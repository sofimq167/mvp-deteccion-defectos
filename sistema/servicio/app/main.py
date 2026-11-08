"""
Controlador de analisis.

Unico componente que conoce la interfaz externa: recibe la peticion del
servidor web, invoca al validador, encarga el almacenamiento de la pista y
ordena la secuencia de la cadena de deteccion. Los demas componentes operan
sobre senales y estructuras de datos, sin saber que hay una peticion HTTP
detras, que es lo que permite reutilizarlos desde un guion por lotes.

El analisis se ejecuta EN SEGUNDO PLANO y el cliente consulta su estado,
que es lo que exige RF-07: la interfaz debe informar el avance mientras el
analisis corre. La validacion, en cambio, sigue siendo sincrona, porque
RF-02 pide que un archivo inadmisible se rechace de inmediato y no tras
una espera.

Se atiende una pista a la vez: el sistema no incorpora cola de trabajos
(antiescenario de la seccion 3.2.4). Si llega una segunda peticion
mientras otra corre, espera su turno y lo declara en su estado.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from nucleo import config as cfg
from app import almacenamiento, consolidador, preprocesador
from app.clasificador import ModeloNoDisponible, clasificador
from app.validador import EntradaInvalida, validar

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("servicio")

app = FastAPI(title="Detección de defectos de audio", version="0.2.0")

# Un verrou evita que dos analisis compitan por la CPU y degraden ambos.
# Es coherente con atender una pista a la vez.
_turno = threading.Lock()

# Estado de los analisis en curso. No guarda audio ni reportes: solo el
# avance. El reporte terminado vive en el almacenamiento temporal.
_tareas: dict[str, dict] = {}
_candado = threading.Lock()

ADVERTENCIA = ("Este resultado no sustituye la escucha del ingeniero. "
               "Un segmento no señalado no garantiza estar libre de defectos.")

ETAPAS = {
    "en_cola": "Esperando turno",
    "preparando": "Preparando la señal",
    "clasificando": "Clasificando segmentos",
    "consolidando": "Consolidando hallazgos",
}


def _fijar(ident: str, **campos) -> None:
    with _candado:
        if ident in _tareas:
            _tareas[ident].update(campos)


def _leer_tarea(ident: str):
    with _candado:
        t = _tareas.get(ident)
        return dict(t) if t else None


@app.on_event("startup")
def iniciar() -> None:
    try:
        clasificador.cargar()
        log.info("modelo cargado: %s", clasificador.metadatos)
    except ModeloNoDisponible as e:
        log.error("%s", e)       # el servicio arranca igual, para poder diagnosticar
    n = almacenamiento.purgar_vencidas()
    if n:
        log.info("sesiones vencidas purgadas al iniciar: %d", n)


@app.get("/api/salud")
def salud() -> dict:
    return {"estado": "ok" if clasificador.listo else "sin_modelo",
            "modelo": clasificador.metadatos,
            "clases": list(cfg.CLASES)}


def _ejecutar(ident: str, contenido: bytes, meta) -> None:
    """Cadena de deteccion completa, fuera del hilo de la peticion."""
    t0 = time.time()
    try:
        with _turno:
            _fijar(ident, etapa="preparando", progreso=0.05)
            try:
                segs = preprocesador.procesar(contenido)
            except Exception:
                log.exception("fallo al decodificar la pista %s", ident)
                raise ValueError("No se pudo decodificar el audio de la pista.")

            if len(segs.espectros) == 0:
                raise ValueError("La pista es más corta que un segundo; "
                                 "no hay nada que analizar.")

            _fijar(ident, etapa="clasificando", progreso=0.15)

            def avance(fraccion: float) -> None:
                # la clasificacion ocupa el grueso del trabajo
                _fijar(ident, progreso=round(0.15 + 0.80 * fraccion, 3))

            clases, confianzas = clasificador.predecir(segs.espectros,
                                                       ondas=segs.ondas,
                                                       avance=avance)

        _fijar(ident, etapa="consolidando", progreso=0.97)
        hallazgos = consolidador.consolidar(clases, confianzas,
                                            segs.inicios_s, segs.silenciosos)

        reporte = {
            "id": ident,
            "estado": "listo",
            "duracion_s": round(segs.duracion_s, 2),
            "frecuencia_original_hz": meta.frecuencia_hz,
            "canales_original": meta.canales,
            "n_segmentos": int(len(clases)),
            "segmentos_por_clase": consolidador.resumen(clases, segs.silenciosos),
            "hallazgos": hallazgos,
            "segundos_de_analisis": round(time.time() - t0, 2),
            "modelo": clasificador.metadatos,
            "advertencia": ADVERTENCIA,
        }
        almacenamiento.guardar_reporte(ident, reporte)
        _fijar(ident, estado="listo", progreso=1.0, etapa="listo")
        log.info("analisis %s: %.1f s de audio, %d hallazgos, %.1f s de computo",
                 ident, segs.duracion_s, len(hallazgos),
                 reporte["segundos_de_analisis"])

    except Exception as e:
        almacenamiento.eliminar(ident)
        _fijar(ident, estado="error", etapa="error",
               detalle=str(e) or "El análisis falló.")
        log.error("analisis %s fallido: %s", ident, e)


@app.post("/api/analisis", status_code=202)
def analizar(archivo: UploadFile = File(...)) -> JSONResponse:
    """Acepta la pista y lanza el analisis; el avance se consulta aparte."""
    if not clasificador.listo:
        raise HTTPException(
            status_code=503,
            detail="El servicio no tiene modelo cargado y no puede analizar.")

    contenido = archivo.file.read()
    try:
        meta = validar(contenido, archivo.filename or "pista.wav")
    except EntradaInvalida as e:
        raise HTTPException(status_code=400, detail=str(e))

    ident = almacenamiento.nuevo_identificador()
    almacenamiento.guardar_pista(ident, contenido)

    with _candado:
        _tareas[ident] = {"id": ident, "estado": "en_proceso",
                          "etapa": "en_cola", "progreso": 0.0,
                          "duracion_s": round(meta.duracion_s, 2)}

    threading.Thread(target=_ejecutar, args=(ident, contenido, meta),
                     daemon=True).start()

    return JSONResponse(status_code=202, content={
        "id": ident,
        "estado": "en_proceso",
        "duracion_s": round(meta.duracion_s, 2),
        "frecuencia_original_hz": meta.frecuencia_hz,
        "canales_original": meta.canales,
    })


@app.get("/api/analisis/{ident}")
def consultar(ident: str) -> dict:
    """Avance mientras corre; el reporte completo cuando termina."""
    reporte = almacenamiento.leer_reporte(ident)
    if reporte is not None:
        return reporte

    tarea = _leer_tarea(ident)
    if tarea is None:
        raise HTTPException(status_code=404,
                            detail="No hay reporte para ese identificador.")

    if tarea.get("estado") == "error":
        raise HTTPException(status_code=400,
                            detail=tarea.get("detalle", "El análisis falló."))

    etapa = tarea.get("etapa", "en_cola")
    return {"id": ident,
            "estado": "en_proceso",
            "progreso": tarea.get("progreso", 0.0),
            "etapa": etapa,
            "etapa_texto": ETAPAS.get(etapa, "Procesando"),
            "duracion_s": tarea.get("duracion_s")}


@app.delete("/api/analisis/{ident}")
def cerrar(ident: str) -> dict:
    """Cierre explícito de la sesión (RF-10)."""
    almacenamiento.eliminar(ident)
    with _candado:
        _tareas.pop(ident, None)
    return {"eliminado": ident}
