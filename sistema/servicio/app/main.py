"""
Controlador de analisis.

Unico componente que conoce la interfaz externa: recibe la peticion del
servidor web, invoca al validador, encarga el almacenamiento de la pista y
ordena la secuencia de la cadena de deteccion. Los demas componentes operan
sobre senales y estructuras de datos, sin saber que hay una peticion HTTP
detras, que es lo que permite reutilizarlos desde un guion por lotes.

El analisis es sincrono y se atiende una pista a la vez: el sistema no
incorpora cola de trabajos (antiescenario de la seccion 3.2.4).
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

app = FastAPI(title="Detección de defectos de audio", version="0.1.0")

# El analisis es sincrono: un verrou evita que dos peticiones compitan por
# la CPU y degraden ambas. Es coherente con atender una pista a la vez.
_turno = threading.Lock()

ADVERTENCIA = ("Este resultado no sustituye la escucha del ingeniero. "
               "Un segmento no señalado no garantiza estar libre de defectos.")


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


@app.post("/api/analisis")
def analizar(archivo: UploadFile = File(...)) -> JSONResponse:
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

    t0 = time.time()
    with _turno:
        try:
            segs = preprocesador.procesar(contenido)
        except Exception:
            almacenamiento.eliminar(ident)
            log.exception("fallo al decodificar la pista %s", ident)
            raise HTTPException(
                status_code=400,
                detail="No se pudo decodificar el audio de la pista.")

        if len(segs.espectros) == 0:
            almacenamiento.eliminar(ident)
            raise HTTPException(
                status_code=400,
                detail="La pista es más corta que un segundo; no hay nada que analizar.")

        clases, confianzas = clasificador.predecir(segs.espectros)

    hallazgos = consolidador.consolidar(clases, confianzas,
                                        segs.inicios_s, segs.silenciosos)

    reporte = {
        "id": ident,
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
    log.info("analisis %s: %.1f s de audio, %d hallazgos, %.1f s de computo",
             ident, segs.duracion_s, len(hallazgos), reporte["segundos_de_analisis"])
    return JSONResponse(reporte)


@app.get("/api/analisis/{ident}")
def consultar(ident: str) -> dict:
    reporte = almacenamiento.leer_reporte(ident)
    if reporte is None:
        raise HTTPException(status_code=404,
                            detail="No hay reporte para ese identificador.")
    return reporte


@app.delete("/api/analisis/{ident}")
def cerrar(ident: str) -> dict:
    """Cierre explícito de la sesión (RF-10)."""
    almacenamiento.eliminar(ident)
    return {"eliminado": ident}
