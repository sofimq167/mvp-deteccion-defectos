"""
Prueba de sistema extremo a extremo (nivel 2 del plan de V&V).

Recorre el flujo completo contra un servicio en ejecucion: salud, analisis
de una pista valida, consulta del reporte, rechazo de entradas invalidas
(RF-02) y cierre de sesion con borrado del material (RF-10).

No es una prueba unitaria: requiere el servicio levantado.

    # en una terminal
    RUTA_MODELO=sistema/servicio/modelo/mejor.pt \\
    PYTHONPATH=".:sistema/servicio" \\
    python -m uvicorn app.main:app --port 8765

    # en otra
    python sistema/servicio/tests/prueba_sistema.py
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.error
import urllib.request
import wave

import numpy as np

BASE = "http://127.0.0.1:8765"
FRONTERA = "----limite-prueba-sistema"


def pista_con_hum(segundos: float = 6.0, sr: int = 32_000) -> bytes:
    """Señal armónica con un zumbido de 60 Hz entre los segundos 2 y 4."""
    t = np.arange(int(segundos * sr)) / sr
    x = 0.25 * np.sin(2 * np.pi * 220 * t) + 0.12 * np.sin(2 * np.pi * 440 * t)
    tramo = (t >= 2.0) & (t < 4.0)
    x[tramo] += 0.08 * np.sin(2 * np.pi * 60 * t[tramo])
    datos = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(datos.tobytes())
    return buf.getvalue()


def pedir(ruta: str, metodo: str = "GET", archivo: tuple | None = None):
    """Devuelve (codigo, cuerpo). No lanza ante un codigo de error HTTP."""
    url = BASE + ruta
    if archivo is None:
        req = urllib.request.Request(url, method=metodo)
    else:
        nombre, contenido = archivo
        cuerpo = ("--%s\r\nContent-Disposition: form-data; name=\"archivo\"; "
                  "filename=\"%s\"\r\nContent-Type: audio/wav\r\n\r\n"
                  % (FRONTERA, nombre)).encode() + contenido + \
                 ("\r\n--%s--\r\n" % FRONTERA).encode()
        req = urllib.request.Request(url, data=cuerpo, method="POST")
        req.add_header("Content-Type",
                       "multipart/form-data; boundary=%s" % FRONTERA)
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def esperar_servicio(intentos: int = 40):
    for _ in range(intentos):
        try:
            return pedir("/api/salud")[1]
        except Exception:
            time.sleep(1)
    raise SystemExit("El servicio no respondió en %d s." % intentos)


def main() -> int:
    fallos = []

    def comprobar(ok: bool, descripcion: str):
        print("  %s  %s" % ("OK   " if ok else "FALLA", descripcion))
        if not ok:
            fallos.append(descripcion)

    print("1. Salud del servicio")
    salud = esperar_servicio()
    comprobar(salud["estado"] == "ok", "el servicio tiene modelo cargado")
    comprobar(len(salud["clases"]) == 5, "declara las cinco clases")

    print("\n2. Análisis de una pista válida")
    cod, rep = pedir("/api/analisis", "POST", ("prueba.wav", pista_con_hum()))
    comprobar(cod == 200, "responde 200")
    comprobar(abs(rep["duracion_s"] - 6.0) < 0.1, "informa la duración correcta")
    comprobar(rep["n_segmentos"] == 6, "produce un segmento por segundo")
    comprobar(bool(rep.get("advertencia")), "incluye la advertencia (RF-08)")
    comprobar(all("inicio_s" in h and "confianza" in h for h in rep["hallazgos"]),
              "cada hallazgo trae ubicación y confianza (RF-04, RF-06)")
    for h in rep["hallazgos"]:
        print("       %-22s %5.1f s a %5.1f s   confianza %.2f"
              % (h["nombre"], h["inicio_s"], h["fin_s"], h["confianza"]))

    ident = rep["id"]

    print("\n3. Consulta del reporte guardado")
    cod, rep2 = pedir("/api/analisis/%s" % ident)
    comprobar(cod == 200 and rep2["id"] == ident, "devuelve el mismo reporte")

    print("\n4. Rechazo de entradas inválidas (RF-02)")
    for nombre, contenido in [("documento.pdf", b"%PDF-1.7 no es audio"),
                              ("vacio.wav", b""),
                              ("corta.wav", pista_con_hum(0.3))]:
        cod, d = pedir("/api/analisis", "POST", (nombre, contenido))
        comprobar(cod == 400 and bool(d.get("detail")),
                  "%s se rechaza con mensaje: %s" % (nombre, str(d.get("detail"))[:60]))

    print("\n5. Cierre de sesión (RF-10)")
    cod, _ = pedir("/api/analisis/%s" % ident, "DELETE")
    comprobar(cod == 200, "la sesión se cierra")
    cod, _ = pedir("/api/analisis/%s" % ident)
    comprobar(cod == 404, "el reporte ya no existe tras el cierre")

    print("\n%s" % ("Todas las comprobaciones pasaron."
                    if not fallos else "%d comprobaciones fallaron." % len(fallos)))
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
