"""
Acceso al almacenamiento temporal (RF-10, RNF-06).

Conserva la pista y su reporte mientras dura la sesion de trabajo y los
elimina al vencer. No es una base de datos y no guarda nada de forma
permanente: al no existir cuentas de usuario, no hay historial que mantener
ni informacion que asociar a una persona.

Los nombres de archivo los genera el sistema y nunca se derivan del nombre
original, que es un control del analisis de amenazas de la seccion 3.2.6.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

RAIZ = Path(os.environ.get("RUTA_TEMPORAL", "/tmp/analisis"))
VIDA_SESION_S = int(os.environ.get("VIDA_SESION_S", str(60 * 60)))   # 1 hora


def _dir(ident: str) -> Path:
    return RAIZ / ident


def nuevo_identificador() -> str:
    return uuid.uuid4().hex


def guardar_pista(ident: str, contenido: bytes) -> Path:
    d = _dir(ident)
    d.mkdir(parents=True, exist_ok=True)
    ruta = d / "pista.wav"              # nombre generado, no el del usuario
    ruta.write_bytes(contenido)
    return ruta


def guardar_reporte(ident: str, reporte: dict) -> None:
    d = _dir(ident)
    d.mkdir(parents=True, exist_ok=True)
    (d / "reporte.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")


def leer_reporte(ident: str) -> dict | None:
    ruta = _dir(ident) / "reporte.json"
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def eliminar(ident: str) -> None:
    shutil.rmtree(_dir(ident), ignore_errors=True)


def purgar_vencidas() -> int:
    """Elimina las sesiones que superaron su vida util. Devuelve cuantas."""
    if not RAIZ.is_dir():
        return 0
    limite = time.time() - VIDA_SESION_S
    n = 0
    for d in RAIZ.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < limite:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        except OSError:
            continue
    return n
