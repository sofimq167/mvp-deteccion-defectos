"""
Validador de entrada (RF-02).

Primera linea de los controles de seguridad de la seccion 3.2.6: comprueba
cabecera, formato, tamano y duracion ANTES de que el archivo llegue al
decodificador de audio, que es donde un WAV malformado podria provocar un
comportamiento inesperado.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import soundfile as sf

from nucleo import config as cfg


class EntradaInvalida(Exception):
    """El archivo no es analizable. El mensaje va dirigido al usuario."""


@dataclass
class Metadatos:
    duracion_s: float
    frecuencia_hz: int
    canales: int
    bytes: int


def validar(contenido: bytes, nombre: str) -> Metadatos:
    if not contenido:
        raise EntradaInvalida("El archivo llegó vacío.")

    if len(contenido) > cfg.MAX_BYTES:
        raise EntradaInvalida(
            "El archivo pesa %.0f MB y el máximo admitido es %.0f MB."
            % (len(contenido) / 1e6, cfg.MAX_BYTES / 1e6))

    # Cabecera RIFF/WAVE: se comprueba antes de decodificar nada.
    if contenido[:4] != b"RIFF" or contenido[8:12] != b"WAVE":
        raise EntradaInvalida(
            "El archivo no es un WAV. Exportá la pista en formato WAV desde "
            "tu estación de trabajo e intentá de nuevo.")

    try:
        info = sf.info(io.BytesIO(contenido))
    except Exception:
        raise EntradaInvalida(
            "El archivo tiene cabecera de WAV pero no se puede leer; "
            "es posible que esté incompleto o dañado.")

    if info.duration < cfg.MIN_DURACION_S:
        raise EntradaInvalida(
            "La pista dura %.2f s y hacen falta al menos %.0f s para analizarla."
            % (info.duration, cfg.MIN_DURACION_S))

    if info.duration > cfg.MAX_DURACION_S:
        raise EntradaInvalida(
            "La pista dura %.1f minutos y el máximo admitido es %.0f minutos."
            % (info.duration / 60, cfg.MAX_DURACION_S / 60))

    return Metadatos(duracion_s=float(info.duration),
                     frecuencia_hz=int(info.samplerate),
                     canales=int(info.channels),
                     bytes=len(contenido))
