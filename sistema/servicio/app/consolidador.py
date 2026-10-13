"""
Consolidador de hallazgos (RF-04, RF-06).

Traduce la salida del clasificador al lenguaje del usuario: una secuencia
de clases por segundo no es un reporte. Agrupa los segmentos contiguos de
una misma clase en hallazgos con instante de inicio y de fin, y deriva su
confianza de la de los segmentos que lo componen.

Decision explicita: dos tramos de la misma clase separados por un unico
segmento de otra clase se unen en un solo hallazgo. Un reporte que parte un
zumbido continuo en seis hallazgos porque un segmento se clasifico distinto
es mas dificil de leer y no describe mejor la pista.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from nucleo import config as cfg

HUECO_TOLERADO = 1          # segmentos de otra clase que no rompen el hallazgo
MIN_SEGMENTOS = 1           # duracion minima de un hallazgo


@dataclass
class Hallazgo:
    clase: str
    nombre: str
    inicio_s: float
    fin_s: float
    duracion_s: float
    confianza: float
    n_segmentos: int


def consolidar(clases: np.ndarray, confianzas: np.ndarray,
               inicios_s: np.ndarray, silenciosos: np.ndarray) -> list[dict]:
    if len(clases) == 0:
        return []

    etiquetas = np.array([cfg.CLASES[i] for i in clases], dtype=object)
    # Los segmentos silenciosos no se reportan: no hay senal que juzgar.
    etiquetas[silenciosos] = "sin_defecto"

    hallazgos: list[Hallazgo] = []
    i = 0
    n = len(etiquetas)
    while i < n:
        clase = etiquetas[i]
        if clase == "sin_defecto":
            i += 1
            continue

        indices = [i]
        j = i + 1
        hueco = 0
        while j < n:
            if etiquetas[j] == clase:
                indices.append(j)
                hueco = 0
            else:
                hueco += 1
                if hueco > HUECO_TOLERADO:
                    break
            j += 1

        if len(indices) >= MIN_SEGMENTOS:
            ini = float(inicios_s[indices[0]])
            fin = float(inicios_s[indices[-1]] + cfg.DURACION_SEGMENTO_S)
            conf = float(np.mean(confianzas[indices]))
            hallazgos.append(Hallazgo(
                clase=clase, nombre=cfg.NOMBRE_LEGIBLE[clase],
                inicio_s=round(ini, 2), fin_s=round(fin, 2),
                duracion_s=round(fin - ini, 2),
                confianza=round(conf, 3), n_segmentos=len(indices)))
        i = indices[-1] + 1

    hallazgos.sort(key=lambda h: h.inicio_s)
    return [asdict(h) for h in hallazgos]


def resumen(clases: np.ndarray, silenciosos: np.ndarray) -> dict:
    """Conteo de segmentos por clase, para el encabezado del reporte."""
    conteo = {c: 0 for c in cfg.CLASES}
    for k, s in zip(clases, silenciosos):
        conteo["sin_defecto" if s else cfg.CLASES[k]] += 1
    return conteo
