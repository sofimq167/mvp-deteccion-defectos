"""
Preprocesador de senal.

No reimplementa la cadena: importa `nucleo.senal`, que es el mismo codigo
que produjo el conjunto de datos. Devuelve tanto el espectrograma, que
consume la red, como la forma de onda, sobre la que se calculan los
descriptores acusticos: estos miden propiedades que el log-mel ya perdio,
de modo que no pueden derivarse de el. Cualquier divergencia entre el
preprocesamiento de entrenamiento y el de inferencia produciria una caida
de desempeno sin sintoma visible.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from nucleo import config as cfg
from nucleo import senal


@dataclass
class Segmentos:
    espectros: np.ndarray      # (n, tramas, n_mel)
    ondas: np.ndarray          # (n, muestras) forma de onda sin transformar
    inicios_s: np.ndarray      # instante de inicio de cada segmento
    silenciosos: np.ndarray    # mascara booleana
    duracion_s: float


def procesar(contenido: bytes) -> Segmentos:
    x, sr = sf.read(io.BytesIO(contenido), dtype="float64")
    y = senal.preparar(x, sr)
    duracion = len(y) / cfg.SR

    segs, inicios = senal.segmentar(y)
    if len(segs) == 0:
        return Segmentos(np.empty((0, 0, 0)), np.empty((0, 0)), np.empty((0,)),
                         np.empty((0,), dtype=bool), duracion)

    silencio = np.array([senal.es_silencio(s) for s in segs])
    espectros = senal.log_mel(segs)
    return Segmentos(espectros=espectros,
                     ondas=segs,
                     inicios_s=inicios / cfg.SR,
                     silenciosos=silencio,
                     duracion_s=duracion)
