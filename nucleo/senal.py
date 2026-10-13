"""
Cadena de preprocesamiento de la senal, compartida por el entrenamiento y
el servicio de analisis.

    forma de onda -> monofonico -> 32 kHz -> segmentos de 1 s -> log-mel

El banco de filtros mel reproduce el de librosa (escala Slaney, norma
Slaney), que es el que uso torchlibrosa en el preentrenamiento de PANN.
Se implementa aqui para no arrastrar librosa ni torchaudio como
dependencia del servicio.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window, resample_poly

from . import config as cfg


# ---------------------------------------------------------------- mel ----

def _hz_a_mel(f) -> np.ndarray:
    """Escala mel de Slaney: lineal bajo 1 kHz, logaritmica por encima."""
    f = np.atleast_1d(np.asarray(f, dtype=np.float64)).copy()
    f_min, f_sp = 0.0, 200.0 / 3
    mels = (f - f_min) / f_sp
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    alto = f >= min_log_hz
    mels[alto] = min_log_mel + np.log(f[alto] / min_log_hz) / logstep
    return mels


def _mel_a_hz(m) -> np.ndarray:
    m = np.atleast_1d(np.asarray(m, dtype=np.float64)).copy()
    f_min, f_sp = 0.0, 200.0 / 3
    freqs = f_min + f_sp * m
    min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    alto = m >= min_log_mel
    freqs[alto] = min_log_hz * np.exp(logstep * (m[alto] - min_log_mel))
    return freqs


def banco_mel() -> np.ndarray:
    """Matriz (n_mel, 1 + ventana/2) con los filtros triangulares."""
    n_fft = cfg.VENTANA
    frecs = np.linspace(0, cfg.SR / 2, 1 + n_fft // 2)
    puntos = _mel_a_hz(np.linspace(_hz_a_mel(cfg.FMIN)[0],
                                   _hz_a_mel(cfg.FMAX)[0],
                                   cfg.N_MEL + 2))
    pesos = np.zeros((cfg.N_MEL, len(frecs)))
    dif = np.diff(puntos)
    rampa = puntos.reshape(-1, 1) - frecs.reshape(1, -1)
    for i in range(cfg.N_MEL):
        inferior = -rampa[i] / dif[i]
        superior = rampa[i + 2] / dif[i + 1]
        pesos[i] = np.maximum(0, np.minimum(inferior, superior))
    # normalizacion Slaney: area unitaria por filtro
    enorm = 2.0 / (puntos[2:cfg.N_MEL + 2] - puntos[:cfg.N_MEL])
    return pesos * enorm.reshape(-1, 1)


_BANCO = banco_mel()
_VENTANA_HANN = get_window("hann", cfg.VENTANA, fftbins=True)


# ------------------------------------------------------------- senal ----

def a_monofonico(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def remuestrear(x: np.ndarray, sr_origen: int) -> np.ndarray:
    if sr_origen == cfg.SR:
        return x
    from math import gcd
    g = gcd(int(sr_origen), cfg.SR)
    return resample_poly(x, cfg.SR // g, sr_origen // g)


def preparar(x: np.ndarray, sr_origen: int) -> np.ndarray:
    """Monofonico, 32 kHz y rango acotado a [-1, 1] sin recortar."""
    y = remuestrear(a_monofonico(np.asarray(x, dtype=np.float64)), sr_origen)
    pico = float(np.max(np.abs(y))) if y.size else 0.0
    if pico > 1.0:
        y = y / pico
    return y


def segmentar(x: np.ndarray):
    """Divide en segmentos consecutivos de 1 s, sin solape.

    Devuelve (segmentos, inicios_en_muestras). El resto final que no
    completa un segundo se descarta, igual que en la construccion del
    conjunto de datos.
    """
    n = cfg.MUESTRAS_SEGMENTO
    total = len(x) // n
    if total == 0:
        return np.empty((0, n)), np.empty((0,), dtype=int)
    inicios = np.arange(total) * n
    segs = np.stack([x[i:i + n] for i in inicios])
    return segs, inicios


def es_silencio(seg: np.ndarray) -> bool:
    p = float(np.mean(seg ** 2))
    if p <= 0:
        return True
    return 10.0 * np.log10(p) < cfg.UMBRAL_SILENCIO_DB


def log_mel(segs: np.ndarray) -> np.ndarray:
    """Espectrograma log-mel de un lote de segmentos.

    Entrada  (n, muestras)  ->  salida (n, tramas, n_mel), en decibelios.
    """
    if segs.ndim == 1:
        segs = segs[None, :]
    n_fft, salto = cfg.VENTANA, cfg.SALTO
    pad = n_fft // 2
    x = np.pad(segs, ((0, 0), (pad, pad)), mode="reflect")
    n_tramas = 1 + (x.shape[1] - n_fft) // salto

    idx = np.arange(n_fft)[None, :] + salto * np.arange(n_tramas)[:, None]
    tramas = x[:, idx] * _VENTANA_HANN            # (n, tramas, n_fft)
    espectro = np.abs(np.fft.rfft(tramas, axis=-1)) ** 2
    mel = espectro @ _BANCO.T                     # (n, tramas, n_mel)

    # power_to_db con las mismas referencias que torchlibrosa
    mel = np.maximum(mel, 1e-10)
    return (10.0 * np.log10(mel)).astype(np.float32)
