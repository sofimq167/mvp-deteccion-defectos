"""
Descriptores acusticos calculados sobre la forma de onda.

Existen porque el log-mel de PANN pierde informacion que estos defectos
necesitan, y la perdida esta medida: su ventana de 32 ms resuelve 125 Hz,
mientras que los armonicos del zumbido distan 50 Hz, y el banco mel suma
en una sola banda todo lo que hay entre 50 y 158 Hz, de modo que la
fundamental del zumbido y su segundo armonico llegan a la red como un
unico numero.

Estos descriptores se calculan antes de esa perdida. Cada uno codifica el
modelo fisico de un defecto, que es como lo hacen los metodos clasicos
descritos en la seccion 2.3.3: una componente tonal estable para el
zumbido, una discontinuidad impulsiva para el clic, una reduccion del
factor de cresta para la saturacion y un piso de ruido de banda ancha
para el siseo. No reemplazan a la red: se combinan con ella.

El segmento dura un segundo entero, asi que una transformada sobre el
segmento completo da 0,98 Hz de resolucion, suficiente para distinguir
una red electrica de 50 Hz de una de 60 Hz y para separar sus armonicos.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from . import config as cfg

# Nombres en el mismo orden que devuelve `calcular`.
NOMBRES = (
    "rms_db", "pico_db", "cresta_db", "centroide", "zcr",
    "grave", "medio", "agudo",
    "hum_50", "hum_60", "hum_estabilidad",
    "clic_pico", "clic_curtosis", "clic_cuenta",
    "planitud_alta", "piso_ruido_db", "dinamica_db",
)

_N_FFT = 32_768                      # 0,98 Hz por bin a 32 kHz
_VENTANA = get_window("hann", cfg.MUESTRAS_SEGMENTO, fftbins=True)
_FRECS = np.fft.rfftfreq(_N_FFT, 1.0 / cfg.SR)
_EPS = 1e-12


def _db(x):
    return 10.0 * np.log10(np.maximum(x, _EPS))


_PASO_BIN = cfg.SR / _N_FFT


def _indices(f0_nom: float, n_arm: int, span: float, paso: float):
    """Indices de bin de los armonicos, para cada frecuencia candidata.

    El zumbido simulado esta desafinado hasta +-0,6 Hz respecto de la
    nominal, y ese desvio se multiplica por el numero de armonico: a la
    octava ya son casi 5 Hz. Por eso no se puede fijar f0; hay que
    barrerla y quedarse con la que mejor alinea el peine completo.
    """
    cand = np.arange(f0_nom - span, f0_nom + span + paso / 2, paso)
    k = np.arange(1, n_arm + 1)
    centro = np.round(cand[:, None] * k[None, :] / _PASO_BIN).astype(int)
    pico = centro[:, :, None] + np.arange(-1, 2)[None, None, :]
    ancho = int(round(40.0 / _PASO_BIN))
    fondo = (np.round(f0_nom * k / _PASO_BIN).astype(int)[:, None]
             + np.arange(-ancho, ancho + 1)[None, :])
    tope = _N_FFT // 2
    return np.clip(pico, 0, tope), np.clip(fondo, 0, tope)


_PEINE = {f0: _indices(f0, 8, 1.5, 0.05) for f0 in (50.0, 60.0)}


def _peine(mag: np.ndarray, f0_nom: float) -> np.ndarray:
    """Prominencia del peine armonico, maximizada sobre la desafinacion.

    Un zumbido deja picos estrechos en f0, 2*f0, 3*f0... Un bajo musical
    tambien tiene armonicos, pero no en esas frecuencias ni de forma
    estable. Se compara cada pico contra la mediana de su entorno, que es
    el nivel de la musica alrededor.
    """
    pico_idx, fondo_idx = _PEINE[f0_nom]
    pot = mag ** 2
    fondo = np.median(pot[:, fondo_idx], axis=2)                  # (n, arm)
    picos = pot[:, pico_idx].max(axis=3)                          # (n, cand, arm)
    puntaje = (_db(picos) - _db(fondo)[:, None, :]).mean(axis=2)  # (n, cand)
    return puntaje.max(axis=1)


def calcular(segs: np.ndarray) -> np.ndarray:
    """Descriptores de un lote de segmentos.

    Entrada (n, muestras) -> salida (n, len(NOMBRES)) en float32.
    """
    if segs.ndim == 1:
        segs = segs[None, :]
    x = np.asarray(segs, dtype=np.float64)
    n = x.shape[0]

    # --- nivel y dinamica -----------------------------------------------
    pot = np.mean(x ** 2, axis=1)
    pico = np.max(np.abs(x), axis=1)
    rms_db, pico_db = _db(pot), _db(pico ** 2)
    cresta_db = pico_db - rms_db

    # --- espectro de alta resolucion sobre el segmento entero ------------
    esp = np.abs(np.fft.rfft(x * _VENTANA, n=_N_FFT, axis=1))
    pot_esp = esp ** 2
    total = pot_esp.sum(axis=1) + _EPS

    centroide = (pot_esp * _FRECS).sum(axis=1) / total
    grave = pot_esp[:, _FRECS < 250].sum(axis=1) / total
    medio = pot_esp[:, (_FRECS >= 250) & (_FRECS < 4000)].sum(axis=1) / total
    agudo = pot_esp[:, _FRECS >= 8000].sum(axis=1) / total

    cruces = np.mean(np.abs(np.diff(np.sign(x), axis=1)) > 0, axis=1)

    # --- zumbido: peine armonico a 50 y a 60 Hz --------------------------
    hum_50 = _peine(esp, 50.0)
    hum_60 = _peine(esp, 60.0)
    # Estabilidad: un zumbido esta presente en las dos mitades por igual.
    mitad = cfg.MUESTRAS_SEGMENTO // 2
    v = get_window("hann", mitad, fftbins=True)
    e1 = np.abs(np.fft.rfft(x[:, :mitad] * v, n=_N_FFT, axis=1))
    e2 = np.abs(np.fft.rfft(x[:, mitad:] * v, n=_N_FFT, axis=1))
    mejor = np.where(hum_50 >= hum_60, 50.0, 60.0)
    est = np.empty(n)
    for f0 in (50.0, 60.0):
        sel = mejor == f0
        if sel.any():
            est[sel] = -np.abs(_peine(e1[sel], f0) - _peine(e2[sel], f0))

    # --- clic: discontinuidad impulsiva en el tiempo ---------------------
    dif = np.diff(x, axis=1)
    mad = np.median(np.abs(dif - np.median(dif, axis=1, keepdims=True)),
                    axis=1, keepdims=True) + _EPS
    norm = np.abs(dif) / mad
    clic_pico = _db(norm.max(axis=1) ** 2)
    d = dif - dif.mean(axis=1, keepdims=True)
    var = d.var(axis=1) + _EPS
    clic_curtosis = np.mean(d ** 4, axis=1) / var ** 2
    clic_cuenta = np.sum(norm > 20.0, axis=1).astype(np.float64)

    # --- siseo y saturacion ----------------------------------------------
    alta = pot_esp[:, _FRECS >= 4000] + _EPS
    planitud_alta = np.exp(np.mean(np.log(alta), axis=1)) / np.mean(alta, axis=1)

    tramas = x[:, :31 * 1024].reshape(n, 31, 1024)
    e_tr = _db(np.mean(tramas ** 2, axis=2))
    piso = np.percentile(e_tr, 10, axis=1)
    dinamica = np.percentile(e_tr, 95, axis=1) - piso

    salida = np.stack([
        rms_db, pico_db, cresta_db, centroide, cruces,
        grave, medio, agudo,
        hum_50, hum_60, est,
        clic_pico, clic_curtosis, clic_cuenta,
        planitud_alta, piso, dinamica,
    ], axis=1)
    return np.nan_to_num(salida, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
