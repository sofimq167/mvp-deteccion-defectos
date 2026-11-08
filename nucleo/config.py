"""
Constantes compartidas entre el entrenamiento y el servicio de analisis.

Estos valores NO se repiten en ningun otro lugar del proyecto. Una
divergencia entre el preprocesamiento de entrenamiento y el de inferencia
produciria una caida de desempeno sin sintoma visible, que es justamente el
riesgo que la seccion 3.2.4 del documento senala. Por eso el paquete
`nucleo` es compartido: el servicio no reimplementa la cadena, la importa.
"""

from __future__ import annotations

# --- representacion de la senal ----------------------------------------
# Parametros del preentrenamiento de PANN sobre AudioSet (Kong et al., 2020).
# Cambiarlos invalida los pesos preentrenados.
SR = 32_000                 # frecuencia de muestreo, Hz
VENTANA = 1_024             # tamano de ventana de la STFT
SALTO = 320                 # salto entre tramas
N_MEL = 64                  # bandas del banco mel
FMIN = 50                   # Hz
FMAX = 14_000               # Hz

DURACION_SEGMENTO_S = 1.0
MUESTRAS_SEGMENTO = int(SR * DURACION_SEGMENTO_S)   # 32 000

# --- clases del modelo --------------------------------------------------
CLASES = ("sin_defecto", "hum", "hiss", "distorsion", "clic")
N_CLASES = len(CLASES)

NOMBRE_LEGIBLE = {
    "sin_defecto": "Sin defecto",
    "hum": "Zumbido (hum)",
    "hiss": "Ruido de fondo (hiss)",
    "distorsion": "Distorsión",
    "clic": "Clic",
}

# --- limites de entrada del servicio ------------------------------------
MAX_BYTES = 200 * 1024 * 1024        # 200 MB
MAX_DURACION_S = 15 * 60             # 15 minutos
MIN_DURACION_S = 1.0

# --- umbral de silencio -------------------------------------------------
UMBRAL_SILENCIO_DB = -60.0

# --- reproducibilidad ---------------------------------------------------
SEMILLA = 20_260_828
