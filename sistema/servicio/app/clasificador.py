"""
Clasificador de segmentos.

Carga los artefactos UNA vez, al iniciar el contenedor, y los mantiene en
memoria: el costo de la carga se paga una sola vez y no en cada peticion.
Es el unico componente que depende de lo que entrega el entorno de
entrenamiento (ADR-02), de modo que sustituir el modelo afecta a este
componente y a ningun otro (RNF-08).

Son dos artefactos y no uno, porque el sistema desplegado es la fusion de
dos fuentes y no solo la red:

  - `mejor.pt`, la red ajustada por transferencia. El punto de control
    registra con que arquitectura se entreno, de modo que cambiar de
    respaldo no obliga a tocar este archivo.
  - `combinador.joblib`, el conjunto de arboles que combina los logitos de
    la red con los diecisiete descriptores acusticos calculados sobre la
    forma de onda. Se ajusto sobre la particion de entrenamiento y aqui
    solo se aplica.

La diferencia entre usar la red sola y usar la fusion es de 0,594 a 0,649
de exactitud balanceada sobre la particion de prueba, de modo que el
combinador no es un adorno. Si falta el archivo el servicio arranca igual
y responde con la red sola, porque es preferible un analisis degradado y
declarado a un servicio caido; `metadatos["fusion"]` dice cual de los dos
caminos se tomo.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

from nucleo import config as cfg
from nucleo import descriptores
from nucleo.modelo import crear

log = logging.getLogger(__name__)

RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", "/app/modelo/mejor.pt"))
RUTA_COMBINADOR = Path(os.environ.get("RUTA_COMBINADOR",
                                      "/app/modelo/combinador.joblib"))
LOTE = int(os.environ.get("LOTE_INFERENCIA", "32"))


class ModeloNoDisponible(RuntimeError):
    pass


class Clasificador:
    def __init__(self, ruta: Path = RUTA_MODELO,
                 ruta_combinador: Path = RUTA_COMBINADOR):
        self.ruta = Path(ruta)
        self.ruta_combinador = Path(ruta_combinador)
        self.modelo = None
        self.combinador = None
        self.metadatos: dict = {}
        self._meta_combinador: dict = {}

    def cargar(self) -> None:
        if not self.ruta.is_file():
            raise ModeloNoDisponible(
                "No se encontró el modelo en %s. El servicio no puede analizar "
                "sin el artefacto que produce el entorno de entrenamiento." % self.ruta)
        punto = torch.load(self.ruta, map_location="cpu", weights_only=False)
        arquitectura = punto.get("arquitectura", "cnn6")
        modelo = crear(arquitectura, cfg.N_CLASES)
        modelo.load_state_dict(punto["modelo"])
        modelo.eval()
        self.modelo = modelo

        self.combinador = self._cargar_combinador()
        self.metadatos = {
            "ruta": str(self.ruta),
            "arquitectura": arquitectura,
            "epoca": punto.get("epoca"),
            "exactitud_balanceada_validacion": punto.get("exactitud_balanceada_validacion"),
            "clases": punto.get("clases", list(cfg.CLASES)),
            "fusion": self.combinador is not None,
        }
        if self.combinador is not None:
            self.metadatos["exactitud_balanceada_prueba"] = \
                self._meta_combinador.get("exactitud_balanceada_prueba")

    def _cargar_combinador(self):
        """Devuelve el combinador, o None si no esta disponible.

        Un fallo aqui degrada el servicio pero no lo impide: la red sola
        sigue siendo un clasificador valido, solo que peor.
        """
        self._meta_combinador = {}
        if not self.ruta_combinador.is_file():
            log.warning("sin combinador en %s: se analizará con la red sola, "
                        "con el desempeño más bajo que eso implica",
                        self.ruta_combinador)
            return None
        try:
            import joblib
            paquete = joblib.load(self.ruta_combinador)
        except Exception:
            log.exception("no se pudo cargar el combinador %s: se analizará "
                          "con la red sola", self.ruta_combinador)
            return None

        if list(paquete.get("descriptores", [])) != list(descriptores.NOMBRES):
            log.error("el combinador espera descriptores distintos de los que "
                      "este código calcula; se descarta para no producir "
                      "predicciones silenciosamente erróneas")
            return None
        self._meta_combinador = paquete
        return paquete["modelo"]

    @property
    def listo(self) -> bool:
        return self.modelo is not None

    def _logitos(self, espectros: np.ndarray, avance=None) -> np.ndarray:
        total = len(espectros)
        salida = []
        with torch.no_grad():
            for i in range(0, total, LOTE):
                x = torch.from_numpy(espectros[i:i + LOTE]).float()
                salida.append(self.modelo(x).numpy())
                if avance is not None:
                    avance(min(1.0, (i + LOTE) / total))
        return np.concatenate(salida).astype(np.float64)

    def predecir(self, espectros: np.ndarray, ondas=None, avance=None):
        """Devuelve (clase_predicha, confianza) por segmento.

        `ondas` son las formas de onda de los mismos segmentos. Sin ellas
        no se pueden calcular los descriptores y la prediccion recae en la
        red sola, igual que si faltara el combinador.

        `avance` es una funcion opcional que recibe la fraccion completada
        tras cada lote. El controlador la usa para informar el estado del
        analisis (RF-07); por lotes y no por segmento, porque llamarla una
        vez por segmento costaria mas que el propio calculo.
        """
        if self.modelo is None:
            raise ModeloNoDisponible("El modelo no está cargado.")
        if len(espectros) == 0:
            return np.empty((0,), dtype=int), np.empty((0,))

        logitos = self._logitos(espectros, avance=avance)

        if self.combinador is None or ondas is None or len(ondas) != len(espectros):
            probs = torch.softmax(torch.from_numpy(logitos), dim=1).numpy()
            return probs.argmax(1), probs.max(1)

        rasgos = descriptores.calcular(np.asarray(ondas, dtype=np.float64))
        probs = self.combinador.predict_proba(np.hstack([logitos, rasgos]))
        return probs.argmax(1), probs.max(1)


clasificador = Clasificador()
