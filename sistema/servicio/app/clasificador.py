"""
Clasificador de segmentos.

Carga el modelo ajustado UNA vez, al iniciar el contenedor, y lo mantiene
en memoria: el costo de la carga se paga una sola vez y no en cada
peticion. Es el unico componente que depende del artefacto que entrega el
entorno de entrenamiento (ADR-02), de modo que sustituir el modelo afecta a
este componente y a ningun otro (RNF-08).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from nucleo import config as cfg
from nucleo.modelo import Cnn6

RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", "/app/modelo/mejor.pt"))
LOTE = int(os.environ.get("LOTE_INFERENCIA", "32"))


class ModeloNoDisponible(RuntimeError):
    pass


class Clasificador:
    def __init__(self, ruta: Path = RUTA_MODELO):
        self.ruta = Path(ruta)
        self.modelo = None
        self.metadatos: dict = {}

    def cargar(self) -> None:
        if not self.ruta.is_file():
            raise ModeloNoDisponible(
                "No se encontró el modelo en %s. El servicio no puede analizar "
                "sin el artefacto que produce el entorno de entrenamiento." % self.ruta)
        punto = torch.load(self.ruta, map_location="cpu", weights_only=False)
        modelo = Cnn6(cfg.N_CLASES)
        modelo.load_state_dict(punto["modelo"])
        modelo.eval()
        self.modelo = modelo
        self.metadatos = {
            "ruta": str(self.ruta),
            "epoca": punto.get("epoca"),
            "exactitud_balanceada_validacion": punto.get("exactitud_balanceada_validacion"),
            "clases": punto.get("clases", list(cfg.CLASES)),
        }

    @property
    def listo(self) -> bool:
        return self.modelo is not None

    def predecir(self, espectros: np.ndarray):
        """Devuelve (clase_predicha, confianza) por segmento."""
        if self.modelo is None:
            raise ModeloNoDisponible("El modelo no está cargado.")
        if len(espectros) == 0:
            return np.empty((0,), dtype=int), np.empty((0,))

        clases, confianzas = [], []
        with torch.no_grad():
            for i in range(0, len(espectros), LOTE):
                x = torch.from_numpy(espectros[i:i + LOTE]).float()
                probs = torch.softmax(self.modelo(x), dim=1)
                conf, idx = probs.max(dim=1)
                clases.append(idx.numpy())
                confianzas.append(conf.numpy())
        return np.concatenate(clases), np.concatenate(confianzas)


clasificador = Clasificador()
