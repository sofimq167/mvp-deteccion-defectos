"""
CNN6 de la familia PANN (Kong et al., 2020), con la cabeza de clasificacion
sustituida por una de cinco clases.

La arquitectura reproduce la del repositorio original para que los pesos
preentrenados sobre AudioSet carguen sin conversion. La diferencia esta en
que aqui el espectrograma log-mel llega ya calculado por `nucleo.senal`, en
vez de computarse dentro de la red: el servicio y el entrenamiento comparten
asi exactamente la misma cadena de preprocesamiento.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as cfg


class BloqueConv5x5(nn.Module):
    """Bloque convolucional de CNN6: una convolucion 5x5 y normalizacion."""

    def __init__(self, entrada: int, salida: int):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, salida, kernel_size=(5, 5),
                               stride=(1, 1), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(salida)

    def forward(self, x, tam_pool=(2, 2), tipo_pool="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        if tipo_pool == "max":
            return F.max_pool2d(x, kernel_size=tam_pool)
        if tipo_pool == "avg":
            return F.avg_pool2d(x, kernel_size=tam_pool)
        raise ValueError("tipo de agrupamiento desconocido: %r" % tipo_pool)


class Cnn6(nn.Module):
    """Cuerpo de CNN6 mas una cabeza lineal de `n_clases` salidas."""

    def __init__(self, n_clases: int = cfg.N_CLASES):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(cfg.N_MEL)
        self.conv_block1 = BloqueConv5x5(1, 64)
        self.conv_block2 = BloqueConv5x5(64, 128)
        self.conv_block3 = BloqueConv5x5(128, 256)
        self.conv_block4 = BloqueConv5x5(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.cabeza = nn.Linear(512, n_clases, bias=True)

    def forward(self, x):
        """x: (lote, tramas, n_mel) en decibelios -> logits (lote, n_clases)."""
        x = x.unsqueeze(1)                 # (lote, 1, tramas, mel)
        x = x.transpose(1, 3)              # (lote, mel, tramas, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)              # (lote, 1, tramas, mel)

        x = self.conv_block1(x, (2, 2), "avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, (2, 2), "avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, (2, 2), "avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, (2, 2), "avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)                       # promedio en frecuencia
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.mean(x, dim=2)                       # agregacion temporal

        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.cabeza(x)


def cargar_preentrenado(modelo: Cnn6, ruta_pesos: str) -> dict:
    """Carga los pesos de AudioSet, descartando la cabeza de 527 clases.

    Devuelve un resumen de que se cargo y que no, para dejarlo registrado:
    cargar en silencio una red mal inicializada es un fallo que solo se
    nota al final, en las metricas.
    """
    bruto = torch.load(ruta_pesos, map_location="cpu", weights_only=False)
    estado = bruto.get("model", bruto)

    propio = modelo.state_dict()
    compatibles, descartados = {}, []
    for k, v in estado.items():
        if k.startswith("spectrogram_extractor") or k.startswith("logmel_extractor"):
            continue                      # el log-mel se calcula fuera de la red
        if k in propio and propio[k].shape == v.shape:
            compatibles[k] = v
        else:
            descartados.append(k)

    faltantes = [k for k in propio if k not in compatibles]
    modelo.load_state_dict(compatibles, strict=False)
    return {"cargados": len(compatibles),
            "descartados": descartados,
            "sin_inicializar": faltantes}
