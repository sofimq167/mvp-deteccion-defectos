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


def cargar_preentrenado(modelo: nn.Module, ruta_pesos: str) -> dict:
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


class BloqueConv3x3(nn.Module):
    """Bloque convolucional de CNN14: dos convoluciones 3x3 encadenadas.

    Difiere del de CNN6 en que aplica dos convoluciones pequenas en vez de
    una grande. A igual campo receptivo resulta mas expresivo y mas hondo,
    que es de donde sale la diferencia de capacidad entre las dos redes.
    """

    def __init__(self, entrada: int, salida: int):
        super().__init__()
        self.conv1 = nn.Conv2d(entrada, salida, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(salida, salida, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(salida)
        self.bn2 = nn.BatchNorm2d(salida)

    def forward(self, x, tam_pool=(2, 2), tipo_pool="avg"):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if tam_pool == (1, 1):
            return x
        if tipo_pool == "max":
            return F.max_pool2d(x, kernel_size=tam_pool)
        if tipo_pool == "avg":
            return F.avg_pool2d(x, kernel_size=tam_pool)
        raise ValueError("tipo de agrupamiento desconocido: %r" % tipo_pool)


class _RedVGG(nn.Module):
    """Base de CNN10 y CNN14, que solo se diferencian en ancho y profundidad.

    Las dos toman la misma entrada que CNN6, el log-mel con los parametros
    del preentrenamiento, de modo que los rasgos precalculados sirven para
    las tres sin volver a extraerlos. Los nombres de los modulos reproducen
    los del repositorio original para que los pesos carguen sin conversion.
    """

    CANALES: tuple = ()
    ULTIMO_REDUCE: bool = True

    def __init__(self, n_clases: int = cfg.N_CLASES):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(cfg.N_MEL)
        self.n_bloques = len(self.CANALES) - 1
        for i in range(self.n_bloques):
            setattr(self, "conv_block%d" % (i + 1),
                    BloqueConv3x3(self.CANALES[i], self.CANALES[i + 1]))
        ancho = self.CANALES[-1]
        self.fc1 = nn.Linear(ancho, ancho, bias=True)
        self.cabeza = nn.Linear(ancho, n_clases, bias=True)

    def forward(self, x):
        """x: (lote, tramas, n_mel) en decibelios -> logits (lote, n_clases)."""
        x = x.unsqueeze(1)                 # (lote, 1, tramas, mel)
        x = x.transpose(1, 3)              # (lote, mel, tramas, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)              # (lote, 1, tramas, mel)

        for i in range(1, self.n_bloques + 1):
            # El ultimo bloque de CNN14 no reduce: a esa altura quedan
            # tres tramas y agruparlas otra vez perderia el eje temporal.
            ultimo = i == self.n_bloques
            tam = (1, 1) if (ultimo and not self.ULTIMO_REDUCE) else (2, 2)
            x = getattr(self, "conv_block%d" % i)(x, tam, "avg")
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


class Cnn10(_RedVGG):
    """CNN10: la misma anchura que CNN6 pero el doble de convoluciones.

    Es la variante intermedia de la Tabla de variantes PANN y la de mejor
    relacion entre costo y ganancia para este trabajo: pasa de 4,8 a 5,2
    millones de parametros, un aumento del ocho por ciento, y sube el mAP
    sobre AudioSet de 0,343 a 0,380. Con una muestra efectiva de ochenta
    canciones, ganar expresividad sin ganar tamano es lo que conviene.
    """

    CANALES = (1, 64, 128, 256, 512)
    ULTIMO_REDUCE = True


class Cnn14(_RedVGG):
    """CNN14: seis bloques y hasta 2048 canales, unos 80 M de parametros.

    Alcanza 0,431 de precision media sobre AudioSet frente a los 0,343 de
    CNN6, a costa de diecisiete veces mas parametros. El ADR-01 del
    documento la habia descartado por el tamano efectivo de la muestra.
    """

    CANALES = (1, 64, 128, 256, 512, 1024, 2048)
    ULTIMO_REDUCE = False


ARQUITECTURAS = {"cnn6": Cnn6, "cnn10": Cnn10, "cnn14": Cnn14}


def crear(arquitectura: str, n_clases: int = cfg.N_CLASES) -> nn.Module:
    """Instancia la red por nombre, para que el guion la elija por bandera."""
    clave = arquitectura.lower()
    if clave not in ARQUITECTURAS:
        raise ValueError("arquitectura desconocida: %r (hay %s)"
                         % (arquitectura, ", ".join(ARQUITECTURAS)))
    return ARQUITECTURAS[clave](n_clases)
