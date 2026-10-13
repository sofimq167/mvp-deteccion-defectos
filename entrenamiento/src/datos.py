"""
Lectura perezosa del conjunto de datos a partir del manifiesto.

Los segmentos no existen como archivos: el manifiesto guarda, para cada uno,
el WAV al que pertenece y sus coordenadas en muestras. Se leen bajo demanda
durante el entrenamiento, que es lo que evita escribir mas de cien mil
archivos de un segundo.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from nucleo import config as cfg
from nucleo import senal


class ConjuntoSegmentos(Dataset):
    def __init__(self, raiz: Path, particion: str, manifiesto: str = "manifiesto_v2.csv",
                 limite_por_clase: int | None = None, semilla: int = cfg.SEMILLA):
        self.raiz = Path(raiz)
        ruta = self.raiz / manifiesto
        if not ruta.is_file():
            raise FileNotFoundError(
                "No se encontro %s. Se espera el manifiesto reetiquetado; "
                "si solo existe manifiesto.csv, pasarlo con --manifiesto." % ruta)

        filas = [r for r in csv.DictReader(ruta.open(encoding="utf-8"))
                 if r["particion"] == particion]

        if limite_por_clase:
            rng = np.random.default_rng(semilla)
            por_clase: dict[str, list] = {}
            for r in filas:
                por_clase.setdefault(r["clase_modelo"], []).append(r)
            filas = []
            for clase, rs in sorted(por_clase.items()):
                idx = rng.permutation(len(rs))[:limite_por_clase]
                filas.extend(rs[i] for i in idx)

        self.filas = filas
        self.indice_clase = {c: i for i, c in enumerate(cfg.CLASES)}

    def __len__(self) -> int:
        return len(self.filas)

    def pesos_por_clase(self) -> torch.Tensor:
        """Peso inverso a la frecuencia, para compensar el desbalance."""
        cuenta = np.zeros(cfg.N_CLASES)
        for r in self.filas:
            cuenta[self.indice_clase[r["clase_modelo"]]] += 1
        cuenta = np.maximum(cuenta, 1)
        w = cuenta.sum() / (cfg.N_CLASES * cuenta)
        return torch.tensor(w, dtype=torch.float32)

    def etiquetas(self) -> np.ndarray:
        return np.array([self.indice_clase[r["clase_modelo"]] for r in self.filas])

    def __getitem__(self, i: int):
        r = self.filas[i]
        ini, fin = int(r["inicio_muestra"]), int(r["fin_muestra"])
        ruta = self.raiz / r["ruta"].replace("\\", "/")
        x, sr = sf.read(str(ruta), start=ini, frames=fin - ini, dtype="float64")
        x = senal.a_monofonico(x)

        if sr != cfg.SR:                      # el conjunto ya esta a 32 kHz
            x = senal.remuestrear(x, sr)
        if len(x) < cfg.MUESTRAS_SEGMENTO:    # relleno defensivo
            x = np.pad(x, (0, cfg.MUESTRAS_SEGMENTO - len(x)))

        espectro = senal.log_mel(x[: cfg.MUESTRAS_SEGMENTO])[0]
        y = self.indice_clase[r["clase_modelo"]]
        return torch.from_numpy(espectro), y
