"""
Calcula los descriptores acusticos de una particion y los guarda.

Son el complemento del log-mel, no su reemplazo. El log-mel de PANN
resuelve 125 Hz con su ventana de 32 ms y suma en una sola banda todo lo
que hay entre 50 y 158 Hz, de modo que la fundamental del zumbido y su
segundo armonico le llegan a la red como un unico numero. Estos
descriptores se calculan sobre la forma de onda, antes de esa perdida.

Se escriben en el ORDEN DEL MANIFIESTO, el mismo que usa `src.exportar`
para los logits, para que las dos matrices se puedan concatenar fila a
fila sin reordenar nada.

Ejecutar:
    PYTHONPATH=.. python -m src.rasgos_acusticos \
        --datos     "C:/.../datos/dataset_defectos" \
        --particion entrenamiento \
        --salida    rasgos/descriptores_entrenamiento.npy
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from nucleo import config as cfg
from nucleo import descriptores as desc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--particion", default="validacion",
                    choices=("entrenamiento", "validacion", "prueba"))
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()

    salida = args.salida or Path("rasgos/descriptores_%s.npy" % args.particion)
    filas = [r for r in csv.DictReader(
        (args.datos / args.manifiesto).open(encoding="utf-8"))
        if r["particion"] == args.particion]
    if not filas:
        print("La particion '%s' no tiene filas en el manifiesto." % args.particion)
        return 1

    # Un archivo se lee una sola vez y se le extraen todos sus segmentos.
    por_ruta: dict[str, list[int]] = {}
    for i, r in enumerate(filas):
        por_ruta.setdefault(r["ruta"], []).append(i)
    print("%d segmentos en %d archivos | partición '%s'"
          % (len(filas), len(por_ruta), args.particion), flush=True)

    X = np.zeros((len(filas), len(desc.NOMBRES)), np.float32)
    t0 = time.time()
    for j, (ruta, idxs) in enumerate(por_ruta.items(), 1):
        audio, sr = sf.read(args.datos / ruta.replace("\\", "/"), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != cfg.SR:
            raise SystemExit("%s esta a %d Hz y se esperaban %d." % (ruta, sr, cfg.SR))
        segs = np.stack([audio[int(filas[i]["inicio_muestra"]):int(filas[i]["fin_muestra"])]
                         for i in idxs])
        X[idxs] = desc.calcular(segs)
        if j % 40 == 0 or j == len(por_ruta):
            transcurrido = time.time() - t0
            print("   %4d / %d archivos | %.0f s | faltan ~%.0f s"
                  % (j, len(por_ruta), transcurrido,
                     transcurrido * (len(por_ruta) - j) / j), flush=True)

    salida.parent.mkdir(parents=True, exist_ok=True)
    np.save(salida, X)
    print("\n%s  %s  %.1f MB" % (salida, X.shape, salida.stat().st_size / 1e6))
    print("columnas: %s" % ", ".join(desc.NOMBRES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
