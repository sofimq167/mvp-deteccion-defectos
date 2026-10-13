"""
Precalculo de los espectrogramas log-mel de todo el conjunto.

Por que hace falta: el audio generado ocupa unos 13 GB, y en cada epoca el
entrenamiento vuelve a decodificar los WAV y a recalcular la FFT de los
mismos segmentos. Precalcular una sola vez resuelve dos problemas a la vez.

  - Tamano. Un segmento de 1 s son 32 000 muestras, pero su log-mel son
    101 x 64 valores. En float16 el conjunto completo baja de 13 GB a
    1,45 GB, que ya cabe en un disco de Drive y se sube a un entorno de
    computo en la nube sin dificultad.

  - Velocidad. El costo por lote pasa a ser una lectura de memoria. En las
    mediciones hechas sobre CPU, el preprocesamiento dominaba el tiempo de
    epoca, no la red.

Los WAV se abren una sola vez cada uno y se recorren en orden, en lugar de
reabrirlos por segmento.

Uso:
    PYTHONPATH=.. python -m src.precalcular \\
        --datos "/ruta/a/datos/dataset_defectos" \\
        --salida "/ruta/a/datos/rasgos"
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from nucleo import config as cfg
from nucleo import senal

PARTICIONES = ("entrenamiento", "validacion", "prueba")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--particiones", nargs="*", default=list(PARTICIONES))
    args = ap.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    indice_clase = {c: i for i, c in enumerate(cfg.CLASES)}

    filas = list(csv.DictReader((args.datos / args.manifiesto).open(encoding="utf-8")))
    print("manifiesto: %d segmentos" % len(filas))

    for particion in args.particiones:
        sub = [r for r in filas if r["particion"] == particion]
        if not sub:
            print("  %s: sin segmentos, se omite" % particion)
            continue

        # Agrupar por archivo: cada WAV se abre una sola vez.
        por_archivo: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(sub):
            por_archivo[r["ruta"]].append(i)

        n = len(sub)
        n_tramas = 1 + cfg.MUESTRAS_SEGMENTO // cfg.SALTO
        rasgos = np.lib.format.open_memmap(
            args.salida / ("rasgos_%s.npy" % particion), mode="w+",
            dtype=np.float16, shape=(n, n_tramas, cfg.N_MEL))
        etiquetas = np.zeros(n, dtype=np.int8)

        t0, hechos = time.time(), 0
        for k, (ruta, indices) in enumerate(sorted(por_archivo.items()), 1):
            completa = args.datos / ruta.replace("\\", "/")
            with sf.SoundFile(str(completa)) as f:
                for i in indices:
                    r = sub[i]
                    ini = int(r["inicio_muestra"])
                    f.seek(ini)
                    x = f.read(cfg.MUESTRAS_SEGMENTO, dtype="float64", always_2d=False)
                    x = senal.a_monofonico(np.asarray(x))
                    if len(x) < cfg.MUESTRAS_SEGMENTO:
                        x = np.pad(x, (0, cfg.MUESTRAS_SEGMENTO - len(x)))
                    rasgos[i] = senal.log_mel(x)[0].astype(np.float16)
                    etiquetas[i] = indice_clase[r["clase_modelo"]]
                    hechos += 1

            if k % 25 == 0 or k == len(por_archivo):
                transcurrido = time.time() - t0
                print("  %s: %d/%d archivos | %d/%d segmentos | %.0f s"
                      % (particion, k, len(por_archivo), hechos, n, transcurrido),
                      flush=True)

        rasgos.flush()
        np.save(args.salida / ("etiquetas_%s.npy" % particion), etiquetas)

        # Metadatos para los desgloses del Capitulo 4: por metodo de
        # simulacion y por numero de clics en el segmento.
        meta = [{"pista": r["pista"],
                 "clase_simulacion": r["clase_simulacion"],
                 "n_clics_segmento": r.get("n_clics_segmento", "")} for r in sub]
        (args.salida / ("meta_%s.json" % particion)).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        mb = (args.salida / ("rasgos_%s.npy" % particion)).stat().st_size / 1e6
        print("  %s: %d segmentos, %.0f MB, %.0f s\n" % (particion, n, mb, time.time() - t0))

    print("Listo. Para entrenar desde aqui, pasar --rasgos %s" % args.salida)


if __name__ == "__main__":
    main()
