"""
Lineas base elementales y prueba de fuga por nivel.

Produce las dos cifras que el documento cita:

  A) PRUEBA DE FUGA. Regresion logistica sobre DOS descriptores de nivel
     (RMS y pico). Si separase las clases, significaria que el modelo puede
     acertar mirando el volumen en lugar de la firma del defecto, que es lo
     que la normalizacion con factor comun por cancion debe impedir.
     Valor medido sobre el manifiesto v2: 0,391.

  B) LINEA BASE ACUSTICA. Regresion logistica sobre SIETE descriptores
     elementales de la forma de onda. Es el piso que el modelo tiene que
     superar para justificar su costo (RNF-02).
     Valor medido sobre el manifiesto v2: 0,463 (sobre el v1 era 0,408).

Que la prueba B funcione no es fuga: significa que los defectos dejan
firmas simples. Lo que seria fuga es que la prueba A funcionara.

Se entrena sobre la particion de ENTRENAMIENTO y se mide sobre la de
VALIDACION. El conjunto de prueba no interviene.

Uso:
    PYTHONPATH=.. python -m src.linea_base \\
        --datos "/ruta/a/datos/dataset_defectos"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from nucleo import config as cfg
from nucleo import senal

SEMILLA = 11                 # la usada para producir las cifras del documento
N_ENTREN, N_VALID = 400, 200   # segmentos por clase

NOMBRES = ["rms_db", "pico_db", "cresta", "centroide", "zcr", "grave", "agudo"]
SOLO_NIVEL = [0, 1]          # rms y pico: la prueba de fuga


def descriptores(x: np.ndarray) -> list[float]:
    """Siete magnitudes elementales de la forma de onda de un segmento."""
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    pico = np.max(np.abs(x)) + 1e-12
    cresta = pico / rms

    esp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + 1e-12
    frec = np.fft.rfftfreq(len(x), 1 / cfg.SR)
    centroide = float(np.sum(frec * esp) / np.sum(esp))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

    pot = esp ** 2
    grave = float(np.sum(pot[frec < 200]) / np.sum(pot))      # pista de hum
    agudo = float(np.sum(pot[frec > 6000]) / np.sum(pot))     # pista de hiss

    return [20 * np.log10(rms), 20 * np.log10(pico), cresta,
            centroide, zcr, grave, agudo]


def reunir(raiz: Path, manifiesto: str, particion: str, por_clase: int,
           rng: random.Random):
    filas = [r for r in csv.DictReader((raiz / manifiesto).open(encoding="utf-8"))
             if r["particion"] == particion]
    agrupadas = defaultdict(list)
    for r in filas:
        agrupadas[r["clase_modelo"]].append(r)

    X, y = [], []
    for clase, rs in sorted(agrupadas.items()):
        for r in rng.sample(rs, min(por_clase, len(rs))):
            i, j = int(r["inicio_muestra"]), int(r["fin_muestra"])
            try:
                x, sr = sf.read(str(raiz / r["ruta"].replace("\\", "/")),
                                start=i, frames=j - i, dtype="float64")
            except Exception:
                continue
            x = senal.a_monofonico(x)
            if len(x) < cfg.MUESTRAS_SEGMENTO:
                continue
            X.append(descriptores(x[: cfg.MUESTRAS_SEGMENTO]))
            y.append(clase)
    return np.array(X), np.array(y)


def ajustar(Xtr, ytr, Xva, yva, columnas, titulo):
    sc = StandardScaler().fit(Xtr[:, columnas])
    modelo = LogisticRegression(max_iter=3000)
    modelo.fit(sc.transform(Xtr[:, columnas]), ytr)
    pred = modelo.predict(sc.transform(Xva[:, columnas]))
    eb = balanced_accuracy_score(yva, pred)

    print("=" * 64)
    print("%s  ->  exactitud balanceada = %.3f" % (titulo, eb))
    print("   descriptores: %s" % ", ".join(NOMBRES[i] for i in columnas))
    clases = sorted(set(yva))
    cm = confusion_matrix(yva, pred, labels=clases)
    print("   matriz de confusión (filas = clase verdadera):")
    print("      %-13s%s" % ("", " ".join("%9s" % c[:9] for c in clases)))
    for c, fila in zip(clases, cm):
        print("      %-13s%s" % (c[:13], " ".join("%9d" % v for v in fila)))
    return float(eb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--informe", type=Path, default=Path("informes/lineas_base.json"))
    args = ap.parse_args()

    rng = random.Random(args.semilla)
    print("extrayendo descriptores...", flush=True)
    Xtr, ytr = reunir(args.datos, args.manifiesto, "entrenamiento", N_ENTREN, rng)
    Xva, yva = reunir(args.datos, args.manifiesto, "validacion", N_VALID, rng)
    print("entrenamiento %s | validación %s\n" % (Xtr.shape, Xva.shape))

    eb_nivel = ajustar(Xtr, ytr, Xva, yva, SOLO_NIVEL,
                       "A) PRUEBA DE FUGA (solo nivel: rms, pico)")
    print()
    eb_base = ajustar(Xtr, ytr, Xva, yva, list(range(len(NOMBRES))),
                      "B) LÍNEA BASE ACÚSTICA (siete descriptores)")

    print("\n" + "=" * 64)
    print("VEREDICTO")
    print("  azar con %d clases              %.3f" % (cfg.N_CLASES, 1 / cfg.N_CLASES))
    print("  A) solo nivel                   %.3f" % eb_nivel)
    print("  B) línea base acústica          %.3f" % eb_base)
    if eb_nivel > 0.60:
        print("  >> FUGA POR NIVEL: las clases se distinguen por volumen.")
    else:
        print("  >> Sin fuga por nivel apreciable.")
    print("  El modelo debe superar %.3f para justificar su costo (RNF-02)." % eb_base)

    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text(json.dumps({
        "semilla": args.semilla,
        "particion_entrenamiento": "entrenamiento",
        "particion_evaluacion": "validacion",
        "segmentos_por_clase": {"entrenamiento": N_ENTREN, "validacion": N_VALID},
        "descriptores": NOMBRES,
        "prueba_de_fuga_solo_nivel": eb_nivel,
        "linea_base_acustica": eb_base,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\ninforme: %s" % args.informe)


if __name__ == "__main__":
    main()
