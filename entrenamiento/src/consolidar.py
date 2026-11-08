"""
Suavizado temporal de las predicciones.

La red decide cada segmento de un segundo por separado, como si el segundo
anterior no dijera nada del siguiente. Pero los defectos de este conjunto
DURAN: la mediana de los intervalos degradados esta entre 7,1 y 7,6
segundos. Un zumbido no aparece y desaparece cada segundo.

Este guion promedia la probabilidad de cada clase sobre una ventana de W
segmentos CONSECUTIVOS DE LA MISMA PISTA y vuelve a decidir. Es una etapa
de posprocesado: no reentrena nada, no conoce donde empiezan ni terminan
los intervalos, y no necesita mas informacion que la que el sistema ya
tiene en produccion.

Lo que NO cambia, que es lo importante para poder comparar:

  - el conjunto de segmentos evaluados es el mismo
  - las cinco clases son las mismas
  - el rango de severidades es el mismo
  - la metrica es la misma de RNF-01

Lo unico que cambia es que el sistema deja de decidir a ciegas segundo a
segundo. El limite W = 1 reproduce exactamente el resultado sin suavizar,
y sirve de comprobacion.

Ejecutar:
    PYTHONPATH=.. python -m src.consolidar \
        --datos        "C:/.../datos/dataset_defectos" \
        --predicciones "C:/.../predicciones_validacion.npz"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from nucleo import config as cfg

SR = 32_000


def bloques(filas: list[dict]) -> list[np.ndarray]:
    """Indices de tramos contiguos de la misma pista y el mismo archivo.

    Un tramo se corta cuando cambia el archivo o cuando el segmento no
    empieza donde terminaba el anterior. Nunca se promedia a traves de una
    frontera de cancion.
    """
    grupos, actual = [], [0]
    for i in range(1, len(filas)):
        a, b = filas[i - 1], filas[i]
        continuo = (a["ruta"] == b["ruta"]
                    and int(b["inicio_muestra"]) == int(a["fin_muestra"]))
        if continuo:
            actual.append(i)
        else:
            grupos.append(np.array(actual)); actual = [i]
    grupos.append(np.array(actual))
    return grupos


def suavizar(prob: np.ndarray, grupos: list[np.ndarray], w: int) -> np.ndarray:
    """Media movil centrada de ancho w dentro de cada tramo contiguo."""
    if w <= 1:
        return prob
    salida = np.empty_like(prob)
    medio = w // 2
    for g in grupos:
        p = prob[g]
        n = len(p)
        # suma acumulada con relleno, para que los bordes promedien solo
        # lo que existe y no se inventen vecinos
        ac = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
        ini = np.clip(np.arange(n) - medio, 0, n)
        fin = np.clip(np.arange(n) - medio + w, 0, n)
        salida[g] = (ac[fin] - ac[ini]) / (fin - ini)[:, None]
    return salida


def balanceada(verdadera: np.ndarray, predicha: np.ndarray) -> tuple[float, list[float]]:
    rec = [float((predicha[verdadera == c] == c).mean()) if (verdadera == c).any()
           else float("nan") for c in range(cfg.N_CLASES)]
    return float(np.nanmean(rec)), rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--predicciones", required=True, type=Path)
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--particion", default="validacion")
    ap.add_argument("--ventanas", type=int, nargs="+",
                    default=[1, 3, 5, 7, 9, 11, 15, 21])
    ap.add_argument("--figura", type=Path, default=None)
    args = ap.parse_args()

    d = np.load(args.predicciones, allow_pickle=True)
    if "logits" not in d:
        print("El archivo no trae 'logits'. Hay que reexportar la salida cruda.")
        return 1
    logits = np.asarray(d["logits"], dtype=np.float64)
    verdadera = np.asarray(d["verdadera"])

    filas = [r for r in csv.DictReader(
        (args.datos / args.manifiesto).open(encoding="utf-8"))
        if r["particion"] == args.particion]
    if len(filas) != len(verdadera):
        print("DESAJUSTE: %d filas contra %d predicciones." % (len(filas), len(verdadera)))
        return 1
    clases = list(cfg.CLASES)
    if not np.array_equal(np.array([clases.index(r["clase_modelo"]) for r in filas]),
                          verdadera):
        print("DESAJUSTE: el orden de las predicciones no es el del manifiesto.")
        return 1

    grupos = bloques(filas)
    largos = np.array([len(g) for g in grupos])
    print("%d segmentos en %d tramos contiguos (mediana %d s, máximo %d s)\n"
          % (len(filas), len(grupos), int(np.median(largos)), int(largos.max())))

    # softmax estable
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z); prob /= prob.sum(axis=1, keepdims=True)

    print("%-9s %9s   %s" % ("ventana", "balanc.",
                             " ".join("%9s" % c[:9] for c in clases)))
    curva = []
    for w in args.ventanas:
        pred = suavizar(prob, grupos, w).argmax(1)
        eb, rec = balanceada(verdadera, pred)
        curva.append((w, eb, rec))
        print("%-9s %9.4f   %s" % ("%d s" % w, eb,
                                   " ".join("%9.3f" % r for r in rec)))

    base, mejor = curva[0], max(curva, key=lambda t: t[1])
    print("\nsin suavizar (W=1): %.4f" % base[1])
    print("mejor ventana W=%d:  %.4f   (%+.4f)" % (mejor[0], mejor[1], mejor[1] - base[1]))
    cumplen = [clases[i] for i, r in enumerate(mejor[2]) if r >= 0.80]
    print("clases que alcanzan el 0,80 de RNF-01: %s"
          % (", ".join(cumplen) if cumplen else "ninguna"))

    if args.figura:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, eje = plt.subplots(figsize=(7.2, 4.2))
        for i, c in enumerate(clases):
            eje.plot([w for w, _, _ in curva], [r[i] for _, _, r in curva],
                     marker="o", lw=1.2, label=cfg.NOMBRE_LEGIBLE.get(c, c))
        eje.plot([w for w, _, _ in curva], [e for _, e, _ in curva],
                 marker="s", lw=2.4, color="0.2", label="Balanceada")
        eje.axhline(0.80, ls="--", lw=1, color="0.5")
        eje.text(curva[0][0], 0.815, " umbral RNF-01", fontsize=8, color="0.4")
        eje.set_xlabel("ventana de suavizado (segundos)")
        eje.set_ylabel("exhaustividad")
        eje.set_ylim(0, 1)
        eje.grid(alpha=.25)
        eje.legend(fontsize=8, ncol=2)
        args.figura.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figura, dpi=200, bbox_inches="tight")
        print("figura: %s" % args.figura)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
