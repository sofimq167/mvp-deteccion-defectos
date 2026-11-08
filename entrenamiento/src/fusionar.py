"""
Combina la salida de la red con los descriptores acusticos.

Las dos fuentes alcanzan por separado un desempeno parecido (0,58 la red,
0,57 los descriptores) y sin embargo combinarlas gana varios puntos, lo
que significa que no son redundantes: cada una ve algo que la otra no.
Esa complementariedad es la evidencia de que el limite del sistema esta
en lo que la representacion log-mel conserva y no en la capacidad del
modelo.

AVISO METODOLOGICO. Los logits de la particion de entrenamiento provienen
de una red que vio esos mismos segmentos, de modo que alli la red parece
mejor de lo que es (0,66 contra 0,58 en validacion). Un fusionador
ajustado sobre ellos aprende a confiar en la red mas de lo debido. Por eso
se comparan dos variantes:

  (a) ajuste sobre ENTRENAMIENTO, con esa distorsion pero mas datos
  (b) ajuste por validacion cruzada AGRUPADA POR CANCION dentro de
      validacion, donde los logits si son fuera de muestra

La variante (b) tiene menos datos pero no sufre el sesgo. Se reportan las
dos y se retiene la que mida mejor sin trampa.

Ejecutar:
    PYTHONPATH=.. python -m src.fusionar \
        --datos      "C:/.../datos/dataset_defectos" \
        --logits-ent "C:/.../predicciones_entrenamiento.npz" \
        --logits-val "C:/.../predicciones_validacion.npz"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

from nucleo import config as cfg
from nucleo import descriptores as desc

C = list(cfg.CLASES)


def nuevo_modelo(semilla: int = 0) -> HistGradientBoostingClassifier:
    # class_weight balanceado porque la metrica de RNF-01 pesa igual a las
    # cinco clases, aunque `sin_defecto` sea casi un tercio de los datos.
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                          class_weight="balanced",
                                          random_state=semilla)


def informe(nombre: str, y: np.ndarray, pred: np.ndarray) -> float:
    eb = balanced_accuracy_score(y, pred)
    rec = [(pred[y == c] == c).mean() for c in range(cfg.N_CLASES)]
    print("%-38s %.4f   %s" % (nombre, eb, " ".join("%9.3f" % v for v in rec)))
    return eb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--logits-ent", required=True, type=Path)
    ap.add_argument("--logits-val", required=True, type=Path)
    ap.add_argument("--rasgos", type=Path, default=Path("rasgos"))
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    args = ap.parse_args()

    todas = list(csv.DictReader((args.datos / args.manifiesto).open(encoding="utf-8")))

    def parte(nombre: str, npz: Path):
        filas = [r for r in todas if r["particion"] == nombre]
        y = np.array([C.index(r["clase_modelo"]) for r in filas])
        grupo = np.array([r["pista"] for r in filas])
        d = np.load(npz, allow_pickle=True)
        lg = np.asarray(d["logits"], np.float64)
        if not np.array_equal(np.asarray(d["verdadera"]), y):
            raise SystemExit("El orden de %s no es el del manifiesto." % npz)
        de = np.load(args.rasgos / ("descriptores_%s.npy" % nombre))
        if len(de) != len(y):
            raise SystemExit("Descriptores de %s: %d filas contra %d."
                             % (nombre, len(de), len(y)))
        return lg, de, y, grupo

    lg_e, de_e, y_e, _ = parte("entrenamiento", args.logits_ent)
    lg_v, de_v, y_v, gr_v = parte("validacion", args.logits_val)
    print("entrenamiento %d | validacion %d | %d descriptores"
          % (len(y_e), len(y_v), de_e.shape[1]))
    print("%d canciones en validacion\n" % len(set(gr_v)))

    cab = "%-38s %7s   %s" % ("", "balanc.", " ".join("%9s" % c[:9] for c in C))
    print(cab); print("-" * len(cab))

    informe("red CNN6 sola", y_v, lg_v.argmax(1))
    m = nuevo_modelo().fit(de_e, y_e)
    informe("descriptores solos", y_v, m.predict(de_v))

    # (a) ajuste sobre entrenamiento
    Xe, Xv = np.hstack([lg_e, de_e]), np.hstack([lg_v, de_v])
    m = nuevo_modelo().fit(Xe, y_e)
    pred_a = m.predict(Xv)
    eb_a = informe("(a) fusion ajustada en entrenamiento", y_v, pred_a)

    # (b) ajuste por validacion cruzada agrupada, dentro de validacion
    pred_b = np.zeros(len(y_v), int)
    for tr, te in GroupKFold(n_splits=5).split(Xv, y_v, gr_v):
        pred_b[te] = nuevo_modelo().fit(Xv[tr], y_v[tr]).predict(Xv[te])
    eb_b = informe("(b) fusion por VC agrupada en validacion", y_v, pred_b)

    print("-" * len(cab))
    mejor, pred = ("(a)", pred_a) if eb_a >= eb_b else ("(b)", pred_b)
    print("diferencia (a) menos (b): %+.4f" % (eb_a - eb_b))
    print("retenida: %s" % mejor)

    print("\nmatriz de confusion de la variante retenida (filas = verdadera):")
    mc = confusion_matrix(y_v, pred, labels=range(cfg.N_CLASES))
    print("%-13s %s" % ("", " ".join("%9s" % c[:9] for c in C)))
    for i, c in enumerate(C):
        print("%-13s %s" % (c, " ".join("%9d" % v for v in mc[i])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
