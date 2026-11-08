"""
Evaluacion final sobre la particion de prueba.

Este guion se ejecuta una vez por apertura declarada del conjunto de
prueba. La primera apertura evaluo CNN6 y dio 0,6121. La segunda evalua
CNN10, arquitectura elegida sobre validacion despues de medir tambien
CNN14. El conjunto de prueba no participo en
ninguna decision: ni en la eleccion de la arquitectura, ni en la de los
hiperparametros del ajuste fino, ni en la del combinador, ni en el
descarte del suavizado temporal. Todas esas decisiones se tomaron sobre
validacion y aqui quedan congeladas.

En consecuencia, el combinador se ajusta sobre ENTRENAMIENTO y se aplica
a prueba sin reajustar nada. Reajustarlo aqui convertiria la estimacion
en otra medida de validacion y perderia su valor.

Ademas del alcance completo se reportan dos analisis de sensibilidad, que
no son recortes del alcance sino descomposiciones de la metrica:

  - sin el clic reductivo, procedimiento de simulacion cuya invalidez se
    midio ANTES de entrenar: sus discontinuidades son 1,01 veces las que
    la musica ya contiene
  - sin el zumbido, clase cuya senal se midio 11,7 dB por debajo de la
    musica en la banda que ambos ocupan

Ejecutar:
    PYTHONPATH=.. python -m src.evaluar_prueba \
        --datos         "C:/.../datos/dataset_defectos" \
        --logits-ent    "C:/.../predicciones_entrenamiento.npz" \
        --logits-prueba "C:/.../predicciones_prueba.npz"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from nucleo import config as cfg

C = list(cfg.CLASES)


def combinador(semilla: int = 0) -> HistGradientBoostingClassifier:
    """La misma configuracion que se fijo mirando validacion. No se toca."""
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                          class_weight="balanced",
                                          random_state=semilla)


def recalls(y: np.ndarray, pred: np.ndarray, clases) -> dict:
    return {C[c]: float((pred[y == c] == c).mean()) for c in clases}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--logits-ent", required=True, type=Path)
    ap.add_argument("--logits-prueba", required=True, type=Path)
    ap.add_argument("--rasgos", type=Path, default=Path("rasgos"))
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--red", default="red sola",
                    help="etiqueta de la arquitectura, solo para el informe")
    args = ap.parse_args()

    todas = list(csv.DictReader((args.datos / args.manifiesto).open(encoding="utf-8")))

    def parte(nombre: str, npz: Path):
        filas = [r for r in todas if r["particion"] == nombre]
        y = np.array([C.index(r["clase_modelo"]) for r in filas])
        sim = np.array([r["clase_simulacion"] for r in filas])
        d = np.load(npz, allow_pickle=True)
        lg = np.asarray(d["logits"], np.float64)
        if not np.array_equal(np.asarray(d["verdadera"]), y):
            raise SystemExit("El orden de %s no es el del manifiesto." % npz)
        de = np.load(args.rasgos / ("descriptores_%s.npy" % nombre))
        if len(de) != len(y):
            raise SystemExit("Descriptores de %s: %d filas contra %d."
                             % (nombre, len(de), len(y)))
        return np.hstack([lg, de]), lg, y, sim

    Xe, _, y_e, sim_e = parte("entrenamiento", args.logits_ent)
    Xp, lg_p, y_p, sim_p = parte("prueba", args.logits_prueba)
    print("entrenamiento %d segmentos | prueba %d segmentos\n" % (len(y_e), len(y_p)))

    ancho = 52
    print("%-*s %8s   %s" % (ancho, "ALCANCE COMPLETO, LAS CINCO CLASES", "balanc.",
                             " ".join("%9s" % c[:9] for c in C)))
    print("-" * (ancho + 60))

    eb_red = balanced_accuracy_score(y_p, lg_p.argmax(1))
    print("%-*s %8.4f   %s" % (ancho, args.red, eb_red,
          " ".join("%9.3f" % v for v in recalls(y_p, lg_p.argmax(1), range(5)).values())))

    m = combinador().fit(Xe[:, 5:], y_e)
    p = m.predict(Xp[:, 5:])
    print("%-*s %8.4f   %s" % (ancho, "descriptores acústicos solos",
          balanced_accuracy_score(y_p, p),
          " ".join("%9.3f" % v for v in recalls(y_p, p, range(5)).values())))

    m = combinador().fit(Xe, y_e)
    pred = m.predict(Xp)
    eb_fus = balanced_accuracy_score(y_p, pred)
    print("%-*s %8.4f   %s" % (ancho, "SISTEMA COMPLETO", eb_fus,
          " ".join("%9.3f" % v for v in recalls(y_p, pred, range(5)).values())))

    print("\nmatriz de confusión del sistema completo (filas = verdadera):")
    mc = confusion_matrix(y_p, pred, labels=range(cfg.N_CLASES))
    print("%-13s %s" % ("", " ".join("%9s" % c[:9] for c in C)))
    for i, c in enumerate(C):
        print("%-13s %s" % (c, " ".join("%9d" % v for v in mc[i])))
    limpio = y_p == 0
    print("\nmaterial limpio marcado como defectuoso: %.1f %%"
          % (100 * (pred[limpio] != 0).mean()))

    print("\n" + "=" * (ancho + 60))
    print("ANÁLISIS DE SENSIBILIDAD (no son recortes del alcance declarado)")
    print("=" * (ancho + 60))
    for nombre, quitar_clase, quitar_sim in [
            ("sin el clic reductivo", (), ("clic_reductivo",)),
            ("sin el zumbido", (1,), ()),
            ("sin el zumbido ni el clic reductivo", (1,), ("clic_reductivo",))]:
        keep = [c for c in range(cfg.N_CLASES) if c not in quitar_clase]
        me = np.isin(y_e, keep) & ~np.isin(sim_e, quitar_sim)
        mp = np.isin(y_p, keep) & ~np.isin(sim_p, quitar_sim)
        mm = combinador().fit(Xe[me], y_e[me])
        pp = mm.predict(Xp[mp])
        eb = balanced_accuracy_score(y_p[mp], pp)
        print("%-*s %8.4f   (%+.4f, %d segmentos)"
              % (ancho, nombre, eb, eb - eb_fus, mp.sum()))

    print("\nEl conjunto de prueba queda abierto. Toda cifra posterior que se")
    print("obtenga sobre él deja de ser una estimación independiente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
