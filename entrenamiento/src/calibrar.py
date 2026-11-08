"""
Calibracion del umbral de decision por clase.

La red decide con `argmax` sobre los logits, lo que supone que el coste de
equivocarse es el mismo en las cinco clases. No lo es: la metrica de la SRS
es la exactitud BALANCEADA, que pesa igual a las cinco aunque `sin_defecto`
sea el 30 % de los datos. El diagnostico del intento 02 muestra el efecto:
el 74,8 % del material limpio se marca como defectuoso.

Este guion busca un sesgo por clase `b` y decide con `argmax(logits + b)`.
Es un parametro de decision, no de aprendizaje: no toca los pesos, no hace
falta GPU y corre en segundos.

El ajuste se hace por ascenso por coordenadas sobre una rejilla, que basta
porque son cinco numeros y la superficie es escalonada.

ADVERTENCIA METODOLOGICA. El sesgo se ajusta sobre VALIDACION, igual que
cualquier otro hiperparametro. Sobre el conjunto de prueba se APLICA el
sesgo ya fijado; volver a ajustarlo alli invalidaria la estimacion.

Ejecutar:
    PYTHONPATH=.. python -m src.calibrar \
        --predicciones "C:/.../predicciones_validacion.npz" \
        --salida       informes/sesgo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nucleo import config as cfg


def exactitud_balanceada(verdadera: np.ndarray, predicha: np.ndarray) -> float:
    rec = []
    for c in range(cfg.N_CLASES):
        m = verdadera == c
        if m.any():
            rec.append(float((predicha[m] == c).mean()))
    return float(np.mean(rec))


def recalls(verdadera: np.ndarray, predicha: np.ndarray) -> list[float]:
    return [float((predicha[verdadera == c] == c).mean()) if (verdadera == c).any()
            else float("nan") for c in range(cfg.N_CLASES)]


def ajustar(logits: np.ndarray, verdadera: np.ndarray,
            rango: float, paso: float, pasadas: int) -> tuple[np.ndarray, float]:
    """Ascenso por coordenadas sobre el sesgo por clase."""
    sesgo = np.zeros(cfg.N_CLASES)
    mejor = exactitud_balanceada(verdadera, (logits + sesgo).argmax(1))
    rejilla = np.arange(-rango, rango + paso / 2, paso)

    for pasada in range(1, pasadas + 1):
        movio = False
        for c in range(cfg.N_CLASES):
            original = sesgo[c]
            candidato, valor = original, mejor
            for v in rejilla:
                sesgo[c] = v
                eb = exactitud_balanceada(verdadera, (logits + sesgo).argmax(1))
                if eb > valor + 1e-9:
                    candidato, valor = v, eb
            sesgo[c] = candidato
            if valor > mejor + 1e-9:
                mejor, movio = valor, True
        print("   pasada %d: exactitud balanceada %.4f" % (pasada, mejor))
        if not movio:
            break

    # El sesgo solo importa por sus diferencias: restar la media lo deja
    # centrado sin cambiar ninguna decision.
    return sesgo - sesgo.mean(), mejor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predicciones", required=True, type=Path,
                    help="npz con 'logits' y 'verdadera'")
    ap.add_argument("--salida", type=Path, default=Path("informes/sesgo.json"))
    ap.add_argument("--rango", type=float, default=4.0)
    ap.add_argument("--paso", type=float, default=0.1)
    ap.add_argument("--pasadas", type=int, default=6)
    ap.add_argument("--aplicar", type=Path, default=None,
                    help="sesgo ya ajustado: lo aplica en vez de ajustar uno nuevo")
    args = ap.parse_args()

    d = np.load(args.predicciones, allow_pickle=True)
    if "logits" not in d:
        print("El archivo no trae 'logits', solo la decision ya tomada.")
        print("Hay que reexportar guardando la salida cruda de la red.")
        return 1
    logits = np.asarray(d["logits"], dtype=np.float64)
    verdadera = np.asarray(d["verdadera"])
    print("%d segmentos, %d clases (época %s, partición %s)\n"
          % (logits.shape[0], logits.shape[1], d.get("epoca"), d.get("particion")))

    antes = logits.argmax(1)
    eb_antes = exactitud_balanceada(verdadera, antes)

    if args.aplicar:
        sesgo = np.array(json.loads(args.aplicar.read_text(encoding="utf-8"))["sesgo"])
        print("sesgo aplicado desde %s (no se reajusta)" % args.aplicar)
        eb_despues = exactitud_balanceada(verdadera, (logits + sesgo).argmax(1))
    else:
        print("ajustando el sesgo por ascenso por coordenadas:")
        sesgo, eb_despues = ajustar(logits, verdadera, args.rango, args.paso, args.pasadas)

    despues = (logits + sesgo).argmax(1)

    r_antes, r_despues = recalls(verdadera, antes), recalls(verdadera, despues)
    print("\n%-13s %7s %9s %9s %8s" % ("clase", "n", "antes", "después", "sesgo"))
    for c, nombre in enumerate(cfg.CLASES):
        n = int((verdadera == c).sum())
        marca = "" if r_despues[c] >= 0.80 else " <- incumple"
        print("%-13s %7d %9.3f %9.3f %+8.2f%s"
              % (nombre, n, r_antes[c], r_despues[c], sesgo[c], marca))
    print("%-13s %7d %9.3f %9.3f  %+.4f"
          % ("BALANCEADA", len(verdadera), eb_antes, eb_despues, eb_despues - eb_antes))

    limpio = verdadera == 0
    print("\nmaterial limpio marcado como defectuoso: %.1f %% -> %.1f %%"
          % (100 * (antes[limpio] != 0).mean(), 100 * (despues[limpio] != 0).mean()))

    if not args.aplicar:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
        args.salida.write_text(json.dumps({
            "clases": list(cfg.CLASES),
            "sesgo": [round(float(v), 4) for v in sesgo],
            "ajustado_sobre": str(d.get("particion")),
            "epoca": int(d.get("epoca", -1)),
            "exactitud_balanceada_sin_sesgo": round(eb_antes, 4),
            "exactitud_balanceada_con_sesgo": round(eb_despues, 4),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nsesgo guardado: %s" % args.salida)
        print("Sobre prueba, aplicarlo con --aplicar; no volver a ajustarlo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
