"""
Evaluacion del modelo sobre el conjunto de prueba, contrastada contra
RNF-01, y comparacion con las lineas base que exige RNF-02.

El conjunto de prueba se abre UNA sola vez, cuando la configuracion ya esta
fijada mediante el de validacion. Este guion no debe usarse para decidir
nada del modelo.

Uso:
    python -m src.evaluar --datos "C:/.../dataset_defectos" \\
                          --modelo modelos/mejor.pt
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)
from torch.utils.data import DataLoader

from nucleo import config as cfg
from nucleo.modelo import Cnn6
from src.datos import ConjuntoPrecalculado, ConjuntoSegmentos

UMBRAL_RNF01_EB = 0.90
UMBRAL_RNF01_EXH = 0.80
LINEA_BASE_DESCRIPTORES = 0.408     # medido en la verificacion del conjunto


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", type=Path, default=None)
    ap.add_argument("--rasgos", type=Path, default=None,
                    help="carpeta con los log-mel precalculados (recomendado)")
    ap.add_argument("--modelo", required=True, type=Path)
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--limite-por-clase", type=int, default=None)
    ap.add_argument("--informe", type=Path, default=Path("informes/evaluacion.md"))
    args = ap.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    if args.rasgos:
        prueba = ConjuntoPrecalculado(args.rasgos, "prueba",
                                      args.limite_por_clase, cfg.SEMILLA)
    elif args.datos:
        prueba = ConjuntoSegmentos(args.datos, "prueba", args.manifiesto,
                                   args.limite_por_clase, cfg.SEMILLA)
    else:
        raise SystemExit("Hay que indicar --rasgos (recomendado) o --datos.")
    cargador = DataLoader(prueba, batch_size=args.lote, num_workers=args.hilos)

    punto = torch.load(args.modelo, map_location=dispositivo, weights_only=False)
    modelo = Cnn6(cfg.N_CLASES).to(dispositivo)
    modelo.load_state_dict(punto["modelo"])
    modelo.eval()

    reales, predichas = [], []
    with torch.no_grad():
        for x, y in cargador:
            predichas.extend(modelo(x.to(dispositivo)).argmax(1).cpu().numpy())
            reales.extend(y.numpy())
    reales, predichas = np.array(reales), np.array(predichas)

    eb = balanced_accuracy_score(reales, predichas)
    cm = confusion_matrix(reales, predichas, labels=range(cfg.N_CLASES))
    rep = classification_report(reales, predichas, labels=range(cfg.N_CLASES),
                                target_names=list(cfg.CLASES),
                                output_dict=True, zero_division=0)

    # --- lineas base ----------------------------------------------------
    cuenta = Counter(reales.tolist())
    mayoritaria = cuenta.most_common(1)[0][0]
    eb_mayoritaria = balanced_accuracy_score(reales, np.full_like(reales, mayoritaria))

    exh = {c: rep[c]["recall"] for c in cfg.CLASES}
    cumple_eb = eb >= UMBRAL_RNF01_EB
    cumple_exh = all(v >= UMBRAL_RNF01_EXH for v in exh.values())

    L = ["# Evaluación del modelo sobre el conjunto de prueba\n",
         "Segmentos evaluados: %d" % len(reales),
         "Modelo: `%s` (época %s)\n" % (args.modelo, punto.get("epoca", "?")),
         "## RNF-01\n",
         "| Métrica | Valor | Umbral | ¿Cumple? |", "|---|---|---|---|",
         "| Exactitud balanceada | %.4f | %.2f | %s |"
         % (eb, UMBRAL_RNF01_EB, "sí" if cumple_eb else "NO")]
    for c in cfg.CLASES:
        L.append("| Exhaustividad, %s | %.4f | %.2f | %s |"
                 % (c, exh[c], UMBRAL_RNF01_EXH,
                    "sí" if exh[c] >= UMBRAL_RNF01_EXH else "NO"))
    L.append("\n**RNF-01: %s**\n" % ("CUMPLE" if (cumple_eb and cumple_exh) else "NO CUMPLE"))

    L.append("## RNF-02: comparación con las líneas base\n")
    L.append("| Línea base | Exactitud balanceada |")
    L.append("|---|---|")
    L.append("| Clase mayoritaria | %.4f |" % eb_mayoritaria)
    L.append("| Descriptores elementales | %.4f |" % LINEA_BASE_DESCRIPTORES)
    L.append("| **Modelo** | **%.4f** |" % eb)
    L.append("\n**RNF-02: %s**\n"
             % ("CUMPLE" if eb > max(eb_mayoritaria, LINEA_BASE_DESCRIPTORES) else "NO CUMPLE"))

    L.append("## Métricas por clase\n")
    L.append("| Clase | Precisión | Exhaustividad | F1 | Apoyo |")
    L.append("|---|---|---|---|---|")
    for c in cfg.CLASES:
        d = rep[c]
        L.append("| %s | %.3f | %.3f | %.3f | %d |"
                 % (c, d["precision"], d["recall"], d["f1-score"], d["support"]))

    L.append("\n## Matriz de confusión\n")
    L.append("Filas: clase verdadera. Columnas: clase predicha.\n")
    L.append("| | " + " | ".join(cfg.CLASES) + " |")
    L.append("|---" * (cfg.N_CLASES + 1) + "|")
    for i, c in enumerate(cfg.CLASES):
        L.append("| **%s** | " % c + " | ".join(str(v) for v in cm[i]) + " |")

    args.informe.parent.mkdir(parents=True, exist_ok=True)
    args.informe.write_text("\n".join(L), encoding="utf-8")
    args.informe.with_suffix(".json").write_text(
        json.dumps({"exactitud_balanceada": float(eb),
                    "por_clase": {c: rep[c] for c in cfg.CLASES},
                    "matriz_confusion": cm.tolist(),
                    "linea_base_mayoritaria": float(eb_mayoritaria)},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(L))
    print("\ninforme: %s" % args.informe)


if __name__ == "__main__":
    main()
