"""
Exporta la salida cruda del modelo sobre una particion.

Guarda los LOGITS y no la decision ya tomada. La diferencia importa: la
calibracion por clase y el suavizado temporal operan sobre probabilidades,
y a partir de un `argmax` ya no se pueden recuperar. Exportar la decision
obliga a volver a correr la red cada vez que se quiere probar una etapa de
posprocesado distinta.

El orden de salida es el del manifiesto, sin barajar, porque `consolidar`
necesita saber que segmentos son contiguos en la misma pista.

Como comprobacion, se recalcula la exactitud balanceada y se compara con
la que el checkpoint trae guardada. Si no coinciden, algo cambio entre el
entrenamiento y esta corrida (otra particion, otro manifiesto, otros
rasgos) y el archivo no sirve.

Ejecutar:
    PYTHONPATH=.. python -m src.exportar \
        --modelo    modelos/mejor.pt \
        --rasgos    /content/rasgos \
        --particion validacion \
        --salida    predicciones_validacion.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from nucleo import config as cfg
from nucleo.modelo import crear
from src.datos import ConjuntoPrecalculado, ConjuntoSegmentos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modelo", required=True, type=Path)
    ap.add_argument("--rasgos", type=Path, default=None,
                    help="carpeta con los log-mel precalculados (recomendado)")
    ap.add_argument("--datos", type=Path, default=None,
                    help="carpeta del conjunto; calcula el log-mel al vuelo")
    ap.add_argument("--particion", default="validacion",
                    choices=("entrenamiento", "validacion", "prueba"))
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--arquitectura", default=None, choices=("cnn6", "cnn10", "cnn14"),
                    help="por omision se lee del propio checkpoint")
    args = ap.parse_args()

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    salida = args.salida or Path("predicciones_%s.npz" % args.particion)

    if args.rasgos:
        conjunto = ConjuntoPrecalculado(args.rasgos, args.particion, None, cfg.SEMILLA)
    elif args.datos:
        conjunto = ConjuntoSegmentos(args.datos, args.particion, args.manifiesto,
                                     None, cfg.SEMILLA)
    else:
        raise SystemExit("Hay que indicar --rasgos (recomendado) o --datos.")

    punto = torch.load(args.modelo, map_location="cpu", weights_only=False)
    if list(punto.get("clases", cfg.CLASES)) != list(cfg.CLASES):
        raise SystemExit("El checkpoint trae otro orden de clases: %s" % punto["clases"])

    # El checkpoint registra con que arquitectura se entreno, de modo que
    # no hace falta recordarlo al exportar.
    arq = args.arquitectura or punto.get("arquitectura", "cnn6")
    modelo = crear(arq, cfg.N_CLASES)
    modelo.load_state_dict(punto["modelo"])
    modelo.to(dispositivo).eval()

    epoca = int(punto.get("epoca", -1))
    eb_guardada = punto.get("exactitud_balanceada_validacion")
    print("modelo: %s, época %d, exactitud balanceada guardada %s"
          % (arq, epoca, "%.4f" % eb_guardada if eb_guardada is not None else "(sin registrar)"))
    print("%d segmentos de la partición '%s' | dispositivo %s"
          % (len(conjunto), args.particion, dispositivo), flush=True)

    # shuffle=False: el orden debe ser el del manifiesto para que
    # `consolidar` pueda reconstruir los tramos contiguos.
    cargador = DataLoader(conjunto, batch_size=args.lote, shuffle=False,
                          num_workers=args.hilos, pin_memory=False)

    trozos, reales = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(cargador, 1):
            trozos.append(modelo(x.to(dispositivo)).cpu().numpy())
            reales.append(y.numpy())
            if i % 50 == 0:
                print("   %d / %d lotes" % (i, len(cargador)), flush=True)

    logits = np.concatenate(trozos).astype(np.float32)
    verdadera = np.concatenate(reales)

    eb = balanced_accuracy_score(verdadera, logits.argmax(1))
    print("\nexactitud balanceada recalculada: %.4f" % eb)
    # La exactitud guardada es la de VALIDACION: solo tiene sentido
    # exigir que coincida cuando se exporta esa misma particion.
    if eb_guardada is not None and args.particion == "validacion":
        dif = abs(eb - float(eb_guardada))
        print("diferencia con la guardada:       %.4f  %s"
              % (dif, "coincide" if dif < 1e-4 else "NO COINCIDE"))
        if dif >= 1e-4:
            print("\nLa salida no reproduce el entrenamiento. Revisar que la\n"
                  "partición, el manifiesto y los rasgos sean los mismos.")
            return 1

    np.savez_compressed(salida, logits=logits, verdadera=verdadera,
                        clases=np.array(cfg.CLASES), epoca=epoca,
                        particion=args.particion,
                        exactitud_balanceada=float(eb))
    print("\narchivo: %s (%.1f MB)" % (salida, salida.stat().st_size / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
