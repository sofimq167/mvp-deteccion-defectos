"""
Ajuste fino de CNN6 sobre el conjunto de defectos.

Registra semilla, particion, hiperparametros y metricas por epoca, que es
lo que exige RNF-09 y sin lo cual los resultados no serian reproducibles.

Uso:
    python -m src.entrenar --datos "C:/.../datos/dataset_defectos" \\
                           --pesos modelos/Cnn6_mAP=0.343.pth \\
                           --epocas 10 --lote 64
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader

from nucleo import config as cfg
from nucleo.modelo import cargar_preentrenado, crear
from src.datos import ConjuntoPrecalculado, ConjuntoSegmentos


def fijar_semilla(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def enmascarar(x: torch.Tensor, max_t: int, max_f: int,
               n_t: int = 2, n_f: int = 2) -> torch.Tensor:
    """SpecAugment: borra franjas de tiempo y de frecuencia del log-mel.

    Obliga a la red a no depender de una banda concreta ni de un instante
    concreto. Es el aumento con el que se entrenaron las PANN originales.

    El relleno es la media de cada espectro, no cero: el log-mel esta en
    decibelios, donde el cero es un valor ALTO, asi que rellenar con cero
    insertaria energia en vez de quitarla.
    """
    relleno = x.mean(dim=(1, 2), keepdim=True)
    lote, tramas, mel = x.shape
    for eje, cuantas, ancho_max, largo in ((1, n_t, max_t, tramas),
                                           (2, n_f, max_f, mel)):
        for _ in range(cuantas):
            w = torch.randint(0, ancho_max + 1, (lote,), device=x.device)
            tope = (largo - w).clamp(min=1).float()
            ini = (torch.rand(lote, device=x.device) * tope).long()
            pos = torch.arange(largo, device=x.device)[None, :]
            m = (pos >= ini[:, None]) & (pos < (ini + w)[:, None])
            m = m[:, :, None] if eje == 1 else m[:, None, :]
            x = torch.where(m, relleno.expand_as(x), x)
    return x


def mezclar(x: torch.Tensor, y: torch.Tensor, alfa: float):
    """Mixup: combina pares de ejemplos y sus etiquetas.

    Crea ejemplos intermedios entre clases, que es justo donde este
    conjunto se confunde: entre limpio, zumbido y saturacion.
    """
    lam = float(np.random.beta(alfa, alfa))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def evaluar(modelo, cargador, dispositivo):
    modelo.eval()
    reales, predichas = [], []
    with torch.no_grad():
        for x, y in cargador:
            logits = modelo(x.to(dispositivo))
            predichas.extend(logits.argmax(1).cpu().numpy())
            reales.extend(y.numpy())
    return balanced_accuracy_score(reales, predichas), np.array(reales), np.array(predichas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", type=Path, default=None,
                    help="carpeta del conjunto; lee el audio en cada epoca")
    ap.add_argument("--rasgos", type=Path, default=None,
                    help="carpeta con los log-mel precalculados (recomendado)")
    ap.add_argument("--pesos", type=Path, default=None,
                    help="pesos de AudioSet; sin ellos la red parte de cero")
    ap.add_argument("--salida", type=Path, default=Path("modelos"))
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--tasa", type=float, default=1e-3,
                    help="tasa de la cabeza de clasificacion")
    ap.add_argument("--tasa-cuerpo", type=float, default=None,
                    help="tasa del cuerpo preentrenado; por omision, la decima parte de --tasa")
    ap.add_argument("--paciencia", type=int, default=6,
                    help="epocas sin mejora antes de detener")
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--limite-por-clase", type=int, default=None,
                    help="submuestrea cada clase; util para una prueba rapida")
    ap.add_argument("--semilla", type=int, default=cfg.SEMILLA)
    ap.add_argument("--arquitectura", default="cnn6", choices=("cnn6", "cnn10", "cnn14"),
                    help="respaldo preentrenado; cnn14 usa el mismo log-mel")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="alfa de la Beta para mixup; 0 lo desactiva (0,2 a 0,4)")
    ap.add_argument("--mascara-tiempo", type=int, default=0,
                    help="ancho maximo de las franjas de tiempo borradas (SpecAugment)")
    ap.add_argument("--mascara-mel", type=int, default=0,
                    help="ancho maximo de las bandas mel borradas (SpecAugment)")
    args = ap.parse_args()

    fijar_semilla(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    args.salida.mkdir(parents=True, exist_ok=True)

    if args.rasgos:
        entren = ConjuntoPrecalculado(args.rasgos, "entrenamiento",
                                      args.limite_por_clase, args.semilla)
        valid = ConjuntoPrecalculado(args.rasgos, "validacion",
                                     args.limite_por_clase, args.semilla)
    elif args.datos:
        entren = ConjuntoSegmentos(args.datos, "entrenamiento", args.manifiesto,
                                   args.limite_por_clase, args.semilla)
        valid = ConjuntoSegmentos(args.datos, "validacion", args.manifiesto,
                                  args.limite_por_clase, args.semilla)
    else:
        raise SystemExit("Hay que indicar --rasgos (recomendado) o --datos.")
    print("segmentos: entrenamiento %d | validacion %d | dispositivo %s"
          % (len(entren), len(valid), dispositivo), flush=True)

    comun = dict(batch_size=args.lote, num_workers=args.hilos, pin_memory=False)
    c_entren = DataLoader(entren, shuffle=True, drop_last=True, **comun)
    c_valid = DataLoader(valid, shuffle=False, **comun)

    modelo = crear(args.arquitectura, cfg.N_CLASES).to(dispositivo)
    resumen_pesos = None
    if args.pesos and args.pesos.is_file():
        resumen_pesos = cargar_preentrenado(modelo, str(args.pesos))
        modelo.to(dispositivo)
        print("pesos preentrenados: %d tensores cargados, %d sin inicializar"
              % (resumen_pesos["cargados"], len(resumen_pesos["sin_inicializar"])))
    else:
        print("AVISO: sin pesos preentrenados. El resultado no es comparable "
              "con el antecedente: se pierde la transferencia de aprendizaje.")

    # El desbalance se compensa con pesos en la perdida y no con remuestreo,
    # para no alterar el numero de segmentos vistos por epoca.
    criterio = nn.CrossEntropyLoss(weight=entren.pesos_por_clase().to(dispositivo))

    # El cuerpo llega preentrenado sobre AudioSet y la cabeza parte de cero:
    # aplicarles la misma tasa destruiria lo aprendido mientras la cabeza
    # todavia no sabe nada. De ahi la tasa diferenciada.
    tasa_cuerpo = args.tasa_cuerpo if args.tasa_cuerpo is not None else args.tasa / 10
    cabeza = [p for n, p in modelo.named_parameters() if n.startswith("cabeza")]
    cuerpo = [p for n, p in modelo.named_parameters() if not n.startswith("cabeza")]
    optim = torch.optim.AdamW(
        [{"params": cuerpo, "lr": tasa_cuerpo},
         {"params": cabeza, "lr": args.tasa}], weight_decay=1e-4)
    print("tasas: cuerpo %.1e | cabeza %.1e" % (tasa_cuerpo, args.tasa))
    plan = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epocas)

    registro = {
        "arquitectura": args.arquitectura,
        "semilla": args.semilla, "epocas": args.epocas, "lote": args.lote,
        "tasa_cabeza": args.tasa, "tasa_cuerpo": tasa_cuerpo,
        "paciencia": args.paciencia, "manifiesto": args.manifiesto,
        "mixup": args.mixup, "mascara_tiempo": args.mascara_tiempo,
        "mascara_mel": args.mascara_mel,
        "origen": "rasgos precalculados" if args.rasgos else "audio",
        "limite_por_clase": args.limite_por_clase,
        "n_entrenamiento": len(entren), "n_validacion": len(valid),
        "pesos_preentrenados": str(args.pesos) if args.pesos else None,
        "resumen_pesos": resumen_pesos, "historia": [],
    }

    mejor, sin_mejora = -1.0, 0
    for epoca in range(1, args.epocas + 1):
        modelo.train()
        t0, perdida_total, n = time.time(), 0.0, 0
        for x, y in c_entren:
            x, y = x.to(dispositivo), y.to(dispositivo)
            optim.zero_grad()
            if args.mascara_tiempo or args.mascara_mel:
                x = enmascarar(x, args.mascara_tiempo, args.mascara_mel)
            if args.mixup > 0:
                # La perdida se interpola en vez de pasar etiquetas blandas,
                # para conservar los pesos por clase del criterio.
                x, ya, yb, lam = mezclar(x, y, args.mixup)
                salida = modelo(x)
                perdida = lam * criterio(salida, ya) + (1 - lam) * criterio(salida, yb)
            else:
                perdida = criterio(modelo(x), y)
            perdida.backward()
            optim.step()
            perdida_total += perdida.item() * len(y)
            n += len(y)
        plan.step()

        eb, _, _ = evaluar(modelo, c_valid, dispositivo)
        fila = {"epoca": epoca, "perdida": perdida_total / max(n, 1),
                "exactitud_balanceada_validacion": float(eb),
                "segundos": round(time.time() - t0, 1)}
        registro["historia"].append(fila)
        print("epoca %2d | perdida %.4f | exactitud balanceada %.4f | %.0fs"
              % (epoca, fila["perdida"], eb, fila["segundos"]), flush=True)

        torch.save({"modelo": modelo.state_dict(), "epoca": epoca,
                    "exactitud_balanceada_validacion": float(eb),
                    "clases": list(cfg.CLASES),
                    "arquitectura": args.arquitectura},
                   args.salida / "ultimo.pt")
        if eb > mejor:
            mejor, sin_mejora = eb, 0
            torch.save({"modelo": modelo.state_dict(), "epoca": epoca,
                        "exactitud_balanceada_validacion": float(eb),
                        "clases": list(cfg.CLASES),
                    "arquitectura": args.arquitectura},
                       args.salida / "mejor.pt")
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print("sin mejora en %d epocas: se detiene." % args.paciencia)
                registro["detenido_en"] = epoca
                break

    registro["mejor_exactitud_balanceada_validacion"] = mejor
    (args.salida / "registro_entrenamiento.json").write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmejor exactitud balanceada en validacion: %.4f" % mejor)
    print("artefacto: %s" % (args.salida / "mejor.pt"))


if __name__ == "__main__":
    main()
