"""
Curva de desempeno frente a la severidad del defecto.

Es la comprobacion que la seccion 3.5 del documento promete y que ninguna
metrica agregada responde: ¿el modelo detecta un defecto leve, o solo los
severos? Si la exhaustividad se derrumba en la banda mas suave, el
conjunto incluye degradaciones por debajo del umbral de detectabilidad, y
eso es un hallazgo sobre el DISENO DEL CONJUNTO, no un fallo del modelo.

Cruza tres fuentes:

  manifiesto_v2.csv   que segmento es cada fila, y en que orden
  intervalos.json     la severidad pedida en cada intervalo degradado
  predicciones .npz   lo que el modelo predijo, exportado del entorno
                      de entrenamiento

El orden importa: `precalcular.py` recorre el manifiesto filtrando por
particion y conserva el orden del CSV, de modo que la fila i del
manifiesto corresponde al indice i de las predicciones. El guion lo
verifica antes de cruzar nada.

El clic reductivo no aparece en la curva: no tiene parametro de severidad
en decibelios, solo densidad y numero de clics. Su dosis se analiza por
`n_clics_segmento`, no aca.

Ejecutar:
    python -m src.curva_severidad \
        --datos        "C:/.../datos/dataset_defectos" \
        --predicciones "C:/.../predicciones_validacion.npz"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from nucleo import config as cfg

SR = 32_000

# campo de severidad de cada defecto y hacia donde crece la dificultad
SEVERIDAD = {
    "hum":          ("snr_db_objetivo", "menos dB = mas severo"),
    "hiss":         ("snr_db_objetivo", "menos dB = mas severo"),
    "distorsion":   ("sdr_db_objetivo", "menos dB = mas severo"),
    "clic_aditivo": ("csr_db_objetivo", "mas dB = mas severo"),
}


def severidad_del_segmento(reg, pista, defecto, ini, fin):
    """Severidad del intervalo que mas solapa con el segmento."""
    d = reg.get(pista, {}).get("defectos", {}).get(defecto)
    if not d:
        return None
    campo = SEVERIDAD[defecto][0]
    mejor, valor = 0, None
    for (a, b), par in zip(d["intervalos_muestras"], d["parametros"]):
        solape = min(fin, b) - max(ini, a)
        if solape > mejor and campo in par:
            mejor, valor = solape, par[campo]
    return valor


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datos", required=True, type=Path)
    ap.add_argument("--predicciones", required=True, type=Path)
    ap.add_argument("--manifiesto", default="manifiesto_v2.csv")
    ap.add_argument("--particion", default="validacion")
    ap.add_argument("--bandas", type=int, default=4,
                    help="cuantos tramos de severidad por defecto")
    ap.add_argument("--figura", type=Path, default=None,
                    help="si se indica, guarda la curva como imagen")
    args = ap.parse_args()

    d = np.load(args.predicciones, allow_pickle=True)
    # Acepta tanto la decision ya tomada como los logits crudos, que es
    # lo que exporta `src.exportar`.
    verdadera = d["verdadera"]
    predicha = (d["predicha"] if "predicha" in d
                else np.asarray(d["logits"]).argmax(1))
    print("predicciones: %d segmentos (época %s, partición %s)"
          % (len(verdadera), d["epoca"], d["particion"]))

    filas = [r for r in csv.DictReader(
        (args.datos / args.manifiesto).open(encoding="utf-8"))
        if r["particion"] == args.particion]

    if len(filas) != len(verdadera):
        print("\nDESAJUSTE: %d filas del manifiesto contra %d predicciones."
              % (len(filas), len(verdadera)))
        print("No se cruza nada: las predicciones y el manifiesto no son "
              "de la misma construcción del conjunto.")
        return 1

    # comprobacion fuerte: la etiqueta del manifiesto debe coincidir
    clases = list(cfg.CLASES)
    desde_csv = np.array([clases.index(r["clase_modelo"]) for r in filas])
    if not np.array_equal(desde_csv, verdadera):
        n = int((desde_csv != verdadera).sum())
        print("\nDESAJUSTE: %d etiquetas no coinciden con el manifiesto." % n)
        print("El orden de las predicciones no es el del manifiesto.")
        return 1
    print("orden verificado: las etiquetas coinciden con el manifiesto\n")

    reg = json.loads((args.datos / "intervalos.json").read_text(encoding="utf-8"))

    resultados = {}
    for defecto, (campo, sentido) in SEVERIDAD.items():
        clase_modelo = "clic" if defecto.startswith("clic") else defecto
        idx = clases.index(clase_modelo)

        sev, acierto = [], []
        for i, r in enumerate(filas):
            if r["clase_simulacion"] != defecto:
                continue
            v = severidad_del_segmento(
                reg, r["pista"], defecto,
                int(r["inicio_muestra"]), int(r["fin_muestra"]))
            if v is None:
                continue
            sev.append(v)
            acierto.append(predicha[i] == idx)

        if len(sev) < args.bandas * 10:
            print("%-14s sin datos suficientes (%d segmentos)" % (defecto, len(sev)))
            continue

        sev, acierto = np.array(sev, float), np.array(acierto, bool)
        cortes = np.quantile(sev, np.linspace(0, 1, args.bandas + 1))
        cortes[-1] += 1e-9

        print("%s  (%s, %s)" % (defecto.upper(), campo, sentido))
        print("   %-18s %7s %10s" % ("banda de severidad", "n", "exhaust."))
        puntos = []
        for k in range(args.bandas):
            m = (sev >= cortes[k]) & (sev < cortes[k + 1])
            if not m.any():
                continue
            r = float(acierto[m].mean())
            puntos.append((float(sev[m].mean()), r, int(m.sum())))
            print("   %6.1f a %6.1f dB %7d %9.3f"
                  % (cortes[k], cortes[k + 1], m.sum(), r))
        print("   %-18s %7d %9.3f\n" % ("todas", len(sev), acierto.mean()))
        resultados[defecto] = puntos

    if args.figura and resultados:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Dos paneles, porque el signo de la severidad se invierte entre
        # familias: en hum, hiss y distorsion menos dB es mas severo, y en
        # el clic mas dB es mas severo. En un solo eje la curva del clic se
        # leeria al reves que las demas.
        inversos = [d for d in resultados if "menos" in SEVERIDAD[d][1]]
        directos = [d for d in resultados if d not in inversos]

        fig, ejes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharey=True,
                                 gridspec_kw={"width_ratios": [1.9, 1]})
        nombre = lambda d: cfg.NOMBRE_LEGIBLE.get(
            "clic" if d.startswith("clic") else d, d)
        # color fijo por defecto: los dos paneles comparten paleta, de modo
        # que ningun color signifique dos cosas distintas en la misma figura
        COLOR = {"hum": "C0", "hiss": "C1", "distorsion": "C2",
                 "clic_aditivo": "C3"}

        for eje, grupo, etiqueta in [
                (ejes[0], inversos, "SNR o SDR pedida (dB)"),
                (ejes[1], directos, "CSR pedida (dB)")]:
            for defecto in grupo:
                puntos = resultados[defecto]
                eje.plot([p[0] for p in puntos], [p[1] for p in puntos],
                         marker="o", color=COLOR.get(defecto),
                         label=nombre(defecto))
            eje.axhline(0.80, ls="--", lw=1, color="0.5")
            eje.set_xlabel(etiqueta)
            eje.set_ylim(0, 1)
            eje.grid(alpha=.25)
            eje.legend(fontsize=8, loc="lower left")

        # Severo a la izquierda en el panel directo tambien no: se invierte
        # el eje de los inversos para que en AMBOS paneles la severidad
        # crezca hacia la derecha.
        ejes[0].invert_xaxis()
        for eje in ejes:
            eje.annotate("más severo", xy=(0.97, -0.17), xycoords="axes fraction",
                         ha="right", fontsize=8, color="0.4")
        ejes[0].set_ylabel("exhaustividad")
        ejes[0].text(ejes[0].get_xlim()[0], 0.82, " umbral RNF-01",
                     fontsize=8, color="0.4")
        args.figura.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figura, dpi=200, bbox_inches="tight")
        print("figura: %s" % args.figura)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
