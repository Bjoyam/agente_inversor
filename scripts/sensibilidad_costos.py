"""Sensibilidad del resultado al costo de transaccion, sin reentrenar ni reajustar.

  uv run python scripts/sensibilidad_costos.py

El estado del agente no incluye el costo, de modo que las posiciones que eligio cada
semilla no cambian si el costo cambia. Este script toma las exposiciones guardadas en
resultados/prueba.json y resultados/prueba_extendida.json y recalcula el capital con la
misma contabilidad del entorno para varios costos. No elige nada: solo mide cuanto del
resultado depende del supuesto de 0,1 %.

Produce resultados/sensibilidad_costos.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agente_inversor import config, datos

RES = Path("resultados")
COSTOS = (0.0, 0.0005, 0.001, 0.002, 0.003, 0.005)


def capital_final(precios: np.ndarray, exposiciones: np.ndarray, costo: float) -> float:
    """Misma contabilidad que EntornoInversion.step: costo sobre lo que se mueve."""
    capital, previa = 1.0, 0.0
    for x, r in zip(exposiciones, precios[1:] / precios[:-1], strict=True):
        capital *= (1.0 - abs(x - previa) * costo) * (1.0 + x * (r - 1.0))
        previa = x
    return capital


def tabla() -> pd.DataFrame:
    filas = []
    for tramo in ("prueba", "prueba_extendida"):
        res = json.loads((RES / f"{tramo}.json").read_text())
        serie, t = datos.tramo(tramo, config.ENTORNO["ventana"])
        p = serie.to_numpy()[t.inicio : t.fin]
        expos = [np.array(res["curvas"][f"exposicion_semilla_{k}"]) for k in range(5)]
        # control: con el costo del proyecto se reproduce exactamente la curva reportada
        reportado = res["curvas"]["semilla_0"][-1]
        assert abs(capital_final(p, expos[0], config.ENTORNO["costo"]) - reportado) < 1e-9
        for c in COSTOS:
            agente = [100 * (capital_final(p, x, c) - 1) for x in expos]
            bh = 100 * (capital_final(p, np.ones(len(p) - 1), c) - 1)
            filas.append({
                "tramo": tramo, "costo_pct": c * 100,
                **{f"semilla_{k}": a for k, a in enumerate(agente)},
                "mediana_agente": float(np.median(agente)), "comprar_y_mantener": bh,
                "semillas_sobre_bh": int(sum(a > bh for a in agente)),
            })
    return pd.DataFrame(filas)


if __name__ == "__main__":
    df = tabla()
    df.to_csv(RES / "sensibilidad_costos.csv", index=False)
    print(df.round(1).to_string(index=False))
