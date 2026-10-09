"""Datos de precios y particion cronologica.

Contenido:
  - cargar()    : cierres diarios de GOOGL (el conjunto que trae gym-anytrading)
  - Tramo       : indices [inicio, fin) de un tramo dentro de la serie
  - particion() : corte cronologico en entrenamiento, validacion y prueba
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TRAMOS = ("entrenamiento", "validacion", "prueba")


def cargar() -> pd.Series:
    """Cierre diario de GOOGL, 2009-05-22 a 2018-08-29 (2.335 dias habiles)."""
    from gym_anytrading.datasets import STOCKS_GOOGL

    serie = STOCKS_GOOGL["Close"].astype("float64")
    serie.index = pd.to_datetime(serie.index)
    return serie


@dataclass(frozen=True)
class Tramo:
    nombre: str
    inicio: int  # primer dia en que se decide
    fin: int  # exclusivo; el ultimo retorno cobrado es el del dia fin - 1


def particion(
    n: int, ventana: int, frac_entreno: float = 0.70, frac_val: float = 0.15
) -> dict[str, Tramo]:
    """Corte por fechas, nunca aleatorio: mezclar el tiempo filtra el futuro al pasado.

    Los tramos comparten frontera: la decision del ultimo dia de un tramo no se toma
    ahi, se toma como primera decision del tramo siguiente. Asi ningun retorno se
    cobra en dos tramos. Validacion y prueba usan los `ventana` dias anteriores solo
    para calcular sus primeros indicadores, lo cual es informacion del pasado.
    """
    a = int(n * frac_entreno)
    b = int(n * (frac_entreno + frac_val))
    return {
        "entrenamiento": Tramo("entrenamiento", ventana, a),
        "validacion": Tramo("validacion", a - 1, b),
        "prueba": Tramo("prueba", b - 1, n),
    }
