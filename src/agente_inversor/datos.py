"""Datos de precios y particion cronologica.

Contenido:
  - cargar()    : cierres diarios de GOOGL (el conjunto que trae gym-anytrading)
  - Tramo       : indices [inicio, fin) de un tramo dentro de la serie
  - particion() : corte cronologico en entrenamiento, validacion y prueba
  - cargar_extendida() : cierres de GOOGL de Yahoo Finance posteriores al conjunto
  - tramo()     : la serie y los indices de cualquier tramo, incluido el extendido
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TRAMOS = ("entrenamiento", "validacion", "prueba")

# Segunda prueba fuera de muestra: desde el dia siguiente al ultimo del conjunto de
# gym-anytrading hasta el 31 de diciembre de 2025 (scripts/descargar_datos.py).
CSV_EXTENDIDO = Path(__file__).resolve().parents[2] / "datos" / "GOOGL_yahoo_2018-07_2025-12.csv"
PRIMERA_DECISION_EXTENDIDA = "2018-08-29"  # ultimo dia del conjunto original


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


def cargar_extendida() -> pd.Series:
    """Cierre diario ajustado de GOOGL de Yahoo Finance, 2018-07-02 a 2025-12-31.

    En los 42 dias que comparte con `cargar()` el cociente entre las dos series es
    constante (la division de acciones de 2022 ya viene ajustada), asi que los
    retornos coinciden y el tramo continua la serie original sin salto.
    """
    serie = pd.read_csv(CSV_EXTENDIDO, index_col="Date", parse_dates=True)["Close"]
    return serie.astype("float64")


def tramo(nombre: str, ventana: int) -> tuple[pd.Series, Tramo]:
    """La serie de precios y el tramo `nombre` dentro de ella.

    `prueba_extendida` decide por primera vez el 29 de agosto de 2018, el ultimo dia
    del conjunto original, y cobra su primer retorno el 30: no comparte ningun
    retorno con los tramos de `particion()`.
    """
    if nombre == "prueba_extendida":
        serie = cargar_extendida()
        inicio = serie.index.get_loc(pd.Timestamp(PRIMERA_DECISION_EXTENDIDA))
        return serie, Tramo(nombre, inicio, len(serie))
    serie = cargar()
    return serie, particion(len(serie), ventana)[nombre]
