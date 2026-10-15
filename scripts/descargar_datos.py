"""Descarga los cierres diarios de GOOGL posteriores al conjunto de gym-anytrading.

  uv run --no-project --with yfinance python scripts/descargar_datos.py

Se corre una sola vez y el CSV queda versionado en datos/, para que la evaluacion
extendida sea reproducible sin depender de la red. El cierre es el ajustado por
divisiones y dividendos (auto_adjust): los indicadores del entorno solo usan
cocientes de precios, asi que lo que importa es que la serie sea continua.
"""
from pathlib import Path

import yfinance as yf

INICIO = "2018-07-01"  # unas semanas antes del 2018-08-29 para los 20 dias de historia
FIN = "2026-01-01"  # exclusivo: el ultimo dia es el 31 de diciembre de 2025
SALIDA = Path("datos/GOOGL_yahoo_2018-07_2025-12.csv")

df = yf.download("GOOGL", start=INICIO, end=FIN, auto_adjust=True, progress=False)
if hasattr(df.columns, "levels"):  # versiones recientes devuelven columnas (campo, ticker)
    df.columns = df.columns.get_level_values(0)
serie = df["Close"].dropna().round(6)
serie.index.name = "Date"
SALIDA.parent.mkdir(exist_ok=True)
serie.to_csv(SALIDA, header=["Close"])
print(f"{len(serie)} dias, {serie.index[0].date()} a {serie.index[-1].date()} -> {SALIDA}")
