"""Figuras del informe, a partir de los archivos de resultados/.

  uv run python scripts/graficar.py

  - validacion.png : retorno en validacion de cada semilla, por configuracion
  - prueba.png     : curva de capital del agente y de comprar y mantener en prueba
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agente_inversor import config, datos, evaluacion

RES = Path("resultados")


def validacion() -> None:
    df = pd.read_csv(RES / "validacion.csv")
    base = evaluacion.baselines("validacion", config.ENTORNO)
    orden = df.groupby("etiqueta")["validacion_retorno_pct"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, etiqueta in enumerate(orden):
        y = df.loc[df["etiqueta"] == etiqueta, "validacion_retorno_pct"]
        ax.scatter([i] * len(y), y, color="tab:blue", alpha=0.7, zorder=3)
        ax.hlines(y.median(), i - 0.25, i + 0.25, color="black", zorder=4)
    estilos = {"comprar_y_mantener": "tab:green", "aleatorio": "tab:red", "efectivo": "gray"}
    for nombre, color in estilos.items():
        ax.axhline(base[nombre]["retorno_pct"], color=color, ls="--", label=nombre)
    ax.set_xticks(range(len(orden)), orden, rotation=20)
    ax.set_ylabel("Retorno acumulado en validacion (%)")
    ax.set_title("Cinco semillas por configuracion; la raya negra es la mediana")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RES / "validacion.png", dpi=150)


def prueba() -> None:
    res = json.loads((RES / "prueba.json").read_text())
    serie = datos.cargar()
    t = datos.particion(len(serie), config.ENTORNO["ventana"])["prueba"]
    fechas = serie.index[t.inicio : t.fin]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre, curva in res["curvas"].items():
        if nombre == "comprar_y_mantener":
            ax.plot(fechas, curva, color="tab:green", lw=2, label="comprar y mantener")
        else:
            ax.plot(fechas, curva, color="tab:blue", alpha=0.6, label=None)
    ax.plot([], [], color="tab:blue", label=f"agente {res['etiqueta']} (5 semillas)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_ylabel("Capital (inicial = 1)")
    ax.set_title("Tramo de prueba")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RES / "prueba.png", dpi=150)


if __name__ == "__main__":
    validacion()
    if (RES / "prueba.json").exists():
        prueba()
