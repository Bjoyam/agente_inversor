"""Interfaz de linea de comandos: `agente-inversor`.

Contenido:
  - cmd_*()         : una funcion por subcomando
  - _build_parser() : el parser de argparse
  - main()          : punto de entrada

Aqui solo hay lectura de argumentos y formato de salida; el comportamiento vive en
envs, evaluacion y experiments.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml

from agente_inversor import config, datos, envs, evaluacion, experiments
from agente_inversor.envs import RECOMPENSAS

COLUMNAS = ["retorno_pct", "max_caida_pct", "sharpe", "exposicion_media", "rotacion"]


def _cfg(args: argparse.Namespace) -> config.Config:
    cfg = config.load(args.config)
    if getattr(args, "recompensa", None):
        cfg.entorno["recompensa"] = args.recompensa
    if getattr(args, "semillas", None):
        cfg.semillas = args.semillas
    if getattr(args, "sufijo", None):
        cfg.sufijo = args.sufijo
    return cfg


def cmd_inspeccionar(args: argparse.Namespace) -> None:
    serie = datos.cargar()
    tramos = datos.particion(len(serie), args.ventana)
    print(f"Activo: GOOGL, cierre diario, {len(serie)} dias\n")
    for t in tramos.values():
        # el primer retorno cobrado es el del dia inicio + 1
        print(
            f"  {t.nombre:<14} {serie.index[t.inicio + 1].date()} a "
            f"{serie.index[t.fin - 1].date()}  ({t.fin - t.inicio - 1} dias)"
        )

    env = envs.make("entrenamiento", ventana=args.ventana)
    print(f"\nEstado   : {env.observation_space}")
    print(f"Acciones : {env.action_space} -> exposicion {envs.EXPOSICIONES.tolist()}\n")
    obs, _ = env.reset(seed=0)
    print(f"  estado inicial: {np.array2string(obs, precision=3)}")
    for paso in range(1, args.pasos + 1):
        accion = env.action_space.sample()
        obs, r, _, _, info = env.step(accion)
        print(
            f"  paso {paso} | accion={accion} | recompensa={r:+.3f} | "
            f"capital={info['capital']:.4f}\n"
            f"           estado -> {np.array2string(obs, precision=3)}"
        )


def cmd_entrenar(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    for semilla in cfg.semillas:
        res = experiments.entrenar(cfg, semilla)
        print(
            f"{cfg.etiqueta} semilla {semilla}: "
            f"entrenamiento {res['entrenamiento_retorno_pct']:+.1f} %, "
            f"validacion {res['validacion_retorno_pct']:+.1f} %"
        )


def cmd_tune(args: argparse.Namespace) -> None:
    estudio = experiments.tune(_cfg(args), n_trials=args.trials)
    print(f"\nMejor mediana en validacion: {estudio.best_value:+.2f} %")
    print("Hiperparametros, para pegar bajo 'hiper:' en el YAML:\n")
    print(yaml.safe_dump(estudio.best_params, sort_keys=False))
    estudio.trials_dataframe().to_csv(experiments.RESULTADOS / "tune.csv", index=False)


def cmd_resumen(_args: argparse.Namespace) -> None:
    """Junta las corridas en validacion.csv y muestra la mediana por configuracion."""
    archivos = sorted((experiments.RESULTADOS / "corridas").glob("*.json"))
    df = pd.DataFrame([json.loads(a.read_text()) for a in archivos])
    df.to_csv(experiments.RESULTADOS / "validacion.csv", index=False)
    cols = [f"entrenamiento_{COLUMNAS[0]}"] + [f"validacion_{c}" for c in COLUMNAS]
    resumen = df.groupby("etiqueta")[cols].median()
    resumen.insert(0, "semillas", df.groupby("etiqueta").size())
    resumen = resumen.sort_values("validacion_retorno_pct", ascending=False)
    resumen.to_csv(experiments.RESULTADOS / "validacion_resumen.csv")
    print(resumen.round(2).to_string())


def cmd_baselines(args: argparse.Namespace) -> None:
    res = evaluacion.baselines(args.tramo, config.ENTORNO)
    print(pd.DataFrame(res).T[COLUMNAS].round(2).to_string())


def cmd_prueba(args: argparse.Namespace) -> None:
    res = experiments.prueba(_cfg(args))
    agente = pd.DataFrame(res["agente"]).set_index("semilla")[COLUMNAS]
    print(f"Agente {res['etiqueta']} en el tramo de prueba:\n")
    print(agente.round(2).to_string())
    print(f"\n  mediana: {agente['retorno_pct'].median():+.2f} %\n\nBaselines:\n")
    print(pd.DataFrame(res["baselines"]).T[COLUMNAS].round(2).to_string())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agente-inversor", description="Agente inversor con Stable-Baselines3"
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def add_cfg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="YAML, p. ej. configs/ppo.yaml")
        p.add_argument("--recompensa", choices=tuple(RECOMPENSAS), help="Cambia la del YAML")
        p.add_argument("--semillas", type=int, nargs="+", help="Cambia las del YAML")
        p.add_argument("--sufijo", help="Se agrega a la etiqueta, p. ej. _tuned")

    p = sub.add_parser("inspeccionar", help="Datos, tramos, espacios y transiciones de muestra")
    p.add_argument("--pasos", type=int, default=3)
    p.add_argument("--ventana", type=int, default=20)
    p.set_defaults(func=cmd_inspeccionar)

    p = sub.add_parser("entrenar", help="Entrena y evalua en validacion (queda en MLflow)")
    add_cfg(p)
    p.set_defaults(func=cmd_entrenar)

    p = sub.add_parser("tune", help="Busca hiperparametros con Optuna sobre validacion")
    add_cfg(p)
    p.add_argument("--trials", type=int, default=None)
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("resumen", help="Tabla de validacion de todas las corridas")
    p.set_defaults(func=cmd_resumen)

    p = sub.add_parser("baselines", help="Baselines en un tramo")
    p.add_argument("--tramo", choices=datos.TRAMOS[:2], default="validacion")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("prueba", help="Evaluacion final en el tramo de prueba")
    add_cfg(p)
    p.set_defaults(func=cmd_prueba)

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    args.func(args)
