"""Simulacion de politicas, metricas y baselines.

Contenido:
  - simular()   : una pasada completa de una politica sobre un entorno
  - metricas()  : retorno acumulado, maxima caida, Sharpe, exposicion y rotacion
  - baselines() : comprar y mantener, 50 % fijo, en efectivo y agente aleatorio

Las metricas se calculan siempre sobre el capital real (neto de costos), nunca
sobre la recompensa moldeada con la que se entreno el agente.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agente_inversor import envs

Politica = Callable[[np.ndarray], int]


def simular(env, politica: Politica) -> dict[str, np.ndarray]:
    """Capital (empieza en 1.0) y exposicion elegida en cada dia."""
    obs, _ = env.reset()
    capital, exposicion = [1.0], []
    fin = False
    while not fin:
        obs, _, terminado, truncado, info = env.step(politica(obs))
        capital.append(info["capital"])
        exposicion.append(info["exposicion"])
        fin = terminado or truncado
    return {"capital": np.array(capital), "exposicion": np.array(exposicion)}


def metricas(sim: dict[str, np.ndarray]) -> dict[str, float]:
    capital, exposicion = sim["capital"], sim["exposicion"]
    r = np.diff(np.log(capital))
    pico = np.maximum.accumulate(capital)
    cambios = np.abs(np.diff(exposicion, prepend=0.0))
    return {
        "retorno_pct": float((capital[-1] - 1.0) * 100.0),
        "max_caida_pct": float((1.0 - capital / pico).max() * 100.0),
        "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)),
        "exposicion_media": float(exposicion.mean()),
        "rotacion": float(cambios.sum()),  # veces que se movio el capital completo
        "operaciones": int((cambios > 0).sum()),
    }


def politica_modelo(modelo) -> Politica:
    return lambda obs: int(modelo.predict(obs, deterministic=True)[0])


def baselines(tramo: str, entorno: dict, n_aleatorios: int = 100) -> dict[str, dict]:
    """Metricas de cada baseline; el aleatorio es la media de `n_aleatorios` semillas."""
    env = envs.make(tramo, ventana=entorno["ventana"], costo=entorno["costo"])
    fijos = {"comprar_y_mantener": 2, "mitad_fija": 1, "efectivo": 0}
    res = {k: metricas(simular(env, lambda _obs, a=a: a)) for k, a in fijos.items()}

    corridas = []
    for semilla in range(n_aleatorios):
        rng = np.random.default_rng(semilla)
        corridas.append(metricas(simular(env, lambda _obs, rng=rng: int(rng.integers(3)))))
    res["aleatorio"] = {k: float(np.mean([c[k] for c in corridas])) for k in corridas[0]}
    res["aleatorio"]["retorno_pct_std"] = float(np.std([c["retorno_pct"] for c in corridas]))
    return res
