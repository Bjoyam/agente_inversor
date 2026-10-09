"""Entrenamiento, busqueda de hiperparametros y evaluacion final.

Contenido:
  - entrenar() : entrena una semilla, la evalua en validacion y la registra en MLflow
  - tune()     : busqueda con Optuna, puntuada en validacion
  - prueba()   : evaluacion en el tramo de prueba, junto a los baselines

El tramo de prueba solo lo toca `prueba()`. La seleccion de recompensa, algoritmo
e hiperparametros se hace con el tramo de validacion.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import optuna

from agente_inversor import config, envs, evaluacion, registry, tracking

RESULTADOS = Path("resultados")


def _env(cfg: config.Config, tramo: str):
    e = cfg.entorno
    if tramo == "entrenamiento_aleatorio":  # ventanas aleatorias, recompensa moldeada
        return envs.make("entrenamiento", **e)
    # evaluacion: una pasada completa; la recompensa no interviene en las metricas
    return envs.make(tramo, ventana=e["ventana"], costo=e["costo"])


def _evaluar(modelo, cfg: config.Config, tramo: str) -> dict:
    sim = evaluacion.simular(_env(cfg, tramo), evaluacion.politica_modelo(modelo))
    return evaluacion.metricas(sim)


def entrenar(cfg: config.Config, semilla: int, *, guardar: bool = True) -> dict:
    """Devuelve las metricas en entrenamiento y validacion de esta semilla."""
    env = _env(cfg, "entrenamiento_aleatorio")
    env.reset(seed=semilla)
    modelo = registry.create(cfg.algo, env, semilla, **cfg.hiper)

    mlflow.set_experiment(cfg.experimento)
    with mlflow.start_run(run_name=f"{cfg.etiqueta}-s{semilla}"):
        mlflow.set_tags({"algo": cfg.algo, "recompensa": cfg.entorno["recompensa"]})
        mlflow.log_params({
            "algo": cfg.algo, "pasos": cfg.pasos, "semilla": semilla,
            **{f"entorno.{k}": v for k, v in cfg.entorno.items()},
            **tracking.model_params(modelo),
        })
        tracking.attach(modelo, stdout=False)
        modelo.learn(total_timesteps=cfg.pasos)

        res = {
            "etiqueta": cfg.etiqueta, "algo": cfg.algo,
            "recompensa": cfg.entorno["recompensa"], "semilla": semilla,
        }
        for tramo in ("entrenamiento", "validacion"):
            m = _evaluar(modelo, cfg, tramo)
            mlflow.log_metrics({f"{tramo}/{k}": v for k, v in m.items()})
            res.update({f"{tramo}_{k}": v for k, v in m.items()})

        if guardar:
            ruta = registry.save_path(cfg.etiqueta, semilla)
            ruta.parent.mkdir(parents=True, exist_ok=True)
            modelo.save(ruta)
            corridas = RESULTADOS / "corridas"
            corridas.mkdir(parents=True, exist_ok=True)
            (corridas / f"{cfg.etiqueta}_semilla_{semilla}.json").write_text(
                json.dumps(res, indent=2)
            )
    return res


def tune(cfg: config.Config, n_trials: int | None = None) -> optuna.Study:
    """Maximiza la mediana del retorno en validacion sobre tune.semillas."""
    espacio = cfg.tune.get("search_space")
    if not espacio:
        raise ValueError(f"{cfg.path}: falta tune.search_space")
    n_trials = n_trials or int(cfg.tune.get("trials", 20))
    semillas = cfg.tune.get("semillas", [0])
    base = dict(cfg.hiper)

    def objetivo(trial: optuna.Trial) -> float:
        cfg_t = config.Config(
            algo=cfg.algo, pasos=int(cfg.tune.get("pasos", cfg.pasos)),
            experimento=f"{cfg.experimento}_tune", entorno=cfg.entorno,
            hiper={**base, **config.sample(trial, espacio)}, sufijo=f"_trial{trial.number}",
        )
        retornos = [
            entrenar(cfg_t, s, guardar=False)["validacion_retorno_pct"] for s in semillas
        ]
        return float(np.median(retornos))

    estudio = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    estudio.optimize(objetivo, n_trials=n_trials)
    return estudio


def prueba(cfg: config.Config) -> dict:
    """Evalua los modelos guardados de cfg en el tramo de prueba, con los baselines."""
    agente, curvas = [], {}
    for semilla in cfg.semillas:
        modelo = registry.load(cfg.algo, cfg.etiqueta, semilla)
        sim = evaluacion.simular(_env(cfg, "prueba"), evaluacion.politica_modelo(modelo))
        agente.append({"semilla": semilla, **evaluacion.metricas(sim)})
        curvas[f"semilla_{semilla}"] = sim["capital"].tolist()

    env = _env(cfg, "prueba")
    curvas["comprar_y_mantener"] = evaluacion.simular(env, lambda _o: 2)["capital"].tolist()
    res = {
        "etiqueta": cfg.etiqueta,
        "agente": agente,
        "baselines": evaluacion.baselines("prueba", cfg.entorno),
        "curvas": curvas,
    }
    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "prueba.json").write_text(json.dumps(res, indent=2))
    return res
