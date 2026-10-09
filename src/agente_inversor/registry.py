"""Algoritmos disponibles y rutas de los modelos guardados.

Contenido:
  - SAVE_DIR     : carpeta de modelos
  - ALGOS        : clave -> clase de Stable-Baselines3
  - save_path()  : archivo de un modelo (etiqueta, semilla)
  - create()     : modelo nuevo, sin entrenar
  - load()       : modelo guardado
"""
from pathlib import Path

import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

SAVE_DIR = Path("saves")

ALGOS: dict[str, type[BaseAlgorithm]] = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
ALGO_CHOICES = tuple(ALGOS)

# Redes de dos capas de 64: un hilo es mas rapido y deja libre el resto para
# correr varias semillas en paralelo.
torch.set_num_threads(1)


def save_path(etiqueta: str, semilla: int) -> Path:
    return SAVE_DIR / etiqueta / f"semilla_{semilla}.zip"


def create(algo: str, env, semilla: int, **hiper) -> BaseAlgorithm:
    """Politica MLP sobre el vector de estado. Las redes son pequenas: CPU."""
    kwargs = {"verbose": 0, "device": "cpu", **hiper}
    return ALGOS[algo]("MlpPolicy", env, seed=semilla, **kwargs)


def load(algo: str, etiqueta: str, semilla: int) -> BaseAlgorithm:
    return ALGOS[algo].load(save_path(etiqueta, semilla), device="cpu")
