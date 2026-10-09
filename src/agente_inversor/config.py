"""Configuracion de experimentos: un YAML por algoritmo, en configs/.

Contenido:
  - Config    : una configuracion ya leida
  - load()    : lee un YAML
  - sample()  : toma hiperparametros del espacio de busqueda (Optuna)

Adaptado de rl_games_sb (Emilio Munoz), con los campos del entorno de inversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from agente_inversor.envs import RECOMPENSAS
from agente_inversor.registry import ALGO_CHOICES

if TYPE_CHECKING:  # solo para la anotacion: importar optuna es lento
    import optuna

ENTORNO = {"ventana": 20, "costo": 0.001, "episodio": 252, "recompensa": "log"}


@dataclass
class Config:
    algo: str
    pasos: int
    experimento: str = "agente_inversor"
    semillas: list[int] = field(default_factory=lambda: [0])
    entorno: dict[str, Any] = field(default_factory=lambda: dict(ENTORNO))
    hiper: dict[str, Any] = field(default_factory=dict)
    tune: dict[str, Any] = field(default_factory=dict)
    sufijo: str = ""
    path: Path | None = None

    @property
    def etiqueta(self) -> str:
        """Nombre de la carpeta de modelos y de las corridas, p. ej. ppo_log."""
        return f"{self.algo}_{self.entorno['recompensa']}{self.sufijo}"


def load(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    faltan = {"algo", "pasos"} - raw.keys()
    if faltan:
        raise ValueError(f"{path}: faltan las claves {sorted(faltan)}")
    if raw["algo"] not in ALGO_CHOICES:
        raise ValueError(f"{path}: algo debe ser uno de {ALGO_CHOICES}")
    entorno = {**ENTORNO, **(raw.get("entorno") or {})}
    if entorno["recompensa"] not in RECOMPENSAS:
        raise ValueError(f"{path}: recompensa debe ser una de {tuple(RECOMPENSAS)}")

    return Config(
        algo=raw["algo"],
        pasos=int(raw["pasos"]),
        experimento=raw.get("experimento", "agente_inversor"),
        semillas=list(raw.get("semillas", [0])),
        entorno=entorno,
        hiper=raw.get("hiper") or {},
        tune=raw.get("tune") or {},
        path=path,
    )


def sample(trial: optuna.Trial, search_space: dict[str, dict]) -> dict[str, Any]:
    """Un valor por entrada del espacio de busqueda. Tipos: float, int, categorical."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            # float() porque YAML lee `1e-5` (sin punto) como texto
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=spec.get("log", False)
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"search_space.{name}: tipo desconocido {kind!r}")
    return params
