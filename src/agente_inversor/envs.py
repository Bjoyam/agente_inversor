"""Entorno de inversion en un activo, compatible con Gymnasium.

Contenido:
  - EXPOSICIONES     : fraccion del capital invertida por cada accion
  - RECOMPENSAS      : las funciones de recompensa disponibles
  - EntornoInversion : el entorno
  - make()           : construye el entorno de un tramo (entrenamiento, validacion, prueba)

Formulacion:
  Estado   : 6 indicadores del precio calculados con los ultimos `ventana` dias,
             mas la exposicion actual del portafolio.
  Acciones : Discrete(3): tener invertido el 0 %, el 50 % o el 100 % del capital.
  Costo    : `costo` por cada unidad de capital que cambia de manos.
  Tiempo   : en el dia t se observa hasta el cierre de t, se decide, y se cobra el
             retorno de t a t+1. Ningun indicador usa precios posteriores a t.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from agente_inversor import datos

EXPOSICIONES = np.array([0.0, 0.5, 1.0])
ESCALA = 100.0  # las recompensas van en puntos porcentuales diarios


def _log(r_cartera: float, r_activo: float) -> float:
    """Retorno logaritmico de la cartera, ya neto del costo."""
    return r_cartera


def _diferencial(r_cartera: float, r_activo: float) -> float:
    """Exceso sobre comprar y mantener: estar 100 % invertido sin operar vale 0."""
    return r_cartera - r_activo


def _aversion(r_cartera: float, r_activo: float) -> float:
    """Las perdidas pesan el doble que las ganancias."""
    return r_cartera if r_cartera >= 0 else 2.0 * r_cartera


RECOMPENSAS = {"log": _log, "diferencial": _diferencial, "aversion": _aversion}


class EntornoInversion(gym.Env):
    """Recorre precios[inicio:fin]. Con `episodio`, cada episodio es una ventana
    aleatoria de esa longitud (entrenamiento); sin el, una pasada completa (evaluacion).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        precios: np.ndarray,
        inicio: int,
        fin: int,
        *,
        ventana: int = 20,
        costo: float = 0.001,
        recompensa: str = "log",
        episodio: int | None = None,
    ) -> None:
        super().__init__()
        if inicio < ventana:
            raise ValueError("inicio debe dejar `ventana` dias de historia")
        self.precios = np.asarray(precios, dtype=np.float64)
        self.inicio, self.fin = inicio, fin
        self.ventana, self.costo, self.episodio = ventana, costo, episodio
        self._recompensa = RECOMPENSAS[recompensa]
        # retornos[t] = log(p[t] / p[t-1]): conocido al cierre del dia t
        self.retornos = np.diff(np.log(self.precios), prepend=np.log(self.precios[0]))

        self.action_space = spaces.Discrete(len(EXPOSICIONES))
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        t, v = self.t, self.ventana
        r = self.retornos[t - v + 1 : t + 1]
        p = self.precios[t - v + 1 : t + 1]
        vol = r.std() + 1e-8
        vol_ventana = vol * np.sqrt(v)  # volatilidad acumulada en la ventana
        return np.array(
            [
                r[-1] / vol,  # ultimo retorno
                r[-5:].sum() / (vol * np.sqrt(5)),  # momento a 5 dias
                r.sum() / vol_ventana,  # momento de la ventana
                (p[-5:].mean() / p.mean() - 1.0) / vol_ventana,  # cruce de medias
                vol * np.sqrt(252),  # volatilidad anualizada
                (p[-1] / p.max() - 1.0) / vol_ventana,  # distancia al maximo
                self.exposicion,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.episodio is None:
            self.t, self.t_fin = self.inicio, self.fin - 1
        else:
            self.t = int(self.np_random.integers(self.inicio, self.fin - self.episodio))
            self.t_fin = self.t + self.episodio
        self.exposicion = 0.0
        self.capital = 1.0
        return self._obs(), {}

    def step(self, action):
        nueva = float(EXPOSICIONES[int(action)])
        costo = abs(nueva - self.exposicion) * self.costo
        self.exposicion = nueva

        self.t += 1
        r_activo = self.retornos[self.t]
        factor = (1.0 - costo) * (1.0 + nueva * (np.exp(r_activo) - 1.0))
        self.capital *= factor

        recompensa = ESCALA * self._recompensa(float(np.log(factor)), float(r_activo))
        # El final de los datos no es un estado terminal del problema: es un corte.
        truncado = self.t >= self.t_fin
        info = {"capital": self.capital, "exposicion": nueva}
        return self._obs(), recompensa, False, truncado, info


def make(
    tramo: str,
    *,
    ventana: int = 20,
    costo: float = 0.001,
    recompensa: str = "log",
    episodio: int | None = None,
) -> Monitor:
    """El entorno de un tramo. `episodio` solo tiene sentido en entrenamiento."""
    precios = datos.cargar().to_numpy()
    t = datos.particion(len(precios), ventana)[tramo]
    env = EntornoInversion(
        precios, t.inicio, t.fin, ventana=ventana, costo=costo,
        recompensa=recompensa, episodio=episodio,
    )
    return Monitor(env)
