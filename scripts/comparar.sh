#!/usr/bin/env bash
# Entrena la rejilla completa: 2 algoritmos x 3 recompensas x 5 semillas.
# Las corridas quedan en MLflow y en resultados/corridas/.
# Uso: scripts/comparar.sh [procesos en paralelo, por defecto 2]
set -euo pipefail
N="${1:-2}"
# La base de MLflow se crea antes: dos procesos creandola a la vez chocan.
uv run --no-sync python -c "import mlflow; mlflow.set_experiment('agente_inversor')"
for algo in ppo dqn; do
  for recompensa in log diferencial aversion; do
    for semilla in 0 1 2 3 4; do
      # una corrida ya hecha no se repite
      [ -f "resultados/corridas/${algo}_${recompensa}_semilla_${semilla}.json" ] && continue
      echo "--config configs/$algo.yaml --recompensa $recompensa --semillas $semilla"
    done
  done
done | xargs -P "$N" -L 1 uv run --no-sync agente-inversor entrenar
uv run --no-sync agente-inversor resumen
