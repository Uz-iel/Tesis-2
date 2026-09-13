# Semana 2 — Ejecución experimental y métricas

## Evidencia incluida

- Notebook reproducible en `notebooks/`.
- Script principal en `src/run_experiment.py`.
- `requirements.txt` con versiones del entorno.
- Cinco folds/corridas por cada uno de los 9 modelos evaluados.
- Tabla de métricas: `results/metricas_5_corridas.xlsx`.
- Comparación agregada: `results/tabla_resultados_final_modelos.xlsx`.

## Pipeline

Carga y limpieza de datos, control de duplicados, imputación, escalado robusto, selección de características, filtro de correlación dentro del pipeline, manejo del desbalance, entrenamiento, validación cruzada estratificada, selección de umbral y cálculo de métricas.

## Métricas principales

PR-AUC, ROC-AUC/C-index, balanced accuracy, sensibilidad, especificidad, precisión, F1, F2 y MCC. Dado el desbalance de clases, accuracy no se utiliza como métrica principal.

## Privacidad

El dataset se mantiene fuera del repositorio público.
