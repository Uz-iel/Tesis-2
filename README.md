# Tesis II — Validación de modelo radiómico predictivo

Repositorio académico del proyecto de Tesis II. Contiene el pipeline reproducible de análisis radiómico, resultados agregados y las ampliaciones de validación técnica trabajadas en las Semanas 2 y 3.

## Estado del repositorio

**Semana 2 — Ejecución experimental y métricas**

- Semilla reproducible (`set_seed`).
- Pipeline de preprocesamiento y modelado.
- Validación cruzada estratificada.
- Cinco folds/corridas por modelo.
- Métricas agregadas con promedio y desviación estándar.
- Resultados y documentación de ejecución.

**Semana 3 — Validación y consistencias técnicas**

El notebook incluye bloques para auditoría de data leakage, consistencia entre semillas, learning curve, boxplots por fold, SHAP y fairness cuando existan variables sensibles.

## Estructura

```text
Tesis-2/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── Tesis_2_Radiomica_Semana2_3.ipynb
├── src/
│   └── run_experiment.py
├── results/
│   ├── metricas_5_corridas.xlsx
│   ├── tabla_resultados_final_modelos.xlsx
│   └── figures/
├── models/
└── docs/
    ├── SEMANA_2.md
    ├── SEMANA_3.md
    └── SUBIR_A_GITHUB_DESDE_COLAB.md
```

## Datos y privacidad

El dataset **no está incluido en este repositorio público**. El archivo de trabajo se mantiene fuera de GitHub y se proporciona al script mediante una ruta local o de Google Drive. No deben subirse identificadores de pacientes, datasets crudos, tokens ni credenciales.

## Reproducibilidad

Crear un entorno e instalar dependencias:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Ejecutar:

```bash
python src/run_experiment.py \
  --data /ruta/radiomica_pancreas_dataset_semana3.csv \
  --output results/run
```

En Google Colab puede utilizarse directamente el notebook ubicado en `notebooks/`.

## Resultados agregados disponibles

La evaluación guardada contiene **9 modelos con 5 folds por modelo**. En la tabla actual, `SVM_RBF_balanced` presenta PR-AUC promedio de aproximadamente **0.965 ± 0.031**. Estos resultados corresponden al análisis piloto almacenado y deben interpretarse junto con el protocolo metodológico y la validación posterior.

## Nota sobre leakage

La imputación, escalado, selección de variables y filtro de correlación se implementan dentro del pipeline evaluado por fold. Si un mismo paciente puede aparecer en más de una fila, debe utilizarse un identificador de paciente y validación agrupada (`GroupKFold`/`StratifiedGroupKFold`) para evitar fuga entre particiones.
