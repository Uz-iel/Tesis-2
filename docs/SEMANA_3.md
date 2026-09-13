# Semana 3 — Validación y consistencias técnicas

El notebook `notebooks/Tesis_2_Radiomica_Semana2_3.ipynb` incorpora una sección específica de Semana 3 con:

1. Auditoría del dataset y controles de data leakage.
2. Consistencia entre semillas (5 seeds × 5 folds).
3. Learning curve para diagnóstico de overfitting/underfitting.
4. Boxplots de métricas por fold.
5. Explicabilidad SHAP del mejor modelo.
6. Fairness básico cuando existan variables sensibles apropiadas.
7. Resumen automático para reporte y peer review.

### Punto crítico

`seg_id` es un identificador de segmentación y no necesariamente un identificador único de paciente. Si un paciente puede aportar más de una fila, debe agregarse `patient_id` y utilizar una estrategia agrupada de partición para impedir leakage entre entrenamiento y validación.
