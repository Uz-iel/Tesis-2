[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Uz-iel/Tesis-2/blob/main/Script_de_radiomica_.ipynb)

# Piloto de radiómica pancreática

Autor: Uzziell Gariazzo Anarcaya. Actualización: 3 de octubre de 2026.

## Proyecto

Validación de un modelo predictivo multivariable a partir de las características radiómicas significativas de estudios tomográficos de pacientes con diagnóstico de adenocarcinoma ductal pancreático atendidos en el Hospital Nacional Guillermo Almenara Irigoyen entre enero de 2020 y junio de 2024.

El análisis principal utiliza radiómica y machine learning. La cohorte definitiva utilizará fase venosa portal; estos resultados corresponden al piloto histórico y no acreditan por sí mismos la fase de adquisición.

## Archivos

- `Script_de_radiomica_.ipynb`: notebook suministrado por el investigador, con las salidas de entrenamiento guardadas. Las dos figuras se regeneraron desde sus tablas agregadas; no se repitió el entrenamiento para ese ajuste visual.
- `run_experiment.py`: código de las celdas computacionales de ese notebook, adaptado para recibir rutas locales mediante argumentos. No carga las métricas de `results` para entrenar.
- `results/`: tablas agregadas extraídas de las salidas HTML del notebook; precisión de seis decimales. La reproducción genera CSV nuevos a precisión completa. No se fabricaron los hiperparámetros por fold: su CSV se debe copiar de la carpeta original de esta misma corrida o generar al repetir el análisis.
- `figures/`: rendimiento e IC95%, y estabilidad entre repeticiones.
- `requirements.txt`: dependencias sin fijar versiones, porque el notebook recibido no registró las versiones exactas de su ejecución. No se garantiza identidad numérica entre entornos distintos.

## Cohorte y método

150 filas y 150 pacientes únicos, 89 casos y 61 controles, 107 características numéricas. Se excluyen `patient_id`, `seg_id` y `y` de los predictores. El CSV privado permanece fuera del repositorio.

Tres modelos: regresión logística Elastic Net, SVM-RBF y XGBoost. Validación cruzada estratificada anidada: cinco folds externos y cinco internos, repetida con semillas 42, 123, 456, 789 y 2026. Pipeline: imputación mediana, varianza nula, escalado robusto, filtro Spearman >0,90 y selección L1 con candidatos 5, 10 y 20 variables. Tuning por ROC-AUC, seguido de calibración sigmoidal con tres folds dentro del entrenamiento externo. Umbral de clasificación 0,5.

El código conservado usa `SVC(probability=True)` además de calibración explícita. XGBoost calcula su peso de clases con toda la tabla; en este piloto el resultado es 1 porque existen 89 casos y 61 controles. Ese comportamiento debe revisarse dentro del entrenamiento antes de ampliar la cohorte. El selector exige que existan al menos tantas variables remanentes como su candidato; una tabla distinta puede requerir ajustar ese límite. Se declaran estas características para no atribuir a esta corrida cambios que no se ejecutaron.

## Resultados exploratorios de esta corrida

Estimaciones sobre probabilidades OOF promediadas entre cinco repeticiones. PR-AUC corresponde a average precision.

| Modelo | ROC-AUC (IC95%) | PR-AUC/AP (IC95%) |
|---|---|---|
| ElasticNet_Logistic | 0.907 (0.852–0.954) | 0.925 (0.874–0.968) |
| SVM_RBF | 0.916 (0.860–0.962) | 0.913 (0.840–0.974) |
| XGBoost | 0.900 (0.832–0.954) | 0.894 (0.816–0.964) |

SVM-RBF: media de PR-AUC entre repeticiones **0,914 ± 0,007**. La desviación estándar describe estabilidad; no es un IC95%. Fracción positiva: 89/150 = 0,593.

Los IC95% son percentiles de 2000 bootstrap estratificados por clase a nivel de paciente sobre OOF promedio, sin reentrenar. Expresan incertidumbre condicional. Las repeticiones comparten pacientes y no son cohortes independientes.

El PR-AUC 0,965 pertenece al análisis histórico y no es el resultado vigente. La diferencia no demuestra que el control de fuga sea su causa; se requiere una comparación controlada cambiando un componente por vez. Esta corrida tampoco debe mezclarse con otra ejecución que reporta SVM PR-AUC 0,910.

## Repetir en Windows

Desde la carpeta del repositorio, con Python instalado:

```powershell
python -m pip install -r requirements.txt
python run_experiment.py --data "C:\ruta_privada\radiomica_piloto_limpia_150.csv" --out "C:\resultados_privados\corrida_nueva"
```

Sustituir las dos rutas por ubicaciones reales. La carpeta de salida debe estar nueva o vacía. La corrida puede tardar; guardar los CSV, los hiperparámetros y las figuras generadas juntos. Para trabajar con el notebook local, abrirlo en Jupyter y sustituir la celda de montaje de Colab por rutas locales; ejecutar todas las celdas. En Colab, ajustar `DATA` y `OUT` a Drive y ejecutar todas las celdas.

Registrar las versiones del entorno usado:

```powershell
python -m pip freeze > versiones_entorno.txt
```

## Datos y autorizaciones

No se publican tomografías, identificadores, filas de pacientes ni predicciones individuales. El script recibe una tabla previamente pseudonimizada; no anonimiza imágenes DICOM.

Según declaración del investigador, la versión presentada previamente en San Marcos obtuvo aprobación ética y permisos del Almenara. Tras cambios metodológicos y de universidad, se consultó y se indicó volver a presentar el proyecto, para lo cual se requiere primero su aprobación por Ricardo Palma. La aprobación previa no se presenta aquí como verificación de aprobación de todas las modificaciones actuales.

IPRIAS es una iniciativa fundada por el investigador y el Dr. Aníbal Apaza. Los gráficos de resultados se obtienen mediante código, sin sello institucional. Se declaró apoyo de IA en programación, organización y redacción.

## Próxima etapa

Fecha propuesta para completar recolección y procesamiento: **5 de diciembre de 2026**, pendiente de aceptación. El piloto es exploratorio; la cohorte confirmatoria, modelo bloqueado y cálculo de poder deben formalizarse antes del análisis confirmatorio. Las conclusiones finales dependerán de sus resultados.
