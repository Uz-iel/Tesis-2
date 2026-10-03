# Datos del proyecto

El CSV utilizado en el análisis se conserva en una ubicación privada y no se publica en GitHub.

## Piloto exploratorio

La tabla contiene 150 pacientes únicos: 89 casos, 61 controles y 107 características radiómicas, con una fila por paciente.

- patient_id: identificador pseudonimizado, excluido de los predictores.
- seg_id: identificador de segmentación, cuando está disponible, excluido de los predictores.
- y: etiqueta binaria, 1 para caso y 0 para control.
- Características radiómicas numéricas utilizadas para desarrollar los modelos.

## Acceso al CSV

El archivo privado se denomina radiomica_piloto_limpia_150.csv.

En Colab se debe montar el Drive autorizado y configurar DATA con la ubicación real del archivo. Abrir el notebook público no concede acceso a los datos privados.

Para ejecutar el script localmente, sustituir las rutas del siguiente ejemplo:

```powershell
python run_experiment.py --data "C:\ruta_privada\radiomica_piloto_limpia_150.csv" --out "C:\resultados_privados\corrida_nueva"
```

La carpeta de salida debe ser nueva o estar vacía.

## Confidencialidad

No publicar tomografías, nombres, DNI, identificadores, filas de pacientes ni predicciones individuales. Solo se comparten resultados agregados y gráficos.

La anonimización de las imágenes se realiza antes de construir la tabla; este notebook y el script de modelamiento no anonimizan archivos DICOM.

## Alcance

Los resultados corresponden al piloto exploratorio, separado de la cohorte definitiva en fase venosa portal.

La fecha propuesta para completar la recolección y el procesamiento es el 5 de diciembre de 2026, pendiente de aceptación.