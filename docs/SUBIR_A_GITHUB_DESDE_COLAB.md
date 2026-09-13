# Subir este proyecto a GitHub desde Google Colab

Repositorio destino:

`https://github.com/Uz-iel/Tesis-2`

## 1. Guarda el token de GitHub como secreto en Colab

En Colab abre el icono de **llave (Secrets)** y crea un secreto llamado `GITHUB_TOKEN`. Usa un token de GitHub con permiso **Contents: Read and write** únicamente para el repositorio `Tesis-2`. No escribas el token directamente en una celda ni lo guardes dentro del notebook.

## 2. Coloca la carpeta `Tesis-2_GitHub_listo` en tu sesión de Colab

Puede estar en `/content/Tesis-2_GitHub_listo` o en Drive. Luego ejecuta, ajustando `PROJECT_DIR` si corresponde:

```python
from pathlib import Path
from google.colab import userdata
import subprocess, os, base64

PROJECT_DIR = Path('/content/Tesis-2_GitHub_listo')
REPO = 'https://github.com/Uz-iel/Tesis-2.git'
TOKEN = userdata.get('GITHUB_TOKEN')

subprocess.run(['git', 'init'], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', 'branch', '-M', 'main'], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', 'config', 'user.name', 'Uz-iel'], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', 'config', 'user.email', 'TU_EMAIL_DE_GITHUB'], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', 'add', '.'], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', 'commit', '-m', 'Tesis II: pipeline radiomico reproducible Semanas 2 y 3'], cwd=PROJECT_DIR, check=True)

# Autenticación sin guardar el token en los archivos del proyecto
auth = base64.b64encode(f'x-access-token:{TOKEN}'.encode()).decode()
subprocess.run(['git', 'remote', 'remove', 'origin'], cwd=PROJECT_DIR, check=False)
subprocess.run(['git', 'remote', 'add', 'origin', REPO], cwd=PROJECT_DIR, check=True)
subprocess.run(['git', '-c', f'http.extraHeader=AUTHORIZATION: basic {auth}', 'push', '-u', 'origin', 'main'], cwd=PROJECT_DIR, check=True)
```

Después actualiza la página de GitHub. Debes ver `README.md`, `src/`, `notebooks/`, `results/`, `docs/`, `data/` y `requirements.txt`.

## Comprobación antes de subir

```bash
git status
git ls-files
```

Confirma que **NO** aparezcan datasets clínicos, archivos `.joblib`, tokens, `.env` ni credenciales.
