
"""
Pipeline radiomico reproducible - Tesis II

Derivado del notebook Colab del proyecto. El dataset NO se incluye en el
repositorio. Ejemplo:

    python src/run_experiment.py --data /ruta/radiomica_pancreas_dataset_semana3.csv --output results/run

El script mantiene el pipeline principal usado en el notebook: limpieza,
control de duplicados, selección de variables, validación cruzada estratificada,
búsqueda de hiperparámetros, métricas y exportación de resultados.
"""

# ==========================================
# 2) Importar librerías
# ==========================================
import os
import re
import json
import shutil
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
    precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from xgboost import XGBClassifier

import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
# ==========================================
# 3) Configuración principal
# ==========================================
TARGET_COL = "y"        # Columna objetivo
POS_LABEL = 1           # Clase positiva/casos
ID_CANDIDATES = ["id", "ID", "Id", "segmentation", "Segmentation", "seg_id", "patient_id", "Paciente", "paciente"]


# Para piloto/desbalance fuerte:
# Se recomienda optimizar por average_precision / PR-AUC y no por accuracy.
MAIN_SCORING = "average_precision"

# Ajusta según tu computador/Colab.
MAX_TOP_FEATURES = 30   # Número de características principales a reportar por modelo
CORR_THRESHOLD_DEFAULT = 0.90

# Umbral de clasificación:
# "f2" prioriza sensibilidad; "f1" balancea precisión/sensibilidad; "balanced_accuracy" balancea sensibilidad/especificidad.
THRESHOLD_METRIC = "f2"

# Si hay duplicados exactos de características:
DROP_DUPLICATED_FEATURE_ROWS = True

# Si hay filas con mismas features pero distinto target, lo más conservador es removerlas.
DROP_CONFLICTING_DUPLICATES = True

import argparse
import random
try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

parser = argparse.ArgumentParser(description="Pipeline radiomico reproducible - Tesis II")
parser.add_argument("--data", required=True, help="Ruta al CSV de radiómica (no se versiona en GitHub)")
parser.add_argument("--output", default="results/run", help="Directorio de resultados/checkpoints")
parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
parser.add_argument("--n-iter", type=int, default=30, help="Iteraciones máximas de RandomizedSearchCV")
args = parser.parse_args()

RANDOM_STATE = args.seed
N_ITER_SEARCH = args.n_iter
set_seed(RANDOM_STATE)

RUN_NAME = Path(args.output).name
USE_GOOGLE_DRIVE = False
RESUME_FROM_CHECKPOINTS = True
SAVE_FINAL_MODELS = True
BASE_RESULTS_DIR = Path(args.output)
FOLD_DIR = BASE_RESULTS_DIR / "folds"
FINAL_MODEL_DIR = BASE_RESULTS_DIR / "final_models"
TABLE_DIR = BASE_RESULTS_DIR / "tables"
EXPORT_DIR = BASE_RESULTS_DIR / "exports"
for d in [BASE_RESULTS_DIR, FOLD_DIR, FINAL_MODEL_DIR, TABLE_DIR, EXPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path(args.data)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"No se encontró el dataset: {DATA_PATH}")
try:
    df_raw = pd.read_csv(DATA_PATH, sep=None, engine="python")
except Exception:
    df_raw = pd.read_csv(DATA_PATH, sep=";")
if df_raw.shape[1] == 1:
    df_raw = pd.read_csv(DATA_PATH, sep=";")
print("Dataset:", DATA_PATH)
print("Dimensiones originales:", df_raw.shape)

def safe_name(name):
    """Nombre seguro para archivos."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(name))


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df_table, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(df_table, pd.DataFrame):
        df_table.to_csv(path, index=False)
    else:
        pd.DataFrame(df_table).to_csv(path, index=False)


def load_table(path):
    return pd.read_csv(path)


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path):
    return joblib.load(path)


def checkpoint_exists(path):
    return Path(path).exists()


def fold_metric_path(model_name, fold_id):
    return FOLD_DIR / f"{safe_name(model_name)}__fold_{fold_id:02d}__metrics.json"


def fold_param_path(model_name, fold_id):
    return FOLD_DIR / f"{safe_name(model_name)}__fold_{fold_id:02d}__best_params.json"


def final_model_path(model_name):
    return FINAL_MODEL_DIR / f"{safe_name(model_name)}__final_model.joblib"


def final_feature_path(model_name):
    return FINAL_MODEL_DIR / f"{safe_name(model_name)}__final_features.csv"


def final_param_path(model_name):
    return FINAL_MODEL_DIR / f"{safe_name(model_name)}__final_params.json"


def export_all_tables(tables_dict, excel_name="resultados_radiomica_colab.xlsx"):
    """
    Exporta tablas a:
    1) Excel consolidado
    2) CSV individual por tabla
    3) Copia con timestamp
    """
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    excel_path = EXPORT_DIR / excel_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_timestamp_path = EXPORT_DIR / f"{Path(excel_name).stem}_{timestamp}.xlsx"

    valid_tables = {}
    for sheet, table in tables_dict.items():
        if table is not None and isinstance(table, pd.DataFrame) and len(table) > 0:
            valid_tables[sheet] = table.copy()
            csv_path = TABLE_DIR / f"{safe_name(sheet)}.csv"
            table.to_csv(csv_path, index=False)

    if len(valid_tables) == 0:
        print("No hay tablas para exportar todavía.")
        return None

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for sheet, table in valid_tables.items():
            table.to_excel(writer, sheet_name=sheet[:31], index=False)

    shutil.copy2(excel_path, excel_timestamp_path)

    manifest = {
        "run_name": RUN_NAME,
        "created_at": timestamp,
        "base_results_dir": str(BASE_RESULTS_DIR),
        "excel_path": str(excel_path),
        "excel_timestamp_path": str(excel_timestamp_path),
        "tables": list(valid_tables.keys())
    }
    save_json(manifest, EXPORT_DIR / "manifest_ultima_exportacion.json")

    print("Excel principal guardado en:", excel_path)
    print("Copia con timestamp guardada en:", excel_timestamp_path)
    print("CSV individuales guardados en:", TABLE_DIR)
    return excel_path


def load_previous_cv_results():
    """Carga métricas y parámetros de folds ya guardados."""
    metric_rows = []
    param_rows = []

    for p in sorted(FOLD_DIR.glob("*__metrics.json")):
        try:
            metric_rows.append(load_json(p))
        except Exception as e:
            print("No se pudo cargar", p, e)

    for p in sorted(FOLD_DIR.glob("*__best_params.json")):
        try:
            param_rows.append(load_json(p))
        except Exception as e:
            print("No se pudo cargar", p, e)

    fold_metrics_prev = pd.DataFrame(metric_rows)
    best_params_prev = pd.DataFrame(param_rows)

    print("Folds previos encontrados:", len(fold_metrics_prev))
    print("Parámetros previos encontrados:", len(best_params_prev))
    return fold_metrics_prev, best_params_prev


def load_previous_final_models_summary():
    """Carga tablas de features/parámetros de modelos finales ya guardados."""
    feature_tables = []
    param_rows = []

    for p in sorted(FINAL_MODEL_DIR.glob("*__final_features.csv")):
        try:
            feature_tables.append(pd.read_csv(p))
        except Exception as e:
            print("No se pudo cargar", p, e)

    for p in sorted(FINAL_MODEL_DIR.glob("*__final_params.json")):
        try:
            param_rows.append(load_json(p))
        except Exception as e:
            print("No se pudo cargar", p, e)

    features_prev = pd.concat(feature_tables, ignore_index=True) if len(feature_tables) else pd.DataFrame()
    params_prev = pd.DataFrame(param_rows)

    print("Tablas de features finales previas:", len(feature_tables))
    print("Parámetros finales previos:", len(params_prev))
    return features_prev, params_prev


def show_checkpoint_status():
    print("Base:", BASE_RESULTS_DIR)
    print("Folds guardados:", len(list(FOLD_DIR.glob("*__metrics.json"))))
    print("Modelos finales guardados:", len(list(FINAL_MODEL_DIR.glob("*__final_model.joblib"))))
    print("Tablas CSV guardadas:", len(list(TABLE_DIR.glob("*.csv"))))
    print("Excels exportados:", len(list(EXPORT_DIR.glob("*.xlsx"))))

show_checkpoint_status()

# ==========================================
# 6) Funciones auxiliares
# ==========================================

def parse_radiomics_feature(col):
    '''
    Intenta separar familia radiomics y nombre específico.
    Ejemplos:
    2_firstorder_median -> familia firstorder, característica median
    3_glcm_correlation -> familia glcm, característica correlation
    '''
    s = str(col).strip()
    s2 = re.sub(r"^\d+_", "", s.lower())
    known_families = ["firstorder", "shape", "shape2d", "shape3d", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    for fam in known_families:
        if s2 == fam or s2.startswith(fam + "_"):
            char = s2.replace(fam + "_", "", 1)
            return fam, char
    parts = s2.split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    return "unknown", s2


def benjamini_hochberg(pvalues):
    '''FDR Benjamini-Hochberg sin depender de statsmodels.'''
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cliffs_delta(x, y):
    '''Tamaño de efecto no paramétrico. Positivo = valores más altos en casos.'''
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Cálculo eficiente razonable para datasets pequeños/medianos.
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return (gt - lt) / (len(x) * len(y))


class CorrelationFilter(BaseEstimator, TransformerMixin):
    '''
    Filtro no supervisado para eliminar features altamente correlacionadas.
    Se ajusta dentro de cada fold para evitar leakage.
    '''
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[1]
        if self.threshold is None or self.threshold >= 1 or n_features <= 1:
            self.keep_mask_ = np.ones(n_features, dtype=bool)
            return self

        corr = pd.DataFrame(X).corr().abs().fillna(0).values
        upper = np.triu(corr, k=1)
        to_drop = np.where((upper > self.threshold).any(axis=0))[0]
        keep = np.ones(n_features, dtype=bool)
        keep[to_drop] = False
        self.keep_mask_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]


def get_scores(estimator, X):
    '''Obtiene score continuo para ROC/PR.'''
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_ap(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def choose_threshold(y_true, y_score, metric="f2"):
    '''
    Elige umbral usando solo datos de entrenamiento.
    Para datos muy desbalanceados, f2 suele ser útil si quieres no perder casos.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    rows = []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        if metric == "f1":
            value = f1_score(y_true, pred, zero_division=0)
        elif metric == "f2":
            value = fbeta_score(y_true, pred, beta=2, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, pred)
        else:
            value = f1_score(y_true, pred, zero_division=0)
        rows.append((thr, value))

    best_thr, best_value = sorted(rows, key=lambda z: z[1], reverse=True)[0]
    return float(best_thr)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": safe_ap(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0,
        "brier": brier_score_loss(y_true, y_score) if np.all((np.asarray(y_score) >= 0) & (np.asarray(y_score) <= 1)) else np.nan,
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def summarize_cv_results(fold_table):
    metric_cols = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
        "precision_ppv", "sensitivity_recall", "specificity",
        "f1", "f2", "mcc", "brier"
    ]
    rows = []
    for model, g in fold_table.groupby("model"):
        row = {"model": model, "n_folds": len(g)}
        for m in metric_cols:
            row[m + "_mean"] = g[m].mean()
            row[m + "_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def get_feature_names_after_pipeline(fitted_pipe, original_feature_names):
    '''
    Recupera nombres tras VarianceThreshold -> SelectKBest -> CorrelationFilter.
    '''
    names = np.array(original_feature_names)

    if "variance" in fitted_pipe.named_steps:
        vt = fitted_pipe.named_steps["variance"]
        if hasattr(vt, "get_support"):
            names = names[vt.get_support()]

    if "select" in fitted_pipe.named_steps:
        sel = fitted_pipe.named_steps["select"]
        if hasattr(sel, "get_support"):
            names = names[sel.get_support()]

    if "corr" in fitted_pipe.named_steps:
        corr = fitted_pipe.named_steps["corr"]
        if hasattr(corr, "keep_mask_"):
            names = names[corr.keep_mask_]

    return list(names)


def extract_model_features(fitted_pipe, original_feature_names, model_name, top_n=30):
    '''
    Extrae importancia interna del modelo:
    - coeficientes para modelos lineales
    - feature_importances_ para árboles
    '''
    names = get_feature_names_after_pipeline(fitted_pipe, original_feature_names)
    model = fitted_pipe.named_steps["model"]

    importance = None
    direction = None
    source = None

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importance = np.abs(coef)
        direction = coef
        source = "coeficiente"
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        direction = importance
        source = "feature_importance"
    else:
        return pd.DataFrame()

    n = min(len(names), len(importance))
    out = pd.DataFrame({
        "model": model_name,
        "feature": names[:n],
        "importance_abs": importance[:n],
        "direction_or_raw_value": direction[:n],
        "importance_source": source
    })

    fam_char = out["feature"].apply(parse_radiomics_feature)
    out["family"] = fam_char.apply(lambda z: z[0])
    out["radiomic_characteristic"] = fam_char.apply(lambda z: z[1])
    out = out.sort_values("importance_abs", ascending=False).head(top_n)
    return out


def make_search(estimator, param_grid, inner_cv):
    candidates = list(ParameterGrid(param_grid))
    if len(candidates) <= N_ITER_SEARCH:
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=MAIN_SCORING,
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring=MAIN_SCORING,
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )

# ==========================================
# 5) Detectar columnas, limpiar target y preparar X/y
# ==========================================
df = df_raw.copy()

# Limpiar nombres de columnas
df.columns = [str(c).strip() for c in df.columns]

if TARGET_COL not in df.columns:
    raise ValueError(f"No encuentro la columna target '{TARGET_COL}'. Columnas disponibles: {list(df.columns)}")

# Detectar columna ID si existe
id_cols = [c for c in df.columns if c in ID_CANDIDATES or "id" in c.lower() or "seg" in c.lower()]
ID_COL = id_cols[0] if len(id_cols) > 0 else None
print("Columna ID detectada:", ID_COL)

# Convertir target a binario 0/1
y_original = df[TARGET_COL]
y = (y_original == POS_LABEL).astype(int)

# Detectar columnas numéricas como features, excluyendo target e ID
exclude_cols = [TARGET_COL]
if ID_COL is not None:
    exclude_cols.append(ID_COL)

candidate_features = [c for c in df.columns if c not in exclude_cols]

# Intentar convertir features a numérico
for c in candidate_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

feature_cols = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]
X = df[feature_cols].copy()

print("Número de features numéricas:", len(feature_cols))
print("Distribución del target:")
balance_table = pd.DataFrame({
    "clase": [0, 1],
    "n": [(y == 0).sum(), (y == 1).sum()],
    "porcentaje": [(y == 0).mean()*100, (y == 1).mean()*100]
})
display(balance_table)

if (y == 1).sum() == 0 or (y == 0).sum() == 0:
    raise ValueError("El target tiene una sola clase. No se puede entrenar un modelo binario.")

prevalence = y.mean()
print(f"Prevalencia de clase positiva: {prevalence:.4f} = {prevalence*100:.2f}%")
print(f"Baseline PR-AUC esperado por azar ≈ prevalencia = {prevalence:.4f}")

# ==========================================
# 6) Funciones auxiliares
# ==========================================

def parse_radiomics_feature(col):
    '''
    Intenta separar familia radiomics y nombre específico.
    Ejemplos:
    2_firstorder_median -> familia firstorder, característica median
    3_glcm_correlation -> familia glcm, característica correlation
    '''
    s = str(col).strip()
    s2 = re.sub(r"^\d+_", "", s.lower())
    known_families = ["firstorder", "shape", "shape2d", "shape3d", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    for fam in known_families:
        if s2 == fam or s2.startswith(fam + "_"):
            char = s2.replace(fam + "_", "", 1)
            return fam, char
    parts = s2.split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    return "unknown", s2


def benjamini_hochberg(pvalues):
    '''FDR Benjamini-Hochberg sin depender de statsmodels.'''
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def cliffs_delta(x, y):
    '''Tamaño de efecto no paramétrico. Positivo = valores más altos en casos.'''
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    # Cálculo eficiente razonable para datasets pequeños/medianos.
    gt = sum(np.sum(xi > y) for xi in x)
    lt = sum(np.sum(xi < y) for xi in x)
    return (gt - lt) / (len(x) * len(y))


class CorrelationFilter(BaseEstimator, TransformerMixin):
    '''
    Filtro no supervisado para eliminar features altamente correlacionadas.
    Se ajusta dentro de cada fold para evitar leakage.
    '''
    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X)
        n_features = X.shape[1]
        if self.threshold is None or self.threshold >= 1 or n_features <= 1:
            self.keep_mask_ = np.ones(n_features, dtype=bool)
            return self

        corr = pd.DataFrame(X).corr().abs().fillna(0).values
        upper = np.triu(corr, k=1)
        to_drop = np.where((upper > self.threshold).any(axis=0))[0]
        keep = np.ones(n_features, dtype=bool)
        keep[to_drop] = False
        self.keep_mask_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]


def get_scores(estimator, X):
    '''Obtiene score continuo para ROC/PR.'''
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def safe_auc(y_true, y_score):
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_ap(y_true, y_score):
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def choose_threshold(y_true, y_score, metric="f2"):
    '''
    Elige umbral usando solo datos de entrenamiento.
    Para datos muy desbalanceados, f2 suele ser útil si quieres no perder casos.
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5

    rows = []
    for thr in thresholds:
        pred = (y_score >= thr).astype(int)
        if metric == "f1":
            value = f1_score(y_true, pred, zero_division=0)
        elif metric == "f2":
            value = fbeta_score(y_true, pred, beta=2, zero_division=0)
        elif metric == "balanced_accuracy":
            value = balanced_accuracy_score(y_true, pred)
        else:
            value = f1_score(y_true, pred, zero_division=0)
        rows.append((thr, value))

    best_thr, best_value = sorted(rows, key=lambda z: z[1], reverse=True)[0]
    return float(best_thr)


def compute_metrics(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan

    return {
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": safe_ap(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0,
        "brier": brier_score_loss(y_true, y_score) if np.all((np.asarray(y_score) >= 0) & (np.asarray(y_score) <= 1)) else np.nan,
        "threshold": threshold,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }


def summarize_cv_results(fold_table):
    metric_cols = [
        "roc_auc", "pr_auc", "accuracy", "balanced_accuracy",
        "precision_ppv", "sensitivity_recall", "specificity",
        "f1", "f2", "mcc", "brier"
    ]
    rows = []
    for model, g in fold_table.groupby("model"):
        row = {"model": model, "n_folds": len(g)}
        for m in metric_cols:
            row[m + "_mean"] = g[m].mean()
            row[m + "_std"] = g[m].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)


def get_feature_names_after_pipeline(fitted_pipe, original_feature_names):
    '''
    Recupera nombres tras VarianceThreshold -> SelectKBest -> CorrelationFilter.
    '''
    names = np.array(original_feature_names)

    if "variance" in fitted_pipe.named_steps:
        vt = fitted_pipe.named_steps["variance"]
        if hasattr(vt, "get_support"):
            names = names[vt.get_support()]

    if "select" in fitted_pipe.named_steps:
        sel = fitted_pipe.named_steps["select"]
        if hasattr(sel, "get_support"):
            names = names[sel.get_support()]

    if "corr" in fitted_pipe.named_steps:
        corr = fitted_pipe.named_steps["corr"]
        if hasattr(corr, "keep_mask_"):
            names = names[corr.keep_mask_]

    return list(names)


def extract_model_features(fitted_pipe, original_feature_names, model_name, top_n=30):
    '''
    Extrae importancia interna del modelo:
    - coeficientes para modelos lineales
    - feature_importances_ para árboles
    '''
    names = get_feature_names_after_pipeline(fitted_pipe, original_feature_names)
    model = fitted_pipe.named_steps["model"]

    importance = None
    direction = None
    source = None

    if hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importance = np.abs(coef)
        direction = coef
        source = "coeficiente"
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        direction = importance
        source = "feature_importance"
    else:
        return pd.DataFrame()

    n = min(len(names), len(importance))
    out = pd.DataFrame({
        "model": model_name,
        "feature": names[:n],
        "importance_abs": importance[:n],
        "direction_or_raw_value": direction[:n],
        "importance_source": source
    })

    fam_char = out["feature"].apply(parse_radiomics_feature)
    out["family"] = fam_char.apply(lambda z: z[0])
    out["radiomic_characteristic"] = fam_char.apply(lambda z: z[1])
    out = out.sort_values("importance_abs", ascending=False).head(top_n)
    return out


def make_search(estimator, param_grid, inner_cv):
    candidates = list(ParameterGrid(param_grid))
    if len(candidates) <= N_ITER_SEARCH:
        return GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=MAIN_SCORING,
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=N_ITER_SEARCH,
        scoring=MAIN_SCORING,
        cv=inner_cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )

# ==========================================
# 7) Calidad de datos, duplicados y familias radiomics
# ==========================================

quality_rows = []

quality_rows.append({"item": "filas_originales", "valor": df.shape[0]})
quality_rows.append({"item": "columnas_originales", "valor": df.shape[1]})
quality_rows.append({"item": "n_features_numericas", "valor": len(feature_cols)})
quality_rows.append({"item": "casos_positivos", "valor": int((y == 1).sum())})
quality_rows.append({"item": "controles_negativos", "valor": int((y == 0).sum())})
quality_rows.append({"item": "prevalencia_positiva_pct", "valor": y.mean() * 100})
quality_rows.append({"item": "faltantes_totales_features", "valor": int(X.isna().sum().sum())})
quality_rows.append({"item": "features_con_faltantes", "valor": int((X.isna().sum() > 0).sum())})
quality_rows.append({"item": "duplicados_filas_completas", "valor": int(df.duplicated().sum())})

if ID_COL is not None:
    quality_rows.append({"item": f"ids_duplicados_en_{ID_COL}", "valor": int(df[ID_COL].duplicated().sum())})

# Duplicados por matriz X
dup_feature_mask = X.duplicated(keep=False)
n_dup_feature_rows = int(dup_feature_mask.sum())
quality_rows.append({"item": "filas_con_features_duplicadas", "valor": n_dup_feature_rows})

# Grupos de features duplicadas con target conflictivo
tmp_dup = pd.concat([X, y.rename("__target__")], axis=1)
group_counts = tmp_dup.groupby(feature_cols, dropna=False)["__target__"].nunique().reset_index(name="n_targets")
conflicting_feature_groups = group_counts[group_counts["n_targets"] > 1]
quality_rows.append({"item": "grupos_features_duplicadas_con_target_conflictivo", "valor": len(conflicting_feature_groups)})

quality_table = pd.DataFrame(quality_rows)
display(quality_table)

# Tabla de familias
feature_meta = pd.DataFrame({"feature": feature_cols})
feature_meta[["family", "radiomic_characteristic"]] = feature_meta["feature"].apply(
    lambda c: pd.Series(parse_radiomics_feature(c))
)
family_count_table = (
    feature_meta.groupby("family")
    .agg(
        n_features=("feature", "count"),
        ejemplos=("feature", lambda s: ", ".join(list(s.head(5))))
    )
    .reset_index()
    .sort_values("n_features", ascending=False)
)
display(family_count_table)

# Limpieza conservadora de duplicados
df_model = df.copy()
y_model = y.copy()
X_model = X.copy()

if DROP_CONFLICTING_DUPLICATES and len(conflicting_feature_groups) > 0:
    # Marcar filas cuyo patrón de features aparece con targets distintos
    conflict_keys = set(map(tuple, conflicting_feature_groups[feature_cols].to_numpy()))
    row_keys = list(map(tuple, X_model[feature_cols].to_numpy()))
    conflict_mask = pd.Series([rk in conflict_keys for rk in row_keys], index=df_model.index)
    print("Removiendo filas conflictivas por mismas features con distinto target:", int(conflict_mask.sum()))
    df_model = df_model.loc[~conflict_mask].copy()
    X_model = X_model.loc[~conflict_mask].copy()
    y_model = y_model.loc[~conflict_mask].copy()

if DROP_DUPLICATED_FEATURE_ROWS:
    before = len(df_model)
    keep_mask = ~X_model.duplicated(keep="first")
    df_model = df_model.loc[keep_mask].copy()
    X_model = X_model.loc[keep_mask].copy()
    y_model = y_model.loc[keep_mask].copy()
    print("Filas removidas por features duplicadas exactas:", before - len(df_model))

print("Dimensiones para modelado:", X_model.shape)
print("Distribución final para modelado:")
display(pd.DataFrame({
    "clase": [0, 1],
    "n": [(y_model == 0).sum(), (y_model == 1).sum()],
    "porcentaje": [(y_model == 0).mean()*100, (y_model == 1).mean()*100]
}))

# ==========================================
# 8) Análisis univariado de features
# ==========================================

univar_rows = []

for feat in feature_cols:
    x0 = X_model.loc[y_model == 0, feat].astype(float)
    x1 = X_model.loc[y_model == 1, feat].astype(float)

    # Imputación simple solo para estadística univariada
    all_values = X_model[feat].astype(float)
    med = all_values.median()
    x0 = x0.fillna(med)
    x1 = x1.fillna(med)

    # Mann-Whitney
    try:
        stat, p = mannwhitneyu(x1, x0, alternative="two-sided")
    except Exception:
        stat, p = np.nan, np.nan

    # AUC univariado
    try:
        scores = X_model[feat].astype(float).fillna(med).values
        auc_raw = roc_auc_score(y_model, scores)
        auc_oriented = max(auc_raw, 1 - auc_raw)
    except Exception:
        auc_raw, auc_oriented = np.nan, np.nan

    fam, char = parse_radiomics_feature(feat)

    univar_rows.append({
        "feature": feat,
        "family": fam,
        "radiomic_characteristic": char,
        "mean_control_0": x0.mean(),
        "mean_case_1": x1.mean(),
        "median_control_0": x0.median(),
        "median_case_1": x1.median(),
        "mannwhitney_p": p,
        "auc_raw": auc_raw,
        "auc_oriented": auc_oriented,
        "cliffs_delta_case_vs_control": cliffs_delta(x1.values, x0.values)
    })

univar_table = pd.DataFrame(univar_rows)
univar_table["fdr_bh"] = benjamini_hochberg(univar_table["mannwhitney_p"].fillna(1).values)
univar_table = univar_table.sort_values(["auc_oriented", "fdr_bh"], ascending=[False, True])

print("Top 20 features univariadas:")
display(univar_table.head(20))

# Top por familia
top_univar_by_family = (
    univar_table.sort_values(["family", "auc_oriented"], ascending=[True, False])
    .groupby("family")
    .head(10)
    .reset_index(drop=True)
)

family_univar_summary = (
    univar_table.groupby("family")
    .agg(
        n_features=("feature", "count"),
        best_auc_oriented=("auc_oriented", "max"),
        median_auc_oriented=("auc_oriented", "median"),
        n_fdr_lt_005=("fdr_bh", lambda s: int((s < 0.05).sum())),
        best_feature=("feature", lambda s: univar_table.loc[s.index].sort_values("auc_oriented", ascending=False)["feature"].iloc[0])
    )
    .reset_index()
    .sort_values("best_auc_oriented", ascending=False)
)

print("Resumen univariado por familia:")
display(family_univar_summary)

# ==========================================
# 9) Definir modelos y grids de hiperparámetros
# ==========================================

X_final = X_model[feature_cols].copy()
y_final = y_model.astype(int).copy()

n_pos = int((y_final == 1).sum())
n_neg = int((y_final == 0).sum())
min_class_count = min(n_pos, n_neg)
pos_weight = n_neg / max(n_pos, 1)

print("n positivos:", n_pos)
print("n negativos:", n_neg)
print("scale_pos_weight sugerido:", pos_weight)

if min_class_count < 2:
    raise ValueError(
        "Hay menos de 2 observaciones en una clase. No es válido hacer CV. "
        "Para un piloto así, reporta solo EDA/descriptivo o consigue más casos positivos."
    )

N_SPLITS_OUTER = min(5, min_class_count)
N_SPLITS_INNER = max(2, min(3, min_class_count - 1))

print("Folds externos:", N_SPLITS_OUTER)
print("Folds internos:", N_SPLITS_INNER)

k_candidates = [5, 10, 20, 30, 50, 75, 100, "all"]
k_options = [k for k in k_candidates if k == "all" or k < len(feature_cols)]
print("Opciones k SelectKBest:", k_options)


def build_pipe(model, sampler=None):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scaler", RobustScaler()),
        ("select", SelectKBest(score_func=f_classif, k=min(20, len(feature_cols)))),
        ("corr", CorrelationFilter(threshold=CORR_THRESHOLD_DEFAULT)),
    ]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


models = {}

# 1) Regresión logística Elastic Net con class_weight
models["LogReg_elasticnet_balanced"] = (
    build_pipe(LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        class_weight="balanced",
        max_iter=10000,
        random_state=RANDOM_STATE
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95, 0.99],
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__l1_ratio": [0.1, 0.5, 0.9]
    }
)

# 2) Regresión logística con RandomOverSampler
models["LogReg_elasticnet_ROS"] = (
    build_pipe(
        LogisticRegression(
            solver="saga",
            penalty="elasticnet",
            max_iter=10000,
            random_state=RANDOM_STATE
        ),
        sampler=RandomOverSampler(random_state=RANDOM_STATE)
    ),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95],
        "model__C": [0.001, 0.01, 0.1, 1, 10],
        "model__l1_ratio": [0.1, 0.5, 0.9]
    }
)

# 3) SVM lineal
models["SVM_linear_balanced"] = (
    build_pipe(SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95],
        "model__C": [0.001, 0.01, 0.1, 1, 10]
    }
)

# 4) SVM RBF
models["SVM_RBF_balanced"] = (
    build_pipe(SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95],
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.001, 0.01, 0.1]
    }
)

# 5) Random Forest balanceado
models["RandomForest_balanced"] = (
    build_pipe(RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95, 0.99],
        "model__max_depth": [None, 2, 3, 5, 8],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5]
    }
)

# 6) Extra Trees balanceado
models["ExtraTrees_balanced"] = (
    build_pipe(ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95, 0.99],
        "model__max_depth": [None, 2, 3, 5, 8],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5]
    }
)

# 7) Balanced Random Forest de imbalanced-learn
models["BalancedRandomForest"] = (
    build_pipe(BalancedRandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        replacement=True
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95],
        "model__max_depth": [None, 2, 3, 5, 8],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5]
    }
)

# 8) XGBoost con scale_pos_weight
models["XGBoost_pos_weight"] = (
    build_pipe(XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
    {
        "select__k": k_options,
        "corr__threshold": [0.85, 0.90, 0.95],
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__min_child_weight": [1, 5, 10],
        "model__scale_pos_weight": [max(1, pos_weight*0.5), max(1, pos_weight), max(1, pos_weight*2)]
    }
)

# SMOTE solo si hay suficientes casos positivos
if n_pos >= 6:
    k_smote = min(5, n_pos - 1)
    models["LogReg_elasticnet_SMOTE"] = (
        build_pipe(
            LogisticRegression(
                solver="saga",
                penalty="elasticnet",
                max_iter=10000,
                random_state=RANDOM_STATE
            ),
            sampler=SMOTE(random_state=RANDOM_STATE, k_neighbors=k_smote)
        ),
        {
            "select__k": k_options,
            "corr__threshold": [0.85, 0.90, 0.95],
            "model__C": [0.001, 0.01, 0.1, 1, 10],
            "model__l1_ratio": [0.1, 0.5, 0.9]
        }
    )
else:
    print("SMOTE omitido: hay muy pocos positivos para generar vecinos de forma segura.")

# ==========================================
# 10) Validación cruzada anidada con búsqueda de hiperparámetros + checkpoints
# ==========================================

outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)

all_fold_rows = []
all_best_params = []

for model_name, (pipe, param_grid) in models.items():
    print("\n==============================")
    print("Modelo:", model_name)
    print("==============================")

    fold_id = 0

    for train_idx, test_idx in outer_cv.split(X_final, y_final):
        fold_id += 1

        m_path = fold_metric_path(model_name, fold_id)
        p_path = fold_param_path(model_name, fold_id)

        # Reanudar si ya existe este fold.
        if RESUME_FROM_CHECKPOINTS and checkpoint_exists(m_path):
            print(f"Fold {fold_id}: cargado desde checkpoint, se salta entrenamiento.")
            metric_row = load_json(m_path)
            all_fold_rows.append(metric_row)
            if checkpoint_exists(p_path):
                all_best_params.append(load_json(p_path))
            continue

        X_train = X_final.iloc[train_idx]
        X_test = X_final.iloc[test_idx]
        y_train = y_final.iloc[train_idx]
        y_test = y_final.iloc[test_idx]

        inner_min_class = min(int((y_train == 0).sum()), int((y_train == 1).sum()))
        inner_splits = max(2, min(N_SPLITS_INNER, inner_min_class))
        inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE + fold_id)

        search = make_search(pipe, param_grid, inner_cv)

        try:
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            best_params = search.best_params_

            # Score en entrenamiento para elegir threshold sin usar test.
            # Intento con predicción out-of-fold del mejor pipeline.
            try:
                train_scores = cross_val_predict(
                    clone(best_est),
                    X_train,
                    y_train,
                    cv=inner_cv,
                    method="predict_proba",
                    n_jobs=-1
                )[:, 1]
                threshold = choose_threshold(y_train, train_scores, metric=THRESHOLD_METRIC)
            except Exception:
                # Fallback menos ideal, pero evita que se caiga el notebook
                train_scores = get_scores(best_est, X_train)
                threshold = choose_threshold(y_train, train_scores, metric=THRESHOLD_METRIC)

            test_scores = get_scores(best_est, X_test)
            metric_row = compute_metrics(y_test, test_scores, threshold=threshold)

            # Convertir tipos numpy a tipos JSON nativos
            metric_row = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
                          for k, v in metric_row.items()}

            metric_row.update({
                "model": model_name,
                "fold": int(fold_id),
                "best_inner_score_pr_auc": float(search.best_score_),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_train_pos": int((y_train == 1).sum()),
                "n_test_pos": int((y_test == 1).sum()),
                "best_params_json": json.dumps(best_params)
            })

            param_row = {
                "model": model_name,
                "fold": int(fold_id),
                "best_inner_score_pr_auc": float(search.best_score_),
                "best_params_json": json.dumps(best_params)
            }

            all_fold_rows.append(metric_row)
            all_best_params.append(param_row)

            # Guardar inmediatamente después de cada fold.
            save_json(metric_row, m_path)
            save_json(param_row, p_path)

            # También guardar resumen acumulado en CSV para revisar mientras corre.
            pd.DataFrame(all_fold_rows).to_csv(TABLE_DIR / "metricas_folds_acumuladas.csv", index=False)
            pd.DataFrame(all_best_params).to_csv(TABLE_DIR / "mejores_parametros_folds_acumulados.csv", index=False)

            print(
                f"Fold {fold_id}: PR-AUC={metric_row['pr_auc']:.3f}, "
                f"ROC-AUC={metric_row['roc_auc']:.3f}, "
                f"BA={metric_row['balanced_accuracy']:.3f}, "
                f"Sens={metric_row['sensitivity_recall']:.3f}, "
                f"Esp={metric_row['specificity']:.3f}, "
                f"thr={threshold:.3f}"
            )

        except Exception as e:
            print(f"Error en {model_name}, fold {fold_id}: {e}")
            error_row = {
                "model": model_name,
                "fold": int(fold_id),
                "error": str(e)
            }
            all_fold_rows.append(error_row)
            save_json(error_row, m_path)

fold_metrics_table = pd.DataFrame(all_fold_rows)
best_params_table = pd.DataFrame(all_best_params)

# Quitar filas fallidas para resumen
fold_metrics_ok = fold_metrics_table[fold_metrics_table["error"].isna()] if "error" in fold_metrics_table.columns else fold_metrics_table.copy()
model_comparison_table = summarize_cv_results(fold_metrics_ok)

# Guardar tablas finales de esta celda
save_table(fold_metrics_table, TABLE_DIR / "06_metricas_folds.csv")
save_table(best_params_table, TABLE_DIR / "08_mejores_parametros_folds.csv")
save_table(model_comparison_table, TABLE_DIR / "07_comparacion_modelos.csv")

print("\nTabla comparativa de modelos:")
display(model_comparison_table)

print("\nMejores hiperparámetros por fold:")
display(best_params_table.head(20))

show_checkpoint_status()

# ==========================================
# 11) Reentrenar modelos completos y extraer mejores features por modelo + checkpoints
# ==========================================

final_model_feature_tables = []
final_model_objects = {}
final_model_params = []

# Reentrenar cada modelo en todos los datos usando búsqueda interna.
full_cv = StratifiedKFold(
    n_splits=max(2, min(N_SPLITS_INNER, min_class_count)),
    shuffle=True,
    random_state=RANDOM_STATE
)

for model_name, (pipe, param_grid) in models.items():
    print("\nRevisando modelo completo:", model_name)

    model_path = final_model_path(model_name)
    features_path = final_feature_path(model_name)
    params_path = final_param_path(model_name)

    # Reanudar si ya existe el modelo final.
    if RESUME_FROM_CHECKPOINTS and checkpoint_exists(model_path) and checkpoint_exists(params_path):
        print("Cargando modelo final desde checkpoint:", model_path)
        try:
            final_model_objects[model_name] = load_pickle(model_path)

            if checkpoint_exists(features_path):
                ft = pd.read_csv(features_path)
                if len(ft) > 0:
                    final_model_feature_tables.append(ft)
                    print("Features cargadas:")
                    display(ft.head(10))

            param_row = load_json(params_path)
            final_model_params.append(param_row)
            continue
        except Exception as e:
            print("No se pudo cargar checkpoint. Se reentrenará.")
            print("Detalle:", e)

    print("Reentrenando modelo completo:", model_name)

    try:
        search = make_search(pipe, param_grid, full_cv)
        search.fit(X_final, y_final)

        final_estimator = search.best_estimator_
        final_model_objects[model_name] = final_estimator

        param_row = {
            "model": model_name,
            "best_cv_pr_auc": float(search.best_score_),
            "best_params_json": json.dumps(search.best_params_)
        }
        final_model_params.append(param_row)

        ft = extract_model_features(
            final_estimator,
            original_feature_names=feature_cols,
            model_name=model_name,
            top_n=MAX_TOP_FEATURES
        )

        if len(ft) > 0:
            final_model_feature_tables.append(ft)
            print("Top features:")
            display(ft.head(10))
        else:
            print("No se pudo extraer importancia interna para este modelo.")

        # Guardar inmediatamente después de cada modelo final.
        if SAVE_FINAL_MODELS:
            save_pickle(final_estimator, model_path)
        save_json(param_row, params_path)
        if len(ft) > 0:
            ft.to_csv(features_path, index=False)

        # Guardar acumulados también.
        pd.DataFrame(final_model_params).to_csv(TABLE_DIR / "09_parametros_finales_acumulados.csv", index=False)
        if len(final_model_feature_tables) > 0:
            pd.concat(final_model_feature_tables, ignore_index=True).to_csv(TABLE_DIR / "10_features_por_modelo_acumuladas.csv", index=False)

    except Exception as e:
        print("Error al reentrenar:", model_name, e)
        error_param = {
            "model": model_name,
            "best_cv_pr_auc": np.nan,
            "best_params_json": "{}",
            "error": str(e)
        }
        final_model_params.append(error_param)
        save_json(error_param, params_path)

final_params_table = pd.DataFrame(final_model_params)

if len(final_model_feature_tables) > 0:
    features_by_model_table = pd.concat(final_model_feature_tables, ignore_index=True)
else:
    features_by_model_table = pd.DataFrame()

# Guardar tablas finales de esta celda
save_table(final_params_table, TABLE_DIR / "09_parametros_finales.csv")
if len(features_by_model_table) > 0:
    save_table(features_by_model_table, TABLE_DIR / "10_features_por_modelo.csv")

print("\nTabla de mejores características por modelo:")
display(features_by_model_table.head(50))

show_checkpoint_status()

# ==========================================
# 12) Tablas por familia radiomics + guardado
# ==========================================

if len(features_by_model_table) > 0:
    # Importancia agregada por modelo y familia
    model_family_importance_table = (
        features_by_model_table
        .groupby(["model", "family"])
        .agg(
            n_top_features=("feature", "count"),
            mean_importance=("importance_abs", "mean"),
            max_importance=("importance_abs", "max"),
            top_features=("feature", lambda s: ", ".join(list(s.head(10))))
        )
        .reset_index()
        .sort_values(["model", "max_importance"], ascending=[True, False])
    )

    # Top características por familia y modelo
    top_features_by_model_family_table = (
        features_by_model_table
        .sort_values(["model", "family", "importance_abs"], ascending=[True, True, False])
        .groupby(["model", "family"])
        .head(5)
        .reset_index(drop=True)
    )

    # Frecuencia de aparición de features entre modelos
    feature_frequency_table = (
        features_by_model_table
        .groupby(["feature", "family", "radiomic_characteristic"])
        .agg(
            n_models=("model", "nunique"),
            models=("model", lambda s: ", ".join(sorted(set(s)))),
            mean_importance=("importance_abs", "mean"),
            max_importance=("importance_abs", "max")
        )
        .reset_index()
        .sort_values(["n_models", "max_importance"], ascending=[False, False])
    )

else:
    model_family_importance_table = pd.DataFrame()
    top_features_by_model_family_table = pd.DataFrame()
    feature_frequency_table = pd.DataFrame()

# Guardar tablas inmediatamente
if len(model_family_importance_table) > 0:
    save_table(model_family_importance_table, TABLE_DIR / "11_importancia_modelo_familia.csv")
if len(top_features_by_model_family_table) > 0:
    save_table(top_features_by_model_family_table, TABLE_DIR / "12_top_features_modelo_familia.csv")
if len(feature_frequency_table) > 0:
    save_table(feature_frequency_table, TABLE_DIR / "13_frecuencia_features_modelos.csv")

print("Importancia agregada por modelo y familia:")
display(model_family_importance_table.head(50))

print("Top features por modelo y familia:")
display(top_features_by_model_family_table.head(50))

print("Features que más se repiten entre modelos:")
display(feature_frequency_table.head(50))

show_checkpoint_status()

# ==========================================
# 13) Recomendación automática de mejor modelo
# ==========================================

if len(model_comparison_table) > 0:
    best_model_name = model_comparison_table.iloc[0]["model"]
    print("Mejor modelo por PR-AUC promedio:", best_model_name)

    best_model_summary = model_comparison_table[model_comparison_table["model"] == best_model_name]
    display(best_model_summary)

    if best_model_name in final_model_objects:
        best_final_model = final_model_objects[best_model_name]
        print("Pipeline final del mejor modelo:")
        print(best_final_model)
else:
    best_model_name = None
    print("No hay comparación de modelos disponible.")

# ==========================================
# 14) Exportar todas las tablas a Excel, CSV y Drive
# ==========================================

output_excel = "resultados_radiomica_colab.xlsx"

tables_to_export = {
    "00_balance_target": balance_table,
    "01_calidad_datos": quality_table,
    "02_familias_features": family_count_table,
    "03_univariado_top": univar_table,
    "04_univariado_por_familia": family_univar_summary,
    "05_top_univariado_familia": top_univar_by_family,
    "06_metricas_folds": fold_metrics_table,
    "07_comparacion_modelos": model_comparison_table,
    "08_mejores_parametros_folds": best_params_table,
    "09_parametros_finales": final_params_table,
    "10_features_por_modelo": features_by_model_table,
    "11_importancia_modelo_familia": model_family_importance_table,
    "12_top_features_modelo_familia": top_features_by_model_family_table,
    "13_frecuencia_features_modelos": feature_frequency_table,
    "14_feature_metadata": feature_meta
}

excel_path = export_all_tables(tables_to_export, excel_name=output_excel)

# En ejecución local/CI no se fuerza descarga automática.

show_checkpoint_status()

print("\nEjecución finalizada. Resultados en:", BASE_RESULTS_DIR.resolve())
