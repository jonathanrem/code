"""
Train script pour le modèle XGBoost csPCa.

Les fichiers `data/train_df.csv` et `data/test_df.csv` lus ici
proviennent d'un split par centre entier réalisé en amont via
`src/build/split_by_center.ipynb`. Le test set est composé de
centres absents du train, garantissant une validation externe
géographique. Voir section 3.8 du rapport TFE pour la liste des
centres affectés au test.
"""
import hashlib
import json
import pickle
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config, cfg_get, cfg_path, CONFIG, BASE_DIR

CONFIG_PATH = BASE_DIR / "config.yaml"
CONFIG = load_config(CONFIG_PATH)

_VALID_FEATURE_SETS = ("with_contralateral", "no_contralateral", "parsimonious")
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--feature-set", choices=_VALID_FEATURE_SETS, default=None)
_args, _ = _parser.parse_known_args()

DATA_PATH_TRAIN = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
DATA_PATH_TEST = cfg_path(CONFIG, "paths.test", "data/test_df.csv")
RUNS_DIR = cfg_path(CONFIG, "paths.runs_dir", "runs")
TARGET_COL = cfg_get(CONFIG, "common.target_col", "outcome")
RANDOM_STATE = cfg_get(CONFIG, "common.random_state", 2)
N_THREADS = cfg_get(CONFIG, "common.n_threads", 32)
RUN_TEST = cfg_get(CONFIG, "model.run_test", False)
_base_features_to_drop = list(cfg_get(CONFIG, "model.features_to_drop", ["Order"]))
early_stopping_metric = cfg_get(CONFIG, "model.early_stopping_metric", "logloss")
common_threshold = cfg_get(CONFIG, "model.common_threshold", 0.5)
EARLY_STOPPING_ROUNDS = cfg_get(CONFIG, "model.early_stopping_rounds", 100)
AUTO_EVALUATE = cfg_get(CONFIG, "model.auto_evaluate", False)
VERBOSE = cfg_get(CONFIG, "model.verbose", True)
MODEL_COMPARISON_PATH = BASE_DIR / "model_comparison.xlsx"
OPTUNA_LABEL = cfg_get(CONFIG, "model.optuna_label", "none")
CONTRALATERAL = cfg_get(CONFIG, "contralateral_features", [])
PARSIMONIOUS = list(cfg_get(CONFIG, "parsimonious_features", []))
FEATURE_SET = _args.feature_set or cfg_get(CONFIG, "optuna.feature_set", "with_contralateral")
if FEATURE_SET not in _VALID_FEATURE_SETS:
    raise ValueError(
        f"feature_set invalide : '{FEATURE_SET}'. "
        f"Valeurs acceptées : {', '.join(_VALID_FEATURE_SETS)}"
    )
FEATURES_LABEL = FEATURE_SET


def log(message: str) -> None:
    if VERBOSE:
        print(message)


def plot_roc_pr(y_true, y_proba, out_path, title_prefix):
    """Plot ROC and PR curves side by side.

    NOTE: The PR panel (AUPRC) is prevalence-dependent and classified as
    improper by Van Calster et al. (2025). Retained for exploratory use;
    not called in the default training pipeline.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    ap_score = average_precision_score(y_true, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={auc_score:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title(f"{title_prefix} ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"AP={ap_score:.3f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title(f"{title_prefix} PR")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    log(f"roc/pr plot saved: {out_path}")


def create_run_directory() -> tuple[Path, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"created run directory: {run_dir}")
    return run_dir, run_id


def compute_file_hash(file_path: Path) -> str:
    if not file_path.exists():
        return "file_not_found"
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_environment(run_dir: Path) -> None:
    try:
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        env_txt = result.stdout if result.returncode == 0 else "pip freeze failed"
    except Exception as exc:
        env_txt = f"failed to capture environment: {exc}"
    with open(run_dir / "env.txt", "w") as f:
        f.write(env_txt)
    log(f"environment saved: {run_dir / 'env.txt'}")


def save_data_info(run_dir: Path, X_train: pd.DataFrame, X_test: pd.DataFrame | None,
                   y_train: pd.Series, y_test: pd.Series | None) -> None:
    data_info = {
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape) if X_test is not None else None,
        "train_target_distribution": {
            "0": int((y_train == 0).sum()),
            "1": int((y_train == 1).sum()),
        },
        "test_target_distribution": {
            "0": int((y_test == 0).sum()),
            "1": int((y_test == 1).sum()),
        } if y_test is not None else None,
        "train_file_hash": compute_file_hash(DATA_PATH_TRAIN),
        "test_file_hash": compute_file_hash(DATA_PATH_TEST),
    }
    with open(run_dir / "data_info.json", "w") as f:
        json.dump(data_info, f, indent=2)
    log(f"data info saved: {run_dir / 'data_info.json'}")


def save_run_artifacts(run_dir: Path, run_id: str, model, xgb_params: dict,
                       scale_pos_weight: float, auc_tr: float, auc_val: float,
                       auc_test: float, brier_tr: float, brier_val: float,
                       brier_test: float, best_iteration: int, threshold_final: float,
                       category_mappings: dict,
                       X_train: pd.DataFrame, X_test: pd.DataFrame | None,
                       y_train: pd.Series, y_test: pd.Series | None) -> None:

    shutil.copy(CONFIG_PATH, run_dir / "config.yaml")
    log(f"config saved: {run_dir / 'config.yaml'}")

    params = {
        "hyperparameters": {k: v for k, v in xgb_params.items()
                            if k not in ["scale_pos_weight", "random_state", "n_jobs", "early_stopping_rounds"]},
        "scale_pos_weight": float(scale_pos_weight),
        "random_state": int(xgb_params.get("random_state", RANDOM_STATE)),
        "n_threads": int(xgb_params.get("n_jobs", N_THREADS)),
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)
    log(f"params saved: {run_dir / 'params.json'}")

    metrics = {
        "auc": {
            "train": float(auc_tr),
            "val": float(auc_val),
            "test": float(auc_test) if auc_test is not None else None,
        },
        "brier": {
            "train": float(brier_tr),
            "val": float(brier_val),
            "test": float(brier_test) if brier_test is not None else None,
        },
        "best_iteration": int(best_iteration),
        "threshold": float(threshold_final),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"metrics saved: {run_dir / 'metrics.json'}")

    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    log(f"model saved: {run_dir / 'model.pkl'}")

    metadata = {
        "model_type": "XGBClassifier",
        "feature_set": FEATURE_SET,
        "contralateral_used": FEATURE_SET == "with_contralateral" and any(c in X_train.columns for c in CONTRALATERAL),
        "features": list(X_train.columns),
        "n_features": X_train.shape[1],
        "target": TARGET_COL,
        "random_state": RANDOM_STATE,
        "best_iteration": int(best_iteration),
        "hyperparameters": {k: v for k, v in xgb_params.items() if k != "scale_pos_weight"},
        "scale_pos_weight": float(scale_pos_weight),
        "category_mappings": {col: list(cats) for col, cats in category_mappings.items()},
        "metrics": {
            "auc": {"train": float(auc_tr), "val": float(auc_val)},
            "brier": {"train": float(brier_tr), "val": float(brier_val)},
        },
        "threshold": {"fixed": float(threshold_final)},
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"metadata saved: {run_dir / 'metadata.json'}")

    save_data_info(run_dir, X_train, X_test, y_train, y_test)
    save_environment(run_dir)

    with open(run_dir / "run_id.txt", "w") as f:
        f.write(run_id)
    log(f"run_id saved: {run_dir / 'run_id.txt'}")


def _normalize_optuna_label(value) -> str | int:
    if value is None:
        return "none"
    value_str = str(value).strip().lower()
    if value_str in ("", "none"):
        return "none"
    if value_str.isdigit():
        return int(value_str)
    return value_str


def update_model_comparison(auc_train: float, auc_val: float) -> None:
    optuna_value = _normalize_optuna_label(OPTUNA_LABEL)
    features_value = FEATURES_LABEL or "all"
    row = {
        "optuna": optuna_value,
        "features": features_value,
        "auc_train": round(float(auc_train), 4),
        "auc_val": round(float(auc_val), 4),
    }
    try:
        if MODEL_COMPARISON_PATH.exists():
            df = pd.read_excel(MODEL_COMPARISON_PATH)
        else:
            df = pd.DataFrame(columns=["optuna", "features", "auc_train", "auc_val"])
    except Exception as exc:
        log(f"model_comparison update skipped (read failed): {exc}")
        return
    if {"optuna", "features", "auc_train", "auc_val"}.issubset(df.columns):
        mask = (df["optuna"] == row["optuna"]) & (df["features"] == row["features"])
        if mask.any():
            df.loc[mask, ["auc_train", "auc_val"]] = row["auc_train"], row["auc_val"]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    try:
        df.to_excel(MODEL_COMPARISON_PATH, index=False)
        log(f"model_comparison updated: {MODEL_COMPARISON_PATH}")
    except Exception as exc:
        log(f"model_comparison update skipped (write failed): {exc}")


run_dir, run_id = create_run_directory()
log(f"run_id: {run_id}")

_XGB_DEFAULTS = {
    "booster": "gbtree",
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1.0,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}
_xgb_config_section = "model_parsimonious" if FEATURE_SET == "parsimonious" else "model"
_cfg_xgb = cfg_get(CONFIG, f"{_xgb_config_section}.xgb_params", {}) or {}
xgb_params = {
    **_XGB_DEFAULTS,
    **_cfg_xgb,
    "random_state": RANDOM_STATE,  # type: ignore
    "n_jobs": N_THREADS,           # type: ignore
    "enable_categorical": True,
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,  # type: ignore
}
if _cfg_xgb:
    log(f"xgb_params: loaded from config[{_xgb_config_section}] ({len(_cfg_xgb)} keys overridden)")
else:
    log("xgb_params: using XGBoost defaults")

log("LOAD DATA")
df_train = pd.read_csv(DATA_PATH_TRAIN, sep=";")
if RUN_TEST:
    df_test = pd.read_csv(DATA_PATH_TEST, sep=";")
    log(f"train: {df_train.shape}, test: {df_test.shape}")
else:
    df_test = None
    log(f"train: {df_train.shape}, test: skipped")

y_train = df_train[TARGET_COL]
X_train = df_train.drop(columns=[TARGET_COL])
if RUN_TEST:
    y_test = df_test[TARGET_COL]
    X_test = df_test.drop(columns=[TARGET_COL])
else:
    y_test = None
    X_test = None

log("DROP FEATURES")
if FEATURE_SET == "parsimonious":
    if not PARSIMONIOUS:
        raise ValueError("parsimonious_features vide dans config.yaml")
    missing = [c for c in PARSIMONIOUS if c not in X_train.columns]
    if missing:
        raise ValueError(f"parsimonious_features absentes du CSV train : {missing}")
    X_train = X_train[PARSIMONIOUS]
    if RUN_TEST:
        X_test = X_test[PARSIMONIOUS]
    log(f"parsimonious features ({len(PARSIMONIOUS)}): {PARSIMONIOUS}")
elif FEATURE_SET == "no_contralateral":
    features_to_drop = list(set(_base_features_to_drop) | set(CONTRALATERAL))
    existing_to_drop = [f for f in features_to_drop if f in X_train.columns]
    if existing_to_drop:
        X_train = X_train.drop(columns=existing_to_drop)
        if RUN_TEST:
            X_test = X_test.drop(columns=existing_to_drop)
        log(f"dropped: {existing_to_drop}")
else:  # with_contralateral
    features_to_drop = [f for f in _base_features_to_drop if f not in CONTRALATERAL]
    existing_to_drop = [f for f in features_to_drop if f in X_train.columns]
    if existing_to_drop:
        X_train = X_train.drop(columns=existing_to_drop)
        if RUN_TEST:
            X_test = X_test.drop(columns=existing_to_drop)
        log(f"dropped: {existing_to_drop}")

log("CATEGORICAL HANDLING")
object_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
category_mappings = {}
if object_cols:
    for col in object_cols:
        X_train[col] = X_train[col].astype("category")
        category_mappings[col] = X_train[col].cat.categories
        if RUN_TEST:
            X_test[col] = pd.Categorical(X_test[col], categories=category_mappings[col])

log("TRAIN/VAL SPLIT FOR EARLY STOPPING")
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)
log(f"train internal: {X_tr.shape}, validation: {X_val.shape}")

n_neg = (y_tr == 0).sum()
n_pos = (y_tr == 1).sum()
scale_pos_weight = n_neg / n_pos
xgb_params["scale_pos_weight"] = scale_pos_weight
log(f"scale_pos_weight: {scale_pos_weight:.3f} (neg={n_neg}, pos={n_pos})")

if early_stopping_metric == "auc":
    xgb_params["eval_metric"] = ["logloss", "auc"]
else:
    xgb_params["eval_metric"] = ["auc", "logloss"]

log("TRAIN XGBOOST")
model = xgb.XGBClassifier(**xgb_params)
model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=50,
)

try:
    best_iteration = model.best_iteration or model.n_estimators
except AttributeError:
    best_iteration = model.n_estimators
log(f"\nbest iteration: {best_iteration}")

log("EVALUATE ON TRAIN/VAL/TEST")
y_tr_pred_proba = model.predict_proba(X_tr)[:, 1]
y_val_pred_proba = model.predict_proba(X_val)[:, 1]

auc_tr = roc_auc_score(y_tr, y_tr_pred_proba)
auc_val = roc_auc_score(y_val, y_val_pred_proba)
brier_tr = brier_score_loss(y_tr, y_tr_pred_proba)
brier_val = brier_score_loss(y_val, y_val_pred_proba)

update_model_comparison(auc_tr, auc_val)

if RUN_TEST:
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, y_test_pred_proba)
    brier_test = brier_score_loss(y_test, y_test_pred_proba)
    print(f"auc    - train: {auc_tr:.4f} | val: {auc_val:.4f} | test: {auc_test:.4f}")
    log(f"brier  - train: {brier_tr:.4f} | val: {brier_val:.4f} | test: {brier_test:.4f}")
else:
    auc_test = None
    brier_test = None
    print(f"auc    - train: {auc_tr:.4f} | val: {auc_val:.4f}")
    log(f"brier  - train: {brier_tr:.4f} | val: {brier_val:.4f}")

threshold_final = common_threshold

log("SAVING RUN ARTIFACTS")
save_run_artifacts(
    run_dir=run_dir,
    run_id=run_id,
    model=model,
    xgb_params=xgb_params,
    scale_pos_weight=scale_pos_weight,
    auc_tr=auc_tr,
    auc_val=auc_val,
    auc_test=auc_test,
    brier_tr=brier_tr,
    brier_val=brier_val,
    brier_test=brier_test,
    best_iteration=best_iteration,
    threshold_final=threshold_final,
    category_mappings=category_mappings,
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
)

log("SAVING CANONICAL COPY")
_canonical_dir = RUNS_DIR / FEATURE_SET
_canonical_dir.mkdir(parents=True, exist_ok=True)
for _fname in ("model.pkl", "metadata.json", "params.json", "config.yaml"):
    _src = run_dir / _fname
    if _src.exists():
        shutil.copy(_src, _canonical_dir / _fname)
log(f"canonical: {_canonical_dir}")

log("FINAL MODEL TRAINING COMPLETE")
log(f"\nrun directory: {run_dir}")
log(f"run_id: {run_id}")
if RUN_TEST:
    log(f"auc test: {auc_test:.4f}")
    log(f"brier test: {brier_test:.4f}")
log(f"threshold: {threshold_final:.3f}")
