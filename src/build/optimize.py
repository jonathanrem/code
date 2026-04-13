"""
XGBoost hyperparameter optimization using Optuna with Microsoft Teams notifications.
Run this before train.py to find optimal hyperparameters.
Results are stored in optuna.db (SQLite) and can be resumed.
"""
import sys
from pathlib import Path

import optuna
import pandas as pd
import pymsteams
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_get, cfg_path, CONFIG

DATA_PATH_TRAIN = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
TARGET_COL = cfg_get(CONFIG, "common.target_col", "outcome")
RANDOM_STATE = cfg_get(CONFIG, "common.random_state", 2)
N_FOLDS = cfg_get(CONFIG, "optuna.n_folds", cfg_get(CONFIG, "common.n_folds", 5))
N_TRIALS = cfg_get(CONFIG, "optuna.n_trials", 100)
N_THREADS = cfg_get(CONFIG, "optuna.n_threads", cfg_get(CONFIG, "common.n_threads", 32))
TREE_METHOD = cfg_get(CONFIG, "optuna.tree_method", "hist")
EARLY_STOPPING_ROUNDS = cfg_get(CONFIG, "optuna.early_stopping_rounds", 50)
OPTUNA_STORAGE = cfg_get(CONFIG, "optuna.storage", "sqlite:///optuna.db")
MSTEAMS_HOOK_URL = cfg_get(CONFIG, "teams.hook_url", "")

# Feature set: "all" or "top5"
FEATURE_SET = "all"
OPTUNA_STUDY_NAME = f"{cfg_get(CONFIG, 'optuna.study_name', 'xgb_optuna')}_{FEATURE_SET}"

myTeamsMessage = (
    pymsteams.connectorcard(MSTEAMS_HOOK_URL, verify=False)
    if MSTEAMS_HOOK_URL
    else None
)

df = pd.read_csv(DATA_PATH_TRAIN, sep=";")
if "Order" in df.columns:
    df = df.drop(columns=["Order"])

if FEATURE_SET == "top5":
    features_to_drop = cfg_get(CONFIG, "model_v2.features_to_drop", ["Order"])
    existing_to_drop = [f for f in features_to_drop if f in df.columns]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
        print(f"top5 mode: dropped {len(existing_to_drop)} features")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].astype("category")

n_neg = (y == 0).sum()
n_pos = (y == 1).sum()
scale_pos_weight = n_neg / n_pos

dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)


def send_teams(msg: str):
    if not myTeamsMessage:
        return
    myTeamsMessage.text(msg)
    myTeamsMessage.send()


def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": TREE_METHOD,
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "seed": RANDOM_STATE,
        "nthread": N_THREADS,
    }

    num_boost_round = trial.suggest_int("num_boost_round", 100, 1000)

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        nfold=N_FOLDS,
        stratified=True,
        num_boost_round=num_boost_round,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        seed=RANDOM_STATE,
        verbose_eval=False,
    )

    trial.set_user_attr("best_n_estimators", len(cv_results))

    if trial.number > 0 and (trial.number + 1) % 100 == 0:
        send_teams(f"Trial {trial.number + 1}/{N_TRIALS} - Best AUC: {trial.study.best_value:.4f}")

    return cv_results["test-auc-mean"].values[-1]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        direction="maximize",
    )

    n_completed = len(study.trials)
    n_remaining = max(0, N_TRIALS - n_completed)
    print(f"trials completed: {n_completed}/{N_TRIALS}")

    if n_remaining > 0:
        send_teams(f"Starting Optuna {FEATURE_SET} features. Completed: {n_completed}/{N_TRIALS}")
        print(f"starting {n_remaining} new trials...")
        study.optimize(objective, n_trials=n_remaining)
    else:
        print(f"study already has {n_completed} trials, skipping optimization")

    best_params_str = ", ".join(f"{k}={v}" for k, v in study.best_params.items())
    send_teams(
        f"XGB done with {len(study.trials)} trials. "
        f"Best trial #{study.best_trial.number} AUC={study.best_value:.4f}\n"
        f"Params: {best_params_str}"
    )
    print("best trial")
    print("  auc:", study.best_value)
    print("  params:", study.best_params)
    print("  best_n_estimators:", study.best_trial.user_attrs["best_n_estimators"])
