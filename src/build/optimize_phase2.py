"""
XGBoost Phase 2 hyperparameter optimization using Optuna.

Workflow:
  1. Analyze top 15% trials from Phase 1 (optuna.db)
  2. Compute refined search ranges: [p10 - 10%, p90 + 10%] clamped to Phase 1 bounds
  3. Run 200-trial Phase 2 study with MedianPruner (resumes if interrupted)
  4. Display comparison vs Phase 1 and save best params to runs/phase2_best_params.json

Usage:
    python src/build/optimize_phase2.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

# Silence Optuna info logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Allow `from common.xxx import ...`  (src/build/ → parent = src/)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_get, cfg_path, CONFIG  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH_TRAIN  = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
TARGET_COL       = cfg_get(CONFIG, "common.target_col", "outcome")
RANDOM_STATE     = cfg_get(CONFIG, "common.random_state", 2)
N_FOLDS          = cfg_get(CONFIG, "optuna.n_folds", 5)
N_THREADS        = cfg_get(CONFIG, "optuna.n_threads", 32)
TREE_METHOD      = cfg_get(CONFIG, "optuna.tree_method", "hist")
OPTUNA_STORAGE   = cfg_get(CONFIG, "optuna.storage", "sqlite:///optuna.db")
STUDY_BASE       = cfg_get(CONFIG, "optuna.study_name", "prostate_xgb")
RUNS_DIR         = cfg_path(CONFIG, "paths.runs_dir", "runs")

PHASE2_STUDY_NAME = f"phase2_{STUDY_BASE}"
OUTPUT_JSON       = RUNS_DIR / "phase2_best_params.json"

TOP_PCT              = 0.15   # top 15% of Phase 1 trials
N_TRIALS_PHASE2      = 200
NUM_BOOST_ROUND      = 1000
EARLY_STOPPING_ROUNDS = 50

# Phase 1 absolute search bounds — used as safety clamps for Phase 2 ranges.
# Mirrors the suggest_* calls in src/build/optimize.py.
PHASE1_BOUNDS: dict[str, dict] = {
    "max_depth":        {"low": 2,    "high": 8,    "is_int": True,  "log": False},
    "learning_rate":    {"low": 1e-3, "high": 0.2,  "is_int": False, "log": True},
    "subsample":        {"low": 0.6,  "high": 1.0,  "is_int": False, "log": False},
    "colsample_bytree": {"low": 0.6,  "high": 1.0,  "is_int": False, "log": False},
    "min_child_weight": {"low": 1.0,  "high": 10.0, "is_int": False, "log": False},
    "gamma":            {"low": 0.0,  "high": 1.0,  "is_int": False, "log": False},
    "reg_lambda":       {"low": 1e-2, "high": 10.0, "is_int": False, "log": True},
    "reg_alpha":        {"low": 0.0,  "high": 1.0,  "is_int": False, "log": False},
    "num_boost_round":  {"low": 100,  "high": 1000, "is_int": True,  "log": False},
}

# num_boost_round is analyzed but fixed in Phase 2 (1000 + early stopping)
FIXED_IN_PHASE2 = {"num_boost_round"}


# ── Pruning callback (inline — no optuna-integration required) ─────────────────
class XGBoostPruningCallback(xgb.callback.TrainingCallback):
    """Report intermediate AUC to Optuna and raise TrialPruned when needed."""

    def __init__(self, trial: optuna.Trial, metric: str = "test-auc") -> None:
        self._trial = trial
        self._metric = metric

    def after_iteration(self, _model, epoch: int, evals_log: dict) -> bool:
        for data_name, metrics in evals_log.items():
            for metric_name, values in metrics.items():
                key = f"{data_name}-{metric_name}"
                if key == self._metric:
                    self._trial.report(float(values[-1]), step=epoch)
                    if self._trial.should_prune():
                        raise optuna.TrialPruned(
                            f"Trial pruned at iteration {epoch}"
                        )
        return False  # False = do not stop training


# ── Helpers ────────────────────────────────────────────────────────────────────
def _find_phase1_study_name() -> str:
    """Return the Phase 1 study name actually present in the DB."""
    try:
        available = optuna.get_all_study_names(OPTUNA_STORAGE)
    except Exception as exc:
        raise RuntimeError(f"Cannot access Optuna storage '{OPTUNA_STORAGE}': {exc}") from exc

    # optimize.py appends _all to the base name
    for candidate in (f"{STUDY_BASE}_all", STUDY_BASE):
        if candidate in available:
            return candidate

    raise RuntimeError(
        f"Phase 1 study not found in '{OPTUNA_STORAGE}'.\n"
        f"  Expected one of: {STUDY_BASE}_all, {STUDY_BASE}\n"
        f"  Available studies: {available}"
    )


def _load_train_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load train CSV and encode object columns as category (mirrors optimize.py)."""
    df = pd.read_csv(DATA_PATH_TRAIN, sep=";")
    if "Order" in df.columns:
        df = df.drop(columns=["Order"])
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")
    return X, y


# ── Step 1: Analyze Phase 1 top trials ─────────────────────────────────────────
def analyze_phase1(study: optuna.Study) -> dict[str, dict]:
    """
    Filter the top 15% completed trials by AUC and compute per-parameter
    statistics (min, max, mean, median, p10, p90).

    Returns a dict  param → stats_dict.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No COMPLETE trials found in Phase 1 study.")

    n_top = max(1, int(len(completed) * TOP_PCT))
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:n_top]

    print(f"\n[Phase 1] study name      : {study.study_name}")
    print(f"[Phase 1] completed trials : {len(completed)}")
    print(f"[Phase 1] top {TOP_PCT*100:.0f}% analyzed   : {n_top}")
    print(
        f"[Phase 1] AUC range (top)  : "
        f"{top_trials[-1].value:.4f} – {top_trials[0].value:.4f}"
    )

    stats: dict[str, dict] = {}
    rows = []

    for param in PHASE1_BOUNDS:
        values = [t.params[param] for t in top_trials if param in t.params]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        p10, p90 = np.percentile(arr, 10), np.percentile(arr, 90)
        stats[param] = {
            "min":    float(arr.min()),
            "max":    float(arr.max()),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "p10":    float(p10),
            "p90":    float(p90),
        }
        rows.append({
            "param":  param,
            "min":    f"{arr.min():.4g}",
            "max":    f"{arr.max():.4g}",
            "mean":   f"{arr.mean():.4g}",
            "median": f"{np.median(arr):.4g}",
            "p10":    f"{p10:.4g}",
            "p90":    f"{p90:.4g}",
        })

    df_stats = pd.DataFrame(rows).set_index("param")
    print("\n── Top-trial hyperparameter statistics ─────────────────────────────────")
    print(df_stats.to_string())
    print()

    return stats


# ── Step 2: Build Phase 2 ranges ───────────────────────────────────────────────
def build_phase2_ranges(stats: dict[str, dict]) -> dict[str, dict]:
    """
    For each numeric hyperparameter:
        low  = p10 - 10% * (p90 - p10),  clamped to Phase 1 lower bound
        high = p90 + 10% * (p90 - p10),  clamped to Phase 1 upper bound

    Returns a dict  param → {low, high, is_int, log}.
    """
    ranges: dict[str, dict] = {}
    rows = []

    for param, s in stats.items():
        meta = PHASE1_BOUNDS[param]
        p10, p90 = s["p10"], s["p90"]
        spread = p90 - p10

        if spread > 0:
            low  = p10 - 0.10 * spread
            high = p90 + 0.10 * spread
        else:
            # All top-trial values are identical — expand by 10% of the value
            ref  = abs(p10) if p10 != 0 else abs(meta["low"] + meta["high"]) / 2
            low  = p10 - 0.10 * ref
            high = p90 + 0.10 * ref

        # Clamp to Phase 1 absolute safety bounds
        low  = max(low,  meta["low"])
        high = min(high, meta["high"])

        # Guarantee low < high
        if low >= high:
            low, high = meta["low"], meta["high"]

        if meta["is_int"]:
            low  = int(np.floor(low))
            high = int(np.ceil(high))

        ranges[param] = {
            "low":    low,
            "high":   high,
            "is_int": meta["is_int"],
            "log":    meta["log"],
        }
        rows.append({"param": param, "low": f"{low:.4g}", "high": f"{high:.4g}"})

    df_ranges = pd.DataFrame(rows).set_index("param")
    print("── Phase 2 search ranges ────────────────────────────────────────────────")
    print(df_ranges.to_string())
    print()

    return ranges


# ── Step 3: Optuna objective ───────────────────────────────────────────────────
def make_objective(
    dtrain: xgb.DMatrix,
    phase2_ranges: dict[str, dict],
    scale_pos_weight: float,
):
    """Return an Optuna objective closure over the DMatrix and Phase 2 ranges."""

    def objective(trial: optuna.Trial) -> float:
        params: dict = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "tree_method":      TREE_METHOD,
            "booster":          "gbtree",   # fixed — required for pruning compatibility
            "scale_pos_weight": scale_pos_weight,
            "seed":             RANDOM_STATE,
            "nthread":          N_THREADS,
        }

        for param, meta in phase2_ranges.items():
            if param in FIXED_IN_PHASE2:
                continue
            if meta["is_int"]:
                params[param] = trial.suggest_int(param, meta["low"], meta["high"])
            elif meta["log"]:
                params[param] = trial.suggest_float(
                    param, meta["low"], meta["high"], log=True
                )
            else:
                params[param] = trial.suggest_float(param, meta["low"], meta["high"])

        pruning_cb = XGBoostPruningCallback(trial, "test-auc")

        cv_results = xgb.cv(
            params=params,
            dtrain=dtrain,
            nfold=N_FOLDS,
            stratified=True,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            seed=RANDOM_STATE,
            verbose_eval=False,
            callbacks=[pruning_cb],
        )

        return float(cv_results["test-auc-mean"].iloc[-1])

    return objective


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("XGBoost Phase 2 Hyperparameter Optimization")
    print("=" * 72)

    # Load data
    X, y = _load_train_data()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = n_neg / n_pos
    print(
        f"\n[data] shape={X.shape}  pos={n_pos}  neg={n_neg}"
        f"  scale_pos_weight={scale_pos_weight:.4f}"
    )

    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)

    # ── Step 1: analyze Phase 1 ────────────────────────────────────────────────
    phase1_name = _find_phase1_study_name()
    phase1_study = optuna.load_study(study_name=phase1_name, storage=OPTUNA_STORAGE)
    phase1_best_auc = phase1_study.best_value
    stats = analyze_phase1(phase1_study)

    # ── Step 2: build Phase 2 ranges ───────────────────────────────────────────
    phase2_ranges = build_phase2_ranges(stats)

    # ── Steps 3 & 4 disabled — set to True to run optimization ────────────────
    RUN_OPTIMIZATION = False
    if not RUN_OPTIMIZATION:
        print("[dry-run] Steps 3 & 4 skipped. Set RUN_OPTIMIZATION = True to launch.")
        return

    # ── Step 3: run Phase 2 ────────────────────────────────────────────────────
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    phase2_study = optuna.create_study(
        study_name=PHASE2_STUDY_NAME,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,   # resume if interrupted
        direction="maximize",
        pruner=pruner,
    )

    n_done = sum(
        1 for t in phase2_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_remaining = max(0, N_TRIALS_PHASE2 - n_done)
    print(f"[Phase 2] study name : {PHASE2_STUDY_NAME}")
    print(f"[Phase 2] completed  : {n_done}/{N_TRIALS_PHASE2}")

    if n_remaining > 0:
        print(f"[Phase 2] launching {n_remaining} new trials …\n")
        phase2_study.optimize(
            make_objective(dtrain, phase2_ranges, scale_pos_weight),
            n_trials=n_remaining,
        )
    else:
        print("[Phase 2] target already reached — loading results from DB.\n")

    # ── Step 4: results ────────────────────────────────────────────────────────
    best_params = phase2_study.best_params
    best_auc_p2 = phase2_study.best_value
    delta       = best_auc_p2 - phase1_best_auc
    arrow       = "↑" if delta >= 0 else "↓"

    print("\n" + "=" * 72)
    print("Results")
    print("=" * 72)
    print(f"  Best AUC Phase 1  : {phase1_best_auc:.4f}")
    print(f"  Best AUC Phase 2  : {best_auc_p2:.4f}  ({arrow} {abs(delta):.4f})")
    print(f"  Best trial #      : {phase2_study.best_trial.number}")
    print("\n  Best hyperparameters (Phase 2):")
    for k, v in sorted(best_params.items()):
        print(f"    {k:25s} = {v}")

    # Save
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "best_auc_phase2":  best_auc_p2,
        "best_auc_phase1":  phase1_best_auc,
        "best_trial":       phase2_study.best_trial.number,
        "params":           best_params,
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\n[saved] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
