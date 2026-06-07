"""
Produce outputs/v2/table_auc_main.csv
======================================
Models: with_contralateral (18 feat) | parsimonious (6 feat, reference)
        Peeters | Kinnaird | ERSPC | logistic_regression

Per model:
  - AUROC point estimate
  - 95 % CI via stratified bootstrap (n=1000, seed=2, patient-level resampling)
  - DeLong paired test vs parsimonious on the intersection of scorable patients

Columns: [model, n_scored, auc, auc_lo, auc_hi,
          delta_vs_parsimonious, delta_lo, delta_hi, pvalue_delong, n_delong]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.append(str(_HERE.parent))   # src/
sys.path.append(str(_HERE))          # src/compare/

from common.data import load_compare_data  # noqa: E402
from models import (                        # noqa: E402
    get_probabilities_xgb_all_features,
    get_probabilities_xgb_parsimonious,
    get_probabilities_peter2022,
    get_probabilities_kinnaird,
    get_probabilities_erspc,
    get_probabilities_logreg_new,
)

_OUT  = _ROOT / "outputs" / "v2"
_PRED = _OUT / "predictions"

MODEL_SPECS = [
    ("with_contralateral",  get_probabilities_xgb_all_features),
    ("parsimonious",        get_probabilities_xgb_parsimonious),
    ("Peeters",             get_probabilities_peter2022),
    ("Kinnaird",            get_probabilities_kinnaird),
    ("ERSPC",               get_probabilities_erspc),
    ("logistic_regression", get_probabilities_logreg_new),
]

REFERENCE = "parsimonious"
N_BOOT    = 1000
SEED      = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _align(y: pd.Series, p: pd.Series):
    """Return (y, p) restricted to their common non-NaN index."""
    idx = y.dropna().index.intersection(p.dropna().index)
    return y.loc[idx], p.loc[idx]


def _bootstrap_auc_ci(y: pd.Series, p: pd.Series,
                      n: int = N_BOOT, seed: int = SEED):
    """Stratified bootstrap 95 % CI for AUROC (patient-level resampling)."""
    rng  = np.random.default_rng(seed)
    yv   = y.values
    pv   = p.values
    pos  = np.where(yv == 1)[0]
    neg  = np.where(yv == 0)[0]

    boot_aucs = []
    for _ in range(n):
        idx = np.concatenate([
            pos[rng.integers(0, len(pos), len(pos))],
            neg[rng.integers(0, len(neg), len(neg))],
        ])
        yb, pb = yv[idx], pv[idx]
        if 0 < yb.sum() < len(yb):
            boot_aucs.append(roc_auc_score(yb, pb))

    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    return float(lo), float(hi)


def _delong_vs_ref(y_ref: pd.Series, p_ref: pd.Series,
                   y_mod: pd.Series, p_mod: pd.Series):
    """
    Paired DeLong test on the intersection of patients scored by both models.

    Returns (delta, delta_lo, delta_hi, pval, n_common).
    CI derived from z-statistic: SE = |ΔAUC / z|, CI = ΔAUC ± 1.96·SE.
    """
    base = pd.DataFrame({"y": y_ref, "p_ref": p_ref}).dropna()
    ext  = p_mod.dropna().rename("p_mod")
    df   = base.join(ext, how="inner").dropna()

    if len(df) < 5:
        return np.nan, np.nan, np.nan, np.nan, 0

    y_c   = df["y"]
    p_c_r = df["p_ref"]
    p_c_m = df["p_mod"]

    auc_mod = roc_auc_score(y_c, p_c_m)
    auc_ref = roc_auc_score(y_c, p_c_r)
    delta   = auc_mod - auc_ref

    z, pval, *_ = Delong_test(y_c, p_c_m, p_c_r)
    z, pval = float(z), float(pval)

    se      = abs(delta / z) if abs(z) > 1e-10 else 0.0
    d_lo    = delta - 1.96 * se
    d_hi    = delta + 1.96 * se

    return delta, d_lo, d_hi, pval, len(df)


def _fmt_p(val) -> str:
    if isinstance(val, str) or val is None:
        return str(val)
    if np.isnan(val):
        return "NA"
    return f"{val:.2e}" if val < 0.001 else f"{val:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _PRED.mkdir(parents=True, exist_ok=True)

    X, y = load_compare_data()
    test_df = X.copy()
    test_df["outcome"] = y

    # ── 1. Score all models ──────────────────────────────────────────────────
    preds: dict[str, tuple[pd.Series, pd.Series]] = {}
    for name, fn in MODEL_SPECS:
        print(f"[{name}] scoring…", end=" ", flush=True)
        try:
            y_m, p_m = fn(test_df)
            y_m, p_m = _align(y_m, p_m)
            preds[name] = (y_m, p_m)
            pd.DataFrame({"y_true": y_m, "y_proba": p_m}).to_csv(
                _PRED / f"{name}_test.csv"
            )
            print(f"n={len(p_m)}  AUC={roc_auc_score(y_m, p_m):.4f}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if REFERENCE not in preds:
        raise RuntimeError(f"Reference model '{REFERENCE}' failed — cannot continue.")

    y_ref, p_ref = preds[REFERENCE]

    # ── 2. Build summary table ───────────────────────────────────────────────
    rows = []
    for name, (y_m, p_m) in preds.items():
        n_scored = len(p_m)
        auc_pt   = roc_auc_score(y_m, p_m)
        lo, hi   = _bootstrap_auc_ci(y_m, p_m)

        if name == REFERENCE:
            row = {
                "model":                 name,
                "n_scored":              n_scored,
                "auc":                   round(auc_pt, 3),
                "auc_lo":                round(lo, 3),
                "auc_hi":                round(hi, 3),
                "delta_vs_parsimonious": 0.0,
                "delta_lo":              0.0,
                "delta_hi":              0.0,
                "pvalue_delong":         "ref",
                "n_delong":              n_scored,
            }
        else:
            delta, d_lo, d_hi, pval, n_dl = _delong_vs_ref(y_ref, p_ref, y_m, p_m)
            row = {
                "model":                 name,
                "n_scored":              n_scored,
                "auc":                   round(auc_pt, 3),
                "auc_lo":                round(lo, 3),
                "auc_hi":                round(hi, 3),
                "delta_vs_parsimonious": round(delta, 3) if not np.isnan(delta) else np.nan,
                "delta_lo":              round(d_lo, 3)  if not np.isnan(d_lo)  else np.nan,
                "delta_hi":              round(d_hi, 3)  if not np.isnan(d_hi)  else np.nan,
                "pvalue_delong":         _fmt_p(pval),
                "n_delong":              n_dl,
            }
        rows.append(row)

    result = (
        pd.DataFrame(rows)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )

    out_path = _OUT / "table_auc_main.csv"
    result.to_csv(out_path, index=False)
    print(f"\n[OK] saved -> {out_path}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
