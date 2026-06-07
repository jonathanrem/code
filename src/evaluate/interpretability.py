"""
Publication-quality SHAP interpretability report for the with_contralateral model.

Outputs
-------
  outputs/v2/shap_with_contralateral.csv
  outputs/v2/figures/shap_beeswarm_with_contralateral.png
  outputs/v2/figures/shap_beeswarm_with_contralateral.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, CONFIG
from common.data import load_test_data, align_to_metadata
from common.model import load_model, load_metadata

ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_CFG_KEY = "compare.paths.xgb_all_features"
MODEL_FALLBACK = "runs/with_contralateral/model.pkl"

OUT_DIR = ROOT / "outputs" / "v2"
FIG_DIR = OUT_DIR / "figures"

MAX_BEESWARM_FEATURES = 15
DPI = 300
DIRECTION_THRESHOLD = 0.1


def _compute_shap(model_path: Path):
    model = load_model(model_path)
    metadata = load_metadata(model_path.parent / "metadata.json")

    X_test, y_test = load_test_data()
    X_test = align_to_metadata(X_test, metadata)

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(
        X_test, feature_names=X_test.columns.tolist(), enable_categorical=True
    )
    shap_raw = booster.predict(dmatrix, pred_contribs=True)

    # Last column is the base value (bias term), exclude it
    shap_values = shap_raw[:, :-1]
    feature_names = X_test.columns.tolist()
    return shap_values, feature_names, X_test, y_test


def _direction(mean_signed: float, mean_abs: float) -> str:
    if mean_abs == 0:
        return "mixed"
    if abs(mean_signed) < DIRECTION_THRESHOLD * mean_abs:
        return "mixed"
    return "+" if mean_signed > 0 else "−"


def build_ranking_table(
    shap_values: np.ndarray,
    feature_names: list,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    sorted_idx = np.argsort(mean_abs)[::-1]

    rows = []
    for rank, i in enumerate(sorted_idx, start=1):
        fname = feature_names[i]
        col = X_test[fname] if fname in X_test.columns else None
        miss_pct = float(col.isna().mean() * 100) if col is not None else float("nan")

        abs_val = float(mean_abs[i])
        signed_val = float(mean_signed[i])

        rows.append(
            {
                "rank": rank,
                "feature_name": fname,
                "mean_abs_shap": round(abs_val, 6),
                "mean_shap_signed": round(signed_val, 6),
                "missingness_test_pct": round(miss_pct, 2),
                "direction": _direction(signed_val, abs_val),
            }
        )

    return pd.DataFrame(rows)


def plot_beeswarm(
    shap_values: np.ndarray,
    feature_names: list,
    X_test: pd.DataFrame,
    save_stem: Path,
) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    top_idx = sorted_idx[:MAX_BEESWARM_FEATURES]
    n_feat = len(top_idx)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    rng = np.random.default_rng(42)

    # Plot bottom→top: index 0 = highest-ranked feature at top of y-axis
    for plot_i, feat_idx in enumerate(top_idx[::-1]):
        fname = feature_names[feat_idx]
        shap_vals = shap_values[:, feat_idx]

        raw = pd.to_numeric(
            X_test[fname] if fname in X_test.columns else pd.Series(dtype=float),
            errors="coerce",
        ).values

        fmin, fmax = np.nanmin(raw), np.nanmax(raw)
        feat_norm = np.where(
            np.isnan(raw),
            0.5,
            np.clip((raw - fmin) / (fmax - fmin + 1e-8), 0.0, 1.0),
        )

        # Beeswarm jitter: scale jitter to local point density
        y_jitter = rng.uniform(-0.28, 0.28, size=len(shap_vals))
        y_pos = plot_i + y_jitter

        ax.scatter(
            shap_vals,
            y_pos,
            c=feat_norm,
            cmap="coolwarm",
            alpha=0.45,
            s=9,
            linewidths=0,
            rasterized=True,
            vmin=0,
            vmax=1,
        )

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([feature_names[i] for i in top_idx[::-1]])

    ax.set_xlabel("SHAP value (log-odds)", fontsize=12, labelpad=8)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.9, zorder=0)
    ax.set_ylim(-0.6, n_feat - 0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(
        cmap="coolwarm", norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, shrink=0.55, aspect=20)
    cbar.set_label("Feature value", fontsize=10, labelpad=6)
    cbar.set_ticks([0.02, 0.98])
    cbar.set_ticklabels(["Low", "High"], fontsize=9)
    cbar.outline.set_linewidth(0.6)

    fig.tight_layout()

    for suffix in (".png", ".pdf"):
        out = save_stem.with_suffix(suffix)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {out}")

    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model_path = cfg_path(CONFIG, MODEL_CFG_KEY, MODEL_FALLBACK)
    print(f"[interpretability] model : {model_path}")

    shap_values, feature_names, X_test, _ = _compute_shap(model_path)
    n_samples, n_feat = shap_values.shape
    print(
        f"[interpretability] SHAP computed — {n_feat} features, {n_samples} samples"
    )

    df = build_ranking_table(shap_values, feature_names, X_test)
    csv_path = OUT_DIR / "shap_with_contralateral.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[interpretability] Ranking table -> {csv_path}")
    print(df.to_string(index=False))

    beeswarm_stem = FIG_DIR / "shap_beeswarm_with_contralateral"
    print(f"\n[interpretability] Beeswarm plot (top {MAX_BEESWARM_FEATURES} features):")
    plot_beeswarm(shap_values, feature_names, X_test, beeswarm_stem)

    print("\n[interpretability] Done.")


if __name__ == "__main__":
    main()
