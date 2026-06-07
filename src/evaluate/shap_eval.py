"""
SHAP analysis for XGBoost models.
Saves bar plot, beeswarm summary and JSON importance into each model's
existing outputs/evaluate/{model_name}/ folder.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, CONFIG
from common.data import load_test_data, align_to_metadata
from common.model import load_model, load_metadata

ROOT = Path(__file__).resolve().parent.parent.parent

# XGB models to analyse: display name → (config path key, fallback path)
MODELS = {
    "All_features": (
        "compare.paths.xgb_all_features",
        "runs/all_features_tuned/model.pkl",
    ),
    "No_contralateral": (
        "compare.paths.xgb_no_contra",
        "runs/no_contralateral_tuned/model.pkl",
    ),
}


def _plot_bar(shap_values_only, feature_names, sorted_idx, title, save_path):
    mean_abs_shap = np.abs(shap_values_only).mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(feature_names)), mean_abs_shap[sorted_idx][::-1])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _plot_beeswarm(X_test, shap_values_only, feature_names, sorted_idx, title, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, feat_idx in enumerate(sorted_idx[::-1]):
        feat_values = pd.to_numeric(X_test.iloc[:, feat_idx], errors="coerce").values
        shap_vals = shap_values_only[:, feat_idx]
        feat_norm = (feat_values - np.nanmin(feat_values)) / (
            np.nanmax(feat_values) - np.nanmin(feat_values) + 1e-8
        )
        y_pos = np.full_like(shap_vals, i) + np.random.uniform(-0.2, 0.2, size=len(shap_vals))
        ax.scatter(shap_vals, y_pos, c=feat_norm, cmap="coolwarm", alpha=0.5, s=10)

    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx[::-1]])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title(title)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Feature value (normalized)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def run_shap_for_model(model_name: str, model_path: Path, output_dir: Path) -> None:
    print(f"\n{'='*50}")
    print(f"  SHAP — {model_name}")
    print(f"{'='*50}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_path)
    metadata = load_metadata(model_path.parent / "metadata.json")

    X_test, _ = load_test_data()
    X_test = align_to_metadata(X_test, metadata)

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(
        X_test, feature_names=X_test.columns.tolist(), enable_categorical=True
    )
    shap_values = booster.predict(dmatrix, pred_contribs=True)

    base_value = float(shap_values[0, -1])
    shap_values_only = shap_values[:, :-1]
    feature_names = X_test.columns.tolist()

    mean_abs_shap = np.abs(shap_values_only).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    _plot_bar(
        shap_values_only, feature_names, sorted_idx,
        title=f"SHAP Feature Importance — {model_name}",
        save_path=output_dir / "shap_feature_importance.png",
    )
    _plot_beeswarm(
        X_test, shap_values_only, feature_names, sorted_idx,
        title=f"SHAP Summary — {model_name}",
        save_path=output_dir / "shap_summary_plot.png",
    )

    shap_summary = {
        "base_value": base_value,
        "feature_importance": {
            feature_names[i]: float(mean_abs_shap[i]) for i in sorted_idx
        },
    }
    json_path = output_dir / "shap_summary.json"
    with open(json_path, "w") as f:
        json.dump(shap_summary, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n  Top-3 features:")
    for rank, idx in enumerate(sorted_idx[:3], 1):
        print(f"    {rank}. {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")


def main():
    for model_name, (cfg_key, fallback) in MODELS.items():
        model_path = cfg_path(CONFIG, cfg_key, fallback)
        output_dir = ROOT / "outputs" / "evaluate" / model_name
        run_shap_for_model(model_name, model_path, output_dir)


if __name__ == "__main__":
    main()
