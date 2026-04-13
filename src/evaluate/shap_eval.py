"""
SHAP analysis for a frozen XGBoost model on a CSV test set.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, CONFIG
from common.data import load_test_data, align_to_metadata, get_feature_label
from common.model import load_model, load_metadata

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = cfg_path(CONFIG, "paths.model", "runs/final_model_all_removed_biopsy/model.pkl")
METADATA_PATH = cfg_path(CONFIG, "paths.metadata", "runs/final_model_all_removed_biopsy/metadata_all_features_removed_biopsy.json")


def get_output_dir() -> Path:
    output_dir = ROOT / "outputs" / "evaluate"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)
label = get_feature_label(metadata)

X_test, y_test = load_test_data()

drop_cols = metadata.get("nan_handling", {}).get("dropped_cols", []) or []
X_test = align_to_metadata(X_test, metadata, drop_cols=drop_cols)

# Get the booster and compute SHAP values using XGBoost's native method
booster = model.get_booster()
import xgboost as xgb

# Create DMatrix for XGBoost native SHAP
dmatrix = xgb.DMatrix(X_test, feature_names=X_test.columns.tolist(), enable_categorical=True)

# Get SHAP values using XGBoost's native implementation (pred_contribs=True)
shap_values = booster.predict(dmatrix, pred_contribs=True)

# Last column is the base value (expected value), remove it for plotting
base_value = shap_values[0, -1]
shap_values_only = shap_values[:, :-1]

feature_names = X_test.columns.tolist()

# Get output directory
output_dir = get_output_dir()

# Bar plot: mean absolute SHAP value per feature
mean_abs_shap = np.abs(shap_values_only).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_names)), mean_abs_shap[sorted_idx][::-1])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx][::-1])
plt.xlabel("Mean |SHAP value|")
plt.title(f"SHAP Feature Importance ({label})")
plt.tight_layout()
plt.savefig(output_dir / "shap_feature_importance.png", dpi=150)
print(f"Saved: {output_dir / 'shap_feature_importance.png'}")
plt.show()

# Summary plot (beeswarm-style) - manual implementation to avoid colorbar issues
plt.figure(figsize=(10, 8))
for i, feat_idx in enumerate(sorted_idx[::-1]):
    feat_values = pd.to_numeric(X_test.iloc[:, feat_idx], errors="coerce").values
    shap_vals = shap_values_only[:, feat_idx]
    # Normalize feature values for coloring
    feat_norm = (feat_values - np.nanmin(feat_values)) / (np.nanmax(feat_values) - np.nanmin(feat_values) + 1e-8)
    # Add jitter to y position
    y_pos = np.full_like(shap_vals, i) + np.random.uniform(-0.2, 0.2, size=len(shap_vals))
    plt.scatter(shap_vals, y_pos, c=feat_norm, cmap="coolwarm", alpha=0.5, s=10)

plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx[::-1]])
plt.xlabel("SHAP value (impact on model output)")
plt.title(f"SHAP Summary Plot ({label})")
plt.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label("Feature value (normalized)")
plt.tight_layout()
plt.savefig(output_dir / "shap_summary_plot.png", dpi=150)
print(f"Saved: {output_dir / 'shap_summary_plot.png'}")
plt.show()

# Waterfall for first sample
idx = 0
print(f"\n=== SHAP values for sample {idx} ===")
print(f"Base value: {base_value:.4f}")
for i, (name, val) in enumerate(zip(feature_names, shap_values_only[idx])):
    if abs(val) > 0.01:
        print(f"  {name}: {val:+.4f}")
print(f"Prediction (log-odds): {shap_values[idx].sum():.4f}")

# Save SHAP values summary to JSON
shap_summary = {
    "base_value": float(base_value),
    "feature_importance": {
        feature_names[i]: float(mean_abs_shap[i]) for i in sorted_idx
    },
}
with open(output_dir / "shap_summary.json", "w") as f:
    json.dump(shap_summary, f, indent=2)
print(f"Saved: {output_dir / 'shap_summary.json'}")
