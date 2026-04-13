"""
Confusion matrices at multiple thresholds for a frozen XGBoost model.
Plots one confusion matrix per threshold and prints clinical metrics.
Edit the paths and THRESHOLDS list below and run the script.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, cfg_get, CONFIG
from common.data import load_test_data, align_to_metadata, get_feature_label
from common.model import load_model, load_metadata

MODEL_PATH = cfg_path(CONFIG, "paths.model", "runs/final_model_all_removed_biopsy/model.pkl")
METADATA_PATH = cfg_path(CONFIG, "paths.metadata", "runs/final_model_all_removed_biopsy/metadata_all_features_removed_biopsy.json")
THRESHOLDS = [cfg_get(CONFIG, "evaluate.common_threshold", 0.2)]



def get_output_dir(model_path: Path) -> Path:
    """Determine output directory based on model name."""
    model_name = model_path.stem
    if "all_features" in model_name:
        output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "all_features"
    elif "top" in model_name:
        output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "top_5"
    else:
        output_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "other"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ── Load data ─────────────────────────────────────────────────────────────
model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

X_test, y_test = load_test_data()

drop_cols = metadata.get("nan_handling", {}).get("dropped_cols", []) or []
X_test = align_to_metadata(X_test, metadata, drop_cols=drop_cols)

y_proba = model.predict_proba(X_test)[:, 1]
label = get_feature_label(metadata)
output_dir = get_output_dir(MODEL_PATH)
n_total = len(y_test)

# ── Evaluate at each threshold ────────────────────────────────────────────
n_thresholds = len(THRESHOLDS)
cols = max(2, n_thresholds)
fig, axes = plt.subplots(1, n_thresholds, figsize=(5 * n_thresholds, 5))
if n_thresholds == 1:
    axes = [axes]

print(f"Model: {MODEL_PATH.stem} ({label})")
print(f"Patients: {n_total} | Prevalence: {np.mean(y_test):.3f}")
print("=" * 75)

metrics_rows = []

for i, thr in enumerate(THRESHOLDS):
    y_pred = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    biopsies_avoided = fp + tn  # patients NOT sent to biopsy = predicted negative
    # Biopsies avoided = predicted negatives / total (those who would have been biopsied under "biopsy all")
    biopsies_avoided_pct = (tn + fn) / n_total * 100

    metrics_rows.append({
        "Threshold": f"{thr:.0%}",
        "Sensitivity": f"{sensitivity:.3f}",
        "Specificity": f"{specificity:.3f}",
        "PPV": f"{ppv:.3f}",
        "NPV": f"{npv:.3f}",
        "FN": fn,
        "Biopsies avoided (%)": f"{biopsies_avoided_pct:.1f}%",
    })

    # ── Print per-threshold summary ───────────────────────────────────────
    print(f"\n--- Threshold = {thr:.0%} ---")
    print(f"  Sensitivity (Recall) : {sensitivity:.3f}")
    print(f"  Specificity          : {specificity:.3f}")
    print(f"  PPV (Precision)      : {ppv:.3f}")
    print(f"  NPV                  : {npv:.3f}")
    print(f"  FN (cancers manqués) : {fn}")
    print(f"  Biopsies évitées     : {biopsies_avoided_pct:.1f}%")

    # ── Plot confusion matrix ─────────────────────────────────────────────
    ax = axes[i]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No biopsy", "Biopsy"],
        yticklabels=["No cancer", "Cancer"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Threshold = {thr:.0%}")

fig.suptitle(f"Confusion Matrices — {label} features", fontsize=14, y=1.02)
fig.tight_layout()

output_path = output_dir / "confusion_matrices_multi_threshold.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to {output_path}")
plt.close()

# ── Summary table ─────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("Summary table:")
df_metrics = pd.DataFrame(metrics_rows)
print(df_metrics.to_string(index=False))
