"""
Precision-Recall curve for a frozen XGBoost model on a CSV test set.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, CONFIG
from common.data import load_test_data, align_to_metadata, get_feature_label
from common.model import load_model, load_metadata

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = cfg_path(CONFIG, "paths.model", "runs/final_model_all_removed_biopsy/model.pkl")
METADATA_PATH = cfg_path(CONFIG, "paths.metadata", "runs/final_model_all_removed_biopsy/metadata_all_features_removed_biopsy.json")


def get_output_dir(model_path: Path) -> Path:
    """Determine output directory based on model name."""
    model_name = model_path.stem
    if "all_features" in model_name:
        output_dir = ROOT / "outputs" / "all_features"
    elif "top" in model_name:
        output_dir = ROOT / "outputs" / "top_5"
    else:
        output_dir = ROOT / "outputs" / "other"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


model = load_model(MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

X_test, y_test = load_test_data()

drop_cols = metadata.get("nan_handling", {}).get("dropped_cols", []) or []
X_test = align_to_metadata(X_test, metadata, drop_cols=drop_cols)

y_pred = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred)
auc_score = auc(recall, precision)
label = get_feature_label(metadata)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker=".", label="PR Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve ({label}, AUC = {auc_score:.2f})")
plt.legend()
plt.tight_layout()

# Save figure
output_dir = get_output_dir(MODEL_PATH)
output_path = output_dir / "precision_recall_curve.png"
plt.savefig(output_path, dpi=150)
print(f"AUPRC: {auc_score:.4f}")
print(f"Figure saved to {output_path}")
plt.show()
