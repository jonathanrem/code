"""
Comprehensive classification metrics from confusion matrix counts.

This script computes TP/TN/FP/FN and derived binary-classification metrics
for one or multiple probability thresholds, then exports tables to CSV/JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, cfg_get, CONFIG
from common.data import load_test_data, align_to_metadata, get_feature_label
from common.model import load_model, load_metadata

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = cfg_path(CONFIG, "paths.model", "runs/final_model_all_removed_biopsy/model.pkl")
METADATA_PATH = cfg_path(CONFIG, "paths.metadata", "runs/final_model_all_removed_biopsy/metadata_all_features_removed_biopsy.json")
THRESHOLDS = [cfg_get(CONFIG, "evaluate.common_threshold", 0.2)]


def get_output_dir(model_path: Path) -> Path:
    model_name = model_path.stem
    if "all_features" in model_name:
        output_dir = ROOT / "outputs" / "all_features"
    elif "top" in model_name:
        output_dir = ROOT / "outputs" / "top_5"
    else:
        output_dir = ROOT / "outputs" / "other"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def to_binary_array(y: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().all():
        y_bin = numeric.astype(int).to_numpy()
        uniques = set(np.unique(y_bin))
        if uniques.issubset({0, 1}):
            return y_bin

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "negative": 0,
        "positive": 1,
    }
    if y_str.isin(mapping).all():
        return y_str.map(mapping).to_numpy(dtype=int)

    raise ValueError(
        "Target labels are not binary 0/1. "
        "Please encode target as 0/1 (or true/false, no/yes)."
    )


def compute_confusion_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    p = tp + fn
    n = tn + fp
    total = p + n
    predicted_positive = tp + fp
    predicted_negative = tn + fn

    prevalence = safe_div(p, total)
    accuracy = safe_div(tp + tn, total)
    tpr = safe_div(tp, p)
    fnr = safe_div(fn, p)
    fpr = safe_div(fp, n)
    tnr = safe_div(tn, n)
    ppv = safe_div(tp, predicted_positive)
    for_rate = safe_div(fn, predicted_negative)
    fdr = safe_div(fp, predicted_positive)
    npv = safe_div(tn, predicted_negative)
    lr_plus = safe_div(tpr, fpr)
    lr_minus = safe_div(fnr, tnr)
    dor = safe_div(lr_plus, lr_minus)
    bm = tpr + tnr - 1 if np.isfinite(tpr) and np.isfinite(tnr) else float("nan")
    mk = ppv + npv - 1 if np.isfinite(ppv) and np.isfinite(npv) else float("nan")
    ba = safe_div(tpr + tnr, 2)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    fm = float(np.sqrt(ppv * tpr)) if np.isfinite(ppv) and np.isfinite(tpr) and ppv >= 0 and tpr >= 0 else float("nan")

    mcc_num = tp * tn - fp * fn
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(mcc_num, mcc_den)

    ts = safe_div(tp, tp + fn + fp)
    if np.isfinite(tpr) and np.isfinite(fpr) and tpr >= 0 and fpr >= 0:
        pt = safe_div(np.sqrt(tpr * fpr) - fpr, tpr - fpr)
    else:
        pt = float("nan")

    return {
        "threshold": float(threshold),
        "total_population": float(total),
        "real_positive_p": float(p),
        "real_negative_n": float(n),
        "predicted_positive": float(predicted_positive),
        "predicted_negative": float(predicted_negative),
        "tp": float(tp),
        "fn": float(fn),
        "fp": float(fp),
        "tn": float(tn),
        "prevalence": prevalence,
        "accuracy_acc": accuracy,
        "tpr_recall_sensitivity": tpr,
        "fnr_miss_rate": fnr,
        "fpr_fall_out": fpr,
        "tnr_specificity": tnr,
        "ppv_precision": ppv,
        "for_false_omission_rate": for_rate,
        "fdr_false_discovery_rate": fdr,
        "npv_negative_predictive_value": npv,
        "lr_positive": lr_plus,
        "lr_negative": lr_minus,
        "dor_diagnostic_odds_ratio": dor,
        "bm_informedness": bm,
        "mk_markedness": mk,
        "ba_balanced_accuracy": ba,
        "f1_score": f1,
        "fm_fowlkes_mallows": fm,
        "mcc_phi": mcc,
        "ts_csi_jaccard": ts,
        "pt_prevalence_threshold": pt,
    }


def metrics_to_table(metrics: dict[str, float]) -> pd.DataFrame:
    metric_labels = [
        ("Threshold", "threshold"),
        ("Total population (P + N)", "total_population"),
        ("Real positive (P)", "real_positive_p"),
        ("Real negative (N)", "real_negative_n"),
        ("Predicted positive", "predicted_positive"),
        ("Predicted negative", "predicted_negative"),
        ("True positive (TP)", "tp"),
        ("False negative (FN)", "fn"),
        ("False positive (FP)", "fp"),
        ("True negative (TN)", "tn"),
        ("Prevalence", "prevalence"),
        ("Accuracy (ACC)", "accuracy_acc"),
        ("TPR / Recall / Sensitivity", "tpr_recall_sensitivity"),
        ("FNR / Miss rate", "fnr_miss_rate"),
        ("FPR / Fall-out", "fpr_fall_out"),
        ("TNR / Specificity", "tnr_specificity"),
        ("PPV / Precision", "ppv_precision"),
        ("FOR / False omission rate", "for_false_omission_rate"),
        ("FDR / False discovery rate", "fdr_false_discovery_rate"),
        ("NPV / Negative predictive value", "npv_negative_predictive_value"),
        ("LR+ / Positive likelihood ratio", "lr_positive"),
        ("LR- / Negative likelihood ratio", "lr_negative"),
        ("DOR / Diagnostic odds ratio", "dor_diagnostic_odds_ratio"),
        ("Informedness (BM)", "bm_informedness"),
        ("Markedness (MK)", "mk_markedness"),
        ("Balanced accuracy (BA)", "ba_balanced_accuracy"),
        ("F1 score", "f1_score"),
        ("Fowlkes-Mallows index (FM)", "fm_fowlkes_mallows"),
        ("Matthews corrcoef (MCC / phi)", "mcc_phi"),
        ("Threat score (TS / CSI / Jaccard)", "ts_csi_jaccard"),
        ("Prevalence threshold (PT)", "pt_prevalence_threshold"),
    ]
    rows = [{"Metric": label, "Value": metrics[key]} for label, key in metric_labels]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)
    label = get_feature_label(metadata)

    X_test, y_test = load_test_data()
    drop_cols = metadata.get("nan_handling", {}).get("dropped_cols", []) or []
    X_test = align_to_metadata(X_test, metadata, drop_cols=drop_cols)

    y_true = to_binary_array(y_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test)

    output_dir = get_output_dir(MODEL_PATH)

    all_metrics = []
    all_tables = []
    for thr in THRESHOLDS:
        metrics = compute_confusion_metrics(y_true, y_proba, threshold=thr)
        all_metrics.append(metrics)

        table = metrics_to_table(metrics)
        table.insert(0, "Threshold", thr)
        all_tables.append(table)

    metrics_wide_df = pd.DataFrame(all_metrics)
    metrics_long_df = pd.concat(all_tables, ignore_index=True)

    table_path = output_dir / "classification_metrics_table.csv"
    wide_path = output_dir / "classification_metrics_wide.csv"
    json_path = output_dir / "classification_metrics.json"

    metrics_long_df.to_csv(table_path, index=False)
    metrics_wide_df.to_csv(wide_path, index=False)

    payload = {
        "model": MODEL_PATH.stem,
        "feature_set": label,
        "thresholds": THRESHOLDS,
        "metrics": all_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved: {table_path}")
    print(f"Saved: {wide_path}")
    print(f"Saved: {json_path}")
    print("\n=== Classification metrics (wide) ===")
    print(metrics_wide_df.to_string(index=False))
