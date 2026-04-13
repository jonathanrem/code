"""
Clinical evaluation script for binary prediction models.

Retains only methodologically sound, clinically meaningful evaluation:
  - AUROC (discrimination, single value)
  - Calibration (LOESS curve, O:E ratio, intercept, slope, ECI, ICI, ECE)
  - Risk distribution (violin plot by outcome class)
  - Decision Curve Analysis (net benefit vs threshold)
  - Expected Cost analysis (FP/FN trade-off)

Based on: Van Calster et al. (2025) — Performance Measures for Binary Classification.
Refactored from the KU Leuven reference notebook.

Inputs:
  - y_true: binary outcome (0/1)
  - y_proba: predicted probabilities (0–1)
  - optional threshold grid and cost grid

Outputs:
  - Saved plots (PNG)
  - Printed key metrics
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.nonparametric.smoothers_lowess import lowess

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, cfg_get, CONFIG
from common.data import load_test_data, align_to_metadata, get_feature_label
from common.model import load_model, load_metadata

MODEL_PATH = cfg_path(CONFIG, "paths.model", "runs/final_model_all_removed_biopsy/model.pkl")
METADATA_PATH = cfg_path(CONFIG, "paths.metadata", "runs/final_model_all_removed_biopsy/metadata_all_features_removed_biopsy.json")
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "evaluate"

CLASSIFICATION_THRESHOLD = cfg_get(CONFIG, "evaluate.common_threshold", 0.2)
COST_RATIO = 9          # FN is this many times worse than FP
N_GROUPS = 10            # bins for grouped calibration / ECE
RANDOM_SEED = 42
N_BOOTSTRAP = 1000       # number of bootstrap iterations

plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "font.size": 12,
})


# ---------------------------------------------------------------------------
# 1. AUROC — single discrimination metric
# ---------------------------------------------------------------------------
def compute_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Clinical purpose: quantify the model's ability to rank patients with
    the condition higher than those without. A single summary of
    discrimination — no threshold dependence.
    """
    return float(roc_auc_score(y_true, y_proba))


# ---------------------------------------------------------------------------
# 2. Calibration assessment
# ---------------------------------------------------------------------------
def compute_calibration_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, n_groups: int = 10
) -> dict:
    """
    Clinical purpose: assess whether predicted probabilities match observed
    event rates. A well-calibrated model lets clinicians trust the predicted
    risk value itself, not just the ranking.

    Returns: O:E ratio, calibration intercept, calibration slope,
             ECI, ICI, ECE.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)

    # O:E ratio
    oe_ratio = float(np.sum(y) / np.sum(p))

    # Calibration intercept (GLM with logit(p) as offset)
    logit_p = np.log(p / (1 - p))
    try:
        glm_int = GLM(y, np.ones_like(y), family=Binomial(), offset=logit_p)
        int_result = glm_int.fit()
        intercept = float(int_result.params[0])
    except Exception:
        intercept = float("nan")

    # Calibration slope (logistic regression on logit probabilities)
    sl_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    sl_model.fit(logit_p.reshape(-1, 1), y)
    slope = float(sl_model.coef_[0][0])

    # Flexible calibration via LOESS
    flc = lowess(y, p, frac=0.75, return_sorted=False)

    # ECI — expected calibration index (continuous, normalised)
    baseline = np.full(y.shape, np.mean(y), dtype=float)
    eci = float(np.mean((flc - p) ** 2) / np.mean((baseline - p) ** 2))

    # ICI — integrated calibration index (mean absolute deviation)
    ici = float(np.mean(np.abs(flc - p)))

    # ECE — expected calibration error (grouped)
    df = pd.DataFrame({"p": p, "y": y})
    df["q"] = pd.qcut(df["p"], q=n_groups, duplicates="drop")
    grouped = df.groupby("q", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean")
    )
    ece = float((grouped["mean_p"] - grouped["mean_y"]).abs().mean())

    return {
        "O:E ratio": oe_ratio,
        "Cal. intercept": intercept,
        "Cal. slope": slope,
        "ECI": eci,
        "ICI": ici,
        "ECE": ece,
    }


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_groups: int = 10,
    save_path: Path | None = None,
) -> None:
    """
    Clinical purpose: visualise agreement between predicted risk and
    observed proportion. The diagonal = perfect calibration. Deviations
    indicate over- or under-estimation of risk.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Grouped calibration points
    df = pd.DataFrame({"p": p, "y": y})
    df["q"] = pd.qcut(df["p"], q=n_groups, duplicates="drop")
    grouped = df.groupby("q", observed=True).agg(
        mean_p=("p", "mean"), mean_y=("y", "mean")
    )
    ax.scatter(grouped["mean_p"], grouped["mean_y"],
               color="black", s=40, marker="^", facecolors="none",
               label="Grouped calibration")

    # Ideal diagonal
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Ideal")

    ax.set_xlabel("Estimated probability")
    ax.set_ylabel("Observed proportion")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Calibration plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Risk distribution
# ---------------------------------------------------------------------------
def plot_risk_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """
    Clinical purpose: visualise how predicted risks are distributed across
    true outcome groups. Good separation = the model assigns high
    probabilities to actual positives and low to negatives.
    """
    df = pd.DataFrame({
        "Outcome": pd.Categorical(
            np.where(y_true == 1, "Positive", "Negative"),
            categories=["Negative", "Positive"],
            ordered=True,
        ),
        "Predicted probability": y_proba,
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(x="Outcome", y="Predicted probability", data=df,
                   inner=None, color="white", ax=ax)
    sns.stripplot(x="Outcome", y="Predicted probability", data=df,
                  jitter=0.1, color="black", size=2, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylabel("Estimated risk")
    ax.set_xlabel("")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Risk distribution plot saved to {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# 4. Decision Curve Analysis
# ---------------------------------------------------------------------------
def decision_curve_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
    save_path: Path | None = None,
) -> dict:
    """
    Clinical purpose: evaluate whether using the model to guide biopsy
    decisions leads to better outcomes (more true positives caught per
    unnecessary biopsy) than default strategies (treat-all / treat-none)
    across a range of decision thresholds.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y)
    prevalence = np.mean(y)

    # Net benefit for the model
    nb_model = np.array([
        np.sum((p >= t) & (y == 1)) / n - np.sum((p >= t) & (y == 0)) / n * (t / (1 - t))
        for t in thresholds
    ])

    # Net benefit: treat all
    nb_all = np.array([
        prevalence - (1 - prevalence) * (t / (1 - t))
        for t in thresholds
    ])

    # Standardised net benefit (divided by prevalence)
    snb_model = nb_model / prevalence

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A — raw net benefit
    axes[0].plot(thresholds, nb_model, color="black", lw=2, label="Model")
    axes[0].plot(thresholds, nb_all, color="gray", lw=1, label="Treat all")
    axes[0].axhline(0, color="red", lw=0.5, label="Treat none")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Net Benefit")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(-0.05, max(0.5, prevalence + 0.05))
    axes[0].legend(loc="upper right")
    axes[0].set_title("A — Decision Curve")

    # Panel B — standardised net benefit
    axes[1].plot(thresholds, snb_model, color="black", lw=2, label="Model")
    axes[1].plot(thresholds, nb_all / prevalence, color="gray", lw=1, label="Treat all")
    axes[1].axhline(0, color="red", lw=0.5, label="Treat none")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Standardised Net Benefit")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].set_title("B — Standardised Decision Curve")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Decision curve plot saved to {save_path}")
    plt.show()

    return {"thresholds": thresholds.tolist(), "net_benefit": nb_model.tolist()}


# ---------------------------------------------------------------------------
# 5. Expected Cost analysis
# ---------------------------------------------------------------------------
def expected_cost_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_grid: np.ndarray | None = None,
    save_path: Path | None = None,
) -> dict:
    """
    Clinical purpose: determine the minimum misclassification cost
    achievable by the model across different assumptions about the
    relative cost of a false positive vs false negative. Useful when
    the cost ratio is uncertain or debated among clinicians.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    prevalence = np.mean(y)

    if cost_grid is None:
        cost_grid = np.arange(1, 100) / 100  # normalised cost of FP

    risksort = np.sort(p)
    ec_model = np.full(len(cost_grid), np.nan)
    ec_threshold = np.full(len(cost_grid), np.nan)

    for i, cost_fp in enumerate(cost_grid):
        cost_fn = 1 - cost_fp
        ec_values = np.array([
            (np.sum(p[y == 1] < t) / np.sum(y == 1)) * prevalence * cost_fn
            + (np.sum(p[y == 0] >= t) / np.sum(y == 0)) * (1 - prevalence) * cost_fp
            for t in risksort
        ])
        best = np.argmin(ec_values)
        ec_model[i] = ec_values[best]
        ec_threshold[i] = risksort[best]

    # Reference strategies
    # Treat all: FN=0, FP = all negatives
    ec_treat_all = cost_grid * (1 - prevalence) / prevalence  # normalised
    # Treat none: FN = all positives, FP=0
    ec_treat_none = (1 - cost_grid) * prevalence / (1 - prevalence)  # normalised

    # Correct normalisation: match KU Leuven formulation
    ec_treat_all = cost_grid * (prevalence / (1 - prevalence))
    ec_treat_none = (1 - cost_grid) * ((1 - prevalence) / prevalence)

    x_vals = 1 - cost_grid  # normalised cost of FN

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_vals, ec_model, color="black", lw=2, label="Model")
    ax.plot(x_vals, ec_treat_all, color="gray", lw=2, label="Treat all")
    ax.plot(x_vals, ec_treat_none, color="gray", lw=1, linestyle="--", label="Treat none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.5, np.nanmax(ec_model) + 0.05))
    ax.set_xlabel("Normalised cost of false negative")
    ax.set_ylabel("Expected cost")
    ax.legend(loc="upper right")
    ax.set_title("Expected Cost Analysis")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Expected cost plot saved to {save_path}")
    plt.show()

    return {
        "cost_grid": cost_grid.tolist(),
        "expected_cost": ec_model.tolist(),
        "optimal_thresholds": ec_threshold.tolist(),
    }


# ---------------------------------------------------------------------------
# 6. Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_evaluation(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    n_groups: int = 10,
    cut: float = 0.1,
    costratio: float = 9.0,
) -> pd.DataFrame:
    """
    Clinical purpose: quantify uncertainty around all evaluation metrics.
    Uses percentile bootstrap (resampling with replacement) to produce
    95% confidence intervals, as in Van Calster et al. (2025).
    """
    np.random.seed(seed)
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    n = len(y)

    # Columns: AUROC + 6 calibration + NB + sNB
    col_names = [
        "AUROC",
        "O:E ratio", "Cal. intercept", "Cal. slope", "ECI", "ICI", "ECE",
        "Net benefit", "Standardized net benefit",
    ]
    boot_results = pd.DataFrame(np.nan, index=range(n_bootstrap), columns=col_names)

    for i in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y[idx]
        p_b = p[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue

        # AUROC
        boot_results.iloc[i, 0] = compute_auc(y_b, p_b)

        # Calibration
        try:
            cal = compute_calibration_metrics(y_b, p_b, n_groups=n_groups)
            boot_results.iloc[i, 1] = cal["O:E ratio"]
            boot_results.iloc[i, 2] = cal["Cal. intercept"]
            boot_results.iloc[i, 3] = cal["Cal. slope"]
            boot_results.iloc[i, 4] = cal["ECI"]
            boot_results.iloc[i, 5] = cal["ICI"]
            boot_results.iloc[i, 6] = cal["ECE"]
        except Exception:
            pass

        # Net benefit at threshold
        tp = np.sum((p_b >= cut) & (y_b == 1))
        fp = np.sum((p_b >= cut) & (y_b == 0))
        nb = tp / n - fp / n * (cut / (1 - cut))
        boot_results.iloc[i, 7] = nb
        boot_results.iloc[i, 8] = nb / np.mean(y_b)

    # Point estimates
    point_auroc = compute_auc(y, p)
    point_cal = compute_calibration_metrics(y, p, n_groups=n_groups)
    tp = np.sum((p >= cut) & (y == 1))
    fp = np.sum((p >= cut) & (y == 0))
    point_nb = tp / n - fp / n * (cut / (1 - cut))
    point_snb = point_nb / np.mean(y)

    point_estimates = [
        point_auroc,
        point_cal["O:E ratio"], point_cal["Cal. intercept"], point_cal["Cal. slope"],
        point_cal["ECI"], point_cal["ICI"], point_cal["ECE"],
        point_nb, point_snb,
    ]

    # 95% CI (percentile method)
    summary = pd.DataFrame(index=col_names, columns=["Point estimate", "LCL", "UCL"])
    for j, name in enumerate(col_names):
        vals = boot_results.iloc[:, j].dropna()
        summary.loc[name, "Point estimate"] = round(point_estimates[j], 4)
        summary.loc[name, "LCL"] = round(float(np.percentile(vals, 2.5)), 4)
        summary.loc[name, "UCL"] = round(float(np.percentile(vals, 97.5)), 4)

    return summary


# ---------------------------------------------------------------------------
# 7. ROC curve with bootstrap 95% CI
# ---------------------------------------------------------------------------
def plot_roc_with_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    save_path: Path | None = None,
) -> dict:
    """
    Clinical purpose: visualise discrimination with uncertainty.
    The shaded band shows the 95% percentile bootstrap CI on the ROC curve,
    interpolated on a common FPR grid so all curves are comparable.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_proba, dtype=float)
    np.random.seed(seed)
    n = len(y)

    mean_fpr = np.linspace(0, 1, 300)
    boot_tprs = []
    boot_aucs = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        y_b, p_b = y[idx], p[idx]
        if len(np.unique(y_b)) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_b, p_b)
        interp_tpr = interp1d(fpr_b, tpr_b, kind="linear", bounds_error=False,
                              fill_value=(0.0, 1.0))(mean_fpr)
        boot_tprs.append(interp_tpr)
        boot_aucs.append(roc_auc_score(y_b, p_b))

    boot_tprs = np.array(boot_tprs)
    tpr_lower = np.percentile(boot_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(boot_tprs, 97.5, axis=0)

    auc_point = roc_auc_score(y, p)
    auc_lcl = float(np.percentile(boot_aucs, 2.5))
    auc_ucl = float(np.percentile(boot_aucs, 97.5))

    fpr_main, tpr_main, _ = roc_curve(y, p)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_main, tpr_main, color="black", lw=2,
            label=f"AUROC = {auc_point:.3f} [95% CI: {auc_lcl:.3f}–{auc_ucl:.3f}]")
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper,
                    color="black", alpha=0.15, label="95% CI (bootstrap)")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random")
    ax.set_xlabel("1 − Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    plt.show()

    return {"auroc": auc_point, "lcl": auc_lcl, "ucl": auc_ucl}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model + data
    model = load_model(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)
    label = get_feature_label(metadata)

    X_test, y_test = load_test_data()
    drop_cols = metadata.get("nan_handling", {}).get("dropped_cols", []) or []
    X_test = align_to_metadata(X_test, metadata, drop_cols=drop_cols)

    y_true = y_test.values
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"Model: {MODEL_PATH.stem} ({label})")
    print(f"Patients: {len(y_true)} | Prevalence: {np.mean(y_true):.3f}")
    print("=" * 60)

    # 1. AUROC
    auroc = compute_auc(y_true, y_proba)
    print(f"\nAUROC: {auroc:.4f}")

    # 2. Calibration
    cal = compute_calibration_metrics(y_true, y_proba, n_groups=N_GROUPS)
    print("\nCalibration metrics:")
    for k, v in cal.items():
        print(f"  {k}: {v:.4f}")

    plot_calibration(y_true, y_proba, n_groups=N_GROUPS, save_path=OUTPUT_DIR / "calibration_curve.png")

    # 3. Risk distribution
    plot_risk_distribution(y_true, y_proba,
                           save_path=OUTPUT_DIR / "risk_distribution.png")

    # 4. Decision Curve Analysis
    dca_results = decision_curve_analysis(
        y_true, y_proba,
        save_path=OUTPUT_DIR / "decision_curve.png",
    )

    # 5. Expected Cost
    ec_results = expected_cost_analysis(
        y_true, y_proba,
        save_path=OUTPUT_DIR / "expected_cost.png",
    )

    # 6. ROC curve with 95% CI
    print(f"\nROC curve with bootstrap CI ({N_BOOTSTRAP} iterations)...")
    roc_ci = plot_roc_with_ci(
        y_true, y_proba,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED,
        save_path=OUTPUT_DIR / "roc_curve.png",
    )

    # 7. Bootstrap metrics table
    print(f"\nBootstrap metrics ({N_BOOTSTRAP} iterations)...")
    boot_ci = bootstrap_evaluation(
        y_true, y_proba,
        n_bootstrap=N_BOOTSTRAP,
        seed=RANDOM_SEED,
        n_groups=N_GROUPS,
        cut=CLASSIFICATION_THRESHOLD,
        costratio=COST_RATIO,
    )
    print("\n95% Confidence Intervals:")
    print(boot_ci.to_string())

    # Save all numeric results
    results = {
        "model": MODEL_PATH.stem,
        "feature_set": label,
        "n_patients": int(len(y_true)),
        "prevalence": float(np.mean(y_true)),
        "auroc": auroc,
        "auroc_ci": roc_ci,
        "calibration": cal,
        "bootstrap_ci": boot_ci.to_dict(orient="index"),
    }
    results_path = OUTPUT_DIR / "clinical_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
