"""
Generate 6 publication-quality figures for TFE Chapter 6.

Outputs (outputs/v2/figures/, PNG + PDF, 300 dpi):
  shap_bar_with_contralateral
  calibration_loess_overlay
  roc_overlay
  forest_subgroups_with_contralateral
  confmat_t020
  proba_hist_with_contralateral
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parent.parent.parent
PRED_DIR = ROOT / "outputs" / "v2" / "predictions"
OUT_DIR  = ROOT / "outputs" / "v2" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

#µµµµ

MODEL_META = {
    # key → (display_name, color, linestyle, linewidth)
    "with_contralateral":  ("With contralateral", "tab:red",    "-",  1.8),
    "parsimonious":        ("Parsimonious",        "tab:brown",  "-",  1.8),
    "logistic_regression": ("Logistic regression", "tab:green",  "-",  1.8),
    "Peeters":             ("Peeters 2022",         "#7f7f7f",   "--", 1.4),
    "Kinnaird":            ("Kinnaird 2021",        "tab:orange","--", 1.4),
    "ERSPC":               ("ERSPC-RC",             "tab:purple","--", 1.4),
}

RC = {
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "font.family":       "sans-serif",
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
}

N_BOOT = 1000
SEED   = 2

FEAT_LABELS = {
    "psa_density":              "PSA density",
    "pirads":                   "PI-RADS",
    "age":                      "Age",
    "prostate_volume":          "Prostate volume",
    "contralateral_suspicious": "Contra. suspicious",
    "ant":                      "Anterior zone",
    "diameter":                 "Lesion diameter",
    "clinical_stage":           "Clinical stage",
    "contralateral_diameter":   "Contra. diameter",
    "psa":                      "PSA",
    "suspicious_trus":          "Suspicious TRUS",
    "base":                     "Base zone",
    "post":                     "Posterior zone",
    "median":                   "Median zone",
    "prev_neg_trus_biopsy":     "Prior neg. biopsy",
    "nb_susp_lesions":          "# suspicious lesions",
    "apical":                   "Apical zone",
    "family_history":           "Family history",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[OK] {path.name}")


def _load_pred(name: str) -> pd.DataFrame:
    return pd.read_csv(PRED_DIR / f"{name}_test.csv", index_col=0)


def _strip_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)


def _loess_with_ci(p: np.ndarray, y: np.ndarray,
                   frac: float = 0.15,
                   n_boot: int = N_BOOT,
                   seed: int = SEED,
                   n_grid: int = 300):
    """LOESS fit + percentile bootstrap CI.

    Returns (p_sorted, loess_fit, p_grid, ci_lo, ci_hi, ici).
    """
    rng     = np.random.default_rng(seed)
    sort_i  = np.argsort(p)
    p_s     = p[sort_i]
    loess_s = lowess(y[sort_i], p_s, frac=frac, return_sorted=False)
    ici     = float(np.mean(np.abs(loess_s - p_s)))

    p_grid      = np.linspace(float(p_s[0]), float(p_s[-1]), n_grid)
    boot_curves = []
    n           = len(p)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        sb = np.argsort(pb)
        fb = lowess(yb[sb], pb[sb], frac=frac, return_sorted=False)
        fn = interp1d(pb[sb], fb, kind="linear",
                      bounds_error=False,
                      fill_value=(float(fb[0]), float(fb[-1])))
        boot_curves.append(np.clip(fn(p_grid), 0.0, 1.0))

    arr   = np.array(boot_curves)
    ci_lo = np.percentile(arr, 2.5,  axis=0)
    ci_hi = np.percentile(arr, 97.5, axis=0)
    return p_s, loess_s, p_grid, ci_lo, ci_hi, ici


def _auc_ci(y: np.ndarray, p: np.ndarray,
            n_boot: int = N_BOOT, seed: int = SEED):
    """Stratified bootstrap 95% CI for AUROC."""
    rng  = np.random.default_rng(seed)
    pos  = np.where(y == 1)[0]
    neg  = np.where(y == 0)[0]
    boot = []
    for _ in range(n_boot):
        idx = np.concatenate([
            pos[rng.integers(0, len(pos), len(pos))],
            neg[rng.integers(0, len(neg), len(neg))],
        ])
        yb, pb = y[idx], p[idx]
        if 0 < yb.sum() < len(yb):
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)

#shap bar

def fig_shap_bar() -> None:
    df = (pd.read_csv(ROOT / "outputs" / "v2" / "shap_with_contralateral.csv")
          .sort_values("rank")
          .head(18))
    df["label"] = df["feature_name"].map(lambda x: FEAT_LABELS.get(x, x))

    fig_w = 10 / 2.54
    fig_h = 11 / 2.54

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        n      = len(df)
        y_pos  = np.arange(n - 1, -1, -1)    # rank 1 at top
        vals   = df["mean_abs_shap"].values

        ax.barh(y_pos, vals, color="#1a4c8b", height=0.62)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["label"].values, fontsize=7)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_xlim(0, vals.max() * 1.12)
        _strip_axes(ax)

        for i, (yp, v) in enumerate(zip(y_pos, vals)):
            ax.text(v + vals.max() * 0.01, yp, f"{v:.3f}",
                    va="center", fontsize=5.5, color="#333333")

        fig.tight_layout(pad=0.5)
        _save(fig, "shap_bar_with_contralateral")
        plt.close(fig)


# calibration loess

def fig_calibration_overlay() -> None:
    models = [
        "with_contralateral", "parsimonious", "logistic_regression",
        "Peeters", "Kinnaird", "ERSPC",
    ]

    # ICI standard (LOESS frac=0.75, cohorte propre) — source: calibration_summary.json
    ICI_STANDARD = {
        "with_contralateral":  0.027,
        "parsimonious":        0.051,
        "logistic_regression": 0.059,
        "Peeters":             0.032,
        "Kinnaird":            0.093,
        "ERSPC":               0.015,
    }

    FOOTNOTE = (
        "ICI calculés selon la convention LOESS frac=0·75 (val.prob, "
        "référence Van Calster et al., 2025). "
        "Tracé : LOESS frac=0·15 avec bandes bootstrap 95 % CI."
    )

    preds = {m: _load_pred(m) for m in models}

    common_idx = preds[models[0]].index
    for m in models[1:]:
        common_idx = common_idx.intersection(preds[m].index)
    print(f"  Common N = {len(common_idx)}")

    fig_w = 17 / 2.54
    fig_h = 11 / 2.54

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for i, m in enumerate(models):
            name, color, ls, lw = MODEL_META[m]
            df_m = preds[m].loc[common_idx]
            p    = df_m["y_proba"].values
            y    = df_m["y_true"].values.astype(float)

            print(f"  [{i+1}/{len(models)}] LOESS + bootstrap for {name}…", flush=True)
            p_s, loess_s, p_grid, ci_lo, ci_hi, _ = _loess_with_ci(p, y)

            ici_std = ICI_STANDARD[m]
            label   = f"{name} (ICI={ici_std:.3f})"
            ax.plot(p_s, loess_s, color=color, ls=ls, lw=lw, label=label)
            ax.fill_between(p_grid, ci_lo, ci_hi, color=color, alpha=0.09)

        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=0.8, label="Idéal")
        ax.set_xlabel("Probabilité prédite")
        ax.set_ylabel("Proportion observée")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _strip_axes(ax)
        ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                  edgecolor="none", handlelength=1.8,
                  borderpad=0.5, labelspacing=0.3)

        # Reserve bottom margin for footnote, then add it
        fig.tight_layout(pad=0.4, rect=[0, 0.07, 1, 1])
        fig.text(
            0.5, 0.015, FOOTNOTE,
            ha="center", va="bottom",
            fontsize=5.5, color="#444444",
            style="italic",
            wrap=True,
        )

        _save(fig, "calibration_loess_overlay")
        plt.close(fig)


# roc

def fig_roc_overlay() -> None:
    models = [
        "with_contralateral", "parsimonious", "logistic_regression",
        "Peeters", "Kinnaird", "ERSPC",
    ]
    auc_table = pd.read_csv(
        ROOT / "outputs" / "v2" / "table_auc_main.csv",
        index_col="model",
    )

    fig_w = 9 / 2.54
    fig_h = 8 / 2.54

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for m in models:
            name, color, ls, lw = MODEL_META[m]
            df_m    = _load_pred(m)
            y       = df_m["y_true"].values.astype(float)
            p       = df_m["y_proba"].values
            auc_val = (auc_table.loc[m, "auc"]
                       if m in auc_table.index
                       else roc_auc_score(y, p))
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=color, ls=ls, lw=lw,
                    label=f"{name} (AUROC={auc_val:.3f})")

        ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.8)
        ax.set_xlabel("1 − Spécificité (FPR)")
        ax.set_ylabel("Sensibilité (TPR)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _strip_axes(ax)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92,
                  edgecolor="none", handlelength=1.8,
                  borderpad=0.5, labelspacing=0.3)
        fig.tight_layout(pad=0.4)
        _save(fig, "roc_overlay")
        plt.close(fig)


# forest plot

def _build_strata(feat_df: pd.DataFrame) -> dict[str, pd.Index]:
    strata: dict[str, pd.Index] = {}

    for pv in [3, 4, 5]:
        idx = feat_df.index[feat_df["pirads"] == pv]
        if len(idx):
            strata[f"PI-RADS = {int(pv)}"] = idx

    if "age" in feat_df.columns:
        valid = feat_df["age"].dropna()
        labels = pd.qcut(valid, q=3,
                         labels=["T1 (young)", "T2 (mid)", "T3 (old)"],
                         duplicates="drop")
        for lbl in labels.cat.categories:
            strata[f"Age {lbl}"] = valid.index[labels == lbl]

    if "psa_density" in feat_df.columns:
        valid = feat_df["psa_density"].dropna()
        labels = pd.qcut(valid, q=3,
                         labels=["T1 (low)", "T2 (mid)", "T3 (high)"],
                         duplicates="drop")
        for lbl in labels.cat.categories:
            strata[f"PSAd {lbl}"] = valid.index[labels == lbl]

    if "Center" in feat_df.columns:
        for raw_center in ["FR (Bordeaux)", "FR (Toulouse)",
                           "FR (Grenoble)", "CZ (Ostrava)"]:
            idx = feat_df.index[feat_df["Center"] == raw_center]
            if len(idx):
                label = (raw_center
                         .replace("FR (", "").replace("CZ (", "")
                         .rstrip(")"))
                strata[label] = idx

    return strata


def fig_forest_plot() -> None:
    df_pred = _load_pred("with_contralateral")

    test_path = ROOT / "data" / "test_df.csv"
    df_test   = pd.read_csv(test_path, sep=";")
    if "Order" in df_test.columns:
        df_test = df_test.set_index("Order")

    center_src  = ROOT / "data" / "dataframe_cleaned_2_with_center.csv"
    center_info = pd.read_csv(center_src, sep=";", usecols=["Order", "Center"])
    center_info = center_info.set_index("Order")["Center"]
    df_test     = df_test.join(center_info, how="left")

    common   = df_pred.index.intersection(df_test.index)
    feat_df  = df_test.loc[common]
    strata   = _build_strata(feat_df)

    GLOBAL_AUC = 0.760

    rows: list[dict] = []
    for stratum_name, stratum_idx in strata.items():
        valid_idx = stratum_idx.intersection(df_pred.index)
        if len(valid_idx) < 30:
            continue
        sub = df_pred.loc[valid_idx]
        y   = sub["y_true"].values.astype(float)
        p   = sub["y_proba"].values
        if len(np.unique(y)) < 2:
            continue
        auc      = roc_auc_score(y, p)
        lo, hi   = _auc_ci(y, p)
        rows.append({
            "stratum": stratum_name,
            "n":       len(valid_idx),
            "auc":     auc,
            "lo":      lo,
            "hi":      hi,
        })
        print(f"  {stratum_name}: AUC={auc:.3f} [{lo:.3f}–{hi:.3f}] n={len(valid_idx)}")

    result = {r["stratum"]: r for r in rows}

    groups = [
        ("PI-RADS",       ["PI-RADS = 3",  "PI-RADS = 4",  "PI-RADS = 5"]),
        ("PSAd tertile",  ["PSAd T1 (low)","PSAd T2 (mid)","PSAd T3 (high)"]),
        ("Age tertile",   ["Age T1 (young)","Age T2 (mid)", "Age T3 (old)"]),
        ("Centre",        ["Bordeaux", "Toulouse", "Grenoble", "Ostrava"]),
    ]

    # Build ordered item list: ("header"|"row", label, data_or_None)
    items: list[tuple] = []
    for grp_name, strata_list in groups:
        items.append(("header", grp_name, None))
        for s in strata_list:
            if s in result:
                items.append(("row", s, result[s]))

    n_items = len(items)
    fig_w   = 12 / 2.54
    fig_h   = max(17 / 2.54, (n_items * 0.88 + 3.0) / 2.54)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = GridSpec(1, 2, figure=fig,
                       width_ratios=[0.47, 0.53], wspace=0.02)
        ax_lbl = fig.add_subplot(gs[0])
        ax_plt = fig.add_subplot(gs[1])

        y_top = n_items - 0.5
        y_bot = -0.5
        ax_lbl.set_xlim(0, 1)
        ax_lbl.set_ylim(y_bot, y_top)
        ax_lbl.set_axis_off()

        ax_plt.set_xlim(0.59, 0.93)
        ax_plt.set_ylim(y_bot, y_top)
        ax_plt.set_yticks([])
        ax_plt.spines["top"].set_visible(False)
        ax_plt.spines["right"].set_visible(False)
        ax_plt.spines["left"].set_visible(False)
        ax_plt.tick_params(direction="out", pad=2)

        y_pos = n_items - 1
        for item_type, label, data in items:
            if item_type == "header":
                ax_lbl.text(0.98, y_pos, label,
                            fontsize=8, fontweight="bold",
                            va="center", ha="right")
                ax_plt.axhline(y_pos - 0.5,
                               color="#cccccc", lw=0.5, zorder=0)
            else:
                ax_lbl.text(0.95, y_pos, label,
                            fontsize=7, va="center", ha="right")
                auc = data["auc"]
                lo  = data["lo"]
                hi  = data["hi"]
                n   = data["n"]
                ax_plt.plot(auc, y_pos, "o",
                            color="#1a4c8b", ms=4.5, zorder=3)
                ax_plt.hlines(y_pos, lo, hi,
                              color="#1a4c8b", lw=1.2, zorder=3)
                ax_plt.text(0.918, y_pos, f"n={n}",
                            fontsize=6, va="center", ha="left",
                            clip_on=False)
            y_pos -= 1

        ax_plt.axvline(GLOBAL_AUC, color="tab:red",
                       ls="--", lw=0.9, zorder=1,
                       label=f"Overall AUROC = {GLOBAL_AUC:.3f}")
        ax_plt.set_xlabel("AUROC")
        ax_plt.legend(loc="lower right", frameon=True,
                      framealpha=0.92, edgecolor="none",
                      fontsize=6.5)

        fig.tight_layout(pad=0.4)
        _save(fig, "forest_subgroups_with_contralateral")
        plt.close(fig)


# confusion matrix

def fig_confmat(threshold: float = 0.20) -> None:
    df     = _load_pred("with_contralateral")
    y      = df["y_true"].values.astype(int)
    p      = df["y_proba"].values
    y_pred = (p >= threshold).astype(int)

    # sklearn CM: rows=true, cols=pred → transpose for rows=pred, cols=true
    cm   = confusion_matrix(y, y_pred)
    cm_T = cm.T      # shape [pred(0,1), true(0,1)]
    n    = len(y)

    fig_w = 9 / 2.54
    fig_h = 8 / 2.54

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(cm_T, cmap="Blues", aspect="auto", vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.80, pad=0.02)

        class_labels = ["Absent (0)", "Présent (1)"]
        for i in range(2):
            for j in range(2):
                v     = cm_T[i, j]
                tc    = "white" if v > cm_T.max() * 0.55 else "black"
                annot = f"{v}\n({100*v/n:.1f}%)"
                ax.text(j, i, annot, ha="center", va="center",
                        fontsize=7.5, color=tc)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)
        ax.set_xlabel("csPCa observé")
        ax.set_ylabel("csPCa prédit")
        ax.set_title(f"Seuil décisionnel = {threshold:.2f}", pad=6)

        fig.tight_layout(pad=0.5)
        _save(fig, "confmat_t020")
        plt.close(fig)


# probability histogram (similar to violon plot)

def fig_proba_hist(threshold: float = 0.20) -> None:
    df  = _load_pred("with_contralateral")
    y   = df["y_true"].values.astype(int)
    p   = df["y_proba"].values
    n0  = int((y == 0).sum())
    n1  = int((y == 1).sum())

    fig_w = 12 / 2.54
    fig_h =  7 / 2.54

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        bins = np.linspace(0, 1, 41)
        ax.hist(p[y == 0], bins=bins, color="tab:blue",   alpha=0.6,
                label=f"csPCa absent  (n={n0})")
        ax.hist(p[y == 1], bins=bins, color="tab:orange", alpha=0.6,
                label=f"csPCa présent (n={n1})")

        y_top = ax.get_ylim()[1]
        ax.axvline(threshold, color="tab:red", ls="--", lw=1.2)
        ax.text(threshold + 0.015, y_top * 0.92,
                f"seuil clinique = {threshold:.2f}",
                color="tab:red", fontsize=7, va="top")

        ax.set_xlabel("Probabilité prédite csPCa")
        ax.set_ylabel("Nombre de patients")
        ax.set_xlim(0, 1)
        _strip_axes(ax)
        ax.legend(frameon=True, framealpha=0.92, edgecolor="none")

        fig.tight_layout(pad=0.4)
        _save(fig, "proba_hist_with_contralateral")
        plt.close(fig)


if __name__ == "__main__":
    print("=== Figure 1/6 : SHAP bar ===")
    fig_shap_bar()

    print("\n=== Figure 2/6 : Calibration LOESS overlay (bootstrap — may take ~1 min) ===")
    fig_calibration_overlay()

    print("\n=== Figure 3/6 : ROC overlay ===")
    fig_roc_overlay()

    print("\n=== Figure 4/6 : Forest plot (bootstrap per stratum) ===")
    fig_forest_plot()

    print("\n=== Figure 5/6 : Confusion matrix (seuil 0.20) ===")
    fig_confmat()

    print("\n=== Figure 6/6 : Histogramme des probabilités ===")
    fig_proba_hist()

    print("\n[DONE] All 6 figures saved to outputs/v2/figures/")
