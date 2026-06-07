import json
import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from MLstatkit import Delong_test

# Desactiver les warnings inutiles
warnings.simplefilter(action="ignore", category=FutureWarning)

# Add src and src/compare to path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.append(str(_HERE.parent))
sys.path.append(str(_HERE))
from common.data import load_compare_data
from common.config import CONFIG, cfg_get
from evaluate.clinical_evaluation import evaluate_all_models

# Importer tous les modeles
from models import (
    get_probabilities_peter2022,
    get_probabilities_erspc,
    get_probabilities_kinnaird,
    get_probabilities_xgb_all_features,
    get_probabilities_xgb_all_features_default,
    get_probabilities_xgb_no_contra,
    get_probabilities_xgb_no_contra_default,
    get_probabilities_xgb_parsimonious,
    get_probabilities_logreg_new,
)

THRESHOLD = cfg_get(CONFIG, "evaluate.common_threshold", 0.2)

MODEL_REGISTRY = {
    "All features":          get_probabilities_xgb_all_features,
    "All features (default)":get_probabilities_xgb_all_features_default,
    "No contralateral":      get_probabilities_xgb_no_contra,
    "No contra (default)":   get_probabilities_xgb_no_contra_default,
    "Parsimonious":          get_probabilities_xgb_parsimonious,
    "LogReg":                get_probabilities_logreg_new,
    "Peeters":               get_probabilities_peter2022,
    "Kinnaird":              get_probabilities_kinnaird,
    "ERSPC":                 get_probabilities_erspc,
}


# Fonction pour calculer les metriques binaires
def compute_metrics(y_true, y_proba, threshold=None):
    if threshold is None:
        threshold = THRESHOLD
    # Intersecter les index
    common_index = y_true.index.intersection(y_proba.index)
    y_true = y_true.loc[common_index]
    y_proba = y_proba.loc[common_index]
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
    }


# Test de DeLong
def compare_auc_delong(y1, p1, y2, p2, name1, name2):

    df1 = pd.DataFrame({"y": y1, "p1": p1}).dropna()
    df2 = pd.DataFrame({"p2": p2}).dropna()

    # Conserver uniquement les lignes ou p1 et p2 sont disponibles
    merged = df1.join(df2, how="inner")

    if merged.empty:
        print(f"[WARN] Aucun cas commun entre {name1} et {name2}")
        return None

    z, p, *_ = Delong_test(merged["y"], merged["p1"], merged["p2"])
    auc1 = roc_auc_score(merged["y"], merged["p1"])
    auc2 = roc_auc_score(merged["y"], merged["p2"])
    print(f"DeLong: {name1} vs {name2} -> z = {z:.2f}, p = {p:.4f}, N = {len(merged)}")
    return {"model1": name1, "model2": name2, "auc1": round(auc1, 4), "auc2": round(auc2, 4),
            "z": round(float(z), 4), "p_value": round(float(p), 4), "n": len(merged)}


def clinical_impact_analysis(proba_dict, thresholds=[0.10, 0.15, 0.20]):
    print("\nValeur clinique des modeles pour differents seuils de biopsie :")
    rows = []
    for threshold in thresholds:
        for name, (y_true, y_proba) in proba_dict.items():
            common_idx = y_true.index.intersection(y_proba.index)
            y = y_true.loc[common_idx]
            p = y_proba.loc[common_idx]

            biopsied     = p >= threshold
            not_biopsied = ~biopsied

            total   = len(y)
            cancers = int(y.sum())
            benign  = total - cancers
            avoided = int(not_biopsied.sum())
            missed  = int(y[not_biopsied].sum())
            tp      = int(y[biopsied].sum())
            fp      = int((y[biopsied] == 0).sum())
            tn      = int((y[not_biopsied] == 0).sum())
            fn      = missed

            sens = tp / cancers       if cancers       > 0 else 0.0
            spec = tn / benign        if benign        > 0 else 0.0
            ppv  = tp / (tp + fp)     if (tp + fp)     > 0 else 0.0
            npv  = tn / (tn + fn)     if (tn + fn)     > 0 else 0.0

            rows.append({
                "Model":            name,
                "Threshold":        threshold,
                "N":                total,
                "Cancers":          cancers,
                "Biopsies_avoided": avoided,
                "Avoided_pct":      round(avoided / total * 100, 1),
                "Cancers_missed":   missed,
                "Sensitivity":      round(sens, 3),
                "Specificity":      round(spec, 3),
                "PPV":              round(ppv, 3),
                "NPV":              round(npv, 3),
            })
            print(
                f"{name:<22} | t={threshold:.2f} | avoided={avoided/total*100:.1f}% "
                f"| missed={missed} | sens={sens:.3f} | spec={spec:.3f} "
                f"| PPV={ppv:.3f} | NPV={npv:.3f}"
            )

    out_path = _ROOT / "outputs" / "compare" / "clinical_impact.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] Clinical impact table saved to: {out_path}")


def compute_net_benefit(y_true, y_proba, thresholds):
    N = len(y_true)
    net_benefit = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()

        nb = (TP / N) - (FP / N) * (t / (1 - t))
        net_benefit.append(nb)

    return net_benefit


def _align_outcomes_and_probabilities(y_true, y_proba):
    y = y_true.copy()
    p = y_proba if isinstance(y_proba, pd.Series) else pd.Series(y_proba, index=y.index)
    idx = y.index.intersection(p.index)
    y = y.loc[idx]
    p = p.loc[idx]
    valid = y.notna() & p.notna()
    return y.loc[valid], p.loc[valid]


def report_net_benefit_table(proba_dict, thresholds=(0.10, 0.15, 0.20, 0.25)):
    aligned = {}
    for name, (y_true, y_proba) in proba_dict.items():
        y, p = _align_outcomes_and_probabilities(y_true, y_proba)
        if y.empty:
            raise ValueError(f"Aucune donnee exploitable pour le modele '{name}'.")
        aligned[name] = (y, p)

    common_idx = None
    for y, _ in aligned.values():
        common_idx = y.index if common_idx is None else common_idx.intersection(y.index)
    if common_idx is None or len(common_idx) == 0:
        raise ValueError("Aucun patient commun entre les modeles.")

    common_idx = common_idx.sort_values()
    ref_name = next(iter(aligned))
    y_common = aligned[ref_name][0].loc[common_idx]
    prevalence = y_common.mean()
    N = len(common_idx)

    col_w = 9
    header = f"{'Modele':<22}" + "".join(f"  t={t:.2f}  " for t in thresholds)
    print(f"\nNet Benefit par seuil (N={N}, prevalence={prevalence:.3f}):")
    print(header)
    print("-" * len(header))

    rows = {}
    for name, (y, p) in aligned.items():
        nbs = compute_net_benefit(y_common, p.loc[common_idx], list(thresholds))
        rows[name] = nbs
        line = f"{name:<22}" + "".join(f"  {nb:+.4f} " for nb in nbs)
        print(line)

    treat_all_nbs = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    print(f"{'Treat all':<22}" + "".join(f"  {nb:+.4f} " for nb in treat_all_nbs))
    print(f"{'Treat none':<22}" + "".join(f"  {0.0:+.4f} " for _ in thresholds))

    print(f"\nDelta NB vs Treat all (modele - treat_all):")
    print(header)
    print("-" * len(header))
    for name, nbs in rows.items():
        deltas = [nb - ta for nb, ta in zip(nbs, treat_all_nbs)]
        line = f"{name:<22}" + "".join(f"  {d:+.4f} " for d in deltas)
        print(line)

    results = {
        name: {f"t={t:.2f}": round(nb, 4) for t, nb in zip(thresholds, nbs)}
        for name, nbs in rows.items()
    }
    results["treat_all"] = {f"t={t:.2f}": round(nb, 4) for t, nb in zip(thresholds, treat_all_nbs)}
    out_path = _ROOT / "outputs" / "compare" / "net_benefit_table.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Net benefit table saved to: {out_path}")


def smooth_net_benefit(thresholds: np.ndarray, nb: list, frac: float = 0.15) -> np.ndarray:
    """Lisse une courbe de Net Benefit par LOESS.

    frac controle la fenetre de lissage (0.10–0.20 recommande pour DCA).
    Plus frac est grand, plus la courbe est lisse mais peut masquer des
    differences reelles entre modeles.
    """
    nb_arr = np.array(nb)
    smoothed = lowess(nb_arr, thresholds, frac=frac, return_sorted=False)
    return smoothed


def plot_decision_curve(proba_dict):
    thresholds = np.linspace(0.01, 0.40, 300)

    aligned = {}
    for name, (y_true, y_proba) in proba_dict.items():
        y, p = _align_outcomes_and_probabilities(y_true, y_proba)
        if y.empty:
            raise ValueError(f"Aucune donnee exploitable pour le modele '{name}'.")
        aligned[name] = (y, p)

    common_idx = None
    for y, _ in aligned.values():
        common_idx = y.index if common_idx is None else common_idx.intersection(y.index)

    if common_idx is None or len(common_idx) == 0:
        raise ValueError(
            "Aucun patient commun entre les modeles. DCA impossible sur une cohorte commune."
        )

    common_idx = common_idx.sort_values()
    ref_name = next(iter(aligned))
    y_common = aligned[ref_name][0].loc[common_idx]

    COLOR_MAP = {
        "All features":          ("tab:red",    "-",  2.0),
        "All features (default)":("tab:red",    ":",  1.5),
        "No contralateral":      ("tab:blue",   "-",  2.0),
        "No contra (default)":   ("tab:blue",   ":",  1.5),
        "Parsimonious":          ("tab:brown",  "-",  2.0),
        "LogReg":                ("tab:green",  "-",  2.0),
        "Peeters":               ("tab:gray",   "--", 1.5),
        "Kinnaird":              ("tab:orange", "--", 1.5),
        "ERSPC":                 ("tab:purple", "--", 1.5),
    }

    plt.figure(figsize=(10, 6))

    for name, (y, p) in aligned.items():
        y_model = y.loc[common_idx]
        if not y_model.equals(y_common):
            raise ValueError(
                f"Incoherence de la variable cible entre '{ref_name}' et '{name}' sur la cohorte commune."
            )
        nb = compute_net_benefit(y_common, p.loc[common_idx], thresholds)
        nb_smooth = smooth_net_benefit(thresholds, nb)
        color, ls, lw = COLOR_MAP.get(name, ("tab:cyan", "-", 1.5))
        plt.plot(thresholds, nb_smooth, label=name, color=color, lw=lw, ls=ls)

    cancer_rate = y_common.mean()
    treat_all = [cancer_rate - t / (1 - t) * (1 - cancer_rate) for t in thresholds]
    treat_all_smooth = smooth_net_benefit(thresholds, treat_all)
    plt.plot(thresholds, [0] * len(thresholds), "k--", label="Ne biopsier personne")
    plt.plot(thresholds, treat_all_smooth, "k:", label="Biopsier tout le monde")

    plt.xlabel("Seuil de probabilite (decision biopsie)")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis (DCA)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()

    output_path = _ROOT / "outputs" / "compare" / "decision_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(
        f"[OK] Courbe sauvegardee dans: {output_path} (N commun = {len(common_idx)}, prevalence = {cancer_rate:.3f})"
    )
    plt.close()


def plot_pr_curves(proba_dict):
    """Plot Precision-Recall curves for all models.

    NOTE: AUPRC is prevalence-dependent and classified as an improper
    scoring rule by Van Calster et al. (2025, Lancet Digital Health).
    Retained for reference only — excluded from the default pipeline.
    """
    plt.figure(figsize=(8, 7))
    COLOR_MAP = {
        "All features":           ("tab:red",    "-",  2.0),
        "All features (default)": ("tab:red",    ":",  1.5),
        "No contralateral":       ("tab:blue",   "-",  2.0),
        "No contra (default)":    ("tab:blue",   ":",  1.5),
        "Parsimonious":           ("tab:brown",  "-",  2.0),
        "LogReg":                 ("tab:green",  "-",  2.0),
        "Peeters":                ("tab:gray",   "--", 1.5),
        "Kinnaird":               ("tab:orange", "--", 1.5),
        "ERSPC":                  ("tab:purple", "--", 1.5),
    }
    prevalences = []
    for name, (y_true, y_proba) in proba_dict.items():
        y, p = _align_outcomes_and_probabilities(y_true, y_proba)
        if y.empty:
            continue
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        color, ls, lw = COLOR_MAP.get(name, ("tab:cyan", "-", 1.5))
        plt.plot(recall, precision,
                 label=f"{name} (AP={ap:.3f}, N={len(y)})",
                 color=color, lw=lw, ls=ls)
        prevalences.append(float(y.mean()))

    if prevalences:
        prev = float(np.mean(prevalences))
        plt.axhline(prev, color="black", linestyle=":", lw=1,
                    label=f"No skill (prev≈{prev:.2f})")

    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True)
    plt.tight_layout()

    out_path = _ROOT / "outputs" / "compare" / "pr_curves.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] PR curves saved to: {out_path}")


def _load_center_info() -> pd.Series:
    source = _ROOT / "data" / "dataframe_cleaned_2_with_center.csv"
    df = pd.read_csv(source, sep=";", usecols=["Order", "Center"])
    return df.set_index("Order")["Center"]


def _define_strata(feat_df: pd.DataFrame) -> dict:
    strata = {}

    for pv in [3, 4, 5]:
        if "pirads" in feat_df.columns:
            idx = feat_df.index[feat_df["pirads"] == pv]
            if len(idx) > 0:
                strata[f"PIRADS={int(pv)}"] = idx

    if "age" in feat_df.columns:
        valid = feat_df["age"].dropna()
        if len(valid) >= 3:
            labels_age = pd.qcut(valid, q=3, labels=["Age T1 (young)", "Age T2 (mid)", "Age T3 (old)"],
                                 duplicates="drop")
            for lbl in labels_age.cat.categories:
                strata[str(lbl)] = valid.index[labels_age == lbl]

    if "psa_density" in feat_df.columns:
        valid = feat_df["psa_density"].dropna()
        if len(valid) >= 3:
            labels_psa = pd.qcut(valid, q=3, labels=["PSAd T1 (low)", "PSAd T2 (mid)", "PSAd T3 (high)"],
                                 duplicates="drop")
            for lbl in labels_psa.cat.categories:
                strata[str(lbl)] = valid.index[labels_psa == lbl]

    if "Center" in feat_df.columns:
        for center in sorted(feat_df["Center"].dropna().unique()):
            idx = feat_df.index[feat_df["Center"] == center]
            strata[f"Center: {center}"] = idx

    return strata


def run_subgroup_analysis(dca_proba: dict, df_test: pd.DataFrame) -> None:
    """APPRAISE-AI item 17: stratified performance analysis."""
    min_size = cfg_get(CONFIG, "evaluate.subgroup_min_size", 30)

    feat_df = df_test.copy()
    center_info = _load_center_info()
    feat_df = feat_df.join(center_info, how="left")

    strata = _define_strata(feat_df)
    if not strata:
        print("[WARN] Subgroup analysis: no strata defined.")
        return

    rows = []
    for model_name, (y_true, y_proba) in dca_proba.items():
        common = y_true.index.intersection(y_proba.index)
        y_all = y_true.loc[common]
        p_all = y_proba.loc[common]

        for stratum_name, stratum_idx in strata.items():
            idx = common.intersection(stratum_idx)
            if len(idx) < min_size:
                continue
            y_s = y_all.loc[idx]
            p_s = p_all.loc[idx]
            if len(y_s.unique()) < 2:
                continue

            auc = roc_auc_score(y_s, p_s)
            oe = float(y_s.sum() / p_s.sum()) if p_s.sum() > 0 else float("nan")
            rows.append({
                "Model":      model_name,
                "Stratum":    stratum_name,
                "N":          len(idx),
                "Events":     int(y_s.sum()),
                "Prevalence": round(float(y_s.mean()), 3),
                "AUC":        round(auc, 3),
                "O:E":        round(oe, 3),
            })

    if not rows:
        print("[WARN] Subgroup analysis: no valid strata (all below min_size or single-class).")
        return

    result_df = pd.DataFrame(rows)
    print("\n=== SUBGROUP ANALYSIS (APPRAISE-AI item 17) ===")
    print(result_df.to_string(index=False))

    out_dir = _ROOT / "outputs" / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "subgroup_analysis.csv"
    result_df.to_csv(out_path, index=False)
    print(f"[OK] Subgroup analysis saved to: {out_path}")


def run_error_analysis(dca_proba: dict, df_test: pd.DataFrame, threshold: float) -> None:
    """APPRAISE-AI item 18: error analysis at the clinical decision threshold."""
    primary_candidates = ["No contralateral", "All features"]
    primary_name = next((n for n in primary_candidates if n in dca_proba), next(iter(dca_proba)))

    y_true, y_proba = dca_proba[primary_name]
    common = y_true.index.intersection(y_proba.index)
    y = y_true.loc[common]
    p = y_proba.loc[common]
    y_pred = (p >= threshold).astype(int)

    def _label(row):
        if row["y"] == 1 and row["pred"] == 1:
            return "TP"
        if row["y"] == 0 and row["pred"] == 0:
            return "TN"
        if row["y"] == 0 and row["pred"] == 1:
            return "FP"
        return "FN"

    err_df = pd.DataFrame({"y": y, "proba": p, "pred": y_pred})
    err_df["classification"] = err_df.apply(_label, axis=1)

    feature_cols = [c for c in ["age", "pirads", "psa_density", "prostate_volume", "clinical_stage"]
                    if c in df_test.columns]
    err_df = err_df.join(df_test[feature_cols], how="left")
    err_df = err_df.join(_load_center_info(), how="left")

    counts = err_df["classification"].value_counts()
    tp, tn, fp, fn = (counts.get(k, 0) for k in ("TP", "TN", "FP", "FN"))
    n_cancers = int(y.sum())
    n_benign = len(y) - n_cancers
    sensitivity = tp / n_cancers if n_cancers > 0 else 0.0
    specificity = tn / n_benign if n_benign > 0 else 0.0

    print(f"\n=== ERROR ANALYSIS (APPRAISE-AI item 18) — {primary_name} @ t={threshold} ===")
    print(f"  N={len(err_df)} | TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}")

    for cls in ("FN", "FP"):
        sub = err_df[err_df["classification"] == cls]
        print(f"\n  {cls} (N={len(sub)}):")
        if "pirads" in sub.columns:
            print(f"    PIRADS distribution: {sub['pirads'].value_counts().sort_index().to_dict()}")
        if "age" in sub.columns:
            print(f"    Age: mean={sub['age'].mean():.1f}, median={sub['age'].median():.1f}")
        if "psa_density" in sub.columns:
            print(f"    PSA density: mean={sub['psa_density'].mean():.3f}, median={sub['psa_density'].median():.3f}")
        if "Center" in sub.columns:
            top = sub["Center"].value_counts().head(5)
            print(f"    Top centers: {top.to_dict()}")
        print(f"    Proba: min={sub['proba'].min():.3f}, mean={sub['proba'].mean():.3f}, max={sub['proba'].max():.3f}")

    out_dir = _ROOT / "outputs" / "evaluate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "error_analysis.csv"
    err_df.to_csv(out_path)
    print(f"\n[OK] Error analysis saved to: {out_path}")


# Fonction principale
def main():
    X_test, y_test = load_compare_data()
    test_df = X_test.copy()
    test_df["outcome"] = y_test

    # Construire le dict de probabilites depuis la config
    all_model_names = cfg_get(CONFIG, "compare.dca_models", list(MODEL_REGISTRY.keys()))
    proba = {}
    for name in all_model_names:
        if name not in MODEL_REGISTRY:
            print(f"[WARN] Modele inconnu ignore: {name}")
            continue
        proba[name] = MODEL_REGISTRY[name](test_df)

    dca_models = cfg_get(CONFIG, "compare.dca_models", list(proba.keys()))
    dca_proba = {name: proba[name] for name in dca_models if name in proba}

    if cfg_get(CONFIG, "compare.run_metrics_table", True):
        print("\nTable des metriques proper (seuil = 0.2) :")
        print(f"{'Modele':<20} | AUC   | Precision | Recall  | Specificity")
        print("-" * 65)
        for name, (y, p) in proba.items():
            m = compute_metrics(y, p)
            print(
                f"{name:<20} | {m['AUC']:.3f} | {m['Precision']:.3f}    | {m['Recall']:.3f}   | {m['Specificity']:.3f}"
            )
        print("\nAUC des modeles utilises pour la DCA :")
        print(f"{'Modele':<20} | {'AUC':<7}")
        print("-" * 30)
        for name, (y, p) in proba.items():
            common_idx = y.index.intersection(p.index)
            auc = roc_auc_score(y.loc[common_idx], p.loc[common_idx])
            print(f"{name:<20} | {auc:.3f}")

    if cfg_get(CONFIG, "compare.run_delong", True):
        delong_base = "All features"
        delong_results = []
        print(f"\nComparaison des AUCs (test de DeLong vs {delong_base}) :")
        base_y, base_p = proba[delong_base]
        for name, (y, p) in proba.items():
            if name != delong_base:
                result = compare_auc_delong(base_y, base_p, y, p, delong_base, name)
                if result is not None:
                    delong_results.append(result)

        # Comparaisons supplémentaires centrées sur Parsimonious
        if "Parsimonious" in proba:
            print("\nComparaison des AUCs (test de DeLong vs Parsimonious) :")
            pars_y, pars_p = proba["Parsimonious"]
            for name in ("No contralateral", "Peeters", "Kinnaird", "ERSPC"):
                if name in proba and name != "Parsimonious":
                    y, p = proba[name]
                    result = compare_auc_delong(pars_y, pars_p, y, p, "Parsimonious", name)
                    if result is not None:
                        delong_results.append(result)

        delong_path = _ROOT / "outputs" / "compare" / "delong_results.json"
        delong_path.parent.mkdir(parents=True, exist_ok=True)
        with open(delong_path, "w", encoding="utf-8") as f:
            json.dump(delong_results, f, indent=2)
        print(f"[OK] DeLong results saved to: {delong_path}")

    if cfg_get(CONFIG, "compare.run_clinical_impact", True):
        clinical_impact_analysis(proba)

    if cfg_get(CONFIG, "compare.run_dca", True):
        print("\nGeneration de la courbe de decision...")
        report_net_benefit_table(dca_proba)
        plot_decision_curve(dca_proba)

    if cfg_get(CONFIG, "compare.run_calibration", True):
        print("\nEvaluation calibration + discrimination par modele...")
        cal_output_dir = _ROOT / "outputs" / "evaluate"
        evaluate_all_models(dca_proba, cal_output_dir, cut=THRESHOLD)

    if cfg_get(CONFIG, "compare.run_subgroup", True):
        print("\nAnalyse par sous-groupes...")
        run_subgroup_analysis(dca_proba, test_df)

    if cfg_get(CONFIG, "compare.run_error_analysis", True):
        print("\nAnalyse des erreurs...")
        run_error_analysis(dca_proba, test_df, THRESHOLD)


if __name__ == "__main__":
    main()
