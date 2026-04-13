import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
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

# Importer tous les modeles
from models import (
    get_probabilities_peter2022,
    get_probabilities_erspc,
    get_probabilities_kinnaird,
    get_probabilities_xgbtuned,
    get_probabilities_xgbdefault,
    get_probabilities_logistic_regression,
)


THRESHOLD = cfg_get(CONFIG, "evaluate.common_threshold", 0.2)


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
        "AUCPR": average_precision_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_true, y_pred),
    }


# Test de DeLong
def compare_auc_delong(y1, p1, y2, p2, name1, name2):

    df1 = pd.DataFrame({"y": y1, "p1": p1}).dropna()
    df2 = pd.DataFrame({"p2": p2}).dropna()

    # Conserver uniquement les lignes ou p1 et p2 sont disponibles
    merged = df1.join(df2, how="inner")

    if merged.empty:
        print(f"[WARN] Aucun cas commun entre {name1} et {name2}")
        return

    z, p, *_ = Delong_test(merged["y"], merged["p1"], merged["p2"])
    print(f"DeLong: {name1} vs {name2} -> z = {z:.2f}, p = {p:.4f}, N = {len(merged)}")


def clinical_impact_analysis(proba_dict, thresholds=[0.10, 0.15, 0.20]):
    print("\nValeur clinique des modeles pour differents seuils de biopsie :")
    header = f"{'Modele':<15} | {'Seuil':<6} | {'Biopsies evitees (%)':<20} | {'Cancers manques':<15} | {'Sensibilite':<12} | {'Specificite':<12}"
    print(header)
    print("-" * len(header))
    for threshold in thresholds:
        for name, (y_true, y_proba) in proba_dict.items():
            # Realigner les index
            common_idx = y_true.index.intersection(y_proba.index)
            y = y_true.loc[common_idx]
            p = y_proba.loc[common_idx]

            biopsied = p >= threshold
            not_biopsied = ~biopsied

            total = len(y)
            cancers = y.sum()
            avoided = not_biopsied.sum()
            missed = y[not_biopsied].sum()

            sens = y[biopsied].sum() / cancers if cancers > 0 else 0
            spec = (
                (y[not_biopsied] == 0).sum() / (y == 0).sum()
                if (y == 0).sum() > 0
                else 0
            )

            print(
                f"{name:<15} | {threshold:<6.2f} | {avoided / total * 100:>18.1f} % | {int(missed):<15} | {sens:.3f}       | {spec:.3f}"
            )


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


def report_net_benefit_at_threshold(proba_dict, threshold=0.01):
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
            "Aucun patient commun entre les modeles. Impossible de calculer NB au meme seuil."
        )

    common_idx = common_idx.sort_values()
    ref_name = next(iter(aligned))
    y_common = aligned[ref_name][0].loc[common_idx]

    print(f"\nNet Benefit a threshold = {threshold:.2f} (cohorte commune):")
    print(f"{'Modele':<15} | {'NB':<8} | {'N':<6} | {'Prevalence':<10}")
    print("-" * 50)

    for name, (y, p) in aligned.items():
        y_model = y.loc[common_idx]
        if not y_model.equals(y_common):
            raise ValueError(
                f"Incoherence de la variable cible entre '{ref_name}' et '{name}' sur la cohorte commune."
            )
        nb = compute_net_benefit(y_common, p.loc[common_idx], [threshold])[0]
        print(f"{name:<15} | {nb:<8.4f} | {len(common_idx):<6} | {y_common.mean():<10.3f}")

    treat_all = y_common.mean() - (1 - y_common.mean()) * (threshold / (1 - threshold))
    print(f"{'Treat all':<15} | {treat_all:<8.4f} | {len(common_idx):<6} | {y_common.mean():<10.3f}")
    print(f"{'Treat none':<15} | {0.0:<8.4f} | {len(common_idx):<6} | {y_common.mean():<10.3f}")


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
    plt.figure(figsize=(10, 6))

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

    for name, (y, p) in aligned.items():
        y_model = y.loc[common_idx]
        if not y_model.equals(y_common):
            raise ValueError(
                f"Incoherence de la variable cible entre '{ref_name}' et '{name}' sur la cohorte commune."
            )
        nb = compute_net_benefit(y_common, p.loc[common_idx], thresholds)
        nb_smooth = smooth_net_benefit(thresholds, nb)
        plt.plot(thresholds, nb_smooth, label=name, lw=2)

    # Reference: biopsier tout le monde vs personne sur la meme cohorte commune
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

    # Sauvegarde la figure
    output_path = _ROOT / "outputs" / "compare" / "decision_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(
        f"[OK] Courbe sauvegardee dans: {output_path} (N commun = {len(common_idx)}, prevalence = {cancer_rate:.3f})"
    )
    plt.close()


# Fonction principale
def main():
    X_test, y_test = load_compare_data()
    test_df = X_test.copy()
    test_df["outcome"] = y_test

    # Recuperer les probabilites pour chaque modele
    proba = {
        "Peeters": get_probabilities_peter2022(test_df),
        "ERSPC": get_probabilities_erspc(test_df),
        "Kinnaird": get_probabilities_kinnaird(test_df),
        "XGB tuned": get_probabilities_xgbtuned(test_df),
        "XGB default": get_probabilities_xgbdefault(test_df),
        "Logistic": get_probabilities_logistic_regression(test_df),
    }

    print("\nTable des metriques (seuil = 0.2) :")
    print(f"{'Modele':<15} | AUC   | Accuracy | Precision | Recall  | Specificity | F1")
    print("-" * 85)
    for name, (y, p) in proba.items():
        m = compute_metrics(y, p)
        print(
            f"{name:<15} | {m['AUC']:.3f} | {m['Accuracy']:.3f}   | {m['Precision']:.3f}    | {m['Recall']:.3f}   | {m['Specificity']:.3f}     | {m['F1']:.3f}"
        )

    print("\nAUCPR des modeles :")
    print(f"{'Modele':<15} | {'AUCPR':<7}")
    print("-" * 27)
    for name, (y, p) in proba.items():
        common_idx = y.index.intersection(p.index)
        aucpr = average_precision_score(y.loc[common_idx], p.loc[common_idx])
        print(f"{name:<15} | {aucpr:.3f}")
    print("\nComparaison des AUCs (test de DeLong vs XGB tuned) :")
    base_y, base_p = proba["XGB tuned"]
    for name, (y, p) in proba.items():
        if name != "XGB tuned":
            compare_auc_delong(base_y, base_p, y, p, "XGB tuned", name)

    models_to_compare = ["Peeters", "ERSPC", "Kinnaird", "XGB tuned", "XGB default", "Logistic"]
    filtered_proba = {name: proba[name] for name in models_to_compare if name in proba}

    clinical_impact_analysis(filtered_proba)

    print("\nAUC des modeles utilises pour la DCA :")
    print(f"{'Modele':<15} | {'AUC':<7}")
    print("-" * 25)
    for name, (y, p) in filtered_proba.items():
        common_idx = y.index.intersection(p.index)
        auc = roc_auc_score(y.loc[common_idx], p.loc[common_idx])
        print(f"{name:<15} | {auc:.3f}")

    print("\nGeneration de la courbe de decision...")
    report_net_benefit_at_threshold(proba, threshold=0.01)
    plot_decision_curve(filtered_proba)


if __name__ == "__main__":
    main()
