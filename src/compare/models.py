import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame, Series
from scipy.special import expit as sigmoid

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import load_config, cfg_get, cfg_path

_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG = load_config(_ROOT / "config.yaml")

def get_probabilities_peter2022(df: DataFrame) -> tuple[Series, Series]:
    # Drop rows where necessary variables are missing
    df = df.copy()
    df["psa_density"] = pd.to_numeric(df["psa_density"], errors="coerce")
    df["prostate_volume"] = pd.to_numeric(df["prostate_volume"], errors="coerce")
    df["pirads"] = pd.to_numeric(df["pirads"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.dropna(subset=["psa_density", "prostate_volume", "pirads", "age"])
    y = df["outcome"].copy()
    # Data transformation
    df["psa_density_log"] = np.log(df["psa_density"])
    df["Prior_biopsy_Yes"] = np.where(df["prev_neg_trus_biopsy"], 1, 0)
    df["Highest_MRI_score_4"] = (df["pirads"] == 4).astype(int)
    df["Highest_MRI_score_5"] = (df["pirads"] == 5).astype(int)

    # Select relevant features and fill missing values if any
    X = df[
        [
            "psa_density_log",
            "Prior_biopsy_Yes",
            "prostate_volume",
            "Highest_MRI_score_4",
            "Highest_MRI_score_5",
            "age",
        ]
    ].copy()
    X["prostate_volume"] = X["prostate_volume"].fillna(
        0
    )  # Assuming missing MRI volume can be treated as 0
    # X = pd.get_dummies(X, columns=['Prior_biopsy'], drop_first=True)

    # Apply the logistic regression model
    coefficients = {
        "intercept": -1.851187,
        "psa_density_log": 1.103418,
        "Prior_biopsy_Yes": -1.020887,
        "prostate_volume": -0.008079,
        "Highest_MRI_score_4": 0.933048,
        "Highest_MRI_score_5": 1.886100,
        "age": 0.052568,
    }

    linear_combination = (
        np.dot(X.to_numpy(), [coefficients.get(col, 0) for col in X.columns])
        + coefficients["intercept"]
    )
    proba = pd.Series(sigmoid(linear_combination), index=df.index)
    return y, proba


def get_probabilities_xgb(
    df: DataFrame, model: xgb.XGBClassifier
) -> tuple[Series, Series]:
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
    X = df.drop(columns=["outcome"]).copy()
    y = df["outcome"].copy()
    expected = list(model.feature_names_in_)
    X = X[expected]
    proba = pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    return y, proba


def get_probabilities_xgbtuned(df: DataFrame) -> tuple[Series, Series]:
    model = joblib.load(_ROOT / "runs" / "xgb_300_trials_old_dataset" / "model.pkl")
    return get_probabilities_xgb(df=df, model=model)


def get_probabilities_xgbdefault(df: DataFrame) -> tuple[Series, Series]:
    model = joblib.load(_ROOT / "runs" / "xgb_default_old_dataset" / "model.pkl")
    return get_probabilities_xgb(df=df, model=model)


def get_probabilities_xgbtunedfe(df: DataFrame) -> tuple[Series, Series]:
    model = joblib.load(_ROOT / "models" / "baselines" / "tuned_fe_model.pkl")
    return get_probabilities_xgb(df=df, model=model)


def get_probabilities_xgbtunedfe_logloss(df: DataFrame) -> tuple[Series, Series]:
    model = joblib.load(_ROOT / "runs" / "xgb_1000trials_fe_logloss" / "model.pkl")
    return get_probabilities_xgb(df=df, model=model)


def get_probabilities_logistic_regression(df: DataFrame) -> tuple[Series, Series]:
    X = df.drop(columns=["outcome"]).copy()
    y = df["outcome"].copy()
    pipeline = joblib.load(_ROOT / "models" / "baselines" / "logistic_regression_old.pkl")
    lr = pipeline.steps[-1][1]
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
    return y, pd.Series(pipeline.predict_proba(X)[:, 1], index=X.index)


def get_probabilities_erspc(df: DataFrame) -> tuple[Series, Series]:
    excel_path = _ROOT / "data" / "external" / "ERSPC_proba.xlsx"
    erspc_df = pd.read_excel(excel_path, engine="openpyxl")
    erspc_df = erspc_df.set_index("Order")[["Probability csPCa"]]
    merged_df = df.join(erspc_df, how="inner")
    merged_df.dropna(subset=["outcome", "Probability csPCa"], inplace=True)
    y = merged_df["outcome"].copy()
    proba = pd.Series(merged_df["Probability csPCa"].values, index=merged_df.index)
    return y, proba


def get_probabilities_xgb_all_features(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.xgb_all_features", "runs/all_features_tuned/model.pkl")
    model = joblib.load(model_path)
    return get_probabilities_xgb(df=df.copy(), model=model)


def get_probabilities_xgb_no_contra(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.xgb_no_contra", "runs/no_contralateral_tuned/model.pkl")
    model = joblib.load(model_path)
    return get_probabilities_xgb(df=df.copy(), model=model)


def get_probabilities_xgb_all_features_default(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.xgb_all_features_default", "runs/all_features_default/model.pkl")
    model = joblib.load(model_path)
    return get_probabilities_xgb(df=df.copy(), model=model)


def get_probabilities_xgb_no_contra_default(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.xgb_no_contra_default", "runs/no_contralateral_default/model.pkl")
    model = joblib.load(model_path)
    return get_probabilities_xgb(df=df.copy(), model=model)


def get_probabilities_xgb_parsimonious(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.xgb_parsimonious", "runs/parsimonious/model.pkl")
    model = joblib.load(model_path)
    return get_probabilities_xgb(df=df.copy(), model=model)


def get_probabilities_logreg_new(df: DataFrame) -> tuple[Series, Series]:
    model_path = cfg_path(_CONFIG, "compare.paths.logreg", "runs/logreg/model.pkl")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    lr = pipeline.steps[-1][1]
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"
    features_to_drop = cfg_get(_CONFIG, "model.features_to_drop", ["Order"])
    df = df.copy()
    df["age"]         = pd.to_numeric(df.get("age",         pd.Series(dtype=float)), errors="coerce")
    df["psa_density"] = pd.to_numeric(df.get("psa_density", pd.Series(dtype=float)), errors="coerce")
    y = df["outcome"].copy()
    X = df.drop(columns=[c for c in features_to_drop + ["outcome"] if c in df.columns])
    return y, pd.Series(pipeline.predict_proba(X)[:, 1], index=X.index)


def get_probabilities_kinnaird(df: DataFrame) -> tuple[Series, Series]:
    """Adapted from the javascript code on https://www.uclahealth.org/departments/urology/iuo/research/prostate-cancer/risk-calculator-mri-guided-biopsy-pcrc-mri
    consulted on 29/03/2024"""

    def calculate_race(row):
        if row["African-american ethnicity"] == 1:
            return 0
        if row["African-american ethnicity"] == 0:
            return 2
        return 4

    def calculate_pirad(row):
        if row["pirads"] == 3:
            return 1
        if row["pirads"] == 4:
            return 2
        if row["pirads"] == 5:
            return 3
        if row["pirads"] == 2:
            return 0
        return np.nan

    # Parameters
    param_MRI_Intercept = -4.62990
    param_MRI_Age = 0.05470
    arr_MRI_Race = [0.23780, -0.77190, 0, 0.18840, -0.08560]
    param_MRI_PSA = 0.01920
    param_MRI_DRE = 0.86460
    param_MRI_PrevBiopsy = -0.61630
    param_MRI_ProstateVolume = -0.01790
    param_MRI_PSADensity = 0.88020
    arr_MRI_PIRAD = [0, 0.51020, 1.37510, 2.76270]

    original_df = pd.read_excel(
        _ROOT / "data" / "external" / "DATABASE_SYNTHESE.xlsx", engine="openpyxl", skiprows=1
    )
    original_df.set_index("Order", inplace=True)
    df = df.merge(
        original_df[["African-american ethnicity"]],
        how="left",
        left_index=True,
        right_index=True,
    )

    for col in ["age", "psa", "clinical_stage", "psa_density", "prostate_volume", "prev_neg_trus_biopsy"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["DRE"] = np.where(df["clinical_stage"] > 0, 1, 0)
    df["Race"] = df.apply(calculate_race, axis=1)
    df["PIRAD"] = df.apply(calculate_pirad, axis=1)
    df["calcPSADensity"] = np.where(df["psa_density"] > 0.15, 1, 0)
    df.rename(
        columns={
            "prostate_volume": "Volume",
            "prev_neg_trus_biopsy": "PrevBiopsy",
        },
        inplace=True,
    )

    df.dropna(
        subset=["age", "psa", "DRE", "PrevBiopsy", "Volume", "calcPSADensity", "PIRAD"],
        inplace=True,
    )
    df["PIRAD"] = df["PIRAD"].astype(int)

    # Calculate the linear combination (strEquation in the JS code)
    df["strEquation"] = (
        param_MRI_Intercept
        + param_MRI_Age * df["age"]
        + df["Race"].apply(lambda x: arr_MRI_Race[x])
        + param_MRI_PSA * df["psa"]
        + param_MRI_DRE * df["DRE"]
        + param_MRI_PrevBiopsy * df["PrevBiopsy"]
        + param_MRI_ProstateVolume * df["Volume"]
        + param_MRI_PSADensity * df["calcPSADensity"]
        + df["PIRAD"].apply(lambda x: arr_MRI_PIRAD[x])
    )

    # Calculate the probability using the logistic function
    df["Probability"] = np.exp(df["strEquation"]) / (1 + np.exp(df["strEquation"]))
    y = df["outcome"].copy()
    proba = pd.Series(df["Probability"].values, index=df.index)
    return y, proba
