"""
Common data utilities for the ML pipeline.

This module provides functions to load, clean, and align datasets.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from .config import CONFIG, cfg_path, cfg_get

def clean_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean input dataframe by dropping 'Order' column if present."""
    if "Order" in df.columns:
        df = df.drop(columns=["Order"])
    return df

def load_train_test_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load and clean train and test datasets."""
    train_path = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
    test_path = cfg_path(CONFIG, "paths.test", "data/test_df.csv")

    df_train = pd.read_csv(train_path, sep=";")
    df_test = pd.read_csv(test_path, sep=";")

    df_train = clean_input_df(df_train)
    df_test = clean_input_df(df_test)

    target_col = cfg_get(CONFIG, "common.target_col", "outcome")
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    y_test = df_test[target_col]
    X_test = df_test.drop(columns=[target_col])

    return X_train, y_train, X_test, y_test

def load_compare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset for model comparison, keeping Order as index.

    Unlike load_test_data(), Order is set as the DataFrame index so that
    ERSPC and Kinnaird can join external data on it via index alignment.
    """
    use_test = cfg_get(CONFIG, "evaluate.use_test_df", True)
    if use_test:
        path = cfg_path(CONFIG, "paths.test", "data/test_df.csv")
    else:
        path = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
    print(f"[data] comparing on: {path.name} (evaluate.use_test_df={use_test})")
    df = pd.read_csv(path, sep=";")
    if "Order" in df.columns:
        df = df.set_index("Order")
    target_col = cfg_get(CONFIG, "common.target_col", "outcome")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y


def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and clean dataset for evaluation.

    Controlled by config evaluate.use_test_df:
      true  → data/test_df.csv   (final evaluation)
      false → data/train_df.csv  (development / overfitting check)
    """
    use_test = cfg_get(CONFIG, "evaluate.use_test_df", True)
    if use_test:
        path = cfg_path(CONFIG, "paths.test", "data/test_df.csv")
    else:
        path = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
    print(f"[data] evaluating on: {path.name} (evaluate.use_test_df={use_test})")
    df = pd.read_csv(path, sep=";")
    df = clean_input_df(df)
    target_col = cfg_get(CONFIG, "common.target_col", "outcome")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return X, y

def align_to_metadata(X: pd.DataFrame, metadata: Dict, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Align dataframe features to model metadata."""
    if drop_cols:
        existing = [c for c in drop_cols if c in X.columns]
        if existing:
            X = X.drop(columns=existing)

    features = metadata.get("features", [])
    if isinstance(features, list) and features:
        for col in features:
            if col not in X.columns:
                X[col] = np.nan
        extra = [c for c in X.columns if c not in features]
        if extra:
            X = X.drop(columns=extra)
        X = X[features]

    category_mappings = metadata.get("category_mappings", {}) or {}
    if category_mappings:
        for col, cats in category_mappings.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=cats)
    else:
        obj_cols = X.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            X[col] = X[col].astype("category")

    return X

def get_feature_label(metadata: Dict) -> str:
    """Determine feature set label from metadata."""
    features = metadata.get("features", [])
    if isinstance(features, list) and features:
        return "top5" if len(features) <= 5 else "all"
    return "unknown"


def assert_center_disjoint(df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
    """Assert no center appears in both train and test sets.

    Guarantees the geographic external validation claim: test patients
    come from centers entirely unseen during training (split_by_center.ipynb).
    Silently skips if no center column is present in both dataframes.
    """
    center_col = next(
        (c for c in ["center", "Center"] if c in df_train.columns and c in df_test.columns),
        None,
    )
    if center_col is None:
        return
    common = set(df_train[center_col].dropna()) & set(df_test[center_col].dropna())
    assert not common, f"Centres communs entre train et test : {common}"