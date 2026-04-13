"""
Stratified 90/10 train/test split of the source dataset.
Run once before training to generate data/train_df.csv and data/test_df.csv.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_get, cfg_path, CONFIG

DATA_PATH_SOURCE = cfg_path(CONFIG, "paths.source", "data/dataframe_cleaned.csv")
DATA_PATH_TRAIN = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
DATA_PATH_TEST = cfg_path(CONFIG, "paths.test", "data/test_df.csv")
TARGET_COL = cfg_get(CONFIG, "common.target_col", "outcome")
RANDOM_STATE = cfg_get(CONFIG, "common.random_state", 2)
TEST_SIZE = cfg_get(CONFIG, "split.test_size", 0.10)

df = pd.read_csv(DATA_PATH_SOURCE, sep=";")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
df.columns = df.columns.str.strip()

if TARGET_COL not in df.columns:
    if "ISUP GG (highest grade)" in df.columns:
        df[TARGET_COL] = df["ISUP GG (highest grade)"] > 1
    else:
        raise ValueError("outcome column not found")

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train
test_df = X_test.copy()
test_df[TARGET_COL] = y_test

train_df.to_csv(DATA_PATH_TRAIN, index=False, sep=";")
test_df.to_csv(DATA_PATH_TEST, index=False, sep=";")

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("target distribution train:")
print(train_df[TARGET_COL].value_counts(normalize=True))
print("target distribution test:")
print(test_df[TARGET_COL].value_counts(normalize=True))
