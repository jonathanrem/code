"""
Table 1 — descriptive statistics for train and test sets.
Saves outputs/table1.csv and outputs/table1.xlsx.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common.config import cfg_path, cfg_get, CONFIG

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = cfg_path(CONFIG, "paths.train", "data/train_df.csv")
TEST_PATH  = cfg_path(CONFIG, "paths.test",  "data/test_df.csv")
TARGET_COL = cfg_get(CONFIG, "common.target_col", "outcome")

CONTINUOUS = [
    "age", "psa", "psa_density", "prostate_volume",
    "diameter", "nb_susp_lesions",
]
BINARY = [
    "suspicious_trus", "family_history", "prev_neg_trus_biopsy",
    "epe", "ant", "mid", "post", "base", "median", "apical",
    "contralateral_suspicious",
]
ORDINAL = {
    "pirads":         [2, 3, 4, 5],
    "clinical_stage": [0, 1, 2, 3],
}


def _fmt_median_iqr(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "—"
    q1, med, q3 = valid.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


def _fmt_n_pct(count: int, total: int) -> str:
    return f"{count} ({count / total * 100:.1f}%)"


def _nan_pct(series: pd.Series) -> str:
    n_nan = series.isna().sum()
    pct   = n_nan / len(series) * 100
    return f"{n_nan} ({pct:.1f}%)" if n_nan > 0 else "0"


def describe_split(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    n = len(df)

    # ── Header ──────────────────────────────────────────────────────────────
    rows.append({"Variable": f"N", label: str(n), "section": "Overview"})

    # ── Outcome ─────────────────────────────────────────────────────────────
    if TARGET_COL in df.columns:
        n_pos = int(df[TARGET_COL].sum())
        rows.append({
            "Variable": "csPCa positive",
            label: _fmt_n_pct(n_pos, n),
            "section": "Outcome",
        })

    # ── Continuous ───────────────────────────────────────────────────────────
    for col in CONTINUOUS:
        if col not in df.columns:
            continue
        rows.append({
            "Variable": f"{col} — median [Q1–Q3]",
            label: _fmt_median_iqr(df[col]),
            "section": "Continuous",
        })
        nan_str = _nan_pct(df[col])
        if nan_str != "0":
            rows.append({
                "Variable": f"  {col} — missing",
                label: nan_str,
                "section": "Continuous",
            })

    # ── Binary ───────────────────────────────────────────────────────────────
    for col in BINARY:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        n_pos = int((valid == 1).sum())
        rows.append({
            "Variable": f"{col} — yes",
            label: _fmt_n_pct(n_pos, len(valid)) if len(valid) > 0 else "—",
            "section": "Binary",
        })
        nan_str = _nan_pct(df[col])
        if nan_str != "0":
            rows.append({
                "Variable": f"  {col} — missing",
                label: nan_str,
                "section": "Binary",
            })

    # ── Ordinal ──────────────────────────────────────────────────────────────
    for col, values in ORDINAL.items():
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        for val in values:
            count = int((valid == val).sum())
            if count == 0:
                continue
            rows.append({
                "Variable": f"{col} = {int(val)}",
                label: _fmt_n_pct(count, len(valid)),
                "section": "Ordinal",
            })
        nan_str = _nan_pct(df[col])
        if nan_str != "0":
            rows.append({
                "Variable": f"  {col} — missing",
                label: nan_str,
                "section": "Ordinal",
            })

    return rows


def main():
    df_train = pd.read_csv(TRAIN_PATH, sep=";")
    df_test  = pd.read_csv(TEST_PATH,  sep=";")
    df_all   = pd.concat([df_train, df_test], ignore_index=True)

    train_rows = describe_split(df_train, "Train")
    test_rows  = describe_split(df_test,  "Test")
    all_rows   = describe_split(df_all,   "Overall")

    # Merge on Variable
    df_t = pd.DataFrame(train_rows).set_index("Variable")
    df_v = pd.DataFrame(test_rows).set_index("Variable")
    df_a = pd.DataFrame(all_rows).set_index("Variable")

    table = (
        df_t[["Train", "section"]]
        .join(df_v[["Test"]], how="outer")
        .join(df_a[["Overall"]], how="outer")
        .reset_index()
    )

    # Print
    print("\n=== TABLE 1 ===")
    print(table[["Variable", "Train", "Test", "Overall"]].to_string(index=False))

    # Save
    csv_path  = OUT_DIR / "table1.csv"
    xlsx_path = OUT_DIR / "table1.xlsx"
    table[["section", "Variable", "Train", "Test", "Overall"]].to_csv(csv_path, index=False)
    table[["section", "Variable", "Train", "Test", "Overall"]].to_excel(xlsx_path, index=False)
    print(f"\n[OK] Table 1 saved to: {csv_path}")
    print(f"[OK] Table 1 saved to: {xlsx_path}")


if __name__ == "__main__":
    main()
