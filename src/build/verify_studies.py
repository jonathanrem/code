"""
Pre-flight check: vérifie que les trois études Optuna utilisent bien des feature sets distincts.
Lancer après ~3 trials par étude, avant de lancer les 300 trials complets.

Usage:
    python src/build/verify_studies.py
"""
import sys
from pathlib import Path

import optuna
from optuna.study import StudyDirection

ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE = f"sqlite:///{ROOT / 'optuna.db'}"
CONTRALATERAL = {"contralateral_suspicious", "contralateral_pirads", "contralateral_diameter"}

optuna.logging.set_verbosity(optuna.logging.WARNING)

errors = []

def check(cond: bool, msg: str) -> None:
    if not cond:
        errors.append(f"FAIL: {msg}")
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK  : {msg}")


print("=== prostate_xgb_with_contralateral ===")
try:
    s_with = optuna.load_study(study_name="prostate_xgb_with_contralateral", storage=STORAGE)
    check(s_with.user_attrs.get("contralateral_used") is True, "contralateral_used == True")
    n_trials_with = len(s_with.trials)
    check(n_trials_with >= 1, f"nb trials >= 1 (obtenu : {n_trials_with})")
    check(s_with.direction == StudyDirection.MAXIMIZE, f"direction == MAXIMIZE (obtenu : {s_with.direction})")
    features_with = s_with.user_attrs.get("features")
    check(features_with is not None, "user_attr 'features' enregistré")
    if features_with is not None:
        for col in CONTRALATERAL:
            check(col in features_with, f"features enregistrées mais '{col}' absent")
    print(f"  n_features: {s_with.user_attrs.get('n_features')}")
except Exception as exc:
    errors.append(f"FAIL: impossible de charger prostate_xgb_with_contralateral — {exc}")
    print(f"  FAIL: {exc}")

print()
print("=== prostate_xgb_no_contralateral ===")
try:
    s_no = optuna.load_study(study_name="prostate_xgb_no_contralateral", storage=STORAGE)
    check(s_no.user_attrs.get("contralateral_used") is False, "contralateral_used == False")
    n_trials_no = len(s_no.trials)
    check(n_trials_no >= 1, f"nb trials >= 1 (obtenu : {n_trials_no})")
    check(s_no.direction == StudyDirection.MAXIMIZE, f"direction == MAXIMIZE (obtenu : {s_no.direction})")
    features_no = s_no.user_attrs.get("features")
    check(features_no is not None, "user_attr 'features' enregistré")
    if features_no is not None:
        for col in CONTRALATERAL:
            check(col not in features_no, f"features enregistrées mais '{col}' présent (devrait être absent)")
    print(f"  n_features: {s_no.user_attrs.get('n_features')}")
except Exception as exc:
    errors.append(f"FAIL: impossible de charger prostate_xgb_no_contralateral — {exc}")
    print(f"  FAIL: {exc}")

print()
print("=== prostate_xgb_parsimonious ===")
_REQUIRED_PARSIMONIOUS = {"psa_density", "pirads", "age", "prostate_volume", "diameter"}
try:
    s_pars = optuna.load_study(study_name="prostate_xgb_parsimonious", storage=STORAGE)
    check(s_pars.user_attrs.get("feature_set") == "parsimonious", "feature_set == 'parsimonious'")
    check(s_pars.user_attrs.get("contralateral_used") is False, "contralateral_used == False")
    n_trials_pars = len(s_pars.trials)
    check(n_trials_pars >= 1, f"nb trials >= 1 (obtenu : {n_trials_pars})")
    check(s_pars.direction == StudyDirection.MAXIMIZE, f"direction == MAXIMIZE (obtenu : {s_pars.direction})")
    features_pars = s_pars.user_attrs.get("features")
    check(features_pars is not None, "user_attr 'features' enregistré")
    n_features_pars = s_pars.user_attrs.get("n_features")
    check(n_features_pars in (5, 6), f"n_features in (5, 6) — obtenu : {n_features_pars}")
    if features_pars is not None:
        for col in _REQUIRED_PARSIMONIOUS:
            check(col in features_pars, f"'{col}' présent dans les features")
        for col in CONTRALATERAL:
            check(col not in features_pars, f"'{col}' absent des features parcimonieuses")
    print(f"  n_features: {n_features_pars}")
    print(f"  features: {features_pars}")
except Exception as exc:
    errors.append(f"FAIL: impossible de charger prostate_xgb_parsimonious — {exc}")
    print(f"  FAIL: {exc}")

print()
print("=== Cross-study diff check ===")
try:
    diff = set(s_with.user_attrs["features"]) - set(s_no.user_attrs["features"])
    check(
        diff == CONTRALATERAL,
        f"diff (with - no) == {CONTRALATERAL} — obtenu : {diff}"
    )
except (NameError, KeyError):
    errors.append("FAIL: une des deux études n'a pas pu être chargée ou ses features ne sont pas enregistrées, diff impossible")

print()
if errors:
    print(f"{'='*50}")
    print(f"ÉCHEC — {len(errors)} assertion(s) failed. NE PAS lancer les 300 trials.")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("Toutes les assertions passées. Safe to launch 300 trials.")
