"""
Usage :
    python check_db.py <fichier.db>
    python check_db.py databases/no_stacking_optuna.db
"""
import sys
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

if len(sys.argv) < 2:
    print("Usage : python check_db.py <fichier.db>")
    sys.exit(1)

db_path = sys.argv[1]
storage = f"sqlite:///{db_path}"

try:
    names = optuna.get_all_study_names(storage)
except Exception as e:
    print(f"Impossible d'ouvrir '{db_path}' : {e}")
    sys.exit(1)

if not names:
    print(f"'{db_path}' ne contient aucune étude.")
    sys.exit(0)

for name in names:
    study    = optuna.load_study(study_name=name, storage=storage)
    df       = study.trials_dataframe()
    complete = (df["state"] == "COMPLETE").sum()
    total    = len(df)

    print(f"\nstudy : {name}  |  {complete} COMPLETE / {total} total")
    if complete > 0:
        print(f"best  : AUC={study.best_value:.5f}  (trial #{study.best_trial.number})")

    cols = ["number", "state", "datetime_start", "datetime_complete", "value"]
    print(df[cols].tail(20).to_string(index=False))