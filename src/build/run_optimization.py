"""
Pipeline d'optimisation automatisé — Phase 1 + Phase 2

Exécution :
    python run_optimization.py

Sorties :
    results/phase1/   — optuna.db snapshot + best_params_phase1.json + log
    results/phase2/   — optuna.db snapshot + phase2_best_params.json  + log

Le script est relançable : chaque phase reprend là où elle s'était arrêtée.
"""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PHASE1_DIR = RESULTS / "phase1"
PHASE2_DIR = RESULTS / "phase2"
OPTUNA_DB  = ROOT / "optuna.db"
PYTHON     = sys.executable   # même interpréteur que celui utilisé pour ce script


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, file=None) -> None:
    line = f"[{ts()}] {msg}"
    print(line)
    if file:
        file.write(line + "\n")
        file.flush()


def run_phase(
    script: Path,
    phase_dir: Path,
    log_name: str,
) -> bool:
    """
    Lance un script Python, capture stdout/stderr en temps réel,
    et retourne True si le processus s'est terminé avec code 0.
    """
    phase_dir.mkdir(parents=True, exist_ok=True)
    log_path = phase_dir / log_name

    with open(log_path, "w", encoding="utf-8") as logf:
        log(f"Démarrage : {script.name}", logf)
        log(f"Log       : {log_path}", logf)

        proc = subprocess.Popen(
            [PYTHON, str(script)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.rstrip()
            print(line)
            logf.write(line + "\n")
            logf.flush()

        proc.wait()

    success = proc.returncode == 0
    status = "OK" if success else f"ERREUR (code {proc.returncode})"
    log(f"{script.name} terminé — {status}")
    return success


def save_snapshot(phase_dir: Path, label: str) -> None:
    """Copie optuna.db dans le dossier de la phase pour archivage."""
    if OPTUNA_DB.exists():
        dest = phase_dir / "optuna_snapshot.db"
        shutil.copy2(OPTUNA_DB, dest)
        log(f"[{label}] snapshot DB → {dest}")


def _load_phase1_study():
    """Charge l'étude Phase 1 depuis optuna.db. Lève RuntimeError si introuvable."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sys.path.append(str(ROOT / "src"))
    from common.config import cfg_get, CONFIG

    study_base   = cfg_get(CONFIG, "optuna.study_name", "prostate_xgb")
    storage      = cfg_get(CONFIG, "optuna.storage", "sqlite:///optuna.db")
    # optimize.py construit le nom comme f"{study_base}_{FEATURE_SET}" avec FEATURE_SET="all"
    candidates   = (f"{study_base}_all", study_base)
    available    = optuna.get_all_study_names(storage)

    for candidate in candidates:
        if candidate in available:
            return optuna.load_study(study_name=candidate, storage=storage), storage

    raise RuntimeError(
        f"Étude Phase 1 introuvable dans optuna.db.\n"
        f"  Cherché : {candidates}\n"
        f"  Disponible : {available}"
    )


def phase1_already_complete() -> bool:
    """Retourne True si l'étude Phase 1 a déjà atteint n_trials."""
    try:
        import sys as _sys
        _sys.path.append(str(ROOT / "src"))
        from common.config import cfg_get, CONFIG
        n_trials = cfg_get(CONFIG, "optuna.n_trials", 100)
        study, _ = _load_phase1_study()
        n_complete = sum(
            1 for t in study.trials
            if t.state.name == "COMPLETE"
        )
        if n_complete >= n_trials:
            log(f"[phase1] déjà complète ({n_complete}/{n_trials} trials) — phase skippée")
            return True
        log(f"[phase1] {n_complete}/{n_trials} trials complétés — lancement")
        return False
    except Exception:
        # Pas encore d'étude → à lancer
        return False


def save_phase1_params(phase_dir: Path) -> None:
    """Lit l'étude Phase 1 dans optuna.db et sauvegarde les meilleurs params."""
    try:
        study, _ = _load_phase1_study()
        output = {
            "study_name": study.study_name,
            "n_trials":   len(study.trials),
            "best_auc":   study.best_value,
            "best_trial": study.best_trial.number,
            "params":     study.best_params,
        }
        dest = phase_dir / "best_params_phase1.json"
        dest.write_text(json.dumps(output, indent=2))
        log(f"[phase1] meilleurs params → {dest}")
        log(f"[phase1] best AUC = {study.best_value:.4f}  (trial #{study.best_trial.number})")
    except RuntimeError as exc:
        # Nom d'étude introuvable — erreur visible, pas silencieuse
        log(f"[phase1] AVERTISSEMENT archivage échoué : {exc}")
    except Exception as exc:
        log(f"[phase1] AVERTISSEMENT archivage échoué : {exc}")


def save_phase2_params(phase_dir: Path) -> None:
    """Copie le JSON de résultats Phase 2 produit par optimize_phase2.py."""
    src = ROOT / "runs" / "phase2_best_params.json"
    if src.exists():
        dest = phase_dir / "phase2_best_params.json"
        shutil.copy2(src, dest)
        log(f"[phase2] best params → {dest}")
    else:
        log("[phase2] phase2_best_params.json introuvable (Phase 2 non complétée ?)")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  Pipeline Optuna — Phase 1 + Phase 2")
    print(f"  Démarré : {ts()}")
    print("=" * 70)

    RESULTS.mkdir(parents=True, exist_ok=True)

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 1 — Optimisation large")
    print("─" * 70)
    t0 = time.time()

    if phase1_already_complete():
        ok1 = True
    else:
        ok1 = run_phase(
            script   = ROOT / "src" / "build" / "optimize.py",
            phase_dir= PHASE1_DIR,
            log_name = "optimize_phase1.log",
        )
        elapsed1 = time.time() - t0
        log(f"Phase 1 durée : {elapsed1/60:.1f} min")

    save_snapshot(PHASE1_DIR, "phase1")
    save_phase1_params(PHASE1_DIR)

    if not ok1:
        print("\n[ARRÊT] Phase 1 a échoué — Phase 2 non lancée.")
        print(f"Consulter le log : {PHASE1_DIR / 'optimize_phase1.log'}")
        sys.exit(1)

    # ── Phase 2 ────────────────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PHASE 2 — Affinement sur top trials")
    print("─" * 70)
    t0 = time.time()

    ok2 = run_phase(
        script   = ROOT / "src" / "build" / "optimize_phase2.py",
        phase_dir= PHASE2_DIR,
        log_name = "optimize_phase2.log",
    )

    elapsed2 = time.time() - t0
    log(f"Phase 2 durée : {elapsed2/60:.1f} min")

    save_snapshot(PHASE2_DIR, "phase2")
    save_phase2_params(PHASE2_DIR)

    # ── Résumé ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Résumé final")
    print("=" * 70)
    for label, ok, d in [("Phase 1", ok1, PHASE1_DIR), ("Phase 2", ok2, PHASE2_DIR)]:
        status = "OK" if ok else "ECHEC"
        print(f"  {label} : {status}  →  {d}")
    print(f"\n  Terminé : {ts()}")
    print("=" * 70)

    if not ok2:
        sys.exit(1)


if __name__ == "__main__":
    main()
