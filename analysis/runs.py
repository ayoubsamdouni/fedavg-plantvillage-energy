"""Shared definitions: the 13 runs, their log files and their saved test-set predictions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
PREDS = ROOT / "results" / "predictions"
SUMMARY = ROOT / "results" / "summary"
FIGURES = ROOT / "figures"

SEEDS = (42, 43, 44)
LOCAL_EPOCHS = {"S1": 3, "S2": 3, "S3": 3, "S4": 1, "S5": 1}
ALPHA = {"S1": "IID", "S2": 0.5, "S3": 0.1, "S4": 0.1, "S5": 0.5}

# (scenario, partition seed) -> file stem shared by the log and the prediction file
RUNS = {
    ("S1", 42): "",
    ("S2", 42): "_dirichlet_a05",
    ("S2", 43): "_dirichlet_a05_seed43",
    ("S2", 44): "_dirichlet_a05_seed44",
    ("S3", 42): "_dirichlet_a01",
    ("S3", 43): "_dirichlet_a01_seed43",
    ("S3", 44): "_dirichlet_a01_seed44",
    ("S4", 42): "_dirichlet_a01_epoch1",
    ("S4", 43): "_dirichlet_a01_epoch1_seed43",
    ("S4", 44): "_dirichlet_a01_epoch1_seed44",
    ("S5", 42): "_dirichlet_a05_e1",
    ("S5", 43): "_dirichlet_a05_e1_seed43",
    ("S5", 44): "_dirichlet_a05_e1_seed44",
}

# bytes uploaded by the five clients in one round (full model state, float32)
BYTES_PER_ROUND = 21_689_440
N_TEST = 8_146


def log_file(sc, seed):
    return LOGS / f"fl_results{RUNS[(sc, seed)]}.csv"


def pred_file(sc, seed):
    return PREDS / f"fl_ckpt{RUNS[(sc, seed)]}_global_weights.npz.preds.npy"


def seeds_of(sc):
    return [s for (k, s) in RUNS if k == sc]
