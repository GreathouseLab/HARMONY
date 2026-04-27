"""
Autoresearch loop for HARMONY genomic language model.

Systematically tests hyperparameter configurations, runs 5-minute training
experiments, and tracks val_bpb to find the best setup.

Usage:
    python autoresearch.py                # run all experiments
    python autoresearch.py --resume       # skip already-completed experiments

Results are logged to experiments/results.csv
"""

import os
import sys
import csv
import json
import re
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
TRAIN_PY = PROJECT_DIR / "train.py"
TRAIN_PY_BACKUP = PROJECT_DIR / "train.py.baseline"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python3.12")

# ---------------------------------------------------------------------------
# Experiment definitions
#
# Each experiment overrides specific hyperparameters in train.py.
# The baseline run (experiment 0) has already been completed:
#   val_bpb = 1.943357, depth=4, aspect_ratio=64, 7.3M params
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # ---- Round 3: re-run timed-out deeper models (timeout now 2400s) ----
    # Current best: r2_lowlr_warmup_wd01 -> val_bpb 1.9325
    #   Config: MATRIX_LR=0.02, WARMUP_RATIO=0.05, WEIGHT_DECAY=0.1

    # -- Re-run deeper models that timed out in round 2 --
    {
        "name": "r3_depth6_lowlr",
        "description": "Depth 6 + lower LR (re-run with longer timeout)",
        "overrides": {
            "DEPTH": 6,
            "MATRIX_LR": 0.02,
        },
    },
    {
        "name": "r3_depth6_best_combo",
        "description": "Depth 6 + full winning combo from round 2",
        "overrides": {
            "DEPTH": 6,
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
        },
    },
    {
        "name": "r3_depth8_smallbatch",
        "description": "Depth 8 + batch_size=8 (re-run with longer timeout)",
        "overrides": {
            "DEPTH": 8,
            "DEVICE_BATCH_SIZE": 8,
        },
    },
    {
        "name": "r3_depth8_best_combo",
        "description": "Depth 8 + full winning combo + small batch",
        "overrides": {
            "DEPTH": 8,
            "DEVICE_BATCH_SIZE": 8,
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
        },
    },

    # -- Explore around the current winner (depth 4) --
    {
        "name": "r3_wd005",
        "description": "Weight decay 0.05 (lower than winner's 0.1)",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.05,
        },
    },
    {
        "name": "r3_wd015",
        "description": "Weight decay 0.15 (between winner's 0.1 and default 0.2)",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.15,
        },
    },
    {
        "name": "r3_warmup_3pct",
        "description": "Warmup 3% instead of 5%",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.03,
            "WEIGHT_DECAY": 0.1,
        },
    },
    {
        "name": "r3_warmup_10pct",
        "description": "Warmup 10% instead of 5%",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.10,
            "WEIGHT_DECAY": 0.1,
        },
    },
    {
        "name": "r3_lr_025",
        "description": "Muon LR 0.025 (between 0.02 winner and 0.04 default)",
        "overrides": {
            "MATRIX_LR": 0.025,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
        },
    },
    {
        "name": "r3_warmdown_70_combo",
        "description": "Winner combo + longer warmdown 70%",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
            "WARMDOWN_RATIO": 0.7,
        },
    },
    {
        "name": "r3_final_lr_01",
        "description": "Winner combo + final LR 10% instead of 0%",
        "overrides": {
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
            "FINAL_LR_FRAC": 0.1,
        },
    },
    {
        "name": "r3_depth5_best",
        "description": "Depth 5 + winning combo (between 4 and 6)",
        "overrides": {
            "DEPTH": 5,
            "MATRIX_LR": 0.02,
            "WARMUP_RATIO": 0.05,
            "WEIGHT_DECAY": 0.1,
        },
    },
]

# ---------------------------------------------------------------------------
# Hyperparameter patching
# ---------------------------------------------------------------------------

# Map of hyperparameter names to their regex patterns in train.py
HYPERPARAM_PATTERNS = {
    "ASPECT_RATIO": r"^(ASPECT_RATIO\s*=\s*).*",
    "HEAD_DIM": r"^(HEAD_DIM\s*=\s*).*",
    "WINDOW_PATTERN": r'^(WINDOW_PATTERN\s*=\s*).*',
    "TOTAL_BATCH_SIZE": r"^(TOTAL_BATCH_SIZE\s*=\s*).*",
    "EMBEDDING_LR": r"^(EMBEDDING_LR\s*=\s*).*",
    "UNEMBEDDING_LR": r"^(UNEMBEDDING_LR\s*=\s*).*",
    "MATRIX_LR": r"^(MATRIX_LR\s*=\s*).*",
    "SCALAR_LR": r"^(SCALAR_LR\s*=\s*).*",
    "WEIGHT_DECAY": r"^(WEIGHT_DECAY\s*=\s*).*",
    "ADAM_BETAS": r"^(ADAM_BETAS\s*=\s*).*",
    "WARMUP_RATIO": r"^(WARMUP_RATIO\s*=\s*).*",
    "WARMDOWN_RATIO": r"^(WARMDOWN_RATIO\s*=\s*).*",
    "FINAL_LR_FRAC": r"^(FINAL_LR_FRAC\s*=\s*).*",
    "DEPTH": r"^(DEPTH\s*=\s*).*",
    "DEVICE_BATCH_SIZE": r"^(DEVICE_BATCH_SIZE\s*=\s*).*",
}


def patch_train_py(overrides):
    """Apply hyperparameter overrides to train.py, return original content."""
    original = TRAIN_PY.read_text()
    patched = original

    for param, value in overrides.items():
        if param not in HYPERPARAM_PATTERNS:
            raise ValueError(f"Unknown hyperparameter: {param}")
        pattern = HYPERPARAM_PATTERNS[param]
        # Keep the comment on the same line if present
        new_line_value = str(value)
        patched = re.sub(
            pattern,
            lambda m: f"{m.group(1)}{new_line_value}",
            patched,
            flags=re.MULTILINE,
        )

    TRAIN_PY.write_text(patched)
    return original


def restore_train_py(original_content):
    """Restore train.py to its original state."""
    TRAIN_PY.write_text(original_content)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def parse_results(output):
    """Parse val_bpb and other metrics from train.py stdout."""
    results = {}
    patterns = {
        "val_bpb": r"val_bpb:\s+([\d.]+)",
        "training_seconds": r"training_seconds:\s+([\d.]+)",
        "total_seconds": r"total_seconds:\s+([\d.]+)",
        "total_tokens_M": r"total_tokens_M:\s+([\d.]+)",
        "num_steps": r"num_steps:\s+(\d+)",
        "num_params_M": r"num_params_M:\s+([\d.]+)",
        "depth": r"depth:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[key] = float(match.group(1))
    return results


def run_experiment(experiment, experiment_idx):
    """Run a single experiment, return results dict."""
    name = experiment["name"]
    desc = experiment["description"]
    overrides = experiment["overrides"]

    print(f"\n{'='*70}")
    print(f"EXPERIMENT {experiment_idx}: {name}")
    print(f"  {desc}")
    print(f"  Overrides: {overrides}")
    print(f"{'='*70}\n")

    # Patch train.py
    original = patch_train_py(overrides)

    # Create experiment output directory
    exp_dir = EXPERIMENTS_DIR / f"{experiment_idx:02d}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save the patched train.py for reference
    shutil.copy2(TRAIN_PY, exp_dir / "train.py")

    try:
        t0 = time.time()
        result = subprocess.run(
            [PYTHON, str(TRAIN_PY)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min max (5 min training + startup + eval on deeper models)
        )
        wall_time = time.time() - t0

        # Save full output
        (exp_dir / "stdout.txt").write_text(result.stdout)
        (exp_dir / "stderr.txt").write_text(result.stderr)

        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
            return {
                "experiment_idx": experiment_idx,
                "name": name,
                "description": desc,
                "overrides": json.dumps(overrides),
                "status": "FAILED",
                "val_bpb": None,
                "wall_time": wall_time,
            }

        # Parse results
        metrics = parse_results(result.stdout)
        val_bpb = metrics.get("val_bpb")

        print(f"  val_bpb: {val_bpb}")
        print(f"  params:  {metrics.get('num_params_M', '?')}M")
        print(f"  steps:   {metrics.get('num_steps', '?')}")
        print(f"  wall:    {wall_time:.0f}s")

        return {
            "experiment_idx": experiment_idx,
            "name": name,
            "description": desc,
            "overrides": json.dumps(overrides),
            "status": "OK",
            "val_bpb": val_bpb,
            "num_params_M": metrics.get("num_params_M"),
            "num_steps": metrics.get("num_steps"),
            "total_tokens_M": metrics.get("total_tokens_M"),
            "training_seconds": metrics.get("training_seconds"),
            "wall_time": wall_time,
            "timestamp": datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (>2400s)")
        return {
            "experiment_idx": experiment_idx,
            "name": name,
            "description": desc,
            "overrides": json.dumps(overrides),
            "status": "TIMEOUT",
            "val_bpb": None,
            "wall_time": 2400,
        }

    finally:
        # Always restore train.py
        restore_train_py(original)


def load_completed_experiments():
    """Load already-completed experiment names from results CSV."""
    completed = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "OK":
                    completed.add(row["name"])
    return completed


def save_result(result, is_first):
    """Append one result row to the CSV."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_idx", "name", "description", "overrides", "status",
        "val_bpb", "num_params_M", "num_steps", "total_tokens_M",
        "training_seconds", "wall_time", "timestamp",
    ]
    mode = "w" if is_first and not RESULTS_CSV.exists() else "a"
    write_header = not RESULTS_CSV.exists() or (is_first and mode == "w")

    with open(RESULTS_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="HARMONY autoresearch loop")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiments that already have OK results")
    args = parser.parse_args()

    # Backup original train.py
    if not TRAIN_PY_BACKUP.exists():
        shutil.copy2(TRAIN_PY, TRAIN_PY_BACKUP)
        print(f"Backed up train.py -> {TRAIN_PY_BACKUP.name}")

    completed = load_completed_experiments() if args.resume else set()
    if completed:
        print(f"Resuming: skipping {len(completed)} completed experiments")

    all_results = []
    is_first = True

    print(f"\nAutoresearch loop: {len(EXPERIMENTS)} experiments queued")
    print(f"Each experiment is a 5-minute training run on MPS")
    print(f"Results will be saved to {RESULTS_CSV}")
    print(f"Estimated total time: ~{len(EXPERIMENTS) * 10} minutes\n")

    for idx, experiment in enumerate(EXPERIMENTS):
        if experiment["name"] in completed:
            print(f"\nSkipping experiment {idx}: {experiment['name']} (already completed)")
            continue

        result = run_experiment(experiment, idx)
        all_results.append(result)
        save_result(result, is_first)
        is_first = False

    # Final summary
    print(f"\n{'='*70}")
    print("AUTORESEARCH RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Idx':<4} {'Name':<25} {'val_bpb':<10} {'Params':<10} {'Status'}")
    print(f"{'-'*70}")

    # Include any previously completed results if resuming
    if args.resume and RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            all_results = list(reader)

    ok_results = [r for r in all_results if r.get("status") == "OK" and r.get("val_bpb")]
    for r in all_results:
        bpb = str(r.get("val_bpb") or "N/A")
        params = str(r.get("num_params_M") or "N/A")
        status = str(r.get("status") or "?")
        idx = str(r.get("experiment_idx") or "?")
        name = str(r.get("name") or "?")
        print(f"{idx:<4} {name:<25} {bpb:<10} {params:<10} {status}")

    if ok_results:
        best = min(ok_results, key=lambda r: float(r["val_bpb"]))
        print(f"\nBEST: {best['name']} with val_bpb = {best['val_bpb']}")
        print(f"  Overrides: {best.get('overrides', '{}')}")
    else:
        print("\nNo successful experiments to compare.")

    print(f"\nFull results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
