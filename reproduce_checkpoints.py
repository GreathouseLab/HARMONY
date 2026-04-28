"""
Reproduce the four checkpoints needed for Probe 4 (and future probes).

Runs each of the four chosen historical configs sequentially, saves a checkpoint
to experiments/<name>/checkpoint.pt, and verifies val_bpb is within ±0.005 of
the originally recorded value. Outputs a summary table.

Configs: baseline_rerun, lower_matrix_lr, r2_lowlr_warmup_wd01, r3_lr_025

Each run is ~5 min training + ~2-5 min eval on M3 Pro. Total ~30-40 min.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
TRAIN_PY = PROJECT_DIR / "train.py"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python3.12")

# Reuse existing patch helpers from the autoresearch loop
sys.path.insert(0, str(PROJECT_DIR))
from autoresearch_llm import (  # type: ignore
    patch_train_py, restore_train_py, parse_results, STEP_DT_RE,
    FAST_FAIL_WARMUP_STEPS, FAST_FAIL_CHECK_STEPS, FAST_FAIL_DT_MS_THRESHOLD,
    SUBPROCESS_TIMEOUT,
)

# (name, overrides, original_val_bpb)
TARGETS = [
    ("baseline_rerun",        {},                                                                    1.951289),
    ("lower_matrix_lr",       {"MATRIX_LR": 0.02},                                                   1.935718),
    ("r2_lowlr_warmup_wd01",  {"MATRIX_LR": 0.02, "WARMUP_RATIO": 0.05, "WEIGHT_DECAY": 0.1},        1.932465),
    ("r3_lr_025",             {"MATRIX_LR": 0.025, "WARMUP_RATIO": 0.05, "WEIGHT_DECAY": 0.1},       1.950751),
]

TOLERANCE = 0.005


def run_one(name: str, overrides: dict, original_val_bpb: float) -> dict:
    print(f"\n{'='*72}")
    print(f"REPRODUCING: {name}")
    print(f"  overrides: {overrides}")
    print(f"  original val_bpb: {original_val_bpb:.6f}")
    print(f"{'='*72}\n")

    exp_dir = EXPERIMENTS_DIR / f"repro_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = exp_dir / "checkpoint.pt"
    stdout_path = exp_dir / "stdout.txt"

    original_src = patch_train_py(overrides)
    shutil.copy2(TRAIN_PY, exp_dir / "train.py")

    sub_env = os.environ.copy()
    sub_env["HARMONY_CHECKPOINT_PATH"] = str(checkpoint_path)

    proc = subprocess.Popen(
        [PYTHON, str(TRAIN_PY)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=sub_env,
    )

    dt_values: list[int] = []
    early_killed = False
    timed_out = False
    output_buf: list[str] = []
    t0 = time.time()

    try:
        with open(stdout_path, "w") as f_out:
            for line in proc.stdout:
                f_out.write(line)
                f_out.flush()
                output_buf.append(line)

                m = STEP_DT_RE.search(line)
                if m:
                    dt_values.append(int(m.group(1)))
                    needed = FAST_FAIL_WARMUP_STEPS + FAST_FAIL_CHECK_STEPS
                    if len(dt_values) == needed:
                        check = dt_values[FAST_FAIL_WARMUP_STEPS:needed]
                        avg = sum(check) / len(check)
                        if avg > FAST_FAIL_DT_MS_THRESHOLD:
                            print(f"\n  EARLY KILL: avg dt = {avg:.0f}ms over post-warmup steps")
                            proc.kill()
                            early_killed = True
                            break

                if time.time() - t0 > SUBPROCESS_TIMEOUT:
                    print(f"\n  TIMEOUT (>{SUBPROCESS_TIMEOUT}s)")
                    proc.kill()
                    timed_out = True
                    break

            try:
                tail, _ = proc.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                tail, _ = proc.communicate()
            if tail:
                f_out.write(tail)
                output_buf.append(tail)
    finally:
        restore_train_py(original_src)

    wall = time.time() - t0
    full_output = "".join(output_buf)
    metrics = parse_results(full_output)
    repro_val_bpb = metrics.get("val_bpb")
    saved = bool(re.search(r"checkpoint_saved:\s+\S+", full_output))

    diff = (repro_val_bpb - original_val_bpb) if repro_val_bpb is not None else None
    within_tol = (diff is not None) and (abs(diff) <= TOLERANCE)

    print()
    print(f"  reproduced val_bpb: {repro_val_bpb}")
    print(f"  diff vs original:   {diff}")
    print(f"  within ±{TOLERANCE}: {within_tol}")
    print(f"  checkpoint exists:  {checkpoint_path.exists()} ({checkpoint_path})")
    print(f"  wall time:          {wall:.0f}s")

    return {
        "name": name,
        "overrides": overrides,
        "original_val_bpb": original_val_bpb,
        "repro_val_bpb": repro_val_bpb,
        "diff": diff,
        "within_tolerance": within_tol,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": checkpoint_path.exists(),
        "wall_seconds": wall,
        "early_killed": early_killed,
        "timed_out": timed_out,
        "returncode": proc.returncode,
    }


def main():
    results = []
    for name, overrides, orig in TARGETS:
        results.append(run_one(name, overrides, orig))

    print("\n" + "=" * 72)
    print("REPRODUCTION SUMMARY")
    print("=" * 72)
    print(f"{'name':<24} {'orig':>10} {'repro':>10} {'diff':>10} {'within':>8} {'ckpt':>6}")
    for r in results:
        repro = f"{r['repro_val_bpb']:.6f}" if r['repro_val_bpb'] is not None else "—"
        diff = f"{r['diff']:+.6f}" if r['diff'] is not None else "—"
        print(f"{r['name']:<24} {r['original_val_bpb']:>10.6f} {repro:>10} {diff:>10} "
              f"{str(r['within_tolerance']):>8} {str(r['checkpoint_exists']):>6}")

    out_json = EXPERIMENTS_DIR / "reproduction_summary.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved full summary to {out_json}")

    failed = [r for r in results if not r["checkpoint_exists"]]
    drifted = [r for r in results if r["checkpoint_exists"] and not r["within_tolerance"]]
    if failed:
        print(f"\nWARNING: {len(failed)} run(s) failed to produce a checkpoint:")
        for r in failed:
            print(f"  - {r['name']}: returncode={r['returncode']} early_killed={r['early_killed']} timed_out={r['timed_out']}")
    if drifted:
        print(f"\nWARNING: {len(drifted)} run(s) produced val_bpb outside ±{TOLERANCE}:")
        for r in drifted:
            print(f"  - {r['name']}: orig={r['original_val_bpb']:.6f} repro={r['repro_val_bpb']:.6f} diff={r['diff']:+.6f}")
    if not failed and not drifted:
        print("\nAll 4 runs reproduced within tolerance and saved checkpoints. Ready for Probe 4.")


if __name__ == "__main__":
    main()
