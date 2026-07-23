"""
LLM-driven autoresearch loop for HARMONY genomic language model.

Uses Claude API to analyze experiment results and propose the next
hyperparameter configuration. Runs autonomously with budget controls.

Usage:
    python autoresearch_llm.py                          # default: 20 experiments max
    python autoresearch_llm.py --max-experiments 10     # limit experiments
    python autoresearch_llm.py --max-api-cost 5.0       # limit API spend to $5
    python autoresearch_llm.py --max-wall-hours 4       # stop after 4 hours
    python autoresearch_llm.py --dry-run                # show what Claude proposes without running

Requires:
    ANTHROPIC_API_KEY environment variable set

Budget controls:
    - --max-experiments: hard cap on number of training runs (default: 20)
    - --max-api-cost: cumulative Claude API cost limit in USD (default: 2.00)
    - --max-wall-hours: total wall clock limit (default: 6)
    - Each Claude API call costs ~$0.01-0.05 (Haiku) or ~$0.05-0.20 (Sonnet)
    - Each training run costs ~0 (local MPS, just electricity + time)

Results: experiments/results.csv (same format as autoresearch.py)
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

import anthropic
from anthropic import APIConnectionError
from anthropic._exceptions import OverloadedError  # 529, not re-exported at top level

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
TRAIN_PY = PROJECT_DIR / "train.py"
TRAIN_PY_BACKUP = PROJECT_DIR / "train.py.baseline"
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python3.12")


def _load_dotenv(path: Path = PROJECT_DIR / ".env"):
    """Minimal .env loader — KEY=VALUE per line, # comments, optional quotes.
    Existing env vars are not overwritten."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# Budget defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_EXPERIMENTS = 50
DEFAULT_MAX_API_COST_USD = 100.00
DEFAULT_MAX_WALL_HOURS = 24
SUBPROCESS_TIMEOUT = 2400  # 40 min per experiment

# Fast-fail: skip JIT/compilation noise, then kill if the next few steps are too slow.
# A doomed depth-8 run otherwise burns the full SUBPROCESS_TIMEOUT before we know.
FAST_FAIL_WARMUP_STEPS = 3        # ignore these step dts (JIT compile)
FAST_FAIL_CHECK_STEPS = 3         # average dt over this many subsequent steps
FAST_FAIL_DT_MS_THRESHOLD = 8000  # kill if avg dt > 8s — too slow to converge in 5min budget

# Pattern matches the train.py step log line, e.g.
#   "step 00012 (1.7%) | loss: 8.31 | lrm: 0.33 | dt: 6509ms | tok/sec: ..."
STEP_DT_RE = re.compile(r"step\s+\d+.*?\bdt:\s*(\d+)ms")

# Claude Sonnet 4.6 — best cost/capability balance for experiment reasoning
CLAUDE_MODEL = "claude-sonnet-4-6"

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

# ---------------------------------------------------------------------------
# Joint score (selection criterion as of 2026-05-07)
# ---------------------------------------------------------------------------

def compute_joint_score(result):
    """joint_score = val_bpb − 5.0 × (probe1_auc − 0.5). Lower is better.

    Missing probe1_auc is treated as 0.5 (neutral) so legacy runs without a
    probe still rank by val_bpb alone. Returns None if val_bpb is missing.
    """
    vb = result.get("val_bpb")
    if vb in (None, "", "None"):
        return None
    try:
        vb = float(vb)
    except (TypeError, ValueError):
        return None
    auc = result.get("probe1_auc")
    if auc in (None, "", "None"):
        return vb
    try:
        return vb - 5.0 * (float(auc) - 0.5)
    except (TypeError, ValueError):
        return vb


def _joint_score_sort_key(result):
    js = compute_joint_score(result)
    return js if js is not None else float("inf")


# ---------------------------------------------------------------------------
# Hyperparameter config (what Claude can modify)
# ---------------------------------------------------------------------------

HYPERPARAMS = {
    "DEPTH": {"type": "int", "range": [2, 12], "default": 4,
              "description": "Number of transformer layers. model_dim = DEPTH * ASPECT_RATIO. Depth >6 needs DEVICE_BATCH_SIZE=8 to avoid OOM on 22GB MPS."},
    "ASPECT_RATIO": {"type": "int", "range": [32, 128], "default": 64,
                     "description": "model_dim = DEPTH * ASPECT_RATIO. Higher = wider model per layer."},
    "HEAD_DIM": {"type": "int", "range": [64, 256], "default": 128,
                 "description": "Attention head dimension."},
    "WINDOW_PATTERN": {"type": "str", "options": ["L", "S", "SL", "SSL", "SSSL"],
                       "default": "L",
                       "description": "Sliding window pattern. L=full context, S=half context."},
    "TOTAL_BATCH_SIZE": {"type": "int", "range": [32768, 262144], "default": 65536,
                         "description": "Tokens per optimizer step. Must be power of 2."},
    "EMBEDDING_LR": {"type": "float", "range": [0.1, 1.0], "default": 0.6,
                     "description": "Learning rate for token embeddings (Adam)."},
    "UNEMBEDDING_LR": {"type": "float", "range": [0.001, 0.01], "default": 0.004,
                       "description": "Learning rate for lm_head (Adam)."},
    "MATRIX_LR": {"type": "float", "range": [0.005, 0.1], "default": 0.04,
                  "description": "Learning rate for matrix parameters (Muon optimizer)."},
    "SCALAR_LR": {"type": "float", "range": [0.1, 1.0], "default": 0.5,
                  "description": "Learning rate for per-layer scalar parameters."},
    "WEIGHT_DECAY": {"type": "float", "range": [0.0, 0.5], "default": 0.2,
                     "description": "Cautious weight decay for Muon optimizer."},
    "WARMUP_RATIO": {"type": "float", "range": [0.0, 0.2], "default": 0.0,
                     "description": "Fraction of time budget for LR warmup."},
    "WARMDOWN_RATIO": {"type": "float", "range": [0.0, 0.9], "default": 0.5,
                       "description": "Fraction of time budget for LR cooldown."},
    "FINAL_LR_FRAC": {"type": "float", "range": [0.0, 0.3], "default": 0.0,
                      "description": "Final LR as fraction of initial after warmdown."},
    "DEVICE_BATCH_SIZE": {"type": "int", "range": [4, 32], "default": 16,
                          "description": "Per-device batch size. Reduce to 8 for deeper models to avoid OOM."},
}

# Regex patterns for patching train.py
HYPERPARAM_PATTERNS = {
    name: rf"^({name}\s*=\s*).*" for name in HYPERPARAMS
}

# ---------------------------------------------------------------------------
# API cost tracking
# ---------------------------------------------------------------------------

class CostTracker:
    def __init__(self, max_cost_usd):
        self.max_cost_usd = max_cost_usd
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.num_calls = 0

    def add(self, input_tokens, output_tokens, model):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.num_calls += 1
        pricing = PRICING.get(model, PRICING[CLAUDE_MODEL])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.total_cost_usd += cost
        return cost

    def budget_remaining(self):
        return self.max_cost_usd - self.total_cost_usd

    def is_over_budget(self):
        return self.total_cost_usd >= self.max_cost_usd

    def summary(self):
        return (f"API calls: {self.num_calls} | "
                f"Tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out | "
                f"Cost: ${self.total_cost_usd:.4f} / ${self.max_cost_usd:.2f}")


# ---------------------------------------------------------------------------
# Hyperparameter patching (same as autoresearch.py)
# ---------------------------------------------------------------------------

def patch_train_py(overrides):
    original = TRAIN_PY.read_text()
    patched = original
    for param, value in overrides.items():
        if param not in HYPERPARAM_PATTERNS:
            raise ValueError(f"Unknown hyperparameter: {param}")
        pattern = HYPERPARAM_PATTERNS[param]
        str_value = str(value)
        if param == "WINDOW_PATTERN" and not str_value.startswith('"'):
            str_value = f'"{value}"'
        patched = re.sub(
            pattern,
            lambda m, sv=str_value: f"{m.group(1)}{sv}",
            patched,
            flags=re.MULTILINE,
        )
    TRAIN_PY.write_text(patched)
    return original


def restore_train_py(original_content):
    TRAIN_PY.write_text(original_content)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def evaluate_probe1(checkpoint_path: Path, exp_dir: Path,
                    val_path: Path = PROJECT_DIR / "output" / "val.txt",
                    n_reads_per_sample: int = 100):
    """Run Probe 1 (sample-coherence) on a freshly-trained checkpoint.
    Returns (auc, dmean) on success, (None, None) on any failure (failure does
    not abort the experiment — the val_bpb result is still useful)."""
    try:
        from probe_sample_coherence import run_probe1
        result = run_probe1(
            checkpoint_path=checkpoint_path,
            val_path=val_path,
            output_path=exp_dir / "probe1.json",
            n_reads_per_sample=n_reads_per_sample,
        )
        return result.get("auc"), result.get("delta_mean")
    except Exception as e:
        print(f"  [probe1] WARNING: probe1 evaluation failed: {e}")
        return None, None


def parse_results(output):
    results = {}
    patterns = {
        "val_bpb": r"val_bpb:\s+([\d.]+)",
        "training_seconds": r"training_seconds:\s+([\d.]+)",
        "total_seconds": r"total_seconds:\s+([\d.]+)",
        "peak_vram_mb": r"peak_vram_mb:\s+([\d.]+)",
        "mfu_percent": r"mfu_percent:\s+([\d.]+)",
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


def run_experiment(name, description, overrides, experiment_idx):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT {experiment_idx}: {name}")
    print(f"  {description}")
    print(f"  Overrides: {overrides}")
    print(f"{'='*70}\n")

    original = patch_train_py(overrides)
    exp_dir = EXPERIMENTS_DIR / f"{experiment_idx:02d}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAIN_PY, exp_dir / "train.py")
    stdout_path = exp_dir / "stdout.txt"
    stderr_path = exp_dir / "stderr.txt"
    checkpoint_path = exp_dir / "checkpoint.pt"

    base_record = {
        "experiment_idx": experiment_idx,
        "name": name,
        "description": description,
        "overrides": json.dumps(overrides),
    }

    # Subprocess env: inherit, plus point train.py's optional save block at the exp dir.
    sub_env = os.environ.copy()
    sub_env["HARMONY_CHECKPOINT_PATH"] = str(checkpoint_path)

    proc = subprocess.Popen(
        [PYTHON, str(TRAIN_PY)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merged so we can stream a single source
        text=True,
        bufsize=1,
        env=sub_env,
    )

    dt_values = []
    early_kill_reason = None
    timed_out = False
    output_buf = []
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
                            early_kill_reason = (
                                f"avg dt over post-warmup steps "
                                f"{FAST_FAIL_WARMUP_STEPS}..{needed - 1} "
                                f"= {avg:.0f}ms > {FAST_FAIL_DT_MS_THRESHOLD}ms threshold "
                                f"(samples: {check})"
                            )
                            print(f"\n  EARLY KILL: {early_kill_reason}")
                            proc.kill()
                            break

                if time.time() - t0 > SUBPROCESS_TIMEOUT:
                    timed_out = True
                    print(f"\n  TIMEOUT (>{SUBPROCESS_TIMEOUT}s) — killing")
                    proc.kill()
                    break

            # Drain anything left and reap the process
            try:
                tail, _ = proc.communicate(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                tail, _ = proc.communicate()
            if tail:
                f_out.write(tail)
                output_buf.append(tail)
    finally:
        restore_train_py(original)

    # We merged stderr into stdout; keep stderr.txt for backward compat
    stderr_path.write_text("")
    wall_time = time.time() - t0
    full_output = "".join(output_buf)

    if early_kill_reason:
        print(f"  EARLY_KILL ({wall_time:.0f}s)")
        return {
            **base_record,
            "status": "EARLY_KILL",
            "error": early_kill_reason,
            "val_bpb": None,
            "wall_time": wall_time,
            "timestamp": datetime.now().isoformat(),
        }

    if timed_out:
        return {
            **base_record,
            "status": "TIMEOUT",
            "val_bpb": None,
            "wall_time": wall_time,
            "timestamp": datetime.now().isoformat(),
        }

    if proc.returncode != 0:
        error_msg = full_output[-300:] if full_output else "unknown error"
        print(f"  FAILED (exit code {proc.returncode})")
        return {
            **base_record,
            "status": "FAILED",
            "error": error_msg,
            "val_bpb": None,
            "wall_time": wall_time,
            "timestamp": datetime.now().isoformat(),
        }

    metrics = parse_results(full_output)
    val_bpb = metrics.get("val_bpb")
    print(f"  val_bpb:      {val_bpb}")
    print(f"  params:       {metrics.get('num_params_M', '?')}M")
    print(f"  steps:        {metrics.get('num_steps', '?')}")
    print(f"  peak_vram_mb: {metrics.get('peak_vram_mb', '?')}")
    print(f"  wall:         {wall_time:.0f}s")

    # Probe 1 (sample-coherence) — diagnostic, runs only if the checkpoint was saved.
    probe1_auc, probe1_dmean = (None, None)
    if checkpoint_path.exists():
        print(f"  [probe1] evaluating sample-coherence on {checkpoint_path.name}…")
        probe1_auc, probe1_dmean = evaluate_probe1(checkpoint_path, exp_dir)
        if probe1_auc is not None:
            print(f"  probe1_auc:   {probe1_auc:.4f}  (Δmean={probe1_dmean:+.4f})")

    return {
        **base_record,
        "status": "OK",
        "val_bpb": val_bpb,
        "num_params_M": metrics.get("num_params_M"),
        "num_steps": metrics.get("num_steps"),
        "total_tokens_M": metrics.get("total_tokens_M"),
        "training_seconds": metrics.get("training_seconds"),
        "peak_vram_mb": metrics.get("peak_vram_mb"),
        "mfu_percent": metrics.get("mfu_percent"),
        "wall_time": wall_time,
        "timestamp": datetime.now().isoformat(),
        "probe1_auc": probe1_auc,
        "probe1_dmean": probe1_dmean,
    }


CSV_FIELDS = [
    "experiment_idx", "name", "description", "overrides", "status",
    "val_bpb", "num_params_M", "num_steps", "total_tokens_M",
    "training_seconds", "peak_vram_mb", "mfu_percent",
    "wall_time", "timestamp", "error",
    # Probe 1 (sample-coherence ROC-AUC, higher = better). Selection criterion
    # since 2026-05-07 is joint_score = val_bpb − 5.0×(probe1_auc−0.5), computed
    # on the fly via compute_joint_score(). See experiments/probes_r4_summary.md.
    "probe1_auc", "probe1_dmean",
]


def migrate_results_csv():
    """Idempotently expand results.csv to the current CSV_FIELDS schema.

    Older rows simply get empty values for new columns. Safe to call on every startup.
    """
    if not RESULTS_CSV.exists():
        return
    with open(RESULTS_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)
    additions = [c for c in CSV_FIELDS if c not in existing]
    if not additions:
        return
    new_fields = existing + additions
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            for col in additions:
                row.setdefault(col, "")
            writer.writerow(row)
    print(f"[csv] migrated results.csv: added columns {additions}")


def save_result(result):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def load_all_results():
    results = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
    return results


# ---------------------------------------------------------------------------
# Claude API integration
# ---------------------------------------------------------------------------

def _api_call_with_retry(client, max_retries=5, **kwargs):
    """client.messages.create with exponential backoff on transient errors
    (HTTP 529 OverloadedError, APIConnectionError). 5 retries, ~62s max wait.
    Added 2026-05-13 after a 529 mid-loop crashed an autoresearch run."""
    for attempt in range(max_retries + 1):
        try:
            return client.messages.create(**kwargs)
        except (OverloadedError, APIConnectionError) as e:
            if attempt == max_retries:
                raise
            delay = 2 ** (attempt + 1)  # 2, 4, 8, 16, 32s
            print(f"  [API] Transient {type(e).__name__}: {e}. "
                  f"Retry {attempt+1}/{max_retries} in {delay}s…")
            time.sleep(delay)


SYSTEM_PROMPT = """You are an AI research scientist optimizing a genomic language model.

You are part of an automated research loop:
1. You analyze past experiment results
2. You propose ONE new hyperparameter configuration to try
3. The system runs a 5-minute training experiment and reports val_bpb AND probe1_auc
4. The system computes joint_score = val_bpb − 5.0 × (probe1_auc − 0.5)
5. Repeat

OBJECTIVE (CHANGED 2026-05-07):
- Minimize joint_score = val_bpb − 5.0 × (probe1_auc − 0.5). Lower is better.
- val_bpb is bits per byte (lower = better LM fit).
- probe1_auc is sample-coherence ROC-AUC (~0.50 = chance, higher = more sample-discriminating embeddings).
- Joint scoring rewards good val_bpb AND high sample-coherence simultaneously.


CALIBRATION ANCHORS:
- r49 (val_bpb champion): val_bpb=1.911, probe1_auc=0.520, joint_score=1.811
- r34_deeper_depth6 (probe1_auc champion): val_bpb=1.947, probe1_auc=0.5255, joint_score=1.819
- These two are nearly tied under joint_score — that is the inflection point.
- 0.01 gain in probe1_auc ≈ 0.05 reduction in val_bpb (worth-it threshold).

AVOID THIS BASIN (R5 36/50 confirmed it's exhausted, including all window-pattern variants):
- DEPTH=2 with ASPECT_RATIO=128 in ANY WINDOW_PATTERN configuration (L, S, SL, SSL all tried)
- TOTAL_BATCH_SIZE=32k with WARMDOWN_RATIO=0.7
- These configurations cluster around joint_score ~1.81 with probe1_auc ~0.52 — they represent a local minimum of val_bpb at the cost of community discrimination collapse.
- The objective is NOT to mine variants of this basin further. We need configs OUTSIDE it.

SPECIFICALLY EXPLORE (underrepresented in n=49):
- Deeper models: DEPTH ∈ {{6, 8}}. Remember DEVICE_BATCH_SIZE=8 for DEPTH>=6.
- Wider models: ASPECT_RATIO ∈ {{96, 128}} (model_dim = DEPTH × ASPECT_RATIO, rounded up to HEAD_DIM=128 multiple — e.g. DEPTH=4×ASPECT_RATIO=128 → model_dim=512).
- Sliding-window attention: WINDOW_PATTERN='S' (biologically grounded for short reads; never tried).
- Longer warmup: WARMUP_RATIO ∈ {{0.05, 0.10}}.

HARDWARE CONSTRAINTS (critical — violating these wastes an experiment):
- Apple Silicon Mac with 22GB MPS memory
- DEPTH >= 6 MUST use DEVICE_BATCH_SIZE=8 (otherwise OOM)
- DEPTH >= 10 will likely OOM even with batch_size=4 — avoid
- Deeper models (DEPTH >= 6) take much longer (~20+ min total wall time)
- Each experiment has a 40-minute timeout

AVAILABLE HYPERPARAMETERS:
{hyperparams_json}

GUIDELINES:
- Change 1-3 hyperparameters at a time
- Learn from failures: OOM means too large, TIMEOUT means too slow
- The genomic data is DNA sequences (A/C/G/T/N) with special tokens — very different from NLP
- DNA has ~2 bits/base of theoretical entropy, but BPE compression and corpus-specific structure can push val_bpb below 2.0 (r49 at 1.911 demonstrates this; the 2.0 figure is not a hard floor)
- Be bold — the val_bpb basin is well-mapped, explore underexplored regions
- Don't repeat failed configurations
- The history below is biased toward val_bpb-corner configs; apply joint_score weighting when reading it, don't just imitate past patterns

RESPONSE FORMAT:
You MUST respond with a JSON object (and nothing else) in this exact format:
{{
    "name": "short_experiment_name",
    "description": "one line explaining the hypothesis",
    "overrides": {{"PARAM_NAME": value, ...}},
    "reasoning": "2-3 sentences on why this might improve joint_score, citing past results"
}}
Do NOT include any text before or after the JSON object."""


def build_results_summary(results):
    """Format experiment history for Claude."""
    lines = ["EXPERIMENT HISTORY (sorted by joint_score, best first):\n"]

    ok_results = [r for r in results if r.get("status") == "OK" and r.get("val_bpb")]
    ok_results.sort(key=_joint_score_sort_key)

    for r in ok_results:
        vram = r.get("peak_vram_mb") or "?"
        auc = r.get("probe1_auc")
        auc_str = f"{float(auc):.4f}" if auc not in (None, "", "None") else "n/a"
        js = compute_joint_score(r)
        js_str = f"{js:.4f}" if js is not None else "n/a"
        lines.append(
            f"  {r['name']}: joint_score={js_str} | val_bpb={r['val_bpb']} | probe1_auc={auc_str} | "
            f"params={r.get('num_params_M', '?')}M | "
            f"steps={r.get('num_steps', '?')} | "
            f"vram_mb={vram} | "
            f"overrides={r.get('overrides', '{}')}"
        )

    failed = [r for r in results if r.get("status") in ("FAILED", "TIMEOUT", "EARLY_KILL")]
    if failed:
        lines.append("\nFAILED/TIMED OUT/EARLY-KILLED experiments (avoid repeating these):")
        for r in failed:
            note = ""
            if r.get("status") == "EARLY_KILL" and r.get("error"):
                note = f" — {r['error']}"
            lines.append(
                f"  {r['name']}: status={r['status']} | "
                f"overrides={r.get('overrides', '{}')}{note}"
            )

    lines.append(f"\nTotal experiments run: {len(results)}")
    if ok_results:
        best_js = compute_joint_score(ok_results[0])
        best_js_str = f"{best_js:.4f}" if best_js is not None else "N/A"
        lines.append(
            f"Best joint_score so far: {best_js_str} "
            f"(name={ok_results[0]['name']}, val_bpb={ok_results[0]['val_bpb']}, "
            f"probe1_auc={ok_results[0].get('probe1_auc', 'n/a')})"
        )
    else:
        lines.append("Best joint_score so far: N/A")

    return "\n".join(lines)


def ask_claude(client, results, cost_tracker, model=CLAUDE_MODEL):
    """Ask Claude to propose the next experiment."""
    history_summary = build_results_summary(results)
    hyperparams_json = json.dumps(HYPERPARAMS, indent=2)

    system = SYSTEM_PROMPT.format(hyperparams_json=hyperparams_json)
    user_msg = f"""Here are all experiment results so far:

{history_summary}

Based on these results, propose ONE new experiment to try. Remember:
- DEPTH >= 6 needs DEVICE_BATCH_SIZE=8
- DEPTH >= 10 will OOM — avoid
- Don't repeat configurations that already failed
- Focus on lowering joint_score = val_bpb − 5.0 × (probe1_auc − 0.5) (not val_bpb alone)

Respond with ONLY a JSON object."""

    response = _api_call_with_retry(
        client,
        model=model,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    # Track cost
    cost = cost_tracker.add(
        response.usage.input_tokens,
        response.usage.output_tokens,
        model,
    )
    print(f"  [API] {response.usage.input_tokens} in / {response.usage.output_tokens} out | cost: ${cost:.4f} | {cost_tracker.summary()}")

    # Parse response
    text = response.content[0].text.strip()
    # Handle potential markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"```json?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

    try:
        proposal = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [API] Failed to parse response: {e}")
        print(f"  [API] Raw response: {text[:500]}")
        return None

    # Validate
    required_keys = {"name", "description", "overrides", "reasoning"}
    if not required_keys.issubset(proposal.keys()):
        print(f"  [API] Missing keys: {required_keys - proposal.keys()}")
        return None

    # Validate overrides
    for param in proposal["overrides"]:
        if param not in HYPERPARAMS:
            print(f"  [API] Unknown hyperparameter: {param}")
            return None

    return proposal


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM-driven HARMONY autoresearch loop")
    parser.add_argument("--max-experiments", type=int, default=DEFAULT_MAX_EXPERIMENTS,
                        help=f"Max experiments to run (default: {DEFAULT_MAX_EXPERIMENTS})")
    parser.add_argument("--max-api-cost", type=float, default=DEFAULT_MAX_API_COST_USD,
                        help=f"Max Claude API cost in USD (default: ${DEFAULT_MAX_API_COST_USD:.2f})")
    parser.add_argument("--max-wall-hours", type=float, default=DEFAULT_MAX_WALL_HOURS,
                        help=f"Max total wall clock hours (default: {DEFAULT_MAX_WALL_HOURS})")
    parser.add_argument("--model", type=str, default=CLAUDE_MODEL,
                        help=f"Claude model to use (default: {CLAUDE_MODEL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show Claude's proposals without running experiments")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Get a key at: https://console.anthropic.com/")
        sys.exit(1)

    client = anthropic.Anthropic()
    cost_tracker = CostTracker(args.max_api_cost)

    # Backup train.py
    if not TRAIN_PY_BACKUP.exists():
        shutil.copy2(TRAIN_PY, TRAIN_PY_BACKUP)

    # Bring results.csv up to the current schema (no-op if already migrated)
    migrate_results_csv()

    # Load existing results
    results = load_all_results()

    t_loop_start = time.time()
    experiments_run = 0

    print(f"HARMONY LLM Autoresearch Loop")
    print(f"{'='*70}")
    print(f"Model:           {args.model}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"Max API cost:    ${args.max_api_cost:.2f}")
    print(f"Max wall time:   {args.max_wall_hours}h")
    print(f"Existing results: {len(results)} experiments")
    print(f"{'='*70}\n")

    while experiments_run < args.max_experiments:
        # Check budget limits
        elapsed_hours = (time.time() - t_loop_start) / 3600
        if elapsed_hours >= args.max_wall_hours:
            print(f"\nWall time limit reached ({args.max_wall_hours}h). Stopping.")
            break

        if cost_tracker.is_over_budget():
            print(f"\nAPI cost limit reached (${args.max_api_cost:.2f}). Stopping.")
            break

        # Ask Claude for next experiment
        print(f"\n--- Asking Claude for experiment {experiments_run + 1}/{args.max_experiments} ---")
        print(f"  Budget remaining: ${cost_tracker.budget_remaining():.4f} API | "
              f"{args.max_wall_hours - elapsed_hours:.1f}h wall time")

        proposal = ask_claude(client, results, cost_tracker, model=args.model)

        if proposal is None:
            print("  Claude returned invalid proposal. Retrying...")
            continue

        print(f"\n  Claude proposes: {proposal['name']}")
        print(f"  Hypothesis: {proposal['description']}")
        print(f"  Reasoning: {proposal['reasoning']}")
        print(f"  Overrides: {proposal['overrides']}")

        if args.dry_run:
            print("  [DRY RUN] Skipping training run.")
            experiments_run += 1
            continue

        # Run the experiment
        experiment_idx = len(results)
        result = run_experiment(
            proposal["name"],
            proposal["description"],
            proposal["overrides"],
            experiment_idx,
        )
        result["reasoning"] = proposal.get("reasoning", "")
        save_result(result)
        results.append(result)
        experiments_run += 1

        # Report progress
        ok_results = [r for r in results if r.get("status") == "OK" and r.get("val_bpb")]
        if ok_results:
            best = min(ok_results, key=_joint_score_sort_key)
            best_js = compute_joint_score(best)
            best_js_str = f"{best_js:.4f}" if best_js is not None else "n/a"
            best_auc = best.get("probe1_auc")
            best_auc_str = (f"{float(best_auc):.4f}"
                            if best_auc not in (None, "", "None") else "n/a")
            print(f"\n  Current best: {best['name']} with joint_score = {best_js_str} "
                  f"(val_bpb={best['val_bpb']}, probe1_auc={best_auc_str})")

    # Final summary
    print(f"\n{'='*70}")
    print("AUTORESEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"Experiments this session: {experiments_run}")
    print(f"Total experiments: {len(results)}")
    print(f"{cost_tracker.summary()}")
    print(f"Wall time: {(time.time() - t_loop_start) / 3600:.2f}h")

    ok_results = [r for r in results if r.get("status") == "OK" and r.get("val_bpb")]
    if ok_results:
        ok_results.sort(key=_joint_score_sort_key)
        print(f"\nTop 5 results (by joint_score):")
        for i, r in enumerate(ok_results[:5]):
            js = compute_joint_score(r)
            js_str = f"{js:.4f}" if js is not None else "n/a"
            auc = r.get("probe1_auc")
            auc_str = (f"{float(auc):.4f}"
                       if auc not in (None, "", "None") else "n/a")
            print(f"  {i+1}. {r['name']}: joint_score={js_str} | "
                  f"val_bpb={r['val_bpb']} | probe1_auc={auc_str} | "
                  f"overrides={r.get('overrides', '{}')}")
        best_js = compute_joint_score(ok_results[0])
        best_js_str = f"{best_js:.4f}" if best_js is not None else "n/a"
        print(f"\nBEST: {ok_results[0]['name']} with joint_score = {best_js_str} "
              f"(val_bpb={ok_results[0]['val_bpb']})")

    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
