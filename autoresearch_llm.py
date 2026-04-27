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
# Budget defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_EXPERIMENTS = 50
DEFAULT_MAX_API_COST_USD = 100.00
DEFAULT_MAX_WALL_HOURS = 24
SUBPROCESS_TIMEOUT = 2400  # 40 min per experiment

# Claude Sonnet 4.6 — best cost/capability balance for experiment reasoning
CLAUDE_MODEL = "claude-sonnet-4-6"

# Pricing per 1M tokens (as of 2025)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
}

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

def parse_results(output):
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

    try:
        t0 = time.time()
        result = subprocess.run(
            [PYTHON, str(TRAIN_PY)],
            cwd=str(PROJECT_DIR),
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )
        wall_time = time.time() - t0

        (exp_dir / "stdout.txt").write_text(result.stdout)
        (exp_dir / "stderr.txt").write_text(result.stderr)

        if result.returncode != 0:
            error_msg = result.stderr[-300:] if result.stderr else "unknown error"
            print(f"  FAILED (exit code {result.returncode})")
            return {
                "experiment_idx": experiment_idx,
                "name": name,
                "description": description,
                "overrides": json.dumps(overrides),
                "status": "FAILED",
                "error": error_msg,
                "val_bpb": None,
                "wall_time": wall_time,
                "timestamp": datetime.now().isoformat(),
            }

        metrics = parse_results(result.stdout)
        val_bpb = metrics.get("val_bpb")
        print(f"  val_bpb: {val_bpb}")
        print(f"  params:  {metrics.get('num_params_M', '?')}M")
        print(f"  steps:   {metrics.get('num_steps', '?')}")
        print(f"  wall:    {wall_time:.0f}s")

        return {
            "experiment_idx": experiment_idx,
            "name": name,
            "description": description,
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
        print(f"  TIMEOUT (>{SUBPROCESS_TIMEOUT}s)")
        return {
            "experiment_idx": experiment_idx,
            "name": name,
            "description": description,
            "overrides": json.dumps(overrides),
            "status": "TIMEOUT",
            "val_bpb": None,
            "wall_time": SUBPROCESS_TIMEOUT,
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        restore_train_py(original)


def save_result(result):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_idx", "name", "description", "overrides", "status",
        "val_bpb", "num_params_M", "num_steps", "total_tokens_M",
        "training_seconds", "wall_time", "timestamp",
    ]
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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

SYSTEM_PROMPT = """You are an AI research scientist optimizing a genomic language model.

You are part of an automated research loop:
1. You analyze past experiment results
2. You propose ONE new hyperparameter configuration to try
3. The system runs a 5-minute training experiment and reports val_bpb (bits per byte, lower is better)
4. Repeat

HARDWARE CONSTRAINTS (critical — violating these wastes an experiment):
- Apple Silicon Mac with 22GB MPS memory
- DEPTH >= 6 MUST use DEVICE_BATCH_SIZE=8 (otherwise OOM)
- DEPTH >= 10 will likely OOM even with batch_size=4 — avoid
- Deeper models (DEPTH >= 6) take much longer for evaluation (~20+ min total wall time)
- Each experiment has a 40-minute timeout — if startup + training + eval exceeds this, it times out

AVAILABLE HYPERPARAMETERS:
{hyperparams_json}

GUIDELINES:
- Focus on val_bpb improvement — that's the only metric that matters
- Change 1-3 hyperparameters at a time to understand what works
- Learn from failures: OOM means too large, TIMEOUT means too slow on this hardware
- The genomic data is DNA sequences (A/C/G/T/N) with special tokens — very different from NLP
- DNA has ~2 bits/base of entropy, so val_bpb near 2.0 is approaching theoretical limits
- Be bold but learn from mistakes — don't repeat failed configurations

RESPONSE FORMAT:
You MUST respond with a JSON object (and nothing else) in this exact format:
{{
    "name": "short_experiment_name",
    "description": "one line explaining the hypothesis",
    "overrides": {{"PARAM_NAME": value, ...}},
    "reasoning": "2-3 sentences on why this might improve val_bpb based on past results"
}}

Do NOT include any text before or after the JSON object."""


def build_results_summary(results):
    """Format experiment history for Claude."""
    lines = ["EXPERIMENT HISTORY (sorted by val_bpb, best first):\n"]

    ok_results = [r for r in results if r.get("status") == "OK" and r.get("val_bpb")]
    ok_results.sort(key=lambda r: float(r["val_bpb"]))

    for r in ok_results:
        lines.append(
            f"  {r['name']}: val_bpb={r['val_bpb']} | "
            f"params={r.get('num_params_M', '?')}M | "
            f"steps={r.get('num_steps', '?')} | "
            f"overrides={r.get('overrides', '{}')}"
        )

    failed = [r for r in results if r.get("status") in ("FAILED", "TIMEOUT")]
    if failed:
        lines.append("\nFAILED/TIMED OUT experiments (avoid repeating these):")
        for r in failed:
            lines.append(
                f"  {r['name']}: status={r['status']} | "
                f"overrides={r.get('overrides', '{}')}"
            )

    lines.append(f"\nTotal experiments run: {len(results)}")
    lines.append(f"Best val_bpb so far: {ok_results[0]['val_bpb'] if ok_results else 'N/A'}")

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
- Focus on lowering val_bpb

Respond with ONLY a JSON object."""

    response = client.messages.create(
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
            best = min(ok_results, key=lambda r: float(r["val_bpb"]))
            print(f"\n  Current best: {best['name']} with val_bpb = {best['val_bpb']}")

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
        ok_results.sort(key=lambda r: float(r["val_bpb"]))
        print(f"\nTop 5 results:")
        for i, r in enumerate(ok_results[:5]):
            print(f"  {i+1}. {r['name']}: val_bpb={r['val_bpb']} | overrides={r.get('overrides', '{}')}")
        print(f"\nBEST: {ok_results[0]['name']} with val_bpb = {ok_results[0]['val_bpb']}")

    print(f"\nFull results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
