"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import json
import math
import time
from dataclasses import dataclass, asdict

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def verify_macos_env():
    if sys.platform != "darwin":
        raise RuntimeError(f"This script requires macOS with Metal. Detected platform: {sys.platform}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS (Metal Performance Shaders) is not available. Ensure you are running on Apple Silicon with a compatible PyTorch build.")
    print("Environment verified: macOS detected with Metal (MPS) hardware acceleration available.")
    print()

verify_macos_env()

from prepare_genomic import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb, get_token_bytes
from model import GPT, GPTConfig, MuonAdamW

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "L"    # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 32768
EMBEDDING_LR = 0.68
UNEMBEDDING_LR = 0.0032
MATRIX_LR = 0.019
SCALAR_LR = 0.4
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.7
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 4               # number of transformer layers
DEVICE_BATCH_SIZE = 16  # per-device batch size (reduce if OOM)

# Convergence monitoring (mini-eval excluded from training_time budget)
EVAL_INTERVAL_STEPS = 10 # mini-eval every N optimizer steps; 0 disables
MINI_EVAL_BATCHES = 4   # fixed val batches per mini-eval (4*B*T ~= 131k tokens)

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Detect device
device_type = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_type)

# Autocast context
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
elif device_type == "cpu":
    autocast_ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
else:
    import contextlib
    autocast_ctx = contextlib.nullcontext()

H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# torch.compile is unstable on MPS, only use on CUDA
if device_type == "cuda":
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

# Pre-fetch a fixed set of val batches for cheap convergence-monitoring evals.
# Reusing the same batches every check makes the curve interpretable (no
# inter-eval batch noise) and keeps overhead off the 300s training budget.
mini_eval_batches = []
if MINI_EVAL_BATCHES > 0 and EVAL_INTERVAL_STEPS > 0:
    _val_loader_curve = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "val")
    for _ in range(MINI_EVAL_BATCHES):
        vx, vy, _ = next(_val_loader_curve)
        mini_eval_batches.append((vx, vy))
    del _val_loader_curve
_token_bytes_curve = get_token_bytes(device=device) if mini_eval_batches else None

def mini_eval_bpb():
    """Bits-per-byte over the fixed mini-eval batches. Toggles eval/train mode."""
    nats = 0.0
    nbytes_total = 0
    model.eval()
    try:
        with autocast_ctx, torch.no_grad():
            for vx, vy in mini_eval_batches:
                loss_flat = model(vx, vy, reduction='none').view(-1)
                y_flat = vy.view(-1)
                nbytes = _token_bytes_curve[y_flat]
                mask = nbytes > 0
                nats += (loss_flat * mask).sum().item()
                nbytes_total += nbytes.sum().item()
    finally:
        model.train()
    if nbytes_total == 0:
        return float('nan')
    return nats / (math.log(2) * nbytes_total)

# Output paths for val curve artifacts (next to checkpoint if set, else cwd).
_ckpt_env = os.environ.get("HARMONY_CHECKPOINT_PATH")
_artifact_dir = os.path.dirname(_ckpt_env) if _ckpt_env else "."
val_curve = []
val_curve_jsonl = os.path.join(_artifact_dir, "val_curve.jsonl") if mini_eval_batches else None
if val_curve_jsonl and os.path.exists(val_curve_jsonl):
    os.remove(val_curve_jsonl)  # fresh file per run

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")
if mini_eval_batches:
    print(f"Mini-eval: every {EVAL_INTERVAL_STEPS} steps on {MINI_EVAL_BATCHES} fixed val batches "
          f"(~{MINI_EVAL_BATCHES * DEVICE_BATCH_SIZE * MAX_SEQ_LEN // 1000}k tokens)")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()

while True:
    sync_device(device_type)
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding
    if train_loss_f > 100:
        print("FAIL")
        exit(1)

    sync_device(device_type)
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # Mini-eval for convergence curve (overhead excluded from total_training_time)
    if mini_eval_batches and (step + 1) % EVAL_INTERVAL_STEPS == 0:
        sync_device(device_type)
        t_eval0 = time.time()
        bpb_mini = mini_eval_bpb()
        sync_device(device_type)
        eval_dt = time.time() - t_eval0
        point = {
            "step": step + 1,
            "training_time": total_training_time,
            "progress": progress,
            "val_bpb": bpb_mini,
            "full_eval": False,
        }
        val_curve.append(point)
        if val_curve_jsonl:
            with open(val_curve_jsonl, "a") as _f:
                _f.write(json.dumps(point) + "\n")
        print(f"\n[mini-eval] step {step+1:>5d} val_bpb={bpb_mini:.4f} (overhead {eval_dt*1000:.0f}ms)", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
if device_type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
elif device_type == "mps":
    # MPS has no peak counter; driver_allocated_memory is the closest proxy
    peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

# Append the final full-budget eval to the curve and render a PNG.
if mini_eval_batches:
    final_point = {
        "step": step,
        "training_time": total_training_time,
        "progress": 1.0,
        "val_bpb": float(val_bpb),
        "full_eval": True,
    }
    val_curve.append(final_point)
    if val_curve_jsonl:
        with open(val_curve_jsonl, "a") as _f:
            _f.write(json.dumps(final_point) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt

        steps_arr = [pt["step"] for pt in val_curve]
        bpbs_arr = [pt["val_bpb"] for pt in val_curve]
        full_idx = [i for i, pt in enumerate(val_curve) if pt.get("full_eval")]

        fig, ax = _plt.subplots(figsize=(8, 5))
        ax.plot(steps_arr, bpbs_arr, "-o", markersize=3, linewidth=1,
                label=f"mini-eval ({MINI_EVAL_BATCHES} val batches, ~{MINI_EVAL_BATCHES*DEVICE_BATCH_SIZE*MAX_SEQ_LEN//1000}k tok)")
        for i in full_idx:
            ax.plot(steps_arr[i], bpbs_arr[i], "*", markersize=18, color="tab:red",
                    label="final full eval (~21M tok)" if i == full_idx[0] else None,
                    zorder=5)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("val_bpb")
        title_bits = [f"depth={DEPTH}", f"matrix_lr={MATRIX_LR}",
                      f"warmup={WARMUP_RATIO}", f"wd={WEIGHT_DECAY}"]
        ax.set_title("val_bpb curve  |  " + ", ".join(title_bits))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        png_path = os.path.join(_artifact_dir, "val_curve.png")
        fig.savefig(png_path, dpi=120)
        _plt.close(fig)
        print(f"val_curve_plot:   {png_path}")
        print(f"val_curve_jsonl:  {val_curve_jsonl}")
    except Exception as _e:
        print(f"val_curve_plot:   FAILED ({_e!r})")

# Across-runs progress chart: regenerated every run so the user sees the
# search trajectory accumulate. Reads existing results.csv + r{N}_*/stdout.txt
# directories. Flush first so this run's stdout file (if redirected) is on
# disk before the subprocess reads it.
sys.stdout.flush()
try:
    import subprocess as _sp_plot
    _plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_progress.py")
    if os.path.exists(_plot_script):
        _r = _sp_plot.run(
            [sys.executable, _plot_script],
            check=False, timeout=60, capture_output=True, text=True,
        )
        if _r.stdout.strip():
            print(_r.stdout.strip())
        if _r.returncode != 0:
            print(f"progress_plot:    FAILED (exit {_r.returncode}) {_r.stderr.strip()[:200]}")
except Exception as _e:
    print(f"progress_plot:    FAILED ({_e!r})")

# ---------------------------------------------------------------------------
# Optional checkpoint save
# Triggered by autoresearch_llm.py via HARMONY_CHECKPOINT_PATH; this block sits
# below the regex-patched hyperparameter region so it survives patch cycles.
# ---------------------------------------------------------------------------
_ckpt_path = os.environ.get("HARMONY_CHECKPOINT_PATH")
if _ckpt_path:
    import subprocess as _subprocess
    try:
        _git_sha = _subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=_subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        _git_sha = None
    _hyperparams = {
        "ASPECT_RATIO": ASPECT_RATIO, "HEAD_DIM": HEAD_DIM, "WINDOW_PATTERN": WINDOW_PATTERN,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE, "EMBEDDING_LR": EMBEDDING_LR,
        "UNEMBEDDING_LR": UNEMBEDDING_LR, "MATRIX_LR": MATRIX_LR, "SCALAR_LR": SCALAR_LR,
        "WEIGHT_DECAY": WEIGHT_DECAY, "ADAM_BETAS": ADAM_BETAS, "WARMUP_RATIO": WARMUP_RATIO,
        "WARMDOWN_RATIO": WARMDOWN_RATIO, "FINAL_LR_FRAC": FINAL_LR_FRAC,
        "DEPTH": DEPTH, "DEVICE_BATCH_SIZE": DEVICE_BATCH_SIZE,
    }
    os.makedirs(os.path.dirname(_ckpt_path) or ".", exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
        "config": asdict(config),
        "hyperparameters": _hyperparams,
        "val_bpb": float(val_bpb),
        "num_params": int(num_params),
        "num_steps": int(step),
        "training_seconds": float(total_training_time),
        "git_sha": _git_sha,
        "seed": 42,
    }, _ckpt_path)
    print(f"checkpoint_saved: {_ckpt_path}")
