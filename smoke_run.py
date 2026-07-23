"""
D7 smoke driver: train_mlm for <=200 steps OR <=15min, then run Probe 1 in-process.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from train_mlm import train, cli, GPT_MLM
from prepare_genomic import Tokenizer
from probe_sample_coherence import run_probe1


if __name__ == "__main__":
    args = cli()
    args.max_steps = 200
    args.max_runtime_hours = 15.0 / 60.0   # 15 minutes
    args.out_dir = "experiments/mlm_smoke"

    print("=" * 70)
    print("D7 SMOKE: train_mlm <=200 steps or <=15min, then Probe 1 in-process")
    print("=" * 70)
    t0 = time.time()
    init_losses, last_losses, ckpt_path = train(args)
    dt = time.time() - t0

    # Re-create model identical to the trained one and re-load its state_dict.
    # train() returned the path; but model was discarded inside train(). Easiest:
    # repeat build + load.
    from train_mlm import build_model, EXTENDED_VOCAB
    from device_utils import get_device
    device = get_device()
    model, config = build_model(device, depth=args.depth, aspect_ratio=args.aspect_ratio, seq_len=args.seq_len)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tk = Tokenizer.from_directory()

    n_steps = ckpt.get("step", "?")
    ms_per_step = (dt / max(1, (n_steps + 1))) * 1000 if isinstance(n_steps, int) else None
    print(f"\n[smoke] training wall time: {dt:.1f}s, steps: {n_steps}")
    if ms_per_step is not None:
        proj_20k = ms_per_step * 20000 / 1000 / 3600
        print(f"[smoke] ms/step: {ms_per_step:.0f}  -> implied wall for 20,000 steps: {proj_20k:.2f} hours")
    print(f"[smoke] step 0 losses (MLM, contrastive): {init_losses}")
    print(f"[smoke] last  losses (MLM, contrastive): {last_losses}")
    assert init_losses is not None and last_losses is not None
    print(f"[smoke] MLM decrease:         {init_losses[0]:.4f} -> {last_losses[0]:.4f}  (Δ={last_losses[0]-init_losses[0]:+.4f})")
    print(f"[smoke] contrastive decrease: {init_losses[1]:.4f} -> {last_losses[1]:.4f}  (Δ={last_losses[1]-init_losses[1]:+.4f})")
    assert last_losses[0] < init_losses[0], "MLM did not decrease"
    assert last_losses[1] < init_losses[1], "Contrastive did not decrease"
    assert all(map(lambda x: x == x, init_losses + last_losses)), "NaN found"
    print("[smoke] decrease+NaN-free assertions: PASS")

    print("\n" + "=" * 70)
    print("Probe 1 on the smoke-trained checkpoint (val.txt, held-out samples)")
    print("=" * 70)
    out_dir = Path(args.out_dir)
    result = run_probe1(
        checkpoint_path=ckpt_path,
        val_path=Path("output/val.txt"),
        output_path=out_dir / "probe1.json",
        n_reads_per_sample=50,
        device=device,
        model=model,
        tokenizer=tk,
        ckpt_meta={"n_embd": config.n_embd, "n_layer": config.n_layer, "vocab_size": config.vocab_size,
                   "seed": args.seed, "hyperparams": {}, "val_bpb": None, "git_sha": None},
    )
    print(f"\n[smoke] Probe 1 AUC: {result.get('auc')}")
    print("[smoke] DONE — wiring + smoke verified; NOT launching 20K MVT.")
