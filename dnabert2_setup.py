"""
Verified DNABERT-2 setup for Apple Silicon (MacBook Pro / Mac Mini, MPS).

DNABERT-2's remote code hard-imports `triton` (its flash-attention kernels), which has no
Apple-Silicon build, so a naive `AutoModel.from_pretrained(..., trust_remote_code=True)` fails
transformers' static import check. The model ALREADY falls back to plain PyTorch attention at
runtime when triton is absent — the only blocker is that static check. This script materializes
the model to a local folder, neuters that one import, and loads from the patched folder.

Verified working: torch 2.6 (MPS), transformers 4.57, einops 0.8, Python 3.12 arm64.
Embedding is ~52 ms/read on CPU (fine for our ~700 val reads); CPU beats MPS for short reads.

USAGE
  # one-time env (isolated):
  #   python3 -m venv ~/dnabert2-env && source ~/dnabert2-env/bin/activate
  #   pip install -U pip && pip install torch "transformers>=4.38,<5" einops
  # then:
  python dnabert2_setup.py            # downloads (~450MB) + patches + smoke-tests
  # and import from your own code:
  #   from dnabert2_setup import load_dnabert2, embed_reads
"""
from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")

REPO = "zhihan1996/DNABERT-2-117M"
LOCAL_DIR = os.environ.get("DNABERT2_DIR", os.path.expanduser("~/dnabert2_model"))
TRITON_IMPORT = "from .flash_attn_triton import flash_attn_qkvpacked_func"
PATCH = "flash_attn_qkvpacked_func = None  # PATCHED (Apple Silicon): skip triton, use pytorch attention"


def ensure_model(local_dir: str = LOCAL_DIR) -> str:
    """Download DNABERT-2 to a plain local folder and patch out the triton import (idempotent)."""
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=local_dir)
    bl = os.path.join(local_dir, "bert_layers.py")
    src = open(bl).read()
    if TRITON_IMPORT in src:
        open(bl, "w").write(src.replace(TRITON_IMPORT, PATCH))
        print(f"[patch] neutered triton import in {bl}")
    else:
        print("[patch] already patched (or import not present) — ok")
    return local_dir


def load_dnabert2(device: str = "cpu", local_dir: str = LOCAL_DIR):
    """Return (tokenizer, model) ready for embedding. device='cpu' recommended for short reads."""
    ensure_model(local_dir)
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(local_dir, trust_remote_code=True).eval().to(device)
    return tok, model


def embed_reads(seqs, tok, model, device: str = "cpu"):
    """Embed a list of raw DNA strings (ACGT..., no special markers) -> (N, 768) mean-pooled.
    HARMONY reads: strip <READ_START>/<PAIRED_END>/<READ_END> and pass the raw bases."""
    import torch
    out = []
    with torch.no_grad():
        for s in seqs:
            ids = tok(s, return_tensors="pt").to(device)
            h = model(**ids)[0]           # (1, tokens, 768)
            out.append(h.mean(dim=1).squeeze(0).cpu())
    return torch.stack(out)               # (N, 768)


if __name__ == "__main__":
    tok, model = load_dnabert2(device="cpu")
    n = sum(p.numel() for p in model.parameters()) / 1e6
    emb = embed_reads(["ACGTACGTACGTTTGCATGCATGCATACGT", "TTTGCACACACGTGTGTACACACGTAC"], tok, model)
    print(f"[ok] DNABERT-2 loaded ({n:.1f}M params); embedded 2 reads -> {tuple(emb.shape)}")
    print("[ok] setup verified — ready to embed HARMONY reads.")
