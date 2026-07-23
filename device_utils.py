"""
Device abstraction for HARMONY — Intel XPU (Aurora), CUDA, Apple MPS, or CPU.

Aurora (ANL) uses Intel Data Center GPU Max ("Ponte Vecchio") via oneAPI, exposed in PyTorch as the
`xpu` device. torch>=2.5 has native XPU support; older stacks need `import intel_extension_for_pytorch`
to register the backend, so we try that import opportunistically and ignore it if absent.

Selection order (first available wins):  xpu  ->  cuda  ->  mps  ->  cpu
Override anywhere with:  HARMONY_DEVICE=xpu|cuda|mps|cpu

All helpers are written generically (via getattr(torch, "<type>")) so the same call works on every
backend — no per-device branching at call sites.

Usage:
    from device_utils import get_device, empty_cache, describe
    device = get_device()
    print(describe())
"""
from __future__ import annotations

import os

import torch

_IPEX_TRIED = False


def _try_ipex() -> None:
    """Import Intel Extension for PyTorch once, if present (registers XPU on older stacks)."""
    global _IPEX_TRIED
    if _IPEX_TRIED:
        return
    _IPEX_TRIED = True
    try:
        import intel_extension_for_pytorch  # noqa: F401
    except Exception:
        pass  # torch>=2.5 has native XPU; on Mac/CPU this simply isn't installed.


def xpu_available() -> bool:
    _try_ipex()
    try:
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def get_device(prefer: str | None = None) -> str:
    """Return the device string to use. `prefer` or $HARMONY_DEVICE overrides autodetection."""
    req = prefer or os.environ.get("HARMONY_DEVICE")
    if req:
        return req
    if xpu_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def device_type(dev) -> str:
    """'xpu:0' -> 'xpu'. Accepts str or torch.device."""
    return str(dev).split(":")[0]


def _backend(dev):
    """torch.xpu / torch.cuda / torch.mps module for this device, or None for cpu."""
    t = device_type(dev)
    return None if t == "cpu" else getattr(torch, t, None)


def empty_cache(dev) -> None:
    """Device-agnostic cache release (used in the OOM-retry path). No-op where unsupported."""
    mod = _backend(dev)
    if mod is not None and hasattr(mod, "empty_cache"):
        try:
            mod.empty_cache()
        except Exception:
            pass


def synchronize(dev) -> None:
    """Device-agnostic synchronize (for honest timing). No-op where unsupported."""
    mod = _backend(dev)
    if mod is not None and hasattr(mod, "synchronize"):
        try:
            mod.synchronize()
        except Exception:
            pass


def manual_seed_all(seed: int, dev=None) -> None:
    """Seed CPU RNG plus the accelerator's RNG, whichever backend is active."""
    torch.manual_seed(seed)
    mod = _backend(dev if dev is not None else get_device())
    for fn in ("manual_seed_all", "manual_seed"):
        if mod is not None and hasattr(mod, fn):
            try:
                getattr(mod, fn)(seed)
                break
            except Exception:
                pass


def describe(dev: str | None = None) -> str:
    """One-line device banner for run logs — record this in every experiment."""
    d = dev or get_device()
    return (f"device={d} | torch={torch.__version__} | "
            f"xpu={xpu_available()} cuda={torch.cuda.is_available()} "
            f"mps={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}")


if __name__ == "__main__":
    print(describe())
