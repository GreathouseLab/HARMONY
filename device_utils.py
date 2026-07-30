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


# ============================================================
# Distributed data-parallel helpers (single-node multi-GPU, extensible to multi-node)
# ============================================================
# We use MANUAL gradient all-reduce ("poor-man's DDP") instead of torch DDP because HARMONY's
# model uses a custom forward_mlm() (not Module.forward, which DDP hooks) + a custom Muon optimizer,
# and with lam=0 the projection head has no gradient (DDP would error on unused params). Manual
# all-reduce sidesteps all three: each rank trains its own data shard, we average grads before step.
# Backend: ccl (xpu/Aurora) · nccl (cuda) · gloo (cpu/mps). Override with HARMONY_DDP_BACKEND.

_DIST = {"active": False, "rank": 0, "world_size": 1, "local_rank": 0}


def _env_int(names, default):
    for n in names:
        v = os.environ.get(n)
        if v is not None:
            try:
                return int(v)
            except ValueError:
                pass
    return default


def init_distributed(device_hint=None):
    """Detect the launcher env (torchrun OR mpiexec/PALS) and init the process group if world_size>1.
    Returns {active, rank, world_size, local_rank, device}. A no-op (active=False) when single-process,
    so single-device runs are completely unchanged."""
    world = _env_int(["WORLD_SIZE", "PMI_SIZE", "PALS_NRANKS", "OMPI_COMM_WORLD_SIZE"], 1)
    rank = _env_int(["RANK", "PMI_RANK", "PALS_RANKID", "OMPI_COMM_WORLD_RANK"], 0)
    lrank = _env_int(["LOCAL_RANK", "PALS_LOCAL_RANKID", "MPI_LOCALRANKID",
                      "OMPI_COMM_WORLD_LOCAL_RANK"], -1)
    dtype = device_type(device_hint or get_device())

    if world <= 1:
        _DIST.update(active=False, rank=0, world_size=1, local_rank=0)
        return {**_DIST, "device": get_device()}

    if lrank < 0:  # infer local rank assuming homogeneous nodes
        mod = _backend(dtype)
        gpn = mod.device_count() if (mod and hasattr(mod, "device_count") and mod.device_count()) else world
        lrank = rank % max(gpn, 1)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")   # multi-node: launcher must export the head node
    os.environ.setdefault("MASTER_PORT", "29500")

    backend = os.environ.get("HARMONY_DDP_BACKEND") or {
        "cuda": "nccl", "xpu": "ccl", "cpu": "gloo", "mps": "gloo"}.get(dtype, "gloo")
    if backend == "ccl":
        try:
            import oneccl_bindings_for_pytorch  # noqa: F401  (registers the 'ccl' backend on Aurora)
        except Exception:
            pass

    device = dtype
    if dtype in ("xpu", "cuda"):
        mod = _backend(dtype)
        try:
            mod.set_device(lrank)
        except Exception:
            pass
        device = f"{dtype}:{lrank}"

    import torch.distributed as dist
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    _DIST.update(active=True, rank=rank, world_size=world, local_rank=lrank)
    return {**_DIST, "device": device}


def is_main() -> bool:
    return _DIST["rank"] == 0


def dist_active() -> bool:
    return _DIST["active"]


def broadcast_parameters(model, src: int = 0) -> None:
    """Make every rank start from identical weights (rank 0's), so manual all-reduce keeps them synced."""
    if not _DIST["active"]:
        return
    import torch.distributed as dist
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p.data, src=src)


def allreduce_gradients(model) -> None:
    """Average gradients across ranks in place (the core of manual data parallelism). No-op if single."""
    if not _DIST["active"]:
        return
    import torch.distributed as dist
    ws = _DIST["world_size"]
    for p in model.parameters():
        if p.grad is not None:                      # unused params (e.g. projection head at lam=0)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)   # are None on ALL ranks -> matched collective
            p.grad.div_(ws)


def barrier() -> None:
    """Keep ranks in lockstep (e.g. while rank 0 runs eval). No-op if single."""
    if not _DIST["active"]:
        return
    import torch.distributed as dist
    try:
        dist.barrier()
    except Exception:
        pass


def cleanup_distributed() -> None:
    if not _DIST["active"]:
        return
    import torch.distributed as dist
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


if __name__ == "__main__":
    print(describe())
