"""Structural check on every rescued checkpoint: can torch actually LOAD it?

This is NOT redundant with the md5 pass, and the distinction matters:
  * md5 vs source proves the COPY is faithful. It says nothing about whether the
    ORIGINAL was usable — a checkpoint truncated by a dying pod copies perfectly.
  * torch.load proves the file is a well-formed checkpoint we can resume/eval from.

Neither alone is sufficient; together they cover both failure modes.
Read-only. Loads to CPU, reports the training step where present, frees immediately.
"""
from __future__ import annotations

import gc
import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path("/workspace/rescue")


def describe(obj) -> str:
    if not isinstance(obj, dict):
        return f"<{type(obj).__name__}>"
    step = obj.get("step", obj.get("global_step"))
    keys = [k for k in ("model", "state_dict", "opt", "optimizer", "cfg", "config") if k in obj]
    n_params = None
    sd = obj.get("model") or obj.get("state_dict")
    if isinstance(sd, dict):
        try:
            n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
        except Exception:
            n_params = None
    bits = []
    if step is not None:
        bits.append(f"step={step}")
    if n_params:
        bits.append(f"params={n_params:,}")
    if keys:
        bits.append("keys=" + ",".join(keys))
    return "  ".join(bits) or f"dict({len(obj)} entries)"


def main() -> int:
    files = sorted(ROOT.rglob("*.pt"))
    print(f"=== LOADCHECK: {len(files)} checkpoints under {ROOT} ===", flush=True)
    ok = bad = 0
    results = []
    for f in files:
        rel = str(f.relative_to(ROOT))
        size = f.stat().st_size
        t0 = time.time()
        try:
            obj = torch.load(f, map_location="cpu", weights_only=False)
            info = describe(obj)
            del obj
            gc.collect()
            dt = time.time() - t0
            print(f"  OK    {rel}  ({size/2**20:.0f} MiB, {dt:.1f}s)  {info}", flush=True)
            results.append({"file": rel, "ok": True, "bytes": size, "info": info})
            ok += 1
        except Exception as e:  # noqa: BLE001 - we want every failure mode reported, not raised
            dt = time.time() - t0
            print(f"  FAIL  {rel}  ({size/2**20:.0f} MiB, {dt:.1f}s)  {type(e).__name__}: {e}", flush=True)
            results.append({"file": rel, "ok": False, "bytes": size, "error": f"{type(e).__name__}: {e}"})
            bad += 1
    print(f"=== LOADCHECK DONE — LOADABLE={ok}  FAILED={bad} ===", flush=True)
    (ROOT / "loadcheck.json").write_text(json.dumps(results, indent=1))
    if bad:
        print("🔴 at least one checkpoint is NOT loadable — do NOT release the old pod.", flush=True)
    else:
        print("✅ every rescued checkpoint loads. Combined with the md5 pass this covers "
              "both 'copy is faithful' and 'original is usable'.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
