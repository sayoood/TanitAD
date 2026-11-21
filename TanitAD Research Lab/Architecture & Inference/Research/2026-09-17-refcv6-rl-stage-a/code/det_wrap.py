"""Run a trainer with PyTorch determinism forced, WITHOUT editing the trainer.

⛔ `ddv2_rl_refcv5.py` is load-bearing and was RUNNING when this was written, and this
repo's standing rule is *never edit a running script*. So the flags are set here, in a
parent process, before the target module is executed — the trainer is read, never
written.

Two modes, and the second is the one that actually names the cause:

* ``--mode strict``  — ``use_deterministic_algorithms(True, warn_only=True)``: run to
  completion and let the caller compare two launches for bit-identity.
* ``--mode name-it`` — ``warn_only=False``: the first non-deterministic kernel
  **RAISES**, and the exception names it. ⭐ This is the cheap discriminator: it turns
  "something amplifies" into "this op is the source" in one short run.

⚠️ ``CUBLAS_WORKSPACE_CONFIG`` must be in the environment BEFORE the CUDA context
exists, so the caller sets it; this script asserts it rather than assuming.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("strict", "name-it", "off"), required=True)
    ap.add_argument("--script", required=True)
    a, rest = ap.parse_known_args()

    import torch
    if a.mode != "off":
        cw = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if cw not in (":4096:8", ":16:8"):
            print(f"ZZREFUSED CUBLAS_WORKSPACE_CONFIG={cw!r} — must be :4096:8 or :16:8 "
                  f"in the ENV before CUDA initialises, or cuBLAS matmul raisesZZ")
            return 3
        torch.use_deterministic_algorithms(True, warn_only=(a.mode == "strict"))
        torch.backends.cudnn.benchmark = False
    print(f"ZZWRAP mode={a.mode} "
          f"deterministic={torch.are_deterministic_algorithms_enabled()} "
          f"warn_only={torch.is_deterministic_algorithms_warn_only_enabled()} "
          f"cudnn_benchmark={torch.backends.cudnn.benchmark} "
          f"tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"tf32_cudnn={torch.backends.cudnn.allow_tf32}ZZ", flush=True)
    sys.argv = [a.script] + rest
    runpy.run_path(a.script, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
