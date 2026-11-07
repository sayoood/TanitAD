"""One stage-5 rollout cell (AGENT R): stage 4's composite recipe with the checkpoints swapped, then the stage-2 cell.

    python -I t2v8f_stage5_cell.py [--executor base|hold] [--hold-eps 0.1] <the t2v8f_run.py rollout arguments>

The cell's argv is EXACTLY the stage-4 cell's (`t2v8f_stage2.cell_argv`); what makes it stage 5 is the arm table, read from the MANIFEST5.json that `T2V8F_S5_MANIFEST` names (the driver sets it for
every child): `arm5.name` and `arm5.ckpt_file` {NAME_s0: file, NAME_s1: file} (file names inside `<pkg>/ckpt`, which `--t2-ckpt-root` points at). `install5(R, ...)` extends `t2v8f_run`'s ARMS /
CKPT_FILE and wraps `R.build_arm` BEFORE `R.setup()` registers the arms (the stage-2 cell calls `R.setup()` inside `apply_executor`). The wrapped arm is built exactly as every model arm is, and then wrapped
in `t2v8f_c0c.CompositePolicy`, the SAME class, smoothing (sg72) and projection (kbox) that stage 4 ran on C0's checkpoint, with the RI tool loaded from `T2V8F_RI_TOOL` by sha256 (0417c838...).
Only the weights differ between stage 4's C0C and a stage-5 arm: `tests/test_t2v8f_stage5.py::check_identity` runs C0's own checkpoint through both and requires the plans bit for bit.
"""
from __future__ import annotations

import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ENV_MANIFEST = "T2V8F_S5_MANIFEST"
SEEDS = (0, 1)


def arm5_from_manifest(path):
    """-> (name, {arm: ckpt file name}) from a MANIFEST5.json."""
    a = json.load(open(path, encoding="utf-8")).get("arm5") or {}
    if not a.get("name") or not a.get("ckpt_file"):
        raise SystemExit(f"[t2v8f_stage5_cell] {path} carries no arm5.name / arm5.ckpt_file: this is not a MANIFEST5.json")
    kind = (a.get("encoder") or {}).get("kind")
    if kind != "frozen_dinov3":
        raise SystemExit(f"[t2v8f_stage5_cell] ENCODER_UNSUPPORTED {kind!r}: the stage-5 cell builds the frozen DINOv3-L encoder only (a merged-LoRA arm needs the LC build, HANDOFF_S5.md)")
    return a["name"], dict(a["ckpt_file"])


def install5(R, name, ckpt_file):
    """Register `<name>_s0` / `<name>_s1` in `t2v8f_run` (module `R`) before `R.setup()`. Idempotent for the same arm; refuses a second, different stage-5 arm in one process."""
    import t2v8f_c0c as C0C
    arms = tuple(f"{name}_s{s}" for s in SEEDS)
    if set(ckpt_file) != set(arms):
        raise SystemExit(f"[t2v8f_stage5_cell] ckpt_file keys {sorted(ckpt_file)} are not the arm's cells {list(arms)}")
    have = getattr(R, "_S5_ARMS", None)
    if have is not None:
        if have != arms:
            raise SystemExit(f"[t2v8f_stage5_cell] a stage-5 arm {list(have)} is already installed in this process; refusing {list(arms)}")
        return R
    clash = [a for a in arms if a in R.ARMS]
    if clash:
        raise SystemExit(f"[t2v8f_stage5_cell] arm name clash with an existing arm: {clash}")
    R.ARMS = tuple(R.ARMS) + arms
    R.CKPT_FILE.update(ckpt_file)
    orig = R.build_arm

    def build_arm(args):
        pol = orig(args)
        if args.arm in arms:
            pol.inner = C0C.CompositePolicy(pol.inner)
        return pol
    R.build_arm = build_arm
    R._S5_ARMS = arms
    return R


def run_cell(argv):
    path = os.environ.get(ENV_MANIFEST)
    if not path or not os.path.isfile(path):
        raise SystemExit(f"[t2v8f_stage5_cell] {ENV_MANIFEST} is not set to an existing MANIFEST5.json ({path!r}); the driver sets it for every cell")
    import t2v8f_run as R
    import t2v8f_stage2_cell as C
    install5(R, *arm5_from_manifest(path))
    return C.run_cell(argv)


if __name__ == "__main__":
    run_cell(sys.argv[1:])
