"""l2b_locus_roll.py -- refcv9 L2-B (PREREG_L2B.md): the brake-LOCUS roll of the shipped refcv8 EMA-48,852 at PER-FRAME nav.  ONE inference-seed os-only NATIVE roll
(no hook, no force, no calibrator) that BANKS, for EVERY window, the whole refined candidate fan and the per-candidate selection scores:

    fan [N=149, 8, 2] float32   (out["anchor_traj"], the 117 base anchors refined + 32 allocated)       sel_score / sel_score_v3 [N]
    reach_keep / ceil_keep / emit_keep [N] (int8; -1 = the key was ABSENT in the model output)           kind / lon6 / lat3 [N] int8
    cons_raw [N, 4]   hyp_alloc [32]   prior_path [8, 2]   bank [117, 8, 2] (fp16, unrefined)           plan8 (= out["traj"]), sel_idx, sel_idx_base, v0

Adapted from the L2 wrapper `l2_roll.py` (same battery child, same remap re-point, same Refcv6Proxy subclass pattern) and from the CPU re-roll's field list
(`brake_locus_reroll_r8.py` FIELD_KEYS, shapes MEASURED in its SMOKE_REPORT.json).  The difference that matters: this is the PER-FRAME-nav battery child
(`roll_seed_r8_pf.py`), the GPU path whose os-only plans are bit-equal to L1's BASE-os bank (L2 K3: 308/308 windows), so every FULL-roll window is reproducible.

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 l2b_locus_roll.py --trace-out F:/refcv9_l2_locus/locus_s0.npz --require-seal -- \
         --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
         --dump-dir <dir> --windows-json <windows_s0.json> --device cuda --out-json <dir>/roll.json --os-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREREG, SEAL = ROOT / "PREREG_L2B.md", ROOT / "PREREG_L2B.sha256"
PF_CODE = Path(os.environ.get("L2_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L2_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
EVAL_REMAP = HERE / "refcv8_eval_remap_devbox_l2.json"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[l2b] usage: l2b_locus_roll.py [l2b opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-out", required=True)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--require-seal", action="store_true")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if "--os-only" not in child:
        raise SystemExit("[l2b] the os-only battery child is required; --os-only missing")
    return a, child


A, CHILD = parse()
SEAL_SHA = None
if SEAL.exists() and PREREG.exists():
    want, got = SEAL.read_text(encoding="utf-8").split()[0].strip(), sha256_file(PREREG)
    SEAL_SHA = got if (len(want) == 64 and got == want) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[l2b] PREREG_L2B.md is missing / does not match its seal: refusing a scored roll")

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402
REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE), "file": str(EVAL_REMAP), "sha256": sha256_file(EVAL_REMAP)}
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (READ-ONLY, UNCHANGED)
RR = PF.RR
RUNNER = PF.main

import torch  # noqa: E402

_Base = RR.Refcv6Proxy
ST = {"W": None, "t0": time.time(), "records": [], "present": {}, "c": {"calls": 0, "batch_gt1": 0, "navnull_calls": 0, "fan_n_seen": set()}}
# (name, model-output key, numpy dtype, kind) ; kind 'row' = batched [B, ...] -> [r] ; 'mask' = int8 with -1 when absent
FIELDS = (("fan", "anchor_traj", np.float32), ("sel_score", "sel_score", np.float32), ("sel_score_v3", "sel_score_v3", np.float32),
          ("reach_keep", "reach_keep", np.int8), ("ceil_keep", "ceil_keep", np.int8), ("emit_keep", "r8_emit_keep", np.int8),
          ("kind", "r8_kind", np.int8), ("lon6", "r8_lon6", np.int8), ("lat3", "r8_lat3", np.int8), ("cons_raw", "r8_cons_raw", np.float32),
          ("hyp_alloc", "r8_hyp_alloc", np.int8), ("prior_path", "residual_prior_path", np.float32), ("bank", "anchor_bank", np.float16))


def _window_len(model):
    core = getattr(getattr(model, "cfg", None), "core", None)
    if core is None or not hasattr(core, "window"):
        for _n, m in model.named_modules():
            core = getattr(getattr(m, "cfg", None), "core", None)
            if core is not None and hasattr(core, "window"):
                break
    if core is None or not hasattr(core, "window"):
        raise SystemExit("[l2b] cannot read the window length W from the model config")
    return int(core.window)


def _row(o, key, r, B):
    """The window's row of output `key`: batched tensors (leading dim B) are indexed at r, batch-less ones (e.g. r8_kind [N]) are taken whole."""
    t = o[key]
    if not torch.is_tensor(t):
        return None
    t = t.detach()
    if t.dim() >= 1 and t.shape[0] == B:
        t = t[r]
    return t.float().cpu().numpy() if t.is_floating_point() else t.cpu().numpy()


class L2BProxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        if ST["W"] is None:
            ST["W"] = _window_len(d["model"])
        if nav_cmd is None:
            ST["c"]["navnull_calls"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n = int(frames.shape[0])
        e_i, t = (int(x) for x in d["ctx"].index)
        ws = t + int(ST["W"]) - 1
        out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        ST["c"]["calls"] += 1
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        if "nav_true" not in conds:
            raise RuntimeError(f"[l2b] no nav_true row among conds {conds}")
        r = conds.index("nav_true")
        rec = {"e_i": e_i, "t": t, "ws": ws, "plan": out["traj"][r].detach().float().cpu().numpy(), "sel": int(out["sel_idx"][r]),
               "sel_base": (int(out["sel_idx_base"][r]) if "sel_idx_base" in out else -1),
               "v0_in": (float(v0.reshape(-1)[r]) if v0 is not None else float("nan"))}
        for name, key, dt in FIELDS:
            if key in out and torch.is_tensor(out[key]):
                rec[name] = _row(out, key, r, n).astype(dt)
                ST["present"][name] = ST["present"].get(name, 0) + 1
            else:
                rec[name] = None
        if rec["fan"] is not None:
            ST["c"]["fan_n_seen"].add(int(rec["fan"].shape[0]))
        ST["records"].append(rec)
        nw = len(ST["records"])
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 250 == 0 or nw <= 2:
            print(f"ZZL2BPROGZZ {nw} el={time.time() - ST['t0']:.0f}s", flush=True)
        return out


def _stack(recs, name, fill, dtype):
    got = [r[name] for r in recs]
    ref = next((g for g in got if g is not None), None)
    if ref is None:
        return None
    return np.stack([g if g is not None else np.full(ref.shape, fill, dtype) for g in got]).astype(dtype)


def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    out = {"e_i": np.array([r["e_i"] for r in recs], np.int64), "t": np.array([r["t"] for r in recs], np.int64), "ws": np.array([r["ws"] for r in recs], np.int64),
           "plan8": np.stack([r["plan"] for r in recs]).astype(np.float32), "sel": np.array([r["sel"] for r in recs], np.int64),
           "sel_base": np.array([r["sel_base"] for r in recs], np.int64), "v0_in": np.array([r["v0_in"] for r in recs], np.float64)}
    for name, _key, dt in FIELDS:
        fill = -1 if name in ("reach_keep", "ceil_keep", "emit_keep") else 0
        arr = _stack(recs, name, fill, dt)
        if arr is not None:
            out[name] = arr
    meta = {"tool": "l2b_locus_roll.py", "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__), "child_argv": CHILD, "remap": REMAP_REC,
            "window_W": ST["W"], "final": bool(final), "n_windows": len(recs), "present_counts": ST["present"],
            "counters": {k: (sorted(v) if isinstance(v, set) else v) for k, v in ST["c"].items()}, "wall_s": round(time.time() - ST["t0"], 1),
            "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.trace_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    RR.Refcv6Proxy = L2BProxy                                   # install_patches' load_model resolves the name at call time
    print(f"[l2b] prereg {'sealed ' + SEAL_SHA[:16] if SEAL_SHA else 'UNSEALED'} child={' '.join(CHILD)} ZZL2BROLLSTARTZZ", flush=True)
    t0 = time.time()
    rc = RUNNER()
    if not ST["records"]:
        print("ZZL2BNOWINDOWSZZ the L2BProxy never ran", flush=True)
        return 4
    save(True, {"runner_rc": rc, "roll_wall_s": round(time.time() - t0, 1),
                "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None)})
    print(f"[l2b] banked {len(ST['records'])} windows present={json.dumps(ST['present'])} ZZL2BBANKEDZZ", flush=True)
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
