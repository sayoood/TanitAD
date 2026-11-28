"""l2_roll.py -- refcv9 L2 (PREREG_L2.md + Amendment 1): ONE inference-seed os-only roll of refcv8 EMA-48,852 (per-frame nav feed) with an ARM applied at the
lon-head seam `RefCV3Model._r8_apply_force` (stack/tanitad/refs/refc_v3.py:2001/2094) -- the seam the force_lon oracle-class arm used.

NEW file (Research Lab / L2 agent, 2026-10-25).  A WRAPPER on the battery children, exactly the L1 pattern (`refcv9_l1/code/l1_ztac_roll.py`, read-only):
`roll_seed_r8_pf.py` (hold_r8pf, the per-frame eval battery child) is IMPORTED and its main() runs with the battery child's own argv (`--os-only`); the dev-box
remap is re-pointed exactly as P4 / L1 did.  `r7_roll.Refcv6Proxy` is SUBCLASSED and re-bound before the roll builds it.  The subclass runs the battery's own
__call__ UNCHANGED and only (a) tells the seam which window is current and (b) banks the window's raw/applied lon logits, 8-slot plan and sel_idx.

ARMS (the hook replaces out["lon_logits"] [B, 8] BEFORE every planner feed; all other model state is untouched):
  native        no hook installed (nothing is wrapped); plan + sel banked only
  ident         hook installed, lc = (l + 0) / 1                       (hook neutrality, K3)
  cal           lc = (l + b8) / T from --calibrator (l2_fit.py json)   (THE LEVER)
  const_cruise  one-hot +-20 on the CRUISE column (v7 id 1) every window (the force_lon F2 semantics, class-only)  (constant-class control)
  oracle        one-hot +-20 on the GT label's column where `lon_v7` is a LON6 hypothesis, else native (force_lon F1 semantics)  (ceiling; privileged)

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 l2_roll.py --arm cal --calibrator <json> --trace-out <dir>/trace_cal_s0.npz --require-seal -- \
         --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
         --dump-dir <dir>/dump_cal_s0 --windows-json <windows.json> --device cuda --out-json <dir>/roll_cal_s0.json --os-only
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
PREREG, SEAL = ROOT / "PREREG_L2.md", ROOT / "PREREG_L2.sha256"
AMEND, AMEND_SEAL = ROOT / "PREREG_L2_AMENDMENT1.md", ROOT / "PREREG_L2_AMENDMENT1.sha256"
PF_CODE = Path(os.environ.get("L2_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L2_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
EVAL_REMAP = HERE / "refcv8_eval_remap_devbox_l2.json"
LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
CRUISE_V7 = 1
BIG = 20.0

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[l2] usage: l2_roll.py [l2 opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("native", "ident", "cal", "const_cruise", "oracle"), required=True)
    ap.add_argument("--trace-out", required=True)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--labels-npz", default=None, help="oracle arm: npz with clip_index, ws, lon_v7 (banked battery extras)")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--require-seal", action="store_true")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if "--os-only" not in child:
        raise SystemExit("[l2] the os-only battery child is required; --os-only missing")
    if a.arm == "cal" and not a.calibrator:
        raise SystemExit("[l2] --arm cal needs --calibrator")
    if a.arm == "oracle" and not a.labels_npz:
        raise SystemExit("[l2] --arm oracle needs --labels-npz")
    return a, child


A, CHILD = parse()
SEALS = {}
for nm, f, s in (("prereg", PREREG, SEAL), ("amendment1", AMEND, AMEND_SEAL)):
    want = s.read_text(encoding="utf-8").split()[0].strip() if s.exists() else ""
    got = sha256_file(f) if f.exists() else ""
    SEALS[nm] = got if (len(want) == 64 and got == want) else None
if A.require_seal and not all(SEALS.values()):
    raise SystemExit(f"[l2] seal mismatch {SEALS}: refusing a scored roll")

CAL = None
if A.arm == "cal":
    CAL = json.load(open(A.calibrator, encoding="utf-8"))
    if len(CAL["b8"]) != 8 or not (CAL["T"] > 0):
        raise SystemExit("[l2] calibrator json malformed")
LABELS = None
if A.arm == "oracle":
    _z = np.load(A.labels_npz)
    LABELS = {(int(c), int(w)): int(l) for c, w, l in zip(_z["clip_index"], _z["ws"], _z["lon_v7"])}

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402
REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE), "file": str(EVAL_REMAP), "sha256": sha256_file(EVAL_REMAP)}
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay; READ-ONLY, UNCHANGED)
RR = PF.RR
RUNNER = PF.main

import torch  # noqa: E402

_Base = RR.Refcv6Proxy
ST = {"installed": False, "armed": False, "cur": None, "W": None, "hook_module": None, "t0": time.time(), "records": [],
      "tmp": None, "c": {"calls": 0, "hook_calls_armed": 0, "hook_calls_unarmed": 0, "hook_applied": 0, "hook_native_fallthrough": 0, "multi_hook_in_call": 0,
                         "batch_gt1": 0, "navnull_calls": 0, "oracle_labelled": 0, "oracle_unlabelled_or_masked": 0}}


def _arm_logits(lg, key):
    """lg: [B, 8] tensor -> the tensor the planner feeds see, or None for 'leave native'."""
    if A.arm == "ident":
        return (lg + torch.zeros(8, device=lg.device, dtype=lg.dtype)) / 1.0
    if A.arm == "cal":
        b8 = torch.as_tensor(CAL["b8"], device=lg.device, dtype=lg.dtype)
        return (lg + b8) / float(CAL["T"])
    if A.arm == "const_cruise":
        t = torch.full_like(lg, -BIG)
        t[:, CRUISE_V7] = BIG
        return t
    if A.arm == "oracle":
        lab = LABELS.get(key, -100)
        if lab in LON6_V7_IDS:
            t = torch.full_like(lg, -BIG)
            t[:, lab] = BIG
            ST["c"]["oracle_labelled"] += 1
            return t
        ST["c"]["oracle_unlabelled_or_masked"] += 1
        return None
    return None


def _install(model):
    mods = [(n, m) for n, m in model.named_modules() if callable(getattr(m, "_r8_apply_force", None))]
    if len(mods) != 1:
        raise SystemExit(f"[l2] expected exactly one module with _r8_apply_force, found {[n for n, _ in mods]}")
    name, mod = mods[0]
    core = getattr(getattr(model, "cfg", None), "core", None)                  # the L1 route (top-level model), then the seam module's own cfg
    if core is None or not hasattr(core, "window"):
        core = getattr(getattr(mod, "cfg", None), "core", None)
    if core is None or not hasattr(core, "window"):
        raise SystemExit("[l2] cannot read the window length W from model.cfg.core.window")
    ST["W"] = int(core.window)
    ST["hook_module"] = name
    ST["installed"] = True
    if A.arm == "native":
        return
    orig = mod._r8_apply_force

    def wrapped(out, b, dev):
        if not ST["armed"] or ST["cur"] is None:
            ST["c"]["hook_calls_unarmed"] += 1
            return orig(out, b, dev)
        ST["c"]["hook_calls_armed"] += 1
        first = ST["tmp"] is None
        if not first:
            ST["c"]["multi_hook_in_call"] += 1
        lg = out["lon_logits"]
        new = _arm_logits(lg, ST["cur"][2])
        if first:                                   # the trace is the FIRST seam call of the window (os-only: the only one)
            ST["tmp"] = {"raw": lg[0].detach().float().cpu().numpy().copy(),
                         "app": (new if new is not None else lg)[0].detach().float().cpu().numpy().copy()}
        if new is None:
            ST["c"]["hook_native_fallthrough"] += 1
            return orig(out, b, dev)
        out = dict(out)
        out["lon_logits"] = new
        ST["c"]["hook_applied"] += 1
        return orig(out, b, dev)
    mod._r8_apply_force = wrapped          # instance attribute: self._r8_apply_force(out, b, dev) resolves here, no binding


class L2Proxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        if not ST["installed"]:
            _install(d["model"])
        if nav_cmd is None:
            ST["c"]["navnull_calls"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n = int(frames.shape[0])
        e_i, t = (int(x) for x in d["ctx"].index)
        ws = t + int(ST["W"]) - 1
        ST["cur"], ST["tmp"], ST["armed"] = (e_i, t, (e_i, ws)), None, True
        try:
            out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        finally:
            ST["armed"] = False
        tmp, ST["tmp"] = ST["tmp"], None
        ST["c"]["calls"] += 1
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        if "nav_true" not in conds:
            raise RuntimeError(f"[l2] no nav_true row among conds {conds}")
        r = conds.index("nav_true")
        if A.arm != "native" and tmp is None:
            raise RuntimeError("[l2] the nav_true call never reached the lon-head seam")
        nan8 = np.full(8, np.nan, np.float32)
        ST["records"].append({"e_i": e_i, "t": t, "ws": ws, "raw": (tmp["raw"] if tmp else nan8), "app": (tmp["app"] if tmp else nan8),
                              "plan": out["traj"][r].detach().float().cpu().numpy(), "sel": int(out["sel_idx"][r]) if "sel_idx" in out else -1,
                              "v0_in": (float(v0.reshape(-1)[r]) if v0 is not None else float("nan"))})
        nw = len(ST["records"])
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 250 == 0 or nw <= 2:
            print(f"ZZL2PROGZZ {A.arm} {nw} el={time.time() - ST['t0']:.0f}s", flush=True)
        return out


def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    out = {"e_i": np.array([r["e_i"] for r in recs], np.int64), "t": np.array([r["t"] for r in recs], np.int64),
           "ws": np.array([r["ws"] for r in recs], np.int64),
           "raw_lon_logits": np.stack([r["raw"] for r in recs]).astype(np.float32), "applied_lon_logits": np.stack([r["app"] for r in recs]).astype(np.float32),
           "plan8": np.stack([r["plan"] for r in recs]).astype(np.float32), "sel": np.array([r["sel"] for r in recs], np.int64),
           "v0_in": np.array([r["v0_in"] for r in recs], np.float64)}
    meta = {"tool": "l2_roll.py", "arm": A.arm, "prereg_sha256": SEALS.get("prereg"), "amendment1_sha256": SEALS.get("amendment1"), "script_sha256": sha256_file(__file__),
            "calibrator": (None if CAL is None else {"file": A.calibrator, "sha256": sha256_file(A.calibrator), "b8": CAL["b8"], "T": CAL["T"], "tag": CAL.get("tag")}),
            "labels_npz": A.labels_npz, "child_argv": CHILD, "remap": REMAP_REC, "hook_module": ST["hook_module"], "window_W": ST["W"], "final": bool(final),
            "n_windows": len(recs), "counters": ST["c"], "wall_s": round(time.time() - ST["t0"], 1), "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.trace_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    RR.Refcv6Proxy = L2Proxy                                    # install_patches' load_model resolves the name at call time
    print(f"[l2] arm={A.arm} prereg {'sealed ' + SEALS['prereg'][:16] if SEALS['prereg'] else 'UNSEALED'} amendment1 "
          f"{'sealed ' + SEALS['amendment1'][:16] if SEALS['amendment1'] else 'UNSEALED'} child={' '.join(CHILD)} ZZL2ROLLSTARTZZ", flush=True)
    t0 = time.time()
    rc = RUNNER()
    if not ST["records"]:
        print("ZZL2NOWINDOWSZZ the L2Proxy never ran", flush=True)
        return 4
    save(True, {"runner_rc": rc, "roll_wall_s": round(time.time() - t0, 1),
                "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None)})
    print(f"[l2] banked {len(ST['records'])} windows arm={A.arm} counters={json.dumps(ST['c'])} ZZL2BANKEDZZ", flush=True)
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
