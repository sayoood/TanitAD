"""fresh2_ztac_roll.py -- COPY of refcv9_l1c/code/l1c_ztac_roll.py; ONLY the seal paths differ (PREREG_FRESH2 checked; L1C Amendment 1 by absolute path).
Original docstring:
l1c_ztac_roll.py -- refcv9 L1C (PREREG_L1C.md + AMENDMENT 1): the L1 eval-mode roll with ONLY the input paths swapped.

DERIVED from refcv9_l1/code/l1_ztac_roll.py (sha256 in CODE_SHA256_AT_ROLL.txt); the diff is `l1c_ztac_roll.DIFF.txt`.  What differs, and nothing else:
  * the seal checked is PREREG_L1C.md (and AMENDMENT 1) instead of PREREG_L1.md;
  * `--kit` sets R8_EVAL_KIT (the eval kit root: eval cache / eval labels / speed-max sidecar / anchors / class weights are READ FROM IT by the battery's own
    constants `run_battery_r7.KIT_EVAL` / `KIT_LABELS` and the loader's PATH_REMAP_SPEC -- a kit directory of IDENTICAL layout holding the FRESH clips);
  * `--remap` is the eval remap JSON with exactly three values re-pointed (extrinsics union, clock sidecar, v9 release);
  * `--v9-md5` is written to `args.r8_v9_eval_md5` (the loader checks the v9 npz against it; argv carries eval139's md5) -- the same one-line patch the train shim makes;
  * train mode is removed.
Everything else (the battery children, roll_seed_r8_pf.py's per-frame nav feed, the lat_head_tac pre-hook, --os-only, inference seed) is the L1 code unchanged.
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
PREREG, SEAL = ROOT / "PREREG_FRESH2.md", ROOT / "PREREG_FRESH2.sha256"          # FRESH2 COPY: the seal checked is PREREG_FRESH2 (sha256 956f87cd...)
_L1C = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c")                                  # FRESH2 COPY: L1C Amendment 1 (the build recipe) still asserted, by absolute path
AMEND, AMEND_SEAL = _L1C / "PREREG_L1C_AMENDMENT1.md", _L1C / "PREREG_L1C_AMENDMENT1.sha256"
PF_CODE = Path(os.environ.get("L1_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L1_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[l1] usage: l1_ztac_roll.py [l1 opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("eval",), required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--remap", required=True)
    ap.add_argument("--v9-md5", required=True)
    ap.add_argument("--ztac-out", required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--require-seal", action="store_true")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if a.mode == "eval" and "--os-only" not in child:
        raise SystemExit("[l1] eval mode rolls the battery's --os-only child; --os-only missing")
    return a, child


A, CHILD = parse()
os.environ["R8_EVAL_KIT"] = str(A.kit)                 # BEFORE r8_repoint is imported (it reads EVAL_KIT at import)
EVAL_REMAP = Path(A.remap)
SEAL_SHA = None
if SEAL.exists() and PREREG.exists():
    want = SEAL.read_text(encoding="utf-8").split()[0].strip()
    got = sha256_file(PREREG)
    SEAL_SHA = got if (len(want) == 64 and got == want) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[fresh2] PREREG_FRESH2.md does not match PREREG_FRESH2.sha256: refusing a scored roll")
if A.require_seal and sha256_file(AMEND) != AMEND_SEAL.read_text(encoding="utf-8").split()[0].strip():
    raise SystemExit("[l1c] PREREG_L1C_AMENDMENT1.md does not match its sha256: refusing a scored roll")

REMAP_REC: dict = {}
if A.mode == "eval":
    sys.path.insert(0, str(R8_CODE))
    import r8_repoint  # noqa: E402
    REMAP_REC = {"kit": str(A.kit), "v9_md5_set_to": str(A.v9_md5), "devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE),
                 "file": str(EVAL_REMAP), "sha256": sha256_file(EVAL_REMAP)}
    r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
    sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
    sys.path.insert(0, str(PF_CODE))
    import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay; READ-ONLY, UNCHANGED)
    RR = PF.RR
    RUNNER = PF.main
    import tanitad.eval.refcv7_loader as _LL  # noqa: E402
    _orig_bed = _LL.build_eval_dataset

    def _bed(model, cfg, args, config, **kw):             # the loader checks the v9 npz against the md5 in the ARGV (eval139's); set the fresh release's
        if os.path.basename(str(getattr(args, "r8_v9_labels_eval", ""))) == os.path.basename(
                str(json.load(open(EVAL_REMAP, encoding="utf-8"))["--r8-v9-labels-eval"])):
            args.r8_v9_eval_md5 = str(A.v9_md5)
        return _orig_bed(model, cfg, args, config, **kw)
    _bed._r8_patched = True
    _bed._r8_orig = getattr(_orig_bed, "_r8_orig", _orig_bed)
    _LL.build_eval_dataset = _bed
else:
    raise SystemExit("[l1c] train mode is not part of L1C")

import torch  # noqa: E402

_Base = RR.Refcv6Proxy
ST = {"installed": False, "armed": False, "caps": [], "records": [], "t0": time.time(), "W": None, "d": None,
      "c": {"own_calls": 0, "navnull_calls": 0, "caps_per_call_hist": {}, "filteroff_cap_maxdiff": 0.0, "batch_gt1": 0,
            "hook_calls_unarmed": 0}}


def _install(model):
    mods = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "lat_head_tac"]
    if len(mods) != 1:
        raise SystemExit(f"[l1] expected exactly one lat_head_tac module, found {[n for n, _ in mods]}")

    def pre_hook(_mod, inp):
        if ST["armed"]:
            ST["caps"].append(inp[0].detach().float().cpu().numpy().copy())
        else:
            ST["c"]["hook_calls_unarmed"] += 1
        return None
    mods[0][1].register_forward_pre_hook(pre_hook)
    core = getattr(getattr(model, "cfg", None), "core", None)
    ST.update(installed=True, hook_module=mods[0][0], W=(int(core.window) if core is not None and hasattr(core, "window") else None),
              no_strategic=(bool(getattr(core, "no_strategic", False)) if core is not None else None))


class ZProxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        if not ST["installed"]:
            _install(d["model"])
        if nav_cmd is None:
            ST["c"]["navnull_calls"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n = int(frames.shape[0])
        ST["armed"], ST["caps"] = True, []
        try:
            out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        finally:
            ST["armed"] = False
        caps = ST["caps"]
        ST["caps"] = []
        h = ST["c"]["caps_per_call_hist"]
        h[str(len(caps))] = h.get(str(len(caps)), 0) + 1
        if not caps:
            raise RuntimeError("[l1] the OWN call never reached lat_head_tac")
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        if "nav_true" not in conds:
            raise RuntimeError(f"[l1] no nav_true row among conds {conds}")
        r = conds.index("nav_true")
        z = caps[0][r]
        for c in caps[1:]:
            ST["c"]["filteroff_cap_maxdiff"] = max(ST["c"]["filteroff_cap_maxdiff"], float(np.abs(c[r] - z).max()))
        ST["d"] = int(z.shape[-1])
        e_i, t = (int(x) for x in d["ctx"].index)
        ST["records"].append({"e_i": e_i, "t": t, "z": z.astype(np.float32), "traj": out["traj"][r].detach().float().cpu().numpy(),
                              "sel": int(out["sel_idx"][r]) if "sel_idx" in out else -1,
                              "v0_in": (float(v0.reshape(-1)[r]) if v0 is not None else float("nan"))})
        ST["c"]["own_calls"] += 1
        nw = len(ST["records"])
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 250 == 0 or nw <= 2:
            print(f"ZZL1PROGZZ {nw} el={time.time() - ST['t0']:.0f}s d={ST['d']}", flush=True)
        return out


def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    out = {"e_i": np.array([r["e_i"] for r in recs], np.int64), "t": np.array([r["t"] for r in recs], np.int64),
           "z_tac": np.stack([r["z"] for r in recs]).astype(np.float32),
           "traj_own": np.stack([r["traj"] for r in recs]).astype(np.float32),
           "sel_own": np.array([r["sel"] for r in recs], np.int64), "v0_in": np.array([r["v0_in"] for r in recs], np.float64)}
    meta = {"tool": "l1c_ztac_roll.py", "mode": A.mode, "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__),
            "child_argv": CHILD, "remap": REMAP_REC, "hook_module": ST.get("hook_module"), "window_W": ST["W"],
            "no_strategic": ST.get("no_strategic"), "d_tac": ST["d"], "final": bool(final), "n_windows": len(recs),
            "counters": ST["c"], "wall_s": round(time.time() - ST["t0"], 1), "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.ztac_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    RR.Refcv6Proxy = ZProxy                                     # install_patches' load_model resolves the name at call time
    print(f"[l1] mode={A.mode} prereg {'sealed ' + SEAL_SHA[:16] if SEAL_SHA else 'UNSEALED'} child={' '.join(CHILD)} ZZL1ROLLSTARTZZ",
          flush=True)
    t0 = time.time()
    rc = RUNNER()
    if not ST["records"]:
        print("ZZL1NOWINDOWSZZ the ZProxy never ran", flush=True)
        return 4
    save(True, {"runner_rc": rc, "roll_wall_s": round(time.time() - t0, 1),
                "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None)})
    print(f"[l1] banked {len(ST['records'])} windows d_tac={ST['d']} counters={json.dumps(ST['c'])} ZZL1BANKEDZZ", flush=True)
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
