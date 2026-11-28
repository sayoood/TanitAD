"""l1_ztac_roll.py -- refcv9 L1 (PREREG_L1.md): ONE inference-seed roll of refcv8 EMA-48,852 (per-frame nav feed, os-only) that BANKS
the shared tactical latent `z_tac` of every window, read-only, beside the battery's own record.

NEW file (Architecture & Inference, 2026-10-25).  It is a WRAPPER on the battery children; no shared file is edited:
  * --mode eval : `roll_seed_r8_pf.py` (hold_r8pf, the per-frame eval battery child) is IMPORTED and its `main()` runs with the battery
    child's own argv (`--os-only`, the battery's windows file) -- the r8oracle / P4 pattern.  The dev-box remap is re-pointed exactly as
    P4 did (the launch tree moved to the rebuilt tree_fb47a58; one path in the remap named it).
  * --mode train: `roll_seed_r8_train_pf.py`'s main() is REPLICATED line for line (overlay check, nav-feed overlay, `--nav-feed
    perframe_full`), around a SHIM COPY of `roll_seed_r8_train.py` that is asserted BYTE-IDENTICAL (sha256) to the battery's file; the shim
    directory carries the train remap with ONE value re-pointed (`--join-defect-masks` -> the rebuilt tree; recorded).  Window rule
    unchanged: A6 train139, t % 5 == 0.
  * `r7_roll.Refcv6Proxy` is SUBCLASSED (`ZProxy`) and re-bound on the module before the roll builds it.  ZProxy runs the battery's own
    `__call__` UNCHANGED; a forward PRE-hook on the model's single `lat_head_tac` module copies its input (= `z_tac`, refs/refc_v3.py:
    1737-1743 under --no-strategic: z_tac = z_tac_raw = phi_tac(pooled) + nav + max-speed terms) for the nav_true row.  Nothing is
    written back; the hook returns None.  The filter-off sensitivity forward (eval) calls the head a second time: both captures are kept
    and their max |difference| is counted (the filter acts downstream of z_tac, so it must read 0).
  * also banked per window: the OWN plan (8 slots) and sel_idx, so the analysis can assert OWN == the banked battery dump (control C1).

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 l1_ztac_roll.py --mode eval --ztac-out <dir>/ztac_eval_s0.npz --require-seal -- \
         --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
         --dump-dir <dir>/eval_dump_s0 --windows-json <bank>/windows_s0.json --device cuda --out-json <dir>/eval_roll_s0.json --os-only
      python -X utf8 l1_ztac_roll.py --mode train --ztac-out <dir>/ztac_train_s0.npz --require-seal -- \
         --ckpt ... --config ... --seed 0 --dump-dir <dir>/train_dump_s0 --out-json <dir>/train_roll_s0.json --device cuda
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
PREREG, SEAL = ROOT / "PREREG_L1.md", ROOT / "PREREG_L1.sha256"
PF_CODE = Path(os.environ.get("L1_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L1_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
SHIM = HERE / "train_shim"
EVAL_REMAP = HERE / "refcv8_eval_remap_devbox_l1.json"     # == P4's re-pointed eval remap (copied; sha256 recorded)

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
    ap.add_argument("--mode", choices=("eval", "train"), required=True)
    ap.add_argument("--ztac-out", required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--require-seal", action="store_true")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if a.mode == "eval" and "--os-only" not in child:
        raise SystemExit("[l1] eval mode rolls the battery's --os-only child; --os-only missing")
    return a, child


A, CHILD = parse()
SEAL_SHA = None
if SEAL.exists() and PREREG.exists():
    want = SEAL.read_text(encoding="utf-8").split()[0].strip()
    got = sha256_file(PREREG)
    SEAL_SHA = got if (len(want) == 64 and got == want) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[l1] PREREG_L1.md does not match PREREG_L1.sha256: refusing a scored roll")

REMAP_REC: dict = {}
if A.mode == "eval":
    sys.path.insert(0, str(R8_CODE))
    import r8_repoint  # noqa: E402
    REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE),
                 "file": str(EVAL_REMAP), "sha256": sha256_file(EVAL_REMAP)}
    r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
    sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
    sys.path.insert(0, str(PF_CODE))
    import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay; READ-ONLY, UNCHANGED)
    RR = PF.RR
    RUNNER = PF.main
else:
    src, shim = R8_CODE / "roll_seed_r8_train.py", SHIM / "roll_seed_r8_train.py"
    if sha256_file(src) != sha256_file(shim):
        raise SystemExit("[l1] the train shim is not byte-identical to the battery's roll_seed_r8_train.py: refusing")
    os.environ["PF_R8_CODE"] = str(SHIM)                    # roll_seed_r8_train_pf.R8_CODE -> the shim (read at import)
    sys.path.insert(0, str(R8_CODE))                        # r8_repoint, r8_battery_patch ... (imported by the shim)
    sys.path.insert(0, str(PF_CODE))
    import roll_seed_r8_train_pf as TPF  # noqa: E402
    REMAP_REC = {"shim_dir": str(SHIM), "shim_roll_sha256": sha256_file(shim), "battery_roll_sha256": sha256_file(src),
                 "train_remap_file": str(SHIM / "refcv8_train_remap_devbox.json"),
                 "train_remap_sha256": sha256_file(SHIM / "refcv8_train_remap_devbox.json"),
                 "battery_train_remap_sha256": sha256_file(R8_CODE / "refcv8_train_remap_devbox.json")}
    sys.argv = [str(shim)] + CHILD
    ov = TPF.check_overlay()
    sys.path.insert(0, str(SHIM))
    import roll_seed_r8_train as T  # noqa: E402  (the shim: TRAIN remap (re-pointed copy) + loader patch at import)
    if Path(T.__file__).resolve() != shim.resolve():
        raise SystemExit(f"[l1] imported {T.__file__}, not the shim")
    sys.path.insert(0, str(TPF.NF_OVERLAY))
    import navfeed_overlay as NO  # noqa: E402
    _R = NO.install_into_r7_roll(T.RR)
    _meta_nf = dict(_R.__navfeed__)
    _orig_main = _R.main

    def _main_with_feed(argv_=None):
        av = list(argv_ or [])
        if "--nav-feed" in av:
            raise SystemExit("[l1] the parent argv already carries --nav-feed")
        return _orig_main(av + ["--nav-feed", TPF.NAV_FEED])
    _R.main = _main_with_feed
    RR = T.RR

    def RUNNER() -> int:                                    # == roll_seed_r8_train_pf.main() after its imports (replicated)
        T.main()
        a = {sys.argv[i]: sys.argv[i + 1] for i in range(1, len(sys.argv) - 1) if sys.argv[i].startswith("--")}
        out_json, dump_dir = a["--out-json"], a["--dump-dir"]
        rec = json.load(open(out_json, encoding="utf-8"))
        cen = TPF.sidecar_census(dump_dir)
        rec["pf"] = {"nav_feed": TPF.NAV_FEED, "ok": cen["ok"], "census": cen, "overlay": ov, "patched_refcv3_arm": _meta_nf,
                     "parent_roll_script": str(shim), "parent_roll_script_sha256": sha256_file(shim),
                     "battery_roll_script_sha256": sha256_file(src), "this_script_sha256": sha256_file(__file__),
                     "spec": "roll_seed_r8_train_pf.main replicated by l1_ztac_roll.py (PREREG_L1.md); shim byte-identical"}
        rec["tool"] = "l1_ztac_roll.py --mode train (replicates roll_seed_r8_train_pf.py around the byte-identical shim)"
        json.dump(rec, open(out_json, "w", encoding="utf-8"), indent=1, default=str)
        print(f"[l1-train] pf.ok={cen['ok']} windows={cen['n_windows']} ZZL1TRAINPF{'OK' if cen['ok'] else 'BAD'}ZZ", flush=True)
        return 0 if cen["ok"] else 3

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
    meta = {"tool": "l1_ztac_roll.py", "mode": A.mode, "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__),
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
