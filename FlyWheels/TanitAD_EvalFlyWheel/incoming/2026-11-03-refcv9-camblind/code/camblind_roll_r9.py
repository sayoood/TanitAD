"""camblind_roll_r9.py -- refcv9 C9: the camera-blind (W4) roll whose swap covers EVERY camera-derived input, not only `_window_u8`'s pixels.

NEW file (refcv9 C9, 2026-11-03).  THE THIN ADAPTER: a copy of `camblind_roll.py` (EvalFlyWheel 2026-10-27, PREREG_ANTICHEAT_REFCV8.md W4; that file is not
edited) in which the three pixels-only pieces are replaced by calls into `stack/tanitad/eval/camblind_r9.py`:
  * `make_dataset_cls_camblind`  ->  `CB.make_camblind_dataset_cls(...)`   (the census-driven swap; fail-closed on an unclassified dataset-item key;
                                      the SAME derangement rule / seed / donor-window rule and controls (a)-(c) as W4)
  * `_other_inputs_digest`       ->  `CB.other_inputs_digest(...)`         (control (c): every NON-camera input, by the census)
  * the bank's meta also records the census' CAMERA keys and the census sha256.
Everything else -- `roll_seed_r8_pf.main()`, the ZProxy's hook on `lat_head_tac`, the sampler seed, the number and order of forwards -- is W4's.
The proof that the swap is COMPLETE is `stack/tests/test_camblind_r9.py` (the sensitivity identity: exactly 0.0 under this instrument, > 0 under a
pixels-only swap, > 0 when the donor's camera is perturbed).  The MM runs this roll at E2; it was NOT run here.

----- the original W4 header, kept for the record -----
camblind_roll.py -- PREREG_ANTICHEAT_REFCV8.md W4 instrument: the os-only inference roll of refcv8 EMA-48,852 (per-frame nav feed) with each
window's CAMERA PIXELS replaced by those of a window of ANOTHER episode (one fixed derangement, seed 0), every ego input unchanged.

  --camera identity    the dataset class is the battery's own, UNCHANGED; the roll must reproduce the BASE bank (ztac_eval_s0.npz) bit for bit  [control (a)]
  --camera derange     the camera-blind roll

Per window the bank also records: the digest of the FED frames (blake2b-8), the donor (e_d, t_d), the digest of EVERY other dataset-item tensor +
nav_cmd / v0 / ego_state as passed to the model (control (c)), and -- for every 25th window of the blind roll -- the digest of the OWN frames
(control (b), self-contained).  No clip id is printed or written; episodes are INDICES into the roll's manifest.

    R8_LAUNCH_TREE=<tree_fb47a58> PYTHONPATH='<tree>/stack;<tree>/taniteval' R8_NO_POSE_SYNC=1 OMP_NUM_THREADS=4 \
      python -X utf8 camblind_roll_r9.py --camera derange --ztac-out <dir>/ztac_blind_s0.npz --require-seal -- \
         --ckpt E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt --config E:/refcv8_eval_kit/ckpt/config.json --seed 0 \
         --dump-dir <dir>/blind_dump_s0 --windows-json <battery tree>/windows_s0.json --device cuda --out-json <dir>/blind_roll_s0.json --os-only
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
PKG = HERE.parents[1] / "2026-10-27-refcv9-anticheat"            # the W4 package: the prereg, its seal and the controls' source live there, unchanged
AC_COMMON = PKG / "code" / "anticheat_common.py"

from tanitad.eval import camblind_r9 as CB  # noqa: E402  (PYTHONPATH carries <tree>/stack)

PREREG, SEAL = PKG / "PREREG_ANTICHEAT_REFCV8.md", PKG / "PREREG_ANTICHEAT_REFCV8.sha256"
PF_CODE = Path(os.environ.get("L1_PF_CODE", "C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-20-refcv8-perframe-final/code"))
R8_CODE = Path(os.environ.get("L1_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
EVAL_REMAP = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/code/refcv8_eval_remap_devbox_l1.json")    # == the L1 / P4 re-pointed eval remap (read-only)
OWN_EVERY = 25

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:      # noqa: BLE001
    pass


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse():
    if "--" not in sys.argv:
        raise SystemExit("[camblind] usage: camblind_roll.py [opts] -- <battery child argv>")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=("identity", "derange"), required=True)
    ap.add_argument("--derange-seed", type=int, default=0)
    ap.add_argument("--ztac-out", required=True)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--require-seal", action="store_true")
    ap.add_argument("--smoke-stop-after", type=int, default=0, help="SMOKE ONLY: bank + hard-exit after N windows (a prefix of the full roll: same sampler offsets)")
    a = ap.parse_args(sys.argv[1:i])
    child = sys.argv[i + 1:]
    if "--os-only" not in child:
        raise SystemExit("[camblind] the eval roll is the battery's --os-only child; --os-only missing")
    return a, child


A, CHILD = parse()
SEAL_SHA = None
if SEAL.exists() and PREREG.exists():
    want = SEAL.read_text(encoding="utf-8").split()[0].strip()
    got = sha256_file(PREREG)
    SEAL_SHA = got if (len(want) == 64 and got == want) else None
if A.require_seal and SEAL_SHA is None:
    raise SystemExit("[camblind] PREREG_ANTICHEAT_REFCV8.md does not match its .sha256 seal: refusing a scored roll")

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402

REMAP_REC = {"devbox_file_replaced": str(r8_repoint.REMAP_OVERRIDES), "launch_tree": str(r8_repoint.LAUNCH_TREE), "file": str(EVAL_REMAP),
             "sha256": sha256_file(EVAL_REMAP)}
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py")] + CHILD
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402  (overlay check, apply_child, nav-feed overlay; READ-ONLY, UNCHANGED)

RR = PF.RR
RUNNER = PF.main

import torch  # noqa: E402

# ---------------------------------------------------------------------------------------------------- the camera swap (dataset level)
DIG: dict = {}          # (e_i, t) -> {"fed": int, "donor": (e_d, t_d), "own": int | 0}
DRANGE: dict = {}       # filled lazily from the restricted window index


_orig_make_dataset_cls = RR.make_dataset_cls


def make_dataset_cls_camblind(tr, ctx):
    """C9: the census-driven swap.  `CB.make_camblind_dataset_cls` is W4's CamBlindWindows (the SAME derangement rule / seed / donor-window rule, the same
    `_window_u8` dict, the same digests) plus (1) a fail-closed census check of every dataset-item key at window 0 and (2) the donor's value for any census
    CAMERA key beyond the window pixels (none today: tests/test_camblind_r9.py::test_the_camera_set_and_the_offline_pixel_derived_set_are_exactly_what_the_census_says)."""
    base = _orig_make_dataset_cls(tr, ctx)
    return CB.make_camblind_dataset_cls(base, camera=A.camera, derange_seed=A.derange_seed, own_every=OWN_EVERY, dig=DIG, drange=DRANGE,
                                        donor_item_fn=lambda ds, j: tr.V3Dataset.__getitem__(ds, j))


RR.make_dataset_cls = make_dataset_cls_camblind          # build_corpus resolves the name in r7_roll's globals at call time

# ---------------------------------------------------------------------------------------------------- the proxy (== l1_ztac_roll.ZProxy + digests)
_Base = RR.Refcv6Proxy
ST = {"installed": False, "armed": False, "caps": [], "records": [], "t0": time.time(), "W": None, "d": None,
      "c": {"own_calls": 0, "navnull_calls": 0, "caps_per_call_hist": {}, "filteroff_cap_maxdiff": 0.0, "batch_gt1": 0, "hook_calls_unarmed": 0}}


def _install(model):
    mods = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "lat_head_tac"]
    if len(mods) != 1:
        raise SystemExit(f"[camblind] expected exactly one lat_head_tac module, found {[n for n, _ in mods]}")

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
        oh = CB.other_inputs_digest(d["ctx"].item, nav_cmd, v0, ego_state)
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
            raise RuntimeError("[camblind] the OWN call never reached lat_head_tac")
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        if "nav_true" not in conds:
            raise RuntimeError(f"[camblind] no nav_true row among conds {conds}")
        r = conds.index("nav_true")
        z = caps[0][r]
        for c in caps[1:]:
            ST["c"]["filteroff_cap_maxdiff"] = max(ST["c"]["filteroff_cap_maxdiff"], float(np.abs(c[r] - z).max()))
        ST["d"] = int(z.shape[-1])
        e_i, t = (int(x) for x in d["ctx"].index)
        dg = DIG.get((e_i, t))
        if dg is None:
            raise RuntimeError(f"[camblind] no frame digest recorded for window ({e_i}, {t})")
        ST["records"].append({"e_i": e_i, "t": t, "z": z.astype(np.float32), "traj": out["traj"][r].detach().float().cpu().numpy(),
                              "sel": int(out["sel_idx"][r]) if "sel_idx" in out else -1,
                              "v0_in": (float(v0.reshape(-1)[r]) if v0 is not None else float("nan")),
                              "fed_h": dg["fed"], "own_h": dg["own"], "other_h": oh, "donor_e": dg["donor"][0], "donor_t": dg["donor"][1]})
        ST["c"]["own_calls"] += 1
        nw = len(ST["records"])
        if nw % int(A.save_every) == 0:
            save(False)
        if nw % 250 == 0 or nw <= 2:
            print(f"ZZCBPROGZZ {nw} el={time.time() - ST['t0']:.0f}s d={ST['d']}", flush=True)
        if A.smoke_stop_after and nw >= A.smoke_stop_after:
            save(True, {"smoke_stop_after": A.smoke_stop_after})
            print(f"ZZCBSMOKESTOPZZ {nw}", flush=True)
            sys.stdout.flush()
            os._exit(0)
        return out


def save(final: bool, extra: dict | None = None):
    recs = ST["records"]
    if not recs:
        return
    u64 = lambda key: np.array([r[key] for r in recs], dtype=np.uint64)  # noqa: E731
    out = {"e_i": np.array([r["e_i"] for r in recs], np.int64), "t": np.array([r["t"] for r in recs], np.int64),
           "z_tac": np.stack([r["z"] for r in recs]).astype(np.float32),
           "traj_own": np.stack([r["traj"] for r in recs]).astype(np.float32),
           "sel_own": np.array([r["sel"] for r in recs], np.int64), "v0_in": np.array([r["v0_in"] for r in recs], np.float64),
           "fed_h": u64("fed_h"), "own_h": u64("own_h"), "other_h": u64("other_h"),
           "donor_e": np.array([r["donor_e"] for r in recs], np.int64), "donor_t": np.array([r["donor_t"] for r in recs], np.int64)}
    meta = {"tool": "camblind_roll_r9.py", "census_camera_keys": list(CB.camera_keys("item")), "census_sha256": CB.census_sha256(), "camblind_r9_sha256": sha256_file(CB.__file__), "camera": A.camera, "derange_seed": A.derange_seed, "derangement": dict(DRANGE),
            "prereg_sha256": SEAL_SHA, "script_sha256": sha256_file(__file__), "common_sha256": (sha256_file(AC_COMMON) if AC_COMMON.exists() else None),
            "child_argv": CHILD, "remap": REMAP_REC, "hook_module": ST.get("hook_module"), "window_W": ST["W"],
            "no_strategic": ST.get("no_strategic"), "d_tac": ST["d"], "final": bool(final), "n_windows": len(recs),
            "own_digest_every": OWN_EVERY, "counters": ST["c"], "wall_s": round(time.time() - ST["t0"], 1), "saved": time.strftime("%FT%T%z"), **(extra or {})}
    out["meta_json"] = np.array(json.dumps(meta, default=str))
    p = Path(A.ztac_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, p)


def main() -> int:
    RR.Refcv6Proxy = ZProxy                                     # install_patches' load_model resolves the name at call time
    print(f"[camblind] camera={A.camera} derange_seed={A.derange_seed} prereg {'sealed ' + SEAL_SHA[:16] if SEAL_SHA else 'UNSEALED'} "
          f"child={' '.join(CHILD)} ZZCBROLLSTARTZZ", flush=True)
    t0 = time.time()
    rc = RUNNER()
    if not ST["records"]:
        print("ZZCBNOWINDOWSZZ the ZProxy never ran", flush=True)
        return 4
    save(True, {"runner_rc": rc, "roll_wall_s": round(time.time() - t0, 1),
                "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_initialized() else None)})
    print(f"[camblind] banked {len(ST['records'])} windows d_tac={ST['d']} derangement={json.dumps(DRANGE)} counters={json.dumps(ST['c'])} ZZCBBANKEDZZ", flush=True)
    return 0 if rc in (0, None) else int(rc)


if __name__ == "__main__":
    sys.exit(main())
