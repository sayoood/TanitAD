"""l2_common.py -- refcv9 L2 (PREREG_L2.md): shared loaders / definitions for the cruise-false-brake lever. CPU only.

Everything is read from banked arrays (no `tools`/`prepare_seed` machinery); the definitions are the battery's own, IMPORTED:
  plan class      = s6_lon_decomp.plan_lon_class (decel/cruise/accel from the LAST chord, +-1.5 m/s vs v0, dt 1.0)
  per-window comps = pair_read_r8.components(ra, P, G, dt) with ra = taniteval/tools/refav1_arm.py loaded by path
  estimator       = pair_read_r8.paired_cell -> taniteval.ci.paired_episode_cluster_bootstrap (B 2000, seed 0, over EPISODES)
  stage           = tanitad.retime_stage.retime_plan with the registered per-frame DEPLOY8 table (S6 slots [1,3,4,5,6,7] of the 8-slot plan)
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

TREE = os.environ.get("R8_LAUNCH_TREE", "C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58")
BATT = os.environ.get("L2_BATT_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                      "2026-10-15-refcv8-milestone-battery/code")
ARCH = Path("C:/Users/Admin/refcv8_eval_archive/battery")
PFPKG = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final")
DEPLOY_DIR = PFPKG / "pfpkg/raw/retime_deploy"
STAGE_FILE = Path(os.environ.get("RETIME_STAGE_FILE", "C:/lgs/retime_stage_scratch/stack/tanitad/retime_stage.py"))
EVAL_TREE = "step48852ema_pf"
TRAIN_TREE30 = Path("C:/Users/Admin/refcv8_pf/trainfit/step30000ema_train139_pf")
LON6 = ("FOLLOW", "CRUISE", "BRAKE_TO", "CREEP", "HOLD", "ACCELERATE")
LON6_V7_IDS = (0, 1, 3, 4, 5, 7)
# groups over LON6 indices: decel-set = BRAKE_TO, CREEP, HOLD ; cruise-set = FOLLOW, CRUISE ; accel = ACCELERATE
G_DECEL, G_CRUISE, G_ACCEL = (2, 3, 4), (0, 1), (5,)
S6_SLOTS = [1, 3, 4, 5, 6, 7]
S6_T = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
ALL_T = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])
CLS = ("decel", "cruise", "accel")


def sha256_file(p) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_STACK = {}


def stack():
    """pair_read_r8 (PR), s6_lon_decomp (S6), refav1_arm (ra), retime_stage (RS). Idempotent."""
    if _STACK:
        return _STACK
    for p in (f"{TREE}/stack", f"{TREE}/taniteval", BATT):
        if p not in sys.path:
            sys.path.insert(0, p)
    import pair_read_r8 as PR
    import s6_lon_decomp as S6
    sp = Path(TREE) / "taniteval/tools/refav1_arm.py"
    spec = importlib.util.spec_from_file_location("refav1_arm", str(sp))
    ra = importlib.util.module_from_spec(spec)
    sys.modules["refav1_arm"] = ra
    spec.loader.exec_module(ra)
    spec2 = importlib.util.spec_from_file_location("_retime_stage_l2", str(STAGE_FILE))
    RS = importlib.util.module_from_spec(spec2)
    sys.modules["_retime_stage_l2"] = RS
    spec2.loader.exec_module(RS)
    _STACK.update(PR=PR, S6=S6, ra=ra, RS=RS)
    return _STACK


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def plan_class(P, v0, dt=1.0):
    return stack()["S6"].plan_lon_class(P, v0, dt)


def deploy_table():
    RS = stack()["RS"]
    return RS.load_gains_table(DEPLOY_DIR / "refcv8_retime_deploy_gains_pf.json")


def apply_stage(plan8, prior8, table=None):
    """The shipped stage as the banked DEPLOY8 npz computed it: the FULL 8-slot call (8 knots), then the S6 slots [1,3,4,5,6,7]
    ([n,8,2] -> [n,6,2], cast=True == what SHIP-R8 scored). MEASURED bit-equal to the banked S6_pos (smoke/stage_repro.py)."""
    RS = stack()["RS"]
    table = table or deploy_table()
    full = RS.retime_plan(np.asarray(plan8), np.asarray(prior8), ALL_T, table, cast=True)
    return full[:, S6_SLOTS]


def load_eval_pf(seed: int, tree: str = EVAL_TREE) -> dict:
    """The S6 surface of the banked 48,852 per-frame battery at inference seed `seed`: panel arms + extras (lon logits) + the banked stage."""
    root = ARCH / tree
    z = np.load(DEPLOY_DIR / f"DEPLOY8_POS_{tree}_s{seed}.npz", allow_pickle=True)
    ex = np.load(root / f"dump_s{seed}" / "refcv6_extras.npz", allow_pickle=True)
    n6 = len(z["S6_ws"])
    pdir = root / f"s6_s{seed}"
    man = jload(pdir / "manifest.json")
    files = sorted(glob.glob(str(pdir / "ep*.npz")))
    g, v0, os_, r7, ws, clip = [], [], [], [], [], []
    for f in files:
        a = np.load(f)
        g.append(a["g"]); v0.append(a["v0"]); os_.append(a["os"]); r7.append(a[f"b_refcv7_50400_s{seed}"]); ws.append(a["ws"])
        clip += [int(a["clip_index"][0])] * len(a["ws"])
    g, v0, os_, r7 = (np.concatenate(x).astype(np.float64 if k == 1 else np.float32) for k, x in enumerate((g, v0, os_, r7)))
    ws = np.concatenate(ws).astype(np.int64)
    clip = np.asarray(clip, np.int64)
    chk = {"s6_n": int(len(ws)), "ws_equal_deploy": bool(np.array_equal(ws, z["S6_ws"])),
           "clip_equal_deploy": bool(np.array_equal(clip, z["S6_clip"])),
           "os_equal_deploy": bool(np.array_equal(os_, z["S6_os"])),
           "plan8_slots_equal_os": float(np.abs(z["S6_plan8"][:, S6_SLOTS] - os_).max())}
    rows = np.asarray(z["S6_rows"], np.int64)
    # extras are in S2 roll order; S6_rows indexes the S2 list
    chk["extras_ws_equal"] = bool(np.array_equal(ex["ws"][rows], z["S6_ws"])) and bool(np.array_equal(ex["clip_index"][rows], z["S6_clip"]))
    lg = ex["nav_true.tacv6_lon_logits"].astype(np.float64)
    out = {"seed": seed, "g": g.astype(np.float64), "v0": v0, "os": os_, "r7": r7, "ws": ws, "clip": clip, "eid": np.asarray(z["S6_eid"]).astype(np.int64),
           "compliant": np.asarray(z["S6_compliant"], bool), "plan8": z["S6_plan8"], "prior8": z["S6_prior8"], "stage_banked": z["S6_pos"],
           "rows": rows, "logits_s2": lg, "logits": lg[rows], "lon_v7": ex["lon_v7"][rows], "lat_v7": ex["lat_v7"][rows],
           "v0_extras": ex["v0"][rows], "traj_s2": ex["nav_true.traj"], "sel_s2": ex["nav_true.sel_idx"], "ex_ws": ex["ws"], "ex_clip": ex["clip_index"],
           "dt": float(man["grid"]["dt_s"]), "checks": chk}
    return out


def arm_metrics(E: dict, P, name=""):
    """Per-window components of a 6-instant plan P against E['g'] (battery definitions)."""
    st = stack()
    c = st["PR"].components(st["ra"], np.asarray(P, np.float32), E["g"].astype(np.float32), E["dt"])
    c = {k: np.asarray(v, np.float64) for k, v in c.items()}
    P64 = np.asarray(P, np.float64)
    c["along_signed_final_m"] = P64[:, -1, 0] - E["g"][:, -1, 0]
    return c


def lon_indicators(P, E, mask):
    """Per-window false-decel (on GT-cruise) and missed-decel (on GT-decel) indicators of plan P (NaN outside the population)."""
    gt = plan_class(E["g"], E["v0"], E["dt"])
    pc = plan_class(np.asarray(P, np.float64), E["v0"], E["dt"])
    fd = np.where(mask & (gt == "cruise"), (pc == "decel").astype(np.float64), np.nan)
    md = np.where(mask & (gt == "decel"), (pc != "decel").astype(np.float64), np.nan)
    return fd, md, gt, pc


# ------------------------------------------------------------------------------------------------------ TRAIN side (A6 train139)
def load_train_dump(dump_dir, tag: str = "") -> dict:
    """S6 train windows of an A6-train139 roll DUMP (`<dump_dir>/refcv6_extras.npz` + `<dump_dir>/decisions/ep*.npz`), keyed by episode file.
    The GT class is derived from the world-frame 60-step future in decisions/gt_future_ext (the same future s6_lon_decomp.load_tree reads):
    plan_lon_class needs only |g(6 s) - g(5 s)| / dt and v0, i.e. steps 50 and 60 (indices 49, 59); S6 windows = those with a fully valid
    60-step future. Compliance is model-free: gt_max_speed_6s <= v_max_ms (reproduces the eval mask exactly, smoke/compliance_probe.py)."""
    dump_dir = Path(dump_dir)
    ex = np.load(dump_dir / "refcv6_extras.npz", allow_pickle=True)
    files = sorted(glob.glob(str(dump_dir / "decisions" / "ep*.npz")))
    ws, clip_i, ge, gv, epi = [], [], [], [], []
    pdir = dump_dir
    for f in files:
        d = np.load(f)
        n = len(d["ws"])
        ws.append(d["ws"]); ge.append(d["gt_future_ext"][:, [49, 59], :3]); gv.append(d["gt_future_valid_ext"].min(1))
        epi += [int(Path(f).name[2:5])] * n
    ws = np.concatenate(ws).astype(np.int64); ge = np.concatenate(ge).astype(np.float64); gv = np.concatenate(gv); epi = np.asarray(epi, np.int64)
    if len(ws) != len(ex["ws"]) or not np.array_equal(ws, ex["ws"]):
        raise SystemExit(f"[l2] {dump_dir}: decisions ws order != extras ws order ({len(ws)} vs {len(ex['ws'])})")
    v_end = np.linalg.norm(ge[:, 1, :2] - ge[:, 0, :2], axis=-1) / 1.0
    v0 = ex["v0"].astype(np.float64)
    dv = v_end - v0
    gt = np.where(dv <= -1.5, "decel", np.where(dv >= 1.5, "accel", "cruise")).astype(object)
    valid = gv >= 1.0
    vm, vv, gm = ex["v_max_ms"], ex["v_max_valid"], ex["gt_max_speed_6s"]
    comp = (gm <= vm) | (vv < 0.5)
    return {"ws": ws, "clip": ex["clip_index"].astype(np.int64), "eid": ex["clip_index"].astype(np.int64), "gt": gt, "v0": v0, "valid_s6": valid, "compliant": comp,
            "logits": ex["nav_true.tacv6_lon_logits"].astype(np.float64), "lon_v7": ex["lon_v7"], "dump": str(dump_dir), "tag": tag or str(dump_dir),
            "extras_sha256": sha256_file(dump_dir / "refcv6_extras.npz"), "n_rows": int(len(ws))}


def lon6(logits8):
    return np.asarray(logits8, np.float64)[:, list(LON6_V7_IDS)]
