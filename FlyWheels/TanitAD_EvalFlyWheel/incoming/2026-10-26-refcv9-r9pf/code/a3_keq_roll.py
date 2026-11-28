"""a3_keq_roll.py -- PREREG_REFCV9_INMODEL_PA K-EQ / K-OFF: a SUBSET roll of refcv8 EMA-48,852 with the R9 head IN THE FORWARD (the patched tip tree),
inference seed 1, per-frame nav feed, --os-only -- the battery child (roll_seed_r8_pf.py) IMPORTED and run unchanged (the l1_ztac_roll.py pattern).
Dev-box GPU (RTX 4060) when nvidia-smi shows room.  NEW file (EvalFlyWheel 2026-10-26).

  * R8_LAUNCH_TREE = F:/refcv9_r9pf_scratch/launch_tree: stack = a COPY of the tip tree + the banked R9 patch (4b5f7fb), taniteval = the patch base
    425762053c75, the refcv7 battery code = fb47a58's (identical tree).  `tanitad` is asserted to be imported from THERE.
  * the windows: the battery's windows_s1.json, filtered IN MEMORY to the first clips in sha12 order until >= --min-windows (no clip id is written).
  * r7_roll.Refcv6Proxy is SUBCLASSED: on the first call the model gets enable_refcv9(R9Config of the REGISTERED head) + its state dict, eval mode;
    every nav_true row records traj (in-forward R9 plan), traj_preretime, r9_decision, r9_s_target, r9_s_prior, r9_x, and z_tac (lat_head_tac pre-hook).
  * --dry-build: the windows filter is EMPTY -> the model is built and the head attached, NO window is rolled (plumbing check, no eval output).
Then (not --dry-build): K-OFF (traj_preretime == the eval bank's traj_own of seed 1, every window) and K-EQ (in-forward traj == the offline package
R9PF_fs0 S2_pos8 of seed 1 on the same window, max |d| <= 1e-5 m) -> result/R9PF_KEQ.json.

    python -X utf8 a3_keq_roll.py [--dry-build] [--min-windows 500]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LT = Path(os.environ.get("R9PF_LT", "F:/refcv9_r9pf_scratch/launch_tree"))
ap = argparse.ArgumentParser()
ap.add_argument("--dry-build", action="store_true")
ap.add_argument("--min-windows", type=int, default=500)
ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
ap.add_argument("--head", default="R9PF_fs0")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-head", action="store_true", help="flags-OFF: the R9 head is NOT built (AMENDMENT1 K-OFF arm on eval139; a diagnostic on fresh)")
ap.add_argument("--built-unused", action="store_true", help="DIAGNOSTIC (fresh only, with --no-head): build + load the head, then detach it (r9_prog=None)")
ap.add_argument("--kit", choices=("eval139", "fresh"), default="eval139",
                help="fresh = PRE-TEST on the CALIBRATION split (300 fresh clips, not eval139): plumbing + in-forward capture + an equality read "
                     "on calibration rows; written to scratch only")
A = ap.parse_args()
FRESH = A.kit == "fresh"
if FRESH:
    os.environ["R8_EVAL_KIT"] = "F:/l1c/kit_fresh"          # BEFORE r8_repoint is imported (l1c_ztac_roll.py's fresh plumbing)
    A.seed = 0
os.environ["R8_LAUNCH_TREE"] = str(LT)
os.environ.setdefault("R8_NO_POSE_SYNC", "1")
for p in (str(LT / "taniteval"), str(LT / "stack")):
    sys.path.insert(0, p)
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

import numpy as np  # noqa: E402

RC.check_prereg()
AMEND = FIXPIN = None
if not FRESH and not A.dry_build:          # AMENDMENT1: an eval139 roll runs only on the pinned FIX tree, with the seal and every pin re-hashed
    AMEND, FIXPIN = RC.check_amendment1(), RC.check_fix_pin()
    if LT.resolve() != RC.FIX_TREE.resolve():
        raise SystemExit(f"[keq] eval139 rolls run on the pinned fix tree {RC.FIX_TREE}, not {LT} -- refusing")
PF_CODE = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code")
R8_CODE = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
EVAL_REMAP = Path("C:/Users/Admin/refcv8_pf/refcv9_l1/code/refcv8_eval_remap_devbox_l1.json")
WIN = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf") / f"windows_s{A.seed}.json"
if FRESH:
    EVAL_REMAP = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c/code/remap_l1c_fresh.json")
    WIN = Path("F:/l1c/kit_fresh/windows_s0.json")
    V9_MD5 = Path("F:/l1c/kit_fresh/data/v9/v9_labels_eval.md5").read_text(encoding="utf-8").strip()
if A.built_unused and not FRESH:
    raise SystemExit("[keq] --built-unused is a fresh-split diagnostic only")
OUTD = RC.SCRATCH / ("keq_dry" if A.dry_build else (((f"keq_fresh_builtunused" if A.built_unused else f"keq_fresh_nohead") if A.no_head else f"keq_fresh_pretest") if FRESH
                                                    else f"keq_s{A.seed}_{'off' if A.no_head else 'on'}"))
OUTD.mkdir(parents=True, exist_ok=True)
REG = RC.WORK / "REGISTERED_R9PF_PA.sha256"
reg_sha = ""
for ln in REG.read_text(encoding="utf-8-sig").splitlines() if REG.is_file() else []:
    parts = ln.strip().split(None, 1)
    if len(parts) == 2 and Path(parts[1].lstrip("*").strip()).as_posix() == (RC.HEADS / f"{A.head}.pt").as_posix():
        reg_sha = RC.hex64(parts[0])
if not A.dry_build and not FRESH and len(reg_sha) != 64:
    raise SystemExit("[keq] the head is not pinned in REGISTERED_R9PF_PA.sha256 -- refusing an eval roll")

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402

if Path(r8_repoint.LAUNCH_TREE).resolve() != LT.resolve():
    raise SystemExit(f"[keq] r8_repoint.LAUNCH_TREE {r8_repoint.LAUNCH_TREE} != {LT}")
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py"), "--ckpt", RC.CKPT, "--config", "E:/refcv8_eval_kit/ckpt/config.json", "--seed", str(A.seed),
            "--dump-dir", str(OUTD / "dump"), "--windows-json", str(WIN), "--device", A.device, "--out-json", str(OUTD / "roll.json"), "--os-only"]
sys.path.insert(0, str(PF_CODE))
import roll_seed_r8_pf as PF  # noqa: E402

import torch  # noqa: E402
import tanitad  # noqa: E402

TAN = Path(tanitad.__file__).resolve()
if LT.resolve() not in TAN.parents:
    raise SystemExit(f"[keq] tanitad imported from {TAN}, not the R9 launch tree")
from tanitad.refs import refcv9_progress as R9  # noqa: E402

if RC.sha256_file(R9.__file__) != RC.sha256_file(RC.R9_ORIG_FILE):
    raise SystemExit("[keq] the launch tree's refcv9_progress.py differs from the banked one")
from tanitad.refs import refc_v3 as _RCV3  # noqa: E402

RCV3_SHA = RC.sha256_file(_RCV3.__file__)
if FIXPIN is not None and FIXPIN["pins"].get(Path(_RCV3.__file__).resolve().as_posix()) != RCV3_SHA:
    raise SystemExit(f"[keq] the imported refc_v3.py ({_RCV3.__file__}, {RCV3_SHA}) is not the pinned fixed file -- refusing")
import run_battery_r7 as RB  # noqa: E402
import tanitad.eval.refcv7_loader as _LP  # noqa: E402

# POSE CLOCK (r8clamp_patch.py's rule, re-implemented): the tip loader enables the pose shift whenever the argv names --pose-sync-sidecar, which
# fb47a58's loader (the banked rolls) never did.  Under R8_NO_POSE_SYNC=1 the sidecar is blanked for the dataset build and asserted NOT attached.
POSE = {}
_cur_bed = _LP.build_eval_dataset


def _bed_pose(model, cfg, args, config, **kw):
    saved = getattr(args, "pose_sync_sidecar", None)
    no_sync = os.environ.get("R8_NO_POSE_SYNC") in ("1", "true", "yes")
    if saved and no_sync:
        args.pose_sync_sidecar = None
    try:
        res = _cur_bed(model, cfg, args, config, **kw)
    finally:
        if saved:
            args.pose_sync_sidecar = saved
    attached = getattr(res[0], "pose_sync", None) is not None
    if no_sync and attached:
        raise SystemExit("[keq] R8_NO_POSE_SYNC=1 but the eval dataset HAS the pose shift attached")
    POSE.update(policy="uncorrected clock: sidecar blanked, shift NOT attached" if no_sync else "corrected", argv_names_sidecar=bool(saved),
                shift_attached=attached)
    return res


_bed_pose._r8_patched = True
_bed_pose._r8_orig = getattr(_cur_bed, "_r8_orig", _cur_bed)
_LP.build_eval_dataset = _bed_pose

if FRESH:                                                     # l1c_ztac_roll.py's one-line v9-md5 patch, verbatim in effect
    import tanitad.eval.refcv7_loader as _LL  # noqa: E402
    _orig_bed = _LL.build_eval_dataset

    def _bed(model, cfg, args, config, **kw):
        if os.path.basename(str(getattr(args, "r8_v9_labels_eval", ""))) == os.path.basename(
                str(json.load(open(EVAL_REMAP, encoding="utf-8"))["--r8-v9-labels-eval"])):
            args.r8_v9_eval_md5 = V9_MD5
        return _orig_bed(model, cfg, args, config, **kw)
    _bed._r8_patched = True
    _bed._r8_orig = getattr(_orig_bed, "_r8_orig", _orig_bed)
    _LL.build_eval_dataset = _bed

RR = PF.RR


def digest_tree(o, pre=""):
    """sha256 of every tensor in a (nested) model output, with dtype and shape; non-tensor leaves by repr (flagged if address-bearing)."""
    res = {}
    if torch.is_tensor(o):
        t = o.detach().cpu().contiguous()
        b = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
        res[pre] = f"{t.dtype}|{tuple(t.shape)}|" + hashlib.sha256(b.numpy().tobytes()).hexdigest()
    elif isinstance(o, dict):
        for k, v in o.items():
            res.update(digest_tree(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            res.update(digest_tree(v, f"{pre}[{i}]"))
    else:
        r = repr(o)
        res[pre] = ("OBJ_ADDR|" if " at 0x" in r else "repr|") + hashlib.sha256(r.encode("utf-8")).hexdigest()
    return res


ST = {"digests": [], "armed": False, "caps": [], "records": [], "attached": None, "c": {"calls": 0, "navnull": 0, "batch_gt1": 0, "no_r9_key": 0}}


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


_orig_roll_one = RB.roll_one


def roll_one_subset(ckpt, config, seed, dump_dir, windows, *a, **k):
    keys = sorted(windows.keys(), key=sha12)
    keep, n = [], 0
    if not A.dry_build:
        for key in keys:
            keep.append(key)
            n += len(windows[key])
            if n >= A.min_windows:
                break
    sub = {key: windows[key] for key in keep}
    ST["subset"] = {"n_clips": len(sub), "n_windows_listed": int(sum(len(v) for v in sub.values())), "clip_sha12": [sha12(x) for x in keep],
                    "rule": f"first clips in sha12 order until >= {A.min_windows} windows (battery windows_s{A.seed}.json)"}
    return _orig_roll_one(ckpt, config, seed, dump_dir, sub, *a, **k)


RB.roll_one = roll_one_subset
_Base = RR.Refcv6Proxy
_orig_build = RB.L.build_model


def build_and_attach(*a, **k):
    res = _orig_build(*a, **k)
    attach(res[0])                                   # the head is attached right after the loader's build, before any window
    return res


RB.L.build_model = build_and_attach


def attach(model):
    import r9pf_common as RCm
    head_p = RC.HEADS / f"{A.head}.pt"
    _h, blob, sh = RCm.load_head(head_p, reg_sha if not (A.dry_build or FRESH) else None)
    cfg = R9.R9Config(**blob["r9_config"])
    dev = next(model.parameters()).device
    n_new = 0
    if A.no_head and A.built_unused:
        n_new = model.enable_refcv9(cfg)
        model.r9_prog.load_state_dict(blob["state_dict"], strict=True)
        model.r9_prog.to(dev)
        model.r9_prog = None
    if not A.no_head:
        n_new = model.enable_refcv9(cfg)
        model.r9_prog.load_state_dict(blob["state_dict"], strict=True)
        model.r9_prog.to(dev)
        model.r9_prog.eval()
    mods = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "lat_head_tac"]
    if len(mods) != 1:
        raise SystemExit(f"[keq] expected one lat_head_tac, found {len(mods)}")

    def pre_hook(_m, inp):
        if ST["armed"]:
            ST["caps"].append(inp[0].detach().float().cpu().numpy().copy())
        return None
    mods[0][1].register_forward_pre_hook(pre_hook)
    ST["attached"] = {"head": str(head_p), "sha256": sh, "n_params": int(n_new), "r9_config": blob["r9_config"], "device": str(dev),
                      "model_training": bool(model.training), "r9_training": (None if model.r9_prog is None else bool(model.r9_prog.training))}
    print(f"[keq] R9 head attached: {ST['attached']}", flush=True)


class R9Proxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        if ST["attached"] is None:
            raise RuntimeError("[keq] a forward ran before the R9 head was attached")
        if nav_cmd is None:
            ST["c"]["navnull"] += 1
            out0 = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
            ST["digests"].append({"navnull": True, "idx": [int(x) for x in d["ctx"].index], "d": digest_tree(out0)})
            return out0
        n = int(frames.shape[0])
        ST["armed"], ST["caps"] = True, []
        try:
            out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        finally:
            ST["armed"] = False
        ST["c"]["calls"] += 1
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        r = conds.index("nav_true")
        e_i, t = (int(x) for x in d["ctx"].index)
        ST["digests"].append({"navnull": False, "idx": [e_i, t], "conds": conds, "d": digest_tree(out)})
        f = lambda k: out[k][r].detach().float().cpu().numpy()  # noqa: E731
        if A.no_head:
            tr_ = f("traj")
            ST["records"].append({"e_i": e_i, "t": t, "traj": tr_, "pre": tr_, "s_t": np.zeros(8, np.float32), "s_p": np.zeros(8, np.float32),
                                  "x": np.zeros(9, np.float32), "dec": -1, "z": ST["caps"][0][r].astype(np.float32)})
            return out
        if "traj_preretime" not in out:
            ST["c"]["no_r9_key"] += 1
            return out
        ST["records"].append({"e_i": e_i, "t": t, "traj": f("traj"), "pre": f("traj_preretime"), "s_t": f("r9_s_target"), "s_p": f("r9_s_prior"),
                              "x": f("r9_x"), "dec": int(out["r9_decision"][r]), "z": ST["caps"][0][r].astype(np.float32)})
        return out


RR.Refcv6Proxy = R9Proxy
t0 = time.time()
if A.dry_build:
    # build the model through the battery's own loader path, attach the head, roll nothing
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        PF.RS.main()
        rc_note = "RS.main returned"
    except Exception as ex:      # noqa: BLE001  (an empty window set may trip the summary; the build + attach is what is checked)
        rc_note = f"RS.main raised {type(ex).__name__}: {str(ex)[:300]}"
    rec = {"dry_build": True, "subset": ST.get("subset"), "tanitad_file": str(TAN), "attached": ST["attached"], "rs_main": rc_note,
           "forward_calls": ST["c"], "wall_s": round(time.time() - t0, 1)}
    print(json.dumps(rec, default=str)[:800], flush=True)
    RC.jdump(rec, OUTD / "DRY_BUILD.json")
    sys.exit(0)
PF.RS.main()
recs = ST["records"]
if not FRESH:
    # AMENDMENT1: the roll banks its records + per-call output digests in scratch; a3_keq_compare.py reads the ON / OFF pair -> result/R9PF_KEQ.json
    np.savez(OUTD / "keq_records.npz", e_i=np.array([r["e_i"] for r in recs]), t=np.array([r["t"] for r in recs]), traj=np.stack([r["traj"] for r in recs]),
             pre=np.stack([r["pre"] for r in recs]), s_t=np.stack([r["s_t"] for r in recs]), s_p=np.stack([r["s_p"] for r in recs]),
             x=np.stack([r["x"] for r in recs]), dec=np.array([r["dec"] for r in recs]), z=np.stack([r["z"] for r in recs]))
    RC.jdump(ST["digests"], OUTD / "OUTPUT_DIGESTS.json")
    rec = {"tool": "a3_keq_roll.py", "script_sha256": RC.sha256_file(__file__), "flags": ("OFF (head NOT built)" if A.no_head else "ON (head built, decision applied)"),
           "seed": A.seed, "device": A.device, "head": ST["attached"], "pose_clock": POSE, "subset": ST["subset"], "counters": ST["c"], "n_records": len(recs),
           "n_digest_calls": len(ST["digests"]), "tanitad_file": str(TAN), "launch_tree": str(LT), "refc_v3_file": str(_RCV3.__file__), "refc_v3_sha256": RCV3_SHA,
           "amendment1": AMEND, "fix_pin": FIXPIN, "records_sha256": RC.sha256_file(OUTD / "keq_records.npz"),
           "digests_sha256": RC.sha256_file(OUTD / "OUTPUT_DIGESTS.json"), "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
    RC.jdump(rec, OUTD / "ROLL_RECORD.json")
    print(f"[keq] {rec['flags']} seed {A.seed}: {len(recs)} records, {len(ST['digests'])} digested calls  ZZKEQROLLDONEZZ", flush=True)
    sys.exit(0)
np.savez(OUTD / "keq_records.npz", e_i=np.array([r["e_i"] for r in recs]), t=np.array([r["t"] for r in recs]), traj=np.stack([r["traj"] for r in recs]),
         pre=np.stack([r["pre"] for r in recs]), s_t=np.stack([r["s_t"] for r in recs]), s_p=np.stack([r["s_p"] for r in recs]),
         x=np.stack([r["x"] for r in recs]), dec=np.array([r["dec"] for r in recs]), z=np.stack([r["z"] for r in recs]))
if FRESH:
    zf = np.load(RC.FRESH_BANK, allow_pickle=False)
    Wf = int(json.loads(str(zf["meta_json"]))["window_W"])
    fk = {(int(c), int(t)): j for j, (c, t) in enumerate(zip(zf["e_i"], zf["t"]))}
    jf = np.array([fk.get((r["e_i"], r["t"]), -1) for r in recs])
    cr = np.load(RC.SCRATCH / "calib_rows.npz", allow_pickle=False)
    ck = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(cr["clip"], cr["ws"]))}
    jc = np.array([ck.get((r["e_i"], r["t"] + Wf - 1), -1) for r in recs])
    m = (jf >= 0) & (jc >= 0)
    h, blob, _sh = RC.load_head(RC.HEADS / f"{A.head}.pt")
    pre = np.stack([r["pre"] for r in recs]).astype(np.float64)
    off = RC.compose(h, blob, zf["z_tac"][jf[m]], cr["x9"][jc[m]], cr["prior8"][jc[m]], zf["traj_own"][jf[m]])
    tr = np.stack([r["traj"] for r in recs])[m].astype(np.float64)
    rec = {"tool": "a3_keq_roll.py --kit fresh (PRE-TEST on the calibration split; NOT a registered control)", "n_records": len(recs), "pose_clock": POSE,
           "n_on_calib_rows": int(m.sum()), "counters": ST["c"], "head": ST["attached"], "subset": ST["subset"],
           "pre_vs_fresh_bank_traj_own_max_abs_m": float(np.abs(pre[jf >= 0] - zf["traj_own"][jf[jf >= 0]].astype(np.float64)).max()),
           "pre_vs_bank_n_windows_differ": int((np.abs(pre[jf >= 0] - zf["traj_own"][jf[jf >= 0]].astype(np.float64)).max(axis=(1, 2)) > 0).sum()),
           "pre_vs_bank_n_windows_differ_gt_1e-4": int((np.abs(pre[jf >= 0] - zf["traj_own"][jf[jf >= 0]].astype(np.float64)).max(axis=(1, 2)) > 1e-4).sum()),
           "z_in_forward_vs_fresh_bank_max_abs": float(np.abs(np.stack([r["z"] for r in recs])[jf >= 0].astype(np.float64)
                                                              - zf["z_tac"][jf[jf >= 0]].astype(np.float64)).max()),
           "x_in_forward_vs_calib_x9_max_abs": float(np.abs(np.stack([r["x"] for r in recs])[m].astype(np.float64) - cr["x9"][jc[m]].astype(np.float64)).max()),
           "s_prior_in_forward_vs_offline_max_abs_m": float(np.abs(np.stack([r["s_p"] for r in recs])[m].astype(np.float64) - off["s_prior"].astype(np.float64)).max()),
           "decision_agreement": float((np.array([r["dec"] for r in recs])[m] == off["decision"]).mean()),
           "traj_in_forward_vs_offline_max_abs_m": float(np.abs(tr - off["traj"].astype(np.float64)).max()),
           "wall_s": round(time.time() - t0, 1)}
    RC.jdump(rec, OUTD / "KEQ_FRESH_PRETEST.json")
    print(json.dumps(rec, default=str)[:1500], flush=True)
    sys.exit(0)
# ---- K-OFF / K-EQ against the banked arrays and the offline package
zb = np.load(RC.EVAL_BANK / f"ztac_eval_s{A.seed}.npz", allow_pickle=False)
W = int(json.loads(str(zb["meta_json"]))["window_W"])
bk = {(int(c), int(t)): j for j, (c, t) in enumerate(zip(zb["e_i"], zb["t"]))}
jb = np.array([bk.get((r["e_i"], r["t"]), -1) for r in recs])
pk = np.load(RC.SCRATCH / "arms" / A.head / f"R9_POS_step48852ema_pf_s{A.seed}.npz", allow_pickle=False)
pkk = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(pk["S2_clip"], pk["S2_ws"]))}
jp = np.array([pkk.get((r["e_i"], r["t"] + W - 1), -1) for r in recs])
ok = bool((jb >= 0).all() and (jp >= 0).all() and len(recs) >= A.min_windows)
traj = np.stack([r["traj"] for r in recs]).astype(np.float64)
pre = np.stack([r["pre"] for r in recs]).astype(np.float64)
dz = float(np.abs(np.stack([r["z"] for r in recs]).astype(np.float64) - zb["z_tac"][jb].astype(np.float64)).max()) if ok else None
koff_d = float(np.abs(pre - zb["traj_own"][jb].astype(np.float64)).max()) if ok else None
keq_d = np.abs(traj - pk["S2_pos8"][jp].astype(np.float64)).max(axis=(1, 2)) if ok else None
dec_agree = float((np.array([r["dec"] for r in recs]) == pk["S2_decision"][jp]).mean()) if ok else None
st_d = float(np.abs(np.stack([r["s_t"] for r in recs]).astype(np.float64) - pk["S2_s_target"][jp].astype(np.float64)).max()) if ok else None
I = np.load(RC.SCRATCH / "arms" / f"EVAL_INPUTS_step48852ema_pf_s{A.seed}.npz", allow_pickle=False)
ik = {(int(c), int(w)): j for j, (c, w) in enumerate(zip(I["S2_clip"], I["S2_ws"]))}
ji = np.array([ik.get((r["e_i"], r["t"] + W - 1), -1) for r in recs])
sp_d = float(np.abs(np.stack([r["s_p"] for r in recs]).astype(np.float64)
                    - RC.r9p().arc_at_slots(torch.from_numpy(I["S2_prior8"][ji].astype(np.float32))).numpy().astype(np.float64)).max()) if ok else None
rec = {"tool": "a3_keq_roll.py", "script_sha256": RC.sha256_file(__file__), "head": ST["attached"], "pose_clock": POSE, "subset": ST["subset"], "counters": ST["c"],
       "n_records": len(recs), "tanitad_file": str(TAN), "launch_tree": str(LT), "seed": A.seed,
       "K-OFF": {"rule": "in-forward traj_preretime == the eval bank's traj_own (L1 os-only roll, seed 1) on every window", "max_abs_m": koff_d,
                 "z_tac_in_forward_vs_bank_max_abs": dz, "PASS": bool(ok and koff_d == 0.0)},
       "K-EQ": {"rule": "in-forward R9 plan == the offline composition (same head file, same inputs) on >= 500 windows, max |d| <= 1e-5 m",
                "n_windows": int(len(recs)), "max_abs_m": (float(keq_d.max()) if ok else None),
                "n_windows_over_1e-5": (int((keq_d > 1e-5).sum()) if ok else None), "decision_agreement": dec_agree,
                "s_target_max_abs_m": st_d, "s_prior_max_abs_m": sp_d,
                "PASS": bool(ok and float(keq_d.max()) <= 1e-5)},
       "wall_s": round(time.time() - t0, 1), "generated": time.strftime("%FT%T%z")}
RC.jdump(rec, RC.RESULT / "R9PF_KEQ.json")
print(f"[keq] K-OFF {rec['K-OFF']['PASS']} ({koff_d}) K-EQ {rec['K-EQ']['PASS']} ({rec['K-EQ']['max_abs_m']})  ZZKEQDONEZZ", flush=True)
