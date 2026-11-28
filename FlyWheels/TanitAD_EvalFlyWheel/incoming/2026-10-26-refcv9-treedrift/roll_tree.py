"""roll_tree.py -- EvalFlyWheel TREEDRIFT bisect rig (2026-10-26).  NEW file, F:/treedrift only.

ONE inference-seed roll of refcv8 EMA-48,852 on the 70 calibration windows (kit_fresh windows_s0.json, first clips in sha12 order until
>= 60 windows), with the planner code of an ARBITRARY tree (a `git archive` of one commit), NO R9 head.  It is a trimmed, tree-parametrised copy of
`refcv9_r9pf/code/a3_keq_roll.py --kit fresh --no-head` (that script is imported by nobody and left unchanged).

Extra instrumentation (all read-only on the model path unless --rng-inject):
  * a SEED LOG of every torch.manual_seed / torch.cuda.manual_seed(_all) / torch.cuda.set_rng_state call (caller file:line, whether CUDA was
    initialised, the CUDA RNG state hash AFTER the call when CUDA is initialised -- never initialises CUDA itself);
  * per window: the CPU + CUDA global RNG state read IMMEDIATELY BEFORE the forward (the planner's sampler noise is a pure function of it:
    refc._sample draws the base eps from the global CUDA stream and the extras from a generator seeded off blake2b(cuda rng state | seed | site));
  * per window: a checksum of the FIRST CUDA torch.randn draw inside the forward (the base noise);
  * --rng-capture / --rng-inject: write the per-window pre-forward RNG states, or SET them before each forward (replay one tree's noise on another).
    That is the RNG-only discriminator: if the paths agree under identical RNG state, the deterministic computation is the same.

    python -X utf8 roll_tree.py --tree F:/treedrift/<commit> --seed 0 --out F:/treedrift/out/<name> [--rng-capture f.npz | --rng-inject f.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--min-windows", type=int, default=60)
ap.add_argument("--rng-capture", default=None)
ap.add_argument("--rng-inject", default=None, help="npz from --rng-capture; sets cpu+cuda RNG state before each forward")
ap.add_argument("--inject-cuda-only", action="store_true")
ap.add_argument("--tag", default="")
A = ap.parse_args()

LT = Path(A.tree).resolve()
OUTD = Path(A.out)
OUTD.mkdir(parents=True, exist_ok=True)
os.environ["R8_EVAL_KIT"] = "F:/l1c/kit_fresh"          # BEFORE r8_repoint is imported (l1c_ztac_roll.py's fresh plumbing)
os.environ["R8_LAUNCH_TREE"] = str(LT)
os.environ.setdefault("R8_NO_POSE_SYNC", "1")
for p in (str(LT / "taniteval"), str(LT / "stack")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

CKPT = "E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt"
PF_CODE = Path("C:/Users/Admin/qland/hold_r8pf/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-20-refcv8-perframe-final/code")
R8_CODE = Path("C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery/code")
EVAL_REMAP = Path("C:/Users/Admin/refcv8_pf/refcv9_l1c/code/remap_l1c_fresh.json")
WIN = Path("F:/l1c/kit_fresh/windows_s0.json")
V9_MD5 = Path("F:/l1c/kit_fresh/data/v9/v9_labels_eval.md5").read_text(encoding="utf-8").strip()

sys.path.insert(0, str(R8_CODE))
import r8_repoint  # noqa: E402

if Path(r8_repoint.LAUNCH_TREE).resolve() != LT:
    raise SystemExit(f"[tdrift] r8_repoint.LAUNCH_TREE {r8_repoint.LAUNCH_TREE} != {LT}")
r8_repoint.REMAP_OVERRIDES = EVAL_REMAP
sys.argv = [str(PF_CODE / "roll_seed_r8_pf.py"), "--ckpt", CKPT, "--config", "E:/refcv8_eval_kit/ckpt/config.json", "--seed", str(A.seed),
            "--dump-dir", str(OUTD / "dump"), "--windows-json", str(WIN), "--device", "cuda", "--out-json", str(OUTD / "roll.json"), "--os-only"]
sys.path.insert(0, str(PF_CODE))

import torch  # noqa: E402

# ---------------------------------------------------------------------------------------------------------------------------------
# SEED LOG -- installed BEFORE the roll imports anything that could seed.  Never initialises CUDA.
SEEDLOG: list[dict] = []


def _cuda_state_hash() -> str | None:
    try:
        if torch.cuda.is_initialized():
            return hashlib.sha256(torch.cuda.get_rng_state(0).numpy().tobytes()).hexdigest()[:12]
    except Exception as ex:  # noqa: BLE001
        return f"ERR:{type(ex).__name__}"
    return None


def _caller() -> str:
    st = traceback.extract_stack(limit=6)[:-2]
    keep = [f"{Path(f.filename).name}:{f.lineno}" for f in st[-3:]]
    return " <- ".join(reversed(keep))


def _wrap(mod, name, label, argfmt):
    orig = getattr(mod, name)

    def w(*a, **k):
        init_before = torch.cuda.is_initialized()
        r = orig(*a, **k)
        SEEDLOG.append({"i": len(SEEDLOG), "call": label, "arg": argfmt(a, k), "cuda_init_before": init_before,
                        "cuda_init_after": torch.cuda.is_initialized(), "cuda_state_after": _cuda_state_hash(), "caller": _caller()})
        return r
    w.__wrapped_orig__ = orig
    setattr(mod, name, w)


_wrap(torch, "manual_seed", "torch.manual_seed", lambda a, k: a[0] if a else k.get("seed"))
_wrap(torch.cuda, "manual_seed_all", "cuda.manual_seed_all", lambda a, k: a[0] if a else k.get("seed"))
_wrap(torch.cuda, "manual_seed", "cuda.manual_seed", lambda a, k: a[0] if a else k.get("seed"))
_wrap(torch.cuda, "set_rng_state", "cuda.set_rng_state", lambda a, k: "state")
_wrap(torch.cuda, "set_rng_state_all", "cuda.set_rng_state_all", lambda a, k: "states")

# the first CUDA torch.randn draw inside a forward (base noise checksum)
_RANDN = {"armed": False, "first": None, "n_cuda_draws": 0}
_orig_randn = torch.randn


def _randn_w(*a, **k):
    r = _orig_randn(*a, **k)
    if _RANDN["armed"] and r.is_cuda:
        _RANDN["n_cuda_draws"] += 1
        if _RANDN["first"] is None:
            _RANDN["first"] = [list(r.shape), float(r.double().sum().item()), float(r.double().flatten()[0].item())]
    return r


torch.randn = _randn_w

# the battery child's imports (roll_seed_r8_pf) -- AFTER the seed log is armed, in the ORIGINAL a3_keq_roll order
import roll_seed_r8_pf as PF  # noqa: E402
import tanitad  # noqa: E402

TAN = Path(tanitad.__file__).resolve()
if LT not in TAN.parents:
    raise SystemExit(f"[tdrift] tanitad imported from {TAN}, not {LT}")
import run_battery_r7 as RB  # noqa: E402
import tanitad.eval.refcv7_loader as _LP  # noqa: E402

POSE: dict = {}
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
        raise SystemExit("[tdrift] R8_NO_POSE_SYNC=1 but the eval dataset HAS the pose shift attached")
    POSE.update(policy="uncorrected clock: sidecar blanked, shift NOT attached" if no_sync else "corrected", argv_names_sidecar=bool(saved),
                shift_attached=attached)
    return res


_bed_pose._r8_patched = True
_bed_pose._r8_orig = getattr(_cur_bed, "_r8_orig", _cur_bed)
_LP.build_eval_dataset = _bed_pose

_orig_bed = _LP.build_eval_dataset


def _bed(model, cfg, args, config, **kw):
    if os.path.basename(str(getattr(args, "r8_v9_labels_eval", ""))) == os.path.basename(
            str(json.load(open(EVAL_REMAP, encoding="utf-8"))["--r8-v9-labels-eval"])):
        args.r8_v9_eval_md5 = V9_MD5
    return _orig_bed(model, cfg, args, config, **kw)


_bed._r8_patched = True
_bed._r8_orig = getattr(_orig_bed, "_r8_orig", _orig_bed)
_LP.build_eval_dataset = _bed

RR = PF.RR
ST: dict = {"armed": False, "caps": [], "records": [], "c": {"calls": 0, "navnull": 0, "batch_gt1": 0}}


def sha12(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


_orig_roll_one = RB.roll_one


def roll_one_subset(ckpt, config, seed, dump_dir, windows, *a, **k):
    keys = sorted(windows.keys(), key=sha12)
    keep, n = [], 0
    for key in keys:
        keep.append(key)
        n += len(windows[key])
        if n >= A.min_windows:
            break
    sub = {key: windows[key] for key in keep}
    ST["subset"] = {"n_clips": len(sub), "n_windows_listed": int(sum(len(v) for v in sub.values())), "clip_sha12": [sha12(x) for x in keep]}
    return _orig_roll_one(ckpt, config, seed, dump_dir, sub, *a, **k)


RB.roll_one = roll_one_subset
_Base = RR.Refcv6Proxy

INJ = None
if A.rng_inject:
    zi = np.load(A.rng_inject, allow_pickle=False)
    INJ = {(int(c), int(t)): j for j, (c, t) in enumerate(zip(zi["e_i"], zi["t"]))}
    INJ_CUDA, INJ_CPU = zi["cuda_state"], zi["cpu_state"]
CAP: list[dict] = []


def attach(model):
    mods = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "lat_head_tac"]
    if len(mods) != 1:
        raise SystemExit(f"[tdrift] expected one lat_head_tac, found {len(mods)}")

    def pre_hook(_m, inp):
        if ST["armed"]:
            ST["caps"].append(inp[0].detach().float().cpu().numpy().copy())
        return None
    mods[0][1].register_forward_pre_hook(pre_hook)
    ST["attached"] = True


_orig_build = RB.L.build_model


def build_and_attach(*a, **k):
    res = _orig_build(*a, **k)
    attach(res[0])
    SEEDLOG.append({"i": len(SEEDLOG), "call": "BUILD_DONE", "cuda_init_after": torch.cuda.is_initialized(), "cuda_state_after": _cuda_state_hash(),
                    "caller": ""})
    return res


RB.L.build_model = build_and_attach


class TProxy(_Base):
    def __call__(self, frames, nav_cmd=None, v0=None, steps=0, ego_state=None, **kw):
        d = self.__dict__
        if not ST.get("attached"):
            raise RuntimeError("[tdrift] a forward ran before attach")
        if nav_cmd is None:
            ST["c"]["navnull"] += 1
            return _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        n = int(frames.shape[0])
        e_i, t = (int(x) for x in d["ctx"].index)
        dev = frames.device
        if INJ is not None:
            j = INJ[(e_i, t)]
            if not A.inject_cuda_only:
                torch.set_rng_state(torch.from_numpy(INJ_CPU[j].copy()))
            torch.cuda.set_rng_state(torch.from_numpy(INJ_CUDA[j].copy()), dev)
        cpu_st = torch.get_rng_state().numpy().copy()
        cuda_st = torch.cuda.get_rng_state(dev).numpy().copy()
        ST["armed"], ST["caps"] = True, []
        _RANDN["armed"], _RANDN["first"], _RANDN["n_cuda_draws"] = True, None, 0
        try:
            out = _Base.__call__(self, frames, nav_cmd=nav_cmd, v0=v0, steps=steps, ego_state=ego_state, **dict(kw))
        finally:
            ST["armed"] = False
            _RANDN["armed"] = False
        ST["c"]["calls"] += 1
        if n != 1:
            ST["c"]["batch_gt1"] += 1
        conds = list(d["conds_fed"][:n])
        r = conds.index("nav_true")
        f = lambda k: out[k][r].detach().float().cpu().numpy()  # noqa: E731
        after = torch.cuda.get_rng_state(dev).numpy().copy()
        ST["records"].append({"e_i": e_i, "t": t, "traj": f("traj"), "z": ST["caps"][0][r].astype(np.float32), "cpu": cpu_st, "cuda": cuda_st,
                              "cuda_after": after, "randn_first": _RANDN["first"], "n_draws": _RANDN["n_cuda_draws"]})
        return out


RR.Refcv6Proxy = TProxy
t0 = time.time()
PF.RS.main()
recs = ST["records"]
np.savez(OUTD / "records.npz", e_i=np.array([r["e_i"] for r in recs]), t=np.array([r["t"] for r in recs]), traj=np.stack([r["traj"] for r in recs]),
         z=np.stack([r["z"] for r in recs]), cpu_state=np.stack([r["cpu"] for r in recs]), cuda_state=np.stack([r["cuda"] for r in recs]),
         cuda_state_after=np.stack([r["cuda_after"] for r in recs]))
if A.rng_capture:
    np.savez(A.rng_capture, e_i=np.array([r["e_i"] for r in recs]), t=np.array([r["t"] for r in recs]),
             cpu_state=np.stack([r["cpu"] for r in recs]), cuda_state=np.stack([r["cuda"] for r in recs]))
# per-window ADE (2 s grid, GT 'g' vs the selected plan 'os'), from the dump the battery wrote
ade = []
for f in sorted((OUTD / "dump").glob("ep*.npz")):
    z = np.load(f, allow_pickle=False)
    ade.append(np.linalg.norm(z["os"].astype(np.float64) - z["g"].astype(np.float64), axis=-1).mean(-1))
ade = np.concatenate(ade)
np.save(OUTD / "ade_per_window.npy", ade)
meta = {"tool": "roll_tree.py", "tree": str(LT), "tanitad_file": str(TAN), "seed": A.seed, "tag": A.tag, "rng_inject": A.rng_inject,
        "inject_cuda_only": A.inject_cuda_only, "rng_capture": A.rng_capture, "pose_clock": POSE, "subset": ST.get("subset"), "counters": ST["c"],
        "n_records": len(recs), "mean_ade_2s_os": float(ade.mean()), "n_ade": int(ade.size), "wall_s": round(time.time() - t0, 1),
        "randn_first_win0": recs[0]["randn_first"], "n_cuda_randn_draws_win0": recs[0]["n_draws"],
        "cuda_state_before_win0": hashlib.sha256(recs[0]["cuda"].tobytes()).hexdigest()[:12],
        "cuda_state_before_win1": hashlib.sha256(recs[1]["cuda"].tobytes()).hexdigest()[:12],
        "seedlog": SEEDLOG}
(OUTD / "META.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
print(f"[tdrift] done {OUTD.name}: n={len(recs)} mean_ade_2s={ade.mean():.4f} cuda_before_w0={meta['cuda_state_before_win0']} ZZTDRIFTDONEZZ", flush=True)
