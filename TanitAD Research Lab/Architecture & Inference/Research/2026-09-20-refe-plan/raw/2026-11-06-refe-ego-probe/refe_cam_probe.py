#!/usr/bin/env python3
"""REFe ego-input reliance probe (T0, open loop).  See PREREG.md in this directory.

ONE scenario + ONE image set per token, then EIGHT arms through the planner's OWN `infer` (batch 1): A0 real | A0r replicate | AX0 | AXP | AXM | DEAD | GREY | SWAP.
Perturbations are monkeypatches made HERE: the ego vector at `planner._ego_vec` (what `infer` calls), the image tensor handed to `infer`, nothing in the package is edited.
Structure copied from PK/eval/aux_navtest_forward.py (refe_navtest_seam loop, shared dev-box GPU lock, O_EXCL).  LAST ACTION: write DONE or FAILED in --out.

    python refe_ego_probe.py --out <dir> [--tokens <subset.json>] [--limit-tokens N] [--smoke] [--no-gpu-lock]
Exit: 0 ok | 2 control FAILED (results are still written, flagged) | 4 refused | 5 GPU busy.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np

PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
REFE_DIR, EVAL_DIR = PK + "/refe", PK + "/eval"
DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
ROOT = "E:/Projects/TanitAD"
CKPT = ROOT + "/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
CKPT_SHA256_PREFIX = "8b4dba4818b5cc96"
FRAMES = ROOT + "/data/refe_navtest/frames"
DB_DIR = ROOT + "/data/nuplan/nuplan-v1.1/splits/test"
C3_DIR = PK + "/raw/2026-10-11-c3-selection-headroom"
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
WAIT_IDLE = "E:/Projects/TanitAD-artifacts/_rlfan_work/wait_idle.py"
GPU_MAX_USED_MIB = 3500
OWNER = "refe_ego_probe"
RULE = "navsim_v1_lane"
ARMS = ("A0", "A0r", "AX0", "AXP", "AXM", "DEAD", "GREY", "SWAP", "F0ONLY", "NOF0")   # + per-camera ablations (2026-11-06 MM): F0ONLY = F0 real, L0/R0/B0 grey; NOF0 = F0 grey, L0/R0/B0 real
TS = (1.0, 2.0, 4.0)
DT = 0.2

# ---- environment: the same as eval_checkpoint.env_driverl(), D: -> E: (set BEFORE the package modules are imported) ----------------------------------------------------------
os.environ.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": ROOT + "/data/nuplan", "NUPLAN_MAPS_ROOT": ROOT + "/data/nuplan-maps/nuplan-maps-v1.0",
                   "REFE_NUPLAN_DB_ROOT": ROOT + "/data/nuplan/nuplan-v1.1/splits",
                   "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": DZ + "/nuplan-devkit",
                   "DRIVERL_EVAL_CONFIG_PATH": DZ + "/release/configs/driverl_teacher.yaml",
                   "DRIVERL_EVAL_CHECKPOINT_PATH": DZ + "/release/checkpoints/checkpoint_2400.pt",
                   "REFE_BACKBONE_ROOT": ROOT + "/data/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
for _p in (DZ + "/src", DZ + "/nuplan-devkit", REFE_DIR, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def navsim_root():
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    if r:
        return r
    for c in ("E:/Archive/devbox-C/navsim", "F:/Archive/devbox-C/navsim", "D:/Archive/devbox-C/navsim"):
        if os.path.isdir(c):
            return c
    raise RuntimeError("no navsim archive root (set TANITAD_NAVSIM_ROOT)")


class GpuBusy(RuntimeError):
    pass


# ======================================================================================================================
# the shared dev-box GPU lock (PK/eval/aux_navtest_forward.py's protocol, own OWNER marker)
# ======================================================================================================================
def gpu_used_mib() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True)
    try:
        return int(r.stdout.decode("utf-8", "replace").strip().splitlines()[0])
    except (ValueError, IndexError):
        return -1


def pid_alive(pid) -> bool:
    r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True)
    return f'"{pid}"' in r.stdout.decode("utf-8", "replace")


def acquire_gpu_lock(wait_timeout_s: float, settle_s: float = 4.0) -> str:
    """Fail CLOSED.  The neighbour's idle waiter must exit 0, the lock is created with O_EXCL; a lock whose holder is alive or foreign is NEVER taken or removed."""
    try:
        r = subprocess.run([sys.executable, WAIT_IDLE], timeout=wait_timeout_s)
    except subprocess.TimeoutExpired:
        raise GpuBusy(f"wait_idle.py did not report an idle GPU within {wait_timeout_s:.0f} s")
    if r.returncode != 0:
        raise GpuBusy(f"wait_idle.py exited {r.returncode}")
    body = json.dumps({"pid": os.getpid(), "owner": OWNER, "t": time.time()})
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    for attempt in (1, 2):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                holder = json.loads(open(LOCK, encoding="utf-8").read())
                age = time.time() - os.path.getmtime(LOCK)
            except (OSError, ValueError):
                raise GpuBusy(f"{LOCK} exists and is unreadable")
            if not isinstance(holder, dict) or holder.get("owner") != OWNER:
                raise GpuBusy(f"{LOCK} is held by a FOREIGN lock (age {age:.0f} s): never removed: {holder}")
            if "pid" not in holder or pid_alive(holder["pid"]):
                raise GpuBusy(f"{LOCK} is held by {holder}")
            if attempt == 2 or age <= 600:
                raise GpuBusy(f"{LOCK} holder pid {holder['pid']} is gone but the lock is only {age:.0f} s old")
            os.remove(LOCK)
    else:
        raise GpuBusy("lock not acquired")
    os.write(fd, body.encode("utf-8"))
    os.close(fd)
    time.sleep(settle_s)
    used = gpu_used_mib()
    if not 0 <= used <= GPU_MAX_USED_MIB:
        release_gpu_lock(body)
        raise GpuBusy(f"memory.used {used} MiB > {GPU_MAX_USED_MIB} after taking the lock")
    return body


def release_gpu_lock(body: str) -> None:
    try:
        cur = json.loads(open(LOCK, encoding="utf-8").read())
        mine = json.loads(body)
        if cur.get("owner") == mine.get("owner") and cur.get("pid") == mine["pid"]:
            os.remove(LOCK)
    except (OSError, ValueError, AttributeError):
        pass


# ======================================================================================================================
# geometry helpers (position only; the analysis re-derives everything from the stored 20-pose arrays)
# ======================================================================================================================
def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def step_speed(poses: np.ndarray, ts=TS) -> np.ndarray:
    """poses [..., 20, >=2] at 0.2 s; one-step backward finite difference |p_i - p_{i-1}|/0.2, p_0 = origin; returns [..., len(ts)] at i = T/0.2."""
    xy = np.asarray(poses, np.float64)[..., :2]
    prev = np.concatenate([np.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], axis=-2)
    v = np.linalg.norm(xy - prev, axis=-1) / DT
    return v[..., [int(round(t / DT)) - 1 for t in ts]]


def gt_future(sc, ego) -> dict:
    """The logged future on the scenario's 20 states: ego-frame poses, TRUE timestamps (s from t0) and state speeds.  NaN rows if the scenario has < 20."""
    f20 = list(sc.get_ego_future_trajectory(0, 4.0, 20))[:20]
    out = {"gt_valid": len(f20) == 20, "gt_pose20": np.full((20, 3), np.nan), "gt_t20": np.full(20, np.nan), "gt_speed20": np.full(20, np.nan)}
    if not out["gt_valid"]:
        return out
    px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    for i, f in enumerate(f20):
        out["gt_pose20"][i] = [(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s, (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c, wrap(f.rear_axle.heading - pyaw)]
        out["gt_t20"][i] = (int(f.time_point.time_us) - int(ego.time_point.time_us)) / 1e6
        v = f.dynamic_car_state.rear_axle_velocity_2d
        out["gt_speed20"][i] = math.hypot(float(v.x), float(v.y))
    return out


# ======================================================================================================================
# trunk memo: ego-only arms reuse A0's image-trunk output iff the input tensor is bit-identical (see PREREG C4)
# ======================================================================================================================
def make_trunk_memo(torch, inner):
    class TrunkMemo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner, self.bypass, self.last_in, self.last_out = inner, False, None, None
            self.hits = self.misses = self.bypasses = 0
            self.grey_in = self.grey_out = None
            self.grey_hits = self.grey_checks = self.grey_check_equal = 0
            self.grey_verify_every = 25     # every 25th grey hit is recomputed fresh and compared bit-for-bit with the cached grey output (C4b)
            self.ref_equal = None          # set when a bypassed (fresh) call is compared with the stored A0 output

        def forward(self, x):
            if self.bypass:
                self.bypasses += 1
                out = self.inner(x)
                if self.last_in is not None and self.last_in.shape == x.shape and torch.equal(self.last_in, x):
                    self.ref_equal = bool(torch.equal(out, self.last_out))
                return out
            if self.grey_in is not None and self.grey_in.shape == x.shape and torch.equal(self.grey_in, x):
                self.grey_hits += 1
                if self.grey_hits % self.grey_verify_every == 0:
                    out = self.inner(x)
                    self.grey_checks += 1
                    self.grey_check_equal += int(torch.equal(out, self.grey_out))
                    return out
                return self.grey_out.clone()
            if self.last_in is not None and self.last_in.shape == x.shape and torch.equal(self.last_in, x):
                self.hits += 1
                return self.last_out.clone()
            self.misses += 1
            out = self.inner(x)
            if bool((x == x.flatten()[0]).all()):          # a constant (grey) image: its trunk output is the same for every token -> own slot
                self.grey_in, self.grey_out = x.clone(), out.clone()
            else:
                self.last_in, self.last_out = x.clone(), out.clone()
            return out
    return TrunkMemo()


def main_run(a) -> int:
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "run.log")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def status(**kw):
        kw["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        kw["pid"] = os.getpid()
        json.dump(kw, open(os.path.join(a.out, "STATUS.json"), "w"), indent=1)

    # ---- inputs ---------------------------------------------------------------------------------------------------------------------------------------
    tok_doc = json.load(open(a.tokens, encoding="utf-8"))
    tokens = [str(t) for t in (tok_doc["tokens"] if isinstance(tok_doc, dict) else tok_doc)]
    token_log, stratum = tok_doc["token_log"], tok_doc.get("stratum", {})
    tab = np.load(a.table, allow_pickle=False)
    tindex = {str(t): i for i, t in enumerate(tab["token"])}
    missing = [t for t in tokens if t not in tindex]
    if missing:
        say(f"REFUSED: {len(missing)} tokens not in table.npz")
        return 4
    if a.limit_tokens:
        by = {}
        for t in tokens:
            by.setdefault(token_log[t], []).append(t)
        keep, n = set(), 0
        for lg in sorted(by, key=lambda l: (-len(by[l]), l)):          # whole logs, biggest first, as aux_navtest_forward.select_tokens
            for t in by[lg]:
                if n < a.limit_tokens:
                    keep.add(t)
                    n += 1
        tokens = [t for t in tokens if t in keep]
    logs = sorted(set(token_log[t] for t in tokens))
    toks_by_log = {lg: sorted(t for t in tokens if token_log[t] == lg) for lg in logs}
    if len(logs) < 2:
        say("REFUSED: SWAP needs >= 2 logs")
        return 4
    h = hashlib.sha256()
    with open(a.ckpt, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    ck_sha = h.hexdigest()
    if not ck_sha.startswith(CKPT_SHA256_PREFIX):
        say(f"REFUSED: checkpoint sha256 {ck_sha[:16]} does not start with {CKPT_SHA256_PREFIX}")
        return 4
    exp = json.load(gzip.open(os.path.join(navsim_root(), "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"), "rt", encoding="utf-8"))["tokens"]
    say(f"tokens {len(tokens)} over {len(logs)} logs; ckpt sha256 {ck_sha[:16]}; arms {ARMS}")
    status(stage="gpu-lock", n_tokens=len(tokens), n_logs=len(logs))

    # ---- GPU lock, then the heavy imports -----------------------------------------------------------------------------------------------------------------
    lock_body = None
    if not a.no_gpu_lock and a.device == "cuda":
        try:
            lock_body = acquire_gpu_lock(a.wait_idle_timeout)
        except GpuBusy as e:
            say(f"GPU BUSY (nothing loaded, nothing written): {e}")
            return 5
    try:
        return _run_locked(a, say, status, tokens, token_log, stratum, logs, toks_by_log, tab, tindex, exp, ck_sha, t_start)
    finally:
        if lock_body:
            release_gpu_lock(lock_body)


def _run_locked(a, say, status, tokens, token_log, stratum, logs, toks_by_log, tab, tindex, exp, ck_sha, t_start) -> int:
    import torch
    import navtrain_scenarios as NS
    from planner import REFePlanner
    import refe_navtest_seam as SEAM
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    planner = REFePlanner(checkpoint=a.ckpt, images_root=FRAMES, db_dir=a.db_dir, backbone="vitl16", device=dev, select="best", rule=RULE)
    memo = make_trunk_memo(torch, planner.model.backbone)
    planner.model.backbone = memo
    memo.grey_verify_every = a.grey_verify_every
    cmd_mode = getattr(planner.model.cfg, "nav_mode", "goal") == "command"
    say(f"planner ready: device {planner.device} nav_mode {'command' if cmd_mode else 'goal'} rule {planner.rule} goal_fix {planner.goal_fix} repair_last_heading {planner.repair_last_heading}")

    # ---- the ego-vector monkeypatch (the planner builds the vector in `_ego_vec`, which `infer` calls) --------------------------------------------------
    cur = {"ego_fn": None}
    orig_ego_vec = planner._ego_vec

    def patched_ego_vec(ego):
        v = orig_ego_vec(ego)
        fn = cur["ego_fn"]
        return v if fn is None else fn(v.clone())
    planner._ego_vec = patched_ego_vec

    def e_set(**slots):
        idx = {"ax": 2, "ay": 3, "yaw": 4, "steer": 5}

        def fn(v):
            for k, x in slots.items():
                v[:, idx[k]] = x
            return v
        return fn
    EGO_FN = {"A0": None, "A0r": None, "AX0": e_set(ax=0.0, ay=0.0), "AXP": e_set(ax=0.5), "AXM": e_set(ax=-0.5), "DEAD": e_set(yaw=0.3, steer=0.2), "GREY": None, "SWAP": None, "F0ONLY": None, "NOF0": None}
    EXPECT = {"AX0": {2: 0.0, 3: 0.0}, "AXP": {2: 0.5}, "AXM": {2: -0.5}, "DEAD": {4: 0.3, 5: 0.2}}

    # ---- resume ----------------------------------------------------------------------------------------------------------------------------------------------
    part_p = os.path.join(a.out, "partial.pkl")
    recs = pickle.load(open(part_p, "rb")) if (a.resume and os.path.exists(part_p)) else {}
    if recs:
        say(f"RESUME: {len(recs)} tokens already done")

    def to_u8(img):
        u8 = (img * 255.0).round().to(torch.uint8)
        return u8 if torch.equal(u8.float() / 255.0, img) else None

    def from_u8(u8):
        return u8.float() / 255.0

    def build(lg, lt):
        """Yield (tok, sc, ego, img) for the tokens of one log, the seam loop up to and including `_image_for`."""
        db = os.path.join(a.db_dir, f"{lg}.db")
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                raise RuntimeError(f"token {tok}: no frames ({planner.frames.miss_reason})")
            yield tok, sc, ego, img

    prev_imgs, prev_log = {}, None          # uint8 images (or float32 if the uint8 round trip is not exact) of the previously processed log's selected tokens

    def donor_images(lg):
        """Images of the DONOR log (= the previous log in sorted order, cyclic), from memory when it was just processed, else rebuilt (first log / resume)."""
        j = logs.index(lg)
        dlg = logs[j - 1]
        if prev_log == dlg and all(t in prev_imgs for t in toks_by_log[dlg]):
            return dlg, prev_imgs
        say(f"  donor images for {lg} rebuilt from {dlg} ({len(toks_by_log[dlg])} tokens, no model)")
        d = {}
        got = set()
        for tok, _sc, _ego, img in build(dlg, toks_by_log[dlg]):
            u8 = to_u8(img)
            d[tok] = u8 if u8 is not None else img
            got.add(tok)
        if got != set(toks_by_log[dlg]):
            raise RuntimeError(f"donor log {dlg}: scenario builder skipped {len(set(toks_by_log[dlg]) - got)} tokens")
        return dlg, d

    def run_arm(arm, ego, img_in):
        cur["ego_fn"] = EGO_FN[arm]
        memo.bypass = (arm == "A0r")
        t0 = time.time()
        traj, score, k = planner.infer(ego, img_in)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        ego_fed = planner.last_inputs[0].detach().cpu().numpy().astype(np.float64)[0]
        return traj.float().cpu().numpy(), score.float().cpu().numpy(), int(k), ego_fed, dt

    def finish_arm_record(tr, sc_, k, ego_fed, dt, img_in):
        im = img_in.detach().double()
        return {"pick": k, "traj_pick": tr[k].astype(np.float32), "logit_pick": sc_[k].astype(np.float32), "speed_all": step_speed(tr).astype(np.float32),
                "ego_fed": ego_fed, "img_mean": float(im.mean()), "img_std": float(im.std()), "infer_s": dt}

    t_loop = time.time()
    n_done_start = len(recs)
    n_seen = n_done_start
    c3_fail = []
    for li, lg in enumerate(logs):
        lt = [t for t in toks_by_log[lg] if t not in recs]
        if not lt:
            continue
        dlg, dimgs = donor_images(lg)
        dtoks = toks_by_log[dlg]
        order = {t: i for i, t in enumerate(toks_by_log[lg])}
        cur_imgs, got = {}, set()
        t_log = time.time()
        for tok, sc, ego, img in build(lg, lt):
            got.add(tok)
            u8 = to_u8(img)
            cur_imgs[tok] = u8 if u8 is not None else img
            SEAM.set_driving_command(planner, tok, exp[tok])
            donor_tok = dtoks[order[tok] % len(dtoks)]
            dimg = dimgs[donor_tok]
            donor_img = from_u8(dimg) if dimg.dtype == torch.uint8 else dimg
            grey = torch.full_like(img, 128.0 / 255.0)
            imgs = {x: img for x in ARMS}
            imgs["GREY"], imgs["SWAP"] = grey, donor_img
            assert img.dim() == 5 and img.shape[1] == 4, f"expected [1, 4 cams, 3, H, W], got {tuple(img.shape)}"   # cams in cfg order F0, L0, R0, B0 (planner._image_for / FrameIndex.resolve)
            f0only = img.clone(); f0only[:, 1:] = 128.0 / 255.0
            nof0 = img.clone(); nof0[:, 0] = 128.0 / 255.0
            imgs["F0ONLY"], imgs["NOF0"] = f0only, nof0
            rec = {"token": tok, "log": lg, "stratum": stratum.get(tok, ""), "donor_token": donor_tok, "donor_log": dlg}
            rec.update(gt_future(sc, ego))
            h8 = np.asarray(exp[tok]["human_future_poses"], np.float64)
            rec["human8"] = h8
            raw = {}
            run_a0r = (n_seen % a.replicate_every == 0)
            n_seen += 1
            rec["a0r_ran"] = run_a0r
            memo.ref_equal = None
            for arm in ARMS:
                if arm == "A0r" and not run_a0r:
                    raw[arm], rec[arm] = raw["A0"], dict(rec["A0"])           # NOT run on this token: a copy of A0 (flagged by a0r_ran=False; the analysis ignores it)
                    continue
                tr, scr, k, ego_fed, dt = run_arm(arm, ego, imgs[arm])
                raw[arm] = (tr, scr, k)
                rec[arm] = finish_arm_record(tr, scr, k, ego_fed, dt, imgs[arm])
            a0 = rec["A0"]["ego_fed"]
            rec["ego_real"] = a0
            # ---- C1 determinism (A0r vs A0), C3 reachability, C2 identity (A0 vs table.npz) ----------------------------------------------------------------
            tr0, sc0, k0 = raw["A0"]
            tr1, sc1, k1 = raw["A0r"]
            rec["c1"] = None if not run_a0r else {"traj_equal": bool(np.array_equal(tr0, tr1)), "logit_equal": bool(np.array_equal(sc0, sc1)), "pick_equal": k0 == k1,
                         "max_abs_traj": float(np.abs(tr0.astype(np.float64) - tr1).max()), "max_abs_logit": float(np.abs(sc0.astype(np.float64) - sc1).max()),
                         "trunk_equal": memo.ref_equal}
            c3 = {}
            for arm in ARMS:
                fed = rec[arm]["ego_fed"]
                if arm in EXPECT:        # the model input is float32: compare against the float32 of the intended value
                    ok = all(fed[i] == float(np.float32(x)) for i, x in EXPECT[arm].items()) and all(fed[i] == a0[i] for i in range(7) if i not in EXPECT[arm])
                elif arm == "GREY":
                    ok = rec[arm]["img_std"] == 0.0 and abs(rec[arm]["img_mean"] - 128.0 / 255.0) < 1e-7 and bool(np.array_equal(fed, a0))
                elif arm in ("F0ONLY", "NOF0"):
                    g_idx, r_idx = ([1, 2, 3], [0]) if arm == "F0ONLY" else ([0], [1, 2, 3])
                    ok = bool((imgs[arm][:, g_idx] == 128.0 / 255.0).all()) and bool(torch.equal(imgs[arm][:, r_idx], img[:, r_idx])) and bool(np.array_equal(fed, a0))
                elif arm == "SWAP":
                    ok = bool(torch.equal(imgs["SWAP"], donor_img)) and (not torch.equal(imgs["SWAP"], img)) and bool(np.array_equal(fed, a0))
                else:
                    ok = bool(np.array_equal(fed, a0))
                c3[arm] = bool(ok)
                if not ok:
                    c3_fail.append((tok, arm))
            rec["c3"] = c3
            ti = tindex[tok]
            tr0_t = torch.from_numpy(tr0)
            props8 = np.stack([SEAM.to_navsim(planner.executed(tr0_t, j).float().cpu().numpy()) for j in range(tr0.shape[0])]).astype(np.float32)
            ref_p, ref_l, ref_k = tab["proposals"][ti], tab["logits"][ti], int(tab["pick"][ti])
            rec["c2"] = {"logit_equal": bool(np.array_equal(sc0.astype(np.float32), ref_l)), "max_abs_logit": float(np.abs(sc0.astype(np.float64) - ref_l).max()),
                         "pick_equal": k0 == ref_k, "max_abs_props8_m": float(np.abs(props8.astype(np.float64)[..., :2] - ref_p.astype(np.float64)[..., :2]).max())}
            rec["traj_all_A0"] = tr0.astype(np.float32)
            rec["logits_all_A0"] = sc0.astype(np.float32)
            recs[tok] = rec
        miss = set(lt) - got
        if miss:
            raise RuntimeError(f"{len(miss)} tokens skipped by the scenario builder in {lg} (first {sorted(miss)[0]})")
        prev_imgs, prev_log = cur_imgs, lg
        n_new = len(recs) - n_done_start
        el = time.time() - t_loop
        eta = el / max(n_new, 1) * (len(tokens) - len(recs))
        say(f"[{li + 1}/{len(logs)}] {lg}: {len(lt)} tokens in {time.time() - t_log:.0f} s | done {len(recs)}/{len(tokens)} | {el / max(n_new, 1):.2f} s/token | ETA {eta / 60:.0f} min | memo hit/miss/bypass {memo.hits}/{memo.misses}/{memo.bypasses}")
        pickle.dump(recs, open(part_p + ".tmp", "wb"), protocol=4)
        os.replace(part_p + ".tmp", part_p)
        status(stage="run", done=len(recs), n_tokens=len(tokens), eta_min=round(eta / 60, 1), last_log=lg)

    # ---- consolidate ---------------------------------------------------------------------------------------------------------------------------------------
    order_t = [t for t in tokens if t in recs]
    R = [recs[t] for t in order_t]
    out = {"token": np.array(order_t), "log": np.array([r["log"] for r in R]), "stratum": np.array([r["stratum"] for r in R]),
           "donor_token": np.array([r["donor_token"] for r in R]), "arms": np.array(ARMS), "ego_real": np.stack([r["ego_real"] for r in R]),
           "gt_valid": np.array([r["gt_valid"] for r in R]), "a0r_ran": np.array([r["a0r_ran"] for r in R]), "gt_pose20": np.stack([r["gt_pose20"] for r in R]), "gt_t20": np.stack([r["gt_t20"] for r in R]),
           "gt_speed20": np.stack([r["gt_speed20"] for r in R]), "human8": np.stack([r["human8"] for r in R]),
           "pick": np.array([[r[x]["pick"] for x in ARMS] for r in R]), "traj_pick": np.stack([np.stack([r[x]["traj_pick"] for x in ARMS]) for r in R]),
           "logit_pick": np.stack([np.stack([r[x]["logit_pick"] for x in ARMS]) for r in R]), "speed_all": np.stack([np.stack([r[x]["speed_all"] for x in ARMS]) for r in R]),
           "ego_fed": np.stack([np.stack([r[x]["ego_fed"] for x in ARMS]) for r in R]), "infer_s": np.array([[r[x]["infer_s"] for x in ARMS] for r in R]),
           "traj_all_A0": np.stack([r["traj_all_A0"] for r in R]), "logits_all_A0": np.stack([r["logits_all_A0"] for r in R])}
    np.savez_compressed(os.path.join(a.out, "results.npz.tmp.npz"), **out)
    os.replace(os.path.join(a.out, "results.npz.tmp.npz"), os.path.join(a.out, "results.npz"))
    R1 = [r for r in R if r["c1"] is not None]
    c1 = {k: [r["c1"][k] for r in R1] for k in ("traj_equal", "logit_equal", "pick_equal", "trunk_equal")}
    c2 = {k: [r["c2"][k] for r in R] for k in ("logit_equal", "pick_equal")}
    meta = {"tool": "refe_ego_probe/1", "tier": "T0", "ckpt": a.ckpt, "ckpt_sha256": ck_sha, "n_tokens": len(R), "n_logs": len(set(out["log"])), "arms": list(ARMS), "device": dev,
            "rule": RULE, "goal_fix": planner.goal_fix, "repair_last_heading": planner.repair_last_heading, "nav_mode": "command" if cmd_mode else "goal",
            "C1_determinism": {"n": len(R1), "n_tokens_total": len(R), "traj_bit_identical": int(sum(c1["traj_equal"])), "logit_bit_identical": int(sum(c1["logit_equal"])), "pick_equal": int(sum(c1["pick_equal"])),
                               "trunk_bit_identical": int(sum(bool(x) for x in c1["trunk_equal"])), "max_abs_traj_m": max(r["c1"]["max_abs_traj"] for r in R1),
                               "max_abs_logit": max(r["c1"]["max_abs_logit"] for r in R1),
                               "PASS": all(c1["traj_equal"]) and all(c1["logit_equal"]) and all(c1["pick_equal"])},
            "C2_identity_vs_table": {"n": len(R), "logit_bit_identical": int(sum(c2["logit_equal"])), "pick_equal": int(sum(c2["pick_equal"])),
                                     "max_abs_logit": max(r["c2"]["max_abs_logit"] for r in R), "max_abs_props8_m": max(r["c2"]["max_abs_props8_m"] for r in R),
                                     "PASS_strict": all(c2["logit_equal"]) and all(c2["pick_equal"]) and max(r["c2"]["max_abs_props8_m"] for r in R) <= 1e-3,
                                     "PASS_fallback": (sum(c2["pick_equal"]) / len(R) >= 0.99) and max(r["c2"]["max_abs_props8_m"] for r in R) <= 1e-3},
            "C3_reachability": {arm: {"n_ok": int(sum(r["c3"][arm] for r in R)), "n": len(R)} for arm in ARMS}, "C3_failures_first10": c3_fail[:10],
            "C4_memo": {"hits": memo.hits, "misses": memo.misses, "bypasses": memo.bypasses, "grey_hits": memo.grey_hits, "grey_fresh_checks": memo.grey_checks,
                        "grey_fresh_checks_bit_identical": memo.grey_check_equal},
            "C5_ego_slots_real": {"frac_yaw_rate_nonzero": float((out["ego_real"][:, 4] != 0).mean()), "frac_steer_nonzero": float((out["ego_real"][:, 5] != 0).mean()),
                                  "ax_pcts_1_50_99": [float(x) for x in np.percentile(out["ego_real"][:, 2], [1, 50, 99])],
                                  "ay_pcts_1_50_99": [float(x) for x in np.percentile(out["ego_real"][:, 3], [1, 50, 99])]},
            "timing": {"loop_s": round(time.time() - t_loop, 1), "wall_s": round(time.time() - t_start, 1), "s_per_token": round((time.time() - t_loop) / max(len(recs) - n_done_start, 1), 3),
                       "mean_infer_s_per_arm": {arm: float(out["infer_s"][:, i].mean()) for i, arm in enumerate(ARMS)}},
            "gt_valid_share": float(out["gt_valid"].mean())}
    meta["controls_pass"] = bool(meta["C1_determinism"]["PASS"] and (meta["C2_identity_vs_table"]["PASS_strict"] or meta["C2_identity_vs_table"]["PASS_fallback"]) and not c3_fail)
    json.dump(meta, open(os.path.join(a.out, "results_meta.json"), "w"), indent=1)
    if os.path.exists(part_p) and not a.keep_partial:
        os.remove(part_p)
    say(f"controls_pass {meta['controls_pass']} | C1 {meta['C1_determinism']} | C2 {meta['C2_identity_vs_table']}")
    return 0 if meta["controls_pass"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=C3_DIR + "/tokens.json")
    ap.add_argument("--table", default=C3_DIR + "/table.npz")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--db-dir", default=DB_DIR)
    ap.add_argument("--limit-tokens", type=int, default=None, help="smoke: only this many tokens, whole logs first")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--wait-idle-timeout", type=float, default=240.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--grey-verify-every", type=int, default=25)
    ap.add_argument("--replicate-every", type=int, default=4, help="run the A0r determinism replicate on every k-th token (k=1: all)")
    ap.add_argument("--keep-partial", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for m in ("DONE", "FAILED"):
        if os.path.exists(os.path.join(a.out, m)) and not a.resume:
            print(f"REFUSED: {m} marker already exists in {a.out} (use a fresh --out, or --resume after deleting the marker)")
            return 4
    rc = 1
    err = None
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        rc = main_run(a)
    except BaseException:                                       # noqa: B902 -- the marker must be written whatever happens
        err = traceback.format_exc()
        rc = 1
    # ---- LAST ACTION: the marker ---------------------------------------------------------------------------------------------------------------------------
    if rc in (0, 2):
        meta = {}
        try:
            meta = json.load(open(os.path.join(a.out, "results_meta.json")))
        except Exception:
            pass
        open(os.path.join(a.out, "DONE"), "w", encoding="utf-8").write(json.dumps({"rc": rc, "controls_pass": meta.get("controls_pass"), "n_tokens": meta.get("n_tokens"),
                                                                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    else:
        open(os.path.join(a.out, "FAILED"), "w", encoding="utf-8").write(f"rc {rc}\n{err or 'see run.log (refused / GPU busy)'}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
