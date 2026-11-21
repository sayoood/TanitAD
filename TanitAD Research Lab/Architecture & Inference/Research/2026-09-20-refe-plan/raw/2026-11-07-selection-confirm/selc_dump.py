#!/usr/bin/env python3
"""u400 dump for the H-SEL-1 fresh confirmation (T0 inputs; GPU under the shared dev-box lock). One SEGMENT per invocation.

The loop is the navtest seam's own (eval/refe_navtest_seam.py:226-262, copied in refe_ego_probe.py's A0 arm, which reproduced the C3 table bit for bit):
build_scenarios_for_log -> planner.initialize -> _image_for -> set_driving_command -> planner.infer (rule navsim_v1_lane, select best, goal_fix default
pdm_route, Amendment 7 heading repair default ON). Nothing in the package is edited or monkeypatched. Per token it keeps, from the SAME forward pass:
  logits [64,7] (raw scorer logits) | pick (planner's own) | traj [64,20,3] (RAW proposals, 0.2 s) | props8 [64,8,3] (EXECUTED proposals on NAVSIM's grid,
  to_navsim(planner.executed(traj, j)), exactly the seam's --dump-proposals) | ego [7] (planner.last_inputs[0], the fed ego vector; [6] = v0, [2] = ax0) |
  the logged future (gt_valid, gt_pose20, gt_t20, gt_speed20: refe_ego_probe.gt_future) | human8 (W3 export human_future_poses) | the seam's frame control.

Segments: each invocation takes the GPU lock (YAX gpu_lock.acquire_blocking: never forced, a foreign lock is never removed), loads the model,
processes whole logs until `--segment-tokens` is reached, writes one npz per log under <out>/chunks/, releases the lock and EXITS (so the VRAM is
freed and a queued job -- e.g. YAXN -- can take the GPU between segments). The FIRST segment processes the smoke tokens first and checks them
in-process against C3's table + the ego probe (bit for bit); a mismatch stops everything before a single fresh token is run (exit 2).
Exit codes: 0 all logs done | 3 segment done, more remain | 2 smoke identity FAILED | 5 GPU busy (gave up) | 4 refused | 1 error.
    <driverl python> selc_dump.py --tokens tokens_confirm.json --smoke tokens_smoke.json --out dump [--device cuda] [--segment-tokens 1000]
    <driverl python> selc_dump.py --consolidate --tokens tokens_confirm.json --out dump
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
REFE_DIR, EVAL_DIR = PK + "/refe", PK + "/eval"
DZ = "C:/Users/Admin/dz/DriveZero/DriveRL"
ROOT = "E:/Projects/TanitAD"
CKPT = ROOT + "/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
CKPT_SHA256_PREFIX = "8b4dba4818b5cc96"
FRAMES = ROOT + "/data/refe_navtest/frames"
DB_DIR = ROOT + "/data/nuplan/nuplan-v1.1/splits/test"
C3_TABLE = PK + "/raw/2026-10-11-c3-selection-headroom/table.npz"
EGO_NPZ = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/full/results.npz"
YAX_LOCK = "E:/Projects/TanitAD-artifacts/yawrate_wp/YAX"
RULE = "navsim_v1_lane"
OWNER = "selc_dump"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
MIN_FREE_GIB = 3.0

os.environ.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": ROOT + "/data/nuplan", "NUPLAN_MAPS_ROOT": ROOT + "/data/nuplan-maps/nuplan-maps-v1.0",
                   "REFE_NUPLAN_DB_ROOT": ROOT + "/data/nuplan/nuplan-v1.1/splits",
                   "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": DZ + "/nuplan-devkit",
                   "DRIVERL_EVAL_CONFIG_PATH": DZ + "/release/configs/driverl_teacher.yaml",
                   "DRIVERL_EVAL_CHECKPOINT_PATH": DZ + "/release/checkpoints/checkpoint_2400.pt",
                   "REFE_BACKBONE_ROOT": ROOT + "/data/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "4", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
for _p in (DZ + "/src", DZ + "/nuplan-devkit", REFE_DIR, EVAL_DIR, YAX_LOCK):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def navsim_root():
    r = os.environ.get("TANITAD_NAVSIM_ROOT")
    cands = [r] if r else [f"{d}:/Archive/devbox-C/navsim" for d in "EFD"]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise RuntimeError("no navsim archive root (set TANITAD_NAVSIM_ROOT)")


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def gt_future(sc, ego) -> dict:
    """refe_ego_probe.gt_future, verbatim: the logged future on the scenario's 20 states (ego frame, TRUE timestamps, state speeds)."""
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


META_DIR = navsim_root() + "/data/openscene/navsim_logs/test/"
FC_TOL_M = 0.001
FC_STEP_S = 0.5


def meta_times(lg):
    """AMENDMENT 1: {token: (anchor_time_us, [time_us of the 8 NEXT frames of the OpenScene list])} for one log (the frames the W3 export took its human_future_poses from)."""
    import pickle
    meta = pickle.load(open(META_DIR + lg + ".pkl", "rb"))
    ts = [int(f["timestamp"]) for f in meta]
    return {f["token"]: (ts[i], ts[i + 1:i + 9]) for i, f in enumerate(meta)}


def _ego_local(states, ego):
    px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    return np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s, (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c] for f in states])


def frame_ctrl_old(sc, ego, theirs):
    """The ORIGINAL reference (card section 6 as first written): export vs the DB future on the NOMINAL 0.5 s grid. Kept only to reproduce the 17 known failures."""
    f8 = list(sc.get_ego_future_trajectory(0, 4.0, 8))[:8]
    if len(f8) != 8:
        return float("nan")
    return float(np.linalg.norm(_ego_local(f8, ego) - theirs[:, :2], axis=1).max())


def frame_ctrl_meta(sc, ego, theirs, anchor_us, next_us):
    """AMENDMENT 1: the export vs the DB future sampled at the export's OWN meta timestamps (DB 4.5 s / 9 rows on the 0.5 s grid, slot = round(dt/0.5) - 1).
    nan (= FAIL) if the anchor time disagrees with the DB, the meta list has < 8 next frames, a step is off the grid by > 20 ms, or a slot is outside the 9 DB rows."""
    if next_us is None or len(next_us) != 8 or int(anchor_us) != int(ego.time_point.time_us):
        return float("nan")
    dts = (np.asarray(next_us, np.int64) - int(anchor_us)) / 1e6
    q = np.rint(dts / FC_STEP_S).astype(int) - 1
    if (np.abs(dts - (q + 1) * FC_STEP_S) > 0.02).any() or (q < 0).any() or (q > 8).any():
        return float("nan")
    f9 = list(sc.get_ego_future_trajectory(0, 4.5, 9))[:9]
    if len(f9) != 9:
        return float("nan")
    return float(np.linalg.norm(_ego_local(f9, ego)[q] - theirs[:, :2], axis=1).max())



def free_gib(path):
    import shutil
    return shutil.disk_usage(os.path.splitdrive(os.path.abspath(path))[0] + "/").free / 2 ** 30


def chunk_path(out, log):
    return os.path.join(out, "chunks", f"{log}.npz")


FIELDS = ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "gt_speed20", "human8", "frame_ctrl_m")


def save_chunk(p, recs):
    tmp = p + ".tmp.npz"
    np.savez(tmp, token=np.array([r["token"] for r in recs]), log=np.array([r["log"] for r in recs]), fingerprint=np.array([r["fingerprint"] for r in recs]),
             **{k: np.stack([np.asarray(r[k]) for r in recs]) for k in FIELDS})
    os.replace(tmp, p)


def smoke_identity(recs) -> dict:
    """Bit-for-bit identity of the smoke tokens against C3's table (logits, pick, executed props8) and the ego probe (raw traj, ego vector).
    RED arm: the same comparison against a cyclically SHIFTED reference must FAIL, or the check cannot see a defect."""
    T = np.load(C3_TABLE, allow_pickle=False)
    Z = np.load(EGO_NPZ, allow_pickle=False)
    ti = {str(t): i for i, t in enumerate(T["token"])}
    zi = {str(t): i for i, t in enumerate(Z["token"])}
    toks = [r["token"] for r in recs]
    I = [ti[t] for t in toks]
    J = [zi[t] for t in toks]
    a0 = [str(x) for x in Z["arms"]].index("A0")

    def cmp(I, J):
        lg = np.stack([r["logits"] for r in recs]).astype(np.float32)
        pk = np.array([r["pick"] for r in recs])
        p8 = np.stack([r["props8"] for r in recs]).astype(np.float32)
        tr = np.stack([r["traj"] for r in recs]).astype(np.float32)
        eg = np.stack([r["ego"] for r in recs]).astype(np.float64)
        rp, rl, rk = T["proposals"][I].astype(np.float32), T["logits"][I].astype(np.float32), T["pick"][I]
        return {"logits_bit_identical": int(sum(np.array_equal(lg[i], rl[i]) for i in range(len(I)))),
                "pick_equal": int((pk == rk).sum()),
                "props8_xy_bit_identical": int(sum(np.array_equal(p8[i, ..., :2], rp[i, ..., :2]) for i in range(len(I)))),
                "props8_all_bit_identical": int(sum(np.array_equal(p8[i], rp[i]) for i in range(len(I)))),
                "props8_max_abs": float(np.abs(p8.astype(np.float64) - rp).max()),
                "traj_bit_identical_vs_ego_probe": int(sum(np.array_equal(tr[i], Z["traj_all_A0"][J[i]]) for i in range(len(J)))),
                "ego_bit_identical_vs_ego_probe": int(sum(np.array_equal(eg[i], Z["ego_real"][J[i]]) for i in range(len(J)))),
                "pick_equal_vs_ego_probe": int(sum(pk[i] == Z["pick"][J[i], a0] for i in range(len(J))))}
    g = {"n": len(recs), "tokens": toks, "ok": cmp(I, J), "RED_shifted": cmp(I[1:] + I[:1], J[1:] + J[:1])}
    n = len(recs)
    ok = g["ok"]
    g["PASS"] = bool(ok["logits_bit_identical"] == n and ok["pick_equal"] == n and ok["props8_xy_bit_identical"] == n and ok["props8_max_abs"] <= 1e-6
                     and ok["traj_bit_identical_vs_ego_probe"] == n and ok["ego_bit_identical_vs_ego_probe"] == n and ok["pick_equal_vs_ego_probe"] == n)
    red = g["RED_shifted"]
    g["RED_FAILS_AS_REQUIRED"] = bool(red["logits_bit_identical"] < n and red["traj_bit_identical_vs_ego_probe"] < n)
    g["GATE"] = bool(g["PASS"] and g["RED_FAILS_AS_REQUIRED"])
    return g


def run_segment(a, say) -> int:
    doc = json.load(open(a.tokens, encoding="utf-8"))
    token_log = doc["token_log"]
    by = {}
    for t in doc["tokens"]:
        by.setdefault(token_log[t], []).append(t)
    os.makedirs(os.path.join(a.out, "chunks"), exist_ok=True)
    smoke_json = os.path.join(a.out, "smoke_identity.json")
    smoke_done = os.path.exists(smoke_json) and bool(json.load(open(smoke_json)).get("GATE"))
    pending = [lg for lg in sorted(by) if not os.path.exists(chunk_path(a.out, lg))]
    if a.smoke_only and smoke_done:
        say("smoke already done")
        return 0
    if not pending and smoke_done:
        say("all logs done")
        return 0
    if free_gib(a.out) < MIN_FREE_GIB:
        say(f"REFUSED: {free_gib(a.out):.2f} GiB free on the output drive < {MIN_FREE_GIB}")
        return 4
    plan, n = [], 0
    for lg in pending:
        if plan and n + len(by[lg]) > a.segment_tokens:
            break
        plan.append(lg)
        n += len(by[lg])
    h = hashlib.sha256()
    with open(a.ckpt, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if not h.hexdigest().startswith(CKPT_SHA256_PREFIX):
        say(f"REFUSED: checkpoint sha256 {h.hexdigest()[:16]} != {CKPT_SHA256_PREFIX}")
        return 4
    exp = json.load(gzip.open(os.path.join(navsim_root(), "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"), "rt", encoding="utf-8"))["tokens"]
    say(f"segment: smoke {'done' if smoke_done else 'FIRST'}; {len(plan)} logs / {n} tokens of {len(pending)} pending logs")
    body = None
    if a.device == "cuda" and not a.no_gpu_lock:
        import gpu_lock as GL
        t0 = time.time()
        while True:
            if os.path.exists(MARK):
                try:
                    holder = open(MARK, encoding="utf-8").read()[:200]
                except OSError:
                    holder = "?"
                if time.time() - t0 > a.give_up_h * 3600:
                    say(f"GPU BUSY: chain marker present for {a.give_up_h} h: {holder}")
                    return 5
                say(f"chain marker present, waiting: {holder}")
                time.sleep(120)
                continue
            try:
                body = GL.acquire_blocking(max(60.0, a.give_up_h * 3600 - (time.time() - t0)), poll_s=60.0, log=say, owner=OWNER, wait_timeout_s=240.0)
                break
            except GL.GpuBusy as e:
                say(f"GPU BUSY (gave up, nothing loaded): {e}")
                return 5
        say(f"GPU lock taken: {body}")
    try:
        import torch
        import navtrain_scenarios as NS
        import refe_navtest_seam as SEAM
        from planner import REFePlanner
        from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
        planner = REFePlanner(checkpoint=a.ckpt, images_root=FRAMES, db_dir=a.db_dir, backbone="vitl16", device=a.device, select="best", rule=RULE)
        say(f"planner: device {planner.device} rule {planner.rule} goal_fix {planner.goal_fix} repair_last_heading {planner.repair_last_heading} "
            f"nav_mode {getattr(planner.model.cfg, 'nav_mode', 'goal')}")
        assert planner.rule == RULE and planner.select == "best"

        def run_log(lg, toks):
            recs = []
            db = os.path.join(a.db_dir, f"{lg}.db")
            got = set()
            mtimes = meta_times(lg)
            for sc in NS.build_scenarios_for_log(db, toks, history_rows=1, future_rows=80):
                tok = sc._initial_lidar_token
                got.add(tok)
                planner._scenario = sc
                planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                ego = sc.get_ego_state_at_iteration(0)
                img = planner._image_for(ego)
                if img is None:
                    raise RuntimeError(f"token {tok}: no frames ({planner.frames.miss_reason})")
                SEAM.set_driving_command(planner, tok, exp[tok])
                traj, score, k = planner.infer(ego, img)
                ego_fed = planner.last_inputs[0].detach().cpu().numpy().astype(np.float64)[0]
                tr = traj.float().cpu().numpy()
                props8 = np.stack([SEAM.to_navsim(planner.executed(traj, j).float().cpu().numpy()) for j in range(traj.shape[0])]).astype(np.float32)
                rec = {"token": tok, "log": lg, "fingerprint": exp[tok]["fingerprint"], "logits": score.float().cpu().numpy().astype(np.float32),
                       "pick": int(k), "traj": tr.astype(np.float32), "props8": props8, "ego": ego_fed, "human8": np.asarray(exp[tok]["human_future_poses"], np.float64)}
                rec.update(gt_future(sc, ego))
                theirs = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)
                mt = mtimes.get(tok)
                rec["frame_ctrl_m"] = frame_ctrl_meta(sc, ego, theirs, mt[0], mt[1]) if mt else float("nan")
                assert props8.shape == (64, 8, 3) and tr.shape == (64, 20, 3) and rec["logits"].shape == (64, 7)
                assert np.isfinite(props8).all() and np.isfinite(tr).all()
                recs.append(rec)
            miss = set(toks) - got
            if miss:
                raise RuntimeError(f"{len(miss)} tokens skipped by the scenario builder in {lg} (first {sorted(miss)[0]})")
            order = {t: i for i, t in enumerate(sorted(toks))}
            return sorted(recs, key=lambda r: order[r["token"]])

        if not smoke_done:
            sdoc = json.load(open(a.smoke, encoding="utf-8"))
            sby = {}
            for t in sdoc["tokens"]:
                sby.setdefault(sdoc["token_log"][t], []).append(t)
            srecs = []
            for lg in sorted(sby):
                srecs += run_log(lg, sby[lg])
            save_chunk(os.path.join(a.out, "smoke_dump.npz"), srecs)
            g = smoke_identity(srecs)
            g["device"] = a.device
            json.dump(g, open(smoke_json, "w"), indent=1)
            say(f"SMOKE identity: PASS {g['PASS']} RED fails {g['RED_FAILS_AS_REQUIRED']} -> GATE {g['GATE']} | {g['ok']}")
            if not g["GATE"] and not a.plumbing:
                return 2
            if a.smoke_only:
                return 0 if g["GATE"] else 2
        t_seg = time.time()
        nd = 0
        for li, lg in enumerate(plan):
            t_log = time.time()
            recs = run_log(lg, by[lg])
            fc = np.array([r["frame_ctrl_m"] for r in recs])
            if not (fc <= FC_TOL_M).all():   # AMENDMENT 1: a nan is a FAIL too (the old nanmax let an all-nan log pass)
                raise RuntimeError(f"frame control FAILED in {lg}: max {np.nanmax(fc) if np.isfinite(fc).any() else float('nan'):.4f} m > 1 mm or nan ({int((~(fc <= FC_TOL_M)).sum())} tokens)")
            save_chunk(chunk_path(a.out, lg), recs)
            nd += len(recs)
            say(f"[{li + 1}/{len(plan)}] {lg}: {len(recs)} tokens in {time.time() - t_log:.0f} s | segment {nd}/{n} | {(time.time() - t_seg) / max(nd, 1):.2f} s/token")
            if free_gib(a.out) < MIN_FREE_GIB:
                say(f"STOP: {free_gib(a.out):.2f} GiB free < {MIN_FREE_GIB}")
                return 4
    finally:
        if body:
            import gpu_lock as GL
            GL.release_gpu_lock(GL.LOCK, body)
            say("GPU lock released")
    left = [lg for lg in sorted(by) if not os.path.exists(chunk_path(a.out, lg))]
    return 0 if not left else 3


def consolidate(a, say) -> int:
    doc = json.load(open(a.tokens, encoding="utf-8"))
    by = {}
    for t in doc["tokens"]:
        by.setdefault(doc["token_log"][t], []).append(t)
    parts = {k: [] for k in ("token", "log", "fingerprint") + FIELDS}
    for lg in sorted(by):
        with np.load(chunk_path(a.out, lg), allow_pickle=False) as z:
            for k in parts:
                parts[k].append(z[k])
    D = {k: np.concatenate(v) for k, v in parts.items()}
    order = np.argsort(D["token"])                  # AMENDMENT 2: chunks are concatenated log by log; the assertion below and selc_score require token order
    D = {k: v[order] for k, v in D.items()}
    tok = [str(t) for t in D["token"]]
    assert tok == sorted(doc["tokens"]) == sorted(tok) and len(set(tok)) == len(tok), "dump tokens != tokens_confirm.json"
    out = os.path.join(a.out, "dump_confirm.npz")
    np.savez(out, sampling=np.array([8, 0.5]), rule=np.array(RULE), ckpt=np.array(CKPT), select=np.array("best"), **D)
    # the C3/selection_quality-format proposals file (token, fingerprint, proposals [N,64,8,3], logits, pick, sampling)
    np.savez(os.path.join(a.out, "proposals_confirm.npz"), token=D["token"], fingerprint=D["fingerprint"], proposals=D["props8"], logits=D["logits"],
             pick=D["pick"].astype(np.int64), sampling=np.array([8, 0.5]), rule=np.array(RULE), ckpt=np.array(CKPT))
    fc = D["frame_ctrl_m"]
    rep = {"n_tokens": len(tok), "n_logs": len(by), "frame_control_max_m": float(np.nanmax(fc)), "frame_control_n": int(np.isfinite(fc).sum()),
           "gt_valid_share": float(D["gt_valid"].mean()), "dump": out}
    json.dump(rep, open(os.path.join(a.out, "consolidate.json"), "w"), indent=1)
    say(f"consolidated: {rep}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--smoke", default=os.path.join(HERE, "tokens_smoke.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--db-dir", default=DB_DIR)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--segment-tokens", type=int, default=1000)
    ap.add_argument("--give-up-h", type=float, default=10.0)
    ap.add_argument("--consolidate", action="store_true")
    ap.add_argument("--smoke-only", action="store_true", help="run ONLY the smoke tokens + their identity gate, then exit (0 = GATE green)")
    ap.add_argument("--plumbing", action="store_true", help="CPU plumbing smoke ONLY: record the smoke identity but do not stop on it (CPU is not bit-identical to the GPU)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    log_p = os.path.join(a.out, "dump.log")

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        return consolidate(a, say) if a.consolidate else run_segment(a, say)
    except BaseException:                                   # noqa: B902
        say("ERROR\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
