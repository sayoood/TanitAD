#!/usr/bin/env python3
"""EXPLORATORY (report only, dev box): SLOWER COPIES of REFe's own proposals as extra candidates (snapshot 015).

MOTIVATION (coordinator, from the E-6 tables): the best-of-64 drop is not slow slots dying -- the whole fan's speed
shifted up at the first on-policy epoch, so in slow scenes even the slowest option is too fast. STOP recovered some
PDMS but collapsed the longitudinal family (speed MAE 1.17 -> 2.52 m/s, progress ratio 1.09 -> 0.76; see
stop_candidate.json). LEVER: keep each proposal's PATH and slow its SPEED PROFILE, and let the planner choose.

THE COPY (`refe/slow_copies.py::slow_copy(traj, factor, dt=0.2)`, the ONE shared implementation; its validity tests
are eval/selftest_slow_copies.py). REFe emits 20 poses (x, y, heading) at t = 0.2 .. 4.0 s in the t0 rear-axle ego
frame; the ego is at the origin (0, 0, 0) at t = 0. The factor-f copy is
        copy_f(t_k) = p(f * t_k),     t_k = 0.2 k,  k = 1..20,
i.e. at time t the copy is where the original was at f*t: the same path, traversed f times as fast (speed profile
v_copy(t) = f * v(f*t)). p(tau) between two samples is evaluated by the SEAM CONVERTER'S OWN RULE (`to_navsim`): a
sample time is copied verbatim; in between, x and y are linear in time and the heading is the start heading plus the
WRAPPED heading difference, wrapped -- with the origin (0, 0, 0) standing in as the sample at t = 0. Linear in time
between two samples is linear in arc length along that polyline segment, so every copy pose lies ON the original's
polyline (the path is kept), and its heading is the original's own heading at that point of the path ("heading
along the path" as the MODEL oriented it -- a polyline-tangent heading would change the pose even at f = 1, and the
f = 1 copy must reproduce the original bit-for-bit). The copy is then handled exactly like a REFe proposal: the
float32 20-pose array is what REFe's scorer sees, and `to_navsim` of it (the seam's converter) is what the harness
scores. f = 1.0 is the identity control. Consequence, stated not hidden: at t = 0 the copy's speed is f x the
original's, so a copy of a proposal that starts at the ego's speed asks the tracker for an immediate slow-down.

STAGES (resumable; each prints a ZZ marker):
  build    from stop_candidate_probe.py's dump of TODAY's ep015 forward (native poses; its proposals reproduce the
           stored table bit-for-bit, re-checked here): copies at f = 0.75, 0.5 for all 64 slots + two f = 1.0
           control seams (each token's shipped pick; slot 13) -> one W3 seam per (factor, slot)
  score    each seam through `score_navtest_refe.py` UNCHANGED (W3's harness, SeamAgentV1, metric_cache_navtest) --
           the E-6 table's own path; parallel workers, retries, never under the GPU child's environment
  gpu      (child, `EC.env_driverl()`) the ep015 forward again, the scoring decoder's context captured, and REFe's
           scorer on: the 64 alone; PLAIN sets (every member attends to every member) 64+c075, 64+c05, 64+both,
           64+both+STOP; and one MASKED set 64+c075+c05+STOP in which the 64 do not attend to any extra member and
           each extra member attends to the 64 and ITSELF only -- so a copy's score does not depend on which other
           copies exist, and every masked subset is read from this one pass. Masked arms keep the 64's SHIPPED logits.
  analyze  picks by the shipped v1 aggregate (the planner's own `_pick`), PDMS of each arm's pick (the table for the
           64, the copies' harness csvs, the STOP csv), paired log-cluster bootstrap vs shipped and vs the STOP
           floor, the ceiling, switches by factor, NAVSIM sub-scores, and the four families (families6.py) per arm

CONTROLS (gates): the dump's proposals == the stored table (bit); today's GPU forward == the dump (poses, logits,
pick: bit); f = 1 copies == the originals (native and NAVSIM grid: bit) and their HARNESS scores == the table's
(two slots' worth: each token's shipped pick, and slot 13; bar 1e-9, E-6 G2's); every harness run PASSes with 200
valid rows; masked-decoder identities (unmasked == score_trajectories bit; masked 64 rows vs the 64 alone <= 1e-4;
masked STOP row vs stop_candidate_probe's V-mask STOP row <= 1e-4).

TOP-16 DESIGN (the coordinator's fallback, used): 128 per-slot harness runs measured ~0.8 runs/min under D: I/O
contention from other sessions, so the arms use, per token, copies of the 16 proposals the SHIPPED scorer ranks
highest (rank 0 = the shipped pick): 16 ranks x 2 factors = 32 harness runs (`build-ranks`, scheduled first). The
per-slot runs continue behind them and an all-slot arm set is reported only if all 128 PASSed.

POST-HOC REFERENCE ROWS (not selections, no bar; added after the top-16 harness scores were read): every token takes
the factor-f copy of its SHIPPED pick -- f = 0.75 and 0.5 from rank 0 of the top-16 design, and a sweep f = 0.6, 0.7,
0.8, 0.85, 0.9, 0.95 (`build-sweep`, `score --set sweep`) -- with PDMS and the four families per factor. Also
descriptive strata by the ego's speed at t0 and by the logged human's 4 s displacement.

⛔ THE CONFOUND THE SWEEP EXPOSED (`build-repair`, `score --set repair`, section `last_heading_defect`): f = 0.95
already gave +18 PDMS, as much as any factor. REFe's LAST pose (t = 4.0 s) carries a corrupted HEADING on most
proposals in EVERY snapshot (median |heading - path tangent| ~1.1 rad there vs ~0.04 rad at every other pose; the
training targets are clean), and `to_navsim` passes that pose verbatim to the harness -- while any copy with
f <= 0.95 never reaches it. The repair arms keep the shipped pick's positions and speed and replace ONLY that one
heading, which separates "slower" from "without the corrupted last heading". ⚠️ The four families cannot see this:
the NavSim adapter drops the heading column and its heading_mae_deg is the PATH TANGENT (adapter docstring), so a
heading-only repair reads identical families by construction.

    python eval/slow_copy_probe.py build
    python eval/slow_copy_probe.py build-ranks
    python eval/slow_copy_probe.py build-sweep
    python eval/slow_copy_probe.py build-repair
    python eval/slow_copy_probe.py score --set all --workers 4
    python eval/slow_copy_probe.py score --set sweep --workers 2
    python eval/slow_copy_probe.py score --set repair --workers 2
    python eval/slow_copy_probe.py gpu
    python eval/slow_copy_probe.py analyze
Writes eval/raw/e6_sub200_ep015/slow_copies.json.
(History, stated: the build/score stages ran under this script's first name `slow_copies.py`, with the copy as a
local scalar loop; the loop was then moved VERBATIM in arithmetic to refe/slow_copies.py -- MEASURED bit-identical on
all 200 x 64 x 2 copies, numpy and torch -- and control C6 re-asserts that on every analysis.)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  one source for the venv + env the seam ran with
import refe_navtest_seam as SEAM    # noqa: E402  to_navsim, the export, the test DBs; puts refe/ on sys.path
import slow_copies as SLOW          # noqa: E402  refe/slow_copies.py -- the ONE implementation of the copy
from slow_copies import slow_copy   # noqa: E402

NAVTEST = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest"
TOK = f"{NAVTEST}/raw/A1_sub200_tokens.json"
STOP_CSV = f"{NAVTEST}/raw/STOP_navtest/STOP_navtest.csv"
NAME = "sub200_ep015"
TABLE = f"{EC.DATA}/proptable/{NAME}/table.npz"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
LANDED_SEAM = f"{EC.DATA}/seams/refe_{NAME}.npz"
LANDED_FAM = f"{EC.DATA}/points/{NAME}/4_families.json"
STOP_DUMP = f"{EC.DATA}/proptable/{NAME}/stop_candidate_dump.npz"       # stop_candidate_probe.py's GPU dump
WD = f"{EC.DATA}/proptable/{NAME}/slow_copies"
SEAM_DIR = f"{EC.DATA}/seams/proptable/{NAME}_slow"
BUILD = f"{WD}/build.npz"
GPU_DUMP = f"{WD}/gpu_dump.npz"
OUT = os.path.join(HERE, "raw", f"e6_{NAME}", "slow_copies.json")
FACTORS = (0.75, 0.5)
CONTROL_SLOT = 13
NRANK = 16
RANKS = f"{WD}/ranks.npz"
SEED = 20260927          # eval/snapshot_pair_under_rule.py's bootstrap seed, copied with its estimator
BAR = 1e-4
CHILD = "REFE_SLOW_COPIES_CHILD"
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def ftag(f):
    return f"f{int(round(f * 100)):03d}"


def label(f, slot):
    return f"refe_{NAME}_{ftag(f)}_{slot if isinstance(slot, str) else f'p{slot:02d}'}"


def seam_path(f, slot):
    return os.path.join(SEAM_DIR, f"{label(f, slot)}.npz")


def csv_path(f, slot):
    lb = label(f, slot)
    return f"{EC.DATA}/score/{lb}/{lb}.csv"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copies_of(traj32, factors):
    """traj32 [..., 64, 20, 3] float32 -> {f: (native float32 [..., 64, 20, 3], navsim float32 [..., 64, 8, 3])}.
    The copy is refe/slow_copies.py::slow_copy -- the ONE implementation (the training-target and on-policy-label
    agents import the same function); the NAVSIM grid is the seam's own `to_navsim`."""
    lead = traj32.shape[:-2]
    res = {}
    for f in factors:
        nat = slow_copy(np.asarray(traj32, dtype=np.float32), f, dt=SEAM.SRC_DT)
        flat = nat.reshape(-1, nat.shape[-2], nat.shape[-1])
        nav = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in flat])
        res[f] = (nat, nav.reshape(lead + nav.shape[1:]))
    return res


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


# ------------------------------------------------------------------------------------------------ build
def build(a) -> int:
    D = np.load(STOP_DUMP, allow_pickle=False)
    T = np.load(TABLE)
    tok = [str(t) for t in D["token"]]
    if tok != [str(t) for t in T["token"]]:
        print("ZZSLOW_FAIL build token order"); return 1
    traj = D["traj"].astype(np.float32)                                      # [200, 64, 20, 3]
    n = len(tok)
    props = np.stack([[SEAM.to_navsim(traj[i, j]).astype(np.float32) for j in range(64)] for i in range(n)])
    c1 = float(np.abs(props.astype(np.float64) - T["proposals"].astype(np.float64)).max())
    cp = copies_of(traj, (1.0,) + FACTORS)
    id_nat = bool(np.array_equal(cp[1.0][0], traj))
    id_nav = bool(np.array_equal(cp[1.0][1], T["proposals"]))
    print(f"  dump proposals vs stored table max |d| {c1}; f=1 copy == original: native {id_nat}, navsim {id_nav}",
          flush=True)
    if c1 != 0.0 or not (id_nat and id_nav):
        print("ZZSLOW_FAIL build identity"); return 1
    S0 = np.load(LANDED_SEAM)
    if [str(t) for t in S0["token"]] != tok:
        print("ZZSLOW_FAIL landed seam token order"); return 1
    os.makedirs(SEAM_DIR, exist_ok=True)
    os.makedirs(WD, exist_ok=True)
    pick = T["pick"]
    ar = np.arange(n)
    seams = {}
    for f in FACTORS:
        for k in range(64):
            seams[label(f, k)] = (seam_path(f, k), cp[f][1][:, k])
    seams[label(1.0, "pick")] = (seam_path(1.0, "pick"), cp[1.0][1][ar, pick])
    seams[label(1.0, CONTROL_SLOT)] = (seam_path(1.0, CONTROL_SLOT), cp[1.0][1][:, CONTROL_SLOT])
    for lb, (sp, poses) in seams.items():
        np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=poses.astype(np.float32),
                 sampling=S0["sampling"], arm=np.array(lb))
    np.savez(BUILD, token=np.array(tok), **{f"nat_{ftag(f)}": cp[f][0] for f in FACTORS},
             **{f"nav_{ftag(f)}": cp[f][1] for f in FACTORS}, factors=np.array(FACTORS),
             stop_dump_sha256=np.array(sha256(STOP_DUMP)))
    print(f"  wrote {len(seams)} seams -> {SEAM_DIR}; build -> {BUILD}")
    print("ZZSLOW_BUILD_OK", len(seams))
    return 0


def build_ranks(a) -> int:
    """THE SCORER'S TOP-16 DESIGN (the coordinator's fallback when 64 x 2 harness runs are too slow -- MEASURED here:
    ~0.8 runs/min under D: I/O contention from other sessions, i.e. ~2.5 h for 128). Per token, the 16 proposals the
    shipped scorer ranks highest (navsim_v1 aggregate of the STORED logits, float64, stable sort -- rank 0 is the
    shipped pick); one seam per (factor, rank) holds, for every token, the copy of THAT token's rank-r proposal."""
    import stop_candidate_probe as SCP
    T = np.load(TABLE)
    B = np.load(BUILD)
    S0 = np.load(LANDED_SEAM)
    order = [str(x) for x in T["head_order"]]
    agg = SCP.agg_v1_np(T["logits"].astype(np.float64), order)
    top = np.argsort(-agg, axis=1, kind="stable")[:, :NRANK]
    if not (top[:, 0] == T["pick"]).all():
        print("ZZSLOW_FAIL rank 0 is not the shipped pick"); return 1
    ar = np.arange(top.shape[0])
    for f in FACTORS:
        nav = B[f"nav_{ftag(f)}"]
        for r in range(NRANK):
            lb = label(f, f"r{r:02d}")
            np.savez(seam_path(f, f"r{r:02d}"), token=S0["token"], fingerprint=S0["fingerprint"],
                     poses=nav[ar, top[:, r]].astype(np.float32), sampling=S0["sampling"], arm=np.array(lb))
    np.savez(RANKS, token=T["token"], top=top, agg=agg)
    print(f"  wrote {len(FACTORS) * NRANK} rank seams; ranks -> {RANKS}")
    print("ZZSLOW_RANKS_OK")
    return 0


# ------------------------------------------------------------------------------------------------ score
SWEEP = (0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def build_sweep(a) -> int:
    """POST-HOC REFERENCE SWEEP (added after the top-16 harness scores showed the f = 0.75 copy of the shipped pick,
    taken on EVERY token, far above the shipped pick): the shipped pick itself at more factors, one seam each, so
    PDMS and the four families can be read as a function of the factor. Not a selection; no bar."""
    SD = np.load(STOP_DUMP, allow_pickle=False)
    T = np.load(TABLE)
    S0 = np.load(LANDED_SEAM)
    n = len(T["token"])
    picked = SD["traj"][np.arange(n), T["pick"]].astype(np.float32)          # [n, 20, 3] the shipped pick, native
    for f in SWEEP:
        nat = slow_copy(picked, f, dt=SEAM.SRC_DT)
        nav = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in nat])
        np.savez(seam_path(f, "pick"), token=S0["token"], fingerprint=S0["fingerprint"], poses=nav,
                 sampling=S0["sampling"], arm=np.array(label(f, "pick")))
    print(f"  wrote {len(SWEEP)} sweep seams")
    print("ZZSLOW_SWEEP_BUILD_OK")
    return 0


def repair_last_heading(p20, how):
    """[..., 20, 3] -> copy with ONLY the last pose's heading replaced ("hold": the previous pose's heading;
    "tangent": the direction of the last path segment, the previous heading when that segment is < 5 cm)."""
    q = np.array(p20, dtype=np.float64, copy=True)
    if how == "hold":
        q[..., -1, 2] = q[..., -2, 2]
    else:
        d = q[..., -1, :2] - q[..., -2, :2]
        tan = np.arctan2(d[..., 1], d[..., 0])
        q[..., -1, 2] = np.where(np.linalg.norm(d, axis=-1) >= 0.05, tan, q[..., -2, 2])
    return q.astype(np.asarray(p20).dtype)


def build_repair(a) -> int:
    """THE CONFOUND CHECK (added after the sweep: f = 0.95 gave +18 PDMS, as much as any factor). REFe's LAST pose
    (t = 4.0 s) carries a corrupted heading on most proposals -- and every copy with f <= 0.95 never reaches that
    pose. So: the shipped pick at f = 1.0 (speed UNCHANGED) with only the last heading repaired, two ways."""
    SD = np.load(STOP_DUMP, allow_pickle=False)
    T = np.load(TABLE)
    S0 = np.load(LANDED_SEAM)
    n = len(T["token"])
    picked = SD["traj"][np.arange(n), T["pick"]].astype(np.float32)
    for how in ("hold", "tangent"):
        rep = repair_last_heading(picked, how)
        nav = np.stack([SEAM.to_navsim(x).astype(np.float32) for x in rep])
        np.savez(seam_path(1.0, f"pick{how}"), token=S0["token"], fingerprint=S0["fingerprint"], poses=nav,
                 sampling=S0["sampling"], arm=np.array(label(1.0, f"pick{how}")))
    print("ZZSLOW_REPAIR_BUILD_OK")
    return 0


def all_labels(which="all"):
    ctrl = [(1.0, "pick"), (1.0, CONTROL_SLOT)]
    ranks = [(f, f"r{r:02d}") for r in range(NRANK) for f in FACTORS]          # both factors per rank, rank order
    slots = [(f, k) for f in FACTORS for k in range(64)]
    sweep = [(f, "pick") for f in SWEEP]
    repair = [(1.0, "pickhold"), (1.0, "picktangent")]
    return {"controls": ctrl, "ranks": ranks, "slots": slots, "all": ctrl + ranks + slots, "sweep": sweep,
            "repair": repair}[which]


def score(a) -> int:
    if os.environ.get(CHILD) == "1":
        print("ZZSLOW_FAIL score must not run under the GPU child's environment"); return 1
    os.makedirs(os.path.join(WD, "logs"), exist_ok=True)
    todo = all_labels(a.set) if not a.only else [x for x in all_labels() if label(*x) in set(a.only)]

    def one(fk):
        f, k = fk
        lb = label(f, k)
        log = os.path.join(WD, "logs", f"{lb}.log")
        for attempt in range(a.tries):
            if os.path.exists(csv_path(f, k)) and os.path.exists(log):
                st = status_of(open(log, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == 200:
                    return lb, st
            if attempt:
                time.sleep(30)
            _rc, txt = EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", seam_path(f, k),
                               "--tokens", a.tokens, "--out", f"{EC.DATA}/score"],
                              HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        return lb, st

    t0 = time.time()
    res = {}
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for i, (lb, st) in enumerate(ex.map(one, todo)):
            res[lb] = st
            if (i + 1) % 4 == 0 or i + 1 == len(todo):
                npass = sum(1 for s in res.values() if s and s.get("status") == "PASS")
                print(f"    scored {i + 1}/{len(todo)}  pass {npass}  {time.time() - t0:.0f} s  "
                      f"local {time.strftime('%H:%M:%S')}", flush=True)
    bad = [lb for lb, st in res.items() if not (st and st.get("status") == "PASS" and st.get("csv_valid_rows") == 200)]
    json.dump({"set": a.set, "runs": len(todo), "pass": len(todo) - len(bad), "failed": bad,
               "seconds": round(time.time() - t0, 1)},
              open(os.path.join(WD, f"score_status_{a.set}.json"), "w"), indent=1)
    print(f"ZZSLOW_SCORE_{'OK' if not bad else 'FAIL'} {len(todo) - len(bad)}/{len(todo)}")
    return 0 if not bad else 1


# ------------------------------------------------------------------------------------------------ gpu
def past(deadline: str) -> bool:
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return (lt.tm_hour, lt.tm_min) >= (hh, mm)


def gpu_stage(a) -> int:
    import gzip
    import torch
    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    from stop_candidate_probe import score_masked          # ONE masked decoder, the one the STOP probe validated

    SD = np.load(STOP_DUMP, allow_pickle=False)
    B = np.load(BUILD, allow_pickle=False)
    RK = np.load(RANKS, allow_pickle=False)
    sd_tok = [str(t) for t in SD["token"]]
    sd_ix = {t: i for i, t in enumerate(sd_tok)}
    if [str(t) for t in RK["token"]] != sd_tok:
        print("ZZSLOW_FAIL ranks token order"); return 1
    RKtop = RK["top"]
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    sub = json.load(open(a.tokens, encoding="utf-8"))
    want = set(sub["tokens"] if isinstance(sub, dict) else sub)
    by_log: dict = {}
    for t in [t for t in exp if t in want]:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    planner = REFePlanner(checkpoint=CKPT, images_root=a.frames, db_dir=SEAM.TEST_DB_DIR, backbone="vitl16",
                          device="cuda", select="best", rule=None)
    model = planner.model
    print(f"  planner: per_sample_calib={planner.per_sample_calib} device={planner.device} rule={planner.rule}",
          flush=True)
    if planner.device != "cuda" or planner.rule != "navsim_v1":
        print("ZZSLOW_FAIL gpu planner"); return 1
    dev = planner.device
    H, Dm = int(model.cfg.horizon_steps), int(model.cfg.traj_dim)
    cap = {"on": False}

    def pre_hook(_mod, args):
        if cap["on"]:
            cap["sctx"] = args[1]
    hnd = model.score_dec[0].register_forward_pre_hook(pre_hook)
    K = 2 * 64 + 1                                            # c075, c05, STOP
    M = 64 + K
    mask = torch.zeros(M, M, dtype=torch.bool, device=dev)
    mask[:64, 64:] = True                                     # the 64 never see an extra member
    mask[64:, 64:] = ~torch.eye(K, dtype=torch.bool, device=dev)   # an extra member sees the 64 and ITSELF only
    R: dict = {k: [] for k in ("token", "s64", "p075", "p05", "pboth", "pboth_stop", "m", "k", "k_s64", "k_p075",
                               "k_p05", "k_pboth", "k_pboth_stop", "k_m075", "k_m05", "k_mboth", "k_mboth_stop",
                               "p16_075", "p16_05", "p16_both", "p16_both_stop", "k_p16_075", "k_p16_05",
                               "k_p16_both", "k_p16_both_stop", "k_m16_075", "k_m16_05", "k_m16_both",
                               "k_m16_both_stop",
                               "traj_maxdiff", "copy_maxdiff", "id_nomask", "id_m64", "ego", "goal", "calib", "sec")}
    stopped = None
    t0 = time.time()
    logs = sorted(by_log.items())
    if a.limit_logs:
        logs = logs[:a.limit_logs]
    for li, (log, lt) in enumerate(logs):
        if stopped:
            break
        for sc in NS.build_scenarios_for_log(os.path.join(SEAM.TEST_DB_DIR, f"{log}.db"), lt,
                                             history_rows=1, future_rows=80):
            if past(a.deadline):
                stopped = f"deadline {a.deadline} local reached at {time.strftime('%H:%M:%S')}"
                break
            ts = time.time()
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(
                route_roadblock_ids=sc.get_route_roadblock_ids(),
                mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                continue
            cap.pop("sctx", None)
            cap["on"] = True
            traj, score, k = planner.infer(ego, img)              # the planner's OWN call and pick, unchanged
            cap["on"] = False
            sctx = cap.pop("sctx")
            i = sd_ix[tok]
            tr = traj.float().cpu().numpy()
            cp = copies_of(tr, FACTORS)                            # rebuilt from TODAY's poses
            d_copy = max(float(np.abs(cp[f][0].astype(np.float64) - B[f"nat_{ftag(f)}"][i]).max()) for f in FACTORS)
            with torch.no_grad():
                T64 = traj[None]
                C = {f: torch.from_numpy(cp[f][0]).to(dev)[None] for f in FACTORS}
                stop = torch.zeros(1, 1, H, Dm, dtype=traj.dtype, device=dev)
                full = torch.cat([T64, C[0.75], C[0.5], stop], 1)                     # [1, 193, H, D]
                s64 = model.score_trajectories(T64, sctx)[0]
                p075 = model.score_trajectories(full[:, :128], sctx)[0]
                p05 = model.score_trajectories(torch.cat([T64, C[0.5]], 1), sctx)[0]
                pboth = model.score_trajectories(full[:, :192], sctx)[0]
                pboth_stop = model.score_trajectories(full, sctx)[0]
                m = score_masked(model, full, sctx, mask)[0]
                m_nomask = score_masked(model, full, sctx, None)[0]
                pk = lambda L: planner._pick(L[None])              # noqa: E731  the SHIPPED v1 pick
                picks = {"k_s64": pk(s64), "k_p075": pk(p075), "k_p05": pk(p05), "k_pboth": pk(pboth),
                         "k_pboth_stop": pk(pboth_stop),
                         "k_m075": pk(torch.cat([s64, m[64:128]], 0)), "k_m05": pk(torch.cat([s64, m[128:192]], 0)),
                         "k_mboth": pk(torch.cat([s64, m[64:192]], 0)),
                         "k_mboth_stop": pk(torch.cat([s64, m[64:193]], 0))}
                # the scorer's TOP-16 design: copies of this token's 16 highest-ranked proposals, in rank order
                top = torch.as_tensor(RKtop[i], dtype=torch.long, device=dev)
                C16 = {f: C[f][:, top] for f in FACTORS}
                p16_075 = model.score_trajectories(torch.cat([T64, C16[0.75]], 1), sctx)[0]
                p16_05 = model.score_trajectories(torch.cat([T64, C16[0.5]], 1), sctx)[0]
                p16_both = model.score_trajectories(torch.cat([T64, C16[0.75], C16[0.5]], 1), sctx)[0]
                p16_both_stop = model.score_trajectories(torch.cat([T64, C16[0.75], C16[0.5], stop], 1), sctx)[0]
                m16_075, m16_05 = m[64 + top], m[128 + top]
                picks.update({"k_p16_075": pk(p16_075), "k_p16_05": pk(p16_05), "k_p16_both": pk(p16_both),
                              "k_p16_both_stop": pk(p16_both_stop),
                              "k_m16_075": pk(torch.cat([s64, m16_075], 0)), "k_m16_05": pk(torch.cat([s64, m16_05], 0)),
                              "k_m16_both": pk(torch.cat([s64, m16_075, m16_05], 0)),
                              "k_m16_both_stop": pk(torch.cat([s64, m16_075, m16_05, m[192:193]], 0))})
            R["token"].append(tok)
            for nm, v in (("s64", s64), ("p075", p075), ("p05", p05), ("pboth", pboth), ("pboth_stop", pboth_stop),
                          ("m", m), ("p16_075", p16_075), ("p16_05", p16_05), ("p16_both", p16_both),
                          ("p16_both_stop", p16_both_stop)):
                R[nm].append(v.float().cpu().numpy())
            R["k"].append(int(k))
            for nm, v in picks.items():
                R[nm].append(int(v))
            R["traj_maxdiff"].append(float(np.abs(tr.astype(np.float64) - SD["traj"][i].astype(np.float64)).max()))
            R["copy_maxdiff"].append(d_copy)
            R["id_nomask"].append(float((m_nomask - pboth_stop).abs().max()))
            R["id_m64"].append(float((m[:64] - s64).abs().max()))
            R["ego"].append(planner._ego_vec(ego)[0].double().cpu().numpy())
            R["goal"].append(planner._goal_for(ego)[0].double().cpu().numpy())
            R["calib"].append(planner._calib_for(planner._log_hint)[0].double().cpu().numpy()
                              if planner.per_sample_calib else np.full((4, 16), np.nan))
            R["sec"].append(time.time() - ts)
        if (li + 1) % 10 == 0 or li + 1 == len(logs) or stopped:
            print(f"    [{li + 1}/{len(logs)}] rows {len(R['token']):,}  {time.time() - t0:.0f} s  "
                  f"local {time.strftime('%H:%M:%S')}", flush=True)
    hnd.remove()
    meta = {"rows": len(R["token"]), "stopped": stopped, "gpu_seconds": round(time.time() - t0, 1),
            "limit_logs": a.limit_logs,
            "finished_local": time.strftime("%Y-%m-%dT%H:%M:%S"), "ckpt": CKPT, "rule": planner.rule,
            "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "mask": "rows<64 see the 64 only; each extra member (c075 x64, c05 x64, STOP) sees the 64 and itself",
            "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "NUPLAN_MAPS_ROOT", "REFE_BACKBONE_ROOT")}}
    np.savez(a.gpu_dump, token=np.array(R["token"]),
             **{k: np.stack(R[k]).astype(np.float32) for k in ("s64", "p075", "p05", "pboth", "pboth_stop", "m",
                                                               "p16_075", "p16_05", "p16_both", "p16_both_stop")},
             **{k: np.array(R[k], dtype=np.int64) for k in ("k", "k_s64", "k_p075", "k_p05", "k_pboth",
                                                            "k_pboth_stop", "k_m075", "k_m05", "k_mboth",
                                                            "k_mboth_stop", "k_p16_075", "k_p16_05", "k_p16_both",
                                                            "k_p16_both_stop", "k_m16_075", "k_m16_05", "k_m16_both",
                                                            "k_m16_both_stop")},
             **{k: np.array(R[k], dtype=np.float64) for k in ("traj_maxdiff", "copy_maxdiff", "id_nomask", "id_m64",
                                                              "sec")},
             ego=np.stack(R["ego"]), goal=np.stack(R["goal"]), calib=np.stack(R["calib"]),
             meta=np.array(json.dumps(meta)))
    print(f"  gpu dump {len(R['token'])} rows -> {a.gpu_dump} ({meta['gpu_seconds']} s)", flush=True)
    print("ZZSLOW_GPU_DONE", len(R["token"]))
    return 0


# ------------------------------------------------------------------------------------------------ analyze
# arms: members in SET ORDER (must match the GPU stage's concatenation) and the GPU dump's pick key.
# TOP16 = the scorer's top-16 design (primary: its 32 harness runs are the ones scheduled first);
# ALL = every slot's copy (reported only when all 128 per-slot harness runs PASSed).
ARMS_TOP16 = {
    "masked_top16_075": (["64", "c075_top16"], "k_m16_075"),
    "masked_top16_050": (["64", "c050_top16"], "k_m16_05"),
    "masked_top16_both": (["64", "c075_top16", "c050_top16"], "k_m16_both"),
    "masked_top16_both_stop": (["64", "c075_top16", "c050_top16", "stop"], "k_m16_both_stop"),
    "plain_top16_075": (["64", "c075_top16"], "k_p16_075"),
    "plain_top16_050": (["64", "c050_top16"], "k_p16_05"),
    "plain_top16_both": (["64", "c075_top16", "c050_top16"], "k_p16_both"),
    "plain_top16_both_stop": (["64", "c075_top16", "c050_top16", "stop"], "k_p16_both_stop")}
ARMS_ALL = {
    "masked_all_075": (["64", "c075_all"], "k_m075"), "masked_all_050": (["64", "c050_all"], "k_m05"),
    "masked_all_both": (["64", "c075_all", "c050_all"], "k_mboth"),
    "masked_all_both_stop": (["64", "c075_all", "c050_all", "stop"], "k_mboth_stop"),
    "plain_all_075": (["64", "c075_all"], "k_p075"), "plain_all_050": (["64", "c050_all"], "k_p05"),
    "plain_all_both": (["64", "c075_all", "c050_all"], "k_pboth"),
    "plain_all_both_stop": (["64", "c075_all", "c050_all", "stop"], "k_pboth_stop")}
PLAIN_KEY = {"plain_top16_075": "p16_075", "plain_top16_050": "p16_05", "plain_top16_both": "p16_both",
             "plain_top16_both_stop": "p16_both_stop", "plain_all_075": "p075", "plain_all_050": "p05",
             "plain_all_both": "pboth", "plain_all_both_stop": "pboth_stop"}


def arm_logits(arm, G, top):
    """the logits an arm's pick was taken over (masked arms: the 64's SHIPPED logits + the extra members' masked
    rows), rebuilt here for the float64 cross-check of the planner's torch pick"""
    n = G["s64"].shape[0]
    ar = np.arange(n)[:, None]
    s64, m = G["s64"].astype(np.float64), G["m"].astype(np.float64)
    if arm.startswith("masked_top16"):
        parts = [s64]
        if "075" in arm or "both" in arm:
            parts.append(m[ar, 64 + top])
        if "050" in arm or "both" in arm:
            parts.append(m[ar, 128 + top])
        if arm.endswith("stop"):
            parts.append(m[:, 192:193])
        return np.concatenate(parts, 1)
    if arm.startswith("masked_all"):
        span = {"masked_all_075": (64, 128), "masked_all_050": (128, 192), "masked_all_both": (64, 192),
                "masked_all_both_stop": (64, 193)}[arm]
        return np.concatenate([s64, m[:, span[0]:span[1]]], 1)
    return G[PLAIN_KEY[arm]].astype(np.float64)


def families_arms(poses_by_arm, S0):
    """families6.py (the pipeline's step 4, TANITAD venv, unchanged) on a seam of each arm's pick"""
    import stop_candidate_probe as SCP
    fam_dir = os.path.join(WD, "families")
    os.makedirs(fam_dir, exist_ok=True)
    res = {"instrument": os.path.join(EC.EV6, "families6.py"), "python": EC.TANITAD_PY, "dir": fam_dir,
           "what": "per-arm blocks vs the logged human future (an imitation view), not paired intervals"}
    for arm, poses in poses_by_arm.items():
        sp, op, lg = (os.path.join(fam_dir, f"seam_{arm}.npz"), os.path.join(fam_dir, f"families_{arm}.json"),
                      os.path.join(fam_dir, f"families_{arm}.log"))
        if os.path.exists(op):
            os.remove(op)
        np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=poses.astype(np.float32),
                 sampling=S0["sampling"], arm=np.array(f"REFe_{NAME}_{arm}"))
        rc, _ = EC.run([EC.TANITAD_PY, "families6.py", "--seam", sp, "--inputs", EC.EXPORT, "--stage", "1",
                        "--label", f"REFe-{NAME}-{arm}", "--out", op, "--n-boot", "2000"],
                       EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), lg)
        res[arm] = SCP.summarize_families(op) if os.path.exists(op) else {"status": f"FAILED rc={rc}", "log": lg}
        print(f"  families {arm}: {'OK' if os.path.exists(op) else 'FAILED'}", flush=True)
    return res


def heading_defect_report(SD, T):
    """REFe's heading channel per pose: its step change and its disagreement with the path tangent, for the ep015
    proposals (native 20-pose grid, today's forward), for every banked E-6 table (NAVSIM grid, where the t = 4.0 s
    pose IS the native last pose, copied verbatim by to_navsim), and for the TRAINING targets (a bank sample)."""
    import glob

    def wrap_(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    def tangent_err(P):
        xy = P[..., :2]
        prev = np.concatenate([np.zeros(xy.shape[:-2] + (1, 2)), xy[..., :-1, :]], axis=-2)
        tan = np.arctan2(xy[..., 1] - prev[..., 1], xy[..., 0] - prev[..., 0])
        return np.abs(wrap_(P[..., 2] - tan)), np.linalg.norm(xy - prev, axis=-1) > 0.5

    tr = SD["traj"].astype(np.float64)
    e, mv = tangent_err(tr)
    dh = np.abs(wrap_(np.diff(tr[..., 2], axis=-1)))
    out = {"ep015_native_per_step": {
        "median_abs_heading_minus_tangent_rad": [round(float(np.median(e[..., k][mv[..., k]])), 4) for k in range(20)],
        "median_abs_heading_step_change_rad": [round(float(np.median(dh[..., k])), 4) for k in range(19)],
        "share_abs_heading_above_pi": [round(float((np.abs(tr[..., k, 2]) > np.pi).mean()), 4) for k in range(20)],
        "what": "index k = the pose at t = 0.2 (k + 1) s; tangent = direction of the segment ending at pose k "
                "(moving segments > 0.5 m)"}}
    per = {}
    for p in sorted(glob.glob(f"{EC.DATA}/proptable/sub200_ep*/table.npz")):
        s = os.path.basename(os.path.dirname(p)).replace("sub200_ep", "")
        Tn = np.load(p)
        P = Tn["proposals"].astype(np.float64)
        e_, mv_ = tangent_err(P)
        per[s] = {"share_abs_heading_above_pi_at_4s": round(float((np.abs(P[..., 7, 2]) > np.pi).mean()), 4),
                  "median_abs_heading_minus_tangent_at_4s_rad": round(float(np.median(e_[..., 7][mv_[..., 7]])), 4),
                  "median_abs_heading_minus_tangent_at_3s_rad": round(float(np.median(e_[..., 5][mv_[..., 5]])), 4),
                  "best_of_64": round(100 * float(Tn["pdms"].max(1).mean()), 2)}
    out["per_snapshot_stored_tables"] = per
    X = []
    for pat in ("r0_s*/targets_rank0.jsonl", "aug_s*/targets*.jsonl"):
        for f in sorted(glob.glob(f"D:/Projects/TanitAD/data/refe_navtrain10/{pat}"))[:3]:
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 2000:
                        break
                    X.append(json.loads(line)["traj"])
    if X:
        X = np.asarray(X, dtype=np.float64)
        e_, mv_ = tangent_err(X)
        out["training_targets_sample"] = {
            "rows": int(len(X)), "source": "D:/Projects/TanitAD/data/refe_navtrain10/{r0_s*,aug_s*} (first 3 files "
                                            "each, first 2,000 rows per file)",
            "share_abs_heading_above_pi_last": round(float((np.abs(X[:, -1, 2]) > np.pi).mean()), 4),
            "median_abs_heading_minus_tangent_step18_19_rad": [round(float(np.median(e_[:, k][mv_[:, k]])), 4)
                                                               for k in (18, 19)]}
    return out


def load_copy_scores(tok, slots):
    """harness csvs for [(factor, slot_label)] columns -> pdms [n, S], sub [n, S, 6], valid [n, S], runs PASSed"""
    n = len(tok)
    pd_, sb_, va_ = np.full((n, len(slots)), np.nan), np.full((n, len(slots), 6), np.nan), np.zeros((n, len(slots)), bool)
    passed = 0
    for j, (f, s) in enumerate(slots):
        log = os.path.join(WD, "logs", f"{label(f, s)}.log")
        st = status_of(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
        ok = bool(st and st.get("status") == "PASS" and st.get("csv_valid_rows") == n and os.path.exists(csv_path(f, s)))
        passed += ok
        if not ok:
            continue
        rows = read_csv(csv_path(f, s))
        for i, t in enumerate(tok):
            va_[i, j], pd_[i, j], sb_[i, j] = rows[t]
    return pd_, sb_, va_, passed


def analyze(a) -> int:
    import stop_candidate_probe as SCP
    T = np.load(TABLE)
    G = np.load(a.gpu_dump, allow_pickle=False)
    B = np.load(a.build, allow_pickle=False)          # --build: the gate self-test feeds a mutated copy
    SD = np.load(STOP_DUMP, allow_pickle=False)
    RK = np.load(RANKS, allow_pickle=False)
    S0 = np.load(LANDED_SEAM)
    gmeta = json.loads(str(G["meta"]))
    tok = [str(t) for t in T["token"]]
    n = len(tok)
    ar = np.arange(n)
    order = [str(x) for x in T["head_order"]]
    top = RK["top"]
    out = {"_label": "EXPLORATORY -- same 200 tokens the question was raised on; adoption would need a pre-registered "
                     "test on disjoint tokens and the PI's decision. Report only.",
           "question": "Do time-rescaled (slower) copies of REFe's own proposals, as extra candidates scored by its own "
                       "scorer and picked by the shipped v1 aggregate, raise PDMS without STOP's longitudinal collapse?",
           "snapshot": NAME, "factors": list(FACTORS), "script": os.path.abspath(__file__),
           "construction": "copy_f(t_k) = p(f * t_k), t_k = 0.2 k, k = 1..20, origin (0,0,0) at t = 0; between two "
                           "samples x, y linear in time and heading = start + wrapped difference (wrapped): the seam "
                           "converter's own rule. Every copy pose lies on the original polyline; heading is the "
                           "model's own heading at that point of the path. The float32 20-pose copy is what REFe's "
                           "scorer sees; to_navsim of it is what the harness scores. f = 1 is the identity.",
           "design": {"top16": "per token, copies of the 16 proposals the SHIPPED scorer ranks highest (navsim_v1 "
                               "aggregate of the stored logits, float64, stable sort; rank 0 = the shipped pick): "
                               "16 x 2 factors = 32 harness runs -- the coordinator's fallback, used because 128 runs "
                               "measured ~0.8 runs/min under D: I/O contention (~2.5 h)",
                      "all": "every slot's copy (128 runs); reported only if every run PASSed in time"},
           "gpu_stage": gmeta, "complete": gmeta["rows"] == n and gmeta["stopped"] is None}
    gtok = [str(t) for t in G["token"]]
    same_order = gtok == tok and [str(t) for t in RK["token"]] == tok
    stop_rows = SCP.read_stop(STOP_CSV)
    stop_s = np.array([stop_rows[t][1] for t in tok])
    stop_sub = np.array([stop_rows[t][2] for t in tok])
    # ---------------------------------------------------------------- harness scores of the copies
    top16 = {f: load_copy_scores(tok, [(f, f"r{r:02d}") for r in range(NRANK)]) for f in FACTORS}
    allc = {f: load_copy_scores(tok, [(f, k) for k in range(64)]) for f in FACTORS}
    top16_ok = all(top16[f][3] == NRANK and top16[f][2].all() for f in FACTORS)
    all_ok = all(allc[f][3] == 64 and allc[f][2].all() for f in FACTORS)
    # ---------------------------------------------------------------- controls
    C = {}
    props_sd = np.stack([[SEAM.to_navsim(SD["traj"][i, j]).astype(np.float32) for j in range(64)] for i in range(n)])
    C["C1_dump_proposals_vs_table"] = {"max_abs_diff_m": float(np.abs(props_sd.astype(np.float64)
                                                                      - T["proposals"].astype(np.float64)).max())}
    C["C1_dump_proposals_vs_table"]["pass"] = C["C1_dump_proposals_vs_table"]["max_abs_diff_m"] == 0.0
    dlog = np.abs(G["s64"].astype(np.float64) - T["logits"].astype(np.float64)).max() if same_order else np.inf
    C["C2_gpu_forward_vs_dump_and_table"] = {
        "token_order_identical": same_order, "poses_max_abs_diff_vs_stop_probe_dump": float(G["traj_maxdiff"].max()),
        "logits_64set_max_abs_diff_vs_table": float(dlog),
        "pick_reproduced": f"{int((G['k'] == T['pick']).sum())}/{n}" if same_order else "n/a",
        "pick_reproduced_by_64set_scoring": f"{int((G['k_s64'] == T['pick']).sum())}/{n}" if same_order else "n/a",
        "copies_rebuilt_on_gpu_vs_build_max_abs_diff": float(G["copy_maxdiff"].max())}
    C["C2_gpu_forward_vs_dump_and_table"]["pass"] = bool(
        same_order and G["traj_maxdiff"].max() == 0 and dlog == 0 and (G["k"] == T["pick"]).all()
        and (G["k_s64"] == T["pick"]).all() and G["copy_maxdiff"].max() == 0)
    cp1 = copies_of(SD["traj"].astype(np.float32), (1.0,))[1.0]
    ctrl = {}
    for slot, pdt, subt in (("pick", T["pdms"][ar, T["pick"]], T["sub"][ar, T["pick"]]),
                            (CONTROL_SLOT, T["pdms"][:, CONTROL_SLOT], T["sub"][:, CONTROL_SLOT])):
        p = csv_path(1.0, slot)
        if os.path.exists(p):
            rows = read_csv(p)
            d = np.array([abs(rows[t][1] - pdt[i]) for i, t in enumerate(tok)])
            ds = np.array([np.abs(np.array(rows[t][2]) - subt[i]).max() for i, t in enumerate(tok)])
            ctrl[str(slot)] = {"max_abs_score_diff": float(d.max()), "max_abs_subscore_diff": float(ds.max()),
                               "tokens_identical": int((d == 0).sum()), "valid": int(sum(rows[t][0] for t in tok))}
        else:
            ctrl[str(slot)] = {"status": "NO_CSV"}
    C["C3_factor1_identity"] = {
        "native_bit_identical": bool(np.array_equal(cp1[0], SD["traj"].astype(np.float32))),
        "navsim_bit_identical_to_table": bool(np.array_equal(cp1[1], T["proposals"])),
        "harness_f1_copy_vs_table": ctrl, "bar": 1e-9}
    C["C3_factor1_identity"]["pass"] = bool(
        C["C3_factor1_identity"]["native_bit_identical"] and C["C3_factor1_identity"]["navsim_bit_identical_to_table"]
        and all(v.get("max_abs_score_diff", 1) <= 1e-9 and v.get("max_abs_subscore_diff", 1) <= 1e-9
                and v.get("valid") == n for v in ctrl.values()))
    C["C4_harness_runs_top16"] = {"runs": len(FACTORS) * NRANK,
                                  "passed_with_200_valid": int(sum(top16[f][3] for f in FACTORS)),
                                  "rank0_is_the_shipped_pick": bool((top[:, 0] == T["pick"]).all()),
                                  "pass": bool(top16_ok and (top[:, 0] == T["pick"]).all())}
    C["C4b_harness_runs_all_slots_NOT_GATED"] = {"runs": len(FACTORS) * 64,
                                                "passed_with_200_valid": int(sum(allc[f][3] for f in FACTORS)),
                                                "complete": bool(all_ok)}
    sd_ix = {t: i for i, t in enumerate(str(x) for x in SD["token"])}
    d_stop = float(np.abs(G["m"][:, 192].astype(np.float64)
                          - SD["s65m"][[sd_ix[t] for t in gtok], 64].astype(np.float64)).max())
    C["C5_masked_decoder_identities"] = {
        "unmasked_reimplementation_vs_score_trajectories_max_abs": float(G["id_nomask"].max()),
        "masked_64_rows_vs_64set_max_abs": float(G["id_m64"].max()),
        "masked_STOP_row_vs_stop_probe_vmask_STOP_row_max_abs": d_stop, "bar": BAR}
    C["C5_masked_decoder_identities"]["pass"] = bool(G["id_nomask"].max() == 0 and G["id_m64"].max() <= BAR
                                                     and d_stop <= BAR)
    # C6: the harness seams were built before the copy moved into refe/slow_copies.py -- the shared function must
    # reproduce every built copy BIT-FOR-BIT, or the harness scores belong to a different construction
    c6 = {ftag(f): bool(np.array_equal(slow_copy(SD["traj"].astype(np.float32), f, dt=SEAM.SRC_DT),
                                       B[f"nat_{ftag(f)}"])) for f in FACTORS}
    nav_ok = {ftag(f): bool(np.array_equal(
        np.stack([SEAM.to_navsim(x).astype(np.float32) for x in B[f"nat_{ftag(f)}"].reshape(-1, 20, 3)]).reshape(
            B[f"nav_{ftag(f)}"].shape), B[f"nav_{ftag(f)}"])) for f in FACTORS}
    C["C6_shared_module_reproduces_the_harness_copies"] = {
        "module": SLOW.__file__, "native_bitwise_equal": c6, "navsim_grid_bitwise_equal": nav_ok,
        "pass": bool(all(c6.values()) and all(nav_ok.values()))}
    C["all_gates_pass"] = bool(all(v["pass"] for v in C.values() if isinstance(v, dict) and "pass" in v))
    out["controls"] = C
    for k_, v in C.items():
        if isinstance(v, dict):
            print(f"  {k_}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items() if kk not in ("harness_f1_copy_vs_table",)),
                  flush=True)
    if not (C["all_gates_pass"] and out["complete"]):
        out["verdict"] = "CONTROL_FAILED or INCOMPLETE -- no PDMS number is reported"
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ZZSLOW_FAIL controls"); return 1
    # ---------------------------------------------------------------- members and arms
    MB = {"64": (T["pdms"].astype(np.float64), T["sub"].astype(np.float64), T["proposals"], np.tile(np.arange(64), (n, 1))),
          "stop": (stop_s[:, None], stop_sub[:, None, :], np.zeros((n, 1, 8, 3), np.float32), np.full((n, 1), -1))}
    for f in FACTORS:
        nm = "c075" if f == 0.75 else "c050"
        nav_all = B[f"nav_{ftag(f)}"]
        MB[f"{nm}_top16"] = (top16[f][0], top16[f][1], nav_all[ar[:, None], top], top)
        if all_ok:
            MB[f"{nm}_all"] = (allc[f][0], allc[f][1], nav_all, np.tile(np.arange(64), (n, 1)))
    ship = T["pdms"][ar, T["pick"]].astype(np.float64)
    tl = json.load(open(a.tokens, encoding="utf-8"))["token_log"]
    logs = np.array([tl[t] for t in tok])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(a.boot)]
    arms, poses_by_arm = {}, {"shipped_rebuilt": T["proposals"][ar, T["pick"]]}
    arm_pdms = {}                                   # per-token PDMS of each arm's pick (for the strata below)
    todo = dict(ARMS_TOP16)
    if all_ok:
        todo.update(ARMS_ALL)
    for arm, (members, key) in todo.items():
        kk = G[key]
        src = np.empty(n, dtype=object)
        col = np.zeros(n, dtype=np.int64)
        for i in range(n):
            off = 0
            for mb in members:
                sz = MB[mb][0].shape[1]
                if kk[i] < off + sz:
                    src[i], col[i] = mb, kk[i] - off
                    break
                off += sz
        p = np.array([MB[src[i]][0][i, col[i]] for i in range(n)])
        sb = np.array([MB[src[i]][1][i, col[i]] for i in range(n)])
        orig_slot = np.array([MB[src[i]][3][i, col[i]] for i in range(n)])
        poses_by_arm[arm] = np.stack([MB[src[i]][2][i, col[i]] for i in range(n)])
        vs, vf = SCP.boot_ci(p - ship, draws), SCP.boot_ci(p - stop_s, draws)
        L = arm_logits(arm, G, top)
        is64 = src == "64"
        sw = ~is64
        rec = {"members": members, "pick_key": key, "pdms": round(100 * p.mean(), 2),
               "minus_shipped": vs["diff"], "ci95_vs_shipped": vs["ci95"], "separated_vs_shipped": vs["separated"],
               "minus_stop_floor": vf["diff"], "ci95_vs_stop_floor": vf["ci95"], "separated_vs_stop_floor": vf["separated"],
               "float64_rule_agrees_with_planner_pick": f"{int((SCP.agg_v1_np(L, order).argmax(1) == kk).sum())}/{n}",
               "picks_by_member": {mb: int((src == mb).sum()) for mb in members},
               "picks_among_64_changed": int((is64 & (col != T["pick"])).sum()),
               "switched_to_extra_member": {
                   "n": int(sw.sum()), "pdms_before_mean": round(100 * ship[sw].mean(), 2) if sw.any() else None,
                   "pdms_after_mean": round(100 * p[sw].mean(), 2) if sw.any() else None,
                   "helped": int((p[sw] > ship[sw]).sum()), "hurt": int((p[sw] < ship[sw]).sum()),
                   "tied": int((p[sw] == ship[sw]).sum()),
                   "copy_of_the_shipped_pick": int((sw & (src != "stop") & (orig_slot == T["pick"])).sum())},
               "navsim_subscores_mean_x100": dict(zip(HEAD, [round(100 * float(x), 2) for x in sb.mean(0)])),
               "_per_token_pick": [f"{src[i]}:{int(orig_slot[i])}" for i in range(n)]}
        arms[arm] = rec
        arm_pdms[arm] = p
    # REFERENCE rows, not selections: the shipped pick itself, time-rescaled, on EVERY token (rank 0 of the top-16
    # design IS the shipped pick -- asserted by C4). Added after the rank copies' harness scores were read: the
    # f = 0.75 copy of the shipped pick alone scored far above the shipped pick, so its four families are the direct
    # test of "more PDMS without the longitudinal collapse".
    for f, nm in ((0.75, "c075_top16"), (0.5, "c050_top16")):
        arm = f"fixed_{nm[:4]}_of_shipped_pick"
        p, sb = MB[nm][0][:, 0], MB[nm][1][:, 0]
        poses_by_arm[arm] = MB[nm][2][:, 0]
        vs, vf = SCP.boot_ci(p - ship, draws), SCP.boot_ci(p - stop_s, draws)
        arms[arm] = {"what": f"NOT a selection: every token takes the factor-{f} copy of its shipped pick "
                             "(post-hoc reference row)",
                     "pdms": round(100 * p.mean(), 2), "minus_shipped": vs["diff"], "ci95_vs_shipped": vs["ci95"],
                     "separated_vs_shipped": vs["separated"], "minus_stop_floor": vf["diff"],
                     "ci95_vs_stop_floor": vf["ci95"], "separated_vs_stop_floor": vf["separated"],
                     "helped": int((p > ship).sum()), "hurt": int((p < ship).sum()), "tied": int((p == ship).sum()),
                     "navsim_subscores_mean_x100": dict(zip(HEAD, [round(100 * float(x), 2) for x in sb.mean(0)]))}
        arm_pdms[arm] = p
    out["arms"] = arms
    # WHERE does it act? tokens split by the ego's speed at t0 (the model's own input) and by how far the logged
    # human drove in 4 s (W3's export) -- the coordinator's reading is "in slow scenes even the slowest option is
    # too fast"; descriptive, same tokens, no interval
    import gzip
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    hum = np.array([float(np.linalg.norm(np.asarray(exp[t]["human_future_poses"], dtype=np.float64)[-1, :2]))
                    for t in tok])
    spd = G["ego"][:, 6].astype(np.float64) if same_order else np.full(n, np.nan)
    strata = {}
    for nm_, v in (("ego_speed_t0_mps", spd), ("human_displacement_4s_m", hum)):
        qs = np.quantile(v, [1 / 3, 2 / 3])
        bins = np.digitize(v, qs)
        st_ = {}
        for b_, lab in enumerate(("low", "mid", "high")):
            msk = bins == b_
            row = {"n": int(msk.sum()), "range": [round(float(v[msk].min()), 2), round(float(v[msk].max()), 2)],
                   "shipped": round(100 * ship[msk].mean(), 2), "stop_floor": round(100 * stop_s[msk].mean(), 2),
                   "best_of_64": round(100 * T["pdms"].max(1)[msk].mean(), 2)}
            for arm_ in ("masked_top16_both", "plain_top16_both", "fixed_c075_of_shipped_pick", "masked_all_both"):
                if arm_ in arms:
                    pp = np.array([float(x) for x in arm_pdms[arm_]])
                    row[arm_] = round(100 * pp[msk].mean(), 2)
            st_[lab] = row
        strata[nm_] = st_
    out["strata_descriptive"] = strata
    # POST-HOC REFERENCE SWEEP: the shipped pick itself at each factor, on every token (f = 1.0 is the control run,
    # 0.75 / 0.5 are rank 0 of the top-16 design -- the same poses). Reported only for factors whose run PASSed.
    sweep = {}
    fac_src = [(1.0, "pick")] + [(f, "pick") for f in SWEEP] + [(0.75, "r00"), (0.5, "r00")]
    for f, s in sorted(fac_src, key=lambda x: -x[0]):
        pd_, sb_, va_, ok_ = load_copy_scores(tok, [(f, s)])
        if not (ok_ == 1 and va_.all()):
            sweep[f"{f:.2f}"] = {"status": "NOT SCORED (run missing or not PASS)"}
            continue
        p = pd_[:, 0]
        vs, vf = SCP.boot_ci(p - ship, draws), SCP.boot_ci(p - stop_s, draws)
        sweep[f"{f:.2f}"] = {"source": label(f, s), "pdms": round(100 * p.mean(), 2), "minus_shipped": vs["diff"],
                             "ci95_vs_shipped": vs["ci95"], "minus_stop_floor": vf["diff"],
                             "ci95_vs_stop_floor": vf["ci95"],
                             "navsim_subscores_mean_x100": dict(zip(HEAD, [round(100 * float(x), 2)
                                                                           for x in sb_[:, 0].mean(0)]))}
        if f not in (1.0, 0.75, 0.5):
            poses_by_arm[f"sweep_{ftag(f)}_of_shipped_pick"] = np.load(seam_path(f, s))["poses"]
    out["sweep_shipped_pick_by_factor"] = {"_label": "POST-HOC reference sweep, not a selection: every token takes "
                                                     "the factor-f copy of its shipped pick", "rows": sweep}
    # ---------------------------------------------------------------- THE CONFOUND: REFe's last-pose heading
    out["last_heading_defect"] = heading_defect_report(SD, T)
    rep = {}
    ref09 = None
    pd09, _, va09, ok09 = load_copy_scores(tok, [(0.9, "pick")])
    if ok09 == 1 and va09.all():
        ref09 = pd09[:, 0]
    for how in ("hold", "tangent"):
        pd_, sb_, va_, ok_ = load_copy_scores(tok, [(1.0, f"pick{how}")])
        if not (ok_ == 1 and va_.all()):
            rep[how] = {"status": "NOT SCORED"}
            continue
        p = pd_[:, 0]
        vs, vf = SCP.boot_ci(p - ship, draws), SCP.boot_ci(p - stop_s, draws)
        rec = {"what": ("the shipped pick at f = 1.0 (positions and speed UNCHANGED) with only the t = 4.0 s heading "
                        "replaced by " + ("the t = 3.8 s heading" if how == "hold" else
                                          "the direction of the last path segment")),
               "pdms": round(100 * p.mean(), 2), "minus_shipped": vs["diff"], "ci95_vs_shipped": vs["ci95"],
               "separated_vs_shipped": vs["separated"], "minus_stop_floor": vf["diff"],
               "ci95_vs_stop_floor": vf["ci95"], "helped": int((p > ship).sum()), "hurt": int((p < ship).sum()),
               "navsim_subscores_mean_x100": dict(zip(HEAD, [round(100 * float(x), 2) for x in sb_[:, 0].mean(0)]))}
        if ref09 is not None:
            rec["f090_copy_minus_this"] = SCP.boot_ci(ref09 - p, draws)
            rec["f090_copy_minus_this_note"] = ("the SPEED effect net of the heading repair: the f = 0.9 copy of the "
                                                "shipped pick never reaches the t = 4.0 s pose, so it carries no "
                                                "corrupted heading either")
        rep[how] = rec
        poses_by_arm[f"repair_{how}_shipped_pick"] = np.load(seam_path(1.0, f"pick{how}"))["poses"]
    out["repair_last_heading_shipped_pick"] = rep
    # the same strata for the speed-unchanged repair and the f = 0.9 copy: is the high-speed gain speed or heading?
    extra = {}
    for nm_, src_ in (("repair_hold", (1.0, "pickhold")), ("sweep_f090", (0.9, "pick"))):
        pd_, _, va_, ok_ = load_copy_scores(tok, [src_])
        if ok_ == 1 and va_.all():
            extra[nm_] = pd_[:, 0]
    for nm_, v in (("ego_speed_t0_mps", spd), ("human_displacement_4s_m", hum)):
        bins = np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
        for b_, lab in enumerate(("low", "mid", "high")):
            msk = bins == b_
            for arm_, pp in extra.items():
                out["strata_descriptive"][nm_][lab][arm_] = round(100 * pp[msk].mean(), 2)
    best64 = T["pdms"].max(1)
    ce = {"shipped": round(100 * ship.mean(), 2), "stop_floor": round(100 * stop_s.mean(), 2),
          "best_of_64": round(100 * best64.mean(), 2),
          "max_best64_stop_reference": round(100 * np.maximum(best64, stop_s).mean(), 2),
          "mean_of_64": round(100 * T["pdms"].mean(), 2)}
    for design, names in (("top16", ("c075_top16", "c050_top16")), ("all", ("c075_all", "c050_all"))):
        if names[0] not in MB:
            continue
        b075, b050 = MB[names[0]][0].max(1), MB[names[1]][0].max(1)
        ce[design] = {"best_copy_075": round(100 * b075.mean(), 2), "best_copy_050": round(100 * b050.mean(), 2),
                      "max_best64_copies075": round(100 * np.maximum(best64, b075).mean(), 2),
                      "max_best64_copies050": round(100 * np.maximum(best64, b050).mean(), 2),
                      "max_best64_both_copies": round(100 * np.maximum.reduce([best64, b075, b050]).mean(), 2),
                      "max_best64_both_copies_stop": round(100 * np.maximum.reduce([best64, b075, b050, stop_s]).mean(), 2),
                      "mean_of_copies_075": round(100 * MB[names[0]][0].mean(), 2),
                      "mean_of_copies_050": round(100 * MB[names[1]][0].mean(), 2)}
    ce["top16"]["copy_of_shipped_pick_075"] = round(100 * MB["c075_top16"][0][:, 0].mean(), 2)
    ce["top16"]["copy_of_shipped_pick_050"] = round(100 * MB["c050_top16"][0][:, 0].mean(), 2)
    ce["top16"]["originals_top16_mean"] = round(100 * T["pdms"][ar[:, None], top].mean(), 2)
    out["ceilings"] = ce
    # the scorer's view of the copies (masked scores: each copy judged against the 64 and itself only)
    agg_m = SCP.agg_v1_np(G["m"].astype(np.float64), order)                # [n, 193]
    agg64 = SCP.agg_v1_np(G["s64"].astype(np.float64), order)
    sc = {"originals_top16": {"pred_agg_mean": round(float(agg64[ar[:, None], top].mean()), 4),
                              "harness_pdms_mean": round(float(T["pdms"][ar[:, None], top].mean()), 4),
                              "corr_pred_vs_harness": round(float(np.corrcoef(agg64[ar[:, None], top].ravel(),
                                                                              T["pdms"][ar[:, None], top].ravel())[0, 1]), 4)}}
    for f, lo in ((0.75, 64), (0.5, 128)):
        nm = "c075_top16" if f == 0.75 else "c050_top16"
        a_ = agg_m[ar[:, None], lo + top]
        h_ = MB[nm][0]
        sc[nm] = {"pred_agg_mean": round(float(a_.mean()), 4), "harness_pdms_mean": round(float(h_.mean()), 4),
                  "corr_pred_vs_harness": round(float(np.corrcoef(a_.ravel(), h_.ravel())[0, 1]), 4),
                  "pred_minus_original_mean": round(float((a_ - agg64[ar[:, None], top]).mean()), 4),
                  "harness_minus_original_mean": round(float((h_ - T["pdms"][ar[:, None], top]).mean()), 4),
                  "share_pred_above_original": round(float((a_ > agg64[ar[:, None], top]).mean()), 4),
                  "share_harness_above_original": round(float((h_ > T["pdms"][ar[:, None], top]).mean()), 4)}
    out["scorer_on_copies"] = sc
    out["navsim_subscores_mean_x100_references"] = {
        "shipped": dict(zip(HEAD, [round(100 * float(x), 2) for x in T["sub"][ar, T["pick"]].mean(0)])),
        "stop_floor": dict(zip(HEAD, [round(100 * float(x), 2) for x in stop_sub.mean(0)]))}
    # the four families of each arm's pick (+ the shipped pick rebuilt, checked against the landed seam first)
    ctl = float(np.abs(poses_by_arm["shipped_rebuilt"].astype(np.float64) - S0["poses"].astype(np.float64)).max())
    fam_arms = {k_: v for k_, v in poses_by_arm.items()
                if k_ in ("shipped_rebuilt",) or k_ in ARMS_TOP16 or k_.startswith(("fixed_", "sweep_", "repair_"))
                or k_ in ARMS_ALL}
    out["four_families"] = ({"status": "NOT RUN (--no-families)"} if a.no_families else families_arms(fam_arms, S0))
    out["four_families"]["control_rebuilt_shipped_vs_landed_seam_max_abs_m"] = ctl
    sj = os.path.join(HERE, "raw", f"e6_{NAME}", "stop_candidate.json")
    if os.path.exists(sj):
        ff = json.load(open(sj, encoding="utf-8")).get("four_families", {})
        out["four_families"]["reference_stop_65set_from_stop_candidate_json"] = ff.get("new_65set")
        out["four_families"]["reference_shipped_banked_by_the_pipeline"] = ff.get("shipped_banked_by_the_pipeline")
    out["provenance"] = {"table_sha256": sha256(TABLE), "build": a.build, "build_sha256": sha256(a.build),
                         "ranks_sha256": sha256(RANKS),
                         "gpu_dump": a.gpu_dump, "gpu_dump_sha256": sha256(a.gpu_dump),
                         "stop_dump_sha256": sha256(STOP_DUMP), "stop_csv_sha256": sha256(STOP_CSV),
                         "script_sha256": sha256(os.path.abspath(__file__)),
                         "shared_module": SLOW.__file__, "shared_module_sha256": sha256(SLOW.__file__),
                         "stop_candidate_probe_sha256": sha256(os.path.join(HERE, "stop_candidate_probe.py"))}
    out["verdict"] = "OK" + ("" if all_ok else " (top16 design; the all-slot design did not complete in time)")
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    for arm, r in arms.items():
        print(f"  {arm:23s} {r['pdms']:6.2f}  vs shipped {r['minus_shipped']:+.2f} [{r['ci95_vs_shipped'][0]:+.2f}, "
              f"{r['ci95_vs_shipped'][1]:+.2f}]  vs STOP {r['minus_stop_floor']:+.2f} [{r['ci95_vs_stop_floor'][0]:+.2f}, "
              f"{r['ci95_vs_stop_floor'][1]:+.2f}]  picks {r.get('picks_by_member', 'reference row')}", flush=True)
    print(f"  wrote {a.out}")
    print("ZZSLOW_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build", "build-ranks", "build-sweep", "build-repair", "score", "gpu", "analyze"))
    ap.add_argument("--set", default="all", choices=("controls", "ranks", "slots", "all", "sweep", "repair"),
                    help="score: which seams, in this order (all = controls, ranks, slots)")
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None, help="score: only these labels")
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--deadline", default="15:45")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-families", action="store_true")
    ap.add_argument("--gpu-dump", default=GPU_DUMP)
    ap.add_argument("--build", default=BUILD, help="analyze: the build npz (the gate self-test passes a mutated copy)")
    ap.add_argument("--limit-logs", type=int, default=0, help="gpu smoke: only the first N logs")
    a = ap.parse_args()
    if a.stage == "build":
        return build(a)
    if a.stage == "build-sweep":
        return build_sweep(a)
    if a.stage == "build-repair":
        return build_repair(a)
    if a.stage == "build-ranks":
        return build_ranks(a)
    if a.stage == "score":
        return score(a)
    if a.stage == "gpu":
        if os.environ.get(CHILD) != "1":
            env = EC.env_driverl()
            env[CHILD] = "1"
            return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=HERE, env=env)
        return gpu_stage(a)
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
