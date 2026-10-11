#!/usr/bin/env python3
"""P4 (eval/PREREG_P4_TEACHER_NAVTEST.md): OUR DriveRL teacher protocol on NAVSIM v1.1 navtest -> a W3 seam npz.

The teacher protocol that built REFe's rank-0 targets, reused UNCHANGED from `refe/build_teacher_rollouts.py`:
`build_planner` (their yaml + the runner's overrides, TTS off, exact caches ON), `build_controller` (bicycle max accel
4.0, max steering rate 0.8), `rollout_from_frame` (ego seeded at the LOGGED state, 2.0 s history read backward out of
the DB, background agents replaying the log, 4.0 s), rank 0 (THEIR route code; DriveRL corrects the route itself on the
first step, feature_builder.py:407-413). The plan is `make_row`'s target geometry -- the rollout's rear-axle poses at
0.2 s in the logged ego frame -- converted to NAVSIM's 8 poses at 0.5 s by `refe_navtest_seam.to_navsim`.

Modes (each a separate process; long runs are launched detached, one process per shard):
  --heldout <json>   control C3: re-roll held-out navtrain samples whose banked teacher target is known; compare.
  --tokens <json> --shard i/N --out-dir <dir>
                     roll navtest tokens (W3-format subset json, or the all-tokens json); writes
                     <out-dir>/rows_s<i>of<N>.jsonl, one line per token, resumable (tokens already written are skipped).
  --merge --tokens <json> --out-dir <dir> --seam <out.npz>
                     merge every shard file into ONE seam (W3 export order) + <seam>.report.json; prints
                     ZZTSEAM_OK <rows> <c1_unexpected> only when C1 passes (the 36 known DB-vs-OpenScene tokens excepted)
                     and coverage misses <= 1 %.
Frame/time control C1 runs in the same breath as every rollout (the logged future on NAVSIM's 0.5 s grid vs W3's
`human_future_poses`, 1 mm), exactly as refe_navtest_seam.py computes it.

Interpreter: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe, environment = eval_checkpoint.env_driverl() with
D:/ -> E:/ (see run_p4.py). CPU only.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REFE = os.path.join(PKG, "refe")
sys.path.insert(0, REFE)
sys.path.insert(0, HERE)

from refe_navtest_seam import FRAME_CONTROL_MAX_M, NAVSIM_DT, NAVSIM_N, to_navsim, wrap  # noqa: E402

EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
TRAINVAL_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
# the 36 tokens in 4 logs whose nuPlan-DB future disagrees with OpenScene (student's full seam, 2026-10-04)
C1_KNOWN = os.path.join(PKG, "raw", "2026-10-04-navtest-final", "frame_control_navtest_full.json")
DB_ROW_DT = 0.05                     # the raw nuPlan lidar_pc stream is 20 Hz (navtrain_scenarios, MEASURED)

_S: dict = {}


def _init(device: str, threads: int, tts: int = 0) -> None:
    import warnings
    # log noise only: nuplan_map/utils.py:413 casts a NaN id column on every map query (smoke log: hundreds of lines)
    warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
    import torch
    torch.set_num_threads(max(1, threads))
    import route_rank
    _S["route"] = route_rank.apply(0)            # rank 0 = THEIR unpatched route; refuses a stale patch
    import build_teacher_rollouts as BTR
    BTR.TEACHER_DEVICE = device
    _S["BTR"] = BTR
    _S["device"] = device
    _S["planner"] = BTR.build_planner(device)
    if tts:
        # ⭐ P5's re-query mechanism, REPORT-ONLY follow-on (PREREG_P4 §9): DriveRL's own value-guided test-time action
        # search (driverl/nuplan/tts.py). The agent and the selector are built lazily at the FIRST initialize()
        # (planner.py:124-210), so setting the config here, before any initialize, is what switches it on.
        # The REGISTERED P4 arm is tts = 0 (build_planner's tts_enabled = False, exactly as the bank was built).
        _S["planner"].config.tts_enabled = True
        _S["planner"].config.tts_num_candidates = int(tts)
    _S["tts"] = int(tts)
    import navtrain_scenarios as NS
    _S["NS"] = NS
    print(f"  {_S['route']}; device {device}; torch threads {torch.get_num_threads()}; "
          f"REFE_SIM_HZ {NS.SIM_HZ}; TTS {'N=' + str(tts) if tts else 'OFF'}", flush=True)


def _ego_frame_poses(states, px, py, pyaw):
    c, s = math.cos(-pyaw), math.sin(-pyaw)
    return np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s,
                      (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c,
                      wrap(f.rear_axle.heading - pyaw)] for f in states], dtype=np.float64)


def roll(sc, pos_in_log: int | None):
    """One teacher rollout from iteration 0 of `sc` -> dict (traj20, exact8, goal, buffer_s, history_clamped, ...).

    `pos_in_log` = the token's 0-based index among the log's lidar_pc rows (20 Hz). The devkit reads the 2.0 s history
    BACKWARD out of the DB and dies on row 0's NULL prev_token (navtrain_scenarios: measured strictly-greater guard),
    so a token closer than 2.0 s to its log's start gets the longest buffer the DB holds, and is FLAGGED.
    """
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    BTR, planner = _S["BTR"], _S["planner"]
    stride, span, _hist = BTR.geometry_for(sc)
    dt = float(sc.database_interval)
    buffer_s = BTR.BUFFER_S
    clamped = False
    if pos_in_log is not None:
        rows_ok = pos_in_log - 1                                 # rows strictly before the guard row
        max_s = math.floor(rows_ok * DB_ROW_DT / dt + 1e-9) * dt
        if max_s + 1e-9 < buffer_s:
            buffer_s, clamped = round(max_s, 6), True
    controller = BTR.build_controller(sc, _S["device"])         # per scenario (their requirement)
    planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                             mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
    poses = BTR.rollout_from_frame(sc, planner, controller, 0, span, buffer_s=buffer_s)
    if poses is None:
        raise RuntimeError(f"scenario too short for span {span} (n={sc.get_number_of_iterations()})")
    st = sc.get_ego_state_at_iteration(0)
    px, py, pyaw = st.rear_axle.x, st.rear_axle.y, st.rear_axle.heading
    sel = poses[stride - 1::stride][:BTR.HORIZON]                # == make_row
    if len(sel) < BTR.HORIZON:
        raise RuntimeError(f"rollout gave {len(sel)} target samples < {BTR.HORIZON}")
    traj = _ego_frame_poses(sel, px, py, pyaw)                   # [20, 3] at 0.2 s == the bank's `traj`
    k05 = int(round(NAVSIM_DT / dt))
    exact = _ego_frame_poses(poses[k05 - 1::k05][:NAVSIM_N], px, py, pyaw)   # report only
    goal = getattr(BTR.rollout_from_frame, "last_goal", None)
    v = st.dynamic_car_state.rear_axle_velocity_2d
    return {"traj20": traj, "exact8": exact, "goal": goal, "buffer_s": buffer_s, "history_clamped": clamped,
            "v0": float(math.hypot(v.x, v.y)), "stride": stride, "span": span, "dt": dt,
            "corrected_route_n": len(getattr(planner._feature_builder, "_corrected_route_roadblock_ids", None) or []),
            "raw_route_n": len(list(sc.get_route_roadblock_ids()))}


def _log_positions(db_path: str) -> dict:
    _map, rows = _S["NS"]._log_rows(db_path)
    return {t: i for i, (t, _) in enumerate(rows)}


# ----------------------------------------------------------------------------------------------- C3 held-out control
def heldout(a) -> int:
    _init(a.device, a.threads, a.tts)
    NS = _S["NS"]
    q = json.load(open(a.heldout, encoding="utf-8"))
    by: dict = {}
    for k, r in q.items():
        by.setdefault(r["log_name"], []).append(r)
    out, t0 = [], time.time()
    for lg in sorted(by):
        rows = by[lg]
        db = os.path.join(TRAINVAL_DB, f"{lg}.db")
        pos = _log_positions(db)
        # the bank's own call: navtrain_scenarios defaults (history/future rows derived from the log's interval)
        scs = {sc.scenario_name: sc for sc in NS.build_scenarios_for_log(db, sorted({r["token"] for r in rows}))}
        for r in rows:
            rec = {"log": lg, "token": r["token"], "rank": r.get("rank", 0), "ckpt_step": r.get("ckpt_step")}
            try:
                if int(r.get("step", 0)) != 0:
                    raise RuntimeError("only step-0 (per-token scenario) samples are comparable")
                sc = scs[r["token"]]
                t1 = time.time()
                g = roll(sc, pos.get(r["token"]))
                ref = np.asarray(r["teacher"], dtype=np.float64)
                d = np.linalg.norm(g["traj20"][:, :2] - ref[:, :2], axis=1)
                rec.update({"max_dxy_m": float(d.max()), "mean_dxy_m": float(d.mean()),
                            "end_dxy_m": float(d[-1]),
                            "max_dyaw_rad": float(np.abs(wrap(g["traj20"][:, 2] - ref[:, 2])).max()),
                            "buffer_s": g["buffer_s"], "history_clamped": g["history_clamped"], "v0": g["v0"],
                            "end_ref": ref[-1, :2].tolist(), "end_mine": g["traj20"][-1, :2].tolist(),
                            "sec": round(time.time() - t1, 2)})
            except Exception as e:                                                   # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out.append(rec)
            print(json.dumps({k: rec.get(k) for k in ("token", "max_dxy_m", "end_dxy_m", "sec", "error")}), flush=True)
    ok_rows = [r for r in out if "max_dxy_m" in r]
    m = np.array([r["max_dxy_m"] for r in ok_rows]) if ok_rows else np.array([np.inf])
    rep = {"what": "C3: my rollout of the released teacher vs the BANKED rank-0 teacher target (on-policy queue field "
                   "`teacher`), max |dxy| over the 20 poses at 0.2 s",
           "rate_hz": _S["NS"].SIM_HZ, "device": a.device, "n": len(out), "n_ok": len(ok_rows),
           "n_error": len(out) - len(ok_rows),
           "median_max_dxy_m": float(np.median(m)), "p90_max_dxy_m": float(np.percentile(m, 90)),
           "max_max_dxy_m": float(m.max()), "share_le_0p25": float((m <= 0.25).mean()),
           "over_0p25": [r for r in ok_rows if r["max_dxy_m"] > 0.25],
           "bar": "median <= 0.05 m AND >= 90 % of samples <= 0.25 m (PREREG_P4 §4 C3)",
           "seconds": round(time.time() - t0, 1), "rows": out}
    rep["pass"] = bool(rep["n_error"] == 0 and rep["median_max_dxy_m"] <= 0.05 and rep["share_le_0p25"] >= 0.90)
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(rep, open(a.report, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("rows", "over_0p25")}, indent=1))
    print(f"ZZC3_{'PASS' if rep['pass'] else 'FAIL'} n={rep['n_ok']} median={rep['median_max_dxy_m']:.4f} "
          f"share<=0.25={rep['share_le_0p25']:.3f}")
    return 0 if rep["pass"] else 1


# ------------------------------------------------------------------------------------------------- navtest rollouts
def load_tokens(path: str, exp: dict) -> list:
    toks = list(exp)                                  # W3 export order
    if path:
        sub = json.load(open(path, encoding="utf-8"))
        want = set(sub["tokens"] if isinstance(sub, dict) else sub)
        toks = [t for t in toks if t in want]
    return toks


def shard_logs(by_log: dict, i: int, n: int) -> list:
    """Deterministic balanced partition BY LOG (greedy on token count, ties by name) -- the same for every shard."""
    load = [0] * n
    assign: dict = {}
    for lg in sorted(by_log, key=lambda x: (-len(by_log[x]), x)):
        j = min(range(n), key=lambda k: (load[k], k))
        assign[lg] = j
        load[j] += len(by_log[lg])
    return sorted(lg for lg, j in assign.items() if j == i)


def rollouts(a) -> int:
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = load_tokens(a.tokens, exp)
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    i, n = (int(x) for x in a.shard.split("/"))
    logs = shard_logs(by_log, i, n)
    os.makedirs(a.out_dir, exist_ok=True)
    out_p = os.path.join(a.out_dir, f"rows_s{i}of{n}.jsonl")
    done = set()
    if os.path.exists(out_p):
        for ln in open(out_p, encoding="utf-8"):
            try:
                done.add(json.loads(ln)["token"])
            except (json.JSONDecodeError, KeyError):
                continue                               # a torn last line is simply re-derived
    todo_n = sum(len([t for t in by_log[lg] if t not in done]) for lg in logs)
    print(f"  shard {i}/{n}: {len(logs)} logs, {sum(len(by_log[l]) for l in logs)} tokens, {todo_n} to do "
          f"({len(done)} already written) -> {out_p}", flush=True)
    if todo_n == 0:
        print("ZZSHARD_DONE", i, n, 0, flush=True)
        return 0
    _init(a.device, a.threads, a.tts)
    NS = _S["NS"]
    # terminate a torn last line before appending (build_teacher_rollouts.open_append)
    f = _S["BTR"].open_append(out_p)
    t0, k = time.time(), 0
    for lg in logs:
        lt = [t for t in by_log[lg] if t not in done]
        if not lt:
            continue
        db = os.path.join(a.db_dir, f"{lg}.db")
        got = set()
        try:
            pos = _log_positions(db)
            scen = NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80)   # == refe_navtest_seam.py
            for sc in scen:
                tok = sc._initial_lidar_token
                got.add(tok)
                rec = {"token": tok, "log": lg, "tts": _S["tts"]}
                t1 = time.time()
                try:
                    g = roll(sc, pos.get(tok))
                    plan = to_navsim(g["traj20"])
                    if not np.isfinite(plan).all():
                        raise RuntimeError("non-finite plan")
                    rec.update({"plan8": plan.tolist(), "traj20": g["traj20"].tolist(), "exact8": g["exact8"].tolist(),
                                "plan_vs_exact_max_m": float(np.linalg.norm(plan[:, :2] - g["exact8"][:, :2],
                                                                            axis=1).max()),
                                "goal": g["goal"], "buffer_s": g["buffer_s"], "history_clamped": g["history_clamped"],
                                "v0": g["v0"], "corrected_route_n": g["corrected_route_n"],
                                "raw_route_n": g["raw_route_n"], "dt": g["dt"]})
                except Exception as e:                                               # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    rec["trace"] = traceback.format_exc()[-400:]
                    _S["BTR"]._reset_planner_state(_S["planner"])
                # C1 frame/time control, same breath, identical to refe_navtest_seam.py
                try:
                    ego = sc.get_ego_state_at_iteration(0)
                    theirs = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)
                    f8 = list(sc.get_ego_future_trajectory(0, 4.0, NAVSIM_N))[:NAVSIM_N]
                    if len(f8) == NAVSIM_N:
                        loc = _ego_frame_poses(f8, ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading)
                        rec["c1_err_m"] = float(np.linalg.norm(loc[:, :2] - theirs[:, :2], axis=1).max())
                except Exception as e:                                               # noqa: BLE001
                    rec["c1_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                rec["sec"] = round(time.time() - t1, 2)
                f.write(json.dumps(rec) + "\n")
                k += 1
        except Exception as e:                                                       # noqa: BLE001
            print(f"    LOG_ERROR {lg}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        for tok in set(lt) - got:
            f.write(json.dumps({"token": tok, "log": lg, "error": "scenario builder skipped the token (margin guard)"})
                    + "\n")
            k += 1
        f.flush()
        os.fsync(f.fileno())
        el = time.time() - t0
        print(f"    {lg[:38]:38s} tokens {len(lt):4d}  total {k:5d}/{todo_n}  {el:7.0f}s  "
              f"{el / max(k, 1):5.2f} s/token", flush=True)
    f.close()
    for mod in ("map_feature_cache", "query_cache", "map_step_speedups"):
        if mod in sys.modules:
            print(f"  {sys.modules[mod].summary()}", flush=True)
    print("ZZSHARD_DONE", i, n, k, flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------ merge
def merge(a) -> int:
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = load_tokens(a.tokens, exp)
    rows: dict = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "rows_s*of*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r["token"] in rows and "plan8" in rows[r["token"]]:
                continue                                   # a good row is never replaced by a later error row
            rows[r["token"]] = r
    known = set(json.load(open(C1_KNOWN, encoding="utf-8")).get("over_bar", {}))
    seam_t, seam_fp, seam_p, misses = [], [], [], []
    c1 = {}
    for t in toks:
        r = rows.get(t)
        if r is None:
            misses.append((t, "not rolled"))
            continue
        if "c1_err_m" in r:
            c1[t] = r["c1_err_m"]
        if "plan8" not in r:
            misses.append((t, r.get("error", "no plan")))
            continue
        seam_t.append(t)
        seam_fp.append(exp[t]["fingerprint"])
        seam_p.append(np.asarray(r["plan8"], dtype=np.float32))
    over = {t: e for t, e in c1.items() if e > FRAME_CONTROL_MAX_M}
    unexpected = sorted(t for t in over if t not in known)
    no_c1 = sorted(t for t in toks if t in rows and t not in c1)
    os.makedirs(os.path.dirname(os.path.abspath(a.seam)), exist_ok=True)
    np.savez(a.seam, token=np.array(seam_t), fingerprint=np.array(seam_fp),
             poses=np.stack(seam_p) if seam_p else np.zeros((0, NAVSIM_N, 3), np.float32),
             sampling=np.array([NAVSIM_N, NAVSIM_DT]), arm=np.array(a.arm))
    sec = [r["sec"] for r in rows.values() if "sec" in r and "plan8" in r]
    pve = [r["plan_vs_exact_max_m"] for r in rows.values() if "plan_vs_exact_max_m" in r]
    rep = {"tokens_asked": len(toks), "rows": len(seam_t), "misses": len(misses), "miss_examples": misses[:20],
           "miss_rate": len(misses) / max(len(toks), 1),
           "c1_frame_control": {"n": len(c1), "bar_m": FRAME_CONTROL_MAX_M,
                                "max_m": max(c1.values()) if c1 else None,
                                "median_m": float(np.median(list(c1.values()))) if c1 else None,
                                "n_over_bar": len(over), "n_over_bar_known36": len([t for t in over if t in known]),
                                "unexpected_over_bar": unexpected[:50], "n_unexpected": len(unexpected),
                                "n_rolled_without_c1": len(no_c1), "known_list": C1_KNOWN},
           "history_clamped": sum(1 for r in rows.values() if r.get("history_clamped")),
           "plan_vs_exact_0p5s": {"n": len(pve), "max_m": max(pve) if pve else None,
                                  "median_m": float(np.median(pve)) if pve else None,
                                  "what": "to_navsim(20 poses at 0.2 s) vs the rollout's own states at 0.5 s multiples"},
           "sec_per_token_mean": float(np.mean(sec)) if sec else None,
           "arm": a.arm, "export": EXPORT}
    json.dump(rep, open(os.path.splitext(a.seam)[0] + ".report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "miss_examples"}, indent=1))
    ok = (not unexpected) and not no_c1 and rep["miss_rate"] <= 0.01 and len(c1) > 0
    print(f"ZZTSEAM_{'OK' if ok else 'FAIL'} {len(seam_t)} {len(unexpected)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--seam", default=None)
    ap.add_argument("--arm", default="TEACHER_driverl_u2400_rank0")
    ap.add_argument("--db-dir", default=TEST_DB)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tts", type=int, default=0, help="REPORT-ONLY follow-on: DriveRL test-time search with N candidates; 0 = the registered arm")
    a = ap.parse_args()
    if a.heldout:
        return heldout(a)
    if a.merge:
        return merge(a)
    if not a.out_dir:
        print("  --out-dir is required"); return 2
    return rollouts(a)


if __name__ == "__main__":
    sys.exit(main())
