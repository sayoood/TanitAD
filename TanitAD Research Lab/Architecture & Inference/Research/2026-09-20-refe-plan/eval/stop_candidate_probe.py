#!/usr/bin/env python3
"""EXPLORATORY (report only, dev box): can a STOP candidate recover REFe's lost ceiling?  (snapshot 015, sub200)

MOTIVATION (`raw/e6_sub200_ep015/oracle_trend.json`, `lost_modes_ep012_ep015.json`): the best of the 64 proposals
fell 91.33 -> 84.19 PDMS from snapshot 012 to 015 because the proposal set lost its slow/short options. W3's STOP
floor beats the best of 64 on 16 of these 200 tokens, and max(best of 64, STOP) averages 88.75.
QUESTION: if the planner scored a 65th candidate, STOP, TOGETHER with its 64 proposals using its OWN scorer and picked
by the shipped NAVSIM-v1 aggregate, what PDMS would its pick reach on the same 200 tokens?

No harness run is needed. The proposals are unchanged, so a pick among the 64 takes the E-6 table's harness PDMS
(`table.pdms[token, idx]`), and a STOP pick takes W3's STOP arm's harness score for that token -- scored by the same
v1.1 tree through the same `run_v1.py cmd_score` and `metric_cache_navtest` as the table's per-proposal runs.

INPUT PATH -- REUSED, NOT REWRITTEN. The loop is `refe_navtest_seam.py`'s own: the same W3 export and token filter,
`navtrain_scenarios.build_scenarios_for_log(history_rows=1, future_rows=80)`, `REFePlanner.initialize`, `_image_for`
and `infer` (the planner's own model call and pick), in the interpreter and environment `eval_checkpoint.py` runs the
seam with (`EC.DRIVERL_PY`, `EC.env_driverl()`; this script re-launches itself under them -- `REFE_SIM_HZ=10` there
is load-bearing, the scenario builder's default is 20). The scoring decoder's visual context `sctx` is captured from
THAT forward pass by a pre-hook on `score_dec[0]` (observation only; no model code is touched), and a set of
trajectories is scored by `REFe.score_trajectories(trajs, sctx)` -- the function `REFe.forward` itself calls for
`score_extra`. The decoder's self-attention spans the set, so the 65-set (64 proposals + STOP) is scored TOGETHER.

STOP'S FORM. W3's `w3_agents_v1.StopAgent` submits `np.zeros((8, 3))`: 8 x (x fwd, y left, heading) = (0, 0, 0) at
t = 0.5 .. 4.0 s, i.e. the t0 rear-axle pose. REFe's own format is 20 x (x, y, heading) at t = 0.2 .. 4.0 s in the
same rear-axle ego frame (the seam's frame control ties REFe's frame to NAVSIM's), so STOP enters the scorer as
`zeros(20, 3)`, and `to_navsim(zeros(20, 3))` must equal the submitted `zeros(8, 3)` bit-for-bit (control C-e).

CONTROLS -- gates; no PDMS number is reported unless every gate passes:
  C-a  the forward reproduces the stored proposals: to_navsim of all 64, float32, every token (bar 1e-4 m = E-6 G1)
  C-b  the 64-set scored alone reproduces the stored logits (bar 1e-4) AND the stored pick on every token, through
       `score_trajectories` (the route the 65-set uses) and through `infer`'s own output
  C-c  the route is the model's own: a second, full forward with `score_extra=<65-set>` gives the same 65 scores
       (bar 1e-4; skipped with --no-extra-forward)
  C-e  STOP's form: the StopAgent source line, its default sampling (4 s / 0.5 s = 8 poses), the tensor actually fed
       (all zeros, REFe's [horizon_steps, traj_dim]) and to_navsim(fed) == zeros(8, 3) bit-for-bit; the STOP csv
       holds every token, valid; the STOP run's Hydra agent passes no trajectory_sampling and its artifact records
       `np.zeros((8, 3))`
  C-f  (--rescore-stop) zeros(8, 3) scored TODAY through the table's own path (score_navtest_refe.py) reproduces
       the banked STOP csv on every token (bar 1e-9, E-6 G2's) -- the substitution of that csv is like for like
Reported, not gated: C-d permutation (STOP first instead of last -- the set decoder carries no positional encoding,
so only float summation-order noise may appear) and the second forward's poses/logits vs the first (inference
determinism).

TWO POST-HOC VARIANTS, same forward, scoring decoder only (added after a 3-token smoke showed the 65th member moving
the 64 proposals' logits by up to 2.5). They are NOT the question asked, carry no bar, and are labelled as such:
  V-dup   the 65th member is a DUPLICATE of the shipped pick (an in-distribution trajectory): how far does ANY extra
          member move the 64? -- the reference magnitude for the STOP shift
  V-mask  STOP attends to the 64 but the 64 may NOT attend to STOP (a boolean self-attention mask, one direction), so
          the 64 keep their shipped scores by construction and only STOP's own score is new -- the lever that removes
          the shift. The decoder is re-implemented for the mask (CrossBlock's three residual lines), so it carries two
          identity checks: unmasked it must reproduce `score_trajectories` on the 65-set, and masked its first 64 rows
          must reproduce the 64-set scored alone (bar 1e-4 each; the variant is withheld if either fails).
  V-gate  (analysis only) take STOP when a scorer statistic crosses a threshold, the threshold chosen
          LEAVE-ONE-LOG-OUT; a hypothesis generator for a pre-registered test, never a result.

FOUR FAMILIES (analysis, complete dumps only; `--no-families` skips it): the eval pipeline's own step 4,
`families6.py` in the TANITAD venv, unchanged, on a seam of each arm's pick (the table's proposal, or zeros(8, 3) for
STOP), with the shipped pick rebuilt the same way and checked bit-for-bit against the LANDED seam first. Per-arm
blocks vs the logged human future (an imitation view), not paired intervals.

    python eval/stop_candidate_probe.py                      # GPU stage + analysis -> raw/e6_sub200_ep015/stop_candidate.json
    python eval/stop_candidate_probe.py --analyze-only       # re-analyse the banked dump, no GPU
    python eval/stop_candidate_probe.py --limit-logs 2 --out <tmp.json> --dump <tmp.npz>     # smoke
Prints ZZSTOPPROBE_OK <new_pdms> <diff> <lo> <hi> or ZZSTOPPROBE_FAIL <why>.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  one source for the venv + env the seam ran with
import refe_navtest_seam as SEAM    # noqa: E402  to_navsim, the export, the test DBs; puts refe/ on sys.path

NAVTEST = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest"
TOK = f"{NAVTEST}/raw/A1_sub200_tokens.json"
STOP_CSV = f"{NAVTEST}/raw/STOP_navtest/STOP_navtest.csv"
STOP_SRC = f"{NAVTEST}/code/w3_agents_v1.py"
STOP_LINE = "poses = np.zeros((self._trajectory_sampling.num_poses, 3), dtype=np.float32)"
STOP_SAMPLING_LINE = "_DEF_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)"
STOP_MANIFEST = f"{NAVTEST}/raw/STOP_navtest/STOP_navtest_manifest.json"
STOP_ARTIFACT = f"{NAVTEST}/raw/artifact_STOP_navtest.json"
STOP_AGENT_YAML = ("D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/"
                   "script/config/common/agent/constant_velocity_agent.yaml")   # the Hydra agent the STOP run used
NAME = "sub200_ep015"
TABLE = f"{EC.DATA}/proptable/{NAME}/table.npz"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/snap_epoch015.pt"
OUT = os.path.join(HERE, "raw", f"e6_{NAME}", "stop_candidate.json")
DUMP = f"{EC.DATA}/proptable/{NAME}/stop_candidate_dump.npz"
SEED = 20260927          # eval/snapshot_pair_under_rule.py's bootstrap seed, copied with its estimator
BAR = 1e-4
CHILD = "REFE_STOP_PROBE_CHILD"
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def agg_v1_np(logits, order):
    """float64 copy of the planner's navsim_v1 rule (as in eval/snapshot_pair_under_rule.py) -- a CROSS-CHECK of
    the planner's own torch pick, never the pick itself"""
    p = sig(np.asarray(logits, dtype=np.float64))
    g = {k: p[..., order.index(k)] for k in HEAD}
    return g["NC"] * g["DAC"] * (5 * g["EP"] + 5 * g["TTC"] + 2 * g["C"]) / 12


def score_masked(model, trajs, sctx, mask):
    """`REFe.score_trajectories` with a self-attention mask (V-mask only). The loop body is CrossBlock.forward's
    three residual lines with `attn_mask` added; `mask=None` must reproduce `score_trajectories` (checked per token)."""
    s = model.score_q_mlp(trajs.flatten(2))
    for blk in model.score_dec:
        h = blk.n1(s)
        s = s + blk.self_attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = blk.n2(s)
        s = s + blk.cross(h, sctx, sctx, need_weights=False)[0]
        s = s + blk.mlp(blk.n3(s))
    return model.score_head(s)


def past(deadline: str) -> bool:
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return (lt.tm_hour, lt.tm_min) >= (hh, mm)


# ------------------------------------------------------------------------------------------------ GPU stage
def gpu_stage(a) -> int:
    import torch
    import navtrain_scenarios as NS
    import planner as PL
    import model as MD
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    t_start = time.time()
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    toks = list(exp)
    sub = json.load(open(a.tokens, encoding="utf-8"))
    want = set(sub["tokens"] if isinstance(sub, dict) else sub)
    toks = [t for t in toks if t in want]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log.items())
    if a.limit_logs:
        logs = logs[:a.limit_logs]
    n_asked = sum(len(v) for _, v in logs)
    print(f"  tokens: {n_asked:,} over {len(logs)} logs; ckpt {a.ckpt}", flush=True)
    planner = REFePlanner(checkpoint=a.ckpt, images_root=a.frames, db_dir=a.db_dir, backbone="vitl16",
                          device="cuda", select="best", rule=None)
    model = planner.model
    print(f"  planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} "
          f"device={planner.device} select={planner.select} rule={planner.rule}", flush=True)
    if planner.device != "cuda":
        print("ZZSTOPPROBE_FAIL no_cuda"); return 1
    H, Dm = int(model.cfg.horizon_steps), int(model.cfg.traj_dim)
    cap = {"on": False}

    def pre_hook(_mod, args):                    # observation only: the scoring decoder's context
        if cap["on"]:
            cap["sctx"] = args[1]
    hnd = model.score_dec[0].register_forward_pre_hook(pre_hook)
    keys = ("token", "props", "traj", "logits_infer", "pick_infer", "s64", "s65", "s65_stopfirst", "ext65",
            "k64", "k65", "k65_among64", "agg64", "agg65", "traj2_maxdiff", "score2_maxdiff", "sec",
            "s65m", "s65m_nomask", "k65m", "agg65m", "s65dup")
    R: dict = {k: [] for k in keys}
    misses, stopped = [], None
    stop_fed = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for li, (log, lt) in enumerate(logs):
        if stopped:
            break
        db = os.path.join(a.db_dir, f"{log}.db")
        got = set()
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            if past(a.deadline):
                stopped = f"deadline {a.deadline} local reached at {time.strftime('%H:%M:%S')}"
                break
            ts = time.time()
            tok = sc._initial_lidar_token
            got.add(tok)
            planner._scenario = sc
            planner.initialize(PlannerInitialization(
                route_roadblock_ids=sc.get_route_roadblock_ids(),
                mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            cap.pop("sctx", None)
            cap["on"] = True
            traj, score, k = planner.infer(ego, img)          # the planner's OWN call and pick, unchanged
            cap["on"] = False
            sctx = cap.pop("sctx")
            with torch.no_grad():
                T64 = traj[None]                                                   # [1, 64, H, D]
                stop = torch.zeros(1, 1, H, Dm, dtype=traj.dtype, device=traj.device)
                T65 = torch.cat([T64, stop], 1)                                    # STOP is index 64
                s64 = model.score_trajectories(T64, sctx)[0]
                s65 = model.score_trajectories(T65, sctx)[0]
                s65f = model.score_trajectories(torch.cat([stop, T64], 1), sctx)[0]
                s65f = torch.cat([s65f[1:], s65f[:1]], 0)                         # back to STOP-last order
                k64 = planner._pick(s64[None])
                k65 = planner._pick(s65[None])                                     # the SHIPPED v1 aggregate
                agg64 = planner.aggregate(s64[None])[0]
                agg65 = planner.aggregate(s65[None])[0]
                k65_64 = int(agg65[:64].argmax())
                # V-mask: the 64 may not attend to STOP (row < 64, column 64); STOP attends to all 65
                mask = torch.zeros(65, 65, dtype=torch.bool, device=traj.device)
                mask[:64, 64] = True
                s65m = score_masked(model, T65, sctx, mask)[0]
                s65m_nomask = score_masked(model, T65, sctx, None)[0]
                k65m = planner._pick(s65m[None])
                agg65m = planner.aggregate(s65m[None])[0]
                # V-dup: the 65th member is a copy of the shipped pick
                s65dup = model.score_trajectories(torch.cat([T64, T64[:, k:k + 1]], 1), sctx)[0]
                if a.no_extra_forward:
                    ext = torch.full_like(s65, float("nan"))
                    d_traj2 = d_score2 = float("nan")
                else:
                    # C-c: the model's OWN score_extra route, same inputs as infer builds them
                    ego_vec = planner._ego_vec(ego)
                    goal = planner._goal_for(ego).to(planner.device)
                    calib = planner._calib_for(planner._log_hint) if planner.per_sample_calib else None
                    traj2, score2, ext = model(img.to(planner.device), ego_vec, goal, calib=calib,
                                               score_extra=T65)
                    ext = ext[0]
                    d_traj2 = float((traj2[0] - traj).abs().max())
                    d_score2 = float((score2[0] - score).abs().max())
            if stop_fed is None:
                stop_fed = stop[0, 0].float().cpu().numpy()
            R["token"].append(tok)
            R["props"].append(np.stack([SEAM.to_navsim(traj[j].float().cpu().numpy())
                                        for j in range(traj.shape[0])]).astype(np.float32))
            R["traj"].append(traj.float().cpu().numpy())
            R["logits_infer"].append(score.float().cpu().numpy())
            R["pick_infer"].append(int(k))
            for nm, v in (("s64", s64), ("s65", s65), ("s65_stopfirst", s65f), ("ext65", ext),
                          ("agg64", agg64), ("agg65", agg65), ("s65m", s65m), ("s65m_nomask", s65m_nomask),
                          ("agg65m", agg65m), ("s65dup", s65dup)):
                R[nm].append(v.float().cpu().numpy())
            R["k64"].append(int(k64))
            R["k65"].append(int(k65))
            R["k65m"].append(int(k65m))
            R["k65_among64"].append(k65_64)
            R["traj2_maxdiff"].append(d_traj2)
            R["score2_maxdiff"].append(d_score2)
            R["sec"].append(time.time() - ts)
        for tok in set(lt) - got:
            if not stopped:
                misses.append((tok, "scenario builder skipped the token (margin guard)"))
        if (li + 1) % 5 == 0 or li + 1 == len(logs) or stopped:
            print(f"    [{li + 1}/{len(logs)}] rows {len(R['token']):,}  misses {len(misses)}  "
                  f"{time.time() - t0:.0f} s  local {time.strftime('%H:%M:%S')}", flush=True)
    hnd.remove()
    n = len(R["token"])
    if n == 0:
        print("ZZSTOPPROBE_FAIL no_rows"); return 1
    import nuplan
    import driverl
    meta = {"n_asked": n_asked, "rows": n, "misses": misses[:20], "n_misses": len(misses), "stopped": stopped,
            "limit_logs": a.limit_logs, "extra_forward": not a.no_extra_forward,
            "gpu_started_local": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
            "gpu_finished_local": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu_seconds": round(time.time() - t_start, 1),
            "sec_per_token_median": round(float(np.median(R["sec"])), 3),
            "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "device_name": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "python": sys.executable, "planner_rule": planner.rule, "planner_select": planner.select,
            "per_sample_calib": bool(planner.per_sample_calib), "ckpt_format": str(getattr(planner, "ckpt_format", "")),
            "horizon_steps": H, "traj_dim": Dm, "traj_dt_s": float(PL.TRAJ_DT_S), "seam_src_dt_s": float(SEAM.SRC_DT),
            "modules": {"model": MD.__file__, "planner": PL.__file__, "navtrain_scenarios": NS.__file__,
                        "refe_navtest_seam": SEAM.__file__, "nuplan": nuplan.__file__, "driverl": driverl.__file__},
            "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "NUPLAN_MAPS_ROOT", "NUPLAN_DATA_ROOT",
                                                    "REFE_BACKBONE_ROOT", "OMP_NUM_THREADS", "PYTHONPATH")},
            "ckpt": a.ckpt, "frames": a.frames, "db_dir": a.db_dir, "export": a.export, "tokens": a.tokens}
    os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
    np.savez(a.dump, token=np.array(R["token"]),
             **{k: np.stack(R[k]).astype(np.float32) for k in ("props", "traj", "logits_infer", "s64", "s65",
                                                               "s65_stopfirst", "ext65", "agg64", "agg65",
                                                               "s65m", "s65m_nomask", "agg65m", "s65dup")},
             **{k: np.array(R[k], dtype=np.int64) for k in ("pick_infer", "k64", "k65", "k65_among64", "k65m")},
             **{k: np.array(R[k], dtype=np.float64) for k in ("traj2_maxdiff", "score2_maxdiff", "sec")},
             stop_fed=stop_fed, meta=np.array(json.dumps(meta)))
    print(f"  dump: {n} rows -> {a.dump}  ({meta['gpu_seconds']} s, max mem {meta['cuda_max_memory_allocated_gb']} GB)",
          flush=True)
    print("ZZSTOPPROBE_GPU_DONE", n, flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ analysis
def read_stop(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def boot_ci(d, draws):
    bs = np.array([100 * d[ix].mean() for ix in draws])
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    return {"diff": round(100 * float(d.mean()), 2), "ci95": [round(lo, 2), round(hi, 2)],
            "separated": bool(lo > 0 or hi < 0)}


def rescore_stop(a, dtok, stop_rows):
    """C-f: the STOP plan (zeros(8, 3)) scored TODAY through the same path the E-6 table's proposals were scored
    by (`score_navtest_refe.py` -> W3's `run_v1.cmd_score`, SeamAgentV1, `metric_cache_navtest`) must reproduce
    the banked STOP csv per token -- so substituting that csv's score for a STOP pick is like for like.
    Resumable: a PASSed run whose csv exists is not re-run."""
    label = f"refe_{NAME}_stopzeros"
    wd = os.path.join(os.path.dirname(os.path.abspath(a.dump)), "stop_candidate_rescore")
    os.makedirs(wd, exist_ok=True)
    S0 = np.load(a.landed_seam)
    if [str(t) for t in S0["token"]] != dtok:
        return {"pass": False, "note": "the landed seam's token order differs from the dump's"}
    seam = os.path.join(wd, "seam_stopzeros.npz")
    np.savez(seam, token=S0["token"], fingerprint=S0["fingerprint"],
             poses=np.zeros((len(dtok), 8, 3), dtype=np.float32), sampling=S0["sampling"],
             arm=np.array(f"REFe_{NAME}_stopzeros"))
    csvp = f"{EC.DATA}/score/{label}/{label}.csv"
    log = os.path.join(wd, "rescore_stop.log")

    def status(txt):
        for line in reversed(txt.splitlines()):
            if line.startswith("{") and '"status"' in line:
                return json.loads(line)
        return None
    st = status(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
    if not (st and st.get("status") == "PASS" and os.path.exists(csvp)):
        _rc, txt = EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", label, "--seam", seam,
                           "--tokens", a.tokens, "--out", f"{EC.DATA}/score"],
                          HERE, dict(os.environ, PYTHONIOENCODING="utf-8"), log)
        st = status(txt)
    if not (st and st.get("status") == "PASS" and os.path.exists(csvp)):
        return {"pass": False, "note": f"the harness run did not PASS (see {log})", "status": st}
    got = read_stop(csvp)
    d = np.array([abs(got[t][1] - stop_rows[t][1]) for t in dtok])
    ds = np.array([np.abs(np.array(got[t][2]) - np.array(stop_rows[t][2])).max() for t in dtok])
    return {"what": "zeros(8, 3) through score_navtest_refe.py today vs the banked STOP_navtest.csv, per token",
            "label": label, "csv": csvp, "log": log, "harness_status": st.get("status"),
            "csv_valid_rows": st.get("csv_valid_rows"), "valid": f"{sum(got[t][0] for t in dtok)}/{len(dtok)}",
            "max_abs_score_diff": float(d.max()), "max_abs_subscore_diff": float(ds.max()),
            "tokens_identical": int((d == 0).sum()), "bar": 1e-9,
            "pass": bool(all(got[t][0] for t in dtok) and d.max() <= 1e-9)}


def summarize_families(p):
    """the headline numbers of one `families6.py` block (the arm under test is keyed `refcv6` by that script)"""
    d = json.load(open(p, encoding="utf-8"))
    r = d["families"]["refcv6"]
    lon, lat, tac = r.get("longitudinal") or {}, r.get("lateral") or {}, r.get("tactical") or {}

    def kap(k):
        return (tac.get(k) or {}).get("kappa")
    return {"n": d.get("n"), "tier": d.get("tier"),
            "longitudinal": {k: lon.get(k) for k in ("speed_mae_mps", "speed_bias_mps", "along_mae_m",
                                                      "along_final_bias_m", "accel_mae_mps2")}
            | {"target_speed_within_1mps": (lon.get("target_speed_acc") or {}).get("within_1.0_mps"),
               "progress_ratio_mean": (lon.get("ego_progress") or {}).get("progress_ratio_mean"),
               "under_progress_rate": (lon.get("ego_progress") or {}).get("under_progress_rate"),
               "distance_keeping": (lon.get("distance_keeping") or {}).get("status")},
            "lateral": {k: lat.get(k) for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm",
                                                 "cross_mae_m", "cross_final_mae_m")},
            "tactical": {"lateral_decision_kappa": kap("lateral_decision"),
                         "longitudinal_decision_kappa": kap("longitudinal_decision"),
                         "maneuver_5way_kappa": kap("maneuver_5way_collapsed"),
                         "goal_point_error_m": (tac.get("goal_setting") or {}).get("goal_point_error_m")},
            "strategic": (r.get("strategic") or {}).get("status")}


def families_step(a, dtok, T, ri, pick, k65, k65m):
    """the eval pipeline's step 4 (`families6.py`, TANITAD venv, unchanged) on seams of each arm's pick: the table's
    proposal for a pick among the 64, zeros(8, 3) for STOP. Per-arm blocks vs the logged human future -- NOT paired
    intervals, and an imitation view: a STOP pick is scored against a human who may have driven on."""
    fam_dir = os.path.join(os.path.dirname(os.path.abspath(a.dump)), "stop_candidate_families")
    os.makedirs(fam_dir, exist_ok=True)
    res = {"what": "the four metric families (families6.py) of each arm's pick, vs the logged human future; per-arm "
                   "blocks, not paired intervals",
           "instrument": os.path.join(EC.EV6, "families6.py"), "python": EC.TANITAD_PY, "dir": fam_dir,
           "landed_seam": a.landed_seam}
    S0 = np.load(a.landed_seam)
    if [str(t) for t in S0["token"]] != dtok:
        res["status"] = "FAILED: the landed seam's token order differs from the dump's"
        return res
    n = len(dtok)
    ar = np.arange(n)
    P = T["proposals"][ri].astype(np.float32)

    def poses_of(k):
        o = np.zeros((n, 8, 3), dtype=np.float32)
        m = k < 64
        o[m] = P[ar[m], k[m]]
        return o
    # control: the shipped pick rebuilt from the table must equal the LANDED seam bit-for-bit
    res["control_rebuilt_shipped_vs_landed_seam_max_abs_m"] = float(
        np.abs(poses_of(pick).astype(np.float64) - S0["poses"].astype(np.float64)).max())
    for arm, k in (("shipped_rebuilt", pick), ("new_65set", k65), ("post_hoc_vmask", k65m)):
        sp = os.path.join(fam_dir, f"seam_{arm}.npz")
        op = os.path.join(fam_dir, f"families_{arm}.json")
        lg = os.path.join(fam_dir, f"families_{arm}.log")
        if os.path.exists(op):
            os.remove(op)                          # never read a stale block as this run's
        np.savez(sp, token=S0["token"], fingerprint=S0["fingerprint"], poses=poses_of(k), sampling=S0["sampling"],
                 arm=np.array(f"REFe_{NAME}_{arm}"))
        rc, _ = EC.run([EC.TANITAD_PY, "families6.py", "--seam", sp, "--inputs", EC.EXPORT, "--stage", "1",
                        "--label", f"REFe-{NAME}-{arm}", "--out", op, "--n-boot", "2000"],
                       EC.EV6, dict(os.environ, PYTHONIOENCODING="utf-8"), lg)
        res[arm] = summarize_families(op) if os.path.exists(op) else {"status": f"FAILED rc={rc}", "log": lg}
        print(f"  families {arm}: {'OK' if os.path.exists(op) else 'FAILED (see ' + lg + ')'}", flush=True)
    res["shipped_banked_by_the_pipeline"] = (summarize_families(a.landed_families)
                                             if os.path.exists(a.landed_families) else None)
    return res


def analyze(a) -> int:
    D = np.load(a.dump, allow_pickle=False)
    meta = json.loads(str(D["meta"]))
    T = np.load(a.table)
    ttok = [str(t) for t in T["token"]]
    dtok = [str(t) for t in D["token"]]
    order = [str(x) for x in T["head_order"]]
    rule = str(T["rule"]) if "rule" in T.files else "v2_shape"
    n = len(dtok)
    tix = {t: i for i, t in enumerate(ttok)}
    missing_in_table = [t for t in dtok if t not in tix]
    out = {"_label": "EXPLORATORY -- same 200 tokens the question was raised on; adoption would need a "
                     "pre-registered test on disjoint tokens and the PI's decision. Report only.",
           "question": "If the planner scored a 65th candidate, STOP, together with its 64 proposals using its OWN "
                       "scorer and picked by the shipped NAVSIM-v1 aggregate, what PDMS would its pick reach?",
           "snapshot": NAME, "table": a.table, "dump": os.path.abspath(a.dump), "script": os.path.abspath(__file__),
           "n_rows": n, "n_table": len(ttok), "complete": (n == len(ttok) and not missing_in_table
                                                            and meta.get("stopped") is None),
           "token_order_identical_to_table": dtok == ttok, "gpu_stage": meta}
    if missing_in_table:
        out["verdict"] = "FAIL_TOKENS_NOT_IN_TABLE"
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ZZSTOPPROBE_FAIL tokens_not_in_table"); return 1
    ri = np.array([tix[t] for t in dtok])
    ar = np.arange(n)
    pick = T["pick"][ri]
    pd = T["pdms"][ri].astype(np.float64)                       # [n, 64] harness PDMS per proposal
    L_st = T["logits"][ri].astype(np.float64)
    stop_rows = read_stop(a.stop_csv)

    # ---------------------------------------------------------------- controls
    C = {}
    dpose = np.abs(D["props"].astype(np.float64) - T["proposals"][ri].astype(np.float64)).reshape(n, -1).max(1)
    C["C_a_proposals_reproduced"] = {
        "what": "to_navsim of all 64 proposals from this run's forward vs the stored table.proposals (float32)",
        "max_abs_diff_m": float(dpose.max()), "tokens_bit_identical": int((dpose == 0).sum()), "n": n,
        "bar_m": BAR, "pass": bool(dpose.max() <= BAR)}
    d_s64 = np.abs(D["s64"].astype(np.float64) - L_st).reshape(n, -1).max(1)
    d_inf = np.abs(D["logits_infer"].astype(np.float64) - L_st).reshape(n, -1).max(1)
    rep_np = agg_v1_np(L_st, order).argmax(1) == pick
    C["C_b_logits_and_pick_reproduced"] = {
        "what": "the 64-set scored alone (score_trajectories on the captured context -- the 65-set's route) and "
                "infer's own logits, vs the stored table.logits; picks by the planner's own _pick vs table.pick",
        "max_abs_logit_diff_score64_vs_stored": float(d_s64.max()),
        "tokens_bit_identical_score64": int((d_s64 == 0).sum()),
        "max_abs_logit_diff_infer_vs_stored": float(d_inf.max()),
        "max_abs_logit_diff_score64_vs_infer": float(np.abs(D["s64"].astype(np.float64)
                                                            - D["logits_infer"].astype(np.float64)).max()),
        "pick_reproduced_score64": f"{int((D['k64'] == pick).sum())}/{n}",
        "pick_reproduced_infer": f"{int((D['pick_infer'] == pick).sum())}/{n}",
        "pick_reproduced_float64_rule_on_stored_logits": f"{int(rep_np.sum())}/{n}",
        "table_rule": rule, "planner_rule": meta.get("planner_rule"), "bar": BAR,
        "pass": bool(d_s64.max() <= BAR and d_inf.max() <= BAR and (D["k64"] == pick).all()
                     and (D["pick_infer"] == pick).all() and rule == meta.get("planner_rule") == "navsim_v1")}
    if meta.get("extra_forward"):
        d_ext = np.abs(D["ext65"].astype(np.float64) - D["s65"].astype(np.float64)).reshape(n, -1).max(1)
        C["C_c_score_extra_route"] = {
            "what": "a second full forward with score_extra=<the 65-set> (REFe.forward's own route) vs the 65-set "
                    "scored on the captured context",
            "max_abs_diff": float(d_ext.max()), "tokens_bit_identical": int((d_ext == 0).sum()), "n": n,
            "bar": BAR, "pass": bool(np.isfinite(d_ext).all() and d_ext.max() <= BAR)}
        C["inference_determinism_second_forward"] = {
            "what": "the second forward's poses and 64 logits vs the first (REFe's forward samples nothing); "
                    "reported, not gated",
            "max_abs_pose_diff_native": float(np.nanmax(D["traj2_maxdiff"])),
            "max_abs_logit_diff": float(np.nanmax(D["score2_maxdiff"]))}
    else:
        C["C_c_score_extra_route"] = {"pass": None, "note": "skipped (--no-extra-forward)"}
    d_perm = np.abs(D["s65_stopfirst"].astype(np.float64) - D["s65"].astype(np.float64)).reshape(n, -1).max(1)
    k65f = agg_v1_np(D["s65_stopfirst"], order).argmax(1)
    C["C_d_permutation"] = {
        "what": "STOP scored FIRST instead of last (set decoder has no positional encoding); reported, not gated",
        "max_abs_logit_diff": float(d_perm.max()), "tokens_bit_identical": int((d_perm == 0).sum()),
        "pick_identical": f"{int((k65f == D['k65']).sum())}/{n}"}
    src = open(a.stop_src, encoding="utf-8").read()
    fed = np.asarray(D["stop_fed"])
    conv = SEAM.to_navsim(fed.astype(np.float64)).astype(np.float32)
    submitted = np.zeros((8, 3), dtype=np.float32)
    stop_valid = [t for t in dtok if t in stop_rows and stop_rows[t][0]]
    C["C_e_stop_form"] = {
        "submitted_by_STOP_arm": "w3_agents_v1.StopAgent.compute_trajectory -> np.zeros((num_poses, 3)), "
                                 "default TrajectorySampling(time_horizon=4, interval_length=0.5) = 8 poses of "
                                 "(x fwd, y left, heading) = (0, 0, 0) at the t0 rear-axle pose",
        "stop_agent_source": a.stop_src, "stop_agent_sha256": sha256(a.stop_src),
        "source_line_present": STOP_LINE in src, "default_sampling_line_present": STOP_SAMPLING_LINE in src,
        "fed_to_scorer": f"zeros[{fed.shape[0]}, {fed.shape[1]}] on REFe's {meta.get('traj_dt_s')} s grid "
                         f"(horizon_steps {meta.get('horizon_steps')}, traj_dim {meta.get('traj_dim')}), same ego frame",
        "fed_all_zero": bool((fed == 0).all()),
        "fed_shape_is_model_format": [int(x) for x in fed.shape] == [meta.get("horizon_steps"), meta.get("traj_dim")],
        "to_navsim_of_fed_equals_submitted_bitwise": bool(conv.shape == submitted.shape
                                                          and np.array_equal(conv, submitted)),
        "stop_csv": a.stop_csv, "stop_csv_sha256": sha256(a.stop_csv),
        "stop_csv_tokens_valid": f"{len(stop_valid)}/{n}"}
    # the STOP run's own records: its Hydra agent config passes NO trajectory_sampling (so the class default --
    # 8 poses -- applied), its artifact states the mechanism, and the source predates the run
    ce = C["C_e_stop_form"]
    yml_txt = open(STOP_AGENT_YAML, encoding="utf-8").read() if os.path.exists(STOP_AGENT_YAML) else None
    man = json.load(open(STOP_MANIFEST, encoding="utf-8")) if os.path.exists(STOP_MANIFEST) else {}
    art = json.load(open(STOP_ARTIFACT, encoding="utf-8")) if os.path.exists(STOP_ARTIFACT) else {}
    mech = str(((art.get("protocol") or {}).get("ego_status_enforcement") or {}).get("mechanism", ""))
    ce["stop_run_agent_yaml"] = STOP_AGENT_YAML
    ce["stop_run_agent_yaml_passes_no_trajectory_sampling"] = (yml_txt is not None
                                                               and "trajectory_sampling" not in yml_txt)
    ce["stop_run_overrides"] = [o for o in man.get("overrides", []) if o.startswith("agent")]
    ce["stop_run_artifact_mechanism"] = mech
    ce["stop_run_artifact_says_zeros_8x3"] = "np.zeros((8, 3))" in mech
    ce["stop_agent_source_mtime_local"] = time.strftime("%Y-%m-%dT%H:%M:%S",
                                                        time.localtime(os.path.getmtime(a.stop_src)))
    ce["stop_run_started_local"] = man.get("started_local")
    ce["pass"] = bool(
        ce["source_line_present"] and ce["default_sampling_line_present"] and ce["fed_all_zero"]
        and ce["fed_shape_is_model_format"] and ce["to_navsim_of_fed_equals_submitted_bitwise"]
        and len(stop_valid) == n and meta.get("traj_dt_s") == meta.get("seam_src_dt_s") == 0.2
        and ce["stop_run_agent_yaml_passes_no_trajectory_sampling"] and ce["stop_run_artifact_says_zeros_8x3"]
        and "agent._target_=w3_agents_v1.StopAgent" in ce["stop_run_overrides"])
    C["C_f_stop_rescored_through_the_refe_path"] = (rescore_stop(a, dtok, stop_rows) if a.rescore_stop else
                                                   {"pass": None, "note": "not run (--rescore-stop)"})
    gates = [C["C_a_proposals_reproduced"]["pass"], C["C_b_logits_and_pick_reproduced"]["pass"],
             C["C_c_score_extra_route"]["pass"] is not False, C["C_e_stop_form"]["pass"],
             C["C_f_stop_rescored_through_the_refe_path"]["pass"] is not False]
    C["all_gates_pass"] = bool(all(gates))
    out["controls"] = C
    for k, v in C.items():
        if isinstance(v, dict):
            print(f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in v.items()
                                          if kk not in ("what", "submitted_by_STOP_arm", "stop_agent_source",
                                                        "stop_csv", "fed_to_scorer", "stop_run_agent_yaml",
                                                        "stop_run_artifact_mechanism")), flush=True)
    if not C["all_gates_pass"]:
        out["verdict"] = "CONTROL_FAILED -- no PDMS number is reported"
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
        print("ZZSTOPPROBE_FAIL controls"); return 1

    # ---------------------------------------------------------------- the question
    stop_s = np.array([stop_rows[t][1] for t in dtok], dtype=np.float64)
    stop_sub = np.array([stop_rows[t][2] for t in dtok], dtype=np.float64)
    k65 = D["k65"]
    is_stop = k65 == 64
    ship = pd[ar, pick]
    new = np.where(is_stop, stop_s, pd[ar, np.minimum(k65, 63)])
    best = pd.max(1)
    k65_np = agg_v1_np(D["s65"], order).argmax(1)
    tl = json.load(open(a.tokens, encoding="utf-8"))["token_log"]
    logs = np.array([tl[t] for t in dtok])
    ul = np.unique(logs)
    idx = {l: np.where(logs == l)[0] for l in ul}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx[l] for l in rng.choice(ul, size=len(ul), replace=True)]) for _ in range(a.boot)]
    head = boot_ci(new - ship, draws)
    res = {"rule": "navsim_v1 (planner.aggregate: NC x DAC x (5 EP + 5 TTC + 2 C) / 12 on the sigmoids), "
                   "picked by the planner's own _pick on the 65 logits",
           "shipped_pick_pdms": round(100 * ship.mean(), 2), "new_pick_pdms": round(100 * new.mean(), 2),
           "new_minus_shipped": head["diff"], "ci95": head["ci95"], "separated": head["separated"],
           "estimator": f"paired log-cluster bootstrap (percentile 2.5/97.5), {a.boot} resamples of the "
                        f"{len(ul)} logs, seed {SEED} -- eval/snapshot_pair_under_rule.py's estimator, copied",
           "variance_answered": "EPISODES (another draw of logs). Not training variance and not inference "
                                "variance: one checkpoint, one deterministic forward.",
           "float64_rule_agrees_with_planner_pick": f"{int((k65_np == k65).sum())}/{n}",
           "references_same_tokens": {
               "best_of_64": round(100 * best.mean(), 2),
               "stop_floor_alone": round(100 * stop_s.mean(), 2),
               "max_best_of_64_and_stop": round(100 * np.maximum(best, stop_s).mean(), 2),
               "switch_oracle_max_shipped_and_stop": round(100 * np.maximum(ship, stop_s).mean(), 2),
               "switch_oracle_note": "a perfect STOP/no-STOP decision on top of the SHIPPED pick -- the most this "
                                     "lever can add without changing the pick among the 64"},
           "stop_floor_minus_shipped": boot_ci(stop_s - ship, draws),
           "new_minus_stop_floor": boot_ci(new - stop_s, draws)}
    out["result"] = res
    # switches to STOP
    sw = np.where(is_stop)[0]
    out["switch_to_stop"] = {
        "n": int(len(sw)), "pdms_before_mean": round(100 * ship[sw].mean(), 2) if len(sw) else None,
        "pdms_after_mean": round(100 * stop_s[sw].mean(), 2) if len(sw) else None,
        "helped": int((stop_s[sw] > ship[sw]).sum()), "hurt": int((stop_s[sw] < ship[sw]).sum()),
        "tied": int((stop_s[sw] == ship[sw]).sum()),
        "sum_delta_pdms_points_over_200": round(100 * float((stop_s[sw] - ship[sw]).sum()) / n, 3),
        "tokens": [{"token": dtok[i], "log": str(logs[i]), "shipped_pick": int(pick[i]),
                    "pdms_before": round(100 * ship[i], 2), "stop_pdms": round(100 * stop_s[i], 2),
                    "best_of_64": round(100 * best[i], 2),
                    "stop_agg": round(float(D["agg65"][i, 64]), 4),
                    "best_agg_among_64": round(float(D["agg65"][i, :64].max()), 4)} for i in sw]}
    # where STOP is genuinely better
    b16 = np.where(stop_s > best)[0]
    out["stop_beats_best_of_64"] = {
        "n": int(len(b16)), "scorer_picks_stop": int(is_stop[b16].sum()),
        "pdms_shipped_mean": round(100 * ship[b16].mean(), 2) if len(b16) else None,
        "pdms_new_mean": round(100 * new[b16].mean(), 2) if len(b16) else None,
        "stop_rank_among_65": [int((D["agg65"][i, :64] > D["agg65"][i, 64]).sum()) for i in b16],
        "tokens": [dtok[i] for i in b16]}
    better = stop_s > ship
    out["stop_decision_confusion"] = {
        "what": "STOP vs the SHIPPED pick, per token, against whether the scorer picked STOP",
        "stop_better_than_shipped": int(better.sum()), "of_which_picked_stop": int((better & is_stop).sum()),
        "stop_worse_than_shipped": int((stop_s < ship).sum()),
        "of_which_picked_stop_harmful": int(((stop_s < ship) & is_stop).sum()),
        "tied": int((stop_s == ship).sum())}
    # the 65th candidate's effect on the 64
    s64 = D["s64"].astype(np.float64)
    s65 = D["s65"].astype(np.float64)
    dl = np.abs(s65[:, :64] - s64)
    dag = np.abs(D["agg65"][:, :64].astype(np.float64) - D["agg64"].astype(np.float64))
    ch64 = np.where(D["k65_among64"] != pick)[0]
    ch_real = np.where((~is_stop) & (k65 != pick))[0]
    out["shift_of_the_64_from_the_65th"] = {
        "what": "the 64 proposals' logits scored in the 65-set vs scored alone, same forward, same context",
        "logit_abs_shift_max": float(dl.max()), "logit_abs_shift_mean": float(dl.mean()),
        "logit_abs_shift_max_per_head": {h: float(dl[..., j].max()) for j, h in enumerate(order)},
        "logit_abs_shift_mean_per_head": {h: float(dl[..., j].mean()) for j, h in enumerate(order)},
        "agg_abs_shift_max": float(dag.max()), "agg_abs_shift_mean": float(dag.mean()),
        "logit_shift_vs_stored_logits_max": float(np.abs(s65[:, :64] - L_st).max()),
        "argmax_among_64_changed": int(len(ch64)),
        "argmax_among_64_changed_tokens": [{"token": dtok[i], "old": int(pick[i]), "new": int(D["k65_among64"][i]),
                                            "pdms_old": round(100 * pd[i, pick[i]], 2),
                                            "pdms_new": round(100 * pd[i, D["k65_among64"][i]], 2),
                                            "final_pick_is_stop": bool(is_stop[i])} for i in ch64],
        "final_pick_is_a_different_proposal": int(len(ch_real))}
    # decomposition of the change
    unchanged = (k65 == pick)
    out["decomposition"] = {
        "unchanged_tokens": int(unchanged.sum()), "switched_to_stop": int(is_stop.sum()),
        "switched_among_64": int(len(ch_real)),
        "points_from_stop_switches": round(100 * float((new - ship)[is_stop].sum()) / n, 3),
        "points_from_switches_among_64": round(100 * float((new - ship)[ch_real].sum()) / n, 3),
        "total_points": round(100 * float((new - ship).sum()) / n, 3)}
    # the scorer's view of STOP against the harness's
    ps = sig(s65[:, 64, :])
    agg_stop = D["agg65"][:, 64].astype(np.float64)
    rank = (D["agg65"][:, :64] > D["agg65"][:, 64:65]).sum(1)
    out["scorer_on_stop"] = {
        "what": "STOP's predicted sub-scores (sigmoid of its logits in the 65-set) vs W3's harness sub-scores",
        "pred_mean": {h: round(float(ps[:, j].mean()), 4) for j, h in enumerate(order)},
        "harness_mean": {h: round(float(stop_sub[:, j].mean()), 4) for j, h in enumerate(order)},
        "pred_agg_mean": round(float(agg_stop.mean()), 4), "harness_pdms_mean": round(float(stop_s.mean()), 4),
        "corr_pred_agg_vs_harness_pdms": round(float(np.corrcoef(agg_stop, stop_s)[0, 1]), 4),
        "rank_among_65_median": float(np.median(rank)), "rank_among_65_min": int(rank.min()),
        "rank_note": "number of proposals the scorer prefers over STOP (0 = STOP picked)"}
    # ---------------------------------------------------------------- post-hoc variants (not the question asked)
    V = {"_label": "POST-HOC VARIANTS -- added after a 3-token smoke showed the 65th member moving the 64's logits by "
                   "up to 2.5; NOT the question asked, no bar, EXPLORATORY on the same 200 tokens; any use needs a "
                   "pre-registration on disjoint tokens"}
    dup = D["s65dup"].astype(np.float64)
    ddup = np.abs(dup[:, :64] - s64)
    kdup64 = agg_v1_np(dup[:, :64], order).argmax(1)
    V["V_dup_65th_is_a_copy_of_the_shipped_pick"] = {
        "what": "an IN-DISTRIBUTION 65th member: how far does any extra member move the 64?",
        "logit_abs_shift_max": float(ddup.max()), "logit_abs_shift_mean": float(ddup.mean()),
        "argmax_among_64_changed_float64_rule": int((kdup64 != pick).sum()),
        "stop_as_65th_for_comparison": {"logit_abs_shift_max": float(dl.max()),
                                        "logit_abs_shift_mean": float(dl.mean()),
                                        "argmax_among_64_changed": int(len(ch64))}}
    sm = D["s65m"].astype(np.float64)
    id1 = float(np.abs(D["s65m_nomask"].astype(np.float64) - s65).max())
    id2 = float(np.abs(sm[:, :64] - s64).max())
    vm = {"what": "STOP attends to the 64, the 64 may not attend to STOP (one-directional self-attention mask): the 64 "
                  "keep their shipped scores, only STOP's own score is new; picked by the planner's own _pick",
          "identity_unmasked_reimplementation_vs_score_trajectories_max_abs": id1,
          "identity_masked_first64_vs_64set_alone_max_abs": id2, "bar": BAR,
          "identities_pass": bool(id1 <= BAR and id2 <= BAR)}
    k65m = D["k65m"]
    is_m = k65m == 64
    newm = np.where(is_m, stop_s, pd[ar, np.minimum(k65m, 63)])
    if vm["identities_pass"]:
        cm = boot_ci(newm - ship, draws)
        swm = np.where(is_m)[0]
        rank_m = (D["agg65m"][:, :64] > D["agg65m"][:, 64:65]).sum(1)
        vm.update({
            "new_pick_pdms": round(100 * newm.mean(), 2), "new_minus_shipped": cm["diff"], "ci95": cm["ci95"],
            "separated": cm["separated"], "minus_stop_floor": boot_ci(newm - stop_s, draws),
            "switch_to_stop": int(len(swm)),
            "switch_helped": int((stop_s[swm] > ship[swm]).sum()), "switch_hurt": int((stop_s[swm] < ship[swm]).sum()),
            "pick_among_64_changed": int(((~is_m) & (k65m != pick)).sum()),
            "stop_beats_best_of_64_scorer_picks_stop": f"{int(is_m[b16].sum())}/{len(b16)}",
            "stop_pred_agg_mean": round(float(D["agg65m"][:, 64].mean()), 4),
            "stop_rank_among_65_median": float(np.median(rank_m)),
            "switched_tokens": [{"token": dtok[i], "pdms_before": round(100 * ship[i], 2),
                                 "stop_pdms": round(100 * stop_s[i], 2)} for i in swm]})
    else:
        vm["verdict"] = "WITHHELD -- an identity check failed, so the re-implemented masked decoder is not the model's"
    V["V_mask_64_blind_to_stop"] = vm
    # V-gate: take STOP when a statistic crosses a threshold -- the threshold chosen LEAVE-ONE-LOG-OUT (tau never
    # sees the log it is applied to), plus the in-sample optimum as an explicitly optimistic bound
    target = stop_s > ship
    conf = D["agg64"][ar, pick].astype(np.float64)                  # the scorer's own aggregate for its pick
    marg = (D["agg65m"][:, 64] - D["agg65m"][ar, pick]).astype(np.float64)   # V-mask: STOP minus the pick

    def auc(score, lab):
        pos, neg = score[lab], score[~lab]
        if len(pos) == 0 or len(neg) == 0:
            return None
        return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

    def gate(stat):                                                 # take STOP when stat > tau
        def best_tau(m):                                            # candidates from the FIT tokens only
            cand = np.concatenate([[-np.inf], np.unique(stat[m]), [np.inf]])
            vals = [(np.where(stat[m] > t, stop_s[m], ship[m]).mean(), t) for t in cand]
            return max(vals)[1]                                     # ties -> the LARGEST tau (fewest STOPs)
        g = np.zeros(n, dtype=bool)
        for l in ul:
            te = logs == l
            g[te] = stat[te] > best_tau(~te)
        cv = np.where(g, stop_s, ship)
        t_in = best_tau(np.ones(n, dtype=bool))
        ins = np.where(stat > t_in, stop_s, ship)
        c = boot_ci(cv - ship, draws)
        return {"auc_for_stop_better_than_shipped": auc(stat, target),
                "lolo_cv": {"pdms": round(100 * cv.mean(), 2), "minus_shipped": c["diff"], "ci95": c["ci95"],
                            "separated": c["separated"], "minus_stop_floor": boot_ci(cv - stop_s, draws),
                            "stop_taken": int(g.sum()), "stop_taken_and_better": int((g & target).sum())},
                "in_sample_optimum_OPTIMISTIC": {"tau": float(t_in), "pdms": round(100 * ins.mean(), 2),
                                                 "stop_taken": int((stat > t_in).sum())}}
    V["V_gate_threshold_on_a_scorer_statistic"] = {
        "what": "take STOP instead of the shipped pick when a statistic exceeds tau; tau chosen per held-out log on "
                "the OTHER logs (leave-one-log-out), so no token is scored by a tau fitted on it. The bootstrap "
                "resamples the CV outcomes and does not re-simulate the tau selection. A HYPOTHESIS GENERATOR for a "
                "pre-registered test on disjoint tokens -- never a result",
        "n_stop_better_than_shipped": int(target.sum()),
        "stat_low_confidence_minus_agg_of_pick": gate(-conf),
        "stat_vmask_stop_minus_pick_agg": gate(marg) if vm["identities_pass"] else None}
    out["post_hoc_variants"] = V
    out["per_token"] = {
        "fields": ["token", "log", "shipped_pick", "new_pick(64=STOP)", "pdms_shipped", "pdms_new", "stop_pdms",
                   "best_of_64", "stop_agg", "best_agg_64_in_65set", "stop_rank", "vmask_pick(64=STOP)"],
        "rows": [[dtok[i], str(logs[i]), int(pick[i]), int(k65[i]), round(100 * ship[i], 3), round(100 * new[i], 3),
                  round(100 * stop_s[i], 3), round(100 * best[i], 3), round(float(agg_stop[i]), 5),
                  round(float(D["agg65"][i, :64].max()), 5), int(rank[i]), int(k65m[i])] for i in range(n)]}
    # NAVSIM's own sub-scores of each arm's pick (harness values: the table's per-proposal `sub`, STOP's csv row)
    subt = T["sub"][ri].astype(np.float64)                                     # [n, 64, 6] in SUB order
    assert tuple(str(x) for x in T["sub_names"]) == SUB

    def subs_of(k):
        return np.where((k == 64)[:, None], stop_sub, subt[ar, np.minimum(k, 63)])
    out["navsim_subscores_mean_x100"] = {
        "order": list(SUB),
        "shipped": [round(100 * float(x), 2) for x in subs_of(pick).mean(0)],
        "new_65set": [round(100 * float(x), 2) for x in subs_of(k65).mean(0)],
        "post_hoc_vmask": [round(100 * float(x), 2) for x in subs_of(k65m).mean(0)],
        "stop_floor": [round(100 * float(x), 2) for x in stop_sub.mean(0)]}
    if out["complete"] and not a.no_families:
        out["four_families"] = families_step(a, dtok, T, ri, pick, k65, k65m)
    else:
        out["four_families"] = {"status": "NOT RUN", "reason": "--no-families or an incomplete dump"}
    out["provenance"] = {"table_sha256": sha256(a.table), "dump_sha256": sha256(a.dump),
                         "tokens_sha256": sha256(a.tokens), "script_sha256": sha256(os.path.abspath(__file__)),
                         "ckpt_sha256": sha256(meta["ckpt"]) if os.path.exists(meta.get("ckpt", "")) else None}
    out["verdict"] = "OK" if out["complete"] else ("SMOKE" if meta.get("limit_logs") else "INCOMPLETE")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"  shipped {res['shipped_pick_pdms']:.2f} -> new {res['new_pick_pdms']:.2f}  diff {res['new_minus_shipped']:+.2f} "
          f"[{res['ci95'][0]:+.2f}, {res['ci95'][1]:+.2f}]  switches to STOP {len(sw)} "
          f"(of the {len(b16)} STOP>best: {int(is_stop[b16].sum())})  wrote {a.out}", flush=True)
    if vm.get("identities_pass"):
        print(f"  [post-hoc V-mask] new {vm['new_pick_pdms']:.2f}  diff {vm['new_minus_shipped']:+.2f} "
              f"[{vm['ci95'][0]:+.2f}, {vm['ci95'][1]:+.2f}]  switches to STOP {vm['switch_to_stop']}", flush=True)
    print(f"  [post-hoc V-dup] 64-shift max {ddup.max():.4f} mean {ddup.mean():.4f}  vs STOP's max {dl.max():.4f} "
          f"mean {dl.mean():.4f}", flush=True)
    print(f"ZZSTOPPROBE_{'OK' if out['complete'] else out['verdict']} {res['new_pick_pdms']:.2f} "
          f"{res['new_minus_shipped']:+.2f} {res['ci95'][0]:+.2f} {res['ci95'][1]:+.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--db-dir", default=SEAM.TEST_DB_DIR)
    ap.add_argument("--export", default=SEAM.W3_EXPORT)
    ap.add_argument("--stop-csv", default=STOP_CSV)
    ap.add_argument("--stop-src", default=STOP_SRC)
    ap.add_argument("--dump", default=DUMP)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--limit-logs", type=int, default=0, help="smoke: only the first N logs")
    ap.add_argument("--no-extra-forward", action="store_true", help="skip control C-c (halves the GPU time)")
    ap.add_argument("--deadline", default="15:45", help="local HH:MM after which the GPU loop stops")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--no-families", action="store_true", help="skip the four-families step (families6.py)")
    ap.add_argument("--rescore-stop", action="store_true",
                    help="control C-f: score zeros(8,3) through score_navtest_refe.py (a CPU harness run, resumable) "
                         "and require the banked STOP csv per token")
    ap.add_argument("--landed-seam", default=f"{EC.DATA}/seams/refe_{NAME}.npz")
    ap.add_argument("--landed-families", default=f"{EC.DATA}/points/{NAME}/4_families.json")
    a = ap.parse_args()
    if a.analyze_only:
        return analyze(a)
    if os.environ.get(CHILD) != "1":
        # the seam's interpreter and environment, from eval_checkpoint.py -- never a hand-copied subset
        env = EC.env_driverl()
        env[CHILD] = "1"
        rc = subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=HERE, env=env)
        return rc
    rc = gpu_stage(a)
    if rc != 0:
        return rc
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
