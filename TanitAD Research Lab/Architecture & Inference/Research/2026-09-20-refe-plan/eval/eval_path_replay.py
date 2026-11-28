#!/usr/bin/env python3
"""EXPLORATORY (report only, dev box): did the EVAL PATH change, or did the WEIGHTS move the fan?

FINDING THAT MOTIVATES IT (coordinator, from the E-6 tables): per proposal SLOT, the median path length over the 200
tokens moved TOGETHER for all 64 slots -- 011->012 every slot shorter (mean -2.85 m), 012->013 every slot LONGER
(mean +4.33 m, 63 of 64 slots > +3 m), then stable. A shift that uniform could be the weights (the first on-policy
epoch) or a change in how the eval builds the model's inputs between the runs that produced those tables.
DISCRIMINATOR: replay each snapshot's weights through TODAY's forward path on the same 200 tokens and compare with
the STORED E-6 table. Bit-identical proposals => the path that produced the stored table is the path today, so the
fan shift between two snapshots is in their weights.

TODAY'S PATH = `refe_navtest_seam.py`'s own loop (the W3 export and token filter, `build_scenarios_for_log(
history_rows=1, future_rows=80)`, `REFePlanner.initialize`, `_image_for`, `infer`), in `eval_checkpoint.py`'s
interpreter and environment (`EC.DRIVERL_PY`, `EC.env_driverl()`; one child process per snapshot). The planner's
selection rule is the one the TABLE shipped with (`table.rule`, absent = "v2_shape", as
`snapshot_pair_under_rule.py` reads it), so the stored pick can be reproduced as well.
Per token it also records the model's non-image inputs as `infer` builds them -- `_ego_vec` (vx, vy, ax, ay,
yaw_rate, steering, speed), `_goal_for` (2 goal points) and the per-log calibration -- so the SPEED INPUT can be
compared across snapshots directly (it is a function of the scenario, not of the weights; the record proves it).

    python eval/eval_path_replay.py --snap 012 013                  # GPU, one child per snapshot, then the analysis
    python eval/eval_path_replay.py --snap 012 013 --analyze-only   # re-analyse the banked dumps
    python eval/eval_path_replay.py --snap 012 --limit-logs 2       # smoke
Prints ZZREPLAY_OK <snap> ... per snapshot and ZZREPLAY_DONE, or ZZREPLAY_FAIL <why>.
"""
from __future__ import annotations

import argparse
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

TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")
CKPT_DIR = "D:/Projects/TanitAD/data/refe_runs_eval"
OUT = os.path.join(HERE, "raw", "e6_sub200_ep015", "eval_path_replay.json")
CHILD = "REFE_PATH_REPLAY_CHILD"
BAR = 1e-4
EGO_FIELDS = ("vx", "vy", "ax", "ay", "yaw_rate", "tire_steering_angle", "speed")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def paths(snap):
    name = f"sub200_ep{snap}"
    return {"name": name, "ckpt": f"{CKPT_DIR}/snap_epoch{snap}.pt",
            "table": f"{EC.DATA}/proptable/{name}/table.npz",
            "dump": f"{EC.DATA}/proptable/{name}/path_replay_dump.npz"}


def past(deadline: str) -> bool:
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return (lt.tm_hour, lt.tm_min) >= (hh, mm)


def gpu_one(a, snap) -> int:
    import torch
    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    P = paths(snap)
    T = np.load(P["table"])
    rule = str(T["rule"]) if "rule" in T.files else "v2_shape"
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    sub = json.load(open(a.tokens, encoding="utf-8"))
    want = set(sub["tokens"] if isinstance(sub, dict) else sub)
    toks = [t for t in exp if t in want]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log.items())
    if a.limit_logs:
        logs = logs[:a.limit_logs]
    planner = REFePlanner(checkpoint=P["ckpt"], images_root=a.frames, db_dir=SEAM.TEST_DB_DIR, backbone="vitl16",
                          device="cuda", select="best", rule=rule)
    print(f"  [{snap}] planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} "
          f"device={planner.device} rule={planner.rule} (table rule {rule})", flush=True)
    if planner.device != "cuda":
        print("ZZREPLAY_FAIL no_cuda"); return 1
    R: dict = {k: [] for k in ("token", "props", "logits", "pick", "ego", "goal", "calib", "sec")}
    misses, stopped = [], None
    t0 = time.time()
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
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            traj, score, k = planner.infer(ego, img)            # the planner's OWN call and pick
            # the non-image inputs exactly as infer built them (same calls; _route_poly is cached by then)
            ego_vec = planner._ego_vec(ego)[0].double().cpu().numpy()
            goal = planner._goal_for(ego)[0].double().cpu().numpy()
            calib = (planner._calib_for(planner._log_hint)[0].double().cpu().numpy()
                     if planner.per_sample_calib else np.full((4, 16), np.nan))
            R["token"].append(tok)
            R["props"].append(np.stack([SEAM.to_navsim(traj[j].float().cpu().numpy())
                                        for j in range(traj.shape[0])]).astype(np.float32))
            R["logits"].append(score.float().cpu().numpy())
            R["pick"].append(int(k))
            R["ego"].append(ego_vec)
            R["goal"].append(goal)
            R["calib"].append(calib)
            R["sec"].append(time.time() - ts)
        if (li + 1) % 10 == 0 or li + 1 == len(logs) or stopped:
            print(f"    [{snap}] [{li + 1}/{len(logs)}] rows {len(R['token']):,}  {time.time() - t0:.0f} s  "
                  f"local {time.strftime('%H:%M:%S')}", flush=True)
    if not R["token"]:
        print("ZZREPLAY_FAIL no_rows"); return 1
    meta = {"snap": snap, "ckpt": P["ckpt"], "table": P["table"], "rule": planner.rule, "table_rule": rule,
            "per_sample_calib": bool(planner.per_sample_calib), "ckpt_format": str(getattr(planner, "ckpt_format", "")),
            "rows": len(R["token"]), "misses": misses[:20], "stopped": stopped, "limit_logs": a.limit_logs,
            "gpu_seconds": round(time.time() - t0, 1), "finished_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cuda_max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "python": sys.executable, "torch": torch.__version__,
            "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "NUPLAN_MAPS_ROOT", "REFE_BACKBONE_ROOT")}}
    dump = a.dump_override or P["dump"]
    np.savez(dump, token=np.array(R["token"]), props=np.stack(R["props"]), logits=np.stack(R["logits"]),
             pick=np.array(R["pick"], dtype=np.int64), ego=np.stack(R["ego"]), goal=np.stack(R["goal"]),
             calib=np.stack(R["calib"]), sec=np.array(R["sec"]), meta=np.array(json.dumps(meta)))
    print(f"  [{snap}] dump {len(R['token'])} rows -> {dump}", flush=True)
    print(f"ZZREPLAY_GPU_DONE {snap} {len(R['token'])}", flush=True)
    return 0


def path_len(props):
    """[..., 8, 3] NAVSIM-grid poses -> polyline length from the origin through the 8 poses"""
    xy = np.concatenate([np.zeros(props.shape[:-2] + (1, 2)), props[..., :2]], axis=-2)
    return np.linalg.norm(np.diff(xy, axis=-2), axis=-1).sum(-1)


def analyze(a) -> int:
    out = {"_label": "EXPLORATORY -- eval-path discriminator for the E-6 fan shift; same 200 tokens",
           "question": "Does TODAY's forward path reproduce the STORED E-6 tables of earlier snapshots bit-for-bit? "
                       "If yes, the path that produced them is unchanged and the fan shift is in the weights.",
           "script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
           "tokens": a.tokens, "snapshots": {}}
    dumps = {}
    for snap in a.snap:
        P = paths(snap)
        dp = a.dump_override or P["dump"]
        if not os.path.exists(dp):
            out["snapshots"][snap] = {"status": "NO_DUMP", "dump": dp}
            continue
        D = np.load(dp, allow_pickle=False)
        meta = json.loads(str(D["meta"]))
        T = np.load(P["table"])
        ttok = [str(t) for t in T["token"]]
        dtok = [str(t) for t in D["token"]]
        tix = {t: i for i, t in enumerate(ttok)}
        ri = np.array([tix[t] for t in dtok])
        n = len(dtok)
        dp_ = np.abs(D["props"].astype(np.float64) - T["proposals"][ri].astype(np.float64)).reshape(n, -1).max(1)
        dl = np.abs(D["logits"].astype(np.float64) - T["logits"][ri].astype(np.float64)).reshape(n, -1).max(1)
        pk = D["pick"] == T["pick"][ri]
        L_rep, L_st = path_len(D["props"].astype(np.float64)), path_len(T["proposals"][ri].astype(np.float64))
        rec = {"ckpt": meta["ckpt"], "ckpt_sha256": sha256(meta["ckpt"]), "table": P["table"],
               "table_sha256": sha256(P["table"]), "dump": dp, "dump_sha256": sha256(dp),
               "table_rule": meta["table_rule"], "planner_rule": meta["rule"],
               "per_sample_calib": meta["per_sample_calib"], "ckpt_format": meta["ckpt_format"],
               "rows": n, "n_table": len(ttok), "stopped": meta["stopped"], "token_order_identical": dtok == ttok,
               "proposals_max_abs_diff_m": float(dp_.max()), "proposals_tokens_bit_identical": int((dp_ == 0).sum()),
               "logits_max_abs_diff": float(dl.max()), "logits_tokens_bit_identical": int((dl == 0).sum()),
               "pick_reproduced": f"{int(pk.sum())}/{n}",
               "per_slot_median_path_m_max_abs_diff": float(np.abs(np.median(L_rep, 0) - np.median(L_st, 0)).max()),
               "gpu_seconds": meta["gpu_seconds"], "cuda_max_memory_allocated_gb": meta["cuda_max_memory_allocated_gb"],
               "env": meta["env"]}
        rec["verdict"] = ("IDENTICAL -- today's path reproduces the stored table bit-for-bit"
                          if rec["proposals_tokens_bit_identical"] == n and rec["logits_tokens_bit_identical"] == n
                          and int(pk.sum()) == n else
                          ("WITHIN_BAR" if dp_.max() <= BAR and dl.max() <= BAR and pk.all() else "DIFFERS"))
        out["snapshots"][snap] = rec
        dumps[snap] = (dtok, D)
        print(f"ZZREPLAY_{'OK' if rec['verdict'].startswith('IDENTICAL') else 'DIFF'} {snap} props "
              f"{rec['proposals_max_abs_diff_m']:.3g} ({rec['proposals_tokens_bit_identical']}/{n} bit-identical) "
              f"logits {rec['logits_max_abs_diff']:.3g} pick {rec['pick_reproduced']}", flush=True)
    # the non-image inputs across snapshots (a function of the scenario; the record proves it). --also-inputs adds
    # another snapshot's recorded inputs (e.g. 015 from slow_copies.py's GPU dump), inputs only.
    for spec in (a.also_inputs or []):
        s, _, p = spec.partition("=")
        if os.path.exists(p):
            X = np.load(p, allow_pickle=False)
            dumps[s] = ([str(t) for t in X["token"]], {k: X[k] for k in ("ego", "goal", "calib")})
            out.setdefault("inputs_only_from", {})[s] = {"dump": p, "dump_sha256": sha256(p)}
    snaps = [s for s in list(a.snap) + [x.partition("=")[0] for x in (a.also_inputs or [])] if s in dumps]
    if len(snaps) >= 2:
        base_tok, B = dumps[snaps[0]]
        cmp = {}
        for s in snaps[1:]:
            tk, X = dumps[s]
            common = [t for t in base_tok if t in set(tk)]
            ib = {t: i for i, t in enumerate(base_tok)}
            ix = {t: i for i, t in enumerate(tk)}
            bi = np.array([ib[t] for t in common])
            xi = np.array([ix[t] for t in common])
            cmp[f"{snaps[0]}_vs_{s}"] = {
                "tokens_compared": len(common),
                "ego_vec_max_abs_diff": float(np.abs(B["ego"][bi] - X["ego"][xi]).max()),
                "goal_max_abs_diff": float(np.abs(B["goal"][bi] - X["goal"][xi]).max()),
                "calib_max_abs_diff": float(np.nanmax(np.abs(B["calib"][bi] - X["calib"][xi])))
                if np.isfinite(B["calib"]).any() else None,
                "ego_vec_tokens_bit_identical": int((B["ego"][bi] == X["ego"][xi]).all(1).sum())}
        out["non_image_inputs_across_snapshots"] = cmp
        def row(s, t):
            j = dumps[s][0].index(t)
            return {"ego": [round(float(v), 6) for v in dumps[s][1]["ego"][j]],
                    "goal": [round(float(v), 4) for v in dumps[s][1]["goal"][j]]}
        out["ego_vector_examples"] = {
            "fields": {"ego": list(EGO_FIELDS), "goal": "2 goal points x (x, y), ego frame, m"},
            "tokens": [{"token": t, **{f"ep{s}": row(s, t) for s in snaps if t in dumps[s][0]}}
                       for t in base_tok[:2]]}
        E = np.asarray(B["ego"])
        out["ego_fields_summary_over_tokens"] = {
            f: {"n_nonzero": int((E[:, j] != 0).sum()), "min": round(float(E[:, j].min()), 4),
                "max": round(float(E[:, j].max()), 4)} for j, f in enumerate(EGO_FIELDS)}
        out["ego_fields_summary_note"] = ("fields that are exactly 0.0 on every token carry no information at "
                                          "inference; compare with the training bank before reading anything into it")
        # the fan shift itself, TODAY's replay (per slot: the median path over the tokens)
        fs = [s for s in snaps if "props" in getattr(dumps[s][1], "files", dumps[s][1])]
        med = {s: np.median(path_len(dumps[s][1]["props"].astype(np.float64)), 0) for s in fs}
        for i in range(len(fs) - 1):
            snaps_i = fs
            d = med[snaps_i[i + 1]] - med[snaps_i[i]]
            out.setdefault("fan_shift_per_slot_median_path_m", {})[f"{fs[i]}_to_{fs[i + 1]}"] = {
                "mean": round(float(d.mean()), 3), "min": round(float(d.min()), 3), "max": round(float(d.max()), 3),
                "slots_longer_than_3m": int((d > 3).sum()), "slots_shorter_than_minus_3m": int((d < -3).sum())}
    ok = all(v.get("verdict", "").startswith("IDENTICAL") for v in out["snapshots"].values())
    out["verdict"] = ("EVAL PATH UNCHANGED for every replayed snapshot: the fan shift is in the WEIGHTS" if ok else
                      "NOT all snapshots reproduced bit-for-bit -- read the per-snapshot records")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"  wrote {a.out}")
    print(f"ZZREPLAY_DONE {'IDENTICAL' if ok else 'NOT_IDENTICAL'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", nargs="+", default=["012", "013"])
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit-logs", type=int, default=0)
    ap.add_argument("--deadline", default="15:45")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dump-override", default=None, help="smoke: write/read this dump instead (one --snap only)")
    ap.add_argument("--also-inputs", nargs="*", default=None,
                    help="<snap>=<npz> with token/ego/goal/calib (e.g. slow_copies.py's GPU dump for 015): added to "
                         "the cross-snapshot INPUT comparison only")
    ap.add_argument("--child-snap", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child_snap:
        return gpu_one(a, a.child_snap)
    if not a.analyze_only:
        env = EC.env_driverl()
        env[CHILD] = "1"
        for snap in a.snap:
            argv = [EC.DRIVERL_PY, os.path.abspath(__file__), "--child-snap", snap, "--tokens", a.tokens,
                    "--frames", a.frames, "--deadline", a.deadline, "--limit-logs", str(a.limit_logs)]
            if a.dump_override:
                argv += ["--dump-override", a.dump_override]
            rc = subprocess.call(argv, cwd=HERE, env=env)
            if rc != 0:
                print(f"ZZREPLAY_FAIL gpu {snap} rc={rc}"); return rc
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
