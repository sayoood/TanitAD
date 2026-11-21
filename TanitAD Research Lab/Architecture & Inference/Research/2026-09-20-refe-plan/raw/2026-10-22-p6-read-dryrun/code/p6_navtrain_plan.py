"""P6-READ-DRYRUN: the NAVTEST SEAM's per-token loop (eval/refe_navtest_seam.py main(), tip 1664143, lines 226-262), run on NAVTRAIN tokens.

WHY A MIRROR AND NOT THE SEAM ITSELF. The seam reads the W3 navtest export (navtest tokens, fingerprints, NAVSIM `driving_command`) and would put a P6
checkpoint on navtest tokens. PREREG_P6 registers ONE navtest look, on `model_final.pt`; an epoch-6 plan on navtest is an unregistered look. So this driver
reuses the seam's OWN functions unchanged (`to_navsim`, `set_driving_command`), the planner class unchanged (`REFePlanner`, same constructor call), the same
scenario builder call (`navtrain_scenarios.build_scenarios_for_log(db, tokens, history_rows=1, future_rows=80)`) and the same `planner.initialize / _image_for /
infer / executed` sequence. What differs, and only this: the token source (navtrain DBs under splits/trainval, frames under refe_proxy/pixels) and the
`driving_command` source (P6's own training command table, HF logs/commands.json: NAVSIM's `driving_command`, the P6-N-SRC rank-0 rows).

HARD GUARD: refuses any token that is in navtest_all_tokens.json and any DB path that is under splits/test.

Modes
  select : CPU, no model. Picks a command-stratified token set that has a scenario AND resolvable 4-camera frames. Writes tokens_sel.json.
  plan   : loads the checkpoint through the planner, runs every token, writes an npz of the RAW outputs (traj [64,20,3], score [64,6], pick), the executed
           NAVSIM-grid poses, what the model was actually fed (ego vector, command), and per-stage timings. GPU runs take the shared dev-box lock (never wait).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p6dry_common as C  # noqa: E402

C.setup_env()
import numpy as np  # noqa: E402

TRAINVAL = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval"
PIXELS = "D:/Projects/TanitAD/data/refe_proxy/pixels"
COMMANDS = "D:/Projects/TanitAD-artifacts/p6dry/hf/logs/commands.json"
NAVTEST_TOKENS = "D:/Projects/TanitAD/data/refe_navtest/navtest_all_tokens.json"
CMD_NAMES = {(1, 0, 0, 0): "LEFT", (0, 1, 0, 0): "STRAIGHT", (0, 0, 1, 0): "RIGHT", (0, 0, 0, 1): "UNKNOWN"}


def navtest_set() -> set:
    nt = json.load(open(NAVTEST_TOKENS, encoding="utf-8"))
    return set(nt["tokens"] if isinstance(nt, dict) else nt)


def guard(tokens, db_dir) -> dict:
    nts = navtest_set()
    hit = sorted(set(tokens) & nts)
    if hit:
        raise SystemExit(f"REFUSED: {len(hit)} token(s) are navtest tokens, e.g. {hit[:3]}")
    if "/splits/test" in db_dir.replace("\\", "/") or db_dir.replace("\\", "/").rstrip("/").endswith("test"):
        raise SystemExit(f"REFUSED: db dir {db_dir} is the navtest split")
    return {"navtest_tokens": len(nts), "tokens_checked": len(tokens), "overlap": 0, "db_dir": db_dir}


def do_select(a) -> int:
    import navtrain_scenarios as NS
    import planner as PLM
    cmds = json.load(open(COMMANDS, encoding="utf-8"))
    by_log = collections.defaultdict(list)
    for k, v in cmds.items():
        lg, tk = k.split("/")
        by_log[lg].append((tk, tuple(int(x) for x in v)))
    logs = sorted(l for l in by_log if os.path.exists(os.path.join(TRAINVAL, f"{l}.db"))
                  and all(os.path.exists(os.path.join(PIXELS, f"{l}_{c}.zip")) for c in PLM.CAMERAS))
    rng = random.Random(a.seed)
    rng.shuffle(logs)
    quota = {"LEFT": a.per_class, "STRAIGHT": a.per_class, "RIGHT": a.per_class}
    got = {k: [] for k in quota}
    fr = PLM.FrameResolver(db_dir=TRAINVAL, images_root=PIXELS)
    examined = 0
    for lg in logs[: a.max_logs]:
        if all(len(got[k]) >= quota[k] for k in quota):
            break
        cand = by_log[lg][:]
        rng.shuffle(cand)
        cand = cand[: a.cand_per_log]
        cmap = dict(cand)
        db = os.path.join(TRAINVAL, f"{lg}.db")
        ok_tok = []
        for sc in NS.build_scenarios_for_log(db, [t for t, _ in cand], history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            ego = sc.get_ego_state_at_iteration(0)
            paths = fr.resolve(lg, int(ego.time_point.time_us))
            if paths and len(paths) == 4:
                ok_tok.append(tok)
        examined += len(cand)
        for tok in ok_tok:
            name = CMD_NAMES[cmap[tok]]
            if name in got and len(got[name]) < quota[name]:
                got[name].append({"log": lg, "token": tok, "command": list(cmap[tok]), "command_name": name})
        print(f"  {lg}: {len(ok_tok)}/{len(cand)} resolvable; have " + " ".join(f"{k}={len(v)}" for k, v in got.items()), flush=True)
    sel = sorted((r for v in got.values() for r in v), key=lambda r: (r["log"], r["token"]))
    g = guard([r["token"] for r in sel], TRAINVAL)
    out = {"seed": a.seed, "per_class": a.per_class, "n": len(sel), "guard": g, "pixels": PIXELS, "db_dir": TRAINVAL, "commands_src": COMMANDS,
           "commands_src_sha256": hashlib.sha256(open(COMMANDS, "rb").read()).hexdigest(), "examined": examined,
           "class_counts": {k: len(v) for k, v in got.items()}, "n_logs": len({r["log"] for r in sel}), "tokens": sel}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"ZZSELECT_OK n={len(sel)} logs={out['n_logs']} classes={out['class_counts']} navtest_overlap=0")
    return 0 if len(sel) >= a.min_n else 1


def h(*arrs) -> str:
    m = hashlib.sha256()
    for x in arrs:
        m.update(np.ascontiguousarray(x).tobytes())
    return m.hexdigest()


def one_pass(planner, by_log, sel, set_driving_command, to_navsim, PlannerInitialization, NS, sync, swap_lr=False):
    keep = {k: [] for k in ("token", "poses", "traj_raw", "score_raw", "pick", "ego_vec", "cmd_seen", "cmd_asked", "tok_hash", "t_image", "t_infer", "t_post")}
    n_cmd_set, misses = 0, []
    t0 = time.perf_counter()
    for log, lt in by_log.items():
        db = os.path.join(sel["db_dir"], f"{log}.db")
        cmap = {r["token"]: r for r in lt}
        got = set()
        for sc in NS.build_scenarios_for_log(db, [r["token"] for r in lt], history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            got.add(tok)
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            a0 = time.perf_counter()
            img = planner._image_for(ego)
            a1 = time.perf_counter()
            if img is None:
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            cmd = list(cmap[tok]["command"])
            if swap_lr:                                  # CONTROL ONLY: LEFT <-> RIGHT, STRAIGHT untouched
                cmd = [cmd[2], cmd[1], cmd[0], cmd[3]]
            n_cmd_set += int(set_driving_command(planner, tok, {"ego_statuses": [{"driving_command": cmd}]}))
            sync()
            b0 = time.perf_counter()
            traj, score, k = planner.infer(ego, img)
            sync()
            b1 = time.perf_counter()
            poses = to_navsim(planner.executed(traj, k).float().cpu().numpy())
            b2 = time.perf_counter()
            ev, gl = planner.last_inputs
            tr = traj.float().cpu().numpy().astype(np.float32)
            sc_ = score.float().cpu().numpy().astype(np.float32)
            keep["token"].append(tok)
            keep["poses"].append(poses.astype(np.float32))
            keep["traj_raw"].append(tr)
            keep["score_raw"].append(sc_)
            keep["pick"].append(int(k))
            keep["ego_vec"].append(ev.reshape(-1).float().cpu().numpy())
            keep["cmd_seen"].append(gl.reshape(-1).float().cpu().numpy())
            keep["cmd_asked"].append(np.asarray(cmd, dtype=np.float32))
            keep["tok_hash"].append(h(poses.astype(np.float32), tr, sc_))
            keep["t_image"].append(a1 - a0)
            keep["t_infer"].append(b1 - b0)
            keep["t_post"].append(b2 - b1)
        for tok in set(cmap) - got:
            misses.append((tok, "scenario builder skipped the token (margin guard)"))
        print(f"    {log}: rows {len(keep['token'])} misses {len(misses)} {time.perf_counter() - t0:.0f} s", flush=True)
    return keep, misses, n_cmd_set, time.perf_counter() - t0


def do_plan(a) -> int:
    import torch
    import navtrain_scenarios as NS
    import planner as PLM
    from refe_navtest_seam import set_driving_command, to_navsim           # the seam's own functions, imported unchanged
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    sel = json.load(open(a.sel, encoding="utf-8"))
    rows = sel["tokens"][: a.limit] if a.limit else sel["tokens"]
    g = guard([r["token"] for r in rows], sel["db_dir"])
    by_log = collections.OrderedDict()
    for r in rows:
        by_log.setdefault(r["log"], []).append(r)
    dev = a.device
    reps = []
    with C.gpu_lock(enabled=(dev == "cuda")) as lockbody:
        t_load = time.perf_counter()
        planner = PLM.REFePlanner(checkpoint=a.ckpt, images_root=sel["pixels"], db_dir=sel["db_dir"], backbone="vitl16", device=dev, select="best",
                                  rule=a.rule, repair_last_heading=None, sanitize_goal=None, goal_fix=None)   # the seam's constructor call (seam:177-183)
        planner.model.eval()
        load_s = time.perf_counter() - t_load
        cfgm = planner.model.cfg
        print(f"  planner: trained={planner.trained} format={planner.ckpt_format} per_sample_calib={planner.per_sample_calib} device={planner.device} "
              f"rule={planner.rule} nav_mode={cfgm.nav_mode} load {load_s:.1f} s", flush=True)
        if a.device == "cuda" and planner.device != "cuda":
            raise SystemExit("REFUSED: cuda asked, planner fell back to cpu")
        cuda = planner.device == "cuda"
        sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
        for r in range(a.repeat):
            print(f"  pass {r}", flush=True)
            reps.append(one_pass(planner, by_log, sel, set_driving_command, to_navsim, PlannerInitialization, NS, sync, swap_lr=a.swap_lr))
    reports = []
    for r, (keep, misses, n_cmd_set, total_s) in enumerate(reps):
        n = len(keep["token"])
        out = {k: (np.array(v) if k in ("token", "tok_hash", "pick") else np.stack(v)) for k, v in keep.items()}
        path = a.out if a.repeat == 1 else a.out.replace(".npz", f".pass{r}.npz")
        np.savez(path, **out)
        ti, tf, tp = (np.array(keep[k]) for k in ("t_image", "t_infer", "t_post"))
        warm = 2 if (n > 4 and r == 0) else 0
        steady = (ti + tf + tp)[warm:]
        rep = {"ckpt": a.ckpt, "pass": r, "device": planner.device, "torch": torch.__version__, "n_asked": len(rows), "n_rows": n, "misses": misses[:10], "n_misses": len(misses),
               "format": planner.ckpt_format, "rule": planner.rule, "nav_mode": cfgm.nav_mode, "per_sample_calib": planner.per_sample_calib,
               "repair_last_heading": planner.repair_last_heading, "sanitize_goal": planner.sanitize_goal, "goal_fix_recorded": planner.goal_fix,
               "driving_command_set": n_cmd_set, "guard": g, "model_load_s": round(load_s, 2), "loop_s": round(total_s, 2),
               "per_token_s": {"all_mean": float(np.mean(ti + tf + tp)) if n else None, "steady_mean": float(steady.mean()) if len(steady) else None,
                               "steady_median": float(np.median(steady)) if len(steady) else None, "warmup_tokens_excluded": warm,
                               "image_mean": float(ti[warm:].mean()) if n > warm else None, "infer_mean": float(tf[warm:].mean()) if n > warm else None,
                               "post_mean": float(tp[warm:].mean()) if n > warm else None, "first_token_s": float((ti + tf + tp)[0]) if n else None},
               "tokens_per_s_steady": float(1.0 / steady.mean()) if len(steady) else None,
               "cmd_seen_equals_cmd_asked_all": bool(n and np.array_equal(out["cmd_seen"], out["cmd_asked"])),
               "gpu_lock_body": lockbody, "cuda_peak_mib": (round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None), "npz": path,
               "flags": {"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                         "cudnn_benchmark": torch.backends.cudnn.benchmark, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}}
        rep["npz_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        json.dump(rep, open(os.path.splitext(path)[0] + ".report.json", "w", encoding="utf-8"), indent=1)
        reports.append(rep)
        print(json.dumps({k: rep[k] for k in ("pass", "n_rows", "n_misses", "format", "nav_mode", "driving_command_set", "cmd_seen_equals_cmd_asked_all", "tokens_per_s_steady", "loop_s")}))
    ok = all(r["n_rows"] == len(rows) and not r["n_misses"] and r["driving_command_set"] == r["n_rows"] for r in reports)
    print(f"ZZPLAN_{'OK' if ok else 'FAIL'} {reports[0]['n_rows']} {reports[0]['tokens_per_s_steady']}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("select", "plan"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--per-class", type=int, default=16)
    ap.add_argument("--max-logs", type=int, default=40)
    ap.add_argument("--cand-per-log", type=int, default=24)
    ap.add_argument("--min-n", type=int, default=32)
    ap.add_argument("--ckpt")
    ap.add_argument("--sel")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--rule", default="navsim_v1", choices=("navsim_v1", "navsim_v1_lane", "v2_shape"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--swap-lr", action="store_true", help="CONTROL: feed LEFT<->RIGHT swapped commands (STRAIGHT unchanged); the output must move for LEFT/RIGHT tokens and stay bit-identical for STRAIGHT")
    ap.add_argument("--repeat", type=int, default=1, help="passes over the token set IN THIS PROCESS (one npz per pass)")
    a = ap.parse_args()
    return do_select(a) if a.mode == "select" else do_plan(a)


if __name__ == "__main__":
    sys.exit(main())
