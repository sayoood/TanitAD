#!/usr/bin/env python3
"""GPU stage of the SAFE-HEAD cache: C1's 64 proposals, post-decoder scorer features `s`, C1's own head logits and pick, per token.

Runs under the driverl-eval venv with the seam's environment (sh_run.denv()). It drives `REFePlanner` EXACTLY as eval/refe_navtest_seam.py does
(scenario at the token -> initialize -> _image_for -> infer; yaw shim attached; NAVSIM driving_command set; rule/repair/goal_fix = planner defaults, which
are what the banked C1 navtest seam used), so its pick and proposals are the planner's own. Two additions, both read-only taps:
  * a forward PRE-hook on model.score_head records its input `s` [1,64,256] (model.py:563 head, model.py:709 `return self.score_head(s)`)
  * props8 = to_navsim(planner.executed(traj, j)) for every j (the seam's --dump-proposals conversion, refe_navtest_seam.py:296-298)

    sh_dump.py --job job.json --ckpt <model_final.pt> --expect-sha256 <hex> --out-dir <dir> [--no-lock] [--device cuda]
job.json: {"frames": root, "yaw": yaw table, "commands": json {"<log>/<token>": [4]}, "logs": [{"log","db","tokens","split"}]}
The GPU lock (devbox_gpu.lock, stage2_sft.take_lock's protocol) is taken PER LOG and held only around that log's forwards; the model sits on the CPU between logs.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh_lib as L  # noqa: E402
import nt_lib as NT  # noqa: E402

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"


def take_lock(me):                                   # stage2_sft.take_lock, verbatim protocol
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode())
                os.close(fd)
                return
            except FileExistsError:
                pass
        print(f"waiting for the GPU lock {time.strftime('%T')}", flush=True)
        time.sleep(60)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK)
            print(f"GPU lock released {time.strftime('%T')}", flush=True)
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--expect-sha256", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jobname", default="refe-safehead-dump")
    a = ap.parse_args()
    got = L.sha256_file(a.ckpt)
    if got != a.expect_sha256:
        print(f"CTRL-SHA FAIL {got} != {a.expect_sha256}")
        return 3
    job = json.load(open(a.job, encoding="utf-8"))
    cmds = json.load(open(job["commands"], encoding="utf-8"))
    os.makedirs(a.out_dir, exist_ok=True)
    pk = os.environ.get("TANITAD_REFE_PKG", L.PK)
    sys.path.insert(0, os.path.join(pk, "refe"))
    sys.path.insert(0, os.path.join(pk, "eval"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("refe_navtest_seam_sh", os.path.join(pk, "eval", "refe_navtest_seam.py"))
    SEAM = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SEAM)                     # definitions only (main is __main__-guarded)
    import torch
    import ego_yaw as EY
    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    cap: dict = {}
    state: dict = {}

    def init_planner():
        planner = REFePlanner(checkpoint=a.ckpt, images_root=job["frames"], db_dir=os.path.dirname(job["logs"][0]["db"]), backbone="vitl16", device="cpu", select="best")
        shim = EY.attach_to_planner(planner, a.ckpt, job["yaw"])
        if shim is None:
            print("REFUSING: C1 declares ego_yaw but no shim was attached")
            return 2
        planner.model.score_head.register_forward_pre_hook(lambda mod, args: cap.__setitem__("s", args[0].detach()))
        meta = {"ckpt": a.ckpt, "ckpt_sha256": got, "rule": planner.rule, "repair_last_heading": bool(planner.repair_last_heading), "goal_fix": planner.goal_fix,
                "nav_mode": getattr(planner.model.cfg, "nav_mode", "goal"), "per_sample_calib": bool(planner.per_sample_calib), "yaw_table": job["yaw"],
                "yaw_table_sha256": L.sha256_file(job["yaw"]), "tool": "sh_dump_nt/1", "device": a.device}
        print("PLANNER " + json.dumps(meta), flush=True)
        if meta["nav_mode"] != "command":
            print("REFUSING: C1 must build in nav_mode=command")
            return 2
        state["planner"], state["meta"] = planner, meta
        return 0

    rc = 0
    for ent in job["logs"]:
        log, db, toks = ent["log"], ent["db"], ent["tokens"]
        outp = os.path.join(a.out_dir, f"{log}.npz")
        if os.path.exists(outp):
            print(f"skip {log} (exists)", flush=True)
            continue
        me = json.dumps({"job": a.jobname, "pid": os.getpid(), "host": socket.gethostname(), "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        NT.memgate(f"navtest {log}")                      # NAVTEST copy: RAM + commit gate before EACH log (before the GPU lock is taken)
        if not a.no_lock:
            take_lock(me)                                 # the model is built only AFTER the first lock (no 3 GB of RAM held while waiting for the GPU)
        print(f"{log}: lock taken {time.strftime('%T')}, {len(toks)} tokens", flush=True)
        if "planner" not in state:
            try:
                r0 = init_planner()
            except Exception as e:                       # noqa: BLE001
                print(f"FAIL planner build: {type(e).__name__}: {e}", flush=True)
                r0 = 1
            if r0:
                if not a.no_lock:
                    release(me)
                return r0
        planner, meta = state["planner"], state["meta"]
        planner.frames.db_dir = os.path.dirname(db)
        rec = {k: [] for k in ("token", "props", "props8", "s", "logits", "pick", "ego", "cmd")}
        misses, t0 = [], time.time()
        try:
            planner.model.to(a.device)
            planner.device = a.device
            built = {sc._initial_lidar_token: sc for sc in NS.build_scenarios_for_log(db, toks, map_root=L.MAPS, history_rows=1, future_rows=80)}
            for tok in toks:
                sc = built.get(tok)
                if sc is None:
                    misses.append((tok, "scenario builder skipped the token (margin guard)"))
                    continue
                planner._scenario = sc
                planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                ego = sc.get_ego_state_at_iteration(0)
                img = planner._image_for(ego)
                if img is None:
                    misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                    continue
                cmd = cmds[f"{log}/{tok}"]
                planner.driving_command = cmd
                planner.goal_diag = None
                cap.pop("s", None)
                traj, score, k = planner.infer(ego, img)
                s = cap["s"]
                if tuple(s.shape) != (1, L.N_PROP, L.S_DIM) or tuple(score.shape) != (L.N_PROP, L.N_HEAD):
                    raise RuntimeError(f"unexpected shapes s {tuple(s.shape)} logits {tuple(score.shape)}")
                with torch.no_grad():
                    d = (planner.model.score_head(s)[0].float() - score.float()).abs().max().item()
                if d != 0.0:
                    raise RuntimeError(f"{tok}: score_head(s) != the forward's logits on the same device (max abs diff {d})")
                p8 = np.stack([SEAM.to_navsim(planner.executed(traj, j).float().cpu().numpy()) for j in range(traj.shape[0])]).astype(np.float32)
                if not np.isfinite(p8).all():
                    misses.append((tok, "non-finite proposals"))
                    continue
                rec["token"].append(tok)
                rec["props"].append(traj.float().cpu().numpy().astype(np.float32))
                rec["props8"].append(p8)
                rec["s"].append(s[0].float().cpu().numpy().astype(np.float32))
                rec["logits"].append(score.float().cpu().numpy().astype(np.float32))
                rec["pick"].append(int(k))
                rec["ego"].append(planner.last_inputs[0].reshape(-1).float().cpu().numpy())
                rec["cmd"].append(np.asarray(cmd, np.float32).reshape(-1))
                if len(rec["token"]) % 25 == 0:
                    print(f"  {log}: {len(rec['token'])}/{len(toks)} {time.time() - t0:.0f} s", flush=True)
            s_dtype = str(s.dtype) if rec["token"] else "n/a"
        except Exception as e:                        # noqa: BLE001 -- loud, then the chain stops on rc
            print(f"FAIL {log}: {type(e).__name__}: {e}", flush=True)
            rc = 1
            s_dtype = "n/a"
        finally:
            planner.model.to("cpu")
            planner.device = "cpu"
            if a.device == "cuda":
                torch.cuda.empty_cache()
            if not a.no_lock:
                release(me)
        if rc:
            return rc
        if not rec["token"]:
            print(f"FAIL {log}: no token produced a record; misses {misses[:3]}", flush=True)
            return 1
        arrs = {"log": np.array(log), "split": np.array("navtest"), "token": np.array(rec["token"]), "props": np.stack(rec["props"]), "props8": np.stack(rec["props8"]), "s": np.stack(rec["s"]), "logits": np.stack(rec["logits"]),
                "pick": np.array(rec["pick"], np.int64), "ego": np.stack(rec["ego"]), "cmd": np.stack(rec["cmd"]),
                "meta": np.array(json.dumps({**meta, "s_dtype_captured": s_dtype, "log": log, "split": ent.get("split")}, sort_keys=True))}
        L.save_npz_det(outp, **arrs)
        json.dump({"log": log, "asked": len(toks), "records": len(rec["token"]), "misses": misses, "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out_dir, f"{log}.report.json"), "w"), indent=0)
        print(f"ZZDUMP {log} records {len(rec['token'])}/{len(toks)} misses {len(misses)} {time.time() - t0:.0f} s", flush=True)
    print("ZZDUMP_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
