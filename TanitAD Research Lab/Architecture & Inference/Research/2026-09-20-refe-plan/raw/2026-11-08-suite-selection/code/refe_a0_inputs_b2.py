#!/usr/bin/env python3
"""Lean A0-input capture for the bundle-2 tokens: the exact (ego, goal) REFe receives at A0, captured by a forward PRE-hook on the model.

It reuses the SAME planner / scenario builder / frame resolver / driving-command seam as E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/refe_a0_inputs.py
(which produced tsf_nuplan_cruise_bundle/a0_model_inputs.jsonl) but drops everything the capture does not need: the table.npz identity check (it only knows the 1,058
C3 tokens), the GREY/SWAP/replicate arms, the donor logs and the GPU lock.  No file of the original is edited.

    CUDA_VISIBLE_DEVICES=-1 python -I refe_a0_inputs_b2.py --tokens <json> --out <dir> [--device cpu] [--forward]

Default (no --forward): the pre-hook records (ego, goal) and ABORTS the forward, so no ViT-L compute runs and `pick` is null.
--forward: the model runs (CPU or GPU) and `pick` is written, as in the cruise bundle's a0_model_inputs.jsonl.
Output line (same keys as the cruise bundle): {"token","log","ego7_as_fed","goal_as_fed","pick"} (+ "device","forward").
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
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
RULE = "navsim_v1_lane"

# same environment as refe_a0_inputs.py (set BEFORE the package modules are imported)
os.environ.update({"DZ_ROOT": DZ, "NUPLAN_DATA_ROOT": ROOT + "/data/nuplan", "NUPLAN_MAPS_ROOT": ROOT + "/data/nuplan-maps/nuplan-maps-v1.0",
                   "REFE_NUPLAN_DB_ROOT": ROOT + "/data/nuplan/nuplan-v1.1/splits",
                   "DRIVERL_EVAL_ROOT": DZ, "DRIVERL_EVAL_NUPLAN_ROOT": DZ + "/nuplan-devkit",
                   "DRIVERL_EVAL_CONFIG_PATH": DZ + "/release/configs/driverl_teacher.yaml",
                   "DRIVERL_EVAL_CHECKPOINT_PATH": DZ + "/release/checkpoints/checkpoint_2400.pt",
                   "REFE_BACKBONE_ROOT": ROOT + "/data/backbones", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "8"),
                   "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"})
for _p in (DZ + "/src", DZ + "/nuplan-devkit", REFE_DIR, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class StopCapture(Exception):
    pass


def navsim_root():
    for c in (os.environ.get("TANITAD_NAVSIM_ROOT"), "E:/Archive/devbox-C/navsim", "F:/Archive/devbox-C/navsim", "D:/Archive/devbox-C/navsim"):
        if c and os.path.isdir(c):
            return c
    raise RuntimeError("no navsim archive root (set TANITAD_NAVSIM_ROOT)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="json: list of tokens, or {token: ...} (e.g. build work/wanted.json 'tokinfo')")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doc = json.load(open(a.tokens, encoding="utf-8"))
    if isinstance(doc, dict) and "tokinfo" in doc:
        doc = doc["tokinfo"]
    tokinfo = doc if isinstance(doc, dict) else None
    tokens = list(doc.keys()) if isinstance(doc, dict) else [str(t) for t in doc]
    exp = json.load(gzip.open(os.path.join(navsim_root(), "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"), "rt", encoding="utf-8"))["tokens"]
    missing = [t for t in tokens if t not in exp]
    assert not missing, f"tokens not in navtest_inputs: {missing[:5]}"
    # token -> log from the DB-name list in the exp record when present, else from the bundle's tokinfo
    if tokinfo is not None:
        token_log = {t: tokinfo[t]["log"] for t in tokens}
    else:
        raise SystemExit("--tokens must carry the token->log map (use work/wanted.json)")
    if a.limit:
        tokens = tokens[:a.limit]
    h = hashlib.sha256()
    with open(CKPT, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    ck = h.hexdigest()
    assert ck.startswith(CKPT_SHA256_PREFIX), f"checkpoint sha256 {ck[:16]} != {CKPT_SHA256_PREFIX}"
    import torch
    import navtrain_scenarios as NS
    from planner import REFePlanner
    import refe_navtest_seam as SEAM
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    planner = REFePlanner(checkpoint=CKPT, images_root=FRAMES, db_dir=DB_DIR, backbone="vitl16", device=dev, select="best", rule=RULE)
    CAP = {}

    def _cap(mod, args, kwargs):
        e = args[1] if len(args) > 1 else kwargs.get("ego")
        g = args[2] if len(args) > 2 else kwargs.get("goal")
        CAP["ego"] = e.detach().float().cpu().numpy().reshape(-1).tolist()
        CAP["goal"] = g.detach().float().cpu().numpy().reshape(-1).tolist()
        if not a.forward:
            raise StopCapture()

    planner.model.register_forward_pre_hook(_cap, with_kwargs=True)
    out_p = os.path.join(a.out, "a0_model_inputs.jsonl")
    done = set()
    if a.resume and os.path.exists(out_p):
        done = {json.loads(l)["token"] for l in open(out_p, encoding="utf-8") if l.strip()}
    log_p = os.path.join(a.out, "run.log")

    def say(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True)
        open(log_p, "a", encoding="utf-8").write(line + "\n")

    say(f"tokens {len(tokens)}; ckpt {ck[:16]}; device {planner.device}; forward {a.forward}; already done {len(done)}")
    by_log = {}
    for t in tokens:
        if t not in done:
            by_log.setdefault(token_log[t], []).append(t)
    n = 0
    t_all = time.time()
    with open(out_p, "a", encoding="utf-8") as fo:
        for lg in sorted(by_log):
            lt = sorted(by_log[lg])
            db = os.path.join(DB_DIR, f"{lg}.db")
            got = set()
            for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
                tok = sc._initial_lidar_token
                planner._scenario = sc
                planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                ego = sc.get_ego_state_at_iteration(0)
                img = planner._image_for(ego)
                if img is None:
                    say(f"  token {tok}: NO FRAMES ({planner.frames.miss_reason})")
                    got.add(tok)
                    continue
                SEAM.set_driving_command(planner, tok, exp[tok])
                pick = None
                CAP.clear()
                t1 = time.time()
                try:
                    _tr, _sc, k = planner.infer(ego, img)
                    pick = int(k)
                except StopCapture:
                    pass
                if "ego" not in CAP:
                    say(f"  token {tok}: hook did not fire")
                    got.add(tok)
                    continue
                fo.write(json.dumps({"token": tok, "log": lg, "ego7_as_fed": CAP["ego"], "goal_as_fed": CAP["goal"], "pick": pick, "device": str(planner.device),
                                     "forward": bool(a.forward)}) + "\n")
                fo.flush()
                got.add(tok)
                n += 1
                say(f"  {lg} {tok}: ego7 {np.round(CAP['ego'], 3).tolist()} goal {np.round(CAP['goal'], 2).tolist()} pick {pick} ({time.time() - t1:.1f} s)")
            miss = set(lt) - got
            if miss:
                say(f"  {lg}: scenario builder skipped {sorted(miss)}")
    say(f"done {n} tokens in {time.time() - t_all:.0f} s")
    open(os.path.join(a.out, "DONE"), "w").write("ok\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        print(err, flush=True)
        try:
            open(os.path.join(sys.argv[sys.argv.index("--out") + 1], "FAILED"), "w").write(err)
        except Exception:
            pass
        sys.exit(1)
