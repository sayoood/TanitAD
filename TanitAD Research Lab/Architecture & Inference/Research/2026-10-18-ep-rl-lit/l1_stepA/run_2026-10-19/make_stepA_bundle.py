#!/usr/bin/env python3
"""THE ONE GPU FORWARD of Step A: P2-s0 on the sampled held-out tokens, capturing what the dump of `p2ho_dump.py` did NOT keep: the RAW trajectory (the scorer's input; the dump stores only
the A7-REPAIRED plan, last heading overwritten: MEASURED in raw/real_scorer_check.json, 1705 x 64 rows) and the scoring decoder's `visual_ctx` (refe/ctx_cache.py's tap).

Derived from `raw/2026-10-18-p2-heldout-table/code/p2ho_dump.py` (same planner, same scenario builder call, same input construction, same pick), one forward per token.  UNTESTED on data in
this package (needs the nuPlan / driverl-eval venv and the GPU lock): it is gated at run time instead, and exits non-zero unless
  G2b-fan   repair(raw trajectory) == the banked proposals20 AND the logits == the banked logits, BIT FOR BIT, on every token  (the dump's own determinism check passed at max abs 0.0)
  tap       scene_proj ran twice and the decoder's context IS scene_proj's second output (ctx_cache.VisualCtxTap)

Writes <out>/bundle.npz {token, log, traj_raw [N,64,20,3], logits [N,64,6], pick, sub, pdms, valid (banked PDM labels of P2's own fan), ...}, <out>/ctx.npy ([N,7680,256] uint16 = raw bf16 bits, np.load(mmap_mode='r');
3.75 MiB per token) and <out>/ctx.json (sha256).  Run ONLY through launch_stepA.py (the dev-box GPU lock).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe make_stepA_bundle.py --ckpt <p2/model_final.pt> --plan <heldout_tokens.json> --sample <sample.json> --table <table_p2_heldout.npz> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.environ.get("L1_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(TREE, "refe"))
sys.path.insert(0, os.path.join(TREE, "eval"))
P_TOK, D_CTX = 7680, 256


def banked_rows(table_path, tokens):
    t = np.load(table_path, allow_pickle=True)
    ix = {str(k): i for i, k in enumerate(t["token"])}
    missing = [x for x in tokens if x not in ix]
    if missing:
        raise SystemExit(f"{len(missing)} sampled tokens are not in the banked table: {missing[:3]}")
    rows = [ix[x] for x in tokens]
    return {k: t[k][rows] for k in ("logits", "proposals", "pick", "sub", "pdms", "valid")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sample", required=True, help="stepA.py sample's sample.json")
    ap.add_argument("--table", required=True)
    ap.add_argument("--proposals20", default="D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/proposals.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    a = ap.parse_args()
    import stepA as A
    A.require_lock_for_gpu(a.device)                         # exit 4 unless launched through launch_stepA.py (L1_GPU_LOCKED=1)
    import torch
    import ctx_cache as CC
    import navtrain_scenarios as NS
    import planner as PLM
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    plan = json.load(open(a.plan, encoding="utf-8"))
    sample = json.load(open(a.sample, encoding="utf-8"))["tokens"]
    toks = sorted(sample)
    N = len(toks)
    bank = banked_rows(a.table, toks)
    p20 = np.load(a.proposals20, allow_pickle=True)
    P20 = p20["proposals20"]                                         # load once (an npz member is re-read on every access)
    p20i = {str(k): i for i, k in enumerate(p20["token"])}
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(plan["token_log"][t], []).append(t)
    os.makedirs(a.out_dir, exist_ok=True)
    first_db = next(iter(plan["db"].values()))
    planner = REFePlanner(checkpoint=a.ckpt, images_root="", db_dir=os.path.dirname(first_db), backbone=a.backbone, device=a.device, select="best", rule="navsim_v1")
    if getattr(planner.model.cfg, "nav_mode", "goal") == "command" or planner.sanitize_goal:
        print("REFUSED: command-mode / sanitising planner", flush=True)
        return 2
    frs = {r: PLM.FrameResolver(db_dir="", images_root=r) for r in sorted(set(plan["token_root"][t] for t in toks))}
    tap = CC.VisualCtxTap(planner.model)
    ctx_mm = np.lib.format.open_memmap(os.path.join(a.out_dir, "ctx.npy"), mode="w+", dtype=np.uint16, shape=(N, P_TOK, D_CTX))
    row = {t: i for i, t in enumerate(toks)}
    raw = np.zeros((N, 64, 20, 3), np.float32)
    logits = np.zeros((N, 64, 6), np.float32)
    pick = np.full(N, -1, np.int64)
    done = np.zeros(N, bool)
    bad = []
    t0 = time.time()
    for li, (log, lt) in enumerate(sorted(by_log.items())):
        db = plan["db"][log]
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            fr = frs[plan["token_root"][tok]]
            fr.db_dir = os.path.dirname(db)
            planner.frames = fr
            img = planner._image_for(ego)
            if img is None:
                bad.append((tok, "no frames"))
                continue
            tap.reset()
            traj, score, k = planner.infer(ego, img)                 # ONE forward; `traj` is the RAW [64,20,3] model output
            # MM plumbing fix 2026-10-19: the planner runs fp32 (no autocast), so visual_ctx arrives fp32; the bundle format is bf16 bits.
            # Round explicitly (round-to-nearest). G2b (fan/logits/pick) is unaffected; the CPU scoring phase is not bit-comparable anyway (C9).
            _v = tap.visual_ctx()
            ctx = CC.to_u16(_v.to(torch.bfloat16) if _v.dtype == torch.float32 else _v)[0]   # raises CaptureError unless the decoder's ctx IS scene_proj's second output
            i = row[tok]
            r = traj.float().cpu().numpy().astype(np.float32)
            lg = score.float().cpu().numpy().astype(np.float32)
            rep = r.copy()
            rep[:, -1, 2] = rep[:, -2, 2]                             # planner.repair_last_heading
            if not (np.array_equal(rep, P20[p20i[tok]]) and np.array_equal(lg, bank["logits"][i]) and int(k) == int(bank["pick"][i])):
                bad.append((tok, "fan/logits/pick differ from the banked P2 table"))
            raw[i], logits[i], pick[i], done[i] = r, lg, int(k), True
            ctx_mm[i] = ctx
        print(f"  [{li + 1}/{len(by_log)}] {log} done {int(done.sum())}/{N} bad {len(bad)} {time.time() - t0:.0f} s", flush=True)
    tap.close()
    ctx_mm.flush()
    del ctx_mm
    if not done.all() or bad:
        print(f"ZZBUNDLE_FAIL missing {int((~done).sum())} bad {len(bad)} {bad[:3]}", flush=True)
        return 1
    log_of = np.array([plan["token_log"][t] for t in toks])
    np.savez(os.path.join(a.out_dir, "bundle.npz"), token=np.array(toks), log=log_of, traj_raw=raw, logits=logits, pick=pick,
             sub=bank["sub"], pdms=bank["pdms"], valid=bank["valid"])
    h = hashlib.sha256()
    with open(os.path.join(a.out_dir, "ctx.npy"), "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    json.dump({"n": N, "ctx_file": "ctx.npy", "dtype": "uint16 (bf16 bits)", "shape": [N, P_TOK, D_CTX], "ctx_sha256": h.hexdigest(), "ckpt": a.ckpt, "tokens_sha256": hashlib.sha256("\n".join(toks).encode()).hexdigest()},
              open(os.path.join(a.out_dir, "ctx.json"), "w"), indent=1)
    print(f"ZZBUNDLE_OK {N} {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
