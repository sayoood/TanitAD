#!/usr/bin/env python3
"""THE ONE GPU FORWARD per navtest chunk of SEARCH-NAVTEST-PIPE: P2-s0 on one chunk's navtest tokens, capturing what the banked navtest seam did NOT keep (the RAW 64-proposal fan, the scorer's six logits per
proposal, the pick index) and the scoring decoder's `visual_ctx` (refe/ctx_cache.py's tap).

Derived from `l1run/code/make_stepA_bundle.py` (the certified held-out builder: same planner, same scenario-builder call, same input construction, same pick, ONE forward per token) with exactly three changes:
  1. the reference for the fail-loud gate is P2's BANKED NAVTEST SEAM (default; `--reference-kind table|s0fan` selects the navtest 64-proposal table or the S0 bank's fan files) (`stage2_p2_full.npz`: the 12,146 executed picks P2 already produced when 88.82 was measured), not the held-out fan table (which does not exist for navtest);
  2. NO label of any kind is read or written: the bundle carries no `sub` / `pdms` / `valid` (the pick's score is joined from the banked CSV by the CPU analysis, never here);
  3. the plan is the navtest plan (`navpipe_core`-pinned token file; DB = nuPlan test split; frames = the navtest frame zips).
It exits non-zero unless, on EVERY token,
  G-IDENT   to_navsim(repair(raw trajectory)[pick]) == the banked seam row, BYTE FOR BYTE   (the pick of the fresh forward IS P2's banked navtest pick)
  tap       scene_proj ran twice and the decoder's context IS scene_proj's second output (ctx_cache.VisualCtxTap)
Writes <out>/bundle.npz {token, log, traj_raw [N,64,20,3], logits [N,64,6], pick, poses_navsim [N,8,3]}, <out>/ctx.npy ([N,7680,256] uint16 = raw bf16 bits; 3.75 MiB per token) and <out>/ctx.json (sha256s).
UNTESTED ON DATA IN THIS PACKAGE (needs the nuPlan / driverl-eval venv and the GPU lock); run ONLY through navpipe_gpu_stage.py / launch_stepA_ctype.py (the dev-box GPU lock).  Its first run is the BRIDGE.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe make_navtest_bundle.py --ckpt <p2/model_final.pt> --plan <navtest_plan.json> --sample <chunk_sample.json> --banked-seam <stage2_p2_full.npz> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
TREE = os.environ.get("L1_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
for p in (HERE, L1_CODE, os.path.join(TREE, "refe"), os.path.join(TREE, "eval")):
    if p not in sys.path:
        sys.path.insert(0, p)
P_TOK, D_CTX = 7680, 256
TEST_DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
FRAMES_ROOT = "D:/Projects/TanitAD/data/refe_navtest/frames"


def make_plan(token_doc: dict, db_dir: str = TEST_DB_DIR, frames_root: str = FRAMES_ROOT) -> dict:
    """the navtest plan in the shape make_stepA_bundle's plan has ({tokens, token_log, db, token_root}); derived from the pinned token file alone, so it needs no pin of its own"""
    logs = sorted({token_doc["token_log"][t] for t in token_doc["tokens"]})
    return {"tokens": list(token_doc["tokens"]), "token_log": dict(token_doc["token_log"]), "db": {l: f"{db_dir}/{l}.db" for l in logs}, "token_root": {t: frames_root for t in token_doc["tokens"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--sample", required=True, help="{'tokens': [...]} = ONE chunk's sorted tokens")
    ap.add_argument("--banked-seam", default=None, help="alias of --reference-path with --reference-kind seam (navtest)")
    ap.add_argument("--reference-kind", default="seam", choices=("seam", "table", "s0fan"), help="the banked P2 reference every token must equal: seam = navtest executed picks; table = navtest 64-proposal table; s0fan = the S0 bank's fan files")
    ap.add_argument("--reference-path", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    a = ap.parse_args(argv)
    import navpipe_core as NC
    import stepA as A
    A.require_lock_for_gpu(a.device)                         # exit 4 unless launched through launch_stepA_ctype.py (L1_GPU_LOCKED=1)
    import torch
    import ctx_cache as CC
    import navtrain_scenarios as NS
    import planner as PLM
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization

    plan = json.load(open(a.plan, encoding="utf-8"))
    toks = sorted(json.load(open(a.sample, encoding="utf-8"))["tokens"])
    N = len(toks)
    ref = NC.make_reference(a.reference_kind, a.reference_path or a.banked_seam)
    miss = [t for t in toks if not ref.has(t)]
    if miss:
        raise SystemExit(f"{len(miss)} chunk tokens are not in the banked {a.reference_kind} reference: {miss[:3]}")
    seam = NC.seam_module()                                  # sha-pinned vendored copy of the seam's to_navsim
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
    poses = np.zeros((N, 8, 3), np.float32)
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
            _v = tap.visual_ctx()
            ctx = CC.to_u16(_v.to(torch.bfloat16) if _v.dtype == torch.float32 else _v)[0]   # raises CaptureError unless the decoder's ctx IS scene_proj's second output
            i = row[tok]
            r = traj.float().cpu().numpy().astype(np.float32)
            lg = score.float().cpu().numpy().astype(np.float32)
            pz = seam.to_navsim(NC.repair_last_heading(r[int(k)])).astype(np.float32)        # exactly the seam's row: to_navsim(planner.executed(traj, k)).astype(float32)
            if not ref.equal(tok, r, lg, int(k), seam):                                      # @GUARD:MB-IDENT-REF
                bad.append((tok, f"fresh forward differs from P2's banked {ref.kind} reference"))
            raw[i], logits[i], pick[i], poses[i], done[i] = r, lg, int(k), pz, True
            ctx_mm[i] = ctx
        print(f"  [{li + 1}/{len(by_log)}] {log} done {int(done.sum())}/{N} bad {len(bad)} {time.time() - t0:.0f} s", flush=True)
    tap.close()
    ctx_mm.flush()
    del ctx_mm
    if not done.all() or bad:
        print(f"ZZBUNDLE_FAIL missing {int((~done).sum())} bad {len(bad)} {bad[:3]}", flush=True)
        return 1
    log_of = np.array([plan["token_log"][t] for t in toks])
    np.savez(os.path.join(a.out_dir, "bundle.npz"), token=np.array(toks), log=log_of, traj_raw=raw, logits=logits, pick=pick, poses_navsim=poses)
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
