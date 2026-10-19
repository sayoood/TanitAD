#!/usr/bin/env python3
"""SH5 ctx build: P2's `visual_ctx` for the PRIVGEOM tokens, captured with `ctx_cache.VisualCtxTap` exactly as L1's `make_stepA_bundle.py` does (one planner forward per token), stored as bf16 bits.

    sh5_ctx_build.py plan  --inputs sh5_inputs.npz --out-dir W           # CPU: which of the tokens have a DB and four resolvable camera frames; coverage; the list of frames still to stage
    sh5_ctx_build.py probe                                               # read-only GPU lock state
    sh5_ctx_build.py run   --inputs ... --plan W/plan.json --ckpt ... --out-dir C --max-wait 21600    # GPU, behind the dev-box lock (sh5_gpulock); resumable

THE FIDELITY GATE (C-FID).  For every token the stored census logits `z6` of its S0 sets are recomputed from the CAPTURED ctx with the very call S0 made (`model.score_trajectories(traj3, sctx)`).  The gate passes iff
>= 99 % of the sets reproduce `z6` to <= 1e-3 (max over the 64x6 logits).  That one number proves the cached ctx IS the ctx S0's scorer saw (same images, same goal, same checkpoint), not merely a ctx of the
right shape.  The bf16 rounding of the stored ctx is measured too (`bf16_maxabs`, diagnostic, no gate).  An early stop after the first 10 sets aborts a run whose fidelity is hopeless (rc 5), before the 70 min are spent.

Never removes a foreign lock (sh5_gpulock).  ⚠️ The planner runs fp32 (no autocast), so `visual_ctx` arrives fp32: it is rounded to bf16 EXPLICITLY before `ctx_cache.to_u16` (the 2026-10-19 fix that L1's
fixed copy carries).  RAM gate 8 GB before the lock (same job class as L1's step 3).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402

TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/par18/l1run/tree_hybrid")           # tree_hybrid = the p2heldout tree's exact code + the tip's ctx_cache.py (L1 attempt-3 fix)
DB_ROOT = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
PROXY_ROOTS = ["D:/Projects/TanitAD/data/refe_m5/pixels", "D:/Projects/TanitAD/data/refe_proxy/pixels"]
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
P_TOK, D_CTX = 7680, 256
EARLY_MIN_SETS, EARLY_MIN_FRAC = 10, 0.5


def index_dbs(extra_dir=None):
    out = {}
    for d in ([extra_dir] if extra_dir else []):
        for p in glob.glob(os.path.join(d, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for s in ("mini", "test", "trainval"):
        for p in glob.glob(os.path.join(DB_ROOT, s, "*.db")):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    for p in glob.glob(os.path.join(DB_ROOT, "*.db")):
        out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p).replace("\\", "/"))
    return out


def token_times(db, toks):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        want = set(toks)
        return {t: int(ts) for t, ts in con.execute("SELECT lower(hex(token)), timestamp FROM lidar_pc") if t in want}
    finally:
        con.close()


# ======================================================================================================================================================================================
def cmd_plan(a) -> int:
    z = np.load(a.inputs, allow_pickle=True)
    toks, tlog = [str(t) for t in z["token"]], [str(l) for l in z["token_log"]]
    tok_log = dict(zip(toks, tlog))
    sys.path.insert(0, f"{TREE}/refe")
    import planner as PL
    roots = list(a.images_root or []) + PROXY_ROOTS
    fr = {r: PL.FrameResolver(db_dir="", images_root=r) for r in roots if os.path.isdir(r)}
    dbs = index_dbs(a.stage_db_dir)
    by_log: dict = {}
    for t, lg in tok_log.items():
        by_log.setdefault(lg, []).append(t)
    ok_tokens, token_root, token_ts, blocked, needed, per_log = [], {}, {}, {}, {}, {}
    for lg in sorted(by_log):
        ts_ = sorted(by_log[lg])
        rec = {"n_tokens": len(ts_), "db": dbs.get(lg)}
        if lg not in dbs:
            rec["status"] = "BLOCKED_NO_DB"
            per_log[lg] = rec
            for t in ts_:
                blocked[t] = {"log": lg, "reason": "no nuPlan DB for this log on this box"}
            continue
        times = token_times(dbs[lg], ts_)
        n_img = 0
        for t in ts_:
            if t not in times:
                blocked[t] = {"log": lg, "reason": "token not in the DB's lidar_pc"}
                continue
            got, why = None, None
            for r, res_ in fr.items():
                res_.db_dir = os.path.dirname(dbs[lg])
                paths = res_.resolve(lg, times[t])
                if paths:
                    got = r
                    break
                why = res_.miss_reason
            if got:
                n_img += 1
                ok_tokens.append(t)
                token_root[t] = got
                token_ts[t] = times[t]
            else:
                blocked[t] = {"log": lg, "reason": f"no camera frames: {why}"}
                r0 = next(iter(fr.values()))
                r0.db_dir = os.path.dirname(dbs[lg])
                nf = {}
                for ch, (ts_arr, names) in r0._index(lg).items():
                    if ts_arr.size:
                        nf[ch] = names[int(abs(ts_arr - times[t]).argmin())]
                needed[t] = {"log": lg, "ts_us": times[t], "frames_filename_jpg": nf}
        rec.update(status="OK" if n_img == len(ts_) else ("PARTIAL" if n_img else "BLOCKED_NO_FRAMES"), n_tokens_with_frames=n_img)
        per_log[lg] = rec
    os.makedirs(a.out_dir, exist_ok=True)
    ok_sorted = sorted(ok_tokens)
    plan = {"rule": "SH5 ctx plan: tokens of the PRIVGEOM subset that have a DB and four resolvable camera frames", "tokens": ok_sorted, "token_log": {t: tok_log[t] for t in ok_sorted},
            "token_root": {t: token_root[t] for t in ok_sorted}, "token_ts_us": {t: token_ts[t] for t in ok_sorted}, "db": {lg: dbs[lg] for lg in sorted({tok_log[t] for t in ok_sorted})}}
    keep = np.array([t in set(ok_tokens) for t in toks])
    sets_keep = keep[z["token_idx"]]
    fold = z["fold"]
    cov = {"n_tokens": len(toks), "n_tokens_with_frames": int(keep.sum()), "n_tokens_blocked": int((~keep).sum()), "n_sets": int(len(sets_keep)), "n_sets_kept": int(sets_keep.sum()),
           "sets_kept_by_fold": [int(sets_keep[fold == k].sum()) for k in range(L.FOLDS)], "sets_by_fold": [int((fold == k).sum()) for k in range(L.FOLDS)],
           "n_logs_with_any": int(len(set(z["log_idx"][sets_keep].tolist()))), "per_log_status": {k: v["status"] for k, v in per_log.items()},
           "blocked_reasons": sorted({b["reason"].split(":")[0] for b in blocked.values()}), "est_ctx_gib_kept": float(int(keep.sum()) * P_TOK * D_CTX * 2 / 2 ** 30),
           "est_ctx_gib_full": float(len(toks) * P_TOK * D_CTX * 2 / 2 ** 30), "image_roots": sorted(fr)}
    json.dump(plan, open(f"{a.out_dir}/plan.json", "w", encoding="utf-8", newline="\n"), indent=0)
    json.dump({"n_tokens": len(needed), "tokens": needed}, open(f"{a.out_dir}/needed_frames.json", "w", encoding="utf-8", newline="\n"), indent=0)
    json.dump(blocked, open(f"{a.out_dir}/blocked_tokens.json", "w", encoding="utf-8", newline="\n"), indent=0)
    json.dump(cov, open(f"{a.out_dir}/coverage.json", "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps({k: cov[k] for k in ("n_tokens", "n_tokens_with_frames", "n_sets", "n_sets_kept", "sets_kept_by_fold", "n_logs_with_any", "est_ctx_gib_kept")}))
    return 0


# ======================================================================================================================================================================================
def fidelity_row(z_new: np.ndarray, z_census: np.ndarray) -> float:
    """max |logit difference| over the 64x6 logits of one set"""
    return float(np.abs(z_new.astype(np.float64) - z_census.astype(np.float64)).max())


def fid_gate(maxabs: np.ndarray) -> dict:
    """C-FID: >= FID_MIN_FRAC of the sets reproduce the census logits to <= FID_TOL."""
    maxabs = np.asarray(maxabs, np.float64)
    frac = float((maxabs <= L.FID_TOL).mean()) if len(maxabs) else float("nan")
    return {"n_sets": int(len(maxabs)), "frac_within_tol": frac, "tol": L.FID_TOL, "min_frac": L.FID_MIN_FRAC, "median": float(np.median(maxabs)) if len(maxabs) else None,
            "max": float(maxabs.max()) if len(maxabs) else None, "pass": bool(len(maxabs) > 0 and frac >= L.FID_MIN_FRAC)}


def cmd_run(a) -> int:
    import sh5_gpulock as GL
    import sh5_guard as G
    G.require_sealed()
    return GL.run_locked_inprocess(lambda: _run_locked(a), max_wait=a.max_wait, min_avail_gb=a.min_avail_gb, owner="sh5_dacprox", job="sh5-ctx-build")


def _run_locked(a) -> int:
    import torch
    import sh5_gpulock as GL
    GL.require_locked(a.device)
    sys.path.insert(0, f"{TREE}/refe")
    sys.path.insert(0, f"{TREE}/eval")
    import ctx_cache as CC
    import navtrain_scenarios as NS
    import planner as PLM
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    plan = json.load(open(a.plan, encoding="utf-8"))
    Z = np.load(a.inputs, allow_pickle=True)
    all_tok = [str(t) for t in Z["token"]]
    tok_sets = {t: np.flatnonzero(Z["token_idx"] == i) for i, t in enumerate(all_tok)}
    toks = sorted(plan["tokens"])
    N = len(toks)
    row = {t: i for i, t in enumerate(toks)}
    os.makedirs(a.out_dir, exist_ok=True)
    ctx_path, done_path, fid_path = f"{a.out_dir}/ctx.npy", f"{a.out_dir}/ctx_done.npy", f"{a.out_dir}/fid.jsonl"
    if os.path.exists(ctx_path) and os.path.exists(done_path):
        ctx_mm = np.lib.format.open_memmap(ctx_path, mode="r+")
        done = np.load(done_path)
        if ctx_mm.shape[0] != N or done.shape[0] != N:
            print(f"REFUSED: existing ctx file {ctx_mm.shape} does not match the plan ({N} tokens); delete it or use another --out-dir", flush=True)
            return 2
        print(f"  resuming: {int(done.sum())}/{N} tokens already built", flush=True)
    else:
        ctx_mm = np.lib.format.open_memmap(ctx_path, mode="w+", dtype=np.uint16, shape=(N, P_TOK, D_CTX))
        done = np.zeros(N, bool)
    by_log: dict = {}
    for t in toks:
        if not done[row[t]]:
            by_log.setdefault(plan["token_log"][t], []).append(t)
    first_db = next(iter(plan["db"].values()))
    planner = REFePlanner(checkpoint=a.ckpt, images_root="", db_dir=os.path.dirname(first_db), backbone=a.backbone, device=a.device, select="best", rule="navsim_v1")
    if getattr(planner.model.cfg, "nav_mode", "goal") == "command" or planner.sanitize_goal:
        print("REFUSED: command-mode / sanitising planner", flush=True)
        return 2
    frs = {r: PLM.FrameResolver(db_dir="", images_root=r) for r in sorted(set(plan["token_root"][t] for t in toks))}
    tap = CC.VisualCtxTap(planner.model)
    fid_all, bad, t0 = [], [], time.time()
    if os.path.exists(fid_path):
        fid_all = [json.loads(l)["maxabs"] for l in open(fid_path, encoding="utf-8") for _ in [0] if l.strip()]
    fid_f = open(fid_path, "a", encoding="utf-8", newline="\n")
    n_new = 0
    for li, (log, lt) in enumerate(sorted(by_log.items())):
        db = plan["db"][log]
        got = set()
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            got.add(tok)
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            fr = frs[plan["token_root"][tok]]
            fr.db_dir = os.path.dirname(db)
            planner.frames = fr
            img = planner._image_for(ego)
            if img is None:
                bad.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            tap.reset()
            planner.infer(ego, img)
            v = tap.visual_ctx()                                                                       # fp32 [1, 7680, 256]; raises CaptureError unless the decoder's ctx IS scene_proj's 2nd output
            vb = v.to(torch.bfloat16)
            ctx_u16 = CC.to_u16(vb)[0]
            # ---- C-FID: the census logits of this token's S0 sets, recomputed from the captured ctx
            for si in tok_sets[tok]:
                x = torch.from_numpy(Z["traj3"][si]).float().unsqueeze(0).to(planner.device)
                with torch.no_grad():
                    z_new = planner.model.score_trajectories(x, v)[0].float().cpu().numpy()
                    z_bf = planner.model.score_trajectories(x, vb.float())[0].float().cpu().numpy()
                rec = {"token": tok, "set": int(si), "maxabs": fidelity_row(z_new, Z["z6"][si]), "bf16_maxabs": fidelity_row(z_bf, z_new)}
                fid_f.write(json.dumps(rec) + "\n")
                fid_all.append(rec["maxabs"])
            fid_f.flush()
            i = row[tok]
            ctx_mm[i] = ctx_u16.cpu().numpy() if hasattr(ctx_u16, "cpu") else ctx_u16
            done[i] = True
            n_new += 1
            if n_new % 20 == 0:
                np.save(done_path, done)
                ctx_mm.flush()
            if len(fid_all) >= EARLY_MIN_SETS and n_new <= EARLY_MIN_SETS * 3 and fid_gate(np.array(fid_all))["frac_within_tol"] < EARLY_MIN_FRAC:
                np.save(done_path, done)
                ctx_mm.flush()
                print(f"ZZCTX_FID_EARLYSTOP fidelity {fid_gate(np.array(fid_all))} after {n_new} tokens: the cached ctx does NOT reproduce the census logits; stopping", flush=True)
                tap.close()
                return 5
        for t in set(lt) - got:
            bad.append((t, "scenario builder skipped the token (margin guard)"))
        print(f"  [{li + 1}/{len(by_log)}] {log} done {int(done.sum())}/{N} bad {len(bad)} {time.time() - t0:.0f} s", flush=True)
    tap.close()
    ctx_mm.flush()
    np.save(done_path, done)
    h = hashlib.sha256()
    with open(ctx_path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    gate = fid_gate(np.array(fid_all))
    man = {"n": N, "tokens": toks, "done_file": done_path, "n_done": int(done.sum()), "ctx_file": ctx_path, "dtype": "uint16 (bf16 bits)", "shape": [N, P_TOK, D_CTX], "ctx_sha256": h.hexdigest(), "ckpt": a.ckpt,
           "c_fid": gate, "bad": bad[:50], "n_bad": len(bad), "seconds": round(time.time() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(man, open(f"{a.out_dir}/ctx_manifest.json", "w", encoding="utf-8", newline="\n"), indent=1)
    ok = bool(gate["pass"]) and not bad
    print(f"ZZCTX_{'OK' if ok else 'FAIL'} {int(done.sum())}/{N} tokens; C-FID {gate}", flush=True)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--inputs", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--stage-db-dir", default=None)
    p.add_argument("--images-root", action="append", default=None)
    sp.add_parser("probe")
    r = sp.add_parser("run")
    r.add_argument("--inputs", required=True, help="sh5_inputs.npz (label-free): token_idx, traj3, z6 for the fidelity gate")
    r.add_argument("--plan", required=True)
    r.add_argument("--ckpt", default=CKPT)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--max-wait", type=float, required=True)
    r.add_argument("--min-avail-gb", type=float, default=L.MIN_AVAIL_GB_BUILD)
    r.add_argument("--device", default="cuda")
    r.add_argument("--backbone", default="vitl16")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "probe":
        import sh5_gpulock as GL
        print(json.dumps(GL.probe(), indent=1))
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
