#!/usr/bin/env python3
"""Extract `scene_ctx` (fp32 [64, 256]) for the 1,058 navtest C3 tokens in the SAME forward that produced `c3_full_ctx` (P2 of the scene-tap probe).

THE FORWARD.  Exactly `eval/aux_navtest_forward.py`'s loop (NS.build_scenarios_for_log -> REFePlanner.initialize -> planner._image_for -> SEAM.set_driving_command -> ONE `planner.infer` per token,
batch 1, rule navsim_v1_lane, checkpoint `model_sft4_B_u400.pt` md5 8e382eeb..., fp32, no autocast).  Three forward hooks are armed on that one call:
    model.scene_proj            (called TWICE per forward: output 1 = scene_ctx [1, 64, 256] = scene_proj(reg_compress(registers, visual)); output 2 = visual_ctx [1, 7680, 256])
    model.dec[0]   pre-hook     args[1] = the context the TRAJECTORY decoder reads = scene_ctx  (model.py:773-774)
    model.score_dec[0] pre-hook args[1] = the context the SCORING decoder reads  = visual_ctx.detach()  (model.py:784)
PROOFS, per token, in the same forward (all recorded in extract_log.jsonl):
    tap identity    AUX.tap_is(dec_ctx, scene_ctx) and AUX.tap_is(score_ctx, visual_ctx)  (same storage and shape): the saved tensor IS what a tap='scene' head would read
    same forward    max |visual_ctx(this forward) - c3_full_ctx cache| must be <= 1e-4 (STOP otherwise: it is not the same forward); also the scorer logits vs table.npz
OUTPUT  <out>/ctx_<token>.npy (scene_ctx) + logits_<token>.npy (the scorer's logits of this forward), the layout of aux_navtest_forward's cache; scene_ctx_manifest.json (sha256 per file) when complete.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe launch_extract.py [--limit-tokens 8] [--max-lock-s 1200]      (launch_extract.py sets the nuPlan environment)
Exit: 0 ok | 4 refused | 5 GPU busy | 6 NOT THE SAME FORWARD (stopped) | 7 identity failure.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PKG, "eval"))
sys.path.insert(0, os.path.join(PKG, "refe"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import aux_navtest_forward as ANF  # noqa: E402  (the forward's own module: constants, GPU lock, ctx tap, cache layout, token selection)
import aux_drivable_head as AUX  # noqa: E402

ART = "D:/Projects/TanitAD-artifacts"
VIS_CACHE = f"{ART}/aux_navtest_forward/c3_full_ctx"
OUT_DEFAULT = f"{ART}/aux_navtest_forward/c3_full_scene_ctx"
MAX_ABS_D = 1e-4


class Taps:
    """Hooks on one `planner.infer` call (reset before each token)."""

    def __init__(self, model):
        self.sp = []
        self.dec_ctx = None
        self.score_ctx = None
        self._h = [model.scene_proj.register_forward_hook(lambda m, i, o: self.sp.append(o)),
                   model.dec[0].register_forward_pre_hook(lambda m, args: setattr(self, "dec_ctx", args[1])),
                   model.score_dec[0].register_forward_pre_hook(lambda m, args: setattr(self, "score_ctx", args[1]))]

    def reset(self):
        self.sp, self.dec_ctx, self.score_ctx = [], None, None

    def close(self):
        for h in self._h:
            h.remove()


def sha256_file(p):
    return ANF.sha256_file(p)


def run(a) -> int:
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    doc = json.load(open(ANF.DEFAULT_TOKENS, encoding="utf-8"))
    tab = np.load(ANF.DEFAULT_TABLE, allow_pickle=False)
    table_tokens = [str(t) for t in tab["token"]]
    token_log = doc["token_log"]
    ref_logits = np.asarray(tab["logits"], np.float32)
    tindex = {t: i for i, t in enumerate(table_tokens)}
    want_all = ANF.select_tokens(table_tokens, token_log, a.limit_tokens)
    need = [t for t in want_all if not ANF.cache_complete(a.out, t)]
    print(f"tokens requested {len(want_all)}, already extracted {len(want_all) - len(need)}, to do {len(need)}", flush=True)
    by_log: dict = {}
    for t in need:
        by_log.setdefault(token_log[t], []).append(t)
    h = hashlib.md5()
    with open(ANF.CKPT, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    if not h.hexdigest().startswith(ANF.CKPT_MD5_PREFIX):
        print(f"REFUSED: checkpoint md5 {h.hexdigest()} does not start with {ANF.CKPT_MD5_PREFIX}")
        return 4
    logf = open(os.path.join(a.out, "extract_log.jsonl"), "a", encoding="utf-8")
    state = {"body": None, "t0": 0.0}

    def lease():
        if state["body"] is not None and time.time() - state["t0"] <= a.max_lock_s:
            return
        if state["body"] is not None:
            ANF.release_gpu_lock(ANF.LOCK, state["body"])
            state["body"] = None
        waited = time.time()
        while True:
            try:
                state["body"] = ANF.acquire_gpu_lock(wait_timeout_s=240.0)
                state["t0"] = time.time()
                print(f"GPU lock taken after {time.time() - waited:.0f} s", flush=True)
                return
            except ANF.GpuBusy as e:
                if time.time() - waited > a.max_wait_s:
                    raise
                print(f"GPU busy ({str(e)[:100]}); retry in 90 s", flush=True)
                time.sleep(90)

    rc = 0
    planner = taps = tap = None
    try:
        if need:
            import navtrain_scenarios as NS
            from planner import REFePlanner
            from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
            import refe_navtest_seam as SEAM
            lease()
            planner = REFePlanner(checkpoint=ANF.CKPT, images_root=ANF.FRAMES, db_dir=ANF.DB_DIR, backbone="vitl16", device="cuda", select="best", rule=ANF.RULE)
            taps = Taps(planner.model)
            cmd_mode = getattr(planner.model.cfg, "nav_mode", "goal") == "command"
            exp = None
            if cmd_mode or planner.sanitize_goal:
                exp = json.load(gzip.open(ANF.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
            exp_get = (lambda t: exp[t]) if exp is not None else (lambda t: {})
            n_done = 0
            for li, (lg, lt) in enumerate(sorted(by_log.items())):
                lease()
                db = os.path.join(ANF.DB_DIR, f"{lg}.db")
                gen = NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80)
                got = set()
                for sc in gen:
                    tok = sc._initial_lidar_token
                    got.add(tok)
                    planner._scenario = sc
                    planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(), mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
                    ego = sc.get_ego_state_at_iteration(0)
                    img = planner._image_for(ego)
                    if img is None:
                        print(f"REFUSED: token {tok}: no frames ({planner.frames.miss_reason})")
                        return 4
                    SEAM.set_driving_command(planner, tok, exp_get(tok))
                    taps.reset()
                    t1 = time.time()
                    _traj, score, _k = planner.infer(ego, img)
                    torch.cuda.synchronize()
                    wall = time.time() - t1
                    if len(taps.sp) < 2 or taps.dec_ctx is None or taps.score_ctx is None:
                        print(f"REFUSED: token {tok}: hooks saw {len(taps.sp)} scene_proj calls (expected 2)")
                        return 4
                    scene, vis = taps.sp[-2], taps.sp[-1]               # scene_proj(scene) first, scene_proj(visual) second (model.py:758-759)
                    scene_np = scene.detach().float().cpu().numpy()
                    vis_np = vis.detach().float().cpu().numpy().reshape(-1, vis.shape[-1])
                    logits = score.float().cpu().numpy().astype(np.float32)
                    ref_vis = np.load(os.path.join(a.visual_cache, f"ctx_{tok}.npy"))
                    d = np.abs(vis_np.astype(np.float64) - ref_vis.astype(np.float64))
                    rel = float(np.linalg.norm(vis_np - ref_vis) / max(float(np.linalg.norm(ref_vis)), 1e-30))
                    lg_ident = bool(np.array_equal(logits, ref_logits[tindex[tok]]))
                    rec = {"token": tok, "log": lg, "scene_shape": list(scene_np.shape), "visual_shape": list(vis_np.shape),
                           "n_scene_proj_calls": len(taps.sp), "tap_is_scene_dec_ctx": bool(AUX.tap_is(taps.dec_ctx, scene)),
                           "tap_is_visual_score_ctx": bool(AUX.tap_is(taps.score_ctx, vis)), "scene_is_not_visual": bool(not AUX.tap_is(scene, vis)),
                           "visual_max_abs_d_vs_cache": float(d.max()), "visual_rel_l2_vs_cache": rel, "visual_bit_identical": bool(d.max() == 0.0),
                           "scorer_logits_bit_identical_vs_table": lg_ident, "scene_abs_mean": float(np.abs(scene_np).mean()), "scene_finite": bool(np.isfinite(scene_np).all()),
                           "forward_s": round(wall, 3), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()
                    if not (rec["tap_is_scene_dec_ctx"] and rec["tap_is_visual_score_ctx"] and rec["scene_is_not_visual"] and rec["scene_finite"] and tuple(scene_np.shape) == (1, 64, 256)):
                        print(f"IDENTITY FAILURE at {tok}: {rec}")
                        return 7
                    if rec["visual_max_abs_d_vs_cache"] > MAX_ABS_D:
                        print(f"NOT THE SAME FORWARD at {tok}: max |d| {rec['visual_max_abs_d_vs_cache']:.3g} > {MAX_ABS_D} (rel L2 {rel:.3g}); STOPPED, nothing further extracted")
                        json.dump(rec, open(os.path.join(a.out, "SAME_FORWARD_FAILED.json"), "w"), indent=1)
                        return 6
                    ANF.cache_save(a.out, tok, scene, logits)
                    n_done += 1
                miss = set(lt) - got
                if miss:
                    print(f"REFUSED: {len(miss)} tokens skipped by the scenario builder in {lg} (first {sorted(miss)[0]})")
                    return 4
                print(f"  [{li + 1}/{len(by_log)}] {lg}: {len(lt)} tokens, {n_done}/{len(need)} done, {time.time() - t_start:.0f} s", flush=True)
    except ANF.GpuBusy as e:
        print(f"GPU BUSY: {e}")
        rc = 5
    finally:
        if taps is not None:
            taps.close()
        if state["body"] is not None:
            ANF.release_gpu_lock(ANF.LOCK, state["body"])
    if rc:
        return rc
    # ---- complete? summary + manifest ------------------------------------------------------------------------------------------------------------------------
    done = [t for t in want_all if ANF.cache_complete(a.out, t)]
    recs = [json.loads(l) for l in open(os.path.join(a.out, "extract_log.jsonl"), encoding="utf-8")]
    last = {r["token"]: r for r in recs}                                   # a re-run token keeps its latest record
    sel = [last[t] for t in want_all if t in last]
    summary = {"tool": "extract_scene_ctx/1", "tokens_requested": len(want_all), "tokens_extracted": len(done), "tokens_with_a_proof_record": len(sel),
               "visual_max_abs_d_max": max((r["visual_max_abs_d_vs_cache"] for r in sel), default=None),
               "visual_bit_identical_tokens": sum(r["visual_bit_identical"] for r in sel), "scorer_logits_bit_identical_tokens": sum(r["scorer_logits_bit_identical_vs_table"] for r in sel),
               "tap_identity_all": all(r["tap_is_scene_dec_ctx"] and r["tap_is_visual_score_ctx"] and r["scene_is_not_visual"] for r in sel),
               "threshold": MAX_ABS_D, "same_forward": bool(sel and max(r["visual_max_abs_d_vs_cache"] for r in sel) <= MAX_ABS_D), "wall_s": round(time.time() - t_start, 1)}
    json.dump(summary, open(os.path.join(a.out, "extract_summary.json" if not a.limit_tokens else "extract_summary_limit.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if not a.limit_tokens and len(done) == len(want_all):
        man = {t: {"sha256": sha256_file(os.path.join(a.out, f"ctx_{t}.npy")), "bytes": os.path.getsize(os.path.join(a.out, f"ctx_{t}.npy"))} for t in sorted(want_all)}
        body = json.dumps(man, sort_keys=True, indent=1) + "\n"
        with open(os.path.join(a.out, "scene_ctx_manifest.json"), "wb") as f:
            f.write(body.encode("utf-8"))
        print(f"MANIFEST {len(man)} files sha256 {hashlib.sha256(body.encode('utf-8')).hexdigest()}")
        print(f"ZZSCENE_OK {len(done)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--visual-cache", default=VIS_CACHE)
    ap.add_argument("--limit-tokens", type=int, default=None, help="smoke: only this many tokens, whole logs first (the same selection as aux_navtest_forward)")
    ap.add_argument("--max-lock-s", type=float, default=1200.0)
    ap.add_argument("--max-wait-s", type=float, default=48 * 3600.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
