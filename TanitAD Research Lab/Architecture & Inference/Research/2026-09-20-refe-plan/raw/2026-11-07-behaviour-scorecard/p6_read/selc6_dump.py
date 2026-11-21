#!/usr/bin/env python3
"""selc6_dump.py -- the GPU half of the pre-registered P6 read (PREREG_P6_SCORER_SPEED.md section 3, stage 1). A P6 VARIANT of selc_dump.py.

What differs from selc_dump.py (the card's differences table, and nothing else):
  checkpoint  the P6 FINAL checkpoint given by --ckpt (no u400 sha prefix); its full sha256 is computed and recorded in the dump
  scorer      6 logits, rule `navsim_v1` (planner DEFAULT_RULE), select "best"
  nav input   nav_mode `command`: NAVSIM driving_command via SEAM.set_driving_command (identical call to selc_dump.py)
  precision   FP32 (Master Mind requirement): refuses to run under autocast, asserts every parameter is float32, records both. bf16 would put the raw plan
              coordinates on a 0.25 m grid at 32-64 m and corrupt v4/v0; selc6_analyze.py gate G6-FP32 re-checks the output.
  tokens      the 1,058 C3 tokens (tokens_all.json); no u400 smoke identity (it would compare a P6 forward to u400's table)
The token loop, the GPU lock policy (YAX lock, never forced, one segment per invocation), gt future, human8 and `ego` vector are selc_dump.py's own helpers,
IMPORTED (not copied). Exit codes as selc_dump: 0 all done | 3 segment done, more remain | 4 refused | 5 GPU busy | 1 error.

    python selc6_dump.py --ckpt <ckpt_last.pt> --out DIR/dump [--device cuda] [--segment-tokens 1100]
    python selc6_dump.py --consolidate --ckpt <ckpt> --out DIR/dump
    python selc6_dump.py --smoke --ckpt <ckpt> --out DIR/dump_smoke    # 3 tokens, GPU, the FIRST runbook step (plumbing + shapes + fp32)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SELC = "E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107"
sys.path.insert(0, SELC)
import selc_dump as SD  # noqa: E402  (pinned by sha256 in selc6_analyze.PINS)

TOKENS_ALL = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json"
RULE = "navsim_v1"
N_OUT = 6
NAV_MODE = "command"
OWNER = "selc6_dump"
FIELDS = SD.FIELDS


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def assert_fp32(planner, say):
    """Master Mind requirement: the dump runs in fp32, never bf16/fp16."""
    import torch
    if torch.is_autocast_enabled() or (hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled()):
        raise RuntimeError("REFUSED: autocast is enabled; the P6 read must run in fp32")
    bad = {str(p.dtype) for p in planner.model.parameters() if p.dtype != torch.float32}
    if bad:
        raise RuntimeError(f"REFUSED: model parameters are not all float32: {sorted(bad)}")
    say("fp32 confirmed: autocast off, all parameters float32")


def run_segment(a, say) -> int:
    doc = json.load(open(a.tokens, encoding="utf-8"))
    token_log = doc["token_log"]
    toks_all = doc["tokens"][: a.limit_tokens] if a.limit_tokens else doc["tokens"]
    by = {}
    for t in toks_all:
        by.setdefault(token_log[t], []).append(t)
    os.makedirs(os.path.join(a.out, "chunks"), exist_ok=True)
    pending = [lg for lg in sorted(by) if not os.path.exists(SD.chunk_path(a.out, lg))]
    if not pending:
        say("all logs done")
        return 0
    if SD.free_gib(a.out) < SD.MIN_FREE_GIB:
        say(f"REFUSED: {SD.free_gib(a.out):.2f} GiB free < {SD.MIN_FREE_GIB}")
        return 4
    plan, n = [], 0
    for lg in pending:
        if plan and n + len(by[lg]) > a.segment_tokens:
            break
        plan.append(lg)
        n += len(by[lg])
    ck_sha = sha256_file(a.ckpt)
    say(f"checkpoint {a.ckpt} sha256 {ck_sha}")
    if a.expect_ckpt_sha and ck_sha != a.expect_ckpt_sha:
        say(f"REFUSED: checkpoint sha256 != --expect-ckpt-sha {a.expect_ckpt_sha}")
        return 4
    import gzip
    exp = json.load(gzip.open(os.path.join(SD.navsim_root(), "exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"), "rt", encoding="utf-8"))["tokens"]
    body = None
    if a.device == "cuda" and not a.no_gpu_lock:
        import gpu_lock as GL
        t0 = time.time()
        while True:
            if os.path.exists(SD.MARK):
                if time.time() - t0 > a.give_up_h * 3600:
                    say("GPU BUSY: chain marker present")
                    return 5
                say("chain marker present, waiting")
                time.sleep(120)
                continue
            try:
                body = GL.acquire_blocking(max(60.0, a.give_up_h * 3600 - (time.time() - t0)), poll_s=60.0, log=say, owner=OWNER, wait_timeout_s=240.0)
                break
            except GL.GpuBusy as e:
                say(f"GPU BUSY (gave up, nothing loaded): {e}")
                return 5
        say(f"GPU lock taken: {body}")
    try:
        import torch
        import navtrain_scenarios as NS
        import refe_navtest_seam as SEAM
        from planner import REFePlanner
        from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
        planner = REFePlanner(checkpoint=a.ckpt, images_root=SD.FRAMES, db_dir=a.db_dir, backbone="vitl16", device=a.device, select="best", rule=RULE)
        nav_mode = getattr(planner.model.cfg, "nav_mode", "goal")
        say(f"planner: device {planner.device} rule {planner.rule} nav_mode {nav_mode} goal_fix {planner.goal_fix} repair_last_heading {planner.repair_last_heading}")
        if planner.rule != RULE or planner.select != "best" or nav_mode != NAV_MODE:
            raise RuntimeError(f"REFUSED: planner rule/select/nav_mode = {planner.rule}/{planner.select}/{nav_mode}, need {RULE}/best/{NAV_MODE}")
        assert_fp32(planner, say)
        meta = dict(rule=np.array(RULE), ckpt=np.array(os.path.abspath(a.ckpt)), ckpt_sha256=np.array(ck_sha), nav_mode=np.array(nav_mode), select=np.array("best"),
                    fp32=np.array(True))

        def run_log(lg, toks):
            recs = []
            db = os.path.join(a.db_dir, f"{lg}.db")
            got = set()
            mtimes = SD.meta_times(lg)
            for sc_ in NS.build_scenarios_for_log(db, toks, history_rows=1, future_rows=80):
                tok = sc_._initial_lidar_token
                got.add(tok)
                planner._scenario = sc_
                planner.initialize(PlannerInitialization(route_roadblock_ids=sc_.get_route_roadblock_ids(), mission_goal=sc_.get_mission_goal(), map_api=sc_.map_api))
                ego = sc_.get_ego_state_at_iteration(0)
                img = planner._image_for(ego)
                if img is None:
                    raise RuntimeError(f"token {tok}: no frames ({planner.frames.miss_reason})")
                SEAM.set_driving_command(planner, tok, exp[tok])
                with torch.no_grad():
                    traj, score, k = planner.infer(ego, img)
                if traj.dtype != torch.float32 or score.dtype != torch.float32:
                    raise RuntimeError(f"REFUSED: output dtype {traj.dtype}/{score.dtype} is not float32")
                ego_fed = planner.last_inputs[0].detach().cpu().numpy().astype(np.float64)[0]
                tr = traj.float().cpu().numpy()
                props8 = np.stack([SEAM.to_navsim(planner.executed(traj, j).float().cpu().numpy()) for j in range(traj.shape[0])]).astype(np.float32)
                rec = {"token": tok, "log": lg, "fingerprint": exp[tok]["fingerprint"], "logits": score.float().cpu().numpy().astype(np.float32), "pick": int(k),
                       "traj": tr.astype(np.float32), "props8": props8, "ego": ego_fed, "human8": np.asarray(exp[tok]["human_future_poses"], np.float64)}
                rec.update(SD.gt_future(sc_, ego))
                theirs = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)
                mt = mtimes.get(tok)
                rec["frame_ctrl_m"] = SD.frame_ctrl_meta(sc_, ego, theirs, mt[0], mt[1]) if mt else float("nan")
                if not (props8.shape == (64, 8, 3) and tr.shape == (64, 20, 3) and rec["logits"].shape == (64, N_OUT)):
                    raise RuntimeError(f"token {tok}: shapes {props8.shape} {tr.shape} {rec['logits'].shape}; need (64,8,3) (64,20,3) (64,{N_OUT})")
                assert np.isfinite(props8).all() and np.isfinite(tr).all()
                recs.append(rec)
            miss = set(toks) - got
            if miss:
                raise RuntimeError(f"{len(miss)} tokens skipped by the scenario builder in {lg} (first {sorted(miss)[0]})")
            order = {t: i for i, t in enumerate(sorted(toks))}
            return sorted(recs, key=lambda r: order[r["token"]])

        t_seg, nd = time.time(), 0
        for i, lg in enumerate(plan):
            t_log = time.time()
            recs = run_log(lg, by[lg])
            fc = np.array([r["frame_ctrl_m"] for r in recs])
            # frame control is RECORDED, not fatal (not a card gate; the card's G6-1 compares picks to the P6 seam). Failures are listed in consolidate.json.
            SD.save_chunk(SD.chunk_path(a.out, lg), recs)
            nd += len(recs)
            say(f"[{i + 1}/{len(plan)}] {lg}: {len(recs)} tokens in {time.time() - t_log:.0f} s | segment {nd}/{n} | {(time.time() - t_seg) / max(nd, 1):.2f} s/token | "
                f"frame ctrl fails {int((~(fc <= SD.FC_TOL_M)).sum())}")
            if SD.free_gib(a.out) < SD.MIN_FREE_GIB:
                say("STOP: disk")
                return 4
        json.dump({k: str(v) for k, v in meta.items()}, open(os.path.join(a.out, "dump_meta.json"), "w"), indent=1)
    finally:
        if body:
            import gpu_lock as GL
            GL.release_gpu_lock(GL.LOCK, body)
            say("GPU lock released")
    left = [lg for lg in sorted(by) if not os.path.exists(SD.chunk_path(a.out, lg))]
    return 0 if not left else 3


def consolidate(a, say) -> int:
    doc = json.load(open(a.tokens, encoding="utf-8"))
    toks_all = doc["tokens"][: a.limit_tokens] if a.limit_tokens else doc["tokens"]
    by = {}
    for t in toks_all:
        by.setdefault(doc["token_log"][t], []).append(t)
    parts = {k: [] for k in ("token", "log", "fingerprint") + FIELDS}
    for lg in sorted(by):
        with np.load(SD.chunk_path(a.out, lg), allow_pickle=False) as z:
            for k in parts:
                parts[k].append(z[k])
    D = {k: np.concatenate(v) for k, v in parts.items()}
    tok = [str(t) for t in D["token"]]
    assert set(tok) == set(toks_all) and len(set(tok)) == len(tok), "dump tokens != the token list"
    meta = json.load(open(os.path.join(a.out, "dump_meta.json")))
    ck_sha = sha256_file(a.ckpt)
    assert meta["ckpt_sha256"] == ck_sha, f"dump was made by a different checkpoint ({meta['ckpt_sha256'][:16]} vs {ck_sha[:16]})"
    np.savez(os.path.join(a.out, "dump_p6.npz"), rule=np.array(meta["rule"]), ckpt=np.array(meta["ckpt"]), ckpt_sha256=np.array(ck_sha),
             nav_mode=np.array(meta["nav_mode"]), select=np.array("best"), sampling=np.array([8, 0.5]), **D)
    fc = D["frame_ctrl_m"]
    rep = {"n_tokens": len(tok), "n_logs": len(by), "frame_control_n_fail_or_nan": int((~(fc <= SD.FC_TOL_M)).sum()), "fail_tokens": [tok[i] for i in np.where(~(fc <= SD.FC_TOL_M))[0]][:50],
           "gt_valid_share": float(D["gt_valid"].mean()), "ckpt_sha256": ck_sha}
    json.dump(rep, open(os.path.join(a.out, "consolidate.json"), "w"), indent=1)
    say(f"consolidated: {rep}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=TOKENS_ALL)
    ap.add_argument("--db-dir", default=SD.DB_DIR)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--segment-tokens", type=int, default=1100)
    ap.add_argument("--give-up-h", type=float, default=10.0)
    ap.add_argument("--expect-ckpt-sha")
    ap.add_argument("--limit-tokens", type=int, default=0, help="first N tokens of the list (the --smoke step uses 3); 0 = all 1,058")
    ap.add_argument("--smoke", action="store_true", help="3 tokens into --out: plumbing, shapes, fp32; consolidates too")
    ap.add_argument("--consolidate", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.limit_tokens = a.limit_tokens or 3
    os.makedirs(a.out, exist_ok=True)

    def say(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(os.path.join(a.out, "dump.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        if a.consolidate:
            return consolidate(a, say)
        rc = run_segment(a, say)
        return consolidate(a, say) if (a.smoke and rc == 0) else rc
    except BaseException:                                   # noqa: B902
        say("ERROR\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
