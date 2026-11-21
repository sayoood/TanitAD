#!/usr/bin/env python3
"""The EVAL split of the shared proxy cache (refe/proxy_cache.py's format): navtest tokens through the PLANNER'S OWN
input path (the path the E-6 tables were made with), one snapshot's frozen encoder side, stored as raw bits.

Same layout as the train split -- shard_NNNN.scene.bin / .visual.bin, frames.jsonl, rows.jsonl, manifest.json (LAST)
-- with rows carrying what an eval readout needs instead of teacher targets: the planner's ego vector and goal, and
the HUMAN future (8 poses at 0.5 s from the W3 export; its last pose is native step 19's time).
fp32 by default: the E-6 tables are fp32 planner forwards, and C2 below compares against them.
CONTROLS (gates): C1 the trajectory decoder on the captured scene_ctx == the planner's forward BIT-FOR-BIT;
C2 that forward reproduces the stored E-6 table's proposals (to_navsim, bar 1e-4 m on the GPU, 5e-3 m on a CPU smoke);
C3 every shard reads back as written.

    python eval/proxy_eval_cache.py --snapshot 015 --out <dir> [--tokens a.json,b.json] [--shards 2 --shard 0]
                                    [--limit-tokens N] [--device cuda|cpu] [--deadline HH:MM]
Each shard also writes shard_NNNN.fwd.npz: the planner's own proposals [n, 64, 20, 3] and logits [n, 64, 6].
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_checkpoint as EC        # noqa: E402  the seam's interpreter + environment
import refe_navtest_seam as SEAM    # noqa: E402  to_navsim, the export, the test DBs; puts refe/ on sys.path
import proxy_cache as PC            # noqa: E402  refe/proxy_cache.py: the ONE bit format

CHILD = "REFE_EVALCACHE_CHILD"
SNAPDIR = "D:/Projects/TanitAD/data/refe_runs_eval"
TABLES = f"{EC.DATA}/proptable"
TOK = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
       "A1_sub200_tokens.json")


def run(a) -> int:
    import torch
    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    want = set()
    for tf in a.tokens.split(","):
        sub = json.load(open(tf, encoding="utf-8"))
        want |= set(sub["tokens"] if isinstance(sub, dict) else sub)
    toks = [t for t in exp if t in want]
    if a.shards > 1:                                   # disjoint by LOG, so each process builds whole logs
        logs_all = sorted({exp[t]["log_name"] for t in toks})
        mine = set(logs_all[a.shard::a.shards])
        toks = [t for t in toks if exp[t]["log_name"] in mine]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    logs = sorted(by_log.items())
    if a.limit_tokens:
        keep, n = [], 0
        for lg, lt in logs:
            if n >= a.limit_tokens:
                break
            keep.append((lg, lt[:a.limit_tokens - n]))
            n += len(keep[-1][1])
        logs = keep
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = out / "frames.jsonl"
    done = {tuple(json.loads(l)["key"]) for l in open(fl, encoding="utf-8")} if fl.exists() else set()
    ck = f"{SNAPDIR}/snap_epoch{a.snapshot}.pt"
    planner = REFePlanner(checkpoint=ck, images_root=a.frames, db_dir=a.db_dir, backbone="vitl16", device=a.device,
                          select="best", rule=None)
    m, dev = planner.model, planner.device
    fp = PC.frozen_in_proxy_fingerprint(m)
    cap = {}
    m.dec[0].register_forward_pre_hook(lambda _m, args: cap.__setitem__("scene", args[1].detach()))
    m.score_dec[0].register_forward_pre_hook(lambda _m, args: cap.__setitem__("visual", args[1].detach()))
    tb = np.load(f"{TABLES}/sub200_ep{a.snapshot}/table.npz") if os.path.exists(
        f"{TABLES}/sub200_ep{a.snapshot}/table.npz") else None
    ti = {t: i for i, t in enumerate(tb["token"])} if tb is not None else {}
    shard = len(list(out.glob("shard_*.scene.bin")))
    buf_s, buf_v, lines, rlines, fwd_t, fwd_l = [], [], [], [], [], []
    c1, c2, n_new, stopped = 0.0, 0.0, 0, None

    def decode(scene_ctx, ego, goal):
        g = goal.unsqueeze(-1) * m.goal_freqs
        gf = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        q = m.queries.expand(1, -1, -1) + m.ego_enc(torch.cat([ego, gf], -1)).unsqueeze(1)
        for blk in m.dec:
            q = blk(q, scene_ctx)
        return m.traj_head(q).view(1, m.cfg.n_proposals, m.cfg.horizon_steps, m.cfg.traj_dim)[0]

    def flush():
        nonlocal shard, buf_s, buf_v, lines, rlines, fwd_t, fwd_l
        if not buf_s:
            return
        # the planner's full-forward record per token: native proposals + scorer logits (gates G2/G3)
        np.savez(out / f"shard_{shard:04d}.fwd.npz", traj=np.stack(fwd_t), logits=np.stack(fwd_l))
        for suffix, buf in (("scene", buf_s), ("visual", buf_v)):
            p = out / f"shard_{shard:04d}.{suffix}.bin"
            b = b"".join(buf)
            with open(p, "wb") as f:
                f.write(b)
                f.flush()
                os.fsync(f.fileno())
            if PC.sha256_file(p) != PC.sha256_bytes(b):
                raise SystemExit(f"C3 FAILED: {p} does not read back as written")
        with open(fl, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(l) + "\n" for l in lines))
        with open(out / "rows.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(l) + "\n" for l in rlines))
        shard += 1
        buf_s, buf_v, lines, rlines, fwd_t, fwd_l = [], [], [], [], [], []
    t0 = time.time()
    for lg, lt in logs:
        if stopped:
            break
        lt = [t for t in lt if (lg, t) not in done]
        if not lt:
            continue
        for sc in NS.build_scenarios_for_log(os.path.join(a.db_dir, f"{lg}.db"), lt, history_rows=1, future_rows=80):
            if PC.past(a.deadline):
                stopped = f"deadline {a.deadline}"
                break
            tok = sc._initial_lidar_token
            planner._scenario = sc
            planner.initialize(PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                                     mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                continue
            traj, logits, _k = planner.infer(ego, img)             # the planner's OWN call
            ego_vec, goal = planner._ego_vec(ego), planner._goal_for(ego).to(dev)
            with torch.no_grad():
                dt = decode(cap["scene"], ego_vec, goal)
            c1 = max(c1, float((dt - traj).abs().max()))
            if tok in ti:
                nav = np.stack([SEAM.to_navsim(traj[j].float().cpu().numpy().astype(np.float64)) for j in range(64)])
                c2 = max(c2, float(np.abs(nav.astype(np.float32) - tb["proposals"][ti[tok]]).max()))
            fwd_t.append(traj.float().cpu().numpy())
            fwd_l.append(logits.float().cpu().numpy())
            sbits, vbits = PC.to_bits(cap["scene"][0]), PC.to_bits(cap["visual"][0])
            buf_s.append(sbits)
            buf_v.append(vbits)
            lines.append({"key": [lg, tok], "shard": shard, "index": len(buf_s) - 1,
                          "scene_sha256": PC.sha256_bytes(sbits)})
            rlines.append({"key": [lg, tok], "ego": ego_vec[0].cpu().tolist(), "goal": goal[0].cpu().tolist(),
                           "human_future_poses": exp[tok]["human_future_poses"],
                           "human_future_sampling": exp[tok]["human_future_sampling"]})
            n_new += 1
            if len(buf_s) >= a.shard_frames:
                flush()
    flush()
    PREV_PASSES = (json.load(open(out / "manifest.json", encoding="utf-8")).get("passes", [])
                   if (out / "manifest.json").exists() else [])
    shards = sorted(out.glob("shard_*.scene.bin"))
    bar2 = 1e-4 if str(dev).startswith("cuda") else 5e-3
    mf = {"version": 1, "split": "eval", "snapshot": ck, "snapshot_sha256": PC.sha256_file(ck),
          "frozen_in_proxy_sha256": fp, "dtype": "fp32", "device": str(dev), "input_path": "planner.REFePlanner.infer",
          "tokens": a.tokens, "export": a.export,
          "shapes": {"scene": [PC.SCENE_TOKENS, PC.WIDTH], "visual": [PC.VISUAL_TOKENS, PC.WIDTH]},
          "shards": [{"scene": p.name, "scene_sha256": PC.sha256_file(p),
                      "visual": p.name.replace(".scene.", ".visual."),
                      "visual_sha256": PC.sha256_file(p.with_name(p.name.replace(".scene.", ".visual.")))} for p in shards],
          "frames": sum(1 for _ in open(fl, encoding="utf-8")) if fl.exists() else 0,
          "last_pass": {"new_frames": n_new, "seconds": round(time.time() - t0, 1), "stopped": stopped,
                        "C1_decode_vs_forward_max_abs": c1, "C2_forward_vs_table_m": c2, "C2_bar_m": bar2},
          "passes": PREV_PASSES + [{"new_frames": n_new, "seconds": round(time.time() - t0, 1), "stopped": stopped,
                        "C1_decode_vs_forward_max_abs": c1, "C2_forward_vs_table_m": c2, "C2_bar_m": bar2}],
          "tool_sha256": PC.sha256_file(__file__), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmpm = out / "manifest.json.tmp"
    with open(tmpm, "w", encoding="utf-8", newline="\n") as f:
        json.dump(mf, f, indent=1)
    os.replace(tmpm, out / "manifest.json")
    ok = c1 == 0.0 and c2 <= bar2
    print(f"  cached {n_new} tokens ({mf['frames']} total) in {mf['last_pass']['seconds']} s; C1 {c1:.3g} "
          f"(bit-exact: {c1 == 0.0}); C2 {c2:.3g} m (bar {bar2}); stopped: {stopped}")
    print("ZZEVALCACHE_OK" if ok else "ZZEVALCACHE_FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="015")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=TOK)
    ap.add_argument("--frames", default=f"{EC.DATA}/frames")
    ap.add_argument("--db-dir", default=SEAM.TEST_DB_DIR)
    ap.add_argument("--export", default=SEAM.W3_EXPORT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-tokens", type=int, default=0)
    ap.add_argument("--shard-frames", type=int, default=64)
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--shards", type=int, default=1, help="split the tokens by LOG into this many disjoint parts")
    ap.add_argument("--shard", type=int, default=0)
    a = ap.parse_args()
    if os.environ.get(CHILD) != "1":
        env = EC.env_driverl()
        env[CHILD] = "1"
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + os.path.join(os.path.dirname(HERE), "refe")
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=HERE, env=env)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
