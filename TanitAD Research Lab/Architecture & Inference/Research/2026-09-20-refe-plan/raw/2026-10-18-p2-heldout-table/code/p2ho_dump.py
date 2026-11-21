"""P2 forward on NAVTRAIN HELD-OUT tokens: every proposal + the scorer's six logits + the scorer head's INPUT (pre-head features), ONE forward per token.

Derived from `eval/refe_navtest_seam.py` (landed; `p2ho_dump_vs_refe_navtest_seam.diff` is the full edit list).  What is the SAME as the landed dump:
the planner (`planner.REFePlanner`, rule navsim_v1, A7 last-heading repair, goal fix pdm_route), the scenario builder call (`history_rows=1, future_rows=80`),
the input construction (`planner._image_for`, `planner.infer`), the proposal conversion (`to_navsim(planner.executed(traj, j))`), the pick (`planner._pick` inside `infer`).
What CHANGED, and why:
  * TOKENS / LOGS / DBs / IMAGES come from the plan (`p2ho_audit.py`), not from W3's navtest export: a held-out navtrain token has no W3 export.  The DB of a log can sit in
    splits/ or splits/trainval/ and the camera frames of different tokens in different roots, so `planner.frames` is switched PER TOKEN (db_dir = the log's DB folder,
    images_root = the root that resolved this token in the audit).  `FrameResolver` itself is untouched.
  * FINGERPRINT: no W3 fingerprint exists; the row's fingerprint is sha256(log | token | ego vector | goal | resolved frame names)[:16] of the inputs the planner actually used.
  * NO frame/time control against W3's `human_future_poses` (there is none for navtrain): the dump's fidelity is established by the driver's DG-DUMP gate instead
    (this same script, on NAVTEST tokens, must reproduce the landed `table_p2_full.npz` rows) and by the scoring-side fidelity gate against the banked held-out labels.
  * NAVSIM `driving_command`: a command-mode checkpoint needs it and navtrain tokens carry none here -> REFUSED loudly (P2 is goal-mode; the report records nav_mode).
  * PRE-HEAD FEATURES: a forward hook on `model.score_head` records its INPUT `s` [1, 64, dec_width] from the SAME forward (no second model call, no model code touched).
    Exactly one hook call per `infer` is asserted.  Saved fp16 beside the dump.
  * `proposals20` (the executed plans on REFe's own 20-pose 0.2 s grid, before `to_navsim`) is saved too: SR-TTC may need the native grid.
Prints ZZP2HODUMP_OK <rows> only when every requested token has a finite row.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe p2ho_dump.py --ckpt <model_final.pt> --plan <plan.json> --out <dump.npz> --feat-out <feat.npz> --device cuda
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
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
sys.path.insert(0, os.path.join(TREE, "refe"))
sys.path.insert(0, os.path.join(TREE, "eval"))
NAVSIM_DT, NAVSIM_N = 0.5, 8
SRC_DT = 0.2


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--plan", required=True, help="{'tokens', 'token_log', 'token_root', 'db'} from p2ho_audit.py (or a smoke plan of the same shape)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat-out", default=None)
    ap.add_argument("--arm", default="REFe_p2heldout")
    ap.add_argument("--select", default="best")
    ap.add_argument("--rule", default="navsim_v1", choices=("v2_shape", "navsim_v1", "navsim_v1_lane"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    toks = sorted(plan["tokens"])
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(plan["token_log"][t], []).append(t)
    print(f"  tokens: {len(toks):,} over {len(by_log)} logs; ckpt {a.ckpt}", flush=True)

    import navtrain_scenarios as NS
    import planner as PLM
    from refe_navtest_seam import to_navsim            # landed conversion, imported unchanged (its main() is guarded)
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    first_db = next(iter(plan["db"].values()))
    planner = REFePlanner(checkpoint=a.ckpt, images_root="", db_dir=os.path.dirname(first_db), backbone=a.backbone, device=a.device, select=a.select, rule=a.rule)
    nav_mode = getattr(planner.model.cfg, "nav_mode", "goal")
    if nav_mode == "command" or planner.sanitize_goal:
        print("  REFUSED: a command-mode / sanitising planner needs NAVSIM's driving_command, which navtrain tokens do not carry here", flush=True)
        return 2
    print(f"  planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} device={planner.device} select={planner.select} rule={planner.rule} "
          f"goal_fix={planner.goal_fix} repair_last_heading={planner.repair_last_heading} nav_mode={nav_mode}", flush=True)
    # one FrameResolver per image root; switched per token (db_dir is per log).  The resolver class is the planner's own, unmodified.
    frs = {r: PLM.FrameResolver(db_dir="", images_root=r) for r in sorted(set(plan["token_root"].values()))}
    feat_box: list = []
    hook = planner.model.score_head.register_forward_hook(lambda m, inp, out: feat_box.append((inp[0].detach().float().cpu().numpy().astype(np.float16), out.detach().float().cpu().numpy())))
    rows_tok, rows_fp, rows_pose, misses = [], [], [], []
    dump_props, dump_props20, dump_logits, dump_pick, dump_feat = [], [], [], [], []
    t0 = time.time()
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
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            feat_box.clear()
            traj, _score, k = planner.infer(ego, img)
            if len(feat_box) != 1:
                raise RuntimeError(f"token {tok}: the score_head hook fired {len(feat_box)} times in one infer (want exactly 1: a second forward would have run)")
            feat = feat_box[0][0][0]                                        # [64, dec_width] fp16
            if not np.array_equal(feat_box[0][1][0], _score.float().cpu().numpy()):
                raise RuntimeError(f"token {tok}: the hooked score_head output is not the logits infer() returned (the hook is on the wrong call)")
            ev, gl = planner.last_inputs
            paths = planner.frames.resolve(planner._log_hint, int(ego.time_point.time_us)) or []
            fp = hashlib.sha256("|".join([log, tok, ev.detach().cpu().numpy().astype(np.float64).tobytes().hex(),
                                          gl.detach().cpu().numpy().astype(np.float64).tobytes().hex(), *[os.path.basename(p.split("::")[-1]) for p in paths]]).encode()).hexdigest()[:16]
            exe20 = [planner.executed(traj, j).float().cpu().numpy() for j in range(traj.shape[0])]
            poses = to_navsim(planner.executed(traj, k).float().cpu().numpy())
            props8 = np.stack([to_navsim(e) for e in exe20]).astype(np.float32)
            if not (np.isfinite(poses).all() and np.isfinite(props8).all() and np.isfinite(_score.float().cpu().numpy()).all()):
                misses.append((tok, "non-finite output"))
                continue
            assert np.array_equal(props8[k].astype(np.float32), poses.astype(np.float32)), "the picked proposal differs from the seam row (conversion path)"
            rows_tok.append(tok)
            rows_fp.append(fp)
            rows_pose.append(poses.astype(np.float32))
            dump_props.append(props8)
            dump_props20.append(np.stack(exe20).astype(np.float32))
            dump_logits.append(_score.float().cpu().numpy().astype(np.float32))
            dump_pick.append(int(k))
            dump_feat.append(feat)
        for tok in set(lt) - got:
            misses.append((tok, "scenario builder skipped the token (margin guard)"))
        print(f"    [{li + 1}/{len(by_log)}] {log} rows {len(rows_tok):,} misses {len(misses)} {time.time() - t0:.0f} s", flush=True)
    hook.remove()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, token=np.array(rows_tok), fingerprint=np.array(rows_fp), proposals=np.stack(dump_props) if dump_props else np.zeros((0, 64, NAVSIM_N, 3), np.float32),
             proposals20=np.stack(dump_props20) if dump_props20 else np.zeros((0, 64, 20, 3), np.float32),
             logits=np.stack(dump_logits) if dump_logits else np.zeros((0, 64, 6), np.float32), pick=np.array(dump_pick, dtype=np.int64),
             log=np.array([plan["token_log"][t] for t in rows_tok]), select=np.array(a.select), rule=np.array(planner.rule),
             repair_last_heading=np.array(planner.repair_last_heading), ckpt=np.array(str(a.ckpt)), sampling=np.array([NAVSIM_N, NAVSIM_DT]))
    if a.feat_out and dump_feat:
        os.makedirs(os.path.dirname(os.path.abspath(a.feat_out)), exist_ok=True)
        np.savez(a.feat_out, token=np.array(rows_tok), log=np.array([plan["token_log"][t] for t in rows_tok]), prehead_features=np.stack(dump_feat),
                 what=np.array("input of model.score_head, [N, 64 proposals, dec_width], fp16, captured by a forward hook in the same forward as the logits"))
    rep = {"tokens_asked": len(toks), "rows": len(rows_tok), "misses": len(misses), "miss_examples": misses[:10], "ckpt": a.ckpt, "rule": planner.rule,
           "repair_last_heading": planner.repair_last_heading, "goal_fix": planner.goal_fix, "nav_mode": nav_mode, "sanitize_goal": planner.sanitize_goal,
           "prehead_feature_shape": list(np.stack(dump_feat).shape) if dump_feat else None, "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    ok = len(rows_tok) == len(toks)
    print(f"ZZP2HODUMP_{'OK' if ok else 'FAIL'} {len(rows_tok)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
