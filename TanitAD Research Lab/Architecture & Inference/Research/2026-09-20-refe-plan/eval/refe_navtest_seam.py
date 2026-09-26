"""REFe on NAVSIM v1.1 navtest: `planner.REFePlanner`'s own inference -> a W3 seam npz (SPEC_NAVTEST).

Per navtest token (grouped by log): the nuPlan TEST-DB scenario at the token
(`navtrain_scenarios.build_scenarios_for_log`, the training code path) -> `REFePlanner.initialize`
(the route) -> `_image_for` + `infer` (the planner's own input construction AND selection -- one
implementation shared with the nuPlan closed-loop planner) -> the selected proposal's 20 poses at
0.2 s (ego frame, rear axle) -> linear interpolation to NAVSIM's 8 poses at 0.5 s -> seam row
{token, fingerprint (W3's export), poses}.

⭐ FRAME/TIME CONTROL, same breath: the LOGGED future from the SAME scenario is put through the SAME
0.5 s grid and compared with W3's exported `human_future_poses` (the devkit's own
`Scene.get_future_trajectory`). They are two routes to one quantity, so they must agree to ~cm; if
they do not, the frame convention or the token -> scenario pairing is wrong and NO REFe number from
this seam is admissible -- the control, not the model, decides that.

  python eval/refe_navtest_seam.py --ckpt <snap_epochNNN.pt | model_final.pt> \
      --frames D:/Projects/TanitAD/data/refe_navtest/frames --out <seam.npz> [--tokens <subset.json>]
  python eval/refe_navtest_seam.py --selftest        # the converter's analytic bars (SPEC E-1)
Prints ZZSEAM_OK <rows> <frame_control_max_m> only when every requested token has a finite row AND
the frame control passes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")
sys.path.insert(0, REFE)

W3_EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_DB_DIR = "D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
SRC_DT = 0.2                                    # REFe's pose grid (planner.TRAJ_DT_S)
NAVSIM_DT, NAVSIM_N = 0.5, 8                    # NAVSIM v1 TrajectorySampling
# ⭐ TWO DIFFERENT QUESTIONS, MEASURED APART (2026-09-24, E-2 first run):
#  * FRAME / TIME / PAIRING: the log's future sampled DIRECTLY on NAVSIM's 0.5 s grid (DB rows 10k,
#    exactly what OpenScene keeps) vs W3's `human_future_poses` -- MEASURED 0.0000 m on 25/25, so the
#    bar is 1 mm and any miss is a real frame or pairing defect.
#  * INTERPOLATION RESIDUAL: the same future on REFe's 0.2 s grid (rows 4i) put through `to_navsim`
#    -- up to 0.123 m MEASURED (3/25 tokens, the odd half-seconds), from uneven lidar-frame timing
#    between rows 8 and 12. It is inherent to converting REFe's output grid, so it is REPORTED with
#    its max / median, never gated. (The first version gated the residual at 5 cm and read a correct
#    frame as a failed one.)
FRAME_CONTROL_MAX_M = 0.001


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def to_navsim(poses20: np.ndarray, dt: float = SRC_DT) -> np.ndarray:
    """[20, 3] (x, y, heading) at t = dt, 2dt, ... -> [8, 3] at t = 0.5, ..., 4.0 by linear
    interpolation of x, y and of the WRAPPED heading difference (never of the raw angle)."""
    p = np.asarray(poses20, dtype=np.float64)
    t_src = dt * np.arange(1, p.shape[0] + 1)
    out = np.zeros((NAVSIM_N, 3), dtype=np.float64)
    for k in range(NAVSIM_N):
        t = NAVSIM_DT * (k + 1)
        j = int(np.searchsorted(t_src, t - 1e-9))
        if j < p.shape[0] and abs(t_src[j] - t) < 1e-9:
            out[k] = p[j]
            continue
        if j == 0 or j >= p.shape[0]:
            raise ValueError(f"t={t} outside the source grid ({t_src[0]}..{t_src[-1]})")
        w = (t - t_src[j - 1]) / (t_src[j] - t_src[j - 1])
        out[k, :2] = (1 - w) * p[j - 1, :2] + w * p[j, :2]
        out[k, 2] = wrap(p[j - 1, 2] + w * wrap(p[j, 2] - p[j - 1, 2]))
    return out


def selftest() -> int:
    """SPEC E-1 (with AMENDMENT 1): straight line exact; arc within its chord sagitta."""
    ok = True
    t = SRC_DT * np.arange(1, 21)
    v = 12.0
    line = np.stack([v * t, np.zeros_like(t), np.zeros_like(t)], -1)
    got = to_navsim(line)
    want = np.stack([v * NAVSIM_DT * np.arange(1, 9), np.zeros(8), np.zeros(8)], -1)
    e_line = float(np.abs(got - want).max())
    print(f"  [{'PASS' if e_line < 1e-6 else 'FAIL'}] straight line 12 m/s: max |err| {e_line:.2e} m (< 1e-6)")
    ok &= e_line < 1e-6
    for R, vv in ((20.0, 10.0), (60.0, 15.0), (8.0, 4.0)):
        w_ = vv / R
        arc = np.stack([R * np.sin(w_ * t), R * (1 - np.cos(w_ * t)), wrap(w_ * t)], -1)
        tq = NAVSIM_DT * np.arange(1, 9)
        want = np.stack([R * np.sin(w_ * tq), R * (1 - np.cos(w_ * tq)), wrap(w_ * tq)], -1)
        got = to_navsim(arc)
        e_xy = float(np.linalg.norm(got[:, :2] - want[:, :2], axis=1).max())
        e_h = float(np.abs(wrap(got[:, 2] - want[:, 2])).max())
        bound = R * (1 - math.cos(w_ * SRC_DT / 2)) + 1e-9
        good = e_xy <= bound and e_h < 1e-9
        print(f"  [{'PASS' if good else 'FAIL'}] arc R={R} v={vv}: max |xy err| {e_xy:.3e} m <= sagitta "
              f"{bound:.3e}; heading err {e_h:.1e} rad")
        ok &= good
    # heading wrap across +-pi must not unwind through 0
    wrapped = np.stack([np.zeros(20), np.zeros(20), wrap(math.pi - 0.05 + 0.01 * np.arange(1, 21))], -1)
    got = to_navsim(wrapped)
    e_w = float(np.abs(wrap(got[:, 2] - wrap(math.pi - 0.05 + 0.01 * (NAVSIM_DT * np.arange(1, 9) / SRC_DT)))).max())
    print(f"  [{'PASS' if e_w < 1e-9 else 'FAIL'}] heading across +-pi: max err {e_w:.1e} rad")
    ok &= e_w < 1e-9
    print("ZZCONVERTER_" + ("EXACTZZ" if ok else "FAILZZ"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--frames", default="D:/Projects/TanitAD/data/refe_navtest/frames")
    ap.add_argument("--db-dir", default=TEST_DB_DIR)
    ap.add_argument("--export", default=W3_EXPORT)
    ap.add_argument("--tokens", default=None, help="W3-format subset json; default = all navtest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--arm", default="REFe")
    ap.add_argument("--select", default="best")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--dump-proposals", default=None,
                    help="REPORT ONLY (SPEC E-6): also write EVERY proposal on NAVSIM's grid and the "
                         "scorer's raw six logits per token to this .npz, for eval/proposal_table.py. "
                         "The seam itself is unchanged.")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.ckpt and a.out):
        print("  --ckpt and --out are required"); return 2
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    toks = list(exp)
    if a.tokens:
        sub = json.load(open(a.tokens, encoding="utf-8"))
        want = set(sub["tokens"] if isinstance(sub, dict) else sub)
        toks = [t for t in toks if t in want]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    print(f"  tokens: {len(toks):,} over {len(by_log)} logs; ckpt {a.ckpt}", flush=True)

    import navtrain_scenarios as NS
    from planner import REFePlanner
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
    planner = REFePlanner(checkpoint=a.ckpt, images_root=a.frames, db_dir=a.db_dir,
                          backbone=a.backbone, device=a.device, select=a.select)
    print(f"  planner: trained={planner.trained} per_sample_calib={planner.per_sample_calib} "
          f"device={planner.device} select={planner.select}", flush=True)
    rows_tok, rows_fp, rows_pose, misses = [], [], [], []
    ctrl_err, n_ctrl, interp_err = [], 0, []
    prop_diag: list = []
    dump_props, dump_logits, dump_pick = [], [], []
    n_prop = 0
    t0 = time.time()
    for li, (log, lt) in enumerate(sorted(by_log.items())):
        db = os.path.join(a.db_dir, f"{log}.db")
        got = set()
        for sc in NS.build_scenarios_for_log(db, lt, history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            got.add(tok)
            planner._scenario = sc
            planner.initialize(PlannerInitialization(
                route_roadblock_ids=sc.get_route_roadblock_ids(),
                mission_goal=sc.get_mission_goal(), map_api=sc.map_api))
            ego = sc.get_ego_state_at_iteration(0)
            img = planner._image_for(ego)
            if img is None:
                misses.append((tok, f"no frames: {planner.frames.miss_reason}"))
                continue
            traj, _score, k = planner.infer(ego, img)
            poses = to_navsim(traj[k].float().cpu().numpy())
            if not np.isfinite(poses).all():
                misses.append((tok, "non-finite poses"))
                continue
            rows_tok.append(tok)
            rows_fp.append(exp[tok]["fingerprint"])
            rows_pose.append(poses.astype(np.float32))
            # ⭐ PROPOSAL DIAGNOSTICS (report only; the seam above is unchanged). While the scorer
            # bank does not exist the pick `k` is a placeholder argmax, so PDMS alone cannot say
            # whether the PROPOSALS are good. Every one of the M proposals is scored against the
            # human future on the same grid: selected / RANDOM (the mean over proposals -- a random
            # selector's expectation, the control an oracle gap needs) / ORACLE (best of M, a bound
            # no scorer exceeds); plus which proposals are ever best -- the WTA-collapse check the
            # trainer's 4-sample `winners` print cannot make. Open-loop, vs the logged human future.
            hum = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)[:, :2]
            allp = np.stack([to_navsim(traj[j].float().cpu().numpy())[:, :2]
                             for j in range(traj.shape[0])])
            ade = np.linalg.norm(allp - hum[None], axis=-1).mean(-1)          # [M]
            ends = allp[:, -1, :]
            prop_diag.append((float(ade[k]), float(ade.mean()), float(ade.min()),
                              int(ade.argmin()), int(k),
                              float(np.linalg.norm(ends - ends.mean(0), axis=-1).mean())))
            n_prop = int(traj.shape[0])
            if a.dump_proposals:
                # SPEC E-6: all M proposals in the SAME to_navsim conversion the seam row used, and
                # the scorer's raw logits from the SAME forward pass -- never a second model call
                dump_props.append(np.stack([to_navsim(traj[j].float().cpu().numpy())
                                            for j in range(traj.shape[0])]).astype(np.float32))
                dump_logits.append(_score.float().cpu().numpy().astype(np.float32))
                dump_pick.append(int(k))
            # the frame/time control: the log's own future through the SAME grid, in REFe's frame
            px, py, pyaw = ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading
            c, s = math.cos(-pyaw), math.sin(-pyaw)

            def _loc(states):
                return np.array([[(f.rear_axle.x - px) * c - (f.rear_axle.y - py) * s,
                                  (f.rear_axle.x - px) * s + (f.rear_axle.y - py) * c,
                                  wrap(f.rear_axle.heading - pyaw)] for f in states])
            theirs = np.asarray(exp[tok]["human_future_poses"], dtype=np.float64)
            f8 = list(sc.get_ego_future_trajectory(0, 4.0, NAVSIM_N))[:NAVSIM_N]
            if len(f8) == NAVSIM_N:
                ctrl_err.append(float(np.linalg.norm(_loc(f8)[:, :2] - theirs[:, :2], axis=1).max()))
                n_ctrl += 1
            f20 = list(sc.get_ego_future_trajectory(0, 4.0, 20))[:20]
            if len(f20) == 20:
                interp_err.append(float(np.linalg.norm(to_navsim(_loc(f20))[:, :2] - theirs[:, :2],
                                                       axis=1).max()))
        for tok in set(lt) - got:
            misses.append((tok, "scenario builder skipped the token (margin guard)"))
        if (li + 1) % 5 == 0 or li + 1 == len(by_log):
            print(f"    [{li + 1}/{len(by_log)}] rows {len(rows_tok):,}  misses {len(misses)}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, token=np.array(rows_tok), fingerprint=np.array(rows_fp),
             poses=np.stack(rows_pose) if rows_pose else np.zeros((0, NAVSIM_N, 3), np.float32),
             sampling=np.array([NAVSIM_N, NAVSIM_DT]), arm=np.array(a.arm))
    ce = np.array(ctrl_err) if ctrl_err else np.array([np.inf])
    rep = {"tokens_asked": len(toks), "rows": len(rows_tok), "misses": len(misses),
           "miss_examples": misses[:10], "ckpt": a.ckpt, "select": a.select,
           "frame_control": {"n": n_ctrl, "max_m": float(ce.max()), "median_m": float(np.median(ce)),
                             "bar_m": FRAME_CONTROL_MAX_M,
                             "what": "log future on NAVSIM's 0.5 s grid vs W3 human_future_poses"},
           "interpolation_residual": {"n": len(interp_err),
                                      "max_m": float(max(interp_err)) if interp_err else None,
                                      "median_m": float(np.median(interp_err)) if interp_err else None,
                                      "what": "log future on REFe's 0.2 s grid -> to_navsim, vs W3; "
                                              "reported, not gated"},
           "seconds": round(time.time() - t0, 1)}
    if prop_diag:
        from collections import Counter
        pd_ = np.asarray(prop_diag, dtype=np.float64)
        orc = Counter(int(x) for x in pd_[:, 3])
        rep["proposals"] = {
            "n": int(len(pd_)), "M": n_prop,
            "ade_selected_m": round(float(pd_[:, 0].mean()), 4),
            "ade_random_m": round(float(pd_[:, 1].mean()), 4),
            "ade_oracle_m": round(float(pd_[:, 2].mean()), 4),
            "oracle_index_distinct": len(orc),
            "oracle_index_top5": orc.most_common(5),
            "selected_index_distinct": len(set(int(x) for x in pd_[:, 4])),
            "endpoint_spread_m": round(float(pd_[:, 5].mean()), 3),
            "what": "open-loop ADE over NAVSIM's 8 poses (0.5 s) vs W3's human future: selected = "
                    "the planner's pick; random = mean over the M proposals (a random selector's "
                    "expectation); oracle = best of M; oracle_index_* = how many proposals are "
                    "ever best (WTA collapse if few); endpoint_spread = mean distance of the M "
                    "endpoints from their centroid"}
    if a.dump_proposals:
        # rows are appended at the same point as the seam's rows, so they align with rows_tok
        assert len(dump_pick) == len(rows_tok), (len(dump_pick), len(rows_tok))
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_proposals)), exist_ok=True)
        np.savez(a.dump_proposals, token=np.array(rows_tok), fingerprint=np.array(rows_fp),
                 proposals=np.stack(dump_props), logits=np.stack(dump_logits),
                 pick=np.array(dump_pick, dtype=np.int64), select=np.array(a.select),
                 ckpt=np.array(str(a.ckpt)), sampling=np.array([NAVSIM_N, NAVSIM_DT]))
        rep["proposal_dump"] = os.path.abspath(a.dump_proposals)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "miss_examples"}, indent=1))
    ok = (len(rows_tok) == len(toks) and n_ctrl > 0 and float(ce.max()) <= FRAME_CONTROL_MAX_M)
    print(f"ZZSEAM_{'OK' if ok else 'FAIL'} {len(rows_tok)} {float(ce.max()):.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
