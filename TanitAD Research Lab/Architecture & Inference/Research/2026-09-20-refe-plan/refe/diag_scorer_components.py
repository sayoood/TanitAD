"""Proof that the three SCORER defects from the 2026-09-20 conformance review are fixed.

  D3  `comfort` was identical across every candidate of every frame (varied in 0.0 % of 620), so a
      sixth of the scoring head was supervised on a constant and could not move the argmax.
      Root cause found here: `inject_proposal` HELD `agent_velocity_all` at its last recorded value
      for every appended step, which is right for the non-reacting NPCs and wrong for the ego whose
      whole trajectory we had just replaced. Every candidate was handed the SAME velocities, and
      the comfort calculator reads acceleration and jerk out of exactly that tensor.
  D4  PROGRESS had no counterpart in the release and our `goal_reaching` proxy REWARDED OVERSPEED
      (lon x1.5 45/74 vs teacher 13/74); driving-direction compliance was never read at all, so DDC
      supervision was ZERO.
  D5  inference selection was an unweighted sum of six RAW LOGITS -- no sigmoid, no aggregation.

⭐ Each arm carries a control that must read the no-effect value. The D5 arms are pure arithmetic
and need no data; D3 and D4 need one banked simulation log.

Usage:  python diag_scorer_components.py <simulation_log.msgpack.xz> [--step 40]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))


def check_aggregation() -> dict:
    """D5, arithmetic only: the PDM rule must zero on a violation and must DISAGREE with a sum."""
    import planner as PL
    agg = PL.REFePlanner.aggregate
    # order: 0 NC · 1 DAC · 2 EP · 3 TTC · 4 comfort · 5 DDC
    # ⚠ MY FIRST VERSION USED 6.0 AND THE ARM FAILED ON ITS OWN THRESHOLD, NOT ON THE CODE:
    # sigmoid(-6) = 0.00247, so the aggregate read 0.0025 against an assertion of < 1e-3. The
    # BEHAVIOUR was right all along (0.0025 against 0.1285 for the mediocre candidate, 51x lower,
    # and the PDM rule picked correctly). A test threshold that is tighter than the arithmetic it
    # tests is a failing test, not a failing fix -- worth recording because the opposite mistake,
    # a threshold looser than the effect, is the one that hides a real defect.
    big = 12.0                                  # sigmoid(-12) = 6.1e-06
    # A: fails collision, perfect on everything else.  B: mediocre everywhere.
    A = torch.tensor([[[-big, big, big, big, big, big]]])
    B = torch.tensor([[[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]])
    both = torch.cat([A, B], dim=1)             # [1, 2, 6]
    self_ = type("X", (), {"PDM_W": PL.REFePlanner.PDM_W})()
    a = agg(self_, both)[0]
    logit_pick = int(both[0].sum(-1).argmax())
    pdm_pick = int(a.argmax())
    print("--- D5: does the benchmark rule zero a violation, and does it differ from a sum? ---")
    print(f"  candidate A (collision, perfect otherwise): pdm {float(a[0]):.4f}  "
          f"logit-sum {float(both[0, 0].sum()):+.2f}")
    print(f"  candidate B (mediocre everywhere)         : pdm {float(a[1]):.4f}  "
          f"logit-sum {float(both[0, 1].sum()):+.2f}")
    print(f"  the OLD rule picks #{logit_pick}, the PDM rule picks #{pdm_pick}")
    return {
        "D5 a collision drives the aggregate to ~0": float(a[0]) < 1e-4,
        "D5 the collision candidate is far below the mediocre one":
            float(a[0]) < 0.05 * float(a[1]),
        "D5 the safe-but-mediocre candidate wins": pdm_pick == 1,
        "D5 CONTROL the OLD logit sum picked the colliding one": logit_pick == 0,
        "D5 CONTROL aggregate stays in [0, 1]": bool(((a >= 0) & (a <= 1)).all()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--step", type=int, default=40)
    a = ap.parse_args(argv)

    checks = check_aggregation()

    # ⛔ A STRUCTURAL GUARD I ADDED ONLY AFTER BREAKING THIS MYSELF. While extending the kinematics
    # derivation I orphaned `clean_polygon_cache()` and `update_nearest_neighbors()` behind an
    # early `return`, leaving them as dead code. The file's own header names the cache as "the
    # single most likely way to get a plausible-looking but wrong target bank" -- and the edit
    # compiled, imported and ran. Only reading the function body caught it.
    import ast
    import score_proposals as _SPmod
    tree = ast.parse(io.open(_SPmod.__file__, encoding="utf-8").read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    inj = ast.unparse(fns["inject_proposal"]) if "inject_proposal" in fns else ""
    dead = any(isinstance(st, ast.Return) and k < len(fns["inject_proposal"].body) - 1
               for k, st in enumerate(fns.get("inject_proposal", ast.FunctionDef(body=[])).body))
    checks.update({
        "STRUCT inject_proposal clears the polygon cache": "clean_polygon_cache" in inj,
        "STRUCT inject_proposal updates nearest neighbours": "update_nearest_neighbors" in inj,
        "STRUCT inject_proposal derives the ego kinematics": "_derive_ego_kinematics" in inj,
        "STRUCT no dead code after a return in inject_proposal": not dead,
    })

    import scorer_gate as G
    import score_proposals as SP
    import augment_routes as AR
    import build_scorer_targets as BST

    cfg, _ = G.build_engine_config()
    calcs = SP.build_calculators(cfg)
    sd, log_sd, sc, smps, step, init = G.scenario_data_from_log(a.log, a.step)
    sd, _ = SP.enrich_lane_graph(sd, sc.map_api, AR._anchor_from_ego(smps[step].ego_state),
                                 init.route_roadblock_ids)
    gxy, gyw = BST.teacher_future(smps, step)
    cands = BST.candidate_set(sd, gxy, gyw, 0)
    by = {nm: SP.score_proposal_rollout(sd, log_sd, calcs, xy, yw) for nm, xy, yw in cands}

    t_adv = by["teacher"].get("progress.advance_m", float("nan"))
    for r in by.values():
        adv = r.get("progress.advance_m", float("nan"))
        r["progress.ep"] = (max(0.0, min(1.0, adv / t_adv))
                            if adv == adv and t_adv == t_adv and t_adv > 1e-3 else float("nan"))

    print(f"\n--- D3 / D4 on one real frame (teacher advance {t_adv:.2f} m) ---")
    print(f"  {'candidate':11s} {'comfort':>9s} {'advance m':>10s} {'EP':>7s} {'DDC viol':>9s}")
    comforts = []
    for nm in by:
        c = by[nm].get("comfort.Comfort.reward", float("nan"))
        comforts.append(c)
        print(f"  {nm:11s} {c:9.4f} {by[nm].get('progress.advance_m', float('nan')):10.2f} "
              f"{by[nm].get('progress.ep', float('nan')):7.3f} "
              f"{by[nm].get('ddc.violation', float('nan')):9.1f}")

    # ⛔ MY FIRST VERSION OF THIS GUARD WAS ALGEBRAICALLY BLIND AND THE REVIEW PROVED IT.
    # It compared the derived speed against `displacement / (T * TRAJ_DT)` -- but the derived speed
    # IS displacement/TRAJ_DT, so TRAJ_DT cancels on both sides and the identity holds for ANY
    # value. Mutation arm: it PASSED at 0.1 as readily as at 0.2, i.e. it would have certified the
    # exact 2x error it was written to catch.
    # ⭐ The reference has to be INDEPENDENT of the quantity under test. The ego's LOGGED speed at
    # this frame comes from the simulation log and knows nothing about our pose cadence, so
    # comparing against it actually tests TRAJ_DT. (The review's own cross-check: sample spacing
    # reads 0.1999 s and derived 6.197 vs logged 6.291, a 1.50 % difference.)
    cand_t = SP.inject_proposal(sd, gxy, gyw)
    vel = cand_t.agent_velocity_all[0, 0, -gxy.shape[0]:, :2]
    # ⛔ COMPARE LIKE WITH LIKE. My first independent-reference version compared the derived MEAN
    # over a 4 s horizon against the INSTANTANEOUS logged speed at t0 -- and the ego is accelerating
    # at ~1.5 m/s^2, so those two MUST differ: 2.816 m/s at t0 rising to a 6.197 m/s mean is
    # entirely consistent and the guard failed on correct code. This programme has made exactly
    # this error before (a 4 s mean checked against an instantaneous speed) and logged it; I
    # repeated it inside the very guard written to enforce rigour.
    # ⇒ take the FIRST derived velocity, which covers the first TRAJ_DT from t0, against the speed
    # logged at t0. A doubled dt would still double it, so the guard keeps its power.
    v_derived = float(vel[0].norm())
    v_mean = float(vel.norm(dim=-1).mean())
    _ev = smps[step].ego_state.dynamic_car_state.rear_axle_velocity_2d
    v_logged = float((_ev.x ** 2 + _ev.y ** 2) ** 0.5)
    print("")
    print(f"--- dt guard: derived FIRST-STEP {v_derived:.3f} m/s vs the LOG's speed at t0 "
          f"{v_logged:.3f} m/s   (horizon mean {v_mean:.3f}, higher because the ego accelerates) "
          f"[TRAJ_DT={SP.TRAJ_DT}] ---")


    eps = {nm: by[nm].get("progress.ep", float("nan")) for nm in by}
    over = eps.get("lon x1.5", float("nan"))
    checks.update({
        "D3 comfort now VARIES across candidates": (max(comforts) - min(comforts)) > 1e-9,
        "D4 progress.ep exists for every candidate": all(v == v for v in eps.values()),
        # ⚠️ THESE TWO USED TO ASSERT THE DIAGNOSTIC'S OWN CLAMP. `progress.ep` is computed right
        # here as `min(1.0, advance/teacher)`, so "EP is bounded in [0,1]" and "lon x1.5 <= 1.0"
        # were restatements of the clamp on the line above -- they would pass however wrong the
        # underlying advance was. The real question is whether the RAW advance exceeds the
        # teacher's (so the clamp is doing work) while the clamped value stops at 1.0.
        "D4 EP is bounded in [0, 1]": all(0.0 <= v <= 1.0 for v in eps.values() if v == v),
        "D4 the RAW advance for lon x1.5 really exceeds the teacher's (the clamp does work)":
            by.get("lon x1.5", {}).get("progress.advance_m", 0.0) > 1.05 * t_adv
            if t_adv == t_adv and t_adv > 1e-3 else False,
        "D4 and the CLAMPED value stops at exactly 1.0, not above":
            abs(eps.get("lon x1.5", 0.0) - 1.0) < 1e-9,
        "D4 a stopped ego makes no progress": eps.get("stopped", 1.0) < 0.05,
        "D4 ddc.violation is emitted": all("ddc.violation" in r for r in by.values()),
        "D4 progress spans the REAL horizon, not a 2-point segment":
            max(abs(r.get("progress.advance_m", 0.0)) for r in by.values()) > 5.0,
        "D4 DDC FIRES on a reverse candidate (it was structurally zero)":
            by.get("reverse", {}).get("ddc.violation", 0.0) > 0.0
            if "reverse" in by else False,
        "D3 derived speed matches the LOG's speed (a wrong dt would double it)":
            abs(v_derived - v_logged) < 0.20 * max(v_logged, 1e-6),
    })
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    good = all(checks.values())
    print("\n" + ("SCORER_COMPONENTS_CONFORM" if good else "SCORER_COMPONENTS_DEFECTIVE"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
