"""Score T2-A rollouts into per-arm JSONs stamped TIER T2, plus paired A-minus-B files.

    python t2a_score.py arm  <rollouts_*.json> --out <arm.json> [--scene-label X]
    python t2a_score.py pair <A rollouts.json> <B rollouts.json> --out <pair.json>
    python t2a_score.py table <out-dir>        # flatten every T2A_*.json under a directory into SUMMARY.md / .json

The metric math is ``cl_metrics.py``'s (unchanged): ADE, LONGITUDINAL, LATERAL, TACTICAL families with episode-cluster
bootstrap CIs; the STRATEGIC family is reported PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE item 25), never as a pass
and never omitted. Collision and off-road are written as UNAVAILABLE WITH THEIR REASON; progress is real.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TIER = "T2"
TIER_TEXT = ("T2 = ACTION-CLOSED, PERCEPTION-CLOSED loop: the arm's own trajectory drives a kinematic bicycle and the next "
             "observation is RE-RENDERED from the resulting pose (EVAL_DOCTRINE D-LOOP-1). In-house NuRec/gsplat harness "
             "(stack/experiments/alpasim-gsplat/closedloop_drive.py), NOT AlpaSim.")
SCOPE = ("The loop is closed on PERCEPTION, not on INTERACTION: nothing in the scene responds to the ego (actors are replayed or "
         "absent), there is no map and no drivable-area polygon, and the vehicle model has no collision response. This "
         "measures HOW WELL THE MODEL DRIVES ITSELF; it cannot measure WHAT HAPPENS WHEN IT DRIVES BADLY, and it cannot say "
         "'safe'. n = 2 scenes; clusters are disjoint rollout starts of ONE clip each, not independent episodes. "
         "WITHIN-SIM RELATIVE: absolute levels do not transfer to real footage (REF-C open-loop ADE 1.5157 on these "
         "reconstructions vs 0.4728 on real footage, 3.21x OOD) -- orderings and paired differences may.")
PAUSED = ("PAUSED -- PI ruling 2026-09-19 (PI_DECISION_QUEUE item 25): the strategic layer is switched off until the tactical "
          "and operative layers drive with high quality. Neither a pass nor omitted.")
COLLISION = {"status": "UNAVAILABLE",
             "reason": "the bicycle has no collision response and the `empty` condition has no actor to hit; the only collision "
                       "metric in the harness is a geometric in-lane headway<=0 test against a SCRIPTED lead "
                       "(synth_lead_collision_rate_inlane) which exists only for the constructed lead conditions"}
OFF_ROAD = {"status": "UNAVAILABLE",
            "reason": "no map in the harness: `grep -iE 'xodr|off_road|drivable'` over closedloop_drive.py, cl_metrics.py, "
                      "actor_map.py returns 0 hits. `route_corridor_departure_rate` is |cross-track to the LOGGED path| > 2 m "
                      "-- a corridor around the log, NOT a lane or a road edge. map.xodr ships in the scene but is not wired in"}


def _sys_path():
    """The SAME search path t2a_run.py builds: cl_metrics imports refb_labels (stack/scripts) and more, so scoring from a clean
    interpreter fails with ModuleNotFoundError unless the whole tree is on the path (found by running run_on_box2.py end to end)."""
    tree = Path(os.environ.get("T2A_TREE") or (HERE.parent / "tree"))
    for p in (tree / "taniteval" / "tools", tree / "taniteval", tree / "stack" / "scripts", tree / "stack",
              tree / "stack" / "experiments" / "nurec-gsplat", tree / "stack" / "experiments" / "alpasim-gsplat", HERE):
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def score_arm(path: str, label: str | None = None) -> dict:
    _sys_path()
    import numpy as np
    from cl_metrics import collect, families
    d, rows, eids, summ = collect(path)
    fam = families(rows, eids, summ)
    strategic = fam.pop("STRATEGIC", None)
    prog = [s["progress_rel"] for s in summ if s.get("progress_rel") is not None]
    out = {
        "tier": TIER, "tier_text": TIER_TEXT, "scope": SCOPE,
        "arm": d["arm"], "scene": d["scene"], "scene_label": label, "condition": d["condition"], "ckpt": d["ckpt"],
        "steps_per_rollout": d["steps"], "n_rollouts": len(d["rollouts"]), "n_windows": len(rows),
        "n_clusters": int(len(set(eids.tolist()))),
        "provenance": {"model_provenance": d.get("model_provenance"), "canon": d.get("canon"), "window": d.get("window"),
                       "f_eff": d.get("f_eff"), "layers": d.get("layers"),
                       "render_quality": d["rollouts"][0].get("render_quality") if d["rollouts"] else None,
                       "transport": d["rollouts"][0].get("transport") if d["rollouts"] else None},
        "families": fam,
        "STRATEGIC": {"status": PAUSED, "harness_values_for_reference_only_NOT_A_RESULT": strategic},
        "progress": {"per_rollout": [{k: s.get(k) for k in ("start", "n", "driven_m", "gt_dist_m", "progress_rel",
                                                            "max_cross_track", "corridor_departure_rate")} for s in summ],
                     "mean_progress_rel": float(np.mean(prog)) if prog else None,
                     "note": "driven distance / logged distance over the matched stretch; 1.0 = the log's pace"},
        "collision": COLLISION, "off_road": OFF_ROAD,
        "estimator": "paired/unpaired episode-cluster bootstrap over ROLLOUT STARTS (cl_metrics._ci). Answers 'would another draw "
                     "of rollout starts say this?' ONLY: it is blind to training-seed and (for stochastic planners) "
                     "inference-seed variance -- see H-ESTIM-SEED-1.",
    }
    return out


def score_pair(a: str, b: str) -> dict:
    _sys_path()
    tmp = Path(a).with_suffix(".pair_tmp.json")
    cmd = [sys.executable, str(next(p for p in (HERE.parent / "tree" / "stack" / "experiments" / "alpasim-gsplat" / "cl_metrics.py",
                                                Path(os.environ.get("T2A_TREE", "x")) / "stack" / "experiments" / "alpasim-gsplat" / "cl_metrics.py")
                                    if p.exists())),
           "--a", a, "--b", b, "--out", str(tmp)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"cl_metrics pair failed rc={r.returncode}\n{r.stderr[-1500:]}")
    res = json.loads(tmp.read_text())
    tmp.unlink()
    res["tier"], res["tier_text"], res["scope"] = TIER, TIER_TEXT, SCOPE
    res["STRATEGIC"] = {"status": PAUSED}
    res["sign"] = "paired_A_minus_B: negative = A has the LOWER error (better) for error metrics"
    return res


def flatten(out_dir: Path):
    rows = []
    for f in sorted(out_dir.rglob("T2A_*.json")):
        d = json.loads(f.read_text())
        if not isinstance(d, dict) or d.get("tier") != TIER or "families" not in d:   # T2A_SUMMARY.json (a list) matches the glob
            continue
        fam = d["families"]
        g = lambda k1, k2: (fam.get(k1, {}).get(k2) or {})
        ci = lambda e: (e.get("mean"), e.get("lo"), e.get("hi")) if isinstance(e, dict) else (None, None, None)
        rows.append({"file": f.name, "scene": d.get("scene_label") or d.get("scene"), "arm": d["arm"],
                     "vmax": (d["provenance"].get("model_provenance") or {}).get("vmax_mode"),
                     "n_windows": d["n_windows"], "n_clusters": d["n_clusters"],
                     "ade_0_2s": ci(g("ADE", "ade_0_2s")), "dist_to_gt_traj_m": ci(g("ADE", "dist_to_gt_traj_m")),
                     "cross_track_abs_m": ci(g("LATERAL", "cross_track_abs_m")),
                     "abs_target_speed_err_ms": ci(g("LONGITUDINAL", "abs_target_speed_err_ms")),
                     "mean_progress_rel": d["progress"]["mean_progress_rel"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pa = sub.add_parser("arm")
    pa.add_argument("rollouts")
    pa.add_argument("--out", required=True)
    pa.add_argument("--scene-label", default=None)
    pp = sub.add_parser("pair")
    pp.add_argument("a")
    pp.add_argument("b")
    pp.add_argument("--out", required=True)
    pt = sub.add_parser("table")
    pt.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "arm":
        Path(a.out).write_text(json.dumps(score_arm(a.rollouts, a.scene_label), indent=1, default=str))
    elif a.cmd == "pair":
        Path(a.out).write_text(json.dumps(score_pair(a.a, a.b), indent=1, default=str))
    else:
        rows = flatten(Path(a.dir))
        Path(a.dir, "T2A_SUMMARY.json").write_text(json.dumps(rows, indent=1))
        f = lambda t: "n/a" if t[0] is None else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
        lines = ["| scene | arm | ceiling | n_win | ade_0_2s [95% CI] | dist_to_log_m | cross_track_m | |speed err| m/s | progress |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {str(r['scene'])[:8]} | {r['arm']} | {r['vmax'] or '-'} | {r['n_windows']} | {f(r['ade_0_2s'])} | "
                         f"{f(r['dist_to_gt_traj_m'])} | {f(r['cross_track_abs_m'])} | {f(r['abs_target_speed_err_ms'])} | "
                         f"{r['mean_progress_rel'] if r['mean_progress_rel'] is None else round(r['mean_progress_rel'], 3)} |")
        Path(a.dir, "T2A_SUMMARY.md").write_text("TIER T2 (perception-closed loop, in-house harness). " + SCOPE + "\n\n" + "\n".join(lines) + "\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
