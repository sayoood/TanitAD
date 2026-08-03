#!/usr/bin/env python3
"""Run the STRATEGIC family's MANDATORY negative control on the REAL banked option sets.

⭐ NEGATIVE CONTROL FIRST — prove the metric can DISCRIMINATE before quoting it.

This is the gate that ``route_head_eq_logged`` never had. It scores synthetic
arms (oracle / three constant predictors / uniform-random / no-head) against the
**real** map-derived option sets in ``results/strategic_gt/`` and reports whether
the oracle is paired-CI-separated above the BEST CONSTANT. If it is not, the
scene set cannot carry a strategic verdict and no accuracy may be quoted from it.

It also drives the closed-loop join end-to-end on the winner scene with two
synthetic per-tick policies (commits early / commits late), so
``decision_lead_distance_m`` — the one strategic metric a flat policy cannot fake
— is exercised on real labels rather than on a fixture.

    python3 strategic_family_control.py --labels results/strategic_gt \
        --out results/strategic_family_control.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _taniteval():
    here = Path(__file__).resolve()
    for up in here.parents:
        if (up / "taniteval" / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(up / "taniteval"))
            break
    from taniteval import strategic_optionset as SO
    return SO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(Path(__file__).parent / "results" / "strategic_gt"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    SO = _taniteval()
    reports = SO.load_label_reports(a.labels)
    scenes = [s for s in reports if not s.startswith("_")]
    events, ev_scenes = SO.scoreable_events(reports)

    control = SO.discrimination_control(reports, n_boot=a.n_boot)

    # every synthetic arm scored through the FULL family, not just the control
    arms = {}
    oracle = {e["event_id"]: {"class": e["route_gt_class"],
                              "road": e["connecting_road_taken"]} for e in events}
    arms["ORACLE"] = SO.strategic_family(reports, oracle, arm="ORACLE", n_boot=a.n_boot)
    for c, name in enumerate(SO.ROUTE_CLASSES[:3]):
        arms[f"CONSTANT_{name}"] = SO.strategic_family(
            reports, {e["event_id"]: {"class": c, "road": None} for e in events},
            arm=f"CONSTANT_{name}", n_boot=a.n_boot)
    arms["NO_HEAD"] = SO.strategic_family(
        reports, {}, arm="NO_HEAD", n_boot=a.n_boot)

    # ---- the night clip, scored on its own: the state that produced the 1.0000
    night = next((s for s in scenes if s.startswith("00040136")), None)
    night_block = None
    if night:
        one = {night: reports[night], "_refused": {}}
        night_block = SO.strategic_family(one, oracle, arm="ORACLE_on_the_night_clip",
                                          n_boot=a.n_boot)

    # ---- closed-loop join, end to end, on the winner scene's real labels
    lead = None
    win = next((s for s in scenes if s.startswith("7c72937c")), None)
    if win:
        rep = reports[win]
        one = {win: rep, "_refused": {}}
        ev1, _ = SO.scoreable_events(one)
        gt = {e["event_id"]: e for e in ev1}
        n_poses = int(rep["n_poses"])

        def ticks(commit_at_m):
            """A per-tick policy that becomes correct `commit_at_m` before the junction."""
            out = []
            for p in rep["per_pose"]:
                if not p.get("admissible"):
                    continue
                e = gt.get(p["event_id"])
                if e is None:
                    continue
                d = p["dist_to_decision_point_m"]
                right = e["route_gt_class"]
                wrong = next((o["class"] for o in e["options"]
                              if o["class"] != right and o["class"] in (0, 1, 2)), 1)
                out.append({"i_gt": p["pose"],
                            "route_head": right if d <= commit_at_m else wrong})
            return out

        lead = {}
        for label, m in (("commits_60m_out", 60.0), ("commits_20m_out", 20.0),
                         ("commits_5m_out", 5.0), ("never_correct", -1.0)):
            preds = SO.event_predictions_from_ticks(ticks(m), rep)
            fam = SO.strategic_family(one, preds, arm=label, n_boot=a.n_boot)
            lead[label] = {
                "route_class_accuracy": fam["route_class_accuracy"],
                "decision_lead_distance_m": fam["decision_lead_distance_m"],
                "n_ticks_joined": preds["_join"]["n_ticks_out_of_range"],
            }
        lead["_n_poses"] = n_poses
        lead["_scene"] = win

    rep_out = {
        "tool": "strategic_family_control.py",
        "evidence_class": "MEASURED",
        "labels_dir": str(a.labels),
        "n_scenes_supplied": len(scenes) + len(reports.get("_refused", {})),
        "n_scenes_admissible": len(scenes),
        "n_scenes_refused_by_selfconsistency": len(reports.get("_refused", {})),
        "refused": reports.get("_refused", {}),
        "n_scoreable_events": len(events),
        "n_scenes_contributing_an_event": len(set(ev_scenes)),
        "NEGATIVE_CONTROL": control,
        "arms_through_the_full_family": {
            k: {kk: v[kk] for kk in (
                "status", "route_class_accuracy", "route_class_per_class",
                "route_class_confusion_gt_x_pred", "floors", "vs_best_constant",
                "beats_best_constant", "label_degenerate",
                "n_events_without_a_prediction", "n_events_gt_outside_head_vocabulary",
                "n_events_class_to_road_ambiguous", "reason", "n")
                if kk in v}
            for k, v in arms.items()},
        "NIGHT_CLIP_ALONE": night_block,
        "DECISION_LEAD_DISTANCE_on_the_winner": lead,
    }
    txt = json.dumps(rep_out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    print(json.dumps({k: v for k, v in rep_out.items()
                      if k not in ("arms_through_the_full_family",)},
                     indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
