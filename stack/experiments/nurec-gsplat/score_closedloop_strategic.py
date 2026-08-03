#!/usr/bin/env python3
"""Score REAL closed-loop rollouts against the map-derived option sets.

This is the wiring, end to end, on genuine data: the closed-loop panel driven on
NuRec scene ``7c72937c`` (the only branch scene we have rendered) -> per-tick
``argmax(s_route_logits)`` -> ``taniteval.strategic_optionset.strategic_family``.

INPUT is the compact extract produced on Thor::

    {"scene": …, "arms": {<run>: {"arm","condition","n_gt_poses","gt_arc_m":[…],
                                  "rollouts":[{"start_frame","ticks":[{"i_gt","route_head"}]}]}}}

⚠️ **THE POSE ALIGNMENT IS NOT ASSUMED, IT IS FITTED AND ITS SENSITIVITY REPORTED.**
The closed-loop ``gt`` track has **199** poses (frames 0-198, arc 125.064 m) while
``strategic_gt`` labelled the **202**-pose clipgt track (arc 127.55 m). Those are
not the same index space, and a silent off-by-one would re-score every decision.
:func:`fit_offset` picks the integer offset that best reproduces the labels' own
``entry_arc_m`` anchors, and :func:`main` re-scores at the neighbouring offsets
too. **If the verdict moves with the offset, the number is not admissible.**
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


def fit_offset(gt_arc, report, span=8):
    """Integer offset ``cl_index = clip_pose + off`` best matching the labels' arcs."""
    anchors = [(e["entry_pose"], e["entry_arc_m"]) for e in report.get("events", [])
               if e.get("entry_arc_m") is not None]
    rows = []
    for off in range(-span, span + 1):
        errs = [abs(gt_arc[p + off] - a) for p, a in anchors
                if 0 <= p + off < len(gt_arc)]
        if len(errs) == len(anchors) and errs:
            rows.append({"offset": off, "max_abs_arc_err_m": round(max(errs), 3),
                         "errs_m": [round(e, 3) for e in errs]})
    if not rows:
        return 0, rows
    best = min(rows, key=lambda r: r["max_abs_arc_err_m"])
    return best["offset"], rows


def score_arm(SO, run, report, scene, offset, n_boot):
    """One arm/condition at one offset. Every rollout is a separate decision instance."""
    per_event, joins = {}, []
    for r in run["rollouts"]:
        preds = SO.event_predictions_from_ticks(r["ticks"], report, pose_offset=offset)
        joins.append({"start_frame": r["start_frame"], **preds["_join"]})
        for eid, p in preds.items():
            if eid == "_join":
                continue
            per_event.setdefault(eid, []).append(p)
    fam = SO.strategic_family({scene: report, "_refused": {}}, per_event,
                              arm=f"{run['arm']}/{run['condition']}", n_boot=n_boot)
    fam["_join"] = {"pose_offset": offset,
                    "n_rollouts": len(run["rollouts"]),
                    "n_ticks_out_of_range_total": sum(j["n_ticks_out_of_range"] for j in joins),
                    "per_rollout": joins}
    return fam


def paired(SO, runA, runB, report, scene, offset, n_boot):
    """Paired A-vs-B on the IDENTICAL (rollout, decision-event) instances.

    ⛔ Keyed on ``(event_id, start_frame)``, not on list position: two arms can
    reach different events from the same start, and a positional zip would then
    compare different decisions and call it a delta.
    """
    def vec(run):
        out = {}
        for r in run["rollouts"]:
            p = SO.event_predictions_from_ticks(r["ticks"], report, pose_offset=offset)
            for eid, v in p.items():
                if eid != "_join":
                    out[(eid, r["start_frame"])] = v.get("class")
        return out
    A, B = vec(runA), vec(runB)
    gt = {e["event_id"]: e["route_gt_class"] for e in report["events"]}
    keys = sorted(set(A) & set(B))
    if not keys:
        return {"n": 0, "reason": "the two arms share no (event, rollout) instance"}
    a = [float(A[k] is not None and A[k] == gt[k[0]]) for k in keys]
    b = [float(B[k] is not None and B[k] == gt[k[0]]) for k in keys]
    out = SO._ci.paired_episode_cluster_bootstrap(a, b, [scene] * len(keys),
                                                  n_boot=n_boot, seed=0)
    SO._guard_single_cluster(out, 1)
    out["n_shared_instances"] = len(keys)
    out["n_instances_A_only"] = len(set(A) - set(B))
    out["n_instances_B_only"] = len(set(B) - set(A))
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", required=True, help="scene2_route_decisions.json from Thor")
    ap.add_argument("--labels", default=str(here / "results" / "strategic_gt"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    SO = _taniteval()
    ex = json.loads(Path(a.extract).read_text(encoding="utf-8"))
    scene = ex["scene"]
    reports = SO.load_label_reports(a.labels)
    if scene not in reports:
        raise SystemExit(f"no option-set label for scene {scene}")
    rep = reports[scene]

    first = next(iter(ex["arms"].values()))
    best_off, scan = fit_offset(first["gt_arc_m"], rep)

    out = {
        "tool": "score_closedloop_strategic.py",
        "evidence_class": "MEASURED",
        "scene": scene,
        "label_n_poses": rep["n_poses"],
        "closedloop_n_gt_poses": first["n_gt_poses"],
        "closedloop_gt_total_arc_m": first["gt_total_arc_m"],
        "ALIGNMENT": {
            "note": ("the closed-loop gt track and the labelled clipgt track are NOT the same "
                     "index space (199 vs 202 poses). The offset is FITTED to the labels' own "
                     "entry_arc_m anchors and the verdict is re-checked at its neighbours."),
            "best_offset": best_off, "scan": scan,
        },
        "arms": {},
        "OFFSET_SENSITIVITY": {},
    }

    for run_name, run in ex["arms"].items():
        out["arms"][run_name] = score_arm(SO, run, rep, scene, best_off, a.n_boot)
    # ⭐ the paired contrast the programme actually cares about: the two closed-loop arms
    out["PAIRED_flagship_vs_refc"] = {}
    for cond in ("empty", "objects"):
        fa = ex["arms"].get(f"flagship-v1_{cond}")
        rb = ex["arms"].get(f"refc-base_{cond}")
        if fa and rb:
            out["PAIRED_flagship_vs_refc"][cond] = paired(
                SO, fa, rb, rep, scene, best_off, a.n_boot)

    for off in sorted({best_off - 1, best_off, best_off + 1}):
        out["OFFSET_SENSITIVITY"][str(off)] = {
            k: (score_arm(SO, v, rep, scene, off, 200)
                .get("route_class_accuracy", {}) or {}).get("mean")
            for k, v in ex["arms"].items()}

    txt = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    for k, v in out["arms"].items():
        if v.get("status") != "OK":
            print(f"{k:28s} {v['status']}: {v.get('reason','')[:150]}")
            continue
        acc = v["route_class_accuracy"]
        print(f"{k:28s} acc={acc['mean']:.4f} n_inst={v['n_decision_instances_scored']} "
              f"n_ev={v['n_events_scoreable']} best_const="
              f"{v['floors']['BEST_CONSTANT']['class']}@{v['floors']['BEST_CONSTANT']['accuracy']}")
    print("offset sensitivity:", json.dumps(out["OFFSET_SENSITIVITY"]))


if __name__ == "__main__":
    main()
