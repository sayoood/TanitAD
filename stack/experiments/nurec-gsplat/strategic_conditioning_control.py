#!/usr/bin/env python3
"""⛔ THE MISSING MANDATORY CONTROL: is a closed-loop STRATEGIC score just the ECHO
of the nav command the harness FED the policy?

Why this file exists
--------------------
``strategic_optionset.discrimination_control`` proves the OPTION-SET LABELS carry
entropy (ORACLE beats the best constant). That is a property of the SCENES. It
says nothing about whether an ARM's score is skill, because the ORACLE arm is
built by copying ``route_gt_class`` — it is a tautology by construction.

The control that decides an ARM's number is different and was never run:

    the harness computes `nav` from the LOGGED GT track
    (closedloop_drive.py:330 `nav_from_route` -> refb_labels.nav_command_v21),
    FEEDS it to the policy, and the policy's route head may simply return it.

MEASURED 2026-08-03 on the banked rollouts of NuRec branch scene 7c72937c
(``stack/experiments/alpasim-gsplat/results/scene2-realclose/rollouts/``):

* ``flagship-v1`` head is an EXACT BIJECTION of nav — nav=1 -> head=0 on 369/369,
  nav=0 -> head=1 on 81/81, both conditions.
* Pushing the **fed nav command alone, with NO MODEL**, through
  ``strategic_optionset.strategic_family`` scores **1.0000** with the IDENTICAL
  confusion ``{LEFT:{LEFT:5}, STRAIGHT:{STRAIGHT:1}}`` as flagship-v1.
* ``refc-base`` head is NOT a function of nav (both conditions).

⇒ flagship-v1's closed-loop strategic 1.0000 on this scene is the echo of its own
conditioning input, not a strategic decision. ``cl_metrics.py`` already computes
``route_head_nav_echo_check`` and stamps ``CIRCULAR_NAV_ECHO`` /
``do_not_quote``; the option-set path had no equivalent, and the compact Thor
extract that fed it (``scene2_route_decisions.json``) carries only ``i_gt`` and
the logits — **the nav field was dropped, so no guard could fire.**

Second control in here, same class:

    CONSTANT_LEFT scored on the SAME (event, rollout) instances as the paired
    flagship-vs-REF-C contrast. All 5 shared instances are ONE decision event
    (``7c72937c|J149|82``) whose option set is 4 roads but only TWO classes
    {LEFT, UTURN} — and UTURN is outside every deployed 3-way head's vocabulary
    (``refb.py:68``). So the only head-emittable answer there is LEFT: a constant
    predictor scores 1.0000 and reproduces the paired ``+1.000`` exactly.

⚠️ ``SCOREABLE`` (strategic_gt.py:312) counts ROADS. ``route_class_accuracy``
scores CLASSES. Those are not the same gate, and the difference is where the
single-option tie comes back.

    python3 strategic_conditioning_control.py \
        --rollouts ../alpasim-gsplat/results/scene2-realclose/rollouts \
        --out results/strategic_conditioning_control.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

#: head index -> nav index. The inverse of the mapping MEASURED on flagship-v1.
#: Emitting the fed nav through this is "the policy that only echoes its input".
NAV_TO_HEAD = {0: 1, 1: 0, 2: 2}

ROUTE_LOGIT_KEYS = ("s_route_logits", "route_logits")


def _taniteval():
    here = Path(__file__).resolve()
    for up in here.parents:
        if (up / "taniteval" / "taniteval" / "ci.py").exists():
            sys.path.insert(0, str(up / "taniteval"))
            break
    from taniteval import strategic_optionset as SO
    return SO


def _logits(extra):
    for k in ROUTE_LOGIT_KEYS:
        if extra.get(k) is not None:
            return k, extra[k]
    return None, None


def load_runs(rollout_dir):
    """``{run: {arm, condition, scene, rollouts:[{start_frame, ticks}]}}``.

    Each tick keeps BOTH the head's decision and the nav it was given, which is
    the whole point — the extract that lost ``nav`` is how the circularity became
    invisible.
    """
    runs = {}
    for f in sorted(glob.glob(str(Path(rollout_dir) / "rollouts_*.json"))):
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        rolls = []
        for ro in r["rollouts"]:
            ticks = []
            for st in ro["steps"]:
                key, rl = _logits(st.get("extra", {}) or {})
                nav = st.get("nav")
                ticks.append({
                    "i_gt": st["i_gt"],
                    "route_head": None if rl is None else int(np.argmax(rl)),
                    "nav": None if nav is None else int(nav),
                    "nav_echo": None if nav is None else NAV_TO_HEAD.get(int(nav)),
                    "constant_left": 0,
                    "_key": key,
                })
            rolls.append({"start_frame": ro.get("start_frame"), "ticks": ticks})
        runs[f"{r['arm']}_{r['condition']}"] = {
            "arm": r["arm"], "condition": r["condition"], "scene": r["scene"],
            "logit_key": ticks[0]["_key"] if ticks else None,
            "n_gt_poses": len(r["gt"]), "rollouts": rolls, "source": f,
        }
    return runs


def nav_echo_check(run):
    """Is the route head a deterministic function of the fed nav command?"""
    m = collections.defaultdict(collections.Counter)
    for ro in run["rollouts"]:
        for t in ro["ticks"]:
            if t["route_head"] is not None:
                m[t["nav"]][t["route_head"]] += 1
    det = bool(m) and all(len(v) == 1 for v in m.values())
    return {
        "head_is_deterministic_function_of_nav": det,
        "nav_to_head_counts": {str(k): dict(v) for k, v in sorted(
            m.items(), key=lambda x: (x[0] is None, x[0]))},
        "n_ticks": sum(sum(v.values()) for v in m.values()),
        "verdict": ("⛔ CIRCULAR — the route head reproduces the nav command the harness FEEDS "
                    "the policy (closedloop_drive.py:330 derives nav from the LOGGED GT track). "
                    "Any strategic accuracy from this head measures the conditioning input, not "
                    "a decision. DO NOT QUOTE."
                    if det else
                    "not an echo — the head is not a function of nav on these ticks"),
    }


def score(SO, run, report, scene, key, offset, n_boot):
    per_event = {}
    for ro in run["rollouts"]:
        preds = SO.event_predictions_from_ticks(ro["ticks"], report,
                                                pose_offset=offset, class_key=(key,))
        for eid, p in preds.items():
            if eid != "_join":
                per_event.setdefault(eid, []).append(p)
    return SO.strategic_family({scene: report, "_refused": {}}, per_event,
                               arm=f"{run['arm']}/{run['condition']}::{key}", n_boot=n_boot)


def instance_vector(SO, run, report, key, offset):
    out = {}
    for ro in run["rollouts"]:
        p = SO.event_predictions_from_ticks(ro["ticks"], report,
                                            pose_offset=offset, class_key=(key,))
        for eid, v in p.items():
            if eid != "_join":
                out[(eid, ro["start_frame"])] = v.get("class")
    return out


def head_vocabulary_branching(report, head_classes=(0, 1, 2)):
    """⛔ SCOREABLE counts ROADS; route_class_accuracy scores CLASSES.

    An event whose options collapse to ONE head-emittable class is a
    constant-predictor tie in the space the headline metric lives in — the exact
    degeneracy the option-set gate was built to refuse, one coordinate over.
    """
    rows = []
    for e in report.get("events", []) or []:
        if not e.get("SCOREABLE"):
            continue
        cls = {o.get("class") for o in (e.get("options") or []) if o.get("class") is not None}
        emit = sorted(c for c in cls if c in head_classes)
        rows.append({
            "event_id": e["event_id"], "n_options_roads": e["n_options"],
            "map_option_classes": sorted(cls), "head_emittable_classes": emit,
            "route_gt_class": e.get("route_gt_class"),
            "DEGENERATE_IN_HEAD_SPACE": bool(len(emit) < 2),
        })
    return rows


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts",
                    default=str(here.parent / "alpasim-gsplat" / "results" /
                                "scene2-realclose" / "rollouts"))
    ap.add_argument("--labels", default=str(here / "results" / "strategic_gt"))
    ap.add_argument("--offset", type=int, default=-1,
                    help="fitted in score_closedloop_strategic.py; accuracy is flat at -2/-1/0")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    SO = _taniteval()
    runs = load_runs(a.rollouts)
    if not runs:
        raise SystemExit(f"no rollouts_*.json under {a.rollouts}")
    scene = next(iter(runs.values()))["scene"]
    reports = SO.load_label_reports(a.labels)
    if scene not in reports:
        raise SystemExit(f"no option-set label for scene {scene}")
    rep = reports[scene]

    out = {
        "tool": "strategic_conditioning_control.py",
        "evidence_class": "MEASURED",
        "scene": scene,
        "pose_offset": a.offset,
        "rollout_dir": str(a.rollouts),
        "WHAT_THIS_CONTROLS_FOR": (
            "discrimination_control proves the LABELS have entropy (its ORACLE is built by "
            "copying route_gt_class — a tautology). It cannot tell whether an ARM's score is "
            "skill. This control scores the FED NAV COMMAND with no model at all, and a CONSTANT "
            "predictor, on the identical instances."),
        "NAV_ECHO": {}, "ARMS": {}, "HEAD_VOCABULARY_BRANCHING": {},
        "PAIRED_INSTANCE_AUDIT": {},
    }

    for name, run in runs.items():
        out["NAV_ECHO"][name] = dict(nav_echo_check(run), logit_key=run["logit_key"])
        block = {}
        for key, label in (("route_head", "the ARM"),
                           ("nav_echo", "the FED NAV COMMAND, no model"),
                           ("constant_left", "CONSTANT_LEFT, no model, no input")):
            fam = score(SO, run, rep, scene, key, a.offset, a.n_boot)
            block[key] = {
                "what": label,
                "route_class_accuracy": (fam.get("route_class_accuracy") or {}).get("mean"),
                "confusion_gt_x_pred": fam.get("route_class_confusion_gt_x_pred"),
                "n_decision_instances_scored": fam.get("n_decision_instances_scored"),
                "n_events_scoreable": fam.get("n_events_scoreable"),
                "n_scenes_scored": fam.get("n_scenes_scored"),
                "status": fam.get("status"),
            }
        arm_acc = block["route_head"]["route_class_accuracy"]
        echo_acc = block["nav_echo"]["route_class_accuracy"]
        block["ARM_IS_INDISTINGUISHABLE_FROM_ITS_OWN_CONDITIONING"] = bool(
            arm_acc is not None and echo_acc is not None and
            abs(arm_acc - echo_acc) < 1e-9 and
            block["route_head"]["confusion_gt_x_pred"] ==
            block["nav_echo"]["confusion_gt_x_pred"])
        out["ARMS"][name] = block

    rows = head_vocabulary_branching(rep)
    out["HEAD_VOCABULARY_BRANCHING"] = {
        "note": ("SCOREABLE (strategic_gt.py:312) is n_options over CONNECTING ROADS in the 4-way "
                 "map vocabulary. route_class_accuracy is scored in CLASS space and a deployed "
                 "head is 3-way (refb.py:68) — it cannot emit UTURN. Where the option set "
                 "collapses to ONE head-emittable class there is nothing to choose and a constant "
                 "predictor ties: the single-option degeneracy, one coordinate over."),
        "n_scoreable_events": len(rows),
        "n_DEGENERATE_in_head_space": sum(r["DEGENERATE_IN_HEAD_SPACE"] for r in rows),
        "per_event": rows,
    }

    # ---- the paired contrast, audited instance by instance
    fa = runs.get("flagship-v1_empty")
    rb = runs.get("refc-base_empty")
    if fa and rb:
        A = instance_vector(SO, fa, rep, "route_head", a.offset)
        B = instance_vector(SO, rb, rep, "route_head", a.offset)
        CL = instance_vector(SO, fa, rep, "constant_left", a.offset)
        gt = {e["event_id"]: e["route_gt_class"] for e in rep["events"]}
        shared = sorted(set(A) & set(B))

        def acc(V):
            return round(float(np.mean(
                [1.0 if (V.get(k) is not None and V[k] == gt[k[0]]) else 0.0
                 for k in shared])), 4) if shared else None
        out["PAIRED_INSTANCE_AUDIT"] = {
            "n_shared_instances": len(shared),
            "n_DISTINCT_DECISION_EVENTS": len({k[0] for k in shared}),
            "distinct_event_ids": sorted({k[0] for k in shared}),
            "start_frames": [k[1] for k in shared],
            "flagship_v1": acc(A), "refc_base": acc(B),
            "CONSTANT_LEFT_no_model": acc(CL),
            "paired_flagship_minus_refc": (None if acc(A) is None
                                           else round(acc(A) - acc(B), 4)),
            "paired_CONSTANT_minus_refc": (None if acc(CL) is None
                                           else round(acc(CL) - acc(B), 4)),
            "⛔_reading": (
                "score_closedloop_strategic.paired() reports n_shared_instances but NOT "
                "n_events — and every shared instance here is the SAME decision, re-driven from "
                "overlapping starts. A constant predictor reproduces the paired delta exactly, so "
                "the delta carries no information about the winning arm."),
        }

    txt = json.dumps(out, indent=1, default=str, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
        print(f"wrote {a.out}")
    for name, v in out["NAV_ECHO"].items():
        b = out["ARMS"][name]
        print(f"{name:24s} key={v['logit_key']:<16s} nav_echo={v['head_is_deterministic_function_of_nav']}"
              f"  arm={b['route_head']['route_class_accuracy']}"
              f"  nav_only={b['nav_echo']['route_class_accuracy']}"
              f"  const_left={b['constant_left']['route_class_accuracy']}"
              f"  INDISTINGUISHABLE={b['ARM_IS_INDISTINGUISHABLE_FROM_ITS_OWN_CONDITIONING']}")
    hv = out["HEAD_VOCABULARY_BRANCHING"]
    print(f"head-space degenerate events: {hv['n_DEGENERATE_in_head_space']}/{hv['n_scoreable_events']}")
    print("paired audit:", json.dumps(
        {k: v for k, v in out["PAIRED_INSTANCE_AUDIT"].items() if not k.startswith("⛔")},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
