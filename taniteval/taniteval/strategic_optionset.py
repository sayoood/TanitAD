"""TanitEval — the STRATEGIC family, scored against MAP-DERIVED OPTION SETS.

⛔ WHAT WAS BROKEN, AND WHY A NUMBER WAS WORSE THAN NO NUMBER
--------------------------------------------------------------
The closed-loop harness reported ``route_head_eq_logged = 1.0000`` on the 20 s
night clip (scene ``00040136``). That is not skill. It is a **constant-predictor
tie**: MEASURED from ``map.xodr``, the ego enters four junctions on that clip and
at **every one of them its own lane admits exactly ONE continuation**
(``stack/experiments/nurec-gsplat/results/junction_00040136.json``). There was
nothing to choose, so agreeing with the label was free. On the IDM streams the
situation is worse still — there is no route/map label at all
(``tanitad/eval/idm_families.py:296-314``), so the family is UNAVAILABLE.

The missing instrument was never "a route classifier" — the harness has one. It
is the **OPTION SET**: at each decision point, *which continuations did the map
admit*, and which did the ego take. Without it there is no way to separate a 1.0
that means *"chose correctly among four"* from a 1.0 that means *"there was one
road and everybody drove down it"*.

``stack/experiments/nurec-gsplat/strategic_gt.py`` produces those option sets
from OpenDRIVE + the clipgt ego track. **This module is the consumer**: it turns
them into the closed-loop STRATEGIC family, with the programme's own estimator
and with the degeneracy that produced the 1.0000 made structurally impossible.

THE LABEL CONTRACT (fields quoted at file:line, not paraphrased)
---------------------------------------------------------------
Produced by ``strategic_gt.strategic_gt()``; consumed here read-only:

===============================================  =========================================
field                                            strategic_gt.py
===============================================  =========================================
``events[].event_id``                            :293   the resampling/join key
``events[].SCOREABLE``                           :312   ``n_options >= 2``
``events[].n_options``                           :304   branching factor
``events[].options[] {road,class,class_name}``   :287-289  the confusion set
``events[].connecting_road_taken``               :299   road-level GT
``events[].route_gt_class``                      :302   class-level GT (map, not ego yaw)
``events[].entry_pose`` / ``entry_arc_m``        :295-296
``per_pose[] {pose,event_id,dist_...,admissible}``  :327-339
``SELFCONSISTENCY_CONTROL`` / ``ADMISSIBLE``     :383-384  the refusal gate
===============================================  =========================================

⛔ SIX TRAPS THIS MODULE EXISTS TO MAKE UNREACHABLE
--------------------------------------------------
0. **THE ARM ECHOES ITS OWN CONDITIONING INPUT.** The one a constant-predictor floor
   structurally CANNOT catch, because an echo is not constant and beats every
   constant. The harness derives ``nav`` from the ego's own logged future
   (``closedloop_drive.py:348`` → ``refb_labels.nav_command_v21``) and feeds it to
   the policy; a route head that hands it back scores perfectly and has decided
   nothing. **MEASURED 2026-08-03 over 78 NuRec T1 branch scenes, by sweeping the
   nav vocabulary at a FIXED observation:** ``flagship-v1``'s argmax moves with the
   nav input at **100 %** of poses (nav-to-image logit-variance ratio ≈ 7×) while
   ``refc-base``'s logits are **bit-identical under all four nav commands** — and
   the source agrees on both counts (``fourbrain.py:77-86`` FiLM-conditions every
   block on ``nav_emb`` and reads ``route_head`` off it; ``refc.py:1140`` reads
   ``route_head(pooled)``, i.e. image features, BEFORE the nav one-hot is fused at
   ``:1130``). :func:`conditioning_echo_control` +
   ``strategic_family(conditioning_sweeps=…)` → ``STRATEGIC_SKILL_ADMISSIBLE``.
   ⚠️ With no sweep the verdict is ``None`` (**UNTESTED**), never a pass.
   ⚠️ And a *permutation* of the conditioning is NOT a substitute: 50 of those 78
   scenes hold exactly ONE decision event, so a within-scene shuffle is the
   identity and the control measures nothing while appearing to run.

1. **A single-option junction is never scored.** ``SCOREABLE`` gates every value.
   A set with no scoreable event returns ``status="UNAVAILABLE"`` **with the
   reason and the n**, never a free 1.0. This is the one rule that killed the
   1.0000.

2. **The cluster is the SCENE and the value is the DECISION EVENT — never the
   pose.** Every pose approaching one junction carries the identical label, so a
   pose-level n overstates the sample by ~50x (on scene ``7c72937c``: 100
   admissible poses, **2** events). :func:`strategic_family` refuses to accept a
   pose-level value vector.

3. **1/n_options is NOT the null.** The programme has already published one
   retraction from comparing against 1/k when the empirical null was higher
   (CLAUDE.md: *"The empirical null is 0.3678, not 1/3"*). The decision-grade
   baseline here is the **BEST CONSTANT PREDICTOR fitted on the same events**,
   compared by a **paired** episode-cluster bootstrap. Fitting the constant on
   the same data makes the baseline stronger, i.e. the arm's win conservative —
   stated, not hidden.

4. **The route head's vocabulary is 3-way; the map's is 4-way.** ``refb.py:68``
   ``ROUTE_CLASSES = ("route_left","route_straight","route_right")`` vs
   ``strategic_gt.py:60`` ``{0:LEFT, 1:STRAIGHT, 2:RIGHT, 3:UTURN}``. The first
   three indices coincide; **UTURN is outside every deployed head's vocabulary**,
   so an event whose GT is UTURN can only ever be scored wrong. That subset is
   counted and reported (``n_events_gt_outside_head_vocabulary``) instead of
   quietly deflating the arm.
   ⚠️ And a THIRD vocabulary collides: the closed-loop harness's own
   ``ROUTE_NAMES = ("left","straight","right","unknown")``
   (``stack/experiments/alpasim-gsplat/cl_metrics.py:39``) uses **index 3 for
   UNKNOWN, not UTURN**. Those labels are a different object and are never mixed
   in here. :func:`assert_vocabularies` pins all of this so a rename breaks
   loudly rather than silently re-scoring.

5. **PRECISION IS REPORTED BESIDE RECALL, and the denominator is stated.** A
   recall-only frontier cannot see what it is paying — that is exactly how the
   brake_stop claim got published and retracted. Every class row carries
   ``n_true`` (support), ``n_pred``, ``recall`` and ``precision``.

Pure numpy + :mod:`taniteval.ci`. No torch, no map dependency, no pod path — so
the family is unit-testable anywhere, which is the point.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ci as _ci

__all__ = [
    "ROUTE_CLASSES",
    "HEAD_CLASSES",
    "HEAD_TO_ROUTE",
    "assert_vocabularies",
    "load_label_reports",
    "scoreable_events",
    "event_predictions_from_ticks",
    "strategic_family",
    "constant_arm",
    "discrimination_control",
    "conditioning_echo_control",
    "NAV_TO_ROUTE",
]

#: ``strategic_gt.py:60`` verbatim. Index IS the class id in the labels.
ROUTE_CLASSES = ("LEFT", "STRAIGHT", "RIGHT", "UTURN")
#: ``stack/tanitad/refs/refb.py:68`` verbatim — what a deployed route head emits.
HEAD_CLASSES = ("route_left", "route_straight", "route_right")
#: head index -> map class id. Identity on 0..2; the head has no UTURN.
HEAD_TO_ROUTE = {0: 0, 1: 1, 2: 2}
#: the class ids no deployed head can emit
CLASSES_OUTSIDE_HEAD_VOCABULARY = tuple(
    c for c in range(len(ROUTE_CLASSES)) if c not in HEAD_TO_ROUTE.values())

#: ``closedloop_drive.NAV_NAMES`` = ("follow","left","right","straight") -> this module's
#: map classes. FOLLOW carries no route meaning; STRAIGHT is the kindest reading of it
#: (and the most common class), which makes the echo baseline STRONGER, not weaker.
NAV_TO_ROUTE = {0: 1, 1: 0, 2: 2, 3: 1}

DEFAULT_N_BOOT = 2000


def assert_vocabularies() -> dict:
    """Pin the three colliding route vocabularies. Fails loud on a rename.

    ⛔ A silent vocabulary drift here does not crash — it re-scores. That is the
    worst failure mode this module has, so it is asserted rather than trusted.
    """
    if ROUTE_CLASSES[:3] != ("LEFT", "STRAIGHT", "RIGHT"):
        raise AssertionError(
            f"map route vocabulary drifted: {ROUTE_CLASSES!r}; strategic_gt.py:60 "
            "pins {0:LEFT, 1:STRAIGHT, 2:RIGHT, 3:UTURN}")
    if HEAD_CLASSES != ("route_left", "route_straight", "route_right"):
        raise AssertionError(
            f"head vocabulary drifted: {HEAD_CLASSES!r}; refb.py:68 pins "
            "('route_left','route_straight','route_right')")
    if HEAD_TO_ROUTE != {0: 0, 1: 1, 2: 2}:
        raise AssertionError("HEAD_TO_ROUTE must stay the identity on 0..2")
    return {
        "map_classes": list(ROUTE_CLASSES),
        "head_classes": list(HEAD_CLASSES),
        "head_index_to_map_class": {int(k): int(v) for k, v in HEAD_TO_ROUTE.items()},
        "classes_no_head_can_emit": [ROUTE_CLASSES[c]
                                     for c in CLASSES_OUTSIDE_HEAD_VOCABULARY],
        "⛔_third_vocabulary": (
            "cl_metrics.py:39 ROUTE_NAMES=('left','straight','right','unknown') uses index 3 "
            "for UNKNOWN, NOT UTURN. Those labels come from route_from_future_v21 (ego-yaw "
            "derived) and are a DIFFERENT OBJECT from the map-derived class. Never mix them."),
    }


# --------------------------------------------------------------------------- #
# Loading the labels                                                           #
# --------------------------------------------------------------------------- #
def _scene_of(report: dict, fallback: str) -> str:
    """Scene id — from the label's own event ids, not from the filename.

    ``event_id`` is ``"<scene>|J<junction>|<entry_pose>"`` (strategic_gt.py:293),
    so the scene travels *inside* the data and a renamed file cannot silently
    split one scene into two bootstrap clusters.
    """
    for e in report.get("events", []) or []:
        eid = e.get("event_id")
        if isinstance(eid, str) and "|" in eid:
            return eid.split("|", 1)[0]
    return fallback


def load_label_reports(source) -> dict:
    """``{scene_id: report}`` from a directory, a glob-able list, or dicts.

    ⛔ Scenes whose ``SELFCONSISTENCY_CONTROL`` did not pass are **REFUSED**, not
    scored: on the surveyed shortlist 2 of 15 failed it (``302c5c99`` 98.21 deg,
    ``d1a25a99`` 179.86 deg). They are returned in the ``_refused`` key so the
    refusal is visible in the output rather than discovered later as a gap.
    """
    reports, refused = {}, {}
    if isinstance(source, dict):
        items = [(k, v) for k, v in source.items()]
    else:
        paths = ([source] if isinstance(source, (str, Path)) else list(source))
        expanded = []
        for p in paths:
            p = Path(p)
            expanded.extend(sorted(p.glob("strategic_gt_*.json")) if p.is_dir() else [p])
        items = [(p.stem, json.loads(Path(p).read_text())) for p in expanded]
    for key, rep in items:
        sid = _scene_of(rep, str(key))
        if rep.get("ADMISSIBLE") is False:
            ctl = rep.get("SELFCONSISTENCY_CONTROL", {}) or {}
            refused[sid] = {
                "reason": "SELFCONSISTENCY_CONTROL did not pass",
                "worst_abs_err_deg": ctl.get("worst_abs_err_deg_untruncated"),
                "tolerance_deg": ctl.get("tolerance_deg"),
            }
            continue
        reports[sid] = rep
    reports["_refused"] = refused
    return reports


def scoreable_events(reports: dict) -> tuple[list, list]:
    """``(events, scene_ids)`` over every SCOREABLE (>=2-option) decision event.

    The pairing is positional and is the ONLY join used downstream, so the
    bootstrap's cluster vector can never drift out of step with its values.
    """
    events, scenes = [], []
    for sid, rep in reports.items():
        if sid.startswith("_"):
            continue
        for e in rep.get("events", []) or []:
            if e.get("SCOREABLE"):
                events.append(e)
                scenes.append(sid)
    return events, scenes


# --------------------------------------------------------------------------- #
# Joining a closed-loop rollout to the decision events                         #
# --------------------------------------------------------------------------- #
#: ⛔ EVERY key a deployed arm has ever used for its route head, because ONE key was
#: not enough and the cost was invisible.
#:
#: **MEASURED 2026-08-03, two independent probes, on the NuRec branch scene 7c72937c:**
#: ``flagship-v1`` writes ``extra["s_route_logits"]`` (3-wide, argmax {LEFT 369,
#: STRAIGHT 81} over 450 ticks) while ``refc-base`` writes ``extra["route_logits"]``
#: (3-wide, argmax {LEFT 99, STRAIGHT 70, RIGHT 281} over 450 ticks — a head that
#: VARIES over all three classes). ``cl_metrics.py:176`` reads only
#: ``ex["s_route_logits"]``, so the closed-loop STRATEGIC family has been emitting
#: *"this arm exposes no strategic route logits at the deploy path"* for REF-C — the
#: arm that BEATS flagship v1 in closed-loop — while its logits sat in the record
#: under a neighbouring name. **INSTRUMENT-FAIL, not a model gap.**
#: This is the programme's "absence found at ONE location is not absence" rule
#: costing a whole family, so the resolution is a LIST and the key that won is
#: reported in ``_join["class_key_resolved"]``.
ROUTE_LOGIT_KEYS = ("route_head", "s_route_logits", "route_logits")


def _first_present(row, keys):
    for k in keys:
        if row.get(k) is not None:
            return k, row[k]
    return None, None


def _as_class(value):
    """A route class from either a decoded int or a logit vector."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(np.argmax(np.asarray(value, dtype=float))) if len(value) else None
    return int(value)


def event_predictions_from_ticks(ticks, report, *, pose_key="i_gt",
                                 class_key=ROUTE_LOGIT_KEYS, road_key=None,
                                 pose_offset=0, horizon_m=None) -> dict:
    """Per-tick closed-loop route decisions -> ONE prediction per decision event.

    ``ticks`` are the rows a closed-loop rollout already emits; each must carry a
    pose index into the **logged** track (``cl_metrics.py:190`` calls it
    ``i_gt``) and the head's route output. ``class_key`` accepts a **list** of
    candidate keys (default :data:`ROUTE_LOGIT_KEYS`) and takes the first
    present — see that constant for the measured reason. The value may be an
    already-decoded class **or** a logit vector, which is argmaxed here.

    ⚠️ ``pose_offset`` exists because the clipgt track and the pose record are
    **not** the same index space — on the night clip the clip window is
    ``pose_record[49:251]`` (STRATEGIC_FAMILY.md §a). The caller must state the
    alignment; this function will not guess one. Ticks that fall outside the
    label's pose range are counted in ``n_ticks_out_of_range``, never wrapped.

    **The scored prediction is the one at the LAST admissible tick before the
    ego enters the junction** — the decision as it stood when it stopped being
    revisable. ``decision_lead_distance_m`` then reports how far back that same
    answer was already stable, which is the hierarchy signal a flat policy
    cannot fake.
    """
    keys = (class_key,) if isinstance(class_key, str) else tuple(class_key)
    per_pose = {p["pose"]: p for p in report.get("per_pose", []) or []}
    n_poses = int(report.get("n_poses", len(per_pose)))
    by_event: dict[str, list] = {}
    out_of_range = 0
    resolved = set()
    for t in ticks:
        raw = t.get(pose_key)
        if raw is None:
            continue
        pose = int(raw) + int(pose_offset)
        if pose < 0 or pose >= n_poses or pose not in per_pose:
            out_of_range += 1
            continue
        pp = per_pose[pose]
        if not pp.get("admissible"):
            continue
        d = pp.get("dist_to_decision_point_m")
        if horizon_m is not None and (d is None or d > horizon_m):
            continue
        hit, raw_cls = _first_present(t, keys)
        if hit:
            resolved.add(hit)
        cls = _as_class(raw_cls)
        by_event.setdefault(pp["event_id"], []).append({
            "pose": pose,
            "dist_m": None if d is None else float(d),
            "class": None if cls is None else int(HEAD_TO_ROUTE.get(cls, cls)),
            "road": None if road_key is None else t.get(road_key),
        })

    preds = {}
    for eid, rows in by_event.items():
        rows.sort(key=lambda r: (r["dist_m"] if r["dist_m"] is not None else 1e9))
        at_entry = rows[0]                      # smallest distance-to-decision
        preds[eid] = {
            "class": at_entry["class"],
            "road": at_entry["road"],
            "at_pose": at_entry["pose"],
            "n_ticks_on_approach": len(rows),
            "_approach": rows,
        }
    preds["_join"] = {
        "pose_key": pose_key, "pose_offset": int(pose_offset),
        "class_key_candidates": list(keys),
        "class_key_resolved": sorted(resolved) or None,
        "road_key": road_key,
        "n_ticks_in": len(list(ticks)) if hasattr(ticks, "__len__") else None,
        "n_ticks_out_of_range": out_of_range,
        "n_events_with_a_prediction": len(preds),
        "rule": ("scored at the LAST admissible tick before junction entry "
                 "(min dist_to_decision_point_m)"),
        "⛔_key_note": (
            "flagship-v1 writes `s_route_logits`, refc-base writes `route_logits` — MEASURED, "
            "same scene, same driver. Reading one key made REF-C's route head invisible to the "
            "closed-loop STRATEGIC family (cl_metrics.py:176). `class_key_resolved` records "
            "which key actually carried the decision, so that failure cannot recur silently."),
    }
    return preds


def _decision_lead_distance(approach, gt_class, gt_road) -> tuple | None:
    """``(lead_m, available_m, censored)`` — how far out the answer was already
    correct and stayed correct all the way in; ``0.0`` if wrong at the entry.

    ⭐ This is the only strategic metric a FLAT policy structurally cannot fake:
    it asks *when* the choice was made, not merely whether it was right once the
    turn was already visible. HYPOTHESIS-class as a discriminator until it has
    been run against two real arms.

    ⚠️ **IT IS RIGHT-CENSORED BY THE CLIP, and that is not cosmetic.** MEASURED
    2026-08-03 on the winner scene ``7c72937c``: a policy that commits **60 m**
    out scores **20.43 m**, because junction 149 sits at ``entry_arc_m = 18.07``
    and the clip simply has no more approach to observe. A censored value is a
    LOWER BOUND — *"correct for the whole observable approach"* — and comparing
    two arms on clips with different approach lengths compares the clips.
    ``n_censored`` and ``available_lead_m`` travel with the number for exactly
    that reason.
    """
    rows = [r for r in approach if r["dist_m"] is not None]
    if not rows:
        return None
    rows.sort(key=lambda r: r["dist_m"])            # nearest first
    lead = 0.0
    for r in rows:
        ok = (r["class"] == gt_class) if r["class"] is not None else False
        if gt_road is not None and r["road"] is not None:
            ok = ok and (r["road"] == gt_road)
        if not ok:
            break
        lead = r["dist_m"]
    avail = float(rows[-1]["dist_m"])
    return round(float(lead), 2), round(avail, 2), bool(lead >= avail - 1e-9)


# --------------------------------------------------------------------------- #
# The family                                                                   #
# --------------------------------------------------------------------------- #
def _per_class(gt, pred, classes=ROUTE_CLASSES) -> dict:
    """Recall AND precision AND both denominators, per class. Never recall alone."""
    gt = list(gt)
    pred = list(pred)
    out = {}
    for c, name in enumerate(classes):
        n_true = sum(1 for g in gt if g == c)
        n_pred = sum(1 for p in pred if p == c)
        hit = sum(1 for g, p in zip(gt, pred) if g == c and p == c)
        if n_true == 0 and n_pred == 0:
            continue
        out[name] = {
            "n_true": n_true,
            "n_pred": n_pred,
            "n_correct": hit,
            "recall": round(hit / n_true, 4) if n_true else None,
            "precision": round(hit / n_pred, 4) if n_pred else None,
        }
    return out


def _confusion(gt, pred, classes=ROUTE_CLASSES) -> dict:
    conf: dict = {}
    for g, p in zip(gt, pred):
        gn = classes[g] if isinstance(g, int) and 0 <= g < len(classes) else str(g)
        pn = (classes[p] if isinstance(p, int) and 0 <= p < len(classes)
              else ("none" if p is None else str(p)))
        conf.setdefault(gn, {})
        conf[gn][pn] = conf[gn].get(pn, 0) + 1
    return conf


def constant_arm(events, class_id) -> list:
    """Predictions of the constant predictor that always answers ``class_id``.

    The baseline that ``route_head_eq_logged = 1.0000`` could not be told apart
    from. It is scored on **every** arm's events, always.
    """
    return [{"class": int(class_id), "road": None} for _ in events]


def _as_pred_list(events, scenes, predictions):
    """``(events, scenes, preds)`` — accepts three prediction shapes.

    * ``{event_id: pred}``                one closed-loop trial
    * ``{event_id: [pred, pred, ...]}``   SEVERAL trials over the same event, e.g. the
      9 overlapping rollouts a closed-loop panel launches at ``start_frame`` 0, 15, 30 …
      Each trial is a separate decision instance, so the event is REPEATED in the value
      vector — the cluster stays the SCENE, which is what keeps the CI honest.
    * a positional list aligned with ``events``

    ⛔ Averaging the trials into one value per event would hide a policy that is
    correct from one start and wrong from the next, which is precisely the
    closed-loop instability the panel exists to expose.
    """
    if not isinstance(predictions, dict):
        pl = list(predictions)
        if len(pl) != len(events):
            raise ValueError(
                f"positional predictions must align with events: {len(pl)} vs {len(events)}")
        return events, scenes, pl
    ev_out, sc_out, pr_out = [], [], []
    for e, s in zip(events, scenes):
        p = predictions.get(e["event_id"])
        for one in (p if isinstance(p, list) else [p]):
            ev_out.append(e)
            sc_out.append(s)
            pr_out.append(one)
    return ev_out, sc_out, pr_out


def _guard_single_cluster(block: dict, n_scenes: int) -> dict:
    """⛔ A one-cluster episode bootstrap returns ``lo == hi == point``.

    That prints as an interval of width **zero** — an infinitely precise verdict
    from a single scene. The programme has already retracted numbers for less.
    The point estimate stays (it is the full-set mean and it is correct); the
    INTERVAL is marked inadmissible.
    """
    if isinstance(block, dict) and n_scenes < 2 and "lo" in block:
        block["CI_NOT_ADMISSIBLE"] = True
        block["⛔_ci_note"] = (
            f"n_scenes = {n_scenes}. The episode-cluster bootstrap resamples SCENES, so with "
            "one cluster every draw is the same set and lo == hi == the point estimate. That "
            "is not a zero-width interval, it is NO interval. The point estimate is valid; "
            "quote it WITHOUT bounds, and get more branch scenes (results/"
            "junction_turn_scenes.tsv lists 225) before claiming separation.")
    return block


def conditioning_echo_control(sweeps) -> dict:
    """⭐ THE SECOND DEGENERACY, AND THE ONE THAT SURVIVED THE FIRST GUARD.

    :func:`discrimination_control` proves the **labels** carry entropy, and
    :func:`strategic_family`'s ``BEST_CONSTANT`` floor kills a head that always
    answers one class. Neither can see the failure that actually happened: the
    harness computes the nav command from **the ego's own logged future**
    (``closedloop_drive.py:348 nav_from_route`` -> ``refb_labels.nav_command_v21``),
    feeds it to the policy, and the policy's route head hands it straight back.
    That head is not constant, it beats every constant, and it is **still not
    strategy** — it is a relabelling of an oracle input.

    ``sweeps`` is a list of per-decision observations, each
    ``{conditioning_value: predicted_class}`` **taken at a FIXED observation**.
    Sweeping the input while the pixels are held constant is a MANIPULATION, so
    it identifies the echo; an observational nav-vs-head contingency table cannot,
    because a competent head and an echo agree whenever the nav is correct.

    ⚠️ The sweep must cover more than one conditioning value or the control
    silently measures nothing — the failure mode that made a within-scene
    permutation useless on the 60 %+ of branch scenes carrying ONE decision.

    Returns ``ECHO`` True when the prediction moves with the input on a majority
    of observations, and ``DETERMINISTIC_ECHO`` when the map input -> output is a
    function with no observation-dependence left at all.
    """
    rows = [s for s in (sweeps or []) if isinstance(s, dict) and len(s) >= 2]
    n_all = len(sweeps or [])
    if not rows:
        return {"ECHO": None, "n_observations": n_all, "n_usable": 0,
                "reason": ("no observation carries >=2 distinct conditioning values, so "
                           "the sweep cannot separate an echo from a decision. This is a "
                           "MISSING CONTROL, not a pass.")}
    moved = sum(1 for s in rows if len({v for v in s.values() if v is not None}) > 1)
    mapping, deterministic = {}, True
    for s in rows:
        for k, v in s.items():
            if v is None:
                continue
            if k in mapping and mapping[k] != v:
                deterministic = False
            mapping.setdefault(k, v)
    rate = moved / len(rows)
    return {
        "n_observations": n_all,
        "n_usable": len(rows),
        "n_observations_whose_answer_MOVED_with_the_input": moved,
        "echo_rate": round(rate, 4),
        "ECHO": bool(rate >= 0.5),
        "DETERMINISTIC_ECHO": bool(deterministic and rate > 0.0),
        "input_to_output_map_if_deterministic": (
            {int(k): int(v) for k, v in sorted(mapping.items())} if deterministic else None),
        "⛔_verdict": (
            "the route head is a relabelling of its own conditioning input: it changes "
            "answer when ONLY the input changes and the observation does not. Any accuracy "
            "scored while that input is an ORACLE is the oracle, not the arm."
            if rate >= 0.5 else
            "not an echo on this sweep — the answer is driven by the observation, not by "
            "the conditioning input."),
        "⭐_why_this_is_not_the_constant_guard": (
            "an echo is NOT constant and BEATS every constant predictor, so BEST_CONSTANT "
            "cannot catch it. It is the second way to score 1.0000 without deciding "
            "anything, and it is the one that survived the first guard."),
    }


def strategic_family(reports, predictions, *, arm="arm", n_boot=DEFAULT_N_BOOT,
                     seed=0, compare_to_best_constant=True,
                     conditioning_sweeps=None) -> dict:
    """THE closed-loop STRATEGIC family, scored on map-derived option sets.

    ``reports``      ``{scene_id: strategic_gt report}`` (:func:`load_label_reports`)
    ``predictions``  ``{event_id: {"class": int|None, "road": str|None,
                       "_approach": [...] optional}}`` — or a positional list.

    Returns the family block. ``status`` is ``"UNAVAILABLE"`` — with the reason
    and the n — whenever the SCENES admit no choice or the ARM emits no route
    decision. **It never returns a number that a constant predictor could tie.**
    """
    assert_vocabularies()
    refused = reports.get("_refused", {}) if isinstance(reports, dict) else {}
    events, scenes = scoreable_events(reports)
    all_events = [e for sid, rep in reports.items() if not sid.startswith("_")
                  for e in (rep.get("events", []) or [])]
    n_single = sum(1 for e in all_events if not e.get("SCOREABLE"))

    base = {
        "family": "STRATEGIC",
        "source": "strategic_gt.py option sets (map.xodr) — MEASURED",
        "n_scenes_supplied": len([s for s in reports if not s.startswith("_")]),
        "n_scenes_refused_by_selfconsistency": len(refused),
        "scenes_refused": refused,
        "n_decision_events_total": len(all_events),
        "n_events_single_option_excluded": n_single,
        "vocabularies": assert_vocabularies(),
    }

    if not events:
        base.update({
            "status": "UNAVAILABLE",
            "n": 0,
            "reason": (
                "no SCOREABLE decision event in this set: every junction offers exactly ONE "
                f"lane-level continuation ({n_single} single-option events over "
                f"{len([s for s in reports if not s.startswith('_')])} scenes). A constant "
                "predictor ties, so no strategic accuracy is DEFINED here — this is a property "
                "of the SCENES, not a missing instrument. ⛔ Do NOT fill this with "
                "route_head_eq_logged: on the night clip that read 1.0000 for exactly this "
                "reason. Select scenes with a branch (the 141 T1 scenes in "
                "results/junction_turn_scenes.tsv) before quoting a strategic accuracy."),
        })
        return base

    events, scenes, preds = _as_pred_list(events, scenes, predictions)
    gt_class = [e.get("route_gt_class") for e in events]
    gt_road = [e.get("connecting_road_taken") for e in events]
    n_opts = [int(e.get("n_options", 0)) for e in events]

    # ⛔ a None-vs-None match is a FREE POINT and is refused, not scored.
    def _class_hit(p, g):
        pc = None if p is None else p.get("class")
        return 0.0 if (pc is None or g is None) else float(int(pc) == int(g))

    def _road_hit(p, g):
        pr = None if p is None else p.get("road")
        return 0.0 if (pr is None or g is None) else float(pr == g)

    cls_ok = [_class_hit(p, g) for p, g in zip(preds, gt_class)]
    has_class = any((p or {}).get("class") is not None for p in preds)
    has_road = any((p or {}).get("road") is not None for p in preds)

    if not has_class and not has_road:
        base.update({
            "status": "UNAVAILABLE",
            "n": len(events),
            "reason": (
                f"the option sets ARE scoreable ({len(events)} events over "
                f"{len(set(scenes))} scenes, branching factors {sorted(set(n_opts))}) but this "
                "arm emitted no route decision on any of them — no `s_route_logits` at the "
                "deploy path, or no tick landed inside a decision's admissible approach. "
                "That is an ARM/JOIN gap, not a scene gap, and it is a WORK ITEM."),
        })
        return base

    boot = dict(n_boot=n_boot, seed=seed)
    out = dict(base)
    out.update({
        "status": "OK",
        "arm": arm,
        # ⚠️ TWO denominators, because one closed-loop panel scores the SAME junction from
        # several overlapping starts. `n_events_scoreable` is the number of distinct
        # DECISIONS in the world; `n_decision_instances_scored` is the number of times an
        # arm was asked. Quoting the second as if it were the first inflates the sample.
        "n_events_scoreable": len({e["event_id"] for e in events}),
        "n_decision_instances_scored": len(events),
        "n_scenes_scored": len(set(scenes)),
        "n_events_without_a_prediction": sum(
            1 for p in preds if p is None or p.get("class") is None),
        "n_events_gt_outside_head_vocabulary": sum(
            1 for g in gt_class if g in CLASSES_OUTSIDE_HEAD_VOCABULARY),
        "⚠️_outside_vocabulary_note": (
            f"a 3-way head ({', '.join(HEAD_CLASSES)}) CANNOT emit "
            f"{[ROUTE_CLASSES[c] for c in CLASSES_OUTSIDE_HEAD_VOCABULARY]}; those events can "
            "only score wrong. Counted, not silently deflating the arm."),
        "branching_factor": {"values": n_opts, "max": max(n_opts),
                             "distinct": sorted(set(n_opts))},
        "route_class_accuracy": _ci.episode_cluster_bootstrap(cls_ok, scenes, **boot),
        "route_class_per_class": _per_class(
            [g for g in gt_class],
            [(p or {}).get("class") for p in preds]),
        "route_class_confusion_gt_x_pred": _confusion(
            gt_class, [(p or {}).get("class") for p in preds]),
        # ⭐ A STRATEGIC ERROR THE CONFUSION MATRIX CANNOT NAME: predicting a manoeuvre
        # the MAP DOES NOT ADMIT here. "wrong branch" and "no such branch" are different
        # failures — the first is a choice, the second means the head is not reading the
        # junction at all — and a class-vs-class confusion table scores them identically.
        # Only an option-set label can tell them apart, which is the point of this module.
        "n_predictions_outside_the_option_set": sum(
            1 for e, p in zip(events, preds)
            if (p or {}).get("class") is not None
            and (p["class"] not in {o.get("class") for o in (e.get("options") or [])})),
        "predictions_outside_the_option_set_note": (
            "the map admitted no continuation of that class at this junction. Distinct "
            "from a wrong-branch error, and invisible to a class-vs-class confusion."),
        "accuracy_by_branching_factor": {
            int(k): {"n": int(sum(1 for v, kk in zip(cls_ok, n_opts) if kk == k)),
                     "acc": round(float(np.mean([v for v, kk in zip(cls_ok, n_opts)
                                                 if kk == k])), 4)}
            for k in sorted(set(n_opts))},
        "CLUSTER": "scene — the VALUE is per decision EVENT, never per pose",
        "⛔_never_per_pose": (
            "every pose approaching one junction carries the identical label; a pose-level n "
            "overstates the sample by ~50x (scene 7c72937c: 100 admissible poses, 2 events)."),
        "estimator": "episode_cluster_bootstrap (taniteval.ci), cluster = scene",
    })

    # ---- FLOORS. Three of them, because 1/k alone has already misled us once.
    uniform_floor = float(np.mean([1.0 / max(k, 1) for k in n_opts]))
    class_floor = float(np.mean(
        [1.0 / max(len({o.get("class") for o in (e.get("options") or [])
                        if o.get("class") is not None}), 1) for e in events]))
    const_scores = {}
    for c in sorted(HEAD_TO_ROUTE.values()):
        v = [_class_hit(p, g) for p, g in zip(constant_arm(events, c), gt_class)]
        const_scores[ROUTE_CLASSES[c]] = round(float(np.mean(v)), 4)
    best_name = max(const_scores, key=lambda k: const_scores[k])
    best_c = ROUTE_CLASSES.index(best_name)
    best_v = [_class_hit(p, g) for p, g in zip(constant_arm(events, best_c), gt_class)]

    out["floors"] = {
        "uniform_over_options": round(uniform_floor, 4),
        "uniform_over_option_CLASSES": round(class_floor, 4),
        "constant_predictor_per_class": const_scores,
        "BEST_CONSTANT": {"class": best_name, "accuracy": const_scores[best_name]},
        "⛔_which_floor_decides": (
            "BEST_CONSTANT. 1/n_options is a *design* floor, not the empirical null — the "
            "programme has already retracted one claim for comparing against 1/k when the "
            "empirical null was higher. The constant is fitted on THESE events, which makes "
            "the baseline stronger and the arm's win CONSERVATIVE."),
    }

    if compare_to_best_constant:
        out["vs_best_constant"] = _ci.paired_episode_cluster_bootstrap(
            cls_ok, best_v, scenes, **boot)
        out["vs_best_constant"]["baseline"] = f"always {best_name}"
        out["beats_best_constant"] = bool(
            out["vs_best_constant"]["separated"] and out["vs_best_constant"]["delta"] > 0)

    # ---- degeneracy, decided on the DATA, and always emitted
    n_lab = len({g for g in gt_class if g is not None})
    n_pr = len({(p or {}).get("class") for p in preds
                if (p or {}).get("class") is not None})
    out["label_degenerate"] = bool(n_lab < 2)
    out["prediction_degenerate"] = bool(n_pr < 2)
    if n_lab < 2:
        out["⛔_degenerate_note"] = (
            f"the map-derived label takes {n_lab} distinct value(s) over {len(events)} "
            "scoreable events. A constant predictor scores the same; read "
            "`vs_best_constant`, not the accuracy. Widen the scene set.")

    # ---- road-level choice, only where a class maps to ONE option road
    if has_road:
        road_ok = [_road_hit(p, g) for p, g in zip(preds, gt_road)]
        out["route_choice_accuracy_road_level"] = _ci.episode_cluster_bootstrap(
            road_ok, scenes, **boot)
        out["route_choice_note"] = "arm emitted an explicit connecting-road/goal identity"
    else:
        amb = 0
        proj_ok, proj_scene = [], []
        for e, p, g, s in zip(events, preds, gt_road, scenes):
            pc = (p or {}).get("class")
            if pc is None:
                continue
            same = [o for o in (e.get("options") or []) if o.get("class") == pc]
            if len(same) != 1:
                amb += 1
                continue
            proj_ok.append(float(same[0].get("road") == g))
            proj_scene.append(s)
        out["route_choice_accuracy_road_level"] = (
            _ci.episode_cluster_bootstrap(proj_ok, proj_scene, **boot) if proj_ok else
            {"n": 0, "reason": "no event where the predicted class maps to exactly one option road"})
        out["route_choice_note"] = (
            "⚠️ DERIVED, not emitted: this arm predicts a CLASS, not a road identity. The class is "
            f"projected onto the option whose class matches. {amb} of {len(events)} events are "
            "AMBIGUOUS (several options share the predicted class — e.g. scene 7c72937c junction "
            "149 has THREE UTURN options) and are EXCLUDED, so the denominator here is "
            f"{len(proj_ok)}, not {len(events)}. A road-level number on a class head is not "
            "comparable to one from a goal-selecting head.")
        out["n_events_class_to_road_ambiguous"] = amb

    # ---- decision lead distance, when the caller kept the approach
    leads, lead_scene, avails, censored = [], [], [], 0
    for e, p, s in zip(events, preds, scenes):
        ap = (p or {}).get("_approach")
        if not ap:
            continue
        d = _decision_lead_distance(ap, e.get("route_gt_class"),
                                    e.get("connecting_road_taken") if has_road else None)
        if d is not None:
            leads.append(d[0])
            avails.append(d[1])
            censored += int(d[2])
            lead_scene.append(s)
    out["decision_lead_distance_m"] = (
        dict(_ci.episode_cluster_bootstrap(leads, lead_scene, **boot),
             **{"definition": ("largest approach distance from which the choice was already "
                               "correct and stayed correct to the junction; 0.0 = wrong at entry"),
                "available_lead_m_mean": round(float(np.mean(avails)), 2),
                "available_lead_m_max": round(float(max(avails)), 2),
                "n_censored_by_clip": censored,
                "⚠️_censoring": (
                    f"{censored} of {len(leads)} events are RIGHT-CENSORED — the arm was correct "
                    "for the entire observable approach, so the value is a LOWER BOUND set by the "
                    "clip, not by the arm. MEASURED on 7c72937c: a policy committing 60 m out "
                    "scores 20.43 m because junction 149 sits at entry_arc_m 18.07. ⛔ Never "
                    "compare this across scene sets with different approach lengths."),
                "evidence_class": "MEASURED value, HYPOTHESIS discriminator — "
                                  "never yet run against two real arms"})
        if leads else
        {"n": 0, "reason": ("no per-tick approach supplied — pass predictions built by "
                            "event_predictions_from_ticks(), which keeps `_approach`")})

    # ⛔ EVERY interval in the block gets the one-cluster guard, not just the headline.
    n_sc = len(set(scenes))
    for k in ("route_class_accuracy", "route_choice_accuracy_road_level",
              "decision_lead_distance_m", "vs_best_constant"):
        if isinstance(out.get(k), dict):
            _guard_single_cluster(out[k], n_sc)
    if n_sc < 2:
        out["⛔_SINGLE_SCENE"] = (
            f"scored on {n_sc} scene. Point estimates are valid; NO interval on this block is "
            "admissible and `beats_best_constant` must not be quoted as a verdict — the "
            "resampling unit is the scene. A +-0.10 strategic verdict needs ~100+ branch "
            "scenes (results/junction_turn_scenes.tsv lists 225 candidates).")
        out["beats_best_constant_ADMISSIBLE"] = False

    # ---- ⛔ the INPUT-ECHO degeneracy, which BEST_CONSTANT structurally cannot see
    echo = conditioning_echo_control(conditioning_sweeps)
    out["conditioning_echo_control"] = echo
    beat = bool(out.get("beats_best_constant"))
    if echo.get("ECHO"):
        out["STRATEGIC_SKILL_ADMISSIBLE"] = False
        out["⛔_ECHO"] = (
            f"echo_rate {echo['echo_rate']}: this arm's route answer moves with its "
            "CONDITIONING INPUT at a fixed observation. The nav command is derived from "
            "the ego's own logged future (closedloop_drive.py:348 -> "
            "refb_labels.nav_command_v21), so an accuracy scored under it is the ORACLE "
            "passing through, NOT a strategic decision. Quote the navFOLLOW (neutral-"
            "conditioning) number, or do not quote one. `beats_best_constant` is "
            f"{beat} and is NOT sufficient here — an echo beats every constant.")
    elif echo.get("ECHO") is None:
        out["STRATEGIC_SKILL_ADMISSIBLE"] = None
        out["⚠️_ECHO_UNTESTED"] = (
            "no conditioning sweep was supplied, so the input-echo degeneracy is UNTESTED. "
            "Pass `conditioning_sweeps=[{nav: predicted_class}, ...]` measured at FIXED "
            "observations. An untested control is a gap, not a pass.")
    else:
        out["STRATEGIC_SKILL_ADMISSIBLE"] = beat

    out["_binding_rule"] = (
        "Sayed 2026-08-02: STRATEGIC is one of FOUR families reported in ADDITION to ADE, "
        "per-family, never pooled, each with its estimator and CI. A family reported "
        "UNAVAILABLE is a WORK ITEM, not a pass.")
    return out


# --------------------------------------------------------------------------- #
# The MANDATORY negative control                                               #
# --------------------------------------------------------------------------- #
def discrimination_control(reports, *, n_boot=DEFAULT_N_BOOT, seed=0) -> dict:
    """⭐ NEGATIVE CONTROL FIRST — prove the metric can DISCRIMINATE before quoting it.

    Scores synthetic arms against the **real** option sets:

    ==================  ======================================================
    arm                 behaviour
    ==================  ======================================================
    ``ORACLE``          always the branch the ego took
    ``CONSTANT_*``      always one class — **the degeneracy that produced
                        route_head_eq_logged = 1.0000**
    ``UNIFORM_RANDOM``  uniform over the option CLASSES (seeded)
    ``NO_HEAD``         emits nothing — must score 0, never be dropped
    ==================  ======================================================

    ``DISCRIMINATES`` is True only when ORACLE is **paired-CI-separated above the
    BEST CONSTANT**. If a constant predictor cannot be beaten on this scene set,
    the set cannot carry a strategic verdict and the number must not be quoted —
    which is precisely the check the night clip failed.
    """
    events, scenes = scoreable_events(reports)
    if not events:
        return {"DISCRIMINATES": False, "n_events": 0,
                "reason": ("no SCOREABLE event: every junction is single-option, so no "
                           "arm can be told from a constant predictor here. This is the "
                           "night-clip state and it must NOT be reported as an accuracy.")}
    gt = [e.get("route_gt_class") for e in events]
    rng = np.random.default_rng(seed)

    arms = {"ORACLE": [{"class": g, "road": e.get("connecting_road_taken")}
                       for g, e in zip(gt, events)],
            "NO_HEAD": [None] * len(events)}
    for c in sorted(HEAD_TO_ROUTE.values()):
        arms[f"CONSTANT_{ROUTE_CLASSES[c]}"] = constant_arm(events, c)
    rnd = []
    for e in events:
        cs = sorted({o.get("class") for o in (e.get("options") or [])
                     if o.get("class") is not None})
        rnd.append({"class": int(rng.choice(cs)) if cs else None, "road": None})
    arms["UNIFORM_RANDOM_OVER_OPTION_CLASSES"] = rnd

    def _hit(preds):
        return [0.0 if (p is None or p.get("class") is None or g is None)
                else float(int(p["class"]) == int(g)) for p, g in zip(preds, gt)]

    scored, curves = {}, {}
    for name, preds in arms.items():
        v = _hit(preds)
        curves[name] = v
        scored[name] = _ci.episode_cluster_bootstrap(v, scenes, n_boot=n_boot, seed=seed)

    const_names = [k for k in scored if k.startswith("CONSTANT_")]
    best_const = max(const_names, key=lambda k: scored[k]["mean"])
    paired = _ci.paired_episode_cluster_bootstrap(
        curves["ORACLE"], curves[best_const], scenes, n_boot=n_boot, seed=seed)

    return {
        "control": "ORACLE vs BEST CONSTANT PREDICTOR on the real option sets",
        "evidence_class": "MEASURED",
        "n_events": len(events), "n_scenes": len(set(scenes)),
        "arms": scored,
        "best_constant": best_const,
        "oracle_minus_best_constant": paired,
        "DISCRIMINATES": bool(paired["separated"] and paired["delta"] > 0),
        "constant_predictor_does_not_score_well": bool(
            scored[best_const]["mean"] < scored["ORACLE"]["mean"]),
        "no_head_scores_zero": bool(scored["NO_HEAD"]["mean"] == 0.0),
        "⛔_why_this_gate": (
            "a constant predictor tying the oracle is EXACTLY the degeneracy that made "
            "route_head_eq_logged = 1.0000 meaningless. If DISCRIMINATES is False the scene "
            "set cannot carry a strategic verdict and no accuracy may be quoted from it."),
        "⚠️_power": (
            "the cluster bootstrap narrows ~1/sqrt(n_scenes); a +-0.10 strategic verdict needs "
            "~100+ scenes WITH a branch. One scene cannot carry a strategic CI."),
    }
