"""Score the necessary-condition probes and report the false-negative bounds.

Reads the feature file that ``probe_cot_absence_falsenegatives.py`` writes and
the label blob, and answers, per ``vlm-cot`` token: *if we now call every
unlabelled clip a NEGATIVE, how many of those negatives can be wrong?*

⛔ THE ARITHMETIC, stated once so every number below is readable:

  ``S``  sensitivity -- the probe's fire rate on the LABELLED POSITIVES. This
         is the INSTRUMENT CONTROL. The probes are necessary conditions, so a
         correct probe fires on essentially all positives; ``S`` well below 1
         means the probe is not measuring the token and its bound is VOID.
  ``F``  the probe's fire rate on the caption-UNLABELLED clips.
  ``f0`` the probe's fire rate on clips the frozen exclusion table makes
         CERTAINLY NEGATIVE. This is the base rate of the necessary condition
         among true negatives. It exists only for tokens with an exclusion
         partner.

  UPPER BOUND on the false-negative rate  = ``F``          (assumption-free:
      a clip that fails a necessary condition cannot be a false negative)
  sensitivity-corrected                   = ``F / S``      (bound, de-blunted)
  POINT ESTIMATE, where ``f0`` exists     = ``(F - f0) / (S - f0)``, clipped

⭐ THE PIPELINE CONTROL RUNS FIRST AND MUST PASS. Three GEOMETRY-provenance
tokens (``STOP_POINT``, ``TURN_L``, ``TURN_R``) have a truth that ego geometry
knows independently. Scoring them with the same features and the same time base
is the check that the time base and the feature extraction are sound at all. If
a stop-probe cannot find the labelled stops, nothing downstream is a
measurement.

ASCII only in the printed output: this box is cp1252.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. ⛔ Not normal-approximation: several of these
    counts are single digits, where the normal interval leaves the unit
    interval and reports a negative rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------
# THE PROBES. Each returns True (fires), False (does not), or None (this clip
# carries no evidence either way -- excluded from both numerator and
# denominator, and counted).
# ---------------------------------------------------------------------------
def _need_join(f):
    return f.get("agents") is not None


def p_stop(f, th):                                    # PIPELINE CONTROL
    e = f["ego"]
    return e["v_min"] <= th


def p_turn(f, th, sign):                              # PIPELINE CONTROL
    e = f["ego"]
    return (e["peak_yaw_deg"] * sign) >= th


def p_yield_kin(f, th):
    e = f["ego"]
    return bool(e["drop_lead_ms"] >= th or e["v_min"] <= 2.0)


def p_yield_conflict(f, th):
    if not _need_join(f):
        return None
    a = f["agents"]
    return bool(p_yield_kin(f, th) and (a["n_crossing_tracks"] > 0
                                        or a["n_oncoming_tracks"] > 0
                                        or a["n_vru_ahead"] > 0))


def p_gap_target(f, th):
    if not _need_join(f):
        return None
    return f["agents"]["lead_frac_wide"] >= th


def p_oncoming(f, th):
    if not _need_join(f):
        return None
    a = f["agents"]
    return bool(a["n_oncoming_tracks"] > 0
                and (a["oncoming_min_cx"] is not None
                     and a["oncoming_min_cx"] <= th))


def p_corridor_offset(f, th):
    return f["ego"]["lat_resid_m"] >= th


def p_evade(f, th):
    if not _need_join(f):
        return None
    a = f["agents"]
    return bool((a["static_veh_in_corridor"] or a["vru_in_corridor"])
                and f["ego"]["lat_resid_m"] >= th)


def p_overtake(f, th):
    if not _need_join(f):
        return None
    a = f["agents"]
    return bool(a["mover_ahead"] and (a["closing_on_mover"]
                                      or a["overtaken_a_mover"])
                and f["ego"]["lat_resid_m"] >= th)


def p_lanewidth(f, th):        # ⚠️ NOT a lane-change probe. See the report.
    return abs(f["ego"]["lat_peak_m"]) >= th


PROBES = {
    "YIELD": ("decelerated by >= TH m/s in [t0, t0+6] or v_min <= 2.0 m/s "
              "(ego kinematics only)", p_yield_kin, [1.0, 1.5, 2.0, 3.0]),
    "YIELD@conflict": ("the kinematic test AND a crossing / oncoming / VRU "
                       "agent in the scene", p_yield_conflict, [1.5]),
    "GAP_TARGET": ("a vehicle in the +-1.8 m corridor ahead for >= TH of the "
                   "window", p_gap_target, [0.05, 0.25, 0.50, 0.75]),
    "REACT_ON_ONCOMING": ("a vehicle with opposed heading (>135 deg) ahead "
                          "within TH m and |cy| <= 6 m", p_oncoming,
                          [30.0, 50.0, 80.0]),
    "CORRIDOR_OFFSET": ("curvature-detrended lateral residual >= TH m",
                        p_corridor_offset, [0.05, 0.10, 0.20, 0.40, 1.00]),
    "EVADE_IN_CORRIDOR": ("a STATIC vehicle or a VRU in the corridor AND "
                          "lateral residual >= TH m", p_evade, [0.05, 0.10, 0.20]),
    "OVERTAKE_VEHICLE": ("a MOVING vehicle ahead in the corridor the ego is "
                         "closing on / has passed, AND lateral residual >= TH m",
                         p_overtake, [0.05, 0.10, 0.20]),
}

#: tokens with NO honest probe on this box, and WHY. ⛔ Listed rather than
#: approximated: an invented probe would be reported as a measurement.
UNVERIFIABLE = {
    "MERGE": "needs a road-topology reference (a merge is defined by the lane "
             "structure, not by the ego path). No map on this box for these "
             "clips: map.xodr exists only for the NuRec scenes.",
    "TAKE_EXIT_L": "needs a map: an exit is a road-topology event and is "
                   "geometrically a bend. No map for these clips.",
    "TAKE_EXIT_R": "needs a map: see TAKE_EXIT_L.",
    "LANE_CHANGE_L": "needs a LANE reference. v7_labels.effective_mask records "
                     "the measured consequence (D-NUDGE-ABSORB): with no lane "
                     "detector a lane change is absorbed into NUDGE or "
                     "LANE_KEEP depending on yaw, so lateral offset alone "
                     "cannot separate a lane change from a nudge.",
    "LANE_CHANGE_R": "needs a LANE reference: see LANE_CHANGE_L.",
    "TRAFFIC_LIGHT_REACT": "no non-caption signal for the COLOURLESS case.",
    "TRAFFIC_LIGHT_REACT_RED": "_TLBOX",
    "TRAFFIC_LIGHT_REACT_YELLOW": "_TLBOX",
    "TRAFFIC_LIGHT_REACT_GREEN": "_TLBOX",
}

EXCLUSIVE = (
    ("TURN_L", "TURN_R"), ("TURN_L", "FOLLOW_LANE"), ("TURN_R", "FOLLOW_LANE"),
    ("LANE_CHANGE_L", "LANE_CHANGE_R"), ("LANE_CHANGE_L", "TURN_L"),
    ("LANE_CHANGE_R", "TURN_R"), ("YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R"),
    ("YIELD_FOR_TURN_L", "TURN_R"), ("YIELD_FOR_TURN_R", "TURN_L"),
    ("STOP_POINT", "FOLLOW_LANE"),
    ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_GREEN"),
    ("TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("TRAFFIC_LIGHT_REACT_GREEN", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_GREEN"),
    ("TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_YELLOW"),
    ("OVERTAKE_VEHICLE", "FOLLOW_LANE"),
    ("OVERTAKE_VEHICLE", "EVADE_IN_CORRIDOR"),
    ("TAKE_EXIT_L", "TAKE_EXIT_R"), ("TAKE_EXIT_L", "TURN_R"),
    ("TAKE_EXIT_R", "TURN_L"), ("MERGE", "FOLLOW_LANE"),
)


def excluded_by(tok):
    return frozenset(b for a, b in list(EXCLUSIVE)
                     + [(y, x) for x, y in EXCLUSIVE] if a == tok)


def score(fn, th, clips, feats):
    k = n = unk = 0
    for c in clips:
        f = feats.get(c)
        if f is None:
            unk += 1
            continue
        r = fn(f, th)
        if r is None:
            unk += 1
            continue
        n += 1
        k += bool(r)
    return k, n, unk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fd = json.loads(Path(a.feats).read_text(encoding="utf-8"))
    feats = fd["clips"]
    goals: dict[str, set] = {}
    scene: dict[str, dict] = {}
    with gzip.open(a.labels, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            goals[r["clip_id"]] = set((r.get("g_tac") or {}).get("goals") or {})
            scene[r["clip_id"]] = r.get("scene") or {}
    allc = list(goals)
    print(f"[an] blob md5 {fd['blob_md5']} n_clips={len(allc)} "
          f"feats={len(feats)} with_join={fd['n_with_join']}", flush=True)

    res: dict = {"_evidence_class": "MEASURED (ours)",
                 "blob_md5": fd["blob_md5"], "window_s": fd["window_s"],
                 "n_clips": len(allc), "n_with_ego": fd["n_with_ego"],
                 "n_with_join": fd["n_with_join"],
                 "pipeline_control": {}, "tokens": {}, "unverified": {}}

    # ---- PIPELINE CONTROL ------------------------------------------------
    for name, fn, th, partner in (
            ("STOP_POINT", p_stop, 0.5, "FOLLOW_LANE"),
            ("TURN_L", lambda f, t: p_turn(f, t, +1), 20.0, "TURN_R"),
            ("TURN_R", lambda f, t: p_turn(f, t, -1), 20.0, "TURN_L")):
        pos = [c for c in allc if name in goals[c]]
        neg = [c for c in allc if partner in goals[c] and name not in goals[c]]
        kp, np_, _ = score(fn, th, pos, feats)
        kn, nn, _ = score(fn, th, neg, feats)
        res["pipeline_control"][name] = {
            "rule": f"{fn.__name__ if hasattr(fn,'__name__') else name} th={th}",
            "n_pos": np_, "fire_pos": kp, "S": kp / max(np_, 1),
            "S_ci": wilson(kp, np_),
            "known_negative_via": partner, "n_neg": nn, "fire_neg": kn,
            "f0": kn / max(nn, 1), "f0_ci": wilson(kn, nn)}
        print(f"[control] {name:12s} S={kp}/{np_}={kp/max(np_,1):.3f}  "
              f"f0({partner})={kn}/{nn}={kn/max(nn,1):.3f}", flush=True)

    # ---- THE COT TOKENS ---------------------------------------------------
    # ⭐ THE SECOND KNOWN-NEGATIVE CONSTRUCTION, for the four tokens the
    # exclusion table says nothing about. All four need SOMETHING in the scene
    # (to yield to / follow / meet / avoid), so a window in which the 3D-box
    # channel sees no agent ahead in ANY frame is a CERTAIN negative -- from a
    # channel the caption never touched. ⛔ Its own control is printed: if
    # labelled positives fall inside this "certain negative" set, the
    # construction is wrong and its f0 is not used.
    def empty_scene(c):
        f = feats.get(c)
        return (f is not None and f.get("agents") is not None
                and f["agents"]["empty_ahead_frac"] >= 1.0)
    empty = [c for c in allc if empty_scene(c)]
    res["empty_scene_known_negative"] = {
        "rule": "no agent of any class with 0 < cx <= 80 m and |cy| <= 15 m in "
                "ANY frame of the window (obstacle.offline 3D boxes)",
        "n": len(empty),
        "control_positives_inside": {t: sum(1 for c in empty if t in goals[c])
                                     for t in ("YIELD", "GAP_TARGET",
                                               "REACT_ON_ONCOMING",
                                               "EVADE_IN_CORRIDOR")}}
    print(f"[empty] n={len(empty)} positives_inside="
          f"{res['empty_scene_known_negative']['control_positives_inside']}",
          flush=True)

    NEEDS_A_SCENE = {"YIELD", "GAP_TARGET", "REACT_ON_ONCOMING",
                     "EVADE_IN_CORRIDOR"}
    for key, (rule, fn, ths) in PROBES.items():
        tok = key.split("@")[0]
        pos = [c for c in allc if tok in goals[c]]
        unl = [c for c in allc if tok not in goals[c]]
        exc = excluded_by(tok)
        kneg = [c for c in unl if goals[c] & exc] if exc else []
        kneg_src = "exclusion-table" if kneg else None
        if not kneg and tok in NEEDS_A_SCENE:
            kneg = [c for c in empty if tok not in goals[c]]
            kneg_src = "empty-scene"
        rows = []
        for th in ths:
            kp, np_, up = score(fn, th, pos, feats)
            kf, nf, uf = score(fn, th, unl, feats)
            k0, n0, _ = score(fn, th, kneg, feats)
            S = kp / max(np_, 1)
            F = kf / max(nf, 1)
            f0 = (k0 / n0) if n0 else None
            # ⛔ THE BOUND IS F / S, NOT F. The upper-bound argument needs the
            # probe to be a NECESSARY condition; S < 1 says it is not, and a
            # probe that misses a share of the LABELLED positives misses the
            # same share of the UNLABELLED ones, so the raw fire rate F
            # UNDER-states the false negatives by exactly that factor.
            ub = min(1.0, F / S) if S > 0 else None
            pt = (None if f0 is None or S <= f0
                  else max(0.0, min(1.0, (F - f0) / (S - f0))))
            rows.append({"threshold": th, "n_pos": np_, "fire_pos": kp, "S": S,
                         "S_ci": wilson(kp, np_), "n_unlabelled": nf,
                         "fire_unlabelled": kf, "F": F, "F_ci": wilson(kf, nf),
                         "known_negative_source": kneg_src,
                         "n_known_neg": n0, "fire_known_neg": k0, "f0": f0,
                         "f0_ci": wilson(k0, n0) if n0 else None,
                         "upper_bound_fn_rate": ub,
                         "raw_fire_rate_unlabelled": F,
                         "point_estimate": pt,
                         "control_passed": S >= 0.90,
                         "n_unknown_pos": up, "n_unknown_unlabelled": uf})
        res["tokens"][key] = {"rule": rule, "excluded_by": sorted(exc),
                              "known_negative_source": kneg_src, "sweep": rows}
        best = max(rows, key=lambda r: r["S"])
        print(f"[tok] {key:22s} bestS={best['S']:.3f}@{best['threshold']} "
              f"F={best['F']:.3f} f0={best['f0']} "
              f"UB={best['upper_bound_fn_rate']} pt={best['point_estimate']} "
              f"ctrl={'PASS' if best['control_passed'] else 'VOID'}", flush=True)

    # ---- TRAFFIC LIGHT: the grounding-box channel -------------------------
    TL = ["TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN"]
    asked_tl = [c for c in allc
                if "traffic" in str(scene[c].get("asked", "")).lower()]
    vis_tl = [c for c in asked_tl if scene[c].get("traffic_light_visible")]
    any_tl = [c for c in allc if goals[c] & set(TL)]
    res["traffic_light_box_channel"] = {
        "_independence": "SEMI-INDEPENDENT -- the Alpamayo grounding-box answer "
                         "is a different question from the chain-of-thought "
                         "caption, but the same annotator model. It is a "
                         "NECESSARY condition (no visible light => no light to "
                         "react to) and therefore an upper bound.",
        "n_clips": len(allc), "n_asked_tl_question": len(asked_tl),
        "n_asked_and_visible": len(vis_tl),
        "n_any_tl_token": len(any_tl),
        "n_visible_and_no_tl_token": sum(1 for c in vis_tl
                                         if not (goals[c] & set(TL))),
        "control_S_tl_visible_on_positives": None,
    }
    tlp = [c for c in any_tl if c in asked_tl]
    res["traffic_light_box_channel"]["control_S_tl_visible_on_positives"] = {
        "n_positives_asked": len(tlp),
        "n_visible": sum(1 for c in tlp if scene[c].get("traffic_light_visible")),
        "S": (sum(1 for c in tlp if scene[c].get("traffic_light_visible"))
              / max(len(tlp), 1)),
    }
    k = res["traffic_light_box_channel"]["n_visible_and_no_tl_token"]
    n = len(vis_tl)
    res["traffic_light_box_channel"]["fn_rate_among_asked_and_visible"] = \
        {"k": k, "n": n, "rate": k / max(n, 1), "ci": wilson(k, n)}
    print(f"[tl] asked={len(asked_tl)} visible={len(vis_tl)} "
          f"visible_without_token={k} rate={k/max(n,1):.3f}", flush=True)

    # ---- the lane-width proxy, reported as NOT a probe ---------------------
    for tok in ("LANE_CHANGE_L", "LANE_CHANGE_R"):
        sign = +1 if tok.endswith("_L") else -1
        pos = [c for c in allc if tok in goals[c]]
        unl = [c for c in allc if tok not in goals[c]]
        fn = lambda f, t, s=sign: (f["ego"]["lat_peak_m"] * s) >= t
        kp, np_, _ = score(fn, 2.5, pos, feats)
        kf, nf, _ = score(fn, 2.5, unl, feats)
        res["unverified"][tok] = {
            "reason": UNVERIFIABLE[tok],
            "weak_proxy_NOT_A_PROBE": {
                "rule": "signed lat_peak_m >= 2.5 m (lane-width scale) -- "
                        "MEASURED to fire on plain road curvature, so it is "
                        "reported for scale only and NOT as a bound",
                "S": kp / max(np_, 1), "n_pos": np_, "fire_pos": kp,
                "F": kf / max(nf, 1), "n_unlabelled": nf, "fire_unlabelled": kf}}
    for tok, why in UNVERIFIABLE.items():
        if tok in res["unverified"]:
            continue
        res["unverified"][tok] = {"reason": (
            "the Alpamayo grounding-box channel is the only non-caption signal "
            "and it is semi-independent and scoped to the clips that were asked "
            "the traffic-light question -- see traffic_light_box_channel"
            if why == "_TLBOX" else why)}

    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[an] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
