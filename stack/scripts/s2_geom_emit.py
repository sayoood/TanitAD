"""Emit strategic + tactical labels from EGO GEOMETRY, keyed by clip UUID.

⭐ THE CORRECTED LABEL PATH (DataFlyWheel, 2026-08-23). It replaces two defects
that were each measured, not suspected:

 1. ⛔ **The episode-cache join was wrong on 20.5 % of clips** (8/39), because it
    ran through the colliding 16-bit `episode_id_legacy`. This emitter uses the
    CLIP UUID end to end — egomotion, labels and frames all key on it, so the
    entire class of error is gone. (`egomotion_source`, RETRACTION_LOG C140.)

 2. ⛔ **Alpamayo's lateral/lane axes are AT CHANCE** against ego kinematics
    (31.2 % vs 23.9 % shuffled, p=0.335; lane 20.0 % vs 19.5 %, p=0.706) and its
    CoT hallucinates objects (3 correct / 2 wrong on visually checkable claims).
    `tac_str_labels.compose()` derives its lateral tier FROM `alpamayo_lane`.
    Here **geometry decides every kinematic token**, and Alpamayo may only
    REFINE the REASON of a stop that geometry has already found.

⭐ AND IT RECOVERS THE STRATEGIC HORIZON. The 20 s episode cache truncated the
key+8…30 s band to 18.2 %; the provider's egomotion runs 20–140 s, so
**94.5 % of clips carry the FULL 22 s band** (median horizon 37.0 s). The
strategic layer can be supervised on OBSERVED future, not extrapolation.

Every emitted label carries `provenance`, the horizon it was derived over, and
an explicit `abstain` where the horizon is too short — never a silent guess.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tanitad.data import alpamayo_semantics as SEM
from tanitad.data import ego_manoeuvre as EM
from tanitad.data import label_guard as LG
from tanitad.data import egomotion_source as ES
from tanitad.data import tactical_goals as TG
from tanitad.models import v6

HZ = 10.0
STRAT_HI_S = 30.0        # strategic band upper edge (PI definition)
STRAT_MIN_S = 8.0        # below this the strategic band is unobserved
TAC_ACTION_S = 6.0       # a_tac spans the FULL plan rollout (HIERARCHY §4b)
TAC_VOCAB = "v6.1"       # appends TURN_L/TURN_R -- a junction has no lanes
STR_VOCAB = "v6.1"       # appends PREPARE_TURN_L/R
LON_GOAL_VOCAB = "v6.1"  # appends ADAPT_SPEED_FOR_CURVE
DV_BRAKE_MS = 1.5        # drop from the anchor that counts as braking
DV_CRUISE_MS = 1.0       # |dv| below this is holding speed
SCHEMA = "s2-geom-v1"


def _tactical_actions(poses, key, hz=HZ, vocab_version=TAC_VOCAB):
    """Factored LAT x LON over the FULL 0-6 s plan horizon, from geometry.

    ⛔ THE WINDOW WAS WRONG, AND IT MADE EVERY JUNCTION TURN READ `LANE_KEEP`.
    This used a 2 s window inherited from `refb_labels.LABEL_HORIZON`. But 2 s
    is the OPERATIVE band — HIERARCHY_VOCABULARY §4b defines the plan as ONE
    unicycle rollout 0->6 s whose 2-6 s segment is the TACTICAL band. MEASURED
    on `0e56dae2`: the turn begins at **t+2.5 s and reaches +120 deg**, so a
    0-2 s window cannot see it and the ego was labelled `LANE_KEEP` while
    traversing an intersection. Same class as C134 — a label scored on a window
    it is not defined over.

    ⛔ AND THE VOCABULARY COULD NOT SAY "TURN" EITHER. v6.0's lateral actions
    are all lane-relative; `LANE_KEEP` inside a junction is false (there are no
    lanes to keep). v6.1 appends TURN_L/TURN_R for exactly this, so the emitter
    selects v6.1 — and any consumer must size its head to match.
    """
    n = int(round(TAC_ACTION_S * hz))
    if key + n >= len(poses):
        return {"lat": "ABSTAIN", "lon": "ABSTAIN",
                "window_s": [0.0, TAC_ACTION_S], "vocab": vocab_version,
                "reason": "plan horizon truncated"}
    vocab = v6.tactical_lat_actions(vocab_version)
    m = EM.analyse(poses[:key + n + 1], key=key, hz=hz)

    if m.lateral_class.startswith("JUNCTION_TURN") and "TURN_L" in vocab:
        lat = "TURN_L" if m.peak_yaw_deg > 0 else "TURN_R"
    elif m.lateral_class.startswith("JUNCTION_TURN"):
        # v6.0 cannot express it; say so rather than emitting a false LANE_KEEP
        lat = "ABSTAIN"
    elif m.lateral_class == "NUDGE_L":
        lat = "NUDGE_L"
    elif m.lateral_class == "NUDGE_R":
        lat = "NUDGE_R"
    else:
        lat = "LANE_KEEP"

    # ⛔ ENDPOINT-TO-ENDPOINT dv IS THE WRONG STATISTIC OVER A 6 s WINDOW.
    # MEASURED on `01bee851` (PI): the ego runs 5.9 -> 4.7 -> 5.1 m/s, so
    # dv = -0.8 and |dv| < 1.0 read as CRUISE — while the frames show it braking
    # hard through a junction (11.1 m/s four seconds earlier, 1.6 m/s six
    # seconds later). A dip that partially recovers nets out to nothing.
    # ⇒ the DEEPEST point of the window decides whether the ego braked, and the
    # endpoint only decides whether it recovered.
    dv_end = m.v_end - m.v_at_key
    dv_min = m.v_min - m.v_at_key
    braked = dv_min <= -DV_BRAKE_MS
    if m.v_at_key <= 0.5 and m.v_end <= 0.5:
        lon = "HOLD"
    elif m.stop_type != "NONE" or braked:
        # braked and stayed slow => BRAKE_TO; braked and recovered => FOLLOW
        lon = "BRAKE_TO" if dv_end <= -DV_BRAKE_MS / 2 else "FOLLOW"
    elif abs(dv_end) < DV_CRUISE_MS:
        lon = "CRUISE"
    else:
        lon = "FOLLOW" if dv_end < 0 else "CRUISE"
    assert lat in vocab or lat == "ABSTAIN", lat
    # ⭐ ACTIONS CARRY THEIR EXTENT TOO (PI 2026-08-23: "lateral tactical
    # actions have no args"). `within_m` is the arc over which the action
    # runs, so a nudge and a full junction traversal are distinguishable
    # without reading the class string.
    arc_m = round(float(np.sum(np.hypot(
        np.diff(poses[key:key + n + 1, 0]),
        np.diff(poses[key:key + n + 1, 1])))), 1)
    return {"lat": lat, "lon": lon, "window_s": [0.0, TAC_ACTION_S],
            "lat_args": {"within_m": arc_m},
            "lon_args": {"v_at_key_ms": round(m.v_at_key, 2),
                         "v_min_ms": round(m.v_min, 2),
                         "v_end_ms": round(m.v_end, 2)},
            "vocab": vocab_version, "reason": None,
            "lateral_class": m.lateral_class}


def emit_one(clip_id: str, *, sem_row: dict | None = None) -> dict:
    tr = ES.load(clip_id, hz=HZ, max_s=ES.RAW_T0_S + STRAT_HI_S + 5.0)
    key = tr.key_index
    poses = tr.poses
    horizon = tr.horizon_available_s

    sem = SEM.extract(sem_row.get("cot")) if sem_row else None
    stop_reason = sem.stop_reason if sem else None

    # ---- STRATEGIC: derived over the FULL observed band ------------------
    if horizon < STRAT_MIN_S:
        g_str = {"token": "NONE_ABSTAIN", "provenance": "abstain",
                 "reason": f"strategic band unobserved (only {horizon:.1f}s)"}
        man = None
    else:
        # ⛔ THE STRATEGIC LABEL IS DERIVED OVER THE STRATEGIC BAND ONLY.
        # PI correction 2026-08-23: "the turning is happening within the 6 s
        # horizon, so it is no strategic action any more, it's tactical. The
        # strategic action in such a situation should be follow road AFTER the
        # turning, which can be extracted."
        #
        # This previously analysed [t0, t0+30 s] — which CONTAINS the tactical
        # band — so a manoeuvre the plan already executes was promoted to the
        # strategic layer. MEASURED on `0e56dae2`, the three bands disagree
        # completely and only the band you ask decides the answer:
        #     operative 0-2 s : STRAIGHT      (approaching)
        #     tactical  2-6 s : JUNCTION_TURN_L +102.6 deg, R=6.4 m
        #     strategic 8-30 s: NUDGE_R -11.5 deg, R=160.6 m  (follows the road)
        # ⇒ g_str = FOLLOW_MAIN_ROAD, and the turn belongs to g_tac/a_tac.
        #
        # ⭐ AND THIS IS WHAT MAKES `PREPARE_TURN_*` MEAN SOMETHING. Scoped to
        # 8-30 s, a turn token is one the ego has NOT STARTED — a decision to
        # prepare for. A turn already under way at t+2.5 s is not a strategic
        # goal; it is the plan being executed. Same class as C134: a label
        # scored over a window it is not defined over.
        lo = key + int(round(STRAT_MIN_S * HZ))
        hi = key + int(round(min(STRAT_HI_S, horizon) * HZ))
        man = EM.analyse(poses[:hi + 1], key=lo, hz=HZ)
        stops = EM.stop_episodes(poses[lo:hi + 1, 3], HZ)
        tok, why = TG.strategic_from_geometry(
            man,
            stop_onset_s=(STRAT_MIN_S + stops[0][0] / HZ if stops else None),
            turn_onset_s=(STRAT_MIN_S + man.yaw_onset_s
                          if man.yaw_onset_s is not None else None))
        why = f"{why} [band {STRAT_MIN_S:.0f}–{min(STRAT_HI_S, horizon):.0f}s]"
        jc = SEM.junction_corroboration(sem_row.get("cot") if sem_row else None)
        g_str = {"token": tok, "provenance": "geometry", "reason": why,
                 "radius_m": man.turn_radius_m, "peak_yaw_deg": man.peak_yaw_deg,
                 "n_turn_segments": man.n_turn_segments,
                 # PI idea 2026-08-23: a turn VERB together with a JUNCTION
                 # referent is a claim about road TOPOLOGY, which ego poses
                 # cannot see. It RAISES CONFIDENCE and never flips the class
                 # (C138: this same source fabricated an intersection once).
                 "vlm_junction_corroboration": jc.as_dict()}

    # ---- STRATEGIC ACTION -------------------------------------------------
    if man is None:
        a_str = {"token": "ABSTAIN", "provenance": "abstain",
                 "reason": "no observed strategic band"}
    # ⛔ THE LAUNCH CASE MUST BE TESTED FIRST. An ego already AT REST at the
    # anchor has a stop episode in its horizon by definition, so a
    # stop-type-first ordering labelled it PREPARE_STOP while it was
    # accelerating away — the exact inversion `label_guard` G3 exists to catch,
    # produced by the emitter itself. MEASURED: 9 of 801 labels (all of the
    # emitter's REFUSE findings) were this one ordering bug, e.g. `17a04e0f`
    # 0.10 -> 11.14 m/s labelled prepare-to-stop.
    # ⚠️ THE GATE IS IMPORTED FROM THE GUARD, NOT RESTATED. The emitter first
    # used its own 0.5 m/s "at rest" and 5.0 m/s "moving" while `label_guard`
    # used 1.0 and +2.0 m/s — two constants for ONE fact, so five clips landed
    # in the gap between them and the emitter's own output failed its own
    # guard. A shared threshold cannot drift; a copied one always does.
    elif (man.v_at_key < LG.V_REST_MS
          and man.v_end - man.v_at_key > LG.DV_ACCEL_MS):
        a_str = {"token": "RESUME_CRUISE", "provenance": "geometry",
                 "reason": f"launch from rest {man.v_at_key:.2f}->"
                           f"{man.v_end:.2f} m/s (stop_type={man.stop_type})"}
    # ⛔ AN ACTION MUST NOT CONTRADICT ITS OWN GOAL. Found by the PI on
    # `0e56dae2`: g_str=TURN_LEFT with a_str=HOLD_CORRIDOR — "hold the corridor"
    # as the action for LEAVING it. v6.0 had no token for committing to a
    # junction turn, so every turn was forced onto one that denies it. v6.1
    # appends PREPARE_TURN_L/R and the turn case is now tested BEFORE the
    # longitudinal fallbacks, because the route decision outranks a speed
    # description.
    elif (g_str["token"] in ("TURN_LEFT", "TURN_RIGHT")
          and "PREPARE_TURN_L" in v6.strategic_action_tokens(STR_VOCAB)):
        side = "L" if g_str["token"] == "TURN_LEFT" else "R"
        onset_t0 = (round(STRAT_MIN_S + man.yaw_onset_s, 1)
                    if man.yaw_onset_s is not None else None)
        # ⭐ THE ACTION CARRIES ITS DISTANCE AND ITS DEADLINE (PI 2026-08-23):
        # "add the arg prepare turn right in x m and y seconds. This will make
        # clear that this action is NOT AFFECTING THE CURRENT TACTICAL
        # MANOEUVRE." `within_m` / `by_time_s` are the vocabulary's own uniform
        # constraint slots (§2), so a reader — and a head — can tell a turn
        # 60 m away from one being executed now.
        within_m = None
        if onset_t0 is not None:
            j = key + int(round(onset_t0 * HZ))
            if j < len(poses):
                seg = poses[key:j + 1]
                within_m = round(float(np.sum(np.hypot(
                    np.diff(seg[:, 0]), np.diff(seg[:, 1])))), 1)
        a_str = {"token": f"PREPARE_TURN_{side}", "provenance": "geometry",
                 "args": {"within_m": within_m, "by_time_s": onset_t0},
                 "reason": f"a {g_str['token']} still AHEAD — "
                           f"{within_m} m / {onset_t0}s away, not yet begun "
                           f"(R={man.turn_radius_m} m)"}
    elif man.stop_type in ("CONTROLLED", "QUEUE"):
        a_str = {"token": "PREPARE_STOP", "provenance": "geometry",
                 "reason": f"stop_type={man.stop_type}"}
    elif man.v_end - man.v_at_key <= -1.5:
        a_str = {"token": "REDUCE_TO", "provenance": "geometry",
                 "reason": f"dv {man.v_end - man.v_at_key:+.2f} m/s"}
    else:
        # ⭐ "FOLLOW THE ROAD AFTER TURNING" (PI). Over the strategic band the
        # ego is simply following the road — which is exactly what
        # HOLD_CORRIDOR means once the band excludes the tactical turn.
        a_str = {"token": "HOLD_CORRIDOR", "provenance": "geometry",
                 "reason": f"follows the road across the strategic band "
                           f"(no junction manoeuvre, no stop)"}

    gt = TG.derive(poses, key=key, hz=HZ, stop_reason=stop_reason,
               lon_vocab=LON_GOAL_VOCAB)
    out = {
        "schema_version": SCHEMA, "vocab": {"tactical_lat": TAC_VOCAB, "strategic_action": STR_VOCAB,
                  "tactical_lon_goal": LON_GOAL_VOCAB}, "clip_id": clip_id, "t0_s": ES.RAW_T0_S,
        "g_str": g_str, "a_str": a_str,
        "g_tac": gt.as_dict(), "a_tac": _tactical_actions(poses, key),
        "manoeuvre": (man.as_dict() if man else None),
        "horizon": {"available_s": round(horizon, 1),
                    "strategic_band_s": [STRAT_MIN_S, STRAT_HI_S],
                    "band_observable_s": round(
                        float(np.clip(horizon - STRAT_MIN_S, 0.0,
                                      STRAT_HI_S - STRAT_MIN_S)), 1),
                    "recording_span_s": round(tr.span_s, 1)},
        "semantics": (sem.as_dict() if sem else None),
        "disjointness": {"situation_classifier_output_used": False},
        "_provenance": {
            "poses": "provider egomotion, keyed by clip UUID (no legacy id)",
            "lateral": "GEOMETRY ONLY — alpamayo lateral/lane measured at chance",
            "stop_reason": ("vlm-cot refinement of a geometric stop"
                            if stop_reason else "geometry only"),
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-index", required=True)
    ap.add_argument("--alpamayo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    clips = json.loads(Path(a.clip_index).read_text(encoding="utf-8"))["clips"]
    alpa = {}
    if a.alpamayo:
        for line in Path(a.alpamayo).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                alpa[r["clip_id"]] = r

    ids = sorted(clips)
    if a.limit:
        ids = ids[:a.limit]
    rows, failed = [], []
    for i, c in enumerate(ids):
        try:
            rows.append(emit_one(c, sem_row=alpa.get(c)))
        except Exception as ex:                                # noqa: BLE001
            failed.append({"clip_id": c, "error": repr(ex)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ids)}", flush=True)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    g = collections.Counter(r["g_str"]["token"] for r in rows)
    at = collections.Counter(r["a_str"]["token"] for r in rows)
    lat = collections.Counter(r["g_tac"]["lat_token"] for r in rows)
    lon = collections.Counter(r["g_tac"]["lon_token"] for r in rows)
    band = np.array([r["horizon"]["band_observable_s"] for r in rows])
    print(f"[emit] {len(rows)} labels -> {outp}   failed {len(failed)}")
    print(f"[emit] strategic band observable: median {np.median(band):.1f}s / 22s"
          f"  full-band {int((band >= 21.95).sum())}/{len(band)}")
    print(f"[emit] g_str {dict(g)}")
    print(f"[emit] a_str {dict(at)}")
    print(f"[emit] g_tac LAT {dict(lat)}")
    print(f"[emit] g_tac LON {dict(lon)}")
    if failed:
        (outp.parent / "emit_failures.json").write_text(
            json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
