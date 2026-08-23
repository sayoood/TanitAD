"""v7 label emitter — ORDINAL layers, multi-label tactical goals, nav commands.

The hierarchy is separated by POSITION IN THE MANOEUVRE SEQUENCE, not by time:

    manoeuvre[0]  -> TACTICAL goal set   (+ anchor args, + lat/lon actions)
    manoeuvre[1]  -> STRATEGIC goal      (+ `_FOLLOW_ROUTE` suffix, + args)
    none          -> FOLLOW_ROUTE / HOLD_MAIN_ROAD

⛔ WHY ORDINAL BEATS TEMPORAL, MEASURED. `01b24287` contains exactly ONE
manoeuvre (+97 deg, t+5.9-11.9 s) which STRADDLES the 6 s/8 s boundary — under a
time split it landed in the strategic band, so a single turn the plan executes
was reported as the route-level decision. `01bee851` contains THREE (+76 deg at
1.0 s, -43 deg at 8.8 s, +69 deg at 28.8 s): its strategic `TURN_RIGHT` was the
right token for the OVERNEXT manoeuvre but carried no args, so nothing said
WHICH of the three it meant.

⚠️ Sequence depth over the corpus: 0 manoeuvres 56.9 %, 1 -> 26.7 %,
2 -> 11.1 %, 3+ -> 5.2 %. Only **16.4 % of clips carry an overnext manoeuvre**,
so the strategic goal is `FOLLOW_ROUTE` on ~84 % — an honest but heavily skewed
target that a consumer must weight.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from tanitad.data import alpamayo_semantics as SEM
from tanitad.data import cot_tokens_v7 as COT
from tanitad.data import ego_manoeuvre as EM
from tanitad.data import egomotion_source as ES
from tanitad.models import vocab_v7 as V7

HZ = 10.0
PLAN_S = 6.0             # the tactical plan rollout (HIERARCHY §4b)
LOOKAHEAD_S = 30.0       # how far ahead a manoeuvre sequence is read
TURN_RATE_DEG_S = 6.0
MIN_SEG_S = 0.8
MIN_TURN_DEG = 20.0
DV_BRAKE_MS = 1.5
DV_CRUISE_MS = 1.0
DV_ACCEL_MS = 1.5
V_STOP_MS = 0.5
SPEED_BAND_TOL_MS = 1.5
CURVE_MAX_R_M = 80.0
SCHEMA = "s2-geom-v7"


def manoeuvre_sequence(poses, key, hz=HZ):
    """Sustained turn segments after ``key``, in TEMPORAL ORDER.

    Returns [(t_start_s, t_end_s, dyaw_deg, radius_m), ...]. This is the
    backbone of the whole v7 design: index 0 is the NEXT manoeuvre and index 1
    the OVERNEXT one.
    """
    p = np.asarray(poses, dtype=np.float64)
    yaw = np.unwrap(p[:, 2])
    v = p[:, 3]
    rel = np.degrees(yaw - yaw[key])
    om = np.gradient(yaw) * hz
    k = max(1, int(round(0.5 * hz)))
    oms = np.convolve(om, np.ones(k) / k, mode="same")
    rate = np.degrees(oms)
    kappa = np.abs(oms) / np.maximum(np.convolve(v, np.ones(k) / k, mode="same"), 1.0)

    act = np.abs(rate[key:]) >= TURN_RATE_DEG_S
    out, i, need = [], 0, int(round(MIN_SEG_S * hz))
    while i < len(act):
        if act[i]:
            j = i
            while j + 1 < len(act) and act[j + 1]:
                j += 1
            if j - i + 1 >= need:
                a, b = key + i, min(key + j, len(rel) - 1)
                d = float(rel[b] - rel[a])
                if abs(d) >= MIN_TURN_DEG:
                    kmax = float(kappa[a:b + 1].max())
                    R = 1.0 / kmax if kmax > 1e-4 else float("inf")
                    out.append((round(i / hz, 1), round((j + 1) / hz, 1),
                                round(d, 1), round(R, 1)))
            i = j + 1
        else:
            i += 1
    return out


def _arc_to(poses, key, t_s, hz=HZ):
    j = min(len(poses) - 1, key + int(round(t_s * hz)))
    seg = poses[key:j + 1]
    if len(seg) < 2:
        return 0.0
    return round(float(np.sum(np.hypot(np.diff(seg[:, 0]),
                                       np.diff(seg[:, 1])))), 1)


def tactical_goals(poses, key, seq, cot, hz=HZ):
    """The tactical goal SET for the NEXT manoeuvre, plus its anchor args."""
    p = np.asarray(poses, dtype=np.float64)
    hi = min(len(p) - 1, key + int(round(PLAN_S * hz)))
    c, s = np.cos(-p[key, 2]), np.sin(-p[key, 2])
    dx, dy = p[:, 0] - p[key, 0], p[:, 1] - p[key, 1]
    ex, ey = c * dx - s * dy, s * dx + c * dy
    anchor = {"goal_x_m": round(float(ex[hi]), 2),
              "goal_y_m": round(float(ey[hi]), 2),
              "t_reach_s": round((hi - key) / hz, 1)}

    v = p[key:hi + 1, 3]
    stops = EM.stop_episodes(v, hz)
    goals: dict[str, dict] = {}

    nxt = seq[0] if seq else None
    # a manoeuvre counts as THE NEXT tactical one if it starts inside the plan
    if nxt and nxt[0] < PLAN_S:
        side = "L" if nxt[2] > 0 else "R"
        held = bool(stops and (stops[0][1] - stops[0][0] + 1) / hz >= 0.5
                    and stops[0][0] / hz < nxt[0])
        tok = f"YIELD_FOR_TURN_{side}" if held else f"TURN_{side}"
        goals[tok] = {"within_m": _arc_to(p, key, nxt[0], hz),
                      "by_time_s": nxt[0], "radius_m": nxt[3],
                      "dyaw_deg": nxt[2]}
        if held:
            goals[f"TURN_{side}"] = {"by_time_s": nxt[0], "radius_m": nxt[3]}
    if stops and not any(k.startswith(("TURN_", "YIELD_FOR_TURN_"))
                         for k in goals):
        i0, i1 = stops[0]
        goals["STOP_POINT"] = {"within_m": _arc_to(p, key, i0 / hz, hz),
                               "hold_for_s": round((i1 - i0 + 1) / hz, 1)}
    # ⭐ SPEED_BAND restored (PI: "the target speed band disappeared"). It is
    # the HELD band across the plan — trivially derivable, and the honest
    # longitudinal answer whenever the ego is neither stopping nor turning.
    if "STOP_POINT" not in goals and not any(
            k.startswith(("TURN_", "YIELD_FOR_TURN_")) for k in goals):
        lo_v, hi_v = float(v.min()), float(v.max())
        # ⛔ THE GOAL AND THE ACTION MUST USE ONE THRESHOLD. SPEED_BAND first
        # allowed a 3.0 m/s spread while the CRUISE action required |dv| < 1.0 —
        # so 103 clips emitted "a held band" alongside ACCELERATE or BRAKE_TO.
        # Two thresholds for one fact is the same defect as the emitter/guard
        # constants earlier today. The band now requires the speed to be held by
        # the SAME measure the action uses.
        held = abs(float(v[-1]) - float(v[0])) < DV_CRUISE_MS
        if held and hi_v - lo_v <= 2 * SPEED_BAND_TOL_MS and hi_v > V_STOP_MS:
            goals["SPEED_BAND"] = {"v_lo_ms": round(max(0.0, lo_v), 2),
                                   "v_hi_ms": round(hi_v, 2)}
    # ⛔ PERCEPTION TOKENS COME ONLY FROM THE CoT, NEVER FROM GEOMETRY.
    for t, a in COT.goals_from_cot(cot).items():
        goals.setdefault(t, {**a, "provenance": "vlm-cot", "disputed": True})
    if not goals:
        goals["FOLLOW_LANE"] = {}
    # ⚠️ the exclusion matrix is CHECKED, not assumed
    viol = V7.validate_goal_set(goals)
    return goals, anchor, viol


def tactical_actions(poses, key, seq, hz=HZ):
    n = int(round(PLAN_S * hz))
    if key + n >= len(poses):
        return {"lat": "LANE_KEEP", "lon": "CRUISE", "truncated": True,
                "lat_args": {}, "lon_args": {}}
    m = EM.analyse(poses[:key + n + 1], key=key, hz=hz)
    lat = ("TURN_L" if m.lateral_class == "JUNCTION_TURN_L" else
           "TURN_R" if m.lateral_class == "JUNCTION_TURN_R" else
           "NUDGE_L" if m.lateral_class == "NUDGE_L" else
           "NUDGE_R" if m.lateral_class == "NUDGE_R" else "LANE_KEEP")

    dv_end, dv_min = m.v_end - m.v_at_key, m.v_min - m.v_at_key
    turning = lat.startswith("TURN_") or (m.turn_radius_m <= CURVE_MAX_R_M
                                          and abs(m.peak_yaw_deg) >= MIN_TURN_DEG)
    if m.v_at_key <= V_STOP_MS and m.v_end <= V_STOP_MS:
        lon = "HOLD"
    elif m.stop_type != "NONE" and m.v_end <= V_STOP_MS:
        lon = "BRAKE_TO"
    elif turning:
        # ⭐ the longitudinal partner of an arc — universal over junction turns
        # and plain bends (PI: keep the CURVE name, not FOR_TURNING)
        lon = "ADAPT_SPEED_FOR_CURVE"
    elif dv_end >= DV_ACCEL_MS:
        lon = "ACCELERATE"          # ⭐ had no token; was filed as CRUISE
    elif dv_min <= -DV_BRAKE_MS:
        lon = "BRAKE_TO" if dv_end <= -DV_BRAKE_MS / 2 else "FOLLOW"
    elif abs(dv_end) < DV_CRUISE_MS:
        lon = "CRUISE"
    else:
        lon = "FOLLOW" if dv_end < 0 else "ACCELERATE"

    assert lat in V7.TACTICAL_LAT_ACTIONS_V7 and lon in V7.TACTICAL_LON_ACTIONS_V7
    return {"lat": lat, "lon": lon, "truncated": False,
            "lat_args": {"within_m": _arc_to(poses, key, PLAN_S, hz)},
            "lon_args": {"v_target_ms": round(m.v_min, 2),
                         "within_m": _arc_to(poses, key, PLAN_S, hz)}}


def strategic(poses, key, seq, hz=HZ):
    """The OVERNEXT manoeuvre, suffixed `_FOLLOW_ROUTE`, with args."""
    inplan = [s for s in seq if s[0] < PLAN_S]
    rest = [s for s in seq if s[0] >= PLAN_S]
    over = rest[0] if rest else None
    if over is None:
        # ⭐ WHAT `REDUCE_TO_FOLLOW_ROUTE` MEANS (PI asked, and it had NO
        # definition and 0 emissions in the first v7 run). With the ordinal
        # design, PREPARE_TURN/PREPARE_STOP already cover "slow for the overnext
        # manoeuvre". The one distinct thing left for a strategic REDUCE_TO is a
        # SUSTAINED speed drop that no manoeuvre explains — the ego entering a
        # slower road class (town, zone) and STAYING slower. That is a
        # route-level fact, so it earns its place; anything else is tactical.
        p = np.asarray(poses, dtype=np.float64)
        hi_i = min(len(p) - 1, key + int(round(LOOKAHEAD_S * hz)))
        early = p[key:key + int(round(6.0 * hz)) + 1, 3]
        late = p[max(key, hi_i - int(round(10.0 * hz))):hi_i + 1, 3]
        sustained = (len(early) > 5 and len(late) > 5
                     and float(late.mean()) <= float(early.mean()) - 3.0
                     and float(late.max()) <= float(early.mean()) - 1.0)
        act = ({"token": "REDUCE_TO_FOLLOW_ROUTE",
                "args": {"v_target_ms": round(float(late.mean()), 2)},
                "provenance": "geometry",
                "reason": f"sustained drop {float(early.mean()):.1f} -> "
                          f"{float(late.mean()):.1f} m/s with no manoeuvre — "
                          f"a slower road class, not a manoeuvre"}
               if sustained else
               {"token": "HOLD_MAIN_ROAD", "args": {},
                "provenance": "geometry", "reason": "nothing to prepare for"})
        return ({"token": "FOLLOW_ROUTE", "args": {},
                 "provenance": "geometry",
                 "reason": "no overnext manoeuvre in the lookahead"}, act)
    side = "LEFT" if over[2] > 0 else "RIGHT"
    args = {"within_m": _arc_to(poses, key, over[0], hz), "by_time_s": over[0]}
    g = {"token": f"TURN_{side}_FOLLOW_ROUTE", "args": args,
         "provenance": "geometry",
         "reason": f"overnext manoeuvre {over[2]:+.0f} deg at t+{over[0]}s "
                   f"(R={over[3]} m); {len(inplan)} in-plan before it"}
    a = {"token": f"PREPARE_TURN_{side[0]}_FOLLOW_ROUTE", "args": args,
         "provenance": "geometry",
         "reason": f"{args['within_m']} m / {over[0]}s ahead, not yet begun"}
    return g, a


def nav_command(poses, key, seq, hz=HZ):
    """⛔ MODEL INPUT derived from the EGO FUTURE — training only.

    See `vocab_v7.NAV_PROVENANCE`. On PhysicalAI the only route supplier is the
    recorded future, so this is ORACLE information: admissible as a training
    input, never as evidence in a vision-only eval arm.
    """
    nxt = seq[0] if seq else None
    if nxt is None:
        return {"token": "NAV_FOLLOW_ROAD", "args": {},
                "provenance": "ego-future", "oracle": True}
    side = "L" if nxt[2] > 0 else "R"
    return {"token": f"NAV_TURN_{side}",
            "args": {"distance_m": _arc_to(poses, key, nxt[0], hz),
                     "time_s": nxt[0]},
            "provenance": "ego-future", "oracle": True}


def emit_one(clip_id: str, *, sem_row=None) -> dict:
    tr = ES.load(clip_id, hz=HZ, max_s=ES.RAW_T0_S + LOOKAHEAD_S + 5.0)
    key, poses = tr.key_index, tr.poses
    cot = sem_row.get("cot") if sem_row else None
    seq = manoeuvre_sequence(poses, key, HZ)
    goals, anchor, viol = tactical_goals(poses, key, seq, cot, HZ)
    g_str, a_str = strategic(poses, key, seq, HZ)
    return {
        "schema_version": SCHEMA, "clip_id": clip_id, "t0_s": ES.RAW_T0_S,
        "vocab": "v7",
        "manoeuvre_sequence": [{"t_start_s": a, "t_end_s": b,
                                "dyaw_deg": d, "radius_m": r}
                               for a, b, d, r in seq],
        "g_str": g_str, "a_str": a_str,
        "g_tac": {"goals": goals, "anchor": anchor, "violations": viol},
        "a_tac": {**(_at := tactical_actions(poses, key, seq, HZ)),
                  # ⭐ PI: link goals to their admissible actions. An action
                  # serving NONE of its goals is the TURN_LEFT+HOLD_CORRIDOR
                  # shape — recorded per clip so it cannot hide.
                  "serves_goals": V7.action_serves_goals(
                      _at["lat"], _at["lon"], goals)},
        "nav_command": nav_command(poses, key, seq, HZ),
        "semantics": (SEM.extract(cot).as_dict() if cot else None),
        "cot_tokens": COT.extract(cot).as_dict() if cot else None,
        "horizon": {"available_s": round(tr.horizon_available_s, 1),
                    "recording_span_s": round(tr.span_s, 1)},
        "_provenance": {
            "layers": "ORDINAL: manoeuvre[0]=tactical, manoeuvre[1]=strategic",
            "kinematics": "ego geometry only",
            "perception_tokens": "vlm-cot, disputed, never geometry-derived",
            "nav_command": "ORACLE (ego-future) — training input only",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-index", required=True)
    ap.add_argument("--alpamayo")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clips = json.loads(Path(a.clip_index).read_text(encoding="utf-8"))["clips"]
    alpa = {}
    if a.alpamayo:
        for line in Path(a.alpamayo).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                alpa[r["clip_id"]] = r
    rows, failed = [], []
    for i, c in enumerate(sorted(clips)):
        try:
            rows.append(emit_one(c, sem_row=alpa.get(c)))
        except Exception as ex:                              # noqa: BLE001
            failed.append({"clip_id": c, "error": repr(ex)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(clips)}", flush=True)
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    gs = collections.Counter(r["g_str"]["token"] for r in rows)
    ast = collections.Counter(r["a_str"]["token"] for r in rows)
    gt = collections.Counter(t for r in rows for t in r["g_tac"]["goals"])
    la = collections.Counter(r["a_tac"]["lat"] for r in rows)
    lo = collections.Counter(r["a_tac"]["lon"] for r in rows)
    nav = collections.Counter(r["nav_command"]["token"] for r in rows)
    nviol = sum(1 for r in rows if r["g_tac"]["violations"])
    print(f"[v7] {len(rows)} labels -> {outp}  failed {len(failed)}")
    print(f"[v7] goal-set violations: {nviol}")
    print(f"[v7] g_str {dict(gs)}")
    print(f"[v7] a_str {dict(ast)}")
    print(f"[v7] g_tac (multi-label) {dict(gt)}")
    print(f"[v7] a_tac LAT {dict(la)}")
    print(f"[v7] a_tac LON {dict(lo)}")
    print(f"[v7] nav {dict(nav)}")
    if failed:
        (outp.parent / "emit_v7_failures.json").write_text(
            json.dumps(failed, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
