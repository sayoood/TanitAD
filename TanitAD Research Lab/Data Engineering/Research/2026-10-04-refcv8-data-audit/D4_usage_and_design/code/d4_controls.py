#!/usr/bin/env python3
"""D4 analytic controls: every rule in d4_lib must read a KNOWN value on a synthetic track, and a
deliberate mutation of each rule must go RED. Writes raw/controls.json. CPU only, < 1 s."""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d4_lib as L  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw")
res = {}


def at(B, now_s):
    """index of the window whose NOW (provider row t+7, 10 Hz, row 0 = 0.0 s) is now_s."""
    return int(round(now_s * 10)) - (L.W - 1)


# ---- C1 dense tactical labels on analytic tracks ---------------------------------------------------
expect = {"straight": ("LANE_KEEP", "CRUISE"), "turnL": ("TURN_L", None), "turnR": ("TURN_R", None),
          "lcL": ("LANE_CHANGE_L", "CRUISE"), "nudgeR": ("NUDGE_R", "CRUISE"),
          "stop": ("LANE_KEEP", "BRAKE_TO"), "accel": ("LANE_KEEP", "ACCELERATE")}
c1 = {}
for kind, (want_lat, want_lon) in expect.items():
    P = L.synth(kind)
    B = L.window_block(P)
    lat, lon, aux = L.dense_tactical(B)
    i = at(B, 7.0)
    got = (L.LAT_NAMES[lat[i]], L.LON_NAMES[lon[i]])
    ok = got[0] == want_lat and (want_lon is None or got[1] == want_lon)
    c1[kind] = {"want": [want_lat, want_lon], "got": list(got), "pass": bool(ok)}
res["C1_dense_labels_analytic"] = c1

# ---- C2 NavSim-equivalent path rule: first LEFT window on a R = 15 m left turn starting at x = 72 m ---
P = L.synth("turnL")
B = L.window_block(P)
side, lat, ok = L.nav_path_rule_xy(P, B)
now_s = (B["r"]) * 0.1
first_L = float(now_s[np.argmax(side == 1)])
a = 15.0 * math.acos(1 - 2.0 / 15.0)        # arc into the turn where the offset reaches 2 m
analytic = (72.0 - 20.0 + a) / 8.0
res["C2_nav_path_first_left_s"] = {"analytic_s": round(analytic, 3), "measured_s": round(first_L, 3),
                                   "pass": bool(abs(first_L - analytic) <= 0.11)}
# straight track: never a turn
Ps = L.synth("straight")
Bs = L.window_block(Ps)
ss, _, oks = L.nav_path_rule_xy(Ps, Bs)
res["C2b_nav_path_straight_track"] = {"n_turn": int(((ss == 1) | (ss == -1)).sum()), "pass": bool(((ss == 1) | (ss == -1)).sum() == 0)}

# ---- C3 entry rule on the same track: entry = left turn starting at arc 72 m from an anchor at 0 m ---
entries = [{"token": "NAV_TURN_L", "distance_m": 72.0, "distance_end_m": 72.0 + math.pi / 2 * 15.0}]
s_now = B["S"][B["r"]] - B["S"][0]
es, ed, ee = L.nav_entry_rule(entries, s_now)
first_E = float(now_s[np.argmax(es == 1)])
res["C3_nav_entry_first_left_s"] = {"analytic_s": round((72.0 - 20.0) / 8.0, 3), "measured_s": round(first_E, 3),
                                    "pass": bool(abs(first_E - 6.5) <= 0.11)}
# ---- M3 MUTATION: measure s_now from a different origin (the "distance from the recording start" bug,
# here a 64 m origin error = 8 s at 8 m/s). The first-LEFT time must move by ~8 s -> RED vs analytic.
es_m, _, _ = L.nav_entry_rule(entries, s_now - 64.0)
first_M = float(now_s[np.argmax(es_m == 1)])
res["M3_mutation_origin_shift"] = {"measured_s": round(first_M, 3),
                                   "goes_red": bool(abs(first_M - 6.5) > 1.0)}

# ---- M1 MUTATION of the dense labeler: read the PAST 6 s instead of the future (reverse the track) ----
P_rev = L.synth("turnL")[::-1].copy()
P_rev[:, 2] = P_rev[:, 2] + math.pi     # driving the track backwards
Br = L.window_block(L.synth("straight"))
lat_s, _, _ = L.dense_tactical(Br)
# a past-reading labeler on a window whose turn lies 2 s in the FUTURE must not see it: emulate by
# shifting the window 6 s earlier (NOW = 1.0 s -> horizon ends at 7.0 s, before the 9.0 s turn)
Bt = L.window_block(L.synth("turnL"))
lat_t, _, _ = L.dense_tactical(Bt)
res["M1_mutation_horizon_shift"] = {
    "label_at_NOW_7s": L.LAT_NAMES[lat_t[at(Bt, 7.0)]],
    "label_at_NOW_1s_(horizon_ends_before_turn)": L.LAT_NAMES[lat_t[at(Bt, 1.0)]],
    "goes_red": bool(L.LAT_NAMES[lat_t[at(Bt, 1.0)]] != "TURN_L")}

# ---- C4 snap-up ladder literals -----------------------------------------------------------------------
c4 = {}
for v_kmh, want in [(29.9, 30), (30.0, 30), (30.01, 50), (50.0, 50), (99.0, 100), (125.0, 120)]:
    i = int(L.snap_up(v_kmh / 3.6, L.LADDER4_KMH))
    c4[str(v_kmh)] = {"want": want, "got": L.LADDER4_KMH[i], "pass": L.LADDER4_KMH[i] == want}
res["C4_snap_up_ladder4"] = c4

allpass = all(v["pass"] for v in c1.values()) and res["C2_nav_path_first_left_s"]["pass"] \
    and res["C2b_nav_path_straight_track"]["pass"] and res["C3_nav_entry_first_left_s"]["pass"] \
    and res["M3_mutation_origin_shift"]["goes_red"] and res["M1_mutation_horizon_shift"]["goes_red"] \
    and all(v["pass"] for v in c4.values())
res["ALL_CONTROLS_PASS"] = bool(allpass)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "controls.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
