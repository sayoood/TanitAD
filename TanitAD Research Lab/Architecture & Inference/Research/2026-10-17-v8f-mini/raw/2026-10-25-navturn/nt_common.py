"""nt_common.py -- shared loaders for the nav/turn audit (EXPLORATORY, CPU only, read-only on every input).

Inputs are the banked eval-141 files exported from the mirror (git show c7d40467) into ./src/.
The builder code (build_v9_labels.py, s2_geom_emit_v7.py, tanitad/) is the mirror's own, imported unmodified.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/"
SRC = HERE + "src/"
TA = SRC + "raw/2026-10-25-t-attrib/"
for p in (SRC + "stack", SRC + "stack/scripts"):
    if p not in sys.path:
        sys.path.insert(0, p)

CONS_NAMES = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
              "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m",
              "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
              "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
CI_ = {n: i for i, n in enumerate(CONS_NAMES)}
LAT_NAMES = {-100: "none", 0: "LANE_KEEP", 1: "TURN_L", 2: "TURN_R", 3: "LC_L", 4: "LC_R"}


def load_inputs():
    J = dict(np.load(TA + "raw_in/join_eval141.npz", allow_pickle=True))
    K = dict(np.load(TA + "raw_in/eval141_kin.npz", allow_pickle=True))
    T = dict(np.load(TA + "table.npz", allow_pickle=True))
    grid = json.load(open(TA + "raw_in/eval_grid_windows.json"))
    return J, K, T, grid


def clip_tracks(K):
    koff = np.concatenate([[0], np.cumsum(K["n"])])
    return [K["poses"][koff[i]:koff[i + 1]].astype(np.float64) for i in range(len(K["n"]))]


def clip_clock(J, c):
    """(g0, dt) of clip c: frame k (raw v2ep row) is at ts(k) = g0 + k*dt on the builder's log clock; recovered from the join's now_s (even k = 2j)."""
    o, n = int(J["off"][c]), int(J["n"][c])
    ns = J["now_s"][o:o + n]
    fin = np.where(np.isfinite(ns))[0]
    dt = float(np.median(np.diff(ns[fin]) / np.diff(fin)) / 2.0)
    g0 = float(ns[fin[0]] - 2 * fin[0] * dt)
    return g0, dt
