"""Shared helpers for the T-attribution scripts (EXPLORATORY, CPU only). Reads table.npz written by s1_build_table.py."""
import os

import numpy as np

S = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, "/") + "/"
CONS_NAMES = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
              "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m",
              "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
              "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
GROUPS = {"latcons": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "lonspeed": [12, 13, 14, 15, 22, 23, 24, 25],
          "stoplead": [16, 17, 18, 19, 20, 21]}
M2B_FAMILY = ["M2b_s0", "M2b_s1", "grid8x20_s0", "hist10_s0", "pgt_s0", "wc_s0", "df50_s0"]
LAT_NAMES = {-100: "none", 0: "LANE_KEEP", 1: "TURN_L", 2: "TURN_R", 3: "LC_L", 4: "LC_R"}
LON_NAMES = {-100: "none", 0: "HOLD", 1: "CREEP", 2: "STOP", 3: "FOLLOW", 4: "DECELERATE", 5: "ACCELERATE", 6: "KEEP"}


def load():
    T = dict(np.load(S + "table.npz", allow_pickle=True))
    return T


def resid(T, arm):
    """T's constraint error in SD units over PRESENT cells (NaN elsewhere): (T_cons_std - GT_std). Absent cells have no GT, so no error (R63)."""
    return T[f"T_{arm}"] - T["gt_std"]


def sq(T, arm):
    r = resid(T, arm)
    return np.where(np.isfinite(r), r ** 2, 0.0), np.isfinite(r)


def huber(r, delta=1.0):
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r * r, delta * (a - 0.5 * delta))


def band(x, edges, labels):
    idx = np.digitize(x, edges)
    return np.array([labels[i] for i in idx])


def episode_cluster_ci(values_by_window, eid, n_boot=2000, seed=0, stat=np.mean):
    """percentile CI of stat over episodes (cluster bootstrap resampling EPISODES; windows within an episode move together)."""
    rng = np.random.default_rng(seed)
    ue = np.unique(eid)
    grp = [np.where(eid == e)[0] for e in ue]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(ue), len(ue))
        idx = np.concatenate([grp[k] for k in pick])
        out[b] = stat(values_by_window[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
