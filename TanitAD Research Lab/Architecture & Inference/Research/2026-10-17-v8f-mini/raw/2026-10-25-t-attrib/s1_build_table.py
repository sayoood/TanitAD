"""s1_build_table.py -- per-window table for the T-constraint attribution (EXPLORATORY, CPU only, read-only on every input).

Builds, on the eval-141 grid (2,399 windows, 141 episodes):
  * GT constraint block in the SD units T is trained and scored in: gt_std[N,26] (NaN where the label is absent),
    from join_eval141.npz row (off[ci] + t - a_t_eff) -- the row the evaluator's `lab_t` reads (v8f_mini_data.py:494-516, ages 'cycle',
    v8f_mini_eval.py:290-297: a_t = i % 3, clamped to min(room, max_age_t=2), room = max(t - window_start, 0), window_start = hist-1 = 3).
  * mu/sd of the standardiser, RECOVERED from the banked GT-oracle code (tacorc-all-goal tac_code) and VERIFIED to reproduce it.
  * T's own constraint block per arm: tacabs tac_code[:, 34:60] (T's std output; absent slots zeroed by the absmask eval) and
    the full block (present + absent) from the base eval's raw `cons_pred` -> (cons_pred - mu)/sd. Both are checked equal on present cells.
  * strata variables: GT lat/lon class, T-tick ego speed, 6 s heading change of the GT path, lead present, episode.
Output: table.npz + this script's stdout (s1_build_table.out).
"""
import json
import os
import sys

import numpy as np

S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/t_attrib/"
L = "C:/Users/Admin/box2_ship/logs/"
C0, NC = 22 + 5 + 7, 26
CONS_NAMES = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
              "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m",
              "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
              "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
GROUPS = {"latcons": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], "lonspeed": [12, 13, 14, 15, 22, 23, 24, 25],
          "stoplead": [16, 17, 18, 19, 20, 21]}

# arm -> (eval dir holding the arm's base + tacabs files, file tag)
ARMS = {
    "M2b_s0": ("v8f_cz", "M2b_s0"), "M2b_s1": ("v8f_cz", "M2b_s1"),
    "grid8x20_s0": ("v8f_grid", "M2b-8x20_s0"), "hist10_s0": ("v8f_hist", "M2b-h10_s0"),
    "pgt_s0": ("v8f_pgt", "M2b-pgt_s0"), "wc_s0": ("v8f_wc", "M2b-wc_s0"), "df50_s0": ("v8f_dscale", "M2b-df50_s0"),
    "M1_s0": ("v8f_cz", "M1_s0"), "M1_s1": ("v8f_cz", "M1_s1"),
}
BASE_DIR = {"M2b_s0": "v8f_mini", "M2b_s1": "v8f_mini", "M1_s0": "v8f_mini", "M1_s1": "v8f_mini", "grid8x20_s0": "v8f_grid",
            "hist10_s0": "v8f_hist", "pgt_s0": "v8f_pgt", "wc_s0": "v8f_wc", "df50_s0": "v8f_dscale"}


def main():
    J = dict(np.load(S + "raw_in/join_eval141.npz", allow_pickle=True))
    K = dict(np.load(S + "raw_in/eval141_kin.npz", allow_pickle=True))
    grid = json.load(open(S + "raw_in/eval_grid_windows.json"))
    order = grid["order"]
    assert J["clip"].tolist() == order == K["clip"].tolist()
    ci, tt = [], []
    for k, c in enumerate(order):
        for t in grid["windows"][c]:
            ci.append(k); tt.append(int(t))
    ci, tt = np.array(ci), np.array(tt)
    N = len(ci)
    assert N == 2399
    i = np.arange(N)
    window_start, max_age_t = 3, 2                                       # hist 4 -> window_start = hist-1 = 3; cfg.max_age_t default 2
    room = np.maximum(tt - window_start, 0)
    a_t = np.minimum(np.minimum(i % 3, room), max_age_t)
    rowT = J["off"][ci] + tt - a_t
    rowN = J["off"][ci] + tt
    print(f"N windows {N}, episodes {len(np.unique(ci))}, a_t counts {np.bincount(a_t)}")

    # ---- GT oracle code (std units) from the banked evaluator, and the join rows -> mu/sd recovery
    zg = np.load(L + "v8f_tacgrp2/eval/eval_M2b_s0-tacorc-all-goal.npz")
    gcode = zg["tac_code"][:, C0:C0 + NC].astype(np.float64)
    fb = np.asarray(zg["tac_oracle_fallback"], bool)
    has = J["has"][rowT]
    assert np.array_equal(fb, ~has), "oracle fallback != ~has[rowT]: row mapping is wrong"
    print(f"row mapping check 1: tac_oracle_fallback == ~has[rowT] on all {N} windows: OK ({int(fb.sum())} windows have no v9 row)")
    raw = J["cons"][rowT].astype(np.float64)
    mu, sd = np.zeros(NC), np.ones(NC)
    maxdev = 0.0
    for c in range(NC):
        m = np.isfinite(raw[:, c]) & has
        if m.sum() < 3:
            continue
        A = np.vstack([raw[m, c], np.ones(m.sum())]).T
        sl, ic = np.linalg.lstsq(A, gcode[m, c], rcond=None)[0]
        sd[c] = 1.0 / sl
        mu[c] = -ic / sl
        maxdev = max(maxdev, float(np.abs(gcode[m, c] - (raw[m, c] - mu[c]) / sd[c]).max()))
    gt_std = np.where(np.isfinite(raw) & has[:, None], (raw - mu) / sd, np.nan)
    chk = np.where(np.isfinite(gt_std), gt_std, 0.0)
    print(f"row mapping check 2 (the {int(has.sum())} windows WITH a v9 row; the {int((~has).sum())} fallback windows carry T's own code in the "
          f"oracle and are excluded from every T-error number): max |GT oracle tac_code - (join cons - mu)/sd| = "
          f"{np.abs(chk[has] - gcode[has]).max():.2e} (per-column recovery residual max {maxdev:.2e}); absent-cell zero fill matches: "
          f"{bool(np.all(gcode[has][~np.isfinite(gt_std[has])] == 0.0))}")
    print("recovered standardiser (mu, sd) per cell:")
    for c in range(NC):
        n = int(np.isfinite(gt_std[:, c]).sum())
        print(f"  {c:2d} {CONS_NAMES[c]:18s} mu {mu[c]:10.4f} sd {sd[c]:9.4f}  present n={n} ({n / N:.1%})")

    # ---- T's own constraint blocks per arm
    out = {"gt_std": gt_std, "mu": mu, "sd": sd, "ci": ci, "tt": tt, "a_t": a_t, "rowT": rowT, "has": has}
    for arm, (d, tag) in ARMS.items():
        pa = L + f"{d}/eval/eval_{tag}-tacabs.npz"
        pb = L + f"{BASE_DIR[arm]}/eval/eval_{tag}.npz"
        za, zb = np.load(pa), np.load(pb)
        assert np.array_equal(za["eid"], ci), f"{arm}: tacabs eid != grid"
        assert np.array_equal(zb["eid"], ci), f"{arm}: base eid != grid"
        t_abs = za["tac_code"][:, C0:C0 + NC].astype(np.float64)                     # std units, absent -> 0
        t_full = (zb["cons_pred"].astype(np.float64) - mu) / sd                      # std units, every slot
        pres = np.isfinite(gt_std)
        dev = float(np.abs(t_full[pres] - t_abs[pres]).max())
        print(f"{arm:12s} base-eval cons_pred (raw, standardised with recovered mu/sd) vs tacabs tac_code on present cells: max|diff| {dev:.2e}"
              f"   ade6 mean {float(zb['ade6'].mean()):.4f}")
        out[f"T_{arm}"] = t_full
        out[f"ade6_{arm}"] = np.asarray(zb["ade6"], np.float64)
    # ---- GT-code ADE6 per window (tacorc) per arm -- the T->GT gap
    gtade = {"M2b_s0": ("v8f_tacgrp2", "M2b_s0"), "M2b_s1": ("v8f_tacgrp2", "M2b_s1"), "grid8x20_s0": ("v8f_grid", "M2b-8x20_s0"),
             "hist10_s0": ("v8f_hist", "M2b-h10_s0"), "pgt_s0": ("v8f_pgt", "M2b-pgt_s0"), "wc_s0": ("v8f_wc", "M2b-wc_s0"),
             "df50_s0": ("v8f_dscale", "M2b-df50_s0"), "M1_s0": ("v8f_tacgrp2", "M1_s0"), "M1_s1": ("v8f_tacgrp2", "M1_s1")}
    for arm, (d, tag) in gtade.items():
        z = np.load(L + f"{d}/eval/eval_{tag}-tacorc.npz")
        assert np.array_equal(z["eid"], ci)
        out[f"ade6orc_{arm}"] = np.asarray(z["ade6"], np.float64)
        out[f"orcfb_{arm}"] = np.asarray(z["tac_oracle_fallback"], bool)
        print(f"{arm:12s} GT-code (tacorc) ade6 mean {float(z['ade6'].mean()):.4f}  own-code ade6 mean {float(out['ade6_' + arm].mean()):.4f}")

    # ---- strata variables
    poses = K["poses"].astype(np.float64)
    koff = np.concatenate([[0], np.cumsum(K["n"])])
    v_T, v_N, dpsi6, kappa6, tN_arr = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    gt6_x, gt6_y = np.zeros(N), np.zeros(N)
    for k in range(len(order)):
        P = poses[koff[k]:koff[k + 1]]
        G = P[0::2]                                                                  # the 0.2 s pose grid (v8f_mini_data.window_kin)
        m = ci == k
        tT = tt[m] - a_t[m]
        v_T[m] = P[2 * tT, 3]
        v_N[m] = P[2 * tt[m], 3]
        end = np.minimum(tT + 30, len(G) - 1)
        dy = G[end, 2] - G[tT, 2]
        dy = (dy + np.pi) % (2 * np.pi) - np.pi
        dpsi6[m] = np.degrees(dy)
        arc = np.hypot(G[end, 0] - G[tT, 0], G[end, 1] - G[tT, 1])
        kappa6[m] = np.abs(dy) / np.maximum(arc, 1.0)
        # GT 6 s end point in the NOW frame (for reference)
    out.update(v_T=v_T, v_N=v_N, dpsi6=dpsi6, kappa6=kappa6)
    out["lat_cls"] = J["lat_cls"][rowT]
    out["lon_cls"] = J["lon_cls"][rowT]
    out["lat_allowed"] = J["lat_allowed"][rowT]
    out["lon_allowed"] = J["lon_allowed"][rowT]
    out["lead_valid"] = J["lead_valid"][rowT]
    out["lead_gap"] = J["lead_gap"][rowT]
    out["nav_tok"] = J["nav"][rowN][:, :3].argmax(1)
    # label temporal stability: |gt_std[t] - gt_std[t+1 cache step]| for cells present at both (a label-side noise probe)
    nxt = np.minimum(rowT + 1, J["off"][ci] + J["n"][ci] - 1)
    raw_next = np.where(J["has"][nxt][:, None], J["cons"][nxt].astype(np.float64), np.nan)
    gt_next = np.where(np.isfinite(raw_next), (raw_next - mu) / sd, np.nan)
    out["gt_std_next"] = gt_next
    print(f"strata vars: v_T [{v_T.min():.2f},{v_T.max():.2f}] median {np.median(v_T):.2f}; dpsi6 deg abs median {np.median(np.abs(dpsi6)):.2f}; "
          f"lead_valid frac {out['lead_valid'].mean():.3f}; lat_cls values {np.unique(out['lat_cls'])}; lon_cls {np.unique(out['lon_cls'])}")
    np.savez_compressed(S + "table.npz", **out)
    print("wrote table.npz keys:", len(out))


if __name__ == "__main__":
    sys.exit(main())
