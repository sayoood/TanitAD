"""PREREG_V8F_TFAR design measurement (labels only, zero training, CPU).

Question: on the eval-141 grid, where does S's 120 m point carry information that T does not already read,
and do T's own tactical labels depend on it there?  T reads the tactical nav token (side only while
d <= max(30 m, 6 s x v)) and S's 30/50/80 m checkpoints.  The 120 m point adds the far side.

Inputs (pinned by sha256 below): the CKPT120 grid truth (side, d_text, token_on, rc_ok at 30..150 m, v, has)
and the eval join (lat_cls, lon_cls).  Output: TFAR_LABELS.json.

Run:  python -I measure_tfar_labels.py --out TFAR_LABELS.json   (about 2 s)
"""
import argparse, hashlib, json
import numpy as np

TRUTH = "F:/TanitAD-scratch/strat1/read/ckpt120/inputs/eval141_grid_truth.npz"
JOIN = "F:/TanitAD-scratch/strat1/read/inputs/join_eval141.npz"
RC120 = "F:/TanitAD-scratch/strat1/read/ckpt120/inputs/rc120_table.npz"
PINS = {TRUTH: "da9529a1f9bd4599ec44bbaff2827e12c033f314a8ff9167e94a66dbfcdccc73",
        RC120: "0f122570be5c0b9a60641ea9938102deb212752f5a67c411049e9dd116e01bf2"}
LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
COMPLETE_M = 20.0


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def dist(x, names):
    out = {}
    for i, n in enumerate(names):
        out[n] = int((x == i).sum())
    out["IGNORE_or_other"] = int(((x < 0) | (x >= len(names))).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p, s in PINS.items():
        got = sha(p)
        if got != s:
            raise SystemExit(f"pin mismatch {p}: {got}")
    T = np.load(TRUTH)
    J = np.load(JOIN)
    rows, eid, side, d, on, ok, v, arcs, has = (T[k] for k in ("rows", "eid", "side", "d_text", "token_on", "rc_ok", "v_now", "arcs", "has"))
    k120 = int(np.where(np.isclose(arcs, 120.0))[0][0])
    lat = J["lat_cls"][rows].astype(int)
    lon = J["lon_cls"][rows].astype(int)
    rep = {"inputs": {TRUTH: sha(TRUTH), JOIN: sha(JOIN)}, "n_grid": int(len(rows)), "n_episodes": int(len(np.unique(eid)))}
    dd = np.where(np.isfinite(d), d, 1e9)
    scor = (side != 0) & (dd + COMPLETE_M <= 120.0) & ok[:, k120] & has
    rep["scorable_120"] = {"n": int(scor.sum()), "episodes": int(len(np.unique(eid[scor])))}
    # control: must reproduce the CKPT120 card's 341 / 27 and 219 / 122
    rep["CONTROL_K_LABEL"] = {"n": int(scor.sum()), "token_on": int((scor & on).sum()), "token_off": int((scor & ~on).sum()),
                              "expect": [341, 219, 122]}
    for nm, m in (("token_on", scor & on), ("token_off", scor & ~on)):
        blk = {"n": int(m.sum()), "episodes": int(len(np.unique(eid[m]))),
               "lat": dist(lat[m], LAT9), "lon": dist(lon[m], LON9)}
        # side agreement of T's own lat label with the announced (far) side: +1 = left, -1 = right
        left = m & (side > 0)
        right = m & (side < 0)
        lat_side = np.zeros_like(lat)
        lat_side[np.isin(lat, (1, 3))] = 1
        lat_side[np.isin(lat, (2, 4))] = -1
        blk["lat_label_names_a_side"] = int((m & (lat_side != 0)).sum())
        blk["lat_label_side_equals_far_side"] = int((m & (lat_side != 0) & (lat_side == side)).sum())
        blk["lat_label_side_opposite_far_side"] = int((m & (lat_side != 0) & (lat_side == -side)).sum())
        blk["turn_label_on_far_side"] = int((m & np.isin(lat, (1, 2)) & (lat_side == side)).sum())
        blk["lanechange_label_on_far_side"] = int((m & np.isin(lat, (3, 4)) & (lat_side == side)).sum())
        blk["decelerate_or_stop_label"] = int((m & np.isin(lon, (2, 4))).sum())
        blk["left_right"] = [int(left.sum()), int(right.sum())]
        rep[nm] = blk
    # all RC-A120-valid windows (route setting basis), split by whether the 120 m far stratum exists
    m_all = ok[:, k120] & has
    rep["rc120_valid_all"] = {"n": int(m_all.sum()), "episodes": int(len(np.unique(eid[m_all]))),
                              "lat": dist(lat[m_all], LAT9), "lon": dist(lon[m_all], LON9)}
    # far-turn stratum where T has no side: the ceiling of a planted T that knows the far side, as a share of all windows
    m_f = scor & ~on
    rep["ceiling_T_side_gain_windows"] = {
        "windows_where_far_side_could_change_T_lat_label": rep["token_off"]["lat_label_side_equals_far_side"],
        "of_scorable_token_off": int(m_f.sum()),
        "of_grid": int(len(rows)),
        "share_of_grid": round(rep["token_off"]["lat_label_side_equals_far_side"] / len(rows), 5),
    }
    # geometry: where the road bends between 80 and 120 m (only the far point carries it), vs the longitudinal label
    R = np.load(RC120)
    ok120 = R["eval_rc120_ok"][rows].astype(bool)
    rep["CONTROL_RC120_ok_identity"] = int((ok120 != ok[:, k120]).sum())          # must read 0
    psi120 = np.degrees(np.abs(R["eval_rc120"][rows, 2].astype(np.float64)))
    psi80 = np.degrees(np.abs(J["rc"][rows, 2, 2].astype(np.float64)))
    ok80 = J["rc_ok"][rows, 2].astype(bool)
    base = has & ok120 & ok80
    dec = np.isin(lon, (2, 4))
    geo = {}
    for nm, m in (("far_bend_only (|psi120|>=25, |psi80|<25)", base & (psi120 >= 25) & (psi80 < 25)),
                  ("bend_by_80 (|psi80|>=25)", base & (psi80 >= 25)),
                  ("no_bend (|psi120|<25)", base & (psi120 < 25))):
        geo[nm] = {"n": int(m.sum()), "episodes": int(len(np.unique(eid[m]))), "decelerate_or_stop": int((m & dec).sum()),
                   "share": round(float((m & dec).sum()) / max(int(m.sum()), 1), 4), "token_on": int((m & on).sum()),
                   "lat": dist(lat[m], LAT9)}
    rep["geometry_strata_rc_valid"] = geo
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
