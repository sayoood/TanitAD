"""D-NAVROUTE-1 -- the disagreement contingency table, ZERO GPU.

Reads the banked refcv4b navflip decisions dump and asks, on the windows where a
candidate "route head" and the commanded nav token DISAGREE, which one the emitted
plan actually tracks.

Two candidate heads, deliberately kept apart (they are different objects):
  R1  route_pred_nav_true   -- the CORE route head argmax (3-way categorical).
                               The arm's own config records graft_route=false.
  R2  gstr_nav_true         -- the CASCADE strategic goal (unit bearing + dist_pref).

ASCII-only output (cp1252-safe). Estimator: paired episode-cluster bootstrap.
"""
import glob
import json
import os
import sys

import numpy as np

DUMP = os.environ.get("DECDIR", "/home/nvidia/navpred/navflip_dump/decisions")
NBOOT = int(os.environ.get("NBOOT", "2000"))
SEED = int(os.environ.get("SEED", "0"))

L, S, R = 0, 1, 2
NAMES = {L: "LEFT", S: "STRAIGHT", R: "RIGHT"}

# NAV_COMMANDS = ("follow", "left", "right", "straight")
NAV_TO_DIR = {0: S, 1: L, 2: R, 3: S}
# N_ROUTE = 3 -> (left, straight, right)
ROUTE_TO_DIR = {0: L, 1: S, 2: R}


def dir_from_y(y, tau):
    out = np.full(y.shape, S, dtype=np.int64)
    out[y > tau] = L
    out[y < -tau] = R
    return out


def load():
    eps = []
    for f in sorted(glob.glob(os.path.join(DUMP, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        eps.append({k: z[k] for k in z.files})
    return eps


def build(eps, tau_y, tau_g):
    rows = []
    for ei, z in enumerate(eps):
        n = int(z["nav_cmd"].shape[0])
        plan_y = z["plan_full_nav_true"][:, -1, 1].astype(np.float64)
        gt_y = z["gt_future_ext"][:, -1, 1].astype(np.float64)
        gt_ok = z["gt_future_valid_ext"][:, -1].astype(bool)
        rows.append({
            "ep": np.full(n, ei, dtype=np.int64),
            "nav": np.array([NAV_TO_DIR[int(v)] for v in z["nav_cmd"]]),
            "nav_valid": z["nav_valid"].astype(bool),
            "r1": np.array([ROUTE_TO_DIR[int(v)]
                            for v in z["route_pred_nav_true"]]),
            "r2": dir_from_y(z["gstr_nav_true"][:, 1].astype(np.float64), tau_g),
            "plan": dir_from_y(plan_y, tau_y),
            "plan_y": plan_y,
            "gt_y": gt_y,
            "gt_ok": gt_ok,
            "gt": dir_from_y(gt_y, tau_y),
        })
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def boot_gap(ep, a, b, nboot, seed):
    """Paired episode-cluster bootstrap of mean(a) - mean(b) over episodes."""
    rng = np.random.default_rng(seed)
    ueps = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in ueps}
    obs = float(a.mean() - b.mean()) if a.size else float("nan")
    draws = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(ueps, size=ueps.size, replace=True)
        sel = np.concatenate([idx[e] for e in pick])
        draws[i] = a[sel].mean() - b[sel].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"point": round(obs, 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_windows": int(a.size), "n_episodes": int(ueps.size)}


def table(D, head_key, tau_y, tau_g, nboot, seed):
    v = D["nav_valid"]
    head, nav, plan = D[head_key][v], D["nav"][v], D["plan"][v]
    ep = D["ep"][v]
    # DISAGREEMENT: head and command name different directions, >=1 non-straight
    dis = (head != nav) & ((head != S) | (nav != S))
    h, c, p, e = head[dis], nav[dis], plan[dis], ep[dis]
    m_head = (p == h).astype(np.float64)
    m_cmd = (p == c).astype(np.float64)
    neither = int(((p != h) & (p != c)).sum())
    out = {
        "head": head_key,
        "tau_y": tau_y, "tau_g": tau_g,
        "n_windows_nav_valid": int(v.sum()),
        "n_agree": int((~dis & ((head != S) | (nav != S))).sum()),
        "n_disagreement": int(dis.sum()),
        "n_episodes_disagreement": int(np.unique(e).size),
        "counts": {"plan_eq_head": int(m_head.sum()),
                   "plan_eq_command": int(m_cmd.sum()),
                   "plan_eq_neither": neither},
        "rate_plan_eq_head": round(float(m_head.mean()), 4) if m_head.size else None,
        "rate_plan_eq_command": round(float(m_cmd.mean()), 4) if m_cmd.size else None,
    }
    if m_head.size:
        out["follow_gap_head_minus_command"] = boot_gap(e, m_head, m_cmd,
                                                        nboot, seed)
    # the joint census, so nothing is hidden behind a scalar
    cen = {}
    for hh in (L, S, R):
        for cc in (L, S, R):
            k = f"head={NAMES[hh]}|cmd={NAMES[cc]}"
            sel = (head == hh) & (nav == cc)
            if sel.sum():
                pp = plan[sel]
                cen[k] = {"n": int(sel.sum()),
                          "plan_LEFT": int((pp == L).sum()),
                          "plan_STRAIGHT": int((pp == S).sum()),
                          "plan_RIGHT": int((pp == R).sum())}
    out["census_head_x_command"] = cen
    return out


def main():
    eps = load()
    res = {"tool": "D-NAVROUTE-1 contingency (zero GPU)",
           "dump": DUMP, "n_episodes": len(eps),
           "tier": "T1 (self-action OPEN loop, 2026-09-02 ruling)",
           "estimator": "paired episode-cluster bootstrap, "
                        f"{NBOOT} resamples, seed {SEED}",
           "variance_answered": "EPISODE DRAW ONLY -- not training-run, not "
                                "inference-run variance"}
    tau_y, tau_g = 1.0, 0.10
    D = build(eps, tau_y, tau_g)
    res["n_windows_total"] = int(D["ep"].size)

    # ---- CONTROL THAT MUST READ A KNOWN VALUE -----------------------------
    # Orientation: if +y is LEFT in the ego frame, then on windows commanded
    # LEFT the GT terminal y must be positive far more often than on windows
    # commanded RIGHT. If this control does not separate, every direction in
    # this file is meaningless and the run is INADMISSIBLE.
    v = D["nav_valid"] & D["gt_ok"]
    gl = D["gt_y"][v & (D["nav"] == L)]
    gr = D["gt_y"][v & (D["nav"] == R)]
    res["ORIENTATION_CONTROL"] = {
        "rule": "+y must be LEFT: mean GT terminal y under nav=LEFT must exceed "
                "that under nav=RIGHT",
        "n_left": int(gl.size), "n_right": int(gr.size),
        "mean_gt_y_navLEFT": round(float(gl.mean()), 4) if gl.size else None,
        "mean_gt_y_navRIGHT": round(float(gr.mean()), 4) if gr.size else None,
        "PASS": bool(gl.size and gr.size and gl.mean() > gr.mean()),
    }
    # base rates, so a "match" is readable against chance
    vv = D["nav_valid"]
    res["base_rates"] = {
        "plan": {NAMES[k]: int((D["plan"][vv] == k).sum()) for k in (L, S, R)},
        "nav": {NAMES[k]: int((D["nav"][vv] == k).sum()) for k in (L, S, R)},
        "r1_core_route_head": {NAMES[k]: int((D["r1"][vv] == k).sum())
                               for k in (L, S, R)},
        "r2_gstr": {NAMES[k]: int((D["r2"][vv] == k).sum()) for k in (L, S, R)},
        "gt": {NAMES[k]: int((D["gt"][vv] == k).sum()) for k in (L, S, R)},
    }
    res["primary"] = {"R1_core_route_head": table(D, "r1", tau_y, tau_g,
                                                  NBOOT, SEED),
                      "R2_gstr": table(D, "r2", tau_y, tau_g, NBOOT, SEED)}
    # ---- SENSITIVITY (pre-registered, not tuned) --------------------------
    sens = []
    for ty in (0.5, 1.0, 2.0):
        for tg in (0.05, 0.10, 0.20):
            Dx = build(eps, ty, tg)
            for hk in ("r1", "r2"):
                t = table(Dx, hk, ty, tg, 400, SEED)
                g = t.get("follow_gap_head_minus_command") or {}
                sens.append({"tau_y": ty, "tau_g": tg, "head": hk,
                             "n_dis": t["n_disagreement"],
                             "rate_head": t["rate_plan_eq_head"],
                             "rate_cmd": t["rate_plan_eq_command"],
                             "gap": g.get("point"),
                             "sep": g.get("separated")})
    res["sensitivity"] = sens
    print(json.dumps(res, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
