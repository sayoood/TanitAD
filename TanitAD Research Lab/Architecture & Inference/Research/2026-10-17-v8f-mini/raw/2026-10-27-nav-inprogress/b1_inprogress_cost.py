"""b1_inprogress_cost.py -- EXPLORATORY (decides nothing): how much of DAG's T1-kin 6 s error sits on windows whose nav says
"a turn is IN PROGRESS" (TURN token, d_next = 0) -- the case of the PI's screenshot (window 2158, clip ed87040c, t = 11.8 s).

Inputs (read-only): DAG's T1-kin dumps (m = 10, causal bucket; cl = executed path, fl = the floor P10, g = GT) and the audit's per-row
nav table (repro_eval141.npz, rows aligned with join_eval141.npz). CPU only. Estimator: episode-cluster bootstrap, B = 2,000, 141 episodes
(answers "another draw of EPISODES" only)."""
import json
import sys

import numpy as np

HERE = sys.path[0].replace("\\", "/") + "/"
T1 = "E:/Projects/TanitAD-artifacts/v8f-dagger-t1/"
R = dict(np.load(HERE + "repro_eval141.npz", allow_pickle=True))
J = dict(np.load(HERE + "src/raw/2026-10-25-t-attrib/raw_in/join_eval141.npz", allow_pickle=True))
rng = np.random.default_rng(0)


def load(arm):
    rows = []
    for e in range(141):
        z = np.load(f"{T1}dump_{arm}_m10_causal_heads/ep{e:04d}.npz")
        for i in range(len(z["ws"])):
            c = int(z["clip_index"][i])
            rows.append((e, c, int(z["ws"][i]), z["g"][i], z["cl"][i], z["fl"][i]))
    return rows


def heading_change(g):
    d = g[-1] - g[-4]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def boot(vals, eps, B=2000):
    vals, eps = np.asarray(vals), np.asarray(eps)
    ue = np.unique(eps)
    idx = {u: np.where(eps == u)[0] for u in ue}
    m = []
    for _ in range(B):
        s = rng.choice(ue, len(ue))
        k = np.concatenate([idx[u] for u in s])
        m.append(vals[k].mean())
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = {}
for arm in ("DAG_s0", "DAG_s1"):
    rows = load(arm)
    E, CL, FL, CLS, TOK, DEND, DYAW, V = [], [], [], [], [], [], [], []
    for e, c, ws, g, cl, fl in rows:
        r = int(J["off"][c]) + ws
        tok = int(np.argmax(J["nav"][r, :3])) if J["nav"][r, 4] > 0 or J["nav"][r, :3].sum() > 0 else 0
        dnext = float(J["nav_d"][r]) if "nav_d" in J else np.nan
        ade_cl = float(np.linalg.norm(cl - g, axis=1).mean())
        ade_fl = float(np.linalg.norm(fl - g, axis=1).mean())
        inprog = tok in (1, 2) and float(R["nav_d_next_m"][r]) <= 0.5
        cls = "IN_PROGRESS" if inprog else ("TURN_AHEAD" if tok in (1, 2) else "FOLLOW")
        E.append(e); CL.append(ade_cl); FL.append(ade_fl); CLS.append(cls); TOK.append(tok)
        DEND.append(float(R["nav_d_end_m"][r])); DYAW.append(heading_change(g)); V.append(float(R["v_now_ms"][r]))
    E, CL, FL, CLS, TOK, DEND, DYAW = map(np.asarray, (E, CL, FL, CLS, TOK, DEND, DYAW))
    N = len(CL)
    # "stale in-progress": the token's turn sign disagrees with the GT's heading change over the last 0.6 s of the 6 s horizon,
    # or the GT ends heading within +-10 deg of straight (the turn is over well inside the horizon)
    sgn = np.where(TOK == 1, 1.0, np.where(TOK == 2, -1.0, 0.0))
    stale = (CLS == "IN_PROGRESS") & ((np.sign(DYAW) != sgn) | (np.abs(DYAW) < 10.0))
    rec = {"n_windows": int(N), "ade6_dag": boot(CL, E), "ade6_p10": boot(FL, E), "delta_dag_p10_all": boot(CL - FL, E)}
    for name, m in (("FOLLOW", CLS == "FOLLOW"), ("TURN_AHEAD", CLS == "TURN_AHEAD"), ("IN_PROGRESS", CLS == "IN_PROGRESS"),
                    ("IN_PROGRESS_stale", stale), ("IN_PROGRESS_consistent", (CLS == "IN_PROGRESS") & ~stale)):
        if m.sum() < 2:
            rec[name] = {"n": int(m.sum())}
            continue
        d = CL - FL
        rec[name] = {"n": int(m.sum()), "n_episodes": int(len(np.unique(E[m]))), "share_of_windows": float(m.mean()),
                     "ade6_dag": boot(CL[m], E[m]), "ade6_p10": boot(FL[m], E[m]), "delta_dag_minus_p10": boot(d[m], E[m]),
                     # contribution to the all-window mean of (DAG - P10): sum over the class / N
                     "contribution_to_mean_delta_m": float(d[m].sum() / N),
                     "contribution_to_mean_ade6_dag_m": float(CL[m].sum() / N),
                     "nav_d_end_m_median": float(np.nanmedian(DEND[m]))}
    w = [i for i, (e, c, ws, *_r) in enumerate(rows) if c == 126 and ws == 59]
    rec["window_2158"] = {"class": str(CLS[w[0]]), "stale": bool(stale[w[0]]), "ade6_dag": float(CL[w[0]]), "ade6_p10": float(FL[w[0]]),
                          "gt_heading_change_deg": float(DYAW[w[0]]), "nav_d_end_m": float(DEND[w[0]])} if w else None
    out[arm] = rec
json.dump(out, open(HERE + "b1_inprogress_cost.json", "w"), indent=1)
for arm, rec in out.items():
    print(arm, "N", rec["n_windows"], "ADE6 DAG %.3f P10 %.3f" % (rec["ade6_dag"][0], rec["ade6_p10"][0]), "2158:", rec["window_2158"])
    for k in ("FOLLOW", "TURN_AHEAD", "IN_PROGRESS", "IN_PROGRESS_stale", "IN_PROGRESS_consistent"):
        x = rec[k]
        if "ade6_dag" in x:
            print(f"  {k:24s} n {x['n']:5d} ep {x['n_episodes']:3d}  DAG {x['ade6_dag'][0]:6.3f}  P10 {x['ade6_p10'][0]:6.3f}  "
                  f"DAG-P10 {x['delta_dag_minus_p10'][0]:+.3f} [{x['delta_dag_minus_p10'][1]:+.3f}, {x['delta_dag_minus_p10'][2]:+.3f}]  "
                  f"contrib to mean DAG-P10 {x['contribution_to_mean_delta_m']:+.4f} m, to mean ADE6 {x['contribution_to_mean_ade6_dag_m']:.4f} m, d_end med {x['nav_d_end_m_median']:.1f}")
        else:
            print(f"  {k:24s} n {x['n']}")
