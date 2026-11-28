"""plan_lead_read.py : note PREREG_V8F_PROOF_ADD39_N1. Does the lead explain any of the plan's speed error at the first tick of the lead windows?
T0 (the plan of the first tick, made from the logged state on logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).
Inputs: the pulled T1-kin tick and episode dumps of LS_s0 and LS_s1, the label join of the headway read, the lead head's per-frame read (LEADREAD_<ARM>.npz).
  python plan_lead_read.py --out-dir OUT [--B 2000]"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "F:/TanitAD-scratch/v8proof/headway")
import headway_read as HR  # noqa: E402

PULL = "C:/Users/Admin/v8proof_c/pull"
HERE = os.path.dirname(os.path.abspath(__file__))
LAMS = (0.01, 0.1, 1.0, 10.0, 100.0)
DT = 0.2


def arc(tr, k):
    """tr [N,30,2] positions at steps 1..30 from the start (0,0) -> path length up to step k."""
    p = np.concatenate([np.zeros((tr.shape[0], 1, 2)), tr[:, :k]], axis=1)
    return np.sqrt((np.diff(p, axis=1) ** 2).sum(-1)).sum(1)


def load_arm(arm):
    d = os.path.join(PULL, arm)
    tag = [x for x in os.listdir(d) if x.startswith("dump_")][0][5:]
    z = np.load(os.path.join(d, "ticks", "ticks_%s.npz" % tag), allow_pickle=True)
    i0 = np.where(z["tick"] == 0)[0]
    order = np.argsort(z["win"][i0]); i0 = i0[order]
    ci, ws = z["ci"][i0].astype(np.int64), z["t0"][i0].astype(np.int64)
    eps = sorted(f for f in os.listdir(os.path.join(d, "dump_" + tag)) if f.endswith(".npz"))
    ws_d = np.concatenate([np.load(os.path.join(d, "dump_" + tag, f))["ws"] for f in eps])
    ci_d = np.concatenate([np.load(os.path.join(d, "dump_" + tag, f))["clip_index"] for f in eps])
    same = bool(len(ws_d) == len(ws) and np.array_equal(ws_d, ws) and np.array_equal(ci_d, ci))
    return {"arm": arm, "plan": z["plan"][i0], "gt": z["gt"][i0], "gt_ok": z["gt_ok"][i0], "v0": z["v0"][i0].astype(np.float64), "ci": ci, "ws": ws,
            "clip_names": [str(c) for c in z["clip_names"]], "ticks_match_episode_dumps": same, "n_windows": int(len(i0))}


def lead_read(arm, A, shift=0):
    L = np.load(os.path.join(HERE, "LEADREAD_%s.npz" % arm), allow_pickle=True)
    name_to = {str(c): i for i, c in enumerate(L["clips"])}
    key = {}
    for r, (c, t) in enumerate(zip(L["clip_idx"], L["t"])):
        key[(int(c), int(t))] = r
    rows = np.array([key.get((name_to.get(A["clip_names"][c], -1), int(w) + shift), -1) for c, w in zip(A["ci"], A["ws"])])
    ok = rows >= 0
    out = np.full((len(rows), 3), np.nan)
    out[ok, 0] = L["presence_logit"][rows[ok]]; out[ok, 1] = L["gap_m"][rows[ok]]; out[ok, 2] = L["closing_ms"][rows[ok]]
    return out


def ridge_fit(X, y, lam):
    G = X.T @ X
    return np.linalg.solve(G + lam * (np.trace(G) / X.shape[1] + 1e-12) * np.eye(X.shape[1]), X.T @ y)


def oof_pred(X, y, ep):
    fold = ep % 5
    pred = np.zeros_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte, ym = (X[tr] - mu) / sd, (X[te] - mu) / sd, y[tr].mean()
        inner = (ep[tr] // 5) % 4
        best, bl = None, None
        for lam in LAMS:
            e = 0.0
            for g in range(4):
                a, b = inner != g, inner == g
                if b.sum() == 0 or a.sum() < X.shape[1] + 2:
                    continue
                w = ridge_fit(Xtr[a], y[tr][a] - y[tr][a].mean(), lam)
                e += float((((Xtr[b] @ w) + y[tr][a].mean() - y[tr][b]) ** 2).sum())
            if best is None or e < best:
                best, bl = e, lam
        pred[te] = Xte @ ridge_fit(Xtr, y[tr] - ym, bl) + ym
    return pred


def feats(A, J, vshift=0, perm=None, plant=None, rng=None):
    Lw = HR.label_windows(J, A["ci"], A["ws"])
    lead = np.asarray(Lw["lead"], bool)
    gap0 = Lw["G"][:, 0]
    m = np.minimum(5, A["ws"])
    rows_p = J["off"][A["ci"]] + A["ws"] - m
    gp = J["lead_gap"][rows_p].astype(np.float64); vp = (J["lead_valid"][rows_p] == 1) & np.isfinite(gp) & (m > 0)
    clos = np.where(vp, (gp - gap0) / np.maximum(m, 1) / DT, 0.0)
    v0 = A["v0"]
    B = np.stack([v0, v0 ** 2], 1)
    Lf = np.stack([gap0, clos, (~vp).astype(float), gap0 / np.maximum(v0, 1.0), clos * v0, gap0 * v0], 1)
    V = lead_read(A["arm"], A, vshift)
    vnan = ~np.isfinite(V).all(1)
    Vf = np.concatenate([np.where(np.isfinite(V), V, 0.0), vnan[:, None].astype(float)], 1)
    return lead, gap0, B, Lf, Vf, V, vnan


def run_seed(A, J, target_k, rng, perm=False, plant=False):
    lead, gap0, B, Lf, Vf, V, vnan = feats(A, J)
    y_all = arc(A["plan"], target_k) - arc(A["gt"], target_k)
    use = lead & A["gt_ok"] & np.isfinite(y_all)
    y, ep = y_all[use], A["ci"][use]
    B, Lf, Vf = B[use], Lf[use], Vf[use]
    if perm:
        p = rng.permutation(len(y)); Lf = Lf[p]; Vf = Vf[p]
    if plant:
        Lf = np.concatenate([Lf, (y + rng.normal(0, y.std(), len(y)))[:, None]], 1)
    eB = (y - oof_pred(B, y, ep)) ** 2
    eL = (y - oof_pred(np.concatenate([B, Lf], 1), y, ep)) ** 2
    eV = (y - oof_pred(np.concatenate([B, Vf], 1), y, ep)) ** 2
    st = (y - y.mean()) ** 2
    return {"ep": ep, "dL": eB - eL, "dV": eB - eV, "st": st, "eB": eB, "y": y, "n": int(use.sum()), "n_ep": int(len(np.unique(ep))), "n_lead": int(lead.sum()),
            "share_v_nan": float(vnan[use].mean())}


def two_level(recs, key, B, seed=0):
    rng = np.random.default_rng(seed)
    S = len(recs)
    per = []
    for r in recs:
        eps = np.unique(r["ep"])
        num = np.array([r[key][r["ep"] == e].sum() for e in eps]); den = np.array([r["st"][r["ep"] == e].sum() for e in eps])
        per.append((num, den))
    pts = [float(n.sum() / d.sum()) for n, d in per]
    boot = np.empty(B)
    for b in range(B):
        vals = []
        for s in rng.integers(0, S, S):
            n, d = per[s]; i = rng.integers(0, len(n), len(n))
            vals.append(n[i].sum() / d[i].sum())
        boot[b] = np.mean(vals)
    return {"per_seed": pts, "pooled": float(np.mean(pts)), "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def above0(c):
    return bool(all(p > 0 for p in c["per_seed"]) and c["ci"][0] > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    J = HR.load_join()
    arms = [load_arm("LS_s0"), load_arm("LS_s1")]
    res = {"note": "PREREG_V8F_PROOF_ADD39_N1", "tier": "T0 (first tick of each window, logged state, logged frames; a diagnostic, never driving performance)",
           "strategic": "STRATEGIC: PAUSED (PI item 25)", "families": "LONGITUDINAL: the plan's along-track error against the logged driver; LATERAL, TACTICAL: not measured by this read",
           "B": a.B, "bootstrap": "two-level (seeds, then episodes), rng seed 0", "arms": ["LS_s0", "LS_s1"]}
    out = {}
    for name, k in (("2.0s", 10), ("4.0s", 20)):
        recs = [run_seed(A, J, k, np.random.default_rng(10 + i)) for i, A in enumerate(arms)]
        out[name] = {"GL": two_level(recs, "dL", a.B), "GV": two_level(recs, "dV", a.B),
                     "R2_B": [float(1 - r["eB"].sum() / r["st"].sum()) for r in recs], "y_mean_m": [float(r["y"].mean()) for r in recs], "y_std_m": [float(r["y"].std()) for r in recs],
                     "n": [r["n"] for r in recs], "n_ep": [r["n_ep"] for r in recs], "n_lead": [r["n_lead"] for r in recs], "share_v_missing": [r["share_v_nan"] for r in recs]}
    res["stats"] = out
    # controls
    rp = [run_seed(A, J, 10, np.random.default_rng(100 + i), perm=True) for i, A in enumerate(arms)]
    d1 = {"GL": two_level(rp, "dL", a.B), "GV": two_level(rp, "dV", a.B)}
    d1["pass"] = bool(not above0(d1["GL"]) and not above0(d1["GV"]))
    rk = [run_seed(A, J, 10, np.random.default_rng(200 + i), plant=True) for i, A in enumerate(arms)]
    d2 = {"GL": two_level(rk, "dL", a.B)}; d2["pass"] = above0(d2["GL"])
    d3 = {"n_lead": out["2.0s"]["n_lead"], "n_used": out["2.0s"]["n"], "n_ep": out["2.0s"]["n_ep"], "expected": [642, 56], "ticks_match_episode_dumps": [A["ticks_match_episode_dumps"] for A in arms]}
    d3["pass"] = bool(all(n == 642 for n in d3["n_lead"]) and all(n == 642 for n in d3["n_used"]) and all(e == 56 for e in d3["n_ep"]) and all(d3["ticks_match_episode_dumps"]))
    d4 = {"corr": []}
    for A in arms:
        lead, gap0, *_ = feats(A, J)
        cs = {}
        for sh in (-5, 0, 5):
            V = lead_read(A["arm"], A, sh); ok = lead & np.isfinite(V[:, 1])
            cs[str(sh)] = float(np.corrcoef(V[ok, 1], gap0[ok])[0, 1]) if ok.sum() > 10 else float("nan")
        d4["corr"].append(cs)
    d4["pass"] = bool(all(c["0"] > 0.3 and c["0"] > c["-5"] and c["0"] > c["5"] for c in d4["corr"]))
    # d4b (note PREREG_V8F_PROOF_ADD39_N2): fixed windows, the head read at the start frame against the label gap k frames later
    d4b = {"k": [0, 2, 4, 5, 6, 8, 10], "corr": [], "n": []}
    for A in arms:
        lead, gap0, *_ = feats(A, J)
        G = HR.label_windows(J, A["ci"], A["ws"])["G"]
        V = lead_read(A["arm"], A, 0)
        ok = lead & np.isfinite(V[:, 1]) & np.isfinite(G[:, d4b["k"]]).all(1)
        d4b["corr"].append([float(np.corrcoef(V[ok, 1], G[ok, k])[0, 1]) for k in d4b["k"]]); d4b["n"].append(int(ok.sum()))
    d4b["pass"] = bool(all(c[0] > 0.3 and c[0] == max(c) for c in d4b["corr"]))
    d5 = {"share_within_0.5_mps": []}
    for A in arms:
        v1 = np.sqrt((A["gt"][:, 0] ** 2).sum(-1)) / DT
        ok = A["gt_ok"]
        d5["share_within_0.5_mps"].append(float((np.abs(v1[ok] - A["v0"][ok]) <= 0.5).mean()))
    d5["pass"] = bool(all(s >= 0.95 for s in d5["share_within_0.5_mps"]))
    res["controls"] = {"d1_permuted": d1, "d2_planted": d2, "d3_counts": d3, "d4_frame_alignment_as_written": d4, "d4b_note_N2": d4b, "d5_one_frame": d5}
    res["controls_pass_as_written"] = bool(all(c["pass"] for k, c in res["controls"].items() if k != "d4b_note_N2"))
    res["controls_pass"] = bool(all(c["pass"] for k, c in res["controls"].items() if k != "d4_frame_alignment_as_written"))
    res["control_form"] = "d4 as written: %s; names under d4b of note PREREG_V8F_PROOF_ADD39_N2 (written after the first run, values seen)" % ("PASS" if d4["pass"] else "FAIL")
    s = out["2.0s"]
    if not res["controls_pass"]:
        res["name"] = None
    elif above0(s["GL"]) and above0(s["GV"]):
        res["name"] = "PLAN-ERROR-LEAD-READABLE"
    elif above0(s["GL"]):
        res["name"] = "PLAN-ERROR-LEAD-NOT-READ"
    else:
        res["name"] = "PLAN-ERROR-NOT-LEAD"
    json.dump(res, open(os.path.join(a.out_dir, "RESULT_PLANLEAD.json"), "w"), indent=1)
    f = lambda c: "per seed %s, pooled %+.4f [%+.4f, %+.4f]" % (["%+.4f" % x for x in c["per_seed"]], c["pooled"], c["ci"][0], c["ci"][1])
    L = ["# VERDICT_PLANLEAD (note PREREG_V8F_PROOF_ADD39_N1)", "", "Tier: " + res["tier"] + ". " + res["strategic"] + ". " + res["families"] + ".", "", "## Controls (first)", ""]
    for k, c in res["controls"].items():
        L.append("* %s: %s %s" % (k, "PASS" if c["pass"] else "FAIL", json.dumps({x: y for x, y in c.items() if x != "pass"})[:420]))
    L += ["", "## Name", "", str(res["name"]), ""]
    for name in ("2.0s", "4.0s"):
        s = out[name]
        L += ["## Target at %s (plan minus logged along-track position, m)" % name, "",
              "* windows %s, episodes %s; mean of y %s m, spread %s m; R2 of the ego-speed base %s" % (s["n"], s["n_ep"], ["%.3f" % x for x in s["y_mean_m"]], ["%.3f" % x for x in s["y_std_m"]], ["%.3f" % x for x in s["R2_B"]]),
              "* GL (label lead added to the base): " + f(s["GL"]), "* GV (the lead head's camera read added to the base): " + f(s["GV"]), ""]
    L.append("Limits: linear fits on a few inputs; the first tick only; two seeds, one corpus.")
    open(os.path.join(a.out_dir, "VERDICT_PLANLEAD.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("plan_lead_read: controls_pass %s; name %s; GL 2.0s %s; GV 2.0s %s" % (res["controls_pass"], res["name"], f(out["2.0s"]["GL"]), f(out["2.0s"]["GV"])))


if __name__ == "__main__":
    main()
