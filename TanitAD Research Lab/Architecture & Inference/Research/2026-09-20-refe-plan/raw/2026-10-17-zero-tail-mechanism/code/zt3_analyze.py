"""zt3_analyze.py -- PREREG_ZT3 Q1-Q7 + gates + committed readings -> RESULT.json (driverl-eval venv).

    python zt3_analyze.py --inroot <scratch in/> --out ../RESULT.json

Every share is a ratio estimator over arrays indexed by the 12,146 navtest tokens, bootstrapped with the landed
`zero_tail.Boot` (136 logs, 10,000 resamples, seed 20260927); differences / enrichments reuse the SAME draws.
Intervals answer "another draw of LOGS" for one checkpoint only.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zero_tail as zt  # noqa: E402
import zt3_lib as L  # noqa: E402

PK = HERE.parent
DATA = PK / "data"
INPUTS_EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
HUMAN_CSV = "D:/Projects/TanitAD/data/refe_navtest/score/refe_p4_human_navtest/refe_p4_human_navtest.csv"
TRACKED = "C:/Users/Admin/tmpauxv/tracked"
TJ = {"A_P": 0, "A_S": 1, "B_P": 2, "B_S": 3, "H": 4}
MATRIX = {  # PREREG section 4 Q5 (P_RD = partial only if ROUTE_DIFF, else N)
    "P6_lane_rank_twins": {"S_STOP": "N", "WRONG_MANOEUVRE": "Y", "CORNER_CUT": "P_RD", "WIDE_TURN": "P_RD",
                           "STRAIGHT_DRIFT": "P_RD", "RUN_LONG": "N", "ON_PATH": "N", "TRACKER_MADE": "N"},
    "P6_trunk_DAC_gradient": {"S_STOP": "P", "WRONG_MANOEUVRE": "Y", "CORNER_CUT": "Y", "WIDE_TURN": "Y",
                              "STRAIGHT_DRIFT": "Y", "RUN_LONG": "Y", "ON_PATH": "Y", "TRACKER_MADE": "P"},
    "P7_1_map_supervision": {"S_STOP": "N", "WRONG_MANOEUVRE": "P", "CORNER_CUT": "Y", "WIDE_TURN": "Y",
                             "STRAIGHT_DRIFT": "Y", "RUN_LONG": "Y", "ON_PATH": "P", "TRACKER_MADE": "N"},
    "proposal_side": {"S_STOP": "Y", "WRONG_MANOEUVRE": "P", "CORNER_CUT": "P", "WIDE_TURN": "P",
                      "STRAIGHT_DRIFT": "P", "RUN_LONG": "P", "ON_PATH": "P", "TRACKER_MADE": "Y"},
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Est:
    """ratio estimators over navtest-indexed arrays, one set of log draws (zero_tail.Boot)."""

    def __init__(self, log_idx, n_logs):
        self.boot = zt.Boot(log_idx, n_logs)
        self.li, self.nl = log_idx, n_logs

    def vec(self, num, den):
        s_n = np.bincount(self.li, weights=np.asarray(num, float), minlength=self.nl)
        s_d = np.bincount(self.li, weights=np.asarray(den, float), minlength=self.nl)
        d = self.boot.draws
        with np.errstate(divide="ignore", invalid="ignore"):
            return s_n.sum() / s_d.sum(), s_n[d].sum(1) / s_d[d].sum(1)

    @staticmethod
    def ci(point, v, scale=1.0):
        v = v[np.isfinite(v)] * scale
        return {"point": round(float(point) * scale, 4), "lo": round(float(np.percentile(v, 2.5)), 4),
                "hi": round(float(np.percentile(v, 97.5)), 4)}

    def share(self, ind, base, scale=100.0):
        p, v = self.vec(np.asarray(ind) & np.asarray(base), base)
        return {"n": int((np.asarray(ind) & np.asarray(base)).sum()), "of": int(np.asarray(base).sum()), **self.ci(p, v, scale)}

    def mean(self, x, base):
        x = np.where(base, np.nan_to_num(x), 0.0)
        p, v = self.vec(x, base)
        return self.ci(p, v)

    def ratio_of(self, numA, denA, numB, denB):
        pa, va = self.vec(numA, denA)
        pb, vb = self.vec(numB, denB)
        return self.ci(pa / pb, va / vb)

    def diff_of(self, numA, denA, numB, denB):
        pa, va = self.vec(numA, denA)
        pb, vb = self.vec(numB, denB)
        return self.ci(pa - pb, va - vb)


def load_geom():
    parts = sorted((DATA / "geom").glob("geom_*of*.npz"))
    G = [np.load(p) for p in parts if "_" not in p.stem[len("geom_"):].replace("of", "")]
    keys = [k for k in G[0].files]
    cat = {k: np.concatenate([g[k] for g in G]) for k in keys}
    lanes = {}
    for p in sorted((DATA / "geom").glob("geom_*of*_lanes.json.xz")):
        with lzma.open(p, "rt", encoding="utf-8") as f:
            lanes.update(json.load(f))
    return cat, lanes, [str(p) for p in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R_in = Path(a.inroot)
    J = np.load(DATA / "jobs.npz")
    G, LANES, gparts = load_geom()
    row2g = {int(r): i for i, r in enumerate(G["row"])}
    assert len(row2g) == len(J["token"]), f"geometry covers {len(row2g)} of {len(J['token'])} job rows"
    A = np.load(R_in / "c3/table.npz")
    B = np.load(R_in / "p2/p2_own_table/table.npz")
    c3 = json.load(open(R_in / "c3/tokens.json", encoding="utf-8"))
    nt = json.load(open(R_in / "navtest_full_tokens.json", encoding="utf-8"))
    NT = list(nt["tokens"])
    N = len(NT)
    nidx = {t: i for i, t in enumerate(NT)}
    logs = sorted(set(nt["token_log"].values()))
    lidx = np.array([logs.index(nt["token_log"][t]) for t in NT])
    E = Est(lidx, len(logs))
    hum = np.load("D:/Projects/TanitAD/data/refe_navtest/seams/refe_p4_human_navtest_all.npz")
    hpos = {str(t): i for i, t in enumerate(hum["token"])}
    dpsi_H_all = np.array([float(L.wrap_deg(np.degrees(hum["poses"][hpos[t], 7, 2]))) for t in NT])
    exp = json.load(gzip.open(INPUTS_EXPORT, "rt", encoding="utf-8"))["tokens"]
    v0_all = np.array([float(np.hypot(*exp[t]["ego_statuses"][-1]["ego_velocity"])) for t in NT])
    cmd_all = np.array([int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in NT])
    jt = [str(t) for t in J["token"]]
    jrow = {t: r for r, t in enumerate(jt)}

    def g(t, key, traj):
        return G[key][row2g[jrow[t]], TJ[traj]]

    # ------------------------------------------------------------------ gates
    gates = {"G0": json.loads(str(J["meta"]))["gates"]}
    hv, ed = J["have"], J["exp_dac"]
    dac_me = np.zeros_like(ed)
    for r in range(len(jt)):
        dac_me[r] = (~G["nd_sim"][row2g[r]].any(-1)).astype(float)
    chk = hv & (ed >= 0)
    gates["G1_all_scored_trajectories"] = f"{int((dac_me[chk] == ed[chk]).sum())}/{int(chk.sum())}"
    hcsv = {}
    for line in open(HUMAN_CSV, encoding="utf-8").read().splitlines()[1:]:
        f = line.split(",")
        hcsv[f[1]] = float(f[4])
    hm = np.array([dac_me[jrow[t], 4] for t in jt])
    hc = np.array([hcsv[t] for t in jt])
    gates["G1_H_vs_harness_csv"] = f"{int((hm == hc).sum())}/{len(jt)}"
    gates["G3_human_DAC1"] = f"{int((hm == 1).sum())}/{len(jt)}"
    gates["G3_human_DAC0_tokens"] = [jt[i] for i in np.nonzero(hm != 1)[0]][:20]

    # ------------------------------------------------------------------ per arm
    arms = {}
    per_token_A = {}
    for X, T, rule in (("A", A, "navsim_v1_lane"), ("B", B, "navsim_v1")):
        toks = [str(t) for t in T["token"]]
        pdms, sub, lg, pick = T["pdms"], T["sub"], T["logits"].astype(np.float64), T["pick"]
        score = L.selection_score(lg, rule)
        n = len(toks)
        zero = pdms[np.arange(n), pick] == 0
        cause = zt.cause_labels(sub[np.arange(n), pick, 0], sub[np.arange(n), pick, 1])
        sel = zt.sel_any(pdms)
        tgt = zero & (cause == "DAC_only") & sel
        P_, S_ = f"{X}_P", f"{X}_S"
        recs = []
        for i in np.nonzero(tgt)[0]:
            t = toks[i]
            S = L.best_nonzero(score[i], pdms[i])
            assert S >= 0 and S != pick[i]
            ndP = g(t, "nd_sim", P_)
            ts = L.first_true(ndP)
            assert ts >= 1, (t, ts)
            cP, cS = g(t, "cxy", P_).astype(np.float64), g(t, "cxy", S_).astype(np.float64)
            LP, LS = L.arc_length(cP), L.arc_length(cS)
            sP = L.arc_length(cP, ts)
            dP = float(L.wrap_deg(np.degrees(g(t, "psi", P_)[-1])))
            dS = float(L.wrap_deg(np.degrees(g(t, "psi", S_)[-1])))
            dH = float(L.wrap_deg(np.degrees(g(t, "psi", "H")[-1])))
            e, _ = L.signed_offset(cP[ts], cS)
            cls = L.classify(LP, LS, dP, dS, e, sP)
            rob = {f"e{et}": L.classify(LP, LS, dP, dS, e, sP, e_thr=et) for et in (0.25, 1.0)}
            rob.update({f"man{mt}": L.classify(LP, LS, dP, dS, e, sP, man_thr=mt) for mt in (20.0, 45.0)})
            ln = LANES[t]["lanes"]
            lanesP, lanesS = ln[str(TJ[P_])], ln[str(TJ[S_])]
            rd = L.route_diff(lanesP, lanesS, ts)
            route = set(LANES[t]["route"])
            offr = any(x not in route for k in range(ts) for x in lanesP[k])
            depth = float(np.max(g(t, "depth", P_)[ts:]))
            tm = (not g(t, "nd_raw", P_).any())
            pp, ps = L.sigmoid(lg[i, pick[i]]), L.sigmoid(lg[i, S])
            dec = L.term_decomposition(lg[i, pick[i]], lg[i, S], rule)
            dom = max(dec, key=lambda k: dec[k])
            dacv = sub[i, :, 1]
            auc = L.within_token_auc(L.sigmoid(lg[i, :, 1]), dacv == 1)
            ni = nidx[t]
            recs.append({"token": t, "ni": ni, "t_star": ts, "d_star": float(np.linalg.norm(cP[ts])), "s_star": sP,
                         "L_P": LP, "L_S": LS, "dpsi_P": dP, "dpsi_S": dS, "dpsi_H_sim": dH, "e_star": e, "class": cls, "rob": rob,
                         "side": L.side_of(g(t, "out4", P_)[ts]), "depth": depth, "tracker_made": bool(tm),
                         "route_diff": bool(rd), "p_off_route": bool(offr),
                         "J": bool(g(t, "inter_c", "H").any()), "exc_junc": bool(g(t, "inter_dist_out", P_)[ts] <= L.EXC_JUNC_M),
                         "cmd": int(cmd_all[ni]), "v0": float(v0_all[ni]), "dpsi_H_seam": float(dpsi_H_all[ni]),
                         "pDAC_P": float(pp[1]), "pDAC_S": float(ps[1]), "rank_S": L.rank_of(score[i], S), "decomp": dec,
                         "dom_term": dom, "auc": auc, "pdms_S": float(pdms[i, S]), "pdms_best": float(pdms[i].max()),
                         "man_P": L.manoeuvre(dP), "man_S": L.manoeuvre(dS), "man_H": L.manoeuvre(dH)})
        # navtest-indexed indicator arrays
        base = np.zeros(N, bool)
        for r in recs:
            base[r["ni"]] = True

        def ind(fn):
            a_ = np.zeros(N, bool)
            for r in recs:
                if fn(r):
                    a_[r["ni"]] = True
            return a_

        def val(fn):
            a_ = np.zeros(N)
            for r in recs:
                a_[r["ni"]] = fn(r)
            return a_

        def med(fn):
            x = np.array([fn(r) for r in recs], float)
            x = x[np.isfinite(x)]
            return {"median": round(float(np.median(x)), 3), "q25": round(float(np.percentile(x, 25)), 3),
                    "q75": round(float(np.percentile(x, 75)), 3), "n": int(x.size)}

        ones = np.ones(N, bool)
        pdmsS = val(lambda r: r["pdms_S"])
        stake = lambda m: Est.ci(*E.vec(np.where(m & base, pdmsS, 0.0), ones), scale=100.0)
        out = {"rule": rule, "n_target": len(recs), "n_table_rows": n, "n_zero": int(zero.sum()),
               "n_DAC_only_zero": int((zero & (cause == "DAC_only")).sum())}
        # Q1
        near = ind(lambda r: r["d_star"] < L.NEAR_M)
        out["Q1"] = {"NEAR": E.share(near, base), "FAR": E.share(~near, base),
                     "t_star_bins_s": {f"[{lo / 10:.0f},{hi / 10:.0f})": E.share(ind(lambda r, lo=lo, hi=hi: lo <= r["t_star"] < hi), base)
                                       for lo, hi in L.TSTAR_BINS},
                     "side": {s: E.share(ind(lambda r, s=s: r["side"] == s), base) for s in ("LEFT", "RIGHT", "MIXED")},
                     "GRAZE_depth_lt_0.5m": E.share(ind(lambda r: r["depth"] < L.GRAZE_M), base),
                     "t_star_s": med(lambda r: r["t_star"] / 10), "d_star_m": med(lambda r: r["d_star"]),
                     "s_star_m": med(lambda r: r["s_star"]), "depth_m": med(lambda r: r["depth"]),
                     "EXC_AT_JUNCTION": E.share(ind(lambda r: r["exc_junc"]), base)}
        # Q6 origin
        tm = ind(lambda r: r["tracker_made"])
        out["Q6_origin"] = {"TRACKER_MADE": E.share(tm, base), "PLAN_INTRINSIC": E.share(~tm, base),
                            "TRACKER_MADE_x_class": dict(Counter(r["class"] for r in recs if r["tracker_made"])),
                            "PLAN_INTRINSIC_x_class": dict(Counter(r["class"] for r in recs if not r["tracker_made"]))}
        # Q2 scene
        q2 = {"command": {}, "v0_mps": {}, "human_dpsi_deg": {}}
        for c, nm in zt.CMD_NAMES.items():
            m_all = cmd_all == c
            if not m_all.any():
                continue
            mt = ind(lambda r, c=c: r["cmd"] == c)
            q2["command"][nm] = {"share_of_target": E.share(mt, base), "navtest_share": E.share(m_all, ones),
                                 "rate_in_class_pct": E.share(base, m_all), "enrichment": E.ratio_of(mt & base, base, m_all, ones)}
        for lo, hi in zt.V0_BINS:
            m_all = (v0_all >= lo) & (v0_all < hi)
            mt = base & m_all
            q2["v0_mps"][f"[{lo},{hi})"] = {"share_of_target": E.share(mt, base), "navtest_share": E.share(m_all, ones),
                                            "enrichment": E.ratio_of(mt, base, m_all, ones)}
        for lo, hi in L.CURV_BINS:
            m_all = (np.abs(dpsi_H_all) >= lo) & (np.abs(dpsi_H_all) < hi)
            mt = base & m_all
            q2["human_dpsi_deg"][f"[{lo:.0f},{min(hi, 999):.0f})"] = {"share_of_target": E.share(mt, base), "navtest_share": E.share(m_all, ones),
                                                                      "enrichment": E.ratio_of(mt, base, m_all, ones)}
        # junction: navtest base from the C3 strata (529 zero census weight 1 + 529 uniform random weight 11617/529)
        w = np.zeros(N)
        Jall = np.zeros(N)
        for t_, s_ in c3["stratum"].items():
            w[nidx[t_]] = 1.0 if s_ != "random" else 11617.0 / 529.0
            Jall[nidx[t_]] = float(G["inter_c"][row2g[jrow[t_]], TJ["H"]].any())
        jt_ = ind(lambda r: r["J"])
        q2["junction_J"] = {"share_of_target": E.share(jt_, base), "navtest_share_stratified": Est.ci(*E.vec(w * Jall, w), scale=100.0),
                            "enrichment": E.ratio_of(jt_ & base, base, w * Jall, w), "strata": {k: int(v) for k, v in Counter(c3["stratum"].values()).items()}}
        out["Q2"] = q2
        # Q3 classes
        q3 = {"class_share": {}, "class_pdms_at_stake": {}}
        for c in L.CLASSES:
            m = ind(lambda r, c=c: r["class"] == c)
            q3["class_share"][c] = E.share(m, base)
            q3["class_pdms_at_stake"][c] = stake(m)
        q3["total_pdms_at_stake"] = stake(base)
        q3["robustness_top_class"] = {}
        for key in ("e0.25", "e1.0", "man20.0", "man45.0"):
            cnt = Counter(r["rob"][key] for r in recs)
            q3["robustness_top_class"][key] = {"top": cnt.most_common(1)[0][0], "counts": dict(cnt)}
        q3["counts"] = dict(Counter(r["class"] for r in recs))
        rdiff = ind(lambda r: r["route_diff"])
        q3["ROUTE_DIFF"] = E.share(rdiff, base)
        q3["P_OFF_ROUTE"] = E.share(ind(lambda r: r["p_off_route"]), base)
        q3["L_P_over_L_S"] = med(lambda r: r["L_P"] / max(r["L_S"], 1e-6))
        q3["P_longer_than_S"] = E.share(ind(lambda r: r["L_P"] > r["L_S"]), base)
        q3["abs_e_star_m"] = med(lambda r: abs(r["e_star"]))
        q3["dpsi_P_minus_S_deg"] = med(lambda r: r["dpsi_P"] - r["dpsi_S"])
        cmd_man = {0: "LEFT", 1: "STRAIGHT", 2: "RIGHT"}
        q3["man_P_matches_command"] = E.share(ind(lambda r: cmd_man.get(r["cmd"]) == r["man_P"]), ind(lambda r: r["cmd"] in cmd_man))
        q3["man_S_matches_command"] = E.share(ind(lambda r: cmd_man.get(r["cmd"]) == r["man_S"]), ind(lambda r: r["cmd"] in cmd_man))
        q3["man_P_eq_man_H"] = E.share(ind(lambda r: r["man_P"] == r["man_H"]), base)
        q3["man_S_eq_man_H"] = E.share(ind(lambda r: r["man_S"] == r["man_H"]), base)
        q3["class_x_command"] = {c: dict(Counter(zt.CMD_NAMES[r["cmd"]] for r in recs if r["class"] == c)) for c in L.CLASSES}
        q3["class_x_NEAR"] = {c: dict(Counter("NEAR" if r["d_star"] < L.NEAR_M else "FAR" for r in recs if r["class"] == c)) for c in L.CLASSES}
        q3["class_x_ROUTE_DIFF"] = {c: dict(Counter(str(r["route_diff"]) for r in recs if r["class"] == c)) for c in L.CLASSES}
        q3["class_x_side"] = {c: dict(Counter(r["side"] for r in recs if r["class"] == c)) for c in L.CLASSES}
        q3["class_x_junction_J"] = {c: dict(Counter(str(r["J"]) for r in recs if r["class"] == c)) for c in L.CLASSES}
        out["Q3"] = q3
        # Q4 discrimination
        inv = ind(lambda r: r["pDAC_P"] >= r["pDAC_S"])
        q4 = {"pDAC_pick": med(lambda r: r["pDAC_P"]), "pDAC_S": med(lambda r: r["pDAC_S"]),
              "pDAC_P_minus_S": med(lambda r: r["pDAC_P"] - r["pDAC_S"]),
              "DAC_INVERTED": E.share(inv, base), "rank_S": med(lambda r: r["rank_S"]),
              "rank_S_eq_2": E.share(ind(lambda r: r["rank_S"] == 2), base), "rank_S_le_4": E.share(ind(lambda r: r["rank_S"] <= 4), base),
              "dominant_positive_term": {k: E.share(ind(lambda r, k=k: r["dom_term"] == k), base) for k in ("NC", "DAC", "W", "lane")},
              "dominant_term_when_not_inverted": dict(Counter(r["dom_term"] for r in recs if r["pDAC_P"] < r["pDAC_S"])),
              "decomp_median": {k: med(lambda r, k=k: r["decomp"][k]) for k in ("NC", "DAC", "W", "lane")},
              "pdms_S": med(lambda r: r["pdms_S"]), "pdms_best": med(lambda r: r["pdms_best"])}
        out["Q4"] = q4
        # Q7 within-token AUC
        aucv = val(lambda r: r["auc"] if np.isfinite(r["auc"]) else 0.0)
        has = ind(lambda r: np.isfinite(r["auc"]))
        q7 = {"target_mean_auc": E.mean(aucv, has), "n_target_mixed": int(has.sum())}
        if X == "A":
            rnd_auc = np.zeros(N)
            rnd = np.zeros(N, bool)
            for i, t in enumerate(toks):
                if c3["stratum"][t] == "random":
                    a_ = L.within_token_auc(L.sigmoid(lg[i, :, 1]), sub[i, :, 1] == 1)
                    if np.isfinite(a_):
                        rnd_auc[nidx[t]], rnd[nidx[t]] = a_, True
            q7["random_mean_auc"] = E.mean(rnd_auc, rnd)
            q7["n_random_mixed"] = int(rnd.sum())
            q7["target_minus_random"] = E.diff_of(aucv * has, has, rnd_auc * rnd, rnd)
        out["Q7"] = q7
        # Q5 coverage
        q5 = {}
        for lev, mat in MATRIX.items():
            def cell(r, use_origin=True, mat=mat):
                c = mat["TRACKER_MADE"] if (use_origin and r["tracker_made"]) else mat[r["class"]]
                return ("P" if r["route_diff"] else "N") if c == "P_RD" else c
            y = ind(lambda r: cell(r) == "Y")
            yp = ind(lambda r: cell(r) in ("Y", "P"))
            ys = ind(lambda r: cell(r, False) == "Y")
            q5[lev] = {"Y_share": E.share(y, base), "Y_or_P_share": E.share(yp, base), "N_share": E.share(~yp, base),
                       "Y_pdms_at_stake": stake(y), "Y_or_P_pdms_at_stake": stake(yp),
                       "sensitivity_class_cell_for_all_Y_share": E.share(ys, base)}
        out["Q5"] = q5
        arms[X] = out
        if X == "A":
            per_token_A = recs
        # per-token dump
        with open(PK / f"data/per_token_{X}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({k: v for k, v in r.items()}, default=float) + "\n")

    # G2 identity vs the banked tracked dump (arm A picks)
    mx, ncmp = 0.0, 0
    tabtok = [str(t) for t in A["token"]]
    for p in sorted(Path(TRACKED).glob("tracked_*of4.npz")):
        z = np.load(p)
        for q, (ix, t) in enumerate(zip(z["idx"], z["token"])):
            t = str(t)
            if t not in jrow:
                continue
            assert tabtok[int(ix)] == t
            mine = G["corners_AP"][row2g[jrow[t]]]
            ref = z["corners_sim_ego"][q, int(A["pick"][int(ix)])]
            mx = max(mx, float(np.nanmax(np.abs(mine - ref))))
            ncmp += 1
    gates["G2_tracked_dump_identity"] = {"n_tokens": ncmp, "max_abs_m": mx, "pass": mx <= 1e-3}

    # committed readings (PREREG section 5)
    def pt(x):
        return x["point"]

    rd = {}
    for X in ("A", "B"):
        o = arms[X]
        recs = [json.loads(l) for l in open(PK / f"data/per_token_{X}.jsonl", encoding="utf-8")]
        route_ = sum(1 for r in recs if r["class"] == "WRONG_MANOEUVRE" or r["route_diff"]) / len(recs) * 100
        geo_ = sum(1 for r in recs if r["class"] in ("CORNER_CUT", "WIDE_TURN", "STRAIGHT_DRIFT", "RUN_LONG", "ON_PATH")
                   and not r["route_diff"]) / len(recs) * 100
        top = max(o["Q3"]["class_share"], key=lambda c: pt(o["Q3"]["class_share"][c]))
        rd[X] = {"R1_route_share_pct": round(route_, 2), "R1_same_route_geometric_share_pct": round(geo_, 2),
                 "R1": "ROUTE" if route_ >= 50 else ("GEOMETRY" if geo_ >= 50 else "MIXED"),
                 "R2": ("EXECUTION_ARM" if pt(o["Q6_origin"]["TRACKER_MADE"]) >= 25 else
                        ("PLANS_LEAVE_ROAD" if pt(o["Q6_origin"]["TRACKER_MADE"]) < 10 else "BETWEEN")),
                 "R3": "HEAD_WRONG" if pt(o["Q4"]["DAC_INVERTED"]) >= 50 else "OVERRIDDEN",
                 "R4": "NEAR" if pt(o["Q1"]["NEAR"]) >= 50 else "FAR",
                 "R5": "NEAR_BLIND" if o["Q7"]["target_mean_auc"]["point"] <= 0.60 else "DISCRIMINATES",
                 "top_class": top, "top_class_share": o["Q3"]["class_share"][top]}
        lev_pdms = {k: pt(v["Y_pdms_at_stake"]) for k, v in o["Q5"].items()}
        lev_cnt = {k: pt(v["Y_share"]) for k, v in o["Q5"].items()}
        rd[X]["lever_largest_Y_pdms"] = max(lev_pdms, key=lev_pdms.get)
        rd[X]["lever_largest_Y_count"] = max(lev_cnt, key=lev_cnt.get)
    tA, tB = rd["A"]["top_class"], rd["B"]["top_class"]
    rd["replication_P2"] = {"same_top_class": tA == tB,
                            "share_diff_pts": round(pt(arms["B"]["Q3"]["class_share"][tA]) - pt(arms["A"]["Q3"]["class_share"][tA]), 2)}
    rd["replication_P2"]["replicates"] = rd["replication_P2"]["same_top_class"] and abs(rd["replication_P2"]["share_diff_pts"]) <= 15

    R = {"package": "2026-10-17-zero-tail-mechanism (ZT3)", "tier": "T1 NAVSIM v1.1 pseudo-sim; geometry on simulated (LQR-tracked) states",
         "estimator": "log-cluster bootstrap, 136 navtest logs, 10,000 resamples, seed 20260927 (zero_tail.Boot); answers another draw of LOGS only",
         "evidence": "MEASURED (ours), this package", "prereg_sha256_chain": open(PK / "PREREG_ZT3.md.sha256").read().split("\n"),
         "gates": gates, "readings": rd, "arms": arms,
         "inputs": {**json.loads(str(J["meta"]))["inputs"], "jobs.npz": sha(DATA / "jobs.npz"),
                    "inputs_export": {"path": INPUTS_EXPORT, "sha256": sha(INPUTS_EXPORT)},
                    "human_csv": {"path": HUMAN_CSV, "sha256": sha(HUMAN_CSV)}, "geom_parts": {p: sha(p) for p in gparts}}}
    json.dump(R, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"gates": gates, "readings": rd}, indent=1, default=float))


if __name__ == "__main__":
    main()
