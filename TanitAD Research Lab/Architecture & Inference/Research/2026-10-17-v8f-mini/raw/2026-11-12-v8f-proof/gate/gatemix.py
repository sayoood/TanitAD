"""gatemix.py -- PREREG_V8F_PROOF_ADD8 (GATEMIX).  CPU, zero training, zero rollouts.  Tier T1-kin (logged frames): a diagnostic, never driving performance.

Usage:  python gatemix.py run [--out-dir DIR] [--nboot 2000] [--g3-seeds 20]
Order: g2 (reproduction, before any read; failing stops the run) -> tick-0 load + alignment -> g1, g4 -> g3 -> reads -> RESULT_GATEMIX.json + VERDICT_GATEMIX.md.
Exit 0 = all controls pass, 2 = a control failed.  Definitions and deviations: GATE_DEFS.md.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
sys.path.insert(0, HERE.rsplit("/", 1)[0] + "/hier")
import numpy as np  # noqa: E402
import hierwhere as H  # noqa: E402  (read-only reuse: loader, ADE6, mode function, bootstrap on the pinned tree)
from scipy.stats import rankdata  # noqa: E402

ART = H.ART
SEEDS = (0, 1)
NB, BSEED = H.NB, H.BSEED
F_SEED, F_TAC = H.F_SEED, H.F_TAC
NFOLD, FOLD_SEED = 5, 0
NLAT, NLON = 5, 7
DUMP = {"DAG": ART + "/v8f-dagger-t1/dump_DAG_s{s}_m10_causal_heads", "DAG0": ART + "/v8f-dagger-t1/dump_DAG0_s{s}_m10_causal_heads",
        "FLATDAG": ART + "/v8f-flatdag-t1/dump_FLATDAG_s{s}_m10_causal_heads", "FLATDAG0": ART + "/v8f-flatdag-t1/dump_FLATDAG0_s{s}_m10_causal_heads"}
TICKS = ART + "/v8f-dagger-t1/ticks/ticks_{h}_s{s}_m10_causal_heads.npz"
PAIRS = {"P1": ("DAG", "FLATDAG"), "P2": ("DAG0", "FLATDAG0")}
PAIR2_BANKED_READ = HERE.rsplit("/", 1)[0] + "/hier/pair2/read/RESULT_HIERWHERE.json"
BANKED_P1 = H.BANKED                                                   # {0: (-0.3451,..), 1: (-0.0544,..)}
BANKED_H1A = {0: (-0.4973, -0.6075, -0.3874), 1: (-0.2450, -0.3553, -0.1316)}   # (DAG, FLATDAG0), the h1a route
METRICS = ("ade6_m", "ade26_m", "speed_mae_mps", "along_mae_m", "cross_mae_m", "heading_mae_deg", "curvature_mae_1pm", "yaw_rate_mae_degps")
FAM = (("LONGITUDINAL", "speed_mae_mps"), ("LONGITUDINAL", "along_mae_m"), ("LATERAL", "cross_mae_m"), ("LATERAL", "heading_mae_deg"),
       ("LATERAL", "curvature_mae_1pm"), ("LATERAL", "yaw_rate_mae_degps"))


# ------------------------------------------------------------------------------------------------ score and cross-fit (pure functions, tested on synthetic data)
def softmax(x):
    x = np.asarray(x, np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def joint_probs(p_lat, p_lon):
    """[n,5] x [n,7] -> [n,35]; column lat*7 + lon, product p_lat(i) * p_lon(j)."""
    return (p_lat[:, :, None] * p_lon[:, None, :]).reshape(len(p_lat), -1)


def common_case(p_lat, p_lon, train):
    """most frequent (argmax lat, argmax lon) over the training windows; ties -> lowest joint index."""
    j = np.argmax(p_lat, 1) * p_lon.shape[1] + np.argmax(p_lon, 1)
    return int(np.bincount(j[train], minlength=p_lat.shape[1] * p_lon.shape[1]).argmax())


def score(J, common):
    """largest joint probability over joint classes other than the common-case one."""
    K = np.array(J, np.float64, copy=True)
    K[:, common] = -1.0
    return K.max(axis=1)


def fold_assign(eid, k=NFOLD, seed=FOLD_SEED):
    """5 folds by episode: unique episode ids (sorted) permuted by default_rng(seed), fold = position % k."""
    uniq = np.unique(eid)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    fo = np.empty(len(uniq), np.int64)
    fo[perm] = np.arange(len(uniq)) % k
    return fo[np.searchsorted(uniq, eid)]


def tau_candidates(s):
    return np.concatenate([[-np.inf], np.quantile(s, np.linspace(0.0, 1.0, 101)), [np.inf]])


def pick_tau(s, eh, ef):
    """tau minimising the composite mean ADE6 on the given (training) windows; hierarchy drives when s >= tau; ties -> first candidate in ascending order."""
    c = tau_candidates(s)
    m = np.array([np.where(s >= t, eh, ef).mean() for t in c])
    return float(c[int(np.argmin(m))])


def crossfit(eh, ef, p_lat, p_lon, fold, shuffle_seed=None):
    """-> {S held-out score per window, tau per window (its fold's), drive, common per fold}.  Every window is decided by a tau and a common-case class fit on the other folds only.
    shuffle_seed (control g3): the score of every fold group is permuted within the group (training groups and the held-out group alike)."""
    J = joint_probs(p_lat, p_lon)
    n = len(eh)
    S, T = np.empty(n), np.empty(n)
    rng = None if shuffle_seed is None else np.random.default_rng(1000 + shuffle_seed)
    commons = []
    for f in range(NFOLD):
        tr = fold != f
        com = common_case(p_lat, p_lon, tr)
        commons.append(com)
        Sf = score(J, com)
        if rng is not None:
            for g in range(NFOLD):
                idx = np.flatnonzero(fold == g)
                Sf[idx] = Sf[rng.permutation(idx)]
        tau = pick_tau(Sf[tr], eh[tr], ef[tr])
        S[~tr], T[~tr] = Sf[~tr], tau
    return {"S": S, "tau": T, "drive": S >= T, "common": commons}


def compose(drive, cH, cF):
    """per-window composite metrics: the hierarchy's value where drive, else the flat arm's (NaN-preserving)."""
    return {k: np.where(drive, cH[k], cF[k]) for k in cH}


def auroc(s, y):
    """P(score of a critical window > score of a bulk window), ties 1/2 (rank form)."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------------------------------------ data
def comps_for(name, s):
    arm = H.load_arm(DUMP[name].format(s=s))
    return arm, H.comps(arm["cl"], arm["g"])


def load_ticks(h, s, arm):
    """tick-0 class probabilities of the hierarchy arm (softmax of t_lat_logits [5], t_lon_logits [7]: only logits are stored); alignment asserted against the dump windows."""
    with np.load(TICKS.format(h=h, s=s)) as z:
        m = z["tick"] == 0
        chk = {"n_tick0": int(m.sum()), "ci_equal": bool((z["ci"][m] == arm["clip_index"]).all()), "t0_equal_ws": bool((z["t0"][m] == arm["ws"]).all()),
               "gt_max_abs_diff": float(np.abs(z["gt"][m] - arm["g"]).max()), "v0_max_abs_diff": float(np.abs(z["v0"][m] - arm["v0"]).max()),
               "age_t_all0": bool((z["age_t"][m] == 0).all()), "gt_ok_all": bool(z["gt_ok"][m].all())}
        ok = chk["n_tick0"] == len(arm["ws"]) and chk["ci_equal"] and chk["t0_equal_ws"] and chk["gt_max_abs_diff"] < 1e-4 and chk["age_t_all0"]
        assert ok, ("tick-0 rows do not align with the dump windows", chk)
        return softmax(z["t_lat_logits"][m]), softmax(z["t_lon_logits"][m]), chk


def critical_label(g):
    lon, lat = H.modes(g)
    return H.joint(lon, lat) != H.KEEP_STRAIGHT


# ------------------------------------------------------------------------------------------------ statistics
def fmt(r):
    return f"{r['point']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"


def sep_neg(r):
    return bool(r["hi"] < 0)


def sep_pos(r):
    return bool(r["lo"] > 0)


def outcome_cell(rh, rf):
    """per pair and seed, on ADE6.  rh, rf: composite minus hierarchy / minus flat."""
    if sep_pos(rh) or sep_pos(rf):
        return "GATE-WORSE"
    if sep_neg(rh) and sep_neg(rf):
        return "GATE-BEATS-BOTH"
    return "GATE-TIES"


def outcome_overall(cells):
    """cells {pair: {seed: name}}."""
    names = [v for p in cells.values() for v in p.values()]
    if any(n == "GATE-WORSE" for n in names):
        return "GATE-COSTS"
    if any(all(p[s] == "GATE-BEATS-BOTH" for s in p) for p in cells.values()):
        return "GATE-DOMINATES"
    return "GATE-TIES"


def read_cell(eid, cH, cF, comp, orc, crit, S, drive, nb):
    """all reads of the card for one pair and seed (one set of bootstrap draws for every statistic)."""
    stats, nn = {}, {}
    allm = np.ones(len(eid), bool)
    groups = {"all": allm, "critical": crit, "bulk": ~crit}
    for nm, X in (("C", comp), ("O", orc)):
        for an, A in (("H", cH), ("F", cF)):
            for met in ("ade6_m", "ade26_m"):
                d = X[met] - A[met]
                for gn, gm in groups.items():
                    stats[f"{nm}-{an}|{met}|{gn}"] = H.mean_over(d, gm)
    for an, A in (("H", cH), ("F", cF)):
        for fam, met in FAM:
            stats[f"C-{an}|{met}|fam"] = H.mean_over(comp[met] - A[met])
            nn[f"C-{an}|{met}|fam"] = H.nums(eid, allm, comp[met] - A[met])
    dr = drive.astype(np.float64)
    stats["share|all"], stats["share|critical"], stats["share|bulk"] = H.mean_over(dr), H.mean_over(dr, crit), H.mean_over(dr, ~crit)
    stats["share_oracle|all"] = H.mean_over(crit.astype(np.float64))
    stats["auroc"] = lambda sel: auroc(S[sel], crit[sel])
    r = H.boot(eid, stats, nb)
    out = {k: H.row(v) for k, v in r.items()}
    for gn, gm in groups.items():
        for k in out:
            if k.endswith("|" + gn) and "share" not in k:
                out[k]["n_windows"], out[k]["n_episodes"] = H.nums(eid, gm)
    for k, v in nn.items():
        out[k]["n_windows"], out[k]["n_episodes"] = v
    out["auroc"]["n_windows"], out["auroc"]["n_episodes"] = H.nums(eid, allm)
    out["auroc"]["n_critical"], out["auroc"]["n_bulk"] = int(crit.sum()), int((~crit).sum())
    return out


# ------------------------------------------------------------------------------------------------ the run
def run(a):
    out_dir = a.out_dir.replace("\\", "/")
    os.makedirs(out_dir + "/stages", exist_ok=True)
    nb = a.nboot
    t0 = time.time()
    res = {"card": "PREREG_V8F_PROOF_ADD8 sha256 807bb2115013", "tier": "T1-kin", "floors": {"F_seed_m": F_SEED, "tactical_replicate_m": F_TAC}, "controls": {}, "cells": {}}

    def bank(name, obj):
        with open(f"{out_dir}/stages/{name}.json", "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)

    # ---- g2 first: the banked ADE6 differences
    ARM = {}
    for s in SEEDS:
        for nm in DUMP:
            ARM[(nm, s)] = comps_for(nm, s)
    ref2 = json.load(open(PAIR2_BANKED_READ, encoding="utf-8"))["controls"]
    g2 = {}
    for s in SEEDS:
        for label, (hn, fn), bk in (("P1", ("DAG", "FLATDAG"), BANKED_P1[s]), ("P2_h1a_route", ("DAG", "FLATDAG0"), BANKED_H1A[s]),
                                    ("P2_read_selfref", ("DAG0", "FLATDAG0"), (ref2[f"h1_s{s}"]["reader"]["delta"], ref2[f"h1_s{s}"]["reader"]["lo"], ref2[f"h1_s{s}"]["reader"]["hi"]))):
            ah, ch = ARM[(hn, s)]
            af, cf = ARM[(fn, s)]
            assert (ah["clip_index"] == af["clip_index"]).all() and (ah["ws"] == af["ws"]).all() and np.array_equal(ah["g"], af["g"])
            pb = H.CI.paired_episode_cluster_bootstrap(ch["ade6_m"], cf["ade6_m"], ah["eid"], n_boot=nb, seed=BSEED)
            dev = max(abs(pb["delta"] - bk[0]), abs(pb["lo"] - bk[1]), abs(pb["hi"] - bk[2]))
            g2[f"{label}_s{s}"] = {"PASS": bool(dev <= 1e-4 + 1e-12), "reproduced": [pb["delta"], pb["lo"], pb["hi"]], "banked": list(bk), "max_abs_dev_m": dev}
    res["controls"]["g2"] = g2
    res["controls"]["g2_PASS"] = bool(all(v["PASS"] for v in g2.values()))
    bank("g2", g2)
    print("g2", json.dumps({k: (v["PASS"], v["max_abs_dev_m"]) for k, v in g2.items()}), f"{time.time() - t0:.0f}s", flush=True)
    if not res["controls"]["g2_PASS"]:
        print("g2 FAILED; stopping before any read")
        return finish(res, out_dir, 2, nb)

    # ---- tick-0 probabilities, alignment, fold, score, composite
    D = {}
    for pn, (hn, fn) in PAIRS.items():
        for s in SEEDS:
            ah, ch = ARM[(hn, s)]
            af, cf = ARM[(fn, s)]
            pl, po, chk = load_ticks(hn, s, ah)
            crit = critical_label(ah["g"])
            fold = fold_assign(ah["eid"])
            cf_ = crossfit(ch["ade6_m"], cf["ade6_m"], pl, po, fold)
            D[(pn, s)] = dict(ah=ah, ch=ch, cf=cf, pl=pl, po=po, chk=chk, crit=crit, fold=fold, cfit=cf_, eid=ah["eid"])
            res["cells"].setdefault(pn, {})[f"s{s}"] = {"alignment": chk, "n_windows": int(len(ah["eid"])), "n_episodes": int(len(np.unique(ah["eid"]))),
                                                        "n_critical": int(crit.sum()), "fold_sizes_windows": np.bincount(fold).tolist(),
                                                        "common_case_joint_per_fold": cf_["common"], "tau_per_fold": sorted({float(t) for t in cf_["tau"]})}
    # ---- g1, g4
    g1, g4 = {}, {}
    for (pn, s), d in D.items():
        S = d["cfit"]["S"]
        lo_t, hi_t = np.full(len(S), S.min() - 1.0), np.full(len(S), S.max() + 1.0)
        cH, cF = d["ch"], d["cf"]
        mh = compose(S >= lo_t, cH, cF)
        mf = compose(S >= hi_t, cH, cF)
        dh = max(float(np.nanmax(np.abs(mh[k] - cH[k]))) for k in METRICS)
        df = max(float(np.nanmax(np.abs(mf[k] - cF[k]))) for k in METRICS)
        g1[f"{pn}_s{s}"] = {"PASS": dh == 0.0 and df == 0.0, "tau_below_vs_hier_max_abs_m": dh, "tau_above_vs_flat_max_abs_m": df}
        crit = d["crit"]
        orc = compose(crit, cH, cF)
        implied = (crit.sum() * cH["ade6_m"][crit].mean() + (~crit).sum() * cF["ade6_m"][~crit].mean()) / len(crit)
        dev = abs(float(orc["ade6_m"].mean()) - float(implied))
        g4[f"{pn}_s{s}"] = {"PASS": dev <= 1e-9, "oracle_mean_ade6_m": float(orc["ade6_m"].mean()), "implied_by_group_means_m": float(implied), "abs_dev_m": dev}
        d["orc"] = orc
    res["controls"]["g1"], res["controls"]["g4"] = g1, g4
    res["controls"]["g1_PASS"] = all(v["PASS"] for v in g1.values())
    res["controls"]["g4_PASS"] = all(v["PASS"] for v in g4.values())
    bank("g1_g4", {"g1": g1, "g4": g4})
    print("g1", res["controls"]["g1_PASS"], "g4", res["controls"]["g4_PASS"], f"{time.time() - t0:.0f}s", flush=True)
    if not (res["controls"]["g1_PASS"] and res["controls"]["g4_PASS"]):
        print("g1/g4 FAILED; stopping before any read")
        return finish(res, out_dir, 2, nb)

    # ---- g3: shuffled score
    g3 = {}
    for (pn, s), d in D.items():
        eh, ef = d["ch"]["ade6_m"], d["cf"]["ade6_m"]
        best = eh if eh.mean() <= ef.mean() else ef
        rows = []
        for k in range(a.g3_seeds):
            cs = crossfit(eh, ef, d["pl"], d["po"], d["fold"], shuffle_seed=k)
            dd = np.where(cs["drive"], eh, ef) - best
            r = H.boot(d["eid"], {"d": H.mean_over(dd)}, nb)["d"]
            rows.append({"seed": k, "point": round(r["point"], 6), "lo": round(r["lo"], 6), "hi": round(r["hi"], 6), "negative_separated": sep_neg(r)})
        npass = sum(1 for r in rows if not r["negative_separated"])
        g3[f"{pn}_s{s}"] = {"PASS": npass >= 18 * a.g3_seeds // 20, "n_not_negative_separated": npass, "n_shuffles": len(rows), "better_arm": "hier" if eh.mean() <= ef.mean() else "flat", "rows": rows}
        print("g3", pn, s, npass, f"{time.time() - t0:.0f}s", flush=True)
    res["controls"]["g3"] = g3
    res["controls"]["g3_PASS"] = all(v["PASS"] for v in g3.values())
    bank("g3", g3)

    # ---- reads
    names = {}
    for (pn, s), d in D.items():
        comp = compose(d["cfit"]["drive"], d["ch"], d["cf"])
        cell = read_cell(d["eid"], d["ch"], d["cf"], comp, d["orc"], d["crit"], d["cfit"]["S"], d["cfit"]["drive"], nb)
        nm = outcome_cell(cell["C-H|ade6_m|all"], cell["C-F|ade6_m|all"])
        cell["outcome"] = nm
        cell["arm_means_ade6_m"] = {"hier": float(d["ch"]["ade6_m"].mean()), "flat": float(d["cf"]["ade6_m"].mean()), "composite": float(comp["ade6_m"].mean()), "oracle": float(d["orc"]["ade6_m"].mean())}
        res["cells"][pn][f"s{s}"]["read"] = cell
        names.setdefault(pn, {})[s] = nm
        bank(f"read_{pn}_s{s}", cell)
        print("read", pn, s, nm, f"{time.time() - t0:.0f}s", flush=True)
    res["outcome_cells"] = {p: {f"s{s}": v for s, v in c.items()} for p, c in names.items()}
    res["outcome_overall"] = outcome_overall(names)
    ctrl_ok = all(res["controls"][k] for k in ("g2_PASS", "g1_PASS", "g3_PASS", "g4_PASS"))
    return finish(res, out_dir, 0 if ctrl_ok else 2, nb)


def finish(res, out_dir, rc, nb):
    res["exit_code"] = rc
    res["n_boot"] = nb
    with open(out_dir + "/RESULT_GATEMIX.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    with open(out_dir + "/VERDICT_GATEMIX.md", "w", encoding="utf-8") as f:
        f.write(verdict(res))
    return rc


def flag(r):
    a = abs(r["point"])
    return ("> F" if a > F_SEED else "<= F") + (", > 0.287" if a > F_TAC else "")


def verdict(res):
    L = []
    P = L.append
    C = res["controls"]
    P("# VERDICT_GATEMIX (PREREG_V8F_PROOF_ADD8, card sha256 807bb2115013; MEASURED, tier T1-kin on logged frames: a diagnostic, never driving performance; CPU, zero training and zero rollouts)")
    P("")
    if "outcome_overall" not in res:
        P("**No read was made: a control stopped the run.** " + json.dumps({k: v for k, v in C.items() if k.endswith("PASS")}))
        return "\n".join(L) + "\n"
    P(f"## Overall outcome: **{res['outcome_overall']}**")
    P("")
    P("Per pair and seed (decided on composite minus arm, ADE6, paired episode-cluster bootstrap, B 2,000, seed 0):")
    P("")
    P("| pair | seed | outcome | composite minus hierarchy | composite minus flat |")
    P("|---|---|---|---|---|")
    for pn in PAIRS:
        for s in SEEDS:
            r = res["cells"][pn][f"s{s}"]["read"]
            P(f"| {pn} ({PAIRS[pn][0]} vs {PAIRS[pn][1]}) | {s} | **{r['outcome']}** | {fmt(r['C-H|ade6_m|all'])} | {fmt(r['C-F|ade6_m|all'])} |")
    P("")
    P("Floors beside every number: F = 0.12075 m (seed floor), tactical replicate floor 0.287 m. Estimator of every interval: paired episode-cluster bootstrap over the 141 episodes (taniteval.ci draws), B 2,000, seed 0; the tau and the common-case class are fixed per fold and are not re-fitted inside the draws. One training run per arm per seed: a separated interval is necessary and not sufficient. STRATEGIC family: **PAUSED** (PI item 25).")
    P("")
    P("## Controls")
    P("")
    P(f"* g1 (tau below every score = hierarchy, above = flat, 0.0 m): {'PASS' if C['g1_PASS'] else 'FAIL'}; max abs deviation " + ", ".join(f"{k} {v['tau_below_vs_hier_max_abs_m']:.1e}/{v['tau_above_vs_flat_max_abs_m']:.1e}" for k, v in C["g1"].items()))
    P(f"* g2 (banked ADE6 differences reproduced, bar 1e-4 m on point and bounds): {'PASS' if C['g2_PASS'] else 'FAIL'}")
    for k, v in C["g2"].items():
        P(f"  * {k}: reproduced {v['reproduced'][0]:+.4f} [{v['reproduced'][1]:+.4f}, {v['reproduced'][2]:+.4f}] vs banked {v['banked'][0]:+.4f} [{v['banked'][1]:+.4f}, {v['banked'][2]:+.4f}], max dev {v['max_abs_dev_m']:.1e} {'PASS' if v['PASS'] else 'FAIL'}")
    P(f"* g3 (shuffled score: composite minus the better arm not negative-and-separated in >= 18 of 20 shuffles): {'PASS' if C['g3_PASS'] else 'FAIL'}; " + ", ".join(f"{k} {v['n_not_negative_separated']}/{v['n_shuffles']}" for k, v in C["g3"].items()))
    P(f"* g4 (ORACLE composite equals the share-weighted group means, bar 1e-9 m): {'PASS' if C['g4_PASS'] else 'FAIL'}; " + ", ".join(f"{k} {v['abs_dev_m']:.1e}" for k, v in C["g4"].items()))
    P("")
    for pn in PAIRS:
        for s in SEEDS:
            c = res["cells"][pn][f"s{s}"]
            r = c["read"]
            P(f"## {pn} ({PAIRS[pn][0]} vs {PAIRS[pn][1]}), seed {s}: {r['outcome']}")
            P("")
            P(f"n = {c['n_windows']:,} windows, {c['n_episodes']} episodes; selection-critical {c['n_critical']:,}; fold sizes (windows) {c['fold_sizes_windows']}; common-case joint class (lat*7+lon) per fold {c['common_case_joint_per_fold']}; tau per fold {[round(t, 4) for t in c['tau_per_fold']]}. Mean ADE6 (m): hierarchy {r['arm_means_ade6_m']['hier']:.4f}, flat {r['arm_means_ade6_m']['flat']:.4f}, composite {r['arm_means_ade6_m']['composite']:.4f}, ORACLE {r['arm_means_ade6_m']['oracle']:.4f}.")
            P("")
            P("| metric | group | n win / n ep | composite minus hier | composite minus flat | ORACLE minus hier | ORACLE minus flat | floors (composite minus flat) |")
            P("|---|---|---|---|---|---|---|---|")
            for met in ("ade6_m", "ade26_m"):
                for gn in ("all", "critical", "bulk"):
                    a_, b_, c_, d_ = (r[f"{x}|{met}|{gn}"] for x in ("C-H", "C-F", "O-H", "O-F"))
                    P(f"| {met} | {gn} | {a_['n_windows']:,} / {a_['n_episodes']} | {fmt(a_)} | {fmt(b_)} | {fmt(c_)} | {fmt(d_)} | {flag(b_)} |")
            P("")
            sh = {g: r[f"share|{g}"] for g in ("all", "critical", "bulk")}
            au = r["auroc"]
            P(f"Share of windows driven by the hierarchy: all {fmt(sh['all'])}, critical {fmt(sh['critical'])}, bulk {fmt(sh['bulk'])}; ORACLE share {r['share_oracle|all']['point']:.4f}. AUROC of s(w) for the logged selection-critical label: {fmt(au)} (n = {au['n_windows']:,}, {au['n_episodes']} episodes; {au['n_critical']:,} critical, {au['n_bulk']:,} bulk; 0.5 = chance).")
            P("")
            P("Four families, composite minus each arm (mean over windows where both are finite; n windows per row):")
            P("")
            P("| family | metric | n win | composite minus hier | composite minus flat |")
            P("|---|---|---|---|---|")
            for fam, met in FAM:
                a_, b_ = r[f"C-H|{met}|fam"], r[f"C-F|{met}|fam"]
                P(f"| {fam} | {met} | {a_['n_windows']:,} | {fmt(a_)} | {fmt(b_)} |")
            P(f"| TACTICAL | AUROC of s(w) | {au['n_windows']:,} | {fmt(au)} | share driven by hier {fmt(sh['all'])} |")
            P("| STRATEGIC | PAUSED (PI item 25) | | | |")
            P("")
    P("## Card bars next to the measured values")
    P("")
    P("| bar | measured | result |")
    P("|---|---|---|")
    for pn in PAIRS:
        for s in SEEDS:
            r = res["cells"][pn][f"s{s}"]["read"]
            h, f_ = r["C-H|ade6_m|all"], r["C-F|ade6_m|all"]
            P(f"| {pn} s{s}: composite minus hierarchy negative, interval excludes 0 | {fmt(h)} | {'PASS' if sep_neg(h) else 'FAIL'} |")
            P(f"| {pn} s{s}: composite minus flat negative, interval excludes 0 | {fmt(f_)} | {'PASS' if sep_neg(f_) else 'FAIL'} |")
            P(f"| {pn} s{s}: no cell GATE-WORSE (neither difference positive with interval excluding 0) | {r['outcome']} | {'PASS' if r['outcome'] != 'GATE-WORSE' else 'FAIL'} |")
    for pn in PAIRS:
        ok = all(res["cells"][pn][f"s{s}"]["read"]["outcome"] == "GATE-BEATS-BOTH" for s in SEEDS)
        P(f"| {pn}: GATE-BEATS-BOTH at both seeds | {[res['cells'][pn][f's{s}']['read']['outcome'] for s in SEEDS]} | {'PASS' if ok else 'FAIL'} |")
    P(f"| overall GATE-DOMINATES | {res['outcome_overall']} | {'PASS' if res['outcome_overall'] == 'GATE-DOMINATES' else 'FAIL'} |")
    for k in ("g1", "g2", "g3", "g4"):
        P(f"| control {k} | see Controls | {'PASS' if C[k + '_PASS'] else 'FAIL'} |")
    P("")
    P("## Limits of the card (stated in advance, kept)")
    P("")
    P("Post-hoc on banked runs whose primary outcome is NOT ESTABLISHED. One training run per arm per seed, no replicate arm: the bootstrap resamples episodes with the models held fixed and cannot see training variance. tau is cross-fitted on the evaluation clips (5 folds by episode), so the composite is an estimate of a rule, not a deployable threshold. The composite needs both models to run. ORACLE is privileged (the logged label) and is a ceiling, not an arm. This read never supports a statement that the hierarchy beats flat as an architecture. Definitions and deviations (R1...): GATE_DEFS.md.")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", default=HERE)
    r.add_argument("--nboot", type=int, default=NB)
    r.add_argument("--g3-seeds", type=int, default=20)
    a = ap.parse_args(argv)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
