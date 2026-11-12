r"""bias_read.py -- READ of BIASEDIT (PREREG_V8F_PROOF_ADD4.md, frozen).  CPU, no model pass.  Tier T1-kin first tick (self-action open loop, m=10, causal, dead-reckoned nav, logged scene), eval-141: a diagnostic.
Estimator: taniteval.ci episode-cluster bootstrap (paired for two arms), B=2000, seed 0 (another draw of EPISODES only; the replicate gap |NULL_rep - NULL_s0| is the run-to-run yardstick the card names).
Inputs: parents NULL_s0 / NULL_rep / G_s0 / G2_s0 from wsp/t1 + wsp/stopdec (the WSP job, same environment, 2 threads); NULL_s1 / G_s1 / G2_s1 / GE{0,1,2}_s{0,1} from bias/t1 + bias/stopdec.
Reuses wsp_read (loaders, cells_of, family labels) and theory/stopbal/read/stopbal_read.py by import; edits neither.
Controls: e2 stop-goal AUROC of GE1/GE2 equals the parent's to 1e-9 (STOPDEC windows, both seeds); e3 GE0 dump (cl, ol, fl, g, v0) and STOPDEC probabilities bit-identical to the parent's; e1 is in BUILD_REPORT.json (re-checked here);
instrument controls of stopdec; one thread count across all dumps; window sets equal.
Reads: (1) hit / fpr at p>=0.5 for parent, GE1, GE2, trained G, G2, per seed, with CIs; replicate gap per metric (|NULL_rep - NULL_s0|, seed-0 pair, used for both seeds: only one replicate pair exists).
(2) b_match: scan b in [-12, 12] step 0.0005 of p -> sigmoid(logit(p) + b) on the PARENT's STOPDEC stop-goal probabilities; the b whose fpr is closest to the trained arm's (middle of the plateau), and the hit there.
(3) T1-kin four families for GE2 vs its parent and vs trained G2 (paired episode-cluster bootstrap, B 2000, seed 0): all windows and the stop-point windows; TAC-LON macro-F1 by the same draws; STRATEGIC PAUSED (PI item 25).
Outcome (card): EQUALS = at both doses and both seeds hit AND fpr of the edit within the replicate gap of the trained arm; else MATCHES-ROC = at b_match hit within the gap of the trained arm for all four (dose, seed) pairs;
else DIFFERS = outside the gap for all four; any other pattern is printed MIXED (card silent) with its per-pair classes.  (Reading choices where the card says both doses and both seeds.)
Usage: python -I bias_read.py [--out RESULT_BIASEDIT.json] [--verdict VERDICT_BIASEDIT.md] [--n-boot 2000]"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bias_common as B  # noqa: E402
import wsp_common as C  # noqa: E402
import wsp_read as W  # noqa: E402  (sets STOPBAL_TREE, imports our tree first, asserts tanitad path)
import numpy as np  # noqa: E402

SR, NB, SEED = W.SR, W.NB, W.SEED
GRID = np.arange(-12.0, 12.0001, 0.0005)


def arm_spec(tag):
    mine = tag in B.MY_DUMP_TAGS
    t1, sd = (B.T1, B.STOPDEC_OUT) if mine else (C.T1, C.STOPDEC_OUT)
    return {"dump": f"{t1}/dump_{tag}_m10_causal_heads", "done": f"{t1}/done_{tag}_m10_causal_heads.txt", "stopdec": f"{sd}/stopdec_{tag}.npz", "stopdec_json": f"{sd}/STOPDEC_{tag}.json", "stopdec_tag": tag}


def shift(p, b):
    p = np.clip(p, 1e-300, 1 - 1e-16)
    return 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + b)))


def macro_f1(pred, gold, classes):
    f = []
    for c in classes:
        tp, fp, fn = np.sum((pred == c) & (gold == c)), np.sum((pred == c) & (gold != c)), np.sum((pred != c) & (gold == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def paired(CI, a, b, eid, mask, nb, higher=False):
    m = mask & np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=nb, seed=SEED)
    d = float(a[m].mean() - b[m].mean())
    return {"status": "OK", "delta": d, "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "mean_edit": float(a[m].mean()), "mean_ref": float(b[m].mean()),
            "better": bool(r["separated"] and ((d > 0) if higher else (d < 0))), "worse": bool(r["separated"] and ((d < 0) if higher else (d > 0))), "n_windows": int(m.sum())}


def f1_paired(CI, pa, pb, gold, eid, mask, nb):
    classes = sorted(set(np.unique(gold[mask])) | set(np.unique(pa[mask])) | set(np.unique(pb[mask])))
    uniq, idx = CI.episode_index(eid[mask])
    pa_, pb_, g_ = pa[mask], pb[mask], gold[mask]
    d = [macro_f1(pa_[s], g_[s], classes) - macro_f1(pb_[s], g_[s], classes) for s in CI._draws(uniq, idx, nb, SEED)]
    pt = macro_f1(pa_, g_, classes) - macro_f1(pb_, g_, classes)
    lo, hi = [float(x) for x in np.percentile(d, [2.5, 97.5])]
    return {"status": "OK", "delta": pt, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "mean_edit": macro_f1(pa_, g_, classes), "mean_ref": macro_f1(pb_, g_, classes), "n_windows": int(mask.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=B.BIAS + "/RESULT_BIASEDIT.json")
    ap.add_argument("--verdict", default=B.BIAS + "/VERDICT_BIASEDIT.md")
    ap.add_argument("--n-boot", type=int, default=NB)
    a = ap.parse_args()
    t0 = time.time()
    CI = SR.load_ci()
    tags = ["NULL_s0", "NULL_rep", "G_s0", "G2_s0", "NULL_s1", "G_s1", "G2_s1"] + B.EDIT_TAGS
    arms = {t: arm_spec(t) for t in tags}
    prob = []
    miss = [t for t in tags if not (os.path.isdir(arms[t]["dump"]) and os.path.isfile(arms[t]["stopdec"]) and os.path.isfile(arms[t]["done"]))]
    rep = {"tool": "bias_read.py", "card": "PREREG_V8F_PROOF_ADD4.md (frozen)", "evidence_class": "MEASURED (ours; T1-kin dumps, zero training)", "tier": "T1-kin, eval-141", "STRATEGIC": "PAUSED (PI item 25)",
           "estimator": f"taniteval.ci paired / episode-cluster bootstrap, B={a.n_boot}, seed {SEED}; answers another draw of EPISODES only", "controls": {"problems": prob}}
    if miss:
        rep["outcome"] = "NOT-YET"
        rep["reason"] = f"dump / STOPDEC / done file missing for {miss}"
        json.dump(rep, open(a.out, "w"), indent=1)
        print("NOT-YET", miss)
        return 3
    ref = SR.load_dump(W.REF)
    geom = SR.build_geom(ref)
    D, S, thr = {}, {}, {}
    for t in tags:
        D[t] = SR.load_dump(arms[t]["dump"])
        for k in ("eid", "clip_index", "ws", "g", "v0"):
            if not np.array_equal(D[t][k], ref[k]):
                prob.append(f"{t}: dump array {k} differs from the banked DAG-OACC dump")
        S[t] = SR.align_stopdec(dict(np.load(arms[t]["stopdec"])), D[t])
        thr[t] = json.load(open(arms[t]["done"], encoding="utf-8")).get("threads")
        if os.path.isfile(arms[t]["stopdec_json"]):
            prob += [f"{t}: {x}" for x in SR.check_instrument(json.load(open(arms[t]["stopdec_json"], encoding="utf-8"))["controls"][t])]
        else:
            prob.append(f"{t}: STOPDEC json missing")
    rep["controls"]["threads_by_tag"] = thr
    if len({str(v) for v in thr.values()}) != 1:
        prob.append(f"dumps not from one environment (thread counts differ): {thr}")
    ok0, sp0 = SR.window_sets(S["NULL_s0"])
    for t in tags:
        ok, sp = SR.window_sets(S[t])
        if not (np.array_equal(ok, ok0) and np.array_equal(sp, sp0)):
            prob.append(f"{t}: window sets differ from NULL_s0")
    eid = ref["eid"]
    gs = {t: S[t]["goal_prob"][:, SR.STOP_GI] for t in tags}
    dec = {t: gs[t] >= SR.P_STOP_MIN for t in tags}
    R = {t: SR.decision_rates(CI, dec[t], eid, sp0, ok0, a.n_boot) for t in tags}
    gap = {m: abs(R["NULL_rep"][m]["point"] - R["NULL_s0"][m]["point"]) for m in ("hit", "fpr")}
    rep["replicate_gap"] = gap
    rep["rates"] = R
    # ---- controls e1 (re-read), e2, e3
    br = json.load(open(B.BIAS + "/BUILD_REPORT.json", encoding="utf-8"))
    e1 = {t: v["e1"]["pass"] for t, v in br["written"].items()}
    ctl = {"e1_from_BUILD_REPORT": e1}
    if not all(e1.values()):
        prob.append("e1 failed in BUILD_REPORT.json")
    sel = ok0 | sp0
    lab = sp0[sel]
    for t in B.EDIT_TAGS:
        par = B.parent_of(t)
        a_e, a_p = SR.auroc(gs[t][sel], lab), SR.auroc(gs[par][sel], lab)
        ctl[f"e2_auroc_{t}"] = {"edit": a_e, "parent": a_p, "abs_diff": abs(a_e - a_p), "pass": bool(abs(a_e - a_p) <= 1e-9)}
        if not ctl[f"e2_auroc_{t}"]["pass"]:
            prob.append(f"e2: AUROC of {t} differs from {par} by {abs(a_e - a_p):.3e}")
    for s in ("s0", "s1"):
        t, par = f"GE0_{s}", B.PARENTS[s]
        dump_eq = {k: bool(np.array_equal(D[t][k], D[par][k])) for k in ("cl", "ol", "fl", "g", "v0")}
        sd_eq = {k: bool(np.array_equal(S[t][k], S[par][k])) for k in ("goal_prob", "lon_prob", "dec_goal_stop")}
        ctl[f"e3_{t}"] = {"dump_bit_identical": dump_eq, "stopdec_bit_identical": sd_eq, "pass": all(dump_eq.values()) and all(sd_eq.values())}
        if not ctl[f"e3_{t}"]["pass"]:
            prob.append(f"e3: GE0_{s} is not bit-identical to {par}: {dump_eq} {sd_eq}")
    rep["controls"].update(ctl)
    # ---- b_match, and per (seed, dose) pair classes
    pairs = {}
    nn = ok0 & ~sp0
    for s in ("s0", "s1"):
        par = B.PARENTS[s]
        p = gs[par]
        for dose, ge, tr, b in (("dose1", f"GE1_{s}", B.TRAINED[s]["G"], B.B1), ("dose2", f"GE2_{s}", B.TRAINED[s]["G2"], B.B2)):
            ft, ht = R[tr]["fpr"]["point"], R[tr]["hit"]["point"]
            fp = np.array([float((shift(p, x) >= 0.5)[nn].mean()) for x in GRID])
            dfp = np.abs(fp - ft)
            near = np.flatnonzero(dfp <= dfp.min() + 1e-15)
            bm = float(GRID[near[len(near) // 2]])
            hm = float((shift(p, bm) >= 0.5)[sp0].mean())
            fm = float((shift(p, bm) >= 0.5)[nn].mean())
            eq = {m: bool(abs(R[ge][m]["point"] - R[tr][m]["point"]) <= gap[m]) for m in ("hit", "fpr")}
            ah, af = float((shift(p, b) >= 0.5)[sp0].mean()), float((shift(p, b) >= 0.5)[nn].mean())
            pairs[f"{dose}_{s}"] = {"edit": ge, "trained": tr, "b": b, "edit_hit": R[ge]["hit"]["point"], "edit_fpr": R[ge]["fpr"]["point"], "trained_hit": ht, "trained_fpr": ft, "equal_within_gap": eq,
                                    "equals": bool(all(eq.values())), "b_match": bm, "b_match_plateau_n_grid_points": int(len(near)), "hit_at_b_match": hm, "fpr_at_b_match": fm,
                                    "hit_gap_at_b_match": abs(hm - ht), "matches_roc": bool(abs(hm - ht) <= gap["hit"]),
                                    "offset_arithmetic_hit_fpr": [ah, af], "edit_equals_offset_arithmetic": bool(abs(R[ge]["hit"]["point"] - ah) < 1e-12 and abs(R[ge]["fpr"]["point"] - af) < 1e-12)}
    rep["pairs"] = pairs
    if all(v["equals"] for v in pairs.values()):
        out = "BIAS-EQUALS-DOSE"
    elif all(v["matches_roc"] for v in pairs.values()):
        out = "BIAS-MATCHES-ROC"
    elif not any(v["matches_roc"] for v in pairs.values()):
        out = "BIAS-DIFFERS"
    else:
        out = "MIXED (card silent): " + ", ".join(f"{k}:{'match' if v['matches_roc'] else 'differs'}" for k, v in pairs.items())
    rep["outcome"] = out
    # ---- four families, GE2 vs parent and vs trained G2
    g = ref["g"]
    glat, glon = geom["G"].labels(g)
    fam = {}
    allw = np.ones(len(eid), bool)
    for s in ("s0", "s1"):
        ge = f"GE2_{s}"
        X = {t: D[t]["cl"] for t in (ge, B.PARENTS[s], B.TRAINED[s]["G2"])}
        cells = {t: W.cells_of(X[t], g, geom, glat, glon) for t in X}
        lonp = {t: geom["arm_arrays"](X[t], g, geom["J"])["_lon"] for t in X}
        for refname, rt in (("parent", B.PARENTS[s]), ("trained_G2", B.TRAINED[s]["G2"])):
            rows = {}
            for cn in cells[ge]:
                higher = cn in W.HIGHER
                for wname, mask in (("all", allw), ("stop_point", sp0)):
                    if wname == "stop_point" and cn not in ("tactical_2_6s:ade_m", "full_0_6s:ade_m", "tactical_2_6s:speed_mae_mps", "full_0_6s:speed_mae_mps"):
                        continue
                    rows[f"{cn}|{wname}"] = {"family": W.FAM.get(cn.split(":")[-1], W.FAM.get(cn, "?")), **paired(CI, cells[ge][cn], cells[rt][cn], eid, mask, a.n_boot, higher)}
            rows["TAC-LON-macroF1|all"] = {"family": "TACTICAL", **f1_paired(CI, lonp[ge], lonp[rt], glon, eid, allw, a.n_boot)}
            fam[f"{ge}_vs_{refname}"] = rows
    rep["families_GE2"] = fam
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    open(a.verdict, "w", encoding="utf-8").write(render(rep))
    print("WROTE", a.out, "outcome:", out, "problems:", len(prob), "wall_s", rep["wall_s"])
    for p_ in prob:
        print("CONTROL PROBLEM:", p_)
    return 0 if not prob else 2


def render(rep):
    f = lambda x, n=4: "n/a" if x is None else f"{x:.{n}f}"
    L = ["# VERDICT BIASEDIT (addendum 4: stop-weighting lever as a one-scalar weight edit); MEASURED, T1-kin first tick (diagnostic), eval-141", "",
         f"Outcome: **{rep['outcome']}**", f"Estimator: {rep['estimator']}", f"Controls: {len(rep['controls']['problems'])} problem(s): " + ("; ".join(rep["controls"]["problems"]) or "none"),
         f"Replicate gap |NULL_rep - NULL_s0|: hit {f(rep['replicate_gap']['hit'])}, fpr {f(rep['replicate_gap']['fpr'])} (one replicate pair, seed 0; used for both seeds).", "", "STRATEGIC family: PAUSED (PI item 25).", "",
         "## STOPDEC hit / false-positive rate at p >= 0.5", "", "| tag | hit [CI] | fpr [CI] |", "|---|---|---|"]
    for t, r in rep["rates"].items():
        L.append(f"| {t} | {f(r['hit']['point'])} [{f(r['hit']['lo'])}, {f(r['hit']['hi'])}] | {f(r['fpr']['point'])} [{f(r['fpr']['lo'])}, {f(r['fpr']['hi'])}] |")
    L += ["", "## Edit vs trained, and b_match", "", "| pair | edit hit / fpr | trained hit / fpr | equal within gap (hit, fpr) | b (ln W) | b_match | hit / fpr at b_match | matches ROC (hit within gap) |", "|---|---|---|---|---|---|---|---|"]
    for k, v in rep["pairs"].items():
        L.append(f"| {k} | {f(v['edit_hit'])} / {f(v['edit_fpr'])} | {f(v['trained_hit'])} / {f(v['trained_fpr'])} | {v['equal_within_gap']['hit']}, {v['equal_within_gap']['fpr']} | {f(v['b'], 3)} | {f(v['b_match'], 3)} | {f(v['hit_at_b_match'])} / {f(v['fpr_at_b_match'])} | {v['matches_roc']} |")
    L += ["", "## Controls", "", "```", json.dumps({k: v for k, v in rep["controls"].items() if k != "problems"}, indent=1, default=float)[:6000], "```", "",
          "## T1-kin four families for GOALEDIT(b2) (edit minus reference; paired episode-cluster bootstrap)", ""]
    for name, rows in rep["families_GE2"].items():
        L += [f"### {name}", "", "| family | cell | windows | delta [CI] | edit / ref | flag |", "|---|---|---|---|---|---|"]
        for cn, v in rows.items():
            if v.get("status") != "OK" or not (cn.startswith(("full_0_6s", "tactical_2_6s", "TAC"))):
                continue
            flag = "better" if v.get("better") else ("WORSE" if v.get("worse") else ("separated" if v["separated"] else ""))
            L.append(f"| {v['family']} | {cn} | {v['n_windows']} | {f(v['delta'])} [{f(v['lo'])}, {f(v['hi'])}] | {f(v['mean_edit'])} / {f(v['mean_ref'])} | {flag} |")
        L += ["| STRATEGIC | | | PAUSED (PI item 25) | | |", ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
