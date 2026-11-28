r"""wsp_read.py -- READ of H-V8F-PROOF-1 (SOUP) and H-V8F-PROOF-2 P2b (dose linearity) from T1-kin dumps and STOPDEC files (PREREG_V8F_PROOF.md, frozen; sha256 in PREREG_V8F_PROOF.sha256).
CPU, zero training, no model pass.  Tier T1-kin (self-action open loop, m = 10, causal speed bucket, dead-reckoned nav, logged scene; deterministic planner), eval-141: a planner diagnostic, never driving performance.
Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B = 2000, seed 0 ("another draw of EPISODES" only; "another training run" needs replicates: here one pair of members, one soup).
The first dose's package (theory/stopbal/read, imported by path, not edited) supplies the loaders, geometry, window sets, decision rates and instrument controls.

H1 (SOUP).  arms: a = NULL_s0, b = NULL_rep (members), SOUP2 = 0.5*(theta_a + theta_b); controls SOUPa0 / SOUPa1 (alpha 0 / 1) must reproduce the member dumps bit for bit.
  Delta per window = soup minus the mean of the two members; primary cells: tactical band (knots 11-30, 2.2-6.0 s) ADE and full 0-6 s ADE.
  G_ens (VC-G1, exact): for squared error  err(mean of outputs) = mean member squared error - 0.25*||y_a - y_b||^2, so G_ens_sq = 0.25*mean||y_a-y_b||^2 (checked window by window against the directly
  computed ensemble error).  ADE is not a squared error: its output-ensemble gain is computed EXACTLY by averaging the two members' plans per window and scoring that plan (convexity gives G_ens_ade >= 0).
  G_soup = member-mean error - soup error (same definition); kappa = G_soup / G_ens, point = ratio of means, interval = the same episode-cluster draws (seed 0, B 2000).
  Outcome (card): SOUP-REALISES = delta < 0 with CI excluding 0 on BOTH primaries and kappa >= 0.5; SOUP-PARTIAL = CI excludes 0 (delta < 0), kappa < 0.5; SOUP-NULL = not separated;
  SOUP-BARRIER = soup worse than the member mean with CI excluding 0.  READING CHOICES the card leaves open (none moves a bar): (i) kappa for the outcome is read on ADE (the primary metric);
  squared-error kappa is printed beside; (ii) REALISES needs kappa >= 0.5 on both primaries, PARTIAL = both primaries separated-better with kappa < 0.5 on at least one;
  (iii) BARRIER = a primary separated-worse and none separated-better; NULL = neither primary separated; every other mixture is printed as UNSPECIFIED-MIXTURE with its per-primary classes (the card is silent).
  Four families per band (PRIMARY ADE/FDE, LONGITUDINAL, LATERAL, TACTICAL accuracy), per cell: delta, CI, replicate gap |NULL_rep - NULL_s0|, cost / gain flags (separated on the worse / better side);
  lateral and tactical must not carry a separated cost.  STRATEGIC: PAUSED (PI item 25), always printed, never a pass, never omitted.
H2 (P2b).  STOPDEC goal decision p(STOP_POINT) >= 0.5: hit rate on the stop-point windows (117), false-positive rate on the other windows with a label (2,234); tags NULL_s0, NULL_rep, G_s0 (= BAL-G, dose 1),
  G2_s0 (= BAL-G2, dose 2), DOSE050, DOSE100.  DOSE-LINEAR iff theta(0.5) reads hit AND fpr between the NULL_s0 and G2_s0 values AND within the replicate gap |NULL_rep - NULL_s0| (that metric) of the
  TRAINED dose-1 arm G_s0; DOSE-NONLINEAR otherwise.  Control: DOSE100 decisions equal G2_s0's.

Usage: python -I wsp_read.py [--out RESULT_WSP.json] [--verdict VERDICT_WSP.md] [--n-boot 2000] [--map arms.json] [--only h1|h2]
  default map: dumps wsp/t1/dump_<TAG>_m10_causal_heads, STOPDEC wsp/stopdec/stopdec_<TAG>.npz (+ STOPDEC_<TAG>.json).   --map overrides: {TAG: {"dump": dir, "stopdec": npz, "stopdec_json": json, "stopdec_tag": tag}}."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wsp_common as C  # noqa: E402

os.environ["STOPBAL_TREE"] = C.TREE
C.use_tree()
THEORY = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
sys.path.insert(0, THEORY + "/stopbal/read")
sys.path.insert(0, THEORY + "/stopbal/dose2")
import numpy as np  # noqa: E402
import stopbal_read as SR  # noqa: E402
import read_lib_stopbal as BL  # noqa: E402
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(C.TREE), tanitad.__file__
REF = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads"
NB, SEED = 2000, 0
BANDS3 = {"operative_0_2s": (0, 10), "tactical_2_6s": (10, 30), "full_0_6s": (0, 30)}
PRIMARY = ("tactical_2_6s:ade_m", "full_0_6s:ade_m")
FAM = dict(SR.FAMILY)
FAM.update({"dk_hw_abs_err_m": "LONGITUDINAL", "dk_tg_abs_err_s": "LONGITUDINAL", "TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"})
HIGHER = ("TAC-LAT", "TAC-LON")


def default_map():
    m = {}
    for t in C.DUMP_TAGS:
        m[t] = {"dump": f"{C.T1}/dump_{t}_m10_causal_heads", "stopdec": f"{C.STOPDEC_OUT}/stopdec_{t}.npz", "stopdec_json": f"{C.STOPDEC_OUT}/STOPDEC_{t}.json", "stopdec_tag": t}
    return m


def cells_of(X, G, geom, glat, glon):
    """{cell: [N]} over three bands x the families' metrics (operative / tactical from stopbal_geom.arm_arrays, full 0-6 s from the band slicer) + the two TAC accuracies."""
    A = geom["arm_arrays"](X, G, geom["J"])
    out = {f"{b}:{m}": np.asarray(A[b][m], np.float64) for b in ("operative_0_2s", "tactical_2_6s") for m in geom["metrics"]}
    full = BL.band_rows(X, G, 0, 30)
    for m, v in full.items():
        out[f"full_0_6s:{m}"] = np.asarray(v, np.float64)
    out["TAC-LAT"] = (A["_lat"] == glat).astype(np.float64)
    out["TAC-LON"] = (A["_lon"] == glon).astype(np.float64)
    return out


def band_err(X, G, a, b):
    """per-window (ADE, squared-error mean) over knots [a, b): ADE = mean ||x - g||, SQ = mean ||x - g||^2."""
    e = np.linalg.norm(np.asarray(X, np.float64)[:, a:b] - np.asarray(G, np.float64)[:, a:b], axis=-1)
    return e.mean(1), (e ** 2).mean(1)


def fin(*xs):
    m = np.ones(len(xs[0]), bool)
    for x in xs:
        m &= np.isfinite(x)
    return m


def cell_vs_mean(CI, soup, ma, mb, eid, name, nb):
    base = 0.5 * (ma + mb)
    m = fin(soup, ma, mb)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.paired_episode_cluster_bootstrap(soup[m], base[m], eid[m], n_boot=nb, seed=SEED)
    higher = name in HIGHER
    pt = float(soup[m].mean() - base[m].mean())
    better, worse = (pt > 0, pt < 0) if higher else (pt < 0, pt > 0)
    return {"status": "OK", "delta": pt, "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]), "gain": bool(r["separated"] and better), "cost": bool(r["separated"] and worse),
            "replicate_gap": float(abs(mb[m].mean() - ma[m].mean())), "mean_soup": float(soup[m].mean()), "mean_member_mean": float(base[m].mean()), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def gain_block(CI, mem, ens, soup, eid, nb):
    """G_ens = mem - ens, G_soup = mem - soup (per-window arrays), kappa = mean(G_soup) / mean(G_ens); intervals from the SAME episode draws (CI._draws, seed 0)."""
    ge, gs = mem - ens, mem - soup
    uniq, idx = CI.episode_index(eid)
    k, e_, s_ = [], [], []
    for sel in CI._draws(uniq, idx, nb, SEED):
        x, y = ge[sel].mean(), gs[sel].mean()
        e_.append(x)
        s_.append(y)
        k.append(y / x if x > 0 else np.nan)
    q = lambda v: [float(t) for t in np.nanpercentile(v, [2.5, 97.5])]
    return {"G_ens": float(ge.mean()), "G_ens_ci": q(e_), "G_soup": float(gs.mean()), "G_soup_ci": q(s_), "kappa": float(gs.mean() / ge.mean()) if ge.mean() > 0 else None, "kappa_ci": q(k),
            "mean_member_err": float(mem.mean()), "ens_err": float(ens.mean()), "soup_err": float(soup.mean())}


def class_of(c):
    if c["status"] != "OK":
        return "NOT-READ"
    return "BETTER" if c["gain"] else ("WORSE" if c["cost"] else "NOT-SEPARATED")


def h1_outcome(prim, kap):
    cls = {n: class_of(prim[n]) for n in PRIMARY}
    if "NOT-READ" in cls.values():
        return "NOT-READ", cls
    v = list(cls.values())
    if all(x == "BETTER" for x in v):
        return ("SOUP-REALISES" if all(kap[n] is not None and kap[n] >= 0.5 for n in PRIMARY) else "SOUP-PARTIAL"), cls
    if all(x == "NOT-SEPARATED" for x in v):
        return "SOUP-NULL", cls
    if "WORSE" in v and "BETTER" not in v:
        return "SOUP-BARRIER", cls
    return "UNSPECIFIED-MIXTURE (card silent)", cls


def read_h1(CI, arms, geom, ref, nb, rep):
    prob = rep["controls"]["problems"]
    tags = ("NULL_s0", "NULL_rep", "SOUP2", "SOUPa0", "SOUPa1")
    miss = [t for t in tags if not os.path.isdir(arms.get(t, {}).get("dump", "/nonexistent"))]
    if miss:
        rep["H1"] = {"outcome": "NOT-YET", "reason": f"dumps missing for {miss}"}
        return
    D = {t: SR.load_dump(arms[t]["dump"]) for t in tags}
    for t, d in D.items():
        for k in ("eid", "clip_index", "ws", "g", "fl", "v0"):
            if not np.array_equal(d[k], ref[k]):
                prob.append(f"control 3: {t}: window or model-free array differs from the banked DAG-OACC dump: {k}")
    # c1 at dump level: alpha 0 / 1 reproduce the member dumps bit for bit
    c1 = {}
    for s, mem in (("SOUPa0", "NULL_s0"), ("SOUPa1", "NULL_rep")):
        c1[s] = {k: bool(np.array_equal(D[s][k], D[mem][k])) for k in ("cl", "ol", "fl", "g", "v0")}
        if not all(c1[s].values()):
            prob.append(f"c1: {s} dump is not bit-identical to {mem}: {c1[s]}")
    rep["controls"]["c1_dump_level"] = c1
    # c3: one environment: thread counts of the done files equal
    thr = {}
    for t in tags:
        done = os.path.join(os.path.dirname(arms[t]["dump"]), f"done_{os.path.basename(arms[t]['dump'])[len('dump_'):]}.txt")
        thr[t] = json.load(open(done, encoding="utf-8")).get("threads") if os.path.isfile(done) else "no done file"
    rep["controls"]["c3_threads_by_tag"] = thr
    if len({str(v) for v in thr.values()}) != 1 or "no done file" in thr.values():
        prob.append(f"c3: dumps not from one environment / missing done files: {thr}")
    g = D["NULL_s0"]["g"]
    eid = D["NULL_s0"]["eid"]
    glat, glon = geom["G"].labels(g)
    X = {t: D[t]["cl"] for t in tags}
    ens = 0.5 * (X["NULL_s0"] + X["NULL_rep"])
    cells = {t: cells_of(X[t], g, geom, glat, glon) for t in ("NULL_s0", "NULL_rep", "SOUP2")}
    cells["ENS_plan"] = cells_of(ens, g, geom, glat, glon)
    fam = {}
    for cn in cells["SOUP2"]:
        base = cn.split(":")[-1]
        fam[cn] = {"family": FAM.get(base, FAM.get(cn, "?")), **cell_vs_mean(CI, cells["SOUP2"][cn], cells["NULL_s0"][cn], cells["NULL_rep"][cn], eid, cn if cn in HIGHER else base, nb)}
    # ---- G_ens / G_soup / kappa on ADE and squared error, per band; VC-G1 identity control
    gains, ident = {}, {}
    for bn, (a, b) in BANDS3.items():
        ea, qa = band_err(X["NULL_s0"], g, a, b)
        eb, qb = band_err(X["NULL_rep"], g, a, b)
        ee, qe = band_err(ens, g, a, b)
        es, qs = band_err(X["SOUP2"], g, a, b)
        msd = (np.linalg.norm(X["NULL_s0"][:, a:b] - X["NULL_rep"][:, a:b], axis=-1) ** 2).mean(1)
        ident[bn] = float(np.max(np.abs(qe - (0.5 * (qa + qb) - 0.25 * msd))))
        gains[bn] = {"ADE": gain_block(CI, 0.5 * (ea + eb), ee, es, eid, nb), "SQ": gain_block(CI, 0.5 * (qa + qb), qe, qs, eid, nb),
                     "VC_G1_identity_max_abs_residual": ident[bn], "G_ens_sq_closed_form": float(0.25 * msd.mean())}
        if ident[bn] > 1e-8:
            prob.append(f"control VC-G1: identity residual {ident[bn]:.3e} in {bn}")
    prim = {n: fam[n] for n in PRIMARY}
    kap = {"tactical_2_6s:ade_m": gains["tactical_2_6s"]["ADE"]["kappa"], "full_0_6s:ade_m": gains["full_0_6s"]["ADE"]["kappa"]}
    out, cls = h1_outcome(prim, kap)
    costs = {k: v for k, v in fam.items() if v.get("cost")}
    lat_tac_costs = [k for k, v in costs.items() if v["family"] in ("LATERAL", "TACTICAL")]
    rep["H1"] = {"outcome": out, "per_primary_class": cls, "primary_cells": prim, "kappa_ade": kap, "kappa_sq": {b: gains[b]["SQ"]["kappa"] for b in gains}, "gains": gains,
                 "families": fam, "named_costs": sorted(costs), "named_gains": sorted(k for k, v in fam.items() if v.get("gain")),
                 "lateral_or_tactical_separated_cost": lat_tac_costs, "STRATEGIC": "PAUSED (PI item 25)",
                 "members_vs_soup_primary_means": {n: {"NULL_s0": float(cells["NULL_s0"][n].mean()), "NULL_rep": float(cells["NULL_rep"][n].mean()), "SOUP2": float(cells["SOUP2"][n].mean()),
                                                      "ENS_plan": float(cells["ENS_plan"][n].mean())} for n in PRIMARY}}


def read_h2(CI, arms, ref, nb, rep):
    prob = rep["controls"]["problems"]
    tags = C.H2_TAGS
    miss = [t for t in tags if not (os.path.isdir(arms.get(t, {}).get("dump", "/x")) and os.path.isfile(arms.get(t, {}).get("stopdec", "/x")))]
    if miss:
        rep["H2"] = {"outcome": "NOT-YET", "reason": f"dump or STOPDEC missing for {miss}"}
        return
    R, S = {}, {}
    for t in tags:
        d = SR.load_dump(arms[t]["dump"])
        for k in ("eid", "clip_index", "ws", "g", "v0"):
            if not np.array_equal(d[k], ref[k]):
                prob.append(f"control 3: {t}: dump array {k} differs from the banked DAG-OACC dump")
        sd = SR.align_stopdec(dict(np.load(arms[t]["stopdec"])), d)
        S[t] = sd
        sj = arms[t].get("stopdec_json")
        if sj and os.path.isfile(sj):
            ctl = json.load(open(sj, encoding="utf-8"))["controls"][arms[t].get("stopdec_tag", t)]
            prob += [f"{t}: {x}" for x in SR.check_instrument(ctl)]
        else:
            prob.append(f"control 3: {t}: STOPDEC json missing")
    sets = {t: SR.window_sets(S[t]) for t in tags}
    ok0, sp0 = sets["NULL_s0"]
    for t, (ok, sp) in sets.items():
        if not (np.array_equal(ok, ok0) and np.array_equal(sp, sp0)):
            prob.append(f"control 3: {t}: window sets differ from NULL_s0")
    cnt = {"n_windows": int(len(ok0)), "n_has": int(np.asarray(S["NULL_s0"]["has"]).sum()), "n_stop_point": int(sp0.sum()), "n_ok": int(ok0.sum())}
    if (cnt["n_windows"], cnt["n_has"], cnt["n_stop_point"]) != (SR.EXPECT["n_windows"], SR.EXPECT["n_has"], SR.EXPECT["n_stop_point"]):
        prob.append(f"control 3: window counts {cnt} differ from {SR.EXPECT}")
    eid = ref["eid"]
    dec = {t: S[t]["goal_prob"][:, SR.STOP_GI] >= SR.P_STOP_MIN for t in tags}
    for t in tags:
        if not np.array_equal(dec[t], np.asarray(S[t]["dec_goal_stop"], bool)):
            prob.append(f"control 3: {t}: decisions differ from the npz's own dec_goal_stop")
        R[t] = SR.decision_rates(CI, dec[t], eid, sp0, ok0, nb)
    gap = {m: abs(R["NULL_rep"][m]["point"] - R["NULL_s0"][m]["point"]) for m in ("hit", "fpr")}
    rows, ok = {}, {}
    for m in ("hit", "fpr"):
        v0, v2, v1, vh = R["NULL_s0"][m]["point"], R["G2_s0"][m]["point"], R["G_s0"][m]["point"], R["DOSE050"][m]["point"]
        between = min(v0, v2) <= vh <= max(v0, v2)
        near = abs(vh - v1) <= gap[m]
        rows[m] = {"NULL_s0": v0, "G2_s0_dose2": v2, "G_s0_dose1_trained": v1, "DOSE050": vh, "replicate_gap": gap[m], "between_NULL_and_G2": bool(between), "abs_diff_to_trained_dose1": abs(vh - v1),
                   "within_gap_of_trained_dose1": bool(near), "pass": bool(between and near)}
        ok[m] = rows[m]["pass"]
    ctrl = {"DOSE100_decisions_equal_G2_s0": bool(np.array_equal(dec["DOSE100"], dec["G2_s0"])), "DOSE100_goal_prob_max_abs_diff_to_G2_s0": float(np.max(np.abs(S["DOSE100"]["goal_prob"] - S["G2_s0"]["goal_prob"])))}
    if not ctrl["DOSE100_decisions_equal_G2_s0"]:
        prob.append("control P2b: DOSE100 decisions differ from BAL-G2_s0")
    banked = {}
    for t, p in (("NULL_s0", THEORY + "/stopbal/read/out/stopdec/stopdec_BAL-NULL_s0.npz"), ("NULL_rep", THEORY + "/stopbal/read/out/stopdec/stopdec_BAL-NULL_rep.npz"),
                 ("G_s0", THEORY + "/stopbal/read/out/stopdec/stopdec_BAL-G_s0.npz"), ("G2_s0", THEORY + "/stopbal/dose2/read/out/stopdec/stopdec_BAL-G2_s0.npz")):
        if os.path.isfile(p):
            z = np.load(p)
            banked[t] = {"decisions_equal_banked": bool(np.array_equal(np.asarray(z["dec_goal_stop"], bool)[np.lexsort((z["ws"], z["clip_index"]))],
                                                                      dec[t][np.lexsort((S[t]["ws"], S[t]["clip_index"]))])), "banked_file": p}
    rep["H2"] = {"outcome": "DOSE-LINEAR" if all(ok.values()) else "DOSE-NONLINEAR", "rates": {t: R[t] for t in tags}, "criteria": rows, "controls": ctrl, "windows": cnt,
                 "own_dumps_vs_banked_stopdec_decisions": banked, "STRATEGIC": "PAUSED (PI item 25)"}


def render(rep):
    f = lambda x, n=4: "n/a" if x is None else f"{x:.{n}f}"
    L = ["# VERDICT WSP (H-V8F-PROOF-1 SOUP, H-V8F-PROOF-2 P2b), generated from the JSON; MEASURED, T1-kin, eval-141", "",
         f"Estimator: {rep['estimator']}", f"Controls: {len(rep['controls']['problems'])} problem(s): " + ("; ".join(rep["controls"]["problems"]) or "none"), "", "STRATEGIC family: PAUSED (PI item 25).", ""]
    h = rep.get("H1", {})
    L.append(f"## H1 SOUP: **{h.get('outcome')}**")
    if "primary_cells" in h:
        L += ["| primary cell | members (s0 / rep) | member mean -> soup | delta [CI] | class | kappa (ADE) [CI] |", "|---|---|---|---|---|---|"]
        for n in PRIMARY:
            c, m = h["primary_cells"][n], h["members_vs_soup_primary_means"][n]
            bn = n.split(":")[0]
            k = h["gains"][bn]["ADE"]
            L.append(f"| {n} | {f(m['NULL_s0'])} / {f(m['NULL_rep'])} | {f(c['mean_member_mean'])} -> {f(c['mean_soup'])} | {f(c['delta'])} [{f(c['lo'])}, {f(c['hi'])}] | {h['per_primary_class'][n]} | {f(k['kappa'], 3)} [{f(k['kappa_ci'][0], 3)}, {f(k['kappa_ci'][1], 3)}] |")
        L += ["", "| band | metric | G_ens | G_soup | kappa [CI] | VC-G1 residual |", "|---|---|---|---|---|---|"]
        for bn, gb in h["gains"].items():
            for kind in ("ADE", "SQ"):
                k = gb[kind]
                L.append(f"| {bn} | {kind} | {f(k['G_ens'], 5)} | {f(k['G_soup'], 5)} | {f(k['kappa'], 3)} [{f(k['kappa_ci'][0], 3)}, {f(k['kappa_ci'][1], 3)}] | {gb['VC_G1_identity_max_abs_residual']:.1e} |")
        L += ["", f"Named costs (soup minus member mean, separated worse): {h['named_costs'] or 'none'}; lateral / tactical separated costs: {h['lateral_or_tactical_separated_cost'] or 'none'}.", f"Named gains: {h['named_gains'] or 'none'}.",
              "Four families (soup minus member mean; delta [CI]; replicate gap):", "", "| family | cell | delta [CI] | gap |", "|---|---|---|---|"]
        for cn, v in h["families"].items():
            if v.get("status") == "OK":
                L.append(f"| {v['family']} | {cn} | {f(v['delta'])} [{f(v['lo'])}, {f(v['hi'])}]{' COST' if v['cost'] else (' gain' if v['gain'] else '')} | {f(v['replicate_gap'])} |")
        L.append("| STRATEGIC | | PAUSED (PI item 25) | |")
    else:
        L.append(str(h.get("reason")))
    h = rep.get("H2", {})
    L += ["", f"## H2 P2b dose linearity: **{h.get('outcome')}**"]
    if "criteria" in h:
        L += ["| metric | NULL_s0 | G2_s0 (dose 2) | G_s0 (dose 1, trained) | DOSE050 | replicate gap | between | within gap of dose 1 |", "|---|---|---|---|---|---|---|---|"]
        for m, r in h["criteria"].items():
            L.append(f"| {m} | {f(r['NULL_s0'])} | {f(r['G2_s0_dose2'])} | {f(r['G_s0_dose1_trained'])} | {f(r['DOSE050'])} | {f(r['replicate_gap'])} | {r['between_NULL_and_G2']} | {r['within_gap_of_trained_dose1']} |")
        L += ["", "| tag | hit [CI] | fpr [CI] |", "|---|---|---|"]
        for t, r in h["rates"].items():
            L.append(f"| {t} | {f(r['hit']['point'])} [{f(r['hit']['lo'])}, {f(r['hit']['hi'])}] | {f(r['fpr']['point'])} [{f(r['fpr']['lo'])}, {f(r['fpr']['hi'])}] |")
        L += ["", f"Control DOSE100 = G2_s0: {h['controls']}", f"Own dumps vs banked STOPDEC decisions: {h['own_dumps_vs_banked_stopdec_decisions']}"]
    else:
        L.append(str(h.get("reason")))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=C.WSP + "/RESULT_WSP.json")
    ap.add_argument("--verdict", default=C.WSP + "/VERDICT_WSP.md")
    ap.add_argument("--n-boot", type=int, default=NB)
    ap.add_argument("--map")
    ap.add_argument("--only", choices=("h1", "h2"))
    a = ap.parse_args()
    t0 = time.time()
    CI = SR.load_ci()
    arms = json.load(open(a.map, encoding="utf-8")) if a.map else default_map()
    rep = {"tool": "wsp_read.py", "card": "PREREG_V8F_PROOF.md (frozen)", "evidence_class": "MEASURED (ours; T1-kin dumps, zero training)", "tier": "T1-kin, eval-141",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap, B={a.n_boot}, seed {SEED}; ci.py from {CI.SOURCE}. It answers 'another draw of EPISODES' only; one member pair and one soup give no run-to-run interval",
           "STRATEGIC": "PAUSED (PI item 25)", "tree": C.TREE, "controls": {"problems": []}}
    ref = SR.load_dump(REF)
    geom = SR.build_geom(ref)
    if a.only != "h2":
        read_h1(CI, arms, geom, ref, a.n_boot, rep)
    if a.only != "h1":
        read_h2(CI, arms, ref, a.n_boot, rep)
    rep["wall_s"] = round(time.time() - t0, 1)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    open(a.verdict, "w", encoding="utf-8").write(render(rep))
    print("WROTE", a.out, "H1:", rep.get("H1", {}).get("outcome"), "H2:", rep.get("H2", {}).get("outcome"), "problems:", len(rep["controls"]["problems"]), "wall_s", rep["wall_s"])
    for p in rep["controls"]["problems"]:
        print("CONTROL PROBLEM:", p)
    return 0 if not rep["controls"]["problems"] else 2


if __name__ == "__main__":
    sys.exit(main())
