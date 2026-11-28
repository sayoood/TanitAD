"""Read-out of the ladder PLANTEST / PLANTEST2 run: applies the two cards' committed tables, writes RESULT_LADDER_PLANTEST.json / .md. NO verdict is written (the Master Mind writes it). Tier T0.
Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the episodes of the read set (141 for PLANTEST at c1; the SCORE half for PLANTEST2; the subset with both past frames for c3),
B 2,000, seed 0, ratio of per-episode sums, 95 % percentile (plantest_lib.boot_*), interval-only separation. Tolerates missing units (rows read INCOMPLETE).
Usage: python report_lp.py [--runs DIR] [--out PREFIX] [--smoke]"""
import argparse, glob, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lp_lib as L
import plantest_lib as P
import plantest2_lib as Q

RUNS = "F:/TanitAD-scratch/ladder_plantest/runs"
FAMILIES = ("C0", "R35", "R35H")
FORMS = ("c1", "c3")
FORM_NAME = {"c1": "context 1 (the cards' form)", "c3": "context 3, real t-2 / t-1 history (beside, no bar)"}
FAM = Q.FAMS
CH = {"LONGITUDINAL": 0, "LATERAL": 1}
UNIT = {"acc_mae": "m/s^2", "yaw_mae": "rad/s", "speed_err": "m/s", "heading_err": "rad", "ade": "m"}
BESIDE = ("speed_err", "heading_err", "ade")
_CTX = {}


def lam_s(l):
    return f"{l:g}"


def fmt(t):
    return "-" if t is None else f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


def ctx_of(name):
    if name not in _CTX:
        _CTX[name] = L.prepare_arm(name)
    return _CTX[name]


def planners_of(fam):
    return ("flat1", "td3") if fam == "C0" else ("flat1",)


def ld(runs, kind, *parts):
    p = os.path.join(runs, "_".join((kind,) + tuple(str(x) for x in parts)) + ".npz")
    return dict(np.load(p)) if os.path.exists(p) else None


def seeds_label(labels):
    if any(l is None for l in labels):
        return "INCOMPLETE"
    return labels[0] if len(set(labels)) == 1 else "SEEDS-DISAGREE(" + "|".join(labels) + ")"


def boot_for(eids):
    uniq, Wt, pos = P.boot_setup(eids)
    return uniq, Wt, pos


def read_family(runs, fam, form, out):
    """PLANTEST and PLANTEST2 for one arm family and one rollout form."""
    names = [f"{fam}_s0", f"{fam}_s1"]
    ctxs = {n: ctx_of(n) for n in names}
    c0 = ctxs[names[0]]
    sel = L.sel_arrays(c0, form)
    for n in names[1:]:
        s2 = L.sel_arrays(ctxs[n], form)
        assert all(np.array_equal(sel[k], s2[k]) for k in sel), "window sets differ across seeds"
    res = {"n_windows_all": int(len(sel["all"])), "n_windows_tune": int(len(sel["tune"])), "n_windows_score": int(len(sel["score"]))}
    # ------------------------------------------------------------- PLANTEST (card 1): all windows of the form
    idx = sel["all"]
    uniq, Wt, pos = boot_for(c0.eid[idx])
    res["n_episodes_all"] = int(len(uniq))
    p1 = {}
    for seed, n in enumerate(names):
        c = ctxs[n]
        mets = lambda plan: P.window_metrics(plan, c.A[idx], c.v0[idx])
        M = {"hold": mets(P.hold_plan(c.A[idx])), "prior": mets(P.prior_plan(c.st, len(idx), c.ch1))}
        runs_ = {}
        for pl in planners_of(fam):
            for g in ("true", "rand"):
                r = ld(runs, "p1", n, form, pl, g)
                if r is None:
                    continue
                assert np.array_equal(r["idx"], idx)
                M[f"{pl}_{g}"] = mets(r["plan_raw"].astype(np.float64))
                runs_[f"{pl}_{g}"] = r
        p1[seed] = (M, runs_)
    rows = {}
    for pl in planners_of(fam):
        for famname, q in FAM.items():
            per = []
            for seed in (0, 1):
                M, runs_ = p1[seed]
                if f"{pl}_true" not in M or f"{pl}_rand" not in M:
                    per.append(None)
                    continue
                tr = M[f"{pl}_true"][q]
                d_hold, d_rand = P.boot_mean_diff(tr, M["hold"][q], pos, Wt), P.boot_mean_diff(tr, M[f"{pl}_rand"][q], pos, Wt)
                d_prior = P.boot_mean_diff(tr, M["prior"][q], pos, Wt)
                sh, sr = P.sep_label(*d_hold), P.sep_label(*d_rand)
                row = dict(unit=UNIT[q], hold=P.boot_mean(M["hold"][q], pos, Wt), prior=P.boot_mean(M["prior"][q], pos, Wt), true=P.boot_mean(tr, pos, Wt), rand=P.boot_mean(M[f"{pl}_rand"][q], pos, Wt),
                           true_minus_hold=d_hold, true_minus_rand=d_rand, true_minus_prior=d_prior, sep_hold=sh, sep_rand=sr, label=P.family_label(sh, sr))
                for qq in BESIDE:
                    row[qq] = dict(hold=P.boot_mean(M["hold"][qq], pos, Wt), true=P.boot_mean(M[f"{pl}_true"][qq], pos, Wt), rand=P.boot_mean(M[f"{pl}_rand"][qq], pos, Wt))
                rt = runs_[f"{pl}_true"]
                for lv in (1, 2, 3):
                    if f"mono{lv}" in rt:
                        row[f"monotonicity_L{lv}"] = P.boot_mean(rt[f"mono{lv}"], pos, Wt)
                        row[f"frac_cost_lowered_L{lv}"] = P.boot_mean((rt[f"cost{lv}"] < rt[f"init{lv}"]).astype(float), pos, Wt)
                per.append(row)
                rows[f"{pl}|{famname}|s{seed}"] = row
            rows[f"{pl}|{famname}|label"] = seeds_label([None if r is None else r["label"] for r in per])
    hier = {}
    if fam == "C0":
        for famname, q in FAM.items():
            hr = []
            for seed in (0, 1):
                M, _ = p1[seed]
                if "td3_true" in M and "flat1_true" in M:
                    d = P.boot_mean_diff(M["td3_true"][q], M["flat1_true"][q], pos, Wt)
                    hr.append(dict(seed=seed, td3_minus_flat1=d, sep=P.sep_label(*d)))
            hier[famname] = dict(rows=hr, HIER_ADDS=bool(len(hr) == 2 and all(h["sep"] == "better" for h in hr)), complete=len(hr) == 2)
    res["plantest"] = dict(rows=rows, hier_adds=hier)
    # ------------------------------------------------------------- PLANTEST2 (card 2): SCORE half of the form
    sc = sel["score"]
    uniq2, Wt2, pos2 = boot_for(c0.eid[sc])
    res["n_episodes_score"] = int(len(uniq2))
    rows2, M2 = {}, {}
    for pl in planners_of(fam):
        for famname, q in FAM.items():
            per = []
            for seed, n in enumerate(names):
                c = ctxs[n]
                chp = os.path.join(runs, f"choice_{n}_{form}_{pl}.json")
                if not os.path.exists(chp):
                    per.append(None)
                    continue
                ch = json.load(open(chp))
                lam = ch["choice"][famname]
                rt, rr = ld(runs, "score", n, form, pl, "true", f"L{lam_s(lam)}"), ld(runs, "score", n, form, pl, "rand", f"L{lam_s(lam)}")
                if rt is None or rr is None:
                    per.append(None)
                    continue
                assert np.array_equal(rt["idx"], sc) and np.array_equal(rr["idx"], sc)
                mets = lambda plan: P.window_metrics(plan, c.A[sc], c.v0[sc])
                mh, mt, mr = mets(P.hold_plan(c.A[sc])), mets(rt["plan_raw"].astype(np.float64)), mets(rr["plan_raw"].astype(np.float64))
                d_hold, d_rand = P.boot_mean_diff(mt[q], mh[q], pos2, Wt2), P.boot_mean_diff(mt[q], mr[q], pos2, Wt2)
                sh, sr = P.sep_label(*d_hold), P.sep_label(*d_rand)
                dt = float(Q.hold_distance(rt["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[famname]])
                dr = float(Q.hold_distance(rr["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[famname]])
                row = dict(lam=lam, unit=UNIT[q], hold=P.boot_mean(mh[q], pos2, Wt2), true=P.boot_mean(mt[q], pos2, Wt2), rand=P.boot_mean(mr[q], pos2, Wt2), true_minus_hold=d_hold, true_minus_rand=d_rand,
                           sep_hold=sh, sep_rand=sr, base_label=P.family_label(sh, sr), d_true=dt, d_rand=dr, collapse=bool(dt < Q.COLLAPSE_FRAC), tune_table=ch["tune_table"])
                for qq in BESIDE:
                    row[qq] = dict(hold=P.boot_mean(mh[qq], pos2, Wt2), true=P.boot_mean(mt[qq], pos2, Wt2), rand=P.boot_mean(mr[qq], pos2, Wt2))
                for lv in (1, 2, 3):
                    if f"mono{lv}" in rt:
                        row[f"monotonicity_L{lv}"] = P.boot_mean(rt[f"mono{lv}"], pos2, Wt2)
                        row[f"frac_cost_lowered_L{lv}"] = P.boot_mean((rt[f"cost{lv}"] < rt[f"init{lv}"]).astype(float), pos2, Wt2)
                per.append(row)
                rows2[f"{pl}|{famname}|s{seed}"] = row
                M2[(seed, pl, famname)] = mt
            labs = [None if r is None else r["base_label"] for r in per]
            lab = seeds_label(labs)
            fl = [f"s{i}" for i, r in enumerate(per) if r is not None and r["collapse"]]
            rows2[f"{pl}|{famname}|label"] = lab + (f" +HOLD-COLLAPSE({','.join(fl)})" if fl else "")
    hier2 = {}
    if fam == "C0":
        for famname, q in FAM.items():
            hr = []
            for seed in (0, 1):
                if (seed, "td3", famname) in M2 and (seed, "flat1", famname) in M2:
                    d = P.boot_mean_diff(M2[(seed, "td3", famname)][q], M2[(seed, "flat1", famname)][q], pos2, Wt2)
                    hr.append(dict(seed=seed, td3_minus_flat1=d, sep=P.sep_label(*d), lam_td3=rows2[f"td3|{famname}|s{seed}"]["lam"], lam_flat1=rows2[f"flat1|{famname}|s{seed}"]["lam"]))
            hier2[famname] = dict(rows=hr, HIER_ADDS=bool(len(hr) == 2 and all(h["sep"] == "better" for h in hr)), complete=len(hr) == 2)
    res["plantest2"] = dict(rows=rows2, hier_adds=hier2)
    out.setdefault(fam, {})[form] = res
    return res


def read_c3_minus_c1(runs, fam, out):
    """Beside, no bar: the G-TRUE plan's MAE under context 3 minus context 1 on the windows both read (the windows with both past frames), PLANTEST (all) and PLANTEST2 (SCORE, each form at its own chosen lambda)."""
    rows = {}
    for seed in (0, 1):
        n = f"{fam}_s{seed}"
        c = ctx_of(n)
        s1, s3 = L.sel_arrays(c, "c1"), L.sel_arrays(c, "c3")
        for pl in planners_of(fam):
            for famname, q in FAM.items():
                # PLANTEST
                a1, a3 = ld(RUNS_G, "p1", n, "c1", pl, "true"), ld(RUNS_G, "p1", n, "c3", pl, "true")
                if a1 is not None and a3 is not None:
                    idx = s3["all"]
                    uniq, Wt, pos = boot_for(c.eid[idx])
                    p1_ = a1["plan_raw"].astype(np.float64)[idx]                    # c1 ran on all windows in order: row i = window i
                    m1, m3 = P.window_metrics(p1_, c.A[idx], c.v0[idx])[q], P.window_metrics(a3["plan_raw"].astype(np.float64), c.A[idx], c.v0[idx])[q]
                    d = P.boot_mean_diff(m3, m1, pos, Wt)
                    rows[f"PLANTEST|{pl}|{famname}|s{seed}"] = dict(c3_minus_c1=d, sep=P.sep_label(*d), n=int(len(idx)))
                # PLANTEST2
                ch1p, ch3p = [os.path.join(RUNS_G, f"choice_{n}_{f}_{pl}.json") for f in FORMS]
                if os.path.exists(ch1p) and os.path.exists(ch3p):
                    l1, l3 = json.load(open(ch1p))["choice"][famname], json.load(open(ch3p))["choice"][famname]
                    b1, b3 = ld(RUNS_G, "score", n, "c1", pl, "true", f"L{lam_s(l1)}"), ld(RUNS_G, "score", n, "c3", pl, "true", f"L{lam_s(l3)}")
                    if b1 is not None and b3 is not None:
                        keep = np.nonzero(np.isin(b1["idx"], s3["score"]))[0]
                        idx = b1["idx"][keep]
                        assert np.array_equal(idx, b3["idx"])
                        uniq, Wt, pos = boot_for(c.eid[idx])
                        m1 = P.window_metrics(b1["plan_raw"].astype(np.float64)[keep], c.A[idx], c.v0[idx])[q]
                        m3 = P.window_metrics(b3["plan_raw"].astype(np.float64), c.A[idx], c.v0[idx])[q]
                        d = P.boot_mean_diff(m3, m1, pos, Wt)
                        rows[f"PLANTEST2|{pl}|{famname}|s{seed}"] = dict(c3_minus_c1=d, sep=P.sep_label(*d), n=int(len(idx)), lam_c1=l1, lam_c3=l3)
    out.setdefault(fam, {})["c3_minus_c1"] = rows


def md(out, controls):
    Lns, a = [], None
    a = Lns.append
    a("# RESULT_LADDER_PLANTEST: PLANTEST and PLANTEST2 on the ladder world models (T0, zero training; agent, 2026-11-09)")
    a("Cards `../PREREG_V8F_PLANTEST.md` and `../PREREG_V8F_PLANTEST2.md` (frozen, not edited; their code imported unchanged), implementation choices `IMPL_CHOICES_LADDER_PLANTEST.md` (sha256 in `IMPL_CHOICES_LADDER_PLANTEST.sha256`, hashed before any arm statistic). "
      "**Tier T0 on every number: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`RESULT_LADDER_PLANTEST.json`, plans banked in `RESULT_LADDER_PLANTEST_plans.npz`). No verdict is written here.")
    a("Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the episodes of the read set, B 2,000, seed 0, ratio of per-episode sums, 95 % percentile; separated-better = interval wholly below 0 (MAE, lower is better). "
      "It answers 'another draw of episodes' and is blind to training-seed and planner-seed variance: both training seeds are required for any label, and the planner-seed replicate was not run (see Scope).")
    a("Arms: `C0` (control, strides (2, 2), one-step loss), `R35` and `R35H` (strides (3, 5), k-step loss; R35H adds a 3-frame history encoder), training seeds 0 and 1, adapter frozen at M2b s{k}. "
      "Read sets: PLANTEST on all windows of the form; PLANTEST2 on SCORE only (lambda chosen on TUNE only, hash split salt `plantest2-v1`, the cards' own split).")
    a("")
    a("## Scope, stated per item (what was and was not run)")
    a("* Rollout forms: **c1** = the cards' form (level-1 rollout conditioned on the current latent). **c3** = beside, labelled, no bar: level 1 conditioned on the REAL encoded latents at t-2 and t-1 (and the logged measured actions taken at them) plus NOW; "
      "windows without both past frames (g < 2) are dropped from c3 and their n is stated. Upper levels of TD3 keep context 1.")
    a("* TD3 as the cards define it needs level steps of 0.2 / 0.4 / 0.8 s. It exists on C0 only. On the (3, 5) arms (R35, R35H) it is **NOT COMPUTABLE** as the card defines it (the solver would run mechanically on those shapes, at horizons the card does not define): the built level-3 step is 3.0 s, so the card's 2.4 s goal is not a level-3 horizon "
      "(a rollout of 3 level-3 steps would span 9.0 s) and the card's level-2 sub-plan of 6 states would span 3.6 s instead of 1.2 s. n = 1,976 windows (c1) / 1,835 (c3) for each such row. HIER-ADDS is therefore computable on C0 only. No variant was substituted.")
    a("* Not run (cost; each is a resumable unit and none decides a committed label): G-STAY, the last-2-state FLAT, the planner-seed replicates, the two untrained stacks. FLAT1 is the flat arm (PLANTEST card log, 2026-11-08).")
    a("* TACTICAL: not computed, n = 0 (this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).")
    a("")
    a("## Controls (`CONTROLS_LADDER_PLANTEST.md` / `.json`)")
    if controls:
        for k, v in controls.items():
            a(f"* {k}: {'PASS' if v.get('passed') else 'FAIL'}" + (f" (regression arm goes RED: {v['regression_goes_red']})" if "regression_goes_red" in v else ""))
    else:
        a("* controls file not found")
    a("")
    for fam in FAMILIES:
        if fam not in out:
            continue
        for form in FORMS:
            r = out[fam].get(form)
            if r is None:
                continue
            a(f"# {fam}, {FORM_NAME[form]}")
            a(f"n windows: PLANTEST {r['n_windows_all']} ({r['n_episodes_all']} episodes); PLANTEST2 SCORE {r['n_windows_score']} ({r['n_episodes_score']} episodes), TUNE {r['n_windows_tune']}.")
            if fam != "C0":
                a(f"TD3: **NOT COMPUTABLE** on this arm (built strides (3, 5); see Scope), n = {r['n_windows_all']} windows.")
            for card, key, title in (("PLANTEST", "plantest", "PLANTEST (card 1): action recovery against the 12 logged actions, all windows"), ("PLANTEST2", "plantest2", "PLANTEST2 (card 2): SEED and REG, SCORE half, lambda from TUNE")):
                rows = r[key]["rows"]
                a(f"## {card}: verdict table (the card's committed outcomes; both training seeds required)")
                a("| planner | family | s0 | s1 | label |")
                a("|---|---|---|---|---|")
                for pl in planners_of(fam):
                    for famname in FAM:
                        r0, r1 = rows.get(f"{pl}|{famname}|s0"), rows.get(f"{pl}|{famname}|s1")
                        k0 = None if r0 is None else r0["label"] if card == "PLANTEST" else r0["base_label"] + (" (collapse)" if r0["collapse"] else "")
                        k1 = None if r1 is None else r1["label"] if card == "PLANTEST" else r1["base_label"] + (" (collapse)" if r1["collapse"] else "")
                        a(f"| {pl.upper()} | {famname} | {k0 or 'not run yet'} | {k1 or 'not run yet'} | **{rows.get(f'{pl}|{famname}|label', 'INCOMPLETE')}** |")
                if fam != "C0":
                    a("| TD3 | LONGITUDINAL, LATERAL | NOT COMPUTABLE | NOT COMPUTABLE | **NOT COMPUTABLE** (built strides (3, 5)) |")
                if r[key]["hier_adds"]:
                    a("")
                    a("HIER-ADDS (TD3 minus FLAT1 under G-TRUE, separated-better at both seeds): " + "; ".join(f"{f}: {'YES' if v['HIER_ADDS'] else ('NO' if v['complete'] else 'INCOMPLETE')}" for f, v in r[key]["hier_adds"].items()))
                else:
                    a("")
                    a("HIER-ADDS: NOT COMPUTABLE (needs TD3).")
                a("")
                a(f"### {card}: deciding rows, MAE against the logged actions (EC-BOOT mean [95 % interval])")
                for famname, q in FAM.items():
                    a(f"**{famname}** ({q}, {UNIT[q]})")
                    if card == "PLANTEST":
                        a("| model | planner | HOLD | PRIOR | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | TRUE - PRIOR (beside) | label |")
                        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
                    else:
                        a("| model | planner | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | plan distance from HOLD, G-TRUE (G-RAND) [prior-std units] | label |")
                        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
                    for seed in (0, 1):
                        for pl in planners_of(fam):
                            x = rows.get(f"{pl}|{famname}|s{seed}")
                            if x is None:
                                a(f"| {fam}_s{seed} | {pl} | not run yet |")
                                continue
                            if card == "PLANTEST":
                                a(f"| {fam}_s{seed} | {pl} | {fmt(x['hold'])} | {fmt(x['prior'])} | {fmt(x['true'])} | {fmt(x['rand'])} | {fmt(x['true_minus_hold'])} | {x['sep_hold']} | {fmt(x['true_minus_rand'])} | {x['sep_rand']} | {fmt(x['true_minus_prior'])} | {x['label']} |")
                            else:
                                a(f"| {fam}_s{seed} | {pl} | {lam_s(x['lam'])} | {fmt(x['hold'])} | {fmt(x['true'])} | {fmt(x['rand'])} | {fmt(x['true_minus_hold'])} | {x['sep_hold']} | {fmt(x['true_minus_rand'])} | {x['sep_rand']} | {x['d_true']:.3f} ({x['d_rand']:.3f}){' COLLAPSE' if x['collapse'] else ''} | {x['base_label']} |")
                    a("")
                if r[key]["hier_adds"]:
                    a(f"### {card}: HIER-ADDS rows (TD3 minus FLAT1 under G-TRUE; negative = TD3 better)")
                    a("| family | seed | TD3 - FLAT1 | sep |")
                    a("|---|---|---|---|")
                    for f, v in r[key]["hier_adds"].items():
                        for h in v["rows"]:
                            a(f"| {f} | {h['seed']} | {fmt(h['td3_minus_flat1'])} | {h['sep']} |")
                    a("")
                a(f"### {card}: beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (EC-BOOT mean [interval])")
                a("| model | planner | ref | speed err [m/s] | heading err [rad] | ADE [m] |")
                a("|---|---|---|---|---|---|")
                seen = set()
                for kk, x in rows.items():
                    if kk.endswith("label") or "|" not in kk:
                        continue
                    pl, famname, s = kk.split("|")
                    if famname != "LONGITUDINAL":
                        continue                                              # the beside metrics do not depend on the family unless lambda differs; the LONGITUDINAL row's lambda is printed
                    for ref in ("hold", "true", "rand"):
                        a(f"| {fam}_{s} | {pl}{(' (lambda ' + lam_s(x['lam']) + ')') if 'lam' in x else ''} | {ref} | {fmt(x['speed_err'][ref])} | {fmt(x['heading_err'][ref])} | {fmt(x['ade'][ref])} |")
                a("")
                a(f"### {card}: beside, no bar: cost monotonicity (-Spearman of goal distance with time; 1.0 = perfectly decreasing) and fraction of windows where planning lowered the cost, G-TRUE")
                a("| run | level | monotonicity | frac lowered |")
                a("|---|---|---|---|")
                for kk, x in rows.items():
                    if kk.endswith("label") or kk.split("|")[1] != "LONGITUDINAL":
                        continue
                    for lv in (1, 2, 3):
                        if f"monotonicity_L{lv}" in x:
                            a(f"| {fam}_{kk.split('|')[2]} {kk.split('|')[0]} | L{lv} | {fmt(x[f'monotonicity_L{lv}'])} | {fmt(x[f'frac_cost_lowered_L{lv}'])} |")
                a("")
                if card == "PLANTEST2":
                    a("### PLANTEST2: the TUNE grid (G-TRUE, mean MAE over the TUNE windows; the selection record, not a result)")
                    a("| model | planner | lambda | acc MAE [m/s^2] | yaw MAE [rad/s] |")
                    a("|---|---|---|---|---|")
                    for kk, x in rows.items():
                        if kk.endswith("label") or kk.split("|")[1] != "LONGITUDINAL":
                            continue
                        for l, v in x["tune_table"].items():
                            a(f"| {fam}_{kk.split('|')[2]} | {kk.split('|')[0]} | {l} | {v['acc_mae']:.4f}{' (chosen LONGITUDINAL)' if lam_s(x['lam']) == l else ''} | {v['yaw_mae']:.4f}{' (chosen LATERAL)' if lam_s(rows[kk.replace('LONGITUDINAL', 'LATERAL')]['lam']) == l else ''} |")
                    a("")
        c31 = out[fam].get("c3_minus_c1")
        if c31:
            a(f"# {fam}: context 3 minus context 1, G-TRUE plan MAE on the windows both read (beside, no bar; negative = context 3 recovers the logged actions better)")
            a("| read | planner | family | seed | c3 - c1 | sep | n | lambda c1 / c3 |")
            a("|---|---|---|---|---|---|---|---|")
            for kk, x in c31.items():
                card, pl, famname, s = kk.split("|")
                a(f"| {card} | {pl} | {famname} | {s} | {fmt(x['c3_minus_c1'])} | {x['sep']} | {x['n']} | {lam_s(x['lam_c1']) + ' / ' + lam_s(x['lam_c3']) if 'lam_c1' in x else '-'} |")
            a("")
    return "\n".join(Lns) + "\n"


def pack(runs, dest):
    """One compressed archive of every unit's plan and diagnostics (float32), so the plans survive a clean-up of the scratch folder."""
    z = {}
    for p in sorted(glob.glob(os.path.join(runs, "*.npz"))):
        n = os.path.basename(p)[:-4]
        if n.endswith(".tmp"):
            continue
        d = np.load(p)
        for k in d.files:
            z[f"{n}/{k}"] = d[k].astype(np.float32) if d[k].dtype == np.float64 and d[k].size > 64 else d[k]
    np.savez_compressed(dest, **z)
    return dest


def main():
    global RUNS_G
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=f"{HERE}/RESULT_LADDER_PLANTEST")
    ap.add_argument("--pack", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--smoke", action="store_true", help="8-window read sets (for the smoke test of this script only; the output is not a result)")
    a_ = ap.parse_args()
    RUNS_G = a_.runs
    if a_.smoke:
        real = L.sel_arrays
        L.sel_arrays = lambda ctx, form: {k: v[:8] for k, v in real(ctx, form).items()}
    out = {"estimator": "EC-BOOT: paired episode-cluster bootstrap, B 2000, seed 0, ratio of per-episode sums, 95 % percentile", "tier": "T0"}
    for fam in [f for f in FAMILIES if not a_.only or f in a_.only.split(",")]:
        for form in FORMS:
            if any(os.path.exists(p) for p in glob.glob(os.path.join(a_.runs, f"p1_{fam}_s*_{form}_*.npz"))):
                read_family(a_.runs, fam, form, out)
        if fam in out:
            read_c3_minus_c1(a_.runs, fam, out)
    cp = f"{HERE}/CONTROLS_LADDER_PLANTEST.json"
    controls = json.load(open(cp)) if os.path.exists(cp) else None
    out["controls"] = controls
    json.dump(out, open(f"{a_.out}.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    text = md(out, controls)
    if a_.smoke:
        text = "**SMOKE RUN of this script on 8-window read sets and copied seed units. These numbers are not results.**" + chr(10) * 2 + text
    open(f"{a_.out}.md", "w", encoding="utf-8").write(text)
    if a_.pack:
        pack(a_.runs, f"{a_.out}_plans.npz")


RUNS_G = RUNS
if __name__ == "__main__":
    main()
