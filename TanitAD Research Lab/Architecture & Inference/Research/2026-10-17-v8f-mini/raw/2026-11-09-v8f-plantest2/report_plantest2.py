"""PREREG_V8F_PLANTEST2 read-out: applies the card's committed table on SCORE ONLY and writes RESULT_PLANTEST2.json / .md. Tier T0.
Estimator on every number: EC-BOOT = paired episode-cluster bootstrap over the SCORE episodes (B 2,000, seed 0, ratio of per-episode sums, 95 % percentile interval), interval-only separation (IMPL section 7).
Usage: python report_plantest2.py [--runs DIR] [--choice FILE] [--out PREFIX] [--n-score N]   (--n-score / --runs / --out are for the smoke test of this script only)"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import plantest2_lib as Q
import plantest2_real as R2
import plantest_lib as P

TAGS = ("M2b_s0", "M2b_s1")
PL = ("td3", "flat1")
FAM = Q.FAMS
CH = {"LONGITUDINAL": 0, "LATERAL": 1}
UNIT = {"acc_mae": "m/s^2", "yaw_mae": "rad/s", "speed_err": "m/s", "heading_err": "rad", "ade": "m"}
BESIDE = ("speed_err", "heading_err", "ade")


def lam_s(l):
    return f"{l:g}"


def fmt(t):
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


def load_run(runs, tag, kind, goal, lam, pseed):
    p = f"{runs}/score_{tag}_{kind}_{goal}_L{lam_s(lam)}_p{pseed}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def main(runs, choice_path, out_prefix, n_score):
    ch = json.load(open(choice_path))
    choice, tune_table = ch["choice"], ch["tune_table"]
    ctxs = {t: R2.prepare2(t) for t in TAGS}
    c0 = ctxs[TAGS[0]]
    assert np.array_equal(c0.score_idx, ctxs[TAGS[1]].score_idx)
    sc = c0.score_idx[:n_score] if n_score else c0.score_idx
    eids = c0.eid[sc]
    uniq, Wt, pos = P.boot_setup(eids)
    out = dict(estimator="EC-BOOT: paired episode-cluster bootstrap over the SCORE episodes, B 2000, seed 0, ratio of per-episode sums, 95 % percentile", tier="T0",
               n_windows=int(len(sc)), n_episodes=int(len(uniq)), split_salt=Q.SALT, subset_smoke=bool(n_score))
    bmean = lambda x: P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)

    def bdiff(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return P.boot_mean_diff(a[m], b[m], pos[m], Wt)

    mets = lambda c, plan: P.window_metrics(plan, c.A[sc], c.v0[sc])
    rows, hier, beside = {}, {}, {"tune_grid": tune_table, "choice": choice}
    M = {}
    for t in TAGS:
        c = ctxs[t]
        M[(t, "hold")] = mets(c, P.hold_plan(c.A[sc]))
        hs = Q.hold_scaled(c.wm, c.A[sc][:, :1])
        hc = Q.hold_scaled(c.wm, c.A[sc][:, :1], c.st.prior_lo_1, c.st.prior_hi_1)
        beside[f"{t}|fraction_windows_HOLD_clipped_to_box"] = float((hs != hc).any(dim=(1, 2)).float().mean())
        for k in PL:
            for fam, q in FAM.items():
                lam = choice[f"{t}|{k}|{fam}"]
                rt, rr, r1 = (load_run(runs, t, k, g, lam, p) for g, p in (("true", 0), ("rand", 0), ("true", 1)))
                if rt is None or rr is None:
                    continue
                if not n_score:
                    assert np.array_equal(rt["idx"], sc) and np.array_equal(rr["idx"], sc), "saved plans are not the SCORE windows"
                mt, mr = mets(c, rt["plan_raw"]), mets(c, rr["plan_raw"])
                d_hold, d_rand = bdiff(mt[q], M[(t, "hold")][q]), bdiff(mt[q], mr[q])
                s_h, s_r = P.sep_label(*d_hold), P.sep_label(*d_rand)
                base = P.family_label(s_h, s_r)
                dt = float(Q.hold_distance(rt["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[fam]])
                dr = float(Q.hold_distance(rr["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[fam]])
                row = dict(lam=lam, unit=UNIT[q], hold=bmean(M[(t, "hold")][q]), true=bmean(mt[q]), rand=bmean(mr[q]), true_minus_hold=d_hold, true_minus_rand=d_rand, sep_hold=s_h, sep_rand=s_r,
                           base_label=base, d_true=dt, d_rand=dr, collapse=bool(dt < Q.COLLAPSE_FRAC), prior_std_raw=float(c.std_raw[CH[fam]]))
                for qq in BESIDE:
                    row[qq] = dict(hold=bmean(M[(t, "hold")][qq]), true=bmean(mt[qq]), rand=bmean(mr[qq]))
                for lv in (1, 2, 3):
                    if f"mono{lv}" in rt:
                        row[f"monotonicity_L{lv}"] = bmean(rt[f"mono{lv}"])
                        row[f"frac_cost_lowered_L{lv}"] = bmean((rt[f"cost{lv}"] < rt[f"init{lv}"]).astype(float))
                if r1 is not None:
                    m1 = mets(c, r1["plan_raw"])
                    dd = bdiff(m1[q], mt[q])
                    row["planner_seed_replicate"] = dict(seed1_minus_seed0=dd, seed1=bmean(m1[q]), sep=P.sep_label(*dd))
                rows[f"{t}|{k}|{fam}"] = row
                M[(t, f"{k}|{fam}|true")] = mt
    out["rows"] = rows
    labels = {}
    for k in PL:
        for fam in FAM:
            rs = [rows.get(f"{t}|{k}|{fam}") for t in TAGS]
            if any(r is None for r in rs):
                labels[f"{k}|{fam}"] = "INCOMPLETE"
                continue
            bl = [r["base_label"] for r in rs]
            lab = bl[0] if len(set(bl)) == 1 else "SEEDS-DISAGREE(" + "|".join(bl) + ")"
            fl = [t for t, r in zip(TAGS, rs) if r["collapse"]]
            labels[f"{k}|{fam}"] = lab + (f" +HOLD-COLLAPSE({','.join(fl)})" if fl else "")
    out["labels"] = labels
    for fam, q in FAM.items():
        hr = {}
        for t in TAGS:
            if f"{t}|td3|{fam}" in rows and f"{t}|flat1|{fam}" in rows:
                d = bdiff(M[(t, f"td3|{fam}|true")][q], M[(t, f"flat1|{fam}|true")][q])
                hr[t] = dict(td3_minus_flat1=d, sep=P.sep_label(*d), lam_td3=rows[f"{t}|td3|{fam}"]["lam"], lam_flat1=rows[f"{t}|flat1|{fam}"]["lam"])
        hier[fam] = dict(rows=hr, HIER_ADDS=len(hr) == 2 and all(v["sep"] == "better" for v in hr.values()))
    out["hier_adds"], out["beside"] = hier, beside
    json.dump(out, open(f"{out_prefix}.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(out, out_prefix)
    return out


def write_md(out, prefix):
    rows, labels, hier, beside = out["rows"], out["labels"], out["hier_adds"], out["beside"]
    L, a = [], None
    a = L.append
    a("# RESULT_PLANTEST2: does a seeded and regularised solver make the level planners beat holding the action? (T0, zero training; agent, 2026-11-08)")
    a("Card `../PREREG_V8F_PLANTEST2.md` (prefix sha256 `d8217dfa...1a89`), IMPL `IMPL_CHOICES_PLANTEST2.md` (sha256 in `IMPL_CHOICES_PLANTEST2.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST2.json`).")
    a(f"Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the {out['n_episodes']} SCORE episodes ({out['n_windows']} windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Lambda was chosen on the TUNE half only (hash split, salt `{out['split_salt']}`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.")
    if out.get("subset_smoke"):
        a("**SMOKE RUN of this script on synthetic plan arrays and a subset of SCORE windows. These numbers are not results.**")
    a("")
    a("## Verdict (the card's committed table, per planner and family; both training seeds required)")
    a("| planner | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |")
    a("|---|---|---|---|---|---|")
    for k in PL:
        for fam in FAM:
            r0, r1 = rows.get(f"M2b_s0|{k}|{fam}"), rows.get(f"M2b_s1|{k}|{fam}")
            if r0 and r1:
                a(f"| {k.upper()} | {fam} | {lam_s(r0['lam'])} / {lam_s(r1['lam'])} | {r0['base_label']}{' (collapse)' if r0['collapse'] else ''} | {r1['base_label']}{' (collapse)' if r1['collapse'] else ''} | **{labels[f'{k}|{fam}']}** |")
    a("TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).")
    a("")
    a("HIER-ADDS (TD3 minus FLAT1 under G-TRUE, each planner at its own chosen lambda, separated-better at both seeds): " + "; ".join(f"{f}: {'YES' if v['HIER_ADDS'] else 'NO'}" for f, v in hier.items()))
    a("")
    a("## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])")
    for fam, q in FAM.items():
        a(f"### {fam} ({q}, {UNIT[q]})")
        a("| model | planner | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | plan distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for t in TAGS:
            for k in PL:
                r = rows.get(f"{t}|{k}|{fam}")
                if r:
                    a(f"| {t} | {k} | {lam_s(r['lam'])} | {fmt(r['hold'])} | {fmt(r['true'])} | {fmt(r['rand'])} | {fmt(r['true_minus_hold'])} | {r['sep_hold']} | {fmt(r['true_minus_rand'])} | {r['sep_rand']} | {r['d_true']:.3f} ({r['d_rand']:.3f}){' COLLAPSE' if r['collapse'] else ''} | {r['base_label']} |")
        a("")
    a("HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel (IMPL section 7); a flagged row does not count as planning.")
    a("")
    a("## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE (negative = TD3 better), SCORE")
    a("| family | model | lambda TD3 / FLAT1 | TD3 - FLAT1 | sep |")
    a("|---|---|---|---|---|")
    for fam, v in hier.items():
        for t, r in v["rows"].items():
            a(f"| {fam} | {t} | {lam_s(r['lam_td3'])} / {lam_s(r['lam_flat1'])} | {fmt(r['td3_minus_flat1'])} | {r['sep']} |")
    a("")
    a("## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])")
    a("| model | planner | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |")
    a("|---|---|---|---|---|---|---|")
    for key, r in rows.items():
        t, k, fam = key.split("|")
        for ref in ("hold", "true", "rand"):
            a(f"| {t} | {k} | {fam} ({lam_s(r['lam'])}) | {ref} | {fmt(r['speed_err'][ref])} | {fmt(r['heading_err'][ref])} | {fmt(r['ade'][ref])} |")
    a("")
    a("## Beside, no bar: cost monotonicity per level (Tab. 10 statistic: -Spearman(goal distance, time); 1.0 = perfectly decreasing) and fraction of windows where planning lowered the cost below the best initial candidate (G-TRUE, SCORE; the cost includes REG when lambda > 0)")
    a("| run | level | monotonicity | frac lowered |")
    a("|---|---|---|---|")
    for key, r in rows.items():
        for lv in (1, 2, 3):
            if f"monotonicity_L{lv}" in r:
                a(f"| {key} (lambda {lam_s(r['lam'])}) | L{lv} | {fmt(r[f'monotonicity_L{lv}'])} | {fmt(r[f'frac_cost_lowered_L{lv}'])} |")
    a("")
    a("## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)")
    for key, r in rows.items():
        if "planner_seed_replicate" in r:
            v = r["planner_seed_replicate"]
            a(f"* {key} (lambda {lam_s(r['lam'])}): {fmt(v['seed1_minus_seed0'])} ({v['sep']})")
    a("")
    a("## Beside: the TUNE grid (G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result)")
    a("| model | planner | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |")
    a("|---|---|---|---|---|")
    for key, row in beside["tune_grid"].items():
        t, k = key.split("|")
        for l, v in row.items():
            mark = lambda fam, q: " (chosen)" if lam_s(beside["choice"][f"{t}|{k}|{fam}"]) == l else ""
            a(f"| {t} | {k} | {l} | {v['acc_mae']:.4f}{mark('LONGITUDINAL', 'acc_mae')} | {v['yaw_mae']:.4f}{mark('LATERAL', 'yaw_mae')} |")
    a("")
    a("Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): " + ", ".join(f"{k.split('|')[0]} {v:.3f}" for k, v in beside.items() if k.endswith("clipped_to_box")))
    a("")
    trig = {"PLANNABLE": "this solver becomes the HWM-PLAN solver, and it is rerun on the new ladder.",
            "GOAL-BLIND": "action sensitivity first (inverse-dynamics weight, action-diverse sampler).",
            "NO-GAIN": "if NO-GAIN holds at every lambda, the next lever is the cost space (distance in decoded, action-relevant quantities instead of the raw latent), read on the ladder world model."}
    a("## What each outcome triggers (committed in the card, Rule Zero)")
    seen = {l.split(" ")[0] for l in labels.values()}
    for lab, txt in trig.items():
        if lab in seen:
            a(f"* {lab} appears in the table above: {txt}")
    a("* Every outcome: PLANTEST and this solver are required gates of the ladder arms; the ladder is not dropped on a negative here (one-step loss, 0.8 s top step).")
    open(f"{prefix}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{HERE}/runs")
    ap.add_argument("--choice", default=f"{HERE}/tune_choice.json")
    ap.add_argument("--out", default=f"{HERE}/RESULT_PLANTEST2")
    ap.add_argument("--n-score", type=int, default=0)
    a_ = ap.parse_args()
    main(a_.runs, a_.choice, a_.out, a_.n_score)
