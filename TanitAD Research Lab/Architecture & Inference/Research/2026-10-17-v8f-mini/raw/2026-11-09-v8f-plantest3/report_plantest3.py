"""PREREG_V8F_PLANTEST3 read-out: applies the card's committed table on SCORE ONLY and writes RESULT_PLANTEST3.json / .md. Tier T0.
Estimator on every number: EC-BOOT = paired episode-cluster bootstrap over the SCORE episodes (B 2,000, seed 0, ratio of per-episode sums, 95 % percentile), interval-only separation (IMPL section 7).
Usage: python report_plantest3.py [--runs DIR] [--choice FILE] [--out PREFIX] [--n-score N]   (--n-score / --runs / --out are for the synthetic smoke test of this script only)"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import plantest3_lib as L3
import plantest3_real as R3
import plantest2_lib as Q
import plantest_lib as P

TAGS = ("M2b_s0", "M2b_s1")
ARMS = ("end", "path")
FORMS = ("plain", "shrink")
FAMS = ("LONGITUDINAL", "LATERAL")
QF = {"LONGITUDINAL": "acc_mae", "LATERAL": "yaw_mae"}
CH = {"LONGITUDINAL": 0, "LATERAL": 1}
UNIT = {"acc_mae": "m/s^2", "yaw_mae": "rad/s", "speed_err": "m/s", "heading_err": "rad", "ade": "m"}
BESIDE = ("speed_err", "heading_err", "ade")
ARM_NAME = {"end": "DEC-END", "path": "DEC-PATH"}


def lam_s(l):
    return f"{l:g}"


def fmt(t):
    return f"{t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]"


def load_run(runs, tag, kind, arm, form, goal, lam, pseed):
    p = f"{runs}/score_{tag}_{kind}_{arm}_{form}_{goal}_L{lam_s(lam)}_p{pseed}.npz"
    return dict(np.load(p)) if os.path.exists(p) else None


def main(runs, choice_path, out_prefix, n_score):
    ch = json.load(open(choice_path))
    choice, tune_table, td3_pick = ch["choice"], ch["tune_table"], ch["td3_pick"]
    ctxs = {t: R3.prepare3(t) for t in TAGS}
    c0 = ctxs[TAGS[0]]
    assert np.array_equal(c0.score_idx, ctxs[TAGS[1]].score_idx)
    sc = c0.score_idx[:n_score] if n_score else c0.score_idx
    uniq, Wt, pos = P.boot_setup(c0.eid[sc])
    out = dict(estimator="EC-BOOT: paired episode-cluster bootstrap over the SCORE episodes, B 2000, seed 0, ratio of per-episode sums, 95 % percentile", tier="T0",
               n_windows=int(len(sc)), n_episodes=int(len(uniq)), split_salt=Q.SALT, subset_smoke=bool(n_score), shrink_c={t: ctxs[t].shrink_c for t in TAGS})
    bmean = lambda x: P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)

    def bdiff(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return P.boot_mean_diff(a[m], b[m], pos[m], Wt)

    mets = lambda c, plan: P.window_metrics(plan, c.A[sc], c.v0[sc])
    # ---- the gate
    gate = {}
    for t in TAGS:
        p = f"{runs}/pre_{t}.json"
        gate[t] = json.load(open(p)) if os.path.exists(p) else None
    out["gate"] = gate
    weak = [t for t in TAGS if gate[t] and gate[t]["DECODER_WEAK_this_seed"]]
    out["DECODER_WEAK"] = bool(weak)
    out["DECODER_WEAK_seeds"] = weak
    out["gate_missing"] = [t for t in TAGS if gate[t] is None]
    # ---- HOLD, KIN-CEILING
    M, beside, rows, kin = {}, {"tune_grid": tune_table, "choice": choice, "td3_pick": td3_pick}, {}, {}
    for t in TAGS:
        c = ctxs[t]
        M[(t, "hold")] = mets(c, P.hold_plan(c.A[sc]))
        hs = Q.hold_scaled(c.wm, c.A[sc][:, :1]); hc = Q.hold_scaled(c.wm, c.A[sc][:, :1], c.st.prior_lo_1, c.st.prior_hi_1)
        beside[f"{t}|fraction_windows_HOLD_clipped_to_box"] = float((hs != hc).any(dim=(1, 2)).float().mean())
        kt = f"{runs}/kin_{t}_true.npz"
        kr = f"{runs}/kin_{t}_rand.npz"
        if os.path.exists(kt) and os.path.exists(kr):
            zt, zr = np.load(kt), np.load(kr)
            if not n_score:
                assert np.array_equal(zt["idx"], sc) and np.array_equal(zr["idx"], sc)
            mk, mkr = mets(c, zt["plan_raw"][:len(sc)]), mets(c, zr["plan_raw"][:len(sc)])
            for fam, q in QF.items():
                d = bdiff(mk[q], M[(t, "hold")][q]); dr = bdiff(mk[q], mkr[q])
                kin[f"{t}|{fam}"] = dict(unit=UNIT[q], hold=bmean(M[(t, "hold")][q]), kin=bmean(mk[q]), kin_rand=bmean(mkr[q]), kin_minus_hold=d, sep_hold=P.sep_label(*d), kin_minus_kinrand=dr, sep_rand=P.sep_label(*dr))
    out["kin"] = kin
    kin_beats = {fam: all(kin.get(f"{t}|{fam}", {}).get("sep_hold") == "better" for t in TAGS) for fam in FAMS}
    out["kin_beats_hold"] = kin_beats
    # ---- the planner rows
    def one_row(t, kind, arm, form, fam, lam):
        c = ctxs[t]
        q = QF[fam]
        rt, rr, r1 = (load_run(runs, t, kind, arm, form, g, lam, p) for g, p in (("true", 0), ("rand", 0), ("true", 1)))
        if rt is None or rr is None:
            return None
        if not n_score:
            assert np.array_equal(rt["idx"], sc) and np.array_equal(rr["idx"], sc), "saved plans are not the SCORE windows"
        mt, mr = mets(c, rt["plan_raw"]), mets(c, rr["plan_raw"])
        d_hold, d_rand = bdiff(mt[q], M[(t, "hold")][q]), bdiff(mt[q], mr[q])
        s_h, s_r = P.sep_label(*d_hold), P.sep_label(*d_rand)
        dt = float(Q.hold_distance(rt["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[fam]])
        dr = float(Q.hold_distance(rr["plan_raw"], P.hold_plan(c.A[sc]), c.std_raw)[CH[fam]])
        row = dict(kind=kind, arm=arm, form=form, lam=lam, unit=UNIT[q], hold=bmean(M[(t, "hold")][q]), true=bmean(mt[q]), rand=bmean(mr[q]), true_minus_hold=d_hold, true_minus_rand=d_rand,
                   sep_hold=s_h, sep_rand=s_r, base_label=P.family_label(s_h, s_r), d_true=dt, d_rand=dr, collapse=bool(dt < Q.COLLAPSE_FRAC), prior_std_raw=float(c.std_raw[CH[fam]]))
        for qq in BESIDE:
            row[qq] = dict(hold=bmean(M[(t, "hold")][qq]), true=bmean(mt[qq]), rand=bmean(mr[qq]))
        for key in ("mono1", "dmono1"):
            if key in rt:
                row[f"{key}_true"] = bmean(rt[key])
        if "cost1" in rt:
            row["frac_cost_lowered_L1"] = bmean((rt["cost1"] < rt["init1"]).astype(float))
        if r1 is not None:
            m1 = mets(c, r1["plan_raw"])
            dd = bdiff(m1[q], mt[q])
            row["planner_seed_replicate"] = dict(seed1_minus_seed0=dd, seed1=bmean(m1[q]), sep=P.sep_label(*dd))
        M[(t, kind, arm, form, fam, "true")] = mt
        return row

    for t in TAGS:
        for a in ARMS:
            for f in FORMS:
                for fam in FAMS:
                    r = one_row(t, "flat1", a, f, fam, choice[f"{t}|{a}|{f}|{fam}"])
                    if r:
                        rows[f"{t}|flat1|{a}|{f}|{fam}"] = r
        for fam in FAMS:
            f, lam = td3_pick[f"{t}|{fam}"]
            r = one_row(t, "td3", "end", f, fam, lam)
            if r:
                rows[f"{t}|td3|end|{f}|{fam}"] = r
    out["rows"] = rows
    labels = {}

    def lab_of(rs):
        bl = [r["base_label"] for r in rs]
        lab = bl[0] if len(set(bl)) == 1 else "SEEDS-DISAGREE(" + "|".join(bl) + ")"
        fl = [t for t, r in zip(TAGS, rs) if r["collapse"]]
        return lab + (f" +HOLD-COLLAPSE({','.join(fl)})" if fl else "")

    for a in ARMS:
        for f in FORMS:
            for fam in FAMS:
                rs = [rows.get(f"{t}|flat1|{a}|{f}|{fam}") for t in TAGS]
                labels[f"flat1|{a}|{f}|{fam}"] = "INCOMPLETE" if any(r is None for r in rs) else lab_of(rs)
    for fam in FAMS:
        rs = [next((r for k, r in rows.items() if k.startswith(f"{t}|td3|end|") and k.endswith(f"|{fam}")), None) for t in TAGS]
        labels[f"td3|end|{fam}"] = "INCOMPLETE" if any(r is None for r in rs) else lab_of(rs)
    out["labels"] = labels
    # ---- HIER-ADDS: TD3 DEC-END minus FLAT1 DEC-END of the same form and lambda, G-TRUE
    hier = {}
    for fam in FAMS:
        hr = {}
        for t in TAGS:
            f, lam = td3_pick[f"{t}|{fam}"]
            kt, kf = f"{t}|td3|end|{f}|{fam}", f"{t}|flat1|end|{f}|{fam}"
            if kt in rows and kf in rows:
                d = bdiff(M[(t, "td3", "end", f, fam, "true")][QF[fam]], M[(t, "flat1", "end", f, fam, "true")][QF[fam]])
                hr[t] = dict(td3_minus_flat1=d, sep=P.sep_label(*d), form=f, lam=lam)
        hier[fam] = dict(rows=hr, HIER_ADDS=len(hr) == 2 and all(v["sep"] == "better" for v in hr.values()))
    out["hier_adds"] = hier
    # ---- the card's committed triggers, evaluated mechanically (the Master Mind applies them; nothing is asserted here)
    def counts(lbl):
        return lbl.startswith("PLANNABLE") and "HOLD-COLLAPSE" not in lbl
    end_plan = {k: counts(v) for k, v in labels.items() if "|end|" in k and v != "INCOMPLETE"}
    path_plan = {k: counts(v) for k, v in labels.items() if "|path|" in k and v != "INCOMPLETE"}
    anyp = lambda d: any(d.values())
    trig = []
    if anyp(end_plan):
        trig.append("PLANNABLE on DEC-END in at least one family: " + ", ".join(k for k, v in end_plan.items() if v) + " -> this cost becomes the HWM-PLAN level-1 cost and is rerun on the ladder world models.")
    if anyp(path_plan) and not anyp(end_plan):
        trig.append("PLANNABLE on DEC-PATH only: a single end goal is too little; the next lever is a goal path from the tactical level's goal knots up to 6 s.")
    decoded = [v for v in labels.values() if v != "INCOMPLETE"]
    if decoded and not anyp(end_plan) and not anyp(path_plan):
        if all(l.startswith("NO-GAIN") or l.startswith("WORSE-THAN-RAND") for l in decoded):
            if out["DECODER_WEAK"] or not any(kin_beats.values()):
                trig.append("NO-GAIN on every decoded arm and (KIN-CEILING does not beat HOLD in any family, or DECODER-WEAK): the decoded quantities are the limit; next is a nonlinear readout and multi-frame decoding on the ladder latents.")
            else:
                trig.append("NO-GAIN on every decoded arm, KIN-CEILING beats HOLD in " + ", ".join(f for f, v in kin_beats.items() if v) + ": the world model's action response is the limit; next is the action-sensitivity lever (inverse-dynamics weight, action-diverse sampler) on the ladder arms.")
        if any(l.startswith("GOAL-BLIND") for l in decoded):
            trig.append("GOAL-BLIND appears: " + ", ".join(k for k, v in labels.items() if v.startswith("GOAL-BLIND")) + " -> action sensitivity first.")
    trig.append("Every outcome: the harness is read on the ladder world models when they finish; PLANTEST and PLANTEST2 stay required gates.")
    out["triggers"] = trig
    out["beside"] = beside
    json.dump(out, open(f"{out_prefix}.json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    write_md(out, out_prefix)
    return out


def write_md(out, prefix):
    rows, labels, hier, beside, kin, gate = out["rows"], out["labels"], out["hier_adds"], out["beside"], out["kin"], out["gate"]
    L = []
    a = L.append
    a("# RESULT_PLANTEST3: does a cost in decoded, action-relevant quantities make the level planners beat holding the action? (T0, zero training; agent, 2026-11-09)")
    a("Card `../PREREG_V8F_PLANTEST3.md` (prefix sha256 `1f9609b4...b348f9`), IMPL `IMPL_CHOICES_PLANTEST3.md` (sha256 in `IMPL_CHOICES_PLANTEST3.sha256`). **Tier T0: a world-model diagnostic with an oracle (true-future) goal, never a driving result.** Evidence class: MEASURED (`runs/*.npz`, `RESULT_PLANTEST3.json`).")
    a(f"Estimator on every interval: EC-BOOT = paired episode-cluster bootstrap over the {out['n_episodes']} SCORE episodes ({out['n_windows']} windows), B 2,000, seed 0, 95 % percentile. Separated-better = interval wholly below 0 (MAE, lower is better). Decoders and lambda were chosen on the TUNE half only (hash split, salt `{out['split_salt']}`); every number below is SCORE. The interval answers 'another draw of episodes'; it is blind to training-seed and planner-seed variance, hence both training seeds are required and the planner-seed replicate is printed beside.")
    if out.get("subset_smoke"):
        a("**SMOKE RUN of this script on synthetic plan arrays and a subset of SCORE windows. These numbers are not results.**")
    a("A read is VALID only if the six controls of `RESULT_PLANTEST3_controls.md` passed; this report does not check that file.")
    a("")
    a("## Decoder gate (before planning; TUNE inner split, held-out episodes)")
    flag = "**DECODER-WEAK: YES** (" + ", ".join(out["DECODER_WEAK_seeds"]) + ")" if out["DECODER_WEAK"] else "DECODER-WEAK: no"
    a(f"{flag}; bars: held-out R2 on encoded latents below {L3.R2_SPEED_MIN:.2f} (speed) or {L3.R2_YAW_MIN:.2f} (yaw rate) at either seed. Missing gate records: {out['gate_missing'] or 'none'}.")
    a("| model | read | R2 speed | R2 yaw rate | n speed / n yaw |")
    a("|---|---|---|---|---|")
    for t in TAGS:
        g = gate.get(t)
        if not g:
            continue
        for key, nm in (("enc_L1", "D_1 encoded, all knots (the DECODER-WEAK quantity)"), ("enc_L1_k12", "D_1 encoded, k = 12"), ("rollout_L1_k12_logged_actions", "D_1 on the rollout at k = 12 under the logged actions"), ("enc_L2", "D_2 encoded, all knots")):
            v = g[key]
            a(f"| {t} | {nm} | {v['r2_speed']:.4f} | {v['r2_yaw']:.4f} | {v['n_speed']} / {v['n_yaw']} |")
    a("")
    a("## Verdict (the card's committed table; per arm, planner and family; both training seeds required)")
    a("| planner | arm | form | family | chosen lambda s0 / s1 | M2b_s0 | M2b_s1 | decision label |")
    a("|---|---|---|---|---|---|---|---|")
    for key, lab in labels.items():
        parts = key.split("|")
        if parts[0] == "flat1":
            _, arm, form, fam = parts
            r0, r1 = rows.get(f"M2b_s0|flat1|{arm}|{form}|{fam}"), rows.get(f"M2b_s1|flat1|{arm}|{form}|{fam}")
            planner, formtxt = "FLAT1", form
        else:
            _, arm, fam = parts
            r0 = next((r for k, r in rows.items() if k.startswith("M2b_s0|td3|") and k.endswith(f"|{fam}")), None)
            r1 = next((r for k, r in rows.items() if k.startswith("M2b_s1|td3|") and k.endswith(f"|{fam}")), None)
            planner, formtxt = "TD3", (f"{r0['form']} / {r1['form']}" if r0 and r1 else "?")
        if r0 and r1:
            a(f"| {planner} | {ARM_NAME[arm]} | {formtxt} | {fam} | {lam_s(r0['lam'])} / {lam_s(r1['lam'])} | {r0['base_label']}{' (collapse)' if r0['collapse'] else ''} | {r1['base_label']}{' (collapse)' if r1['collapse'] else ''} | **{lab}** |")
    a("TACTICAL: not computed (n = 0: this read has no decision output). STRATEGIC: PAUSED (PI item 25, `PI_DECISION_QUEUE.md`).")
    a("")
    a("HIER-ADDS (TD3 DEC-END minus FLAT1 DEC-END of the same form and lambda, G-TRUE, separated-better at both seeds): " + "; ".join(f"{f}: {'YES' if v['HIER_ADDS'] else 'NO'}" for f, v in hier.items()))
    a("")
    a("## KIN-CEILING (no world model, no planner): actions from the decoded true-future speed differences and yaw rates, MAE against the 12 logged actions, SCORE")
    a("| model | family | unit | HOLD | KIN | KIN-RAND (donor's decoded future) | KIN - HOLD | sep | KIN - KIN-RAND | sep |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for key, v in kin.items():
        t, fam = key.split("|")
        a(f"| {t} | {fam} | {v['unit']} | {fmt(v['hold'])} | {fmt(v['kin'])} | {fmt(v['kin_rand'])} | {fmt(v['kin_minus_hold'])} | {v['sep_hold']} | {fmt(v['kin_minus_kinrand'])} | {v['sep_rand']} |")
    a("KIN beats HOLD (separated-better at both seeds): " + "; ".join(f"{f}: {'YES' if v else 'NO'}" for f, v in out["kin_beats_hold"].items()))
    a("")
    a("## Deciding rows: MAE against the 12 logged actions, SCORE (EC-BOOT mean [95 % interval])")
    for fam, q in QF.items():
        a(f"### {fam} ({q}, {UNIT[q]})")
        a("| model | planner | arm | form | lambda | HOLD | G-TRUE | G-RAND | TRUE - HOLD | sep | TRUE - RAND | sep | distance from HOLD, G-TRUE (G-RAND) [prior-std units] | base label |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key, r in rows.items():
            t, kind, arm, form, f = key.split("|")
            if f != fam:
                continue
            a(f"| {t} | {kind} | {ARM_NAME[arm]} | {form} | {lam_s(r['lam'])} | {fmt(r['hold'])} | {fmt(r['true'])} | {fmt(r['rand'])} | {fmt(r['true_minus_hold'])} | {r['sep_hold']} | {fmt(r['true_minus_rand'])} | {r['sep_rand']} | {r['d_true']:.3f} ({r['d_rand']:.3f}){' COLLAPSE' if r['collapse'] else ''} | {r['base_label']} |")
        a("")
    a("HOLD-COLLAPSE flag: mean absolute distance of the G-TRUE plan from HOLD below 0.10 of the TRAIN prior std of the channel; a flagged row does not count as planning.")
    a("")
    a("## HIER-ADDS rows: TD3 minus FLAT1 under G-TRUE, DEC-END (negative = TD3 better), SCORE")
    a("| family | model | form / lambda | TD3 - FLAT1 | sep |")
    a("|---|---|---|---|---|")
    for fam, v in hier.items():
        for t, r in v["rows"].items():
            a(f"| {fam} | {t} | {r['form']} / {lam_s(r['lam'])} | {fmt(r['td3_minus_flat1'])} | {r['sep']} |")
    a("")
    a("## Beside, no bar: speed error, heading error and ADE at 2.4 s of the unicycle integration, plan vs logged actions (SCORE; at the family's chosen lambda; EC-BOOT mean [interval])")
    a("| model | planner | arm | form | family (lambda) | ref | speed err [m/s] | heading err [rad] | ADE [m] |")
    a("|---|---|---|---|---|---|---|---|---|")
    for key, r in rows.items():
        t, kind, arm, form, fam = key.split("|")
        for ref in ("hold", "true", "rand"):
            a(f"| {t} | {kind} | {ARM_NAME[arm]} | {form} | {fam} ({lam_s(r['lam'])}) | {ref} | {fmt(r['speed_err'][ref])} | {fmt(r['heading_err'][ref])} | {fmt(r['ade'][ref])} |")
    a("")
    a("## Beside, no bar: cost monotonicity (Tab. 10 statistic: -Spearman(distance, time); 1.0 = perfectly decreasing) along the chosen G-TRUE plan, level 1: decoded distance to the END goal, and latent distance to the END goal; fraction of windows where planning lowered the cost below the best initial candidate (the cost includes REG when lambda > 0)")
    a("| run | decoded monotonicity | latent monotonicity | frac lowered |")
    a("|---|---|---|---|")
    for key, r in rows.items():
        dm = fmt(r["dmono1_true"]) if "dmono1_true" in r else "-"
        lm = fmt(r["mono1_true"]) if "mono1_true" in r else "-"
        fl = fmt(r["frac_cost_lowered_L1"]) if "frac_cost_lowered_L1" in r else "-"
        a(f"| {key} (lambda {lam_s(r['lam'])}) | {dm} | {lm} | {fl} |")
    a("")
    a("## Beside: planner-seed replicate (planner seed 1 minus seed 0, G-TRUE, same checkpoint and lambda, SCORE, EC-BOOT; a difference of this size is inference noise, not an effect)")
    for key, r in rows.items():
        if "planner_seed_replicate" in r:
            v = r["planner_seed_replicate"]
            a(f"* {key} (lambda {lam_s(r['lam'])}): {fmt(v['seed1_minus_seed0'])} ({v['sep']})")
    a("")
    a("## Beside: the TUNE grid (FLAT1, G-TRUE, planner seed 0, mean MAE over the TUNE windows; lambda is chosen as the row minimum per family; this is the selection record, not a result) and the TD3 form pick")
    a("| model | arm | form | lambda | LONGITUDINAL acc MAE [m/s^2] | LATERAL yaw MAE [rad/s] |")
    a("|---|---|---|---|---|---|")
    for key, row in beside["tune_grid"].items():
        t, arm, form = key.split("|")
        for l, v in row.items():
            mk = lambda fam: " (chosen)" if lam_s(beside["choice"][f"{t}|{arm}|{form}|{fam}"]) == l else ""
            a(f"| {t} | {ARM_NAME[arm]} | {form} | {l} | {v['acc_mae']:.4f}{mk('LONGITUDINAL')} | {v['yaw_mae']:.4f}{mk('LATERAL')} |")
    a("TD3 form and lambda (the FLAT1 DEC-END choice per training seed and family): " + "; ".join(f"{k}: {v[0]} / {lam_s(v[1])}" for k, v in beside["td3_pick"].items()))
    a("")
    a("Fraction of SCORE windows whose HOLD lies outside the level-1 box (HOLD_c differs from HOLD): " + ", ".join(f"{k.split('|')[0]} {v:.3f}" for k, v in beside.items() if k.endswith("clipped_to_box")))
    a("")
    a("## What each outcome triggers (committed in the card, Rule Zero; evaluated mechanically on the table above, the Master Mind applies it)")
    for t_ in out["triggers"]:
        a(f"* {t_}")
    open(f"{prefix}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=f"{HERE}/runs")
    ap.add_argument("--choice", default=f"{HERE}/tune_choice.json")
    ap.add_argument("--out", default=f"{HERE}/RESULT_PLANTEST3")
    ap.add_argument("--n-score", type=int, default=0)
    a_ = ap.parse_args()
    main(a_.runs, a_.choice, a_.out, a_.n_score)
