"""PREREG_V8F_PT3M read-out: PLANTEST3's committed report applied to mlp4/runs (ARMS restricted to DEC-END), then the validity block, the paired PT3M minus PT3N table and the END ceilings on the same SCORE windows. Tier T0. Writes RESULT_PT3M.json / .md."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PT3 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PT3)
import numpy as np
import report_plantest3 as RR
import plantest3_real as R3
import plantest_lib as P

RUNS = f"{HERE}/runs"
N1053 = f"{PT3}/n1053"
PREFIX = f"{HERE}/RESULT_PT3M"
RR.ARMS = ("end",)
TAGS, FORMS, FAMS, QF, UNIT = RR.TAGS, RR.FORMS, RR.FAMS, RR.QF, RR.UNIT


def validity():
    rows, ok, lr = [], True, []
    for t in TAGS:
        g = json.load(open(f"{RUNS}/pre_{t}.json", encoding="utf-8"))
        for k, c in g["control1_tie_decn"].items():
            ok &= c["passed"]
            rows.append(f"* Control 1, tie to DECN ({t}, {k}): {c['mine']:+.5f} against {c['decn']:+.5f}, difference {c['abs_diff']:.5f} (bar 0.01): {'pass' if c['passed'] else 'FAIL'}.")
        lr.append(f"{t}: chosen {g['lr_chosen']:g}; " + ", ".join(f"lr {k}: early-stopping loss {v['es_loss']:.5f}, held-out R2 {v['r2_held']:+.3f}, epochs {v['epochs']}" for k, v in g["lr_grid"].items()) + f"; shuffled-target yaw rate at the chosen lr {g['yaw_shuffled_at_chosen_lr']:+.3f} (no bar)")
        c = json.load(open(f"{RUNS}/ctl_{t}.json", encoding="utf-8"))
        ok &= c["control2_passed"] and c["control3_passed"]
        rows.append(f"* Control 2, reachability ({t}): mean absolute gradient per channel {c['grad_abs_mean_by_channel'][0]:.3e}, {c['grad_abs_mean_by_channel'][1]:.3e}; under SHRINK(0) largest absolute gradient {c['grad_abs_max_shrink0']:.1e}: {'pass' if c['control2_passed'] else 'FAIL'}.")
        rows.append(f"* Control 3, harness tie ({t}): largest relative cost difference {c['cost_rel_diff_max']:.2e} (bar 1e-4), gradient-norm difference {c['grad_norm_rel_diff']:.2e} (bar 1e-3): {'pass' if c['control3_passed'] else 'FAIL'}.")
    rows.append("* Control 4, split: DECN's `split_guard_n` on real ids before each fit; `guard_mutation` requires four planted violations to raise, run at start. Pass (the run reached planning).")
    rows.append("* Control 5, clips: the 1,000 shards are DECN's; count and order asserted in `clip_order`. Pass.")
    st = json.load(open(f"{HERE}/run_pt3m.status.json", encoding="utf-8"))
    iso = bool(st.get("tune_choice_shas_unchanged"))
    ok &= iso and st.get("errors") == 0
    rows.append(f"* Control 6, isolation: PLANTEST3's and PT3N's `tune_choice.json` sha256 unchanged {iso}; run errors {st.get('errors')}: {'pass' if iso and st.get('errors') == 0 else 'FAIL'}.")
    return ok, rows, lr


def compare(choice_new, choice_old):
    ctxs = {t: R3.prepare3(t) for t in TAGS}
    sc = ctxs[TAGS[0]].score_idx
    uniq, Wt, pos = P.boot_setup(ctxs[TAGS[0]].eid[sc])

    def bdiff(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return P.boot_mean_diff(a[m], b[m], pos[m], Wt)

    def bmean(x):
        return P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)

    out, ceil, beats = {}, {}, {}
    for t in TAGS:
        c = ctxs[t]
        mets = lambda plan: P.window_metrics(plan, c.A[sc], c.v0[sc])
        for f in FORMS:
            for fam in FAMS:
                ln, lo = choice_new[f"{t}|end|{f}|{fam}"], choice_old[f"{t}|end|{f}|{fam}"]
                pn = np.load(f"{RUNS}/score_{t}_flat1_end_{f}_true_L{RR.lam_s(ln)}_p0.npz")["plan_raw"]
                po = np.load(f"{N1053}/runs/score_{t}_flat1_end_{f}_true_L{RR.lam_s(lo)}_p0.npz")["plan_raw"]
                q = QF[fam]
                d = bdiff(mets(pn)[q], mets(po)[q])
                out[f"{t}|{f}|{fam}"] = dict(unit=UNIT[q], lam_new=ln, lam_pt3n=lo, pt3m_minus_pt3n=d, sep=P.sep_label(*d))
        z = np.load(f"{RUNS}/ceil_{t}.npz")
        hold, orc, rd = mets(P.hold_plan(c.A[sc])), mets(z["oracle"]), mets(z["read"])
        for fam in FAMS:
            q = QF[fam]
            do, dr = bdiff(orc[q], hold[q]), bdiff(rd[q], hold[q])
            ceil[f"{t}|{fam}"] = dict(unit=UNIT[q], hold=bmean(hold[q]), oracle=bmean(orc[q]), read=bmean(rd[q]), oracle_minus_hold=do, sep_oracle=P.sep_label(*do), read_minus_hold=dr, sep_read=P.sep_label(*dr), n_oracle=int(np.isfinite(orc[q]).sum()), n_read=int(np.isfinite(rd[q]).sum()))
    for fam in FAMS:
        beats[fam] = dict(oracle=all(ceil[f"{t}|{fam}"]["sep_oracle"] == "better" for t in TAGS), read=all(ceil[f"{t}|{fam}"]["sep_read"] == "better" for t in TAGS))
    return out, ceil, beats


def main():
    ok, vrows, lr = validity()
    out = RR.main(RUNS, f"{HERE}/tune_choice.json", PREFIX, 0)
    cn = json.load(open(f"{HERE}/tune_choice.json", encoding="utf-8"))["choice"]
    co = json.load(open(f"{N1053}/tune_choice.json", encoding="utf-8"))["choice"]
    cmp_, ceil, beats = compare(cn, co)
    out.update(pt3m_valid=bool(ok), pt3m_minus_pt3n=cmp_, end_ceilings=ceil, end_ceiling_beats_hold=beats)
    json.dump(out, open(PREFIX + ".json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    body = open(PREFIX + ".md", encoding="utf-8").read().split("\n", 1)[1]
    body = body.split("## What each outcome triggers")[0]
    body = "\n".join(l for l in body.split("\n") if not l.startswith("A read is VALID only if"))
    head = ["# RESULT_PT3M: FLAT1 DEC-END with the four-frame per-target reader in the cost, fitted on N1053", "Card `../../PREREG_V8F_PT3M.md`. Tier **T0**, an oracle-goal diagnostic, never a driving result. TD3 and DEC-PATH are not run, so their lines below read INCOMPLETE or NO by construction. In the gate table D_1 is this reader on four frames; D_2 is not fitted (nan). In the KIN-CEILING table knots 3 to 11 come from the reader and knots 0 to 2 stay at HOLD. The triggers are in the card; the Master Mind applies them. MEASURED: `plantest3/mlp4/runs/`.", "", "## Validity first (the read is INVALID if one control fails)", f"* **Verdict on validity: {'all controls pass' if ok else 'INVALID (see rows)'}.**"] + vrows + ["", "Reader fit (no bar): "] + [f"* {x}" for x in lr]
    head += ["", "## END ceilings (no world model, no planner; SCORE; paired EC-BOOT, B 2,000)", "ORACLE-END: constant acceleration to the true speed at knot 12, yaw-rate ramp to the true value at knot 12. READ-END: the same with those two values from the reader on the encoded true future, starting from the measured current speed and yaw rate.", "", "| model | family | unit | HOLD | ORACLE-END | READ-END | ORACLE - HOLD | sep | READ - HOLD | sep |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in ceil.items():
        t, fam = k.split("|")
        head.append(f"| {t} | {fam} | {v['unit']} | {RR.fmt(v['hold'])} | {RR.fmt(v['oracle'])} | {RR.fmt(v['read'])} | {RR.fmt(v['oracle_minus_hold'])} | {v['sep_oracle']} | {RR.fmt(v['read_minus_hold'])} | {v['sep_read']} |")
    head.append("Separated-better than HOLD at both training seeds: " + "; ".join(f"{fam}: ORACLE-END {'YES' if b['oracle'] else 'NO'}, READ-END {'YES' if b['read'] else 'NO'}" for fam, b in beats.items()))
    head += ["", "## PT3M minus PT3N, G-TRUE, same SCORE windows (paired EC-BOOT, B 2,000; beside, no bar; negative is lower error)", "", "| seed | form | family | lambda (PT3M / PT3N) | difference [95 %] | separated |", "|---|---|---|---|---|---|"]
    for k, v in cmp_.items():
        t, f, fam = k.split("|")
        head.append(f"| {t} | {f} | {fam} | {RR.lam_s(v['lam_new'])} / {RR.lam_s(v['lam_pt3n'])} | {RR.fmt(v['pt3m_minus_pt3n'])} {v['unit']} | {v['sep']} |")
    open(PREFIX + ".md", "w", encoding="utf-8").write("\n".join(head) + "\n\n" + body.rstrip() + "\n")
    print("valid", ok)
    print(json.dumps(out["labels"], indent=0))
    print(json.dumps(beats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
