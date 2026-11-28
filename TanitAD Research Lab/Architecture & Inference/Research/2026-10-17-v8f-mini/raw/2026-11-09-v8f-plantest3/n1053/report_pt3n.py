"""PREREG_V8F_PT3N read-out: PLANTEST3's committed report applied to n1053/runs (ARMS restricted to DEC-END), then the validity block and the paired PT3N minus PLANTEST3 table on the same SCORE windows. Tier T0. Writes RESULT_PT3N.json / .md."""
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
PT3_RUNS = f"{PT3}/runs"
PREFIX = f"{HERE}/RESULT_PT3N"
RR.ARMS = ("end",)
TAGS, FORMS, FAMS, QF, UNIT = RR.TAGS, RR.FORMS, RR.FAMS, RR.QF, RR.UNIT


def validity():
    rows, ok = [], True
    for t in TAGS:
        g = json.load(open(f"{RUNS}/pre_{t}.json", encoding="utf-8"))
        for tgt, c in g["control1_tie_decn"].items():
            ok &= c["passed"]
            rows.append(f"* Control 1, tie to DECN ({t}, {tgt}): N1053 D_1 {c['mine']:+.5f}, DECN ridge m=1 {c['decn']:+.5f}, difference {c['abs_diff']:.5f} (bar 0.01): {'pass' if c['passed'] else 'FAIL'}.")
        for tgt, c in g["control2_reference_n53"].items():
            ok &= c["passed"]
            rows.append(f"* Control 2, N53 reference ({t}, {tgt}): {c['mine']:+.5f} against PLANTEST3 {c['plantest3']:+.5f}, difference {c['abs_diff']:.6f} (bar 0.001): {'pass' if c['passed'] else 'FAIL'}.")
    r = json.load(open(f"{RUNS}/repro_M2b_s0.json", encoding="utf-8"))
    ok &= r["identical"]
    rows.append(f"* Control 3, harness reproduction ({r['unit']}): identical {r['identical']}, largest absolute difference {r['max_abs_diff']:.3e}: {'pass' if r['identical'] else 'FAIL'}.")
    rows.append("* Control 4, split: asserted in `run_pt3n.py` `split_guard` on real ids before every fit; `guard_mutation` requires three planted violations to raise, run at start. Pass (the run reached planning).")
    rows.append("* Control 5, clips: the 1,000 shards are DECN's, md5-verified at pull (DECN control 6); count and order asserted in `clip_order`. Pass.")
    st = json.load(open(f"{HERE}/run_pt3n.status.json", encoding="utf-8"))
    iso = bool(st.get("tune_choice_sha_unchanged"))
    ok &= iso and st.get("errors") == 0
    rows.append(f"* Control 6, isolation: PLANTEST3's `tune_choice.json` sha256 unchanged {iso}; run errors {st.get('errors')}: {'pass' if iso and st.get('errors') == 0 else 'FAIL'}.")
    return ok, rows


def vs_pt3(choice_new, choice_old):
    ctxs = {t: R3.prepare3(t) for t in TAGS}
    sc = ctxs[TAGS[0]].score_idx
    uniq, Wt, pos = P.boot_setup(ctxs[TAGS[0]].eid[sc])

    def bdiff(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return P.boot_mean_diff(a[m], b[m], pos[m], Wt)

    out = {}
    for t in TAGS:
        c = ctxs[t]
        for f in FORMS:
            for fam in FAMS:
                ln, lo = choice_new[f"{t}|end|{f}|{fam}"], choice_old[f"{t}|end|{f}|{fam}"]
                pn = np.load(f"{RUNS}/score_{t}_flat1_end_{f}_true_L{RR.lam_s(ln)}_p0.npz")["plan_raw"]
                po = np.load(f"{PT3_RUNS}/score_{t}_flat1_end_{f}_true_L{RR.lam_s(lo)}_p0.npz")["plan_raw"]
                mn, mo = P.window_metrics(pn, c.A[sc], c.v0[sc]), P.window_metrics(po, c.A[sc], c.v0[sc])
                q = QF[fam]
                d = bdiff(mn[q], mo[q])
                out[f"{t}|{f}|{fam}"] = dict(unit=UNIT[q], lam_new=ln, lam_plantest3=lo, pt3n_minus_plantest3=d, sep=P.sep_label(*d))
    return out


def main():
    ok, vrows = validity()
    out = RR.main(RUNS, f"{HERE}/tune_choice.json", PREFIX, 0)
    cn = json.load(open(f"{HERE}/tune_choice.json", encoding="utf-8"))["choice"]
    co = json.load(open(f"{PT3}/tune_choice.json", encoding="utf-8"))["choice"]
    cmp_ = vs_pt3(cn, co)
    out["pt3n_valid"] = bool(ok)
    out["pt3n_minus_plantest3"] = cmp_
    json.dump(out, open(PREFIX + ".json", "w", encoding="utf-8"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    body = open(PREFIX + ".md", encoding="utf-8").read().split("\n", 1)[1]
    head = ["# RESULT_PT3N: PLANTEST3's FLAT1 DEC-END with the decoder fitted on N1053", "Card `../../PREREG_V8F_PT3N.md`. Tier **T0**, an oracle-goal diagnostic, never a driving result. TD3 and DEC-PATH are not run, so their lines below read INCOMPLETE or NO by construction. MEASURED: `plantest3/n1053/runs/`.", "", "## Validity first (the read is INVALID if one control fails)", f"* **Verdict on validity: {'all controls pass' if ok else 'INVALID (see rows)'}.**"] + vrows + ["", "## PT3N minus PLANTEST3, G-TRUE, same SCORE windows (paired EC-BOOT, B 2,000; beside, no bar; negative is lower error)", "", "| seed | form | family | lambda (N1053 / N53-to-70 fit) | difference [95 %] | separated |", "|---|---|---|---|---|---|"]
    for k, v in cmp_.items():
        t, f, fam = k.split("|")
        head.append(f"| {t} | {f} | {fam} | {RR.lam_s(v['lam_new'])} / {RR.lam_s(v['lam_plantest3'])} | {RR.fmt(v['pt3n_minus_plantest3'])} {v['unit']} | {v['sep']} |")
    open(PREFIX + ".md", "w", encoding="utf-8").write("\n".join(head) + "\n\n" + body)
    print("valid", ok)
    print(json.dumps(out["labels"], indent=0))
    for t in out["triggers"]:
        print("TRIGGER", t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
