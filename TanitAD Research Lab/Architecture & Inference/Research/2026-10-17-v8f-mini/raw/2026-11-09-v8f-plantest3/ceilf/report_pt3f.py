"""PREREG_V8F_PT3F report: reads runs/pt3f_<tag>.npz, checks the controls, writes RESULT_PT3F.md and RESULT_PT3F.json. T0, SCORE, paired EC-BOOT B 2,000."""
import json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PT3 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, PT3); sys.path.insert(0, os.path.join(PT3, "mlp4"))
import numpy as np
import pt3m_lib as PM
from pt3m_lib import P
import plantest3_real as R3

TAGS = ("M2b_s0", "M2b_s1")
RUNS = f"{HERE}/runs"; MLP4 = f"{PT3}/mlp4"
FAM = (("LONGITUDINAL", "acc_mae", "m/s^2"), ("LATERAL", "yaw_mae", "rad/s"))
ARMS = (("f_decide", "F-DECIDE (deciding)"), ("f_other", "F-OTHER (beside)"), ("n1053_lr1e3", "N1053-LR1E3 (beside)"), ("pt3m", "PT3M (tie)"), ("f_decide_readdiff", "F-DECIDE, read difference (beside; acceleration only)"), ("oracle", "ORACLE-END (beside)"))


def main():
    res, md, valid = {}, [], True
    for t in TAGS:
        c = R3.prepare3(t)
        sc = np.asarray(c.score_idx)
        z = np.load(f"{RUNS}/pt3f_{t}.npz")
        pre = json.load(open(f"{RUNS}/pre_{t}.json", encoding="utf-8"))
        assert np.array_equal(z["idx"], sc)
        _, Wt, pos = P.boot_setup(c.eid[sc])
        bm = lambda x: [float(v) for v in P.boot_mean(x[np.isfinite(x)], pos[np.isfinite(x)], Wt)]
        def bd(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            d = P.boot_mean_diff(a[m], b[m], pos[m], Wt)
            return [float(v) for v in d] + [P.sep_label(*d)]
        mets = lambda p: P.window_metrics(p, c.A[sc], c.v0[sc])
        M = {k: mets(z[k]) for k in ["hold"] + [a for a, _ in ARMS]}
        ceil = np.load(f"{MLP4}/runs/ceil_{t}.npz")
        cm = {"read": mets(ceil["read"]), "oracle": mets(ceil["oracle"])}
        hold_ref = mets(P.hold_plan(c.A[sc]))
        def maxd(a, b):
            m = np.isfinite(a) & np.isfinite(b)
            return float(np.abs(a[m] - b[m]).max()), bool((np.isfinite(a) == np.isfinite(b)).all())
        ctl = {}
        for q in ("acc_mae", "yaw_mae"):
            ctl[f"c2_pt3m_tie_{q}"] = maxd(M["pt3m"][q], cm["read"][q])
            ctl[f"c3_oracle_tie_{q}"] = maxd(M["oracle"][q], cm["oracle"][q])
            ctl[f"c3_hold_tie_{q}"] = maxd(M["hold"][q], hold_ref[q])
        c1 = pre["control1_tie_fulln"]
        ok = all(v[0] <= 1e-9 and v[1] for v in ctl.values()) and all(v["passed"] for v in c1.values())
        valid &= ok
        rec = dict(controls=ctl, control1=c1, valid=bool(ok), speed_form=pre["speed_form_deciding"], held_r2=pre["held_r2"], epochs=pre["epochs"], fam={})
        for fam, q, unit in FAM:
            h = M["hold"][q]
            row = dict(unit=unit, hold=bm(h), arms={})
            for a, _ in ARMS:
                row["arms"][a] = dict(mean=bm(M[a][q]), minus_hold=bd(M[a][q], h))
            rec["fam"][fam] = row
        rec["read_err_k12"] = dict(speed_abs=float(np.nanmean(np.abs(z["v12_decide"] - z["v12_true"]))), yaw_abs=float(np.nanmean(np.abs(z["w12_decide"] - z["w12_true"]))))
        rec["beside"] = {a: dict(ade=bm(M[a]["ade"]), speed_err=bm(M[a]["speed_err"]), heading_err=bm(M[a]["heading_err"])) for a in ("hold", "oracle", "f_decide", "pt3m")}
        res[t] = rec
    labels = {}
    for fam, _, _ in FAM:
        s = [res[t]["fam"][fam]["arms"]["f_decide"]["minus_hold"][3] for t in TAGS]
        n = sum(x == "better" for x in s)
        labels[fam] = "READS" if n == 2 else ("SEEDS-DISAGREE" if n == 1 else "NOT-READ")
    res["labels"], res["valid"] = labels, bool(valid)
    json.dump(res, open(f"{HERE}/RESULT_PT3F.json", "w", encoding="utf-8"), indent=1)
    md += ["# RESULT_PT3F: READ-END with readers fitted on every trained clip (N4565)", "Card `PREREG_V8F_PT3F.md`. Tier **T0**, an oracle-goal ceiling with no world-model rollout and no planner, never a driving result. MEASURED: `RESULT_PT3F.json`, `runs/`. Paired EC-BOOT over the 71 SCORE episodes (994 windows), B 2,000. TACTICAL: not computed (n = 0, no decision output). STRATEGIC: PAUSED (PI item 25).", "", f"**Controls: {'all pass' if valid else 'FAILED, the read is INVALID'}.**", ""]
    for t in TAGS:
        r = res[t]
        md.append(f"* {t}: control 1 " + ", ".join(f"{k} {v['mine']:+.4f} against {v['fulln']:+.4f}" for k, v in r["control1"].items()) + "; controls 2 and 3 largest window difference " + ", ".join(f"{k} {v[0]:.1e}" for k, v in r["controls"].items()))
        md.append(f"* {t}: held-out R2 speed ridge {r['held_r2']['speed_ridge']:+.3f}, speed MLP {r['held_r2']['speed_mlp']:+.3f} (deciding form: {r['speed_form']}); yaw MLP {r['held_r2']['yaw_mlp']:+.3f}; N1053 yaw MLP at 1e-3 {r['held_r2']['n1053_yaw_mlp_lr1e3']:+.3f}; absolute read error at knot 12 of F-DECIDE: speed {r['read_err_k12']['speed_abs']:.2f} m/s, yaw rate {r['read_err_k12']['yaw_abs']:.4f} rad/s")
    for fam, _, unit in FAM:
        md += ["", f"## {fam} ({unit}; MAE against the 12 logged actions)", "| seed | arm | HOLD | arm MAE | arm - HOLD | separated |", "|---|---|---|---|---|---|"]
        for t in TAGS:
            f = res[t]["fam"][fam]
            for a, nm in ARMS:
                x = f["arms"][a]
                md.append(f"| {t} | {nm} | {f['hold'][0]:.4f} | {x['mean'][0]:.4f} [{x['mean'][1]:.4f}, {x['mean'][2]:.4f}] | {x['minus_hold'][0]:+.4f} [{x['minus_hold'][1]:+.4f}, {x['minus_hold'][2]:+.4f}] | {x['minus_hold'][3]} |")
    md += ["", "## Beside, no bar: at 2.4 s (EC-BOOT mean)", "| seed | plan | ADE [m] | speed err [m/s] | heading err [rad] |", "|---|---|---|---|---|"]
    for t in TAGS:
        for a in ("hold", "oracle", "f_decide", "pt3m"):
            b = res[t]["beside"][a]
            md.append(f"| {t} | {a} | {b['ade'][0]:.4f} | {b['speed_err'][0]:.4f} | {b['heading_err'][0]:.4f} |")
    md += ["", "## Outcome per family (F-DECIDE, both seeds)"] + [f"* **{k}: {v}**" for k, v in labels.items()]
    open(f"{HERE}/RESULT_PT3F.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("valid", valid, json.dumps(labels))
    for t in TAGS:
        print(t, res[t]["speed_form"], res[t]["held_r2"], res[t]["read_err_k12"])


if __name__ == "__main__":
    main()
