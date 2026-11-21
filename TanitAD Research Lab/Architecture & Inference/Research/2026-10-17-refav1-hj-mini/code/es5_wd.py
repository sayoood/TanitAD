"""PREREG_RA1_ES Addendum C (ES-5): weight decay at 1,000 steps. Selection on the A1 HOLDOUT, then the A1x gate, then K.

usage: es5_wd.py <es dir> <out.json>
  wd 1e-3 = the banked ES-4 runs (st1000/es2_verdict.json, pw/st1000_*.npz); wd 0.3 / 3.0 = st1000_wd<W>/es5_verdict.json,
  pw/st1000_wd<W>_*.npz.  Paired episode-cluster bootstrap (B = 2000, seed 0) on the SAME 2,399 eval-141 windows (asserted).
  Four families for the selected A1 (seed 0) and P10 come from that run's eval.json (`families`)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..")), "taniteval"))

import numpy as np  # noqa: E402

from taniteval import ci  # noqa: E402

TIE_M = 0.01
WDS = ("1e-3", "0.3", "3.0")


def pair(a, b, eid):
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=2000, seed=0)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated") if k in r}


def main(d, out):
    pw = os.path.join(d, "pw")
    src = {"1e-3": (os.path.join(d, "st1000", "es2_verdict.json"), "st1000_", os.path.join(d, "st1000"))}
    for w in ("0.3", "3.0"):
        src[w] = (os.path.join(d, f"st1000_wd{w}", "es5_verdict.json"), f"st1000_wd{w}_", os.path.join(d, f"st1000_wd{w}"))
    res = dict(schema="ra1-es5-wd-v1", prereg="PREREG_RA1_ES.md Addendum C", budget_steps=1000, cell="b0.08g50",
               estimator="paired episode-cluster bootstrap B=2000 seed 0 (EPISODE variance only); training-seed floor |A1_s0 - A1_s1|",
               tier="T1-class open-loop trajectory", wd={})
    ref = None
    r0 = {s: np.load(os.path.join(pw, f"st1000_R0_s{s}.npz"), allow_pickle=True)["ade6"] for s in (0, 1)}
    for w in WDS:
        vpath, pre, rundir = src[w]
        v = json.load(open(vpath, encoding="utf-8"))
        hold = v["holdout"]["ade6_mean_by_run"]
        z = {r: np.load(os.path.join(pw, pre + r + ".npz"), allow_pickle=True) for r in ("A1_b0.08g50_s0", "A1_b0.08g50_s1", "A1x_b0.08g50_s0")}
        for r, x in z.items():
            ref = x if ref is None else ref
            assert np.array_equal(x["eid"], ref["eid"]) and np.array_equal(x["t"], ref["t"]), f"wd {w} {r}: eval windows differ"
        eid, p6 = ref["eid"], ref["p10_ade6"]
        a1 = [z[f"A1_b0.08g50_s{s}"]["ade6"] for s in (0, 1)]
        floor = abs(float(a1[0].mean()) - float(a1[1].mean()))
        k = [pair(x, p6, eid) for x in a1]
        g = pair(z["A1x_b0.08g50_s0"]["ade6"], a1[0], eid)
        res["wd"][w] = dict(A1_holdout_ade6=[hold["A1:b0.08g50@0"], hold["A1:b0.08g50@1"]],
                            A1_holdout_seed_mean=(hold["A1:b0.08g50@0"] + hold["A1:b0.08g50@1"]) / 2,
                            A1x_holdout_ade6=hold.get("A1x:b0.08g50@0"), A1_eval_ade6=[float(x.mean()) for x in a1],
                            gate_A1x_minus_A1=g, gate_ok=bool(g["separated"] and g["delta"] > 0),
                            A1_minus_P10=k, A1_minus_R0_banked=[pair(a1[s], r0[s], eid) for s in (0, 1)], seed_floor=floor,
                            K_pass_if_gate=bool(all(x["separated"] and x["delta"] < 0 and abs(x["delta"]) > 2 * floor for x in k)),
                            verdict_text=v["A1x_control"]["verdict"], run_dir=rundir)
    hm = {w: res["wd"][w]["A1_holdout_seed_mean"] for w in WDS}
    m = min(hm.values())
    sel = min((w for w in WDS if hm[w] <= m + TIE_M), key=float)
    res["selected_wd"], res["holdout_seed_mean_by_wd"] = sel, hm
    res["rule"] = f"lowest seed-mean A1 holdout ADE@6s; ties within {TIE_M} m -> the SMALLER wd"
    s = res["wd"][sel]
    if not s["gate_ok"]:
        res["reading"] = f"READING 1 FAILS at wd {sel}: the A1x gate does not pass ({s['verdict_text']}). K is not read: regularisation trades the instrument away."
        res["outcome"] = "GATE_FAIL"
    elif s["K_pass_if_gate"]:
        res["reading"] = (f"GATE OK and K PASS at wd {sel}: A1 - P10 = " + " / ".join("%+.4f [%+.4f, %+.4f]" % (x["delta"], x["lo"], x["hi"]) for x in s["A1_minus_P10"])
                          + f", |delta| > 2 x floor {s['seed_floor']:.4f}. Next: A2 and SENS (with A0) at wd {sel}, 1,000 steps.")
        res["outcome"] = "K_PASS"
    else:
        res["reading"] = (f"GATE OK, K FAIL at wd {sel}: A1 - P10 = " + " / ".join("%+.4f [%+.4f, %+.4f]" % (x["delta"], x["lo"], x["hi"]) for x in s["A1_minus_P10"])
                          + ". wd alone does not decouple the two effects. Next: readout dropout / a smaller readout (gate-then-K), then data scale.")
        res["outcome"] = "K_FAIL"
    ev = os.path.join(src[sel][2], "A1_b0.08g50_s0", "eval", "eval.json")
    res["families_selected_A1_seed0"] = json.load(open(ev, encoding="utf-8")).get("families") if os.path.exists(ev) else "eval.json not pulled"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for w in WDS:
        b = res["wd"][w]
        print(f"wd {w:>5} hold {b['A1_holdout_seed_mean']:.4f} | gate {b['gate_A1x_minus_A1']['delta']:+.4f} [{b['gate_A1x_minus_A1']['lo']:+.4f}, "
              f"{b['gate_A1x_minus_A1']['hi']:+.4f}] ok={b['gate_ok']} | A1-P10 " + " / ".join("%+.4f [%+.4f, %+.4f]%s" % (x["delta"], x["lo"], x["hi"], " SEP" if x["separated"] else "") for x in b["A1_minus_P10"])
              + f" | floor {b['seed_floor']:.4f}")
    print("selected wd", sel, "|", res["reading"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
