#!/usr/bin/env python3
"""n1dwprobe_read.py -- prints the PROBE-N1D-FEAT result (PREREG_V8F_N1DW.md sec 10): every contrast with its CI, the per-unit labels exactly as the card's table commits them,
the headline labels (HOLDS at all four units / ONE-SEED), and every control with its value. Exit 1 when a card control did not read. usage: python n1dwprobe_read.py PROBE_RESULT.json"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n1dwprobe_outcome as OC  # noqa: E402


def fmt_c(c):
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] {'separated' if c['separated'] else 'not separated'}  (n {c['n_windows']}/{c['n_episodes']} ep)"


def main(path):
    r = json.load(open(path))
    bad = []
    p = print
    p(f"PROBE-N1D-FEAT  {'SMOKE (code check, never a result)' if r.get('smoke') else 'FULL RUN'}")
    p(f"card: {r['card']}")
    p(f"tier: {r['tier']}   evidence: {r['evidence_class']}")
    p(f"estimator: {r['estimator']}")
    p(f"windows {r['n_windows']} / episodes {r['n_episodes']}; class counts {r['class_counts']}; majority share {r['majority_share']:.6f} (card 0.338215); bar {r['bar']}")
    p(f"feature dims {r['feat_dims']}; gt_ok False windows {r['gt_ok_false']}")
    p("")
    p("== accuracy (point; the CIs are in PROBE_RESULT.json) ==")
    for arm, d in r["accuracy"].items():
        for probe, ds in d.items():
            for seed, cells in ds.items():
                row = "  ".join(f"{k}={v['point']:.4f}" for k, v in sorted(cells.items()) if not k.startswith("SHUF:"))
                p(f"{arm} {probe} seed{seed}: {row}")
    p("")
    p("== contrasts A-B on the same windows (paired episode-cluster bootstrap, B 2000 seed 0) ==")
    for u, cs in r["contrasts"].items():
        p(f"-- {u}")
        for n, c in cs.items():
            p(f"   {n:14s} {fmt_c(c)}")
    for probe in ("tree", "logit"):
        o = r["outcome_" + probe]
        p("")
        p(f"== OUTCOME, {probe} probe {'(DECIDES)' if o['deciding'] else '(printed beside; a linear negative is not a negative about learnability)'} ==")
        for u, ls in o["combine"]["per_unit"].items():
            p(f"   {u}: {'INCOMPLETE' if ls is None else (', '.join(ls) if ls else 'no label holds')}")
        for lab in OC.LABELS:
            c = o["combine"]["labels"][lab]
            extra = "" if c["status"] in ("HOLDS", "NOT-HELD") else f"  holds at {c['units_holding']}; not at {c['units_not']}; incomplete {c['units_incomplete']}"
            tag = "one-seed" if c["status"] == "ONE-SEED" else c["status"]
            p(f"   {lab:24s} {tag}{extra}")
        h = o["headline"]
        p(f"   headline: HOLDS {h['holds'] or 'none'}; one-seed {h['one_seed'] or 'none'}")
        if o["deciding"]:
            for l in h["holds"]:
                p(f"   lever after RBW-NO-GAIN ({l}): {OC.NEXT_LEVER[l]}")
    p("")
    p("== controls ==")
    c = r["controls"]
    k = c["K-CONST"]
    p(f"K-CONST: majority share live {k['majority_share_live']:.6f} vs card {k['majority_share_card']}; fold-wise train-majority predictor pooled accuracy {k['foldwise_train_majority_accuracy_pooled']:.6f}; pass {k['pass']}")
    if k["pass"] is False:
        bad.append("K-CONST")
    s = c["SHUF"]
    p(f"SHUF: max |acc - 0.338215| over {len(s['cells'])} cells = {s['max_abs_diff']:.4f} (must be <= 0.02); pass {s['pass']}")
    for kk, v in s["cells"].items():
        if not v["within_0.02"]:
            p(f"   OUT OF BAND {kk}: {v['accuracy']:.4f}")
    if s["pass"] is False:
        bad.append("SHUF")
    d = c["K-DET"]
    p(f"K-DET: pass {d['pass']}  {d.get('cells', d.get('note'))}")
    if d["pass"] is False:
        bad.append("K-DET")
    g = c["REGRESSION-ARM"]
    p(f"REGRESSION-ARM ({g['definition']}): pass {g['pass']}")
    for kk, v in g["cells"].items():
        p(f"   {kk}: {fmt_c(v)}")
    if g["pass"] is False:
        bad.append("REGRESSION-ARM")
    ra = c.get("REGRESSION-ARM-ALL-ZERO (informational)")
    if ra:
        p("REGRESSION-ARM-ALL-ZERO (informational, not a card control): " + "; ".join(f"{kk}: {fmt_c(v)}" for kk, v in ra["cells"].items()))
    t = c["TREE-SEED-IDENTITY"]
    p(f"TREE-SEED-IDENTITY (informational): all identical between probe seeds {t['all_identical']}")
    for kk, v in c.items():
        if kk.startswith("K-LABEL"):
            p(f"{kk}: {json.dumps(v)}")
    p("")
    p(f"CONTROL STATUS: {'ALL CARD CONTROLS READ' if not bad else 'PROBLEM ' + ','.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
