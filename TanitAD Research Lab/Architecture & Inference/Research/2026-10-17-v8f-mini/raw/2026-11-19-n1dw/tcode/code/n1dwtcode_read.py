#!/usr/bin/env python3
"""n1dwtcode_read.py -- prints RESULT_TCODE.json: every contrast with its CI, the committed outcome (PREREG_N1D_TCODE sec 4) per N1D seed and at both, every control. Exit 1 if a card control did not read.
usage: python n1dwtcode_read.py RESULT_TCODE.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n1dwtcode_outcome as OC  # noqa: E402


def fc(c):
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] {'separated' if c['separated'] else 'not separated'} (n {c['n_windows']}/{c['n_episodes']} ep)"


def main(path):
    r = json.load(open(path))
    p, bad = print, []
    p(f"PREREG_N1D_TCODE  {'SMOKE (code check, never a result)' if r['smoke'] else 'FULL RUN'}   card: {r['card']}")
    p(f"tier: {r['tier']}   evidence: {r['evidence_class']}\nestimator: {r['estimator']}")
    p(f"windows {r['n_windows']} / episodes {r['n_episodes']}; class counts {r['class_counts']}; majority share {r['majority_share']:.6f} (card 0.338215); bar {r['bar']}; probe seed {r['probe_seed']} ({r['probe_seed_note']})")
    p(f"feature dims {r['feat_dims']}; F-CODE-LAT columns {r['code_lat_cols']}")
    p("\n== accuracy (point [95% episode-cluster CI]) ==")
    for arm, d in r["accuracy"].items():
        for probe, cells in d.items():
            p(f"{arm} {probe}: " + "  ".join(f"{k}={v['point']:.4f}[{v['lo']:.3f},{v['hi']:.3f}]" for k, v in sorted(cells.items())))
    p("\n== contrasts A-B, same windows (paired episode-cluster bootstrap, B 2000 seed 0) ==")
    for u, cs in r["contrasts"].items():
        p(f"-- {u}")
        for n, c in cs.items():
            p(f"   {n:13s} {fc(c)}")
    for probe in ("tree", "logit"):
        o = r["outcome_" + probe]
        p(f"\n== OUTCOME, {probe} probe {'(DECIDES)' if o['deciding'] else '(beside; a linear negative is not a negative about learnability)'} ==")
        for u, ls in o["combine"]["per_unit"].items():
            p(f"   {u}: {'INCOMPLETE' if ls is None else (', '.join(ls) if ls else 'none')}")
        for lab in OC.LABELS:
            c = o["combine"]["labels"][lab]
            p(f"   {lab:18s} {'one-seed' if c['status'] == 'ONE-SEED' else c['status']}" + ("" if c["status"] in ("HOLDS", "NOT-HELD") else f"  holds at {c['units_holding']}; not at {c['units_not']}"))
        h = o["headline"]
        p(f"   HEADLINE: {h['outcome']}   (holds {h['holds'] or 'none'}; one-seed {h['one_seed'] or 'none'})")
        if o["deciding"]:
            for l in h["next_lever"]:
                p(f"   next lever (needs its own card and a PI go): {l}")
    p("\n== printed beside: F-PLAN - F-CODE and F-PLAN - F-TTRUNK ==")
    for u, cs in r["contrasts"].items():
        p(f"   {u}: PLAN-CODE {fc(cs['PLAN-CODE'])}\n   {u}: PLAN-TTRUNK {fc(cs['PLAN-TTRUNK'])}")
    p("\n== controls ==")
    c = r["controls"]
    k = c["K-CONST"]
    p(f"K-CONST: live {k['majority_share_live']:.6f} vs card {k['majority_share_card']}; pass {k['pass']}")
    if k["pass"] is False:
        bad.append("K-CONST")
    d = c["K-DET"]
    p(f"K-DET (n {d['n_windows']} windows, second extraction {d['second_extraction']}): pass {d['pass']}  {json.dumps(d['cells'])}")
    if d["pass"] is False:
        bad.append("K-DET")
    q = c["K-PARITY"]
    p(f"K-PARITY (re-extracted F-TEXT, F-PLAN vs banked {q['banked']}): pass {q['pass']}  {json.dumps(q['cells'])}")
    if q["pass"] is False:
        bad.append("K-PARITY")
    g = c["REGRESSION-ARM"]
    p(f"REGRESSION-ARM ({g['definition']}): pass {g['pass']}")
    for arm, v in g["cells"].items():
        p(f"   {arm}: F-CODE clears F-TEXT: {v['F-CODE-clears-F-TEXT']}; reference {v['reference_used']}; ZERO-F-CODE {fc(v['ZERO_minus_F-CODE'])}; ZERO-F-TEXT {fc(v['ZERO_minus_F-TEXT'])}")
    if g["pass"] is False:
        bad.append("REGRESSION-ARM")
    s = c["SHUF (beside; moves no outcome)"]
    p(f"SHUF (beside): max |acc - 0.338215| = {s['max_abs_diff']:.4f}; {s['note']}")
    for kk, v in s["cells"].items():
        p(f"   {kk}: {v['accuracy']:.4f}{'' if v['within_0.02'] else '  (outside 0.02 band)'}")
    p(f"\nCONTROL STATUS: {'ALL CARD CONTROLS READ' if not bad else 'PROBLEM ' + ','.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
