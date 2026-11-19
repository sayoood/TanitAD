#!/usr/bin/env python3
"""n1dwceil_read.py -- prints RESULT_CEIL.json (PREREG_N1D_CEIL sec 3 outcome per seed and combined, beside lines, sec 4 controls). Exit 1 if a card control did not read."""
import json
import sys


def fc(c):
    return f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}] {'separated' if c['separated'] else 'not separated'} (n {c['n_windows']}/{c['n_episodes']} ep)"


def main(path):
    r = json.load(open(path))
    p, bad = print, []
    p(f"PREREG_N1D_CEIL  {'SMOKE (code check, never a result)' if r['smoke'] else 'FULL RUN'}   card: {r['card']}")
    p(f"tier: {r['tier']}   evidence: {r['evidence_class']}\nestimator: {r['estimator']}")
    p(f"windows {r['n_windows']} / episodes {r['n_episodes']}; class counts {r['class_counts']}; majority share {r['majority_share']:.6f}; bar {r['bar']}; probe seed {r['probe_seed']}")
    p(f"feature dims {r['feat_dims']}; F-EGO columns {r['ego_columns']}")
    p("\n== accuracy (point [95% episode-cluster CI]) ==")
    for arm, d in r["accuracy"].items():
        for probe, cells in d.items():
            p(f"{arm} {probe}: " + "  ".join(f"{k}={v['point']:.4f}[{v['lo']:.3f},{v['hi']:.3f}]" for k, v in sorted(cells.items())))
    p("\n== contrasts A-B, same windows (paired episode-cluster bootstrap, B 2000 seed 0) ==")
    for u, cs in r["contrasts"].items():
        p(f"-- {u}")
        for n, c in cs.items():
            p(f"   {n:9s} {fc(c)}")
    for probe in ("tree", "logit"):
        o = r["outcome_" + probe]
        p(f"\n== OUTCOME, {probe} probe {'(DECIDES)' if o['deciding'] else '(beside)'} ==  rule on F-TE - F-PLAN: ABOVE if delta >= {r['bar']} and lo > 0")
        for u, l in o["per_unit"].items():
            p(f"   {u}: {l}   [F-TE - F-PLAN {fc(r['contrasts'][u + '|' + probe]['TE-PLAN'])}]")
        for lab, s in o["labels"].items():
            p(f"   {lab:16s} {'one-seed' if s == 'ONE-SEED' else s}")
        p(f"   HEADLINE: {o['headline']}")
    p("\n== printed beside ==")
    for probe in ("tree", "logit"):
        for arm in ("N1D_s0", "N1D_s1"):
            a = r["accuracy"][arm][probe]
            c = r["contrasts"][f"{arm}|{probe}"]
            p(f"   {probe} {arm}: F-EGO alone {a['F-EGO']['point']:.4f}; F-TE-F-TEXT {fc(c['TE-TEXT'])}; F-ALL-F-PLAN {fc(c['ALL-PLAN'])}")
    p("\n== controls ==")
    c = r["controls"]
    k = c["K-CONST"]
    p(f"K-CONST: live {k['majority_share_live']:.6f} vs card {k['majority_share_card']}; pass {k['pass']}")
    if k["pass"] is False:
        bad.append("K-CONST")
    q = c["K-PARITY"]
    p(f"K-PARITY: pass {q['pass']}; join_ok {q['join_ok']}; live acc {q['live_accuracies_4dp']} vs card {q['card_accuracies']}; {json.dumps(q['cells'])}")
    if q["pass"] is False or not q["join_ok"]:
        bad.append("K-PARITY")
    d = c["K-EGO-DET"]
    p(f"K-EGO-DET: pass {d['pass']}; bit-identical across all four {d['F_EGO_bit_identical_across_all_four']}; 2nd vs 1st {json.dumps(d['second_extraction_vs_first'])}")
    if not d["pass"]:
        bad.append("K-EGO-DET")
    i = c["IDENTITY-ARM"]
    p(f"IDENTITY-ARM (binding): {i['n_differing_windows']} of {i['n_windows']} windows differ; pass {i['pass']}")
    if not i["pass"]:
        bad.append("IDENTITY-ARM")
    g = c["REGRESSION-ARM"]
    p(f"REGRESSION-ARM (binding): {g['definition']}; pass {g['pass']}")
    for arm, v in g["cells"].items():
        p(f"   {arm}: {v['status']}; F-TE clears F-EGO: {v['F-TE_clears_F-EGO']}; TEXTZERO-F-TE {fc(v['TEXTZERO_minus_F-TE'])}; TEXTZERO-F-EGO {fc(v['TEXTZERO_minus_F-EGO'])}; preds differing from F-EGO's {v['n_windows_TEXTZERO_pred_differs_from_F-EGO_pred']}")
    if not g["pass"]:
        bad.append("REGRESSION-ARM")
    p(f"\nCONTROL STATUS: {'ALL CARD CONTROLS READ' if not bad else 'PROBLEM ' + ','.join(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
