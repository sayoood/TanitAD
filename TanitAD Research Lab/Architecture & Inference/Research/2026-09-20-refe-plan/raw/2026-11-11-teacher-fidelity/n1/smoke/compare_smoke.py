"""N1 smoke on navtrain held-out: the build -> transp_score path against TF-1 / TF-2 banked rows, per token, per sub-score."""
import glob, json, sys
TF = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity"
def load(pat):
    d = {}
    for p in sorted(glob.glob(pat)):
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if "error" not in r: d.setdefault(r["token"], r)
    return d
tf1, tf2 = load(f"{TF}/tf1/rows/tf1_s*.jsonl"), load(f"{TF}/tf2/rows/tf2_s*.jsonl")
rows = load(sys.argv[1])
SUB = ("nc", "dac", "ddc", "ttc", "ep", "c", "mult", "pdms")
PAIRS = [("N1T", "tracked", tf2, "T_tracked"), ("N1T", "tracked", tf1, "T_tracked"), ("N1T", "transparent", tf2, "T_transparent"),
         ("N1C", "tracked", tf1, "Tc20s_tracked"), ("N1C", "tracked", tf2, "Tc20s_tracked"), ("N1C", "transparent", tf1, "Tc20s_transparent"),
         ("N1W", "tracked", tf1, "Tw_tracked"), ("N1M", "tracked", tf2, "M_tracked"), ("N1M", "transparent", tf2, "M_transparent"),
         ("human", "tracked", tf1, "H_tracked"), ("human", "tracked", tf2, "H_tracked")]
out = {"n_tokens": len(rows), "pairs": {}}
for arm, fr, ref, fld in PAIRS:
    worst, n, nd = 0.0, 0, 0
    for t, r in rows.items():
        if t not in ref or fld not in ref[t]: continue
        d = max(abs(r["arms"][arm][fr][m] - ref[t][fld][m]) for m in SUB)
        worst, n, nd = max(worst, d), n + 1, nd + (d != 0.0)
    out["pairs"][f"{arm}.{fr} vs {'TF1' if ref is tf1 else 'TF2'}.{fld}"] = {"n": n, "tokens_not_bit_equal": nd, "max_abs_diff": worst}
def cert_tf2(r):
    for c in ("Tc20s_tracked", "T_tracked", "M_tracked"):
        if r[c]["dac"] == 1.0: return {"Tc20s_tracked": "N1C", "T_tracked": "N1T", "M_tracked": "N1M"}[c]
    return "NONE"
def cert_n1(r):
    for c in ("N1C", "N1T", "N1M"):
        if r["arms"][c]["tracked"]["dac"] == 1.0: return c
    return "NONE"
ch = [(cert_tf2(tf2[t]), cert_n1(r)) for t, r in rows.items() if t in tf2]
out["cert_choice_equal"] = sum(a == b for a, b in ch); out["cert_n"] = len(ch)
out["cert_choices"] = {k: sum(b == k for _, b in ch) for k in ("N1C", "N1T", "N1M", "NONE")}
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out, indent=1))
