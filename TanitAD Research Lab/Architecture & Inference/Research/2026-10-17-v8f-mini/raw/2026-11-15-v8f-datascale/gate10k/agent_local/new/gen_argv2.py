import json, os
D = "/root/dscale10k"
L = lambda n: json.load(open(f"{D}/argv/{n}.json"))["argv"]
def setf(t, f, vals):
    t = list(t); i = t.index(f); j = i + 1
    while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2): j += 1
    return t[:i + 1] + list(vals) + t[j:]
def has(t, f): return f in t
def W(n, t): json.dump({"argv": t}, open(f"{D}/argv/{n}.json", "w"), indent=1)
PH = {"--init-ckpt": ["@INIT@"], "--dagger-bank": ["@BANK@"], "--dagger-bank-sha256": ["@BANKSHA@"]}
def tmpl(t, out):
    t = setf(t, "--out", [out])
    for f, v in PH.items(): t = setf(t, f, v)
    return t
# B_rep: B_s0 with seed 2 (stage B: --seed 2, --dagger-seed 0)
a = setf(L("B_s0.A"), "--seed", ["2"]); a = setf(a, "--out", [f"{D}/runs/B_rep/A"]); W("B_rep.A", a)
b = setf(L("B_s0.B"), "--seed", ["2"]); W("B_rep.B.tmpl", tmpl(b, f"{D}/runs/B_rep/B"))
W("B_s0.B.tmpl", tmpl(L("B_s0.B"), f"{D}/runs/B_s0/B"))
W("U_s0.B.tmpl", tmpl(L("U_s0.B"), f"{D}/runs/U_s0/B"))
# R_s0: U-like, the R release and its own join (stage A only; NOT minted: the MM rules on R_COLUMNS.md first)
mr = json.load(open(f"{D}/merged/v9_labels_R_s0.derangement.json"))
r = list(L("U_s0.A"))
r = setf(r, "--out", [f"{D}/runs/R_s0/A"]); r = setf(r, "--v9-train", [f"{D}/merged/v9_labels_R_s0.npz"]); r = setf(r, "--v9-train-md5", [mr["md5"]])
r = setf(r, "--join-train", [f"{D}/merged/join_R_s0.npz"]); W("R_s0.A", r)
for n in ("B_rep.A", "B_rep.B.tmpl", "B_s0.B.tmpl", "U_s0.B.tmpl", "R_s0.A"): print(n, " ".join(json.load(open(f"{D}/argv/{n}.json"))["argv"])[:2000]); print()
