"""P6-READ-DRYRUN Q5 support: the paired read P6 will get, exercised on two EXISTING arms (no P6 involved).

Reads the C2 (P2) and C1 (SFT-4 arm B u400) per-token score CSVs from the dev box and pairs them with the SAME estimator PREREG_P6 registers
(proxy_eval.boot: paired log-cluster bootstrap over the 136 logs, 10,000 resamples, seed 20260927). KNOWN VALUE: PREREG_P6 quotes C2 - C1 = +0.57 [+0.02, +1.18]
(eval/RESULT_P2.md). The same function with (P6, C1) and (P6, C2) is the P6 read. Records md5 of each input so a `p2` baseline can be pinned.
"""
import gzip, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/par18/p6dry/pkx/eval")
import proxy_eval as PE
D = "D:/Projects/TanitAD/data/refe_navtest"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
F = {"C1_u400": f"{D}/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv",
     "C2_p2": f"{D}/score/refe_stage2_p2_full/refe_stage2_p2_full.csv"}
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
E = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
tl = {t: E[t]["log_name"] for t in E}
T = {k: {t: 100.0 * float(r["score"]) for t, r in PE.read_csv(p).items()} for k, p in F.items()}
toks = sorted(set(T["C1_u400"]) & set(T["C2_p2"]))
out = {"files": {k: {"path": p, "md5": md5(p), "n": len(T[k])} for k, p in F.items()}, "n_common": len(toks), "n_logs": len({tl[t] for t in toks}),
       "pdms_x100": {k: float(np.mean([T[k][t] for t in toks])) for k in T}}
d = {t: T["C2_p2"][t] - T["C1_u400"][t] for t in toks}
mu, lo, hi = PE.boot(d, tl)
out["C2_minus_C1"] = {"mean": mu, "lo": lo, "hi": hi, "better": sum(v > 0 for v in d.values()), "worse": sum(v < 0 for v in d.values()), "tied": sum(v == 0 for v in d.values())}
out["registered_in_PREREG_P6"] = {"C1": 88.25, "C2": 88.82, "C2_minus_C1": "+0.57 [+0.02, +1.18]"}
json.dump(out, open("C:/Users/Admin/par18/p6dry/raw/pair_known_value_C2_vs_C1.json", "w"), indent=1)
print(json.dumps(out, indent=1))
