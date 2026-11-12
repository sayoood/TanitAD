"""N4-DAC Q2: certified selection over every banked candidate. Order rule frozen in CERT_ORDER_RULE.md before this ran."""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import n4_common as N

ORDER = ["RJ5.R", "RJ5.C", "R0.C", "RJ3.R", "RJ3.C", "R0.R", "R5HZ.R", "RJ5H5.R", "RG8.R", "R5HZ.C", "RJ5H5.C", "RG8.C", "R0.T", "RJ3.T", "RJ5.T"]
S = N.load_scores()
common = sorted(set.intersection(*[set(v) for v in S.values()]))
logs = [S["RJ5"][t]["log"] for t in common]


def cand(t, nm):
    run, f = nm.split(".")
    return S[run][t]["arms"][f]["tracked"]


def pick(t, order, strict):
    for nm in order:
        c = cand(t, nm)
        if c["dac"] == 1.0 and (not strict or (c["nc"] == 1.0 and c["ddc"] == 1.0)):
            return nm
    return None


def vec(sel, k):
    return np.array([cand(t, sel[i] or "RJ5.R")[k] for i, t in enumerate(common)])


out = {"n": len(common), "n_logs": len(set(logs)), "order": ORDER}
base = {k: np.array([cand(t, "RJ5.R")[k] for t in common]) for k in N.SUB}
rules = {"CERT-A": (ORDER, False), "CERT-B": (ORDER, True), "CERT-4": (["RJ5.R", "RJ5.C", "R0.C", "R0.T"], False)}
sels = {}
for nm, (o, st) in rules.items():
    sels[nm] = [pick(t, o, st) for t in common]
sels["ORACLE"] = []
for t in common:
    ok = [(cand(t, n)["pdms"], -ORDER.index(n), n) for n in ORDER if cand(t, n)["dac"] == 1.0]
    sels["ORACLE"].append(max(ok)[2] if ok else None)
res = {}
for nm, sel in sels.items():
    r = {"none_pass": int(sum(s is None for s in sel)),
         "choice_counts": {k: int(sum(1 for s in sel if s == k)) for k in sorted({s for s in sel if s})},
         "levels_x100": {k: 100 * float(vec(sel, k).mean()) for k in N.SUB},
         "zero_dac_tokens": int((vec(sel, "dac") == 0).sum()),
         "paired_vs_RJ5.R_x100": {k: [100 * x for x in N.boot(vec(sel, k), logs, base[k])] for k in ("pdms", "dac", "nc", "ep", "ttc", "c", "ddc")},
         "worse_flips_vs_RJ5.R": {k: int((vec(sel, k) < base[k] - 1e-9).sum()) for k in ("nc", "dac", "c", "ddc")},
         "better_flips_vs_RJ5.R": {k: int((vec(sel, k) > base[k] + 1e-9).sum()) for k in ("nc", "dac", "c", "ddc")}}
    res[nm] = r
out["rules"] = res
out["single_arm_levels_x100"] = {n: {k: 100 * float(np.mean([cand(t, n)[k] for t in common])) for k in ("pdms", "dac", "ep", "ttc")} for n in ORDER}
out["single_arm_zero_dac"] = {n: int(sum(cand(t, n)["dac"] == 0 for t in common)) for n in ORDER}
# which tokens: RJ5.R failing DAC -> who passes
fail = [t for t in common if cand(t, "RJ5.R")["dac"] == 0]
out["rj5r_dac_fail_tokens"] = {t: {n: [cand(t, n)["dac"], round(cand(t, n)["pdms"], 4)] for n in ORDER} for t in fail}
all_fail = [t for t in common if all(cand(t, n)["dac"] == 0 for n in ORDER)]
out["tokens_failing_all_15_candidates"] = all_fail
out["any_form_pass_but_RJ5R_fails"] = [t for t in fail if any(cand(t, n)["dac"] == 1 for n in ORDER)]
json.dump(out, open("Q2_cert_wider.json", "w", encoding="utf-8"), indent=1)
for nm, r in res.items():
    print(nm, "none_pass", r["none_pass"], "zeroDAC", r["zero_dac_tokens"], "PDMS %.2f DAC %.2f" % (r["levels_x100"]["pdms"], r["levels_x100"]["dac"]),
          "dPDMS", [round(x, 2) for x in r["paired_vs_RJ5.R_x100"]["pdms"]], "dDAC", [round(x, 2) for x in r["paired_vs_RJ5.R_x100"]["dac"]],
          "worse", r["worse_flips_vs_RJ5.R"], "better", r["better_flips_vs_RJ5.R"], r["choice_counts"])
print("RJ5.R fail:", fail, "all15 fail:", all_fail)
print(json.dumps(out["single_arm_zero_dac"]))
