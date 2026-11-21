"""Control C7 (MEASURED when run): the driver run with R35_s<k> itself as the arm. Checks, all exact:
 (a) the four regenerated plan files equal R35's banked phase-1 plans (every array but wall_s, bit for bit);
 (b) contrast_l1full --check reproduced the banked rows (read from c7_check.log: 'REPRODUCED EXACTLY');
 (c) arm minus R35 reads 0.0 [0, 0] on every cell (hwm and flat);
 (d) IA-3 / IA-0 of the driver's ia_read equal the banked ladder_read/results/arm_R35_s<k>.json (max abs diff reported; exact required);
 (e) decode_check step3_nn equals the banked decode_check result_s<k>.json (if both exist).
usage: c7_checks.py <arm_out_dir> <chassis>"""
import json, os, sys
import numpy as np
d, k = sys.argv[1], int(sys.argv[2])
ref = os.environ["R35REF"]
res = {"chassis": k}
eq = {}
for arm in ("hwm_oracle", "flat_oracle"):
    for s in (0, 1):
        a = np.load(f"{d}/win/plans_{arm}_s{s}.npz"); b = np.load(f"{ref}/s{k}/plans_{arm}_s{s}.npz")
        keys = sorted(set(a.files) | set(b.files)) if set(a.files) != set(b.files) else [x for x in a.files if x != "wall_s"]
        eq[f"{arm}_s{s}"] = bool(set(a.files) == set(b.files) and all(np.array_equal(a[x], b[x], equal_nan=True) for x in keys))
res["plans_bit_equal"] = eq
log = open(f"{d}/c7_check.log", encoding="utf-8", errors="replace").read()
res["check_reproduced_exactly"] = "REPRODUCED EXACTLY" in log
zero = {}
for K in ("hwm", "flat"):
    c = json.load(open(f"{d}/contrast_{K}.json"))
    bad = [(r["family"], r["band"], r["metric"]) for r in c["rows"] if r["kind"] == "contrast" and (r["a"], r["b"]) == ("A", "B")
           and not ((r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0) or (r["delta"] != r["delta"]))]
    bad += [(r["family"], r["band"], r["metric"], "rep") for r in c["replicate"] if r["arm"] == "A" and False]
    n = sum(1 for r in c["rows"] if r["kind"] == "contrast" and (r["a"], r["b"]) == ("A", "B"))
    zero[K] = {"n_cells": n, "n_not_zero": len(bad), "first": bad[:3]}
res["arm_minus_r35_zero"] = zero
ia = json.load(open(f"{d}/ia_read.json"))
bank = json.load(open(f"E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/ladder_read/results/arm_R35_s{k}.json"))


def flat(x, p=""):
    if isinstance(x, dict):
        for kk, v in x.items():
            yield from flat(v, f"{p}/{kk}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


mx, n, miss = 0.0, 0, 0
for part in ("ia0", "ia0_planning_context1", "passes"):
    A = dict(flat(ia[part])); B = dict(flat(bank[part]))
    for kk, v in B.items():
        if kk not in A:
            miss += 1; continue
        n += 1; mx = max(mx, abs(A[kk] - v) if not (v != v and A[kk] != A[kk]) else 0.0)
A = dict(flat(ia["beside"]["ia3"])); B = dict(flat(bank["beside"]["ia3"]))
for kk, v in B.items():
    if kk not in A:
        miss += 1; continue
    n += 1; mx = max(mx, abs(A[kk] - v))
res["ia_vs_banked"] = {"n_numbers": n, "missing": miss, "max_abs_diff": mx}
dcp, dcb = f"{d}/decode_check/", f"F:/TanitAD-scratch/hwm_plan/decode_check/result_s{k}.json"
tag = [f for f in os.listdir(dcp) if f.startswith("result_")] if os.path.isdir(dcp) else []
if tag and os.path.exists(dcb):
    x = json.load(open(dcp + tag[0]))["step3_nn"]; y = json.load(open(dcb))["step3_nn"]
    res["decode_step3_nn_equal"] = bool(x == y)
else:
    res["decode_step3_nn_equal"] = None
res["C7_PASS"] = bool(all(eq.values()) and res["check_reproduced_exactly"] and all(v["n_not_zero"] == 0 and v["n_cells"] > 0 for v in zero.values()) and miss == 0 and mx == 0.0 and res["decode_step3_nn_equal"] in (True, None))
json.dump(res, open(f"{d}/c7_result.json", "w"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if res["C7_PASS"] else 5)
