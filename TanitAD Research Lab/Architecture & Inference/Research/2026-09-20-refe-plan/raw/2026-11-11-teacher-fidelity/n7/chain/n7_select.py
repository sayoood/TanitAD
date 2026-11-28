"""n7_select.py - the N7 generator as a CASCADE over the recorded levers (smallest change first), from n7_run.py outputs. Plain python.
Within the FIRST tier that has a tolerance-certified candidate, the TF-N6 rule picks: maximum builder PDMS, ties to the earliest (generation order).
A token whose TF-N6 tier (T0 = REPAIR-D + L1 at their A1 / N5 settings) already certifies is NEVER touched by a later tier (structural no-harm).
usage: n7_select.py out.json files... [--floor 0.5]"""
import argparse
import json
import sys

sys.dont_write_bytecode = True
import n7_analyze as A  # noqa: E402

S_GRID = (0.95, 0.90, 0.85, 0.80)
S_DEEP = (0.70, 0.60, 0.50, 0.40, 0.30, 0.20)
OFF_DY = (0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30, 0.50, -0.50, 0.75, -0.75, 1.0, -1.0)
OFF_RAMP = (0.2, 1.0, 2.0)
S_MILD = (0.90, 0.80)

# (tier name, [(lever, cap filter k <= cap or None)])   the frozen generator; every lever's parameters live in n7_lib.LEVERS
TIERS = [
    ("T0_N6", [("D_c20_m05", 5), ("L1_c20_m05", 5)]),
    ("T1_L2", [("D_c20_m05", 20), ("D_c20_m10", 20), ("D_c20_m20", 20), ("L1_c20_m05", 20), ("L1_c20_m10", 20), ("L1_c20_m20", 20)]),
    ("T2_STR", [("STR_label", None), ("STR_target", None)]),
    ("T3_OFF", [("OFF_label", None), ("STR_OFF", None)]),
    ("T4_DEEP", [("STR_deep", None)]),
]


def describe(lever, k):
    if lever == "STR_label" or lever == "STR_target":
        return {"s": S_GRID[k]}
    if lever == "STR_deep":
        return {"s": S_DEEP[k]}
    if lever == "OFF_label":
        return {"dy_m": OFF_DY[k // 3], "ramp_s": OFF_RAMP[k % 3]}
    if lever == "STR_OFF":
        per = len(OFF_DY) * len(OFF_RAMP)
        return {"s": S_MILD[k // per], "dy_m": OFF_DY[(k % per) // 3], "ramp_s": OFF_RAMP[k % 3]}
    return {"iterate": k}


def cascade(row, tiers=TIERS, floor=None, spec_full=False):
    """-> None or dict(tier, lever, k, param, m) for one token row; floor = minimal admissible s of T4 (None = no floor)"""
    ref = row["ref"]
    for name, levers in tiers:
        best = None
        for lv, cap in levers:
            for x in row["levers"].get(lv, []):
                if cap is not None and x["k"] > cap:
                    continue
                p = describe(lv, x["k"])
                if name == "T4_DEEP" and floor is not None and p["s"] < floor:
                    continue
                if not A.tol(x["m"], ref, spec_full):
                    continue
                if best is None or x["m"]["pdms"] > best["m"]["pdms"]:
                    best = {"tier": name, "lever": lv, "k": x["k"], "param": p, "m": x["m"]}
        if best is not None:
            return best
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--floor", type=float, default=None)
    a = ap.parse_args(argv)
    rows = A.derive_caps(A.load(a.files))
    res = {t: cascade(r, floor=a.floor) for t, r in sorted(rows.items())}
    json.dump({"floor": a.floor, "n": len(res), "n_rescued": sum(v is not None for v in res.values()), "per_token": res}, open(a.out, "w"), indent=1)
    cnt = {}
    for v in res.values():
        key = v["tier"] if v else "NONE"
        cnt[key] = cnt.get(key, 0) + 1
    print(len(res), cnt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
