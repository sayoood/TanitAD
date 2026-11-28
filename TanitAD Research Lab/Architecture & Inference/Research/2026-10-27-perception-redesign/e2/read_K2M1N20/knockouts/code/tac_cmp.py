"""Thor-side, read-only: TACTICAL family for K2 vs its three planner knockouts (battery tactical_v6_s{0,1}.json, DEV+HOLD as the battery writes it,
episode-cluster bootstrap CI from the battery). Prints every *.acc.mean / kappa leaf so the rows can be compared; writes JSON to argv[1]."""
import json
import sys

R = "/home/nvidia/r8ns/r9e2/runs/"
ARMS = {"K2": R + "E2_READ_K2M1N20/E2_K2_pf", "K2_mapzero": R + "E2_ABL_K2M1N20/E2_K2abl_mapzero_pf",
        "K2_boxzero": R + "E2_ABL_K2M1N20/E2_K2abl_boxzero_pf", "K2_drvzero": R + "E2_ABL_K2M1N20/E2_K2abl_drvzero_pf"}


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else k)
    elif isinstance(o, (int, float)):
        yield p, o


out = {"class": "DIAGNOSTIC", "source": "battery tactical_v6_s{seed}.json per arm", "rows": {}}
for s in ("0", "1"):
    for arm, d in ARMS.items():
        try:
            j = json.load(open(f"{d}/tactical_v6_s{s}.json", encoding="utf-8"))
        except FileNotFoundError:
            out["rows"][f"s{s} {arm}"] = "MISSING"
            print(f"s{s} {arm}: MISSING")
            continue
        sel = {k: round(v, 4) for k, v in leaves(j) if k.endswith(".acc.mean") or k.endswith(".kappa") or k.endswith(".acc.lo") or k.endswith(".acc.hi")}
        out["rows"][f"s{s} {arm}"] = sel
        print(f"s{s} {arm}: " + " | ".join(f"{k}={v}" for k, v in sel.items() if k.endswith(".acc.mean") or k.endswith(".kappa")))
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
