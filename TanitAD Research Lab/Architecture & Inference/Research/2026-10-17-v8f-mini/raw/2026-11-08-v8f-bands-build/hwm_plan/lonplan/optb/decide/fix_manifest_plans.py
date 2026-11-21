"""MM 2026-11-18: the read chain's manifest step looked for plans under <out>/plans/<tag>/ but the chain wrote them to
F:/TanitAD-scratch/actrsp/optb/plans/<tag>/, so `arms` came out empty. Rebuild the manifest with readchain_tools' own cmd_manifest
(evidence, bandmet, mech, controls from <out>), then fill `arms` from the real plans dir with the SAME rule (DONE + both seed npz).
Also records the window-identity check (eid, now) of every arm and both baselines against the manifest's windows npz."""
import hashlib
import json
import os
import sys
import types

import numpy as np

RC = "F:/TanitAD-scratch/actrsp/optb/readchain"
OUT = RC + "/out"
PLANS = "F:/TanitAD-scratch/actrsp/optb/plans"
sys.path.insert(0, RC)
import readchain_tools as T  # noqa: E402

tmp = OUT + "/manifest_decide.builder.json"
T.cmd_manifest(types.SimpleNamespace(root=OUT, out=tmp))
man = json.load(open(tmp, encoding="utf-8"))
C = T.load_checker()


def sig(p):
    z = np.load(p, allow_pickle=False)
    return [hashlib.sha256(np.ascontiguousarray(z[k]).tobytes()).hexdigest() for k in ("eid", "now")]


ref = sig(man["windows"])
arms, wcheck = {}, {}
for t in C.RUN_ORDER:
    d = f"{PLANS}/{t}"
    p0, p1 = f"{d}/plans_hwm_oracle_s0.npz", f"{d}/plans_hwm_oracle_s1.npz"
    if os.path.isfile(f"{d}/DONE") and os.path.isfile(p0) and os.path.isfile(p1):
        arms[t] = {"s0": p0, "s1": p1}
        wcheck[t] = sig(f"{d}/windows.npz") == ref
for b, v in man["baselines"].items():
    wcheck[b] = sig(os.path.dirname(v["s0"]) + "/windows.npz") == ref
assert all(wcheck.values()), wcheck
man["arms"] = arms
man["note"] = man["note"] + "; arms filled by fix_manifest_plans.py from " + PLANS + " (same DONE + both-seed rule); (eid, now) equal to the windows npz for every arm and baseline"
man["window_identity"] = wcheck
json.dump(man, open(OUT + "/manifest_decide_fixed.json", "w", encoding="utf-8"), indent=1)
os.remove(tmp)
print(json.dumps({"arms": sorted(arms), "n": len(arms), "window_identity_all": all(wcheck.values()), "controls": man["controls"]}))
