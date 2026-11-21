"""smoke_check.py DIR -- compares the dumps of smoke R0 and RRid (planted constant reader) and prints the rule-log facts. Pod CPU, numpy."""
import glob
import json
import os
import sys

import numpy as np

d = sys.argv[1]
a = sorted(glob.glob(os.path.join(d, "R0", "dump_SMKR0_m10_causal_heads", "ep*.npz")))
b = sorted(glob.glob(os.path.join(d, "RRid", "dump_SMKRRid_m10_causal_heads", "ep*.npz")))
diff = 0
for pa in a:
    pb = os.path.join(os.path.dirname(b[0]), os.path.basename(pa)) if b else ""
    if not os.path.exists(pb):
        diff += 1
        continue
    A, B = np.load(pa), np.load(pb)
    if sorted(A.files) != sorted(B.files) or any(not np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f") for k in A.files):
        diff += 1
print("dump files R0", len(a), "RRid", len(b), "differing", diff)
rl = glob.glob(os.path.join(d, "RRid", "ticks", "rule_log_*.npz"))
side = [p for p in glob.glob(os.path.join(d, "RRid", "ticks", "*_rd.npz"))]
if rl:
    L = np.load(rl[0], allow_pickle=True)
    print("rule log rows", len(L["ci"]), "present", int(L["present"].sum()), "active", int(L["active"].sum()), "g_src unique", np.unique(L["g_src"][L["present"]])[:3], "p_src range", float(L["p_src"].min()), float(L["p_src"].max()))
if side:
    S = np.load(side[0])
    print("side rows", len(S["p"]), "p unique", np.unique(np.round(S["p"], 4))[:3], "g_read unique", np.unique(np.round(S["g_read"], 3))[:3], "ok all", bool(S["ok"].all()), "g_used unique", np.unique(S["g_used"])[:3])
for k in ("R0", "RRid"):
    j = glob.glob(os.path.join(d, k, "t1v8f_*.json"))
    if j:
        m = json.load(open(j[0]))["intervals"]["metrics"]
        print(k, "fde_last_m", m.get("fde_last_m"), "ade_dense_m", m.get("ade_dense_m"))
