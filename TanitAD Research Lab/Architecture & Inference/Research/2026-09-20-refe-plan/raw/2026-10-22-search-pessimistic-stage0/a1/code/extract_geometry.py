"""One-off: extract NON-LABEL geometry (log names, token counts, Step A membership) of the real 1,705-token table so synthetic worlds can mimic it.
Reads only the `log` and `token` columns of the table and the `token` column of the Step A bundle. No label array is touched."""
import json, sys, collections
import numpy as np
A = sys.argv[1]
t = np.load(A + "/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz", allow_pickle=False)
b = np.load(A + "/l1_stepA_run/bundle/bundle.npz", allow_pickle=False)
log = t["log"].astype(str); tok = t["token"].astype(str)
sa = set(b["token"].astype(str).tolist())
g = collections.OrderedDict()
for lg in sorted(set(log.tolist())):
    m = log == lg
    g[lg] = {"n_tokens": int(m.sum()), "n_stepA": int(sum(1 for x in tok[m] if x in sa))}
json.dump({"what": "non-label geometry of the real 24-log table", "logs": g, "n_tokens": int(len(tok)), "n_stepA": int(len(sa))}, open(sys.argv[2], "w"), indent=1)
print(len(g), sum(v["n_tokens"] for v in g.values()), sum(v["n_stepA"] for v in g.values()))
