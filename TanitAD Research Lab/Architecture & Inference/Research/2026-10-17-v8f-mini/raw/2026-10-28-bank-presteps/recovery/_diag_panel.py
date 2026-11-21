import sys, math, collections
import numpy as np
sys.path.insert(0, ".")
import labeller as L
import test_labeller as TL
panel = TL._panel()
cnt = collections.Counter(); byv = collections.defaultdict(collections.Counter)
for gg, vv, r, own, v_own in panel:
    res = L.recovery_label(gg, vv, r, own, v_own)
    cnt[res["status"]] += 1
    byv[int(v_own // 3)][res["status"]] += 1
print(cnt)
for k in sorted(byv): print("v_own bin", k*3, "-", k*3+3, dict(byv[k]))
# for refused: which check fails at the last rung?
fails = collections.Counter()
for gg, vv, r, own, v_own in panel:
    res = L.recovery_label(gg, vv, r, own, v_own, check=False, ladder=(30,))
    if res["label_own"] is not None:
        v = L.violations(res["label_own"], v_own, 31)
        fails.update({k: v[k] > 0 for k in ("accel","curv","fric")} if False else {k: int(v[k] > 0) for k in ("accel","curv","fric")})
    else: fails[res["status"]] += 1
print("rung-30 failure modes", fails)
