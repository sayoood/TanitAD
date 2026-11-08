import os, sys, json, numpy as np
A6 = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6"
sys.path.insert(0, A6); sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code")
import select_suite_tokens as SS
log = [e for e in json.load(open(f"{A6}/SUITE_CANDIDATES_A6.json"))["slots"]["TL-S"]["ordered_one_per_log"] if e["token"].startswith("090d6a17")][0]
ctx = SS.load_log(f"{SS.DB_DIR}/{log['log']}.db"); r0 = ctx.tok2rank[log["token"]]
K = np.arange(-20, 151); idx = r0 + 2 * K
yaw = ctx.eyaw[idx]; v = ctx.ev[idx]
x = ctx.ex[idx] + np.cos(yaw) * SS.EGO_COG_OFF; y = ctx.ey[idx] + np.sin(yaw) * SS.EGO_COG_OFF
def arc(x, y): return np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
cs = arc(x, y); i0 = 77
print("arc to tick77 from tick k (k, K, secs, arc, v_k):")
for k in range(0, 40, 2): print(k, K[k], K[k]/10, round(cs[i0]-cs[k], 3), round(float(v[k]), 2))
# which start tick gives 28.01 to the stop tick? and 28.11 to the last pose
for tgt, end in ((28.01, i0), (28.11, 170)):
    d = np.abs((cs[end] - cs) - tgt); k = int(np.argmin(d[:60])); print("target", tgt, "best start tick", k, "t=", K[k]/10, "arc", round(cs[end]-cs[k], 3))
# 2 Hz subsample from t0 and from t0-2
for st in (0, 20):
    s = slice(st, None, 5); print("2Hz from tick", st, "arc to last", round(arc(x[s], y[s])[-1], 3), "to stop-ish", round(arc(x[st:i0+1:5], y[st:i0+1:5])[-1], 3))
# rear-axle path length vs box centre
xr, yr = ctx.ex[idx], ctx.ey[idx]; print("rear-axle arc to last", round(arc(xr, yr)[-1], 3), "box", round(cs[-1], 3))
# time stamps / dt regularity
try:
    ts = ctx.t[idx] if hasattr(ctx, "t") else None
    print("dt min/max", None if ts is None else (np.diff(ts).min(), np.diff(ts).max()))
except Exception as ex: print("no t", ex)
print("v integral to stop (trapz, dt .1):", round(float(np.trapz(v[:i0+1], dx=0.1)), 3), " v at 0..:", np.round(v[:25:4], 2))
