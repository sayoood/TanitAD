import sys, json, numpy as np
A6 = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6"
sys.path.insert(0, A6); sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code")
import select_suite_tokens as SS
c = json.load(open(f"{A6}/SUITE_CANDIDATES_A6.json"))["slots"]["TL-S"]["ordered_one_per_log"]
for pre, diff in (("090d6a17", 3.82), ("15715849", 1.90)):
    e = [e for e in c if e["token"].startswith(pre)][0]
    ctx = SS.load_log(f"{SS.DB_DIR}/{e['log']}.db"); r0 = ctx.tok2rank[e["token"]]
    idx = r0 + 2 * np.arange(-20, 151); T = np.asarray(ctx.T)[idx]; dt = np.diff(T); dt = dt / (1e6 if dt.mean() > 1e3 else 1)
    x, y = ctx.ex[idx], ctx.ey[idx]; cs = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    print(pre, "dt min/max", dt.min().round(4), dt.max().round(4), "first tick whose arc>=diff", int(np.argmax(cs >= diff)), "arc[k]", np.round(cs[:8], 2))
