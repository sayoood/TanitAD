import sys, json, numpy as np
A6 = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6"
sys.path.insert(0, A6); sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code")
import select_suite_tokens as SS
c = json.load(open(f"{A6}/SUITE_CANDIDATES_A6.json"))["slots"]["TL-S"]["ordered_one_per_log"]
for pre in ("090d6a17", "15715849", "7a7ff4cc"):
    e = [e for e in c if e["token"].startswith(pre)][0]
    ctx = SS.load_log(f"{SS.DB_DIR}/{e['log']}.db"); r0 = ctx.tok2rank[e["token"]]
    idx = r0 + 2 * np.arange(-20, 151)
    tn = [a for a in dir(ctx) if not a.startswith("_")]
    ts = getattr(ctx, "ets", None)
    if ts is None:
        ts = next((getattr(ctx, a) for a in tn if a in ("t", "ts", "time", "etime", "ets_us", "lts")), None)
    print(pre, "attrs", tn[:25])
    if ts is not None:
        dt = np.diff(np.asarray(ts)[idx]) / (1e6 if np.asarray(ts)[idx][0] > 1e12 else 1)
        print("  dt min/max/mean", dt.min(), dt.max(), dt.mean())
    st = np.hypot(np.diff(ctx.ex[idx]), np.diff(ctx.ey[idx])); print("  max step", st.max().round(2), "argmax", int(st.argmax()), "len idx range", idx[0], idx[-1], "n ego", len(ctx.ex))
