import sys, json, numpy as np
A6 = "E:/Projects/TanitAD-artifacts/tsf_bundle2_20261107/a6"
sys.path.insert(0, A6); sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code")
import select_suite_tokens as SS
c = json.load(open(f"{A6}/SUITE_CANDIDATES_A6.json"))["slots"]["TL-S"]["ordered_one_per_log"]
mf = json.load(open(f"{A6}/MAP_FIELDS_A6.json"))["fields"]["TL-S"]
for pre, sim_line in (("090d6a17", 34.54), ("15715849", 61.10), ("7a7ff4cc", 76.08), ("b7c35b12", 33.31)):
    e = [e for e in c if e["token"].startswith(pre)][0]
    ctx = SS.load_log(f"{SS.DB_DIR}/{e['log']}.db"); r0 = ctx.tok2rank[e["token"]]
    idx = r0 + 2 * np.arange(-20, 151); yaw = ctx.eyaw[idx]
    x = ctx.ex[idx] + np.cos(yaw) * SS.EGO_COG_OFF; y = ctx.ey[idx] + np.sin(yaw) * SS.EGO_COG_OFF
    cs = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    ours = [v for k, v in mf.items() if k.startswith(pre)][0]["stop_line_s"] if True else None
    print(pre, "ours line", round(ours, 2), "sim", sim_line, "diff", round(ours - sim_line, 2), "| arc t-2->t-1.5", round(cs[5], 3), "t-2->t0", round(cs[20], 2), "v0", round(float(ctx.ev[idx][0]), 2))
