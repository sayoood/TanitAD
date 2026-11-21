"""token_census.py -- counts of the strategic tokens (rows and clips) per split; names the tokens that never occur."""
import json, sys
import numpy as np
out = {}
for sp in ("train", "eval"):
    z = np.load(f"F:/strat_out/band_labels/strategic_tok_{sp}.npz")
    G = [str(x) for x in z["g_tokens"]]; A = [str(x) for x in z["a_tokens"]]
    g, a, gv, av = z["g_str_id"], z["a_str_id"], z["g_valid"], z["a_valid"]
    off, n = z["clip__off"], z["clip__n"]
    clip_of = np.repeat(np.arange(len(off)), n)
    r = {"rows": int(len(g)), "rows_g_valid": int(gv.sum()), "rows_unclaimed": int((g < 0).sum())}
    for name, ids, valid, V in (("g_str", g, gv, G), ("a_str", a, av, A)):
        rows = {V[i]: int(((ids == i) & valid).sum()) for i in range(len(V))}
        clips = {V[i]: int(len(np.unique(clip_of[(ids == i) & valid]))) for i in range(len(V))}
        r[name] = {"rows_valid": rows, "clips_with_valid_row": clips, "never_occur": [t for t in V if rows[t] == 0]}
    by, w = z["by_time_s"], z["within_m"]
    pos = (g > 0) & gv
    r["by_time_s_pctl_5_25_50_75_95_positive_rows"] = np.nanpercentile(by[pos], [5, 25, 50, 75, 95]).round(1).tolist()
    r["within_m_pctl_5_25_50_75_95_positive_rows"] = np.nanpercentile(w[pos], [5, 25, 50, 75, 95]).round(1).tolist()
    r["positive_rows_by_time_in_6_8"] = int(((by >= 6) & (by < 8) & pos).sum())
    r["positive_rows_by_time_in_8_30"] = int(((by >= 8) & pos).sum())
    r["positive_rows_total"] = int(pos.sum())
    # first-frame view (step 0 of each clip): comparable to the v7.2 per-clip record counts
    s0 = off
    r["step0_g_str"] = {G[i]: int(((g[s0] == i) & gv[s0]).sum()) for i in range(len(G))}
    out[sp] = r
json.dump(out, open("F:/strat_out/band_labels/strategic_tok_census.json", "w"), indent=1)
print(json.dumps(out, indent=1))
