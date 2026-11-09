"""Temporal treatment of the pilot's weather, surface and visibility labels (CPU only). Candidates: raw probe, 1 s median (the pilot's rule), sticky forward-backward
with a fixed switch rate, one label per clip. Committed choice, written before this ran: the candidate with the highest agreement with Qwen at the 1 Hz frames in
PROBE-ONLY mode (a clip's own Qwen answers are not used; probes out of fold by clip), ties to the larger smoothing. The 60 PI frames are a check, not the selector.
Output: SL/SMOOTH_EVAL.json."""
import base64, glob, json, os, re, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, pandas as pd
import merge_context as M, extract_context as X, audit_prompts as AP
SW = (0.05, 0.02, 0.01, 0.005, 0.002)
CL = {"weather": X.WEATHER, "surface": X.SURFACE, "visibility": X.VISIB}
VK = {"weather": "v_weather", "surface": "v_surface", "visibility": "v_visibility"}
tab = pd.read_parquet(D + "/splits/clips_table.parquet")
clips = sorted(os.path.basename(p)[:-4] for p in glob.glob("F:/strat_out/pilot/ctxB/*.npz"))
data = {c: M.load_parts("F:/strat_out/pilot", c) for c in clips}
train = [c for c in tab[tab.split == "train4512"].clip_id if c in data]
P = M.fit_probes(data, train)
A = json.load(open(D + "/pilot_audit_answers_PI_60.json"))["frames"]
def treat(p, name):
    if name == "raw": return p
    if name == "median": return M.median_smooth(p, np.zeros(len(p), bool))
    if name == "clip": return np.repeat(p.mean(0, keepdims=True), len(p), 0)
    return M.sticky_posterior(p, float(name))
res = {}
for f, cl in CL.items():
    oof = P[f][0]
    res[f] = {}
    for name in ["raw", "median", "clip"] + [str(s) for s in SW]:
        agree = n = flips = steps = 0; lab = {}
        for c, p in oof.items():
            B = data[c][1][VK[f]]
            q = treat(p, name).argmax(1)
            lab[c] = q
            flips += int((q[1:] != q[:-1]).sum()); steps += len(q) - 1
            i = np.arange(len(B)) * 5
            i = i[i < len(q)]
            agree += int((q[i] == B[: len(i)].argmax(1)).sum()); n += len(i)
        row = {"agree_qwen_pct": round(100 * agree / n, 2), "n": n, "flip_pct_per_step": round(100 * flips / steps, 3)}
        for scope, sel in (("random", lambda a: a["stratum"] == "random"), ("all", lambda a: True)):
            pairs = [(a[f], cl[int(lab[a["clip_id"]][a["frame_idx"]])]) for a in A if sel(a) and a[f] in cl and a["clip_id"] in lab]
            k = sum(y == x for y, x in pairs)
            row[f"pi_{scope}"] = [k, len(pairs)] + AP.wilson(k, len(pairs))
        res[f][name] = row
    names = list(res[f])
    order = ["raw", "median"] + [str(s) for s in SW] + ["clip"]
    res[f]["_chosen"] = max(order, key=lambda m: (res[f][m]["agree_qwen_pct"], order.index(m)))
json.dump(res, open(D + "/SMOOTH_EVAL.json", "w"), indent=1)
for f in res:
    print(f, "chosen", res[f]["_chosen"])
    for k, v in res[f].items():
        if k != "_chosen": print("  ", k, v)
