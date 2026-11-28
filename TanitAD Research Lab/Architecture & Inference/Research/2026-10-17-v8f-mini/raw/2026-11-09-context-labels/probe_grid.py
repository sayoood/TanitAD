"""Probe settings for the per-frame labels (CPU). Pilot finding that prompted it: Qwen says rain on 6.8 % of 1 Hz frames, the merged label said 0.0 %; accuracy hid it.
Committed selector, written before this ran: per field, the highest macro-F1 against Qwen at the 1 Hz frames, out of fold by clip (5 folds, 187 pilot train clips).
The PI's frames are a check only. Output: SL/PROBE_GRID.json."""
import glob, json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
import merge_context as M, extract_context as X
CL = {"weather": (X.WEATHER, "v_weather"), "surface": (X.SURFACE, "v_surface"), "visibility": (X.VISIB, "v_visibility")}
GRID = [(c, w) for c in (0.5, 5.0, 50.0, 500.0) for w in (None, "balanced")]
tab = pd.read_parquet(D + "/splits/clips_table.parquet")
clips = sorted(os.path.basename(p)[:-4] for p in glob.glob("F:/strat_out/pilot/ctxB/*.npz"))
data = {c: M.load_parts("F:/strat_out/pilot", c) for c in clips}
train = [c for c in tab[tab.split == "train4512"].clip_id if c in data]
A = json.load(open(D + "/pilot_audit_answers_PI_60.json"))["frames"]
res = {}
for f, (cl, vk) in CL.items():
    Xa, ya, ga = [], [], []
    for c in train:
        a, b = data[c]; F = M.feats(a)[::5][: len(b[vk])]
        Xa.append(F); ya.append(b[vk].argmax(1)[: len(F)]); ga += [c] * len(F)
    Xa, ya, ga = np.concatenate(Xa), np.concatenate(ya), np.array(ga)
    res[f] = {"n": int(len(ya)), "qwen_share": {cl[i]: round(float((ya == i).mean()), 4) for i in range(len(cl))}}
    for C, w in GRID:
        pred = np.zeros(len(ya), int); oof = {}
        for tr, te in GroupKFold(5).split(Xa, ya, ga):
            m = LogisticRegression(max_iter=1000, C=C, class_weight=w).fit(Xa[tr], ya[tr])
            pred[te] = m.classes_[m.predict_proba(Xa[te]).argmax(1)]
            for c in set(ga[te].tolist()):
                oof[c] = M._proba(m, M.feats(data[c][0]), len(cl)).argmax(1)
        row = {"macro_f1": round(float(f1_score(ya, pred, average="macro", labels=range(len(cl)), zero_division=0)), 4), "acc": round(float((pred == ya).mean()), 4),
               "recall": {cl[i]: round(float((pred[ya == i] == i).mean()), 3) if (ya == i).any() else None for i in range(len(cl))},
               "pred_share": {cl[i]: round(float((pred == i).mean()), 4) for i in range(len(cl))}}
        pairs = [(a[f], cl[int(oof[a["clip_id"]][a["frame_idx"]])]) for a in A if a[f] in cl and a["clip_id"] in oof]
        row["pi_all"] = [sum(y == p for y, p in pairs), len(pairs)]
        row["pi_recall"] = {k: [sum(p == k for y, p in pairs if y == k), sum(1 for y, p in pairs if y == k)] for k in cl}
        res[f][f"C{C}_{w}"] = row
    keys = [k for k in res[f] if k.startswith("C")]
    res[f]["_chosen"] = max(keys, key=lambda k: res[f][k]["macro_f1"])
json.dump(res, open(D + "/PROBE_GRID.json", "w"), indent=1)
for f in res:
    print(f, res[f]["n"], res[f]["qwen_share"], "chosen", res[f]["_chosen"])
    for k, v in res[f].items():
        if k.startswith("C"): print("  ", k, v["macro_f1"], v["acc"], v["recall"], "PI", v["pi_all"], v["pi_recall"])
