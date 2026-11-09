"""Which clips get Qwen at 1 Hz in the full run (about 1,000 of 4,659), written after the pilot and the PI's audit, before any full-run Qwen answer exists.

Always in: the 147 eval clips and the 200 pilot clips (the PI's audited frames are in them). The rest come from two buckets of train clips, half each:
 RARE    the pilot-trained probe (C = 5, unweighted) gives a clip-mean P(not clear) of at least 0.30, or P(not dry) of at least 0.40, or the Alpamayo text mentions snow, fog, rain, wet or snow/ice.
 COMMON  everything else, sampled per country in proportion to the country's share with at least 2 per country.
Each train clip's probe weight is 1 / (selected fraction of its bucket), so oversampling rare clips does not move the probe's class prior.
Output: splits/qwen_clips.json {"clips": [...], "weights": {clip: w}, "report": {...}}.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_context as M
import merge_full as F

TARGET, SEED, OUT = 1000, 20261108, "F:/strat_out"


def main():
    from sklearn.linear_model import LogisticRegression
    tab = pd.read_parquet(HERE + "/splits/clips_table.parquet")
    pilot = json.load(open(HERE + "/splits/pilot_clips.json"))["clips"]
    data = {c: M.load_parts("F:/strat_out/pilot", c) for c in pilot}
    train_ids = tab[tab.split == "train4512"].clip_id.tolist()
    fit = {}
    for f, vk in (("weather", "v_weather"), ("surface", "v_surface")):
        Xs, ys = [], []
        for c in train_ids:
            if c in data and data[c] is not None:
                a, b = data[c]
                Fe = M.feats(a)[::5][: len(b[vk])]
                Xs.append(Fe)
                ys.append(b[vk].argmax(1)[: len(Fe)])
        fit[f] = LogisticRegression(max_iter=1000, C=F.PROBE_C).fit(np.concatenate(Xs), np.concatenate(ys))
    txt = pd.read_parquet(HERE + "/alpamayo_text/alpamayo_weak.parquet")
    rare_cols = [c for c in txt.columns if any(w in c for w in ("snow", "fog", "rain", "wet"))]
    mined = set()
    for c in rare_cols:
        if txt[c].dtype == bool:
            mined |= set(txt.loc[txt[c], "clip_id"])
    rows = []
    for c in train_ids:
        p = f"{OUT}/ctxA/{c}.npz"
        A = dict(np.load(p))
        fe = M.feats(A)
        pw = fit["weather"].predict_proba(fe).mean(0)
        ps = fit["surface"].predict_proba(fe).mean(0)
        rows.append((c, 1 - float(pw[list(fit["weather"].classes_).index(0)]), 1 - float(ps[list(fit["surface"].classes_).index(0)])))
    d = pd.DataFrame(rows, columns=["clip_id", "p_not_clear", "p_not_dry"]).merge(tab[["clip_id", "country"]], on="clip_id")
    d["rare"] = (d.p_not_clear >= 0.30) | (d.p_not_dry >= 0.40) | d.clip_id.isin(mined)
    fixed = set(tab[tab.split == "eval"].clip_id) | set(pilot)
    n_more = TARGET - len(fixed)
    pool = d[~d.clip_id.isin(fixed)]
    rng = np.random.default_rng(SEED)
    rare_pool = pool[pool.rare].clip_id.to_numpy()
    n_rare = min(n_more // 2, len(rare_pool))
    picked = list(rng.choice(np.sort(rare_pool), n_rare, replace=False))
    n_common = n_more - n_rare
    common = pool[~pool.rare]
    share = common.groupby("country").size() / len(common)
    alloc = (share * n_common).round().astype(int).clip(lower=2)
    while alloc.sum() > n_common:
        alloc[alloc.idxmax()] -= 1
    while alloc.sum() < n_common:
        alloc[alloc.idxmax()] += 1
    for country, k in alloc.items():
        ids = np.sort(common[common.country == country].clip_id.to_numpy())
        picked += list(rng.choice(ids, min(int(k), len(ids)), replace=False))
    chosen = sorted(fixed | {str(x) for x in picked})
    sel = d.clip_id.isin(chosen)
    frac = {True: float(sel[d.rare].mean()), False: float(sel[~d.rare].mean())}
    weights = {c: round(1.0 / frac[bool(r)], 3) for c, r in zip(d.clip_id[sel], d.rare[sel])}
    rep = {"n_clips": len(chosen), "n_eval": int((tab.split == "eval").sum()), "n_pilot": len(pilot), "train_rare": int(d.rare.sum()), "train_total": len(d), "alpamayo_mined_train": int(d.clip_id.isin(mined).sum()),
           "selected_rare": int(sel[d.rare].sum()), "selected_common": int(sel[~d.rare].sum()), "selected_fraction": {"rare": round(frac[True], 4), "common": round(frac[False], 4)},
           "countries_selected": int(d[sel].country.nunique()), "rare_cols": rare_cols, "seed": SEED, "evidence_class": "MEASURED (this script)"}
    json.dump({"clips": chosen, "weights": weights, "report": rep}, open(HERE + "/splits/qwen_clips.json", "w"), indent=0)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
