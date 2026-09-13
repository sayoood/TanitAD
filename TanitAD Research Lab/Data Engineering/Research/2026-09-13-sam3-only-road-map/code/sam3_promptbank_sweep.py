"""Offline sweep over the SAM3 prompt bank on the nuPlan demo frames that carry the TRUE map.

True map classes: 2 road_line, 4 crosswalk. Scored inside the observed ground only, 0.45 m tolerance (3 cells at
0.15 m), pooled over frames:
  precision = predicted cells within 0.45 m of a true cell of the class; recall = true cells within 0.45 m of a
  prediction. Families are swept as prompt subset x score threshold x on-road rule; a COMPETITION arm assigns each
  cell to the family of the highest-scoring instance covering it. Controls: the best arm mirrored left-right.
Hatched and symbol prompts have no class in the nuPlan map; for them we report WHERE they land (true line /
true crosswalk / neither), which is exactly the confusion the PI saw.
Output: /home/nvidia/sam3map/promptbank/sweep_demo.json
"""
import itertools, json
from pathlib import Path
import numpy as np
from scipy import ndimage

B = Path("/home/nvidia/sam3map/promptbank")
TOL = 3
FAM_CLASS = {"line": 2, "crosswalk": 4}


def load():
    frames = []
    for f in sorted((B / "demo").glob("*.npz")):
        d = np.load(f)
        meta = json.loads(str(d["meta"]))
        R = np.unpackbits(d["rasters"], axis=1)[:, :80000].reshape(-1, 200, 400).astype(bool)
        frames.append((f.stem, meta, R, d["gt"].astype(int), d["observed"].astype(bool)))
    return frames


def union(meta, R, prompts, thr, onroad_rule):
    m = np.zeros((200, 400), bool)
    for row, r in zip(meta, R):
        if row["prompt"] in prompts and row["score"] >= thr and (not onroad_rule or (row["onroad"] >= 0.6 and row["ring_road"] >= 0.35)):
            m |= r
    return m


def pr_counts(pred, true, obs):
    near_t = ndimage.binary_dilation(true, iterations=TOL); near_p = ndimage.binary_dilation(pred, iterations=TOL)
    pp, tp = pred & obs, true & obs
    return np.array([(pp & near_t).sum(), pp.sum(), (tp & near_p).sum(), tp.sum()], np.int64)


def summarise(c):
    p = c[0] / max(c[1], 1); r = c[2] / max(c[3], 1)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(2 * p * r / max(p + r, 1e-9)), 4),
            "pred_cells": int(c[1]), "true_cells": int(c[3])}


def main():
    frames = load()
    out = {"frames": len(frames), "true_cells": {}}
    for fam, cls in FAM_CLASS.items():
        out["true_cells"][fam] = int(sum(((gt == cls) & obs).sum() for _, _, _, gt, obs in frames))
    bank = {"line": ["lane marking", "road marking", "stop line", "dashed line", "solid line", "white line on road", "yellow line on road"],
            "crosswalk": ["crosswalk", "zebra crossing", "pedestrian crossing", "crosswalk stripes"]}
    sweeps = {}
    for fam, prompts in bank.items():
        cls = FAM_CLASS[fam]
        subsets = [(p,) for p in prompts] + [tuple(prompts[:2]), tuple(prompts)]
        rows = []
        for sub, thr, rule in itertools.product(subsets, (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (False, True)):
            c = np.zeros(4, np.int64); wrong = 0
            for _, meta, R, gt, obs in frames:
                pred = union(meta, R, set(sub), thr, rule)
                if fam == "crosswalk":
                    pred = ndimage.binary_closing(pred, iterations=2)
                c += pr_counts(pred, gt == cls, obs)
                other = 4 if cls == 2 else 2
                wrong += int((pred & obs & ndimage.binary_dilation(gt == other, iterations=TOL) & ~ndimage.binary_dilation(gt == cls, iterations=TOL)).sum())
            s = summarise(c); s.update({"prompts": list(sub), "thr": thr, "onroad_rule": rule, "cells_on_the_other_paint_class": wrong})
            rows.append(s)
        rows.sort(key=lambda r: -r["f1"])
        sweeps[fam] = rows
    out["sweeps"] = {fam: rows[:25] for fam, rows in sweeps.items()}
    # ---- honesty checks: 2-fold held-out selection (pick on two frames, score on the other two) + SHUFFLED control
    folds = [((0, 1), (2, 3)), ((2, 3), (0, 1))]
    heldout = {}
    for fam, prompts in bank.items():
        cls = FAM_CLASS[fam]
        subsets = [(p,) for p in prompts] + [tuple(prompts[:2]), tuple(prompts)]
        grid = list(itertools.product(subsets, (0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), (False, True)))
        per_frame = {}
        for gi, (sub, thr, rule) in enumerate(grid):
            for fi, (_, meta, R, gt, obs) in enumerate(frames):
                pred = union(meta, R, set(sub), thr, rule)
                if fam == "crosswalk":
                    pred = ndimage.binary_closing(pred, iterations=2)
                per_frame[(gi, fi)] = pr_counts(pred, gt == cls, obs)
        tot_sel, picks = np.zeros(4, np.int64), []
        for fit, test in folds:
            f1 = [summarise(sum(per_frame[(gi, fi)] for fi in fit))["f1"] for gi in range(len(grid))]
            gi = int(np.argmax(f1)); picks.append({"prompts": list(grid[gi][0]), "thr": grid[gi][1], "rule": grid[gi][2]})
            tot_sel += sum(per_frame[(gi, fi)] for fi in test)
        v1 = next(gi for gi, g in enumerate(grid) if list(g[0]) == prompts[:2] and g[1] == {"line": 0.4, "crosswalk": 0.5}[fam] and g[2])
        tot_v1 = sum(per_frame[(v1, fi)] for fi in range(len(frames)))
        heldout[fam] = {"picked_on_fit_folds": picks, "heldout_score_of_picks": summarise(tot_sel), "v1_rule_all_frames": summarise(tot_v1)}
    out["heldout_2fold"] = heldout
    shuf = {}
    for fam in bank:
        best = sweeps[fam][0]; c = np.zeros(4, np.int64)
        for fi, (_, meta, R, gt, obs) in enumerate(frames):
            _, meta2, R2, _, _ = frames[(fi + 1) % len(frames)]
            pred = union(meta2, R2, set(best["prompts"]), best["thr"], best["onroad_rule"])
            if fam == "crosswalk":
                pred = ndimage.binary_closing(pred, iterations=2)
            c += pr_counts(pred, gt == FAM_CLASS[fam], obs)
        shuf[fam] = summarise(c)
    out["best_arm_SHUFFLED_control"] = shuf
    out["v1_rule"] = {fam: next(r for r in sweeps[fam] if r["prompts"] == bank[fam][:2] and r["thr"] == {"line": 0.4, "crosswalk": 0.5}[fam] and r["onroad_rule"])
                      for fam in bank}
    # ---- where do hatch / symbol prompts land?
    land = {}
    for fam_name, prompts in {"hatch": ["hatched road marking", "chevron road marking", "diagonal stripes on road", "painted traffic island"],
                              "symbol": ["arrow painted on road", "road arrow", "text painted on road", "word on road surface",
                                         "bicycle symbol painted on road", "triangle painted on road"]}.items():
        for p in prompts:
            tot = np.zeros(3, np.int64)
            for _, meta, R, gt, obs in frames:
                pred = union(meta, R, {p}, 0.45, True) & obs
                nl = ndimage.binary_dilation(gt == 2, iterations=TOL); nc = ndimage.binary_dilation(gt == 4, iterations=TOL)
                tot += [int((pred & nc).sum()), int((pred & nl & ~nc).sum()), int((pred & ~nl & ~nc).sum())]
            land[p] = {"on_true_crosswalk": int(tot[0]), "on_true_line": int(tot[1]), "on_neither": int(tot[2])}
    out["hatch_symbol_landing"] = land
    # ---- competition: each cell -> family of the highest-scoring instance (line / crosswalk / hatch / symbol)
    fams = {"line": ["lane marking", "road marking", "stop line"], "crosswalk": ["crosswalk", "zebra crossing", "pedestrian crossing"],
            "hatch": ["hatched road marking", "chevron road marking"], "symbol": ["arrow painted on road", "text painted on road"]}
    thr = {"line": 0.4, "crosswalk": 0.5, "hatch": 0.45, "symbol": 0.45}
    comp = {"line": np.zeros(4, np.int64), "crosswalk": np.zeros(4, np.int64)}
    comp_mirror = {"line": np.zeros(4, np.int64), "crosswalk": np.zeros(4, np.int64)}
    for _, meta, R, gt, obs in frames:
        best = np.zeros((200, 400)); lab = np.full((200, 400), "", object)
        for row, r in zip(meta, R):
            for fam, ps in fams.items():
                if row["prompt"] in ps and row["score"] >= thr[fam] and row["onroad"] >= 0.6 and row["ring_road"] >= 0.35:
                    upd = r & (row["score"] > best)
                    best[upd] = row["score"]; lab[upd] = fam
        for fam, cls in FAM_CLASS.items():
            pred = lab == fam
            if fam == "crosswalk":
                pred = ndimage.binary_closing(pred, iterations=2)
            comp[fam] += pr_counts(pred, gt == cls, obs)
            comp_mirror[fam] += pr_counts(pred[::-1, :], gt == cls, obs)
    out["competition"] = {fam: summarise(c) for fam, c in comp.items()}
    out["competition_MIRRORED_control"] = {fam: summarise(c) for fam, c in comp_mirror.items()}
    (B / "sweep_demo.json").write_text(json.dumps(out, indent=1))
    compact = lambda r: f"P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} thr {r['thr']} rule {r['onroad_rule']} {'+'.join(r['prompts'])}"
    print("true cells", out["true_cells"])
    for fam in bank:
        print(f"== {fam}: v1 {compact(out['v1_rule'][fam])}")
        for r in out["sweeps"][fam][:4]:
            print("   best", compact(r))
        print("   held-out picks", out["heldout_2fold"][fam]["picked_on_fit_folds"], "-> held-out score", out["heldout_2fold"][fam]["heldout_score_of_picks"])
        print("   shuffled control of best", out["best_arm_SHUFFLED_control"][fam])
    print("ZZSWEEP-DONEZZ")


if __name__ == "__main__":
    main()
