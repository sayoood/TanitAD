"""e0a_gap.py -- E0a (REFCV9_DESIGN.md s4.1 / s2.5): the over-fit gap of refcv8's box head -- AP@2 m on TRAIN vs HELD-OUT windows, equal window counts.

Reuses pbn_q3_box.py's AP definition (`detection_metrics.greedy_rows(pk, 2.0)` + `_ap`, class-agnostic 2 m greedy, all-point AP) on the banked PERC8 packs of refcv8 EMA-48,852 (ckpt md5 b4364029...):
  TRAIN  = `perc8/out/T_nosync/train_diag.packs.pkl` (dict {'box3d': [pack ...]}; 139 TRAIN episodes x 8 windows, 1,074 packs),
  HELD   = `perc8/out/A_nosync/perc_det.pkl`          (list of records {sha12, ws, box3d: [pack]}; eval139, 4,532 packs).
No GPU: the packs are the banked output of the Thor forward passes (they were produced on Thor; this probe is the CPU analysis).

    PYTHONPATH=<tree>/stack python -X utf8 e0a_gap.py --train train_diag.packs.pkl --heldout perc_det.pkl --out E0A_RESULT.json

Frozen readout (E0a_CARD.md): equal counts = the HELD set is cut to exactly len(TRAIN) packs (per-episode 8 evenly spaced windows by `ws`, then a seeded drop); gap = AP_train - AP_heldout;
95 % interval = TWO-SAMPLE episode-cluster bootstrap (each group's episodes resampled independently), B 1000, seed 0.  A1 (design s2.5 augmentation) RUNS iff the gap's lower bound > 0.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

BANDS = ("0_20", "20_40", "40_60")


def window_rows(pk):
    from tanitad.eval import detection_metrics as dm
    rr = dm.greedy_rows(pk, 2.0)
    pos = pk["pos"]
    ng = {"all": int(pos.sum())}
    for b in BANDS:
        ng[b] = int(sum(1 for g in np.nonzero(pos)[0] if dm._band_of(float(pk["gt_xy"][g, 0])) == b))
    conf = np.array([r[0] for r in rr if r[1] >= 0], np.float64)
    kind = np.array([r[1] for r in rr if r[1] >= 0], np.int8)
    band = np.array([BANDS.index(r[2]) if r[2] in BANDS else -1 for r in rr if r[1] >= 0], np.int8)
    return conf, kind, band, ng


def ap_cells(wins, idx, key):
    """all-point AP over the windows `idx`; key 'all' or a band name (rows of that band, GT count of that band) -- the same arithmetic as box3d_head.ap_from_rows."""
    cf = np.concatenate([wins[i][0] for i in idx]) if len(idx) else np.zeros(0)
    kd = np.concatenate([wins[i][1] for i in idx]) if len(idx) else np.zeros(0, np.int8)
    if key == "all":
        n = sum(wins[i][3]["all"] for i in idx)
    else:
        m = np.concatenate([wins[i][2] for i in idx]) == BANDS.index(key)
        cf, kd, n = cf[m], kd[m], sum(wins[i][3][key] for i in idx)
    if n == 0:
        return float("nan")
    o = np.argsort(-cf, kind="stable")
    k = kd[o].astype(np.float64)
    tp, fp = np.cumsum(k), np.cumsum(1 - k)
    rec, prec = tp / n, tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def equal_count_subset(ws_by_ep, n_target, per_ep, seed):
    """HELD windows -> exactly n_target indices: per episode `per_ep` evenly spaced windows (by ws order), then a seeded drop to n_target."""
    pick = []
    for e, lst in sorted(ws_by_ep.items()):
        lst = sorted(lst)                                                  # (ws, index)
        pos = np.unique(np.round(np.linspace(0, len(lst) - 1, min(per_ep, len(lst)))).astype(int))
        pick += [lst[i][1] for i in pos]
    rng = np.random.default_rng(seed)
    if len(pick) < n_target:
        raise ValueError("held-out pool smaller than the target count")
    return sorted(rng.permutation(pick)[:n_target].tolist())


def boot_gap(wa, ia, ea, wb, ib, eb, key, B, seed):
    """two-sample episode-cluster bootstrap of AP_a - AP_b."""
    rng = np.random.default_rng(seed)
    ca, cb = defaultdict(list), defaultdict(list)
    for i, e in zip(ia, ea):
        ca[e].append(i)
    for i, e in zip(ib, eb):
        cb[e].append(i)
    ka, kb = sorted(ca), sorted(cb)
    d = []
    for _ in range(B):
        sa = [i for e in rng.choice(ka, len(ka)) for i in ca[e]]
        sb = [i for e in rng.choice(kb, len(kb)) for i in cb[e]]
        d.append(ap_cells(wa, sa, key) - ap_cells(wb, sb, key))
    d = np.array(d)
    d = d[np.isfinite(d)]
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), int(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--n-seeds", type=int, default=20)
    a = ap.parse_args()
    tr = pickle.load(open(a.train, "rb"))["box3d"]
    ho = pickle.load(open(a.heldout, "rb"))
    ho_packs = [(r["sha12"], int(r["ws"]), pk) for r in ho for pk in (r.get("box3d") or [])]
    eps = sorted({s for s, _, _ in ho_packs})
    ho_ep = [eps.index(s) for s, _, _ in ho_packs]
    W_tr = [window_rows(pk) for pk in tr]
    W_ho = [window_rows(pk) for pk in (p for _, _, p in ho_packs)]
    tr_ep = [int(pk["ep"]) for pk in tr]
    n_tr = len(tr)
    out = {"tool": "e0a_gap.py", "n_train_packs": n_tr, "n_train_episodes": len(set(tr_ep)), "n_heldout_packs_all": len(ho_packs), "n_heldout_episodes": len(eps)}
    # controls
    out["K_REDERIVE_heldout_all_ap2"] = {"read": ap_cells(W_ho, range(len(W_ho)), "all"), "banked_PERC8_Q3": 0.328}
    out["K_REDERIVE_heldout_all_ap2"]["pass"] = abs(out["K_REDERIVE_heldout_all_ap2"]["read"] - 0.328) < 0.0015
    by = defaultdict(list)
    for i, ((_, ws, _), e) in enumerate(zip(ho_packs, ho_ep)):
        by[e].append((ws, i))
    res = {}
    seeds = list(range(a.n_seeds))
    for key in ("all",) + BANDS:
        pts = []
        for sd in seeds:
            sub = equal_count_subset(by, n_tr, 8, sd)
            pts.append(ap_cells(W_tr, range(n_tr), key) - ap_cells(W_ho, sub, key))
        sub0 = equal_count_subset(by, n_tr, 8, 0)
        lo, hi, nb = boot_gap(W_tr, range(n_tr), tr_ep, W_ho, sub0, [ho_ep[i] for i in sub0], key, a.draws, 0)
        res[key] = {"ap_train": ap_cells(W_tr, range(n_tr), key), "ap_heldout_equal_count_draw0": ap_cells(W_ho, sub0, key), "gap_draw0": res_gap(W_tr, W_ho, n_tr, sub0, key),
                    "gap_ci95": [lo, hi], "n_boot_finite": nb, "gap_over_20_draws": {"min": float(np.nanmin(pts)), "median": float(np.nanmedian(pts)), "max": float(np.nanmax(pts))},
                    "n_pos_train": sum(w[3][key] for w in W_tr), "n_pos_heldout_draw0": sum(W_ho[i][3][key] for i in sub0)}
    out["equal_count"] = res
    out["A1_runs"] = bool(res["all"]["gap_ci95"][0] > 0)
    # null control: two disjoint random halves of the HELD episodes, same code, same n -> the gap interval must include 0 (checked over 20 splits; <= 10 % may exclude by chance)
    rng = np.random.default_rng(123)
    excl = 0
    for s in range(20):
        perm = rng.permutation(len(eps))
        ha, hb = set(perm[: len(eps) // 2].tolist()), set(perm[len(eps) // 2:].tolist())
        ia = [i for i, e in enumerate(ho_ep) if e in ha]
        ib = [i for i, e in enumerate(ho_ep) if e in hb]
        lo, hi, _ = boot_gap(W_ho, ia, [ho_ep[i] for i in ia], W_ho, ib, [ho_ep[i] for i in ib], "all", 300, s)
        excl += int(lo > 0 or hi < 0)
    out["K_NULL_heldout_halves_gap_ci_excludes_0"] = {"n_splits": 20, "n_excluding": excl, "pass": excl <= 2}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("K_REDERIVE_heldout_all_ap2", "K_NULL_heldout_halves_gap_ci_excludes_0", "A1_runs")}))
    for k, v in res.items():
        print(k, round(v["ap_train"], 4), round(v["ap_heldout_equal_count_draw0"], 4), round(v["gap_draw0"], 4), [round(x, 4) for x in v["gap_ci95"]])


def res_gap(W_tr, W_ho, n_tr, sub, key):
    return ap_cells(W_tr, range(n_tr), key) - ap_cells(W_ho, sub, key)


if __name__ == "__main__":
    main()
