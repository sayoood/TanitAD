"""WP-D PROBE-0b -- the CORRECTED same-box ceiling for a quality-aware presence score (POST-HOC, disclosed).

Why this file exists (disclosed in RESULT sec. 2.4): PROBE-0's Q4 oracle multiplied the score of every slot that has
a GT within 2 m by q = exp(-d^2 / 2 s^2) and left every other slot untouched. That DEMOTES true positives against far
false positives (which keep their score) -- a design error of the oracle, not a property of the head: it read
AP@2 m 0.105 vs T0 0.248. It is kept in raw/box_probe0.json as run, and replaced here.

The corrected oracle answers exactly the question a quality-aware target can answer: "if, among the slots that sit on
the same object, the HIGHEST score went to the NEAREST slot, what would AP be?" For each window, every slot within
2 m of a GT (POS or IGNORE) is assigned to its nearest GT; within each GT's cluster the cluster's own scores are
PERMUTED so that they are sorted by distance (nearest slot <- highest score). The multiset of scores, every score
outside the clusters, and therefore every cluster's rank against the rest of the list are unchanged. It uses GT, so
it is a CEILING (inadmissible as a result).

Controls: (C5) the permutation preserves each window's sorted score vector exactly; (C6) a "permutation" that keeps
the original order reproduces T0 AP exactly. Same estimator as PROBE-0 (paired episode bootstrap, B = 300 here).
Run: PYTHONPATH=<tip>/stack python box_probe0b.py --packs <dir>
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import box_probe0 as P0  # noqa: E402

PKG = Path(__file__).resolve().parents[1]


def perm_scores(pk, radius=2.0, identity=False):
    p = P0.sig(pk["logit"]).copy()
    gi = np.nonzero(pk["pos"] | pk["ign"])[0]
    if gi.size == 0:
        return p
    xy = pk["xy"].astype(np.float64)
    d = np.sqrt(((xy[:, None, :] - pk["gt_xy"][gi][None].astype(np.float64)) ** 2).sum(-1))   # [N, A]
    near = d.min(1) <= radius
    owner = d.argmin(1)
    out = p.copy()
    for a in range(gi.size):
        idx = np.nonzero(near & (owner == a))[0]
        if idx.size < 2:
            continue
        scores = np.sort(p[idx])[::-1]                    # highest first
        if identity:
            continue
        by_dist = idx[np.argsort(d[idx, a], kind="stable")]  # nearest first
        out[by_dist] = scores
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    t0 = time.time()
    with open(Path(a.packs) / "eval_final.packs.pkl", "rb") as fh:
        packs = pickle.load(fh)
    eps = sorted({pk["sha12"] for pk in packs["box3d"]})
    eix = {e: i for i, e in enumerate(eps)}
    ep_of_win = [eix[pk["sha12"]] for pk in packs["box3d"]]
    W = P0.boot_weights(len(eps), a.boot, seed=1)
    out = {"definition": __doc__.split("The corrected oracle")[1].split("Controls")[0].strip(),
           "evidence": "MEASURED packs; ORACLE rows are CEILINGS (use GT)", "n_boot": a.boot, "controls": {}, "arms": {}}
    for hd in ("box3d", "agent"):
        P = packs[hd]
        # C5 / C6
        for pk in P[:200]:
            s = perm_scores(pk)
            assert np.array_equal(np.sort(s), np.sort(P0.sig(pk["logit"]))), "C5 FAIL"
        npos = [P0.npos_band(pk, None) for pk in P]
        for thr in (0.5, 2.0):
            t0a = P0.APTable([P0.greedy(pk, thr) for pk in P], npos, ep_of_win).ap()
            t0b = P0.APTable([P0.greedy(pk, thr, score=perm_scores(pk, identity=True)) for pk in P], npos,
                             ep_of_win).ap()
            assert abs(t0a - t0b) < 1e-12, "C6 FAIL"
        out["controls"][hd] = {"C5_score_multiset_preserved": True, "C6_identity_permutation_is_T0": True}
        res = {}
        base = {}
        arms = {"T0": {}, "ORACLE_perm": {"score": True}, "nms_r1": {"nms": 1.0}, "nms_r2": {"nms": 2.0},
                "ORACLE_perm+nms_r1": {"score": True, "nms": 1.0}, "ORACLE_perm+nms_r2": {"score": True, "nms": 2.0}}
        for name, spec in arms.items():
            res[name] = {}
            for thr in P0.THRS:
                rows = []
                for pk in P:
                    sc = perm_scores(pk) if spec.get("score") else None
                    if spec.get("nms"):
                        # NMS on the (possibly permuted) score order
                        p = P0.sig(pk["logit"]) if sc is None else sc
                        order = np.argsort(-p, kind="stable")
                        xy = pk["xy"].astype(np.float64)
                        keep = np.zeros(len(p), bool)
                        kept = []
                        for s_ in order:
                            if kept and (np.sqrt(((xy[kept] - xy[s_]) ** 2).sum(-1)) < spec["nms"]).any():
                                continue
                            kept.append(s_)
                            keep[s_] = True
                    else:
                        keep = None
                    rows.append(P0.greedy(pk, thr, score=sc, keep=keep))
                tab = P0.APTable(rows, npos, ep_of_win)
                if name == "T0":
                    base[thr] = tab
                    res[name][f"ap{thr:g}"] = tab.ap()
                else:
                    dd = [tab.ap(w) - base[thr].ap(w) for w in W]
                    res[name][f"ap{thr:g}"] = tab.ap()
                    res[name][f"d_vs_T0_ap{thr:g}"] = {"point": tab.ap() - base[thr].ap(), "ci95": P0.ci(dd)}
        out["arms"][hd] = res
        print(hd, {k: {kk: (round(v, 4) if isinstance(v, float) else round(v["point"], 4)) for kk, v in r.items()}
                   for k, r in res.items()}, flush=True)
    out["wall_s"] = time.time() - t0
    p = PKG / "raw" / "box_probe0b.json"
    p.write_text(json.dumps(out, indent=1, allow_nan=True), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
