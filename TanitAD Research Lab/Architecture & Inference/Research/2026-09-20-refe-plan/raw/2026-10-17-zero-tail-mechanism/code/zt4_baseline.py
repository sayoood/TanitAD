"""zt4_baseline.py -- POST-HOC (not in PREREG_ZT3): the CONTROL-ARM baseline of the next experiment proposed in REPORT.md
(ZT4, the near-twin DAC readout), plus one descriptive cross-check of the TRACKER_MADE share. No decision rests on it.

    python zt4_baseline.py --inroot <scratch in/> --out ../ZT4_BASELINE.json

near-twin pair (i, j) of one token: max over the 8 PLANNED poses of |xy_i - xy_j| <= 0.5 m AND NAVSIM DAC_i != DAC_j.
pairwise DAC accuracy = share of pairs with p_DAC(DAC=1 member) > p_DAC(DAC=0 member), ties 0.5.
Estimator: ratio of per-token sums, log-cluster bootstrap (zero_tail.Boot, 136 logs, 10,000, seed 20260927).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zero_tail as zt  # noqa: E402
import zt3_lib as L  # noqa: E402

TWIN_M = 0.5


def twin_pairs(props, dac, pdac, twin_m=TWIN_M):
    """-> (n_pairs, n_correct) for one token. props [M,8,3], dac [M] in {0,1}, pdac [M]."""
    xy = props[:, :, :2].astype(np.float64)
    d = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1).max(-1)          # [M, M]
    safe, bad = dac == 1, dac == 0
    m = (d <= twin_m) & safe[:, None] & bad[None, :]
    if not m.any():
        return 0, 0.0
    i, j = np.nonzero(m)
    corr = (pdac[i] > pdac[j]).sum() + 0.5 * (pdac[i] == pdac[j]).sum()
    return int(m.sum()), float(corr)


def selftest():
    props = np.zeros((3, 8, 3))
    props[1, :, 1] = 0.3                     # twin of 0
    props[2, :, 1] = 5.0                     # far
    dac = np.array([1, 0, 0])
    n, c = twin_pairs(props, dac, np.array([0.9, 0.2, 0.1]))
    ok = (n == 1 and c == 1.0)
    n2, c2 = twin_pairs(props, 1 - dac, np.array([0.9, 0.2, 0.1]))   # mutation: labels swapped -> pair (1,0) wrong way
    ok &= (n2 == 1 and c2 == 0.0)
    n3, _ = twin_pairs(props, dac, np.array([0.9, 0.2, 0.1]), twin_m=0.1)   # threshold below the 0.3 m offset -> no pair
    ok &= (n3 == 0)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert selftest(), "zt4 twin_pairs self-test / mutation check FAILED"
    R = Path(a.inroot)
    nt = json.load(open(R / "navtest_full_tokens.json", encoding="utf-8"))
    NT = list(nt["tokens"])
    nidx = {t: i for i, t in enumerate(NT)}
    logs = sorted(set(nt["token_log"].values()))
    lidx = np.array([logs.index(nt["token_log"][t]) for t in NT])
    boot = zt.Boot(lidx, len(logs))
    c3 = json.load(open(R / "c3/tokens.json", encoding="utf-8"))
    out = {"evidence": "MEASURED (ours); POST-HOC, not pre-registered in ZT3; baseline/control arm for the proposed ZT4",
           "definition": __doc__.strip().splitlines()[3:5], "selftest_and_mutation": True}
    tgt = {"A": {json.loads(l)["token"] for l in open(HERE.parent / "data/per_token_A.jsonl", encoding="utf-8")},
           "B": {json.loads(l)["token"] for l in open(HERE.parent / "data/per_token_B.jsonl", encoding="utf-8")}}
    for X, path in (("A_u400", R / "c3/table.npz"), ("B_P2", R / "p2/p2_own_table/table.npz")):
        T = np.load(path)
        toks = [str(t) for t in T["token"]]
        pdac = L.sigmoid(T["logits"][:, :, 1].astype(np.float64))
        dac = T["sub"][:, :, 1]
        npair, ncorr = np.zeros(len(NT)), np.zeros(len(NT))
        grp = {"target_DAC_SEL": np.zeros(len(NT), bool), "random_stratum": np.zeros(len(NT), bool), "all_rows": np.zeros(len(NT), bool)}
        for i, t in enumerate(toks):
            n, c = twin_pairs(T["proposals"][i], dac[i], pdac[i])
            k = nidx[t]
            npair[k], ncorr[k] = n, c
            grp["all_rows"][k] = True
            grp["target_DAC_SEL"][k] = t in tgt[X[0]]
            if X.startswith("A") and c3["stratum"][t] == "random":
                grp["random_stratum"][k] = True
        res = {}
        for gname, gm in grp.items():
            if not gm.any():
                continue
            r = boot.ratio(ncorr * gm, npair * gm)
            res[gname] = {"tokens": int(gm.sum()), "tokens_with_twin_pair": int(((npair > 0) & gm).sum()),
                          "pairs": int(npair[gm].sum()), "pairwise_DAC_accuracy": {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}
        out[X] = res
    # descriptive cross-check: tracker-made share among ALL DAC=0 proposals of the T_A fans (tracked dump) vs at the pick
    A = np.load(R / "c3/table.npz")
    tok = [str(t) for t in A["token"]]
    num = den = num_p = den_p = 0
    for p in sorted(Path("C:/Users/Admin/tmpauxv/tracked").glob("tracked_*of4.npz")):
        z = np.load(p)
        for q, ix in enumerate(z["idx"]):
            t = tok[int(ix)]
            if t not in tgt["A"]:
                continue
            d0, d1 = z["d0_sim_poly"][q], z["d1_raw_poly"][q]
            bad = d0 == 0
            num += int((bad & (d1 == 1)).sum())
            den += int(bad.sum())
            pk = int(A["pick"][int(ix)])
            num_p += int(d1[pk] == 1)
            den_p += 1
    out["tracker_made_share_among_all_DAC0_proposals_in_T_A_fans"] = {"n": num, "of": den, "pct": round(100 * num / den, 2)}
    out["tracker_made_share_at_the_pick_T_A"] = {"n": num_p, "of": den_p, "pct": round(100 * num_p / den_p, 2)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
