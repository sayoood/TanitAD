#!/usr/bin/env python3
"""EXPOSURE DIAGNOSTIC (descriptive; NOT a Step A quantity).  G0 found the fit universe SEEN at the scene level (manifest M-SCENE).  Does that show up in the scorer's IN-FAN ranking?
In-fan top-8 pairwise concordance of the EP and TTC heads (the prereg's own reference numbers 0.887 / 0.722, D-REFE-SELRANK-1) on P2's navtest table (UNSEEN, 12,146 tokens / 136 logs)
versus P2's held-out table (SEEN, 1,705 tokens / 24 logs), same definition, log-cluster bootstrap (10,000, seed 20260927).  It uses banked labels of P2's own fan and none of the perturbed pairs, so it
cannot bias A1-A5.  Definition (reproduces the registered navtest 0.887): the 8 proposals with the highest aggregate, pairs whose TRUE value differs by > 0.05, scorer tie = 0.5.
    python exposure_diagnostic.py --unseen <table_p2_full.npz> --seen <table_p2_heldout.npz> --out <json>"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stepA as A


def topk_concordance(z, comp, k=8, thr=A.THR_TRUE_PAIR):
    lg, sub = z["logits"].astype(np.float64), z["sub"].astype(np.float64)
    valid = z["valid"] if "valid" in z.files else np.ones(lg.shape[:2], bool)
    agg = A.aggregate(lg)
    logs = np.array([str(x) for x in z["log"]])
    names = sorted(set(logs))
    li = {l: i for i, l in enumerate(names)}
    num, den = np.zeros(len(names)), np.zeros(len(names))
    for i in range(lg.shape[0]):
        top = np.argsort(-agg[i], kind="stable")[:k]
        top = top[valid[i][top]]
        t, s = sub[i, top, comp], lg[i, top, comp]
        d, ds = t[:, None] - t[None, :], s[:, None] - s[None, :]
        m = np.triu(np.ones_like(d, bool), 1) & (np.abs(d) > thr)
        c = np.where(ds[m] == 0, 0.5, (np.sign(ds[m]) == np.sign(d[m])).astype(float))
        num[li[logs[i]]] += c.sum()
        den[li[logs[i]]] += m.sum()
    return num, den, len(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--seen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"definition": "top-8 by aggregate (navsim_v1), pairs with |true diff| > 0.05, scorer tie 0.5; log-cluster bootstrap B=10000 seed 20260927 (another draw of LOGS only; P2 one training seed)",
           "registered_reference": {"EP": "0.887 [0.876, 0.899] navtest (D-REFE-SELRANK-1)", "TTC": "0.722 [0.686, 0.759] navtest"}}
    for tag, path in (("unseen_navtest_p2_full", a.unseen), ("seen_heldout_p2", a.seen)):
        z = np.load(path, allow_pickle=True)
        out[tag] = {"path": path, "sha256": A.sha256_file(path), "n_tokens": int(len(z["token"]))}
        for comp, nm in ((2, "EP"), (3, "TTC")):
            num, den, L = topk_concordance(z, comp)
            out[tag][nm] = A.ratio_ci(num, den, A.boot_indices(L, A.N_BOOT, A.SEED))
            out[tag][nm]["n_logs"] = L
    json.dump(out, open(a.out, "w"), indent=1)
    f = lambda t, n: f"{out[t][n]['point']:.4f} [{out[t][n]['lo']:.4f}, {out[t][n]['hi']:.4f}]"
    print("ZZEXPOSURE", {n: (f("unseen_navtest_p2_full", n), f("seen_heldout_p2", n)) for n in ("EP", "TTC")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
