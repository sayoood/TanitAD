#!/usr/bin/env python3
"""POWER for the fresh-token confirmation of H-SEL-1's frozen rule gBG(0.20, 0.05). CPU only, reads banked C3 + YSEL data only.

Nothing here scores a fresh token. Inputs: the C3 table (1,058 tokens, TRUE per-proposal PDMS), the ego probe's raw proposals and ego,
YSEL's split.json, the W3 navtest export (token -> log) and the banked u400 full-navtest csv (which tokens are zeroed).

Why the variance model is built on the RANDOM stratum only: every zeroed u400 token (529 of 529, banked PDMS = 0) is already in C3,
so a fresh token is always non-zeroed. The confirmation can only measure the random-stratum effect.
Model (one-way random effects, method of moments) for the per-token paired delta d_i = PDMS(rule pick) - PDMS(banked pick):
    d_ij = mu + b_j + e_ij,  Var(b) = s2b (between logs), Var(e) = s2w (within log).
For a sample with n_j tokens in log j (N = sum n_j), the ratio estimator's variance is (sum n_j^2 s2b + N s2w) / N^2.
Check of the model: the same formula on P2's held-out allocation must reproduce P2's own bootstrap half-width.
Writes power.json.
"""
import gzip
import json
import os
import sys

import numpy as np
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
YSEL = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
sys.path.insert(0, YSEL)
sys.path.insert(0, HERE)
import sel_common as sc  # noqa: E402
import selc_rule as R  # noqa: E402

BANKED_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"
EXP = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
W_R = 11617 / 12146
BAR = -0.5            # x100, on the mix
Z = norm.ppf(0.975)


def var_components(d, log):
    ul, li = np.unique(log, return_inverse=True)
    n = np.bincount(li).astype(float)
    m = np.bincount(li, weights=d) / n
    N, k = len(d), len(ul)
    grand = d.mean()
    ssw = ((d - m[li]) ** 2).sum()
    ssb = (n * (m - grand) ** 2).sum()
    msw = ssw / max(N - k, 1)
    msb = ssb / max(k - 1, 1)
    n0 = (N - (n ** 2).sum() / N) / max(k - 1, 1)
    s2b = max((msb - msw) / n0, 0.0)
    return {"N": int(N), "logs": int(k), "mean": float(grand), "s2w": float(msw), "s2b": float(s2b),
            "s2b_raw": float((msb - msw) / n0), "n0": float(n0)}


def se_alloc(nj, s2b, s2w):
    nj = np.asarray(nj, float)
    N = nj.sum()
    return float(np.sqrt(((nj ** 2).sum() * s2b + N * s2w) / N ** 2))


def boot_halfwidth(d, log, B=10000, seed=20260927):
    ul, li = np.unique(log, return_inverse=True)
    s = np.bincount(li, weights=d)
    c = np.bincount(li).astype(float)
    dr = np.random.default_rng(seed).choice(len(ul), (B, len(ul)), replace=True)
    r = s[dr].sum(1) / c[dr].sum(1)
    return float((np.percentile(r, 97.5) - np.percentile(r, 2.5)) / 2), float(r.std())


def power(mu, se, bar):
    """P(point - Z*se > bar) when the point ~ N(mu, se^2)."""
    return float(norm.cdf((mu - bar) / se - Z))


def main():
    D = sc.load()
    N = len(D["token"])
    r = np.arange(N)
    pick_b = D["pick"]
    pick_r = R.rule_vec(D["logits"], D["v4_all"], D["v0"], D["ax0"])
    split = json.load(open(YSEL + "/split.json"))
    tr = np.isin(D["log"], split["train_logs"])
    ho = np.isin(D["log"], split["heldout_logs"])
    rnd = D["stratum"] == "random"
    d = 100.0 * (D["pdms"][r, pick_r] - D["pdms"][r, pick_b])        # x100
    out = {"what": __doc__.splitlines()[0], "W_R": W_R, "bar_mix_x100": BAR, "bar_on_random_stratum_x100": BAR / W_R}
    vc = {}
    for name, m in (("random_heldout", rnd & ho), ("random_train", rnd & tr), ("random_all", rnd)):
        vc[name] = var_components(d[m], D["log"][m])
        hw, sd = boot_halfwidth(d[m], D["log"][m])
        _, nj = np.unique(D["log"][m], return_counts=True)
        vc[name]["boot_halfwidth_x100"] = hw
        vc[name]["boot_sd_x100"] = sd
        vc[name]["model_se_same_alloc_x100"] = se_alloc(nj, vc[name]["s2b"], vc[name]["s2w"])
        vc[name]["share_changed"] = float((pick_r[m] != pick_b[m]).mean())
        vc[name]["share_changed_better"] = float(((d[m] > 1e-9)).mean())
        vc[name]["share_changed_worse"] = float(((d[m] < -1e-9)).mean())
        vc[name]["d_min_max_x100"] = [float(d[m].min()), float(d[m].max())]
    out["variance_components"] = vc
    # ---- CRUISE slow-down on NON-ZEROED tokens (the only kind a fresh token can be)
    v4b = D["v4_all"][r, pick_b] / np.maximum(D["v0"], 1e-6)
    v4r = D["v4_all"][r, pick_r] / np.maximum(D["v0"], 1e-6)
    slow_b, slow_r = (v4b < 0.85).astype(float), (v4r < 0.85).astype(float)
    cr = {}
    for name, m in (("random_heldout", rnd & ho), ("random_train", rnd & tr), ("random_all", rnd), ("zero_all", ~rnd), ("all_heldout", ho)):
        mm = m & D["cruise"]
        ds = slow_r[mm] - slow_b[mm]
        vcs = var_components(ds, D["log"][mm]) if mm.sum() > 2 else None
        hw = boot_halfwidth(ds, D["log"][mm])[0] if mm.sum() > 2 else None
        cr[name] = {"n_cruise": int(mm.sum()), "n_logs": int(len(np.unique(D["log"][mm]))), "share_cruise_of_stratum": float(mm.sum() / max(m.sum(), 1)),
                    "slow_banked": float(slow_b[mm].mean()) if mm.any() else None, "slow_rule": float(slow_r[mm].mean()) if mm.any() else None,
                    "d_slow": float(ds.mean()) if mm.any() else None, "boot_halfwidth": hw, "vc": vcs}
    out["cruise_slow_nonzeroed"] = cr
    # ---- the fresh pool: navtest non-zeroed tokens NOT in C3, by log class
    exp = json.load(gzip.open(EXP, "rt", encoding="utf-8"))["tokens"]
    c3 = set(D["token"].tolist())
    import csv
    rows = {x["token"]: x for x in csv.DictReader(open(BANKED_CSV, encoding="utf-8")) if x["token"] != "average"}
    zero = {t for t, x in rows.items() if float(x["no_at_fault_collisions"]) * float(x["drivable_area_compliance"]) == 0.0}
    trl, hol = set(split["train_logs"]), set(split["heldout_logs"])
    pool = {"heldout": [], "new": [], "train": []}
    for t in sorted(exp):
        if t in c3 or t in zero:
            continue
        lg = exp[t]["log_name"]
        pool["train" if lg in trl else ("heldout" if lg in hol else "new")].append((t, lg))
    out["fresh_pool"] = {k: {"tokens": len(v), "logs": len({lg for _, lg in v})} for k, v in pool.items()}
    # ---- designs: a uniform draw of N tokens from the NON-TRAIN pool (heldout + new logs); expected allocation n_j = N * pool_j / pool
    nt = pool["heldout"] + pool["new"]
    _, cnt = np.unique([lg for _, lg in nt], return_counts=True)
    designs = {}
    for Nd in (600, 1000, 1500, 2000, 2500, 3000, 4000, 5879):
        nj = cnt * Nd / cnt.sum()
        row = {}
        for vname in ("random_heldout", "random_all"):
            s2b, s2w = vc[vname]["s2b"], vc[vname]["s2w"]
            se = se_alloc(nj, s2b, s2w) * W_R                      # on the mix scale (zeroed term at its structural floor 0)
            row[vname] = {"se_mix_x100": se, "halfwidth_mix_x100": Z * se,
                          "power_true_mix_+0.306(=W_R*heldout_random_+0.32)": power(W_R * vc["random_heldout"]["mean"], se, BAR),
                          "power_true_mix_pooled_random": power(W_R * vc["random_all"]["mean"], se, BAR),
                          "power_true_0": power(0.0, se, BAR),
                          "power_true_+0.47": power(0.47, se, BAR)}
        designs[str(Nd)] = row
    out["designs_nontrain_uniform"] = designs
    json.dump(out, open(os.path.join(HERE, "power.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
