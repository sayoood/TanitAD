"""A-TEST: the sign-admissibility test of a privileged-geometry TARGET block ("could the DAC label be recomputed from this target?"), answered by a SEARCH with PLANTED positive and negative controls,
never by inspection (MM ruling 2026-10-19, after PRIVGEOM-SAFE RESULT 5.2: the 10-column `DAC-nosign` block re-derived the verdict's sign through the pair (|c_min|, |c_t|)).

The recomputer is a RULE SEARCH.  A rule is "predicted class X when <condition on the block's columns>"; the conditions are the families that a sign proof can take:
  unary   C_a < theta | C_a > theta                          theta in THETAS
  pair    C_a + m < C_b                                        every ordered pair (a, b), m in MARGINS     (the family that caught |c_t| < |c_min|)
  V       C_m + m < min(C_a, C_b)  with a < m < b              (a dip between two larger values: a crossing's signature)
  agg     max_k C_k < theta | min_k C_k > theta
For each rule and each class X in {DAC-unsafe, DAC-safe} the PRECISION (P(label = X | rule fires)) and the RECALL (P(rule fires | label = X)) over ALL proposals are computed.
A block is NOT ADMISSIBLE (it re-derives the label's SIGN) iff some rule PROVES DAC-UNSAFE: precision >= PREC_BAR with recall >= RECALL_MIN for the UNSAFE class.
Why one-sided: "every key clearance is large, so the proposal is DAC-safe" is what proximity MEANS (the ego starts inside the drivable area; a path that never comes near the boundary cannot have left it),
so a SAFE-class proof by large |clearance| is the intended information, and it is reported (descriptive).  A proof of UNSAFE from unsigned magnitudes (|c_t| < |c_min|, a dip between larger values, ...)
is a re-derivation of the verdict's sign and is the leak PRIVGEOM-SAFE RESULT 5.2 measured.  (MEASURED while building this test: the within-set-permuted NEGATIVE control still shows a SAFE-class rule at
precision 0.9992, a SET-level base-rate effect - wide-road sets are safe throughout - which is exactly why the two-sided form was dropped and the controls were kept.)

Controls (a test is trusted only if it reads right on both):
  POS-1  the registered 10-column DAC-nosign block [|c_min|, 9 x |c_t|]      MUST be flagged (the rule |c_t| < |c_min|: precision 1.000, recall 88.7 % in PRIVGEOM RESULT 5.2)
  POS-2  the candidate block + one PLANTED column = label * 5 + noise        MUST be flagged (a sign-bearing channel planted by construction)
  NEG    the candidate block with every column permuted WITHIN each set      MUST NOT be flagged
The flexible recomputer (a small MLP, 5-fold log-grouped OOF within-set F-AUC) is DESCRIPTIVE only: it has no authority over admissibility.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402

PREC_BAR, RECALL_MIN = 0.999, 0.005
THETAS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0)
MARGINS = (0.0, 0.25, 0.5, 1.0, 2.0)


def _eval(mask: np.ndarray, unsafe: np.ndarray) -> tuple:
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    tp_u = int((mask & unsafe).sum())
    tp_s = n - tp_u
    n_u, n_s = int(unsafe.sum()), int((~unsafe).sum())
    return tp_u / n, tp_u / max(n_u, 1), tp_s / n, tp_s / max(n_s, 1)       # precision_unsafe, recall_unsafe, precision_safe, recall_safe


def rules(C: np.ndarray):
    """yields (name, boolean mask [M]) for every rule of the families above.  C [M, K] float."""
    M, K = C.shape
    for a in range(K):
        for th in THETAS:
            yield f"C{a} < {th}", C[:, a] < th
            yield f"C{a} > {th}", C[:, a] > th
    for a, b in itertools.permutations(range(K), 2):
        for mg in MARGINS:
            yield f"C{a} + {mg} < C{b}", C[:, a] + mg < C[:, b]
    for a, m, b in itertools.combinations(range(K), 3):
        for mg in MARGINS[:3]:
            yield f"V(C{m} + {mg} < min(C{a}, C{b}))", C[:, m] + mg < np.minimum(C[:, a], C[:, b])
    mx, mn = C.max(1), C.min(1)
    for th in THETAS:
        yield f"max C < {th}", mx < th
        yield f"min C > {th}", mn > th


def search(C: np.ndarray, label_dac: np.ndarray, prec_bar: float = PREC_BAR, recall_min: float = RECALL_MIN, keep: int = 5) -> dict:
    """C [..., K] block (proposals on the leading axes), label_dac [...] (0 = unsafe, 1 = safe, 0.5 excluded) -> {"flagged": bool, "n_rules", "top": [...], "best_by_class": ...}"""
    K = C.shape[-1]
    C2 = np.asarray(C, np.float64).reshape(-1, K)
    lab = np.asarray(label_dac).reshape(-1)
    ok = (lab == 0.0) | (lab == 1.0)
    C2, unsafe = C2[ok], (lab[ok] == 0.0)
    hits, n_rules = [], 0
    best = {"unsafe": (0.0, 0.0, ""), "safe": (0.0, 0.0, "")}
    for name, mask in rules(C2):
        n_rules += 1
        pu, ru, ps, rs = _eval(mask, unsafe)
        for cls, p, r in (("unsafe", pu, ru), ("safe", ps, rs)):
            if r >= recall_min and p > best[cls][0]:
                best[cls] = (p, r, name)
            if p >= prec_bar and r >= recall_min:
                hits.append({"rule": name, "class": cls, "precision": p, "recall": r})
    hits.sort(key=lambda h: -h["recall"])
    unsafe_hits = [h for h in hits if h["class"] == "unsafe"]
    return {"flagged": bool(unsafe_hits), "n_hits_safe_class_descriptive": len(hits) - len(unsafe_hits), "n_rules": n_rules, "n_hits": len(unsafe_hits), "top": unsafe_hits[:keep], "top_safe_class_descriptive": [h for h in hits if h["class"] == "safe"][:keep], "best_precision_at_recall_min": {k: {"precision": v[0], "recall": v[1], "rule": v[2]} for k, v in best.items()},
            "prec_bar": prec_bar, "recall_min": recall_min, "n_proposals": int(len(lab)), "n_unsafe": int(unsafe.sum())}


def permute_columns_within_set(C: np.ndarray, seed: int) -> np.ndarray:
    """every column independently permuted across the 64 proposals of each set (the NEGATIVE control: marginals kept, every proposal-wise relation destroyed)"""
    rng = np.random.default_rng(seed)
    out = np.array(C, copy=True)
    for i in range(out.shape[0]):
        for k in range(out.shape[-1]):
            out[i, :, k] = out[i, rng.permutation(out.shape[1]), k]
    return out


def a_test(block: np.ndarray, label_dac: np.ndarray, known_leaky: np.ndarray = None, seed: int = 20261019) -> dict:
    """The full admissibility test of one candidate block [N,64,K] against the labels [N,64], WITH controls.  known_leaky = the registered 10-column nosign block (POS-1), if available.
    -> {"admissible": bool, "controls_ok": bool, "candidate": ..., "POS-1": ..., "POS-2": ..., "NEG": ...}"""
    rng = np.random.default_rng(seed)
    res = {"candidate": search(block, label_dac)}
    planted = (np.where(label_dac == 0.0, 0.0, 5.0) + rng.normal(0.0, 0.5, label_dac.shape))[..., None]
    res["POS-2"] = search(np.concatenate([block, planted], axis=-1), label_dac)
    res["NEG"] = search(permute_columns_within_set(block, seed), label_dac)
    pos1_ok = True
    if known_leaky is not None:
        res["POS-1"] = search(known_leaky, label_dac)
        pos1_ok = res["POS-1"]["flagged"]
    res["controls_ok"] = bool(res["POS-2"]["flagged"] and not res["NEG"]["flagged"] and pos1_ok)
    res["admissible"] = bool(res["controls_ok"] and not res["candidate"]["flagged"])
    return res


# ---- the flexible recomputer: DESCRIPTIVE (no authority) ----------------------------------------------------------------------------------------------------------------------------------
def mlp_recompute_auc(block: np.ndarray, label_dac: np.ndarray, fold: np.ndarray, code: np.ndarray, log_idx: np.ndarray, n_logs: int, epochs: int = 12, seed: int = 0) -> dict:
    """5-fold (census fold) OOF within-set F-AUC (Q4) and Q1-3 of a 2 x 64 MLP trained to recompute the DAC label from the block alone; paired log-cluster CIs.  CPU torch."""
    import torch
    import torch.nn as nn
    N, P, K = block.shape
    oof = np.zeros((N, P), np.float32)
    X = torch.tensor(block, dtype=torch.float32)
    y = torch.tensor((label_dac == 1.0).astype(np.float32))
    ok = torch.tensor((label_dac == 0.0) | (label_dac == 1.0))
    mu, sd = X.reshape(-1, K).mean(0), X.reshape(-1, K).std(0).clamp_min(1e-6)
    for k in range(L.FOLDS):
        te = torch.tensor(fold == k)
        tr = ~te
        if not bool(te.any()):
            continue
        torch.manual_seed(seed + k)
        net = nn.Sequential(nn.Linear(K, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        xt = ((X[tr] - mu) / sd)[ok[tr]]
        yt = y[tr][ok[tr]]
        idx = torch.arange(len(xt))
        for _ in range(epochs):
            perm = idx[torch.randperm(len(idx))]
            for i in range(0, len(perm), 4096):
                b = perm[i:i + 4096]
                loss = nn.functional.binary_cross_entropy_with_logits(net(xt[b]).squeeze(-1), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            oof[te.numpy()] = net((X[te] - mu) / sd).squeeze(-1).numpy()
    an = L.Analyzer(label_dac, code, log_idx, n_logs)
    st = an.stats(oof)
    return {"F": an.auc(st, "Q4"), "Q123": an.auc(st, "Q123")}
