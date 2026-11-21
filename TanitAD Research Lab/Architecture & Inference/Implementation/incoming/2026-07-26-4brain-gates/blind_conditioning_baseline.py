#!/usr/bin/env python3
"""
blind_conditioning_baseline -- the REQUIRED circularity firewall
(STRATEGIC_TACTICAL_PROBLEM_SPEC.md sec 0.1).

    "A target that is computable from the model's inference-time inputs by a
     fixed rule is not a target. It is a copy."

Given (X_cond, Y) it trains a SCENE-BLIND predictor on the conditioning channels
only -- no pixels, no history -- and reports:

    acc_blind   accuracy of the blind predictor (leave-one-CLUSTER-out)
    acc_major   majority-class rate
    verdict     REFUSED if acc_blind >= REFUSE_FRAC * acc_ceiling

The label is admissible only if a scene-blind predictor CANNOT reach ceiling, and
the reported capability is always  skill = acc_model - acc_blind , never
acc_model alone.

This would have caught  route_target = _NAV_TO_ROUTE[nav_cmd]  (acc_blind = 1.0000)
for CPU-minutes, before it cost months.

Leave-one-cluster-out is used rather than a random split because the resampling
unit of this program is the episode/scene cluster; a random split would leak
sibling decision points from the same scene and UNDER-state acc_blind, i.e. it
would fail unsafely.

Includes a self-test (`--self-test`) asserting that a synthetic ECHO label is
REFUSED and a synthetic unrelated label is ADMITTED. Per the 07-26 C10 lesson:
a guardrail that has never rendered its failing verdict is a comment.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

REFUSE_FRAC = 0.98          # spec sec 0.1 step 5


# --------------------------------------------------------------------------- #
# a tiny multinomial logistic scorer over a VARIABLE-ARITY option set           #
# --------------------------------------------------------------------------- #
class OptionScorer:
    """Scores each option from its own feature vector; softmax over the option set.

    Variable arity is handled natively (this is exactly the property a fixed
    {L,S,R} head does not have). Trained by full-batch gradient descent on the
    multinomial cross-entropy -- n is tiny, so this is exact and deterministic.
    """

    def __init__(self, dim: int, l2: float = 1e-3, iters: int = 4000, lr: float = 0.20, seed: int = 0):
        self.dim, self.l2, self.iters, self.lr = dim, l2, iters, lr
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.01, size=dim)
        self.b = 0.0

    def _probs(self, feats):
        s = feats @ self.w + self.b
        s = s - s.max()
        e = np.exp(s)
        return e / max(e.sum(), 1e-12)

    def fit(self, groups):
        """groups: list of (feats [K,dim], y int)."""
        if not groups:
            return self
        for _ in range(self.iters):
            gw = np.zeros(self.dim)
            for feats, y in groups:
                p = self._probs(feats)
                onehot = np.zeros(len(p))
                onehot[y] = 1.0
                gw += (p - onehot) @ feats
            gw = gw / len(groups) + self.l2 * self.w
            self.w -= self.lr * gw
        return self

    def predict(self, feats):
        return int(np.argmax(self._probs(feats)))


# --------------------------------------------------------------------------- #
# the firewall                                                                 #
# --------------------------------------------------------------------------- #
def run_firewall(groups, clusters, name, acc_ceiling=1.0, seed=0):
    """groups: [(feats [K,dim], y)] ; clusters: [cluster_id] aligned with groups.

    Returns a dict carrying its own provenance. `correct_blind` and
    `correct_major` are per-item 0/1 vectors so the caller can run the paired
    episode-cluster bootstrap on them.
    """
    n = len(groups)
    if n == 0:
        return {"name": name, "n": 0, "verdict": "NO-DATA"}
    dim = groups[0][0].shape[1]

    # ---- majority class (over option INDEX, the only class notion a variable-arity
    #      option set has). Computed leave-one-cluster-out as well, so it is a fair
    #      reference rather than an oracle.
    ys = [g[1] for g in groups]
    correct_blind, correct_major, preds = [], [], []
    uniq = sorted(set(clusters))
    for c in uniq:
        tr = [i for i in range(n) if clusters[i] != c]
        te = [i for i in range(n) if clusters[i] == c]
        if not tr:
            continue
        m = OptionScorer(dim, seed=seed).fit([groups[i] for i in tr])
        tr_y = [ys[i] for i in tr]
        maj = max(set(tr_y), key=tr_y.count)
        for i in te:
            feats, y = groups[i]
            p = m.predict(feats)
            preds.append(p)
            correct_blind.append(int(p == y))
            correct_major.append(int(min(maj, feats.shape[0] - 1) == y))

    acc_blind = float(np.mean(correct_blind))
    acc_major = float(np.mean(correct_major))
    chance = float(np.mean([1.0 / g[0].shape[0] for g in groups]))
    refused = bool(acc_blind >= REFUSE_FRAC * acc_ceiling)
    return {
        "name": name,
        "n": n,
        "n_clusters": len(uniq),
        "acc_blind": round(acc_blind, 4),
        "acc_major": round(acc_major, 4),
        "acc_chance": round(chance, 4),
        "acc_ceiling": acc_ceiling,
        "refuse_threshold": round(REFUSE_FRAC * acc_ceiling, 4),
        "verdict": "REFUSED (target is recoverable from conditioning alone)" if refused else "ADMITTED",
        "refused": refused,
        "cv": "leave-one-cluster-out",
        "correct_blind": correct_blind,
        "correct_major": correct_major,
    }


# --------------------------------------------------------------------------- #
# self-test -- the evaluator MUST render the failing verdict on a known echo    #
# --------------------------------------------------------------------------- #
def self_test():
    rng = np.random.default_rng(0)
    ok = True

    # (1) ECHO label: one feature column IS the answer -> must be REFUSED
    groups, clusters = [], []
    for i in range(60):
        k = rng.integers(2, 4)
        y = int(rng.integers(0, k))
        f = rng.normal(size=(k, 3))
        f[:, 0] = 0.0
        f[y, 0] = 1.0                      # the echo channel
        groups.append((f, y))
        clusters.append(i % 12)
    r = run_firewall(groups, clusters, "SELFTEST-echo")
    print("  echo   : acc_blind=%.4f -> %s" % (r["acc_blind"], r["verdict"]))
    if not r["refused"]:
        print("  FAIL: an echo label was ADMITTED"); ok = False

    # (2) UNRELATED label: features carry no information -> must be ADMITTED
    groups, clusters = [], []
    for i in range(60):
        k = int(rng.integers(2, 4))
        y = int(rng.integers(0, k))
        groups.append((rng.normal(size=(k, 3)), y))
        clusters.append(i % 12)
    r = run_firewall(groups, clusters, "SELFTEST-noise")
    print("  noise  : acc_blind=%.4f -> %s" % (r["acc_blind"], r["verdict"]))
    if r["refused"]:
        print("  FAIL: an uninformative label was REFUSED"); ok = False

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    print(__doc__)
