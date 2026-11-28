#!/usr/bin/env python3
"""Mutation self-test for `selection_readout.py`: the verdict must flip when the truth flips.

Synthetic tables on the REAL 200-token / 93-log structure (the cluster bootstrap is exercised as used):
  ORACLE_PICK   the scorer ranks exactly by the true PDMS    -> NOT selection-bound, Spearman ~1, no FAILING
  RANDOM_PICK   the scorer's logits are noise                 -> SELECTION_BOUND, Spearman ~0, DAC FAILING
  DAC_ONLY_BAD  every head perfect except DAC (noise)         -> DAC is the ONLY failing output
Prints ZZSELFTEST_OK or ZZSELFTEST_FAIL <case>; exit 0 / 1.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENS = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")
HEAD = ("NC", "DAC", "EP", "TTC", "C", "DDC")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def make(case, tok, rng, M=64):
    N = len(tok)
    sub = np.ones((N, M, 6))
    for c in (0, 1, 3, 4):                                   # binary components, varying within tokens
        sub[:, :, c] = (rng.random((N, M)) < 0.7).astype(float)
    sub[:, :, 5] = 1.0
    sub[:, :, 2] = rng.random((N, M))                        # EP continuous
    pdms = sub[..., 0] * sub[..., 1] * sub[..., 5] * (5 * sub[..., 2] + 5 * sub[..., 3] + 2 * sub[..., 4]) / 12
    # a perfect head: confident logits that reproduce the truth per component
    perfect = logit(np.where(sub == 1.0, 0.98, np.where(sub == 0.0, 0.02, sub)))
    if case == "ORACLE_PICK":
        L = perfect.copy()
        L[:, :, 2] = logit(0.02 + 0.96 * sub[:, :, 2])
    elif case == "RANDOM_PICK":
        L = rng.normal(0, 1, (N, M, 6))
    elif case == "DAC_ONLY_BAD":
        L = perfect.copy()
        L[:, :, 2] = logit(0.02 + 0.96 * sub[:, :, 2])
        L[:, :, 1] = rng.normal(0, 1, (N, M))
    else:
        raise ValueError(case)
    sys.path.insert(0, HERE)
    import selection_readout as R
    pick = R.aggregate(L).argmax(1)
    P = np.cumsum(rng.normal(0, 1, (N, M, 8, 3)), axis=2).astype(np.float32)
    return dict(token=np.array(tok), pdms=pdms, sub=sub, valid=np.ones((N, M), bool), logits=L.astype(np.float32),
                proposals=P, pick=pick, sub_names=np.array(HEAD), head_order=np.array(HEAD))


def run(case, tok, rng):
    d = tempfile.mkdtemp(prefix=f"selftest_{case}_")
    np.savez(os.path.join(d, "table.npz"), **make(case, tok, rng))
    p = subprocess.run([sys.executable, os.path.join(HERE, "selection_readout.py"), "--name", case,
                        "--tokens", TOKENS, "--boot", "2000", "--table-dir", d],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        return None
    return json.load(open(os.path.join(d, "readout.json")))


def main() -> int:
    sj = json.load(open(TOKENS, encoding="utf-8"))
    tok = list(sj["tokens"])
    rng = np.random.default_rng(7)
    fails = []
    r = run("ORACLE_PICK", tok, rng)
    ok = (r is not None and r["verdict"]["outcome"] == "NOT_SELECTION_BOUND" and r["verdict"]["failing_scorer_outputs"] == []
          and r["b_ranking"]["within_token_spearman_agg_vs_pdms"]["mean"] > 0.8
          and abs(r["a_pdms"]["actual"]["mean"] - r["a_pdms"]["oracle"]["mean"]) < 1.0)
    print(f"  ORACLE_PICK   {'ok' if ok else 'FAIL'}  " + (json.dumps(r["verdict"]) + " rho " +
          str(r["b_ranking"]["within_token_spearman_agg_vs_pdms"]["mean"]) if r else "no readout"))
    fails += [] if ok else ["ORACLE_PICK"]
    r = run("RANDOM_PICK", tok, rng)
    ok = (r is not None and r["verdict"]["outcome"] == "SELECTION_BOUND"
          and "DAC" in r["verdict"]["failing_scorer_outputs"]
          and abs(r["b_ranking"]["within_token_spearman_agg_vs_pdms"]["mean"]) < 0.1
          # the ORIGINAL Amendment-3 clause cannot fire here -- the reason Amendment 3a exists
          and not r["verdict"]["original_clause_amendment3"])
    print(f"  RANDOM_PICK   {'ok' if ok else 'FAIL'}  " + (json.dumps(r["verdict"]["outcome"]) + " skill " +
          json.dumps(r["a_pdms"]["selection_skill"]) if r else "no readout"))
    fails += [] if ok else ["RANDOM_PICK"]
    r = run("DAC_ONLY_BAD", tok, rng)
    ok = r is not None and r["verdict"]["failing_scorer_outputs"] == ["DAC"]
    print(f"  DAC_ONLY_BAD  {'ok' if ok else 'FAIL'}  " + (json.dumps(r["verdict"]) if r else "no readout"))
    fails += [] if ok else ["DAC_ONLY_BAD"]
    print("ZZSELFTEST_OK" if not fails else f"ZZSELFTEST_FAIL {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
