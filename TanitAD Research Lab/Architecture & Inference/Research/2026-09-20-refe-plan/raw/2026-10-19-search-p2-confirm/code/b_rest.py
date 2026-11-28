#!/usr/bin/env python3
"""SEARCH-P2-CONFIRM deliverable 1: the B-REST universe, enumerated (never assumed).  Reads NO label and NO score.

B-REST = P2's held-out PLAN tokens (1,705 tokens / 24 logs, `heldout_tokens.json`) minus Step A's 400 (L1's `sample.json`, seed 20260927) minus carve B's 400 (L2-FRESH `fresh_sample_token_disjoint.json`, seed 20261019).
The output is written in the SAME schema as `fresh_sample*.json` so `make_stepA_bundle.py --sample` accepts it unchanged (the bundle builder sorts the tokens, so `tokens` is stored sorted and the
token-list sha256 is over the sorted list joined by newlines = the bundle's `ctx.json` `tokens_sha256`).

Assertions (any failure exits 3 and writes nothing):  |plan| = 1705; Step A and carve B each 400 unique tokens, both inside the plan; Step A and carve B disjoint; B-REST disjoint from both; the three partition the
plan (400 + 400 + |B-REST| = 1705); every B-REST token has a log; every B-REST token has a banked row in P2's held-out table (the bundle builder's G2b needs it); the carve-B token list hashes to its sealed sha.
Also reports (MEASURED, no label): per-log token counts for the three sets, the log count, and G0 exactly as `fresh_carve.g0_for` computes it for carve B (M-PDM = the PDM-label bank of P2's scorer loss, M-SCENE = the
final-run target bank).

    CUDA_VISIBLE_DEVICES=-1 python b_rest.py --out ../raw/b_rest.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

W = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
TABLE = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
CARVE_B = "C:/Users/Admin/par18/l2fresh/raw/fresh_sample_token_disjoint.json"
CARVE_B_SHA = "0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c"   # sealed (PREREG_L2_FRESH.md; Step B's data clause)
STEPA_SHA = "211e26406f5b4eb083ae38b1c73839826d560042cd75ccf9ff6656262e7c0ffd"      # L1 Step A's 400 (sample.json tokens_sha)
EXPECT_REST = 905                                                                    # the MM ruling's number: verified here, not assumed
FRESH_CODE = "C:/Users/Admin/par18/l2fresh/code"


def sha_tokens(tokens) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def b_rest(plan_tokens, token_log, stepA_tokens, carveB_tokens):
    """-> (sorted B-REST tokens, aligned logs).  Pure; raises AssertionError on any broken invariant."""
    P, A, B = list(plan_tokens), list(stepA_tokens), list(carveB_tokens)
    assert len(set(P)) == len(P), "plan tokens not unique"
    assert len(set(A)) == len(A) and len(set(B)) == len(B), "Step A / carve B tokens not unique"
    sP, sA, sB = set(P), set(A), set(B)
    assert sA <= sP, f"{len(sA - sP)} Step A tokens are not in the plan"
    assert sB <= sP, f"{len(sB - sP)} carve B tokens are not in the plan"
    assert not (sA & sB), f"Step A and carve B overlap on {len(sA & sB)} tokens"
    rest = sorted(sP - sA - sB)
    assert not (set(rest) & sA) and not (set(rest) & sB), "B-REST overlaps Step A or carve B"
    assert len(sA) + len(sB) + len(rest) == len(sP), "the three sets do not partition the plan"
    assert all(t in token_log for t in rest), "a B-REST token has no log"
    return rest, [token_log[t] for t in rest]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-g0", action="store_true")
    a = ap.parse_args(argv)
    plan = json.load(open(PLAN))
    sA = json.load(open(f"{W}/sample.json"))
    cb = json.load(open(CARVE_B, encoding="utf-8"))
    tl = plan["token_log"]
    try:
        assert sha_tokens(cb["tokens"]) == CARVE_B_SHA == cb["tokens_sha256"], "carve B token list does not hash to the sealed sha"
        assert sha_tokens(sA["tokens"]) == STEPA_SHA, "Step A token list does not hash to its recorded sha"
        rest, logs = b_rest(plan["tokens"], tl, sA["tokens"], cb["tokens"])
        t = np.load(TABLE, allow_pickle=True)
        ix = {str(k) for k in t["token"]}
        miss = [x for x in rest if x not in ix]
        assert not miss, f"{len(miss)} B-REST tokens have no banked row in the held-out table: {miss[:3]}"
        n_ok = len(rest)
    except AssertionError as e:
        print("ZZBREST_FAIL", e, flush=True)
        return 3
    cnt = lambda toks: dict(sorted(collections.Counter(tl[x] for x in toks).items()))   # noqa: E731
    cA, cB, cR = cnt(sA["tokens"]), cnt(cb["tokens"]), cnt(rest)
    rec = {"mode": "token_disjoint_rest", "rule": "plan tokens in NEITHER Step A's 400 NOR carve B's 400 (the same 24 logs)", "seed": None, "n_requested": None,
           "universe_tokens": len(rest), "universe_logs": len(set(logs)), "sampler": "none: the whole remainder (no sampling, no seed)", "status": "OK", "n": len(rest), "tokens": rest, "logs": logs,
           "tokens_sha256": sha_tokens(rest), "table_sha256": sha_file(TABLE), "stepA_sample_tokens_sha256": sha_tokens(sA["tokens"]), "stepA_seed": sA["seed"],
           "carveB_tokens_sha256": cb["tokens_sha256"], "carveB_seed": cb["seed"], "plan_tokens": len(plan["tokens"]), "plan_logs": len(set(tl[x] for x in plan["tokens"])),
           "disjointness": {"B-REST_and_StepA": len(set(rest) & set(sA["tokens"])), "B-REST_and_carveB": len(set(rest) & set(cb["tokens"])), "StepA_and_carveB": len(set(sA["tokens"]) & set(cb["tokens"])),
                            "partition_sum": 400 + 400 + len(rest), "plan_tokens": len(plan["tokens"]), "banked_table_rows_present": n_ok},
           "verified_vs_ruling": {"ruling_says": EXPECT_REST, "measured": len(rest), "equal": len(rest) == EXPECT_REST},
           "per_log_counts": {"stepA": cA, "carveB": cB, "B-REST": cR},
           "evidence": "MEASURED (ours): code/b_rest.py reading heldout_tokens.json, l1_stepA_run/sample.json, fresh_sample_token_disjoint.json and the banked table; NO label, NO score"}
    if not a.no_g0:
        sys.path.insert(0, FRESH_CODE)
        import fresh_carve as FC  # noqa: E402   (imports L1's stepA; reads the two PDM / scene manifests, no label)
        rec["g0"] = FC.g0_for({"tokens": rest, "logs": logs}, sA["tokens"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZBREST_OK n=%d logs=%d sha=%s equal_to_905=%s -> %s" % (len(rest), len(set(logs)), rec["tokens_sha256"][:16], rec["verified_vs_ruling"]["equal"], a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
