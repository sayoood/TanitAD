#!/usr/bin/env python3
"""L2-FRESH-STEPA step 1: the fresh carve + G0, with Step A's OWN sampler (`stepA.sample_stratified`) and a NEW seed.  Reads NO label and NO score.

The brief's recipe is mode `literal`: universe = P2's held-out plan (1,705 tokens / 24 logs), minus every token whose LOG is one of Step A's 24 logs, then 400 tokens by Step A's sampler.
MEASURED (probe_fresh_universe.py): that universe is EMPTY, because Step A's 24 logs ARE the plan's 24 logs (and `sft_heldout_logs.txt`'s), so `literal` exits 3 and writes an explicit EMPTY record
(never a silent n=0 sample).  Two other modes exist so that the Master Mind's choice is ONE flag, but they are CANDIDATES, not the brief:
  token_disjoint   B  universe = plan tokens not in Step A's 400 (1,305 tokens, the SAME 24 logs)                       -> log-shared, scene-seen, full banked fan (G2b literal)
  log_disjoint_c   C  universe = `c_resolvable_tokens.json` of probe_fresh_universe.py (navtrain tokens of logs OUTSIDE the 24, DB present, four frames resolve) -> log-disjoint; G0 FAILS A1.2 by
                      construction (the 24 logs are exactly the complement of M-PDM in navtrain), no banked P2 fan (G2b needs a bridge)
G0 is evaluated exactly as g0_overlap.py did (M-PDM = the PDM-label bank of P2's scorer loss; M-SCENE = the final-run target bank), with its deliberate-overlap control, PLUS the overlap with Step A's sample.

    python fresh_carve.py --mode literal|token_disjoint|log_disjoint_c --out <dir> [--seed 20261019] [--n 400] [--unserved-only]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
sys.path.insert(0, L1_CODE)
import stepA  # noqa: E402

NEW_SEED = 20261019        # the date, the convention of Step A's 20260927; fixed here BEFORE any fresh label or score exists.  The bootstrap seed stays 20260927 (Step A's).
W = "D:/Projects/TanitAD-artifacts/l1_stepA_run"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
TABLE = "D:/Projects/TanitAD-artifacts/p2_heldout_table_full_2026-10-18/table_p2_heldout.npz"
PK = "C:/Users/Admin/par18/l1prep/work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
SERVED = f"{PK}/raw/rl_pool/served_keys.tsv.gz"
TOKTAB = f"{PK}/raw/rl_pool/navtrain_token_table.tsv.gz"
SCAN = "C:/Users/Admin/par18/l1run/code/raw/g0_traingrow_scan2.json"
C_LIST = "C:/Users/Admin/par18/l2fresh/raw/c_resolvable_tokens.json"
MODES = ("literal", "token_disjoint", "log_disjoint_c")


def sha_tokens(tokens) -> str:
    return hashlib.sha256("\n".join(tokens).encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(head, line.rstrip("\n").split("\t")))


def universe(mode, plan, stepA_tokens, c_rows=None, unserved_only=False):
    """-> (tokens, log of each token, description).  Pure (tested)."""
    tl = plan["token_log"]
    logsA = {tl[t] for t in stepA_tokens}
    if mode == "literal":
        toks = [t for t in plan["tokens"] if tl[t] not in logsA]
        return toks, [tl[t] for t in toks], "plan tokens whose log is NOT one of Step A's logs"
    if mode == "token_disjoint":
        sA = set(stepA_tokens)
        toks = [t for t in plan["tokens"] if t not in sA]
        return toks, [tl[t] for t in toks], "plan tokens that are NOT in Step A's sample (same logs)"
    if mode == "log_disjoint_c":
        rows = [r for r in c_rows if r["log"] not in logsA and not (unserved_only and r["served"])]
        rows = sorted(rows, key=lambda r: r["token"])
        return [r["token"] for r in rows], [r["log"] for r in rows], "navtrain tokens of logs outside Step A's, DB present, four frames resolve" + (" (unserved only)" if unserved_only else "")
    raise ValueError(f"unknown mode {mode!r}")


def carve(mode, plan, stepA_tokens, n, seed, c_rows=None, unserved_only=False):
    """-> dict(status, tokens, logs, ...).  status EMPTY when the universe has no token (n=0 is NEVER returned as a sample), SHORT when it has fewer than n (all taken, stated), OK otherwise."""
    toks, logs, desc = universe(mode, plan, stepA_tokens, c_rows, unserved_only)
    tlA = plan["token_log"]
    out = {"mode": mode, "rule": desc, "seed": seed, "n_requested": n, "universe_tokens": len(toks), "universe_logs": len(set(logs)), "sampler": "stepA.sample_stratified (every log 1, rest proportional to size-1, largest remainder, default_rng(seed))"}
    if len(toks) == 0:
        out.update(status="EMPTY", n=0, tokens=[], logs=[])
        return out
    if len(toks) <= n:
        sel = sorted(toks)
        out["status"] = "SHORT"
    else:
        if len(set(logs)) > n:
            raise ValueError(f"{len(set(logs))} logs > n={n}: Step A's sampler needs n >= #logs (not altered here)")
        sel = stepA.sample_stratified(toks, logs, n, seed)
        out["status"] = "OK"
    lmap = dict(zip(toks, logs))
    out.update(n=len(sel), tokens=sel, logs=[lmap[t] for t in sel])
    out["stepA_overlap"] = {"tokens": len(set(sel) & set(stepA_tokens)), "logs": len(set(lmap[t] for t in sel) & {tlA[t] for t in stepA_tokens})}
    return out


def g0_for(carve_rec, stepA_tokens):
    """G0 exactly as g0_overlap.py: M-PDM (A) and M-SCENE (B), each with the deliberate-overlap control; vacuous passes on an empty set are NOT reported as passes."""
    sel, slog = carve_rec["tokens"], carve_rec["logs"]
    ttab = {r["token"]: r for r in read_tsv(TOKTAB)}
    train_tok_A = {r["token"] for r in read_tsv(SERVED) if r["heldout_log"] == "0"}
    train_log_A = {ttab[t]["log"] for t in train_tok_A if t in ttab}
    scan = json.load(open(SCAN, encoding="utf-8"))["token_to_logs"]
    train_tok_B, train_log_B = set(scan), {l for v in scan.values() for l in v}
    out = {"evidence": "MEASURED (ours); manifests = those of g0_overlap.py (L1 Step A)", "n_fit_tokens": len(sel), "n_fit_logs": len(set(slog))}
    if not sel:
        out["evaluable"] = False
        out["note"] = "EMPTY carve: g0_check on an empty set passes vacuously; that is NOT a pass"
        return out
    out["evaluable"] = True
    for name, (tt, ll) in {"M-PDM": (train_tok_A, train_log_A), "M-SCENE": (train_tok_B, train_log_B)}.items():
        out[name] = stepA.g0_check(sel, set(slog), tt, ll)
    out["control_deliberate_overlap_on_M-PDM"] = stepA.g0_check(sel, set(slog), train_tok_A | {sel[0]}, train_log_A | {slog[0]})
    out["control_must_fail"] = bool(not out["control_deliberate_overlap_on_M-PDM"]["passed"])
    out["A1_2_pass_rule"] = "G0 PASSES iff the fit logs are disjoint from M-PDM (L1 Amendment A1.2); M-SCENE overlap is a DECLARED EXPOSURE (A1.3)"
    out["A1_2_verdict"] = "PASS" if out["M-PDM"]["passed"] else "FAIL"
    out["exposure_declared"] = {"M-SCENE_overlap_tokens": out["M-SCENE"]["tokens_overlap"], "M-SCENE_overlap_logs": out["M-SCENE"]["logs_overlap"]}
    sA = set(stepA_tokens)
    out["overlap_with_stepA_sample"] = {"tokens": len(set(sel) & sA), "n_stepA": len(sA)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=NEW_SEED)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--unserved-only", action="store_true")
    ap.add_argument("--name", default=None, help="file stem; default: fresh_sample (literal) or fresh_sample_<mode>")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    plan = json.load(open(PLAN))
    sA = json.load(open(f"{W}/sample.json"))
    c_rows = json.load(open(C_LIST)) if a.mode == "log_disjoint_c" else None
    rec = carve(a.mode, plan, sA["tokens"], a.n, a.seed, c_rows, a.unserved_only)
    rec["tokens_sha256"] = sha_tokens(rec["tokens"])
    rec["parent_prereg_sha256"] = stepA.PREREG_SHA256
    rec["table_sha256"] = sha_file(TABLE) if a.mode != "log_disjoint_c" else None
    rec["stepA_sample_tokens_sha256"] = sha_tokens(sA["tokens"])
    rec["stepA_seed"] = sA["seed"]
    rec["g0"] = g0_for(rec, sA["tokens"])
    stem = a.name or ("fresh_sample" if a.mode == "literal" else f"fresh_sample_{a.mode}")
    p = os.path.join(a.out, stem + ".json")
    json.dump(rec, open(p, "w", encoding="utf-8"), indent=1)
    print("ZZCARVE", a.mode, rec["status"], "n=", rec["n"], "logs=", len(set(rec["logs"])), "universe=", rec["universe_tokens"], "sha", rec["tokens_sha256"][:12], "->", p, flush=True)
    return 3 if rec["status"] == "EMPTY" else 0


if __name__ == "__main__":
    sys.exit(main())
