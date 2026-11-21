#!/usr/bin/env python3
"""The C-ID control sample: 16 carve-B tokens (8 where Step B's P2HEAD search MOVED the pick in at least one seed, 8 where it did not), chosen by a fixed rule BEFORE any B-REST output exists.
It reads only the BANKED carve-B search output's `moved` flags and the carve's token / log lists (already-read data); no B-REST token, no label.

Rule: order the carve-B tokens by sha256(token + '|spc-cid'); take the first 8 whose search moved in >= 1 seed and the first 8 whose search never moved; the sample is the 16 tokens sorted (the bundle builder sorts anyway).
Why both kinds: an identity on tokens the search never moved is vacuous for the search's output (the output IS the pick); an identity on moved tokens alone would not show that an unmoved token stays unmoved.
Output schema = `fresh_sample*.json` (tokens, logs, tokens_sha256), so `make_stepA_bundle.py --sample` accepts it.
    CUDA_VISIBLE_DEVICES=-1 python make_cid_sample.py --out ../raw/cid_sample.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

CARVE_B = "C:/Users/Admin/par18/l2fresh/raw/fresh_sample_token_disjoint.json"
SEARCH = "C:/Users/Admin/par18/l2stepb/run/search_P2HEAD.npz"
N_MOVED, N_UNMOVED = 8, 8


def pick_cid(tokens, logs, moved_any, n_moved=N_MOVED, n_unmoved=N_UNMOVED):
    order = sorted(range(len(tokens)), key=lambda i: hashlib.sha256((tokens[i] + "|spc-cid").encode()).hexdigest())
    mv = [i for i in order if moved_any[i]][:n_moved]
    um = [i for i in order if not moved_any[i]][:n_unmoved]
    if len(mv) < n_moved or len(um) < n_unmoved:
        raise ValueError(f"not enough tokens: moved {len(mv)}/{n_moved}, unmoved {len(um)}/{n_unmoved}")
    sel = sorted(mv + um, key=lambda i: tokens[i])
    return [tokens[i] for i in sel], [logs[i] for i in sel], [bool(moved_any[i]) for i in sel]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cb = json.load(open(CARVE_B, encoding="utf-8"))
    z = np.load(SEARCH, allow_pickle=False)
    stoks = [str(t) for t in z["token"]]
    assert stoks == cb["tokens"], "the banked search output is not on carve B's token list in carve B's order"
    moved_any = np.asarray(z["moved"]).any(1)
    toks, logs, mv = pick_cid(cb["tokens"], cb["logs"], moved_any)
    rec = {"what": "C-ID control sample of SEARCH-P2-CONFIRM: 16 carve-B tokens (8 moved by the banked P2HEAD search, 8 never moved), fixed rule, chosen before any B-REST output", "rule": __doc__.split("Rule:")[1].split("\n")[0].strip(),
           "n": len(toks), "tokens": toks, "logs": logs, "moved_in_banked_search": mv, "tokens_sha256": hashlib.sha256("\n".join(toks).encode()).hexdigest(),
           "carve_tokens_sha256": cb["tokens_sha256"], "search_banked": SEARCH, "n_moved": int(sum(mv)), "n_logs": len(set(logs))}
    json.dump(rec, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    print("ZZCID_SAMPLE_OK n=%d moved=%d logs=%d sha=%s" % (rec["n"], rec["n_moved"], rec["n_logs"], rec["tokens_sha256"][:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
