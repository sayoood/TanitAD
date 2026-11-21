#!/usr/bin/env python3
"""draw_e2.py -- draw E2 of PREREG_SCORER1P6 section 3: 2,000 navtest tokens, uniform, token-disjoint from C3, seed 20261108.

It ONLY draws a list. It reads the navtest token export (token ids + log names) and C3's token list; it never opens a score, a csv of results,
a table or a proposal file. Output (W3 token-file format, usable as `score_e.py --tokens`):
    <out>/e2_tokens.json   {"tokens": [sorted ids], "token_log": {id: log_name}}
    <out>/e2_meta.json     seed, pool size, sha256 of the token list, of the json file, of the export and of the C3 list
    <out>/e2_tokens.sha256 one line: <sha256 of e2_tokens.json>  e2_tokens.json

DEFINITION (so another implementation draws the same list): pool = sorted(navtest export token ids) minus C3 token ids;
idx = numpy.random.default_rng(20261108).choice(len(pool), size=2000, replace=False); E2 = sorted(pool[i] for i in idx).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/draw_e2.py --out <dir>
"""
import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

SEED, N_E2 = 20261108, 2000
DEFAULT_EXPORT = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DEFAULT_C3 = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def draw(all_tokens, c3_tokens, n=N_E2, seed=SEED):
    """pure function: -> sorted list of n tokens drawn from sorted(all) minus c3"""
    c3 = set(c3_tokens)
    pool = sorted(t for t in all_tokens if t not in c3)
    if len(pool) < n:
        raise ValueError(f"pool of {len(pool)} tokens is smaller than {n}")
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return sorted(pool[int(i)] for i in idx), len(pool)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_EXPORT)
    ap.add_argument("--c3", default=DEFAULT_C3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_E2)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    exp = json.load(gzip.open(a.export, "rt", encoding="utf-8"))["tokens"]
    c3 = json.load(open(a.c3, encoding="utf-8"))["tokens"]
    toks, pool_n = draw(list(exp), c3, a.n, a.seed)
    assert len(set(toks)) == a.n and not (set(toks) & set(c3))
    os.makedirs(a.out, exist_ok=True)
    body = json.dumps({"tokens": toks, "token_log": {t: str(exp[t]["log_name"]) for t in toks}}).encode("utf-8")
    p = os.path.join(a.out, "e2_tokens.json")
    open(p, "wb").write(body)
    meta = {"seed": a.seed, "n": a.n, "pool_n": pool_n, "n_navtest": len(exp), "n_c3": len(c3), "n_logs": len({exp[t]["log_name"] for t in toks}),
            "sha256_token_list": sha256_bytes("\n".join(toks).encode("utf-8")), "sha256_e2_tokens_json": sha256_bytes(body),
            "sha256_export": sha256_file(a.export), "sha256_c3_list": sha256_file(a.c3), "export": a.export, "c3": a.c3,
            "reads_scores": False}
    json.dump(meta, open(os.path.join(a.out, "e2_meta.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(a.out, "e2_tokens.sha256"), "w", encoding="utf-8").write(f"{meta['sha256_e2_tokens_json']}  e2_tokens.json\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
