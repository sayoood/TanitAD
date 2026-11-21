#!/usr/bin/env python3
"""P2 OWN 64-PROPOSAL TABLE -- STEP 1: write the token files BEFORE any P2 proposal is dumped or scored.

Population: every token whose P2 NAVSIM v1.1 navtest row has score == 0.0 EXACTLY in the banked original per-token csv
(`refe_stage2_p2_full.csv`, sha256 61073a61...). The landed analysis (`../zero_tail.py`, `zero tail` PREREG section 3) defines a zero token this
way, and RESULT_P2 says there are 509.  Output files are W3-format ({'rule','tokens','token_log'}) so `refe_navtest_seam.py` and
`score_navtest_refe.py` consume them unchanged:

  tokens_p2zero.json         the 509 P2 zero tokens (sorted ascending by token string)
  tokens_p2zero_smoke8.json  8 of the 509, np.random.default_rng(20261015).choice(509, 8, replace=False) over the sorted list

Nothing here reads a model output other than the already-banked P2 csv.  It is written ONCE (refuses to overwrite).
"""
import csv
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = "D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv"
CSV_SHA256 = "61073a61e44a3dcf41fad6dc8c5b261ce97eb8f8d8b49c1adc077dd7c04b61e6"
TOKENS_FULL = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"
SMOKE_SEED = 20261015
N_ZERO_EXPECTED = 509


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    assert sha(CSV) == CSV_SHA256, "the P2 csv is not the banked original"
    rows, avg = {}, None
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["token"] == "average":
            avg = float(r["score"])
            continue
        rows[r["token"]] = float(r["score"])
    assert len(rows) == 12146, len(rows)
    mean100 = float(np.mean(list(rows.values())) * 100)
    assert abs(mean100 - 88.82) <= 0.01 and abs(mean100 - avg * 100) < 1e-6, (mean100, avg)
    zero = sorted(t for t, s in rows.items() if s == 0.0)
    assert len(zero) == N_ZERO_EXPECTED, f"P2 zero count {len(zero)} != {N_ZERO_EXPECTED}"
    full = json.load(open(TOKENS_FULL, encoding="utf-8"))
    assert set(full["tokens"]) == set(rows), "the csv token set differs from navtest_full_tokens.json"
    token_log = {t: full["token_log"][t] for t in zero}
    rng = np.random.default_rng(SMOKE_SEED)
    idx = np.sort(rng.choice(len(zero), size=8, replace=False))
    smoke = [zero[i] for i in idx]
    prov = {"csv": CSV, "csv_sha256": CSV_SHA256, "n_csv_tokens": len(rows), "p2_mean_x100": mean100,
            "n_zero_score_exactly_0": len(zero), "ckpt": CKPT, "ckpt_sha256": CKPT_SHA256,
            "n_logs": len(set(token_log.values())), "navtest_full_tokens_json": TOKENS_FULL,
            "navtest_full_tokens_json_sha256": sha(TOKENS_FULL)}
    for name, toks, rule in (("tokens_p2zero.json", zero, "P2 zero-PDMS tokens: score == 0.0 exactly in refe_stage2_p2_full.csv"),
                             ("tokens_p2zero_smoke8.json", smoke, "8 of the 509 P2 zero tokens (seed 20261015, plumbing + same-pick smoke)")):
        p = os.path.join(HERE, name)
        assert not os.path.exists(p), f"{name} already exists -- it is written ONCE, before any P2 proposal is dumped"
        doc = {"rule": rule, "tokens": toks, "token_log": {t: token_log[t] for t in toks},
               "provenance": dict(prov, n_tokens=len(toks), smoke_seed=SMOKE_SEED if "smoke" in name else None)}
        json.dump(doc, open(p, "w", encoding="utf-8"), indent=0)
        print(name, len(toks), "tokens, sha256", sha(p))
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
