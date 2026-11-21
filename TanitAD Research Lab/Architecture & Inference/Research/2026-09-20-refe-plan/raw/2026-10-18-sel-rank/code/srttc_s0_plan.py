#!/usr/bin/env python3
"""SR-TTC S0 -- the MASTER PLAN: which logs / tokens / SETS the registered 10 % fit subsample (Amendment 2, A2.1) consists of.

Pure bookkeeping (metadata only: no feature, no label, no score is opened).  Reads
  * raw/a2_subsample/ranked_fit_logs.tsv + selection.json     (the registered selection, A2.1; its selected-list sha256 is VERIFIED here)
  * rl_pool/navtrain_token_table.tsv.gz  (sha256 c7456c88...)  token -> log, served_train_sets, in_navtest, heldout_log
  * rl_pool/served_keys.tsv.gz           (sha256 a9cbdf29...)  the 177,836 served keys (token, step, rank, ckpt_step, heldout_log); heldout_log == 0 -> P2's 174,912 training sets
and writes plan/master_plan.json:
  logs  [{rank, log, fold, n_tokens, n_sets}]  in key order (the batching order),  tokens {token: log},  sets [(log, token, step, rank, ckpt_step)].
Guards (any miss = exit 1): list sha256 == selection.json's; 117 logs / 8,922 tokens / 15,808 sets; every set's token is a fit token of a selected log; 0 navtest / held-out flags.

    python -I srttc_s0_plan.py --plan-dir <run/plan>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys

FRACTION = "0.10"
FRAC_KEY = "10%"
TOKEN_TABLE_SHA256 = "c7456c882b3d297c5ba202947d82b850e640a5b08648a3f4dea3c7c3dd176cd1"
SERVED_KEYS_SHA256 = "a9cbdf29a7cd12dab2afae2ab0c65ce09524d98e16e2c689aa5ea9372c962afc"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            yield dict(zip(hdr, ln.rstrip("\n").split("\t")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan-dir", required=True)
    a = ap.parse_args(argv)
    pd = a.plan_dir
    sel = json.load(open(f"{pd}/selection.json", encoding="utf-8"))
    want = sel["fractions"][FRAC_KEY]
    rows = [r for r in csv.DictReader(open(f"{pd}/ranked_fit_logs.tsv", encoding="utf-8"), delimiter="\t") if r["first_fraction"] == FRACTION]
    rows.sort(key=lambda r: int(r["rank"]))
    logs = [r["log"] for r in rows]
    got_sha = hashlib.sha256("\n".join(sorted(logs)).encode("utf-8")).hexdigest()
    assert got_sha == want["selected_logs_sha256"], f"selected-list sha256 {got_sha} != registered {want['selected_logs_sha256']}"
    assert len(logs) == want["n_logs"] == 117, len(logs)
    assert [int(r["fold"]) for r in rows] == [int(r["rank"]) % 5 for r in rows], "fold rule (rank mod 5) violated"
    tt_p, sk_p = f"{pd}/rlpool/navtrain_token_table.tsv.gz", f"{pd}/rlpool/served_keys.tsv.gz"
    assert sha256_file(tt_p) == TOKEN_TABLE_SHA256, "token table sha256"
    assert sha256_file(sk_p) == SERVED_KEYS_SHA256, "served_keys sha256"
    lset = set(logs)
    tok_log, flags = {}, {"in_navtest": 0, "heldout_log": 0}
    for r in read_tsv_gz(tt_p):
        if r["log"] in lset and int(r["served_train_sets"]) > 0:
            tok_log[r["token"]] = r["log"]
            flags["in_navtest"] += int(r["in_navtest"])
            flags["heldout_log"] += int(r["heldout_log"])
    sets = []
    for r in read_tsv_gz(sk_p):
        if r["heldout_log"] == "0" and r["token"] in tok_log:
            sets.append((tok_log[r["token"]], r["token"], int(r["step"]), int(r["rank"]), int(r["ckpt_step"])))
    sets.sort()
    assert len(set((s[0], s[1], s[2], s[3]) for s in sets)) == len(sets), "duplicate set key"
    assert len(tok_log) == want["n_tokens"] == 8922, len(tok_log)
    assert len(sets) == want["n_sets"] == 15808, len(sets)
    assert flags == {"in_navtest": 0, "heldout_log": 0}, flags
    per_log_tok, per_log_set = {}, {}
    for t, l in tok_log.items():
        per_log_tok[l] = per_log_tok.get(l, 0) + 1
    for s in sets:
        per_log_set[s[0]] = per_log_set.get(s[0], 0) + 1
    for r in rows:      # the per-log counts must equal the registered ranked table's
        assert per_log_tok.get(r["log"], 0) == int(r["n_fit_tokens"]) and per_log_set.get(r["log"], 0) == int(r["n_fit_sets"]), f"count mismatch {r['log']}"
    doc = {"what": "SR-TTC S0 master plan: the registered 10 % fit subsample (Amendment 2, A2.1)", "selected_logs_sha256": got_sha,
           "token_table_sha256": TOKEN_TABLE_SHA256, "served_keys_sha256": SERVED_KEYS_SHA256,
           "n_logs": len(logs), "n_tokens": len(tok_log), "n_sets": len(sets), "flags_in_navtest_heldout": flags,
           "logs": [{"rank": int(r["rank"]), "log": r["log"], "fold": int(r["fold"]), "n_tokens": per_log_tok[r["log"]], "n_sets": per_log_set[r["log"]]} for r in rows],
           "tokens": tok_log, "sets": sets}
    out = f"{pd}/master_plan.json"
    with open(out, "w", encoding="utf-8", newline="\n") as w:
        json.dump(doc, w)
    print(json.dumps({k: doc[k] for k in ("n_logs", "n_tokens", "n_sets", "selected_logs_sha256", "flags_in_navtest_heldout")}, indent=1))
    print("master_plan.json sha256", sha256_file(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
