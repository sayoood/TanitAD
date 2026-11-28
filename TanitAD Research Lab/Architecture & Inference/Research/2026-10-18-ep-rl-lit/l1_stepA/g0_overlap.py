#!/usr/bin/env python3
"""G0 (PREREG_L1_SEARCH_ADMISSION.md section 2): did P2-s0 train on the L1 fit sets?  Read-only, CPU, stdlib + numpy.

Two independent manifests of what P2-s0 saw, measured against the fit universe (the 24 SFT held-out logs / 1,705 tokens of P2's own held-out table):
  SOURCE A  the PDM-labelled fine-tune P2 ran (served_keys.tsv.gz + navtrain_token_table.tsv.gz: the 174,912 sets of the carved bank)
  SOURCE B  the FINAL-RUN target bank P2 was warm-started from AND re-trains its trajectory loss on
            (refe_train_grow.tar.zst, streamed by g0_scan_traingrow.py: 103,037 tokens / 1,191 logs)
The check function is `g0_check`; `--control` injects an overlap and must make it FAIL.

    python g0_overlap.py --heldout-logs <txt> --heldout-index <npz> --served <tsv.gz> --token-table <tsv.gz> --scan <json> --table <npz> --out <json>
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stepA  # noqa: E402


def sha256_file(p):
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--heldout-index", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--token-table", required=True)
    ap.add_argument("--scan", required=True, help="g0_traingrow_scan.json (final-run target bank: token -> logs)")
    ap.add_argument("--table", required=True, help="table_p2_heldout.npz (the fit universe: P2's own fan, 1,705 tokens)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--reload-at", default="2026-09-30T18:57:53Z", help="rl_pool_by_base.json last_bank_reload_at of the MAIN run")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    ho_logs = set(open(a.heldout_logs, encoding="utf-8").read().split())
    idx = np.load(a.heldout_index, allow_pickle=True)
    tab = np.load(a.table, allow_pickle=True)
    uni_tok = [str(t) for t in tab["token"]]
    uni_log = {str(t): str(l) for t, l in zip(tab["token"], tab["log"])}
    sets_tok = [str(t) for t in idx["token"]]                    # the 3,137 labelled sets of the older (final-model) fan

    # SOURCE A: the carved training bank (P2's PDM fine-tune)
    ttab = {r["token"]: r for r in read_tsv(a.token_table)}
    train_tok_A, ho_flagged, n_train_sets, n_ho_sets, ho_sets_before_reload = set(), set(), 0, 0, 0
    for r in read_tsv(a.served):
        if r["heldout_log"] == "0":
            n_train_sets += 1
            train_tok_A.add(r["token"])
        else:
            n_ho_sets += 1
            ho_flagged.add(r["token"])
            if r["first_label_at"] <= a.reload_at:
                ho_sets_before_reload += 1
    train_log_A = {ttab[t]["log"] for t in train_tok_A if t in ttab}
    A_missing_log = sum(1 for t in train_tok_A if t not in ttab)

    # SOURCE B: the final-run target bank
    scan = json.load(open(a.scan, encoding="utf-8"))
    tl = scan["token_to_logs"]
    train_tok_B = set(tl)
    train_log_B = {l for v in tl.values() for l in v}

    sample = stepA.sample_stratified(uni_tok, [uni_log[t] for t in uni_tok], a.n, a.seed)
    s_log = [uni_log[t] for t in sample]
    out = {"what": "G0 provenance: overlap of the L1 fit sets with what P2-s0 trained on", "evidence": "MEASURED (ours); inputs hashed below",
           "inputs": {k: {"path": getattr(a, k.replace('-', '_')), "sha256": sha256_file(getattr(a, k.replace('-', '_')))} for k in
                      ("heldout-logs", "heldout-index", "served", "token-table", "scan", "table")},
           "universe": {"tokens": len(uni_tok), "logs": len(set(uni_log.values())), "heldout_logs_listed": len(ho_logs),
                        "universe_logs_equal_heldout_logs": set(uni_log.values()) == ho_logs,
                        "label_sets_of_older_fan": len(sets_tok), "label_set_tokens_subset_of_universe": set(sets_tok) <= set(uni_tok)},
           "sample": {"n": len(sample), "seed": a.seed, "rule": "stepA.sample_stratified (proportional, largest remainder, rng=default_rng(seed))",
                      "tokens_sha256": hashlib.sha256("\n".join(sample).encode()).hexdigest(), "logs": len(set(s_log)),
                      "per_log": {l: s_log.count(l) for l in sorted(set(s_log))}, "tokens": sample},
           "sourceA_pdm_finetune": {"served_train_sets": n_train_sets, "train_tokens": len(train_tok_A), "train_logs": len(train_log_A), "tokens_without_table_log": A_missing_log,
                                    "served_sets_in_heldout_logs_(carved_out)": n_ho_sets, "heldout_log_sets_first_labelled_before_main_run_last_reload": ho_sets_before_reload},
           "sourceB_final_run_bank": {"tokens": len(train_tok_B), "logs": len(train_log_B), "members": scan["n_members"], "rows_without_token_or_log": sum(m["rows_without_token_or_log"] for m in scan["per_member"].values()),
                                      "per_member": scan["per_member"]}}
    for name, (tt, ll) in {"A_pdm_finetune": (train_tok_A, train_log_A), "B_final_run_bank": (train_tok_B, train_log_B)}.items():
        out[f"g0_{name}"] = {"universe": stepA.g0_check(uni_tok, set(uni_log.values()), tt, ll),
                             "sample400": stepA.g0_check(sample, set(s_log), tt, ll),
                             "label_sets_3137": stepA.g0_check(sets_tok, {str(l) for l in idx["log"]}, tt, ll)}
    # control: a deliberately overlapping manifest must make G0 FAIL (prereg section 2)
    inj_tok = set(train_tok_A) | {sample[0]}
    inj_log = set(train_log_A) | {s_log[0]}
    out["control_deliberate_overlap_on_A"] = stepA.g0_check(sample, set(s_log), inj_tok, inj_log)
    out["control_note"] = "control must read passed=False with tokens_overlap>=1 and logs_overlap>=1"
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print("ZZG0_DONE", json.dumps({k: out[k] for k in ("g0_A_pdm_finetune", "g0_B_final_run_bank", "control_deliberate_overlap_on_A")})[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
