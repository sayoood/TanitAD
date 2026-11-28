#!/usr/bin/env python3
"""P2 HELD-OUT 64-PROPOSAL TABLE -- gate 1: PREFLIGHT and the DISJOINTNESS PROOF (navtest AND P2's own training set).

Reads only banked files; writes `disjointness_proof.json` and (with --write-tokens) the held-out token plan.  No GPU, no model.

Held-out set  = the 3,137 navtrain held-out sets (24 logs) whose u400-fan PDM labels are banked at
                `s1_read_cache/labels/x/heldout_pdm/pdm_onpolicy_r{0,1}_hw{0..6}.jsonl` (key = log, token, step, rank).
navtest       = `data/refe_navtest/navtest_full_tokens.json` (12,146 tokens / 136 logs, sha256 927bbe38...).
P2 training   = the 174,912 SERVED TRAINING SETS (PREREG_P2 / PREREG_P2_AMENDMENT_1; `refe/p2r_carve.py` removed the 2,924 sets of the 24
                held-out logs from the 177,836-set bank before P2 trained).  Their keys are `raw/rl_pool/served_keys.tsv.gz` (col `heldout_log`),
                token -> log from `raw/rl_pool/navtrain_token_table.tsv.gz`; that table reproduces the carve (177,836 = 174,912 + 2,924).

Evidence classes, per line of the proof:
  * "held-out logs/tokens vs the carved training bank" ... MEASURED here, from the two banked tables above.
  * "P2's run really used the carved bank" ................ INHERITED (PREREG_P2 design + P2R_POD_RUNBOOK + RESULT_P2 sec 4 '174,912' + PREREG_RL_S1 [M]);
    the pod's own carve.log is on the pod and was NOT read (the pod is off limits to this job).  The proof says so.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time

REFE_PLAN = "D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
LABELS = "D:/Projects/TanitAD-artifacts/s1_read_cache/labels/x/heldout_pdm"
SETS = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/sets"
HELDOUT_LOGS_TXT = f"{REFE_PLAN}/splits/sft_heldout_logs.txt"
HELDOUT_LOGS_TXT2 = "D:/Projects/TanitAD-artifacts/_tlane_work/ho/refe_heldout/heldout_logs.txt"
NAVTEST_TOKENS = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
NAVTEST_TOKENS_SHA256 = "927bbe3838dd70af5fa27eadceb09f55acc9c1cfd04cf0a6b520f1519175a955"
SERVED_KEYS = f"{REFE_PLAN}/raw/rl_pool/served_keys.tsv.gz"
TOKEN_TABLE = f"{REFE_PLAN}/raw/rl_pool/navtrain_token_table.tsv.gz"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"
N_HELDOUT_SETS, N_HELDOUT_LOGS = 3137, 24
N_NAVTEST_TOKENS, N_NAVTEST_LOGS = 12146, 136
N_SERVED, N_SERVED_TRAIN, N_SERVED_HELDOUT = 177836, 174912, 2924


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def read_tsv_gz(p):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for ln in f:
            yield dict(zip(hdr, ln.rstrip("\n").split("\t")))


def heldout_keys():
    """[(log, token, step, rank)] of every banked held-out label, in file order (file name -> n)."""
    keys, per_file = [], {}
    for p in sorted(glob.glob(f"{LABELS}/pdm_onpolicy_*.jsonl")):
        n = 0
        for ln in open(p, encoding="utf-8"):
            k = json.loads(ln)["key"]
            keys.append((str(k[0]), str(k[1]), int(k[2]), int(k[3])))
            n += 1
        per_file[os.path.basename(p)] = n
    return keys, per_file


def proof(check_ckpt_sha=True):
    t0 = time.time()
    R = {"what": "P2 held-out table, gate 1: preflight + disjointness proof", "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    keys, per_file = heldout_keys()
    ho_tokens = [k[1] for k in keys]
    ho_logs = sorted({k[0] for k in keys})
    listed = sorted(l for l in open(HELDOUT_LOGS_TXT, encoding="utf-8").read().split())
    listed2 = sorted(l for l in open(HELDOUT_LOGS_TXT2, encoding="utf-8").read().split())
    R["heldout"] = {"n_label_keys": len(keys), "n_unique_keys": len(set(keys)), "n_unique_tokens": len(set(ho_tokens)), "n_logs": len(ho_logs),
                    "labels_per_file": per_file, "logs_equal_splits_sft_heldout_logs_txt": ho_logs == listed, "logs_equal_tlane_heldout_logs_txt": ho_logs == listed2,
                    "sft_heldout_logs_sha256": sha256_file(HELDOUT_LOGS_TXT), "expected_sets": N_HELDOUT_SETS, "expected_logs": N_HELDOUT_LOGS}
    ok = len(keys) == len(set(keys)) == N_HELDOUT_SETS and len(ho_logs) == N_HELDOUT_LOGS and ho_logs == listed == listed2
    # ---- navtest
    ntp = NAVTEST_TOKENS
    nt = json.load(open(ntp, encoding="utf-8"))
    nt_tok = set(nt["tokens"])
    nt_logs = set(nt["token_log"].values())
    R["navtest"] = {"path": ntp, "sha256": sha256_file(ntp), "sha256_expected": NAVTEST_TOKENS_SHA256, "n_tokens": len(nt_tok), "n_logs": len(nt_logs),
                    "tokens_shared_with_heldout": len(nt_tok & set(ho_tokens)), "logs_shared_with_heldout": len(nt_logs & set(ho_logs)),
                    "shared_log_names": sorted(nt_logs & set(ho_logs))[:5], "shared_token_examples": sorted(nt_tok & set(ho_tokens))[:5]}
    ok_nt = (R["navtest"]["sha256"] == NAVTEST_TOKENS_SHA256 and len(nt_tok) == N_NAVTEST_TOKENS and len(nt_logs) == N_NAVTEST_LOGS
             and R["navtest"]["tokens_shared_with_heldout"] == 0 and R["navtest"]["logs_shared_with_heldout"] == 0)
    R["navtest"]["PASS"] = bool(ok_nt)
    # ---- P2's training set
    served, n_ho_flag = [], 0
    for r in read_tsv_gz(SERVED_KEYS):
        served.append(r)
        n_ho_flag += int(r["heldout_log"])
    train_tok = {r["token"] for r in served if r["heldout_log"] == "0"}
    heldout_flagged_tok = {r["token"] for r in served if r["heldout_log"] == "1"}
    tab = {r["token"]: r for r in read_tsv_gz(TOKEN_TABLE)}
    train_logs = {tab[t]["log"] for t in train_tok if t in tab}
    train_tok_no_log = sorted(t for t in train_tok if t not in tab)
    ho_tok_set, ho_log_set = set(ho_tokens), set(ho_logs)
    tab_train_logs = {r["log"] for r in tab.values() if int(r["served_train_sets"]) > 0}
    tab_ho_tokens_with_train = sorted(t for t in ho_tok_set if t in tab and int(tab[t]["served_train_sets"]) > 0)
    tab_heldout_log_rows_with_train = sorted({r["log"] for r in tab.values() if r["log"] in ho_log_set and int(r["served_train_sets"]) > 0})
    R["p2_training_set"] = {
        "what": "the carved on-policy bank = the 174,912 sets P2 trained on (PREREG_P2_AMENDMENT_1 sec 'labels'; refe/p2r_carve.py)",
        "served_keys_path": SERVED_KEYS, "served_keys_sha256": sha256_file(SERVED_KEYS), "token_table_path": TOKEN_TABLE, "token_table_sha256": sha256_file(TOKEN_TABLE),
        "bank_origin": "D:/Projects/TanitAD-artifacts/refe-pod-final-2026-10-04/onpolicy_sets_logs_full.tar.zst (md5 4fc979939c5f9b5140ba59a952da77fb, per rl_pool_by_base.json)",
        "n_served_keys": len(served), "n_served_train": len(served) - n_ho_flag, "n_served_in_heldout_logs": n_ho_flag,
        "reproduces_carve_177836_174912_2924": bool((len(served), len(served) - n_ho_flag, n_ho_flag) == (N_SERVED, N_SERVED_TRAIN, N_SERVED_HELDOUT)),
        "n_train_tokens": len(train_tok), "n_train_logs": len(train_logs), "n_train_tokens_without_log_in_table": len(train_tok_no_log),
        "heldout_tokens_in_train_tokens": len(ho_tok_set & train_tok),
        "heldout_logs_in_train_logs": len(ho_log_set & train_logs), "heldout_log_names_in_train_logs": sorted(ho_log_set & train_logs),
        "token_table_cross_check": {"heldout_tokens_with_served_train_sets_gt_0": len(tab_ho_tokens_with_train),
                                    "heldout_logs_with_any_token_served_train_gt_0": len(tab_heldout_log_rows_with_train),
                                    "heldout_tokens_found_in_table": sum(t in tab for t in ho_tok_set)},
        "served_heldout_flagged_tokens_subset_of_banked_heldout_tokens": bool(heldout_flagged_tok <= ho_tok_set),
        "n_heldout_flagged_tokens": len(heldout_flagged_tok), "n_banked_heldout_tokens_not_flagged_in_bank": len(ho_tok_set - heldout_flagged_tok),
        "note_flagged_vs_banked": "the bank's 2,924 held-out-log sets (1,667 tokens) are a subset of the 3,137 banked held-out label sets (1,705 tokens); the difference "
                                  "(213 sets / 38 tokens) is sets the relabeller's served_keys does not serve; none of them is in the training bank (checked above)",
        "p2_training_logs_vs_navtest_logs_shared": len(train_logs & nt_logs), "p2_training_tokens_vs_navtest_tokens_shared": len(train_tok & nt_tok)}
    P = R["p2_training_set"]
    ok_tr = (P["reproduces_carve_177836_174912_2924"] and P["heldout_tokens_in_train_tokens"] == 0 and P["heldout_logs_in_train_logs"] == 0
             and P["token_table_cross_check"]["heldout_tokens_with_served_train_sets_gt_0"] == 0
             and P["token_table_cross_check"]["heldout_logs_with_any_token_served_train_gt_0"] == 0
             and P["served_heldout_flagged_tokens_subset_of_banked_heldout_tokens"] and P["n_train_tokens_without_log_in_table"] == 0)
    P["PASS_bank_level"] = bool(ok_tr)
    P["evidence_that_P2_RUN_used_the_carved_bank"] = {
        "class": "INHERITED (documents, not re-verified; the pod's carve.log was not read)",
        "sources": ["eval/PREREG_P2.md Design: '--pdm-only: the bank serves ONLY PDM-labelled sets'; scorer loss over '174,912 served sets'",
                    "eval/P2R_POD_RUNBOOK.md lines 99-103: relabel guard refused 177,836 and the carve removed the 24 held-out logs (2,924 sets) IN PLACE",
                    "eval/RESULT_P2.md sec 4.1: 'P2 trained on all 174,912 PDM labels'",
                    "eval/PREREG_RL_S1.md line 31 [M]: 'P2 trains 3 epochs on all 174,912 served sets = 99,495 tokens / 1,167 logs'",
                    "raw/rl_pool/rl_pool_by_base.json: served_train_sets 174912, served_in_heldout_logs 2924, reproduces_p2r_carve true"],
        "n_train_tokens_this_script": len(train_tok), "n_train_logs_this_script": len(train_logs), "register_says": "99,495 tokens / 1,167 logs",
        "counts_match_register": bool(len(train_tok) == 99495 and len(train_logs) == 1167)}
    # ---- the checkpoint
    ck = {"path": CKPT, "sha256_expected": CKPT_SHA256}
    if check_ckpt_sha:
        ck["sha256"] = sha256_file(CKPT)
        ck["PASS"] = ck["sha256"] == CKPT_SHA256
    R["checkpoint"] = ck
    R["PASS"] = bool(ok and ok_nt and ok_tr and (ck.get("PASS", True)))
    R["PASS_parts"] = {"heldout_set": bool(ok), "navtest_disjoint": bool(ok_nt), "p2_training_bank_disjoint": bool(ok_tr), "checkpoint": ck.get("PASS")}
    R["seconds"] = round(time.time() - t0, 1)
    R["reading"] = ("the 24 held-out logs / 3,137 held-out tokens share NO log and NO token with navtest, and none with the 174,912-set bank P2 trained on"
                    if R["PASS"] else "NOT DISJOINT or an input differs from the banked one -- read PASS_parts")
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-ckpt-sha", action="store_true")
    a = ap.parse_args()
    R = proof(check_ckpt_sha=not a.skip_ckpt_sha)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: R[k] for k in ("PASS", "PASS_parts", "reading", "seconds")}, indent=1))
    return 0 if R["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
