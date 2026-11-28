#!/usr/bin/env python3
"""Write the FRESH confirmation token list (tokens_confirm.json) and the smoke list (tokens_smoke.json) ONCE, before any u400 run on them.

Population (PREREG_SEL_CONFIRM.md section 2), a CENSUS, no sampling choice:
  every NAVSIM v1.1 navtest token that is
    (a) NOT one of the 1,058 C3 tokens (C3 tokens.json),
    (b) NON-ZEROED in the banked u400 full-navtest csv (NC x DAC != 0 -- every zeroed token is already in C3, so this is implied by (a); asserted),
    (c) in a log that is NOT one of YSEL's 61 TRAIN logs (split.json) -- i.e. a YSEL held-out log or one of the 13 navtest logs C3 never touched.
  Tokens ascending by string. Harness chunks = whole logs packed greedily (logs ascending) to <= 1,000 tokens per chunk, because W3's harness
  passes the token list on the Hydra command line (C3 fitted 1,058; Windows caps a command line at 32,767 characters).
Smoke list: 6 C3 tokens over 2 logs (the sim token's TRAIN log + one held-out log), chosen by a fixed rule from the C3 table (below); they have banked
TRUE per-proposal rows, so the GPU dump and the 64 harness columns must reproduce them bit for bit.
"""
import csv
import gzip
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YSEL = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
sys.path.insert(0, YSEL)
sys.path.insert(0, HERE)
EXP = "F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
C3 = ("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-11-c3-selection-headroom")
BANKED_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"
CHUNK_MAX = 1000
# Two fresh tokens were dumped on CPU and their banked/rule columns scored in a PLUMBING test BEFORE the prereg was frozen (their outcome was
# seen). They are EXCLUDED, so no token whose result was seen before the freeze can enter the confirmation (PREREG_SEL_CONFIRM.md section 0).
EXCLUDE_SEEN = {"019eaadd84cb56f8": "CPU plumbing dump + verdict columns scored 2026-11-07 19:13, before the freeze",
                "335576eae9705bfc": "CPU plumbing dump + verdict columns scored 2026-11-07 19:13, before the freeze"}
SIM_TOKEN = "41d7b533797c5209"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump_lf(obj, p):
    with open(p, "wb") as f:
        f.write(json.dumps(obj, indent=0).encode("utf-8"))


def main():
    out_p = os.path.join(HERE, "tokens_confirm.json")
    smoke_p = os.path.join(HERE, "tokens_smoke.json")
    assert not os.path.exists(out_p), "tokens_confirm.json already exists -- it is written ONCE"
    exp = json.load(gzip.open(EXP, "rt", encoding="utf-8"))["tokens"]
    c3 = json.load(open(C3 + "/tokens.json", encoding="utf-8"))
    split = json.load(open(YSEL + "/split.json", encoding="utf-8"))
    rows = {r["token"]: r for r in csv.DictReader(open(BANKED_CSV, encoding="utf-8")) if r["token"] != "average"}
    assert len(exp) == 12146 and len(rows) == 12146
    zero = {t for t, r in rows.items() if float(r["no_at_fault_collisions"]) * float(r["drivable_area_compliance"]) == 0.0}
    c3s = set(c3["tokens"])
    assert len(zero) == 529 and zero <= c3s, "every zeroed u400 token must already be in C3"
    trl, hol = set(split["train_logs"]), set(split["heldout_logs"])
    assert not (trl & hol) and (trl | hol) == set(c3["token_log"].values())
    toks = sorted(t for t in exp if t not in c3s and t not in zero and exp[t]["log_name"] not in trl and t not in EXCLUDE_SEEN)
    assert all(t in exp and t not in c3s and exp[t]["log_name"] not in trl for t in EXCLUDE_SEEN)
    token_log = {t: exp[t]["log_name"] for t in toks}
    log_class = {lg: ("heldout" if lg in hol else "new") for lg in set(token_log.values())}
    assert not (set(toks) & c3s) and not (set(token_log.values()) & trl)
    by = {}
    for t in toks:
        by.setdefault(token_log[t], []).append(t)
    chunks, cur, n = [], [], 0
    for lg in sorted(by):
        if cur and n + len(by[lg]) > CHUNK_MAX:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(lg)
        n += len(by[lg])
    if cur:
        chunks.append(cur)
    chunk_of_log = {lg: i for i, c in enumerate(chunks) for lg in c}
    doc = {"rule": "H-SEL-1 fresh confirmation: ALL navtest tokens not in C3 (1,058), non-zeroed in the banked u400 csv, in logs outside YSEL's 61 TRAIN logs",
           "tokens": toks, "token_log": token_log, "stratum": {t: "random" for t in toks}, "log_class": log_class,
           "chunks": [{"chunk": i, "logs": c, "n_tokens": sum(len(by[lg]) for lg in c)} for i, c in enumerate(chunks)],
           "chunk_of_log": chunk_of_log,
           "provenance": {"export": EXP, "export_sha256": sha(EXP), "c3_tokens": C3 + "/tokens.json", "c3_tokens_sha256": sha(C3 + "/tokens.json"),
                          "split": YSEL + "/split.json", "split_sha256": sha(YSEL + "/split.json"), "banked_csv": BANKED_CSV, "banked_csv_sha256": sha(BANKED_CSV),
                          "n_tokens": len(toks), "n_logs": len(by), "n_logs_heldout": sum(1 for v in log_class.values() if v == "heldout"),
                          "n_logs_new": sum(1 for v in log_class.values() if v == "new"),
                          "n_tokens_heldout_logs": sum(1 for t in toks if log_class[token_log[t]] == "heldout"),
                          "n_tokens_new_logs": sum(1 for t in toks if log_class[token_log[t]] == "new"),
                          "n_chunks": len(chunks), "chunk_max": CHUNK_MAX, "excluded_seen_before_freeze": EXCLUDE_SEEN}}
    dump_lf(doc, out_p)
    # ---- smoke: 6 C3 tokens, fixed rule. Log A = the sim token's log: the sim token + the 2 lexicographically first other C3 tokens of that log.
    # Log B = the held-out log with the most C3 tokens (ties -> smallest name): its first zeroed token, its first random token, and its first
    # token whose frozen-rule pick differs from the banked pick (else its next token).
    import sel_common as sc
    import selc_rule as R
    D = sc.load()
    pr = R.rule_vec(D["logits"], D["v4_all"], D["v0"], D["ax0"])
    tl = {str(t): str(l) for t, l in zip(D["token"], D["log"])}
    idx = {str(t): i for i, t in enumerate(D["token"])}
    la = tl[SIM_TOKEN]
    A = [SIM_TOKEN] + [t for t in sorted(t for t in tl if tl[t] == la) if t != SIM_TOKEN][:2]
    ho_counts = {}
    for t, lg in tl.items():
        if lg in hol:
            ho_counts[lg] = ho_counts.get(lg, 0) + 1
    lb = sorted(ho_counts, key=lambda lg: (-ho_counts[lg], lg))[0]
    tb = sorted(t for t in tl if tl[t] == lb)
    z = [t for t in tb if D["stratum"][idx[t]] == "zero"][:1]
    rr = [t for t in tb if D["stratum"][idx[t]] == "random" and t not in z][:1]
    ch = [t for t in tb if pr[idx[t]] != D["pick"][idx[t]] and t not in z + rr][:1]
    B = z + rr + ch
    for t in tb:
        if len(B) >= 3:
            break
        if t not in B:
            B.append(t)
    sm = sorted(A + B)
    sdoc = {"rule": "smoke: 6 C3 tokens (make_confirm_tokens.py docstring)", "tokens": sm, "token_log": {t: tl[t] for t in sm},
            "stratum": {t: str(D["stratum"][idx[t]]) for t in sm}, "rule_changes_pick": {t: bool(pr[idx[t]] != D["pick"][idx[t]]) for t in sm}}
    dump_lf(sdoc, smoke_p)
    print(json.dumps(doc["provenance"], indent=1))
    print("chunks", [c["n_tokens"] for c in doc["chunks"]])
    print("smoke", json.dumps(sdoc, indent=1))
    print("tokens_confirm.json sha256", sha(out_p))
    print("tokens_smoke.json sha256", sha(smoke_p))


if __name__ == "__main__":
    main()
