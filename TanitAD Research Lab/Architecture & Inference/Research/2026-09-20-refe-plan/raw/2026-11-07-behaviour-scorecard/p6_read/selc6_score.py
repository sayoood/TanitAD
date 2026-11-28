#!/usr/bin/env python3
"""selc6_score.py -- the CPU harness half of the P6 read (prereg section 3 stage 2): W3's NAVSIM v1.1 harness on all 64 proposal columns of the 1,058 C3 tokens
(+ one RED column per chunk: the dump's pick poses shifted +2.0 m in y, for G6-3). It REUSES selc_score.py's functions unchanged (imported) and overrides only
its work dir and label prefix, so no u400 CSV under F:/TanitAD_selc_20261107 or label `refe_selcf_*` can ever be re-read as a P6 result.

    python selc6_score.py --dump DIR/dump/dump_p6.npz --out DIR/score [--workers 3]       (CPU; CUDA_VISIBLE_DEVICES=-1)
Resumable: a harness call whose log says PASS with n valid rows is not redone. 2 chunks x (64 + 1) = 130 calls, ESTIMATED 1.3-1.8 h.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SELC = "E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107"
sys.path.insert(0, SELC)
import selc_score as S  # noqa: E402

TOKENS_ALL = "E:/Projects/TanitAD-artifacts/refe_ego_probe_20261106/tokens_all.json"
S.PREFIX = "refe_p6read"
M = 64


def chunk_doc(tokens_json):
    d = json.load(open(tokens_json, encoding="utf-8"))
    by = {}
    for t in d["tokens"]:
        by.setdefault(d["token_log"][t], []).append(t)
    chunks, cur, n = [], [], 0
    for lg in sorted(by):
        if cur and n + len(by[lg]) > 1000:
            chunks.append({"chunk": len(chunks), "logs": cur, "n_tokens": n})
            cur, n = [], 0
        cur.append(lg)
        n += len(by[lg])
    chunks.append({"chunk": len(chunks), "logs": cur, "n_tokens": n})
    return {"tokens": d["tokens"], "token_log": d["token_log"], "chunks": chunks}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default=TOKENS_ALL)
    ap.add_argument("--work", default="F:/TanitAD_p6read_work/score")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit-tokens", type=int, default=0, help="smoke only: score only the tokens present in the dump (chunks are rebuilt from them)")
    a = ap.parse_args(argv)
    S.WORK = a.work
    os.makedirs(a.out, exist_ok=True)
    logp = os.path.join(a.out, "score.log")
    D = np.load(a.dump, allow_pickle=False)
    tok = [str(t) for t in D["token"]]
    idx = {t: i for i, t in enumerate(tok)}
    doc = chunk_doc(a.tokens)
    if len(tok) != len(doc["tokens"]):                        # a partial (smoke) dump: score exactly its tokens
        keep = set(tok)
        doc["tokens"] = [t for t in doc["tokens"] if t in keep]
        doc["token_log"] = {t: doc["token_log"][t] for t in doc["tokens"]}
        by = {}
        for t in doc["tokens"]:
            by.setdefault(doc["token_log"][t], []).append(t)
        doc["chunks"] = [{"chunk": 0, "logs": sorted(by), "n_tokens": len(doc["tokens"])}]
    P, pick = D["props8"], D["pick"].astype(np.int64)
    cf = S.chunk_files(doc)
    jobs = []
    for c, ctoks, tj in cf:
        I = [idx[t] for t in ctoks]
        for k in range(M):
            lab = f"{S.PREFIX}_c{c}_p{k:02d}"
            sp = f"{S.WORK}/seams/table/{lab}.npz"
            S.write_seam(sp, D["token"][I], D["fingerprint"][I], P[I, k], lab)
            jobs.append((lab, sp, tj, len(I)))
        red = P[I, pick[I]].astype(np.float64).copy()
        red[..., 1] += 2.0
        lab = f"{S.PREFIX}_c{c}_red"
        sp = f"{S.WORK}/seams/table/{lab}.npz"
        S.write_seam(sp, D["token"][I], D["fingerprint"][I], red, lab)
        jobs.append((lab, sp, tj, len(I)))
    S.log_to(logp, f"p6 table: {len(jobs)} harness calls on {len(tok)} tokens, work {S.WORK}, prefix {S.PREFIX}")
    bad = S.run_jobs(jobs, a.workers, os.path.join(a.out, "logs_table"), logp)
    if bad:
        S.log_to(logp, f"{len(bad)} calls failed: {bad[:10]}")
        return 3
    N = len(tok)
    pdms, sub, valid = np.full((N, M), np.nan), np.full((N, M, 6), np.nan), np.zeros((N, M), bool)
    redp = np.full(N, np.nan)
    for c, ctoks, tj in cf:
        for k in range(M):
            rows = S.read_csv(S.csv_of(f"{S.PREFIX}_c{c}_p{k:02d}"))
            for t in ctoks:
                v, s, cc = rows[t]
                pdms[idx[t], k], sub[idx[t], k], valid[idx[t], k] = s, cc, v
        rows = S.read_csv(S.csv_of(f"{S.PREFIX}_c{c}_red"))
        for t in ctoks:
            redp[idx[t]] = rows[t][1]
    np.savez_compressed(os.path.join(a.out, "table_p6.npz"), token=np.array(tok), pdms=pdms, sub=sub, valid=valid, red_pick_pdms=redp, sub_names=np.array(S.SUB))
    ok = bool(valid.all() and np.isfinite(pdms).all())
    S.log_to(logp, f"table assembled: all valid {ok}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
