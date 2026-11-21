#!/usr/bin/env python3
"""build_e2_table.py -- the E2 truth table of PREREG_SCORER1P6 section 3: the true per-proposal NAVSIM v1.1 scores of P6's own 64 proposals on the 2,000 E2 tokens.
CPU only (CUDA_VISIBLE_DEVICES=-1), resumable, RAM-guarded. It REUSES selc_score.py (seam writer, harness caller, resumable job runner) exactly as selc6_score.py does,
with its own work dir and label prefix `refe_p6e2`, so no other run's csv can be re-read as E2.

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe eval/build_e2_table.py \
        --proposals F:/TanitAD_p6read_work/pass2/proposals_p6final.npz --tokens ../e2/e2_tokens.json --out ../e2 [--work F:/TanitAD_p6eval/e2work] [--workers 3]

  Input  : P6's second-pass proposal dump (written by refe_navtest_seam.py --dump-proposals: token, fingerprint, proposals [N,64,8,3] executed plans, logits, pick, ...).
  Work   : 2,000 tokens in chunks of whole logs (<= 1,000 tokens), 64 proposal columns + 1 RED column (the pick poses shifted +2.0 m in y) per chunk = 130 harness calls,
           ESTIMATED 2.5 h CPU (PREREG section 7). A call whose log says PASS with n valid rows is not redone (re-run the same command to resume).
  RAM    : before every harness call it waits while free RAM < 6 GB (selc_score.wait_resources, MIN_RAM_GIB = 6.0), and while a drive in C:, E:, F: has < 3 GB.
  Output : <out>/e2_table_p6final.npz   token, pdms [N,64], sub [N,64,6], valid [N,64], red_pick_pdms [N], sub_names
           <out>/e2_g0.json             gate G0-E2 (the table reproduces P6's picks and per-token PDMS/sub-scores of refe_p6final.csv on 100 % of the tokens; RED shifted column differs)
  Exit   : 0 table built and G0-E2 PASS; 2 harness calls failed; 3 proposals file absent; 4 G0-E2 FAILED (the table is written, the read tool refuses it).
Do not print or inspect the table's statistics: only G0-E2's equalities are read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SELC = "E:/Projects/TanitAD-artifacts/refe_selection_confirm_20261107"
sys.path.insert(0, SELC)
import selc_score as S  # noqa: E402  (reused unchanged; only its work dir, prefix and RAM floor are set below)
import read_scorer1p6 as R  # noqa: E402

DEFAULT_PROPOSALS = "F:/TanitAD_p6read_work/pass2/proposals_p6final.npz"
S.PREFIX = "refe_p6e2"
S.MIN_RAM_GIB = 6.0
M = 64


def chunk_doc(tokens, token_log, max_tokens=1000):
    by = {}
    for t in tokens:
        by.setdefault(token_log[t], []).append(t)
    chunks, cur, n = [], [], 0
    for lg in sorted(by):
        if cur and n + len(by[lg]) > max_tokens:
            chunks.append({"chunk": len(chunks), "logs": cur, "n_tokens": n})
            cur, n = [], 0
        cur.append(lg)
        n += len(by[lg])
    chunks.append({"chunk": len(chunks), "logs": cur, "n_tokens": n})
    return {"tokens": list(tokens), "token_log": dict(token_log), "chunks": chunks}


def load_e2(proposals_npz, tokens_json):
    d = json.load(open(tokens_json, encoding="utf-8"))
    Z = np.load(proposals_npz, allow_pickle=False)
    key = "proposals" if "proposals" in Z.files else "props8"
    idx = {str(t): i for i, t in enumerate(Z["token"])}
    miss = [t for t in d["tokens"] if t not in idx]
    if miss:
        raise SystemExit(f"REFUSING: {len(miss)} E2 tokens absent from {proposals_npz} (e.g. {miss[:3]})")
    ix = np.array([idx[t] for t in d["tokens"]])
    P = Z[key][ix]
    if P.shape[1:] != (M, 8, 3):
        raise SystemExit(f"REFUSING: proposals shape {P.shape}, expected [N, 64, 8, 3]")
    return d, dict(token=np.array(d["tokens"]), fingerprint=Z["fingerprint"][ix], props=P, logits=Z["logits"][ix],
                   pick=(Z["pick"][ix].astype(np.int64) if "pick" in Z.files else None))


def build(a, score_one=None):
    if score_one is not None:                      # tests inject a fake harness; production never does
        S.score_one = score_one
    os.makedirs(a.out, exist_ok=True)
    S.WORK = a.work
    logp = os.path.join(a.out, "e2_build.log")
    if not os.path.exists(a.proposals):
        S.log_to(logp, f"proposals file absent: {a.proposals}")
        return 3
    doc, D = load_e2(a.proposals, a.tokens)
    tok = [str(t) for t in D["token"]]
    doc = chunk_doc(tok, doc["token_log"])
    cf = S.chunk_files(doc)
    idx = {t: i for i, t in enumerate(tok)}
    pick = R.pick_logits(D["logits"])
    jobs = []
    ncol = a.columns or M
    for c, ctoks, tj in cf:
        I = [idx[t] for t in ctoks]
        for k in range(ncol):
            lab = f"{S.PREFIX}_c{c}_p{k:02d}"
            sp = f"{S.WORK}/seams/table/{lab}.npz"
            S.write_seam(sp, D["token"][I], D["fingerprint"][I], D["props"][I, k], lab)
            jobs.append((lab, sp, tj, len(I)))
        if not a.columns:
            red = D["props"][I, pick[I]].astype(np.float64).copy()
            red[..., 1] += 2.0
            lab = f"{S.PREFIX}_c{c}_red"
            sp = f"{S.WORK}/seams/table/{lab}.npz"
            S.write_seam(sp, D["token"][I], D["fingerprint"][I], red, lab)
            jobs.append((lab, sp, tj, len(I)))
    S.log_to(logp, f"E2 table: {len(jobs)} harness calls on {len(tok)} tokens, work {S.WORK}, prefix {S.PREFIX}, RAM floor {S.MIN_RAM_GIB} GiB")
    bad = S.run_jobs(jobs, a.workers, os.path.join(a.out, "logs_table"), logp)
    if bad:
        S.log_to(logp, f"{len(bad)} calls failed: {bad[:10]}")
        return 2
    if a.columns:
        S.log_to(logp, f"smoke: {len(jobs)} calls PASS; no table assembled (--columns {a.columns})")
        return 0
    N = len(tok)
    pdms, sub, valid, redp = np.full((N, M), np.nan), np.full((N, M, 6), np.nan), np.zeros((N, M), bool), np.full(N, np.nan)
    for c, ctoks, tj in cf:
        for k in range(M):
            rows = S.read_csv(S.csv_of(f"{S.PREFIX}_c{c}_p{k:02d}"))
            for t in ctoks:
                v, s, cc = rows[t]
                pdms[idx[t], k], sub[idx[t], k], valid[idx[t], k] = s, cc, v
        rows = S.read_csv(S.csv_of(f"{S.PREFIX}_c{c}_red"))
        for t in ctoks:
            redp[idx[t]] = rows[t][1]
    table = dict(token=np.array(tok), pdms=pdms, sub=sub, valid=valid, red=redp)
    p6_rows = R.read_harness_csv(a.csv)

    class _W:                                      # the gate only needs token -> row position
        index = idx

    g = R.gate_e2(table, tok, _W, p6_rows, D["logits"], pick)
    if D["pick"] is not None:
        g["n_pick_equal_to_dump_pick"] = int((pick == D["pick"]).sum())
        g["PASS"] = bool(g["PASS"] and g["n_pick_equal_to_dump_pick"] == N)
    json.dump(g, open(os.path.join(a.out, "e2_g0.json"), "w", encoding="utf-8"), indent=1, default=R.jdefault)
    np.savez_compressed(os.path.join(a.out, "e2_table_p6final.npz"), token=np.array(tok), pdms=pdms, sub=sub, valid=valid, red_pick_pdms=redp, sub_names=np.array(S.SUB))
    S.log_to(logp, f"table written; G0-E2 {'PASS' if g['PASS'] else 'FAIL'}")
    return 0 if g["PASS"] else 4


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default=DEFAULT_PROPOSALS)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--csv", default=R.DEFAULT_P6_CSV)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="F:/TanitAD_p6eval/e2work")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--columns", type=int, default=0, help="smoke only: run the first K proposal columns, build no table")
    a = ap.parse_args(argv)
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
