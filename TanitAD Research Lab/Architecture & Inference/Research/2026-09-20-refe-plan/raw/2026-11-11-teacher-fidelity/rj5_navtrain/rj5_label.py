#!/usr/bin/env python3
"""RJ5 labels: run the UNCHANGED dual labeller (Tc20s `iterate20`, then P5-R lambda 3.0; PK/raw/2026-11-11-p5c-readiness/a1/p5c_dual_labels.py)
over the RJ5 rank-0 rows, then merge its shards into one table keyed like PREREG_P5C 4.1: (log, token, rank, sha256 of the row's own traj).

  python rj5_label.py prep  --rj5 <rj5_rank0.jsonl.gz> --out-bank <dir> [--logs-file f]     writes <dir>/targets_rank0.jsonl + EMPTY targets_rank1.jsonl
  python rj5_label.py merge --rj5 <rj5_rank0.jsonl.gz> --labels-dir <dir with labels_s*of*.jsonl> --out <table.jsonl.gz> [--logs-file f]
Run the labeller itself as (P5R_BLAS_THREADS=1, all BLAS vars 1; DBs for the logs under P5C_DBROOT; one process per shard):
  P5C_BANKDIR=<out-bank> P5C_DBROOT=<db dir> P5C_PKG=<pk> P5R_DIR=<pk>/raw/2026-11-12-p5r  python p5c_dual_labels.py label --shard i --nshard N --out <labels-dir>
`merge` REFUSES unless every (log, token) of the (filtered) rj5 table has a label record with the matching traj sha, and counts raw_fallback rows.
A row whose RJ5 roll fell back to the bank's traj (source raw_fallback) is labelled from that traj and keeps `rj5_source: raw_fallback` in the table.
"""
import argparse
import glob
import gzip
import json
import os
import sys


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def logs_set(fp):
    return {x.strip() for x in open(fp, encoding="utf-8") if x.strip()} if fp else None


def cmd_prep(a):
    want = logs_set(a.logs_file)
    os.makedirs(a.out_bank, exist_ok=True)
    n = 0
    with open(os.path.join(a.out_bank, "targets_rank0.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in read_jsonl(a.rj5):
            if want is not None and r["log_name"] not in want:
                continue
            f.write(json.dumps({"log_name": r["log_name"], "token": r["token"], "rank": 0, "sim_hz": 10, "step": 0,
                                "traj": r["traj"]}) + "\n")
            n += 1
    open(os.path.join(a.out_bank, "targets_rank1.jsonl"), "w", encoding="utf-8").close()
    print(f"ZZLABEL_PREP rows={n}")
    return 0


def cmd_merge(a):
    want = logs_set(a.logs_file)
    rj5 = {}
    for r in read_jsonl(a.rj5):
        if want is None or r["log_name"] in want:
            rj5[(r["log_name"], r["token"])] = r
    lab = {}
    for fp in sorted(glob.glob(os.path.join(a.labels_dir, "labels_s*of*.jsonl"))):
        for r in read_jsonl(fp):
            if "tc20s" in r or "p5r" in r or "error" in r or "p5r_error" in r:
                lab[(r["log"], r["token"])] = r
    miss = [k for k in rj5 if k not in lab]
    bad_sha = [k for k in rj5 if k in lab and lab[k].get("traj_sha256") != rj5[k]["traj_sha256"]]
    n_fb = sum(1 for k in rj5 if k in lab and ("p5r" not in lab[k] or "tc20s" not in lab[k]))
    with gzip.GzipFile(a.out, "wb", mtime=0) as g:
        for k in sorted(rj5):
            if k in lab:
                o = dict(lab[k])
                o["rj5_source"] = rj5[k]["source"]
                g.write((json.dumps(o) + "\n").encode("utf-8"))
    rep = {"rows": len(rj5), "labelled_records": len(lab), "missing": len(miss), "traj_sha_mismatch": len(bad_sha),
           "label_fallbacks_tc20s_or_p5r": n_fb, "fallback_share": n_fb / max(1, len(rj5)),
           "rj5_roll_fallbacks": sum(1 for r in rj5.values() if r["source"] == "raw_fallback")}
    print(json.dumps(rep))
    ok = not miss and not bad_sha and rep["fallback_share"] <= 0.01
    print("ZZLABEL_MERGE_OK" if ok else "ZZLABEL_MERGE_REFUSED")
    return 0 if ok else 3


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--rj5", required=True)
    p.add_argument("--out-bank", required=True)
    p.add_argument("--logs-file", default=None)
    m = sp.add_parser("merge")
    m.add_argument("--rj5", required=True)
    m.add_argument("--labels-dir", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--logs-file", default=None)
    a = ap.parse_args(argv)
    return cmd_prep(a) if a.cmd == "prep" else cmd_merge(a)


if __name__ == "__main__":
    sys.exit(main())
