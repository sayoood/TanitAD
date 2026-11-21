#!/usr/bin/env python3
"""Merge RJ5 roll shards into ONE bank-keyed table and a report. Pure python (no numpy, no nuplan): runs anywhere.

  python rj5_merge.py --bank bank_p7_rank0_slim.jsonl.gz --rows-dir <dir with rj5_*.jsonl> [--rows-dir ...] --out-dir <dir> [--min-coverage 1.0]

Writes:
  rj5_rank0.jsonl.gz   one line per bank key (log_name, token, rank 0): traj = the RJ5 traj if an ok RJ5 row exists, else the bank's own traj with
                       source "raw_fallback" (counted); traj_sha256 (float64 LE bytes, the P5-C key), bank_traj_sha256, dmax_vs_bank, source
  rj5_merge_report.json  coverage, fallbacks, duplicates (identical rows only; conflicting duplicates REFUSE), R0 identity control, RJ5 movement,
                       per-input-file sha256, the output's sha256
Exit 0 only if coverage >= --min-coverage AND no conflicting duplicate. The control is READ here, never gated: its bar belongs to the PLAN.
"""
import argparse
import glob
import gzip
import hashlib
import json
import math
import os
import struct
import sys


def traj_sha(traj):
    h = hashlib.sha256()
    for p in traj:
        for v in p:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue                      # torn last line of a killed shard


def dmax(a, b):
    return max(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb))


def dxy(a, b):
    return max(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa, pb in zip(a, b))


def quant(v, q):
    if not v:
        return None
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def merge(bank_path, row_files, min_coverage=1.0):
    bank = {}
    for r in read_jsonl(bank_path):
        if int(r.get("rank", 0)) == 0:
            bank[(r["log_name"], r["token"])] = r["traj"]
    rj5, r0 = {}, {}
    n_err = n_dup = 0
    conflicts = []
    errs = {}
    for fp in row_files:
        for r in read_jsonl(fp):
            k = (r["log_name"], r["token"])
            if "traj" not in r:
                n_err += 1
                errs[r.get("error", "?")[:60]] = errs.get(r.get("error", "?")[:60], 0) + 1
                continue
            tgt = rj5 if r["arm"] == "RJ5" else r0 if r["arm"] == "R0" else None
            if tgt is None:
                continue
            if k in tgt:
                n_dup += 1
                if tgt[k]["traj"] != r["traj"]:
                    conflicts.append(k)
                continue
            tgt[k] = r
    out_rows, fallbacks, moved, mv_dxy = [], 0, 0, []
    for k, bt in sorted(bank.items()):
        r = rj5.get(k)
        if r is not None and k in bank:
            traj, src = r["traj"], "rj5"
            d = dxy(traj, bt)
            mv_dxy.append(d)
            moved += d > 1e-6
        else:
            traj, src, fallbacks = bt, "raw_fallback", fallbacks + 1
        out_rows.append({"log_name": k[0], "token": k[1], "rank": 0, "traj": traj, "traj_sha256": traj_sha(traj),
                         "bank_traj_sha256": traj_sha(bt), "dmax_vs_bank": dmax(traj, bt), "source": src})
    cov = (len(bank) - fallbacks) / max(1, len(bank))
    ctl = [r["bank_dmax"] for k, r in r0.items() if "bank_dmax" in r and k in bank]
    ctl_dxy = [dxy(r["traj"], bank[k]) for k, r in r0.items() if k in bank]
    rep = {"n_bank_keys": len(bank), "n_rj5_ok": len(rj5), "n_fallback": fallbacks, "coverage": cov, "n_error_rows": n_err,
           "error_kinds": errs, "n_duplicate_rows": n_dup, "n_conflicting_duplicates": len(conflicts),
           "extra_keys_not_in_bank": sorted(set(rj5) - set(bank))[:5],
           "control_R0": {"n": len(ctl), "max_abs": max(ctl) if ctl else None, "max_dxy_m": max(ctl_dxy) if ctl_dxy else None,
                          "p50": quant(ctl, .5), "p99": quant(ctl, .99), "n_exact_zero": sum(1 for x in ctl if x == 0.0)},
           "rj5_movement": {"n": len(mv_dxy), "n_moved_gt_1e-6_m": moved, "share_moved": (moved / len(mv_dxy)) if mv_dxy else None,
                            "dxy_p50": quant(mv_dxy, .5), "dxy_p90": quant(mv_dxy, .9), "dxy_p99": quant(mv_dxy, .99),
                            "dxy_max": max(mv_dxy) if mv_dxy else None},
           "inputs": {os.path.basename(f): sha256_file(f) for f in row_files}}
    ok = cov >= min_coverage and not conflicts
    return out_rows, rep, ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--rows-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-coverage", type=float, default=1.0)
    a = ap.parse_args(argv)
    files = sorted(f for d in a.rows_dir for f in glob.glob(os.path.join(d, "rj5_*.jsonl")))
    if not files:
        print("ZZMERGE_REFUSE no rows files")
        return 2
    rows, rep, ok = merge(a.bank, files, a.min_coverage)
    os.makedirs(a.out_dir, exist_ok=True)
    outp = os.path.join(a.out_dir, "rj5_rank0.jsonl.gz")
    with gzip.GzipFile(outp, "wb", mtime=0) as g:         # mtime 0: the file's sha256 depends on content only
        for r in rows:
            g.write((json.dumps(r) + "\n").encode("utf-8"))
    rep["output_sha256"] = sha256_file(outp)
    rep["bank_sha256"] = sha256_file(a.bank)
    with open(os.path.join(a.out_dir, "rj5_merge_report.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in ("n_bank_keys", "n_rj5_ok", "n_fallback", "coverage", "n_conflicting_duplicates", "control_R0")}))
    print("ZZRJ5_MERGE_OK" if ok else "ZZRJ5_MERGE_REFUSED")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
