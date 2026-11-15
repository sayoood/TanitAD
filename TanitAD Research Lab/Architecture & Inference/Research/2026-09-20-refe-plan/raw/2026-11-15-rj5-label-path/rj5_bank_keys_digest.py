#!/usr/bin/env python3
"""Per-log digest of the bank's rank-0 keys, to diff the dev-box bank copy against the POD's P7 bank (stdlib only; run on both, diff the outputs).
  python rj5_bank_keys_digest.py --bank <targets_rank0.jsonl[.gz] or a dir with targets_rank*.jsonl> --out <tsv> [--logs-file f]
Line = log_name <TAB> n_rank0_rows <TAB> sha256 of the sorted 'token|sha256(traj)' lines.  Rank-1 rows are ignored.  Last line: ALL <TAB> n <TAB> digest of the per-log lines.
(The P7 rank-0 set is the bank minus the 16 reserve sessions; pass --logs-file with the 1,075 P7 logs, one per line, to restrict, or compare only the logs both sides hold.)"""
import argparse, glob, hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rj5_bank_join as J

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank", required=True); ap.add_argument("--out", required=True); ap.add_argument("--logs-file")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.bank, "targets_rank0.jsonl"))) if os.path.isdir(a.bank) else [a.bank]
    want = {x.strip() for x in open(a.logs_file) if x.strip()} if a.logs_file else None
    per = {}
    for fp in files:
        for r in J.read_jsonl(fp):
            if int(r.get("rank", 0)) != 0 or (want is not None and r["log_name"] not in want):
                continue
            per.setdefault(r["log_name"], []).append(f"{r['token']}|{J.traj_sha256(r['traj'])}")
    lines, allh = [], hashlib.sha256()
    for lg in sorted(per):
        d = hashlib.sha256("\n".join(sorted(per[lg])).encode()).hexdigest()
        lines.append(f"{lg}\t{len(per[lg])}\t{d}"); allh.update((lines[-1] + "\n").encode())
    lines.append(f"ALL\t{sum(len(v) for v in per.values())}\t{allh.hexdigest()}")
    open(a.out, "w", newline="\n").write("\n".join(lines) + "\n"); print(lines[-1])
main()
