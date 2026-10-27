#!/usr/bin/env python3
"""PREREG_V8F_DAGGER section 8: the two helpers the box-2 queue's DAgger stages call. No model, no data, no GPU of its own.

    sha  FILE
         prints the sha256 (lower-case hex) of the file. The bank's identity is the sha256 of its ``bank.json`` (``DaggerBank.sha256``); the ``gatedag`` stage hashes the file ON BOX 2 with this and puts the value in
         the arm's argv as ``--dagger-bank-sha256``.

    bank --n-shards N --out-dir DIR [--bank-script PATH] -- <v8f_dagger_bank.py gen arguments, WITHOUT --shard / --n-shards / --out-dir>
         the clip-sharded bank run: N ``v8f_dagger_bank.py gen --shard K --n-shards N`` child processes at once (a shard whose ``.npz`` AND ``.provenance.json`` already exist is complete and is skipped; the provenance is
         written last), then ``v8f_dagger_bank.py merge --out-dir DIR``. Each child logs to DIR/gen_shard_K.log. Exit 0 only if every shard and the merge exited 0 and DIR/bank.json exists; on the first failing
         shard the others are terminated (a half bank is worth nothing) and the exit code is 1. Nothing is conditional on a number: this is plumbing.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORBIDDEN = ("--shard", "--n-shards", "--out-dir")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def shard_done(out: Path, k: int, n: int) -> bool:
    stem = out / f"shard_{k:03d}of{n:03d}"
    return Path(str(stem) + ".npz").exists() and Path(str(stem) + ".provenance.json").exists()


def cmd_bank(a, gen_args: list, log=print) -> int:
    bad = [x for x in gen_args if x in FORBIDDEN]
    if bad:
        log(f"REFUSED: {bad} are set by the tool (shards) or by --out-dir/--n-shards before the `--`, not in the gen arguments")
        return 2
    if a.n_shards < 1:
        log("REFUSED: --n-shards must be >= 1")
        return 2
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    script = str(Path(a.bank_script) if a.bank_script else HERE / "v8f_dagger_bank.py")
    if (out / "bank.json").exists() and (out / "PROVENANCE.json").exists():
        log(f"[skip] merged bank exists in {out}")
        return 0
    todo = [k for k in range(a.n_shards) if not shard_done(out, k, a.n_shards)]
    log(f"bank: {a.n_shards} shards, {len(todo)} to run ({[k for k in range(a.n_shards) if k not in todo]} already complete)")
    procs = {}
    for k in todo:
        cmd = [sys.executable, script, "gen", *gen_args, "--out-dir", str(out), "--shard", str(k), "--n-shards", str(a.n_shards)]
        procs[k] = (subprocess.Popen(cmd, stdout=open(out / f"gen_shard_{k}.log", "ab"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), cmd)
        log(f"shard {k} started pid {procs[k][0].pid}")
    failed = {}
    live = dict(procs)
    while live:
        for k, (p, _) in list(live.items()):
            rc = p.poll()
            if rc is None:
                continue
            del live[k]
            log(f"shard {k} rc={rc}")
            if rc != 0:
                failed[k] = rc
        if failed and live:                                         # a half bank is worth nothing: free the box
            for k, (p, _) in live.items():
                p.terminate()
            for k, (p, _) in live.items():
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                log(f"shard {k} terminated (shard {sorted(failed)} failed)")
            live = {}
        if live:
            time.sleep(a.poll_s)
    if failed:
        log(f"FAILED: shards {sorted(failed)} exited {failed}; no merge")
        return 1
    missing = [k for k in range(a.n_shards) if not shard_done(out, k, a.n_shards)]
    if missing:
        log(f"FAILED: shards {missing} exited 0 without their .npz / .provenance.json")
        return 1
    rc = subprocess.run([sys.executable, script, "merge", "--out-dir", str(out)], stdin=subprocess.DEVNULL, stdout=open(out / "merge.log", "ab"), stderr=subprocess.STDOUT).returncode
    log(f"merge rc={rc}")
    if rc != 0:
        return 1
    if not (out / "bank.json").exists():
        log("FAILED: merge exited 0 without bank.json")
        return 1
    log(f"bank sha256 {sha256_file(out / 'bank.json')}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    gen_args = []
    if "--" in argv:
        k = argv.index("--")
        argv, gen_args = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sha")
    s.add_argument("file")
    b = sub.add_parser("bank")
    b.add_argument("--n-shards", type=int, required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--bank-script", default="")
    b.add_argument("--poll-s", type=float, default=5.0)
    a = ap.parse_args(argv)
    if a.mode == "sha":
        if not os.path.isfile(a.file):
            print(f"REFUSED: {a.file} is not a file", file=sys.stderr)
            return 2
        print(sha256_file(a.file))
        return 0
    return cmd_bank(a, gen_args)


if __name__ == "__main__":
    sys.exit(main())
