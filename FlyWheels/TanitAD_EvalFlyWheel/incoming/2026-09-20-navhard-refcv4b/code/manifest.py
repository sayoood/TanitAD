#!/usr/bin/env python3
"""The DELIVERABLE MANIFEST — every artifact, where it lives, its size, and a sha256 for anything
that stays off-repo.

⛔ The operating standard requires this table and requires it to mark anything that exists in only
ONE place: the programme's dominant failure mode was never bad work, it was good work stranded
outside git. A bank of 11 GB cannot go in the repo, so the repo carries its **sha256 + the command
that rebuilds it** — a URL is a claim about a disk, a hash is a claim about bytes.

    python manifest.py --run <run dir> --bank <frame bank> --pkg <package dir> --out MANIFEST.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

BIG = 20 * 1024 * 1024


def sha256_file(p: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dir_stat(root: str, *, hash_big: bool = False) -> dict:
    n, total, big = 0, 0, []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                s = os.path.getsize(p)
            except OSError:
                continue
            n += 1
            total += s
            if s > BIG:
                big.append({"path": os.path.relpath(p, root).replace(os.sep, "/"), "bytes": s,
                            **({"sha256": sha256_file(p)} if hash_big else {})})
    return {"n_files": n, "bytes": total, "gb": round(total / 2 ** 30, 3),
            "files_over_20MB": sorted(big, key=lambda x: -x["bytes"])[:40]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hash-big", action="store_true")
    a = ap.parse_args(argv)
    man = {"_what": "every artifact this package produced and WHERE it lives",
           "utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
           "entries": []}

    def add(name, where, path, in_repo, note="", stat=None):
        e = {"artifact": name, "where": where, "path": path.replace(os.sep, "/"),
             "in_repo": in_repo, "note": note}
        if stat:
            e.update(stat)
        man["entries"].append(e)

    add("bench run directory (bench_run.json, summary.json, scores/, artifacts/, criteria/, "
        "plans/, report/, raw/)", "repo", os.path.relpath(a.run, "D:/Projects/TanitAD"), True,
        "the suite writes >20 MB files off-repo itself with a .offrepo.json sidecar",
        dir_stat(a.run))
    b = dir_stat(a.bank)
    prov = os.path.join(a.bank, "frames_provenance.parquet")
    build = os.path.join(a.bank, "BUILD.json")
    add("navhard stage-2 FRAME BANK (5,462 scenes)", "dev box (OFF-REPO — too large)", a.bank, False,
        "rebuild: code/build_frames.py --split navhard_two_stage --stage 2 --data-root "
        "C:/Users/Admin/navsim-crun/data/openscene --out <bank>; then code/finalize_bank.py. "
        "Its per-scene sha256[:16] index IS in the repo as raw/BUILD.json's corpus_id + the bank's "
        "frames_provenance.parquet (hashed below).", b)
    for p, nm in ((prov, "frames_provenance.parquet (per-scene sha256[:16])"),
                  (build, "BUILD.json (frame tag + content assertion)")):
        if os.path.exists(p):
            add(nm, "dev box", p, False, f"sha256 {sha256_file(p)} · {os.path.getsize(p)} B")
    add("W7 package (RESULT.md, code/, raw/)", "repo",
        os.path.relpath(a.pkg, "D:/Projects/TanitAD"), True, "", dir_stat(a.pkg))
    add("refcv4b checkpoint", "dev box (OFF-REPO, 1.29 GB)",
        "D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt", False,
        "md5 99b573e8277d94a5e3bfbf630cb4d751 (MODEL_REGISTRY.md §4.6); the run refuses a mismatch")
    add("devkit runtime (Windows-patched navsim @0a380a9)", "dev box", "C:/Users/Admin/navsim-crun",
        False, "devkit-side wrapper + blobs are recorded per arm in raw/<arm>/<arm>.counts.json")
    add("NavSim metric cache (navhard_two_stage)", "dev box",
        "C:/Users/Admin/navsim-crun/exp/metric_cache_navhard_two_stage", False,
        "5,912 tokens, verified by counts at preflight — NOT rebuilt (5,187 s)")
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps({"entries": [{k: e[k] for k in ("artifact", "where", "in_repo")
                                   if k in e} for e in man["entries"]]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
