#!/usr/bin/env python3
"""COPY this package's deliverables from the development tree (ev6) into the D: package path, verify
every copy by CONTENT, and — only when asked — declare a batch LANDING-READY.

⛔ NO GIT. Superseded 2026-09-26 by the Master Mind (PI: all work is committed and pushed, and the
commit/push is COORDINATED): the Master Mind is the single committer and lands through its guarded
lander (scratch index seeded from the branch tip, leak scan, compare-and-swap push); D:'s shared
index holds ~2,800 stale entries and landing does not use it. So this tool never runs `git`.

    python code/stage_to_repo.py                         # copy everything allow-listed, content-verified
    python code/stage_to_repo.py --reconcile             # ev6 vs D: by CONTENT, copy nothing
    python code/stage_to_repo.py --only 'raw/milestones/step5000/*.json' \
        --landing "2026-09-26 step-5000 NavSim reading, verified"   # copy + append to LANDING_READY.txt

JSON/JSONL over --gzip-mb are copied as ``.gz`` (deterministic bytes, mtime 0); the verification
decompresses and compares the sha256 of the SOURCE bytes. A ``--landing`` batch is refused per file
when the file is over 20 MB or carries a canonical UUID (the devkit scorer logs carry random
``thread_id`` UUIDs — not clip ids, but exactly what the lander's leak scan counts); such files stay
LOCAL and are named in the output, never silently dropped.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = "D:/Projects/TanitAD"
REL = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/navsim"
ALLOW = ["*.md", "code/*.py", "code/*.sh", "tests/*.py", "raw/*.json", "raw/*.txt",
         "raw/*/*.json", "raw/*/*.txt", "raw/*/*.log", "raw/*/*.csv", "raw/*/*.jsonl",
         "raw/*/*.npz", "raw/*/*_wrapper/*.json", "raw/*/*_wrapper/*.csv",
         "raw/*/*/*.json", "raw/*/*/*.csv", "raw/*/*/*.log", "raw/*/*/*.jsonl",
         "raw/*/*/*/*.json", "raw/*/*/*/*.csv", "raw/*/*/*/*.jsonl"]
DENY = ["raw/floors/*/score_*.csv", "raw/floors/*/*_wrapper/*", "*PRE_AGGREGATION*",
        "*__pycache__*", "*.pyc", "LANDING_READY.txt"]
MAX_LAND_BYTES = 20 * 2**20
UUID_RE = re.compile(rb"(?<![0-9a-fA-F])[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     rb"[0-9a-f]{12}(?![0-9a-fA-F])")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def collect(gzip_mb: float) -> list:
    out = []
    for root, _dirs, files in os.walk(PKG):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, PKG).replace("\\", "/")
            if not any(fnmatch.fnmatch(rel, p) for p in ALLOW):
                continue
            if any(fnmatch.fnmatch(rel, p) for p in DENY):
                continue
            big = rel.endswith((".json", ".jsonl")) and os.path.getsize(src) > gzip_mb * 2**20
            out.append((src, rel + (".gz" if big else ""), big))
    return sorted(out, key=lambda x: x[1])


def dst_content_sha(dst: str, big: bool) -> str:
    """sha256 of the SOURCE-equivalent bytes held at dst ('' if absent/unreadable)."""
    try:
        b = open(dst, "rb").read()
        return sha(gzip.decompress(b) if big else b)
    except Exception:                                                     # noqa: BLE001
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reconcile", action="store_true", help="compare ev6 vs D: by content; copy nothing")
    ap.add_argument("--gzip-mb", type=float, default=5.0)
    ap.add_argument("--only", nargs="*", default=[], help="restrict to these relative globs")
    ap.add_argument("--landing", default="", help="one-line heading: append the batch to LANDING_READY.txt")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    items = collect(a.gzip_mb)
    if a.only:
        items = [it for it in items if any(fnmatch.fnmatch(it[1].removesuffix(".gz"), g)
                                           for g in a.only)]
    dst_root = os.path.join(REPO, REL)
    if a.reconcile:
        same, differ, missing = 0, [], []
        for src, rel, big in items:
            s = sha(open(src, "rb").read())
            d = dst_content_sha(os.path.join(dst_root, rel), big)
            if not d:
                missing.append(rel)
            elif d == s:
                same += 1
            else:
                differ.append(rel)
        rep = {"when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n": len(items),
               "same": same, "differ": differ, "missing_on_D": missing}
        print(json.dumps({"n": rep["n"], "same": same, "n_differ": len(differ),
                          "n_missing_on_D": len(missing)}))
        for r in (differ + missing)[:40]:
            print(("DIFFER  " if r in differ else "MISSING ") + r)
        with open(os.path.join(PKG, "raw", "RECONCILE_RECORD.json"), "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
        return 0
    copied, verdict = [], {}
    for src, rel, big in items:
        dst = os.path.join(dst_root, rel)
        if a.dry_run:
            print(("GZIP " if big else "COPY ") + rel)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = open(src, "rb").read()
        if big:
            buf = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=buf, mtime=0) as fo:
                fo.write(data)                                     # deterministic bytes
            payload = buf.getvalue()
        else:
            payload = data
        if not (os.path.exists(dst) and open(dst, "rb").read() == payload):
            with open(dst, "wb") as fh:
                fh.write(payload)
        ok = dst_content_sha(dst, big) == sha(data)
        verdict[f"{REL}/{rel}"] = "VERIFIED" if ok else "MISMATCH"
        copied.append((f"{REL}/{rel}", dst))
    if a.dry_run:
        print(f"{len(items)} files")
        return 0
    rep = {"when_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": "copy-only (no git)",
           "n": len(copied), "n_verified": sum(v == "VERIFIED" for v in verdict.values()), "paths": verdict}
    landed, local = [], []
    if a.landing:
        for relp, dst in copied:
            b = open(dst, "rb").read()
            if verdict[relp] != "VERIFIED":
                local.append((relp, "copy not verified"))
            elif len(b) > MAX_LAND_BYTES:
                local.append((relp, f"{len(b)} B > 20 MB"))
            elif UUID_RE.search(gzip.decompress(b) if dst.endswith(".gz") else b):
                local.append((relp, "carries canonical UUIDs (devkit thread ids) — kept local"))
            else:
                landed.append(relp)
        if landed:
            with open(os.path.join(dst_root, "LANDING_READY.txt"), "a", encoding="utf-8", newline="\n") as fh:
                fh.write(f"\n## {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} — {a.landing}\n")
                for relp in landed:
                    fh.write(relp + "\n")
        rep["landing"] = {"heading": a.landing, "n_listed": len(landed), "kept_local": local}
    with open(os.path.join(PKG, "raw", "STAGING_RECORD.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("when_utc", "mode", "n", "n_verified")}
                     | ({"n_listed": len(landed), "n_kept_local": len(local)} if a.landing else {})))
    for relp, why in local[:30]:
        print("KEPT LOCAL:", relp, "—", why)
    bad = [k for k, v in verdict.items() if v != "VERIFIED"]
    for k in bad[:20]:
        print("NOT VERIFIED:", k)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
