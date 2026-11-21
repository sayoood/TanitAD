#!/usr/bin/env python3
"""verify_landing_package.py -- check a LANDING_READY file against the package it points into, WITHOUT building a tree:
every listed package file exists, its ``git hash-object --no-filters`` equals the declared "new" blob, its EOL equals
the declared EOL (and a CRLF file has no bare LF), and the declared base equals the tip's blob (absent for NEW).

usage: python verify_landing_package.py --git-dir <GIT_DIR> --tip <sha> --package <pkg> --landing <LANDING_*.txt>
Exit 0 and one summary line on success; exit 1 naming the first mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HEAD = re.compile(r"^# (?P<repo>\S+): base (?P<base>NEW|[0-9a-f]{40}) \| new (?P<new>[0-9a-f]{40}) \| "
                  r"(?P<eol>CRLF|LF)$")
MAP = re.compile(r"^(?P<src>.+?)  ->  (?P<repo>\S+)$")


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--landing", required=True)
    a = ap.parse_args()
    pkg = Path(a.package)
    lines = Path(a.landing).read_text(encoding="utf-8").splitlines()
    n = n_new = 0
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        if not m:
            continue
        mm = MAP.match(lines[i + 1]) if i + 1 < len(lines) else None
        if mm is None or mm["repo"] != m["repo"]:
            print(f"FAIL line {i + 1}: header without its mapping line")
            return 1
        p = pkg / mm["src"]
        if not p.is_file():
            print(f"FAIL {m['repo']}: package file missing: {p}")
            return 1
        data = p.read_bytes()
        if blob(data) != m["new"]:
            print(f"FAIL {m['repo']}: package blob {blob(data)} != declared {m['new']}")
            return 1
        crlf, lf = data.count(b"\r\n"), data.count(b"\n")
        eol = "CRLF" if crlf else "LF"
        if eol != m["eol"] or (crlf and crlf != lf):
            print(f"FAIL {m['repo']}: EOL {eol} (crlf {crlf} / lf {lf}) != declared {m['eol']}")
            return 1
        r = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", "--verify", "-q", f"{a.tip}:{m['repo']}"],
                           capture_output=True, text=True)
        tip_blob = r.stdout.strip() if r.returncode == 0 else None
        if m["base"] == "NEW":
            if tip_blob is not None:
                print(f"FAIL {m['repo']}: declared NEW but present on the tip ({tip_blob})")
                return 1
            n_new += 1
        elif tip_blob != m["base"]:
            print(f"FAIL {m['repo']}: base {m['base']} != tip blob {tip_blob}")
            return 1
        n += 1
    if not n:
        print("FAIL: no file pairs parsed")
        return 1
    print(f"OK {Path(a.landing).name}: {n} files ({n - n_new} shared, {n_new} new) verified against tip {a.tip[:12]} "
          f"and package {pkg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
