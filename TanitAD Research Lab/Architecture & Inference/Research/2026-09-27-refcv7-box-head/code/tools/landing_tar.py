#!/usr/bin/env python3
"""landing_tar.py -- a tar of a LANDING_READY file's files at their REPO paths (straight from the package, no tree),
plus the "<blob> <repo path>" list the run host verifies after extraction.

usage: python landing_tar.py <package> <LANDING_*.txt> <out.tar> <out_blobs.txt>
"""
import re
import sys
import tarfile
from pathlib import Path

HEAD = re.compile(r"^# (\S+): base (?:NEW|[0-9a-f]{40}) \| new ([0-9a-f]{40}) \| (?:CRLF|LF)$")
MAP = re.compile(r"^(.+?)  ->  (\S+)$")

pkg, landing, out_tar, out_blobs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
lines = landing.read_text(encoding="utf-8").splitlines()
pairs = []
for i, ln in enumerate(lines):
    m = HEAD.match(ln)
    if m:
        mm = MAP.match(lines[i + 1])
        assert mm and mm.group(2) == m.group(1), ln
        pairs.append((mm.group(1), m.group(1), m.group(2)))
with tarfile.open(out_tar, "w", format=tarfile.PAX_FORMAT) as tf:
    for src, repo, _ in pairs:
        tf.add(str(pkg / src), arcname=repo)
Path(out_blobs).write_text("".join(f"{b} {r}\n" for _, r, b in pairs), encoding="utf-8", newline="\n")
print(f"{out_tar}: {len(pairs)} files")
