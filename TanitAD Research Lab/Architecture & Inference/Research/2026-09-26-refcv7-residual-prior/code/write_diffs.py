#!/usr/bin/env python3
"""Unified diffs of the shared-file edits for review: base = the RAW blob at <rev> (git show),
new = the file under <patched_dir>. EOL-normalised for reading; the full files keep the blob's EOL.

usage: python write_diffs.py <git_dir> <rev> <patched_dir> <out_dir> <repo_path> [<repo_path> ...]
"""
import difflib
import os
import subprocess
import sys

git_dir, rev, patched, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for path in sys.argv[5:]:
    base = subprocess.run(["git", f"--git-dir={git_dir}", "show", f"{rev}:{path}"],
                          check=True, capture_output=True).stdout
    new = open(os.path.join(patched, path), "rb").read()
    a = base.decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True)
    b = new.decode("utf-8").replace("\r\n", "\n").splitlines(keepends=True)
    d = list(difflib.unified_diff(a, b, f"a/{path}", f"b/{path}", n=3))
    name = path.replace("/", "_") + ".diff"
    with open(os.path.join(out, name), "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(d)
    print(f"{name}: +{sum(1 for l in d if l.startswith('+') and not l.startswith('+++'))} "
          f"-{sum(1 for l in d if l.startswith('-') and not l.startswith('---'))}")
