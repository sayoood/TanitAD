#!/usr/bin/env python3
"""write_repo_ready.py -- the Master Mind's REPO format of a landing list (2026-09-27):

    ## <heading>
    # <repo path>: base <tip blob or NEW> | new <blob> | <CRLF|LF>
    <REPO-relative source in this package>  ->  <repo path>
    ...
    <REPO-relative path of each of the package's OWN files>      (added at their own paths)

The code entries are read from a package-relative LANDING_READY file (verified first by ``verify_landing_package``'s
rules: base == tip blob / NEW absent, new == hash-object, EOL); the package's own files are every file under the
package EXCEPT the landed sources' copies (``code/fix*``, ``code/new``, ``code/hqs/new``), caches, and anything > 5 MB.
Refuses a raw UUID-shaped string in any listed package file (clip ids are sha12 only).

usage: python write_repo_ready.py --git-dir G --tip SHA --package PKG --repo-prefix "TanitAD Research Lab/..." \
                                  --landing LANDING_READY_HQS.txt --heading "..." --out LANDING_READY_ANCHORED.txt
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
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
EXCLUDE_PREFIXES = ("code/fix/", "code/fix_r6/", "code/fix_hqs/", "code/new/", "code/hqs/new/")
MAX_BYTES = 5 * 1024 * 1024


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--repo-prefix", required=True)
    ap.add_argument("--landing", required=True)
    ap.add_argument("--heading", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pkg, prefix = Path(a.package), a.repo_prefix.rstrip("/")
    lines = (pkg / a.landing).read_text(encoding="utf-8").splitlines()
    out = [f"## {a.heading}",
           f"# Paths are REPO-relative. Code entries: full files for tip {a.tip}; shared files keep the tip blob's line",
           "# endings; new files are LF. Below them: this package's own files, added at their own paths.", ""]
    n_code = 0
    for i, ln in enumerate(lines):
        m = HEAD.match(ln)
        if not m:
            continue
        mm = MAP.match(lines[i + 1])
        if mm is None or mm["repo"] != m["repo"]:
            raise SystemExit(f"line {i + 1}: header without its mapping")
        src = pkg / mm["src"]
        data = src.read_bytes()
        if blob(data) != m["new"]:
            raise SystemExit(f"{m['repo']}: package blob {blob(data)} != {m['new']}")
        r = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", "--verify", "-q", f"{a.tip}:{m['repo']}"],
                           capture_output=True, text=True)
        tip_blob = r.stdout.strip() if r.returncode == 0 else None
        if (m["base"] == "NEW") != (tip_blob is None) or (m["base"] != "NEW" and tip_blob != m["base"]):
            raise SystemExit(f"{m['repo']}: base {m['base']} vs tip {tip_blob}")
        out.append(ln)
        out.append(f"{prefix}/{mm['src']}  ->  {m['repo']}")
        n_code += 1
    out += ["", f"# ---- the package's own files ({prefix}/) ----"]
    own = []
    for p in sorted(pkg.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rel = p.relative_to(pkg).as_posix()
        if rel.startswith(EXCLUDE_PREFIXES) or p.stat().st_size > MAX_BYTES or rel == Path(a.out).name:
            continue
        if UUID.search(p.read_bytes()):
            raise SystemExit(f"{rel}: a raw UUID-shaped string -- clip ids must be sha12")
        own.append(f"{prefix}/{rel}")
    out += own
    (pkg / a.out).write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.out}: {n_code} code entries + {len(own)} package files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
