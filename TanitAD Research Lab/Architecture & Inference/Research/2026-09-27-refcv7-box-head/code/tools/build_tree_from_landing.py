#!/usr/bin/env python3
"""build_tree_from_landing.py -- apply a LANDING_READY file to a FRESH tree exactly as the committer would, and
verify every blob on the way (the landing's own check, run before it is handed over).

    python build_tree_from_landing.py --git-dir <GIT_DIR> --tip <sha> --package <pkg> --landing <LANDING_*.txt>
                                      --out <fresh dir>

1. ``git -c core.autocrlf=false archive <tip> -- <paths> | tar -x`` into --out (must NOT exist), streamed (the full
   tip is 6.6 GB: only ``stack`` and ``taniteval`` by default -- every landed path must lie under them);
2. for each ``# <repo>: base <b> | new <n> | <EOL>`` + ``<pkg path>  ->  <repo>`` pair: base == the tip's blob (or
   the path is absent for NEW), the package file's ``hash-object --no-filters`` == <n>, its EOL == <EOL>; copy it;
3. print a one-line summary; exit non-zero on the first mismatch (nothing is "skipped").
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAIR_HEAD = re.compile(r"^# (?P<repo>\S+): base (?P<base>NEW|[0-9a-f]{40}) \| new (?P<new>[0-9a-f]{40}) \| "
                       r"(?P<eol>CRLF|LF)$")
PAIR_MAP = re.compile(r"^(?P<src>.+?)  ->  (?P<repo>\S+)$")


def git(git_dir: str, *args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", f"--git-dir={git_dir}", "-c", "core.autocrlf=false", *args], **kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--landing", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--paths", nargs="+", default=["stack", "taniteval"])
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists -- the tree must be FRESH")
    out.mkdir(parents=True)
    g = subprocess.Popen(["git", f"--git-dir={a.git_dir}", "-c", "core.autocrlf=false", "archive", a.tip, "--",
                          *a.paths], stdout=subprocess.PIPE)
    t = subprocess.run(["tar", "-x", "-C", str(out)], stdin=g.stdout, check=False)
    g.stdout.close()
    if g.wait() != 0 or t.returncode != 0:
        raise SystemExit(f"archive | tar failed (git {g.returncode}, tar {t.returncode})")
    pkg = Path(a.package)
    lines = [ln.rstrip("\n") for ln in Path(a.landing).read_text(encoding="utf-8").splitlines()]
    pairs = []
    for i, ln in enumerate(lines):
        m = PAIR_HEAD.match(ln)
        if not m:
            continue
        nxt = PAIR_MAP.match(lines[i + 1]) if i + 1 < len(lines) else None
        if nxt is None or nxt["repo"] != m["repo"]:
            raise SystemExit(f"line {i + 1}: a header without its '<pkg path>  ->  {m['repo']}' line")
        pairs.append((m["repo"], m["base"], m["new"], m["eol"], nxt["src"]))
    if not pairs:
        raise SystemExit("no file pairs parsed")
    n_shared = n_new = 0
    for repo, base, new, eol, src in pairs:
        if not any(repo == q or repo.startswith(q.rstrip("/") + "/") for q in a.paths):
            raise SystemExit(f"{repo} lies outside the archived paths {a.paths}")
        p = pkg / src
        tip_blob = git(a.git_dir, "rev-parse", "--verify", "-q", f"{a.tip}:{repo}", capture_output=True, text=True)
        if base == "NEW":
            if tip_blob.returncode == 0:
                raise SystemExit(f"{repo}: declared NEW but EXISTS on the tip")
            n_new += 1
        else:
            if tip_blob.stdout.strip() != base:
                raise SystemExit(f"{repo}: base {base} != tip blob {tip_blob.stdout.strip()}")
            n_shared += 1
        got = subprocess.run(["git", "hash-object", "--no-filters", str(p)], capture_output=True, text=True,
                             check=True).stdout.strip()
        if got != new:
            raise SystemExit(f"{repo}: package file blob {got} != declared {new}")
        b = p.read_bytes()
        got_eol = "CRLF" if b.count(b"\r\n") else "LF"
        if got_eol != eol or (eol == "CRLF" and b.count(b"\n") != b.count(b"\r\n")):
            raise SystemExit(f"{repo}: EOL {got_eol} (mixed?) != declared {eol}")
        dst = out / repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    print(f"applied {len(pairs)} files ({n_shared} shared, {n_new} new) from {Path(a.landing).name} on "
          f"{a.tip[:12]} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
