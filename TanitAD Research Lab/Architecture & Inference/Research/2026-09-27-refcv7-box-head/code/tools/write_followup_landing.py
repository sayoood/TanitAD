#!/usr/bin/env python3
"""write_followup_landing.py -- a FOLLOW-UP landing of the box-head package on a tip that already carries it: every
changed repo file is a FULL file copied into ``<package>/code/<subdir>/<repo path>``; the list is written in the Master
Mind's repo format:

    ## <heading>
    # <repo path>: base <tip blob> | new <blob> | <CRLF|LF>
    <REPO-relative source>  ->  <repo path>
    ...
    <REPO-relative package file>            (only package files NEW or CHANGED vs the tip)

Each changed file must keep the tip blob's line endings (a CRLF/LF flip is refused). No raw UUID-shaped string may
appear in a listed package file. A landing's SOURCES are not package files: every file that stands left of
``  ->  `` in any ``LANDING_READY*.txt`` of the package lies under a SOURCE ROOT (the source path minus its repo
target, e.g. ``code/fix/``), and every file under a source root -- the full-file copies and their edit manifests --
is left out of the package-file list, as are the ``LANDING_READY*.txt`` files themselves: the committer's practice
on 28d8365, where neither was committed.

usage: python write_followup_landing.py --git-dir G --tip SHA --repo-root D:/Projects/TanitAD --package-rel REL
         --subdir a17 --src-root <dir holding the changed files at repo paths> --files <repo path> ... --heading H
         --out LANDING_READY_A17.txt
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def eol(data: bytes) -> str:
    return "CRLF" if b"\r\n" in data else "LF"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--package-rel", required=True)
    ap.add_argument("--subdir", required=True)
    ap.add_argument("--src-root", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--heading", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, rel = Path(a.repo_root), a.package_rel.rstrip("/")
    pkg = repo / rel
    code_dir = pkg / "code" / a.subdir

    def git(*args, **kw):
        return subprocess.run(["git", f"--git-dir={a.git_dir}", *args], capture_output=True, **kw)
    lines = [f"## {a.heading}",
             f"# Paths are REPO-relative. Full files for tip {a.tip}; each changed file keeps its tip blob's line",
             "# endings. Below: this package's NEW or CHANGED files (vs the tip), added at their own paths.", ""]
    for rp in a.files:
        src = Path(a.src_root) / rp
        data = src.read_bytes()
        r = git("show", f"{a.tip}:{rp}")
        if r.returncode != 0:
            raise SystemExit(f"{rp}: not on the tip -- a follow-up changes files the tip carries")
        base = git("rev-parse", f"{a.tip}:{rp}", text=True).stdout.strip()
        if eol(r.stdout) != eol(data) or (eol(data) == "CRLF" and data.count(b"\r\n") != data.count(b"\n")):
            raise SystemExit(f"{rp}: line endings {eol(data)} vs the tip's {eol(r.stdout)}")
        if blob(data) == base:
            raise SystemExit(f"{rp}: unchanged vs the tip -- not a follow-up file")
        dst = code_dir / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        lines.append(f"# {rp}: base {base} | new {blob(data)} | {eol(data)}")
        lines.append(f"{rel}/code/{a.subdir}/{rp}  ->  {rp}")
    roots = {f"{rel}/code/{a.subdir}/"}
    for lr in pkg.glob("LANDING_READY*.txt"):
        for ln in lr.read_text(encoding="utf-8").splitlines():
            if "  ->  " in ln and not ln.startswith("#"):
                src, target = ln.split("  ->  ", 1)
                src = src if src.startswith(rel + "/") else f"{rel}/{src}"
                if not src.endswith("/" + target.strip()):
                    raise SystemExit(f"{lr.name}: source {src!r} does not end in its target {target!r}")
                roots.add(src[: len(src) - len(target.strip())])
    lines += ["", f"# ---- this package's NEW or CHANGED files ({rel}/) ----"]
    n_pkg = n_src = 0
    for p in sorted(pkg.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        rrel = f"{rel}/{p.relative_to(pkg).as_posix()}"
        if p.name.startswith("LANDING_READY") or p.stat().st_size > 5 * 1024 * 1024:
            continue
        if any(rrel.startswith(r) for r in roots):
            n_src += 1
            continue
        data = p.read_bytes()
        r = git("rev-parse", "--verify", "-q", f"{a.tip}:{rrel}", text=True)
        if r.returncode == 0 and r.stdout.strip() == blob(data):
            continue
        if UUID.search(data):
            raise SystemExit(f"{rrel}: a raw UUID-shaped string -- clip ids must be sha12")
        lines.append(rrel)
        n_pkg += 1
    (pkg / a.out).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{a.out}: {len(a.files)} repo files + {n_pkg} new/changed package files on tip {a.tip[:12]} "
          f"({n_src} landing sources left out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
