#!/usr/bin/env python
"""Rebase NEW-2 onto a new tip: a fresh candidate tree = the new tip's export + NEW-2.

* the new tip tree must be a RAW export (``git -c core.autocrlf=false archive``);
* batch-1 and new-dependent NEW-2 files are copied from the old candidate as authored;
* every EDITED shared file is 3-way merged (base = the old tip blob the edits were made
  on, ours = the old candidate, theirs = the new tip blob), all three normalised to LF
  first (a CRLF working copy against an LF blob is a fake total conflict), and written
  back in the NEW tip blob's own line-ending convention. A file the new tip did not
  change is taken as ours. Any conflict is written WITH markers and the script exits 2.

Usage: rebase_onto_tip.py --old-cand <tree> --old-base <ref> --new-tip-tree <export>
                          --new-ref <ref> --git-dir <.git> --out <fresh dir>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_fix_bundle import BATCH1, EDITED, NEWDEP  # noqa: E402


def blob(gd: str, ref: str, f: str) -> bytes | None:
    r = subprocess.run(["git", "--git-dir", gd, "cat-file", "blob", f"{ref}:{f}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--old-cand", "--old-base", "--new-tip-tree", "--new-ref", "--git-dir",
              "--out"):
        ap.add_argument(k, required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"--out {out} exists: use a FRESH directory name")
    shutil.copytree(a.new_tip_tree, out)
    old = Path(a.old_cand)
    for f in BATCH1 + NEWDEP:
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(old / f, out / f)
    n_conf = 0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in EDITED:
            base = blob(a.git_dir, a.old_base, f)
            theirs = blob(a.git_dir, a.new_ref, f)
            ours = (old / f).read_bytes()
            lf = [x.replace(b"\r\n", b"\n") for x in (base, ours, theirs)]
            crlf = b"\r\n" in theirs
            if lf[0] == lf[2]:
                res, how = lf[1], "tip unchanged -> ours"
            else:
                for name, data in zip(("base", "ours", "theirs"), lf):
                    (td / name).write_bytes(data)
                r = subprocess.run(["git", "merge-file", "-p", str(td / "ours"),
                                    str(td / "base"), str(td / "theirs")],
                                   capture_output=True)
                res = r.stdout
                if r.returncode < 0 or r.returncode > 127:
                    raise SystemExit(r.stderr.decode())
                how = f"3-way merged, {r.returncode} conflict(s)"
                n_conf += r.returncode
            if crlf:
                res = res.replace(b"\n", b"\r\n")
            (out / f).write_bytes(res)
            print(f"{how:34s} {'CRLF' if crlf else 'LF  '} {f}")
    print(f"rebased candidate: {out}  (conflicts: {n_conf})")
    return 0 if n_conf == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
