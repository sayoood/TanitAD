#!/usr/bin/env python3
"""Build the landing: full shared files on a named tip (via apply_box_head_edits.py) + the new files, and write
LANDING_READY.txt in the Master Mind's parsed format:

    # <repo path>: base <tip blob or NEW> | new <blob = git hash-object --no-filters> | <CRLF|LF>
    <package path of the full file>  ->  <repo path>

usage: python make_landing_ready.py --git-dir <GIT_DIR> --tip <sha> --package <package dir>
The +R6 patch (code/r6_banked/) is deliberately NOT listed (A10.1: banked unlanded).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def hash_object(path: Path) -> str:
    return subprocess.run(["git", "hash-object", "--no-filters", str(path)], check=True, capture_output=True,
                          text=True).stdout.strip()


def eol(path: Path) -> str:
    b = path.read_bytes()
    return "CRLF" if b.count(b"\r\n") else "LF"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--with-r6", action="store_true", help="also write LANDING_READY_R6.txt (the A10.1 variant)")
    ap.add_argument("--with-hqs", action="store_true", help="also write LANDING_READY_HQS.txt (the A14 variant)")
    a = ap.parse_args()
    pkg = Path(a.package)
    fix = pkg / "code" / "fix"
    gen = pkg / "code" / "apply_box_head_edits.py"
    r = subprocess.run([sys.executable, str(gen), "--base-git", a.git_dir, "--rev", a.tip, "--out", str(fix)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        raise SystemExit("the patch generator FAILED on this tip -- resolve the named anchor by reading")
    man = json.loads((fix / "EDIT_MANIFEST.json").read_text(encoding="utf-8"))
    lines = [f"# refcv7 A9/A10 box head -- landing on tip {a.tip}. Full files only; each shared file keeps its tip",
             "# blob's line endings; new files are LF. The +R6 patch (code/r6_banked/) is NOT part of this landing.", ""]
    n = 0
    for repo_path, info in man["files"].items():
        p = fix / repo_path
        base = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", f"{a.tip}:{repo_path}"], check=True,
                              capture_output=True, text=True).stdout.strip()
        if base != info["base_blob_sha1"]:
            raise SystemExit(f"{repo_path}: base blob {info['base_blob_sha1']} != tip {base}")
        lines.append(f"# {repo_path}: base {base} | new {hash_object(p)} | {eol(p)}")
        lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
        n += 1
    new_root = pkg / "code" / "new"
    for p in sorted(new_root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        repo_path = p.relative_to(new_root).as_posix()
        exists = subprocess.run(["git", f"--git-dir={a.git_dir}", "cat-file", "-e", f"{a.tip}:{repo_path}"],
                                capture_output=True).returncode == 0
        if exists:
            raise SystemExit(f"{repo_path} EXISTS on the tip -- a new file may not overwrite it")
        if b"\r\n" in p.read_bytes():
            raise SystemExit(f"{repo_path}: a new file must be LF")
        lines.append(f"# {repo_path}: base NEW | new {hash_object(p)} | {eol(p)}")
        lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
        n += 1
    (pkg / "LANDING_READY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"LANDING_READY.txt: {n} files on tip {a.tip}")
    if a.with_r6:
        write_r6(a, pkg, fix, man)
    if a.with_hqs:
        write_hqs(a, pkg, fix, man)
    return 0


def write_r6(a, pkg: Path, fix: Path, man: dict) -> None:
    """LANDING_READY_R6.txt -- the +R6 VARIANT (A10.1): the A9 landing with the R6 edits applied on top, the R6 new
    files added. Used ONLY if a SPEC amendment makes +R6 the launch configuration; otherwise it stays banked."""
    r6 = pkg / "code" / "r6_banked"
    fix_r6 = pkg / "code" / "fix_r6"
    # the R6 generator reads the A9-patched tree: stage one from the tip + code/fix + code/new
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="r6base_"))
    r6_only = ("stack/tests/test_cls_weight_stamp.py",)          # edited by R6 only: base = the TIP blob
    for repo_path in ("stack/tanitad/models/refcv6_perception_branch.py", "stack/scripts/refc_v3_train.py",
                      "stack/tanitad/train/declared_vs_built.py", "stack/tests/test_declared_vs_built.py",
                      "stack/tests/test_occ_knob_is_stamped.py", *r6_only):
        dst = tmp / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if repo_path in r6_only:
            dst.write_bytes(subprocess.run(["git", f"--git-dir={a.git_dir}", "show", f"{a.tip}:{repo_path}"],
                                           check=True, capture_output=True).stdout)
        else:
            shutil.copyfile(fix / repo_path, dst)
    r = subprocess.run([sys.executable, str(r6 / "apply_r6_edits.py"), "--base-dir", str(tmp), "--out", str(fix_r6)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        raise SystemExit("the R6 generator FAILED on the A9 files")
    r6_files = set(json.loads((fix_r6 / "R6_EDIT_MANIFEST.json").read_text(encoding="utf-8"))["files"])
    lines = [f"# refcv7 A9/A10 box head + R6 (A10.1) VARIANT -- landing on tip {a.tip}. Use ONLY if a SPEC amendment",
             "# makes +R6 the launch configuration. Full files; shared files keep the tip's line endings.", ""]
    n = 0
    for repo_path in man["files"]:
        p = (fix_r6 / repo_path) if repo_path in r6_files else (fix / repo_path)
        base = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", f"{a.tip}:{repo_path}"], check=True,
                              capture_output=True, text=True).stdout.strip()
        lines.append(f"# {repo_path}: base {base} | new {hash_object(p)} | {eol(p)}")
        lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
        n += 1
    for repo_path in sorted(r6_files - set(man["files"])):             # R6-only shared files (tip base)
        p = fix_r6 / repo_path
        base = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", f"{a.tip}:{repo_path}"], check=True,
                              capture_output=True, text=True).stdout.strip()
        lines.append(f"# {repo_path}: base {base} | new {hash_object(p)} | {eol(p)}")
        lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
        n += 1
    for root in (pkg / "code" / "new", r6 / "new"):
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            repo_path = p.relative_to(root).as_posix()
            lines.append(f"# {repo_path}: base NEW | new {hash_object(p)} | {eol(p)}")
            lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
            n += 1
    (pkg / "LANDING_READY_R6.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"LANDING_READY_R6.txt: {n} files on tip {a.tip}")


def write_hqs(a, pkg: Path, fix: Path, man: dict) -> None:
    """LANDING_READY_HQS.txt -- MAIN + HQS (SPEC_REFCV7 §19 A14): the A9 landing with the HQS edits applied on top
    (code/fix_hqs), plus the HQS new files (code/hqs/new). Gated as its own variant; the MAIN variant stays intact."""
    import shutil
    import tempfile
    hq = pkg / "code" / "hqs"
    fix_h = pkg / "code" / "fix_hqs"
    tmp = Path(tempfile.mkdtemp(prefix="hqsbase_"))
    for repo_path in ("stack/tanitad/models/refcv6_perception_branch.py", "stack/scripts/refc_v3_train.py",
                      "stack/tanitad/train/declared_vs_built.py", "stack/tests/test_declared_vs_built.py",
                      "stack/tests/test_occ_knob_is_stamped.py"):
        dst = tmp / repo_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fix / repo_path, dst)
    r = subprocess.run([sys.executable, str(hq / "apply_hqs_edits.py"), "--base-dir", str(tmp), "--out", str(fix_h)],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        raise SystemExit("the HQS generator FAILED on the A9 files")
    h_files = set(json.loads((fix_h / "HQS_EDIT_MANIFEST.json").read_text(encoding="utf-8"))["files"])
    lines = [f"# refcv7 A9/A10 box head + HQS (A14) VARIANT -- landing on tip {a.tip}. Gated as its own variant; the",
             "# MAIN variant (LANDING_READY.txt) stays intact. Full files; shared files keep the tip's line endings.", ""]
    n = 0
    for repo_path in man["files"]:
        p = (fix_h / repo_path) if repo_path in h_files else (fix / repo_path)
        base = subprocess.run(["git", f"--git-dir={a.git_dir}", "rev-parse", f"{a.tip}:{repo_path}"], check=True,
                              capture_output=True, text=True).stdout.strip()
        lines.append(f"# {repo_path}: base {base} | new {hash_object(p)} | {eol(p)}")
        lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
        n += 1
    for root in (pkg / "code" / "new", hq / "new"):
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            repo_path = p.relative_to(root).as_posix()
            if b"\r\n" in p.read_bytes():
                raise SystemExit(f"{repo_path}: a new file must be LF")
            lines.append(f"# {repo_path}: base NEW | new {hash_object(p)} | {eol(p)}")
            lines.append(f"{p.relative_to(pkg).as_posix()}  ->  {repo_path}")
            n += 1
    (pkg / "LANDING_READY_HQS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"LANDING_READY_HQS.txt: {n} files on tip {a.tip}")


if __name__ == "__main__":
    sys.exit(main())
