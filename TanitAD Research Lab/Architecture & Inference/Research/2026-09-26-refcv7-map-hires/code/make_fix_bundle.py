#!/usr/bin/env python
"""Build ``code/fix/`` (every NEW-2 file at its repo path) + the LANDING_READY blocks.

* EDITED shared files: the FULL proposed file, written in the SAME line-ending convention
  as the file's CURRENT TIP BLOB (tip blobs are CRLF or LF -- a CRLF working copy must
  not leak into an LF blob), one unified diff against the RAW tip blob, then VERIFIED:
  the diff applied with ``git apply`` (autocrlf off) to a scratch tree of the raw tip
  blobs must reproduce every full file byte for byte, or the script exits 2;
* BATCH1 (independent new files) and NEWDEP (new files that need the shared edits):
  copied as authored (LF), base ``NEW`` (refused if the tip already has the path);
* ``code/fix/LANDING_BLOCKS.txt``: the Master Mind's Thor-gate format, per file
  ``# <repo path>: base <tip blob|NEW> | new <git hash-object --no-filters> | <EOL>``
  then ``<package path>  ->  <repo path>``.

Usage: make_fix_bundle.py --cand <tree> --pkg <package dir> --git-dir <.git> --ref <ref>
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
BATCH1 = [
    "stack/tanitad/data/semantic_map_gt_fine.py",
    "stack/tanitad/models/map_head_hires.py",
    "taniteval/taniteval/map_hires_metrics.py",
    "stack/scripts/compute_map_class_weights.py",
    "stack/tests/test_semantic_map_gt_fine.py",
    "stack/tests/test_map_head_hires.py",
    "stack/tests/test_compute_map_class_weights.py",
    "taniteval/tests/test_map_hires_metrics.py",
]
EDITED = [
    "stack/tanitad/models/timm_trunk.py",
    "stack/tanitad/refs/refc.py",
    "stack/tanitad/refs/refc_v3.py",
    "stack/scripts/refc_v3_train.py",
    "stack/tanitad/models/refcv6_perception_branch.py",
    "stack/tanitad/train/declared_vs_built.py",
    "taniteval/tools/refcv3_arm.py",
    "stack/tests/test_refc_v3_agent_provenance.py",
    "stack/tests/test_declared_vs_built.py",
    "stack/tests/test_occ_knob_is_stamped.py",
    "stack/tests/test_grad_probe_tacgoal.py",   # the rounding pin follows _train_row_scalars
]
#: NEW-2 R2 (SPEC_REFCV7 §17, A12: the 0.1 m near-range lift): every file is a TIP file by
#: then (NEW-2 landed as cef9709), so all are EDITED; `--set r2 --fix-dir code/fix_r2`.
R2_EDITED = [
    "stack/tanitad/models/map_head_hires.py",
    "stack/scripts/refc_v3_train.py",
    "taniteval/tools/refcv3_arm.py",
    "stack/scripts/map_hires_overfit.py",
    "stack/tests/test_map_head_hires.py",
    "stack/tests/test_map_hires_wiring.py",
    "stack/tests/test_map_hires_overfit.py",
    "stack/tests/test_declared_vs_built.py",
]
NEWDEP = [
    "stack/tests/test_map_hires_wiring.py",
    "taniteval/tests/test_map_hires_rebuild.py",
    "stack/scripts/map_hires_overfit.py",
    "stack/tests/test_map_hires_overfit.py",
]


def git(gd: str, *a) -> bytes:
    return subprocess.run(["git", "--git-dir", gd, *a], check=True,
                          capture_output=True).stdout


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, type=Path)
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--git-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--set", choices=("new2", "r2"), default="new2")
    ap.add_argument("--fix-dir", default="code/fix",
                    help="package-relative output dir (code/fix for NEW-2, code/fix_r2 for R2)")
    a = ap.parse_args(argv)
    global BATCH1, EDITED, NEWDEP
    if a.set == "r2":
        BATCH1, EDITED, NEWDEP = [], list(R2_EDITED), []
    fix = a.pkg / a.fix_dir
    fix_rel = Path(a.fix_dir).as_posix().strip("/")
    if fix.exists():
        assert fix.resolve().parts[-2] == "code" and fix.resolve().parts[-1].startswith("fix"), fix
        shutil.rmtree(fix)
    fix.mkdir(parents=True)
    tip = git(a.git_dir, "rev-parse", a.ref).decode().strip()
    rows, diffs = {}, []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for f in EDITED:
            bid = git(a.git_dir, "rev-parse", f"{a.ref}:{f}").decode().strip()
            raw = git(a.git_dir, "cat-file", "blob", bid)
            crlf = b"\r\n" in raw
            cand = (a.cand / f).read_bytes().replace(b"\r\n", b"\n")
            if crlf:
                cand = cand.replace(b"\n", b"\r\n")
            for root, data in ((td / "a", raw), (td / "b", cand), (fix, cand)):
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_bytes(data)
            rows[f] = (bid, blob_id(cand), "CRLF" if crlf else "LF")
            r = subprocess.run(["git", "-c", "core.autocrlf=false", "diff", "--no-index",
                                "--no-color", "--no-prefix", f"a/{f}", f"b/{f}"],
                               cwd=td, capture_output=True)
            if r.returncode not in (0, 1):
                raise SystemExit(r.stderr.decode())
            diffs.append(r.stdout)
        for f in BATCH1 + NEWDEP:
            r = subprocess.run(["git", "--git-dir", a.git_dir, "rev-parse", "--verify",
                                "-q", f"{a.ref}:{f}"], capture_output=True)
            if r.returncode == 0:
                raise SystemExit(f"{f} already exists at {a.ref}: not a NEW file any more "
                                 f"-- move it to EDITED")
            data = (a.cand / f).read_bytes()
            if b"\r\n" in data:
                raise SystemExit(f"{f} carries CRLF: new files are authored LF")
            (fix / f).parent.mkdir(parents=True, exist_ok=True)
            (fix / f).write_bytes(data)
            rows[f] = ("NEW", blob_id(data), "LF")
        dpath = fix / ("NEW2_shared_edits.diff" if a.set == "new2" else "NEW2R2_edits.diff")
        dpath.write_bytes(b"".join(diffs))
        # ---- verification: apply to the RAW tip blobs, compare byte for byte ----
        vt = td / "verify"
        for f in EDITED:
            (vt / f).parent.mkdir(parents=True, exist_ok=True)
            (vt / f).write_bytes((td / "a" / f).read_bytes())
        subprocess.run(["git", "init", "-q"], cwd=vt, check=True)
        r = subprocess.run(["git", "-c", "core.autocrlf=false", "apply",
                            "--whitespace=nowarn", str(dpath)], cwd=vt, capture_output=True)
        if r.returncode:
            print(r.stderr.decode(), file=sys.stderr)
            return 2
        bad = [f for f in EDITED if (vt / f).read_bytes() != (fix / f).read_bytes()]

    def block(files):
        out = []
        for f in files:
            b, n, eol = rows[f]
            out.append(f"# {f}: base {b} | new {n} | {eol}")
            out.append(f"{PKG_REL}/{fix_rel}/{f}  ->  {f}")
        return out
    base_lines = [f"# tip {tip} ({a.ref})"] + [
        f"{rows[f][0]}  {f}  ({rows[f][2]} blob)" for f in EDITED]
    (fix / "BASE_BLOBS.txt").write_bytes(("\n".join(base_lines) + "\n").encode("utf-8"))
    if a.set == "new2":
        blocks = (["# BATCH 1 (independent: new files only)"] + block(BATCH1)
                  + ["# BATCH 2 (shared-file edits + the new files that need them)"]
                  + block(EDITED + NEWDEP))
    else:
        blocks = ["# R2 (every file EDITED against the tip)"] + block(EDITED)
    (fix / "LANDING_BLOCKS.txt").write_bytes(("\n".join(blocks) + "\n").encode("utf-8"))
    n_lines = dpath.read_bytes().count(b"\n")
    print(f"tip {tip}: {len(BATCH1)} batch-1 + {len(EDITED)} edited + {len(NEWDEP)} "
          f"new-dependent files; diff {n_lines} lines; verify "
          f"{'MATCH' if not bad else 'MISMATCH ' + str(bad)}")
    return 0 if not bad else 2


if __name__ == "__main__":
    raise SystemExit(main())
