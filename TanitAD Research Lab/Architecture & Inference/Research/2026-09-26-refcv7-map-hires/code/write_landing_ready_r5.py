#!/usr/bin/env python
"""Compose the NEW-2 R5 LANDING_READY.txt (SPEC_REFCV7 §22.1, A17.1: G-MAP-OVERFIT's lr decays
over the final 10 %; folds in R4's harness generalisation -- edge_w0 and a spec-registered
weights definition -- which never landed on its own) in the Master Mind's Thor-gate format: per
file ``# <repo path>: base <tip blob|NEW> | new <blob> | <EOL>`` then ``<package path>  ->  <repo
path>`` (from code/fix_r5/LANDING_BLOCKS.txt, written by ``make_fix_bundle.py --set r5 --fix-dir
code/fix_r5``); package files listed bare -- only what is NEW or CHANGED after the tip
(classified by per-path blob assertion, never a listing) -- each checked to exist, to be LF where
it is text, and to carry no raw clip id.

Usage: write_landing_ready_r5.py --pkg <package dir> --tip <sha> --date <YYYY-MM-DD>
                                 --b1-check <what the tip is> [--with-a171-record]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"


def pkg_files(with_a171_record: bool) -> list:
    files = [
        "BUILD.md", "LANDING_READY.txt",
        # the complete A12 record (paused 2 h for the box head, paused_s stamped) + evidence
        "raw/gmo_early/g_map_overfit_A12.EARLY_NONBINDING.json", "raw/gmo_early/a12_launch3.log",
        "raw/gmo_early/a12_PAUSED_BY_MASTER_MIND.txt", "raw/gmo_early/a12_RESUMED_AT.txt",
        "code/stamp_paused.py", "code/watch_resume.sh", "code/resume_a12.sh",
        # the A15 arm's record (the FAIL that selected lever (4))
        "raw/gmo_early/g_map_overfit_A15.EARLY_NONBINDING.json", "raw/gmo_early/a15_launch.log",
        "code/a15_tables.py",
        # A16 (lever 4, the TRAIN mf weights): spec, weights + provenance, the PARTIAL record
        # (stopped after MAIN's FAIL, §22.1) with its log and done-marker; the R4 candidate it
        # ran on (1b8170e + code/fix_r4: diff + base blobs) and the scripts behind it
        "raw/gmo_spec_A16.json",
        "raw/map_hires_class_weights_train_100x30_MF.json",
        "raw/map_hires_class_weights_train_100x30_MF.PROVENANCE.json",
        "raw/gmo_early/g_map_overfit_A16.PARTIAL_NONBINDING.json",
        "raw/gmo_early/a16_launch.log", "raw/gmo_early/a16_A16_DONE.json",
        "code/a16_partial_record.py",
        "code/make_r4_arm.py", "code/gmo_r4_runner.py", "code/gmo_r4_launch.sh",
        "code/fix_r4/NEW2R4_edits.diff", "code/fix_r4/BASE_BLOBS.txt",
        "code/fix_r4/LANDING_BLOCKS.txt",
        # A17.1 (the lr decay, R5): the arm's spec and scripts; the landing tools
        "raw/gmo_spec_A171.json", "raw/gmo_early/a171_launch.PARTIAL_at_MAIN.log",
        "code/make_r5_arm.py", "code/gmo_r5_runner.py", "code/gmo_r5_launch.sh",
        "code/make_fix_bundle.py", "code/write_landing_ready_r5.py",
        "code/fix_r5/NEW2R5_edits.diff", "code/fix_r5/BASE_BLOBS.txt",
        "code/fix_r5/LANDING_BLOCKS.txt",
        # informative, zero GPU: how much of a step-1,000 verdict is the constant-lr read-out
        "code/readout_noise.py", "raw/gmo_early/readout_noise.json",
    ]
    if with_a171_record:
        files += ["raw/gmo_early/g_map_overfit_A171.EARLY_NONBINDING.json",
                  "raw/gmo_early/a171_launch.log"]
    return files


UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--b1-check", required=True)
    ap.add_argument("--with-a171-record", action="store_true")
    a = ap.parse_args(argv)
    PKG_FILES = pkg_files(a.with_a171_record)
    bad = []
    for f in PKG_FILES:
        if f == "LANDING_READY.txt":
            continue
        q = a.pkg / f
        if not q.is_file():
            bad.append(f"MISSING {f}")
            continue
        b = q.read_bytes()
        if UUID.search(b):
            bad.append(f"RAW CLIP ID in {f}")
        if b"\r\n" in b and not f.endswith(".diff"):
            bad.append(f"CRLF in {f}")
    if bad:
        raise SystemExit("\n".join(bad))
    blocks = (a.pkg / "code" / "fix_r5" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    body = blocks.replace("# R5 (every file EDITED against the tip)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: tip base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} NEW-2 R5 (SPEC_REFCV7 §22.1, A17.1, the PI's 'same rule for the map'): "
        f"G-MAP-OVERFIT's lr decays over the final 10 % -- a spec key lr_decay {{kind: "
        f"cosine_to_zero, start_step}}; no key = the constant-lr path, which never touches the lr "
        f"-- folding in R4's harness generalisation (must-fail arm edge_w0; a spec-registered "
        f"weights definition, refused before any trunk is built), which never landed on its own; "
        f"the A16 and A17.1 specs (both pinned by md5); the TRAIN mf weights with provenance; the "
        f"complete A12 and A15 early records and A16's partial one. Harness + its test only: no "
        f"model, trainer or registry change.",
        f"# Base blobs read at {a.tip} ({a.b1_check}). Full files under code/fix_r5/; line "
        f"endings as each tip blob (EOL column). The unified diff (code/fix_r5/NEW2R5_edits.diff) "
        f"re-applies to the RAW tip blobs byte for byte (git apply, core.autocrlf=false).",
        "# If the tip moves: code/make_fix_bundle.py --set r5 --fix-dir code/fix_r5 regenerates "
        "this block (both files are EDITED; a 3-way is needed only if the tip changes either).",
        col, body.rstrip("\n"),
        "# package files NEW or CHANGED after the tip (all LF except the diffs):"]
    out += [f"{PKG_REL}/{f}" for f in PKG_FILES] + [""]
    (a.pkg / "LANDING_READY.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY.txt: {len(out)} lines, "
          f"{sum(1 for x in chr(10).join(out).split(chr(10)) if x.startswith('# ') and ': base ' in x)} code entries, "
          f"{len(PKG_FILES)} package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
