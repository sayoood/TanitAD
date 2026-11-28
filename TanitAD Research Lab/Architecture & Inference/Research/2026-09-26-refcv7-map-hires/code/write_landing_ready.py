#!/usr/bin/env python
"""Compose LANDING_READY.txt from code/fix/LANDING_BLOCKS.txt (make_fix_bundle.py) in the
Master Mind's Thor-gate format: per file ``# <repo path>: base <tip blob|NEW> | new
<blob> | <EOL>`` then ``<package path>  ->  <repo path>``; package files listed bare.

Usage: write_landing_ready.py --pkg <package dir> --tip <sha> --date <YYYY-MM-DD>
                              --base-label <what the tip is> --b1-check <independence run>
                              [--b2-note <landing-order note for the batch-2 header>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
PKG_FILES = ["BUILD.md", "LANDING_READY.txt",
             "code/cost_map_hires.py", "code/loss_budget_w_map_hires.py",
             "code/oracle_05m_at_10cm.py", "code/analytic_control_evidence.py",
             "code/d4_weighting_proposals.py", "code/make_fix_bundle.py",
             "code/rebase_onto_tip.py", "code/write_landing_ready.py", "code/v3_reader_control.py", "code/smoke_train_real.py",
             "code/fix/NEW2_shared_edits.diff", "code/fix/BASE_BLOBS.txt",
             "code/fix/LANDING_BLOCKS.txt",
             "raw/class_weights_DRYRUN_eval5.json", "raw/cost_analytic_cpu.json",
             "raw/cost_analytic_a6_cpu.json", "raw/cost_analytic_a6_cpu.log",
             "raw/loss_budget_w_map_hires.json", "raw/analytic_control_fine_reader.json",
             "raw/d4_weighting_proposals.json", "raw/v3_reader_control_eval.json",
             "raw/train_smoke_real_files.json"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--base-label", required=True,
                    help="what the tip IS, e.g. 'fixes batch 2 b4a59b9'")
    ap.add_argument("--b1-check", required=True,
                    help="the MEASURED batch-1 independence run, e.g. 'b4a59b9 + batch 1 only, 131 passed'")
    ap.add_argument("--b2-note", default="",
                    help="landing-order note appended to the batch-2 header")
    a = ap.parse_args(argv)
    blocks = (a.pkg / "code" / "fix" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    b1, b2 = blocks.split("# BATCH 2 (shared-file edits + the new files that need them)\n")
    b1 = b1.replace("# BATCH 1 (independent: new files only)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: tip base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} batch 1: refcv7 NEW-2 -- the 10 cm map's NEW modules "
        f"(A6/A7/A8: extent, /3 reader, shared encoder, sqrt_mf weights, the NEW-2 G-DVB "
        f"checks), their tests, the weights script, the build record. Independent of every "
        f"shared file (MEASURED: {a.b1_check}). Full files under code/fix/.",
        f"# Base blobs read at {a.tip}. New files only (base NEW); LF.",
        "# Supersedes every earlier NEW-2 list and the stale 22:37 copies at the same repo "
        "paths. RESULT.md is untouched.",
        col, b1.rstrip("\n"),
        "# package files:"] + [f"{PKG_REL}/{f}" for f in PKG_FILES] + [
        "# REMOVE from the package if present (superseded by code/make_fix_bundle.py): "
        f"{PKG_REL}/code/make_fix_bundle.sh",
        "",
        f"## {a.date} batch 2 (on {a.base_label}{a.b2_note}): refcv7 NEW-2 "
        f"shared-file edits -- A6 one lift in the forward (hook), the planner pool, the "
        f"stride-8 tap + fmap_s8 pass-through, --bev-source / --bev-planner-crop-m / extent "
        f"/ grad-ckpt flags + pins + G-DVB, the eval-loader rebuild, the ga_mh_* reach keys "
        f"DECLARED for D3's first-row check -- plus the new files "
        f"that need them.",
        f"# Base blobs read at {a.tip} ({a.base_label}). Full files; line endings as "
        f"each tip blob (EOL column). The unified diff (code/fix/NEW2_shared_edits.diff) "
        f"re-applies to the RAW tip blobs byte for byte (git apply, core.autocrlf=false).",
        "# If the tip moves: code/rebase_onto_tip.py (3-way, LF-normalised) + "
        "code/make_fix_bundle.py regenerate this block; a conflict exits 2.",
        col, b2.rstrip("\n"), ""]
    (a.pkg / "LANDING_READY.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY.txt: {len(out)} lines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
