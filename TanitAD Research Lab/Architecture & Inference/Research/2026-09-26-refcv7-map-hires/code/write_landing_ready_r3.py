#!/usr/bin/env python
"""Compose the NEW-2 R3 (SPEC_REFCV7 §20, A15) LANDING_READY.txt in the Master Mind's Thor-gate
format: per file ``# <repo path>: base <tip blob|NEW> | new <blob> | <EOL>`` then
``<package path>  ->  <repo path>`` (from code/fix_r3/LANDING_BLOCKS.txt, written by
``make_fix_bundle.py --set r3 --fix-dir code/fix_r3``); package files listed bare -- only what
is NEW or CHANGED after 2374cd2 (R2 as landed) -- each checked to exist, to be LF where it is
text, and to carry no raw clip id.

Usage: write_landing_ready_r3.py --pkg <package dir> --tip <sha> --date <YYYY-MM-DD>
                                 --b1-check <what the tip is>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
PKG_FILES = [
    "BUILD.md", "LANDING_READY.txt",
    # the A15 arm's registered spec
    "raw/gmo_spec_A15.json",
    # the decoder lever's design input and the scripts behind R3's arm
    "raw/gmo_early/decoder_lever_cost.json", "code/decoder_lever_cost.py",
    "code/gmo_r3_runner.py", "code/gmo_r3_launch.sh", "code/a12_tables.py",
    "code/gmo_r2_launch2.sh",
    "code/make_fix_bundle.py", "code/write_landing_ready_r3.py",
    "code/fix_r3/NEW2R3_edits.diff", "code/fix_r3/BASE_BLOBS.txt",
    "code/fix_r3/LANDING_BLOCKS.txt",
]
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--b1-check", required=True)
    a = ap.parse_args(argv)
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
    blocks = (a.pkg / "code" / "fix_r3" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    body = blocks.replace("# R3 (every file EDITED against the tip)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: tip base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} NEW-2 R3 (SPEC_REFCV7 §20, A15): the 10 cm map's DECODER lever, stacked on "
        f"R2's near lift -- NearRefineBlock (3x3 dilation 2 -> GN -> GELU -> 3x3 dilation 4, last "
        f"conv zero-initialised) on the NEAR rows at 0.1 m, map-only; "
        f"--map-hires-near-refine-blocks (G-HYG field MapHiresConfig.near_refine_blocks, G-DVB "
        f"map_hires_near_refine_blocks: registry 215); the eval loader's rebuild; the harness's "
        f"--near-refine-blocks and must-fail arm near_block_zeros; the A15 spec.",
        f"# Base blobs read at {a.tip} ({a.b1_check}). Full files under code/fix_r3/; line "
        f"endings as each tip blob (EOL column). The unified diff (code/fix_r3/NEW2R3_edits.diff) "
        f"re-applies to the RAW tip blobs byte for byte (git apply, core.autocrlf=false).",
        "# If the tip moves: code/rebase_onto_tip.py (3-way, LF-normalised) + "
        "code/make_fix_bundle.py --set r3 --fix-dir code/fix_r3 regenerate this block.",
        col, body.rstrip("\n"),
        "# package files NEW or CHANGED after 2374cd2 (all LF except the diff):"]
    out += [f"{PKG_REL}/{f}" for f in PKG_FILES] + [""]
    (a.pkg / "LANDING_READY.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY.txt: {len(out)} lines, "
          f"{sum(1 for x in chr(10).join(out).split(chr(10)) if x.startswith('# ') and ': base ' in x)} code entries, "
          f"{len(PKG_FILES)} package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
