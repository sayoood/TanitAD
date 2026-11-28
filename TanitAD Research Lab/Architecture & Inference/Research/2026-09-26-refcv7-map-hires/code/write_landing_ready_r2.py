#!/usr/bin/env python
"""Compose the NEW-2 R2 (SPEC_REFCV7 §17, A12) LANDING_READY.txt in the Master Mind's Thor-gate
format: per file ``# <repo path>: base <tip blob|NEW> | new <blob> | <EOL>`` then
``<package path>  ->  <repo path>`` (from code/fix_r2/LANDING_BLOCKS.txt, written by
``make_fix_bundle.py --set r2 --fix-dir code/fix_r2``); package files listed bare, each checked
to exist, to be LF where it is text, and to carry no raw clip id.

Usage: write_landing_ready_r2.py --pkg <package dir> --tip <sha> --date <YYYY-MM-DD>
                                 --b1-check <what the tip is>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
PKG_FILES = [
    "BUILD.md", "LANDING_READY.txt",
    # the A12 arm's registered spec
    "raw/gmo_spec_A12.json",
    # the early (non-binding) G-MAP-OVERFIT record, MAIN_long, the grid oracle, the lever cost
    "raw/gmo_early/g_map_overfit.EARLY_NONBINDING.json",
    "raw/gmo_early/g_map_overfit_MAIN_long.INFORMATIVE.json",
    "raw/gmo_early/gmo_run.log", "raw/gmo_early/launch_long2.log",
    "raw/gmo_early/grid_oracle_16frames.json", "raw/gmo_early/lift_lever_cost.json",
    "raw/gmo_early_inputs_check.json",
    "raw/map_hires_class_weights_train_100x30.json",
    "raw/map_hires_class_weights_train_100x30.PROVENANCE.json",
    # the scripts that produced them
    "code/gmo_early_runner.py", "code/gmo_early_launch.sh", "code/gmo_early_launch2.sh",
    "code/gmo_launch_long2.sh", "code/gmo_main_long.py", "code/gmo_inputs_check.py",
    "code/grid_oracle.py", "code/analyse_long.py", "code/record_tables.py",
    "code/gmo_summarise.py", "code/lift_lever_cost.py",
    "code/gmo_r2_runner.py", "code/gmo_r2_launch.sh", "code/gmo_r2_launch3.sh",
    "code/make_fix_bundle.py", "code/write_landing_ready_r2.py",
    "code/fix_r2/NEW2R2_edits.diff", "code/fix_r2/BASE_BLOBS.txt",
    "code/fix_r2/LANDING_BLOCKS.txt",
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
    blocks = (a.pkg / "code" / "fix_r2" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    body = blocks.replace("# R2 (every file EDITED against the tip)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: tip base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} NEW-2 R2 (SPEC_REFCV7 §17, A12): the 10 cm map's lift lever -- a 0.1 m "
        f"NEAR-RANGE LIFT of the stride-8 map over x 0-20 m (full width), added to the 10 cm "
        f"decoder only (map-only, zero-initialised, geometry derived from the 0.25 m lift); "
        f"--map-hires-near-lift-m (G-HYG field MapHiresConfig.near_lift_x_m, G-DVB "
        f"map_hires_near_lift_m: registry 214); the eval loader's rebuild; the harness's "
        f"--near-lift-m and must-fail arm near_zeros; the A12 spec; and the EARLY (non-binding) "
        f"G-MAP-OVERFIT record that chose the lever.",
        f"# Base blobs read at {a.tip} ({a.b1_check}). Full files under code/fix_r2/; line "
        f"endings as each tip blob (EOL column). The unified diff (code/fix_r2/NEW2R2_edits.diff) "
        f"re-applies to the RAW tip blobs byte for byte (git apply, core.autocrlf=false).",
        "# If the tip moves: code/rebase_onto_tip.py (3-way, LF-normalised) + "
        "code/make_fix_bundle.py --set r2 --fix-dir code/fix_r2 regenerate this block.",
        col, body.rstrip("\n"),
        "# package files (NEW unless already landed; all LF except the diff):"]
    out += [f"{PKG_REL}/{f}" for f in PKG_FILES] + [""]
    (a.pkg / "LANDING_READY.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY.txt: {len(out)} lines, "
          f"{sum(1 for x in chr(10).join(out).split(chr(10)) if x.startswith('# ') and ': base ' in x)} code entries, "
          f"{len(PKG_FILES)} package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
