#!/usr/bin/env python
"""Compose the NEW-2 MAP-LIFT CLOSURE's LANDING_READY_MAPLIFT.txt (SPEC_REFCV7 §23, A18) in the
Master Mind's Thor-gate format: per file ``# <repo path>: base <tip blob|NEW> | new <blob> | <EOL>``
then ``<package path>  ->  <repo path>`` (from code/fix_maplift/LANDING_BLOCKS.txt, written by
``make_fix_bundle.py --set maplift --fix-dir code/fix_maplift``); package files listed bare.

⛔ R5 lands as its OWN LANDING_READY.txt (md5 7200d3dc, the map harness decay + R4); the Master
Mind concatenates the READY files into one gate. This file carries the gate's closure only: the
canonical argv (the FINAL 154-token list), `launch_gate.py` (MAP-LIFT closed; the G-MAP-OVERFIT
judge binds the A18 protocol; the BOM-safe source reads), the gate's tests, and the A18 spec pin
as a NEW test file. It lands ONLY if the early A18 MAIN passes.

Usage: write_landing_ready_maplift.py --pkg <package dir> --tip <sha> --date <YYYY-MM-DD>
                                      --b1-check <what the tip is> [--with-a18-record]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"


def pkg_files(with_a18_record: bool) -> list:
    files = [
        "BUILD.md", "LANDING_READY_MAPLIFT.txt",
        # A18 (§23): the registered spec, its generator, the early MAIN's runner + launcher
        "raw/gmo_spec_A18.json", "code/make_a18_arm.py",
        "code/gmo_a18_main.py", "code/gmo_a18_launch.sh", "code/a18_tables.py",
        # the MAP BINDING run (the Master Mind's chain starts it after the box binding)
        "code/binding/run_gmo_binding.sh",
        # the complete A17.1 record (the edge FAIL that sent the budget to the PI)
        "raw/gmo_early/g_map_overfit_A171.EARLY_NONBINDING.json", "raw/gmo_early/a171_launch.log",
        "raw/gmo_early/a171_A171_DONE.json",
        # prepared, NOT run: the second near refine block arm (a DRAFT spec) + its cost
        "code/make_block2_arm.py", "raw/gmo_spec_BLOCK2_DRAFT.json",
        "raw/gmo_spec_BLOCK2_A18_DRAFT.json",
        "code/block2_cost.py", "raw/gmo_early/block2_cost.json",
        # the landing tools
        "code/make_fix_bundle.py", "code/write_landing_ready_maplift.py",
        "code/fix_maplift/NEW2_MAPLIFT_edits.diff", "code/fix_maplift/BASE_BLOBS.txt",
        "code/fix_maplift/LANDING_BLOCKS.txt",
    ]
    if with_a18_record:
        files += ["raw/gmo_early/g_map_overfit_A18_MAIN.EARLY_NONBINDING.json",
                  "raw/gmo_early/a18_launch.log", "raw/gmo_early/a18_A18_DONE.json"]
    return files


UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--tip", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--b1-check", required=True)
    ap.add_argument("--with-a18-record", action="store_true")
    a = ap.parse_args(argv)
    PKG_FILES = pkg_files(a.with_a18_record)
    bad = []
    for f in PKG_FILES:
        if f == "LANDING_READY_MAPLIFT.txt":
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
    blocks = (a.pkg / "code" / "fix_maplift" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    body = blocks.replace("# MAPLIFT (every file EDITED against the tip)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: tip base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} NEW-2 MAP-LIFT CLOSURE (SPEC_REFCV7 §23, A18; lands ONLY if the early A18 "
        f"MAIN passes): the canonical argv carries the FINAL 154-token list (gate sha256 "
        f"6402d33de75b7f1c6dbdeb9aeedd46a00a82e7325eec420fa179f366213bd5cd: the box A17 argv + "
        f"--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1 after --map-hires-grad-ckpt "
        f"on); launch_gate.py closes MAP-LIFT (both flags required values), binds G-MAP-OVERFIT to "
        f"the A18 protocol (spec sha256, steps 3,000, lr 1e-3, batch 4, seed 0, lr_decay from "
        f"2,700, near lift 20 + 1 block, must-fail near_block_zeros) and reads sources utf-8-sig "
        f"(import_closure + static_eager_own_files: tanitad/eval/__init__.py carries a BOM); the "
        f"gate's tests; the A18 spec pin as a NEW test file. R5 (LANDING_READY.txt) lands with "
        f"it. No change to the trainer, stack/tanitad/** or either harness (the binding closures).",
        f"# Base blobs read at {a.tip} ({a.b1_check}). Full files under code/fix_maplift/; line "
        f"endings as each tip blob (EOL column). The unified diff "
        f"(code/fix_maplift/NEW2_MAPLIFT_edits.diff) re-applies to the RAW tip blobs byte for byte "
        f"(git apply, core.autocrlf=false).",
        "# If the tip moves: code/make_fix_bundle.py --set maplift --fix-dir code/fix_maplift "
        "regenerates this block (a 3-way is needed only if the tip changes one of the three gate "
        "files).",
        col, body.rstrip("\n"),
        "# package files NEW or CHANGED (all LF except the diff):"]
    out += [f"{PKG_REL}/{f}" for f in PKG_FILES] + [""]
    (a.pkg / "LANDING_READY_MAPLIFT.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY_MAPLIFT.txt: {len(out)} lines, "
          f"{sum(1 for x in chr(10).join(out).split(chr(10)) if x.startswith('# ') and ': base ' in x)} code entries, "
          f"{len(PKG_FILES)} package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
