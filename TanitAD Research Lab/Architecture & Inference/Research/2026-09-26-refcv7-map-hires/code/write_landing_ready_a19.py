#!/usr/bin/env python
"""Compose LANDING_READY_A19.txt (SPEC_REFCV7 §24, A19) in the Master Mind's Thor-gate format. Its
code entries are STACKED ON the MAPLIFT candidate's new blobs (from code/fix_a19/LANDING_BLOCKS.txt,
written by make_a19_bundle.py): apply LANDING_READY_MAPLIFT.txt first, then this. Package files
listed bare, each checked to exist, to be LF, and to carry no raw clip id.
Usage: write_landing_ready_a19.py --pkg <package dir> --date <YYYY-MM-DD> --maplift-md5 <md5>"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PKG_REL = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-map-hires"
PKG_FILES = [
    "BUILD.md", "LANDING_READY_A19.txt",
    # the A19 MAIN-only map spec (the A18 spec minus its must-fail rows) + its generator and check
    "raw/gmo_spec_A19_MAP_MAIN.json", "code/make_a19_map_spec.py", "code/a19_spec_accept_check.py",
    # the judge on a real harness-shaped record, and on the real map binding record (record half)
    "code/a19_real_shape_check.py", "code/a19_prejudge_binding.py",
    # the MAIN-only map binding (the Master Mind's chain ran it 18:05 Berlin)
    "code/binding/run_gmo_binding_a19.sh",
    # the landing tools
    "code/make_a19_bundle.py", "code/write_landing_ready_a19.py",
    "code/fix_a19/NEW2_A19_edits.diff", "code/fix_a19/BASE_BLOBS.txt",
    "code/fix_a19/LANDING_BLOCKS.txt",
]
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, type=Path)
    ap.add_argument("--date", required=True)
    ap.add_argument("--maplift-md5", required=True)
    a = ap.parse_args(argv)
    bad = []
    for f in PKG_FILES:
        if f == "LANDING_READY_A19.txt":
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
    blocks = (a.pkg / "code" / "fix_a19" / "LANDING_BLOCKS.txt").read_text(encoding="utf-8")
    body = blocks.replace("# A19 (every file EDITED against the MAPLIFT blobs)\n", "")
    col = ("# <package path>  ->  <repo path>     (comment above each: MAPLIFT base blob | "
           "new blob = git hash-object --no-filters | EOL)")
    out = [
        f"## {a.date} NEW-2 A19 (SPEC_REFCV7 §24, the PI's \"Binding = MAIN only, both\"): the launch "
        f"gate accepts a BINDING overfit record whose must-fail arms were NOT RUN (the profile's "
        f"`overfit_main_only` policy: its SPEC source and the must-fail evidence it INHERITS, carried "
        f"in the PASS text); MAIN is re-judged from its OWN literals (map: 8 bars on the declared "
        f"rule, >= 1,000 cells, the CE ratio and the finite loss from the record's own numbers, "
        f"C1-C3, the 1 ms guard, the A18 or A19 MAIN-only spec; box: the six criteria at step "
        f"2,000, 113/77, the A17 schedule), never from the harness's overall verdict; a must-fail "
        f"arm that RAN and PASSED is still a VOID. Plus the refcv7 PROFILE's eval_loader default "
        f"(the loader agent's integration diff). launch_gate.py + its tests only.",
        f"# STACKED ON the MAPLIFT candidate (LANDING_READY_MAPLIFT.txt md5 {a.maplift_md5}): the "
        f"base blobs below are MAPLIFT's NEW blobs -- apply MAPLIFT first. Full files under "
        f"code/fix_a19/; the unified diff (code/fix_a19/NEW2_A19_edits.diff) re-applies to the "
        f"MAPLIFT blobs byte for byte (git apply, core.autocrlf=false).",
        "# The eval loader itself (stack/tanitad/eval/refcv7_loader.py + its test) lands from its "
        "own package (…/2026-09-27-refcv7-eval-loader/); the profile default resolves in the "
        "launch tree once both land.",
        col, body.rstrip("\n"),
        "# package files NEW or CHANGED (all LF except the diff):"]
    out += [f"{PKG_REL}/{f}" for f in PKG_FILES] + [""]
    (a.pkg / "LANDING_READY_A19.txt").write_bytes(("\n".join(out)).encode("utf-8"))
    print(f"LANDING_READY_A19.txt: {len(out)} lines, "
          f"{sum(1 for x in chr(10).join(out).split(chr(10)) if x.startswith('# ') and ': base ' in x)} code entries, "
          f"{len(PKG_FILES)} package files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
