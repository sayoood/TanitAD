#!/usr/bin/env python3
"""make_canonical_box_argv.py -- SPEC_REFCV7 §18 (A13): the G-BOX-OVERFIT config = the refcv7 CANONICAL argv (the
launch-gate package's ``stack/ops/runs.d/refcv7-r101-s0.argv.json``) + the A9 box flags (R1-R4) + the VIS-1 sidecar.

The canonical file's own ``todo_box_head`` says the box flags are ADDED when the box head lands; until then this tool
writes a DERIVED object file for the non-binding harness runs, with its provenance (the canonical file's sha256 and
the exact tokens added). ``--out`` is pointed at a scratch path (the harness writes nothing there; a launch path in a
diagnostic argv is a loaded gun). ``--trunk-compile`` is KEPT: the harness drops it itself and records that.

usage: python make_canonical_box_argv.py <canonical.argv.json> <sidecar path on the run host> <out.json> [extra tokens]
(extra tokens, e.g. ``--slot-query-select learned_ref``, are appended after the A9 flags and recorded)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

A9_FLAGS = ["--slot-presence-loss", "focal", "--slot-presence-prior", "0.01", "--slot-deep-supervision", "--slot-vis1"]
SCRATCH_OUT = "/tmp/gbo_scratch_out_never_written"


def derive(canonical: dict, sidecar: str, extra: list | None = None) -> tuple[list, dict]:
    argv = list(canonical["argv"])
    present = [f for f in ("--slot-presence-loss", "--slot-presence-prior", "--slot-deep-supervision", "--slot-vis1",
                           "--vis1-sidecar") if f in argv]
    if present:
        raise SystemExit(f"the canonical argv already carries {present}: use it as-is")
    if "--out" in argv:
        i = argv.index("--out")
        argv[i + 1] = SCRATCH_OUT
    added = A9_FLAGS + ["--vis1-sidecar", sidecar] + list(extra or [])
    return argv + added, {"added": added, "out_replaced_with": SCRATCH_OUT}


def main() -> int:
    src, sidecar, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    extra = sys.argv[4:]
    raw = src.read_bytes()
    canonical = json.loads(raw.decode("utf-8"))
    if not isinstance(canonical, dict) or "argv" not in canonical:
        raise SystemExit("expected the canonical OBJECT file (with 'argv')")
    argv, info = derive(canonical, sidecar, extra)
    rec = {"schema": canonical.get("schema", "tanitad.launch_argv/1"),
           "arm": f"{canonical.get('arm', '?')} + A9 box flags (G-BOX-OVERFIT, NON-BINDING; SPEC_REFCV7 §18 A13)",
           "derived_from": {"file": "stack/ops/runs.d/refcv7-r101-s0.argv.json (launch-gate package)",
                            "sha256": hashlib.sha256(raw).hexdigest(), "n_tokens": len(canonical["argv"])},
           **info, "argv": argv}
    out.write_text(json.dumps(rec, indent=1), encoding="utf-8", newline="\n")
    print(f"{out}: {len(argv)} tokens (canonical {len(canonical['argv'])} + {len(info['added'])}); "
          f"canonical sha256 {rec['derived_from']['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
