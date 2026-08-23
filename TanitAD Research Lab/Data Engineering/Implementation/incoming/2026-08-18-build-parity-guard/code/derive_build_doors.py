"""Re-derive the corpus-materialiser population and its gate status.

Standalone runner for the derivation that `stack/tests/test_build_parity_guard.py`
asserts on. It deliberately IMPORTS that module rather than reimplementing the
walk: a re-implementation tests the rebuild, not the rule that runs
(RETRACTION_LOG C87 — "re-implementing a step and finding it sound tests your
version, not the step that ran").

Usage (from anywhere):
    python derive_build_doors.py [--repo <repo root>] [--out raw/build_doors.json]

Emits, for every module under `stack/` that names a corpus artifact:
  writes   — the (lineno, callee) pairs where an artifact-shaped path is published
  gated    — whether it CALLS parity.guard_corpus_build (an AST call, not a grep)
  verdict  — GATED / CONSUMER / CANNOT_GATE / UNCLASSIFIED
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_repo = os.path.abspath(os.path.join(here, *[".."] * 6))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=default_repo)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(here), "raw", "build_doors.json"))
    a = ap.parse_args(argv)

    stack = os.path.join(a.repo, "stack")
    if not os.path.isdir(stack):
        raise SystemExit(f"no stack/ under {a.repo!r} — pass --repo")
    sys.path.insert(0, os.path.join(stack, "tests"))
    sys.path.insert(0, stack)
    from test_build_parity_guard import (                    # noqa: PLC0415
        NOT_AN_INGEST_DOOR, KNOWN_DOORS, derive_corpus_writers)
    from tanitad.data import parity                          # noqa: PLC0415

    derived = derive_corpus_writers(stack)
    rows = {}
    for rel, v in sorted(derived.items()):
        if v["gated"]:
            verdict = "GATED"
        elif not v["writes"]:
            verdict = "MENTION_ONLY"
        elif rel in NOT_AN_INGEST_DOOR:
            verdict = ("CANNOT_GATE"
                       if NOT_AN_INGEST_DOOR[rel].startswith("CANNOT_GATE")
                       else "CONSUMER")
        else:
            verdict = "UNCLASSIFIED"
        rows[rel] = {"writes": v["writes"], "gated": v["gated"],
                     "verdict": verdict,
                     "reason": NOT_AN_INGEST_DOOR.get(rel, "")}

    write_pop = {k: v for k, v in rows.items() if v["writes"]}
    out = {
        "_evidence_class": "MEASURED (ours; this script, over stack/ at HEAD+worktree)",
        "_rule_write": ("an artifact-shaped path (CORPUS_ARTIFACT) flows into a "
                        "publishing call (WRITE_CALLS), two-hop local dataflow"),
        "_rule_mention": "the module merely NAMES a corpus artifact and writes something",
        "_known_limitation": (
            "static derivation cannot cross a helper-function boundary: "
            "v2_compressed.build hands the <clip>.v2ep.pt path to build_compressed(), "
            "which writes it through a parameter. That door is therefore in the "
            "MENTION census, not the WRITE population, and is pinned instead by a "
            "direct behavioural test "
            "(test_v2_compressed_build_refuses_before_any_download)."),
        "oracles": {
            "parity_train_clips": len(parity.parity_train_clip_digests()),
            "deployed_val_clips": len(parity.deployed_val_clip_digests()),
            "intersection": len(parity.parity_train_clip_digests() &
                                parity.deployed_val_clip_digests()),
        },
        "counts": {
            "mention_census": len(rows),
            "write_population": len(write_pop),
            "gated": sum(1 for v in rows.values() if v["gated"]),
            "unclassified": sum(1 for v in rows.values()
                                if v["verdict"] == "UNCLASSIFIED"),
        },
        "known_doors": list(KNOWN_DOORS),
        "modules": rows,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["counts"], indent=1))
    print(f"-> {a.out}")
    for rel, v in rows.items():
        if v["writes"]:
            print(f"  {v['verdict']:<12} {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
