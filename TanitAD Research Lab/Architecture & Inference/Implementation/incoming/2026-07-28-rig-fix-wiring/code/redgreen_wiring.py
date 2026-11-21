"""RED/GREEN: remove the wiring, one guard at a time, and show the suite FAIL.

A test that passes proves nothing about a guard unless the guard can fail. This
script MUTATES the shipped source in place (backing it up first), re-runs the
wiring tests, and records exactly which tests died and with what message —
then restores the file byte-for-byte and re-runs to prove the restore.

Three mutations, each the realistic way the fix would be lost:

  1. drop_frame_arg        — ``build_v2_data`` stops passing ``frame=`` to the
                             loader. THE ACTUAL PRE-FIX STATE OF THE WORLD.
  2. drop_parent_arg       — the geometry binding is no longer told the parent,
                             so the shape check has nothing to compare against
                             and the "declared but never applied" case goes
                             silent.
  3. drop_preflight        — the flag can be forgotten again.

Usage:  python redgreen_wiring.py --stack <path to stack/> --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

TRAINER = "scripts/train_flagship_v4.py"
TESTS = "tests/test_v5_frame_wiring.py"

MUTATIONS = {
    "drop_frame_arg": {
        "file": TRAINER,
        "old": """    train_eps = build_v2_providers(a.v2_train_cache, lru_size=a.v2_lru,
                                   frame=slice_frame, verbose=verbose)
    val_eps = build_v2_providers(a.v2_val_cache, lru_size=a.v2_lru,
                                 frame=slice_frame, verbose=verbose)""",
        "new": """    train_eps = build_v2_providers(a.v2_train_cache, lru_size=a.v2_lru,
                                   verbose=verbose)
    val_eps = build_v2_providers(a.v2_val_cache, lru_size=a.v2_lru,
                                 verbose=verbose)""",
        "why": "the verified fix exists and no trainer passes it — verbatim",
    },
    "drop_parent_arg": {
        "file": TRAINER,
        "old": """        providers=train_eps, parent=cache_frame)""",
        "new": """        providers=train_eps)""",
        "why": "the binding can no longer tell a slice from a mismatch",
    },
    "drop_preflight": {
        "file": TRAINER,
        "old": """    if getattr(a, "v2_train_cache", None) and \\
            getattr(a, "v2_subframe", None) is None:""",
        "new": """    if False and getattr(a, "v2_train_cache", None) and \\
            getattr(a, "v2_subframe", None) is None:""",
        "why": "the flag becomes losable by omission again",
    },
}


def _run(stack: Path) -> dict:
    p = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q",
                        "--no-header", "-p", "no:cacheprovider"],
                       cwd=str(stack), capture_output=True, text=True)
    tail = p.stdout.strip().splitlines()
    failed = sorted({ln.split("::")[1].split(" ")[0]
                     for ln in tail if ln.startswith("FAILED ")})
    return {"exit": p.returncode, "summary": tail[-1] if tail else "",
            "failed_tests": failed,
            "first_error_lines": [ln for ln in tail if ln.lstrip().startswith("E ")][:6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    stack = Path(a.stack).resolve()
    out: dict = {"stack": str(stack)}

    out["green_before"] = _run(stack)
    print("GREEN before:", out["green_before"]["summary"], flush=True)

    out["mutations"] = {}
    for name, m in MUTATIONS.items():
        f = stack / m["file"]
        original = f.read_text(encoding="utf-8")
        digest = hashlib.sha256(original.encode()).hexdigest()
        assert m["old"] in original, f"{name}: anchor not found (source moved)"
        try:
            f.write_text(original.replace(m["old"], m["new"], 1),
                         encoding="utf-8")
            res = _run(stack)
        finally:
            f.write_text(original, encoding="utf-8")
            assert hashlib.sha256(
                f.read_text(encoding="utf-8").encode()).hexdigest() == digest
        res["why"] = m["why"]
        res["restored_sha256_matches"] = True
        out["mutations"][name] = res
        print(f"RED [{name}]: {res['summary']}  killed={res['failed_tests']}",
              flush=True)
        assert res["exit"] != 0, f"{name} did NOT fail — the guard is inert"

    out["green_after"] = _run(stack)
    print("GREEN after :", out["green_after"]["summary"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
