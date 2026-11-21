#!/usr/bin/env python3
"""After the NT change set is landed: assert, per path, that the commit holds EXACTLY the resolved content.

    python landing_verify.py --repo D:/Projects/TanitAD --rev HEAD \
        --landing out/patch/landing_d0f8719/LANDING_CHECK.json

For every path: ``git rev-parse --verify <rev>:<path>`` must be 40 HEX chars (a failing rev-parse echoes its argument,
and ``HEAD:stack/tanitad/train/refcv8_train.py`` is exactly 40 characters -- measured 2026-10-21) and must equal
``landed_blob_expected``. A path whose blob cannot be read is INCONCLUSIVE, never VERIFIED. Exit 0 only when every
path is VERIFIED; the JSON on stdout is the evidence (read it, not the exit code through a pipe)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--landing", required=True)
    a = ap.parse_args()
    rep = json.loads(Path(a.landing).read_text(encoding="utf-8"))
    res = {}
    for rel, row in rep["files"].items():
        want = row.get("landed_blob_expected")
        r = subprocess.run(["git", "-C", a.repo, "rev-parse", "--verify", "--quiet", f"{a.rev}:{rel}"],
                           capture_output=True, text=True, timeout=180)
        got = r.stdout.strip()
        if not (want and HEX40.match(want)):
            res[rel] = "NO_EXPECTATION"
        elif r.returncode != 0 or not HEX40.match(got):
            res[rel] = "INCONCLUSIVE"
        else:
            lfw = row.get("landed_blob_lf")
            res[rel] = ("VERIFIED" if got == want else
                        # same content, CRLF -> LF rewrite of a CRLF-committed file (core.autocrlf normalised it on
                        # add): content is right, but the commit is a whole-file EOL diff -- named, not hidden
                        "EOL_NORMALISED_TO_LF" if (lfw and got == lfw and lfw != want) else f"MISMATCH {got}")
    ok = all(v == "VERIFIED" for v in res.values())
    print(json.dumps({"rev": a.rev, "ALL_VERIFIED": ok, "paths": res}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
