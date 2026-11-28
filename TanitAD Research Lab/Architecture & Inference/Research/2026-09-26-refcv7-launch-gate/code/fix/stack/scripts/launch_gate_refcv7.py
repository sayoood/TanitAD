#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`launch_gate_refcv7.py` -- the name SPEC_REFCV7 section 2 gives the gate: `launch_gate.py` with
the refcv7 profile pinned. Every subcommand is the generic gate's; this file only refuses to run
under any other profile, so a refcv7 launch cannot be gated against the refcv6 rules by a typo.

    python stack/scripts/launch_gate_refcv7.py run --stage devbox --tree ... --commit ... \
        --argv-file ... --out-dir ...          # == launch_gate.py run --profile refcv7 ...
    python stack/scripts/launch_gate_refcv7.py verify --token ... --argv-file ... --tree ...
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_gate  # noqa: E402

PROFILE = "refcv7"


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if a and a[0] in ("run", "bind"):
        if "--profile" in a:
            i = a.index("--profile")
            if i + 1 >= len(a) or a[i + 1] != PROFILE:
                print(f"[gate] REFUSED: launch_gate_refcv7.py runs the {PROFILE} profile only "
                      f"(got {a[i + 1] if i + 1 < len(a) else None!r})", flush=True)
                return 3
        else:
            a = [a[0], "--profile", PROFILE, *a[1:]]
    return launch_gate.main(a)


if __name__ == "__main__":
    raise SystemExit(main())
