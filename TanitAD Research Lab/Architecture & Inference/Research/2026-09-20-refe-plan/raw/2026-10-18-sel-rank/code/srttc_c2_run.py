#!/usr/bin/env python3
"""SR-TTC C2: the ONLY sanctioned entry for the shuffled-label negative-control fit (PREREG_SELRANK_ARM.md:71).  CPU only.

    C:\\Users\\Admin\\venvs\\driverl-eval\\Scripts\\python.exe -I <code>\\srttc_c2_run.py run --fraction 10 --out-dir ... --threads 8 --epochs 10 --batch-sets 256 --seeds 0,1 --criterion pooled_oof --both-rule flat
        (the exact command is printed by `srttc_c2_gate.py argv --fraction 10`)

It REFUSES (exit 3) before importing or fitting anything unless a signed PASS launch-gate token (stage c2, profile srttc-c2) matches THIS command, the deployed code, the S0 bank manifest and the
registered r1_result.json (Master Mind R5: C2 needs a gate token exactly as EP-R1's `--stage c2` does).  `--no-gate-stage-ack` is not accepted (exit 2).  When the token verifies, it runs the landed
`srttc_c2.run_c2` from the bytes the token binds and writes `r1_c2_attestation.json` next to the C2 result.

WHY A WRAPPER: the landed `srttc_c2.py` (frozen) cannot check a token without being edited, and its own command line only starts with `--no-gate-stage-ack`.  This file does not edit it; it is the entry
a launch uses.  `srttc_c2.py` run directly with its acknowledgement flag would still fit without a token: that path is NOT sanctioned, and the dev gate's pre-read check (`srttc_c2_gate.verify_attestation`)
refuses a C2 result that has no attestation of a PASS token.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import srttc_c2_gate as CG   # noqa: E402

if __name__ == "__main__":
    sys.exit(CG.run_main(sys.argv[1:], __file__))
