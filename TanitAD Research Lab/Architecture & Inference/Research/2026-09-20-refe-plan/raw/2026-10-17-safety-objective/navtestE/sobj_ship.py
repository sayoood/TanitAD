#!/usr/bin/env python3
"""sobj_ship: the free-point shipper (fp/fp_ship.py, a byte-identical COPY) pointed at ANOTHER pod directory and with NO scorer states to ship.

    python sobj_ship.py {plan|prep-pod|run|status} --work W [--remote /workspace/ops/sobjE]

Why a wrapper and not an edit: fp_ship.py hard-codes REMOTE = /workspace/ops/freept (the LIVE free-point run's directory: its in/ and out/ would collide) and STATES = {C0A, R0B} (dev-box
files).  The SAFEOBJ states are already on the pod (the arms trained there), so STATES is emptied and the worker reads them in place.  Everything else -- the chunking, the space gate, the
pull-with-sha256 -- is the free-point code unchanged.  The pod address is read from <W>/../pod_ssh.txt by fp_ship (never written into this package).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fp"))
import fp_ship as SH  # noqa: E402

LIVE_REMOTE = "/workspace/ops/freept"
DEFAULT_REMOTE = "/workspace/ops/sobjE"


def configure(remote: str) -> None:
    if remote.rstrip("/") == LIVE_REMOTE:
        raise SystemExit(f"REFUSED: {LIVE_REMOTE} is the free-point run's pod directory")
    if not remote.startswith("/workspace/ops/"):
        raise SystemExit(f"REFUSED: {remote!r} is not under /workspace/ops/")
    SH.REMOTE = remote
    SH.STATES = {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "prep-pod", "run", "status"))
    ap.add_argument("--work", required=True)
    ap.add_argument("--remote", default=DEFAULT_REMOTE)
    a = ap.parse_args(argv)
    configure(a.remote)
    sys.argv = ["fp_ship.py", a.cmd, "--work", a.work]
    return SH.main()


if __name__ == "__main__":
    sys.exit(main())
