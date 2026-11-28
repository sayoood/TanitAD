#!/usr/bin/env python3
"""DATASCALE-10K per-window build wrapper: ds_build.py UNEDITED, ROOT overridden (workers are forked after the override).
usage: ds_build_w.py <windir> <workers>. No address, port or pod id in this file."""
import sys
sys.path.insert(0, "/root/datascale")
import ds_build as Bd

W, NW = sys.argv[1], sys.argv[2]
Bd.ROOT = f"{W}/stage"
sys.argv = ["ds_build.py", "--clips", f"{W}/okclips.txt", "--v2ep-root", f"{W}/v2ep", "--ledger", f"{W}/ledger_build.jsonl",
            "--log", f"{W}/build.log", "--workers", NW]
sys.exit(Bd.main())
