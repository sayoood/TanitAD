#!/usr/bin/env python3
"""DATASCALE-10K per-window stage wrapper: ds_stage.py UNEDITED, ROOT/LEDGER overridden per window dir. usage: ds_stage_w.py <windir> <threads>
No address, port or pod id in this file."""
import sys
sys.path.insert(0, "/root/datascale")
import ds_stage as S

W, TH = sys.argv[1], sys.argv[2]
S.ROOT = f"{W}/stage"
S.LEDGER = f"{W}/ledger_stage.jsonl"
sys.argv = ["ds_stage.py", "--lists", f"{W}/clips.txt", "--threads", TH]
sys.exit(S.main())
