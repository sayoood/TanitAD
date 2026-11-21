#!/usr/bin/env python3
"""Apply step-cost lever 2 (per-class 10 cm signal + box census on LOGGED train steps only) to
stack/scripts/refc_v3_train.py -- byte-exact, CRLF preserved, self-verifying.

    python apply_lever2.py <path/to/refc_v3_train.py>            # dry run: verifies only
    python apply_lever2.py <path/to/refc_v3_train.py> --write

REFUSES unless the input is EXACTLY the tip blob it was built on (git blob hash, not a summary),
and checks the output hashes to the recorded result blob. Proposal only: the Master Mind lands it.
"""
import hashlib, json, sys
from pathlib import Path
here = Path(__file__).resolve().parent
spec = json.loads((here / "lever2_pairs.json").read_text(encoding="utf-8"))
f = Path(sys.argv[1])
raw = f.read_bytes()
def gh(b): return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
if gh(raw) != spec["base_blob"]:
    sys.exit(f"REFUSED: {f} is blob {gh(raw)}, the patch was built on {spec['base_blob']}")
s = raw.decode("utf-8")
for o, n in spec["pairs"]:
    if s.count(o) != 1:
        sys.exit("REFUSED: an anchor is not unique / missing")
    s = s.replace(o, n)
out = s.encode("utf-8")
if gh(out) != spec["result_blob"]:
    sys.exit(f"REFUSED: result blob {gh(out)} != recorded {spec['result_blob']}")
print("OK: base", spec["base_blob"][:12], "-> result", spec["result_blob"][:12])
if "--write" in sys.argv:
    f.write_bytes(out)
    print("written", f)
