#!/usr/bin/env python3
"""nt_run_argv -- run ``refc_v3_train.py`` of a given tree with the argv stored in an ``nt_argv.py`` record (plus
optional extra tokens), IN-PROCESS, so the argv that ran is byte-for-byte the recorded one.

    python nt_run_argv.py --tree <tree>/stack --argv-json nt-smoke.argv.json -- --device cpu
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def main() -> int:
    a = sys.argv[1:]
    extra = []
    if "--" in a:
        i = a.index("--")
        a, extra = a[:i], a[i + 1:]
    opts = dict(zip(a[::2], a[1::2]))
    stack = Path(opts["--tree"]).resolve()
    rec = json.loads(Path(opts["--argv-json"]).read_text(encoding="utf-8"))
    argv = list(rec["argv"]) + extra
    for p in (str(stack), str(stack / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(str(stack / "scripts"))
    import tanitad
    tf = str(Path(tanitad.__file__).resolve()).lower()
    if not tf.startswith(str(stack).lower()):
        raise SystemExit(f"[nt-run] tanitad imported from {tanitad.__file__}, not {stack}")
    print(f"[nt-run] tree {stack} | argv_sha256 {rec.get('argv_sha256')} | {len(argv)} tokens (+{len(extra)} extra)",
          flush=True)
    sys.argv = [str(stack / "scripts" / "refc_v3_train.py")] + argv
    runpy.run_path(sys.argv[0], run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
