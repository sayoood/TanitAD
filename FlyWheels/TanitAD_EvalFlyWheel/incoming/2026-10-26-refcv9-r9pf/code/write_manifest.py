"""write_manifest.py -- MANIFEST_R9PF_PA.sha256: '<sha256>  <absolute path>' for every artifact of PREREG_REFCV9_INMODEL_PA Phase A (work dir + the
scratch packages and rolls).  Every hash is asserted 64 hex; an unreadable file FAILS the manifest (never silently skipped).  NEW (EvalFlyWheel 2026-10-26)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

OUT = RC.WORK / "MANIFEST_R9PF_PA.sha256"
paths = []
for sub in ("", "code", "heads", "result", "logs"):
    d = RC.WORK / sub if sub else RC.WORK
    paths += [p for p in sorted(d.iterdir()) if p.is_file() and p != OUT and p.suffix not in (".pyc",)]
for pat in ("arms/**/*.npz", "keq_s*_o*/*", "calib_rows.npz", "fits/**/*", "REFCV9_INMODEL.tip.patch",
            "launch_tree_fix/stack/tanitad/refs/refc_v3.py", "launch_tree_fix/stack/tanitad/refs/refcv9_progress.py",
            "launch_tree/stack/tanitad/refs/refc_v3.py", "R9PF_PA_BARS_PROVISIONAL_NO_KEQ.json", "patches/*.py"):
    paths += [p for p in sorted(RC.SCRATCH.glob(pat)) if p.is_file()]
lines, bad = [], []
for p in paths:
    h = RC.sha256_file(p)
    if len(h) != 64:
        bad.append(str(p))
    lines.append(f"{h}  {p.as_posix()}")
if bad:
    raise SystemExit(f"[manifest] unreadable: {bad}")
OUT.write_text("# MANIFEST_R9PF_PA -- PREREG_REFCV9_INMODEL_PA Phase A (+ AMENDMENT1); sha256  path; EvalFlyWheel 2026-10-26\n" + "\n".join(lines) + "\n",
               encoding="utf-8")
print(f"[manifest] {len(lines)} files -> {OUT}  sha256 {RC.sha256_file(OUT)}")
