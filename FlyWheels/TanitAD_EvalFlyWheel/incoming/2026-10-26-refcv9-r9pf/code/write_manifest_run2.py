"""write_manifest_run2.py -- MANIFEST_R9PF_RUN2.sha256 (LF line endings): '<sha256>  <absolute path>' for every NEW artifact of run 2
(K-EQ under AMENDMENT2, the PA verdict AMEND2 reading, PREREG_R9PF_LH run 2).  Every hash asserted 64 hex; an unreadable file FAILS the manifest.
NO log is listed (the a3_keq_* / keq_* roll logs carry clip-id prefixes); F: scratch entries are provenance, not landing.
NEW (EvalFlyWheel run 2, 2026-10-26)."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9pf_common as RC  # noqa: E402

OUT = RC.WORK / "MANIFEST_R9PF_RUN2.sha256"
W = RC.WORK
paths = [W / p for p in (
    "PREREG_REFCV9_INMODEL_PA_AMENDMENT2.md", "PREREG_REFCV9_INMODEL_PA_AMENDMENT2.sha256", "PREREG_R9PF_LH.md", "PREREG_R9PF_LH.sha256",
    "code/a3_keq_roll_amend2.py", "code/run_keq_amend2.sh", "code/a3_keq_amend2_compare.py", "code/pa_verdict_amend2.py", "code/lh_r9pf_run2.py",
    "code/lh_verdict_run2_md.py", "code/lh_verdict_void_mm.py", "code/write_manifest_run2.py", "code/deviations_r9pf_run2.json",
    "result/R9PF_LH_VERDICT.json", "result/R9PF_LH_VERDICT.md", "result/R9PF_LH_RUN2_PROVISIONAL_NO_KEQ.json")]
missing = [str(p) for p in paths if not p.is_file()]
if missing:
    raise SystemExit(f"[manifest2] missing: {missing}")
# scratch (F:, NOT for landing): the composed LH / DER packages and the plumbing-test K-EQ diagnostic
for pat in ("arms_lh_run2/**/*.npz", "KEQ_AMEND2_PLUMBING_TEST.json"):
    paths += [p for p in sorted(RC.SCRATCH.glob(pat)) if p.is_file()]
lines, bad = [], []
for p in paths:
    h = RC.sha256_file(p)
    if len(h) != 64:
        bad.append(str(p))
    lines.append(f"{h}  {p.as_posix()}")
if bad:
    raise SystemExit(f"[manifest2] unreadable: {bad}")
body = ("# MANIFEST_R9PF_RUN2 -- K-EQ under AMENDMENT2 + PA verdict AMEND2 + PREREG_R9PF_LH run 2; sha256  path; EvalFlyWheel 2026-10-26\n"
        + "\n".join(lines) + "\n")
OUT.write_bytes(body.encode("utf-8"))
assert b"\r\n" not in OUT.read_bytes()
print(f"[manifest2] {len(lines)} files -> {OUT}  sha256 {RC.sha256_file(OUT)}")
