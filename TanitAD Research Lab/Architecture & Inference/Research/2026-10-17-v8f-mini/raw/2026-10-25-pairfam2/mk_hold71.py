"""Build qland/hold_v8f_pairfam2: the paired TACTICAL / STRATEGIC family tool (9 NEW, from S/pairfam2) + its first use (MIXO s0),
a closing note on RESULT_V8F_MIXO §5.3 / §5.4's work item, register note on H-V8F-MIXO-1. python mk_hold71.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "pairfam2/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_pairfam2"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RP = f"{PK}/raw/2026-10-25-pairfam2"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
NEW = ["stack/scripts/v8f_paired_families.py", "stack/tests/test_v8f_paired_families.py"] + [f"{RP}/{f}" for f in (
    "PAIRFAM_MIXO.md", "pairfam_mixo_s0.json", "pairfam_mixo_s0_tables.md", "pairfam_mixo_s0.out", "mutate_pairfam.py", "mutate_pairfam.out",
    "make_pairfam_mixo_md.py")]


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_pairfam2", "landed_hold_v8f_pairfam2"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for p in NEW:
    assert git("rev-parse", f"{TIP}:{p}") is None, p
    b = open(T + p, "rb").read()
    assert b"\r\n" not in b or p.endswith(".out"), f"CRLF in {p}"
    put("stage", p, b.replace(b"\r\n", b"\n"))
put("stage", f"{RP}/HANDOFF.md", open(T + "HANDOFF.md", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RP}/check_pairfam.py", open(SC + "check_pairfam.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RP}/mk_hold71.py",open(A2 + "mk_hold71.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RP}/MM_CHECK.md", (
    "# MM check of the paired tool (2026-10-25)\n\n"
    "Independent re-derivation of one strategic cell (S's 50 m checkpoint position error, X = M2b-mixo_s0 minus B = M2b_s0) with a hand-written\n"
    "episode-cluster bootstrap (`check_pairfam.py` beside this file, 2,000 resamples, its own RNG), outside `taniteval.ci` and outside the tool:\n"
    "n 2,319, 137 episodes, delta +0.0245 m (X 8.0368, B 8.0122), 95 % [-0.1468, +0.2039]. The tool: +0.0245 [-0.1547, +0.2015], n 2,319 / 137.\n"
    "Point and n identical; interval endpoints agree to resampling noise. `pytest stack/tests/test_v8f_paired_families.py`: 24 passed in 35.37s.\n").encode())

rm = f"{PK}/RESULT_V8F_MIXO.md"
old = git("show", f"{TIP}:{rm}")
assert old is not None and b"## ADDENDUM (2026-10-25, MM): paired tactical / strategic intervals" not in old
put("mods-rw", rm, old.replace(b"\r\n", b"\n").rstrip(b"\n") + (
    "\n\n## ADDENDUM (2026-10-25, MM): paired tactical / strategic intervals (the work item in §5.3 / §5.4, closed)\n\n"
    "`stack/scripts/v8f_paired_families.py` now computes paired X − B intervals for every tactical and strategic cell from the two eval dumps"
    " (paired episode-cluster bootstrap, B = 2,000, kappa and every F1 re-computed per resample). Run on X = `M2b-mixo_s0` vs B = `M2b_s0`"
    " (`raw/2026-10-25-pairfam2/PAIRFAM_MIXO.md`, `pairfam_mixo_s0.json`):\n"
    "* **No tactical or strategic cell is separated.** 37 paired intervals (31 tactical, 6 strategic); none excludes 0.\n"
    "* 8 sparse goal tokens carry a point difference only; 5 cells are UNAVAILABLE with their reasons (3 tokens with no positive label,"
    " anchor selection, selected-vs-executed).\n"
    "* The 16 point differences printed in §5.3 / §5.4 equal the paired read's.\n"
    "* Strategic n below 2,399 is now explained from the mask: 77 / 80 windows at 30 / 50 m have no valid checkpoint label (34 without a v9"
    " label row, the rest without a valid checkpoint at that arc).\n"
    "* Controls: X vs X reads exactly 0.0 on all 45 cells; per-arm values reproduce each eval JSON (119 checks per arm, 0 failed); 31 of 31"
    " mutants RED. MM re-derived the 50 m strategic cell by an independent bootstrap (`raw/2026-10-25-pairfam2/MM_CHECK.md`).\n\n"
    "One training seed, so this answers \"another draw of episodes\" only. Descriptive: no bar was registered on these cells, and the MIXO"
    " outcome (FAIL, O LEARNED) is unchanged.\n").encode("utf-8"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Paired tactical / strategic read added 2026-10-25** (`stack/scripts/v8f_paired_families.py`; `raw/2026-10-25-pairfam2/PAIRFAM_MIXO.md`):"
    " no tactical or strategic cell separated (37 paired intervals); descriptive, outcome unchanged. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("Paired tactical / strategic family intervals (v8f_paired_families.py) + first use on MIXO s0 (no cell separated)\n\n"
       "Closes RESULT_V8F_MIXO 5.3 / 5.4's work item: paired episode-cluster bootstrap for every tactical and strategic cell, from the eval dumps.\n"
       "24 tests, 31/31 mutants RED, identity 0.0 on 45 cells; MM re-derived one cell independently. Register: H-V8F-MIXO-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
