"""Build qland/hold_v8f_kindx_s1_instr: the KINDX seed-1 box-2 instrument (3 NEW, 2 MODIFIED vs d3d314c7, from S/kindx_s1) + PREREG_V8F_KINDX Amendment 1
(the s1 outcome rows, committed before any s1 data) + register note. python mk_hold69.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "kindx_s1/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kindx_s1_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
BASE = "d3d314c7fd13e1c3a941720d9417d188cc2effc9"
NEW = [f"{PK}/code/box2_ops/chain_v8f_kindx_s1.sh", "stack/scripts/v8f_mini_eval_kindx.py", "stack/tests/test_v8f_kindx_s1.py"]
MOD = [f"{PK}/raw/2026-10-25-kindx/kindx_read.py", f"{PK}/raw/2026-10-25-kindx/kindx_controls.py"]


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def blob(rev, p):
    b = git("rev-parse", f"{rev}:{p}")
    return b.decode().strip() if b else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_kindx_s1_instr", "landed_hold_v8f_kindx_s1_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for p in NEW:
    assert blob(TIP, p) is None, p
    put("stage", p, open(T + p, "rb").read())
for p in MOD:
    assert blob(BASE, p) == blob(TIP, p), f"{p} moved on the tip since {BASE[:8]}"
    put("mods-rw", p, open(T + p, "rb").read())
put("stage", f"{PK}/raw/2026-10-25-kindx/mk_hold69.py", open(A2 + "mk_hold69.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{PK}/raw/2026-10-25-kindx/s1_instrument_HANDOFF.md", open(T + "HANDOFF.md", "rb").read().replace(b"\r\n", b"\n"))

pr = "Project Steering/PREREG_V8F_KINDX.md"
old = git("show", f"{TIP}:{pr}")
assert old is not None and b"Amendment 1" not in old
amend = ("\n\n## Amendment 1 (2026-10-25, MM): the seed-1 rows, committed BEFORE any seed-1 data\n\n"
         "The CANDIDATE row sends SPEED (gain_speed(X_s0) +0.716 [+0.623, +0.818] m, `RESULT_V8F_KINDX.md`) to a seed-1 read and names the CARRIER branch only."
         " The other s1 outcomes are fixed here, before X_s1 exists. The read is gain_speed(X_s1) on the same 2,399 windows, T0, the same estimator.\n\n"
         "| s1 outcome | condition | next (committed) |\n|---|---|---|\n"
         "| **CARRIER** | gain_speed(X_s1) ≥ 2F = 0.2415 m and separated | `PREREG_V8F_SPEEDAVG` runs (its condition is met) |\n"
         "| **NOT A CARRIER** | anything else | SPEED is not a carrier (a one-seed effect, `H-ESTIM-SEED-1`); `PREREG_V8F_SPEEDAVG` does not run; single-kind T-side arms stay unranked; the longitudinal line (KSCORE, then OACC) stays the top lever. Both seeds' gains are reported side by side |\n\n"
         "The instrument (`code/box2_ops/chain_v8f_kindx_s1.sh`, `stack/scripts/v8f_mini_eval_kindx.py`, `kindx_read.py --seed 1`) applies these two rows"
         " verbatim. X_s1 = MIXO's argv with `--seed 1` (and its own `--out`); the chain asserts that the argv differs from X_s0's in exactly those two flags.\n")
put("mods-rw", pr, old.replace(b"\r\n", b"\n").rstrip(b"\n") + amend.encode("utf-8"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KINDX-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Seed-1 instrument landed 2026-10-25** (`chain_v8f_kindx_s1.sh`; 188 tests, 11 / 11 mutations RED); PREREG Amendment 1 fixes the s1 rows"
    " (CARRIER → SPEEDAVG runs; NOT A CARRIER → the longitudinal line stays top) before any s1 data. Box 2 after KSCORE. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("KINDX seed-1 instrument (box-2 chain, eval driver, s1 reader) + PREREG_V8F_KINDX Amendment 1 (s1 rows, before data)\n\n"
       "Train X at seed 1 (MIXO's argv + --seed 1, behind its launch gate), 7 MIXO evals + 10 KINDX evals, controls, READ_s1.json.\n"
       "188 tests, 11/11 mutations RED. Register: H-V8F-KINDX-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
