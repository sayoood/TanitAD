"""Build qland/hold_v8f_speedavg_result: SPEEDAVG's real run (MIXED) banked under raw/2026-10-25-speedavg/run/ (reads, controls, evals,
speed-cell dumps, logs, the paired tactical / strategic output + its substitute reproduction control), RESULT_V8F_SPEEDAVG.md,
PREREG_V8F_SPEEDCAL.md (written before any SPEEDCAL number), register H-V8F-SPEEDAVG-1 (MIXED) + new row H-V8F-SPEEDCAL-1.
python mk_hold76.py <tip>"""
import glob
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
H = "C:/Users/Admin/qland/hold_v8f_speedavg_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RK = f"{PK}/raw/2026-10-25-speedavg/run"
SA = "C:/Users/Admin/sa_run/"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def new(src, rel, text=True):
    assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    b = open(src, "rb").read()
    put("stage", rel, b.replace(b"\r\n", b"\n") if text else b)


for n in ("hold_v8f_speedavg_result", "landed_hold_v8f_speedavg_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
assert b"%%" not in open(SC + "speedavg_res/RESULT_V8F_SPEEDAVG.md", "rb").read(), "unfilled placeholder in the RESULT"
for f in ("READ.json", "READ.md", "READ_controls.json", "check.json", "ckpt_paths.txt", "run_speedavg_all.sh", "run_speedavg_all.cmd"):
    new(SA + f, f"{RK}/{f}", text=not f.endswith(".cmd"))
for p in sorted(glob.glob(SA + "logs/*")):
    new(p, f"{RK}/logs/{os.path.basename(p)}")
for p in sorted(glob.glob(SA + "eval/*")):
    n = os.path.basename(p)
    new(p, f"{RK}/eval/{n}", text=not n.endswith(".npz"))
for p in sorted(glob.glob(SA + "pairfam/*")):
    n = os.path.basename(p)
    new(p, f"{RK}/pairfam/{n}")
put("stage", f"{RK}/mk_hold76.py", open(SC + "grid_a2/mk_hold76.py", "rb").read().replace(b"\r\n", b"\n"))

new(SC + "speedavg_res/RESULT_V8F_SPEEDAVG.md", f"{PK}/RESULT_V8F_SPEEDAVG.md")
r = open(SC + "speedavg_res/RESULT_V8F_SPEEDAVG.md", "rb").read()
assert b"%%" not in r, "unfilled placeholder in the RESULT"
new(SC + "speedavg_res/PREREG_V8F_SPEEDCAL.md", "Project Steering/PREREG_V8F_SPEEDCAL.md")

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-SPEEDAVG-1** (2026-10-25) |")]
assert len(ix) == 1 and not any(x.startswith("| **H-V8F-SPEEDCAL-1**") for x in lines)
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **MEASURED 2026-10-25: MIXED** (`RESULT_V8F_SPEEDAVG.md`, `raw/2026-10-25-speedavg/run/`; T1-class open loop, paired episode-cluster"
    " bootstrap, 141 episodes). PRIMARY ADE6 −0.0414 [−0.0730, −0.0110] (s0, separated) / +0.0189 [−0.0116, +0.0483] (s1); bar ≤ −F at both"
    " seeds not met; SECONDARY fails at s1 (accel MAE +0.0169, separated). Controls PASS. Mechanism (descriptive, no bar): the two seeds'"
    " speed-cell ERRORS correlate 0.71–0.85, so T's speed error is systematic, not variance. Next lever: PREREG_V8F_SPEEDCAL (H-V8F-SPEEDCAL-1). |")
lines.insert(ix[0] + 1, (
    "| **H-V8F-SPEEDCAL-1** (2026-10-25) | T's speed error is a head bias: an affine map per speed cell (5 cells, 10 numbers per seed), fitted on"
    " HBLEND's TRAINING windows only and applied to T's own output at inference, cuts own-code ADE6 by at least F = 0.1207 m, separated, at BOTH"
    " seeds, without worsening speed / accel / distance keeping / LEAD-VIOL | **PRE-REGISTERED, NOT RUN (2026-10-25).**"
    " `Project Steering/PREREG_V8F_SPEEDCAL.md`, written before any SPEEDCAL number. Zero training, dev-box CPU. Outcomes: HEAD BIAS PAYS"
    " (calibration joins `forward_infer`) / PARTIAL BIAS (isotonic calibration) / MAP DOES NOT TRANSFER (fit on all 4,512 clips) /"
    " INFORMATION-LIMITED (the lever moves to perception: PI go) / MIXED. CEILING (eval-fitted map) is a T0 diagnostic only. |"))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("SPEEDAVG: MIXED -- the two seeds make the same speed errors (corr 0.71-0.85); PREREG_V8F_SPEEDCAL before any calibration number\n\n"
       "PRIMARY ADE6 -0.041 (s0, separated) / +0.019 (s1); the both-seed bar is not met and s1's accel MAE worsens. T1-class open loop,\n"
       "paired episode-cluster bootstrap, controls PASS. Systematic, not variance: next lever is a train-fitted affine speed-cell map.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
