"""Build qland/hold_v8f_t1kin_prereg: PREREG_V8F_T1KIN_HACC (before any data), register row H-V8F-T1KIN-HACC-1, a note on D-V8F-T1KIN-1,
and D-V8F-HACC-T1KIN marked MOOT (the eval cache is on the dev box, so the read needs no box-2 slot).  python mk_hold84.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
H = "C:/Users/Admin/qland/hold_v8f_t1kin_prereg"
MIR = "C:/Users/Admin/tanitad-push"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_t1kin_prereg", "landed_hold_v8f_t1kin_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
pre = "Project Steering/PREREG_V8F_T1KIN_HACC.md"
assert git("rev-parse", f"{TIP}:{pre}") is None
put("stage", pre, open(SC + "t1kin/PREREG_V8F_T1KIN_HACC.md", "rb").read().replace(b"\r\n", b"\n"))
put("stage", "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-t1kin-hacc/mk_hold84.py",
    open(SC + "grid_a2/mk_hold84.py", "rb").read().replace(b"\r\n", b"\n"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
assert not any(x.startswith("| **H-V8F-T1KIN-HACC-1**") for x in lines)
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-1** (2026-10-25) |")]
assert len(ix) == 1
row = ("| **H-V8F-T1KIN-HACC-1** (2026-10-26) | HACC's margin over the P10 floor survives self-action: at T1-kin (own executed kinematics,"
       " cameras logged, nav dead-reckoned, replan every m = 10 steps), HACC − P10 6 s ADE of the executed track is ≤ −2F = −0.2415 m with the"
       " interval excluding 0 at BOTH seeds (DOMINANCE adds: no `analyze()` row separated-worse) | **PRE-REGISTERED, NOT RUN (2026-10-26).**"
       " `Project Steering/PREREG_V8F_T1KIN_HACC.md`, written before any real-camera T1-kin number for any v8F-mini checkpoint. Tier T1"
       " (self-action OPEN loop; never \"closed loop\"). Arms: the baked `M2b_s{0,1}_hacc.pt` vs the floor driven by the same executor; plain"
       " M2b beside. Controls first: m = 30 reproduces open loop and the banked floor (2.48699 m), the post-map is in the loop. Brackets in the"
       " same read: `--scene frozen`, `--limit off`. Outcomes: T1 DOMINATES / T1 BEATS THE FLOOR / MARGIN COLLAPSES / MIXED. Venue: the"
       " dev-box CPU, because the pooled eval cache is on the dev box (`scratchpad/sfresh/cache/eval`, 141 / 141 clips, md5-verified);"
       " box 2's queue is unchanged. |")
lines.insert(ix[0] + 1, row)
jx = [k for k, x in enumerate(lines) if x.startswith("| **D-V8F-T1KIN-1** (2026-10-22) |")]
assert len(jx) == 1
lines[jx[0]] = lines[jx[0]].rstrip().rstrip("|").rstrip() + (
    " **UNBLOCKED (2026-10-26, MEASURED):** the pooled eval cache this row waited for is on the dev box (`scratchpad/sfresh/cache/eval`,"
    " pipeline `ra1a-pooled-v1`, 141 / 141 clips, 0 md5 mismatches), so the model-vs-floor read runs on the dev-box CPU:"
    " `PREREG_V8F_T1KIN_HACC.md` (`H-V8F-T1KIN-HACC-1`). |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))

q = "Project Steering/PI_DECISION_QUEUE.md"
qt = git("show", f"{TIP}:{q}").decode("utf-8")
old = "## ⏳ OPEN (2026-10-26 ~00:15 Berlin) -- D-V8F-HACC-T1KIN: run HACC's T1-kin read on box 2, before or after OACC?\n"
assert qt.count(old) == 1
new = ("## ✅ MOOT (MM, 2026-10-26 ~00:45 Berlin; filed ~00:15) -- D-V8F-HACC-T1KIN: run HACC's T1-kin read on box 2, before or after OACC?\n\n"
       "**Why moot (MEASURED):** the pooled eval cache is already on the dev box (`scratchpad/sfresh/cache/eval`, 141 / 141 clips, md5-verified),"
       " so the read needs no box-2 slot. It runs on the dev-box CPU inside the standing allocation (`PI-V8F-2026-10-25d`), pre-registered as"
       " `PREREG_V8F_T1KIN_HACC.md`. Box 2's queue (FLAT, then OACC) is unchanged. The question as filed is kept below.\n")
put("mods-rw", q, qt.replace(old, new).encode("utf-8"))
msg = ("PREREG_V8F_T1KIN_HACC (before any data): HACC vs the P10 floor at T1-kin, m = 10, both seeds; dev-box CPU (the eval cache is there)\n\n"
       "Register H-V8F-T1KIN-HACC-1 (pre-registered, not run); D-V8F-T1KIN-1 unblocked; PI item D-V8F-HACC-T1KIN moot (no box-2 slot needed).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
