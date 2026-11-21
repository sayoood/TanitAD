"""Build qland/hold_v8f_hacc_prereg: PREREG_V8F_HACC (before any HACC number) + the descriptive accel profile that motivates it
(raw/2026-10-25-hacc/accel_profile.{py,log,json}) + register row H-V8F-HACC-1.  python mk_hold78.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
H = "C:/Users/Admin/qland/hold_v8f_hacc_prereg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RK = f"{PK}/raw/2026-10-25-hacc"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def new(src, rel):
    assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))


for n in ("hold_v8f_hacc_prereg", "landed_hold_v8f_hacc_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
new(SC + "hacc/PREREG_V8F_HACC.md", "Project Steering/PREREG_V8F_HACC.md")
for f in ("accel_profile.py", "accel_profile.log", "accel_profile.json"):
    new(SC + "hspec/" + f, f"{RK}/{f}")
put("stage", f"{RK}/mk_hold78.py", open(SC + "grid_a2/mk_hold78.py", "rb").read().replace(b"\r\n", b"\n"))
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-SPEEDCAL-1** (2026-10-25) |")]
assert len(ix) == 1 and not any(x.startswith("| **H-V8F-HACC-1**") for x in lines)
lines.insert(ix[0] + 1, (
    "| **H-V8F-HACC-1** (2026-10-25) | H's accel / curvature gap to P10 is mostly an artefact of HBLEND's monotone per-step weight: a per-step"
    " weight fitted on TRAINING windows with an added second-difference (accel) term and no monotone constraint makes the blended plan dominate"
    " P10 on HSPEC's row (ADE2, ADE6, accel MAE, curvature MAE separated-better, nothing separated-worse) at BOTH seeds, keeping ADE6 within F/2"
    " of H | **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_HACC.md`, written before any HACC number. Zero training,"
    " dev-box CPU. Motivation (MEASURED, descriptive, `raw/2026-10-25-hacc/accel_profile.*`, T1-class open loop): H's accel gap to P10 sits in"
    " three bands, 2.2–2.6 s (where HBLEND's along weight ramps 0.12 → 0.50 and H is worse than both B and P10), 5.2–5.8 s (B's rough tail kept"
    " by the isotonic maximum) and 0.8–1.6 s; on 2.8–5.0 s H already beats P10 on accel. Outcomes: DOMINATES P10 (deploy the 30 × 2 weights;"
    " re-render the videos; T1-kin read needs a PI go) / SMOOTHER, NOT DOMINANT (OACC + HACC on its checkpoint) / NO SMOOTHING (OACC is the"
    " lever) / MIXED. |"))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_HACC before any number: per-step blend weight with an accel term, no monotone constraint (zero training)\n\n"
       "Motivated by a descriptive per-step accel profile (reproduces the banked family metric to float32 epsilon): H's accel gap\n"
       "to P10 sits at 2.2-2.6 s (HBLEND's weight ramp), 5.2-5.8 s (B's rough tail) and 0.8-1.6 s; on 2.8-5.0 s H already beats P10.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
