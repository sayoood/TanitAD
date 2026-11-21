"""Build qland/hold_v8f_r72_video21_speedavg: R72 (strategic family is UN-PAUSED for v8F; four documents/one video said PAUSED), video v2.1 (the fix + segment-5 S-age chart),
correction notes on RESULT_V8F_KINDX / RESULT_V8F_LISTEN2, register label fixes, and PREREG_V8F_SPEEDAVG (conditional, before data) + H-V8F-SPEEDAVG-1. python mk_hold68.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
V = SC + "viz2/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_r72_video21_speedavg"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RV = f"{PK}/raw/2026-10-25-video-v2"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def new(src, rel, text=True):
    assert show(TIP, rel) is None, rel
    b = open(src, "rb").read()
    put("stage", rel, lf(b) if text else b)


for n in ("hold_v8f_r72_video21_speedavg", "landed_hold_v8f_r72_video21_speedavg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# video v2.1 beside v2 (v2 stays as the record of what was first published)
new(V + "v8f_evidence_v2.mp4", f"{PK}/report/2026-10-25/v8f_evidence_v2_1.mp4", text=False)
for f in ("build_video_v2fix.log", "build_video_v2fix2.log"):
    new(V + f, f"{RV}/rebuild_v2_1/{f}")
for f in ("qa_rebuilt2_S_card.png", "qa_rebuilt2_closing.png", "qa_rebuilt_S_card.png"):
    new(V + f, f"{RV}/rebuild_v2_1/{f}", text=False)
put("stage", f"{RV}/rebuild_v2_1/NOTE.md", (
    "# Video v2.1 (2026-10-25, MM)\n\n`report/2026-10-25/v8f_evidence_v2_1.mp4` = v2 rebuilt from `scripts/v2_scenes.py` with:\n"
    "1. the three fixes VIDEO_NOTES.md section 0 lists (segment 5's S-age chart; the shortened segment-5 caption; the 23-of-24 wording);\n"
    "2. R72: segment 5 and the closing card said the STRATEGIC family was PAUSED (2026-09-19 ruling). PI-V8F-2026-10-17 un-paused it for v8F;\n"
    "   the cards now say so and state the family's reading (held S loses to the nav-only baseline at 50 m, +3.28 / +3.21 m; fresh S beats it).\n"
    "No number changed. v2 stays in place as the record of what was first published. Build logs and QA stills are in this folder.\n").encode())
put("mods-rw", f"{RV}/scripts/v2_scenes.py", lf(open(V + "v2_scenes.py", "rb").read()))

# correction notes (appended; nothing above them is edited)
CORR = {
    f"{PK}/RESULT_V8F_KINDX.md": "No number changes: the strategic block is bit-identical across every oracle arm by construction (the oracle replaces only the code O reads; S is the arm's own), so the family is reported as identical across arms, not as paused.",
    f"{PK}/RESULT_V8F_LISTEN2.md": "No number changes: the S row (S's checkpoint error under each intervention) is the strategic family's reading and is reported above; it is not paused.",
}
for rel, what in CORR.items():
    old = show(TIP, rel)
    assert old is not None and b"CORRECTION (2026-10-25, MM; R72)" not in old, rel
    add = ("\n\n## CORRECTION (2026-10-25, MM; R72)\n\nThe STRATEGIC family is NOT paused for v8F. `PI-V8F-2026-10-17` (register): \"R6 (strategic OFF) is superseded"
           " **for v8F**; the STRATEGIC family is un-paused for v8F\". The label above quoted the superseded 2026-09-19 ruling. " + what + "\n")
    put("mods-rw", rel, lf(old).rstrip(b"\n") + add.encode("utf-8"))

# R72
rl = "Project Steering/RETRACTION_LOG.md"
old = show(TIP, rl)
assert old is not None and b"### R72 " not in old and b"### R71 " in old
r72 = ("\n\n### R72 -- \"the STRATEGIC family is PAUSED\" (evidence video v2 segment 5 and closing card; the report's Video 2 caption; `RESULT_V8F_KINDX.md`;"
       " `RESULT_V8F_LISTEN2.md`) (2026-10-25, MM)\n\n"
       "**Wrong for v8F.** The 2026-09-19 pause (`PI_DECISION_QUEUE.md` item 25, quoted in CLAUDE.md) was superseded for this model line by `PI-V8F-2026-10-17`:"
       " \"R6 (strategic OFF) is superseded for v8F; the STRATEGIC family is un-paused for v8F\". MIXO's reader (`v8f_mixo_read.py`: `strategic_family = NOT PAUSED`)"
       " and the report body had it right; four documents and one video copied the older ruling from CLAUDE.md and the agent briefs that quote it.\n\n"
       "**Consequence:** no number changes. KINDX's strategic block is identical across its oracle arms by construction; LISTEN2's S row is the family's"
       " reading and was reported. Fixed: video v2.1 (`report/2026-10-25/v8f_evidence_v2_1.mp4`), correction notes appended to both results, register"
       " `D-V8F-VIDEO2-1`, the report's caption.\n\n"
       "**Class:** *a standing rule quoted from the always-loaded file after a later, scoped ruling in the register superseded it for this model line.*\n"
       "* ⇒ Before stating a family's status for an arm, check the register's PI rulings for that model line, not only CLAUDE.md.\n")
put("mods-rw", rl, lf(old).rstrip(b"\n") + r72.encode("utf-8"))

# PREREG_V8F_SPEEDAVG
new(A2 + "PREREG_V8F_SPEEDAVG.md", "Project Steering/PREREG_V8F_SPEEDAVG.md")
new(A2 + "mk_hold68.py", f"{PK}/raw/2026-10-25-speedavg/mk_hold68.py")

# register
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = show(TIP, g).decode("utf-8").split("\n")
iv = [k for k, x in enumerate(lines) if x.startswith("| **D-V8F-VIDEO2-1** (2026-10-25) |")]
assert len(iv) == 1
a = " Strategic family PAUSED (PI, 2026-09-19). |"
assert lines[iv[0]].count(a) == 1
lines[iv[0]] = lines[iv[0]].replace(a, " Strategic family reported in segment 5 (un-paused for v8F, `PI-V8F-2026-10-17`; v2 said PAUSED, corrected in v2.1, R72). |")
ik = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KINDX-1** (2026-10-25) |")]
assert len(ik) == 1 and not any(x.startswith("| **H-V8F-SPEEDAVG-1**") for x in lines)
row = ("| **H-V8F-SPEEDAVG-1** (2026-10-25) | T's speed error is variance: averaging the two seeds' T speed cells at inference (everything else each seed's own) cuts"
       " own-code ADE6 by at least F = 0.1207 m, separated, at BOTH seeds, without worsening speed / accel / distance keeping / LEAD-VIOL |"
       " **PRE-REGISTERED, CONDITIONAL, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_SPEEDAVG.md`. Runs only if KINDX's seed-1 read makes SPEED a"
       " carrier. Zero training. Outcomes: VARIANCE PAYS (distil into one T) / PARTIAL (a temporal average of T's own speed cells) / BIAS, NOT VARIANCE"
       " (a bias-targeting speed-head arm) / MIXED. |")
lines[ik[0] + 1:ik[0] + 1] = [row]
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("R72: the strategic family is un-paused for v8F (video v2.1, corrections); PREREG_V8F_SPEEDAVG (conditional, before data)\n\n"
       "Video v2.1 adds segment 5's S-age chart and corrects the PAUSED label (PI-V8F-2026-10-17); correction notes on KINDX / LISTEN2;\n"
       "R72 in RETRACTION_LOG. SPEEDAVG: is T's speed error variance? Runs only if speed is a carrier at seed 1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
