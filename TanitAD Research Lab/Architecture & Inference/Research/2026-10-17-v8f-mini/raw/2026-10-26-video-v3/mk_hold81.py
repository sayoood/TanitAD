"""Build qland/hold_v8f_video3: evidence video v3 (HACC's committed next step 2) banked under raw/2026-10-26-video-v3/ from viz3/FILES.txt
(blob-checked: mp4, controls, notes, sources, QA stills); register H-V8F-HACC-1 notes it.  python mk_hold81.py <tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
V = SC + "viz3/"
H = "C:/Users/Admin/qland/hold_v8f_video3"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RV = f"{PK}/raw/2026-10-26-video-v3"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")
BIN = (".mp4", ".png", ".npz", ".json")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    if not rel.endswith((".mp4", ".png", ".npz")):
        assert not TOKEN.search(data), f"token-shaped string in {rel}"
    if kind == "stage":
        assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_video3", "landed_hold_v8f_video3"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for ln in open(V + "FILES.txt", encoding="utf-8"):
    if ln.startswith("#") or not ln.strip():
        continue
    path, nbytes, sha = ln.rstrip("\n").split("\t")
    b = open(path, "rb").read()
    assert len(b) == int(nbytes) and hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == sha, f"FILES.txt stale: {path}"
    rel = os.path.relpath(path, V).replace("\\", "/")
    assert not rel.startswith("..")
    put("stage", f"{RV}/{rel}", b if rel.endswith(BIN) else b.replace(b"\r\n", b"\n"))
    n += 1
put("stage", f"{RV}/mk_hold81.py", open(SC + "grid_a2/mk_hold81.py", "rb").read().replace(b"\r\n", b"\n"))
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Committed next step 2 DONE (2026-10-26):** evidence video v3 (`raw/2026-10-26-video-v3/v8f_evidence_v3.mp4`, 216 s): HACC drawn on the"
    " four pre-declared windows and a dominance chapter (the 15-row table, both seeds; where the accel gap went: the 0.8–1.6 s and 2.2–2.6 s bumps"
    " gone, the 5.2–5.8 s bump reduced, not gone: HACC still +0.233 / +0.127 m/s² above P10 there). Controls PASS (`CONTROLS_V3.json`: HACC as drawn"
    " = the banked plan, 0.0 m; 1,089 on-screen numbers = their JSON; an independent bootstrap reproduces the four primary rows); the strategic"
    " captions corrected to IN FORCE for v8F. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("Evidence video v3: HACC on the pre-declared windows + a dominance-over-P10 chapter (controls PASS; T1-class open loop)\n\n"
       "216 s, 1080p. HACC as drawn = the banked plans (0.0 m); every on-screen number = its JSON source; strategic captions corrected\n"
       "(in force for v8F). The 5.2-5.8 s accel bump is reduced, not gone, and the video says so.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "files")
