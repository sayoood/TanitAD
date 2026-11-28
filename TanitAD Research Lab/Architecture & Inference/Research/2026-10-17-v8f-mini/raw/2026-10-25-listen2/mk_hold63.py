"""Build qland/hold_v8f_listen2_result: RESULT_V8F_LISTEN2.md + raw/2026-10-25-listen2/ (READ, controls, per-window arrays, means, harness, logs) + register H-V8F-LISTEN2-1 status. python mk_hold63.py <tip>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
L = SC + "listen2/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_listen2_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-listen2"
TEXT = (".md", ".json", ".py", ".log", ".txt", ".sh", ".ps1")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, rel
    b = open(src, "rb").read()
    if src.endswith(TEXT):
        b = b.replace(b"\r\n", b"\n")
    put("stage", rel, b)


for n in ("hold_v8f_listen2_result", "landed_hold_v8f_listen2_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(L + "LISTEN2_RESULT.md", f"{PK}/RESULT_V8F_LISTEN2.md")
for f in sorted(os.listdir(L)):
    if os.path.isfile(L + f) and f != "LISTEN2_RESULT.md":
        stage(L + f, f"{RAW}/{f}")
for f in sorted(glob.glob(L + "runs/*")):
    stage(f, f"{RAW}/runs/{os.path.basename(f)}")
stage(A2 + "mk_hold63.py", f"{RAW}/mk_hold63.py")
put("stage", f"{RAW}/NOT_BANKED.txt", b"Not banked: code/ (66 MB), an export of LISTEN's tip (stack/, taniteval/), which is in git already.\n")

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LISTEN2-1** (2026-10-25) |")]
assert len(ix) == 1
old = "**PRE-REGISTERED, NOT RUN (2026-10-25).**"
assert lines[ix[0]].count(old) == 1
lines[ix[0]] = lines[ix[0]].replace(old, (
    "**E1 ROUTE FLOWS DOWN; E2 PARTLY ROBUST; E3 INCOHERENCE (MEASURED 2026-10-25).** T1-class open loop, DIAGNOSTIC; paired episode-cluster bootstrap,"
    " B = 2,000 (another draw of EPISODES; the two training seeds side by side). E1 (PRIMARY): a donor nav at S only drops T-lateral's macro-F1 by"
    " +0.442 [+0.353, +0.507] (s0, 2,107 windows / 139 episodes) and +0.419 [+0.349, +0.483] (s1), ~5x the donor-seed floor 0.0825; T's and O's own"
    " tokens bit-identical on 2,399 / 2,399 windows in every donor run. ⇒ nav → S → T-lateral is a working chain. E2: 3 of 4 rows keep the grid's"
    " argmax under training-mean replacement; O's does not (T code +4.15 vs ego +3.18 m at s0, overlapping) ⇒ O's ego-vs-T-code order is not robust."
    " E3: O with ego, S checkpoints and T code all from ONE donor is better than its worst single swap by −3.835 [−4.959, −2.766] / −3.174 [−4.234, −2.221] m"
    " ⇒ O is hurt by the contradiction between its inputs more than by a coherent foreign state. Controls: identity / own-value exactly 0.0;"
    " clean levels reproduce LISTEN to ≤ 9.5e-7; 14 / 14 source mutations RED. `RESULT_V8F_LISTEN2.md`, `raw/2026-10-25-listen2/READ.json`."))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("RESULT_V8F_LISTEN2: the route flows down the hierarchy (nav -> S -> T-lateral), O reacts to incoherent inputs\n\n"
       "E1 ROUTE FLOWS DOWN at both seeds (T-lateral F1 drop +0.44 / +0.42, ~5x the donor-seed floor); E2 PARTLY ROBUST (O's order flips);\n"
       "E3 INCOHERENCE at both seeds. Controls exact; 14/14 mutations RED. Register: H-V8F-LISTEN2-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
