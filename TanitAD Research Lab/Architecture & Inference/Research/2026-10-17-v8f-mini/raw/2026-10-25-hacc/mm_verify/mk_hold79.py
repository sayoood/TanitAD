"""Build qland/hold_v8f_hacc_result: PREREG_V8F_HACC's read -- DOMINATES P10 at both seeds (T1-class open loop) -- RESULT_V8F_HACC.md (+ the MM
independent re-derivation), raw/2026-10-25-hacc/run/ (every file the agent's FILES.txt lists, blob-checked), raw/2026-10-25-hacc/mm_verify/,
register H-V8F-HACC-1.  python mk_hold79.py <tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
A = SC + "hacc/"
H = "C:/Users/Admin/qland/hold_v8f_hacc_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RK = f"{PK}/raw/2026-10-25-hacc"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    assert git("rev-parse", f"{TIP}:{rel}") is None or kind == "mods-rw", rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_hacc_result", "landed_hold_v8f_hacc_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for ln in open(A + "FILES.txt", encoding="utf-8"):
    if ln.startswith("#") or not ln.strip():
        continue
    path, nbytes, sha = ln.rstrip("\n").split("\t")
    b = open(path, "rb").read()
    assert len(b) == int(nbytes) and hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == sha, f"FILES.txt stale: {path}"
    rel = os.path.relpath(path, A).replace("\\", "/")
    assert not rel.startswith(".."), path
    put("stage", f"{RK}/run/{rel}", b if rel.endswith((".npz", ".json")) else b.replace(b"\r\n", b"\n"))
    n += 1
for f in ("verify_mm.py", "verify_mm.log", "verify_mm.json", "ADDENDUM_MM.md"):
    put("stage", f"{RK}/mm_verify/{f}", open(SC + "hacc_mm/" + f, "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RK}/mm_verify/mk_hold79.py", open(SC + "grid_a2/mk_hold79.py", "rb").read().replace(b"\r\n", b"\n"))
res = open(A + "RESULT_V8F_HACC.md", "rb").read().replace(b"\r\n", b"\n").rstrip(b"\n") + b"\n" + open(SC + "hacc_mm/ADDENDUM_MM.md", "rb").read().replace(b"\r\n", b"\n")
put("stage", f"{PK}/RESULT_V8F_HACC.md", res)

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **MEASURED 2026-10-25: DOMINATES P10 at BOTH seeds** (`RESULT_V8F_HACC.md`, `raw/2026-10-25-hacc/`; T1-class open loop, own held outputs,"
    " NOT closed loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes). HACC − P10-yaw-4512: ADE2 −0.0106 / −0.0083, ADE6 −0.503 / −0.471,"
    " accel MAE −0.0215 [−0.0345, −0.0087] / −0.0402 [−0.0509, −0.0293] m/s², curvature MAE −0.0056 / −0.0051, all separated-better; speed, along,"
    " cross-track, heading, yaw rate, both plan-derived decision agreements and CONSISTENCY separated-better too; distance keeping not separated;"
    " nothing separated-worse. KEEP (ADE6 HACC − H, upper bound ≤ +F/2 = +0.0604) passes: +0.0320 (s0) / +0.0586 (s1, a thin margin; HACC is"
    " separated-worse than H by +0.031 m there). λ\* by the train-only rule: m = 4 (s0) / 16 (s1). Controls 1–7 PASS; 15/15 mutants RED by"
    " assertion. The Master Mind re-derived it independently (different solver, own bootstrap): w to 1.5e-5, every interval to the bootstrap draw."
    " The baseline is the programme's kinematic floor, not the state of the art. Committed next (verbatim): fold the 30 × 2 weights into"
    " `forward_infer` with tests; re-render the evidence videos on HACC; the T1-kin closed-loop read on HACC needs a PI go. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("HACC: DOMINATES P10 at both seeds -- first full dominance row over the kinematic prior (T1-class open loop, zero training)\n\n"
       "HACC - P10: ADE6 -0.50 / -0.47 m, accel MAE -0.022 / -0.040, curvature -0.0056 / -0.0051, every other row better or not\n"
       "separated, nothing worse; KEEP vs H passes (s1 thin: +0.0586 vs +0.0604). Train-only lambda rule; controls PASS; 15/15 mutants\n"
       "RED; independently re-derived (different solver, own bootstrap). Open loop, not closed loop; the baseline is P10, not SOTA.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "run files")
