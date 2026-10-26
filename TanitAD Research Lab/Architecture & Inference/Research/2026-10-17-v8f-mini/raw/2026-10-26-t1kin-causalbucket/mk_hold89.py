"""Build qland/hold_v8f_causalbucket_prereg: PREREG_V8F_T1KIN_CAUSALBUCKET (before any number with the causal bucket) and its register row
H-V8F-CAUSALBUCKET-1.  python mk_hold89.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
H = "C:/Users/Admin/qland/hold_v8f_causalbucket_prereg"
MIR = "C:/Users/Admin/tanitad-push"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert not TOKEN.search(data), rel
    if kind == "stage":
        assert git("rev-parse", f"{TIP}:{rel}") is None, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data.replace(b"\r\n", b"\n"))


for n in ("hold_v8f_causalbucket_prereg", "landed_hold_v8f_causalbucket_prereg"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
put("stage", "Project Steering/PREREG_V8F_T1KIN_CAUSALBUCKET.md", open(SC + "t1kin/PREREG_V8F_T1KIN_CAUSALBUCKET.md", "rb").read())
put("stage", "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-26-t1kin-causalbucket/mk_hold89.py",
    open(SC + "grid_a2/mk_hold89.py", "rb").read())
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-LAT-1** (2026-10-26) |")]
assert len(ix) == 1
assert not any(x.startswith("| **H-V8F-CAUSALBUCKET-1**") for x in lines)
lines.insert(ix[0] + 1, (
    "| **H-V8F-CAUSALBUCKET-1** (2026-10-26) | HACC's T1-kin margin over P10 survives a speed bucket built only from the past: with `--limit causal`"
    " (the sidecar's containing-window rule applied to the max measured ego speed over the last 4 s, executed after t0), HACC − P10 ADE6 at"
    " T1-kin m = 10 is ≤ −2F with the interval excluding 0, at BOTH seeds | **PRE-REGISTERED, NOT RUN (2026-10-26).**"
    " `Project Steering/PREREG_V8F_T1KIN_CAUSALBUCKET.md`, written before any number with the causal bucket. Why: with the bucket off, the margin"
    " is gone (+0.24 / +0.08 m, `RESULT_V8F_T1KIN_HACC.md`), and the logged bucket is the max of the ego's realised speed over [t0+2 s, +6 s]"
    " (read from the sidecar's meta). A present-but-causal bucket separates the bucket's information from the model's response to an absent"
    " input. Outcomes: CAUSAL KEEPS THE MARGIN / CAUSAL PARTIAL (⇒ train O on the causal bucket, box 2) / CAUSAL LOSES IT (⇒ the same, plus"
    " the perception-side speed lever) / MIXED. Zero training, dev-box CPU. |"))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(
    b"PREREG_V8F_T1KIN_CAUSALBUCKET (before any number): is HACC's T1-kin margin the oracle speed bucket's? Register H-V8F-CAUSALBUCKET-1\n\n"
    b"Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
print("hold ok")
