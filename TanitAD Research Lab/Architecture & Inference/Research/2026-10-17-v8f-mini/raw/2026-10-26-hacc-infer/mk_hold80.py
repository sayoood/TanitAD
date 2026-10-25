"""Build qland/hold_v8f_hacc_infer: HACC's committed next step 1 -- the post-map folded into V8FMini.forward_infer (an optional hacc_w buffer,
ABSENT by default => bit-identical), its tests, mutation proof and suite comparison; register H-V8F-HACC-1 updated.  python mk_hold80.py <tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
HI = SC + "hacc_infer/"
X = HI + "export/"
H = "C:/Users/Admin/qland/hold_v8f_hacc_infer"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD = f"{PK}/raw/2026-10-26-hacc-infer"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def tipblob(rel):
    b = git("rev-parse", f"{TIP}:{rel}")
    return b.decode().strip() if b else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_hacc_infer", "landed_hold_v8f_hacc_infer"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
st = open(HI + "chain_status.txt", encoding="utf-8").read()
assert "CHAIN_DONE" in st, "the suite chain has not finished"
cmp_ = open(X + RD + "/suite_compare.txt", encoding="utf-8").read()
assert re.search(r"REGRESSIONS[^\n]*:\s*0\b", cmp_), "suite_compare.txt does not read 0 regressions"
n = 0
for ln in open(HI + "FILES.txt", encoding="utf-8"):
    if ln.startswith("#") or not ln.strip():
        continue
    f = ln.rstrip("\n").split("\t")
    st_, rel, nbytes, sha = f[:4]
    b = open(X + rel, "rb").read()
    assert len(b) == int(nbytes) and hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == sha, f"FILES.txt stale: {rel}"
    if st_ == "MODIFIED":
        assert tipblob(rel) == f[4], f"the tip moved under {rel}"
        put("mods-rw", rel, b)
    else:
        assert tipblob(rel) is None, rel
        put("stage", rel, b)
    n += 1
put("stage", f"{RD}/mk_hold80.py", open(SC + "grid_a2/mk_hold80.py", "rb").read().replace(b"\r\n", b"\n"))
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HACC-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Committed next step 1 DONE (2026-10-26):** the post-map is folded into `V8FMini.forward_infer` / `forward_diag` as an optional `hacc_w`"
    " [30, 2] buffer, ABSENT by default (bit-identical to the old code on both real checkpoints, 36 / 36 output tensors) and never used by `_forward`"
    " (training untouched); with the real weights it reproduces the banked HACC plans on all 2,399 windows to 7.8e-6 / 7.6e-6 m"
    " (`stack/tests/test_v8f_hacc_infer.py`, 17 tests; 18 / 18 mutants RED by assertion; suites `raw/2026-10-26-hacc-infer/suite_compare.txt`)."
    " Not yet: the weights baked into the deployed checkpoints and the map's provenance in the eval JSON. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("HACC post-map in forward_infer: optional hacc_w buffer, absent by default (bit-identical), reproduces the banked plans to 8e-6 m\n\n"
       "Inference only (never in _forward); validated weights; 17 tests, 18/18 mutants RED by assertion, suites compared on base/final.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "files")
