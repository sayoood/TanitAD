"""Hold: PREREG_V8F_SPEEDSUB_T1 Amendment 1 (mods-rw, appended before any rollout) + its instrument (fit, wrapper, pin step and their tests in
stack/; the reader, the chain, the probes and the evidence under raw/2026-10-28-speedsub-t1-build/).
python mk_hold_speedsub_t1_build.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
D = "E:/Projects/TanitAD-artifacts/dagland/speedsub_t1/"
B = D + "build/"
H = "C:/Users/Admin/qland/hold_speedsub_t1_build/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-speedsub-t1-build/"
assert not os.path.exists(H), H
BANNED = (b"void", b"superseded", b"retracted", b"formerly", b"no longer", b"correction", b"withdrawn")


def blob(p):
    r = subprocess.run(GIT + ["show", f"{TIP}:{p}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(rel, b):
    assert len(b) > 0 and len(H + rel) < 260, (rel, len(H + rel))
    assert not re.search(rb"hf_[A-Za-z0-9]{8,}", b), rel
    os.makedirs(os.path.dirname(H + rel), exist_ok=True)
    open(H + rel, "wb").write(b)


rd = lambda p: open(p, "rb").read()
sha = lambda b: hashlib.sha256(b).hexdigest()

# ---- Amendment 1 (the body stays as promoted; only the amendments line is replaced by the appended amendment)
PR = "Project Steering/PREREG_V8F_SPEEDSUB_T1.md"
cur = blob(PR)
assert sha(cur) == "5edf6bc0acf86192e271721d064c835e4ed5f22a31a1ee213f7ac7fc248d95c2", "the tip's PREREG is not the promoted text"
new = rd(D + "PREREG_V8F_SPEEDSUB_T1.amended.md")
assert sha(new) == "69187967e82bf64dac85bf9ae81810b6f8aac7161cd2558d719d45cdf61fd861" and b"\r" not in new
assert new.startswith(cur[:-len(b"* Amendments: none.\n")]) and cur.endswith(b"* Amendments: none.\n")
assert new == rd("F:/TanitAD-scratch/speedsub_t1/ref/PREREG_V8F_SPEEDSUB_T1.md")
for w in BANNED:
    assert w not in new[len(cur) - 20:].lower(), w
put("mods-rw/" + PR, new)

# ---- stack files (new)
STACK = ["stack/scripts/v8f_speedsub_t1_fit.py", "stack/scripts/v8f_t1_speedsub.py", "stack/scripts/v8f_speedsub_t1_pin.py",
         "stack/tests/test_v8f_t1_speedsub.py", "stack/tests/mutation_speedsub_wrapper.py", "stack/tests/test_v8f_speedsub_t1_pin.py",
         "stack/tests/mutation_speedsub_pin.py"]
for rel in STACK:
    b = rd(B + rel)
    assert b"\r" not in b and blob(rel) is None, rel
    put("stage/" + rel, b)

# ---- the package
rdr = rd(B + "read/speedsub_t1_read.py")
assert b'PREREG_SHA256 = "69187967e82bf64dac85bf9ae81810b6f8aac7161cd2558d719d45cdf61fd861"' in rdr
n = 0
for dp, _, fs in os.walk(B):
    for f in sorted(fs):
        pth = os.path.join(dp, f).replace("\\", "/")
        rel = pth[len(B):]
        if rel.startswith("stack/") or "__pycache__" in rel or f.endswith((".pyc", ".pt", ".npz")):
            continue
        b = rd(pth)
        if rel.endswith((".py", ".sh", ".md")):
            assert b"\r" not in b, rel
        assert b and len(b) < 400_000, (rel, len(b))
        assert blob(PK + rel) is None, rel
        put("stage/" + PK + rel, b)
        n += 1
for extra in ("AMEND1.md", "speedsub_t1_read.pre_amend1.py", "mk_hold_speedsub_t1_build.py"):
    put("stage/" + PK + "mm/" + extra, rd(D + extra))
    n += 1
put("msg.txt", ("PREREG_V8F_SPEEDSUB_T1 Amendment 1 (A1-2026-10-28, appended before the smoke and before any rollout): guard G-VIOL (the"
                " executed path's rule-violation rate, egoc - identity, must not be separated-worse at >= 3 of 4 runs; motivated by"
                " RESULT_V8F_SPEEDSUB section 4.2), the tick-frame, G-NEW and smoke-control readings, the C-FIT tolerance class. Its"
                " instrument: mlp_ego_c fitted on the causal speed bucket (readout_c.pt sha256 799f65ed..., not in git; C-FIT 9.412e-6 <= 1e-5,"
                " C-CAUSAL 0 mismatches, C-COVER 306,566 / 306,566), the forward_infer wrapper around the pinned T1 tool (33 tests,"
                " 29 / 29 mutants killed), the pin step (52 tests, 25 / 25), the reader (73 tests, 110 / 110 on the pre-amendment text;"
                " the amendment changes its PREREG pin and one checked sentence; dry run reproduces the banked DAG-OACC Read-A table on"
                " 38 / 38 rows exactly) and the chain (dry run ARGCHECK 18 / 18, self-test 14 / 14). Nothing of this experiment has been"
                " rolled out. Package raw/2026-10-28-speedsub-t1-build/ (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n, "package files")
