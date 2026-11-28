"""Hold: the LAUNCH-INTENT trainer flag `--launch-intent-w` (default 0.0 = off; every existing argv builds and trains bit-identically),
its tests and flag-off golden, the compose test's lever-key allowance, and the verification package. The PREREG is NOT promoted here.
python mk_hold_launch_intent_flag.py <live tip>"""
import hashlib
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
GIT = ["git", "-C", "C:/Users/Admin/tanitad-push", "-c", "safe.directory=*"]
P = "E:/Projects/TanitAD-artifacts/dagland/launch_intent/"
S = "F:/TanitAD-scratch/launch_intent/"
H = "C:/Users/Admin/qland/hold_launch_intent_flag/"
V8 = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PK = V8 + "raw/2026-10-28-launch-intent-flag/"
assert not os.path.exists(H), H


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

# the build base: the tip's files must be the ones the flag was built and verified on (d7208328)
BASE = {"stack/scripts/v8f_mini_train.py": "1c88828094d4", "stack/tanitad/models/v8f_mini.py": "8ec5e19037b6"}
for rel, h in BASE.items():
    assert sha(blob(rel)).startswith(h), (rel, "the tip moved since the flag was verified")
tipc = blob("stack/tests/test_v8f_ego_dagger_compose.py")
assert tipc == rd(S + "stopped_0445/test_v8f_ego_dagger_compose.tip.py"), "the compose test at the tip is not the one the allowance was patched from"
PIN = {"stack/scripts/v8f_mini_train.py": "0d708b3cb841", "stack/tanitad/models/v8f_mini.py": "b337ccecd3ab",
       "stack/tests/test_v8f_launch_intent.py": "1acc33d69980"}
for rel in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tests/test_v8f_ego_dagger_compose.py"):
    b = rd(P + "stage/" + rel)
    assert b"\r" not in b and b == rd(S + "tree/" + rel), rel
    if rel in PIN:
        assert sha(b).startswith(PIN[rel]), rel
    put("mods-rw/" + rel, b)
for rel in ("stack/tests/test_v8f_launch_intent.py", "stack/tests/launch_intent_golden.json"):
    b = rd(P + "stage/" + rel)
    assert b"\r" not in b and blob(rel) is None and b == rd(S + "tree/" + rel), rel
    if rel in PIN:
        assert sha(b).startswith(PIN[rel]), rel
    put("stage/" + rel, b)

# the verification package (the PREREG draft is NOT included: it is being redesigned and lands on promotion)
n = 0
for f in ("RESULTS.md", "mutation.txt", "diff.patch", "f_measure.json", "compare_rep_argv.out", "enumerate_cells_li.out",
          "enumerate_cells_li_mutation.txt", "MM_REDESIGN_2026-10-28.md", "mk_hold_launch_intent_flag.py"):
    b = rd(P + f)
    assert b and len(b) < 400_000, f
    put("stage/" + PK + f, b)
    n += 1
for f in ("final_green.log", "final_red_on_tip.log", "suite_test_v8f_ego_dagger_compose_fixed.log", "pristine_ego_dagger_compose.log",
          "pristine_flat.log", "suite_test_v8f_flat.log", "suite_test_v8f_mini.log", "suite_test_v8f_oacc.log"):
    put("stage/" + PK + "logs/" + f, rd(S + "logs/" + f))
    n += 1
for f in os.listdir(P + "tools"):
    if f.endswith((".py", ".sh")):
        put("stage/" + PK + "tools/" + f, rd(P + "tools/" + f))
        n += 1
put("msg.txt", ("LAUNCH-INTENT trainer flag --launch-intent-w (default 0.0 = off): O's along-axis, knots 1-12 ADE term on GT-launch train samples"
                " (v0 < 6 m/s and GT v(29->30) >= v0 + 1), with its G-DVB check and G-LIVE probe. Verified on the build: its tests 36 GREEN,"
                " 35 RED on the tip; mutation sweep 30 / 30 killed; flag OFF trains bit-identically (golden written by the unmodified tip;"
                " weights, logs, summary and checkpoint keys equal). The compose test's lever-key allowance widens by the four launch-intent"
                " record keys (the built definition is recorded, W 0.0 when off); the compose suite passes 36 / 36 on the build. Suites"
                " test_v8f_mini / test_v8f_oacc / test_v8f_flat fail one test each in the stack-only test tree for missing Research Lab"
                " files; test_v8f_flat fails identically on the unmodified tip. The PREREG is not promoted here (redesign of its bar and"
                " arms in progress, raw/2026-10-28-launch-intent-flag/MM_REDESIGN_2026-10-28.md). Package (%d files).\n\n"
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n" % n).encode())
print("hold written", H, n)
