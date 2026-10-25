"""Build qland/hold_v8f_kscore_result: RESULT_V8F_KSCORE (NOT PREDICTABLE) + the banked run, the generator's strategic-label fix (R73) with its
test, a correction note on RESULT_V8F_KHYP, R73 in RETRACTION_LOG, register H-V8F-KSCORE-1. python mk_hold70.py <tip>"""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
KR = SC + "kscore_res/"
KF = SC + "kfix/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kscore_result"
MIR = "C:/Users/Admin/tanitad-push"
BASE = "4f561a2c1942a41be1150aca9e5a6e3f0748fe79"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-kscore/run"
LOGS = "C:/Users/Admin/box2_ship/logs/"
KS = LOGS + "v8f_kscore/"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def blob(rev, p):
    b = git("rev-parse", f"{rev}:{p}")
    return b.decode().strip() if b else None


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def new(src, rel, text=True):
    assert blob(TIP, rel) is None, rel
    b = open(src, "rb").read()
    put("stage", rel, lf(b) if text else b)


for n in ("hold_v8f_kscore_result", "landed_hold_v8f_kscore_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# the result + the banked run
new(KR + "RESULT_V8F_KSCORE.md", f"{PK}/RESULT_V8F_KSCORE.md")
for f in ("READ.json", "KSCORE_RESULT.md", "kscore_numbers.json", "telemetry_v8fkscore.csv"):
    new(KS + f, f"{RUN}/{f}")
for f in ("chain_v8f_kscore.out", "chain_v8f_kscore.driver_died_1530.out", "chain_v8f_kscore.stop1_guardcopy.out", "v8fkscore_guard.out"):
    new(LOGS + f, f"{RUN}/{f}")
sha = []
for s in ("s0", "s0_perm", "s1", "s1_perm"):
    for f in ("argv.json", "summary.json", "train_log.jsonl", "gate_token.json", "selfcheck.json", "sel_const.json"):
        new(KS + f"fit/{s}/{f}", f"{RUN}/fit/{s}/{f}")
    new(KS + f"fit/{s}/ckpt_last.pt", f"{RUN}/fit/{s}/ckpt_last.pt", text=False)
    import hashlib
    sha.append(f"{hashlib.sha256(open(KS + f'fit/{s}/ckpt_last.pt', 'rb').read()).hexdigest()}  fit/{s}/ckpt_last.pt")
put("stage", f"{RUN}/CKPT_SHA256.txt", ("\n".join(sha) + "\n").encode())
new(KR + "label_fix_tests.sh", f"{RUN}/label_fix_tests.sh")
new(KR + "label_fix_tests.out", f"{RUN}/label_fix_tests.out")
put("stage", f"{RUN}/mk_hold70.py", lf(open(A2 + "mk_hold70.py", "rb").read()))

# the generator's label fix + its test (both unchanged on the tip since BASE)
for p in ("stack/scripts/v8f_kscore.py", "stack/tests/test_v8f_kscore.py"):
    assert blob(BASE, p) == blob(TIP, p), f"{p} moved on the tip since {BASE[:8]}"
    b = open(KF + p, "rb").read()
    assert b"PAUSED (PI ruling 2026-09-19" not in b or p.endswith("test_v8f_kscore.py"), p
    put("mods-rw", p, b)

# correction note on RESULT_V8F_KHYP
rk = f"{PK}/RESULT_V8F_KHYP.md"
old = git("show", f"{TIP}:{rk}")
assert old is not None and b"CORRECTION (2026-10-25, MM; R73)" not in old
put("mods-rw", rk, lf(old).rstrip(b"\n") + (
    "\n\n## CORRECTION (2026-10-25, MM; R73)\n\nThe STRATEGIC family is NOT paused for v8F. `PI-V8F-2026-10-17` (register): \"R6 (strategic OFF) is superseded"
    " **for v8F**; the STRATEGIC family is un-paused for v8F\". The two family tables and the bullet above quoted the superseded 2026-09-19 ruling."
    " No number changes: every selector here deploys one of O's plans with S as B's own, so the family is identical across arms by construction."
    " The banked generator (`raw/2026-10-25-khyp/khyp.py`) and its outputs stay as written.\n").encode("utf-8"))

# R73
rl = "Project Steering/RETRACTION_LOG.md"
old = git("show", f"{TIP}:{rl}")
assert old is not None and b"### R73 " not in old and b"### R72 " in old
r73 = ("\n\n### R73 -- \"the STRATEGIC family is PAUSED\", again: `RESULT_V8F_KHYP.md` (two family tables and a bullet) and KSCORE's own output"
       " (`raw/2026-10-25-kscore/run/KSCORE_RESULT.md`, `READ.json` -> `strategic_family`) (2026-10-25, MM)\n\n"
       "**Wrong for v8F**, as R72. R72 fixed four documents and a video. Its sweep searched the DOCUMENTS, not the GENERATORS that write them:"
       " `raw/2026-10-25-khyp/khyp.py` and `stack/scripts/v8f_kscore.py` hard-coded the label. So KHYP's result (written before R72) was missed,"
       " and KSCORE (run after R72) printed the label again.\n\n"
       "**Consequence:** no number changes. Both are B-frozen selector reads with S as B's own, so the family is identical across arms by"
       " construction. Fixed:\n"
       "* `v8f_kscore.py`'s label now reads `NOT PAUSED for v8F`. `test_v8f_kscore.py` pins it and requires the old ruling's text to be absent;"
       " restoring the old label goes RED (`raw/2026-10-25-kscore/run/label_fix_tests.out`).\n"
       "* A correction note on `RESULT_V8F_KHYP.md`.\n"
       "* `RESULT_V8F_KSCORE.md` states the fix.\n\n"
       "`khyp.py` and the banked run outputs stay as written: they are records. Sweep of the live generators on tip `4f561a2c`: `grep -i paused` over"
       " `stack/scripts/v8f*`, the package `code/`, and the live readers in `raw/2026-10-25-{kindx,listen2,mixo}` finds no other hit"
       " (`v8f_mini_eval.py:94` writes `\"paused\": False`).\n\n"
       "**Class:** *a correction applied to a generator's outputs, and not to the generator, re-emits on the generator's next run.*\n"
       "* ⇒ When retracting a label, grep the code that writes it, not only the documents that carry it.\n")
put("mods-rw", rl, lf(old).rstrip(b"\n") + r73.encode("utf-8"))

# register
g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KSCORE-1** (2026-10-25) |")]
assert len(ix) == 1
a = "**PRE-REGISTERED, NOT RUN (2026-10-25).**"
assert lines[ix[0]].count(a) == 1
lines[ix[0]] = lines[ix[0]].replace(a, (
    "**MEASURED 2026-10-25: NOT PREDICTABLE** (`RESULT_V8F_KSCORE.md`; open loop on B's own held outputs, paired episode-cluster bootstrap, 141 episodes)."
    " PRIMARY-B fails at both seeds: −0.1001 [−0.1463, −0.0583] / +0.0036 [−0.0273, +0.0318] m, against ≤ −F."
    " PRIMARY-SEL fails: KSCORE − SEL-CONST −0.0005 / −0.0072 m, not separated."
    " The s0 gain is the constant class 5 (picked on 86.6 % of windows; CONST − B −0.0996 at s0 and +0.0108 at s1)."
    " s0 also worsens CONSISTENCY(a) and LEAD-VIOL, both separated. Controls 1–4 PASS."
    " Committed next: OACC. R73: the harness printed the strategic family as PAUSED; fixed. Pre-registered 2026-10-25."))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))

msg = ("KSCORE: NOT PREDICTABLE (a trained scorer cannot pick O's best longitudinal hypothesis); R73 (the generator's PAUSED label)\n\n"
       "PRIMARY-B fails at both seeds (-0.1001 / +0.0036 m vs <= -F); KSCORE = SEL-CONST (the s0 gain is the constant class); controls 1-4 PASS.\n"
       "Banked run under raw/2026-10-25-kscore/run/. Next (committed): OACC. v8f_kscore.py's strategic label fixed + pinned (123 passed; mutation RED).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
