"""Build qland/hold_v8f_speedavg_pairfix: SPEEDAVG's pair check compares the configs the loader BUILDS (a key a checkpoint predates is its
default), so X_s0 (pre-FLAT, no `o_interface`) and X_s1 (`o_interface: full`) are recognised as one model; + the test and its RED mutation.
python mk_hold74.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "speedavg/"
H = "C:/Users/Admin/qland/hold_v8f_speedavg_pairfix"
MIR = "C:/Users/Admin/tanitad-push"
BASE = "ea67b7a0137f2b1c591da355072541064f4af58d"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RS = f"{PK}/raw/2026-10-25-speedavg"


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def blob(rev, p):
    b = git("rev-parse", f"{rev}:{p}")
    return b.decode().strip() if b else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_speedavg_pairfix", "landed_hold_v8f_speedavg_pairfix"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for p in ("stack/scripts/v8f_mini_eval_speedavg.py", "stack/tests/test_v8f_speedavg.py"):
    assert blob(BASE, p) == blob(TIP, p), f"{p} moved on the tip since {BASE[:8]}"
    put("mods-rw", p, open(T + p, "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RS}/pairfix/fix_speedavg_pair.py", open(SC + "fix_speedavg_pair.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RS}/pairfix/cfgdiff.py", open(SC + "cfgdiff.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RS}/pairfix/mk_hold74.py", open(SC + "grid_a2/mk_hold74.py", "rb").read().replace(b"\r\n", b"\n"))
put("stage", f"{RS}/pairfix/NOTE.md", (
    "# SPEEDAVG pair-check fix (2026-10-25 21:45, MM), before any SPEEDAVG number\n\n"
    "The real arm's first run (21:40) STOPPED at the driver's pair check, before any eval:\n"
    "`REFUSED: the two checkpoints' configs differ in ['o_interface']`.\n\n"
    "`cfgdiff.py` on the two checkpoints gives exactly one difference: `o_interface` is `<MISSING>` in X_s0 and `'full'` in X_s1.\n"
    "* X_s0 = `M2b-mixo_s0`, sha c6a62876…. It was trained before PREREG_V8F_FLAT's lever existed.\n"
    "* X_s1 = `M2b-mixo_s1`, sha 68a0f335…. It was trained after the lever landed.\n\n"
    "`v8f_mini_eval.py` builds `M.V8FMiniConfig(**ck[\"cfg\"])`, so X_s0's missing key builds the default `\"full\"`. That default is"
    " \"every earlier arm, bit for bit\" (`v8f_mini.py:183`). The two are one model, and the raw-dict compare was stricter than the loader.\n\n"
    "**Fix:** `pair_violations` compares `built_cfg(...)`, the stored keys over the dataclass defaults.\n"
    "* Test: a pre-key checkpoint against `full` passes; against `cut` it is refused.\n"
    "* Mutation: with the raw-dict compare restored, `test_pair_violations_on_checkpoints` goes RED by its assertion.\n"
    "* `test_v8f_speedavg.py`: 53 passed.\n\n"
    "No arm, cell set, bar or outcome changes. The run's logs from the refused attempt are kept beside the rerun's.\n").encode())
msg = ("SPEEDAVG: the pair check compares the configs the loader builds (X_s0 predates o_interface; same model as X_s1)\n\n"
       "The real arm's first run refused before any eval on a key X_s0's checkpoint predates (default 'full' = every earlier arm).\n"
       "Test both directions; the raw-dict mutant goes RED. No bar or outcome changes.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
