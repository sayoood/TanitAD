"""Build qland/hold_v8f_kscore_pred: KSCORE's chain decides on FLAT's stops with one predicate (after_decision); FLAT's 21 dev-box pre-flight stops PROCEED.
python mk_hold62.py <tip> "<pytest summary line>" """
import os
import subprocess
import sys

TIP, SUMMARY = sys.argv[1], sys.argv[2]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
K = SC + "kscore_pred/"
A2 = SC + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_kscore_pred"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-kscore/predecessor"
BASE = {"stack/scripts/chain_v8f_kscore.sh": "f1cce72642b794893938e134d78c8cf4c3b58d91",
        "stack/tests/test_v8f_kscore.py": "01858b7f72511ebe0b56c4d6c9df4f30ce3aba8c"}


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def lf(b):
    return b.replace(b"\r\n", b"\n")


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_kscore_pred", "landed_hold_v8f_kscore_pred"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for rel, blob in BASE.items():   # the agent built on exactly these blobs: refuse if the tip moved them
    assert git("rev-parse", f"{TIP}:{rel}").decode().strip() == blob, rel
put("mods-rw", "stack/scripts/chain_v8f_kscore.sh", lf(open(K + "chain_v8f_kscore.sh", "rb").read()))
put("mods-rw", "stack/tests/test_v8f_kscore.py", lf(open(K + "test_v8f_kscore.py", "rb").read()))
for src, name in (("NOTES.md", "NOTES.md"), ("tests_pf.out", "tests_pf.out"), ("run_tests_pf.sh", "run_tests_pf.sh")):
    rel = f"{RAW}/{name}"
    assert git("cat-file", "-e", f"{TIP}:{rel}") is None, rel
    put("stage", rel, lf(open(K + src, "rb").read()))
put("stage", f"{RAW}/mk_hold62.py", lf(open(A2 + "mk_hold62.py", "rb").read()))

lines = git("show", f"{TIP}:Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KSCORE-1** (2026-10-25) |")]
assert len(ix) == 1
old = "as built it holds on ANY predecessor STOP, which would idle box 2."
assert lines[ix[0]].count(old) == 1
lines[ix[0]] = lines[ix[0]].replace(old, (
    "as built it held on ANY predecessor STOP, which would idle box 2. **FIXED 2026-10-25:** one predicate, `after_decision`, used by the wait loop and by"
    " `--classify-predecessor`: PROCEED on DONE, on FLAT's run-level `<job> rc=<n>` (gate / train / eval), `flat read`, `gate FLAT assertions` and FLAT's 21"
    " dev-box PRE-FLIGHT stops (the box was never asked to run anything; KSCORE re-checks its own tip); HOLD on box-level evidence"
    " (ZZGUARD-STOP / ZZCRASH / ZZUNREACHABLE / ZZTIMEOUT), FLAT's refresh-job rc and any text not listed (fail closed). Every one of FLAT's 42 + 4 stop texts"
    f" is classified by a test that reads FLAT's source; mutations go RED. `{SUMMARY}` (`{RAW}/tests_pf.out`)."))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("KSCORE chain: one predecessor predicate; FLAT's run / verdict / gate and dev-box pre-flight stops proceed, box-level evidence holds\n\n"
       "after_decision is used by the wait loop and by --classify-predecessor. Every FLAT stop text is classified from FLAT's source;\n"
       f"unknown texts hold (fail closed). Tests: {SUMMARY}. Register: H-V8F-KSCORE-1 note.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
