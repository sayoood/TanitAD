"""Build qland/hold_v8f_mixo_mutproof: the PREREG_V8F_MIXO 162-mutant proof rerun banked (148/162 RED, 14 SURVIVED, listed) with a
   register note on H-V8F-MIXO-1. python mk_hold54.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_mixo_mutproof"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-mixo/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_mixo_mutproof", "landed_hold_v8f_mixo_mutproof"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for src, dst in (("mixo_mutate_rerun.out", "mutation_proof_rerun.out"), ("mk_hold54.py", "mk_hold54.py")):
    assert show(TIP, RAW + dst) is None, dst
    put("stage", RAW + dst, open(A2 + src, "rb").read().replace(b"\r\n", b"\n"))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1**")]
assert len(ix) == 1 and lines[ix[0]].rstrip().endswith("|")
lines[ix[0]] = lines[ix[0]].rstrip()[:-1].rstrip() + (
    " **Mutation proof (2026-10-25, rerun on a cmp-identical throwaway tree; `…/raw/2026-10-25-mixo/mutation_proof_rerun.out`): 148 of 162 RED,"
    " 14 SURVIVED.** Survivors: 5 G-LIVE probe refusal paths (the probe's PASS on the real gate, partly-GT in 124 of 128 cells, is a direct"
    " measurement and stands); 5 in the reader (the exact −2F boundary, the O-LEARNED verdict rule, a next-step text, two window-agreement"
    " checks); 4 input checks (dose identity cells, a non-finite dose ADE6, the GT-code check, B being a whole-code run; B's argv carries no"
    " `--code-mix`, so the last holds by inspection). ⇒ The reader's s0 verdict is re-derived by hand from its raw numbers before it is"
    " quoted, and killing tests are a work item after FLAT's instruments land (both touch `test_v8f_mixo.py`). |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_MIXO mutation proof rerun: 148/162 RED, 14 survived (listed in the register)\n\n"
       "Survivors: 5 probe refusal paths, 5 reader branches (incl. the -2F boundary and the O-LEARNED rule), 4 input checks. The s0 verdict\n"
       "will be re-derived by hand; killing tests follow FLAT's landing (shared test file).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
