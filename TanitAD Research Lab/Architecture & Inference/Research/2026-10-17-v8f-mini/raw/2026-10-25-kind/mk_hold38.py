"""Build qland/hold_v8fkind_instr: PREREG_V8F_KIND's instruments (CODE_KINDS, reader, chain, queue stages, tests, raw/2026-10-25-kind/) + Amendment 2
   + the register note on H-V8F-KIND-1.   python mk_hold38.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "c7d40467"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "kind/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fkind_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-kind/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8fkind_instr", "landed_hold_v8fkind_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
q = f"{PK}/code/v8f_mini_box2_queue.cmd"
for rel in ("stack/tanitad/models/v8f_mini.py", q, "stack/tests/test_v8f_lora_ar2.py"):
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    new = open(X + rel, "rb").read()
    assert new != old, rel
    if rel == q:
        assert new.count(b"\r\n") == new.count(b"\n"), "the queue must stay CRLF"
    else:
        new = lf(new)
    put("mods-rw", rel, new)
NEW = [f"{PK}/code/v8f_kind_read.py", f"{PK}/code/box2_ops/chain_v8f_kind.sh", "stack/tests/test_v8f_kind.py"]
NEW += [RAW + f for f in ("window_strata_eval141.npz", "mk_window_strata.py", "mk_window_strata.out", "mutate_kind.py", "mutate_kind.out",
                          "mutate_kind_run1.out", "dryrun_queue_kind.out", "dryrun_chain_kind.out", "check_inputs_real.out", "check_inputs_real.json",
                          "suites.out")]
for rel in NEW:
    assert show(TIP, rel) is None, rel
    b = open(X + rel, "rb").read()
    assert b, rel
    put("stage", rel, b if rel.endswith(".npz") else lf(b))
assert b"\r" not in open(os.path.join(H, "stage", f"{PK}/code/box2_ops/chain_v8f_kind.sh"), "rb").read()
put("stage", RAW + "mk_hold38.py", lf(open(A2 + "mk_hold38.py", "rb").read()))
put("stage", RAW + "FILES.txt", lf(open(X + "FILES.txt", "rb").read()))
pr = "Project Steering/PREREG_V8F_KIND.md"
old = show(TIP, pr)
assert old.endswith(b"\n") and b"## Amendment 1" in old and b"## Amendment 2" not in old
put("mods", pr, old + lf(open(A2 + "kind_amend2.md", "rb").read()))
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KIND-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
lines[ix[0]] = lines[ix[0]][:-2] + (" **Instruments landed with Amendment 2 (2026-10-25):** `code/v8f_kind_read.py`, `code/box2_ops/chain_v8f_kind.sh`,"
                                    " `CODE_KINDS` in `stack/tanitad/models/v8f_mini.py`. The eval needed no change (it already routes `--tac-groups`"
                                    " through `code_group_index`). 32 new tests, 57 / 57 mutants RED, `--check-inputs` PASS on the real banked M2b side"
                                    " (identity controls exactly +0 [0, 0]). Box-2 run: under the PI's go of 2026-10-25, after AR-2. |")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_KIND instruments + Amendment 2 (before any data): CODE_KINDS, reader, eval-only box-2 chain, queue stages, tests\n\n"
       "- stack/tanitad/models/v8f_mini.py: CODE_KINDS, a second partition of the 26 constraint cells (speed, angle, tte, timegap,\n"
       "  geom, leadgap); code_group_index takes a kind anywhere it takes a group; every existing spec returns the same indices.\n"
       "- v8f_mini_eval.py unchanged: --tac-groups already routes through code_group_index (pinned by an AST and a functional test).\n"
       "- code/v8f_kind_read.py: gain_k, necessity_k, additivity A, the outcome table, the SPEED side-read and Amendment 1's\n"
       "  per-stratum read (junction 298 / curve 140 / no-turn 1,961 windows); refusals rc 2; --check-inputs PASS on the real B side.\n"
       "- code/box2_ops/chain_v8f_kind.sh + queue labels kind/kindseed/kindown/kindorc (CRLF kept, old labels byte-identical):\n"
       "  30 evals on the banked M2b s0/s1, atomic busy marker, AFTER_OUT override. test_v8f_lora_ar2.py: one label-set assert widened.\n"
       "- Tests: 32 new; test_v8f*.py + test_launch_gate_v8f.py 837 passed, 0 failed (805 on the base + 32). Mutants 57/57 RED.\n"
       "Register: H-V8F-KIND-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", len(NEW), "new files")
