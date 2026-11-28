"""Build qland/hold_v8fmixo_instr: PREREG_V8F_MIXO's instruments (the --code-mix lever, its G-DVB entry, tests, chain, queue stages,
   reader, raw/2026-10-25-mixo/), PREREG_V8F_FLAT (pre-registered before any instrument or data), and the register: the PI's go
   (PI-V8F-2026-10-25b), H-V8F-MIXO-1 (instruments landed), H-V8F-FLAT-1 (new).   python mk_hold42.py <remote tip sha>
   Every file the MIXO agent changed is found by mtime (newer than the export) AND by bytes (differs from the export's base);
   each changed file must be unchanged on the tip since that base."""
import os
import subprocess
import sys

TIP = sys.argv[1]
BASE = "c0769eaa"
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
X, A2 = S + "mixo/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fmixo_instr"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-mixo/"
REF = X + "stack/tanitad/__init__.py"          # untouched by the agent: its mtime is the export's
QUEUE = f"{PK}/code/v8f_mini_box2_queue.cmd"


import builtins


def LP(p):   # Windows long-path form: the package holds paths over 260 chars, where plain Python file calls fail and os.walk silently skips
    p = os.path.abspath(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


_open = builtins.open


def open(p, *a, **k):
    return _open(LP(p), *a, **k)


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(LP(os.path.dirname(p)), exist_ok=True)
    open(p, "wb").write(data)


lf = lambda b: b.replace(b"\r\n", b"\n")
for n in ("hold_v8fmixo_instr", "landed_hold_v8fmixo_instr"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
t0 = os.path.getmtime(REF)
assert show(BASE, "stack/tanitad/__init__.py") == open(REF, "rb").read()
changed, new = [], []
RESEARCH = LP(X + "TanitAD Research Lab/Architecture & Inference/Research")
XL = LP(X)
walk_errors = []
for root, dirs, files in os.walk(XL, onerror=walk_errors.append):
    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", "_pt", "pt_tmp")]
    if root == RESEARCH:   # the 11 packages added for the suites are byte-identical base exports (c0769eaa); only v8f-mini is the agent's
        dirs[:] = [d for d in dirs if d == "2026-10-17-v8f-mini"]
    for f in files:
        p = os.path.join(root, f)
        if os.path.getmtime(p) <= t0 + 1 or f.endswith(".pyc"):
            continue
        rel = os.path.relpath(p, XL).replace("\\", "/")
        old = show(BASE, rel)
        b = open(p, "rb").read()
        if old is None:
            new.append(rel)
        elif old != b:
            changed.append(rel)
assert not walk_errors, walk_errors        # a directory the walk could not read would hide a changed file
assert "FILES.txt" in new
new.remove("FILES.txt")                     # the agent's manifest at the tree root: banked under raw/ below
print("CHANGED", *changed, sep="\n  ")
print("NEW", *new, sep="\n  ")
ALLOW_CHANGED = {"stack/tanitad/models/v8f_mini.py", "stack/scripts/v8f_mini_train.py", QUEUE, "stack/tests/test_v8f_kind.py",
                 "stack/tests/test_v8f_lora_ar2.py", "stack/tests/test_v8f_window_start.py", "stack/scripts/launch_gate.py",
                 "stack/tests/test_launch_gate_v8f.py"}
assert set(changed) <= ALLOW_CHANGED, set(changed) - ALLOW_CHANGED
for rel in new:
    assert rel.startswith((RAW, f"{PK}/code/", "stack/tests/test_v8f_mixo")), f"unexpected new file {rel}"
for rel in changed:
    old = show(TIP, rel)
    assert old is not None and old == show(BASE, rel), f"{rel}: the tip moved since {BASE}"
    b = open(X + rel, "rb").read()
    if rel == QUEUE:
        assert b.count(b"\r\n") == b.count(b"\n"), "the queue must stay CRLF"
        # every pre-existing label block must survive byte for byte: the old file's lines are a subsequence of the new one's
        ol, nl = old.split(b"\r\n"), b.split(b"\r\n")
        it = iter(nl)
        assert all(any(x == y for y in it) for x in ol), "an old queue line was changed or dropped"
    else:
        b = lf(b)
    put("mods-rw", rel, b)
for rel in new:
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(X + rel, "rb").read()
    assert b, rel
    put("stage", rel, b if rel.endswith((".npz", ".pt")) else lf(b))
ch = os.path.join(H, "stage", f"{PK}/code/box2_ops/chain_v8f_mixo.sh")
assert os.path.exists(LP(ch)) and b"\r" not in open(ch, "rb").read()
put("stage", RAW + "mk_hold42.py", lf(open(A2 + "mk_hold42.py", "rb").read()))
# the reruns replace the two proofs a full C: invalidated; the dead runs are kept beside them, renamed, as the record
put("stage", RAW + "FILES.txt", lf(open(X + "FILES.txt", "rb").read()))
put("stage", RAW + "suites_rerun1_missing_pkgs.out", lf(open(A2 + "mixo_suites_rerun.out", "rb").read()))   # 960 passed: 31 read packages the export lacked
for dead, rerun in (("suites.out", "mixo_suites_rerun2.out"), ("mutate_mixo.out", None)):
    put("stage", RAW + dead.replace(".out", "_enospc.out"), lf(open(X + RAW + dead, "rb").read()))
    if rerun:
        put("stage", RAW + dead, lf(open(A2 + rerun, "rb").read()))
    else:   # the 162-mutant rerun (about 2 h) is still running: it lands with the run's bank, not here
        os.remove(LP(os.path.join(H, "stage", RAW + dead)))   # the hold's OWN staged copy of the dead run, written by this script a moment ago
assert show(TIP, "Project Steering/PREREG_V8F_FLAT.md") is not None   # landed on its own (25037c45), before any data

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1** (2026-10-25) |")]
assert len(ix) == 1 and lines[ix[0]].endswith(" |")
SUITES = sys.argv[2]                       # the rerun's own summary line, e.g. "880 passed, 2 skipped (rerun 08:49)"
note = open(A2 + "mixo_register_note.txt", "rb").read().decode("utf-8").strip().replace("%%SUITES%%", SUITES)
assert "\n" not in note and "%%" not in note, "a register row is one line"
lines[ix[0]] = lines[ix[0]][:-2] + " " + note + " |"
assert sum(x.startswith(("| **PI-V8F-2026-10-25b**", "| **H-V8F-FLAT-1**", "| **H-V8F-HBLEND-1**")) for x in lines) == 3
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = open(A2 + "mixo_msg.txt", "rb").read().decode("utf-8").replace("%%SUITES%%", SUITES)
assert "%%" not in msg
assert msg.rstrip().endswith("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
open(os.path.join(H, "msg.txt"), "wb").write(lf(msg.encode("utf-8")))
print("hold ok", len(changed), "changed,", len(new), "new")
