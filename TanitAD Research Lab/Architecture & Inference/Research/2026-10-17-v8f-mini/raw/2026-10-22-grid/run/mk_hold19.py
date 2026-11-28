"""Build qland/hold_v8fgrid_run: PREREG_V8F_GRID's run banked (RESULT_V8F_GRID.md + raw/2026-10-22-grid/run/) and the register row.
   python mk_hold19.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fgrid_run"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-22-grid/run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_grid/"
EVALS = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def stage(src, rel, binary=False):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    assert b, src
    put("stage", rel, b if binary else lf(b))


for n in ("hold_v8fgrid_run", "landed_hold_v8fgrid_run"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

stage(S + "RESULT_V8F_GRID.md", f"{PK}/RESULT_V8F_GRID.md")
stage(S + "raw_readme_grid_run.md", RUN + "README.md")
stage(S + "mk_hold19.py", RUN + "mk_hold19.py")
for f in ("grid_read_s0.json", "base.json", "check_inputs.json", "cache_check.json", "manifest_eval.jsonl", "telemetry_v8fgrid.csv"):
    stage(E + f, RUN + f)
stage(E + "gate_M2b-8x20/selfcheck.json", RUN + "gate_M2b-8x20/selfcheck.json")
for f in ("chain_v8f_grid_run1.out", "chain_v8f_grid.out", "v8fgrid_guard_run1.out", "v8fgrid_guard.out", "v8f_grid_HANDOFF.md"):
    stage(LG + f, RUN + "box2_logs/" + f)
stage("C:/Users/Admin/box2_ship/run_chain_v8f_grid.cmd", RUN + "box2_logs/run_chain_v8f_grid.cmd")
logs = sorted(glob.glob(LG + "v8fgrid1022215*_*.log") + glob.glob(LG + "v8fgrid1023032*_*.log"))
assert len(logs) == 8, logs                      # two chain runs x (refresh, gate, train, eval)
for p in logs:
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
got = sorted(os.path.basename(p) for p in glob.glob(E + "eval/eval_M2b-8x20_s0*"))
want = sorted(f"eval_M2b-8x20_s0{x}.{e}" for x in EVALS for e in ("npz", "json", "log"))
assert got == want, (got, want)
for f in got:
    stage(E + "eval/" + f, RUN + "eval/" + f, binary=f.endswith(".npz"))
for f in ("summary.json", "selfcheck.json", "argv.json", "train_log.jsonl", "train.log"):
    stage(E + "runs/M2b_s0_8x20/" + f, RUN + "runs/M2b_s0_8x20/" + f)

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ig = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-GRID-1** (2026-10-22) |")]
assert len(ig) == 1
pre = "| **PRE-REGISTERED, NOT RUN (PI go 2026-10-22; box 2 after PGT's chain ends, ~6–8 h per seed, ESTIMATED).** "
assert lines[ig[0]].count(pre) == 1
new = open(S + "reg19.txt", "rb").read().decode("utf-8").strip()
assert new.startswith("| **") and "|" not in new[2:]
lines[ig[0]] = lines[ig[0]].replace(pre, new + " ~~PRE-REGISTERED, NOT RUN (PI go 2026-10-22; box 2 after PGT's chain ends, ~6–8 h per seed, ESTIMATED).~~ ")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg19.txt", "rb").read()))
print("hold ok")
