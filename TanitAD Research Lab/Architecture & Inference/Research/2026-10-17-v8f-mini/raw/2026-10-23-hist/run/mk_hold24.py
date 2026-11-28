"""Build qland/hold_v8fhist_run: PREREG_V8F_HIST's run banked (RESULT_V8F_HIST.md + raw/2026-10-23-hist/run/), register rows H-V8F-HIST-1 and H-V8F-D1-1.
   python mk_hold24.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fhist_run"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-23-hist/run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_hist/"


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


for n in ("hold_v8fhist_run", "landed_hold_v8fhist_run"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(S + "RESULT_V8F_HIST.md", f"{PK}/RESULT_V8F_HIST.md")
stage(S + "raw_readme_hist_run.md", RUN + "README.md")
stage(S + "mk_hold24.py", RUN + "mk_hold24.py")
stage(S + "hist_facts.py", RUN + "hist_facts.py")
stage(S + "hist_facts.out", RUN + "hist_facts.out")
for f in ("hist_read_s0.json", "check_inputs.json", "telemetry_v8fhist.csv"):
    stage(E + f, RUN + f)
stage(E + "gate_M2b-h10/selfcheck.json", RUN + "gate_M2b-h10/selfcheck.json")
for f in ("chain_v8f_hist.out", "v8fhist_guard.out", "v8f_hist_HANDOFF.md"):
    stage(LG + f, RUN + "box2_logs/" + f)
stage("C:/Users/Admin/box2_ship/run_chain_v8f_hist.cmd", RUN + "box2_logs/run_chain_v8f_hist.cmd")
logs = sorted(glob.glob(LG + "v8fhist10230700_*.log"))
assert len(logs) == 4, logs
for p in logs:
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
got = sorted(os.path.basename(p) for p in glob.glob(E + "eval/eval_M2b-h10_s0*"))
want = sorted(f"eval_M2b-h10_s0{x}.{e}" for x in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons") for e in ("npz", "json", "log"))
assert got == want, (got, want)
for f in got:
    stage(E + "eval/" + f, RUN + "eval/" + f, binary=f.endswith(".npz"))
for f in ("summary.json", "selfcheck.json", "argv.json", "train_log.jsonl", "train.log"):
    stage(E + "runs/M2b_s0_h10/" + f, RUN + "runs/M2b_s0_h10/" + f)

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
for rid, pre, newf in (("H-V8F-HIST-1", "| **PRE-REGISTERED, NOT RUN (box 2, ~4.5–5 h per seed ESTIMATED; awaits the PI's go).** ", "reg24_hist.txt"),
                       ("H-V8F-D1-1", "| **PRE-REGISTERED, NOT RUN (box 2 after HIST's chain; awaits the PI's go).** ", "reg24_d1.txt")):
    ix = [k for k, x in enumerate(lines) if x.startswith(f"| **{rid}** (2026-10-23) |")]
    assert len(ix) == 1, rid
    assert lines[ix[0]].count(pre) == 1, rid
    new = open(S + newf, "rb").read().decode("utf-8").strip()
    assert new.startswith("| **") and "|" not in new[2:]
    lines[ix[0]] = lines[ix[0]].replace(pre, new + " ~~" + pre[2:].strip() + "~~ ")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg24.txt", "rb").read()))
print("hold ok")
