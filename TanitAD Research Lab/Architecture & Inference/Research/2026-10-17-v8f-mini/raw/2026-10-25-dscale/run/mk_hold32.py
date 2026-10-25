"""Build qland/hold_v8fdscale_run: PREREG_V8F_DSCALE's run banked (RESULT_V8F_DSCALE.md + raw/2026-10-25-dscale/run/), R69, register rows
   H-V8F-DSCALE-1 (verdict), H-DATA-SCALE-1 and H-V8F-HIST-1 (notes).
   python mk_hold32.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8fdscale_run"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-25-dscale/run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_dscale/"


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


for n in ("hold_v8fdscale_run", "landed_hold_v8fdscale_run"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
assert os.path.isdir(E) and os.path.isdir(S), (E, S)
stage(S + "RESULT_V8F_DSCALE.md", f"{PK}/RESULT_V8F_DSCALE.md")
stage(S + "raw_readme_dscale_run.md", RUN + "README.md")
for f in ("mk_hold32.py", "dscale_facts.py", "dscale_facts.out", "accel_all_arms.py", "accel_all_arms.out"):
    stage(S + f, RUN + f)
for f in ("dscale_read.json", "check_inputs.json", "manifest_train.jsonl", "telemetry_v8fdscale.csv"):
    stage(E + f, RUN + f)
stage(E + "gate_M2b-df50/selfcheck.json", RUN + "gate_M2b-df50/selfcheck.json")
for f in ("chain_v8f_dscale.out", "v8fdscale_guard.out", "v8f_dscale_HANDOFF.md"):
    stage(LG + f, RUN + "box2_logs/" + f)
assert b"## CLOSED" in open(LG + "v8f_dscale_HANDOFF.md", "rb").read(), "close the handoff first"
stage("C:/Users/Admin/box2_ship/run_chain_v8f_dscale.cmd", RUN + "box2_logs/run_chain_v8f_dscale.cmd")
logs = sorted(glob.glob(LG + "v8fdscale10250125_*.log"))
assert len(logs) == 4, logs
for p in logs:
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
got = sorted(os.path.basename(p) for p in glob.glob(E + "eval/eval_M2b-df50_s0*"))
want = sorted(f"eval_M2b-df50_s0{x}.{e}" for x in ("", "-tacabs", "-tacorc", "-tacorc-all-latcons") for e in ("npz", "json", "log"))
assert got == want, (got, want)
for f in got:
    stage(E + "eval/" + f, RUN + "eval/" + f, binary=f.endswith(".npz"))
for f in ("summary.json", "selfcheck.json", "argv.json", "train_log.jsonl", "train.log"):
    stage(E + "runs/M2b_s0_df50/" + f, RUN + "runs/M2b_s0_df50/" + f)

rl = "Project Steering/RETRACTION_LOG.md"
old = show(TIP, rl)
assert old.endswith(b"\n") and b"### R69 " not in old and b"### R68 " in old
put("mods", rl, old + lf(open(S + "r69.md", "rb").read()))

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-DSCALE-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (box 2, ~4–4.5 h ESTIMATED; awaits the PI's go).** "
assert lines[ix[0]].count(pre) == 1
new = open(S + "reg32_dscale.txt", "rb").read().decode("utf-8").strip()
assert new.startswith("| **") and "|" not in new[2:]
lines[ix[0]] = lines[ix[0]].replace(pre, new + " ~~" + pre[2:].strip() + "~~ ")
for rid, f in (("| **H-DATA-SCALE-1** |", "reg32_datascale.txt"), ("| **H-V8F-HIST-1** (2026-10-23) |", "reg32_hist.txt")):
    ix = [k for k, x in enumerate(lines) if x.startswith(rid)]
    assert len(ix) == 1 and lines[ix[0]].endswith(" |"), rid
    note = open(S + f, "rb").read().decode("utf-8").strip()
    assert "|" not in note, f
    lines[ix[0]] = lines[ix[0]][:-2] + " " + note + " |"
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg32.txt", "rb").read()))
print("hold ok")
