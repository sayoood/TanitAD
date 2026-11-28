"""Build qland/hold_v8fpgt_run: PREREG_V8F_PGT's run banked (RESULT_V8F_PGT.md + raw/2026-10-22-pgt/run/) and the register row.
   python mk_hold17.py <remote tip sha> <seeds: 0 | 0,1> <read json: pgt_read_s0.json | pgt_read.json>
   Written BEFORE the read landed (only the paths are fixed here); the RESULT, the run README, the register text and the message are
   separate files written after the read: dkland/RESULT_V8F_PGT.md, raw_readme_pgt_run.md, reg17.txt, msg17.txt."""
import glob
import os
import subprocess
import sys

TIP, SEEDS, READ = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
assert SEEDS in (["0"], ["0", "1"]), SEEDS
assert READ in ("pgt_read_s0.json", "pgt_read.json"), READ
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/dkland/"
H = "C:/Users/Admin/qland/hold_v8fpgt_run"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RUN = f"{PK}/raw/2026-10-22-pgt/run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_pgt/"
EVALS = ("", "-tacabs", "-tacorc", "-tacorc-all-latcons")      # the four evals evalpgt runs per seed


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


for n in ("hold_v8fpgt_run", "landed_hold_v8fpgt_run"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

stage(S + "RESULT_V8F_PGT.md", f"{PK}/RESULT_V8F_PGT.md")
stage(S + "raw_readme_pgt_run.md", RUN + "README.md")
stage(S + "mk_hold17.py", RUN + "mk_hold17.py")
TO = f"{PK}/raw/2026-10-22-pgt/tail_oracle/"                     # EXPLORATORY: does T's code still move the tail after PGT (rule written before the run)
for f in ("pgt_tail_oracle.py", "pgt_tail_oracle.json", "pgt_tail_oracle.out"):
    stage(S + f, TO + f)
for f in ("check_inputs.json", "pgt_read_s0.json") + (("pgt_read.json",) if SEEDS == ["0", "1"] else ()):
    stage(E + f, RUN + f)
stage(E + "gate_M2b-pgt/selfcheck.json", RUN + "gate_M2b-pgt/selfcheck.json")
if os.path.exists(E + "telemetry_v8fpgt.csv"):
    stage(E + "telemetry_v8fpgt.csv", RUN + "telemetry_v8fpgt.csv")
for f in ("chain_v8f_pgt.out", "v8fpgt_guard.out", "v8fpgt_after.out"):   # _after.out: the hand-written DONE line the chain waited on (AFTER_OUT override)
    stage(LG + f, RUN + "box2_logs/" + f)
stage("C:/Users/Admin/box2_ship/run_chain_v8f_pgt.cmd", RUN + "box2_logs/run_chain_v8f_pgt.cmd")
logs = sorted(glob.glob(LG + "v8fpgt10221802_*.log"))
assert len(logs) >= 4, logs                                      # refresh, gate, train s0, eval s0 at the least
for p in logs:
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
for s in SEEDS:
    got = sorted(os.path.basename(p) for p in glob.glob(E + f"eval/eval_M2b-pgt_s{s}*"))
    want = sorted(f"eval_M2b-pgt_s{s}{x}.{e}" for x in EVALS for e in ("npz", "json", "log"))
    assert got == want, (s, got, want)
    for f in got:
        stage(E + "eval/" + f, RUN + "eval/" + f, binary=f.endswith(".npz"))
    for f in ("summary.json", "selfcheck.json", "argv.json", "train_log.jsonl", "train.log"):
        stage(E + f"runs/M2b-pgt_s{s}/" + f, RUN + f"runs/M2b-pgt_s{s}/" + f)

reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
ip = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-PGT-1** (2026-10-22) |")]
assert len(ip) == 1
pre = ("| **RUNNING on box 2 (PI go 2026-10-22; chain launched 18:02 at tip 14a27de; --check-inputs passed on the real banked M2b files "
       "before training).** ")
assert lines[ip[0]].count(pre) == 1
new = open(S + "reg17.txt", "rb").read().decode("utf-8").strip()
assert new.startswith("| **") and "|" not in new[2:], "reg17.txt is ONE cell's opening text"
lines[ip[0]] = lines[ip[0]].replace(pre, new + " ~~RUNNING on box 2 (chain launched 18:02 at tip 14a27de).~~ ")
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg17.txt", "rb").read()))
print("hold ok")
