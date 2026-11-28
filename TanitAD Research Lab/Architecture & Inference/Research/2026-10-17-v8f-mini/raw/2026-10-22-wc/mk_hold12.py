"""Build qland/hold_v8fwc_s0: PREREG_V8F_WC seed 0 (STOP, FAIL-MECHANISM-OK) banked + Amendment 1 (reader fix, --check-inputs).
   python mk_hold12.py <remote tip sha> <n RED> <n mutations>"""
import glob
import os
import subprocess
import sys

TIP, NRED, NMUT = sys.argv[1], sys.argv[2], sys.argv[3]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/tac/"
T4 = S + "t4/"
BASE = "39e21e50d90b050881facf8adf8dd786b6dd366a"          # the land that carried the instruments; every edited file is checked against it
H = "C:/Users/Admin/qland/hold_v8fwc_s0"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-22-wc/"
RUN = RAW + "run/"
LG = "C:/Users/Admin/box2_ship/logs/"
E = LG + "v8f_wc/"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def lf(b):
    return b.replace(b"\r\n", b"\n")


def stage(src, rel, binary=False):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if binary else lf(b))


def rw(rel, new, base_check=True):
    old = show(TIP, rel)
    assert old is not None, rel
    if base_check:
        assert old == show(BASE, rel), f"{rel}: the tip moved since {BASE[:8]}"
    new = lf(new)
    assert new != old, rel
    put("mods-rw", rel, new)


for n in ("hold_v8fwc_s0", "landed_hold_v8fwc_s0"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- Amendment 1: the reader fix, its tests, the chain's pre-training input check, the PREREG note
for rel in (f"{PK}/code/v8f_wc_read.py", "stack/tests/test_v8f_wc.py", f"{PK}/code/box2_ops/chain_v8f_wc.sh"):
    rw(rel, open(T4 + rel, "rb").read())
pz = "Project Steering/PREREG_V8F_WC.md"
assert show(TIP, pz) == open(S + "PREREG_V8F_WC.tip.md", "rb").read(), "the PREREG moved since Amendment 1 was appended to it"
rw(pz, open(S + "PREREG_V8F_WC.a1.md", "rb").read(), base_check=False)
for f in ("mutate_wc_a1.py", "mutate_wc_a1.out", "mk_hold12.py"):
    stage(S + f, RAW + f)

# ---- the result + the run's raw
stage(S + "RESULT_V8F_WC.md", f"{PK}/RESULT_V8F_WC.md")
stage(S + "raw_readme_wc_run.md", RUN + "README.md")
stage(E + "wc_read_s0.json", RUN + "wc_read_s0.json")
stage(E + "gate_M2b-wc/selfcheck.json", RUN + "gate_M2b-wc/selfcheck.json")
stage(E + "telemetry_v8fwc.csv", RUN + "telemetry_v8fwc.csv")
for f in ("chain_v8f_wc.out", "v8fwc_guard.out"):
    stage(LG + f, RUN + "box2_logs/" + f)
for p in sorted(glob.glob(LG + "v8fwc10220759_*.log")):
    stage(p, RUN + "box2_logs/" + os.path.basename(p))
n = 0
for p in sorted(glob.glob(E + "eval/eval_M2b-wc_s0*")):
    stage(p, RUN + "eval/" + os.path.basename(p), binary=p.endswith(".npz"))
    n += 1
assert n == 9, f"{n} M2b-wc_s0 eval files, expected 3 x (npz, json, log)"
for f in ("summary.json", "selfcheck.json", "argv.json", "train_log.jsonl", "train.log"):
    stage(E + "runs/M2b-wc_s0/" + f, RUN + "runs/M2b-wc_s0/" + f)

# ---- the register
reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
iw = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-WC-1** (2026-10-22) |")]
assert len(iw) == 1
pre = "| **PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 stops it, ~10 h in all; awaits the PI's go).** "
assert lines[iw[0]].count(pre) == 1
lines[iw[0]] = lines[iw[0]].replace(pre, (
    "| **FAIL-MECHANISM-OK (as committed), STOP at seed 0 (2026-10-22); seed 1 not trained. MEASURED, T1-class open loop, paired "
    "episode-cluster bootstrap, 141 episodes; `…/2026-10-17-v8f-mini/raw/2026-10-22-wc/run/wc_read_s0.json`.** ADE6 M2b-wc − M2b "
    "**−0.0035 [−0.0795, +0.0850] ns** (bar −0.2415). Mechanism: T present-slot RMSE 0.536 → 0.523 SD, separated, only **−2.4 %**, while "
    "T's TRAIN constraint loss fell 18–25 % (0.125 → 0.103): the extra fit did not generalise. Descriptive: LON speed −0.066 m/s and accel "
    "−0.169 m/s² separated-better, LAT cross-track +0.050 m separated-worse. ⚠️ The mechanism bar had no magnitude threshold (my PREREG "
    "flaw), so the label overstates the fix; the measured guide is T's INPUT (the cached `pool8x20` grid, PI go). Amendment 1 (before "
    "any number): the reader read M2b's GT from Amendment 2's all-goal oracle; `--check-inputs` now runs before training. "
    "`RESULT_V8F_WC.md`. ~~PRE-REGISTERED, NOT RUN (box 2, ~5 h if seed 0 stops it, ~10 h in all; awaits the PI's go).~~ "))
rw("Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"), base_check=False)
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(S + "msg12.txt", "rb").read()).replace(b"<NRED>", NRED.encode()).replace(b"<NMUT>", NMUT.encode()))
print("hold ok")
