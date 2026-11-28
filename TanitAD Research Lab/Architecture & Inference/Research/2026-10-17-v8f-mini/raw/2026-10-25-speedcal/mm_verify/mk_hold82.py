"""Build qland/hold_v8f_speedcal: PREREG_V8F_SPEEDCAL's instruments and its RUN (outcome INFORMATION-LIMITED), the MM re-read, the result
file, register H-V8F-SPEEDCAL-1, and two OPEN PI decisions (D-V8F-HACC-T1KIN, D-V8F-SPEED-PERCEPTION).  python mk_hold82.py <tip>
Every FILES.txt row is NEW (asserted absent at the tip); sources are read through extended-length paths (the tree is deeper than MAX_PATH)."""
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
SP = SC + "speedcal/"
MM = SC + "speedcal_mm/"
H = "C:/Users/Admin/qland/hold_v8f_speedcal"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RO = f"{PK}/raw/2026-10-25-speedcal"
LP = "\\\\?\\"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def tipblob(rel):
    b = git("rev-parse", f"{TIP}:{rel}")
    return b.decode().strip() if b else None


def rd(path):
    return open(LP + os.path.normpath(os.path.abspath(path)), "rb").read()


def put(kind, rel, data):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_speedcal", "landed_hold_v8f_speedcal"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
assert open("C:/Users/Admin/sc_run/OUTCOME.txt", encoding="utf-8").read().strip() == "INFORMATION-LIMITED"
pre = "Project Steering/PREREG_V8F_SPEEDCAL.md"
assert git("show", f"{TIP}:{pre}") == rd(SC + "speedavg_res/PREREG_V8F_SPEEDCAL.md"), "the PREREG on the tip differs from the one the run applied"

n = 0
for ln in open(SP + "FILES.txt", encoding="utf-8"):
    if not ln.strip():
        continue
    rel, nbytes = ln.rstrip("\r\n").split("\t")[:2]
    b = rd(SP + rel)
    assert len(b) == int(nbytes), f"FILES.txt stale: {rel}"
    dst = f"{RO}/HANDOFF.md" if rel == "HANDOFF.md" else rel
    assert dst.startswith((f"{RO}/", f"{PK}/code/", "stack/scripts/", "stack/tests/")), dst
    assert tipblob(dst) is None, f"NEW path exists at the tip: {dst}"
    put("stage", dst, b)
    n += 1
assert n == 104, n
for f in ("verify_sc.py", "verify_sc.log", "verify_sc.json", "ADDENDUM_MM.md"):
    put("stage", f"{RO}/mm_verify/{f}", rd(MM + f).replace(b"\r\n", b"\n"))
put("stage", f"{RO}/mm_verify/mk_hold82.py", rd(SC + "grid_a2/mk_hold82.py").replace(b"\r\n", b"\n"))
read_md = rd(SP + f"{RO}/run/READ.md")
assert read_md == rd("C:/Users/Admin/sc_run/READ.md")
res = f"{PK}/RESULT_V8F_SPEEDCAL.md"
assert tipblob(res) is None
put("stage", res, read_md.rstrip(b"\n") + b"\n" + rd(MM + "ADDENDUM_MM.md").replace(b"\r\n", b"\n"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-SPEEDCAL-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **RUN (2026-10-26): INFORMATION-LIMITED** (the PREREG table applied verbatim; T1-class open loop, CEILING and the `gt` arm T0;"
    " paired episode-cluster bootstrap, B = 2,000, 141 episodes / 2,399 windows). PRIMARY ADE6 SPEEDCAL − X_s: s0 +0.0374 [−0.0100, +0.0864],"
    " s1 −0.0058 [−0.0156, +0.0036], separated at neither seed. CEILING (an eval-GT-fitted map, T0): s0 +0.0296, s1 −0.0075, above −F at both."
    " SECONDARY: separated-worse at s0 (speed MAE, accel MAE, headway-min, time-gap-min), clean at s1. Controls 1–4 PASS at both seeds before"
    " any number was read. The MM re-read (`raw/2026-10-25-speedcal/mm_verify/`) reproduces every point estimate exactly and the intervals to"
    " the draw; it also recomputed ADE6 from the plans against HSPEC's GT (max abs 0.0) and checked that the lever ran (the calibrated plans"
    " moved by up to 6.80 / 1.56 m; the T code shifted on 100 % of windows). Descriptive: the map cuts `v_a_ms` RMSE to the GT from 1.92 to"
    " 1.33 m/s at s0 while ADE6 does not move, and the GT speed cells (T0) would cut ADE6 by 0.72 / 0.67 m. ⇒ The speed error O responds to is"
    " per-window, not a global affine bias, and speed is still the largest lever. Disclosed: the fit-set GT is the trainer's `join_train.npz`"
    " (the same v9 md5), not `v9_labels_train.npz`. Result: `RESULT_V8F_SPEEDCAL.md`. **Committed next: the speed lever moves to perception,"
    " which needs compute beyond box 2.** Its arm is `PREREG_V8F_LORA.md`: AR-2 PASSED in run 3 and projects 470 h per seed on box 2, so its"
    " venue rule names the A40. With HACC, AR-1 of the A40 readiness gate also holds, so the A40 go is asked for (`D-V8F-SPEED-PERCEPTION`,"
    " `PI_DECISION_QUEUE.md`). |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))

q = "Project Steering/PI_DECISION_QUEUE.md"
qt = git("show", f"{TIP}:{q}").decode("utf-8")
assert "D-V8F-HACC-T1KIN" not in qt and "D-V8F-SPEED-PERCEPTION" not in qt
qt = qt.rstrip("\n") + "\n" + open(MM + "PIQ_ADD.md", encoding="utf-8").read().replace("\r\n", "\n")
put("mods-rw", q, qt.encode("utf-8"))

msg = ("SPEEDCAL run: INFORMATION-LIMITED at both seeds (no affine speed map moves ADE6); MM re-read; two OPEN PI decisions filed\n\n"
       "PRIMARY ADE6 s0 +0.0374 [-0.0100,+0.0864], s1 -0.0058 [-0.0156,+0.0036]; CEILING (T0) above -F at both; 69/69 mutants killed.\n"
       "PI queue: D-V8F-HACC-T1KIN (HACC's T1-kin read on box 2, before or after OACC) and D-V8F-SPEED-PERCEPTION (the A40 for LoRA).\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "files +", 6)
