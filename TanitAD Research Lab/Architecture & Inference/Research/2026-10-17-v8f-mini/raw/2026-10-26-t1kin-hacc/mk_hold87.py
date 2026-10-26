"""Build qland/hold_v8f_t1kin_result: the T1-kin read of HACC (outcome T1 BEATS THE FLOOR), its runs and reads, the MM verify and the
descriptive diagnosis, PREREG_V8F_HACC_LAT (before any of its data), the register rows, and the PI item D-V8F-SPEEDBUCKET-T1.  python mk_hold87.py <tip>"""
import glob
import os
import re
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
T = SC + "t1kin/"
RUN = "C:/Users/Admin/t1k_run/"
H = "C:/Users/Admin/qland/hold_v8f_t1kin_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RD = f"{PK}/raw/2026-10-26-t1kin-hacc"
TOKEN = re.compile(rb"hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}")


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data, text=True):
    assert not TOKEN.search(data), f"token-shaped string in {rel}"
    if kind == "stage":
        assert git("rev-parse", f"{TIP}:{rel}") is None, f"NEW path exists at the tip: {rel}"
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data.replace(b"\r\n", b"\n") if text else data)


for n in ("hold_v8f_t1kin_result", "landed_hold_v8f_t1kin_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
chain = open(RUN + "logs/chain.out", encoding="utf-8").read()
assert "ZZT1K-DONE 12" in chain, "the chain has not finished all 12 runs"
put("stage", f"{PK}/RESULT_V8F_T1KIN_HACC.md", open(T + "RESULT_V8F_T1KIN_HACC.md", "rb").read())
put("stage", "Project Steering/PREREG_V8F_HACC_LAT.md", open(T + "PREREG_V8F_HACC_LAT.md", "rb").read())
for f in ("t1k_read.py", "consolidate.py", "curv_profile.log", "curv_profile.json", "viewpoint_probe.log", "speed_probe.log"):
    put("stage", f"{RD}/{f}", open(T + f, "rb").read())
for f in ("verify_t1k_mm.log", "verify_t1k_mm.json"):
    put("stage", f"{RD}/mm_verify/{f}", open(T + f, "rb").read())
for sub in ("read2", "read3", "read4"):
    for f in glob.glob(T + sub + "/*"):
        put("stage", f"{RD}/{sub}/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(RUN + "t1v8f_*.json")):
    put("stage", f"{RD}/run/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(RUN + "logs/*")):
    put("stage", f"{RD}/run/logs/{os.path.basename(f)}", open(f, "rb").read())
for f in sorted(glob.glob(T + "consolidated/*")):
    put("stage", f"{RD}/run/dumps/{os.path.basename(f)}", open(f, "rb").read(), text=not f.endswith(".npz"))
put("stage", f"{RD}/run/t1k_chain.sh", open("C:/Users/Admin/t1k_chain.sh", "rb").read())
put("stage", f"{RD}/run/t1k_chain.cmd", open("C:/Users/Admin/t1k_chain.cmd", "rb").read(), text=False)
put("stage", f"{RD}/mk_hold87.py", open(SC + "grid_a2/mk_hold87.py", "rb").read())

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-T1KIN-HACC-1** (2026-10-26) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **RUN (2026-10-26): T1 BEATS THE FLOOR** (the table applied verbatim; MEASURED, T1-kin m = 10, paired episode-cluster bootstrap, B = 2,000,"
    " 141 episodes; `RESULT_V8F_T1KIN_HACC.md`). PRIMARY HACC − P10 ADE6 **−0.8291 [−0.9711, −0.6894] / −0.7945 [−0.9379, −0.6575] m**, separated"
    " at both seeds, beyond −2F. ADE2, speed, along-track and cross-track are separated-better at both seeds, and accel and heading at s1."
    " DOMINANCE fails on LATERAL: curvature +0.0400 / +0.0155 1/m and yaw rate +4.20 deg/s (s0), separated-worse, concentrated at GT speed below"
    " 5.2 m/s (descriptive). Controls all PASS after Amendment 3 (`R74`); the refused read is banked unopened and the re-read equals it except"
    " C-1's verdict. Independently re-derived. Beside: the cameras frozen at t0 keep −0.79 / −0.71 m. **⚠ With the speed bucket removed the"
    " margin is not separated (+0.24 / +0.08 m):** the T1 margin rests on the declared bucket, which on PhysicalAI comes from the ego's realised"
    " future speed (`D-V8F-SPEEDBUCKET-T1`). Plain M2b at T1 is +1.22 / +1.08 m/s² rougher than the floor: the post-map is what keeps the loop"
    " smooth. Next (committed): LATERAL, `PREREG_V8F_HACC_LAT.md`. |")
lines.insert(ix[0] + 1, (
    "| **H-V8F-HACC-LAT-1** (2026-10-26) | A zero-training refinement of HACC removes its T1-kin lateral deficit against P10 while keeping the margin:"
    " at T1-kin m = 10, arm − P10 meets DOMINANCE (ADE6 ≤ −2F, interval excluding 0, no four-family row separated-worse) and arm − HACC ADE6 has"
    " its upper bound ≤ +F/2, at BOTH seeds | **PRE-REGISTERED, NOT RUN (2026-10-26).** `Project Steering/PREREG_V8F_HACC_LAT.md`, written before"
    " any number of either arm. Arms, both fitted on HBLEND's 141 TRAINING clips at T1-kin ticks: A, HACC's weights and λ rule re-fitted on the"
    " tick samples (committed by `PREREG_V8F_T1KIN_HACC`'s outcome row); B, the cross axis falls back to P10 below a speed gate v\\* ∈ {2, 4, 6, 8}"
    " m/s chosen on train. Outcomes: T1 DOMINANCE REACHED / LATERAL FIXED AT A COST / NOT FIXED (⇒ a lateral training lever on box 2) / MIXED."
    " Dev-box CPU. |"))
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
q = "Project Steering/PI_DECISION_QUEUE.md"
qt = git("show", f"{TIP}:{q}").decode("utf-8")
assert "D-V8F-SPEEDBUCKET-T1" not in qt
put("mods-rw", q, (qt.rstrip("\n") + "\n" + open(T + "PIQ_ADD2.md", encoding="utf-8").read().replace("\r\n", "\n")).encode("utf-8"))
msg = ("T1-kin read of HACC: T1 BEATS THE FLOOR at both seeds (ADE6 -0.83 / -0.79 m vs P10); not dominance (low-speed curvature); PREREG_V8F_HACC_LAT\n\n"
       "Self-action open loop, m = 10, paired episode-cluster bootstrap. Camera frozen keeps -0.79 / -0.71 m; speed bucket off removes the margin\n"
       "(filed for the PI). Controls pass after Amendment 3 (R74); refused read banked unopened; independently re-derived.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
