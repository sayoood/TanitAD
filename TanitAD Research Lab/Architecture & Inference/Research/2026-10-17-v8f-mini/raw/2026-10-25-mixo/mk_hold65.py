"""Build qland/hold_v8f_mixo_result: RESULT_V8F_MIXO.md + raw/2026-10-25-mixo/run/ (read, evals, logs, run metadata; no checkpoint) + hand/ re-derivation + register H-V8F-MIXO-1 status. python mk_hold65.py <tip>"""
import glob
import hashlib
import os
import subprocess
import sys

TIP = sys.argv[1]
SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
M = SC + "mixo_hand/"
A2 = SC + "grid_a2/"
LOG = "C:/Users/Admin/box2_ship/logs/"
E = LOG + "v8f_mixo/"
H = "C:/Users/Admin/qland/hold_v8f_mixo_result"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-mixo"
TEXT = (".md", ".json", ".py", ".log", ".txt", ".out", ".jsonl", ".csv", ".sh", ".cmd")


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, rel
    b = open(src, "rb").read()
    if src.endswith(TEXT):
        b = b.replace(b"\r\n", b"\n")
    put("stage", rel, b)


for n in ("hold_v8f_mixo_result", "landed_hold_v8f_mixo_result"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
stage(M + "RESULT_V8F_MIXO.md", f"{PK}/RESULT_V8F_MIXO.md")
for f in ("mixo_read_s0.json", "check_inputs.json", "ckpt_sha256.json", "gate_token.json", "telemetry_v8fmixo.csv"):
    stage(E + f, f"{RAW}/run/{f}")
for f in sorted(glob.glob(E + "eval/*")):
    stage(f, f"{RAW}/run/eval/{os.path.basename(f)}")
for f in sorted(glob.glob(E + "gate_M2b-mixo/*")):
    if os.path.isfile(f):
        stage(f, f"{RAW}/run/gate_M2b-mixo/{os.path.basename(f)}")
for g in ("M2b_s0", "M2b_s0_mixo"):
    for f in sorted(glob.glob(E + f"runs/{g}/*")):
        if not f.endswith(".pt"):
            stage(f, f"{RAW}/run/runs/{g}/{os.path.basename(f)}")
ck = E + "runs/M2b_s0_mixo/ckpt_last.pt"
sha = hashlib.sha256(open(ck, "rb").read()).hexdigest()
assert sha == "c6a628762790b4fc3f7634867fc15b8167b035692ff518203c9fc0b3c126250d", sha
put("stage", f"{RAW}/run/runs/M2b_s0_mixo/NOT_BANKED.txt", (
    f"ckpt_last.pt ({os.path.getsize(ck)} bytes) is not in git. sha256 {sha}.\n"
    "Copies: box 2 D:/tanitad/runs/v8f_mini_full/M2b_s0_mixo/ckpt_last.pt and the dev box C:/Users/Admin/box2_ship/logs/v8f_mixo/runs/M2b_s0_mixo/ckpt_last.pt.\n").encode())
for f in ("chain_v8f_mixo.out", "v8fmixo_guard.out", "v8fmixo10250925_gatemixo.log", "v8fmixo10250925_refresh.log",
          "v8fmixo10250925_t_M2b_s0_mixo.log", "v8fmixo10250925_e_M2b_s0_mixo.log"):
    if os.path.exists(LOG + f):
        stage(LOG + f, f"{RAW}/run/logs/{f}")
for f in ("verify_mixo_s0.py", "verify_mixo_s0.out"):
    stage(M + f, f"{RAW}/hand/{f}")
stage(A2 + "mk_hold65.py", f"{RAW}/mk_hold65.py")

lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-MIXO-1** (2026-10-25) |")]
assert len(ix) == 1
old = "**PRE-REGISTERED, CONDITIONAL, NOT RUN (2026-10-25).**"
assert lines[ix[0]].count(old) == 1
lines[ix[0]] = lines[ix[0]].replace(old, (
    "**FAIL, O LEARNED (s0, MEASURED 2026-10-25; seed 1 not trained, as committed).** PRIMARY own-code ADE6 X − B +0.022 [−0.059, +0.101] m"
    " (T1-class open loop; paired episode-cluster bootstrap, B = 2,000, 141 episodes; bar ≤ −0.2415 missed). D-M r_X(0.5) 0.644 against r_B 0.365"
    " (committed 0.36): rise +0.284 ≥ 0.10 (T0). HARM GT-code oracle +0.400 [+0.347, +0.449] m (T0). Verdict RE-DERIVED BY HAND from the per-window"
    " arrays (the reader's −2F and O-LEARNED branches had survived its mutation proof): identical to 1e-15 on points. Beside: accel MAE −0.116"
    " [−0.156, −0.074] m/s² (X better); yaw-rate MAE +0.526 [+0.179, +0.965] °/s (X worse); Λ 0.268 → 0.047 m. Next: `PREREG_V8F_KINDX`."
    " `RESULT_V8F_MIXO.md`, `raw/2026-10-25-mixo/run/mixo_read_s0.json`, `raw/2026-10-25-mixo/hand/`. Was:"
    " **PRE-REGISTERED, CONDITIONAL, NOT RUN (2026-10-25).**"))
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("RESULT_V8F_MIXO s0: FAIL, O LEARNED (verdict re-derived by hand)\n\n"
       "PRIMARY ADE6 X - B +0.022 [-0.059, +0.101] m (bar -0.2415 missed); r(0.5) 0.644 vs 0.365 (O now uses partial codes);\n"
       "HARM +0.400 m. Seed 1 not trained, as committed. Next: PREREG_V8F_KINDX. Run files, evals and the hand re-derivation banked.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
