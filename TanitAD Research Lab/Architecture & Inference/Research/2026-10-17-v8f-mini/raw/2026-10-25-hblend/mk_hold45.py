"""Build qland/hold_v8f_hblend_pass: PREREG_V8F_HBLEND's run banked (RESULT_V8F_HBLEND.md + raw/2026-10-25-hblend/) and the register verdict on
   H-V8F-HBLEND-1 (PASS at both seeds; the CONSISTENCY cost and the accel non-dominance stated beside it).   python mk_hold45.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
HB, A2 = S + "hblend/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_hblend_pass"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-hblend/"
BIN = (".npz", ".npy", ".pt")
SKIP_DIRS = ("cache/", "deps/", "inputs/banked/")                 # the train-clip cache (2.8 GB), copies of repo code, copies of banked evals
SKIP = {"box2/join_train.npz", "box2/manifest_train.jsonl", "fit_windows_smoke.npz"}   # box-2 inputs copied for the fit (provenance: box2/*.sftp, get_clips.out)


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    assert len(rel) <= 220, rel
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    b = open(src, "rb").read()
    put("stage", rel, b if rel.endswith(BIN) else b.replace(b"\r\n", b"\n"))


for n in ("hold_v8f_hblend_pass", "landed_hold_v8f_hblend_pass"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for root, dirs, files in os.walk(HB):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), HB).replace("\\", "/")
        if rel.startswith(SKIP_DIRS) or rel in SKIP or "__pycache__" in rel:
            continue
        assert os.path.getsize(HB + rel) < 10_000_000, rel
        stage(HB + rel, RAW + rel); n += 1
stage(A2 + "mk_hold45.py", RAW + "mk_hold45.py")
stage(HB + "HBLEND_RESULT.md", f"{PK}/RESULT_V8F_HBLEND.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HBLEND-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **PASS at both seeds (as committed), 2026-10-25. MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap B = 2,000;"
       " `…/2026-10-17-v8f-mini/raw/2026-10-25-hblend/READ.json`, `RESULT_V8F_HBLEND.md`.**"
       " ADE2 H − B = −0.103 [−0.141, −0.075] / −0.078 [−0.115, −0.053] m (s0 / s1, separated). ADE6 H − B = −0.207 [−0.277, −0.142] /"
       " −0.070 [−0.101, −0.042] m (upper bounds far below the +0.060 bar). Accel MAE H − B = −0.464 / −0.228 m/s²."
       " **Against P10:** ADE6 H − P10 = −0.519 [−0.635, −0.407] / −0.502 [−0.621, −0.385] m, separated, about 2 × the 2F bar; ADE2 is"
       " also below P10's. **H does NOT dominate P10:** accel MAE is still worse (+0.104 / +0.066, separated)."
       " The weights were fitted on 9,582 TRAINING windows (k = 32, 141 clips), one closed form per step and axis on the 0.2 s grid: 60 constants per"
       " seed, not the 120 the PREREG text says. Along-track w ≈ 0 to 1 s, ≈ 0.1 to 2.4 s, then 0.43–0.64; cross-track 0.66–0.89. So O's"
       " longitudinal residual is the over-confident part. The train fit sits within 0.004 m ADE6 of the eval-oracle weights. Identity controls"
       " pass: w ≡ 1 gives Δ = 0.0 and 0 mismatches on the eval's own `evaluate()`; w ≡ 0 gives ADE6 2.48699 = P10."
       " **A COST BESIDE THE BARS, and it blocks deployment as is:** CONSISTENCY(a), the share of plans violating a GT-derived constraint, is"
       " WORSE, separated, at both seeds: 'any' +0.101 / +0.075, almost all of it SPEED (+0.100 / +0.098); self-consistency (b) +0.140 / +0.074."
       " Distance keeping is mixed (s0 worse on headway and time gap, s1 nothing separated). Mechanism (HYPOTHESIS): B's headline metrics are"
       " computed on the UNPROJECTED plan, so the cost is not a projection being undone. Shrinking O's along-track residual toward P10, a"
       " kinematic prior that knows no limit or stop, removes part of the residual's constraint-following."
       " ⇒ Before deployment, the deployable hard projection (`project_speed`) is applied AFTER H under its own PREREG (`PREREG_V8F_HPROJ`);"
       " then the committed next (weights as a buffer; FLAT and MIXO re-scored) uses that form. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
stage(A2 + "PREREG_V8F_HPROJ.md", "Project Steering/PREREG_V8F_HPROJ.md")
assert not any(x.startswith("| **H-V8F-HPROJ-1**") for x in lines)
hp = ("| **H-V8F-HPROJ-1** (2026-10-25) | HBLEND followed by the deployable hard projection (`project_speed`: speed ≤ the max-speed input, Kamm"
      " circle; geometry kept, timing changed) keeps H's 2 s and 6 s gains over B without H's consistency cost. Zero training, no weight re-fitted."
      " Bars: CONSISTENCY(a) 'any' HP − B upper 95 % bound ≤ +0.02 AND ADE6 / ADE2 HP − B < 0 separated, at BOTH seeds |"
      " **PRE-REGISTERED, NOT RUN (2026-10-25).** `Project Steering/PREREG_V8F_HPROJ.md`. Offline from banked outputs: traj60 is recovered"
      " exactly from the banked 30-step plan, under a reconstruction control that must reproduce B's banked `projected_r1_kamm`. Dev-box CPU,"
      " under `PI-V8F-2026-10-25b`. Outcomes: PASS (HP is the deployed v8F output) / CONSISTENCY NOT RESTORED / GAIN LOST, both of which lead to"
      " a constraint-aware fit. |")
lines[ix[0] + 1:ix[0] + 1] = [hp]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_HBLEND: PASS at both seeds -- zero-training residual shrinkage beats B at 2 s AND 6 s, and P10 by ~0.5 m at 6 s\n\n"
       "MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap. Weights fitted on TRAINING windows only (9,582 windows).\n"
       "ADE2 H-B -0.103/-0.078 m sep.; ADE6 H-B -0.207/-0.070 m sep.; accel MAE -0.464/-0.228. vs P10: ADE6 -0.519/-0.502 sep. (~2x 2F).\n"
       "Not dominance: accel MAE still worse than P10 (+0.104/+0.066 sep.). Cost beside the bars: CONSISTENCY(a) worse +0.10/+0.075\n"
       "(speed): shrinking toward a constraint-blind prior -> PREREG_V8F_HPROJ (the deployable hard projection after H) before deployment.\n"
       "Register: H-V8F-HBLEND-1.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files")
