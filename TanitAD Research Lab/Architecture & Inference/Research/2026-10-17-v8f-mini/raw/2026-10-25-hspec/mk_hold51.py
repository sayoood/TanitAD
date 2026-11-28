"""Build qland/hold_v8f_hspec: PREREG_V8F_HSPEC's run banked (RESULT_V8F_HSPEC.md + raw/2026-10-25-hspec/), its committed next
   PREREG_V8F_OACC (before any OACC data), and the register: H-V8F-HSPEC-1 (SMOOTHER, NOT DOMINANT), H-V8F-OACC-1 (new).
   python mk_hold51.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
HS, A2 = S + "hspec/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_hspec"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-hspec/"
BIN = (".npz", ".npy", ".pt")


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


for n in ("hold_v8f_hspec", "landed_hold_v8f_hspec"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n = 0
for f in sorted(os.listdir(HS)):
    if not os.path.isfile(HS + f) or f.startswith("_v8f_mini_") or f.startswith("."):
        continue
    assert os.path.getsize(HS + f) < 10_000_000, f
    stage(HS + f, RAW + f); n += 1
stage(A2 + "mk_hold51.py", RAW + "mk_hold51.py")
stage(HS + "HSPEC_RESULT.md", f"{PK}/RESULT_V8F_HSPEC.md")
stage(A2 + "PREREG_V8F_OACC.md", "Project Steering/PREREG_V8F_OACC.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-HSPEC-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **SMOOTHER, NOT DOMINANT (stage 2, final), 2026-10-25. MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap;"
       " `…/2026-10-17-v8f-mini/raw/2026-10-25-hspec/READ.json`, `RESULT_V8F_HSPEC.md`.** Stage 1 PRIMARY-DOM failed at both seeds, so stage 2"
       " ran automatically (λ = 2500 / 2308 by the train-only rule). Stage 2 HS − P10: ADE2 −0.0051 [−0.0081, −0.0019] / −0.0053 [−0.0090,"
       " −0.0012]; ADE6 −0.511 [−0.617, −0.411] / −0.490 [−0.598, −0.385]; accel MAE +0.0258 [+0.0105, +0.0444] (s0, separated-worse) / +0.0057"
       " [−0.011, +0.028]; curvature MAE +0.0012 / +0.0031 (not separated). Every other family row is not separated-worse at both seeds,"
       " consistency 'any' and distance keeping included. HS − H on accel and curvature is separated-better at both seeds (accel −0.078 / −0.060)."
       " PRIMARY-KEEP passes (ADE6 HS − H +0.007 / +0.012, bar ≤ +0.060). Controls 1–4 pass; one disclosed flip on a hard-threshold count"
       " (`zero_accel_frac`, 7 of 69,571 cells, float32). Post hoc, no bar: weights fitted on the eval itself still leave accel separated-worse at s0."
       " HS keeps H's consistency cost against B ('any' +0.101 / +0.069). ⇒ Committed next: `PREREG_V8F_OACC`. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
assert not any(x.startswith("| **H-V8F-OACC-1**") for x in lines)
row = ("| **H-V8F-OACC-1** (2026-10-25) | Training O with an added acceleration-matching term (second difference of its 0.2 s plan against the GT's,"
       " weight fixed by a pre-launch rule) lowers accel MAE against B (separated) without losing ADE6 (upper bound ≤ +F/2), and its deployed"
       " H-blended form dominates P10-yaw-4512 on HSPEC's row at both seeds | **PRE-REGISTERED, NOT RUN (2026-10-25).**"
       " `Project Steering/PREREG_V8F_OACC.md`. HSPEC's committed next. New flag `--w-acc` with G-DVB, G-LIVE and a deliberate regression."
       " Box 2 after FLAT; the build waits for FLAT's instruments (same files). |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_HSPEC: SMOOTHER, NOT DOMINANT -> PREREG_V8F_OACC pre-registered (acceleration-matching loss on O)\n\n"
       "MEASURED, T1-class open loop, eval-141, paired episode-cluster bootstrap; controls passed. Stage 2 HS - P10: ADE2 -0.005/-0.005,\n"
       "ADE6 -0.511/-0.490 (separated); accel +0.026 (s0, sep. worse) / +0.006 (ns); curvature ns at both seeds; every other family row\n"
       "not separated-worse. Post-processing cannot reach P10's accel; O is trained on positions only. OACC adds --w-acc (weight fixed by a\n"
       "pre-launch rule), box 2 after FLAT. Register: H-V8F-HSPEC-1 verdict, H-V8F-OACC-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files")
