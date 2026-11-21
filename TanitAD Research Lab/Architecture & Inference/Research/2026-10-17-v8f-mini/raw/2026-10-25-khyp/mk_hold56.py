"""Build qland/hold_v8f_khyp: PREREG_V8F_KHYP's run banked (RESULT_V8F_KHYP.md + raw/2026-10-25-khyp/, without the code export and the
   two >10 MB training-window hypothesis arrays), its committed next PREREG_V8F_KSCORE, and the register: H-V8F-KHYP-1 (HEADROOM, NO
   SELECTOR), H-V8F-KSCORE-1 (new). python mk_hold56.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
KH, A2 = S + "khyp/", S + "grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_khyp"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RAW = f"{PK}/raw/2026-10-25-khyp/"
BIN = (".npz", ".npy", ".pt", ".png")


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


for n in ("hold_v8f_khyp", "landed_hold_v8f_khyp"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
n, big = 0, []
for root, dirs, files in os.walk(KH):
    for f in files:
        rel = os.path.relpath(os.path.join(root, f), KH).replace("\\", "/")
        if rel.startswith("code/") or "__pycache__" in rel or ".pytest_cache" in rel:
            continue
        if os.path.getsize(KH + rel) >= 10_000_000:
            big.append(f"{rel} ({os.path.getsize(KH + rel)} bytes)"); continue
        stage(KH + rel, RAW + rel); n += 1
put("stage", RAW + "NOT_BANKED.txt", ("Kept on the dev box only (each over the 10 MB per-file cap; regenerable with run_phaseB.sh):\n"
    + "\n".join(big) + "\n").encode("utf-8"))
stage(A2 + "mk_hold56.py", RAW + "mk_hold56.py")
stage(KH + "KHYP_RESULT.md", f"{PK}/RESULT_V8F_KHYP.md")
stage(A2 + "PREREG_V8F_KSCORE.md", "Project Steering/PREREG_V8F_KSCORE.md")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-KHYP-1** (2026-10-25) |")]
assert len(ix) == 1
pre = "| **PRE-REGISTERED, NOT RUN (2026-10-25).** "
assert lines[ix[0]].count(pre) == 1
new = ("| **HEADROOM, NO SELECTOR, 2026-10-25. MEASURED, eval-141, paired episode-cluster bootstrap; `…/2026-10-17-v8f-mini/raw/2026-10-25-khyp/READ.json`,"
       " `RESULT_V8F_KHYP.md`.** HEADROOM (T0, oracle best of 7 longitudinal codes through O) − B, ADE6: −0.423 [−0.466, −0.383] / −0.359 [−0.407,"
       " −0.315] m. PRIMARY SEL-LEARN − B (T1-class open loop): −0.098 [−0.141, −0.059] (separated, short of −F) / +0.020 [−0.006, +0.047] (not"
       " separated): fails. SECONDARY (speed, accel) met. Controls: B's own-code ADE6 reproduced to 8e-8 m; harness identity 0.0; control 3 (permuted"
       " labels) FAILED at s0 (−0.107 vs SEL-ARGMAX, separated), so the s0 near-miss is not selection. Post hoc, no bar: SEL-LEARN picks one class on"
       " 89 % of s0 windows and equals 'always class 5' (−0.0996); the best constant differs at s1 (class 6, −0.019); its accuracy against k* (0.513 /"
       " 0.358) is the majority class's (0.504 / 0.356). Declared deviation: the distance-keeping reader's tolerance was relaxed from 1e-5 to 1e-4 m"
       " (float32), a reader work item. ⇒ Committed next: `PREREG_V8F_KSCORE`. ~~" + pre[2:].strip() + "~~ ")
lines[ix[0]] = lines[ix[0]].replace(pre, new)
assert not any(x.startswith("| **H-V8F-KSCORE-1**") for x in lines)
row = ("| **H-V8F-KSCORE-1** (2026-10-25) | A small scorer trained on the full training set (B frozen), reading O's 7 longitudinal hypotheses, T's"
       " softmax, B's scene tokens and the ego token, picks a hypothesis that beats B by ≤ −F AND beats the best train-chosen constant class, both"
       " separated at both seeds, without a separated-worse speed, accel, consistency or LEAD-VIOL | **PRE-REGISTERED, NOT RUN (2026-10-25).**"
       " `Project Steering/PREREG_V8F_KSCORE.md`. KHYP's committed next; the constant-class baseline guards against a bias shift passing as"
       " selection. Box 2 after OACC, behind a scorer launch gate. |")
lines[ix[0] + 1:ix[0] + 1] = [row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_KHYP: HEADROOM, NO SELECTOR -> PREREG_V8F_KSCORE pre-registered (trained scene-reading scorer)\n\n"
       "MEASURED, eval-141, paired episode-cluster bootstrap. Oracle best-of-7 longitudinal codes through O: ADE6 -0.42 / -0.36 m vs B (T0\n"
       "headroom). The deployable selector fails (-0.098 sep. / +0.020 ns); at s0 it equals a constant class, and the permuted-label control\n"
       "fails there. KSCORE trains a small scorer on the full training set with B frozen, and must beat the best constant class too.\n"
       "Register: H-V8F-KHYP-1 verdict, H-V8F-KSCORE-1.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok", n, "raw files; not banked:", big)
