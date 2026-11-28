"""Build qland/hold_v8f_dominance_preregs: PREREG_V8F_HSPEC, PREREG_V8F_KHYP, PREREG_V8F_DSEFF (before any of their data)
   and register rows H-V8F-HSPEC-1, H-V8F-KHYP-1, H-V8F-DSEFF-1, plus PI-V8F-2026-10-25c (the PI's go, verbatim).
   python mk_hold49.py <remote tip sha>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_dominance_preregs"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def show(rev, path):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", "show", f"{rev}:{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


def stage(src, rel):
    assert show(TIP, rel) is None, f"{rel} already exists on the tip"
    put("stage", rel, open(src, "rb").read().replace(b"\r\n", b"\n"))


for n in ("hold_v8f_dominance_preregs", "landed_hold_v8f_dominance_preregs"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
for p in ("HSPEC", "KHYP", "DSEFF"):
    stage(A2 + f"PREREG_V8F_{p}.md", f"Project Steering/PREREG_V8F_{p}.md")
stage(A2 + "mk_hold49.py", f"{PK}/raw/2026-10-25-dominance/mk_hold49.py")
lines = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-LISTEN-1** (2026-10-25) |")]
assert len(ix) == 1
for r in ("H-V8F-HSPEC-1", "H-V8F-KHYP-1", "H-V8F-DSEFF-1", "PI-V8F-2026-10-25c"):
    assert not any(x.startswith(f"| **{r}**") for x in lines), r
PRE = " **PRE-REGISTERED, NOT RUN (2026-10-25).** "
rows = [
    ("| **H-V8F-HSPEC-1** (2026-10-25) | H with its residual also shrunk per cosine coefficient (weights fitted on training windows; a"
     " second-difference objective as an automatic stage 2) dominates P10-yaw-4512 on the four families: separated-better on ADE2, ADE6, accel"
     " MAE and curvature MAE, and not separated-worse on any other family metric, at both seeds, while keeping H's ADE6 (upper bound ≤ +F/2) |"
     + PRE + "`Project Steering/PREREG_V8F_HSPEC.md`. Why: H is separated-worse than P10 on exactly two metrics, accel (+0.104 / +0.066)"
     " and curvature (+0.0074 / +0.0110), both second-derivative (`RESULT_V8F_HBLEND.md` §6). Zero training, dev-box CPU. Dominance over"
     " the KINEMATIC FLOOR only, never a state-of-the-art claim. |"),
    ("| **H-V8F-KHYP-1** (2026-10-25) | Enumerating T's longitudinal class (7 one-hot codes) through O and picking with a deployable"
     " selector (logistic regression on model outputs, fitted on training windows) improves ADE6 over B by ≤ −F, separated, at both seeds,"
     " with speed and accel MAE not separated-worse |" + PRE + "`Project Steering/PREREG_V8F_KHYP.md`. The ORACLE best-of-7 is reported"
     " as T0 headroom only. Zero model training, dev-box CPU. Outcomes: MULTIMODAL PAYS (⇒ a trained K-hypothesis head on box 2) /"
     " HEADROOM, NO SELECTOR / NO HEADROOM. |"),
    ("| **H-V8F-DSEFF-1** (2026-10-25) | The hierarchy is more data-efficient than FLAT: halving the training clips costs FLAT more ADE6"
     " than it costs M2b (paired interaction > 0, separated, both seeds), and M2b at 50 % is at least as good as FLAT at 100 % |" + PRE +
     "`Project Steering/PREREG_V8F_DSEFF.md`. New arms: FLAT-50 s0 / s1 and M2b-50 s1 (the DSCALE arm is s0 iff its argv diff is"
     " exactly `--data-frac 0.5`). Box 2 after FLAT, each behind a launch-gate PASS. |"),
    ("| **PI-V8F-2026-10-25c** (2026-10-25) | PI go on the five proposed dominance ideas | **RECORDED.** Sayed, verbatim: *\"Go ahead with"
     " your proosed ideasss Rememer that our prioriy is to proof ry quickly doominance\"*. It answers the five ideas in the 2026-10-25 answer"
     " to his report questions. Idea 1 (train T through O) was found to be ALREADY ON in B (`detach_upper` default False, not set by the M2b"
     " preset), so it adds no arm. The A40 / NAVSIM part (idea 5) is scoped, and the A40 is asked for separately. |"),
]
lines[ix[0] + 1:ix[0] + 1] = rows
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
msg = ("Dominance pre-registrations: HSPEC (accel/curvature fix, zero training), KHYP (longitudinal hypotheses + selector), DSEFF (data efficiency vs FLAT)\n\n"
       "Written before any of their data, under the PI's 2026-10-25 go on the proposed dominance ideas (recorded verbatim as PI-V8F-2026-10-25c).\n"
       "HSPEC: H is separated-worse than P10 on exactly accel and curvature; shrink the residual per cosine coefficient (stage 2: a second-\n"
       "difference objective). KHYP: 7 one-hot longitudinal codes through O, a deployable logistic selector, PRIMARY ADE6 <= -F vs B.\n"
       "DSEFF: FLAT-50 and M2b-50 against the 100 % arms, PRIMARY = the paired data-cost interaction.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
