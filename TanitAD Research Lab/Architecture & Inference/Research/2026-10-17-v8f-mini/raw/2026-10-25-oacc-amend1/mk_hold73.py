"""Build qland/hold_v8f_oacc_amend1: PREREG_V8F_OACC Amendment 1 (eight readings the instrument build had to fix, written before any OACC
data; no bar changes) + register note on H-V8F-OACC-1. python mk_hold73.py <tip>"""
import os
import subprocess
import sys

TIP = sys.argv[1]
A2 = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/grid_a2/"
H = "C:/Users/Admin/qland/hold_v8f_oacc_amend1"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"


def git(*a):
    r = subprocess.run(["git", "-C", MIR, "-c", "safe.directory=*", *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def put(kind, rel, data):
    p = os.path.join(H, kind, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(data)


for n in ("hold_v8f_oacc_amend1", "landed_hold_v8f_oacc_amend1"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n
pr = "Project Steering/PREREG_V8F_OACC.md"
old = git("show", f"{TIP}:{pr}")
assert old is not None and b"Amendment 1" not in old
amend = (
    "\n\n## Amendment 1 (2026-10-25, MM): written BEFORE any OACC data (the instruments are built, not landed; nothing has trained)\n\n"
    "Building the instruments needed eight readings that the text leaves open. They are fixed here. **No bar, rule, mask or seed order"
    " changes.**\n\n"
    "**The weight rule**\n"
    "* **R1.** $\\mathcal{L}_{traj}$ is the trainer's own trajectory term, `w_traj·(ADE/3 + 0.5·FDE/6)`, as it enters the `total` loss of"
    " `v8f_mini_train.py`. It is evaluated with B's s0 checkpoint on the training windows of HBLEND's fit set.\n"
    "* **R2.** $\\mathcal{L}_{acc}$ is this PREREG's term at unit weight: the second difference on the 0.2 s grid, origin knot included,"
    " averaged over GT-valid steps.\n"
    "* **R3.** $w_{acc}$ is rounded to 6 significant figures and written into X's argv. The box-2 stage recomputes it on the checkpoint, and the"
    " result quotes that value.\n"
    "  * Offline dry check on HBLEND's banked B_s0 arrays (9,582 windows): $\\mathcal{L}_{traj}$ 1.24598, $\\mathcal{L}_{acc}$ 0.0145114,"
    " so $w_{acc}$ = **42.9311** (`raw/2026-10-25-oacc/wacc_offline_dry_check.json`).\n"
    "  * An ADE-only $\\mathcal{L}_{traj}$ would give 74.6976. It is shown beside and not used.\n\n"
    "**The reading**\n"
    "* **I1.** PRIMARY-KEEP's bar is F/2, with F the exact |ADE6(B_s0) − ADE6(B_s1)| = 0.12074 m. The bar is therefore **+0.06037 m**; the"
    " table's \"+0.060\" is its rounding.\n"
    "* **I2. Outcome partition**, applied in this order:\n"
    "  1. PRIMARY-ACC fails at any seed that has run ⇒ **NO EFFECT**.\n"
    "  2. Else PRIMARY-KEEP fails at any seed that has run ⇒ **COSTS POSITION**.\n"
    "  3. Else only seed 0 has run ⇒ **PENDING SEED 1**. This is not an outcome: seed 1 runs, as the table's DOMINATES row requires.\n"
    "  4. Else DOMINANCE at both seeds ⇒ **DOMINATES ITS PRIOR**.\n"
    "  5. Else **SMOOTHER, NOT DOMINANT**.\n"
    "* **I3.** A pass at seed 0 alone is PENDING SEED 1, never a dominance claim (this is I2's third step).\n"
    "* **I4.** \"Separated-better\", \"not separated-worse\" and the distance-keeping directions follow HSPEC's reader conventions, so"
    " DOMINANCE is HSPEC's PRIMARY-DOM row verbatim.\n"
    "* **I5.** The STRATEGIC family is **read** for X and HX. S is retrained in X, so it is not identical to B by construction. It is not"
    " paused for v8F (`PI-V8F-2026-10-17`; R72 / R73).\n\n"
    "The instruments land only after their mutation proof is clean. The first proof had 44 surviving mutants, several of them on the reading"
    " above (the ACC and KEEP metrics, HX's re-weighting, DOMINANCE's pair). Killing tests are being written, and the proof is then re-run.\n")
put("mods-rw", pr, old.replace(b"\r\n", b"\n").rstrip(b"\n") + amend.encode("utf-8"))
put("stage", f"{PK}/raw/2026-10-25-oacc-amend1/mk_hold73.py", open(A2 + "mk_hold73.py", "rb").read().replace(b"\r\n", b"\n"))

g = "Project Steering/GOALS_AND_CLAIMS.md"
lines = git("show", f"{TIP}:{g}").decode("utf-8").split("\n")
ix = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-OACC-1** (2026-10-25) |")]
assert len(ix) == 1
lines[ix[0]] = lines[ix[0]].rstrip().rstrip("|").rstrip() + (
    " **Amendment 1 (2026-10-25, before data):** fixes eight readings: the terms of the w_acc rule (offline dry check w_acc = 42.9311), the"
    " KEEP bar (+0.06037 m) and the order of the outcome partition. No bar changes. Instruments built and NOT landed: the first mutation"
    " proof had 44 survivors, and killing tests are in progress. |")
put("mods-rw", g, "\n".join(lines).encode("utf-8"))
msg = ("PREREG_V8F_OACC Amendment 1: eight readings fixed before any OACC data (w_acc rule terms, KEEP bar, outcome order)\n\n"
       "No bar changes. Offline dry check w_acc = 42.9311 (B_s0, HBLEND fit windows). Instruments not landed: the mutation proof is being hardened.\n\n"
       "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
open(os.path.join(H, "msg.txt"), "wb").write(msg.encode("utf-8"))
print("hold ok")
