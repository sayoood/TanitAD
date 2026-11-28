"""Build qland/hold_v8fdk_t1: v8F-mini distance keeping reached (reader, test, K=30 block, read, oracle split) + the T1-kin eval tool.
   python mk_hold13.py <remote tip sha>"""
import glob
import os
import subprocess
import sys

TIP = sys.argv[1]
S = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/"
DK, DL, T1 = S + "dk/", S + "dkland/", S + "t1v8f/"
H = "C:/Users/Admin/qland/hold_v8fdk_t1"
MIR = "C:/Users/Admin/tanitad-push"
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
RDK = f"{PK}/raw/2026-10-22-dk/"
RT1 = f"{PK}/raw/2026-10-22-t1/"


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


def append(rel, text):
    old = show(TIP, rel)
    assert old is not None, rel
    put("mods", rel, old.rstrip(b"\n") + b"\n" + lf(text.encode("utf-8")))


def rw(rel, pairs):
    old = show(TIP, rel)
    assert old is not None, rel
    s = old.decode("utf-8")
    for a, b in pairs:
        assert s.count(a) == 1, (rel, s.count(a), a[:80])
        s = s.replace(a, b)
    put("mods-rw", rel, lf(s.encode("utf-8")))


for n in ("hold_v8fdk_t1", "landed_hold_v8fdk_t1"):
    assert not os.path.exists("C:/Users/Admin/qland/" + n), n

# ---- distance keeping: the reader, its test, the result, the raw
stage(DK + "code/v8f_dk_read.py", f"{PK}/code/v8f_dk_read.py")
stage(DK + "tests/test_v8f_dk.py", "stack/tests/test_v8f_dk.py")
stage(DL + "RESULT_V8F_DK.md", f"{PK}/RESULT_V8F_DK.md")
stage(DL + "README_dk.md", RDK + "README.md")
stage(DK + "raw/lead_k30/b1_eval_lead_block_k30.npz", RDK + "b1_eval_lead_block_k30.npz", binary=True)
stage(DK + "raw/lead_k30/b1_eval_lead_block_k30.npz.report.json", RDK + "b1_eval_lead_block_k30.npz.report.json")
stage(S + "dkv_out/dk_read_mine.json", RDK + "dk_read.json")
stage(S + "dkv_out/dk_read_mine.out", RDK + "dk_read.out")
stage(S + "dkv_out/dk_pw_mine.npz", RDK + "dk_perwindow.npz", binary=True)
stage(DK + "raw/test_v8f_dk.out", RDK + "test_v8f_dk.out")
stage(DK + "mutate_dk.py", RDK + "mutate_dk.py")
stage(S + "dkv_mut/raw/mutation_dk.json", RDK + "mutation_dk.json")
stage(S + "dkv_mut/raw/mutation_dk_v.out", RDK + "mutation_dk.out")
for f in ("k30_control.py", "k30_control.out", "ttc_decomp.py", "ttc_decomp.json", "ttc_decomp.out", "ttc_tail_ci.py", "ttc_tail_ci.json",
          "ttc_tail_ci.out", "dk_oracle_read.py", "dk_oracle_read.json", "dk_oracle_read.out", "dk_oracle_loo.py", "dk_oracle_loo.json",
          "dk_oracle_loo.out", "latcons_mech.py", "latcons_mech.json", "latcons_mech.out", "mk_hold13.py"):
    stage(DL + f, RDK + f)

# ---- T1-kin: the tool, its test, the design and probes
stage(T1 + "tree/stack/scripts/v8f_t1_eval.py", "stack/scripts/v8f_t1_eval.py")
stage(T1 + "tree/stack/tests/test_v8f_t1.py", "stack/tests/test_v8f_t1.py")
stage(DL + "README_t1.md", RT1 + "README.md")
for f in ("DESIGN_T1_V8F.md", "mutate.py", "mutation_results.json", "smoke_real_ckpt.py"):
    stage(T1 + f, RT1 + f)
n = 0
for p in sorted(glob.glob(T1 + "probes/*")):
    stage(p, RT1 + "probes/" + os.path.basename(p))
    n += 1
assert n == 10, n
for p in sorted(glob.glob(T1 + "smoke_out/*.json")):
    stage(p, RT1 + "smoke_out/" + os.path.basename(p))

# ---- errata (append-only) on the docs that said "not instrumented"
append(f"{PK}/RESULT_FULL.md", """
## Erratum (2026-10-22): distance keeping is no longer UNAVAILABLE

* The LONGITUDINAL distance-keeping row above, and the two notes that repeat it, read UNAVAILABLE. That was true of this eval, not of the programme. The instrument (`taniteval/lead_metrics.py` on the B1 lead block, BACKLOG R1, 2026-09-02) existed; v8F-mini never called it.
* It is now read post hoc from these same eval files by `code/v8f_dk_read.py`: `RESULT_V8F_DK.md`, `raw/2026-10-22-dk/`. The table and notes above are left as written.
* Headline (MEASURED, T1-class open loop, descriptive, no bar):
  * at 6 s, M2b keeps more headway than P10: +0.53 / +0.42 m, separated at both seeds;
  * at 2 s, the learned arms close on the lead faster: min-TTC −1.76 / −1.19 s against P10, separated;
  * an exploratory T0 oracle read traces the 2 s closing to T's lateral constraint group (`latcons`).
""")
append(f"{PK}/RESULT.md", """
## Erratum (2026-10-22): distance keeping

* The families note above says the lead-track distance-keeping input "is not built -- a work item". It is now read post hoc from the v8F-mini evals by `code/v8f_dk_read.py` (`taniteval.lead_metrics` on the B1 lead block): see `RESULT_V8F_DK.md`.
""")
append(f"{PK}/RESULT_V8F_CZ.md", """
## Update (2026-10-22): distance keeping

* "Not instrumented" above is out of date: `code/v8f_dk_read.py` now reads distance keeping for M1, M2b and P10 (`RESULT_V8F_DK.md`). The M2b-cz arms were not read.
""")
append(f"{PK}/RESULT_V8F_WC.md", """
## Update (2026-10-22): distance keeping, and what it says about the next lever

* "Distance keeping is not instrumented here" is out of date. `code/v8f_dk_read.py` now reads it for M1, M2b and P10 (`RESULT_V8F_DK.md`). M2b-wc_s0 was not read.
* That read bears on the choice of next lever. An exploratory T0 oracle read finds T's LATERAL constraint error leaking into O's longitudinal profile: O's plan is further along at 2 s, and the early closing on the lead comes with it.
  * That is a measured, O-side defect for the committed lever (O trained against T's own code) to remove. The section above found no such defect.
  * It is also an accuracy target for the `pool8x20` input arm.
""")

# ---- code docstrings that said UNAVAILABLE without a pointer
rw("stack/scripts/v8f_mini_eval.py", [(
    "    CONSISTENCY LEAD rule here; the family's own lead-track input is not built (UNAVAILABLE, a work item).\"\"\"",
    "    CONSISTENCY LEAD rule here; the family's own lead-track input is not built (UNAVAILABLE, a work item) IN THIS EVAL.\n"
    "    Distance keeping proper (headway / time-gap / min-TTC / LEAD-VIOL: `taniteval.lead_metrics` on the B1 lead block) is read from this\n"
    "    eval's saved npz by `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_dk_read.py` (2026-10-22).\"\"\"")])
rw(f"{PK}/code/v8f_mini_full_report.py", [(
    "(n = {dk.get('n')}). A work item, not a pass.\",",
    "(n = {dk.get('n')}). A work item, not a pass. \"\n          \"Since 2026-10-22 it is read post hoc from these evals by `code/v8f_dk_read.py` (RESULT_V8F_DK.md).\",")])

# ---- the register: two new rows at the top of the v8F block
reg = show(TIP, "Project Steering/GOALS_AND_CLAIMS.md").decode("utf-8")
lines = reg.split("\n")
iw = [k for k, x in enumerate(lines) if x.startswith("| **H-V8F-WC-1** (2026-10-22) |")]
assert len(iw) == 1
assert not any(x.startswith("| **D-V8F-DK-1**") or x.startswith("| **D-V8F-T1KIN-1**") for x in lines)
dk_row = (
    "| **D-V8F-DK-1** (2026-10-22) | v8F-mini's LONGITUDINAL distance keeping (headway / time-gap / min-TTC / LEAD-VIOL): is it reached, and what "
    "does it read? | **READ. MEASURED, descriptive (no bar registered), T1-class open loop, paired episode-cluster bootstrap over the 77 episodes "
    "that carry a lead (743 of 2,399 windows); `…/2026-10-17-v8f-mini/raw/2026-10-22-dk/dk_read.json`, `RESULT_V8F_DK.md`.** The B1 lead-block "
    "instrument (R1, 2026-09-02) was built, tested and never called by v8F-mini; `code/v8f_dk_read.py` is the call site (25 tests, 14/14 "
    "mutants killed, re-run bit for bit from the tip's banked inputs). **6 s:** M2b keeps more headway than P10, +0.53 / +0.42 m, separated at "
    "both seeds and above 2F; P10 sits 0.195 s under the human's time-gap (separated). **2 s:** the learned arms close on the lead faster: "
    "min-TTC M2b − P10 −1.76 / −1.19 s, separated (2F met at seed 0 only); the share of LEAD windows under 5 s TTC is +4.4 / +2.5 pp over P10 "
    "(separated), and M2b − M1 is flat ⇒ the shared S→T→O stack, not the WM lever. **Why (EXPLORATORY, T0; each reading rule written before "
    "its run):** with the GT tactical code, the 2 s tail excess over P10 is no longer separated (`UPPER`); leave-one-out puts it on T's LATERAL "
    "constraint group `latcons` at all four checkpoints (+1.8 to +3.7 pp, separated). With T's own latcons, O's plan is 0.17–0.27 m further "
    "along at 2 s, with lateral position and heading flat ⇒ T's lateral error leaks into O's longitudinal profile. My stoplead hypothesis is "
    "refuted. 6 s LEAD-VIOL M2b − P10 is separated at seed 0 only, under a failed progress control. |")
t1_row = (
    "| **D-V8F-T1KIN-1** (2026-10-22) | Can v8F-mini be read at T1 on its own executed kinematic state? | **BUILT AND CPU-TESTED; NOT RUN ON "
    "REAL CAMERAS.** `stack/scripts/v8f_t1_eval.py` + `stack/tests/test_v8f_t1.py` (23 tests, re-run before landing; 6 source mutations RED); "
    "design `…/2026-10-17-v8f-mini/raw/2026-10-22-t1/DESIGN_T1_V8F.md`. The doctrine-correct name is **T1-kin**: `EVAL_DOCTRINE.md` makes T1 "
    "self-action OPEN loop, and the cameras stay logged. It replans every m × 0.2 s, executes the first m knots, rebuilds `p10f` / `base` / "
    "`pose_*` from the executed track and dead-reckons `nav`. `limit` is a declared oracle and `feats` carry the human's ego motion; `--limit "
    "off` and `--scene frozen` bracket them. Controls: a logged-motion executor reproduces every key the model reads bit for bit over 30 ticks; "
    "m = 30 equals the open-loop plan; NaN-poisoned logged ego after t0 changes nothing. **MEASURED by the T1 stream (camera-free, 141 clips / "
    "2,399 windows; `probes/floor_real_probe.json`, not re-run by the Master Mind):** the P10 floor reproduces its banked open-loop ADE6 2.487 m "
    "at m = 30, and RUNS AWAY in its own loop at short replan periods: 3.44 m (m = 10), 5.41 (m = 5), 23.3 (m = 2), 21.7 (m = 1, where 2,397 of "
    "2,399 windows hit the 60 m/s bound). State-rebuild noise is not the cause. ⇒ the informative T1-kin sweep is m ∈ {30, 10}, with 5 as a "
    "stress point. No model-vs-floor T1-kin number yet: it needs the pooled eval cache (box 2) and a PI go. |")
lines[iw[0]:iw[0]] = [dk_row, t1_row]
put("mods-rw", "Project Steering/GOALS_AND_CLAIMS.md", "\n".join(lines).encode("utf-8"))
open(os.path.join(H, "msg.txt"), "wb").write(lf(open(DL + "msg13.txt", "rb").read()))
print("hold ok")
