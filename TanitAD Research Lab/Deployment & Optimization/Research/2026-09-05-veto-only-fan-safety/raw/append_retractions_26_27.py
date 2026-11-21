#!/usr/bin/env python3
"""Append retractions #26 and #27 to `Project Steering/RETRACTION_LOG.md` (append-only).

Refuses if either heading is already present, so a re-run cannot duplicate them.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "RETRACTION_LOG.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
NE, ST, W, ED, X = chr(9940), chr(11088), chr(9888), chr(8212), chr(215)
ARR = chr(8594)
TH = chr(952)
D_ = chr(916)

TEXT = """

# 2026-09-05 (#26) {ED} *"the RL stage did not even improve its own objective out of sample (`R1` -0.0058)"* {ED} WITHDRAWN: `R1` was a BATCH {X} BATCH OUTER PRODUCT

**What was claimed.** `.../2026-09-05-refc-rl-readiness/RESULT.md` {SEC}11.2, in the supporting-readout
line: *"**R1 (the composed reward itself, on held-out EVAL windows) -0.0058** {ED} the stage did not
even improve its own objective out of sample."* The same defect touches `R2` (fan collision rate)
wherever it is quoted.

**What is true.** MEASURED 2026-09-05, reproduced in three lines and then fixed.
`rl_refcv3_min.reward_ctx()` shapes every scene fact for the TRAINING trajectory
`[B, N, G, S, 2]` {ED} three leading axes. `readout()` hands it a `[B, N, S, 2]` **fan** {ED} two.
Right-alignment then broadcasts

    lead_path [B, 1, 1, S, 2]   against   traj [B, N, S, 2]    {ARR}   [B, B, N, S, 2]
    v0        [B, 1, 1]         against   along [B, N]         {ARR}   [B, B, N]

so `_collision`, `_headway` and `_progress` all return `[B, B, N]`: **every window's fan scored
against every window's lead**. The driver then indexes `r1[j]` {ED} window j's row of a B {X} B slab
{ED} and means over it. The readout batch is **4**, so the quoted `R1`/`R2` are a batch-mixed
statistic, not "the composed reward on this window's fan".

{NE} **SCOPED, NOT BLANKET-VOIDED** {ED} three positive checks, each of which had to be run rather
than assumed:
1. **TRAINING is unaffected.** `make_sample_fn` really does hand `traj2 [B, N, G, 5, 2]`, which is
   exactly the rank `reward_ctx` builds for. Verified by shape, not by reading the code twice.
2. **The PRIMARY fan-safety endpoint is unaffected.** `fan_safety.score_paths` is called with
   `lead5 = b["lead_track"].reshape(-1, 1, 5, 2)` and `v0 = b["v0"]` ([B]), and unsqueezes `v0`
   itself {ED} correct rank on both operands.
3. **The affected set is exactly `R1` and `R2`** in every `readout_*.json` of that panel, plus
   their paired deltas.

{ST} **The corroboration is the part worth keeping.** After the fix, the very next arm's BEFORE
readout on the identical 120 windows reads `R1` **0.5876 {ARR} 0.6071** and `R2` **0.03612 {ARR}
0.02923**, while `R3`, `R_FAN`, `R_REACH` and `R_ORACLE` reproduce the banked base **to every printed
digit** (`R3` 0.47414546824681264 both times). Exactly the two metrics the defect could touch moved,
and nothing else did {ED} which is what makes the scoping a measurement rather than an argument.

**Root-cause CLASS {ED} the `df` / Thor `free` / cgroup `usage_in_bytes` / `step_s` / cylindrical-FOV
/ units family, with the scope being TENSOR RANK.** A correct quantity computed against the wrong
axis returns a **number**, not an error; NumPy/torch broadcasting is designed to succeed. Nothing in
the pipeline objected: the array had a plausible dtype, a plausible magnitude, and one more axis
than anybody looked at. {W} And the shape was *documented correctly in a comment* on the very line
that was wrong (`r1 = spec(fan2, ctx)  # [B, N]`) {ED} a comment is not an assertion.

{ARR} **Durable fixes, both shipped:**
1. `reward_ctx(..., cand_dims=N)` makes the candidate rank an ARGUMENT rather than a silent
   assumption, with the failure mode written into its docstring.
2. `readout()` now **asserts** `r1.shape == fan.shape[:2]` and raises. A rank mismatch that returns
   a number must be converted into one that raises, or the next reader inherits the same trap.
{ST} **The generalisation worth carrying:** wherever a helper builds broadcast-shaped context for
one consumer and a second consumer appears, the rank is a CONTRACT {ED} assert it at the boundary.

{ARR} **Pinned:** `stack/scripts/rl_refcv3_min.py` (`reward_ctx`, `readout`);
`.../2026-09-05-veto-only-fan-safety/raw/patch_p1a_readout_ctx_rank.py`, `SPEC.md` {SEC}3,
`RESULT.md` {SEC}0; register row `D-RL-READOUT-CTXRANK-1`.


# 2026-09-05 (#27) {ED} *"a zero-weight spec with the veto OFF is an actual null"* (my own SPEC {SEC}7, guard G1) {ED} FALSE for a SECOND, unrelated reason: AdamW

**What was claimed.** `.../2026-09-05-veto-only-fan-safety/SPEC.md` {SEC}7, guard **G1**:
*"`ctrl_null` {ED} `veto_rate_mean` must be **exactly 0.0**. If it is not, P2's fix did not take and
nothing below is quotable."* The implied premise {ED} and it is the premise the whole P2 work item
was built on {ED} is that removing the veto's information makes the arm a NULL.

**What is true.** The `veto_rate_mean` half is right and it PASSED: `ctrl_null` reads **0.0000
exactly** where this morning's `ctrl_const` read **0.0897**. The premise behind it is wrong.
MEASURED on `ctrl_null` (zero reward, veto off, `w_anchor` 1.0, lr 1e-5, 200 steps):

| | |
|---|---|
| advantage | identically zero {ARR} policy-gradient contribution **exactly 0** |
| `final_loss` | **0.04459** (the anchor trust region, alone) |
| trainable tensors that moved | **71 of 71** (478 compared) |
| mean \\|{DTH}{TH}\\| over decoder tensors | **3.1e-05**, max **5.7e-04**, against a typical \\|{TH}\\| of **1.8e-02** |
| fan-safety metrics separated vs base | **35 of 57** |

**The mechanism.** The only live loss is the anchor trust region, whose step-0 divergence between the
model and its own frozen deepcopy is ~**9e-11 m{SUP2}** (INHERITED, `.../refc-rl-readiness` {SEC}3.2).
A gradient that small should move nothing {ED} **except AdamW normalises by the gradient's own second
moment**, so `m / (sqrt(v) + eps)` {AP} `sign(g)` and the update is of order `lr` however small `g`
is. Predicted movement ~1e-5 per step; MEASURED **3.1e-05** after 200 steps. A 1e-10-scale gradient
producing a 1e-10-scale movement would have read ~1e-10; it read **five orders larger**.

{NE}{NE} **THE CONSEQUENCE IS BIGGER THAN THE RETRACTION: on this rig a ZERO-INFORMATION arm
separates on 35 of 57 fan-safety metrics {ED} MORE than the 14 of 57 that fired the previous panel's
VOID gate.** "Separated vs base" is therefore close to uninformative here on its own, and `ctrl0`
(lr = 0) is the ONLY arm that cannot move. Any arm with a live optimizer and `w_anchor > 0` drifts
whatever its reward says.

**Root-cause CLASS {ED} the `H-ESTIM-SEED-1` family: an estimator answering a narrower question than
the claim hung on it** {ED} with a new source of nuisance movement. `H-ESTIM-SEED-1` identified the
SAMPLER (episodes resampled, models fixed). This adds the **OPTIMIZER**: a scale-invariant update
rule converts numerical noise into full-size steps, so "the information was removed" does not imply
"the weights held still". {W} Note the shape of my error: I fixed the mechanism I had just found
(key-membership) and then asserted the null WITHOUT measuring it, in the same document that exists
because someone else asserted a null without measuring it.

{ARR} **Durable fixes:**
1. **Two floors, always.** `raw/analyze_veto.py` requires every lever to clear BOTH the
   **seed-replicate** floor (`H-ESTIM-SEED-1`) and the **zero-information** floor (`ctrl_null`), and
   reports `quotable_as_lever` (the committed rule, unchanged) beside `quotable_strict` (+ this
   floor). A post-hoc STRENGTHENING; the committed text is still reported as written.
2. **The floor must be DOSE-MATCHED.** Nuisance drift accumulates with steps, so a 200-step null
   cannot floor a 2,000-step lever; `analyze_veto.py --null-dir` takes the matched-dose arm and
   records `dose_matched` in its output.
3. **A null is a MEASUREMENT, never a property of a configuration.** Any arm asserted to be a null
   must produce its own before/after readout and be reported with it.

{ARR} **Pinned:** `.../2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}3, `raw/analyze_veto.py`,
`raw/run/s0/ctrl_null/arm_summary.json`, `raw/run/n2k/ctrl_null/arm_summary.json`; register row
`D-RL-VETO-EXPLICIT-1` (the fix) and `H-VETO-FAN-1` (the two floors in force).
""".replace("{NE}", NE).replace("{ST}", ST).replace("{W}", W).replace("{ED}", ED) \
   .replace("{X}", X).replace("{ARR}", ARR).replace("{TH}", TH).replace("{DTH}", D_) \
   .replace("{SEC}", chr(167)).replace("{SUP2}", chr(178)).replace("{AP}", chr(8776))

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
if "(#26)" in s or "(#27)" in s:
    raise SystemExit("[retraction] #26/#27 already present - refusing to duplicate")
s = s.rstrip(LF) + LF + TEXT
open(PATH, "wb").write((s.replace(LF, CRLF) if is_crlf else s).encode("utf-8"))
print("[retraction] #26 and #27 appended (%s)" % ("CRLF" if is_crlf else "LF"))
