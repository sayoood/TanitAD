# D-VOCAB-L3 — the panel is blocked, the vocabulary is built, and the prize is still unbounded

**Agent:** TanitAD Architecture & Inference FlyWheel · **Date:** 2026-09-05
**Branch:** `agent/arch-inf-20260803`
**Executing:** `PREREG_D-VOCAB-L3_FLOOR_GAP` + **ERRATUM-1**, and M15 / M16 / M11
**Register rows:** `D-VOCAB-L3-POWER`, `D-VOCAB-L3-IMPL`, `D-REFAV1-ORACLE-DECONF`

---

## The one-paragraph answer

The pre-registered panel **cannot be run as written**, and ERRATUM-1 §E2's blocking check is what
found it: on a representative panel the L=1 turning-window deficit to `ha0_ext` is **−0.0628 m** —
the planner **beats** the floor there — so the committed *"shrink the deficit by ≥ 50 %"* criterion
is both **6.1× inside the planner's own inference-seed noise** and **undefined in sign**, on
**10** windows. Worse for the prereg, its motivating fact — *"3.6–5.9× below the trivial floors on
**turning** windows"* — is, on its own 28-window panel, **equally true of straights** (3.77–5.13×),
so it cannot be caused by a sustained-curvature vocabulary. M15's L=3 vocabulary is nonetheless
**implemented, parity-pinned, stamped, and validated against the approval it came from** — and it
has surfaced a gap the approval does not close: **the trained head cannot choose among three
magnitudes**, so an L=3 *arm* needs a chooser that does not exist, and the hint is therefore a
required argument whose tier must be declared. The de-confounded oracle arm — the gate on
everything — **exists, is tested and is running**; until it lands, `D-REFAV1-DRIVE-ORACLE`'s
**ADE 7.4708 still bounds nothing**.

---

## 1. ⛔ THE HEADLINE THE BRIEF ASKED FOR FIRST — the de-confounded oracle

**Status: IMPLEMENTED, TESTED, RUNNING. No number is claimed here, and none should be quoted
from the confounded arm in the meantime.**

`plan(goal_keeps_seed=True)` supplies the true-future goal **through** the seed path: the head
still runs and still builds its canonical seed — bit-identical to the shipped arm's, same
`lat_logit_bias`, same `goal_kappa_turn` — and **only** `goal_t` is replaced. `goal_source`
becomes `supplied+seed`, and `refav1_arm.py` verify-gates that **per window**, because a stale
`plan()` would ignore the keyword-only argument silently and bank the **confounded** arm under the
de-confounded name.

⭐ **The pairing is exact by construction, and that is worth stating:** `icem_plan` seeds a
**fresh** `torch.Generator` from `PlanConfig.seed` (`refa_v1_plan.py:208`), so every arm in a
window draws the *same* coloured noise. The extra tactical rollout the flag costs cannot perturb
it. ⇒ a per-window difference of exactly 0 between two arms is a genuine tie, not a coincidence.

**Panel in flight** (`tools/run_oracle.sh`, dev-box 4060):
`refav1_margin/p4` — 8 episodes, **40 windows at stride 16, 17 real turns** at |gt_κ| > 4e-2, a
4.5× higher turn density than the standard grid; arms `cl` · `cl_oraclegoal` (the confounded
control, on the same windows for the first time) · `cl_oracleseed`, plus `ha`/`ha0`/`ha0_ext`/`ol`;
`ccos`, weights `(0, 0, 64.297)`, plan-seed 0 then 1 (ERRATUM §E1's inference-seed replicate).

⚠️ **Two scope statements that must travel with the number when it lands.**
1. `cl_oraclegoal` and `cl_oracleseed` read the **true future** ⇒ **T0**, diagnostics and bounds,
   **never** driving numbers (EVAL_DOCTRINE).
2. The weight triple has **`W_KAPPA` exactly 0**, which `D-REFAV1-P4-COS-THRASH` MEASURED the same
   day to make the planner saturate `kappa_max` on **90 %** of straight windows. The **paired**
   contrast is unaffected — every arm shares the triple — but the **absolute level** must not be
   read as the road's difficulty.

⭐ **The de-confounding will bite:** on this panel the head decodes a curvature-carrying token on
**22/40 = 55 %** of windows (`TURN_L` 9, `TURN_R` 13) against **13.5 %** on the representative
grid, and only **10 %** of windows decode `CRUISE` longitudinally — so the restored seed is a real
candidate on essentially every window rather than a zero duplicate of the `cv` baseline.

⛔ **Recovery, so the GPU is never paid twice:** the dumps persist per episode at
`C:/Users/Admin/refav1_drive/oracle/dump_oracle_s0/`; `tools/oracle_deconf.py --dump <dir>` reads
them (partial or complete), and `refav1_arm.py --analyze-only <dump>` re-derives the four-family
record with **zero GPU**.

### 1a. ⛔ THE HARNESS, CHECKED BEFORE THE NUMBERS — the floors are bit-identical across runs

`tools/floors_cross_run.py`, `raw/floors_cross_run.json`. The brief's control, run against the
goal-margin stream's **independent** `cos` run over the **same p4 episodes at the same stride**:

| key | max abs diff | |
|---|---|---|
| `g` (GT waypoints), `v0` | **0.0** | the inputs the floors are built from |
| `ha`, `ha0`, **`ha0_ext`** (the INTEGRATOR form, M11), `ol` | **0.0** | ⭐ **BIT-IDENTICAL** |
| **`cl` — the control that MUST differ** | **7.2806 m** | ⭐ the two runs differ in the cost metric (`cos` vs `ccos`), so the comparison can detect a difference |

⇒ the floors are the **same object** in both runs, and the equality is **not vacuous**. *(On this
mount an all-equal report is otherwise indistinguishable from reading nothing — a `0` from a file
that could not be read looks exactly like a genuine `0`.)*

### 1b. ⚠️ PARTIAL READ — 3 of 8 episodes, n = 10 valid windows. DIRECTIONAL ONLY, NO CLAIM

The run is 3/8 done at hand-off. This is banked because the dumps must not be re-paid for and
because the saturation readout is already attributable; **it is not a result.** No interval, no
bootstrap, no four-family table — those come from `refav1_arm.py --analyze-only` on the finished
dump. 15 windows, **5 excluded** at v0 < 1 m/s, **4 turning** and **6 straight** at
|gt_κ| > 4e-2.

⛔ **Both controls PASS on all 15 windows:** `goal_source` is `tactical_imagined` 10/10 (`cl`),
`supplied` 10/10 (`cl_oraclegoal`), **`supplied+seed` 10/10** (`cl_oracleseed`); and the decoded
(lat, lon) seed token is **identical** between `cl` and `cl_oracleseed` on every window. ⇒ the goal
really is the only difference.

| arm | tier | ALL (n = 10) | TURNING (n = 4) | STRAIGHT (n = 6) |
|---|---|---|---|---|
| `cl` (shipped) | T1 | **1.5158** | **0.7868** | 2.0018 |
| `cl_oraclegoal` (CONFOUNDED — no seed) | T0 | 2.9696 | 1.9486 | 3.6503 |
| ⭐ `cl_oracleseed` (DE-CONFOUNDED) | T0 | **2.3902** | **1.9486** | 2.6845 |
| `ha` | T1 | 1.3556 | 1.9446 | 0.9629 |
| `ha0` | T1 | 1.2255 | 1.3630 | 1.1338 |
| **`ha0_ext`** (INTEGRATOR, M11) | T1 | 1.2483 | **1.8567** | 0.8427 |
| `ol` | ⛔ T0 | 1.0860 | 1.8975 | 0.5449 |

**On the headline question, so far and on n = 4 turning windows:** a perfect goal delivered
honestly reads **1.9486** against the `ha0_ext` floor's **1.8567** — ratio **1.049**, it does
**NOT** beat the floor — while the *shipped* arm reads **0.7868**, ratio **0.424**, and does.
Paired against `cl` the perfect goal is worse **everywhere**: **+0.874** overall, **+1.162** on
turning, **+0.683** on straight. ⛔ **n = 4. This is an anecdote, not an answer** — but it is the
same anecdote at 2 and at 3 episodes (ratio 1.046 → 1.049), which is at least not noise chasing.

⭐⭐ **THE ATTRIBUTABLE PART — ERRATUM §E3's SATURATION READOUT, and it survives de-confounding.**
Fraction of windows whose planned controls touch `kappa_max = 0.2`:

| arm | ALL | TURNING | STRAIGHT | mean max \|accel\| (ALL) |
|---|---|---|---|---|
| `cl` | **0.100** | 0.000 | 0.167 | **0.234** |
| `cl_oraclegoal` (confounded) | 0.500 | 0.250 | 0.667 | 2.364 |
| `cl_oracleseed` (de-confounded) | **0.400** | 0.250 | 0.500 | **2.342** |

⇒ **the saturation is caused by the GOAL FIELD, not by the missing seed.** Restoring the seed
moves it 0.500 → 0.400 and leaves the acceleration channel essentially untouched (2.364 → 2.342,
against the shipped arm's 0.234); on turning windows it does not move it **at all** (0.250 both). `D-REFAV1-DRIVE-ORACLE`'s reading — *"an optimiser climbing a
cosine toward a target no control sequence can realise"* — is therefore **not** an artefact of the
confound, which is the single most useful thing this partial says.

⛔⛔ **SELF-CORRECTION, SAME TURN — A MECHANISM I COMMITTED ONE COMMIT AGO IS WRONG.**
Commit `0ac99f4` (and this document's first version) explained the exact ties between
`cl_oracleseed` and `cl_oraclegoal` as *"precisely the `LANE_KEEP` decodes, where the restored seed
is the all-zero profile"*. **The window-level data refutes it:** all **4** turning windows decode
`TURN_*` and **0** decode `LANE_KEEP`; across all windows only **1 of the 10** tied windows is a
`LANE_KEEP` decode. The story was plausible, fitted the first episode's
5 windows, and was **not checked against the joint** before it was written.
⭐ **And the true reading is stronger than the false one:** on those turning windows the seed is a
genuine, non-zero, correctly-signed `TURN_*` candidate — and **the plan comes out bit-identical
without it**, i.e. the canonical turn seed **loses** under an oracle goal field. That is the
goal→plan pathway failing to consume a good goal, measured at the level of a single candidate.
*(Class: a root cause derived from a handful of runs, which `CLAUDE.md` says not to do — three of
its own readings were falsified the same way in one session.)*

## 2. ⛔ THE POWER CHECK — blocking, and it found more than low power

Full table and controls in **`POWER_CHECK.md`**. In brief, on the representative panel
(`dump_ccos_comp`, 282 windows / 141 episodes, `ccos`, ckpt 21,109) at the **MEASURED** crossover:

| | |
|---|---|
| turning windows | **10** of 266 valid (3.76 %) |
| deficit `cl − ha0_ext` on turning | **−0.0628 m** (ratio **0.934** — `cl` beats the floor) |
| 50 % target | **−0.0314 m** |
| inference-seed floor | **0.1907 m** |
| **runnable as pre-registered** | ⛔ **NO** — 6.1× inside the noise, wrong sign, n = 10 |

⭐ **And the motivating fact does not generalise.** The prereg's 3.6–5.9× reproduces on its own
28-window `abt` panel (3.15–5.74×) — but **the straights there read 3.77–5.13×**.
A gap equally present on straight road **cannot** be caused by a sustained-curvature vocabulary,
which is not in play on a straight. ⚠️ The panel-to-panel comparison is **confounded** with the
cost triple (`W_KAPPA` 32.149 vs 0.0) and is reported as such, not attributed.

## 3. ⭐ THE L=3 VOCABULARY — built, pinned, and validated against its own approval

Full detail in **`L3_IMPLEMENTATION.md`**. `GOAL_KAPPA_TURN_LEVELS = (0.01155, 0.01942, 0.05805)`,
`canonical_controls_levels`, `choose_kappa_level`, `plan(goal_kappa_levels=, goal_kappa_hint=)`,
`refav1_arm.py --goal-kappa-levels m15 --goal-kappa-hint gt`, and the **parity stamp**
(`res.goal_kappa_vocab`, `goal_action["kappa_turn_used"]`, `manifest["goal_vocab"]`) that M16 (1)
requires before any cross-vocabulary comparison is admissible.

**Validated, not assumed** (`tools/validate_l3_constant.py`): the constant reproduces the approved
`L=3` row (medAE-on-turns **0.00420751**, expressible **1.0000**) and the **shipped control
reproduces to every printed digit** (0.01551106 / 0.38664596) on the same dense panel; and the
**live code path** produces exactly the analytic profile the approval scored — duty **20/30**,
magnitude to 1e-6, effective curvature = `level × duty` to 1e-7. Known-value controls: `L=0` floor
= road RMS curvature to **1e-12**; continuous ceiling **exactly 0.0**.

⛔⛔ **The gap the approval does not close: there is no chooser.** The v7.0 head has ONE
`TURN_L`/`TURN_R` slot and `kappa_turn` never touches its logits — which is why `turns goaled
correctly` is **0.2811 for every `kappa_turn`**. So `goal_kappa_hint` is **required, not
defaulted**, and the refusal names the reason: a default would silently choose the arm's **tier**.

## 4. ⛔ THE PANEL — not run, and why that is the correct outcome

ERRATUM §E2 makes the power check **blocking**, and it blocked. Running the pre-registered A/B
would have produced a confident-looking null on 10 windows against a criterion that cannot be
satisfied, and *"a confident-looking null from an underpowered panel is worse than no panel,
because it will be quoted."*

⭐ **There is also no L=3 arm to run yet** — only an L=3 vocabulary — because the chooser does not
exist. Both blocks are structural, and neither is dissolved by more GPU.

**The redesign, in the order it should be funded:**

1. ⭐ **Finish the de-confounded oracle** (§1). It gates everything: if a perfect goal delivered
   honestly cannot beat `ha0_ext` on turning windows, **no goal-side work — vocabulary included —
   can make refav1 drive**, and that is the most valuable single sentence available today.
2. ⭐ **The L=3 arm with an ORACLE κ hint (T0).** Cheaper than the pre-registered A/B and strictly
   more informative: it realises M15's oracle bound **at plan level**, which nobody has measured.
   An oracle-chosen L=3 that does not improve the plan refutes the vocabulary as the mechanism
   **without anyone building a chooser first**.
3. **Curvature discipline before vocabulary.** `D-REFAV1-P4-COS-THRASH` shows a planner saturating
   κ = 0.2 on 90 % of *straight* road at `W_KAPPA = 0`. A goal vocabulary topping out at 0.08
   cannot be the binding constraint on a search that already exceeds it by 2.5×; and the
   representative panel's own gap is **on straights**. Both point the same way.
4. Only then a κ chooser (T1), whose own error composes with the vocabulary table.

## 5. What is claimed, and what is not

| | |
|---|---|
| ✅ **MEASURED** | the power check and the floor-gap anatomy (banked dumps, zero GPU, controls passed); the parity and mechanism of the L=3 implementation; the reproduction of the approved design table by the shipped constant **and** the shipped code path |
| ⛔ **NOT CLAIMED** | any ADE, any four-family number, any driving claim for L=3 — nothing has been run on a checkpoint; the de-confounded oracle number, which is still in flight |
| ⛔ **NOT RETRACTED** | `D-MM-VOCAB-1` (L=3's approval rests on the oracle design table, not on the floor gap); `D-MM-VOCAB-3` (the `abt` A/B is a paired contrast whose arms share the triple); `D-REFAV1-DRIVE-GATE2` |
| ⚠️ **WITHDRAWN** | the prereg's §2 *explanation* of the floor gap, and its §1 fact as a statement about **turning** |

**Does refav1 turn, and does it drive?** It turns — `D-REFAV1-P4-COS-THRASH` shows it turning on
90 % of windows, at the clip bound, half of them the wrong way. **It does not drive**, and nothing
in this package changes that. What this package changes is *where to look*: not at the lateral
goal vocabulary, which is now built and cannot be the binding constraint on a planner that already
commands 2.5× its top magnitude on straight road.

---

## Deliverable manifest

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-vocab-l3/`

| artifact | where | what it establishes |
|---|---|---|
| `POWER_CHECK.md` | repo | ⛔ the panel is blocked; the floor gap is not about turning |
| `L3_IMPLEMENTATION.md` | repo | the L=3 build, its parity stamp, its validation, and the missing chooser |
| `RESULT.md` | repo | this synthesis |
| `tools/power_check.py` | repo | ERRATUM §E2's blocking check, with the bit-identical-floors control |
| `tools/floor_gap_anatomy.py` | repo | PANEL × THRESHOLD, straights beside turns, each panel's own cost triple |
| `tools/validate_l3_constant.py` | repo | the constant **and** the code path against the banked approval |
| `tools/oracle_deconf.py` | repo | the de-confounded oracle reader (splits, paired deltas, §E3 saturation, 2 controls) |
| `tools/run_oracle.sh` | repo | the P1 driver (waits for the sibling GPU job by explicit PID) |
| `raw/power_check.json`, `raw/floor_gap_anatomy.json`, `raw/validate_l3_constant.json` | repo | the evidence |
| `stack/tanitad/refs/refa_v1.py` | repo | `goal_keeps_seed`, `GOAL_KAPPA_TURN_LEVELS`, `canonical_controls_levels`, `choose_kappa_level`, the stamps |
| `stack/tests/test_refa_v1_oracle_seed.py` | repo | **7/7** — the confound pinned, the fix pinned, the guard pinned |
| `stack/tests/test_refa_v1_kappa_levels.py` | repo | **9/9** — parity, the level surface, the mandatory chooser |
| `taniteval/tools/refav1_arm.py` | repo | `--with-oracle-seed-arm`, `--goal-kappa-levels`, `--goal-kappa-hint`, two verify-gates, `manifest["goal_vocab"]` |
| ⚠️ **the oracle dumps** | **dev box only** — `C:/Users/Admin/refav1_drive/oracle/` | recover with `tools/oracle_deconf.py` or `refav1_arm.py --analyze-only`; the GPU is already paid |

**Suite:** 341 passed, 1 skipped
(`-k "refa_v1 or cost_ccos or cost_chord or ego_plan or seed_goal or steer_curv"`).

## ESCALATED

1. ⛔ **`PREREG_D-VOCAB-L3_FLOOR_GAP` needs a second ERRATUM or a replacement**, because its §1
   fact and its §2 mechanism do not survive. Its §3 outcomes cannot be evaluated as written.
2. ⭐ **The de-confounded oracle is the gate on the whole goal-side programme** and is the one
   number to read next.
3. ⛔ **M15's approval needs a companion decision on the CHOOSER**, or L=3 stays a vocabulary with
   no arm. The cheapest first step is the **T0 oracle-hint arm**, which needs no new training.
4. ⚠️ **`run_ab.sh` / `run_ab_targeted.sh` pass `W_KAPPA = 0`** and every arm banked through them
   sits on a planner that curves at the clip bound on straight road — flagged by
   `D-REFAV1-P4-COS-THRASH`, and it is the most likely cause of the `abt` panel's uniform 3–6×
   deficit that this prereg was built to explain.
