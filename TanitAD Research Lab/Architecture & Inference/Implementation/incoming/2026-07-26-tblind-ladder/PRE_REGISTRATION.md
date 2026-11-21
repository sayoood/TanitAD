# T_BLIND LADDER — pre-registration

**Written 2026-07-26 (Europe/Berlin; pods log UTC), BEFORE any number in this folder was computed.**
Every file in `artifacts/` and every table in `TBLIND_LADDER.md` carries a later mtime.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` (one path, unreproduced) · `CONFIRMED` (independent reproduction) ·
`DECISION-GRADE` (CONFIRMED + pre-registered + estimator named + falsifier stated).

---

## 0. The question, and the one thing that is genuinely untested

`…/2026-07-26-blind-imagination/` MEASURED (`artifacts/t_blind.json`, re-read here from the raw
JSON, not from its prose):

| regime | `T_blind` | CI95 | beats CV over |
|---|---|---|---|
| (i) true future actions — **privileged** | 6.5 s (65 steps) | [6.5, 8.9] | 0.4 – 7.4 s |
| **(ii) the model's own actions — deployable** | **0.8 s (8 steps)** | **[0.8, 1.0]** | **never (0/185)** |
| (ii-0) held last action — deployable, no policy | 3.2 s (32 steps) | [3.2, 5.0] | 0.4 – 3.8 s |
| (ii-c) convention control (true motion, my inverse) | 5.4 s (54 steps) | [5.4, 7.3] | — |
| A2 — readout `step["str"]`, **true** actions | 18.5 s ⚠️ saturated | [18.5, 18.5] | 0.6 – 18.5 s |

The readout swap (`step["op"]`, calibrated over `op_fwd_k = 4` steps → `step["tac"]` / `step["str"]`,
calibrated over 16 / 20) was measured **only in the privileged regime**. ⛔ **Whether it extends the
DEPLOYABLE `T_blind` was never computed** — and not by oversight in the reporting, but because the
regime was *structurally skipped*: `bi_analyze.REGIMES["A2_readout_str_own_actions"]` declares
`{"a": "a_imagination__own__roSTR", "b": None}` and `analyse_sweep` `continue`s on a missing `b`.
`t_blind.json` therefore contains **five** regimes and the deployable-STR one is not among them.
`MEASURED` — verified by loading both the script and the committed JSON.

---

## 1. ⛔ What the zero-GPU dump can and CANNOT do — declared before it is used

`perwindow/bi_perwindow_compact.pt` (12.1 MB, in the repo) holds **dense per-window `de` [599 × 185]
for 20 arms** plus reporting-grid `de`/`along`/`cross` for 49. Loaded and enumerated before this
document was written. What matters for Rung 0:

| arm needed | present? |
|---|---|
| `a_imagination__own__roSTR` — deployable imagination, 20-step readout | ✅ **dense** |
| `a_imagination__hold__roSTR` — held-action imagination, 20-step readout | ✅ **dense** |
| `a_imagination__true__roSTR` / `__roTAC` | ✅ **dense** |
| `b_frozenlast__true__roSTR` — matched control, privileged regime | ✅ **dense** |
| `d_constant_velocity` — the floor (**no readout at all**) | ✅ **dense** |
| ⛔ **`b_frozenlast__own__roSTR`** — the *matched* control for the deployable regime | ❌ **DOES NOT EXIST** (neither dense nor grid) |
| ⛔ **`a_imagination__own__roTAC`** — the k=16 readout under own actions | ❌ **DOES NOT EXIST** |
| ⛔ **`b_frozenlast__hold__roSTR`** | ❌ **DOES NOT EXIST** |

⇒ **The brief's "tactical (k=16) **and** strategic (k=20) readouts" cannot both be honoured at zero
GPU.** `roTAC` was only ever rolled out under **true** actions. This is stated here, in advance,
rather than quietly narrowed later. Rung 0 therefore runs on **`str` (k=20)**, and `tac` (k=16) under
own actions moves into Rung 1's pod job.

### 1.1 The comparator problem, and how it is handled — declared in advance

`T_blind` needs a **frozen-last-frame** comparator, and the readout-matched one does not exist for
own actions. Two contrasts are pre-registered, and **the unconfounded one is the tiebreak**:

* **P1 (primary, ⚠️ UNMATCHED COMPARATOR):** `a_imagination__own__roSTR` vs **`b_frozenlast__own`**
  (which carries the `op` readout). Declared **a LOWER BOUND**, on a sign argument that is itself
  checked rather than assumed: in the privileged regime the `str` readout makes the **frozen-last**
  arm *worse* at every horizon (§2.5 of the blind-imagination report, `−0.287 … −2.799` m). If that
  sign transfers, the true matched `de_b` is larger, Δ = `de_b − de_a` is larger, and the matched
  `T_blind` is **≥** what P1 reports. ⚠️ The transfer of that sign to the own-action regime is a
  `HYPOTHESIS`; its magnitude is bounded by **P1-S**, below.
* **P2 (secondary, ✅ EXACTLY MATCHED, no confound possible):** `a_imagination__own__roSTR` vs
  **`d_constant_velocity`**. The CV floor is pure kinematics and **has no readout**, so this contrast
  cannot be confounded by the swap. Current value: **never, 0/185 steps.**
* **P1-S (sensitivity, MEASURED not argued):** in the **privileged** regime, where both comparators
  exist, compute `T_blind` for `a_imagination__true__roSTR` against **both** `b_frozenlast__true__roSTR`
  (matched, committed 18.5 s) and `b_frozenlast__true` (unmatched, `op`). The difference between those
  two is the **measured** size of the comparator mismatch, in the one regime where it is observable.

---

## 2. ⭐ The pre-registered adjudication — buckets fixed before the numbers exist

Primary statistic: **`T_blind`** on `de_N`, rule **A4** (`largest N with paired-bootstrap lower bound
> 0 contiguously from N = 2`), estimator **paired episode-cluster bootstrap**, `taniteval/ci.py`,
**B = 2000, seed 0**, resampling unit = **episode cluster** (596 of them, 599 windows), identical
windows for every arm. `overlapping_holdout_se` appears nowhere.

Baseline to beat: the deployable **8 steps (0.8 s)**.

| verdict | rule, fixed now |
|---|---|
| ⭐ **CONFIRM** | `T_blind(P1) ≥ 16 steps (1.6 s)` — at least a doubling ⇒ **the horizon mismatch is a primary cause of the deployable failure** and the Rung-2 training fix (`--op-fwd-k 20`) is justified. |
| **PARTIAL** | `10 ≤ T_blind(P1) ≤ 15 steps` (+25 % … < 2×) ⇒ the mismatch **contributes**; policy divergence still dominates. |
| ⛔ **REFUTE** | `T_blind(P1) ≤ 9 steps` (≤ +1 step — not a material extension) ⇒ **the horizon mismatch is NOT the deployable bottleneck.** Lever 2 (policy divergence) becomes primary and Rung 2 is re-aimed. **This will be stated plainly. The horizon hypothesis will not be rescued.** |

**Corroboration, reported whichever way P1 lands:** P2. If P1 says CONFIRM and P2 still says *never
beats CV*, the headline is **"extends the horizon over which imagination beats a frozen percept, and
still never beats the trivial floor"** — both halves, in the same sentence.

**Tie/conflict rule, fixed now:** if P1 and P2 disagree in direction, **neither is promoted**; the
disagreement is the result and the tier is capped at `PROVISIONAL`.

---

## 3. ⚠️ THE C13 CHECK — what would make my rule return a FAILING value, and proof that it can

The blind-imagination agent's own pre-registration carried a **C13** defect: a `T_blind` contiguity
rule anchored at N = 1 that **could not fire**, because all arms are bit-identical at step 1 by
construction. Amendment **A4** moved the anchor to N = 2. Before adjudicating anything I re-derive
that rule rather than trusting it, and I state its failing value:

* **`_t_contiguous` returns `start_idx = 1` (i.e. 1 step = 0.1 s) — never 0 — when the FIRST evaluable
  horizon (step 2) already fails.** That is the failing value.
* **The concrete result that would produce it:** the `str` readout is MEASURED to be *worse* than
  `op` at short horizon under own actions (`−0.045 m at 0.5 s`, §2.5). If that early penalty puts
  `a_imagination__own__roSTR` behind `b_frozenlast__own` at step 2, the rule returns **0.1 s** and the
  verdict is **REFUTE**. This is a live outcome, not a hypothetical.
* 🔴 **And a residual C13 in the committed artifact, flagged in advance:** `t_blind.json` prints
  `frac_draws_T_blind_is_zero = 0.000` for **all five** regimes. Under A4 that is **structurally
  guaranteed** — the rule's minimum return is 1, so the counter can never be non-zero, and it is a
  vacuous diagnostic. The informative statistic is `frac_draws_T_blind == 1 step` (the floor), and it
  is emitted here instead.

**Both-direction validation of my re-implementation (M3), run before any new number is read:**

| direction | test | required result |
|---|---|---|
| **fidelity** | re-derive the 4 pre-registered `T_blind` values + the A2 arm from the compact dump | **exactly** 65 / 8 / 32 / 54 / 185 steps and CI 6.5–8.9 / 0.8–1.0 / 3.2–5.0 / 5.4–7.3 / 18.5–18.5 s |
| **fidelity** | re-derive `ade_0_2s` for `a_imagination__true`, `__roTAC`, `__roSTR`, `c2_observedpair__true` | **0.3839 / 0.1865 / 0.1950 / 3.6093** |
| **deliberate failure** | feed `de_a ≡ de_b` (identical tensors ⇒ Δ ≡ 0) | must return **1 step**, not a large horizon |
| **deliberate failure** | feed the arms **swapped** (a known-worse arm as "imagination") | must return **1 step** |
| **deliberate failure** | feed a curve that wins only from step 40 (isolated late win) | must return **1 step** — contiguity must not be manufacturable |

⛔ **If the fidelity gate does not reproduce, this pre-registration is void and Rung 0 is reported as
BLOCKED, not adjusted.**

---

## 4. Rung 1 — the v4 canary check, pre-registered

v4's `wm_canary_ade_2s` = **1.1409**, Bar B ≤ **0.55** ⇒ must fall **2.069×** (`INHERITED` from
`BOOST_PROGRAM.md:246` + the 2026-07-26 gate; re-verified against the raw gate JSON before use).

| verdict | rule, fixed now |
|---|---|
| **MOVES** | the swap lowers v4's `wm_canary_ade_2s` by **≥ 25 %**, paired-separated |
| **DOES NOT MOVE** | < 25 %, or not separated |
| **CLEARS BAR B** | ⛔ **only** if the swapped value is **≤ 0.55** — a fall of any other size is *not* a clearance and will not be reported as one |

⚠️ Pre-committed: a canary improvement that does not cross 0.55 is reported as **"moves the canary,
does not clear Bar B"**. No ratio measured on v1 is a prediction for v4.

---

## 5. Rung 2 — the attribution, pre-registered

A **2 × 3 factorial on arm (a)**, all six cells on the identical 599 windows, comparator-free
(`de_N` and `ade_0_2s` need no control arm), so no readout confound can enter:

| | `op` readout (k=4) | `str` readout (k=20) |
|---|---|---|
| own actions (deployable) | committed | **NEW** |
| held last action | committed | **NEW** |
| true actions (privileged) | committed | committed |

Reported as **main effects + interaction** on `de@2s` and on `ade_0_2s`, each with its paired
episode-cluster bootstrap. ⚠️ Pre-committed: `T_blind` is **not** the attribution metric — it is a
non-linear functional of a *contrast*, and four of the six cells lack a matched comparator. `de_N` is.

**Falsifier for the whole ladder (a stream that cannot say what would end it is an activity):** if the
readout main effect on `de@2s` under own actions is **not separated from zero**, the horizon lever is
dead for the deployable regime and Rung 2's L1 (`--op-fwd-k 20`) is **withdrawn from the ladder**, not
demoted.

---

## 6. What is NOT done here

No training is launched. Rung 2 is a **ladder** — cheapest discriminating experiment per lever, with
costs and pre-registered bars — not an execution. Any intervention it ranks is a separate,
separately-approved run.
