# T_BLIND RUNG 1 — the planner-action sweep: pre-registration

**Written 2026-07-26 (Europe/Berlin; pods log UTC), BEFORE any number in this folder was computed.**
Every file in `artifacts/` and every table in `TBLIND_RUNG1.md` carries a later mtime.
🔒 PhysicalAI-AV is gated-confidential: no clip UUID and no raw content appears in this folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED`
(another agent/doc, **not** re-verified) · `ESTIMATED` · `HYPOTHESIS`.
**Tiers:** `PROVISIONAL` (one path, unreproduced) · `CONFIRMED` (independent reproduction) ·
`DECISION-GRADE` (CONFIRMED + pre-registered + estimator named + falsifier stated **and shown to be
reachable**).

**Estimator, everywhere:** **paired episode-cluster bootstrap** — `taniteval/ci.py`, **B = 2000,
seed 0**, resampling unit = **episode cluster**, identical windows for every arm.
`overlapping_holdout_se` appears nowhere in this folder.

---

## 0. The two measured endpoints, and the budget between them

Re-read from the raw JSON, not from prose (`…/2026-07-26-tblind-ladder/artifacts/
rung0c_matched_tblind.json`), all with **matched comparators** — both arms decoded with the same
readout, differing in exactly one tensor — on **599 windows / 596 episode clusters**:

| regime, calibrated (`str`, k=20) readout | `T_blind` | CI95 | paired Δ at 2 s |
|---|---:|---|---|
| **own kinematic actions — the deployable policy** | **25 steps (2.5 s)** | [2.5, 3.9] | +0.4130 [+0.2651, +0.5673] |
| **hold last action — no policy at all** | **115 steps (11.5 s)** | [11.5, 17.4] | +1.3785 [+1.2503, +1.5122] |

⇒ **The gap is 90 steps = 9.0 s, and it is the registered ceiling for this rung.** Merely *removing*
the model from the action loop is worth **4.6×** more blind horizon than letting it choose.
`INHERITED-MEASURED`, re-verified here against the raw JSON before use.

Two further facts this rung is aimed at:

* **89.8 %** of the own-action penalty is the model rather than the measurement convention
  (`gt_kinematic` control, `INHERITED` from `…/2026-07-26-blind-imagination/`). Re-measured here at
  the **calibrated** readout, because §2.3 of the ladder showed one headline of that same run to be
  decoder-conditional.
* The v5 stream MEASURED that v4's imagination fan spans **108.7 m** of 2 s along-track displacement
  with candidates up to **181 km/h**, and that *"the world model does not veto an implausible plan,
  it obediently simulates it"* (`INHERITED`, `V5_IMAGINATION_SELECTION.md` §2.2 /
  `raw/v5_posthoc.json:per_window_span_mean_m = 108.736`). ⚠️ **Declared in advance:** in
  `taniteval/blindimag.py` the own-action inverse is **already clamped** —
  `steer ∈ [−0.05, +0.05] rad`, `accel ∈ [−3, +3] m/s²` (`STEER_CLAMP` / `ACCEL_CLAMP`,
  `kinematic_action_from_dpose`). So the blow-up hypothesis testable here is **saturation against an
  existing clamp**, not unbounded actions, and it is worded that way throughout.

---

## 1. The question, and what "no retraining" means

**What about the model's own action sequence destroys the rollout, and how much of the 9.0 s is
recoverable with no retraining?** "No retraining" = the checkpoint's weights are untouched; the only
change is a **filter on the action tensor fed back into the predictor**. Every intervention below is
a function of the action the deployed inverse would already have produced.

⚠️ **Declared in advance: what this rung CANNOT establish.** `own_kinematic` is one action policy —
the kinematic inverse of the model's own decoded motion. It is **not** v1's tactical planner
(`closedloop.wp_to_control`). The ladder's R1 row names the planner variant; this rung sweeps the
**filter** axis, which is strictly cheaper and interpolates the two *measured* endpoints. A planner
arm is a different experiment and is not silently folded in here.

---

## 2. ⛔ The window-set identity gate — declared before anything is built

The new arms come from a **second encode pass on pod2** and are only poolable with the committed
dense dumps if the window set is bit-identical. Four arms that **already exist** are re-rolled as
anchors, exactly as Rung 0b did:

| anchor | committed in |
|---|---|
| `a_imagination__own__roSTR` | `…/2026-07-26-blind-imagination/perwindow/bi_perwindow_compact.pt` |
| `a_imagination__hold__roSTR` | same |
| `b_frozenlast__own__roSTR` | `…/2026-07-26-tblind-ladder/perwindow/perwindow_matched_K185.pt` |
| `b_frozenlast__hold__roSTR` | same |

**Gate = PASS** requires all of: `n_windows == 599`; `eid` ordering identical; `t0` ordering
identical; and **every anchor's dense `de` within `1e-4` m** (Rung 0b measured `3.05e-05` m of
float-kernel noise between two encode passes; the same tolerance is used, not a looser one).
⛔ **If the gate fails the run is reported BLOCKED, not pooled.** This is the check whose absence
drove Rung 0's unmatched contrast from 185 steps to 1.

---

## 3. ⛔ The plumbing self-test — in BOTH directions, run before any result is read

A filter knob that is silently a no-op would produce a flat, confident, wrong curve. The two
endpoints of the blend axis are **structurally identical to arms that already exist**, which makes
this checkable exactly rather than by eyeball:

| direction | test | required |
|---|---|---|
| **fidelity (lower endpoint)** | `own_kinematic\|blend=0.0` | **max abs Δ = 0.0** against the re-rolled `a_imagination__own__roSTR` in the same process (bit-identical, not within tolerance) |
| **fidelity (upper endpoint)** | `own_kinematic\|blend=1.0` | **max abs Δ = 0.0** against the re-rolled `a_imagination__hold__roSTR` — the blend at α = 1 reduces algebraically to the zero-order hold |
| **anti-no-op** | every ELIGIBLE intervention arm | must differ from `own` by **more than the anchor tolerance** — a knob that changes nothing is a knob that was not read |
| **deliberate failure** | `own_kinematic\|every=1` | must be **bit-identical** to `own` (recompute every step = no filter), so the `every` knob is shown to have a null setting AND a non-null one |

⛔ **If the self-test fails, this pre-registration is void and Rung 1 is reported as BLOCKED, not
adjusted.**

---

## 4. The intervention families — fixed now, no arm added after a result is seen

All arms use the **calibrated `str` (k=20) readout**, i.e. Rung 0's adopted decoder, and every one is
rolled in **both** `state_source = imagination` (a) and `state_source = frozen_last` (b) unless marked
*a-only*, so that **every `T_blind` has a comparator matched in the action filter as well as the
readout**. K = 185, stride 8, 599 windows, v1 = `flagship4b-speedjerk-30k` @ 29999.

| # | family | configs | eligible? |
|---|---|---|---|
| **C** | **blend toward hold-last** — `a_fed = (1−α)·a_own + α·a_hold0` on (steer, accel) | α ∈ {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875} | ✅ |
| **D** | **channel decomposition** — one channel from the model, the other held | `chan=steer`, `chan=accel` | ⛔ diagnostic |
| **E** | **clipping** to a tighter physically-plausible band | `steer_clip` ∈ {0.02, 0.005, 0.0} rad; `accel_clip` ∈ {1.0, 0.3, 0.0} m/s² | ✅ except the `0.0` rows (⛔ diagnostic) |
| **F** | **temporal smoothing** — first-order low-pass `a_fed_j = β·a_fed_{j−1} + (1−β)·a_own_j` | β ∈ {0.5, 0.8, 0.95} | ✅ |
| **G** | **reduced action-update frequency** — recompute every *m* steps, zero-order hold between | m ∈ {2, 5, 20} | ✅ |
| **H** | **onset / switch time** (*a-only*) — own for the first *m* steps then hold (`own_before`), and hold for the first *m* then own (`own_after`) | m ∈ {5, 10, 20, 40} | ⛔ diagnostic |
| **I** | **convention & speed channel** | `gt_kinematic` at `str`; `own` at `str` with `update_speed_channel=True` | `gt_kinematic` ⛔ privileged; `vupd` ✅ |

**ELIGIBLE = may be adopted and may set the verdict.** The eligibility rule, fixed now: *both fed
channels must remain a non-degenerate function of the model's own decoded motion.* That excludes
α = 1 (the ceiling — no policy at all), `chan=` (one channel amputated), `steer_clip = 0` /
`accel_clip = 0` (one channel amputated), the switch arms (an oracle over the horizon) and
`gt_kinematic` (privileged: it reads the true motion). Those are reported, and can never be quoted as
"a no-retraining fix".

---

## 5. ⭐ The pre-registered adjudication — buckets fixed before the numbers exist

**Primary statistic:** deployable **`T_blind`** on `de_N` for the best ELIGIBLE intervention against
its **action-matched, readout-matched** frozen-last comparator; rule **A4** (largest N with
paired-bootstrap lower bound > 0 contiguously from N = 2), estimator as §0.
**Baseline to beat: 25 steps (2.5 s).** **Ceiling: 115 steps (11.5 s).**

| verdict | rule, fixed now |
|---|---|
| ⭐ **CONFIRM** | best eligible `T_blind` **≥ 50 steps (5.0 s)** — at least a doubling of the deployable baseline — **and** the paired bootstrap of (`T_blind_int − T_blind_own`) has a 2.5th percentile **> 0** ⇒ a no-retraining intervention materially extends deployable blind driving. Report it with its interval and its position on the 2.5 → 11.5 s scale, and adopt it. |
| **PARTIAL** | best eligible `T_blind` ∈ **[31, 49] steps** (> +25 %, < 2×) with a separated improvement, **or** ≥ 50 steps with an improvement that is **not** separated ⇒ it lifts it, below the +9.0 s ceiling; state exactly where it lands. |
| ⛔ **REFUTE** | best eligible `T_blind` **≤ 30 steps** (≤ +20 % — not a material extension) ⇒ **nothing without retraining materially helps; the action loop needs R3 scheduled sampling, and this will be said plainly.** The filter hypothesis will not be rescued by re-cutting the buckets. |

**Multiplicity, handled rather than ignored.** The best-of-N is a maximum over ~18 eligible arms and
is biased upward. Three pre-registered mitigations: (i) the **primary evidence is the dose–response
shape of the blend curve**, which a selection effect cannot manufacture — a monotone rise from 25 to
115 steps as α → 1 is a curve, not a lucky arm; (ii) the adopted point additionally reports the
**Bonferroni-adjusted** requirement `frac_draws(T_int > T_own) ≥ 1 − 0.05/n_eligible` beside the
nominal 0.975; (iii) `n_eligible` is fixed **now** by §4 and no arm is added later.

### 5.1 ⚠️ The capability cap — reported in the same breath, per the ladder's standing rule

`T_blind` is an extension **against a frozen percept only**. Two statistics need **no comparator arm**
and therefore cannot be moved by any of this:

* **beats-CV**: the horizon interval over which the arm is separated-better than
  `d_constant_velocity`. Currently **0 / 185 steps** for the deployable arm.
* **`T_useful@1 m`**: the last horizon with mean `de` below 1 m. Currently **1.4 s**.

**Pre-committed:** if the best intervention leaves beats-CV at **0/185**, the headline says
*"extends the horizon over which imagination beats a frozen percept, and still never beats the
trivial floor"* — both halves, in one sentence — and the **capability** claim is capped at
`PROVISIONAL`/negative however large the `T_blind` gain is. A capability **CONFIRM** requires
beats-CV > 0 steps with a separated interval **or** `T_useful@1 m` > 1.4 s.

---

## 6. ⚠️ THE C13 CHECK — what would make my rules return a FAILING value, and proof they can

Two predecessors shipped diagnostics that could not fire: a `T_blind` contiguity rule anchored at
N = 1 (bit-identical by construction), and `frac_draws_T_blind_is_zero = 0.000`, structurally
impossible under the repaired rule. **This section exists so there is no third.**

**(a) The primary statistic's failing value is 1 step (0.1 s), not 0.** `t_contiguous` returns
`start_idx = 1` whenever the first evaluable horizon (step 2) already fails. **The concrete result
that produces it:** an over-aggressive filter — `β = 0.95` smoothing, or `accel_clip = 0.3` — lags
the fed action far behind the model's decoded intent, which can put the imagination arm behind its
own frozen-last twin at step 2. This is a live outcome for individual arms.

**(b) The verdict rule's failing bucket is REFUTE, and it is reachable.** The concrete result that
produces it: the blend curve is **flat at ~25 steps for every α < 1** — i.e. the damage is done by
whatever fraction of the model's action survives, and only its complete removal helps. Nothing in
the two measured endpoints rules this out; the endpoints constrain α = 0 and α = 1 **only**.

**(c) Every diagnostic emitted in this folder is audited for vacuity before it is printed**, and the
audit is itself an artifact (`artifacts/rung1_diagnostic_audit.json`). A diagnostic is admissible
only if **both** of its outcomes are attainable given the rule that produced it. Two that are
declared **inadmissible in advance** and will therefore NOT be quoted as evidence:

* `frac_draws_T_blind_is_zero` — structurally 0 under A4 (already retracted by the ladder).
* `blend=0.0 == own` **on its own** — it is satisfied by a no-op implementation. It is admissible
  only as one half of the §3 pair, beside the anti-no-op requirement.

---

## 7. Mechanism separation — both outcomes committed in advance

The brief names four candidate mechanisms. Each gets a signature that would confirm it **and** a
signature that would refute it, fixed now:

| mechanism | CONFIRMED by | REFUTED by |
|---|---|---|
| **compounding drift** | the penalty `de_own − de_hold` grows **super-linearly** in the horizon; `own_before=m` arms track `own` while the loop is closed and then relax toward hold's **slope** | the penalty is a **level shift** established in the first handful of steps, growing no faster than linearly afterwards |
| **action-magnitude blow-up** (saturation) | fed \|accel\| (and/or \|steer\|) **saturation fraction rises with step index** and far exceeds the true-action arm's; and tightening `accel_clip` / `steer_clip` recovers a large share of the 9.0 s | saturation is flat and low, and the clip family does not move `T_blind` |
| **feedback instability** | step-to-step action **jitter grows** with the step index; `ema` / `every` — which break the per-step feedback without changing the action's mean — recover a large share | jitter is flat; `ema` and `every` do not move `T_blind` |
| **horizon dependence / onset** | `own_before=m` shows a **knee**: little damage for m below some m\*, rapid growth above it | damage accrues from step 2 with no onset — a straight line through the switch sweep |

The action statistics are read from **`fed_actions`**, dumped densely by a separate small audit job,
and — for the full 599-window set — reconstructed from the saved `psi` / `pred_speed` / `v_last`.
⚠️ **That reconstruction is not assumed:** it is proved exact against `fed_actions` on a CPU fixture
(`taniteval/tests/test_blindimag.py`) *and* on the pod's own audit windows before it is used.

---

## 8. Priority order, and incremental banking

A killed agent must still yield value, so the order is fixed now and each stage writes its artifact
before the next begins: **(1)** window-identity + plumbing gates → **(2)** the own↔hold blend curve
(it interpolates the two measured endpoints and is the single most informative curve) → **(3)**
mechanism separation → **(4)** the clip / smooth / update-frequency families → **(5)** the capability
cap and the verdict.

## 9. What is NOT done here

No training is launched. No pod other than **pod2** is touched (pod1 is training, pod3 is running a
situation-classifier build, the eval pod is running situation-semantics); the val cache is read only.
No file under `stack/` is modified. The one library change is **additive** to
`taniteval/taniteval/blindimag.py` — new optional action-filter arguments whose default is a no-op,
pinned bit-identical by §3 and by the existing certification test.
