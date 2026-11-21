# SPEC — THE ZERO-TRAINING KINEMATIC GATE: promote a post-hoc finding into a pre-registered product

`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-kinematic-gate/SPEC.md`
Architecture & Inference FlyWheel · 2026-09-05 · dev-box RTX 4060 / CPU only · ⛔ `tanitad-refcv3` (refcv4b live) and Thor are never touched.
Register: `H-KINGATE-1` (new, this SPEC) · executes `Decisions/2026-09-05-mm-decisions.md` **§M23.2** · cites `D-RL-FANSAFE-1`, `H-ESTIM-SEED-1`, `RETRACTION #30`.
Evidence classes: **MEASURED** (ours + artifact path) · **INHERITED** (another WP, not re-run) · **HYPOTHESIS**.
Tiers: **T0** = readout on the emitted fan / selected path (never a driving claim) · **T1** = self-action OPEN loop through the taniteval harness (PI ruling 2026-09-02: a planner feeding its own predictor is STILL open loop).

⛔ **THIS DOCUMENT IS BANKED BEFORE ANY ARM RUNS. Both outcomes are written below and neither is preferred.**

---

## 1. Why this SPEC exists — and why the finding is not already shipped

The PI's standing correction is that the programme's product is **driving**, not refutation:

> *"its about achieving excellent results and really driving autonomously with a reference implementation"*

The predecessor package (`…/2026-09-05-veto-only-fan-safety/`) ran a 2,000-step RL post-training
arm that **FAILED** its committed exit (`ade_m` **+0.0362 [+0.0261, +0.0460] separated** at T1).
Out of the same package fell something the stream was **not looking for**: a **top-2 kinematic
gate** over the model's own candidate ranking, requiring **no training at all**, which reads
(INHERITED, `…/veto-only-fan-safety/RESULT.md` §8, 480 EVAL windows / 138 episodes):

| metric | model (as shipped) | gate2 | delta |
|---|---|---|---|
| `sel_envelope` | 0.1062 | **0.0729** | **−31 %**, separated |
| `sel_peak_g` | 0.1815 g | **0.1459 g** | **−20 %** |
| `ade_m` (2 s) | 0.4742 | 0.4705 | **+0.0037, NOT separated** |

⛔ **Three reasons that is a candidate and not a result.**

1. **It was found POST-HOC**, by a stream whose SPEC was about an RL veto. `H-ESTIM-SEED-1` is
   the programme's standing rule that a separated CI from one arm is **necessary, not
   sufficient**; the same package watched a replicate kill **four** metrics a single seed would
   have shipped, including `top32_infeasible`, which spans **13.6×** across three runs.
2. **The numbers are quoted from a QUARANTINED artifact.** MEASURED 2026-09-05 18:2xZ: the
   §8 table is read from `raw/fan_rerank_WITHDRAWN_wrong_lambda_operand.json`, and
   `RESULT.md` line 304's claim that it is *"re-confirmed by the corrected re-run
   (`raw/fan_rerank_base.json`)"* is **not yet true** — that file does not exist on disk
   (`/c/Users/Admin/veto_run/raw/` listed; `chain_final2.sh` is still blocked waiting for
   `ZZEVAL-COMPLETE`). §13 of the same document correctly lists it as *queued*. The scoping
   argument for the withdrawal is sound (the selection-rule block never touches
   `out["offset"]`), but **a quoted number whose confirming artifact does not exist is
   INHERITED, not MEASURED**, and this SPEC re-measures it rather than re-quoting it.
3. **The gate's own identity control is described but never executed.** MEASURED at source
   (AST, `raw/no_scene_input.json`): `rl_fan_rerank_probe.py`'s docstring says *"k = 1 is the
   model itself by construction — a built-in identity control"*, and
   `GATE_KS = (2, 4, 8, 16, 32, 128)` — **k = 1 is not in the tuple**, so the control is
   documented and not run. ⛔ That is precisely the deliberate-regression control M23 requires,
   and it is missing.

## 2. ⭐ THE ONE VARIABLE

> **`one_variable`: the gate is ON or OFF. Nothing else moves.**

The gate is a **selection rule over the model's own emitted fan**. Both arms therefore share, by
construction and not by care:

| held | how it is held |
|---|---|
| checkpoint | `ckpt_step40284_frozen.pt`, md5 asserted, `step == 40284` asserted |
| corpus / windows | the identical window list, drawn once and reused by every rule |
| forward pass | **the SAME forward.** Every rule reads one banked fan; no arm re-runs the model |
| cost metric | one `fan_safety.score_paths` call per window, shared by every rule |
| weights, decoder steps, `nav_cmd`, `v0`, lead mode | untouched — the gate is post-decode |
| gradients | **none exist.** Zero training steps, zero optimiser, zero parameter change |

⇒ Because both arms are **the same forward through the same weights**, the comparison is exactly
paired at the window level: the only difference between `model` and `gate2` on any window is
*which of the 128 already-emitted candidates is returned*. That is a far tighter pairing than the
two-checkpoint comparisons the predecessor package ran, and it is why the guards below can demand
**bit-identity** rather than a tolerance.

## 3. ⛔ WHICH VARIANCE THE INTERVAL ANSWERS — all three, named

`H-ESTIM-SEED-1` exists because an interval was quoted against a question it does not answer.
This SPEC names all three variances **before** any number is produced.

| # | variance | does it apply to this lever? | instrument |
|---|---|---|---|
| **V1** | **episode / window draw** — *"would another draw of episodes say this?"* | **YES** — this is the live one | paired **episode-cluster bootstrap** (`taniteval/ci.py`), cluster = clip, n_boot 4,000, seed 11. ⛔ never `overlapping_holdout_se`. **Plus** an independent, **episode-disjoint second draw** (§6 `G-REP`), because the bootstrap resamples *within* one draw |
| **V2** | **training run** — *"would another training run say this?"* — the variance `H-ESTIM-SEED-1` was written about | ⭐ **STRUCTURALLY ABSENT.** There is no training run. Both arms are the same frozen checkpoint, asserted by md5 in the artifact. A "replicate arm with the same flags and a different seed" **cannot differ**, because no seed enters. This is not a claim that the gate is noise-free — it is a claim that *this particular* hole in the estimator is closed **by construction** for this lever, and it is the strongest structural property the gate has | md5 assertion + `G-OFF` bit-identity |
| **V3** | **inference sampling** — *"does the same checkpoint evaluated twice disagree?"* | ⚠️ **MUST BE ASSERTED, NOT ASSUMED.** `refc.py` contains stochastic ops that are **not all** guarded by `self.training`: `:2001` is (`torch.randn_like(x) if self.training else torch.zeros_like(x)`) and `:1687` is, but `:1720` (`x_in = bank + … + torch.randn_like(bank) * cfg.noise_std`, the metre-space **sampler** branch) and `:1501` (`torch.randint` under `anchor_withheld_bank='random'`) are **config-gated, not eval-gated**. Whether this checkpoint's decode is deterministic is a property of ITS config and must be measured | `G-DET` (§6): two forwards, same process, same batch, compared **bitwise** |

⇒ **Committed statement of scope:** the primary interval answers **V1**. **V2 does not exist for
this lever.** **V3 is measured by `G-DET`**; if `G-DET` shows the decode is stochastic, an
**inference-seed replicate** becomes a real estimate and its spread is added to the floor of §5,
and the SPEC's headline is downgraded accordingly. If `G-DET` shows bit-identity, the
inference-seed replicate is a **structural identity, not an estimate** — and it is reported as
such, never as a passed test that carried information. *(That distinction is `H-ECHO-4`'s
structural-zero rule: an identity is not a measurement of zero.)*

## 4. The arms — committed before any of them runs

Base: `/c/Users/Admin/rl_refcv3_min/base/ckpt_step40284_frozen.pt`, `--expect-step 40284`.
Corpus: the **EVAL** split, `/c/Users/Admin/run_refcv3_ol/data/eval` +
`s2_labels_v7.2_eval.jsonl.gz` + `b1_eval_lead_block.npz` — the same three the predecessor used,
**NON-PARITY, as the base itself is** (`D-RL-READY-1`).

Every "arm" below is a **selection rule evaluated on one banked forward**. They are not separate
runs, and that is the point.

| arm | rule | role |
|---|---|---|
| `model` | `out["sel_idx"]` — refcv3 as deployed | ⭐ the FLOOR. Every delta is against this |
| ⛔ `gate1` | top-**1** by the model's own `sel_score` (reach-masked), re-ranked by `feasibility + comfort` | **THE DELIBERATE-REGRESSION CONTROL.** The gate DISABLED. Must reproduce `model` **bit-identically**, and must do so **on the selected INDEX**, not only on the metric values (§6 `G-OFF`) |
| ⭐ `gate2` | top-**2**, re-ranked by `feasibility + comfort` | **THE PRODUCT.** The arm this SPEC exists to accept or reject |
| `gate4`, `gate8`, `gate16`, `gate32`, `gate128` | the same at wider k | the frontier — reported so the k choice is a MEASUREMENT, not an assumption |
| `kin_only` | argmax of `feasibility + comfort` over all 128 (= `gate128`) | the no-model-ranking extreme |
| `oracle` | the fan's best-ADE candidate | T0 ceiling, **never deployable**, reported to show the tension |

⚠️ **`gate1` is not free real estate.** `model` is `out["sel_idx"]`; `gate1` is
`argmax(sel_score.masked_fill(~reach_keep, -inf))`. If those two differ on any window, the gate's
candidate ordering is **not** the model's own selection, and part of any measured gain would come
from re-deriving the ranking rather than from the kinematic tiebreak. **That is a confound, and
`G-OFF` is the instrument that finds it.** Its disagreement rate is reported whatever it is.

## 5. ⛔ SEPARATION FLOORS AND THE REPLICATE FLOOR — per metric, from its own quantum and units

Rate metrics (`envelope`, `infeasible`, `kamm_over`, `off_reach`, `contact`, `ttc_below`,
`flagged`): floor **1e-4**. Metrics in **g** (`peak_g`): **1e-3 g**. Metrics in **m**
(`ade_m`, cross-track, along): **1e-3 m**. Metrics in **m/s** / **m/s²** / **rad** / **rad/s**:
**1e-3** of the unit. ⛔ **Any metric whose `model` value is below its own floor is stamped
`UNDETECTABLE-DOWNWARD` and is reported as neither a null nor a win.**

**The replicate floor**, per metric, is the disagreement between the two **episode-disjoint**
draws (§6 `G-REP`):
```
floor(m) = | delta_A(m) - delta_B(m) |          # same rule, same estimator, disjoint episodes
```
A gate effect is quotable **only** when all three hold:
1. `gate2 − model` is separated on draw **A** and on draw **B**, and
2. the two point estimates have the **same sign**, and
3. `min(|delta_A|, |delta_B|) > floor(m)`.

⚠️ A metric failing (3) is reported **WITHIN-NOISE** — not a null, and not a win.

## 6. ⛔ GUARDS — checked in this order, each VOIDS the panel before any outcome is selected

| id | guard | pass condition |
|---|---|---|
| **G-CKPT** | the object under test is the one named | ckpt md5 recorded; `prov["step"] == 40284`; **both** oids/hashes 32 hex chars or the guard reports **INCONCLUSIVE**, never PASS |
| **G-DET** | **inference determinism (V3)** | the model is run **twice on the same batch in the same process**; `anchor_traj`, `sel_score`, `sel_idx` compared **bitwise** (`torch.equal`). Result recorded either way — a FAIL does not void, it **re-classifies V3 from identity to estimate** and forces the inference-seed replicate |
| ⛔ **G-OFF** | **the deliberate-regression control** | with the gate DISABLED (`k = 1`), **every** reported metric must equal `model`'s **exactly** (`max abs diff == 0.0`, not a tolerance) **AND** `gate1_idx == model_idx` on **every** window. ⭐ **The index assertion is the one that matters**: RETRACTION #30's λ-sweep control **passed while interpolating the wrong tensor** because it checked the arithmetic and not the OBJECT. Two different candidates can carry equal metric values; only the index says the control operated on the same object. **Both halves must pass; the disagreement count is printed whatever it is** |
| **G-PAIR** | the pairing is real | every rule's row set is the **same window list in the same order**; asserted by comparing the `wi` vector per rule, not assumed from the loop |
| **G-REP** | **the replicate exists and is independent** | two draws, **episode-disjoint** (`set(eid_A) ∩ set(eid_B) == ∅`, asserted and printed). A result quoted from one draw is VOID |
| **G-CTRL** | the probe can see a difference at all | at least one rule in the panel must move a metric past its floor (`oracle` on `ade_m` is the designated positive control). ⛔ A panel in which **nothing** moves is a broken probe, not a null result |
| **G-READ** | no absence claim rests on a failed read | every count is paired with a same-breath control that must read non-zero; a zero from an unread file is reported **INCONCLUSIVE** |

## 7. ⭐ ACCEPTANCE — both outcomes committed in advance, neither preferred

**PRIMARY endpoint (T0, selected path):** `sel_envelope` and `sel_peak_g` on the selected
trajectory, with `ade_m` as the **cost**. Paired episode-cluster bootstrap, n_boot 4,000, seed 11.
Deltas are reported as **gate2 − model**, so a **negative** `sel_envelope`/`sel_peak_g` means the
gate is **safer** and a **positive** `ade_m` means the gate is **worse on ADE**.

> ### ✅ SUCCESS
> **All four of:**
> 1. `sel_envelope` (gate2 − model) is **negative** and **separated** on **both** draws, same
>    sign, and clears the replicate floor of §5; **and**
> 2. `sel_peak_g` is **negative** on both draws (separated on at least one) and clears its floor;
>    **and**
> 3. `ade_m` (gate2 − model) is **NOT separated**, **or** separated with
>    `|delta| < 0.010 m` — a tenth of the RL arm's failing `+0.0362 m` and an order of magnitude
>    below the `model`→`ha0` gap; **and**
> 4. every guard in §6 passes, `G-OFF` included.
>
> ### ⛔ FAILURE
> **Any one of:**
> 1. `sel_envelope` does not clear the replicate floor on both draws (⇒ WITHIN-NOISE — the
>    finding was a draw artifact and M23's promotion decision was right to withhold it); **or**
> 2. `ade_m` is separated **and** `|delta| >= 0.010 m` (⇒ the gate buys safety with driving
>    accuracy, which is the trade the predecessor's `kin_only` row already shows is real at
>    −0.0655 m); **or**
> 3. **`G-OFF` fails** — the gate disabled does **not** reproduce the ungated numbers
>    bit-identically, or picks a different index. ⛔ **This is a HARD failure and it voids the
>    measurement rather than merely losing it**: a gate that changes something when it is off is
>    not the thing that was measured, and no delta computed against it is interpretable.

**SECONDARY endpoint — ⛔ THE FOUR FAMILIES, per family, never pooled** (binding, PI 2026-08-02).
ADE is reported **beside** them and never as "the result".

| family | metrics on the selected path | expectation (HYPOTHESIS, committed now) |
|---|---|---|
| **LONGITUDINAL** | speed MAE, along-track MAE, accel MAE, headway/TTC to the lead where a lead exists (n reported per family) | NULL to slightly better: the gate's `comfort` term penalises jerk, which is longitudinal |
| **LATERAL** | cross-track MAE, heading error, curvature error, yaw-rate error | ⭐ **the family the gate should move**: `feasibility` caps `|kappa|` and `comfort` caps `lat_acc`, so a curvature/lat-acc reduction is the mechanism's own signature. A `sel_envelope` gain with **no** lateral movement would be evidence the gain is bookkeeping, not geometry |
| **TACTICAL** | selected-vs-executed manoeuvre, lat/lon manoeuvre correctness, **anchor/goal selection** — the gate IS a selection rule, so `sel_idx` agreement with the model and with the oracle is this family's core row | the gate changes the selection on ~48 % of windows (INHERITED `agree 0.523`); the question is whether the manoeuvre **class** changes with it |
| **STRATEGIC** | route/nav correctness | ⭐ **readable for the first time** — a defect that silently removed it from every refcv3 arm was fixed today (`resolve_labels_path`, `D-NAVCOMP-SHAPE-1`). ⛔ **If it still refuses, the block is read for `defect: true`: a `TypeError` out of our own module is a DEFECT, reported as one — never as a refusal and never silently dropped** |

⛔ **A family that cannot be computed is reported per family with the reason and the `n`** — never
silently dropped (binding rule 5 of the four-families directive).

**TIER DISCIPLINE.** The panel above is **T0**: a readout of which already-emitted candidate is
returned, scored against the logged future. ⛔ **It is not a driving claim and must never be
compared against a T1 number** (`ade_m` here is the 2 s prefix ADE over 4 slots; T1's `ade_m` is
the harness's, and the two are different objects). The **T1** leg — the gate wired into the
inference path and run through `openloop_suite.py` / `paired_openloop.py` against the banked base
— is specified here and run **only if the dev-box GPU frees**; if it does not, it is reported
**NOT RUN**, with the instrument staged, and no T1 claim is made. ⛔ Reporting a T0 number under a
T1 heading is the failure this paragraph exists to prevent.

## 8. What this SPEC does NOT claim, whatever the result

1. Nothing about **closed loop**. Even the T1 leg is self-action OPEN loop (PI ruling 2026-09-02).
2. ⛔ **The gate is a MITIGATION, not the fix.** MEASURED (INHERITED, `…/veto-only-fan-safety/`
   §6): the frozen anchor vocabulary is drivable at **0.48 g** while the decode emits **4.11 g** —
   **8.56×** — so the infeasibility is manufactured **downstream of the vocabulary**. Whatever
   fraction of that gap the gate closes will be **stated as a number** in the RESULT, not
   gestured at. The real work item is a **feasibility-aware decode** (§9), and a successful gate
   does not retire it.
3. `kamm_over` is a **LOWER bound**: the emitted fan is free waypoints, so
   `flyability.friction_load` cannot be applied and the finite difference under-reports by
   1.21–1.85×. Ranks are invariant to a monotone under-report; **levels are not**.
4. The corpus is **NON-PARITY** (`physicalai-train-e438721ae894` is not the split used here), as
   the base itself is. No cross-arm comparison outside this panel is licensed.
5. **`sel_envelope` is not contact.** Selected-path contact already ties the human
   (`os − g` not separated over 1,466 lead windows); this SPEC is about the **envelope**, and a
   gain here is a comfort/feasibility gain, not a collision-rate gain.
6. ⚠️ **"No scene input" is a claim about the RANKING FUNCTION, not about the candidate set.**
   MEASURED at source (`raw/no_scene_input.json`, AST + permutation, three positive controls):
   `feasibility` and `comfort` read only the candidate's own waypoints and six programme
   CONSTANTS (`dt`, `a_max`, `kappa_max`, `jerk_max`, `lat_acc_max`, `min_motion_m`) — no `v0`,
   no `lead_path`, no `obstacles`, no `gt_traj`. But the **candidate set** the gate ranks over is
   the model's own top-k by `sel_score`, reach-masked — both scene-conditioned model outputs.
   ⇒ The correct claim is **"adds no new scene input and no new perception"**, and that is what
   the RESULT will say. It is still admissible under the vision-only-at-inference rule, because
   every input it uses the deployed model already computes.

## 9. The successor, scoped not built (⛔ NOT launched by this SPEC)

A **feasibility-aware decode** is the real work item, and §8.2 prices why: the gate can only
choose among candidates the decode already emitted, and the decode emits a fan whose mean peak-g
is **8.56×** its own vocabulary's. The one-page design and its cheapest discriminating experiment
are written into the RESULT's §"successor". ⛔ **It is not launched here** — launching it inside a
validation SPEC would be exactly the scope creep that makes a post-hoc finding unfalsifiable.

## 10. Cost and machine discipline

⛔ `tanitad-refcv3` (refcv4b-b1-v72-40k live, ~62 %) is **never touched**. Thor is **not used**.
The dev-box RTX 4060 is at **100 % / ~7.0 of 8.2 GB** (MEASURED 2026-09-05 16:24Z: four
`refav1_arm.py` + two `openloop_suite.py` for `s1/veto200`), so **the banking forward runs on CPU**
unless the GPU frees, `OMP_NUM_THREADS=6`, **one arm at a time**. Device is **recorded in the
artifact** and is held constant across every rule and both draws — a CPU/GPU mix inside one panel
would break `one_variable`. The whole rule panel, both draws, all families and all bootstraps are
computed **offline from the banked tensors at zero further model cost**, which is why the guards
above can be this strict.
