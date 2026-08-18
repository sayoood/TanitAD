# F-9 (interaction curriculum) and F-11 (multi-tick strategic rollout) — BUILT

**2026-08-18** · branch `agent/arch-inf-20260803` · base HEAD `45b8e44` ·
evidence class per claim · dev-box CPU only, **no training run, Thor untouched (PID 25477 live)**

---

## 0. Headline — and the one thing that needs a PI decision

**Both cells are built, measured, wired and pinned. The default build is UNCHANGED at
87,893,449 params / 405 keys** — MEASURED four ways through the *real* `build_stack_from_args`
path (default, F-9 on, F-11 on, both on: **delta (0, 0) every time**, §4). The live tensor-strict
v6F S-W resume is untouched by either.

⛔ **ESCALATION — THE CATALOG'S 8–30 s STRATEGIC HORIZON IS NOT REACHABLE ON THIS CORPUS, AND
THAT IS ARITHMETIC, NOT AN OPINION.** F-11 as a *mechanism* is expressible and is built. Its
*specified horizon* is not. `t_max = frames − window − max_horizon` and a K-tick roll needs
`max_horizon = K·stride_str`, so on the 120-frame cache windows/episode is **114 − 20K**:

| K | horizon | windows/episode | vs the 1-tick baseline (94) |
|---|---|---|---|
| 1 | 2 s | 94 | — |
| 2 | 4 s | 74 | −21 % |
| 3 | 6 s | 54 | −43 % |
| **4** | **8 s** | **34** | **−64 %** ← the catalog's *floor*, and the only usable point |
| 5 | 10 s | 14 | −85 % |
| **6** | **12 s** | **0** | ⛔ **the corpus is exhausted** |

The catalog asks for **4–15 ticks**. **Only K=4 and K=5 exist at all; 30 s is longer than a
12 s episode.** This needs a **longer re-extraction of the SAME 2376 episodes** (admissible per PI
decision D4) — never a re-pick of which episodes enter. **This is a CATALOG/CORPUS decision, not
something an implementation may silently resolve by truncating the ladder.**

⚠️ **AND PAST `max_k` THE FAILURE IS A PARITY BREAK, NOT AN ERROR.** D4 settled that `max_horizon`
is a windowing choice *inside* episodes parity already selected — but that reasoning holds only
**while every episode still contributes**. Past `max_k` an episode contributes zero windows, so a
corpus of unequal-length episodes drops its **short** ones silently: an effective re-selection.
The guard therefore refuses on the **shortest** episode and refuses again on any non-zero drop-out
count, rather than training on the survivors.

⭐ **A SECOND FINDING, MEASURED, that changed F-9's implementation.** The obvious reading of
*"multi-agent kinematic entropy"* — Shannon entropy of the normalised occupancy raster — is
**MAXIMAL ON AN EMPTY ROAD**. Normalising a near-zero field divides noise by noise and returns
near-uniform, whose entropy is the functional's maximum. MEASURED on synthetic rasters:

| raster | bare spatial entropy | shipped `multi_agent_kinematic_entropy` |
|---|---|---|
| empty road (1e-4 noise) | **0.9649** | **0.0064** |
| one moving agent | — | 0.1863 |
| four moving agents | **0.2500** | **0.4886** |

⇒ the naive functional is **INVERTED by 3.9×**, and a curriculum built on it would drive training
*towards empty scenes while its name said the opposite*. Pinned by
`test_bare_spatial_entropy_is_maximal_on_an_empty_raster`, which demonstrates the failure on the
bare functional and then shows it absent from the shipped one.

---

## 1. The spec, quoted — two independent locations per cell

Established **before a line was written**, per "absence found at ONE location is not absence".

### F-9 / catalog T3

| source | quote |
|---|---|
| `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:66` | *"T3 \| interaction curriculum: windows ranked by MULTI-AGENT kinematic entropy measured from the O-layer's own predicted occupancy (self-supervised, after O2/O3 make it non-degenerate) \| curriculum from free-flow → dense interaction \| P7 calibration ρ ≥0.3 held on interaction-rich strata, not just pooled"* |
| `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:59` | *"O4 is the ego-kinematic version only; T3's multi-agent extension needs the P8 occupancy readout in the loop. Fix F-9 (gated on P8 maturity)"* |
| `…/DIAGRAM_CONFORMANCE.md:214` | *"F-9 \| P3 \| T3 interaction curriculum (multi-agent entropy from the P8 occupancy readout) — gated on P8 maturity."* |

### F-11 / catalog S1

| source | quote |
|---|---|
| `…/V6_TRAINING_MEASURES.md:79` | *"S1 \| long-horizon latent prediction (own predictor, Δt ≈ 1 s ticks) on the T-layer's latent sequence \| strategic dynamics = evolution of manoeuvre context, not pixels \| ADE(8–30 s) vs CV/corridor baselines at T1"* |
| `…/DIAGRAM_CONFORMANCE.md:70` | *"training target is **one strategic tick ahead** (`stride_str = 20` steps = 2.0 s at the 0.5 Hz clock) — a 1-tick loss; the 8–30 s capability appears only as gate-reported `S1_ade_8_30s`… Multi-tick strategic rollout training is not built. Fix F-11"* |
| `…/DIAGRAM_CONFORMANCE.md:101` | *"horizon 8–30 s+ \| 🟨 PARTIAL \| trained horizon is one 2 s tick…; 8–30 s is gate-reported only. Fix F-11"* |
| `…/DIAGRAM_CONFORMANCE.md:216` | *"F-11 \| P3 \| S1 multi-tick strategic rollout (8–30 s = 4–15 strategic ticks) — currently 1-tick; the gate reports `S1_ade_8_30s` against a capability the loss never exercises."* |

### What the spec does NOT say — the assumptions, named

| # | gap | narrowest defensible reading taken | declared at |
|---|---|---|---|
| A1 | *"kinematic entropy"* is not defined operationally | entropy over cells of the **occupancy CHANGE** across ticks, **mass-gated** so an empty scene scores exactly 0 | `multi_agent_kinematic_entropy` docstring |
| A2 | *"free-flow → dense"* — a direction, not a schedule | linear ramp of a **signed** weighting exponent, `alpha_start=−1` (free-flow-biased) → `alpha_end=+1`, over `warmup_frac` | `T3Curriculum` docstring |
| A3 | where the T3 score comes from at training time | a **precomputed, provenance-stamped artifact** (`--t3-scores`), never an inline P8 pass | `load_t3_scores` docstring |
| A4 | the strategic ACTION at ticks k>1 | the model's **own** `act_head_str` on each tick's predicted latent, through the same planner cut `forward` applies (there is no GT `a_str`: S2 is `NOT BUILT` by recorded decision) | `s1_rollout_loss` docstring |
| A5 | how ticks are weighted in the multi-tick loss | **uniform mean**, declared — and every tick logged separately so the degradation curve is auditable rather than pooled | `s1_rollout_loss` docstring |

---

## 2. ⛔ C115 — asked before implementing, and it changed the reading of F-11

C115 (MEASURED 2026-08-17): `z_tac` — hence `z_str`, which is uplinked from it — is a function of
the **LAST FRAME ALONE**; `encode_window` flattens `[B, W]` into the batch axis so no frame sees
another. The brief flagged F-11 as the cell at risk. **It is not, and the reason is worth stating
precisely**, because the same test rules *out* half of catalog T2 and rules *in* all of S1:

> **The invariance is in the LATENT. The temporal structure F-11 needs is in the PREDICTOR.**

`predictor_str` is a genuine `z(t) → z(t + stride_str)` map. A multi-tick rollout is that map
composed with itself — exactly the shape `o5_rollout` already uses one layer down — and the targets
are per-frame encodes of frames 2 s apart, which is what `s1_latent` already does at k=1. Nothing
in F-11 asks a latent to integrate a window. ⇒ **expressible, and built.**

⚠️ **The rule the brief states generalises, and it lands on the OTHER cell here.** *"Prove the
representation is SENSITIVE to the thing your loss acts on."* F-9's analogue is not a latent
question but a **functional** question — is the ranking signal sensitive to *interaction*? — and the
answer at the naive reading is **no, it is anti-sensitive** (§0). That is why the entropy is
mass-gated and why `t3_rank_control` exists.

⚠️ **One spec deviation noted, NOT introduced by this change.** `V6_TRAINING_MEASURES.md:79` says
S1 predicts *"on the T-layer's latent **sequence**"*. The existing code predicts a strategic latent
derived from a **single** frame, and F-11 inherits that unchanged rather than inventing a sequence
encoder. The multi-tick roll is a *sequence of predictions*, not a *prediction from a sequence*.
Recorded here; not resolved by fiat.

---

## 3. MEASURED — the numbers this deliverable rests on

**Parameter cost, MEASURED BY BUILDING** through `build_stack_from_args` (the real launch path),
never estimated:

| build | params | state_dict keys | delta |
|---|---|---|---|
| default `V6Config()` | **87,893,449** | **405** | — |
| `--t3-scores … --o4-alpha 0 --t3-alpha-start -1.0` (F-9 on) | 87,893,449 | 405 | **(0, 0)** |
| `--w-s1-multi 1.0 --s1-multi-k 4` (F-11 on) | 87,893,449 | 405 | **(0, 0)** |
| both on | 87,893,449 | 405 | **(0, 0)** |

⭐ **Both cells are structurally zero-parameter, and that is not a coincidence to be re-checked
each release — it is pinned by construction.** F-9 introduces no `nn.Module` at all (a curriculum
is a schedule; a score is a number). F-11 re-rolls `predictor_str`/`act_head_str`, both **already**
`layer_str`.

**The reachability table** (§0) — MEASURED by executing `reachable_strategic_ticks(120, window=6,
stride_str=20)`. Its K=1 row (**94 windows/episode**) reproduces the independently MEASURED figure
in `PI_DECISIONS_2026-08-12.md` §D4, which is the corroboration that the 120-frame episode length
(INHERITED from `V6_TRAINER_DESIGN.md §3.6` + the `-w120-` cache name) is the right input.

**The entropy inversion** (§0) — MEASURED on synthetic rasters, `test_v6_t3_curriculum.py`.

**End-to-end execution, F-11** — MEASURED, a real `--dry-run` of an S-S launch:

```
[v6 dry 1] {"stage":"S-S","s1_latent":0.8682,"s1_multi":0.8478,"s1_multi_k":3,
            "s1_multi_k1":0.8519,"s1_multi_k2":0.8509,"s1_multi_k3":0.8408,
            "loss":1.7160,"terms":["s1","s1_multi"],"gnorm":1.7471}
[v6] X3 isolation pass=True violations={'planner_to_encoder':0,'tactical_to_below':0,'strategic_to_below':0}
```

⭐ The isolation probe still passes with the roll live — the multi-tick composition opens no
forbidden edge.

---

## 4. Stage contract — the earliest legal insertion point, stated plainly

`STAGE_MAY_INTRODUCE` and `RESUME_CONTRACT` live in **`stack/scripts/train_v6_staged.py`**, not in
`v6.py` (the brief's pointer; same correction F-7 filed).

| fact | value | consequence |
|---|---|---|
| `STAGE_MAY_INTRODUCE["S-W"]` / `["S-J"]` | `()` | nothing may be introduced into an S-W or S-J resume |
| `STAGE_GROUPS["S-S"]` | `("layer_str",)` | the stage that trains `predictor_str` **and** `act_head_str` |
| `encoder`/`readout`/`aux` in S-T and S-S | frozen | neither cell touches them |
| `predictor_str.` / `act_head_str.` group | `layer_str` (existing `_GROUP_PREFIXES` entries) | F-11 needs **no** new prefix |

⇒ **NEITHER CELL NEEDS AN INSERTION POINT, and both are the F-8 case rather than the F-7 case.**
F-7 needed `STAGE_MAY_INTRODUCE["S-T"] += ("t2_head.",)` because it added five keys. These add
**zero keys**, so there is nothing for the allowlist to adjudicate, and both may be enabled over an
existing checkpoint loaded **tensor-strict**. Pinned by `test_F9_needs_no_stage_may_introduce_entry`
and `test_F11_needs_no_stage_may_introduce_entry`.

**Where each is in force:**

- **F-11 is an S-S (or S-J) measure.** `for_stage("S-W")` and `for_stage("S-T")` zero
  `w_s1_multi` for exactly the reason they zero `s1_latent`: `layer_str` is frozen there, so the
  term would be advertised in the launch line and train nothing. `preflight` refuses it in both.
- **F-9 is a SAMPLER measure and is therefore NOT stage-gated** — it changes the data mix, not a
  loss, so there is no frozen module for a stage check to protect. ⚠️ **This is a judgement call and
  it is flagged, not hidden:** the catalog lists T3 in the **LAYER T** table
  (`V6_TRAINING_MEASURES.md §2`), which would argue for gating it to S-T. I did not, because gating
  a sampler by stage asserts *which stage benefits from an interaction-rich mix*, and no evidence
  supports that claim yet. **PI may prefer the gate; it is a one-line change either way.**

### The `06b8782` defect class — checked, and it cannot apply

Commit `06b8782` changed what S-J trains by **appending to `MODULE_GROUPS`** without touching S-J's
declaring line. F-7 avoided it by adding a `_GROUP_PREFIXES` entry to an **existing** group.

**These two cells avoid it more strongly: they introduce no parameter to assign to a group at all.**
`MODULE_GROUPS` and `LADDER_UNTRAINED_GROUPS` are asserted unchanged in both suites
(`test_the_06b8782_class_does_not_apply_to_F9` / `_F11`). There is no mechanism by which a
zero-parameter loss can move a stage's trainable set.

---

## 5. Guards — wired, and every refusal paired with its opposite (C95/C97)

| guard | where it FIRES | its paired "legal launch raises NOTHING" test |
|---|---|---|
| **F-11** K < 2 (*"a multi-tick term with one tick is a duplicate weight"*) | `s1_rollout_loss` + `preflight` | `test_a_legal_call_raises_NOTHING`, `test_preflight_passes_a_legal_F11_launch` |
| **F-11** in a stage that freezes `layer_str` | `preflight` | ditto |
| **F-11** K ≥ 6 (**zero windows**) — before the corpus mounts | `preflight` | ditto |
| **F-11** K unreachable on the **realised** corpus | `train()`, on the shortest episode | — |
| **F-11** any episode dropped to zero windows (**the parity break**) | `train()` | — |
| **F-11** missing `z_str_multi_target` | `v6_loss_step` | `test_v6_loss_step_runs_the_term_when_the_target_is_present` |
| **F-9** entropy on 1 tick / a snapshot / **logits** | `multi_agent_kinematic_entropy` | `test_a_legal_call_raises_NOTHING` |
| **F-9** curriculum REVERSED (`alpha_end < alpha_start`) | `T3Curriculum` + `preflight` | `test_a_legal_curriculum_construction_raises_NOTHING` |
| **F-9** `floor ≤ 0` (infinite weight **and** a reachability break) | `T3Curriculum` + `preflight` | ditto |
| **F-9** **UNDECLARED** score artifact (no provenance) | `load_t3_scores` | `test_the_loader_accepts_a_well_formed_declared_artifact` |
| **F-9** score/window length mismatch, non-finite, negative, all-zero | `load_t3_scores` | ditto |
| **F-9** `--t3-scores` with `--o4-alpha > 0` (**two levers, one axis**) | `train()` + `preflight` | — |
| **F-9** `--t3-alpha-*` without `--t3-scores` | `preflight` | — |

⭐ **The admissibility guard is the one to read twice.** T3's score descends from the P8 decoder,
which trains on the obstacle join — a **label** path — while O4's own docstring says the join is
*"frozen-probe/eval-strata material, **never a training-time selector**"*. The resolution is the
**F-10 precedent** (`DIAGRAM_CONFORMANCE.md:57`): a label-derived *sampler* input is admissible
**because it is a data mix and not a model input**, but *"must be declared"*. `load_t3_scores`
**refuses an artifact with no provenance stamp**, and the stamp is written into `config.json`, so
the declaration survives the console rather than living in a line nobody re-reads.

⭐ **`--o4-alpha` DEFAULTS TO 1.0**, so a T3 arm must pass `--o4-alpha 0` explicitly. That is not an
inconvenience — it *is* the declaration that this run swapped one saliency lever for another.

---

## 6. The degeneracy checks — both cells, both wired

**F-9 — the functional's degenerate input.** §0: the naive entropy is *maximal* on an empty road.
The mass gate `1 − exp(−M/scale)` is **exactly 0** at zero mass, so an empty raster scores exactly
0 whatever its entropy is (`test_an_exactly_empty_raster_scores_exactly_zero`). Parked agents —
occupancy without kinematics — also score **exactly 0**.

**F-11 — the degenerate SOLUTION, and it is not the same shape as F-8's.** F-8 turned out to sit at
its own global minimum at init. F-11 does not: at random init the predictor is far from the target
and the loss is large. **But its easiest descent direction is the identity.** MEASURED
(`test_the_HOLD_rollout_beats_an_untrained_one_on_slow_drift_targets`): on slow-drift targets
(`z + 0.05·noise`, the realistic regime for a per-frame encode 2 s apart) the **HOLD** rollout —
emit `z_str` and never move — **beats the untrained roll**, and `s1_persistence_control` returns
`NO_BETTER_THAN_HOLD`.

⚠️ **And the control's own positive test surfaced a sharper fact.** Against `z + noise` targets the
identity is **Bayes-optimal** and *no* predictor can beat it, at any amount of training — the
control only becomes informative once the strategic latent has **learnable** dynamics (pinned with
a `z + k·d` drift, where 300 Adam steps do flip the verdict to `OK`). ⇒ **an S1 result is
uninterpretable without this control**, and a `NO_BETTER_THAN_HOLD` verdict may mean the latent has
no strategic dynamics rather than that the predictor failed.

**Both controls carry a minimum-n rule** — `S1_CONTROL_MIN_N = T3_CONTROL_MIN_N = 32`, refusing
below it and returning **no ratio at all** so there is no number to quote out of context. Rationale
is the sibling T2 measurement at the null (true ratio 1 by construction): at n=4 the ratio spanned
**0.397–3.361**. `t3_rank_control` additionally reports per-side SEM and names the `INVERTED` and
`DEGENERATE_ALL_ZERO` cases explicitly — and its own inversion test **feeds it the bare degenerate
functional's scores and asserts the control catches them**.

---

## 7. What was built, file by file

### `stack/tanitad/models/v6.py` (+ ~230 lines, zero parameters)

- `multi_agent_kinematic_entropy(occ[B,K,H,W]) → [B]` — mass-gated entropy of the occupancy-change
  field; refuses K<2, non-`[B,K,H,W]` shapes, and **logits** (P8 emits logits; entropy of a logit
  field is not a specified quantity).
- `T3Curriculum` — frozen dataclass; `alpha_at(progress)` / `weights_at(scores, progress)`.
- `t3_rank_control` + `T3_CONTROL_MIN_N`, `T3_MASS_SCALE`.
- `__all__` extended.

⭐ **`saliency_weights` (O4's shared helper) was NOT touched.** T3 needs a *negative* exponent and
O4's guard refuses one; the resolution was T3's own weighting, so O4's contract is exactly what it
was. Pinned by `test_o4s_saliency_weights_guard_was_NOT_weakened` — F-7's lesson about not editing
a shared module to fit a new cell into it.

### `stack/scripts/train_v6_staged.py`

- `V6LossWeights.w_s1_multi = 0.0`; zeroed in `for_stage("S-W")` and `for_stage("S-T")`.
- `reachable_strategic_ticks`, `s1_rollout_loss`, `s1_persistence_control`, `S1_CONTROL_MIN_N`.
- `load_t3_scores` — the artifact loader and its six refusals.
- `v6_loss_step`: the `w_s1_multi` term, **not nested under `s1_latent`** (nesting would make the
  multi-tick arm unattributable — the `--v2` conflation failure).
- `need` now accounts for `K·stride_str`; the per-tick targets are built in **one** encoder pass
  under `no_grad` via `layer_targets`, matching the `need_k` block's batching discipline.
- The corpus reachability + episode-drop-out refusals; the T3 score load, curriculum construction
  and sampler swap; the per-step curriculum refresh (recomputed only when the exponent moves at
  3 dp — `progress` is over the **whole** run, so a resume re-enters the curriculum rather than
  restarting it).
- `synthetic_train_batch(..., s1_multi_k=0)` so `--dry-run` exercises the same path the pod will,
  and **omits** the key when the flag is off (a target that is always present cannot prove a guard
  fires).
- `preflight`: nine new refusals. CLI: `--w-s1-multi`, `--s1-multi-k`, `--t3-scores`,
  `--t3-alpha-start`, `--t3-alpha-end`, `--t3-warmup-frac`, `--t3-floor`.
- `config.json` gains `t3` (with the provenance stamp) and `s1_multi` (the reachability census).

⛔ **PARITY IS UNTOUCHED BY BOTH.** F-9 **reweights** the draw and never removes a window —
`floor > 0` keeps every window strictly reachable at every alpha, pinned at the weight level *and*
at the **draw** level (`test_free_flow_windows_are_still_drawn_at_full_dense_bias`). F-11 re-selects
no episode and **refuses** the horizon at which episodes would start dropping out.

⚠️ **RNG discipline.** With both cells off, nothing in the new blocks runs; `gen` is consumed
exactly as before, so every other term stays bit-for-bit.

---

## 8. Suites

Run **separately** with `PYTHONUTF8=1`, on an otherwise idle box. ⛔ **Exit codes were not
trusted** — the tails were read.

| suite | result |
|---|---|
| `stack/tests/test_v6_t3_curriculum.py` (F-9) | **40 passed** |
| `stack/tests/test_v6_s1_multitick.py` (F-11) | **29 passed** |
| the two above + `test_v6_stage_init_introduction` + `test_v6_ladder_edges` + `test_v6_chain` + `test_loss_determinism` + `test_v6_t2_contrastive` + `test_v6_t5_consistency` + `test_runbook_commands` | **269 passed**, 95 s |
| **`stack` (full)** | **4084 passed, 7 skipped, 2 xfailed**, 561 s |
| **`taniteval` (full)** | **1136 passed**, 140 s |

⚠️ **The full `stack` suite was run TWICE, and only the second run counts.** The first
(4082 passed) was launched before the ordering fix below and before the two tests that pin it; the
second (**4084**, the same total +2) is the one that covers the shipped tree. Reporting the earlier
number would have been a green result for code that is not what shipped. *(Exit codes were not
trusted either way — both tails were read; the F-7 stream measured a `stack` run reporting 3
failures while exiting 0.)*

⚠️ **An untracked test from a concurrent stream (`stack/tests/test_build_parity_guard.py`) is
inside both totals.** It is not mine; it passes.

⭐ **A DEFECT IN MY OWN WIRING, caught by review and fixed in the same change.** F-9's curriculum
refresh was originally placed **after** `idx = sample(...)`, so every step drew under the
**previous** step's exponent and the final update was never used at all. ⛔ **It would not have
shown up in any log**: the alpha printed and the alpha drawn under are both "correct", one step
apart. Fixed, and pinned against the source by
`test_the_curriculum_refresh_precedes_the_draw_in_the_TRAIN_LOOP` — the same idiom F-8 uses to
assert a loss never reads `waypoints`. *(Exit codes are not evidence; nor are passing tests, when
no test was looking.)*

⭐ **Unlike F-7, `test_v6_stage_init_introduction.py` did NOT need extending** — its pin on
`STAGE_MAY_INTRODUCE` is exact, and neither cell adds an entry, so the pin passes unmodified. That
is the allowlist working as designed: a zero-key cell must not grow it.

---

## 9. Escalations — decisions this change does NOT make

1. ⛔ **The 8–30 s strategic horizon** (§0) — catalog/corpus decision. Options: re-extract longer
   clips from the **same** 2376 episodes (D4-admissible), accept K=4 (8 s) at a 64 % window cost,
   or amend the catalog row. **PI / diagram owner.**
2. ⛔ **`w_s1_multi`'s numeric value is a declared decision, not a default.** It is commensurate
   with `s1_latent` (both latent L1) — deliberately, since K=1 *is* `s1_latent` — but the
   trade-off against it is an experiment, not an invention. No value is pre-registered.
3. ⛔ **F-9's SCORE PRODUCER IS NOT BUILT HERE.** The artifact contract, its validation, the
   entropy functional and the curriculum all exist; the script that runs the P8 decoder over the
   corpus to *emit* the `.pt` does not. It needs (a) a matured P8 checkpoint and (b) the pod-side
   obstacle-join file — neither reachable from this box. **This is the "gated on P8 maturity" half,
   isolated so the ungated half could land.** ⚠️ P8's prior is `retention 0.932 at k=10` with
   **absolute IoU ≈ 0.02** (`OVERNIGHT_RESULTS_2026-08-12.md`); an entropy computed over a raster
   at that absolute quality is exactly what `t3_rank_control` must adjudicate before any T3 arm is
   quotable.
4. ⚠️ **Neither cell's TRAINER-SIDE F-9 path is execution-tested.** The score loader, the entropy,
   the curriculum and the sampler draw all are; the *in-`train()`* wiring (loader → curriculum →
   sampler swap) needs a corpus, which `--dry-run` does not mount. F-11's full path **is**
   execution-tested (§3).
5. ⚠️ **Neither cell has been trained.** Everything here is construction-, contract- and
   control-level evidence. **No claim is made that T3 or S1-multi improves anything.**
6. ⚠️ **F-11's compute cost when ON: K extra future-frame encodes per step** (one batched pass) plus
   K predictor rolls. Zero when off. Not benchmarked on Thor — Thor is training and off-limits.
7. ⛔ **F-9's OWN GATE ROW IS NOT COMPUTABLE TODAY — a separate instrument work item.** The T3
   row's gate is *"P7 calibration ρ ≥0.3 held on **interaction-rich strata, not just pooled**"*.
   P7 lives in `stack/scripts/w7_roll_rerank.py` (`P7_GATE_RHO = 0.3`) and **has no stratification
   support at all** — MEASURED, two probes: `grep strat` over that file returns zero, and a
   repo-wide sweep for a stratified P7 across `stack/` and `taniteval/` returns zero. ⇒ a per-stratum
   P7 read is a prerequisite for adjudicating any T3 arm, and it is not in this deliverable.
8. ⚠️ **`w_s1_multi` is not chain-enforced.** `v6_chain.assert_geometry_carry` enumerates its levers
   first-class and there is no `Step.s1_multi`. Because the cell adds **no keys**, this is a
   *reproducibility* gap, not a load-failure one — unlike `t2_head.`, an S-S run with the flag
   produces a checkpoint indistinguishable in shape from one without it. Same gap `agent_slots.`
   documents.

---

## 10. Deliverable manifest

**Everything below exists in the repo working tree AND is staged. Nothing lives only on a pod or in
a worktree. STAGED, NEVER PUSHED — no commit was made.**

| artifact | location | state |
|---|---|---|
| F-9 entropy, `T3Curriculum`, `t3_rank_control`, constants, `__all__` | `repo:stack/tanitad/models/v6.py` | MODIFIED |
| F-11 loss + reachability + persistence control; F-9 loader + curriculum wiring; weights, CLI, preflight, run-row provenance | `repo:stack/scripts/train_v6_staged.py` | MODIFIED |
| F-9 suite — 39 tests | `repo:stack/tests/test_v6_t3_curriculum.py` | NEW |
| F-11 suite — 28 tests | `repo:stack/tests/test_v6_s1_multitick.py` | NEW |
| this report | `repo:TanitAD Research Hub/…/incoming/2026-08-18-f9-f11-cells/F9_F11_CELLS.md` | NEW |

**Staging verified at the END of the turn, not when first staged** — three of these five were
edited *after* their first `git add` (the ordering fix and the two tests that pin it), and the index
does not follow. NEW files by `git ls-files --cached`; MODIFIED files by **blob comparison**
(`git ls-files --stage` vs `git hash-object`), because `--cached` answers "is this path in the
index?", which is `yes` for any tracked file including one whose index blob is the pre-edit version.
All five: **index blob == worktree blob**.

⚠️ **Foreign entries are in the index from concurrent streams** (`CLAUDE.md`,
`Project Steering/MODEL_REGISTRY.md`, `…/2026-08-18-monitor-fixes/*`). **Not mine, not touched.**
Whoever commits must apply the CLAUDE.md git-hygiene rule: check for foreign staged entries first,
and use `stack/scripts/scoped_commit.py`.
