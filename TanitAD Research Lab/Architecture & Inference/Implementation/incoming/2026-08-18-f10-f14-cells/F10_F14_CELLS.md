# F-10 (domain-diverse mix) BUILT · F-14 (sign/OCR target speed) REFUSED WITH THE ARITHMETIC

**2026-08-18** · branch `agent/arch-inf-20260803` · base HEAD `8cf49ab` ·
evidence class per claim · dev-box CPU only, **no training run, Thor untouched (PID 25477 live)**

---

## 0. Headline

**F-10 is built, measured, wired and pinned. The default build is UNCHANGED at 87,893,449 params /
405 keys** — MEASURED four ways through the *real* `build_stack_from_args` path (default, F-10 on,
F-10 control arm, F-10+F-11: **delta (0, 0) every time**, §3). The live tensor-strict v6F S-W resume
is untouched.

**F-14 is NOT built, and that is the finding.** Both derivation inputs its spec names are
unavailable on this corpus, one of them **forbidden rather than missing**, and the file the
conformance row names as *"the sign/OCR + corridor-prior source"* contains **no sign, no OCR and no
corridor content at all**. §2 is the arithmetic. Per the brief's rule, **no plausible substitute was
implemented** — and §2.5 says precisely which substitute was available and why wiring it would have
been the wrong answer.

---

### ⛔ FINDING 1 — THE OBVIOUS F-10 IMPLEMENTATION IS EXACTLY A NO-OP, AND IT IS SILENT

`InteractionSampler` draws **episodes uniformly** and consults its `weights` vector only *inside* the
already-chosen episode (`v6.py:548-557`; its own docstring: *"Episodes are drawn uniformly so no
episode is starved"*). A domain label is an **episode** property, hence constant within an episode —
and a constant vector through `torch.multinomial` is exactly uniform.

**MEASURED**, before a line of the cell was written — a domain-balanced per-window weight against an
all-ones weight, same seed, 4,000 draws:

| | domain-balanced per-window weight | all-ones weight |
|---|---|---|
| draw sequence | **bit-identical, element for element** | — |
| achieved domain share (target 0.500 / 0.500) | **0.675 / 0.325** | 0.675 / 0.325 |

0.675 is the **corpus proportion** (4 episodes of 6). ⇒ **Writing F-10 into `sample.weights` changes
nothing about the quantity F-10 exists to change, and no log would show it.** Same class as C115 (a
loss over `z_tac`'s non-existent temporal extent) — *a term over an invariant*. F-10 therefore
introduces its **own episode draw** (`StratifiedEpisodeSampler`), and `InteractionSampler` is
byte-unchanged.

⭐ **Independently re-derived.** A separate read-only survey of the sampler surface reached the same
conclusion from the source alone (*"a per-EPISODE stratum pushed through the existing per-window
weight vector is a no-op"*). Two independent derivations, one by execution and one by reading.
Pinned by `test_a_per_window_domain_weight_is_EXACTLY_a_noop_on_the_episode_mix`, which is paired
with `test_the_stratified_sampler_DOES_move_the_episode_mix` so "the no-op test passes" cannot also
be satisfied by a sampler that can move nothing.

### ⛔ FINDING 2 — THE DIVERSITY OBJECTIVE IS MAXIMAL AT **TWO OPPOSITE** DEGENERATE INPUTS

The brief's C119 warning lands, and harder than expected. A "perfect stratum balance" reading is
achieved, **at every temperature**, in both of these cases — and both are **exactly the uniform
draw**:

| stratification | balance gap (max share − min share) at τ=1 | actual draw |
|---|---|---|
| **ONE stratum** over N episodes | **0.000000** — perfect | uniform (q₁ = 1) |
| **EVERY EPISODE ITS OWN stratum** | **0.000000** — perfect | uniform (every nₖ = 1 ⇒ nₖ^(−τ) = 1) |
| a working 6-stratum mix | 0.000000 — perfect | balanced ✅ |

⇒ **No threshold on a balance metric can separate a working mix from either degenerate one.** Both
are therefore **refused**, not warned about, and the demonstration is in the suite
(`test_BOTH_degenerate_stratifications_score_PERFECTLY_BALANCED`). The instrument that *can* see them
is `report()`'s amplification and `n_eff_episodes`, which is why they are in the run row.

### ⭐ FINDING 3 — F-10's PRICE IS ENORMOUS ON THE ONLY STRATA THAT EXIST, AND IT IS AN ESCALATION

The catalog row's whole claim is *"geographic/domain diversity beats volume"*. `n_eff_episodes`
(`1/Σw²`) is **the volume**, and on the real corpus it is brutal. **MEASURED** on the only domain
strata that touch `physicalai-train-e438721ae894` — the aug120 `road_type` census, **201 of 2376
episodes = 8.46 %**:

| τ | max amplification | n_eff episodes | n_eff % | mass on the 201 labelled episodes |
|---|---|---|---|---|
| 0.0 | 1.00× | 2376.0 | 100.0 % | 8.4 % |
| 0.3 | 3.10× | 1991.7 | 83.8 % | 20.0 % |
| 0.5 | 6.01× | 1219.3 | 51.3 % | 33.2 % |
| 0.7 | 10.43× | 602.6 | 25.4 % | 50.2 % |
| **1.0** | **18.56×** | **243.2** | **10.2 %** | **75.0 %** |

⇒ **A fully balanced mix on today's strata spends 75 % of every batch on 8.46 % of the corpus and
discards 89.8 % of the effective volume.** That is not a reason to refuse the cell — it is the trade
the catalog row asserts is worth making, now with a number on both sides for the first time. **It is
a PI/data decision which τ is defensible, and it cannot be made without the strata artifact that
does not yet exist** (§5.1).

⚠️ **And the guards fire on the real data, correctly.** With the raw 4-way census kept intact
(`unclear` holds **2** episodes), `min_stratum_episodes` **REFUSES** — and it should: without the
guard that shape amplifies **237.6×**, collapses the effective corpus to **44.2 episodes (1.9 %)**
and puts **80 %** of every batch on the 201. Merging `unclear` into `unknown` is what makes the arm
legal at all.

---

## 1. The spec, quoted — two independent locations per cell

Established **before a line was written**, per *"absence found at ONE location is not absence"*.

### F-10 / catalog S3

| source | quote |
|---|---|
| `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:81` | *"S3 \| domain-stratified training mix (geographic/domain diversity beats volume — arXiv 2607.04500) \| the S1 scaling-ladder data-mix arm folds in here \| cross-domain P-battery deltas reported per stratum"* |
| `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:69` | *"domain-diverse mix (catalog S3) \| ⬜ NOT BUILT \| no domain-stratified sampling in `train()` (episode draw is uniform / O4-weighted only). Needs the VLM/scena strata as a SAMPLER input — which is admissible for the data MIX (it is not a model input) but must be declared. Fix F-10"* |
| `…/DIAGRAM_CONFORMANCE.md:215` | *"F-10 \| P3 \| S3 domain-stratified mix — VLM strata as SAMPLER input (admissible: data mix, not a model input; declare it)."* |

### F-14 / SPEED_BAND derivation

| source | quote |
|---|---|
| `…/2026-08-07-hierarchical-wm-redesign/HIERARCHY_VOCABULARY.md:87` | *"`SPEED_BAND` \| v_lo, v_hi \| LON axis — **SET BY THE TACTICAL LAYER (PI decision 2026-08-11): target speed is a tactical responsibility, computed from traffic-sign inputs (VLM/OCR speed-limit fields) and prior speed information (corridor speed statistics), bounded by the strategic layer's `REDUCE_TO` only as an upper envelope**"* — quoted verbatim in `stack/tanitad/models/v6.py:173-176` |
| `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:114` | *"**owns target speed (signs + priors)** \| 🟨 PARTIAL \| … Supervision/derivation ⬜ NOT BUILT: no sign/OCR prior enters any loss or label path in v6; the vision-derived speed-band work sits unintegrated in `…/2026-08-04-target-speed/code/vt_band_from_vision.py` (untracked incoming stream). Fix F-14"* |
| `…/DIAGRAM_CONFORMANCE.md:219` | *"F-14 \| P4 \| SPEED_BAND derivation: integrate the sign/OCR + corridor-prior source (`…/2026-08-04-target-speed/code/vt_band_from_vision.py`, currently untracked incoming) into the g_tac label path — coordinate with the target-speed stream owner."* |

⚠️ **A citation drift found in passing, not fixed silently.** `load_t3_scores`'s docstring (F-9,
`train_v6_staged.py`) cites *"the F-10 precedent (`DIAGRAM_CONFORMANCE.md:57`)"*. At HEAD, line 57 is
the **T4 roll-cost row**; the F-10 row is at **:69**. The rule F-9 invokes is real and is quoted
correctly — only the line number drifted. My own citations point at `:69`.

### What F-10's spec does NOT say — the assumptions, named

| # | gap | narrowest defensible reading taken | declared at |
|---|---|---|---|
| A1 | *"domain-stratified"* names no mixing rule | a **temperature** τ: stratum mass ∝ `n_k^(1−τ)`, so τ=0 is proportional (the control) and τ=1 is fully balanced. Neither extreme is privileged by fiat | `DomainMix` docstring |
| A2 | what to do with the unlabelled majority | **REFUSE**, never drop — dropping is a corpus re-selection, and an explicit `OTHER`/`unknown` stratum is a legitimate label the artifact may carry | `load_domain_strata` docstring |
| A3 | the join key | `stable_episode_id` **only**; the legacy 16-bit id is refused (it collides on **69 of 2400** train clips) | `load_domain_strata` docstring |
| A4 | whether the mix conflicts with O4/T3 | it does **not** — F-10 acts on the EPISODE axis, O4/T3 on the WINDOW axis, so they compose. This is why F-10 is *not* refused alongside `--o4-alpha` the way `--t3-scores` is | `DomainMix` docstring + `train()` block comment |
| A5 | *"diversity beats volume"* names no ceiling | an amplification cap and a minimum stratum size, both **calibrated by measurement** (§3), both overridable as declared decisions | the two constants' docstrings |

---

## 2. ⛔ F-14 — WHY IT IS NOT BUILDABLE, AND WHY I DID NOT BUILD SOMETHING ELSE

### 2.1 The sign/OCR half is FORBIDDEN, not missing

A speed-limit prior needs a detection's **`kind == "speed"`** and its **`text`** (the number). Those
are exactly the two fields the programme has ruled out.

| fact | source | evidence class |
|---|---|---|
| the sign channel is released **only as a presence flag** (per-clip 0.5; ≥0.70 per-detection, *not tuned*); ⛔ **"KIND and TEXT stay forbidden"** | `Project Steering/RETRACTION_LOG.md` C87 (:4499-4504) | MEASURED (theirs), re-read |
| **"G1's ~71 % is not withdrawn and G1 read its crops correctly. Sign KIND and TEXT stay forbidden"** | `…/2026-08-18-data-strategy-refresh/DATA_STRATEGY_REFRESH.md:134` | MEASURED (theirs) |
| the G1 sign-**text** gate is **CLOSED at 0/31**; `sign_text_status == pending_g1_gate` on **201/201**, so *"sign text stays extraction-only and never reaches a goal token"* | `Project Steering/MODEL_REGISTRY.md:3599`; `…/2026-08-17-w120val-sign-adjudication/W120VAL_SIGN_ADJUDICATION.md:256` | MEASURED (theirs) |
| a threshold **removes the harmless errors and keeps the harmful ones** — the two highest-scoring false positives are a **dashboard `30` roundel (0.927)** and a commercial hoarding (0.778), both **above** true signs | `W120VAL_SIGN_ADJUDICATION.md:29, 256` | MEASURED (theirs) |

⭐⭐ **AND THE WORST FALSE POSITIVE IS AN EGO ECHO WEARING A SIGN'S COSTUME — this is F-14's own
failure mode, not a generic caveat.** A **dashboard `30` roundel** is the ego's **speedometer**. A
sign-OCR-derived target speed on this corpus would, *at its highest-confidence detections*, be
reading the ego's own current speed off the instrument cluster and calling it a regulatory limit. The
brief's echo test has fired three times (route head 1.0000 as a nav bijection; lateral skill as an
action echo; a readout margin that was ego speed) — this would be the fourth, and it is the one that
**arrives through the vision channel**, where the binding vision-only rule looks like it is
satisfied. ⇒ **"Vision-only at inference" does not protect against an ego echo the camera can see.**
That generalisation is the reusable part of this finding.

### 2.2 The corridor half has no corridor

Two probes, per the absence rule:

1. `grep -rn "corridor_speed|speed_prior|corridor_stat|speed_limit" --include=*.py stack/ taniteval/`
   → **two hits, neither a corridor statistic**: `stack/scripts/vlm_route_labels.py:95` (a VLM
   enum listing `speed_limit` as a sign *type*) and one test fixture.
2. Repo-wide `speed_limit` over `*.md`/`*.json` → hits only in the **NuRec/OpenDRIVE map research**
   (`…/Research/2026-08-02-nurec-xodr-map/`, a *different* corpus — AlpaSim) and in VLM sidecars
   built against `physicalai-val-f1b378f295ae`, not the parity train corpus.

And the settled fact, asserted **in code** rather than only in the dataset card —
`taniteval/taniteval/corridor.py:223-224`: *"A **kinematic signature**, never a topology. There is no
map, no lane graph and no junction annotation in this corpus."* CLAUDE.md records the same at five
independent probes.

### 2.3 The named source file is misidentified — THREE MEASURED errors in one conformance row

`DIAGRAM_CONFORMANCE.md:114` and `:219` both rest on `…/2026-08-04-target-speed/code/vt_band_from_vision.py`.

| # | the row says | MEASURED | how |
|---|---|---|---|
| 1 | *"the **sign/OCR + corridor-prior** source"* | ⛔ the file contains **no sign, no OCR, no corridor and no speed-limit content whatsoever**. It is a **leave-one-episode-out linear readout** asking whether the pooled *image* feature predicts the VTARGET band better than repeating `v0`'s band | `grep -in "sign\|ocr\|corridor\|limit"` → **exit 1, zero matches** |
| 2 | *"currently **untracked** incoming"* | ⛔ it **is tracked**, added in `3039c6a` | `git ls-files --cached` lists it |
| 3 | (implied) that a vision-derived speed band exists to integrate | ⛔ **the probe has NEVER BEEN RUN** — its output exists nowhere | two probes: repo-wide `grep -rl vt_band_from_vision` over `*.json/*.md/*.py` returns only the conformance doc itself + one site-inventory listing; repo-wide `grep -rl identity_v0_band --include=*.json` returns **nothing**. It is also the ONE `code/` file absent from `TARGET_SPEED_LABEL.md` §11's own deliverable manifest |

### 2.4 The measurement that would decide F-14, and why it is not recoverable here

The question is sharp and already priced: **on ego inputs, nothing beats repeating the current
speed's band.** MEASURED, `…/2026-08-04-target-speed/raw/vt_four_families.json` (637 windows / 40
episodes):

| arm | band top-1 (23-token goal-setting) |
|---|---|
| **`hold_v0` — free, 0 params** | **0.4066** |
| `past_ridge` (LOEO ridge on the causal past) | 0.3673 |
| the CE band classifier's **argmax** — what a banded input consumes | **0.2465** |

⇒ *"`refc1`'s `speed_cls` … on ego-only inputs is a **dead parameter**"*. **A SPEED_BAND head must
clear 0.4066 from VISION or it is a dead parameter too**, and `vt_band_from_vision.py` is exactly
that measurement.

⚠️ **I attempted to run it and it is NOT recoverable from in-repo artifacts. MEASURED:**

| artifact | grid | n |
|---|---|---|
| `…/2026-08-03-dtac1-tactical-head/dtac1_substrate_refc-base-30k.pt` (`pooled[1364,704]`, `v0`, `eid`) | **stride 5** — `window_last_indices(T,5)`, l = 7, 12, 17, 22, … (35 windows at T=199) | 1364 windows / **39 episodes** |
| `…/2026-08-04-target-speed/raw/vt_labels_val40.json` (`vt_guarded`, `band_guarded`) | **stride 8** — l = 7, 15, 23, 31, … (22 windows at T=199) | 881 windows / **40 episodes** |

The two grids coincide only every 40 steps (LCM), and the substrate is missing one episode. The
`val40_poses_view.npz` that reconciles them lives on **`tanitad-thor:/tmp`** — a *scratch* path, on
the one host that is training and off-limits. ⇒ **the probe needs the poses view re-materialised; it
is ~2 minutes of CPU once it is.** Recipe in §5.3.

### 2.5 ⛔ THE SUBSTITUTE THAT WAS AVAILABLE, AND WHY IT WAS NOT BUILT

There **is** an admissible SPEED_BAND label: `stack/tanitad/lake/vtarget.py::vtarget_guarded`, banked
for **all 2376 parity-train episodes** (`…/2026-08-04-target-speed/raw/train_vtarget_guarded.npz`,
in repo, staged) and MEASURED **✅ ADMISSIBLE AS A LABEL** (`raw/vt_admissibility.json`). Wiring it
into `goal_head_tac` would have produced a plausible, green, well-tested "F-14".

**It would have been wrong, for a reason that is not stylistic:**

1. `vtarget_guarded` is **hindsight ego geometry** — the 85th percentile of the *ego's own future
   free-flow speed*. It answers *"what speed did this driver settle at"*, not *"what speed is
   permitted here"*. Those differ **exactly where this token matters**: a sign says 30 in free-flowing
   traffic; a settled speed says 30 because of the car in front. Substituting one for the other
   silently converts a **regulatory** prior into a **behavioural** one while keeping the name.
2. It is measured **⛔ INADMISSIBLE as a SUPPLIED input** (privileged increment ΔR² 0.0996) — only
   the **predicted** form is admissible — so the wiring is not a label join but a *new head plus a
   GPU-day*, which is
3. already an **open PI escalation**, filed by the target-speed stream itself
   (`TARGET_SPEED_LABEL.md` §10.1: *"the decision is whether to spend a GPU-day on the predicted
   form"*). An implementation stream may not answer that by building it.

⇒ **F-14 stays unbuilt. What ships instead is the blocker, recorded where the next implementer looks
and pinned so it cannot rot** (§4.2) — the stale-blocker class CLAUDE.md flags twice.

---

## 3. MEASURED — the numbers this deliverable rests on

**Parameter cost, MEASURED BY BUILDING** through `build_stack_from_args` (the real launch path),
never estimated:

| build | params | state_dict keys | delta |
|---|---|---|---|
| default (`--stage S-S`) | **87,893,449** | **405** | — |
| `--domain-strata …` (F-10 on) | 87,893,449 | 405 | **(0, 0)** |
| `--domain-strata … --domain-tau 0` (control arm) | 87,893,449 | 405 | **(0, 0)** |
| F-10 + F-11 (`--w-s1-multi 1.0 --s1-multi-k 4`) | 87,893,449 | 405 | **(0, 0)** |

⭐ **F-10 is structurally zero-parameter and that is pinned by construction, not re-checked per
release: a sampling mix is a schedule over the DATA.** Nothing in the cell is an `nn.Module`, and
`test_F10_is_not_a_LOSS_and_touches_no_weight` additionally asserts it never entered `V6LossWeights`
— an entry there would make it look like a term a stage could zero.

**The amplification ceiling's calibration** — MEASURED at N = 2376, τ = 1, four stratum shapes:

| shape | smallest stratum | amp @ τ=1 | n_eff | n_eff % | default 20× ceiling |
|---|---|---|---|---|---|
| 6-stratum realistic | 36 | 11.00× | 650.6 | 27.4 % | ✅ admits |
| 3-stratum coarse | 176 | 4.50× | 1030.1 | 43.4 % | ✅ admits |
| 10-stratum fine | 16 | 14.85× | 705.6 | 29.7 % | ✅ admits |
| **long-tail 12** | **8** | **24.75×** | 339.0 | 14.3 % | ⛔ **refuses** |

⇒ 20× admits full balance on every shape whose smallest stratum holds ≥16 episodes and refuses the
long-tail shape — the case the cap exists for. **The two guards are not redundant:** the long-tail
shape's smallest stratum is *exactly* 8, so it **passes** `min_stratum_episodes` and is caught only
by the ceiling (`test_min_stratum_and_amplification_are_NOT_redundant_guards`). The whole table is
re-derived in the suite rather than quoted from this document.

**The real-corpus census** (§0 Finding 3) — MEASURED against the aug120 `road_type` counts
(urban 129 · highway 38 · rural 32 · unclear 2 = 201).

---

## 4. What was built, file by file

### 4.1 F-10

**`stack/tanitad/models/v6.py`** (+ ~330 lines, **zero parameters**)

- `DomainMix` — frozen dataclass; `episode_weights(strata)`, `report(strata)`. Carries the two
  degenerate-stratification refusals, the unlabelled-episode refusal, the minimum-stratum-size
  refusal and the amplification ceiling.
- `domain_mix_control` + `DOMAIN_MIX_CONTROL_MIN_N = 32` — the trivial-proxy control, **refusing
  below n=32 and returning no ratio at all**, with per-stratum SEM and the named
  `UNINFORMATIVE` / `DEGENERATE_ONE_STRATUM` / `DEGENERATE_ZERO_WITHIN` / `UNDERPOWERED_STRATUM`
  verdicts.
- `StratifiedEpisodeSampler` — subclasses `InteractionSampler`, overriding **only** the episode
  `pick`. The window draw is inherited untouched, so F-10 composes with O4/T3.
- `DOMAIN_MIX_MAX_AMPLIFICATION = 20.0`, `DOMAIN_MIX_MIN_STRATUM_EPISODES = 8`; `__all__` extended.

⭐ **`InteractionSampler` was NOT edited.** Its contract *"episodes are drawn uniformly so no episode
is starved"* is a real guarantee other arms rely on, and F-10 is precisely the arm that breaks it —
so F-10 subclasses. That is F-7's lesson (do not edit a shared module to fit a new cell into it) and
T3's precedent (it carried its own weighting rather than weakening `saliency_weights`). Pinned
against the source by `test_the_shared_InteractionSampler_contract_was_NOT_weakened`.

**`stack/scripts/train_v6_staged.py`**

- `load_domain_strata` + `DOMAIN_STRATA_SCHEMA = "domain-strata-v1"` — the artifact loader and its
  seven refusals (unreadable JSON, wrong/missing schema, missing provenance, non-integer key, legacy
  16-bit key on either side, unlabelled episode, empty map).
- The `train()` block: load → mix → **sampler replacement**, placed **after** T3 (so it wraps
  whatever window weighting is in force) and **before** the T5 pair block (which mutates
  `sample.weights`). Order pinned against the source.
- CLI: `--domain-strata`, `--domain-tau`, `--domain-max-amp`, `--domain-min-stratum`.
- `preflight`: five refusals, all reachable **without mounting the corpus**.
- `config.json` gains `domain_mix` — the provenance stamp **and the price** (`n_eff_episodes`).

⚠️ **A defect in my own wiring, caught before it could fire and fixed in the same change.** The block
originally read `sample.weights` unguarded. With `--o4-alpha 0` (the O4 **control arm**) `sample` is
`make_sampler`'s plain **closure**, which has no `.weights` attribute — so F-10 would have
`AttributeError`ed on exactly the arm most likely to be run first, and only *after* the corpus
mounted. Now `getattr(sample, "weights", None)` with a uniform fallback, pinned against the source by
`test_the_F10_block_reads_the_window_weights_DEFENSIVELY`.

⛔ **PARITY IS UNTOUCHED.** F-10 **reweights the episode draw and removes no episode**: every weight
is strictly positive at τ ≤ 1, pinned at the weight level *and* at the **draw** level
(`test_the_rarest_stratums_episodes_are_actually_DRAWN_at_full_balance` — all 50 of 50 episodes
reached). An unlabelled episode is **refused**, never dropped.

⚠️ **RNG discipline.** With `--domain-strata` unset, nothing in the new block runs, `gen` is consumed
exactly as before, and every other term stays bit-for-bit. ⚠️ **There are TWO controls and they are
different things**: `--domain-strata` **absent** is byte-identical to the incumbent (`randint`);
`--domain-tau 0` is the **matched** control (same code path, same RNG shape, `multinomial`) and is
distributionally — not stream — identical. `train()` prints which one is in force, because a
stratum-share report cannot distinguish τ=0 from a live mix.

### 4.2 F-14 — the blocker, made durable

- **`stack/tanitad/models/v6.py`** — the `SPEED_BAND` vocabulary comment (which quotes the PI
  decision verbatim) now carries the measured blocker beneath it: both unavailable halves, the
  dashboard-roundel ego-echo mechanism, the substitute and why it is not this, and a pointer here.
  The vocabulary itself is **unchanged**.
- **`stack/tests/test_speed_band_derivation_blocker.py`** (NEW, 11 tests) — pins it so it cannot rot:
  the absence is **executable** (two independent paths, and the searcher strips comments first so the
  pin cannot match its own documentation — the self-matching-filter trap in a test's costume); the
  admissibility verdicts are read from the **banked artifact**, not re-quoted; the `hold_v0 = 0.4066`
  bar is asserted **against every other arm**, so if any arm ever beats it the pin fails and forces a
  re-read; and the three conformance-row errors are each pinned, including a **self-retiring** one
  that fails the moment somebody runs the vision probe.

---

## 5. Escalations — decisions this change does NOT make

1. ⛔ **THE F-10 STRATA ARTIFACT DOES NOT EXIST FOR THE PARITY CORPUS, and this is the blocker.**
   The only domain/weather/illumination strata that touch `physicalai-train-e438721ae894` cover
   **201 of 2376 episodes = 8.46 %** (the aug120 fused records' `semantics.scene` block; closed enums
   at `stack/scripts/ph0_v2.py:55-70`). The producer that assigns a domain to the other 2,175 is a
   separate work item — this is the **F-9 escalation-3 shape**: the contract, the validation, the
   mix, its price and its control all ship; the artifact does not. `preflight` says so by name when
   the path is missing.
   ⚠️ **And there is a second-best that is NOT admissible here:** `train_candidate_census.json` is the
   only all-2376 per-window stratum artifact, but its tags are **kinematic** (`brake`, `accel`,
   `sharp_turn`, `high_speed`) — that is **O4's axis**, so using it would be two levers on one axis,
   the exact conflation `--t3-scores` vs `--o4-alpha` already refuses.
2. ⛔ **`MODEL_REGISTRY.md:3563` (and `:3248`/`:3456`) are STALE about aug120's parity status** —
   they read *"NOT the training-parity corpus"*, superseded 2026-08-18 by
   `…/2026-08-18-alpamayo-parity-exclusion/ALPAMAYO_PARITY_EXCLUSION.md:16-19`, which measures the
   201 aug120 clips as **201/201 = 100 % INSIDE** the parity train corpus (set identity, not a
   coincidence of count). ⚠️ **Not edited here**: `MODEL_REGISTRY.md` is already `MM` in the index
   from a concurrent stream, and CLAUDE.md's git-hygiene rule forbids sweeping a sibling's work.
   **Escalated to the registry owner**, not left in a doc — this inverts the premise anyone building
   an F-10 artifact will start from.
3. ⛔ **F-14's decisive measurement is one CPU-run away and is blocked on one file.** Re-materialise
   `val40_poses_view.npz` (a poses-only view of `physicalai-val-0c5f7dac3b11`) on any host with the
   val cache, then run `vt_band_from_vision.py <substrate> <poses> <out.json>` — ~2 min CPU, no GPU.
   ⚠️ Note the two banked grids disagree (§2.4), so the poses view is genuinely required and cannot
   be reconstructed from `vt_labels_val40.json`. **The result decides whether SPEED_BAND has any
   deployable form at all**, and it is the cheapest discriminating experiment in this package.
4. ⛔ **DIAGRAM_CONFORMANCE's F-14 row needs rewriting** (§2.3): three measured errors. It should
   read that the sign/OCR half is **forbidden by C87 + the closed G1 text gate**, the corridor half
   has **no map**, and the open question is the **predicted vision band** — not an "integration".
5. ⛔ **τ's value is a declared decision, not a default.** `--domain-tau` defaults to 1.0 because that
   is the literal catalog reading, but §0 Finding 3 shows what τ=1 costs on today's strata. **No
   value is pre-registered**, and the trade-off is an experiment.
6. ⚠️ **F-10's in-`train()` path is not execution-tested end to end** — the loader, mix, sampler,
   control and their composition all are (`test_the_train_block_composition_runs_end_to_end`), but
   the block *inside* `train()` needs a mounted corpus, which `--dry-run` does not provide. Same
   position F-9's sampler path is in.
7. ⚠️ **F-10 has not been trained.** Everything here is construction-, contract- and control-level
   evidence. **No claim is made that a domain-stratified mix improves anything.**
8. ⚠️ **F-10 is not chain-enforced.** `v6_chain.assert_geometry_carry` enumerates levers first-class
   and there is no `Step.domain_mix`. Because the cell adds **no keys**, this is a *reproducibility*
   gap, not a load-failure one — the same gap `w_s1_multi` and `agent_slots.` document.
9. ⚠️ **`stack/tanitad/lake/curation.py` already holds `stratum_key` + `inverse_frequency_weights`
   and is DEAD** — imported by no trainer (two probes), and its own docstring says the scene axis
   *"reads `unknown` for now"*. F-10 does not use it: it is per-window inverse-frequency with a
   clamp, i.e. the shape §0 Finding 1 proves is a no-op on the episode mix. **Flagged rather than
   deleted** — that is the owner's call.

---

## 6. Suites

Run **separately** with `PYTHONUTF8=1`, on an otherwise idle box. ⛔ **Exit codes were not trusted —
the tails were read.**

| suite | result |
|---|---|
| `stack/tests/test_v6_domain_mix.py` (F-10) | **60 passed** |
| `stack/tests/test_speed_band_derivation_blocker.py` (F-14) | **11 passed** |
| the two above + `test_v6_t3_curriculum` + `test_v6_s1_multitick` + `test_v6_t2_contrastive` + `test_v6_t5_consistency` + `test_v6_stage_init_introduction` + `test_v6_ladder_edges` + `test_v6_chain` + `test_loss_determinism` + `test_v6_s2_loss` + `test_runbook_commands` | **381 passed**, 97 s |
| **`stack` (full)** | ⛔ **NOT OBTAINED — the repo's Drive mount went down mid-turn. §6.1.** |
| **`taniteval` (full)** | ⛔ **NOT OBTAINED — same cause. §6.1.** |

### ⛔ 6.1 THE FULL SUITES COULD NOT BE RUN — THE DRIVE MOUNT FAILED, AND EVERY FAILURE EXITED 0

⛔ **This is stated as a GAP, not smoothed over.** The targeted runs above are real and complete; the
two full-suite rows are **not obtained**, and no number is quoted for them.

**MEASURED 2026-08-18.** The repo lives on a Google Drive File Stream mount (`G:`), which entered a
sustained **content-fetch outage** during this turn: **directory listings kept working while every
file-content read failed** (`Invalid request code` / `OSError: [Errno 22] Invalid argument`).
`git rev-parse` reported *"not a git repository"* against a repo that was plainly there. Recovery
took ~250 s the first time and had not returned by the end of the turn the second time.

⭐ **AND EVERY ONE OF THOSE FAILURES EXITED 0 — a new instance of the exit-code trap, worse than the
known ones.** Three distinct shapes, all in one turn:

| # | where it failed | what the shell reported |
|---|---|---|
| 1 | `pytest_sessionfinish` writing `.pytest_cache` | **exit 0**, and the **pass/fail summary line was never printed at all** |
| 2 | `locate_config` reading `pyproject.toml` — the session never started | **exit 0**, output was a bare traceback |
| 3 | a full run that *did* start: **413 failed / 63 errors**, every one an `OSError: [Errno 22]` | a red suite that says nothing about the code |

⇒ The documented instances of this trap are *"N failures while exiting 0"*. **Shape 1 is ZERO
INFORMATION while exiting 0**, which is strictly harder to notice, because there is nothing to
disbelieve. And **shape 3 is the mirror image**: 413 failures that are *entirely* the mount and
*none* of them the code — a suite that would have been catastrophic to quote in either direction.

⇒ **Operational rules this earns on this box:**
1. **Run pytest with `-p no:cacheprovider`** — the cache write is on the Drive and is the first thing
   to fail. (The same write-flakiness made the file-writing tool return `EEXIST` on freshly created
   paths twice this turn; the report was drafted in the scratchpad and copied in.)
2. **Classify the failure signature before believing a red suite.** `--tb=line` and a `uniq -c` over
   the error lines separate "the code broke" from "the filesystem broke" in one command.
3. **Gate a long suite on mount stability first** (N consecutive successful content reads), because a
   suite that starts healthy and degrades halfway produces the most expensive artifact of all: a
   plausible-looking failure list.

⚠️ **The first full run was ALSO a torn snapshot** (C114) — launched before the last three edits of
this turn. It would not have counted even had it completed.

⚠️ **What the missing rows would have been measured against:** the concurrent stream's commit
`74fc23a` records the `stack` suite green at **4112**, and this turn adds **71** tests (60 + 11), so
the expected figure is **~4183**. ⛔ **That is an EXPECTATION, not a measurement, and it is labelled
as one.** Whoever next has a healthy mount should run both suites and record the tails.

⚠️ **Two of my own tests failed first and were fixed, not loosened** — a float32-vs-exact rounding
comparison at 1e-6 (widened to 1e-5 **with the reason in the test**), and two wrong assumptions about
repo internals (`MODULE_GROUPS` is a tuple of group *names*, not `(prefix, group)` pairs;
`_GROUP_PREFIXES` is a `V6Stack` class attribute, not module-level). Both were **my** errors, and
both are the kind a test written from memory rather than from the source produces.

⚠️ The `@pytest.mark.slow` build-parity test is **deselected by default**; it was run explicitly
(`-k default_build`) and **passes** — 87,893,449 / 405 through `build_stack_from_args`, F-10 off and
on. A slow marker that silently skips is a green suite that never checked the thing it exists for.

---

## 7. Deliverable manifest

⛔ **READ THE STAGING ROW FIRST. Everything below is IN THE REPO WORKING TREE. It is NOT STAGED, and
that is an environment failure, not a decision** — the Drive mount's content-fetch outage (§6.1) took
`.git` with it: `git rev-parse` reports *"not a git repository"* and `head .git/HEAD` returns
*"Invalid request code"*, so `git add` cannot run at all. Nothing lives only on a pod or in a
worktree; every artifact is on the repo disk with the content verified by md5 during the mount's
earlier recovery window.

**⇒ THE FIRST ACTION FOR WHOEVER PICKS THIS UP — stage the five paths, then verify (never trust the
exit code):**

```
git add -- stack/tanitad/models/v6.py stack/scripts/train_v6_staged.py \
           stack/tests/test_v6_domain_mix.py \
           stack/tests/test_speed_band_derivation_blocker.py \
           "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-f10-f14-cells/F10_F14_CELLS.md"

# NEW files -> presence is sufficient
git ls-files --cached stack/tests/test_v6_domain_mix.py \
                      stack/tests/test_speed_band_derivation_blocker.py
# MODIFIED files -> presence is NOT sufficient; compare the BLOBS
git ls-files --stage stack/tanitad/models/v6.py stack/scripts/train_v6_staged.py
git hash-object       stack/tanitad/models/v6.py stack/scripts/train_v6_staged.py
```

⚠️ **And re-run the two full suites first** (§6.1) — they were never obtained, so this work has not
had a whole-repo regression check.

⚠️ **`git add` can SILENTLY NO-OP on a file in a newly created directory** (CLAUDE.md), and
`F10_F14_CELLS.md` is exactly that case: its parent `2026-08-18-f10-f14-cells/` was created this
turn. It was written through a **fresh inode from Bash** (drafted in the scratchpad and `cp`-ed in,
because the file-writing tool hit `EEXIST` on the Drive twice) — which is the documented un-poison —
but verify it with `git ls-files --cached` regardless.

**File states below are working-tree states, NOT index states.**

| artifact | location | state |
|---|---|---|
| F-10 `DomainMix`, `StratifiedEpisodeSampler`, `domain_mix_control`, constants, `__all__` · F-14 `SPEED_BAND` blocker note | `repo:stack/tanitad/models/v6.py` | MODIFIED |
| F-10 `load_domain_strata`, `train()` wiring, CLI, preflight, run-row provenance | `repo:stack/scripts/train_v6_staged.py` | MODIFIED |
| F-10 suite — 60 tests | `repo:stack/tests/test_v6_domain_mix.py` | NEW |
| F-14 blocker pin — 11 tests | `repo:stack/tests/test_speed_band_derivation_blocker.py` | NEW |
| this report | `repo:TanitAD Research Hub/…/incoming/2026-08-18-f10-f14-cells/F10_F14_CELLS.md` | NEW |

⚠️ **NO `suite_tails.txt` is banked, deliberately** — the two full suites were never obtained (§6.1),
and banking a file of mount-failure tracebacks under that name would read as a suite result to
anyone who found it later. The targeted-run counts in §6 are the complete, honest record.

**Integrity of the four code artifacts, MEASURED during the mount's recovery window** (after the
first outage, before the second) — this is why "not staged" is a *bookkeeping* gap and not a
*content* one:

| file | lines | md5 (first 12) |
|---|---|---|
| `stack/tanitad/models/v6.py` | 5607 | `e4c56f9c5248` |
| `stack/scripts/train_v6_staged.py` | 5860 | `812a665c1188` |
| `stack/tests/test_v6_domain_mix.py` | 823 | `d3edb1fc16e3` |
| `stack/tests/test_speed_band_derivation_blocker.py` | 243 | `d3892cd262c9` |

In that same window `git status --short` reported exactly `M stack/scripts/train_v6_staged.py`,
`M stack/tanitad/models/v6.py`, `?? stack/tests/test_v6_domain_mix.py`,
`?? stack/tests/test_speed_band_derivation_blocker.py` — the expected shape, nothing missing.

⚠️ **THE REPO ADVANCED UNDER THIS TURN.** HEAD moved `8cf49ab` → `74fc23a` → **`a603936`** while the
work was in flight (a concurrent P7-per-stratum stream). **`git show --stat a603936` was checked and
it touches NONE of the five paths above** — no sweep occurred. ⭐ Incidentally that stream **closes
F-9's escalation #7** (*"P7 has no stratification support at all"*): `taniteval/taniteval/p7_strata.py`
now exists.

⚠️ **Foreign entries were in the index from concurrent streams** (`CLAUDE.md`,
`Project Steering/MODEL_REGISTRY.md`, `…/2026-08-18-monitor-fixes/*`). **Not mine, not touched.**
Whoever commits must apply the CLAUDE.md git-hygiene rule: check for foreign staged entries first,
and use `stack/scripts/scoped_commit.py`.
