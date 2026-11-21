# PRE-REGISTRATION — THE TWO ENCODER EXPERIMENTS (E-XENC-1, E-XENC-2)

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Origin** C104 (`Project Steering/RETRACTION_LOG.md:5452`)
**Tier doctrine** every number below carries **T0** (WM diagnostic) or **T1** (primary, action-closed loop).
**Estimator** paired episode-cluster bootstrap over the 40 val episodes (`taniteval/ci.py`). ⛔ **NEVER `overlapping_holdout_se`.**
**Ridge** `intercept_col=-1` on every `pc6` fit. ⛔ The default is the incumbent biased behaviour (C92) and is inadmissible for a new finding.

---

## READ THIS FIRST — four things landed, and one of them amends C104

1. ⛔ **`E-ENC` is already taken in code for a different experiment.** These are **E-XENC-1 /
   E-XENC-2**. §0.
2. ⭐ **E-XENC-1 is costed by BUILDING it**: **+1 065 984 total params (+0.317 %)**, and it
   **trains 85 514 496 FEWER** — the arm cannot win on capacity. A real forward was run. §2.3.
   ⛔ And the build found a live trap: `apply_stage_freeze` **un-freezes the foreign backbone**;
   86 580 480 foreign params would have trained silently. §2.4.
3. ⭐ **E-XENC-2's legality is settled from source, and it cuts both ways.** A **parameterised**
   distill head ⇒ **a fresh S-W from step 0 is the ONLY legal insertion** (order of magnitude).
   But a **relational, parameter-free** form introduces **no state-dict keys**, so it is legal
   warm-started via `--init-from` — a short arm, not a 30 000-step rung. §3.2–3.3.
4. ⛔⛔ **THE FALSIFIER BATTERY RAN, AND IT AMENDS C104.** At **identical width and identical
   feature count**, one-sub-frame DINOv2 keeps **95–99 %** of the headline gap (so the
   concatenation explains nothing), an **untrained** DINOv2 collapses **24–46×** (so pretraining
   does the work) — and **our own encoder's RANDOM INITIALISATION beats its TRAINED self 3.6×
   on both rungs, on all three seeds.** The objective is not failing to add geometry; it is
   **subtracting** it. §7.4.1, logged as **C106**.
   ⚠️ Read §7.5 before quoting: **neither of our arms passes K1** — the claim is about r², the
   metric C104 quoted.

> ## ⛔ CORRECTION 2026-08-18 (citation sweep) — ITEM 4 IS **HALF WITHDRAWN** BY C109. THE "3.6× ON BOTH RUNGS" MUST NOT BE QUOTED.
>
> **Cite this block by its heading, never by line number.**
> C106 was attacked five ways with refutation as the default posture
> (`…/incoming/2026-08-18-c106-adversarial/C106_ADVERSARIAL.md`; reproduction gate against the
> producing harness: `ego_v0` / `lead_gap` / `lead_closing` all Δ **0.00000**). **Half survived.**
>
> | C106's component | verdict | the measurement |
> |---|---|---|
> | ⭐ **`ego_v0` — random init reads better than trained** | ✅ **SURVIVES, and is now STRONGER because it finally has an estimator** | **Δr²c +0.150 [+0.055, +0.226], p(Δ>0) = 1.000** in C106's own cell — **paired episode-cluster bootstrap**, n = 1 507 / 1 362 windows in **70 clusters**, 2 000 draws — and **positive in 27/27** cells (3 init × 3 projection × 3 ridge seeds) |
> | ⛔ **`lead_gap` — the same claim** | ⛔ **DIES** | **0 of 27 cells CI-separated**, p(Δ>0) only **0.71–0.76**, and **the sign FLIPS in 9/27** |
> | ⛔ **"3.6×", the number itself** | ⛔ **WITHDRAW THE RATIO** | It compares a **near-constant** predictor (`pred_sd/gt_sd` **0.014**) to a **live** one (**0.89**), and re-drawing the **ridge inner split** — the seed C106 held fixed, worth **10× more variance** than the projection seed it did vary — moves it to **2.8× / 2.0×** |
> | ⚠️ **C106's bracket `[0.1736, 0.2011]`** | ⛔ **NOT AN INTERVAL** | It is the **PROJECTION-SEED SPREAD** — a dispersion quoted where a confidence interval was implied. *"Never quote an interval without its estimator"* (`CLAUDE.md`). |
> | ⚠️ **"our arm's readout is a flat line"** (§7.5 limit 2) | ⛔ **ONE INNER-SPLIT DRAW** | At `ridge_seed=2` the SAME arm chooses α=10 and reads `pred_sd/gt_sd` **0.93–0.95**, r²c **0.069–0.077** |
> | ✅ **the LayerScale mechanism** | ✅ **VERIFIED FROM THE WEIGHTS** | random-init residual fraction **0.0002**, cos(full, linear) **1.0000** — it **IS** the raw-pixel linear map; and the trained arm **has moved** (`ls` **70× init**, residual **0.38**), the half C106 asserted without checking |
>
> ⛔⛔ **THE FINDING THAT REFRAMES IT:** **our trained arm is NOT CI-separated from its own
> matched-random null** — `lead_gap` **0/9**, `ego_v0` **3/9 and not in C106's own cell**
> (Δ +0.052 [−0.0007, +0.206]) — **while the random arm IS (9/9, Δ +0.164 [+0.074, +0.367]).**
> ⇒ ⭐ **The admissible claim is SIGNAL vs NO-SIGNAL on ONE rung, not a ratio.**
> **A ratio whose denominator is not separated from noise is not interpretable**, however many
> seeds it holds across.
>
> ⚠️ **Two corrections that travel with this and bear on §7's plan:**
> * ⛔ **The step-0→9 250 sweep is UNAVAILABLE, not merely unretrieved** — a whole-filesystem probe
>   of Thor found **nothing before ≈ step 9 100**. It needs a **new run**, not a download. ⭐ But the
>   programme's own `z_op` ladder has a **step-2 000** point: `ego_v0` **0.1346 → 0.0801** and
>   `lead_gap` **0.0123 → 0.0059** by step 9 000, **then flat**, while `nearest_any` stays flat
>   throughout ⇒ **a real decline over 2 000→9 000 — on `ego_v0`, over that window, and NOT as a ratio.**
> * ⭐ **An attack that FAILED, reported because it failed:** widening α to **1e13** changed these rows
>   by ≤ **0.0008**. That **favours C106** and closes C107's alpha-grid concern **for these rows only**
>   — ⛔ **do not inherit it onto the 176 ladder rows, which are a different row set.**
> * ⛔ **`PC-2OBJ`, cited by both C104 and C106, is INERT AT THE DEPLOYED POOLING RATIO by
>   construction** (opposing plants in one cell cancel; at p40 it reproduced the un-planted arm to
>   **5e-05**). Cite **`PC-LOCAL` / `PC-DIST`** instead (**0.0596 → 1.0000, K1 9/9**).
>
> ⭐ **NEW AND MONITORABLE FROM STEP 0 — the one actionable thing this adds to the pre-registration:**
> the trained token field is **RANK-COLLAPSED** — **97.6 % of token-channel variance in ONE
> direction**, effective rank **1.22 against 67–68**; design-matrix rank **6.7 vs 16.4**. PCA-whitening
> lifts *both* arms ~3× and closes nothing ⇒ **a co-symptom, not the explanation** — but unlike *"the
> objective subtracts geometry"* it is **observable from step 0**, and the existing `z_op` spectrum
> monitor demonstrably does **not** cover it.
>
> ⏳ **OPEN, and it is this document's own item:** the frozen-external guard (§2.4) is built and pinned
> both ways (9 tests; catches the 86.6 M un-freeze and **cannot be satisfied by freezing everything**)
> but is **NOT YET CALLED** by `train_v6_staged.py`. ⛔ **It must be wired in the SAME change that
> introduces E-XENC-1**, or it is a guard that never runs — the `pod_git_drift.py` failure mode (C108)
> in advance.
>
> **Sources:** `Project Steering/RETRACTION_LOG.md` **C109** ·
> `…/incoming/2026-08-18-c106-adversarial/C106_ADVERSARIAL.md` ·
> `…/Benchmarks & Eval/Implementation/incoming/2026-08-18-citation-sweep/CITATION_SWEEP.md`.

---

## ⛔ 0. A NAMING COLLISION THAT MUST BE FIXED BEFORE THIS DOC IS CITED

**`E-ENC` IS ALREADY TAKEN, IN CODE, FOR A DIFFERENT EXPERIMENT.** MEASURED — it is
first-class in the stack, not prose:

| where | what `E-ENC` means there |
|---|---|
| `stack/tanitad/models/v6.py:2501` | *"E-ENC (the pre-registered arm, §0 Q1)"* — `shared_encoder` |
| `stack/tanitad/models/v6.py:4482` | `matched_param_config` — *"E-ENC helper"*, matched-total-param arm selection |
| `stack/tanitad/models/v6.py:4468` | *"E-ENC decides at MATCHED TOTAL PARAMS"* — shipped in every `param_report()` |
| `stack/scripts/train_v6_staged.py:3761` | `--per-layer-encoders`, *"E-ENC arm (b)"* |
| `stack/tests/test_v6_staged.py:118,130` | two tests named for it |
| the live checkpoint | `param_report.arm == "shared-encoder+adapters"` |

That `E-ENC` is **shared-encoder vs per-layer-encoders**. It has nothing to do with an
external backbone. Reusing the name would put two different experiments behind one string in
a programme whose registry rule exists because names drifted before.

⇒ **This document registers `E-XENC-1` and `E-XENC-2` (X = eXternal).** The brief's
`E-ENC-1`/`E-ENC-2` are their aliases and are recorded here once, so a reader of either name
lands in the same place. ⏳ **Escalated to the orchestrator** (§10): the brief's naming should
be corrected at source before it propagates.

---

## 1. WHAT THE EVIDENCE ACTUALLY INDICTS, AND WHAT IT DOES NOT

C104 measured, through the **same deployed `AvgPool2d((4,10))`**, on the **same windows**:

| rung | DINOv2-B/14 (86.6 M) | ours (87.3 M) | ratio |
|---|---|---|---|
| `lead_closing` r² | 0.01713 | 0.00000 | — |
| `lead_gap` r² | 0.44997 | 0.00496 | **91×** |
| `ego_v0` r² | 0.71733 | 0.05240 | 13.7× |

⛔ **But that comparison differed in FOUR ways, not one**, and only two of them are "the
encoder":

| axis | ours | the DINOv2 arm |
|---|---|---|
| pretraining | our S-W objective, 2 376 episodes | LVD-142M self-supervised |
| architecture | ViT-5 (RMSNorm/QK-Norm/LayerScale/RoPE), /16 | vanilla ViT, /14 |
| ⛔ **input format** | ONE 9-channel tensor (D-015 stack) | **THREE** 3-channel sub-frames |
| ⛔ **token width** | 768 | **2304** (3 × 768 concatenated) |

A 3-view concatenation makes inter-frame differences available to a **linear** readout by
construction. A single fused 9-channel patch conv does not. ⇒ *"the encoder is the
constraint"* was **one of four readings**, it is one night old, and it rests on a single
external encoder. §7 is the battery that separates them — and **§7 has already run**.

---

## 2. E-XENC-1 — THE FROZEN-EXTERNAL-ENCODER READOUT ARM

> **Question.** How much of the gap is recoverable by swapping the **encoder alone**, holding
> the predictor and the readout fixed?

### 2.1 The interface, established from source

| what the swap must satisfy | source |
|---|---|
| `V6Stack` reads exactly `.n_tokens` and `.grid_shape` off the encoder | `stack/tanitad/models/v6.py:3328-3332` |
| the readout's `out_dim` **must equal** `cfg.d_op` or the build RAISES | `stack/tanitad/models/v6.py:3333-3336` |
| the encoder contract is `[B,C,H,W] -> [B,N,D]` patch tokens, row-major | `stack/tanitad/models/encoder.py:97-110`, `:152-172` |
| `encode_window` is the ONE call the trunk makes; it does `readout(encoder(flat))` | `stack/tanitad/models/v6.py:3691-3708` |
| the geometry is declared by `EncoderConfig`, and a mismatched input is REFUSED | `stack/tanitad/config.py:16-46`; guard at `encoder.py:154-160` |
| encoders **and readouts** are the planner-protected set | `stack/tanitad/models/v6.py:3681-3686` |

⭐ **The live geometry makes the swap unusually clean.** From the live checkpoint's own
`v6_config` (MEASURED, not retyped): `in_channels 9`, `256×640`, `patch 16`, `d_model 768`,
`depth 12`, `n_heads 12`, `vit5_encoder true`, `n_registers 4`, `readout grid 4 × d_readout
128` ⇒ **`n_tokens` 640, grid (16,40), `d_op` 2048**. `facebook/dinov2-base` at **224×560**
tiles at patch 14 into **exactly 16×40 = 640** tokens at an identical 0.4000 aspect.

### 2.2 The arm, and the one deliberate design decision

```
frames [B,9,256,640]
  -> 3 × (3ch sub-frame -> resize 224×560 bilinear+antialias -> ImageNet norm)
  -> FROZEN facebook/dinov2-base -> drop CLS -> [B,640,768] each
  -> concat -> [B,640,2304]
  -> TRAINABLE Linear(2304 -> 768)          ⭐ 768 OUT, NOT 2304
  -> THE SAME SpatialGridReadout -> z_op [B,2048] -> THE SAME predictor
```

⭐ **The adapter emits 768 on purpose.** It makes the swap a **one-variable change**:
`n_tokens`, `d_model`, `grid_shape` and therefore `d_op` are all unchanged, so the readout,
the predictor, every uplink and every loss are shape-identical — and the readout can be
**warm-started from the live checkpoint**. An adapter emitting 2304 would have moved the
readout, the state width and the predictor in one edit: the `--v2` conflation this programme
refuses.

### 2.3 ⭐ THE PARAMETER DELTA — MEASURED BY BUILDING IT

**MEASURED (ours)** · `raw/eenc1_param_delta.json` · `code/eenc1_build_and_count.py`
Instantiated from the live checkpoint's own `v6_config`; every figure is `sum(p.numel())` over
an object that exists.

| | incumbent | E-XENC-1 | delta |
|---|---:|---:|---:|
| **total params** | **336 542 025** | **337 608 009** | **+1 065 984 (+0.317 %)** |
| `encoder` group | 87 284 736 | 88 350 720 | +1 065 984 |
| ⤷ of which **frozen** | 0 | **86 580 480** | — |
| ⤷ of which **trainable** | 87 284 736 | **1 770 240** | **−85 514 496** |
| within `param_budget` 350 000 000 | ✅ | ✅ (headroom 12 391 991) | — |

**Forward verified, not merely built:** `[1,9,256,640] → tokens [1,640,768] → z_op [1,2048]`,
`z_op_matches_d_op: true`, on the dev-box RTX 4060, `torch.cuda.max_memory_allocated()`
**0.405 GB** at batch 1 (⛔ the only admissible device-memory probe here).

⭐ **This arm CANNOT win on trainable capacity — it trains 85.5 M FEWER parameters.** For once
the C6 confound runs in the favourable direction, and that is worth stating before any result.

### 2.4 ⛔ A TRAP THIS BUILD FOUND, WHICH WOULD HAVE SILENTLY VOIDED THE ARM

**MEASURED** (`raw/eenc1_param_delta.json` → `s_w_freeze`): `apply_stage_freeze`
(`stack/tanitad/models/v6.py:3170`) sets `requires_grad` **from the group map**, and the
external backbone lives in group `encoder`, which S-W trains
(`STAGE_GROUPS["S-W"]`, `v6.py:3135`). So it **UN-FREEZES the foreign backbone**:

| | n_trainable |
|---|---:|
| incumbent under S-W freeze | 278 993 667 |
| swap, after `apply_stage_freeze` | **280 059 651** ⛔ backbone trainable |
| swap, after re-asserting the freeze | **193 479 171** |

⇒ **86 580 480 foreign parameters would have trained while the run called itself
"frozen external encoder".** The arm would not have been the arm it claimed to be, and
nothing in the ladder would have said so. ⇒ **RUNBOOK STEP: re-assert
`encoder.backbone.requires_grad_(False)` AFTER `apply_stage_freeze`, and assert
`n_trainable == 193 479 171` before step 1.** Same family as the `df` / Thor `free` traps: a
mechanism that reports the wrong scope.

### 2.5 ⛔ THE PRIOR IS NEGATIVE, AND IT IS IN OUR OWN REPO — REGISTER IT BEFORE RUNNING

**This programme has already built E-XENC-1 once, and it plateaued.** `REF-A` is
*"the frozen-encoder arm (H4)"* (`Project Steering/MODEL_REGISTRY.md:1894`) and its
`DinoGridAdapter` (`stack/tanitad/refs/refa.py:131`) states the design verbatim: it *"reuses
THAT class unchanged (grid=4, d_readout=128 → out_dim 2048, mirroring the main stack's state
geometry) **so the REF-A comparison isolates the ENCODER, not the readout**"* — which is
§2.2's design, built in July.

| REF-A arm | result | registry |
|---|---|---|
| `refa-4brain-speed-30k` — canonical frozen-DINOv2 reference | ADE@2s **2.1675 full-set** (2.1322 ± 0.1821 `heldout` 🟥) · **does not beat CV** · *"14 k = 2.05 → 30 k = 2.14 → plateaued"* | `MODEL_REGISTRY.md:1918-1929` |
| `refa-dynin-4brain-30k` — the H4 **final answer** | ADE@2s **3.0471 full-set** [2.4984, 3.6878] | `MODEL_REGISTRY.md:1946-1966` |
| status | ✅ **ACCEPTED AS REFERENCE (ceiling proven)** | `MODEL_REGISTRY.md:1920` |

⚠️ **Quote the `full_set` column.** Both rows publish a `heldout` split-mean from the
deprecated `overlapping_holdout_se`; the registry flags the cross-arm comparison of two
split-means as invalid at its own §4.1b.

⇒ **A pre-registered honest statement: the base rate for "swap in a frozen external encoder"
in this programme is a plateau.** What is different this time, stated in advance so it is not
invented afterwards:

1. **A different predictor and objective.** REF-A was a supervised head recipe on a pre-v6
   predictor; E-XENC-1 sits under the v6 S-W trunk and the label-free O1–O6 objective.
2. **A different level of evidence.** REF-A's ceiling was measured on *driving ADE*. C104/§7
   measure *what the tokens carry* — upstream of anything REF-A tested. The two are
   compatible: an encoder can carry geometry that the arm above it fails to use.
3. **Per-token, not per-grid.** REF-A's adapter pooled to the readout grid; §2.2's adapter is
   per-token (2304→768) and leaves the 16×40 grid intact.

⛔ **AND THE CRITERION THAT FOLLOWS: if E-XENC-1 lands at REF-A's plateau (~2.1 m full-set
ADE@2s, not beating CV), that is the DROPPED outcome and it was PREDICTED here.** It must not
be re-reported as a new finding.

### 2.6 Cost class

E-XENC-1 changes **which module produces tokens**, not the ladder. **A swapped encoder is a
different module, so the arm is a fresh S-W from step 0 regardless** of §3's allowance
analysis. Its cheapness relative to a full S-W is that the trainable set falls by 85.5 M and
the frozen backbone needs no gradients and no optimiser state.

⏳ **NO wall-clock claim is made for the arm.** For scale only, the *incumbent's* rate is
**MEASURED** — `…/incoming/2026-08-18-o2-live-and-ridge-reread/raw/v6F-SW-30k_train_log.jsonl`,
marginal over the last 100 steps (12 550→12 650): **26.3597 s/step** on the Thor; the same file
gives the A40 producer segment at **17.4604 s/step cumulative** and the width-matched A40
marginal is **20.46 s/step** (`…/incoming/2026-08-15-v6-thor-resume/THOR_VS_A40_TRAINING_SPEED.md:13`).
⇒ a 30 000-step S-W at Thor's marginal rate is **≈219.7 h ≈ 9.15 days**. ⛔ **Do NOT quote
"17.37 s/step"** — it is prose with no window or estimator (`STOP_2026-08-15_RESUME_RUNBOOK.md:16`),
and the doc that echoes it already flags it *"Not a first-call number"*.

⚠️ **A CLAUDE.md TRAP DOES NOT APPLY TO THIS TRAINER, AND ASSUMING IT DOES INVERTS THE READING.**
`step_s` in `train_v6_staged.py:3034-3039` is **ALREADY DIVIDED** — by `max(step - start_step, 1)`,
the steps *this process* ran — and the source comment names the `--log-every` accumulation as
*"the trap this avoids (the false '430 s/step' alarm)"*. So the real hazard here is the
opposite one: `step_s` is a **cumulative mean over the process**, inflated by startup after a
resume and blind to drift. ⇒ **quote MARGINALS computed between log rows, and segment by
producer** — the log is append-only across machines and step numbers OVERLAP (A40 ends 6 300,
Thor resumes 6 250), which is retraction class **C68** (`RETRACTION_LOG.md:3360`).

---

## 3. E-XENC-2 — DINOv2 TOKEN DISTILLATION AS AN `aux` LOSS

> **Question.** Does asking our encoder to reproduce a high-diversity encoder's token
> structure put the missing geometry into our tokens — without changing the deployed
> inference stack at all?

### 3.1 The target and the loss, stated precisely

**Teacher.** `facebook/dinov2-base`, **frozen**, eval mode, at **224×560** (patch 14 ⇒ the
identical 16×40 = 640 grid), **bilinear+antialias isotropic** resize (aspect asserted equal
to 6 decimals), **ImageNet mean/std**, CLS/register prefix dropped. Per-token features
`T ∈ R^{640×d_t}`.

**Student.** Our `ViT5Encoder` output `S ∈ R^{640×768}`, i.e. `self.encoder(flat)` at
`v6.py:3704` — **pre-readout, pre-pool**. That placement is the point: C104 showed the pool
is not the constraint, so the loss belongs where the deficit is.

⭐ **The `aux` group is the ONE non-trunk group permitted to backprop into the encoder, and it
is declared as data:** `ISOLATION_MATRIX` at `stack/tanitad/models/v6.py:2455`, the row
`"aux": ("encoder", "readout", "aux")` at **`:2459`**, measured by `V6Stack.assert_isolation`. ⇒ **placing this
loss in `aux` is legal by the X3 matrix as written — no matrix edit is needed.** ⚠️ Contrast
`interp`, which reaches nothing but itself precisely because its supervision is a perception
label. A DINOv2 token is **not** a perception label: it is a label-free function of the same
pixels the trunk already sees, which is why it belongs in `aux` beside `masked_cells`
(O3) and `sigreg` (O6) and not in `interp`.

⚠️ **A DECLARED TENSION, not hidden:** `stack/tanitad/models/sigreg.py:3-5` states SIGReg was
adopted specifically to avoid *"EMA / stop-gradient / teacher-student heuristics"*. E-XENC-2
adds exactly such a teacher. The defence is that the teacher is **frozen, external and
label-free** — it is not a self-distillation loop and cannot collapse — but the tension is
real and the `w_distill = 0` twin (§4.2) is what would detect it as harm.

⚠️ **Existing teacher machinery is NOT reusable here.** `_EmaCopy` (`v6.py:3217`, *"the V-JEPA
teacher pattern"*) is a weight-EMA of our own adapter, and the live run does not even use it
(`v6_config.uplink == "stopgrad"`). ⛔ **No loss term in `train_v6_staged.py` consumes an
external feature target today** — E-XENC-2 would be the first, which is why §5.2's
gradient-reach control is mandatory rather than a formality.

**Two loss forms, and they have DIFFERENT legality (§3.2 — this is the whole cost question):**

| form | definition | new state-dict keys? |
|---|---|---|
| **(P) projected** | `L = 1 − mean_i cos( W·S_i , T_i )`, `W ∈ R^{768×d_t}` a **trainable** linear head in group `aux` | ⛔ **YES** (`distill.weight`, `distill.bias`) |
| **(R) relational** | `G^S = Ŝ Ŝ^T`, `G^T = T̂ T̂^T` (row-L2-normalised, 640×640); `L = ‖G^S − G^T‖_F² / 640²` | ✅ **NO — zero parameters, zero buffers** |

⭐ **(R) is dimension-agnostic**, so it works between our 768 and the teacher's 768 or 2304
with no projection at all, and it is exactly the structure the ER10 evidence indicts: *which
patches belong to one object at what distance*. **(R) is the registered primary; (P) is the
declared secondary.**

**Weight.** A new `V6LossWeights` field `w_distill: float = 0.0` — **default 0.0 = OFF
everywhere, so the incumbent loss is bit-identical** and the live resume is untouched (the
pattern `w_select`/`w_anchor`/`w_s2_goal` already establish, `train_v6_staged.py:198-237`).
⛔ It must also be zeroed in `for_stage` for S-T and S-S (`train_v6_staged.py:239`), because
those stages do not train `encoder` and a term advertised in the launch line that trains
nothing is the exact lie that dataclass exists to prevent.

### 3.2 ⛔ WHICH STAGE MAY LEGALLY INTRODUCE IT — AND THE ANSWER CHANGES THE COST BY AN ORDER OF MAGNITUDE

**Established from source, not from the brief.**

```
STAGE_GROUPS            (stack/tanitad/models/v6.py:3134-3140)
  S-W: ("encoder", "readout", "predictor_op", "aux")     ← the ONLY early stage that trains them
  S-T: ("layer_tac", "planner")
  S-S: ("layer_str",)
  S-J: MODULE_GROUPS − LADDER_UNTRAINED_GROUPS           ← trains encoder/readout/aux too

STAGE_MAY_INTRODUCE     (stack/scripts/train_v6_staged.py:299-343)
  S-W: ()      S-T: ("cand_score.", "cond_tac_dyn.", "prop_diffusion.", "fallback.", "agent_slots.")
  S-S: ()      S-J: ()
```

The allowance is adjudicated in `load_stage_init` (`train_v6_staged.py:3419-3444`), which is
called for **every `--init-from`** (`:2625`). A resume takes the other path and is
`strict=True` with no allowance at all (`train_v6_staged.py:3341`).

| insertion path | trains `encoder`? | new keys admitted? | verdict |
|---|---|---|---|
| **resume into the LIVE S-W 30k** | yes | ⛔ no — `load_state_dict(..., strict=True)` `:3341` | ⛔ and **forbidden by standing order**: the live 30k is untouchable |
| **S-T** or **S-S** | ⛔ **no** — `encoder`/`readout`/`aux` are not in their `STAGE_GROUPS` | n/a | ⛔ **the loss would train nothing** |
| **S-J** | yes | ⛔ no — `STAGE_MAY_INTRODUCE["S-J"] == ()` `:342` | form **(R) only**; and S-J is a *brief joint polish* at the end of a ladder we have not reached |
| **fresh S-W with `--init-from <live ckpt>`** | yes | ⛔ no — `STAGE_MAY_INTRODUCE["S-W"] == ()` `:300` | ⭐ **form (R) only — and this is the cheap path** |
| **fresh S-W from step 0** | yes | ✅ yes — nothing to load, so no allowance is consulted | ✅ the **only** legal home for form **(P)** |

### ⭐⭐ 3.3 THE FINDING, STATED PLAINLY — IT CUTS *BOTH* WAYS

> ⛔ **For the PARAMETERISED form (P), the brief's suspicion is CONFIRMED: a fresh S-W run
> from step 0 is the ONLY legal insertion.** `STAGE_MAY_INTRODUCE["S-W"] == ()` and
> `["S-J"] == ()`, `encoder`/`readout`/`aux` are frozen in S-T and S-S, and a resume is a
> strict load. There is no path that adds a parameter to an existing checkpoint. **That is a
> full S-W run — an order of magnitude more expensive than E-XENC-1's premise.**

> ⭐ **But there IS a cheaper legal path the brief did not name, and it is the one to
> register: the RELATIONAL form (R) introduces NO state-dict keys, so it can be turned on in
> a fresh S-W run that `--init-from`s the live checkpoint.** The allowance list never fires,
> because nothing is missing. This warm-starts from a trained trunk and runs as a SHORT arm
> instead of a 30 000-step ladder rung.

The precedent is already in the source, one line above the allowance list itself: *"The MPC
refiner needs NO entry: it holds no parameters and no buffers, so flipping it changes no
state_dict key at all"* (`train_v6_staged.py:313-315`). Form (R) is that case.

⚠️ **Three conditions travel with the cheap path, or it is not cheap and not legal:**
1. A fixed random projection is **not** an escape hatch unless registered
   `persistent=False`; a persistent buffer is a state-dict key and is refused like any other.
2. `--init-from` starts a **NEW run at step 0** with a fresh schedule
   (`RESUME_CONTRACT["labelled"]`, `train_v6_staged.py:402-407`). It is not a continuation and
   must never be reported as one.
3. ⛔ **It still may not touch the live run.** A warm-start READS the checkpoint; the live
   30k keeps training untouched, and the arm needs its own `--out`.

### 3.4 The teacher's compute cost — MEASURED, and it decides the design

**MEASURED (ours)** on the dev-box RTX 4060, `raw/log_build_dino1f_s0.txt` /
`raw/log_build_dinorand_s0.txt`:

| teacher variant | rows | wall | throughput |
|---|---:|---:|---|
| DINOv2-B/14, **1 sub-frame**, 224×560 | 2 809 | **89.2 s** | **31.5 row/s** |
| DINOv2-B/14, **3 sub-frames** (C104's headline arm) | 2 809 | 295.4 s | 9.5 row/s |

⇒ **Register the ONE-SUB-FRAME teacher.** It is **3.3× cheaper**, and §7.2 measures that it
loses essentially nothing of what the teacher knows.

⛔ **Do NOT pre-cache the teacher for the training corpus.** A 640×2304 fp16 token map is
**2.95 MB per frame** (768-wide: 0.98 MB). At the corpus scale that is a **hundreds-of-GB**
artifact, and the `df`-trap sibling here is the MooseFS quota that has already killed a
flagship mid-checkpoint. ⇒ **teacher on the fly, one sub-frame, `torch.no_grad()`,
fp16 autocast** — and the per-step cost on the training device is a **work item to MEASURE
before launch**, not a number to estimate here.

---

## 4. KILL CRITERIA — COMMITTED BEFORE ANY MEASUREMENT

⚠️ **Why these are NO-HARM criteria and not trade-off clauses.** R1's flagship citation showed
its own isolated intervention **costing ~10 points on IN1K and SSv2 while gaining on dense
tasks**. A criterion that permits "it got worse at the objective but better at the probe" would
license exactly that, and the WM objective is the thing the programme is actually building.

### 4.1 E-XENC-1

| | criterion |
|---|---|
| **PROCEEDS** if | **(a)** `lead_gap` r² at the deployed pool ≥ **0.10** (T0), i.e. ≥ **20×** the incumbent 0.0049, **AND** the CI excludes the incumbent under the paired episode-cluster bootstrap on the same windows; **AND (b)** `lead_gap` **partial-r² after `v0` is accounted for** ≥ **0.03**; **AND (c) NO HARM on the WM objective**: `g_op_fwd_ade_m` (T0) and the T1 `ade_0_2s` are each **not worse** than the matched-step incumbent, CI containing zero or better. |
| **DROPPED** if | any of (a), (b), (c) fails at the matched step. |
| **INCONCLUSIVE** if | the positive control §5 does not fire — then nothing is read at all. |

⛔ **(b) is not optional.** `C-V0` measured ego speed *alone* beating **all four** ladder arms
on `lead_gap` (0.467 vs 0.075). Any latent-content claim states what it beats **after `v0`**.

### 4.2 E-XENC-2

| | criterion |
|---|---|
| **PROCEEDS** if | **(a)** at matched step, our own encoder's `lead_gap` r² (T0, deployed pool, `intercept_col=-1`) rises to ≥ **0.05** — i.e. ≥ **10×** the incumbent — with the paired CI excluding zero on **≥3 seeds**; **AND (b)** partial-r² after `v0` ≥ **0.015**; **AND (c) NO HARM**: `g_op_fwd_ade_m` (T0) and T1 `ade_0_2s` not worse than the `w_distill = 0` control trained for the same number of steps from the same init and the same seed. |
| **DROPPED** if | (c) fails — **regardless of how large (a) is**. A distillation loss that buys probe r² by degrading the world model has answered a different question. |
| **DROPPED** if | (a) and (b) fail while the control §5.2 fires — the loss reaches the encoder and does not move the geometry. |
| **INCONCLUSIVE** if | §5.2's gradient-reach control shows the term never reached `encoder.*`. |

⚠️ **The comparator is a `w_distill = 0` twin, not the live 30k run.** Different init, different
schedule, different step count — comparing to the live run would confound the loss with the
warm-start.

---

## 5. CONTROLS — A POSITIVE CONTROL **AND** A TRIVIAL-PROXY CONTROL PER EXPERIMENT

This is not optional, and it is precisely why E-R1-0's negative was admissible while D1's was
withdrawn (C79): `PC-2OBJ` proved the instrument could see the effect, and `C-V0` showed ego
speed alone beat all four arms.

### 5.1 E-XENC-1

| control | what it is | what it proves | fires if |
|---|---|---|---|
| **PC-2OBJ** (inherited, re-run) | two *opposing* planted objects inside one deployed cell | the ladder can see a pooling-destroyed signal | r² steps 0.0000 → ≈1.0 (banked: **0.9998**) |
| **C-V0** (trivial proxy) | ego speed **scalar alone** as the only feature | how much of any r² is just `v0` | banked `lead_gap` **0.467** — the bar every claim must clear |
| **C-ADAPTER-RAND** (new, required) | the **same** adapter over a **randomly-initialised** DINOv2 | the arm's gain is the pretrained weights, not the adapter or the plumbing | must **NOT** reach the PROCEEDS threshold |
| **C-FREEZE-ASSERT** (new, required) | assert `n_trainable == 193 479 171` after `apply_stage_freeze` | §2.4's silent un-freeze did not happen | equality holds |

### 5.2 E-XENC-2

| control | what it is | what it proves | fires if |
|---|---|---|---|
| **PC-DISTILL-ID** (positive) | run the loss with the **teacher replaced by the student's own detached tokens** | the term is wired, differentiable, and reaches `encoder.*` | `L → 0` within N steps **and** the measured gradient-reach set is exactly `encoder.*` |
| **PC-GRADREACH** (positive) | backprop `L` alone and enumerate parameters with non-`None` grad | the loss trains what it claims and nothing else | set == `encoder.*` (+ `distill.*` in form (P)); ⛔ the pattern of `tests/test_v6_s2_loss.py`, which does exactly this for `w_s2_goal` |
| **C-V0** (trivial proxy) | as above | any `lead_gap` gain survives partialling `v0` | partial-r² ≥ 0.015 |
| **C-SHUFFLE** (trivial proxy, new) | distil against a teacher whose tokens are **shuffled across the batch** | the gain is the *content* of the teacher, not the mere presence of a dense regulariser | must **NOT** reach the PROCEEDS threshold |

⛔ **C-SHUFFLE is the one that would otherwise be missed.** Any dense per-token target acts as
a regulariser; without it, "distillation works" and "an extra smoothness term works" are
indistinguishable.

---

## 6. PARITY, TIERS, AND THE FOUR METRIC FAMILIES

### 6.1 Parity — how each arm preserves it

**Canonical train corpus `physicalai-train-e438721ae894`, 2 376 episodes, skip-hash
`f09e44db`.** ⛔ Anything that re-selects episodes must be refused.

| arm | how parity is preserved |
|---|---|
| **E-XENC-1** | changes only the module that maps frames→tokens. **No dataset code is touched**: the corpus key, the episode list and the skip-hash are inputs to the data layer, which the swap never reaches. The run's `config.json` must carry the corpus key and the skip-hash, and a launch whose key differs is refused. |
| **E-XENC-2** | adds a **loss term** and (form P) one head. The teacher consumes the **same decoded frames the student sees**, inside the training step — it introduces no new data source, no new episode, and no new sampling. |
| **§7 falsifier battery (already run)** | the row set is **COPIED** from the banked v6 token cache — same clips, same frame indices, same order, same targets, same split — with **only `tokens` replaced**, and the `(clip_id, frame_idx)` sequence asserted equal. **NO episode is selected.** |

⚠️ The §7 battery reads a **130-episode window cache** (2 809 windows, 70 eval clusters), which
is the E-R1-0 probe set, **not** the 40-episode val set. It is a **T0 diagnostic on the probe
corpus** and is quotable as such and no further.

### 6.2 Tier stamps

| measurement | tier |
|---|---|
| linear readout r² on frozen latents (every number in §7 and §4's (a)/(b)) | **T0 — WM diagnostic. NEVER driving performance.** |
| `g_op_fwd_ade_m` | **T0** |
| `ade_0_2s` and every family in §6.3 | **T1 — PRIMARY**, `taniteval/tools/t1_eval.py` |

⛔ Comparisons across tiers are invalid. A registry row quoting §7 as evidence about driving
is malformed.

### 6.3 The four metric families — per family, never pooled

Binding (Sayed, 2026-08-02). **ADE stays; these are ADDED.** Each carries its own paired
episode-cluster CI on the same windows.

**⭐ THE INSTRUMENT'S ACTUAL STATE, ESTABLISHED FROM SOURCE — so no arm is planned against a
family that cannot be produced.** `t1_eval.py:499` delegates to
`taniteval/taniteval/four_families.py:1133 all_families(...)`.

| family | status **today** | what is computed / what is missing | source |
|---|---|---|---|
| **LATERAL** | ✅ **COMPUTED (all four axes)** | `heading_mae_deg`, `yaw_rate_mae_degps`, `curvature_mae_1pm` + bias, `cross_mae_m` / bias / final | `four_families.py:422 lateral()` |
| **LONGITUDINAL** | ⚠️ **PARTIAL** | ✅ `target_speed_acc`, speed MAE/bias/RMSE, along-track, `accel_mae`, `ego_progress`, anti-echo. ⛔ **headway / time-gap / min-TTC return `UNAVAILABLE` unless a `--lead` block is supplied** | `:230 longitudinal()`; `:343 _distance_keeping` returns `{"status":"UNAVAILABLE"}` at `:351`; block built by `taniteval/tools/build_lead_block.py` |
| **TACTICAL** | ⚠️ **PARTIAL** | ✅ manoeuvre-decision quality (factored lat-3 × lon-3 + collapsed legacy 5-way, κ). ⛔ **anchor/goal SELECTION is UNAVAILABLE on a trajectory dump** — *"an arm that commits to one path has no fan to score"* — closed only when the dump carries `<arm>_fan_err` / `<arm>_sel_idx` | `:948 tactical()` → `:776 tactical_from_trajectory()`; `:686 tactical_goal()`, `:691-695`; hook at `t1_eval.py:606-629` |
| **STRATEGIC** | ⛔ **ABSENT — and SETTLED** | `strategic_unavailable()`: no map, no lane graph, no route signal on PhysicalAI-AV. Returns `status`, `reason`, `n_windows_it_would_have_had`, `instrument_that_would_close_it`; marked `_settled` at five independent probes | `four_families.py:1018` → `:881`; stated at `t1_eval.py:485-487` |

⇒ **BINDING ON BOTH ARMS: `--lead` is MANDATORY.** ⛔ Without it the LONGITUDINAL family — the
family these experiments exist to move (§6.3 note) — silently reports `UNAVAILABLE`, and the
eval would be incomplete on exactly the axis under test. `build_lead_block.py` runs first or
the arm does not run.

⚠️ **STRATEGIC will report `UNAVAILABLE` with its reason and its n, and that is CORRECT, not a
gap in this prereg** — it is settled at five probes that the corpus carries no map or route.
It is reported, never dropped.

⚠️ **A missing family is a work item, not an excuse**, and ⛔ **a horizon sweep of ADE is never
"the result"** — it is one row of four. The binding string travels in every output
(`t1_eval.py:454-456`).

**Estimator entry points, named so the wrong one cannot be reached for:**
`paired_episode_cluster_bootstrap(a, b, eid, ...)` — `taniteval/taniteval/ci.py:261`, **the
decision-grade two-arm interval**; `episode_cluster_bootstrap(...)` — `:225`, single arm, point
estimate is the **`full_set`** value; `bootstrap_metrics(...)` — `:320`, one shared resampling
across a whole suite (what `t1_eval.py:518` calls).
⛔ `overlapping_holdout_se` — `ci.py:121`, named `_OLD_ESTIMATOR` at `:56`, **DEPRECATED,
anti-conservative, reproduction only. It must not appear in any number from these arms.**

⭐ **LONGITUDINAL is the family these experiments are actually about.** `lead_gap` and
`lead_closing` are the T0 shadows of headway and TTC; 88.7 % of the oracle gap is
longitudinal. If E-XENC-1 or E-XENC-2 moves anything, **the longitudinal family is where it
must show up at T1**, and a probe gain with a flat longitudinal family is a negative result
wearing a positive one's clothes.

---

## 7. ⭐ WHAT WOULD FALSIFY "THE ENCODER IS THE CONSTRAINT" — AND THE CHEAPEST CHECK, ALREADY RUN

The claim is one night old and rests on a single external encoder. §1 lists four axes on which
the comparison differed. Three cheap controls separate them — **all zero-training, same
windows, same rows, same deployed pool, same harness, `--proj-seeds 0 1 2`:**

| control | isolates | could overturn C104 by showing |
|---|---|---|
| **C-CONCAT** (`dino1f`) | the 3-view concatenation and the 3× token width | DINOv2 at **our exact width (768)** collapses toward ours |
| **C-PRETRAIN** (`dinorand`) | pretraining vs architecture+input-format | an **untrained** DINOv2 reads nearly as well as the pretrained one |
| **C-RANDENC** (`randenc`, 3 init seeds) | our objective vs our architecture | random-init **ours** ≈ DINOv2 (the gap is not pretraining at all), or random-init ours ≫ trained ours (**S-W REMOVED** readable geometry) |

⇒ **C-CONCAT is the cheapest of the three and the one that attacks the largest confound**, so
it is the answer to "name the cheapest check that could still overturn it".

### 7.1 Repro baseline — this harness, this command

**MEASURED (ours)** · `raw/fals_ours.json` · deployed pool `AvgPool2d((4,10))`, 1 302 train /
1 507 eval windows in **70 episode clusters**, 3 projection seeds, `intercept_col=-1`.

`ego_v0` **0.05207**, `lead_gap` **0.00490**, `lead_closing` **0.00000** — reproducing C104's
ours-column (0.05240 / 0.00496 / 0.00000) on the same rows. **The instrument agrees with the
producer, not merely with itself.**

### 7.2 ⛔ C-CONCAT — THE LARGEST CONFOUND IS MEASURED, AND IT EXPLAINS ESSENTIALLY NOTHING

**MEASURED (ours)** · `raw/fals_dino1f.json` · DINOv2-B/14 on **ONE** sub-frame ⇒ token width
**768** and **12 288 raw features — byte-for-byte the same shapes as ours.**

| rung | ours | **DINOv2, ONE sub-frame** | × vs ours | C104's 3-sub-frame headline | fraction of it recovered by ONE frame |
|---|---:|---:|---:|---:|---:|
| `ego_v0` | 0.05207 | **0.70593** | **13.6×** | 0.71733 | **98.4 %** |
| `lead_gap` | 0.00490 | **0.42743** | **87.2×** | 0.44997 | **95.0 %** |
| `lead_closing` | 0.00000 | **0.01707** | — | 0.01713 | **99.7 %** |

⇒ ⭐ **THE 3-VIEW CONCATENATION AND THE 3× TOKEN WIDTH EXPLAIN ESSENTIALLY NONE OF THE GAP.**
At **identical width and identical feature count**, a foreign frozen encoder still reads
`lead_gap` **87×** better than ours. **C-CONCAT fails to overturn C104 and materially
strengthens it**, because the headline no longer depends on the one axis that most obviously
favoured the teacher.

⚠️ **Two things this does NOT license.**
1. It says nothing about `dinorand`/`randenc` — **C-PRETRAIN and C-RANDENC remain live and are
   reported in §7.3.** Until they land, "the encoder is the constraint" is better supported but
   still not isolated to *pretraining*.
2. ⚠️ **DINOv2 reads `ego_v0` at r² 0.706 from a SINGLE FRAME.** Ego speed is not, on this
   corpus, a purely multi-frame quantity — appearance carries it. Any future argument of the
   form *"only a temporal model can read speed"* is refuted by this row.

### 7.3 C-PRETRAIN and C-RANDENC — THE READING RULE, REGISTERED BEFORE THE NUMBERS

⛔ **This table was written and committed to the file BEFORE `dinorand` and `randenc` returned**
(`raw/fals_dino1f.json` timestamped ahead of `raw/fals_randenc_s*.json`), precisely so the
outcome could not be rationalised after the fact:

| if | then |
|---|---|
| `randenc` ≈ `ours` (both ≈ 0) and `dinorand` ≪ `dino1f` | C104's reading **SURVIVES and sharpens**: the constraint is the **objective**, and pretrained weights carry the geometry |
| `randenc` ≫ `ours` | ⛔ **S-W actively REMOVED linearly-readable geometry** — a different and much sharper claim, and a **RETRACTION of C104's phrasing** |
| `randenc` ≈ `dino1f` | ⛔ **C104's headline reading is REFUTED**: the gap is not pretraining but architecture/input format |
| `dinorand` ≈ `dino1f` | ⛔ pretraining is not the lever; the **vanilla-ViT + 3-channel + ImageNet-norm** stack is |

### 7.4 ⭐⭐ THE RESULT — ROW 1 AND **ROW 2** BOTH FIRE

**MEASURED (ours)** · `raw/falsifier_summary.json` · every arm through the **same** command, the
**same** rows, the **same** deployed `AvgPool2d((4,10))`, `--proj-seeds 0 1 2`,
`intercept_col=-1`, 1 302 train / 1 507 eval windows in **70 episode clusters**.

| arm | token width | raw features | `ego_v0` r² | `lead_gap` r² | `lead_closing` r² | K1 PASS |
|---|---:|---:|---:|---:|---:|---:|
| **ours** — trained S-W @11 250 | 768 | 12 288 | 0.05207 | 0.00490 | 0.00000 | **0/3** |
| `dino1f` — DINOv2 **pretrained**, 1 frame | 768 | 12 288 | **0.70593** | **0.42743** | 0.01707 | **3/3** |
| `dinorand` — DINOv2 arch, **untrained**, 3 frames | 2304 | 36 864 | 0.02660 | 0.01843 | 0.00037 | 0/3 |
| **`randenc_s0`** — **OUR arch, RANDOM INIT** | 768 | 12 288 | **0.20110** | **0.01597** | 0.00000 | 0/3 |
| `randenc_s1` | 768 | 12 288 | 0.17360 | 0.01667 | 0.00000 | 0/3 |
| `randenc_s2` | 768 | 12 288 | 0.19357 | 0.02030 | 0.00000 | 0/3 |

**C-PRETRAIN fires as predicted by row 1.** Removing *only* the pretrained weights collapses
the teacher: `ego_v0` **0.71733 → 0.02660 (27×)**, `lead_gap` **0.44997 → 0.01843 (24×)**,
`lead_closing` **0.01713 → 0.00037 (46×)**. ⇒ **Pretraining does essentially all the work;
architecture, input format and 3× width buy almost none of it.**

> ### ⛔⛔ 7.4.1 BUT ROW 2 ALSO FIRES — AND IT RETRACTS C104's PHRASING
>
> **Our own encoder's RANDOM INITIALISATION beats our TRAINED encoder on BOTH rungs, on ALL
> THREE SEEDS**, at identical width and identical feature count:
>
> | rung | trained S-W @11 250 | random init (3 seeds, mean [min, max]) | ratio |
> |---|---:|---:|---:|
> | `ego_v0` | 0.05207 | **0.1894 [0.1736, 0.2011]** | **3.6×** |
> | `lead_gap` | 0.00490 | **0.0176 [0.0160, 0.0203]** | **3.6×** |
>
> ⇒ **C104's *"the encoder/objective is the constraint"* is too weak and mis-placed. On C104's
> own metric, S-W training moved the representation from its random initialisation to
> something ~3.6× WORSE. The objective is not failing to ADD linear geometry — it is
> SUBTRACTING it.**

⭐ **AND THE MECHANISM MAKES `randenc` THE HONEST RAW-PIXEL CONTROL.** `ViT5Encoder` uses
LayerScale with `ls_init = 1e-5` (`stack/tanitad/models/encoder.py:303`, `:307`, `:312`), and
the residual branches are `x + ls * f(x)` (`:315-316`). **At initialisation every one of the 12
blocks contributes ~1e-5 of its output**, so a random-init `ViT5Encoder` is approximately
`RMSNorm(patch_conv(x) + pos)` — **a fixed random LINEAR map of the raw patch pixels.**

⇒ The correct statement of the finding is therefore the strongest and simplest one:
**raw pixels through a random linear map read `ego_v0` and `lead_gap` ~3.6× better than our
trained encoder's tokens do.**

This also explains `dinorand`, which would otherwise look inconsistent: vanilla ViT has **no**
LayerScale, so an untrained one is a genuinely deep random network, and deep random networks
destroy linear readability (`ego_v0` 0.027 — below even ours). Near-linear random map: 0.19.
Deep random net: 0.027. Pretrained: 0.71. The ordering is coherent.

### 7.5 ⚠️ WHAT THIS DOES **NOT** SAY — THREE LIMITS, STATED BEFORE THE FINDING TRAVELS

1. ⛔ **NEITHER OUR ARM PASSES K1 — ONLY DINOv2 DOES.** `K1 PASS` (does the readout beat the
   constant in MAE?) is **0/3 for ours, 0/3 for `randenc`, 0/3 for `dinorand`, 3/3 for
   `dino1f`.** So `randenc`'s advantage is **on r² — the metric C104 quoted — and does not
   amount to beating a constant.** The admissible claim is *"on C104's own metric our trained
   encoder ranks below its own random initialisation"*, **not** *"the random encoder works"*.
2. ⚠️ **Our arm's predictions are nearly FLAT and that is part of the story.**
   `pred_sd / gt_sd` is **0.0141–0.0222** for ours against **0.89–0.92** for `randenc` and
   0.58–0.77 for `dino1f`, and the ridge chose `alpha = 1e7` **at the grid edge**
   (`raw/fals_ours.json`). Under §8's own layer-3 screen (`SD_RATIO_FLAT_FLOOR = 0.05`) **our
   arm is itself a flat line.** The regulariser winning is the honest description.
3. ⚠️ **ONE CHECKPOINT, at step 11 250 of 30 000 (37.5 %).** A single step cannot distinguish
   *"training subtracts"* from *"11 250 is a transient". **§7.6 tests exactly that.**

### 7.6 The step trend — is readability FALLING over S-W training? **NO — IT WAS ALREADY GONE**

Registered reading, written before the numbers:

| if | then |
|---|---|
| readability **falls monotonically** across the window | §7.4.1's *"training subtracts"* is confirmed as a **trajectory** ⇒ retraction-grade |
| readability is **flat or rising** | §7.4.1 stands only as *"the trained arm sits below its own init"*, and the subtraction happened **earlier** than the window |
| the trend is **non-monotonic / mixed** | ⛔ **INCONCLUSIVE**; do **not** fit an exponent — CLAUDE.md forbids quoting one below R² 0.80 and forbids extrapolating >2× beyond the fitted range |

⚠️ **A DEAD END, RECORDED SO IT IS NOT RE-TRIED.** The first attempt used the five banked
**cell** caches (steps 2 000 / 9 000 / 9 250 / 10 000 / 11 250) with `--arms cells`. All five
returned rc=1: *"⛔ this cache banked no tokens"* (`raw/log_trend_s*.txt`) — `er10_pool_ladder.py`
derives cells **from tokens** and those caches bank cells only. ⇒ the trend had to be
**re-encoded** from checkpoints, and only **three** are local (9 250 / 10 000 / 11 250).

#### ⭐ THE GATE PASSES FIRST (C94 — agree with the PRODUCER, not with yourself)

`trained@11250`, re-encoded from the fp16 checkpoint through a fresh `ViT5Encoder` with a
verified strict weight load, reproduces the **banked** token cache:

| | banked (`fals_ours.json`) | re-encoded (`trend_trained_011250.json`) | Δ |
|---|---:|---:|---:|
| `ego_v0` | 0.05207 | 0.05197 | 0.00010 |
| `lead_gap` | 0.00490 | 0.00490 | **0.00000** |
| `lead_closing` | 0.00000 | 0.00000 | 0.00000 |

⇒ the `trained` mode is validated; the other two steps are readable.

#### The result

**MEASURED (ours)** · `raw/trend_trained_*.json` · same 2 809 rows, same deployed pool, 3 seeds.

| step | `ego_v0` r² | spread | `lead_gap` r² | spread | `lead_closing` r² | K1 | `pred_sd/gt_sd` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **step 0** (`randenc`, 3 seeds) | **0.1894** | 0.0275 | **0.0176** | 0.0043 | 0.00000 | 0/9 | 0.89–0.92 |
| 9 250 | 0.05573 | 0.00140 | 0.00470 | 0.00000 | 0.00000 | 0/3 | 0.0153 |
| 10 000 | 0.05453 | 0.00080 | 0.00477 | 0.00010 | 0.00000 | 0/3 | 0.0150 |
| 11 250 | 0.05197 | 0.00110 | 0.00490 | 0.00000 | 0.00000 | 0/3 | 0.0147 |

⇒ **BRANCH 2 + BRANCH 3. Within the window the two rungs move in OPPOSITE directions**
(`ego_v0` falls 0.0557→0.0520, `lead_gap` rises 0.0047→0.0049), and both movements are **~2 %
of the gap to step 0**. ⛔ **Reported as INCONCLUSIVE as a trajectory, and NO exponent is fit** —
the window is 2 000 steps, **6.7 % of the run**, and the direction is not even shared.

> ⭐ **THE SUBSTANTIVE READING: THE 3.6× DEFICIT DID NOT ACCUMULATE HERE — IT WAS ALREADY FULLY
> ESTABLISHED BY STEP 9 250.** Linear readability is not draining away slowly; it is **gone
> early and then stable**. §7.4.1 therefore stands as *"the trained arm sits far below its own
> initialisation"*, and the mechanism is an **early-training** one.

⚠️ **AND THAT CHANGES E-XENC-2's DESIGN, WHICH IS WHY THE NEGATIVE WAS WORTH THE COMPUTE.** If
the loss of readability happens before step 9 250, a distillation term **warm-started late has
little left to preserve** — it would be repairing after the fact rather than preventing.
⇒ **§3.3's "cheap path" (form (R) warm-started via `--init-from`) is now the WEAKER arm on
mechanistic grounds, not merely the cheaper one**, and the pre-registration must carry that:
if the short warm-started arm fails, that is **NOT** evidence against distillation from step 0.
⏳ The decisive missing measurement is a **step-0-to-9 250 sweep**, which needs checkpoints that
are not local — an artifact-retrieval work item, not a compute one.

---

## 8. ⚠️ THE C97 GUARD HOLE — MEASURED IN BOTH DIRECTIONS, CONSTANT **NOT** CHANGED

`SD_RATIO_FLAT_FLOOR = 0.05` (`taniteval/taniteval/degeneracy.py:144`) does not flag C97's own
headline case. **Reproduced from the bank** (`raw/c97_floor_blast_radius.json`, 580 banked
screen rows): the matched-random **null** on `n_agents_all` has `sd_ratio = 0.091089` and
`K1_PASSES: true` **on all five arms** (p40/p10/p4/p1/cells) — and `0.091 > 0.05`, so
`flat_line` is False and it is not screened.

**MEASURED sweep — what each candidate floor would do, in BOTH directions:**

| floor | flat rows /580 | newly flagged | **of which currently claim a K1 PASS** | un-flagged |
|---|---:|---:|---:|---:|
| 0.02 | 296 | 0 | 0 | 16 |
| **0.05 (incumbent)** | **312** | — | — | — |
| 0.07 | 325 | 13 | **13** | 0 |
| 0.091 | 325 | 13 | **13** | 0 |
| **0.10** | 330 | 18 | **18** | 0 |
| 0.12–0.15 | 331 | 19 | **19** | 0 |
| 0.20 | 339 | 27 | 22 | 0 |

⭐⭐ **THE FLOOR IS THE WRONG INSTRUMENT, NOT A MIS-SET ONE — and that is the finding.**
The null's case sits at **0.0911**, *above* the rows it would sweep up: the 18 rows newly
flagged at 0.10 are dominated by **`n_agents_grid` in our own real arms** (`er10_main.json`),
at `sd_ratio` **0.050–0.055**. ⇒ **No floor can flag the null's PASS without first screening
out our own arms' passes**, because the null is on the *high* side of them. And note the
off-by-one that would have been shipped: the test is strict `<`, so a floor of exactly
**0.091 does not flag a ratio of 0.091** — catching it needs ≥ 0.10, which costs all 18.

⛔ **THE CONSTANT IS DELIBERATELY NOT CHANGED.** `degeneracy.py:141-144` calls it *"the one
tunable in the module"*, and C95/C97 is the pair that established that **every correction to a
criterion is a candidate bias in the opposite direction** — this programme built a
rejects-everything guard and a passes-everything guard **within one day**. Moving a threshold
to catch one case, when the sweep shows it costs 18 claimed passes and still cannot separate
the case from our real arms, is that mistake made a third time.

⇒ **RECOMMENDATION (PI decision, §10).** Do not move the floor. **A matched-random NULL that
claims a K1 PASS is itself the diagnostic**, and the module already has the right instrument
for it: `k1_guard`'s **layer 2**. Require layer 2 for any quoted K1 PASS, and record in
`degeneracy.py` that the layer-3 floor **provably cannot** separate this case — with this
sweep as the evidence.

⚠️ **Unchanged and still standing:** our `n_agents_all` K1 PASS is **not quotable as
latent-attributable** until layer 2 runs.

---

## 9. ⏳ DINOv3 — A PI ITEM, NOT A WORKAROUND

`facebook/dinov3-vitb16-pretrain-lvd1689m` is **`gated: manual`** and our token gets **403**
on the weights while the metadata is public — three probes, `…/2026-08-18-pooling-ladder-ER10/
raw/dino_availability.json`. **A human must accept the licence on huggingface.co.**

⛔ **No mirror was used and no gate was bypassed**, and none should be.

⭐ **Why it is worth the PI's minute.** DINOv3 is **/16**, so it reproduces our 16×40 grid
*exactly* with **no resize at all** — removing the last geometric difference between the arms
— and LVD-1689M is the high-diversity corpus the discriminator was specified for. Every §7
number would become strictly stronger evidence. ⚠️ Until then, §7's right-column verdicts may
be quoted **only** as *"…to DINOv2-B/14"*.

---

## 10. ⏳ ESCALATED FOR INTEGRATION — NOT LEFT IN A DOC

Per the operating standard, these need a decision or a merge and are named in the report
headline, not buried here.

1. ⛔ **The `E-ENC` name collision (§0).** Two different experiments behind one string, one of
   them already first-class in `v6.py`, `train_v6_staged.py` and two tests. **Fix the brief at
   source before it propagates.**
2. ⛔ **The `apply_stage_freeze` un-freeze (§2.4)** is a **live trap for any frozen-external
   arm**, not a quirk of this prereg. It wants a guard in the trainer or a test, not a runbook
   line: MEASURED, 86 580 480 foreign params would train silently.
3. ⏳ **E-XENC-2 form (P) needs a `STAGE_MAY_INTRODUCE["S-W"]` decision** if it is ever to run
   warm-started. Registering form (R) as primary avoids it; do not relax the allowance quietly.
4. ⏳ **The C97 recommendation (§8)** — a PI decision on layer-2-required vs a floor change,
   with the sweep as evidence.
5. ⏳ **The DINOv3 licence acceptance (§9)** — a human action.
6. ⚠️ **A registry correction (§11.2)** — C104's live-model param count is off by 17 280.
7. ⭐⭐ **THE `BASELINE-OF-SELF` RULE (§11, C106) IS PROGRAMME-WIDE, NOT LOCAL TO THIS PREREG.**
   *Whenever an arm is compared to a foreign model, also compare it to ITSELF AT
   INITIALISATION.* It is one forward pass, matched by construction, and it is the only control
   that can detect a training objective making a representation **worse**. ⏳ It belongs in
   `AGENT_OPERATING_STANDARD.md`'s controls guidance, not only in a retraction entry — **that
   is an orchestrator decision, and it is why this is escalated rather than left here.**
8. ⏳ **A step-0-to-9 250 checkpoint sweep (§7.6)** — the decisive follow-up, blocked on
   checkpoints that are not local. **Artifact retrieval, not compute.**

---

## 11. CORRECTIONS THIS WORK PRODUCES — LOGGED AS **C106**

⭐ **Appended to `Project Steering/RETRACTION_LOG.md` as C106**, with the root-cause class, not
just the correction. The class:

> **A COMPARISON THAT DIFFERED IN FOUR WAYS, READ AS EVIDENCE ABOUT ONE OF THEM — AND THE
> MISSING CONTROL WAS THE CHEAPEST ONE IN THE BUILDING: THE MODEL'S OWN INITIALISATION.**
> C104 had a positive control and a trivial-proxy control. What it lacked was a
> **BASELINE-OF-SELF**. ⭐ **RULE: whenever an arm is compared to a foreign model, also compare
> it to ITSELF AT INITIALISATION.** One forward pass; width- and format-matched by
> construction; and the **only** control that can detect a training objective making a
> representation *worse* — which no external comparison can, because "the foreign model is
> better" is equally consistent with ours merely being weaker.

### 11.1 The C104 headline gains two controls it did not have
§7.2 removes the concatenation/width confound and §7.4 removes the pretraining-vs-architecture
ambiguity. **Both are strengthenings, not retractions**, and C104's table should cite
`raw/fals_dino1f.json` and `raw/fals_dinorand.json` beside it.

### 11.1b ⛔ And C104's *phrasing* is amended
*"The encoder/objective is the constraint"* is too weak and located one level too high. The
measured statement is **"the objective is SUBTRACTING linear geometry"** — §7.4.1 — with the
three limits of §7.5 attached.

### 11.2 ⚠️ A param-count correction — MEASURED, doubly-sourced
C104 states the live checkpoint is **336 559 305** params. **Two independent sources say
336 542 025**: the checkpoint's OWN recorded `_meta.config.param_report.total`, and a fresh
`V6Stack` instantiated from that same checkpoint's `v6_config` (`raw/eenc1_param_delta.json`,
`incumbent.matches_recorded: true`). **Δ = 17 280.**
⇒ The **12.2 % over the "Sub-300M" headline** conclusion is **unaffected** (336.5 M is still
12.2 % over 300 M), so C104's substantive point stands and only the digits need fixing.

---

## 12. WHAT IS **NOT** CLAIMED HERE

- ⛔ **No wall-clock or GPU-hour figure** for either arm on Thor or an A40. Nothing was
  measured on the training device, and an estimate would be exactly the INHERITED-deciding-a-
  GPU-day violation the standard forbids.
- ⛔ **No T1 number.** Everything measured in this document is **T0**. Neither experiment has
  been run.
- ⛔ **No claim that DINOv2 would make a better driver.** A linear-probe r² is a T0 diagnostic;
  the four families at T1 are what would settle it, and §4's no-harm clause exists precisely
  because a probe gain is not a driving gain.
- ⚠️ **No claim about DINOv3**, whose weights we cannot read.
- ⚠️ **The §7 battery is on the 130-episode probe corpus (2 809 windows, 70 clusters)**, not
  the 40-episode val set.
- ⚠️ **No exponent, anywhere.** §7.6's window is 6.7 % of the run and its two rungs move in
  opposite directions; it is reported INCONCLUSIVE as a trajectory.

---

## 13. SUITES AND DELIVERABLE MANIFEST

**Suites — MEASURED here, not inherited** (`stack_suite.txt`; taniteval run inline):

| suite | result |
|---|---|
| `stack` | **3868 passed, 7 skipped, 2 xfailed, 0 failed** (611 s) |
| `taniteval` | **1136 passed, 0 failed** (248 s) |

⚠️ The brief quoted `stack` **3842+**; the actual count today is **3868**. Reported rather than
inherited, as instructed. ⛔ **Exit codes are not evidence** — the first `stack` invocation was
backgrounded through a pipe and left a **0-byte** output file while exiting 0; it was re-run
writing to a real file, and the line above is from that file.

**Every artifact and where it lives.** Base
`repo:TanitAD Research Hub/Architecture & Inference/Research/2026-08-18-encoder-experiments/`.
⭐ **Nothing lives in only one place** — every finding is in the repo and staged; the scratchpad
holds only regenerable intermediates.

| artifact | path | what it is |
|---|---|---|
| **this pre-registration** | `PREREG_ENCODER_EXPERIMENTS.md` | the deliverable |
| retraction entry | `repo:Project Steering/RETRACTION_LOG.md` **C106** (appended) | the correction + root-cause class |
| E-XENC-1 param delta | `raw/eenc1_param_delta.json` | MEASURED by instantiation + a real forward |
| ⤷ builder | `code/eenc1_build_and_count.py` | includes `ExternalTokenEncoder`, the swap itself |
| falsifier collation | `raw/falsifier_summary.json` | the §7.4 table |
| ⤷ per-arm ladders | `raw/fals_{ours,dino1f,dinorand,randenc_s0,randenc_s1,randenc_s2}.json` | + `raw/log_fals_*.txt` |
| ⤷ cache builder | `code/eenc_falsifier_cache.py` | modes `ours/dino1f/dinorand/randenc/trained/dino3f` |
| ⤷ chain | `code/chain_falsifier.sh` | `ours` runs FIRST as the repro gate |
| ⤷ collator | `code/summarise_falsifier.py` | |
| step trend | `raw/trend_trained_{009250,010000,011250}.json` | + `raw/log_trend_trained_*.txt` |
| ⤷ chain | `code/chain_steptrend.sh` | gate-first |
| ⤷ **the dead end, kept on purpose** | `raw/log_trend_s{10000,11250}.txt` | *"⛔ this cache banked no tokens"* — so it is not re-tried |
| C97 floor sweep | `raw/c97_floor_blast_radius.json` | + `code/c97_floor_blast_radius.py` |
| suite output | `scratchpad:eenc/stack_suite.txt` | ⚠️ **regenerable**; the numbers are in the table above |
| token caches, row index | `scratchpad:eenc/` | ⚠️ **deliberately NOT banked** — 2.8–8.3 GB each, fully regenerable from the chains |

⛔ **STAGED, NEVER COMMITTED, NEVER PUSHED.** No branch was switched. `Project Steering/Mission
Plan.md` was not touched.

⚠️ **Nothing was installed** — `transformers 5.15.0` and `facebook/dinov2-base` were already
present, and `torch 2.11.0+cu128` was verified with a **real CUDA `conv2d`** (not an import)
before any GPU work.

⛔ **Thor was not touched.** All compute ran on the dev-box RTX 4060; peak
`torch.cuda.max_memory_allocated()` **0.92 GB**. The live 30k run is untouched and no parameter
of it was changed.
