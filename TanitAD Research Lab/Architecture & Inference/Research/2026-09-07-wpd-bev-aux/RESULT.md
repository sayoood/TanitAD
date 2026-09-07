# WP-D — a BEV auxiliary loss that puts AGENTS into REF-C's map

**Date** 2026-09-07 · **Owner** Architecture & Inference FlyWheel · **Branch** `agent/arch-inf-20260803`
**Evidence class** MEASURED (ours), artifact named per number, unless a line says otherwise.
**Compute** ⛔ **ZERO GPU.** Every number below was produced on CPU. Nothing was added to the A40
(training refcv5-v2), to Thor (training refav1), or to the dev-box 4060 (occupied by an
inference-seed measurement). Nothing was pulled from HuggingFace.

⛔ **NO EVAL TIER ON ANYTHING HERE, AND THAT IS NOT AN OMISSION.** No model in this package emits a
trajectory. This is a target/instrument delivery plus a corpus census, so a T0/T1 stamp and a
four-family table would be category errors (`EVAL_DOCTRINE.md` binds *capability* claims). The eval
tiers and the four families bind the **arms**, and they are pre-registered in
`Project Steering/PREREG_WPD_BEV_AUX.md` §5B.

---

## 0. The answer, in four lines

1. ⭐ **The aux head supervises a POLAR agent-occupancy map — 24 range bins × 20 azimuth columns —
   registered COLUMN-TO-COLUMN onto REF-C's own 8×20 ResNet map**, built from the B1
   `obstacle.offline` join the trainer already loads. It is **training-only** and its removal is
   **BIT-IDENTICAL** at the planner's output — MEASURED, not asserted.
2. ⭐⭐ **The third state is REAL and large, and that is the load-bearing measurement of this
   package.** On the full B1 EVAL join (26,394 frames / 905,512 boxes): **17.656 %** of cells are
   agent-occluded, and **27.958 %** of GT-OCCUPIED cells are — so a two-state target **asserts that
   more than a quarter of the agents in the grid are free road**. WP-D masks them, and the two-state
   variant ships as a named deliberate-regression arm.
3. ⭐ **The gate can go RED: 6 of 6 mutants killed**, each re-introducing a defect this programme
   actually suffered — and one of them (`M5`) was a **REAL defect in this module**, found by its own
   control on the first run.
4. ⛔ **Nothing was trained.** The GPU arms are specified, costed at a **MEASURED 4.0 s/step**, and
   the cheapest discriminating cut is **13.3 h on one card**. The blocker is provision, not design.

---

## 1. What exactly does the head supervise, and from which artifact

| | |
|---|---|
| **artifact** | `C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz`, md5 **`3ddb42ecbd3926066795a94587af2aed`** — the B1 EVAL `obstacle.offline` join. Read here **directly** through `lzma` + `json` rather than through `train_p8_occupancy.JoinFileReader`: a second read path is a second probe. Its own totals reproduce exactly: **139 clips · 26,394 labelled frames · 905,512 boxes**. |
| **train-side twin** | `C:\Users\Admin\tanitad-caches\b1-train-join-20260906\b1train_agents.jsonl.xz` (**4,427/4,572 clips · 849,263 rows · 28,053,187 boxes**, md5 `1c985e6d6ad34e605c4ebd30cb353558`) — INHERITED from the DataFlyWheel's package, not re-hashed here; the arms use `joins/train2400_agents.jsonl.xz`, already on the A40 |
| **target** | polar **[24 range × 20 azimuth]** = 480 cells: 2.5 m bins to 60 m (`bev_raster.GRID_DEFAULT.x_fwd_m`), 6.0°/column over the rig's true **120°** field, ego frame **+x forward, +y LEFT** |
| **predicate** | **exactly** `bev_raster.rasterize`'s oriented-rectangle test, re-used rather than re-written; a cell is occupied iff any of its `sub_r × sub_az = 6` sub-samples is inside a footprint |
| **head** | `Conv2d(F→64, 1)`, then per azimuth column the whole elevation stack `[64·gh] → 256 → 24`. **182,616 parameters** at REF-C-base, on its own `param_breakdown["bev_aux"]` line and subtractable — the DEPLOYED count is unchanged |
| **loss** | masked `BCEWithLogits`, `pos_weight` a pre-registered **CONSTANT 30.61** = `(1−p)/p` at the MEASURED base rate `p = 0.031634` ⛔ never computed from the batch |

### 1.1 Why POLAR, and not the Cartesian [120, 64] raster

Two reasons, one MEASURED and one geometric.

* **MEASURED (WP-A §6.1):** mapping the Cartesian 0.5 m grid into the encoder's token grid puts a
  **median of 4 BEV cells (max 313)** into one token cell, and only **242 of 640** token cells
  receive any ground-plane cell at all. A dense Cartesian BEV indexed off an image-token grid
  inherits a quantisation floor no training removes.
* **The corpus is CYLINDRICAL** — the image column is **linear in azimuth** — so a polar target is
  registered to the encoder's own columns with **zero resampling**. ⛔ `BEVAuxHead` **REFUSES**
  `gw != n_az` rather than interpolating: a silent resample keeps the loss curve healthy while
  mis-registering every agent.

⚠️ **Scope, or this travels wrong:** column-linear-in-azimuth is a property of **our** 256×640
`f_ref 305.5775` corpus, not of BEV lifting. The pinhole formula gives **92.641°** on the same data
and looks entirely plausible — which is why it is one of the six mutants (§3), where it silently
deletes every agent between 46.32° and 60° of bearing.

⚠️ **And it must NOT be quoted as a v6/v7 fact.** REF-C has **no `SpatialGridReadout` at all**; the
"4 readout columns / 30°/bin" figure is a v6/v7 number (WP-A §1.2). refcv5's map is **8×20**.

---

## 2. ⭐⭐ THE THIRD STATE — MEASURED, and it decides the design

`code/s1_third_state_census.py` · `raw/third_state_census.json` · CPU, **16.7 s**, **all 26,394
labelled frames**, spec 24×20.

### 2.1 There is no occlusion label anywhere in what we hold

⛔ **The join's per-agent `occ` column IS the camera FIELD-OF-VIEW mask, not an occlusion flag** —
MEASURED bit-identical to `bev_raster.fov_mask`'s predicate, **0 of 7,680 cells disagreeing at every
half-angle tried** (`build_obstacle_join.py` `P4_PREDICATE_IDENTITY`,
`stack/tests/test_p4_fov_predicate.py`, which fails if a twin is added). And `obstacle.offline`
itself carries none: `bev_raster.agents_at_time` says *"occ = -1 always"*.

⇒ **Object-object occlusion must be DERIVED, and it is derivable** from what the join does carry —
ego-frame footprints. A ray from the ego origin to a cell either does or does not cross a nearer
footprint. **No `z` is needed**, and none is available: the join schema drops it.

### 2.2 The cost of each option, as numbers

| option | cost | what it teaches |
|---|---|---|
| **(a) MASK the shadow** ⭐ **CHOSEN, default** | **17.656 %** of cells unsupervised (per frame median **11.667 %**, **p90 45.833 %**); **27.958 %** of positives deleted (**128,070 / 458,083**) | nothing wrong — it says less |
| **(b) TWO-STATE** ⛔ *the deliberate regression* | free | ⛔ **asserts that 27.958 % of the occupied cells are FREE ROAD** |
| **(c) THREE-CLASS** | a third logit; a metric no longer comparable to any occupancy AP; the head must also predict *visibility*, mixing a scene fact with a camera fact | more — but it changes what is measured before the first thing is known to work |

⇒ **(a).** (b) ships as `--bev-aux-occlusion none` so the regression the gate must catch is runnable,
and detecting it is a committed criterion (PREREG §5D). (c) is out of scope and named as the
successor.

⚠️ **The derived mask is a LOWER BOUND**, stated three ways rather than hidden: it is **2-D** (a low
`stroller` shadows a `bus`); the shadow starts at a **cell boundary**, quantised to 2.5 m; and **only
labelled agents occlude** — `obstacle.offline`'s 10 classes are all **dynamic**, so buildings and
walls cast nothing and a real urban scene is under-shadowed.

⭐ **OUT-OF-FIELD does not arise at all** in this space — every polar column is inside the 120° field
by construction, while the Cartesian raster loses **590 of 7,680** cells to it. Pinned by
`test_no_cell_of_the_polar_grid_is_out_of_field`.

⚠️ **NO_LABEL is a fourth state and is kept apart from labelled-clear by the MASK, not by the
occupancy array** — both read `occ.sum() == 0`, which is exactly why a check on occupancy alone
would be permanently green. **2.239 %** of B1 EVAL frames are labelled-clear and are fully
supervised; a frame absent from the join contributes nothing.

### 2.3 The numbers the controls must be read against

| quantity | value | note |
|---|---|---|
| base rate over SUPERVISED cells | **0.031634** | the literal the constant control must read |
| ⛔ all-zero predictor's ACCURACY | **96.837 %** | ⇒ **no headline number in this package is an accuracy** |
| `pos_weight` = (1−p)/p | **30.61** | the pre-registered constant, stamped in `config.json` |
| two-state base rate (all cells) | 0.036157 | what arm **D4** would optimise instead |
| boxes per labelled frame | 34.307 | |
| **Cartesian in-field control** | **0.018382** | ⚠️ a **different geometry**, computed on the same frames as an order-of-magnitude cross-check — and it lands beside WP-A's **1.62 % / 1.74 %**, which is corroboration, not replication |

---

## 3. ⛔ THE MUTATION PROOF — the gate CAN go RED. 6 of 6 killed.

`code/s2_mutation_proof.py` · `raw/mutation_proof.json`. Each mutant applies ONE textual edit that
re-introduces a defect this programme actually suffered, runs the WHOLE suite, and records the
deaths. The edit is asserted to have applied **exactly once** — a mutation that silently did not
apply produces a green run indistinguishable from a surviving mutant.

**Baseline GREEN (36 passed). 6 / 6 mutants KILLED.**

| mutant | the real defect it restores | tests killed |
|---|---|---|
| **M1** mirrored world (`+y` sign) | *"a sign error here does not crash and does not show in a loss curve, it teaches a MIRRORED world"* — `E-DEC-18`'s build measured the convention for this reason | **3** |
| **M2** pinhole FOV 92.641° | the retracted 2026-08-21 optics error. It silently deletes **every agent between 46.32° and 60°** of bearing — the near-lateral band where cut-ins live | **6** |
| **M3** two-state merge | WP-A's flag; §2 measures it at **27.958 %** of positives mislabelled | **2** |
| **M4** NO_LABEL read as clear road | the join's own named defect | **1** |
| **M5** tie-blind AP | ⭐ **a REAL defect in this module** (§4) | **2** |
| **M6** head not constructed last | the one-variable violation the removability proof exists to prevent | **2** |

⚠️ **The first M6 SURVIVED, and correctly.** It inserted an *unconditional* RNG draw, which shifts
both arms equally and is therefore **not** the defect; it had to be made **arm-conditional** to be
the real one. ⭐ A surviving mutant is as often information about the mutant as about the suite, and
recording that is cheaper than re-learning it.

### 3.1 ⭐ The three load-bearing ones are now in the REPO'S OWN standing instrument

`stack/scripts/guard_mutation_audit.py` already exists — *"the only evidence in the repo that the
M18 provenance guards are load-bearing rather than decorative"* — with a registry whose anchors are
themselves pinned by `stack/tests/test_guard_mutation_audit.py`. Leaving WP-D's proof in a
package-local script would have been a second instrument for the same job, so the three load-bearing
mutants are **registered there** (`bev_mirrored_world`, `bev_two_state_merge`,
`bev_head_not_constructed_last`), and `tests/test_bev_aux.py` was added to its `TEST_FILES` — an
audit's verdicts are only as wide as the files it runs, and a defect caught by a suite it never
invokes reads as `ESCAPED`.

**MEASURED, by the repo's instrument rather than mine:**

```
anchors: 3/3 present exactly once
baseline: rc=0  125 passed, 2 skipped
CAUGHT bev_mirrored_world            <-- both named guards
CAUGHT bev_two_state_merge           <-- both named guards
CAUGHT bev_head_not_constructed_last <-- both named guards
=== 3/3 defects CAUGHT by the named guard ===   tree restored and verified by sha256.
```

### 3.2 ⛔ AND THAT INSTRUMENT IMMEDIATELY CAUGHT A REAL GAP IN THIS PACKAGE

Its first run **refused to proceed**: *"the suite is ALREADY red"* on
`test_P2_every_knob_is_recoverable_from_the_stamp_BY_VALUE`. The cause was mine, and it was a
genuine provenance defect, not a test artefact:

* `agent_knob_dests` is **derived from argparse** (`--agent*` / `--w-*`), so it picked up
  `--w-bev-aux` automatically — but **only** that one. Every other WP-D knob starts with
  `--bev-aux`, so the family would have reached `config.json` **with its weight recorded and its
  policy absent**.
* ⛔ **`--bev-aux-occlusion` is exactly what separates the pre-registered arm from its
  deliberate-regression twin.** A run record that cannot say which of the two it was makes the whole
  panel unfalsifiable — the M18 finding with a different knob in it. The same holds for
  `--bev-aux-detach` and `--bev-aux-shuffle`, the other two control arms.

**Fixed at the source, not around the test:** the derived predicate now includes `--bev-aux*`;
`_seam_stamp` carries a structural `bev_aux` block (including `shuffled_target`); and the P2 probe's
`base` argv enables the BEV seam for exactly the reason it already enables `--agents oracle` — so a
knob is never refused for a reason that has nothing to do with the knob. All of
`test_refc_v3_agent_provenance.py`, `test_refc_v3_refcv5_wiring.py`, `test_refc_v3_agent_join.py`,
`test_guard_mutation_audit.py`, `test_tac_goal_trainer_flag.py` and `test_bev_aux.py` then read
**136 passed, 1 skipped**.

⭐ **This is the argument for registering in the shared instrument rather than shipping a private
one.** My own mutation script reported 6/6 killed and was blind to this, because it only ran
`test_bev_aux.py`.

### 3.3 The regression check — attributed against a pristine baseline, not asserted

`raw/regression_attribution.json`. The 70 `stack/tests` files that import any module this package
touches, run twice in the SAME environment: once against a **pristine tree** (the 4 tracked changed
files restored from `git HEAD`, the 3 new files removed) and once against the delivered tree.

| | failed | passed | skipped | errors |
|---|---|---|---|---|
| pristine (HEAD) | **12** | 1,188 | 59 | 9 |
| delivered | **12** | **1,224** | 59 | 9 |

⇒ **NEW FAILURES ATTRIBUTABLE TO WP-D: none.** The failure *sets* are identical, and the +36 passed
are `test_bev_aux.py`'s own. The 12 failures and 9 collection errors are pre-existing and
environment-shaped in this off-Drive tree (`ModuleNotFoundError: No module named 'taniteval'`,
absent data sidecars) — they reproduce on HEAD.

⚠️ **This is the ATTRIBUTABLE SUBSET, not the whole 7,370-test suite, and the difference is stated
rather than glossed.** The full `pytest -q` run against the repo on the G: mount reached **85 % over
~2 h** and was stopped under mount contention with another agent's live eval job
(`taniteval/tools/refcv3_arm.py`). ⛔ Its tail is not quotable anyway: pytest imported the trainer at
collection time, i.e. **before** the P2 provenance fix below, so any failure it reports there is
stale by construction. **A clean full-suite run against the repo is an open item** and is named as
such in §11.

### 3.4 ⛔ A second real defect of mine, caught by a repo test the same way

`test_p14_trainer_help_lists_the_flags_and_mine_are_ascii` runs a real `--help` subprocess and
asserts its **return code**. It went RED: `TypeError: %o format: an integer is required, not dict`.
Cause: argparse formats a help string against a dict, so my `"17.66 % of cells"` became a **`%o`
conversion** and `--help` died — for the whole trainer, every flag, not just mine. I had escaped
`%%` in `--bev-aux-pos-weight` and missed the two in `--bev-aux-occlusion`. Fixed, `--help` exits 0,
and the comment at the site now names the failure so the next person writing a percentage there sees
it. ⭐ Worth stating plainly: **two of this package's defects were found by the repo's existing
tests, not by mine** — which is the argument for running the attributable subset rather than only
the suite you wrote.

---

## 4. ⭐ A defect the controls found in this package's own metric, before any arm ran

The naive per-sample average precision breaks ties by array order, so a **CONSTANT** score — which
cannot rank at all — read:

| fixture | naive AP | true base rate | error |
|---|---|---|---|
| 4 positives / 480 cells | **0.010236** | 0.008333 | **+22.8 %** |
| 1 positive / 480 cells | **0.009434** | 0.002083 | **4.5×** |

⇒ the **no-information control would have read HIGHER than the value it is supposed to define**, and
every arm would have been ranked against an inflated floor. Fixed by summing over **tie groups**
(`AP = Σ_g P(g)·ΔR(g)` over distinct scores); the constant control now reads the base rate exactly,
and a perfect ranker reads exactly 1.0.

⭐ **This is the argument for controls that must read a LITERAL** rather than controls that are
merely "lower than the arm": a "lower than the arm" control would have passed.

⚠️ **THE SAME SHAPE IS VISIBLE IN WP-A'S BANKED PANEL.** Its `all-zero` control reads AP
**0.016256** against a stated test base rate of **0.016124** — a **+0.8 %** gap of exactly this
mechanism. Small there, and it does not overturn WP-A's conclusions (its arms sit 5.7× above the
marginal). But it is the same defect, and it is worth re-checking wherever an AP is quoted in this
programme. **Flagged, not retracted** — I did not re-run WP-A's probe.

---

## 5. ⛔ REMOVABILITY — proven, not asserted

The PI's binding rule: *labels may use ego and other agents; **INFERENCE IS VISION-ONLY***. The
published precedent for a training-only head is PhyLatent's Physical State Grounding, *"used only
during training and not required by the planner"* (`LIT-2`, banked `2608.05720`).

**MEASURED** (`test_planner_output_bit_identical`, `test_shared_params_bit_identical`):

| assertion | result |
|---|---|
| every parameter the aux-on and aux-off builds share, at the same seed | **136 / 136 BIT-IDENTICAL** |
| every tensor the planner emits, same seed, same input, eval mode | **BIT-IDENTICAL** |
| keys in the aux-on output that are not in the aux-off output | exactly **one**: `bev_logits` |
| keys in the aux-off output missing from aux-on | **none** |
| `param_breakdown` non-aux lines | **unchanged**, and `total` still sums exactly |

**Two design constraints make that true, and both are load-bearing:**

1. ⛔ **The head is constructed LAST in `RefCModel.__init__`.** Module construction draws from the
   global RNG, so a head inserted anywhere earlier silently changes every subsequent module's
   **initial weights** — and the aux-on/aux-off A/B would then differ in the **SEED** as well as in
   the lever, invisibly, in every log. `M6` is that defect, and it is killed.
2. ⛔ **It is called LAST in `forward` and consumes no RNG** (no dropout, no sampling), so it cannot
   perturb the decoder's draws.

⚠️ **`enable=False` means NOT CONSTRUCTED, not constructed-and-idle.** A disabled-but-present module
still consumes RNG at `__init__` and still lands in `state_dict`, and both break the proofs above
(`test_off_is_not_constructed_at_all`).

---

## 6. The controls, and whether each was verified to read its known value

| control | must read | verified |
|---|---|---|
| constant score (logits −9 / 0 / +9) | **exactly** the base rate | ✅ reads `4/480` exactly at all three |
| all-zero predictor | IoU **0.000**, F1 **0.000** | ✅ |
| perfect ranker | AP **1.0**, IoU **1.0**, F1 **1.0** | ✅ |
| no positives in the scored set | AP **NaN**, never 0.0 | ✅ |
| all-NO_LABEL batch | loss **exactly 0.0**, with `n_supervised == 0` saying why | ✅ |
| IGNORE cells | a logit change under one may not move the loss **by one bit** | ✅ |
| detached trunk (`D3`) | trunk gradient **exactly 0** | ✅ (and `> 0` with detach off) |
| `--bev-aux-shuffle` (`D2`) | no gain over aux-off | ⏳ needs the GPU arm |
| raw-pixel floor | the trunk must BEAT it | ⏳ needs the GPU arm — ⛔ `E-DEC-18-R1` FAILED this exact bar |

### 6.1 The refusals, verified to FIRE (CPU, `raw/trainer_wiring_check.json`)

| refusal | fired |
|---|---|
| `--bev-aux col --w-bev-aux 0` → *"ZERO gradient into the trunk"* | ✅ |
| `--bev-aux col` without `--agent-join` → *"NO LABELS"* | ✅ |
| `--bev-aux off --w-bev-aux 0.5` → *"SILENTLY SKIPPED while config.json stamps the weight"* | ✅ |
| `BEVAuxHead` with `gw != n_az` → refuse, never resample | ✅ |
| `BEVAuxConfig(enable=True, w=0)` | ✅ |

⭐ **All five are the `w_agent` defect in different costumes** — a seam declared in `config.json`
whose loss term is silently skipped — which has already happened in this trainer and is documented
there by name.

### 6.2 ⭐ `E-DEC-18b` moved to BEFORE the GPU-days

`refc_bev_aux.assert_loss_parity` refuses a run whose `w_bev · bev_loss / traj_loss` is outside
**[0.02, 3.0]**. The measured PSG failure was a **10–30×** ratio, at **every** weight tested, and it
destroyed the encoder — visible only after the compute was spent. On the wiring check's real numbers
the ratio reads **0.406** ✅.

⛔ **AND IT IS CALLED FROM THE TRAINING LOOP, not merely available.** A guard that exists and is
never invoked is the `w_agent` defect wearing a different hat, and shipping one inside a package
whose whole subject is inert guards would have been the joke telling itself. It fires **once**, at
the **first step that actually had supervised cells** — ⛔ not at step 0, because a step whose
windows are all NO_LABEL yields `bev_loss == 0.0` exactly, and a ratio computed on that would pass
the gate while measuring nothing. That is the same vacuity `n_supervised` is printed in every log
row to expose.

⚠️ **Necessary, not sufficient.** A term inside the band can still be harmful; that is what the
pre-registered planner-regression failure twin is for.

---

## 7. The pre-registered criteria, in one line each

Full text: `Project Steering/PREREG_WPD_BEV_AUX.md`. Hypothesis `E-BEV-AUX-1`.

* **SUCCESS** = **A1 ∧ A2 ∧ A3 ∧ A4** on a frozen-feature probe of the trained trunk (WP-A §4.2's
  instrument, unchanged): beat the **marginal** control by ≥ +0.010 test AP with a separated paired
  CI; beat the **raw-pixel floor**; beat the aux-off arm by ≥ **3×** a replicate floor measured *in
  the same panel*; and beat the **shuffled-target** arm by the same margin. **AND** non-regression
  at **T1** on the four families, with `D-REFAV1-LON-ACTS`'s "it must still ACT" clause attached.
* **FAILURE TWINS**, each with its next lever named in advance: **F1** no transferable content ⇒
  refuted, next lever is `E-DEC-8` (DINOv3 distillation, MEASURED `n_agents` **−1.04 → +0.33** at no
  ego cost); **F2** representation up, planner down ⇒ the `E-DEC-18b` shape on REF-C, reported and
  **not tuned around**; **F3** the detached arm matches ⇒ the claim is void whatever the numbers
  say; **F4** the shuffled arm matches ⇒ capacity, not content; **F5** everything at the marginal ⇒
  **underpowered, not negative** — report the bound.
* ⭐ **`D4` (two-state) must be DETECTABLE.** If it is not, §2's design decision is **retracted for
  WP-D** and said so plainly — the driving argument for masking survives on its own.

---

## 8. The GPU arm — specified, costed, NOT started

⛔ **Nothing was launched.** A40 training refcv5-v2 (~17,750 / 40,284); Thor training refav1; the
dev-box 4060 occupied. No load was added to any of them.

| | |
|---|---|
| rate | **4.0 s/step MEASURED** on this exact configuration (`…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md`, 50 steps per 200 s sustained — that receipt **retracts its own earlier ~1.2 s/step**, which came from inside `--warmup`). Independently corroborated: launch 23:10:12Z → ~17,750 steps by ~18:00Z ⇒ **3.82 s/step** |
| ⭐ **the cut I would launch first** | **D0 (aux off) / D1 (aux on) / D2 (shuffled target)** at **4,000 steps** = **4.4 h/arm ⇒ 13.3 h on one card**. It answers **A4** — information or capacity — and the **sign** of A3, before the replicate arms are funded. **It fits in one night.** |
| the full panel | 7 arms × 12,000 steps = **93 h ≈ 3.9 days** on one card |
| the probe afterwards | ~**50 GPU-minutes total**, on the dev-box 4060 — **no A40 time** |
| storage | ~4 GB/arm; the join adds **0** (already on the box) |
| added dataloader cost | **MEASURED 0.63 ms/frame** (26,394 frames in 16.7 s) ⇒ ~13 ms per batch of 20 over 6 workers, against a 4.0 s step — negligible |
| ⚠️ host RAM | `--agent-join` loads the TRAIN join at ~**2.1 GB RSS** (`JoinFileReader` docstring, MEASURED 2026-09-05). Budget it or pass `episode_ids` |

⇒ **Ready to run the moment a GPU frees.** The one thing that is NOT verified end-to-end is a real
`--v2-cache` batch flowing through the new dataset path — the wiring check exercises the refusals,
the config assembly, the head, the target and the backward, but on **synthetic** tensors, because
the v2 cache and the TRAIN join are not on this box. **First action when a GPU frees: a 20-step
smoke on the real cache, checking `bev_n_supervised > 0` and `bev_n_pos > 0` in the log row** — both
are already emitted per step for exactly this reason.

---

## 9. What this package does NOT claim

* ⛔ **It does not build the waypoint index (WP-B).** That is the successor, gated on WP-D's §5A.
* ⛔ **It does not claim any effect on driving.** Nothing was trained. §5B is a NON-REGRESSION bar,
  and saying so in advance is what stops a representation result being sold as a driving one.
* ⛔ **It is an AGENT BEV, not a scene BEV.** PhysicalAI-AV publishes no map, lane graph, junction
  annotation or route (*"we do not include open maps data"*, settled at five probes), and
  `obstacle.offline`'s 10 classes are all **dynamic agents**.
* ⚠️ **Every occlusion number is a LOWER BOUND** (§2.2's three stated limits).
* ⚠️ **The polar base rate 3.1634 % is NOT comparable to WP-A's Cartesian numbers** — different
  geometry, different denominator. The comparable figure is this census's own Cartesian control,
  **1.8382 %**.
* ⚠️ **The census is B1 EVAL.** The arms train on the B1 TRAIN join, whose base rate is **not
  measured here**. ⛔ The `pos_weight` constant 30.61 is therefore derived from the EVAL side and
  must be re-derived on TRAIN before the full panel — it is a **stamped constant**, so the
  substitution is auditable, and it is named as a work item rather than left implicit.
* ⚠️ `sub_r=2 / sub_az=3` sub-sampling is a stated lattice choice. It is fine enough that the
  analytic literals hold, and coarse enough that a 1 m-deep object can fall between samples — which
  is why the tests use realistic vehicle footprints and say so.

---

## 10. Deliverable manifest

| artifact | where |
|---|---|
| this report | repo `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-wpd-bev-aux/RESULT.md` |
| the pre-registration | repo `Project Steering/PREREG_WPD_BEV_AUX.md` |
| the register row `E-BEV-AUX-1` | repo `Project Steering/GOALS_AND_CLAIMS.md` |
| **the target builder** | repo `stack/tanitad/data/bev_aux.py` |
| **the head + loss + metrics + parity guard** | repo `stack/tanitad/refs/refc_bev_aux.py` |
| **gated model wiring** (config field, head LAST, `bev_logits` LAST, breakdown line) | repo `stack/tanitad/refs/refc.py` |
| **trainer** (flags, refusals, dataset target, loss term, shuffled control, `bev_aux` seam stamp, widened knob-provenance predicate) | repo `stack/scripts/refc_v3_train.py` |
| **the suite — 36 tests** | repo `stack/tests/test_bev_aux.py` |
| WP-D's 3 load-bearing mutants in the repo's standing audit | repo `stack/scripts/guard_mutation_audit.py` |
| the P2 probe's enabling context for the BEV seam | repo `stack/tests/test_refc_v3_agent_provenance.py` |
| census / mutation / wiring scripts | repo `…/2026-09-07-wpd-bev-aux/code/` |
| result JSONs + logs | repo `…/2026-09-07-wpd-bev-aux/raw/` |
| stack snapshot the scripts imported | dev box `C:\Users\Admin\wpd-bev\stack_snapshot\` — **a copy; nothing lives only there** |
| the join (unmodified, pre-existing) | dev box `C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz` |

⛔ **Nothing in this package exists in only one place.** Every artifact that took effort is in the
repo and staged.

## 11. ⛔ ESCALATION — what needs a decision, not a doc

1. ⭐ **GPU for the 13.3-hour three-arm cut** (D0 / D1 / D2 at 4,000 steps). It is the cheapest
   experiment that can still discriminate, and it is blocked purely on a free card. **This is a
   provision decision.**
2. ⚠️ **The AP tie-handling flag (§4) touches WP-A's banked panel and possibly others.** Whether to
   re-run any banked AP is a call I did not make unilaterally; the defect is +0.8 % on the one panel
   I could check and does not overturn its conclusions.
3. ⚠️ **`pos_weight = 30.61` is derived from the EVAL join** and must be re-derived on the TRAIN
   join before the full panel. Cheap (~2 min CPU), but it changes a stamped constant, so it should
   happen in the same turn as the launch, not after it.
4. ⚠️ **A clean full-suite `pytest -q` against the repo is OPEN.** The attributable subset (70 files,
   1,224 tests) is green with a pristine-baseline diff showing **zero** new failures (§3.3), and
   that is the check that can see this change — but it is not the whole suite. The G: run stalled at
   85 % under contention with another agent's live eval; it should be re-run when the box is quiet,
   or from an off-Drive clone with `taniteval` and `tools` present.
