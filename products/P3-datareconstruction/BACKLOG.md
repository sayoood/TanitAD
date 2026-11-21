# P3 — TanitAD_DataReconstruction — BACKLOG

`Owner: TanitAD_DataFlyWheel. Companion to products/P3-datareconstruction/SPEC.md.`
`Ranked by LEVERAGE (value ÷ effort × unblockedness), not by wish.`
`Each item: why · MEASURED evidence of the gap · effort · GPU · dependencies · definition of done.`

**Reading rules** (CLAUDE.md / TANITAD_PROGRAMME.md §6):
- Effort **S** ≤ 1 day · **M** 1–3 days · **L** > 3 days or a GPU campaign.
- **GPU** `0` = always executable. A blocked headline item never blocks a `0`-GPU item —
  *"gated ≠ idle"*.
- Done = **in the repo, staged, with its provenance** — verified by `git ls-files --cached`,
  never by an exit code.

---

## Priority summary

| tier | what it buys | items |
|---|---|---|
| **T0** | bank what exists; make its numbers quotable | P3-1, P3-2 |
| **T1** | ⛔ **guards that stop P3 manufacturing a poisoned corpus** — cheap, unblocked, irreversible in value | P3-3 · P3-4 · P3-5 · P3-6 · P3-7 |
| **T2** | make the IDM's claims admissible under the programme's own rules | P3-8, P3-9 |
| **T3** | ⛔ **fix measured defects in the shipped harvest** — three of these are live rule violations | P3-10 · P3-11 · P3-12 · P3-13 |
| **T4** | the science that decides whether the product works | P3-14 · P3-15 · P3-16 · P3-17 |
| **T5** | the headline verdict, re-run correctly | P3-18 |
| **T6** | PI-gated decisions | P3-19 · P3-20 · P3-21 |
| **T7** | production polish | P3-22 |

⚠️ **T1 outranks T2–T4 deliberately.** P3's output is *training data*. A defect in a model is
one arm; a defect in a corpus is **every arm trained on it**, found late, with no way to tell
which results it contaminated. §3.6 of the SPEC is the proof: **two leaked episodes out of
twenty-two moved a headline from −0.75 to +0.33** and the claim had to be withdrawn.

---

## T0 — bank and verify what exists

### P3-1 — Consolidate the reconstruction line into `stack/`
**Why.** Everything P3 depends on except `idm_head.py`, `idm_families.py` and the
`run_idm_*.py` orchestrators lives under `TanitAD Research Hub/…/Implementation/incoming/`.
That is the programme's most expensive measured failure pattern — AGENT_OPERATING_STANDARD
rule 3, with LAL-v2 (12 days), an orthogonality instrument (10 days), TanitEval and REF-B v2's
architecture as precedents.

**MEASURED evidence.** **13** work-package directories carry P3-relevant code, none importable
as `tanitad.*`: `2026-07-22-idm-proof`, `2026-07-26-idm-v2`, `2026-07-27-idm-v3`,
`2026-07-27-fleet-refill`, `2026-07-27-fleet-sync-idm-steer`, `2026-08-03-idm-accel-recoverability`,
`2026-08-03-idm-derived-accel`, `2026-08-03-idm-four-families`, `2026-07-22-youtube-idm-pipeline`,
`2026-07-24-youtube-idm-pilot`, `2026-07-25-youtube-idm-scaleup`, `2026-07-25-geocalib`,
`2026-07-28-youtube-geocalib-idm`. The load-bearing ones to promote: `geocalib_intrinsics.py`,
`harvest_scaleup.py`, `yt_idm_reconstruct.py`, `horizon_probe.py`.

**Effort** M · **GPU** 0 · **Deps** none — the unblocked entry point.

**Done when.** A real `stack/tanitad/recon/` package, importable, each module headed with the
`incoming/` dir it came from; `git ls-files --cached stack/tanitad/recon/` lists every file
(⛔ `git add` exit codes are not evidence — the new-directory silent-no-op trap); `pytest -q`
green; and a one-line disposition for **all 13** dirs (*promoted* / *superseded by X* /
*evidence only*). No dir left unadjudicated.

---

### P3-2 — Re-verify every quoted number against its raw JSON
**Why.** Parts of SPEC §3.2 originate in **docstrings** that cite artifacts. CLAUDE.md: *"any
number in any report cites the registry or the raw eval JSON, never a summary"* — a docstring is
prose. Three errors propagated for days in this programme from exactly this.

**MEASURED evidence.** `stack/scripts/idm_head.py:85-155` carries ~12 decision-grade figures.
Two already **disagree in framing** with the JSON and are flagged in the SPEC: the docstring's
*"−0.15 to −0.42"* for `long_accel` vs `idm5_ensemble.json`'s pooled **−0.05906**
(per-domain −0.037 / −0.226), and the `seed_mean`-is-an-ensemble trap (0.79926 vs 0.78716).

**Effort** S · **GPU** 0 · **Deps** none.

**Done when.** `raw/verify_idm_numbers.json` maps every quoted figure → artifact path → JSON key
path → value read. Discrepancies filed in `Project Steering/RETRACTION_LOG.md` **with root-cause
class**, and the SPEC corrected in the same change.

---

## T1 — ⛔ guards, before any corpus is manufactured

### P3-3 — Add the **raw-input floor (L1)** — the one genuinely missing ladder rung
**Why.** TANITAD_PROGRAMME.md §6.2 requires a raw-input floor on every probe panel. Without it
you cannot tell the *encoder's* contribution from the *corpus's*.

**MEASURED evidence — and a correction to an earlier draft of this backlog.** I initially listed
L0 (constant-only) and L3 (guard able to fail) as missing. **That was wrong**, and the error came
from probing `stack/tests/` and not the `incoming/` work packages:

| rung | actual status |
|---|---|
| **L0 constant-only** | ✅ **EXISTS** — `arms.NULL_train_mean` reads **−0.0626** (not 0) on held-out; also `NEG2_blind_mean_predictor`, v3 `Ccorp`/`Crign` |
| **L2 shuffled-latent** | ✅ EXISTS — 13 `__CTRL` arms, `NEG_shuffled_latents`, `NEG1_…` |
| **L3 able-to-fail** | ✅ **EXISTS and is strong** — a planted rank-1 signal at **1e-5 of latent RMS** moves held-out R² from −0.06259 to **+0.94335**; `NEG3_gt_with_lateral_sign_flipped` keeps scalar R² = 1.0 while lateral BA collapses to chance |
| **L1 raw-input floor** | ⛔ **MISSING.** The `ORACLEIN_*` arms are an *oracle ceiling* (true CAN speed as input), **not** a raw-pixel floor — a different object |

⚠️ **A second, sharper defect found in the same probe.**
`stack/tests/test_idm_head.py:191`
`test_deriving_accel_beats_regressing_it_on_a_corpus_where_a_equals_dv_dt` **passes on synthetic
data** while the real-corpus result is the **opposite** (derived accel separated-WORSE,
Δ−0.25298 [−0.48314, −0.09967]). A test that can only pass is not a guard.

**Effort** S–M · **GPU** 0 (dev-box 4060) · **Deps** none.

**Done when.** L1 implemented (ridge on downsampled frame-difference features, no learned
encoder), run by every `run_idm_*.py`, printed with `n` and `d`; the IDM shown
**separated-better than L1** per channel or the channel is masked; the synthetic test renamed to
say it pins a *contract on a synthetic corpus*, never evidence about the real one.

---

### P3-4 — The provenance sidecar, and P3-namespaced corpus keys
**Why.** A reconstructed episode is currently **indistinguishable downstream from a real one**.
Without a sidecar, SPEC §6.1, §7.1, §7.3, §7.4 are unenforceable prose.

**MEASURED evidence.** `stack/tanitad/data/_contract.py:11-16` — the entire contract is
`frames` / `actions[T,2]` / `poses[T,4]` / `episode_id`. Nothing else exists. And the harvested
corpus demonstrates every field that is missing: **licence `None` on 343/343**, **63.3 % not
`fully_canonical`**, median achieved `f_eff` **337.6** vs target **266**.

**Effort** M · **GPU** 0 · **Deps** P3-1.

**Done when.** Schema + writer + reader; all SPEC §6.1 fields present including
**`source_video_id`** (the cluster unit) and `fully_canonical`; `is_parity: false` constant; key
namespaced `p3-<source>-<hash>`; round-trip test; and the export **refuses with a typed
exception** if any mandatory field is absent.

---

### P3-5 — Regression test: no P3 corpus key can collide with a parity key
**Why.** `parity.corpus_key_of` (`stack/tanitad/data/parity.py:211`) resolves keys by **path
substring**, and the module header records that the pre-existing check *"passes for a correctly
named directory holding the wrong number of episodes"* — the class of silent failure that
already cost this programme a truncated corpus.

**MEASURED evidence.** `parity.py:78-80` defines
`PARITY_TRAIN_KEY = "physicalai-train-e438721ae894"` / `PARITY_SKIP_HASH = "f09e44db"`; `:211`
does substring resolution. **No test asserts P3/parity key disjointness** — P3 keys do not exist
yet, which is exactly why the guard should land *before* they do.

**Effort** S · **GPU** 0 · **Deps** P3-4.

**Done when.** A test asserting no P3-namespaced key contains any registered parity key as a
substring, **shown able to fail** by a deliberately colliding fixture; plus a positive test that
a P3 corpus triggers the loud `[parity] NON-PARITY` line and returns `parity=False`.

---

### P3-6 — Regression test: the IDM cannot be fed an ego channel
**Why.** The vision-only invariant is a **comment**, not a guard. `idm_head.py:12-15` states
*"The encoder is FROZEN and PURELY VISUAL … takes NO action/speed channel"*. The IDM's labels
**are** CAN speed and steering, so an ego input converts its speed R² from a capability into a
leak — the sitclf-leak and REF-A-I-JEPA-leak family, both found late.

**MEASURED evidence.** `stack/tests/test_idm_head.py` — 14 tests, none asserting input-channel
provenance.

**Effort** S · **GPU** 0 · **Deps** none.

**Done when.** A test that fails if any ego/action/speed channel is wired into the IDM's encoder
input path, **demonstrated able to fail** by a fixture that wires one in. Its docstring records
the practical reason the rule can never be relaxed: **on internet video there is no ego channel**
— an IDM that needs one cannot do the job it exists for.

---

### P3-7 — ⭐ Content-hash disjointness, enforced (not id-disjointness)
**Why.** This is the guard whose absence **already inverted a published headline**.

**MEASURED evidence.** `…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json`, key
`anchor_settlement_2026_07_27` (retraction class **C43**): 2 of 22 comma val episodes were
**bit-identical** to 2 training clips — caught only by sha256 of raw poses *and* `frames_u8`.

| slice | comma yaw R² |
|---|---:|
| ALL22 (as published) | **+0.330822** |
| **CLEAN20 content-disjoint** | ⛔ **−0.745999 [−1.574, −0.177]** |
| the 2 leaked episodes alone | +0.856185 |

⇒ **Two episodes out of twenty-two flipped the sign and the claim was WITHDRAWN.** Reconstructed
corpora are *structurally* more exposed: one source video yields many near-duplicate clips. And
the harvest census confirms it — **343 clips from only 55 videos**.

**Effort** S–M · **GPU** 0 · **Deps** P3-4.

**Done when.** Every P3 split check hashes **frames and poses**, not ids; a test **shown able to
fail** by planting a duplicated episode across the split; and near-duplicate detection (perceptual
hash) reported per split, since bit-identity is the easy case.

---

## T2 — make the IDM's claims admissible

### P3-8 — Wire the four families into every IDM eval, and build the two missing ones
**Why.** ⛔ **BINDING** (CLAUDE.md, PI 2026-08-02, after asking repeatedly): every eval reports
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC **in addition to** ADE, per family, never pooled;
*"a missing metric is a work item, not an excuse"*. It matters most here, because the IDM's
purpose is to **mint labels** — it can be excellent on speed R² while labelling the wrong
manoeuvre on every turn.

**MEASURED evidence.** The instrument exists (`stack/tanitad/eval/idm_families.py`) and was run
once (`…/2026-08-03-idm-four-families/results_idm_four_families.json`, n 6,900 / 50 comma2k19
episodes). Two families are incomplete and the wiring is absent:

| family | status |
|---|---|
| LONGITUDINAL | computed, but ⛔ `distance_keeping.status = "UNAVAILABLE"`, **n = 0** — no lead-agent track |
| LATERAL | ✅ complete (heading_mae 0.01756 rad, curvature_mae 0.0065 1/m, cross_track_mae 0.1952 m) |
| TACTICAL | computed (lateral BA **0.7722**, longitudinal BA **0.68654**), `goal_setting.status = "PARTIAL"` |
| **STRATEGIC** | ⛔ **`UNAVAILABLE`** — no route/goal/map label on either substrate |

My own probe (`grep -rn "four_families\|idm_families" stack/scripts/ stack/tanitad/`) found **20
hits, all world-model-side** (`fuse_situation_scores.py`, `lan_probe.py`, `refc_obj_probe.py`,
`refc_sel_probe.py`) — **no `idm_*` or `run_idm_*` script imports either module**, corroborating
the module docstring's own claim at `:8-10` with an independent second probe.

⭐ **The reason this is worth real effort, measured:** `NEG3_gt_with_lateral_sign_flipped` feeds
*ground truth* with the lateral sign flipped — **scalar R² stays 1.0 by construction** while
lateral BA collapses to **0.3333 (chance)** and ADE degrades separated. **The lateral family
catches an error the scalars structurally cannot.**

**Effort** M · **GPU** 0 · **Deps** P3-1.

**Done when.** Every `run_idm_*.py` emits four families with per-family `n`; cadence passed
explicitly (`IDM_DT_S = 0.5`, **not** the 10 Hz default) and asserted by test; `distance_keeping`
either computed from a lead-agent track or its `n=0` justified per family; and **STRATEGIC either
computed or its absence recorded as an open work item** — the file's own `_contract` says *"A
family marked UNAVAILABLE is a WORK ITEM, not a pass."*

---

### P3-9 — Publish per-channel admissibility (E-P3-IDM-1)
**Why.** P3 must state, per channel, whether the IDM may mint it — and today that answer is
scattered across docstrings and five artifacts that do not all agree.

**MEASURED evidence.** Absolute R² at rung 757 (`idm5_ensemble.json`, n 4,195 / 36 eps):
`speed` **+0.865**, `yaw_rate` **+0.919**, `steer` **+0.799**, `long_accel` ⛔ **−0.059**.
Separation vs shuffled-latent (`results_idm_derived_accel.json`, 33/17 episode-disjoint):
`speed` **+0.7187**, `yaw_rate` **+0.2252**, `long_accel` **−0.0984 [−0.3087, +0.0179]**.
⇒ P3 can supply **1 of the 2** operative control channels (`metric_dynamics.py:473`).

⚠️ **And one channel's failure has TWO causes that must be separated.**
`…/2026-07-26-idm-v2/labels2.json` key `long_accel.pai`: `corr_label_vs_dv_dt = 0.4335`,
`r2_ceiling_of_best_affine_in_dv_dt = **0.18794**` — **on PhysicalAI the LABEL itself caps a
perfect kinematic estimator at R² 0.188** (comma2k19: 0.8547). Part of the failure is a bad
target, not a missing signal.

**Effort** M · **GPU** dev-box · **Deps** P3-2, P3-3, P3-8.

**Done when.** A per-channel admissibility table with the full L0–L4 ladder on route-disjoint,
episode-disjoint **and content-disjoint** splits, ≥3 seeds, paired episode-cluster bootstrap
B=2000; each channel ADMISSIBLE / **MASKED**; the PhysicalAI-vs-comma label-quality split stated
per channel; the table becomes the default `channel_admissibility` in P3-4's sidecar.

---

## T3 — ⛔ fix measured defects in the shipped harvest

### P3-10 — ⚖️ Fix the licence gate — a written recommendation and shipped behaviour disagree
**Why.** The licence analysis recommends a clean first slice of **CC-BY channels only**
(`LICENSING_TIER_ANALYSIS.md:136`). The shipped pipeline does the opposite.

**MEASURED evidence.** `harvest_scaleup.py:217` sets `--allow-noncc default=True`; the resulting
corpus carries **licence `None` on 343/343 clips** (`YT_DB_RETRY.md:167`). The result note itself
calls it *"a posture for the PI to confirm, not one to adopt silently"*
(`YOUTUBE_GEOCALIB_IDM_RESULT.md:66-69`). This directly violates SPEC §7.3.

**Effort** S · **GPU** 0 · **Deps** PI **Q1(c)** for the *policy*; the *default flip and the
mandatory field* need no decision.

**Done when.** Licence class is a **required, non-null** field at ingest; `--allow-noncc`
defaults **False**; a test refuses an ingest with a null licence; and the tier vocabulary
(`ship`/`ship-sa`/`nc`/`refuse`) plus the augmentation rule
`tier(derivative) = strictest(source, generator, conditioning_labels)` is enforced in code, not
prose.

---

### P3-11 — Fix the contamination filter — 12.83 % contamination, **zero** catch rate
**Why.** A quality gate that catches nothing is worse than none, because it looks like one.

**MEASURED evidence.** `corpus_census_n343.json:79-81`: `contamination_frac = **0.1283**`,
**`caught_by_shipped: []`** — literally zero efficacy. Root causes are concrete:
`BAD_TITLE` (`harvest_scaleup.py:104-106`) lists `"5x"`, `"10x"`, `"4x speed"`, `"2x speed"` but
**`"3x"` is absent**, and there is **no game/sim filter at all** — BeamNG footage passed.

**Effort** S–M · **GPU** 0 · **Deps** none.

**Done when.** Filters extended (speed-multiplier patterns incl. `3x`, a game/sim detector, and
a **content-based** check rather than title-only); re-scored against the same 343-clip census;
**catch rate reported with a CI and shown separated above zero**; and a regression test with
known-contaminated fixtures — a filter that has never caught a planted positive does not ship.

---

### P3-12 — Re-canonicalise with GeoCalib — the corpus was built on the fixed-HFOV fallback
**Why.** The geometry front-end the pipeline was designed around **was not used** for the corpus
it produced.

**MEASURED evidence.** `geocalib_shim.py:5-7` records that at scale-up launch the real GeoCalib
deliverable *"had NOT landed, so the harvest ran with the fixed-HFOV fallback (100 deg)"*.
Fixed 100° was wrong for **11 of 12** real clips; measured hfov median **66.56°**
(`youtube_geocalib_measurement.json:2-33`). Consequence: **63.3 % of clips `fully_canonical ==
false`**, median achieved `f_eff` **337.6** against target **266** (`YT_DB_RETRY.md:177-190`).

⚠️ **Blocked in practice by P3-21**: the latents this would re-derive from are stranded on a pod
and likely gone.

**Effort** M · **GPU** dev-box · **Deps** P3-21, and P3-14 for the estimator's own bound.

**Done when.** Canonicalisation runs from estimated per-video intrinsics with the confidence
gate; `fully_canonical` and `achieved_f_eff` recorded per clip in the sidecar; and the fraction
reaching the canonical frame reported **before and after**.

---

### P3-13 — The P3 → P4 export contract, cadence-pinned
**Why.** A measured 5× error sits in the seam, and a naming convention would ship a second one.

**MEASURED evidence.** IDM emits **4** waypoints at 0.5 s spacing over 2 s (`idm_head.py:38`,
`idm_families.py:57-58`); the operative layer runs at `dt = 0.1` ⇒ **10 Hz**
(`metric_dynamics.py:624`) — **20** waypoints over the same 2 s. `idm_families.py:30-38` records
that feeding IDM trajectories to `four_families` unchanged *"reads every speed and yaw-rate 5×
too large"*.

⚠️ **The naming trap.** The operative control is `(accel, **yaw_rate**)`, **not**
`(accel, curvature)` — `metric_dynamics.py:473`, `:488`. An exporter written to the `(a, κ)`
shorthand would divide by `v` and reintroduce exactly the ill-conditioning the readout was
redesigned to remove (curvature kurtosis **38.9**, low/high-speed tail ratio **9.5×**, vs yaw
rate's 10.4 and 2.3×).

**Effort** S · **GPU** 0 · **Deps** P3-4.

**Done when.** Exporter emits `(accel, yaw_rate)` + `speed` — **never κ** — with `cadence_hz`,
plus `upsample_method` **and its measured error** if not native 10 Hz; a test pins cadence the way
`test_idm_families.py` pins `four_families._seq_geometry`; a test asserts no κ channel is ever
produced.

---

## T4 — the science that decides the product

### P3-14 — ⭐ E-P3-CALIB-1: re-score calibration on a VARYING-focal arm (live false-positive risk)
**Why.** Not a build task — GeoCalib estimation **exists and is validated**
(`…/2026-07-25-geocalib/`, verdict **QUALIFIED PASS**). It is a **re-scoring** task, and it is
urgent because the headline is quotable in a way that would be a false positive.

**MEASURED evidence** (`VALIDATION_REPORT.md` §P2 / `geocalib_validation_results.json`):

| arm | n | focal \|err\| median |
|---|---:|---:|
| comma_native (GT always 910) | 12 | **6.8 %** |
| comma_480p (GT always 910) | 12 | **7.1 %** |
| **comma_focalsweep (GT VARIES)** | 4 | **25.0 %**, **Pearson r = 0.41** |

⛔ **In both passing arms the true focal never changes**, so a constant predictor emitting ~850 px
scores the same 6.8 %. The only arm testing *tracking* fails, and the module's own header records
that GeoCalib *"regresses toward a ~50-55 deg vFoV prior"* (`geocalib_intrinsics.py:26-28`).
Also unresolved: **extrinsics** — `calib.py:17-18` says *"NOT yet fully normalized"*; the one fix
uses `(cx,cy)` **read from PhysicalAI's `calibration/`** (`:19-23`), supplied not estimated; and
**height is not observable from a single image at all**.

**Effort** M · **GPU** dev-box · **Deps** P3-1.

**Done when.** Pre-registration with both outcomes **before the run**; the Y0 falsifier closed as
a gate **on a varying-focal arm** (n ≫ 4) with a **constant-predictor control** and a
deliberate-regression arm; **focal-tracking `r` reported as a headline number**, not the median
alone; a stated verdict on **fisheye** sources and what fraction of a real harvest they exclude;
and E-P3-CALIB-2 (pitch/roll) scoped.

---

### P3-15 — ⚠️ E-P3-SCALE-1: resolve and validate metric scale
**Why.** ⛔ **The highest-risk item in P3.** Every metric quantity P3 emits is downstream of the
monocular scale ambiguity, and the programme's worst measured failure mode is exactly this axis.

**MEASURED evidence of the gap.** No scale-resolution code exists (SPEC §3.4). What passes for
it today is **reading a known camera height**: `idm3_geom.py:95-102` takes `cam_h_m` from
PhysicalAI's `calibration/`, and comma2k19's **1.22 m is hard-coded and flagged "INHERITED and
UNVERIFIED"** (`:73-77`). `geom_sanity.py:281-290` looks like scale recovery but its `dd` comes
from **GT poses** (`:231`) — the inverse of the problem.

⛔ **One candidate route is already REFUTED**: the ground-plane metric-scale route, per
`…/2026-07-27-latent-action-models/LATENT_ACTION_RESEARCH.md:364` — *"ALREADY REFUTED on our
corpus"*. **Do not re-propose it.** The remaining candidates (speedometer OCR, lane-pitch ruler,
metric monocular depth, telemetry) are **HYPOTHESIS class — none implemented, none measured on
our frames.** The one route with published in-domain support is supervised grounding: Cosmos-3
reports ATE **0.98 m** vs VGGT/DepthAnything3 **23.46 / 9.29 m** *because* scale comes from
supervised in-domain logs (`…/2026-07-22-own-dynamics-encoder/DESIGN.md:64`) — which is precisely
what P3 lacks for a new source.

**Why it is urgent, in numbers.** The deployed flagship already shows a systematic over-speed
prior: `speed_bias` **+0.484 m/s**, **71.95 %** of windows ahead at 2 s, **75.51 %** faster than
the human (`MODEL_REGISTRY.md:1148`); **88.7 %** of the oracle gap is longitudinal. A pipeline
that resolves scale badly **manufactures a corpus that actively teaches the wrong speeds**.

**Effort** L · **GPU** dev-box · **Deps** P3-14 (calibration bounds scale — they are **not**
independent).

**Done when.** SPEC §4.2 executed in full: scale-normalised trajectory + **separate explicit λ**;
per-clip `log(λ̂/λ_true)` distribution; within-clip drift; constant-only and raw-input controls;
deliberate-regression arm shown red; paired bootstrap clustered on **source video**. Both
outcomes pre-committed — **including the REFUTED fallback** (scale-free deliverable, longitudinal
channel **masked**, not down-weighted).

---

### P3-16 — Quality gate and the cross-agreement score (E-P3-AGREE-1)
**Why.** SPEC stage ⑧ is **MISSING as code**. Without a per-clip gate, P3 cannot decide inclusion
or loss weight, and every downstream claim is about an unfiltered corpus.

**MEASURED evidence.** The design exists — `YOUTUBE_DASHCAM_STRATEGY.md:45-48` specifies IDM ↔ VO
cross-agreement as the per-clip action-quality score, with the falsifier *"pseudo-labels on
held-out comma (pretending no CAN) must correlate with real CAN at r>0.8 for steer"* (`:47-48`).
**No module implements it.** Half the input exists: `yt_idm_reconstruct.py:105-125` gives a
GT-free **rotational** VO signal (optical-flow yaw + FOE) — enough for a *lateral* agreement
score, not a longitudinal one.

**Effort** M · **GPU** dev-box · **Deps** P3-15.

**Done when.** The gate implemented; the falsifier run as a pre-registered gate; and **the score
validated as predictive** — rank correlation between agreement and held-out label error, with a
CI. A quality score that does not predict quality is worse than none.

---

### P3-17 — E-P3-XDOM-1: cross-domain transfer — the standing blocker
**Why.** P3's entire value proposition assumes an IDM that generalises to unseen cameras. The
registry says the substrate built to do that **failed**.

**MEASURED evidence.** `…/2026-07-22-idm-proof/results.json`: `go_no_go.PASS = **false**` — the
pre-registered gate has **never** passed. PhysicalAI → comma2k19 speed **+0.657**, yaw
**+0.0005**, ADE 2.40× in-domain. rig-A → rig-B speed ⛔ **−2.465**, ADE 4.01×. Neither remedy
closes it: light fine-tune 0.4058 → 0.4111; multi-domain co-train Δ+0.0392.
`results_multirig.json.overall_verdict.data_diversity_hypothesis = **"REFUTED on existing
assets"**` — the collapse is *representational*. `results_regate.json`:
`fix1_frontend = "NO-OP … rig-B collapse NOT intrinsics-driven"`, `NEITHER_FIX_REACHES_GATE =
true`. And `MODEL_REGISTRY.md:3093` registers the own-encoder line **🟥 FAIL — REFUTED**, stating
*"The own-encoder / YouTube-IDM thesis resting on it is not supported."*

**Effort** L · **GPU** campaign · **Deps** PI **Q3**.

**Done when.** Either a substrate that passes the gate, or a recorded decision to accept a
**per-domain IDM** cost model with its price stated. ⚠️ A third outcome is legitimate and should
be pre-registered: that P3's reach is bounded to camera families resembling its training corpora
— which is a *scoped* product, not a failed one.

---

## T5 — the headline verdict

### P3-18 — ⭐ E-P3-DOWNSTREAM-1: re-run the downstream A/B with the correct cluster unit
**Why.** This is the **only** criterion that decides whether P3 is the data moat. A version has
already run and is **positive but not safely quotable**.

**MEASURED evidence.** `…/2026-07-27-yt-dB-retry/pod_artifacts/results_scaleup_downstream.json`
(343 clips / 38,416 windows / 4 seeds): `speed_r2` floor **−0.4387** → pseudo-YT **+0.7264**;
`yaw_r2` 0.5505 → 0.7285; `ade_2s` 12.6098 → **4.9776**; `ci_excludes_0_all_seeds: true`.
`YT_DB_RETRY.md:322` scores it *"① HOLDS — DECISION-GRADE WIN"*. **Three defects block the
quote:**

1. ⛔ **Wrong cluster unit.** `bootstrap_gap_ci()` (`run_youtube_pilot_downstream.py:147-163`)
   resamples **clips**; the corpus is 343 clips from **55 videos / 43 channels**,
   `design_effect: 9.717`, `kish_neff: 35.298` ⇒ intervals ~3× too narrow. The same error
   **already manufactured a false separation** on `long_accel`: clip [1.018, 1.337] excludes 1;
   **video [0.976, 1.372] does not**.
2. ⛔ **The "ceiling" is not a ceiling.** Pre-registered criterion (c) fraction-of-ceiling =
   **1.0695 > 1** — the pseudo arm beats the real-label reference. The doc lists five violated
   preconditions (`:328-343`) and itself recommends treating the claim as **② PARTIAL/BOUND**
   (`:345-351`).
3. ⛔ **A separate NEGATIVE result**: yaw-rate spread ratio YT/PhysicalAI **0.547 [0.419, 0.931]**,
   excluding 1 under the corrected **video**-cluster bootstrap, replicated at n = 100/200/343 —
   the reconstructed corpus is measurably **less dynamic** than the real one.

**Effort** L · **GPU** campaign · **Deps** P3-15, P3-16, and PI **Q1** + **Q4**.

**Done when.** SPEC §8.1 executed: Arm A (parity fine-tune only) vs Arm B (P3 prefix +
**bit-identical** parity fine-tune), ≥3 seeds, four families per-family never pooled, paired
bootstrap **clustered on source video**; **plus the label-shuffled control** — if shuffled-label
pretraining helps as much as real, the gain is from pixels and the *product* claim fails even if
ADE improves. N fixed **in advance** and the verdict scoped to it, counted in **videos, not
clips**.

---

## T6 — PI-gated

### P3-19 — ⚖️ Licence decision brief (blocks all harvesting)
**Why.** ⛔ **Not my decision.** The tier framework exists and routes three rows *"to Sayed +
legal before any public or commercial step"* (`LICENSING_TIER_ANALYSIS.md:142-143`); raw frames
are `refuse` to re-host (`:31-48`).

**Effort** S to write · **GPU** 0 · **Deps** none to write; the PI to answer.

**Done when.** A brief posing SPEC §9 Q1(a)–(e) with options and consequences, filed in
`Project Steering/`, and **flagging the live discrepancy** in Q1(c): the analysis recommends
CC-BY-only, the pipeline shipped `--allow-noncc=True`, and 343/343 clips carry licence `None`.
⛔ Until (a)–(c) are answered P3 does not harvest at scale; the prior authorization is recorded
**SPENT** — *"Any further YouTube harvesting is a NEW decision for the PI"* (`LOOP_STATE.md:693-703`).

---

### P3-20 — Decide the longitudinal channel (Q2) and the cross-domain posture (Q3)
**Why.** `accel` is **1 of 2** operative control channels and sits at the null. The three
responses have very different costs and only the first is free.

**MEASURED evidence.** The **capacity control is decisive** and rules out the cheap answers: the
identical closed-form protocol on the *true* speed window recovers `accel` at **R² +0.92623
[0.88764, 0.95074]** while all 17 architecture/capacity arms sit at or below the **−0.0626**
null ⇒ **the input is the limit, not the head**. Longer context (2.5 s) does not help (−0.0665).
⚠️ **But separate the second cause first**: on PhysicalAI the **label** caps a perfect kinematic
estimator at **R² 0.188**.

**Effort** S to write · **GPU** 0 · **Deps** P3-9.

**Done when.** Options paper with costs; PI decision recorded in
`Project Steering/GOALS_AND_CLAIMS.md` with an ID.

---

### P3-21 — 🔴 The stranded corpus — recover, re-harvest, or accept the loss
**Why.** The only copy of P3's entire harvested output is on a pod, is not in the repo, and is
probably gone. This is the stranding rule's exact failure, at corpus scale.

**MEASURED evidence.** 343 clip latents only at `pod3:/workspace/tmp/yt_scaleup/latents/`,
described as *"(regenerable; transient by design)"* (`YT_DB_RETRY.md:376`); 20 clips / 2.8 GB at
`pod3:/workspace/tmp/yt_geo/clips/`. **Nothing on HF**
(`YOUTUBE_GEOCALIB_IDM_RESULT.md:78`, `YT_DB_RETRY.md:380`). Grepping the entire
`stack/experiments/pod-rescue-20260802/` tree for `yt_scaleup|yt_geo|youtube` returns **zero
hits** ⇒ not rescued. "Regenerable" is optimistic: re-harvest needs an egress **bot-blocked
2026-07-26** with authorization **SPENT**. And `run_scaleup_parallel.sh` truncates
`w*/harvest.log` with `>` each round, so **rounds 5–8 are unrecoverable** and the archived
round-9 logs are the only surviving block proof — *a re-run overwrites them*
(`YT_DB_RETRY.md:148-153`).

**Effort** S to probe, then PI decision · **GPU** 0 · **Deps** PI **Q4**.

**Done when.** A probe establishes whether pod3 still exists and holds the data; the round-9 logs
are **banked into the repo before anything re-runs**; and the PI records recover / re-harvest /
accept-loss.

---

## T7 — production

### P3-22 — The reconstruction demo, in the standing viz format
**Why.** TANITAD_PROGRAMME.md §1: *"Every product is production-oriented: spec'd, tested,
documented, demoable."*

**Effort** S–M · **GPU** dev-box · **Deps** P3-13.

**Done when.** A video on a *reconstructed* clip: camera overlay + metric BEV inset + decoded
tactical manoeuvre + strategic route/goal, with the reconstructed track drawn **against CAN**
where CAN exists, so the error is visible rather than narrated. ⚠️ `*.mp4` is git-ignored —
needs `git add -f`, a documented past stranding cause.

---

## Honest statement of what is missing

1. **The longitudinal control cannot be reconstructed.** Not "is noisy" — `long_accel` sits at
   the empirical null, 17 capacity/architecture arms all land at or below it, and the oracle
   control (**+0.926** given true speed) proves more model will not fix it. P3 today supplies
   **1 of the 2** operative control channels. ⚠️ And on PhysicalAI part of the cause is that the
   **label itself** caps a perfect kinematic estimator at **R² 0.188**.
2. **Metric scale has no implementation and no measurement**, and one candidate route is already
   **REFUTED on our corpus**. This is the largest unquantified risk in the product.
3. **Cross-domain transfer is a live REFUTED result**, not an open question. `go_no_go.PASS =
   false` has never passed; data-diversity is refuted; the registry says the thesis resting on
   the own-encoder is **not supported**.
4. **Extrinsics estimation does not exist**, and camera height is not observable from a single
   image without a metric anchor — so §4.1's scale coupling currently inherits an unbounded term.
5. **The shipped harvest has three live rule violations**: licence `None` on 343/343 against its
   own CC-BY recommendation; contamination **12.83 %** with a **zero** catch rate; and **63.3 %**
   of clips never reached the canonical frame because the geometry front-end had not landed.
6. **The one positive downstream result is scored at the wrong cluster unit** (clips, not the 55
   source videos; design effect **9.717**) and its "ceiling" is exceeded at **1.0695**, so it
   cannot be quoted as settled — the same unit error already manufactured a false separation
   elsewhere in this corpus.
7. **The harvested corpus is stranded on a pod and probably gone**, and re-harvesting is
   bot-blocked with the authorization spent.
8. **No provenance sidecar exists**, so reconstructed and real episodes are currently
   indistinguishable downstream.
9. **The raw-input floor (L1) has never been run.** L0, L2 and L3 do exist — and L3 is unusually
   strong — but without L1 the encoder's contribution is unseparated from the corpus's.
10. **The legal posture is unresolved and is a PI decision.** Listed last not because it matters
    least, but because it is the one item no amount of engineering closes.
