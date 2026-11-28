# SPEC_R1 (DRAFT for the Master Mind to register) — head-only retraining on a frozen refcv7 trunk

*Drafted 2026-10-04 by the R1 agent, BEFORE any arm was trained or scored. The Master Mind registers it (hash + time);
the R1 agent trains and scores nothing until told it is registered. DESIGN.md (same package) is the harness this
SPEC refers to; its cut point, cache and identity controls are not repeated here.*

## 1. Question and decision it feeds

Does retraining ONLY the planner-side heads of refcv7-r101-s0 @ 50,400 (20.15 M parameters: tactical behaviour
decoder, z_tac heads, anchored-diffusion decoder incl. its DDIM sampler, selection grafts, E9) on frozen trunk features,
with corrected labels / inputs, lift route following to PLAN §6's R1 gate? Yes ⇒ the information is in the trunk and the
planner-side levers can be validated (and warm-started) without trunk training. No ⇒ the trunk must learn it.

## 2. Data (fixed; DESIGN.md §4)

* TRAIN: 7,000 windows from 700 refcv7-train clips (350 turn-containing + 350 others, seed 0; ≤ 5 GT-turn windows +
  fill to 10 per clip); 1,727 GT-turn. Never an eval clip.
* EVAL (scored): 4,634 held-out eval139 windows = every GT-turn window (2,317; 849 L / 1,468 R; 38 episodes) + an
  equal seeded sample of the rest (2,317). GT classes by the route package's SPEC §3 rule.
* All arms read the SAME stored tensors (fp16 kv, int8 BEV) — identity control I-S measures the codec; H0 run through
  it is the paired baseline, so the codec is common to every arm.

## 3. Arms (all start from the 50,400 weights; identical trainable set, optimizer, schedule, steps, batch order)

| arm | lateral / longitudinal tactical target | nav input (train AND eval) | selection loss | decoder condition | role |
|---|---|---|---|---|---|
| **H0** | — (no training) | per-clip token | — | — | paired baseline; must replay its capture bit-identically (I-D) |
| **H1** | OLD: `lat_v7`/`lon_v7` as the trainer applies them (±2 s band, IGNORE elsewhere) | per-clip token | E9 selection CE (as trained) | as built | the harness's own fine-tuning / training-noise control |
| **H2** | DENSE on every window (r1_lib literals, below) | per-clip token | as built | as built | lever L1 |
| **H2s** | H2's labels **time-shuffled within clip** (seeded) | per-clip token | as built | as built | deliberate regression — must NOT clear |
| **H3** | DENSE | **announced time-localised** (A6 nav_ann, H = 6 s; A7 semantics; G1′ PASSED) | as built | as built | + lever L2 |
| **H4** | DENSE | announced | **listwise** over the emitted fan (below) | as built | + lever X1 (selector) |
| **H5** | DENSE | announced | listwise | **+ tactical action condition** (R8-4) | + conditioning |
| **H5c** | DENSE | announced | listwise | condition taken from a **deranged** other window of the batch | controllability control — must lose controllability |
| **H6** | H5 + **route checkpoint** input (R8-3) | announced | listwise | + checkpoint | **BLOCKED** until the Data FlyWheel's v9 checkpoint exists; echo control = drive to the checkpoint at constant v0 |
| H0n (diagnostic) | — (no training) | announced at inference only | — | — | separates an inference-input effect from training |

**Dense labels (H2…H6; literals, `code/r1_lib.py`, analytic tests `code/test_r1_lib.py` 7/7 incl. a mutation):**
lateral 3-way from the ego's OWN next 6 s — segment headings of origin→slot 5→…→slot 60; TURN_L if max ≥ +30°, TURN_R if
min ≤ −30° (both: larger |θ|), else LANE_KEEP; LANE_KEEP if the 6-s path < 5 m; IGNORE if slot 60 invalid. Written into
the v7 8-way head at LANE_KEEP / TURN_L / TURN_R (other 5 classes never targeted). Longitudinal on ONE declared window
[NOW, NOW+6 s]: HOLD if max v < 0.5 m/s; CREEP if max v ≤ 2.0; ACCELERATE if Δv ≥ +1.5 (the documented bar, K11);
BRAKE_TO if Δv ≤ −1.5; else CRUISE. Goal-token targets unchanged (anchor-bound). The SAME targets feed the z_tac heads
and the tactical behaviour decoder (both read `lat_v7`/`lon_v7`).

**Listwise selection target (H4…H6)** replacing E9's argmin-ADE CE (`refc_v3.selection_ce`, weight `SEL_V3_WEIGHT` 1.0)
on the E9 blended score over the reach-kept candidates: cost_k = mean over valid slots |s_k(j) − s_GT(j)| / 1.0 m
(s = cumulative arc length at slot j: the SPEED PROFILE) + |wrap(θ_k − θ_GT)| / 15° (terminal heading: the DIRECTION);
target p*_k ∝ exp(−cost_k / 1.0); loss = −Σ_k p*_k log softmax(score)_k.

**Tactical condition (H5):** zero-init `Linear(16 → 384)` on the 8 + 8 lat/lon action vector, added to the decoder's
`cond` (every decoder layer, the classifier pass and both DDIM passes). TRAIN: one-hot of the window's DENSE GT label
(teacher forcing; IGNORE → zeros). EVAL: the tactical behaviour decoder's own posteriors from the same forward,
detached. H5c: the vector of a deranged other window of the same batch (train) — eval as H5.

## 4. Training budget (fixed at registration from the timing smoke; stated here as the proposal)

AdamW, lr 5e-5 (half the run's peak 1e-4), 100-step linear warmup, cosine to 0, weight decay 0, batch 32, training
seed 0, **1,320 steps (= 6 epochs of 7,000 windows)** per arm. Proposal ceiling: ≤ 30 min of Thor GPU per arm
(ESTIMATED; the timing smoke — a few H1 steps, no scoring — replaces the step count before registration if a step
exceeds 1.35 s). Eval per arm: 4,634 windows × 2 sampler seeds, batch 1, head-only (ESTIMATED ≈ 6 min). Total for
H1–H5c: ≈ 7 × 36 min ≈ 4.2 h of Thor GPU, one job at a time under `thor_gpu.lock`.

## 5. Measures (per arm, EVAL, sampler seeds 0 and 1; four families per family, never pooled)

* **Route / LATERAL (the gate):** turn direction-correct pick on GT-turn windows (pick terminal heading class with the
  run's τ 10.35° = GT side); heading within 15° of GT at 6 s; terminal heading error; cross-track at 6 s; curvature
  and yaw-rate MAE 0–2 s (`four_families._seq_geometry`).
* **Selection:** fan contains a correct candidate; oracle-117 and random-in-reach picks on the same windows; pick regret.
* **LONGITUDINAL:** |along-track| and signed along-track at 6 s; speed MAE 0–2 s; ADE / FDE per class.
  Distance keeping: UNAVAILABLE (no lead tracks in this harness) — reported as absent with the reason.
* **TACTICAL:** lat / lon accuracy and macro-F1 of the tactical behaviour decoder AND the z_tac heads against the DENSE
  labels on every scored window, and against the OLD labels on the in-band windows; per-class support printed;
  `four_families.tactical_from_trajectory` κ (0–2 s).
* **STRATEGIC:** N/A — strategic layer OFF (PI ruling R5, 2026-09-27); reported as absent.
* **H5 controllability (R8-4 ii/iii):** on every classified scored window (GT path ≥ 5 m), force TURN_L / TURN_R /
  LANE_KEEP on the same scene (the decoder condition AND the tactical posterior feeding the tac8 selection prior);
  read the pick's direction class and the share of the 117 candidates with that direction.

**Estimator:** paired episode-cluster bootstrap over the eval episodes (B = 2000, seed-0 draws, the route package's
`boot_ratio` / `boot_mean`), every arm vs H0 on the same windows and the same sampler seed. The seed-1 column answers
"another INFERENCE draw"; the episode bootstrap answers "another draw of EPISODES"; "another TRAINING run" is answered
only by §6's replicate (H-ESTIM-SEED-1).

## 6. Bars (committed now)

An arm **clears the R1 gate** iff, on BOTH sampler seeds:
1. turn direction-correct pick **≥ 0.95** (point estimate; refcv7 0.84) AND its paired gain vs H0 has CI lower > 0;
2. heading within 15° on GT-turn windows **≥ 0.70** (refcv7 0.51);
3. GT-straight ΔADE vs H0 **≤ +0.05 m** with CI upper ≤ +0.10 m.

**Training replicate:** the first arm (in the order H2 → H3 → H4 → H5) that clears is re-trained with training seed 1 and
must clear again; otherwise it is reported as "cleared on one training seed" and not as a lever effect.
**Controls that must hold or nothing is quotable:** I-0, I-X, I-D PASS (DESIGN.md §5); **H2s must NOT clear** the gate
(if it does, the harness is broken); dense-label coverage ≥ 0.95 of scored windows and H2s's label agreement with the
GT-turn class must fall below 0.95 (the analytic-control twin of PLAN L1's validation).
**H5 controllability bar:** the pick follows the forced direction on ≥ 0.95 of classified windows for EACH of TURN_L,
TURN_R and LANE_KEEP (LANE_KEEP read on GT-straight + GT-turn windows), on both sampler seeds; **H5c must fall below
0.95 on at least one forced class AND below H5 with a separated paired CI** on the pooled forced-direction rate.

## 7. Reading rule (fixed now)

* **H1 clears** ⇒ fine-tuning the heads on this sample alone lifts route following: the trunk carries it, but the label
  and input levers are NOT attributed by R1 (each later arm is then read against H1, not H0).
* **H1 fails and the first of H2 / H3 / H4 / H5 to clear does so with H2s failing** ⇒ the information is in the trunk;
  the cleared arm names the levers a warm-started refcv8 needs (it does not need trunk training for route following).
* **No arm clears** ⇒ report each criterion's gap per arm; then the TRAIN-FIT diagnostic (each arm scored on 1,000 of
  its own training windows, secondary): an arm that cannot reach the gate on windows it was trained on says the frozen
  features or the head capacity do not separate the turn — the trunk must learn it (full / warm-started run with the
  trunk trainable). An arm that fits TRAIN but not EVAL says generalisation (more data or regularisation), not trunk.
* **H5 controllability passes and H5c fails** ⇒ the tactical layer CAN condition fan generation and selection on this
  trunk (R8-4 ii) — independent of whether the route gate is cleared.
* **H2s clears, or H5c passes controllability** ⇒ the instrument is broken; nothing from R1 is quotable.

## 8. Stamps every R1 number carries

OPEN-LOOP single-shot on logged frames (no closed loop); one TRAINING seed of the trunk; nav and the dense labels are
ego-future derived (the nav input is an ORACLE route supplier, optimistic on PhysicalAI by construction); the eval
turn set is 38 episodes, so a denser window set narrows within-episode noise only.
