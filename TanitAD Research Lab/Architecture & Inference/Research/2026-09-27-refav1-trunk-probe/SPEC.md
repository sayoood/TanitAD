# SPEC — does the frozen refav1 trunk carry trajectory information BEYOND t0 kinematics? (pre-registered)

**Written 2026-09-27 ~14:05 Berlin (dev-box clock; first staged 14:07:38) by the TrainingFlyWheel, BEFORE any probe is fitted.** Staged, never committed.
The per-window features are being extracted while this is written (extraction output is inputs, not results); no
readout has been fitted, scored or looked at.

## 1. Why this, and why now (Rule Zero: the next lever after a FAIL)
The pre-registered full-grid test `H-REFAV1-LONCOMB3-FULL` FAILED (FAIL-WORSE, both inference seeds;
`../2026-09-27-refav1-fullgrid-loncomb3/RESULT.md`). Its exploratory follow-up eliminated every zero-training PLANNER
lever on this checkpoint: on the same damped path the planner's longitudinal profile loses to holding the measured
acceleration (+0.078 / +0.095 m ADE, separated), and its lateral decisions lose to planning straight (the shipped
cost, which always plans κ = 0, beats `loncomb3` laterally by 0.20 m cross-track). The world model's lateral
cost contribution is 1.63e-10 (`refa_v1.py:403`, `D-REFAV1-COST-SURFACE`), so no cost re-weighting can make the
planner steer from vision. ⇒ the next lever is a TRAINED trajectory readout (PI requirement R5: a 6 s combined
trajectory), and the question that decides whether a heads-only retrain is worth storage and compute is this probe.
The planner was never given one: the checkpoint's proposal head is untrained (`w_aux_head 0.0` in `config.json`).

## 2. Hypothesis
> **`H-REFAV1-TRUNK-TRAJ`** — the frozen refav1 step-21,109 trunk state at t0 (`pooled`, exactly what `plan()`
> feeds its heads: `encode(feats).mean(tokens)[:, -1]`, 1,024-d), together with the t0 kinematics, predicts the
> 2 s future trajectory better than the t0 kinematics alone, on HELD-OUT episodes.

## 3. Data, inputs, and what each arm may see
- Surface: the 141-episode v7.2 EVAL grid (`refav1-fp8-eval`), windows at `--window-stride 4` (2,399 windows,
  superset of the 282-window stride-40 grid; the extractor's GT and floors are **bit-identical** to the banked A1
  dump on every shared window — checked on 2 episodes before this SPEC).
- Kinematics `kin` = (v0, a0, κ0) at t0: v0 measured (PI ruling 2026-09-02), (a0, κ0) the backward differences
  `ha0_ext` holds (past measurements only). Ego state at inference is admissible for the planner under that
  ruling; the vision-only rule (2026-08-03) binds the scenario CLASSIFIER, and no classifier output enters any arm.
- Target: the residual `r = g − kd_x` over the strongest measured floor `kd_x` (the damped floor's path —
  0.5·`ha0` + 0.5·`ha0_ext` — re-timed to `ha0_ext`'s travelled distance; EXPLORATORY 2026-09-27 on the stride-40
  grid: `kd_x − ha0_ext` ADE −0.2282 [−0.2946, −0.1702]).

| arm | features (before PCA) | what it answers |
|---|---|---|
| **P0 `kd_x`** | none (zero residual) | the floor; must equal `kd_x` exactly (known-value control) |
| **P1 `const`** | intercept only | the no-information control |
| **P2 `kin`** | `kin` + (v0², v0·a0, v0·κ0, v0²·κ0, a0·κ0) | a LEARNED kinematic readout — the floor with the same affordance as the arms |
| **P3 `raw`** | mean-pooled raw DINOv3 features of frames t0, t0−1, t0−2 (3×1,024) + P2's features | does refav1's adapter add anything over the frozen encoder, at the SAME pooling? |
| **P4 `trunk`** | `pooled` (1,024) + P2's features | THE ARM |
| **P5 `shuf`** | P4 with the trunk rows permuted across training windows (the targets and kin stay aligned) | leakage / capacity control: must NOT beat P2 |

Readout: **ridge** on standardised features; the image block reduced by **PCA** (basis fit on the training folds
only). λ ∈ {1e-2 … 1e4} (13 log-steps) and PCA m ∈ {8, 16, 32, 64, 128} chosen by **inner 4-fold
grouped-by-episode CV on the training folds only**. Outer: **5-fold grouped-by-episode CV** (fold = episode index
mod 5), so every window's prediction is out-of-fold. `n` and `d` printed per arm and fold.
Function class: LINEAR. A negative is "not linearly decodable from mean-pooled state", never "not learnable".

## 4. Bars — committed now
**Estimator:** paired episode-cluster bootstrap over the 141 episodes, n_boot 2,000
(`taniteval/tools/refav1_paired_delta.py` on stride-4 dumps built from the out-of-fold predictions), all four
families; STRATEGIC UNAVAILABLE (no route label on this surface). **Tier:** T1-equivalent — emitted from t0 state
only, no future input, no action rollout. ⚠️ Windows at stride 4 overlap in time (0.8 s apart); the episode
cluster is the resampling unit, so the overlap is inside a cluster and does not shrink the interval.

- **PRIMARY `BAR-T1`:** `P4 − P2` ADE@2 s < 0 with the CI excluding 0.
- **`BAR-T2` (the floor):** `P4 − P0` ADE@2 s < 0 with the CI excluding 0.
- **Validity (VOID if violated):** P0 reproduces `kd_x` exactly; `P5 − P2` must NOT be separated-better (if the
  shuffled trunk beats the kinematic readout, the gain is capacity/leakage, not information); P1 must not beat
  P2 separated.
- **Secondary, never pooled:** `P3 − P2` and `P4 − P3` (ADE, four families); `P2 − P0`; 6 s ADE/FDE on the
  windows whose 6 s GT exists (n stated), same arms, same folds.

## 5. Outcomes, all committed
| outcome | reading | consequence |
|---|---|---|
| **POSITIVE** | BAR-T1 and BAR-T2 met | the trunk carries trajectory information a trained readout can use and it clears the strongest kinematic floor ⇒ the next lever is a HEADS-ONLY R5 trajectory head (residual over a kinematic prior) on the frozen trunk; its training input is `pooled` (~4 KB/window), not the ~155 GB fp8 cache |
| **PARTIAL** | BAR-T1 met, BAR-T2 not | information exists but a linear readout does not clear the floor ⇒ a nonlinear head (MLP, same CV) is the next arm |
| **NEGATIVE** | BAR-T1 not met | the mean-pooled trunk adds nothing linearly decodable over kinematics ⇒ a retrain must change the trunk or the pooling (spatial tokens), not only the heads |
| **VOID** | a validity control fails | no reading |

`P4 − P3` is reported under every outcome; if P3 ≥ P4 the adapter adds nothing over raw DINOv3 at this pooling.

## 6. Provenance
Checkpoint `refav1-b1-v72-ep3-speed` step 21,109, md5 `1189bc020018c2c67ce03d566c390285`; code
`stack/` + `taniteval/` at `b3f7ea6f` (refav1 files unchanged at origin `c36b6ddd`); extractor, probe and dumps
banked beside this file. Dev box CPU only (the GPU is another session's). Not a training run of any trainer:
a ridge readout on frozen features, fitted per fold; nothing is deployed from it.

## 7. Amendment A1 (2026-09-27, dev-box clock, first staged 14:16:23 Berlin) — the committed NEGATIVE branch's next arm: SPATIAL pooling
**Added AFTER §4's primary result was read and BEFORE any spatially pooled feature exists.** §5 committed that a
NEGATIVE sends the next lever to *"the trunk or the pooling (spatial tokens), not only the heads"*. MEASURED (§4 as
written; `raw/probe_result.json`): `P4 − P2` ADE@2 s **+0.0042 [+0.0016, +0.0071]** (BAR-T1 NOT met — the trunk
readout is separated-WORSE than kinematics alone); controls valid (`P0` exact, `P5 − P2` +0.0022 separated-worse,
`P1 − P2` +0.0434 separated-worse); inner CV chose the SMALLEST PCA size (m = 8) in every fold at 2 s.
**The mean over all 640 tokens is the coarsest pooling that exists** — and it is exactly what refav1's heads read
(`plan()`'s `pooled`). This amendment tests whether the information is present but destroyed by that pooling.

- **Token layout:** DINOv3 patch tokens, specials dropped, row-major 16 × 40 (`dinov3_fp8_encode_ship.py:110-125`,
  `DINOV3_GEOMETRY` `grid_h 16, grid_w 40`). **Regions: 4 × 10** (4 bands of 4 token-rows = 64 px; 10 columns of
  4 token-columns = 12° of the 120° cylindrical azimuth), each region the mean of its 16 tokens.
- **P6 `trunk_sp`:** `encode(feats)[:, -1]` pooled 4 × 10 (40 × 1,024) + P2's kinematic features.
- **P7 `raw_sp`:** raw DINOv3 at frames t0 and t0−1, each pooled 4 × 10 (2 × 40 × 1,024) + P2's features — the
  matched floor (the trunk's `tmix` sees ≤ 3 frames; raw gets the two frames that carry motion).
- **P8 `shuf_sp`:** P6 with trunk rows permuted across training windows (validity control).
- Readout, folds, λ/m grid, inner CV, estimator: UNCHANGED from §3–§4 (PCA by the Gram trick because d is large;
  same basis-on-training-rows-only rule).
- **PRIMARY `BAR-S1`:** `P6 − P2` ADE@2 s < 0, CI excluding 0. **Validity:** `P8 − P2` not separated-better.
  **Secondary:** `P6 − P7` (adapter vs raw at the same pooling), `P6 − P4` (spatial vs global mean), 6 s ADE/FDE.
- **Outcomes:** POSITIVE ⇒ the information is in the token field and the global mean destroys it: the next lever is
  a heads-only R5 trajectory head that READS THE TOKEN FIELD (region/attention pooling), trained on the frozen trunk.
  NEGATIVE ⇒ no linearly decodable trajectory information beyond kinematics at 12° × 64 px either; the next arm is a
  NONLINEAR readout (MLP over the same regions, same CV) — function class stated either way.

## 8. Amendment A2 (2026-09-27, dev-box clock, first staged 14:32:15 Berlin, BEFORE any train-split feature exists and BEFORE A1's result was read) — TRAIN → EVAL, the power fix
**Why.** §4 and A1 fit every readout on ~113 EVAL episodes per fold (5-fold CV inside the 141). A negative there cannot
separate "the trunk has no trajectory information" from "113 episodes cannot fit a readout over it" — inner CV chose
the SMALLEST image capacity in every fold, which is also what a power limit looks like. A heads-only retrain would fit
on the refav1 TRAIN split, so the decision-grade question is asked in that regime.
- **Fit set:** the first **600** clips, by sha12 order, of the refav1 TRAIN split (B1's 4,713 source episodes on Thor
  minus the 141 eval clips = 4,572; the set is checked equal to the v7.2 TRAIN label release's clip set before use),
  windows at stride 8. Features are produced by EXACTLY the eval path: DINOv3-L bf16 → fp8 e4m3
  (`dinov3_fp8_encode_ship.encode_episode`, imported), the same extractor, the same trunk. Read-only on Thor (pull
  only; nothing is written there). ⚠️ A PROBE READOUT, not a model arm: the subset selection re-selects nothing that
  any training arm is compared on.
- **Scored set:** all 141 EVAL episodes at stride 4 (the identical 2,399 windows of §4/A1) — no eval window is ever
  fitted on.
- **Arms:** P0 `kd_x`; P2 `kin`; P4 `trunk` (mean); P6 `trunk_sp` (4 × 10); P3 `raw` (mean, 3 frames); P7 `raw_sp`
  (4 × 10, 2 frames); P5 / P8 shuffles of P4 / P6; plus **P9 `trunk_sp_mlp`**: a one-hidden-layer MLP (256 units,
  GELU, weight decay and PCA m selected by grouped inner CV on the fit set, early stopping on an inner fold) over P6's
  inputs — the NONLINEAR function class A1 committed to. Ridge grid, inner CV (4-fold grouped by episode, on the fit
  set only) and estimator unchanged.
- **PRIMARY `BAR-A2`:** `P6 − P2` ADE@2 s < 0, CI excluding 0 (paired episode-cluster bootstrap over the 141 EVAL
  episodes). **Secondary:** `P4 − P2`, `P9 − P2`, `P6 − P7`, `P9 − P6`, `P2 − P0`; four families; 6 s ADE/FDE.
  **Validity:** P0 exact; `P5/P8 − P2` not separated-better; the eval-side floors bit-identical to §4's dumps.
- **Outcomes:** POSITIVE ⇒ the trunk carries trajectory information a readout FITTED ON TRAIN extracts on held-out
  eval: the heads-only R5 head is the next build (its inputs cost ~4 KB/window mean-pooled or ~80 KB spatial, not the
  ~155 GB fp8 cache). NEGATIVE at 600 train clips, linear AND nonlinear ⇒ refav1's frozen trunk does not carry
  decodable trajectory information beyond t0 kinematics at this scale; a heads-only retrain is not the lever, and the
  trunk itself (its training objective) is — stated with the function classes tried and n.

## 9. Amendment A2′ (2026-09-27, staged before `probe_te` has run — the train features are still being encoded)
**Descriptive secondary for A2, no bar:** the pooled 2 s mean is dominated by windows where kinematics already work
(PUBLISHED: `2305.10430`, `2312.03031`), so A2 also reports `P6 − P2`, `P9 − P2` and `P2 − P0` (ADE and FDE, 2 s and 6 s,
paired episode-cluster bootstrap, n stated) inside strata defined from the GT path ONLY, fixed now:
- **turn:** |heading change of the GT path over the horizon| > 15°; **straight:** < 5°;
- **stop:** v0 > 3 m/s and GT terminal speed < 1 m/s; **accelerate:** GT terminal speed − v0 > +1.5 m/s;
- **hard:** `kd_x`'s own ADE above its 75th percentile on the eval windows.
Heading and terminal speed are finite differences of the last two GT points (0.2 s grid). A stratum with fewer than 10
episode clusters is reported as UNDERPOWERED (n stated), never read.

## 10. Amendment A3 — the CAPACITY controls for the nonlinear arm (staged BEFORE either control has been run)
**Why, and what is already known.** A2 read (MEASURED, `raw/out_te/verdict_a2.txt`): PRIMARY `BAR-A2` (linear, `P6 − P2`)
NOT MET (+0.0012 [−0.0014, +0.0040]); validity controls OK. The pre-registered NONLINEAR arm `P9` (MLP over the 4 × 10
trunk field + kinematics, fitted on TRAIN) is separated-better than `P2`: 2 s ADE −0.0236 [−0.0343, −0.0144], 6 s ADE
−0.4037 [−0.5502, −0.2569], 6 s FDE −1.3279 [−1.7641, −0.9046]. ⛔ `P2` is a LINEAR readout, so `P9 − P2` confounds the
trunk's information with the MLP's capacity on the 8 kinematic features — the C6 confound ("an arm that wins on capacity
read as winning on mechanism"). No claim about the trunk is made from `P9 − P2`.
- **P10 `kin_mlp`:** the IDENTICAL MLP (width 256, GELU, zero-init output, AdamW lr 1e-3, weight decay {1e-4, 1e-2},
  epochs chosen on the inner folds, same grouped inner CV on TRAIN) on the kinematic features ONLY — the floor with the
  same function class and capacity as `P9` minus the trunk.
- **P11 `shuf_sp_mlp`:** `P9` with the trunk rows permuted across training windows (drawn after P9, arm order).
- **PRIMARY `BAR-A3`:** `P9 − P10` ADE < 0 with the CI excluding 0 **at 6 s** (R5's horizon, where kinematics fail) AND
  `P9 − P10` ADE ≤ 0 at 2 s (not separated-worse). **Validity:** `P11 − P10` NOT separated-better at either horizon.
  Secondary: FDE, the A2′ strata, four families at 2 s.
- **Outcomes:** POSITIVE ⇒ the frozen refav1 trunk carries trajectory information beyond t0 kinematics that a NONLINEAR
  readout fitted on the train split extracts on held-out episodes — the heads-only R5 head (retrain option H, dev box,
  no new storage) is the next build. NEGATIVE (`P9 ≈ P10`) ⇒ the A2 gain was nonlinear KINEMATICS, not vision; the lever
  becomes a learned nonlinear kinematic prior plus a trunk that must change (option T). VOID ⇒ `P11` beats `P10`
  (capacity leak).
