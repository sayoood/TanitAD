# PRE-REGISTRATION — IDM cross-domain recovery (IDM_VIDEO_PRETRAIN_DESIGN §5)

**Written:** 2026-07-22, **BEFORE any result was seen.** This file is the committed
gate for the cheapest discriminating experiment on the inverse-dynamics (IDM) /
YouTube line. Program rule 5: both outcomes committed in advance; the number this
produces is the go/no-go for the whole IDM/YouTube spend.

**Author:** agent `idm-proof`. **Evidence class of every threshold below: HYPOTHESIS
(pre-committed decision rule).** Every measured number that follows lands in the
results JSON as `MEASURED (ours + artifact path)`.

---

## 0. The claim under test

> A supervised predictive IDM head on our **trained, frozen** flagship-v1 encoder
> recovers ego action/motion (speed, yaw-rate, steer, longitudinal accel) and the
> 2 s metric ego-trajectory **across a domain / camera-rig shift**, well enough to
> justify scaling the supervised IDM to pseudo-label YouTube.

The single dominant, well-documented failure mode for IDM pseudo-labeling is the
**domain/intrinsics gap** (IDM_VIDEO_PRETRAIN_DESIGN §4, [2602.02762, 2502.00379]).
This experiment measures it directly, cheaply, on data we already hold, before any
YouTube engineering.

## 1. Fixed apparatus (frozen before running)

- **Encoder:** flagship-v1 `flagship4b-speedjerk-30k` ViT + spatial readout, loaded
  from `Sayood/tanitad-flagship-4b-speedjerk/ckpt.pt` (MODEL_REGISTRY §1.2, the
  DEPLOYED arm, ADE@2s 0.4522). **FROZEN**; only `encoder.*` + `readout.*` weights
  used. `encode_window`: frames `[B,W,9,256,256] → z [B,W,2048]`. Note the encoder
  is purely visual — it takes NO action/speed channel, so the v1 speed-input
  (`action_dim=3`) is irrelevant to the IDM substrate.
- **IDM head:** small NON-CAUSAL temporal transformer over a window of `2k+1` encoder
  latents, `k=4` → 9 frames `z_{t-4..t+4}`, bidirectional attention (no causal mask),
  read out at the window CENTER `t`. Continuous regression (Huber), a few M params —
  a readout, not a backbone. Outputs at `t`: `speed`, `yaw_rate`, `steer`,
  `long_accel` (continuous) + the 2 s ego-frame metric trajectory at horizons
  `{5,10,15,20}` = `{0.5,1,1.5,2}` s.
- **Ground truth (free, CAN-derived, already in cache):** `speed = poses[t,3]`,
  `steer = actions[t,0]` (road-wheel rad), `long_accel = actions[t,1]` (m/s²),
  `yaw_rate = wrap(yaw[t+1]-yaw[t-1]) / (2·dt)`, `dt=0.1`. Trajectory targets =
  ego-frame waypoints `ego_frame(xy[t+h]-xy[t], yaw[t])` (the repo `_ego` /
  `refb_labels.waypoint_targets` convention exactly).
- **Parity firewall:** the IDM is a SIDE model. It reads frames + the encoder only;
  it does NOT touch the WM train corpus, its parity key `e438721ae894`, or the
  skip-hash `f09e44db`. Its own train/eval splits are by RIG / by CORPUS and are
  orthogonal to the WM parity selection.

## 2. Metrics (mirror VPT held-out R²)

Per target, on the eval window set: coefficient of determination
`R² = 1 - SS_res/SS_tot` (predicted vs CAN GT). Reported for **speed, yaw-rate,
steer, long_accel**. Trajectory: **ADE@2s** = mean over windows of the mean over the
4 horizons of `‖pred_wp_h − gt_wp_h‖₂` (metres). Also report MAE per scalar (R² is
distribution-relative; absolute error is the honest cross-corpus companion).

## 3. The two cross-domain contrasts (both run; both reported)

- **(#2) rig-A → rig-B, WITHIN PhysicalAI** *(cheap pre-probe)*. PhysicalAI-AV front-
  wide has two camera rigs by principal point: `cy≈543` rig-A vs `cy≈755` rig-B
  (`filtering.RIG_CLUSTERS`, split 650). Train the IDM on **rig-A clips only** (zero
  rig-B in training); eval on held-out **rig-B**. In-rig ceiling = eval on held-out
  **rig-A**. Isolates (closest to) pure intrinsics shift with the same corpus/labels.
- **(#3) PhysicalAI → comma2k19** *(the main, real rig gap — the go/no-go)*. Train on
  **PhysicalAI CAN only** (all rigs), zero comma2k19 in training; eval on held-out
  **comma2k19** (different vehicle/camera/intrinsics, already CAN-labeled). In-rig
  ceiling = eval on held-out **PhysicalAI val** (episode-disjoint from train).

## 4. COMMITTED DECISION RULE (frozen before any number was seen)

For each cross-domain contrast, with `X = {speed, yaw}`:

> **PASS** iff cross-domain **speed R² > 0.90 AND yaw-rate R² > 0.90 AND
> traj ADE@2s < 1.5 × the in-domain (same-rig / same-corpus) held-out ADE@2s.**
>
> **FAIL** otherwise.

- **PASS ⇒** the domain/intrinsics gap is tolerable ⇒ **recommend scaling the
  supervised IDM to YouTube** (proceed to canonicalize + IDM-label a YouTube slice).
- **FAIL ⇒** the domain-gap / intrinsics mode dominates ⇒ **do NOT spend on YouTube
  yet; first add the f-theta canon front-end + speed-prior scale head**
  (IDM_VIDEO_PRETRAIN_DESIGN §4) and/or a VO scale auxiliary, then re-test.

**Program go/no-go follows contrast #3 (PhysicalAI→comma2k19)** — the real rig gap and
VPT's held-out-R² protocol — with #2 (rig-A/B) as the corroborating pre-probe. If #2
and #3 disagree, that disagreement is itself the finding (it localises the gap to
vehicle/label vs pure intrinsics) and is reported, not averaged away.

The **in-rig / in-corpus held-out number is reported too, as the ceiling** — it says
how much of any degradation is the encoder/head vs the domain shift.

## 5. Pre-registered confounds (C6 discipline — named before reading the contrast)

The cross-domain contrasts vary more than one thing. Committed handling:

1. **`steer` R² is confounded across corpora** and is NOT in the PASS rule. comma2k19
   derives road-wheel steer with a **constant 15.3 steering ratio**
   (`comma2k19.STEER_RATIO`), whereas PhysicalAI uses `atan(wheelbase·curvature)`
   (`physicalai.WHEELBASE=2.9`). A steer-R² drop across #3 can be a units/derivation
   mismatch, not an encoder failure. speed + yaw (physically comparable, same SI
   units both corpora) carry the decision. steer/accel reported as secondary only.
2. **R² is distribution-relative.** comma2k19 (highway commute) and PhysicalAI speed
   distributions differ; equal absolute error yields different R². Both R² and MAE are
   reported so a narrow-distribution R² penalty cannot masquerade as a domain failure.
3. **rig-A/B is not guaranteed scene-identical.** Rig may correlate with
   geography/time-of-day; #2 measures "rig-A-trained transfer to rig-B", not literally
   "only intrinsics changed". Noted, not overclaimed.
4. **Trajectory scale in comma2k19** rides on the ENU/velocity pose derivation, a
   different pipeline from PhysicalAI's; ADE degradation may include pose-noise, not
   only encoder transfer. Reported alongside the speed/yaw R², which do not depend on
   the trajectory integration.

## 6. Protocol constants (frozen)

- Window: `k=4` (9 frames), one labelled window per valid center `t` (needs 4 frames
  of past, 4 of future, and 20 future poses for the 2 s trajectory), window stride 2.
- Head: `d_model 256`, 3 bidirectional layers, 4 heads, GELU MLP ×4; ~3 M params.
  Optimiser AdamW lr 3e-4, wd 0.01, cosine, ~6–10 epochs over the train windows,
  batch 256 on cached latents. Scalar targets standardised by TRAIN mean/std for the
  Huber loss; R²/MAE computed in raw physical units. Seed 0.
- Splits are clip/episode-disjoint. rig-A train/heldout split 85/15 by clip; comma is
  entirely held out; PhysicalAI val (80 eps) is the in-corpus held-out for #3.
- Encoder ckpt md5 recorded in the results JSON. One GPU under `gpu_lock.sh acquire
  idm-proof` on a NON-training pod (pod3, A40). Frozen-encoder latents are cached once
  then reused for every head fit (the encode is the only heavy step).

---

**Nothing below the decision rule may be edited after the first result is read.**
Results land in `results.json`; the verdict + recommendation in `REPORT.md`.

---

# RE-GATE ADDENDUM (2026-07-22, written BEFORE the re-gate ran)

The baseline gate FAILED (both contrasts). Sayed's directive + coordinator brief: build the two
pre-registered fixes and re-run **this exact gate** (same thresholds) as a 2×2, to learn which fix (if
any) recovers cross-domain transfer. Both outcomes committed here before any re-gate number was seen.

## R1. Disposition of fix #1 (f-theta canonicalization front-end) — determined by CODE READ, committed

The sibling front-end (`…/2026-07-22-youtube-idm-pipeline/ftheta_frontend_prototype.py`) canonicalizes
via `calib.focal_crop_resize` (rectilinear) and `calib.ftheta_crop_resize(center="principal")` (fisheye).
**These are the exact primitives that built the baseline cache** (`physicalai.py _decode_mp4` →
`ftheta_crop_resize(center="principal")`; `comma2k19.py _decode_video` → `focal_crop_resize`), both at
`f_eff≈266`, both PhysicalAI rigs principal-point-aligned (`build.log` `fallback=0`; `PARITY_OK`). So the
front-end's crop-canonicalization is **ALREADY APPLIED in the baseline** — the cross-domain failure
happened *with* it on. **Committed consequence:** the "+front-end" arms are a **no-op on this cached data**
(re-applying an already-applied crop cannot change the encoder input), and the rig-A→rig-B collapse
**cannot be intrinsics-driven** (both rigs share one f-theta poly + principal-point crop; speed
distributions measured near-identical). The only *stronger* variant is full rectilinear
`calib.ftheta_undistort`, which (a) needs the **native fisheye frames** (rolling-deleted; 4 mp4s left on
pod3) and (b) would feed the FROZEN, fisheye-crop-trained encoder **out-of-distribution** input — so it is
meaningful only paired with an encoder retrain, and is booked as a **bounded follow-up**, not a
frozen-encoder lever. This disposition is committed; the re-gate will *verify* the no-op by measurement.

## R2. Fix #2 (light-FT) — the runnable lever, thresholds unchanged

Unfreeze the **last 2 ViT blocks + final norm + readout** (freeze patch+pos+blocks[0:10], run that prefix
in `no_grad`), joint-train with the IDM head on the **training split only** (rig-A for #2; PhysicalAI for
#3), encoder-suffix lr **1e-5**, head lr **3e-4**, short warmup. Then eval **cross** (rig-B / comma) and
the **in-domain held-out** ceiling, encoding eval frames through the FT'd encoder. A paired **frozen arm on
the identical reduced train/eval sets** is run in the same script so the delta is apples-to-apples.

## R3. COMMITTED verdicts (frozen before the re-gate)

Same gate as baseline: an arm **PASSES** iff cross-domain **speed R² > 0.9 AND yaw-rate R² > 0.9 AND
traj ADE@2s < 1.5× that arm's in-domain held-out ADE@2s** (program go/no-go on #3, comma).

- **If light-FT PASSES the gate** on #3 → the fix is *adapt the encoder*; recommend light-FT before scaling
  to YouTube, and re-confirm on the full data.
- **If light-FT improves transfer but does NOT reach the gate** → partial lever; recommend the deeper fix
  (encoder retrain on undistorted, multi-domain data + speed-prior scale head) before YouTube.
- **If NEITHER fix reaches the gate** → committed reading: **the frozen-encoder supervised-IDM line needs a
  rethink** — a from-frozen readout on our PhysicalAI-only encoder does not transfer across a real rig gap,
  and the YouTube line should not proceed on this recipe. Report plainly; do NOT force a pass.

Re-gate results land in `results_regate.json`; the front-end no-op is verified by a measured f_eff check.

---

# MULTI-DOMAIN CO-TRAIN ADDENDUM (2026-07-22, written BEFORE the multi-rig run)

The re-gate committed reading was "retrain on multi-domain data". The **cheapest** discriminating test of
that direction — proposed in `Research/2026-07-22-encoder-strategy-and-vjepa2ac.md` §C.4 — resolves the
biggest open question **before** any GPU-day encoder spend:

> **Is the cross-rig collapse a DATA-DIVERSITY problem (fixable by training on more domains) or a
> REPRESENTATION problem (needs an expensive V-JEPA2-scale video-SSL encoder)?**

This decides whether the SSL bet (design option 2) is even necessary. V-JEPA2-AC (`PUBLISHED`,
arXiv:2506.09985 §limitations) reports the identical camera-pose sensitivity **at 1M+ h pretraining**, so
SSL scale does **not** buy rig-invariance for free — making this the right fork to test cheaply first.

## M1. The change — exactly ONE thing vs the re-gate

The re-gate trained on a **single** domain and tested cross. This co-trains on **multiple** domains jointly
and tests on a **held-out** one. Same IDM head, same harness (`run_idm_ft.py`), same frozen+light-FT arms,
**same §5 gate**. No parity impact; existing assets only.

- **Primary arm — {rig-A + comma2k19} → held-out rig-B.** Co-train on PhysicalAI rig-A **and** comma2k19
  jointly; test on **rig-B (never in training)**. Direct comparison to the single-domain re-gate
  rig-A→rig-B (frozen −3.21 / light-FT −1.65). *Does adding a second, very different domain to training
  recover generalization to a never-seen rig?* In-domain ceiling = rig-A held-out (same corpus).
- **Symmetric arm — {rig-A + rig-B} → held-out comma2k19.** Co-train on both PhysicalAI rigs; test on
  comma (fully held out). Robustness check (≈ the baseline #3 setup, re-confirmed under balanced co-train).

Balanced by ~train-window count (comma clips are 300-frame vs PhysicalAI 199). Light-FT stays on (the
re-gate's best frozen-encoder lever, last 4 blocks, lr 5e-5). Secondary read: the design's speed-prior
scale head is deferred (speed R² is scale-invariant to a global scale, so a scale head cannot change R²;
it would change MAE/ADE — noted, not run here).

## M2. COMMITTED reading (frozen before any multi-domain number)

Gate unchanged: PASS iff held-out cross **speed R² > 0.9 AND yaw R² > 0.9 AND ADE@2s < 1.5× in-domain**.

- **PASS (held-out cross speed R² > 0.9)** ⇒ the collapse is **DATA-DIVERSITY**: multi-domain co-training of
  OUR encoder fixes it, **no V-JEPA2-scale SSL needed**. IDM-v2 recipe = multi-domain-trained encoder +
  speed-prior scale head → proceed toward YouTube. Retires design option (2) as unnecessary.
- **FAIL** ⇒ the collapse is **REPRESENTATIONAL**: data diversity alone is insufficient → this is the
  **pre-registered justification** for the expensive video-SSL (V-JEPA2-style) encoder investment.
- **PARTIAL (recovers materially but misses 0.9)** ⇒ report the trend + how much diversity closed the gap;
  the go/no-go on SSL is then a magnitude judgement, reported plainly, not forced either way.

The **delta vs the single-domain re-gate** (rig-B cross speed −1.65 → multi-domain ?) is the headline
number. Results land in `results_multirig.json`; the verdict is reported plainly regardless of direction.
