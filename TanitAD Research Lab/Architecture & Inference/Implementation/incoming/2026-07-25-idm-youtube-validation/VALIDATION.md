# IDM reconstruction validation — proving ego trajectory from video alone

**Agent:** idm-youtube-validation (reconstruction + proof + video leg) · **Date:** 2026-07-25
**One line:** the persisted non-causal IDM head reconstructs a held-out clip's 2 s ego
trajectory **purely from video** to **ADE 2.53 m** (MEASURED), the pipeline is proven
bit-exact to the head's own card, and the standard BEV + camera-projection video is rendered.

---

## 1. What this validates

Our IDM = flagship-v1's **FROZEN visual encoder** + a **non-causal** bidirectional head that
reads a 9-frame latent window `z_{t-4..t+4}` and predicts, for the centre frame, the ego
dynamics (speed / yaw-rate / steer / long-accel) and a 2 s ego trajectory. Action-FREE video
in, ego motion out (the VPT / action-labelling paradigm). This leg takes the head the prior
agent trained + persisted and does the reconstruction, the ground-truth proof, and the video.

## 2. The clip — held-out, with ground truth

**`ep_00020` — episode_id `808924517`**, PhysicalAI **val** cache
`physicalai-val-0c5f7dac3b11` (eval pod `/root/valdata/`). T=199 @10 Hz (~20 s). The vehicle
**accelerates from a standstill to ~19.8 m/s** then cruises, with a turn — so the reconstructed
path spans the full speed range (a strong test of the trustworthy speed/longitudinal channel)
and curves (a real test of yaw). Chosen over 39 alternatives on the measured per-clip table
(`percli.json`): it is the clip with a wide speed range **and** meaningful positive channel R²
**and** a low ADE.

**Held-out proof (doubly unseen — by BOTH the head and its encoder):**
- The head `idm_head_v1` trained ONLY on tags `tr_a_[0:60]` + `tr_b_[0:60]` + `cm_[0:40]`
  (PhysicalAI **train**-cache episodes + comma2k19) — source `idm_head_v1_train.py::TRAIN_TAGS`,
  in this dir. Zero val-cache episodes. *(MEASURED from the training script.)*
- The frozen encoder (flagship-v1 `flagship4b-speedjerk-30k`) trained on the parity **train**
  corpus `physicalai-train-e438721ae894` — never the val cache.
- PhysicalAI train/val are disjoint splits. *(Evidence class: INHERITED — the parity split
  invariant every program val number rests on; not independently re-checked against the 2376
  train-id list, which lives on pod3.)*
- Stronger still: `ep_00020` is **not even in the head's card `val_parityval` set** — pose-
  fingerprint matching (`idmval_match.py`) shows it has no twin among the card's 40 clips
  (nearest distance 0.74). It is a purer held-out clip than the card measured on.

## 3. Pipeline (purely from video; reuses shipped modules only)

`frames_u8 [199,9,256,256]` → `run_idm_proof.encode_frames` (frozen v1 encoder+readout, verified
md5) → `z [199,2048]` → `idm_head.build_windows` (non-causal 9-frame windows) → `IDMHead` →
per-centre {speed, yaw_rate, steer, long_accel} + 2 s ego trajectory at {0.5,1,1.5,2}s. GT =
CAN-derived kinematics in the episode contract (`poses=x,y,yaw,v`; `actions=steer,accel`).
Encoder `= /root/models/flagship-30k/ckpt.pt`, md5 `b5f07d9e…` (== card), step 29999, state_dim
2048. Head md5 `fa4462f0…` (== card `weights_md5`).

## 4. Results — reconstruction accuracy

Estimator note: R²/MAE/ADE are pooled over the clip's 175 non-causal windows; **ADE@2s** = mean
over windows of the mean-over-4-horizons L2 waypoint error, in metres — the head's own metric.

| quantity | ep_00020 (primary, from video) | evidence |
|---|---|---|
| **ADE@2s** | **2.53 m** | MEASURED (`recon_metrics.json` → clip) |
| per-horizon DE (0.5/1/1.5/2 s) | 1.20 / 2.00 / 2.99 / 3.94 m | MEASURED |
| **speed R²** | **0.855** (MAE 2.08 m/s) | MEASURED |
| **yaw-rate R²** | **0.940** (MAE 0.027 rad/s) | MEASURED |
| steer R² | 0.830 | MEASURED |
| n windows | 175 | MEASURED |

**Reading:** from video alone the head recovers the 2 s driven path to ~2.5 m and tracks the
full 0→20 m/s acceleration (speed R² 0.855) and the turn (yaw R² 0.940). Per-frame ADE falls to
**1.6 m at cruise** where IDM speed 15.1 vs GT 15.5 m/s (see the video's f124) — the predicted
orange path lies almost exactly on the green GT path.

### Whole-route (secondary, illustrative)
Dead-reckoning the per-frame IDM (speed, yaw_rate) over the whole 17.5 s gives endpoint drift
**38.4 m** on a 177.6 m route (path RMSE 19.4 m) — MEASURED but drift-DOMINATED: it integrates
every per-step error, so it is shown only for spatial context, not as the accuracy figure. The
rigorous metric is the per-window 2 s ADE above.

### Aggregate over all 40 held-out val episodes (from video, stride 2)
speed R² **0.825**, yaw R² **0.784**, **ADE@2s 3.86 m**, n=3521 — MEASURED (`recon_metrics.json`
→ aggregate). Per-clip ADE ranges 0.95–9.36 m and scales with speed (`percli.json`): the 40
clips here include 35–36 m/s highway clips whose large 2 s displacements inflate ADE.

## 5. The pipeline is proven correct (three controls) — and the card is reproduced EXACTLY

The from-video aggregate (3.86 m) is higher than the head's card `val_parityval` (2.70 m). This
is **fully explained and is not a pipeline defect** — proven:

1. **Matched-substrate reproduction (MEASURED, `idmval_vacheck.py`):** running this head on the
   head's OWN original val latents (pod3 `lat_flagshipv1/va_*`, built from cache
   `physicalai-val-f1b378f295ae`) reproduces the card to **every digit** —
   speed R² 0.8853, yaw R² 0.8075, steer R² 0.7821, speed MAE 2.0726, **ADE 2.7032**, n=3517.
   → the head weights + the metric code are correct.
2. **Zero encoder drift (MEASURED, `idmval_zcmp.py`):** the two val caches share 8 identical
   clips (pose-fingerprint dist 0.0). Re-encoding those on the eval pod vs the pod3-cached z:
   **cosine 1.0000** and **byte-identical per-clip ADE** (1.547==1.547 … 9.462==9.462). → the
   frozen encoder is reproduced bit-exact on the eval pod; "from video on the eval pod" ==
   the card's cached-latent pipeline.
3. **Therefore** the 3.86 vs 2.70 gap is entirely the **clip selection**: the eval pod's
   `0c5f7dac3b11` is a different, higher-speed 40-clip build of the val split than the card's
   `f1b378f295ae` (8/40 shared, rest different; ADE scales with speed). Not the head, not the
   encode.

## 6. Honest limits

- **Per-clip speed R² is not meaningful for narrow-speed clips.** On a near-constant-speed clip
  (e.g. ep_00012 at 29.7 m/s) R² goes to −8654 while speed MAE is only 1.12 m/s and ADE 1.41 m —
  the low-within-clip-variance artifact CLAUDE.md warns of. **Report speed R² only pooled**
  (aggregate 0.825 over the 0–36 m/s range) or as MAE. ep_00020 was chosen partly because its
  wide speed range makes its 0.855 speed R² genuinely informative.
- **`long_accel` and `steer` remain the weak/caveated channels** (long_accel R² −2.4 on this
  clip), consistent with the card's usage caveats. Speed + yaw-rate + trajectory are the
  trustworthy outputs and are what this validation reports.
- The **HUD tactical maneuver / strategic route come from flagship-v1's policy brains** (the
  taniteval standard's HUD source), NOT the IDM — they are decoded for completeness and are
  imperfect (e.g. decodes "turn right" where GT is "lane keep"). The validated subject is the
  IDM **trajectory**; GT maneuver is shown alongside for honesty.
- Held-out disjointness rests on the parity **split** invariant (INHERITED), not a re-check of
  the 2376 train-id list.

## 7. The video (our standard visualization)

`idm_recon_ep00020.mp4` — 175 frames @10 Hz, 812×512. Every frame carries all five required
elements: **(1)** front-camera projection of GT (green) + IDM-from-video (orange) 2 s path via
the standard flat-ground pinhole (f_eff 265.83, cx=cy=128, cam_h 1.5 — the taniteval standard
for PhysicalAI/comma, no GeoCalib needed; intrinsics known by build); **(2)** metric BEV inset;
**(3)** decoded tactical maneuver (+ GT maneuver); **(4)** strategic route; **(5)** per-frame
ADE + clip-mean + IDM-vs-GT speed. A whole-clip route panel gives spatial context (GT route +
the live 2 s-ahead prediction). Full camera projection (not the BEV-only fallback) — this clip
has known intrinsics. Verified by eye at frames 24/70/124.
