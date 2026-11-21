# PREREG_WPD_A1 — amendment A1 to PREREG_WPD_PROBES.md (P-BOX), registered under the Master Mind's ruling

*WP-D, 2026-10-04. Written BEFORE the P-BOX capture finished and BEFORE any P-BOX arm produced a number (no arm has
run; the capture has not started; P-GRAD is queued on the Thor lock). Base registration: `PREREG_WPD_PROBES.md`,
sha256 c054190bee4df5f266c99318d5374fbad9821a17ae7e44830c4b5f8feffc6f87, 2026-10-04T12:31:35Z. Everything not named
below is unchanged.*

## The ruling (Master Mind, 2026-10-04 ~14:58 Berlin), quoted

> Master Mind rulings on your three P-BOX items. All three were caught before any arm number, which is the right time.
>
> 1. **R-pres0: amendment A1 APPROVED as you propose.** The A1 must-fail control resets the presence row (weight row
>    zeroed, bias = logit(0.01)) AND sets the presence loss weight to 0. Scores become constant, so AP@2 m must fall
>    ≥ 50 % vs PB0. This is a structural known value.
>    - Also run the registered R-pres0 and report it as written.
>    - The registered R-pres0 can no longer VOID P-BOX: under a warm start it is ill-posed. If it fails to fall, label
>      it "ill-posed under warm start (A1 governs)". This is a correction of the instrument, not a moved bar.
> 2. **R-memshuf: CONFIRMED.** The memory is permuted across the windows of each batch in training AND at eval. Bar
>    unchanged (AP@2 m ≤ 0.05).
> 3. **PB1 CDN: APPROVED.** Use centre noise only, plus one learned DN content vector replacing DINO's label-noise
>    embedding. Our anchors carry (x, y) only, so size and yaw noise have no input path. Disclose both in RESULT.
> 4. **Cache 5.1 GB: approved.** Keep bev_feats (PB4h and PB8 need them); Thor has ~52 GB free.
> 5. **C-cache fidelity control: approved and binding.** The 50,400 decoder on the cached memory must reproduce EVAL T0
>    AP@2 m 0.2483 within ±0.003 before any arm runs.

## A1 — the binding text

1. **R-pres0z (governing must-fail):** PB0 with the presence logit row of the box head RESET (weight row = 0,
   bias = logit(0.01)) AND the presence loss weight 0, for the full 4,000 steps. Must read AP@2 m (raw) ≤ 0.5 × PB0's
   AP@2 m (raw), else P-BOX is VOID. The registered **R-pres0** (presence weight 0, no reset) also runs and is reported
   as written; it no longer voids P-BOX; if it does not fall ≥ 50 % it is labelled "ill-posed under warm start
   (A1 governs)".
2. **R-memshuf:** the decoder memory of each batch is rolled by one window across the batch (a derangement) in
   training AND at evaluation. Bar unchanged: EVAL AP@2 m (raw) ≤ 0.05.
3. **PB1 CDN:** 3 denoising groups; per group the window's VIS-1 positives (≤ 64) as noised positives (centre
   ± U(0, 0.4)·max(l, w) per axis) and as noised negatives (centre ± U(0.4, 1.0)·max(l, w) per axis, random sign);
   query content = ONE learned DN vector (initialised to the mean of the trained query table); positives
   reconstructed with the fixed assignment through `box3d_set_loss` + focal presence target 1, negatives presence
   target 0; DN queries see their own group and the matching group 0, nothing sees them; training only. Size and yaw
   noise are NOT applied (no input path) — disclosed in RESULT.
4. **Cache:** memory int8-affine per (window, image | BEV block, channel) + bev_feats int8-affine per (window,
   channel). EVAL + TRAIN ≈ 5.1 GB (approved). ⚠️ **Disclosed for the Master Mind's objection, not covered by item 4
   of the ruling:** the registered P = R gate FIT SET is TRAIN-DIAG (1,112 windows / 139 train clips), which is not in
   the TRAIN capture; capturing it adds ≈ 1.3 GB (≈ 6.4 GB total, Thor 52 GB free). Without it the count metrics
   (boxes per detected object, conf_ratio at the gate) could only be fitted on the arms' own training windows.
5. **Controls that bind before any arm runs** (all three must PASS, else no arm runs):
   * **C-cache:** the refcv7-50,400 decoder, unchanged, on the CACHED EVAL memory → AP@2 m (class-agnostic, raw)
     within **0.2483 ± 0.003** (the diagnostics' T0, `…/2026-10-04-refcv7-map-box-diagnostics/raw/B_box.json`).
   * **C-fwd** (added by WP-D): this probe's decoder pass (`pbox_arms.Arm`, PB0 configuration) equals the library's
     `slot_query_select.anchored_forward` on the same memory: max |Δ presence logit| and max |Δ box| ≤ 1e-5.
   * **C-loss** (added by WP-D): the rebuilt PB0 loss equals `slot_presence.refined_box3d_losses(...)["total"]` on
     the same slots and targets, relative error ≤ 1e-6.

## Code at registration (md5, `code/`)

Recorded in `raw/SPEC_SHA256.txt` beside this file's sha256 (`pbox_prep.py`, `pbox_capture.py`, `pbox_arms.py`).
Any later code change before an arm runs is disclosed in RESULT with its reason.
