# PREREG_V8F_PROOF addendum 10: LATVIS, pictures decoded from the world-model latent and from the planner tokens (frozen 2026-11-13, before its data; file time in TIMES_NOTE.md)

Card: PREREG_V8F_PROOF.md (sha256 38d33abd...c3a7). The PI asked for decoding and visualisation of predicted latent representations. H-V8F-LATSEM-1 (the other session) named a per-level pixel decoder as its next lever and H-V8F-LATSEM2B-1 measured no scene imagining with semantic probes. This addendum adds the pixel view on the frames found on the dev box on 13 Nov. Tier T0 (a world-model and perception diagnostic on logged frames), never driving performance. No model is trained or changed; only small decoders (probes) are fitted.

## Data
* Frames: `E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl`, grid frame g = raw frame 2g, area-downsampled to 64x160 RGB.
* Z: the level-1 world-model latent (128 numbers per frame) of M2b seeds 0 and 1, from the banked `latents.pt` of WMACT (windows of eval-141 with frames).
* G: the 4x10 token grid of the same frame (L40 of addendum 9: 40 tokens x 128 reduced channels).
* Predicted latents at 0.8 s and 2.0 s from the window start under the logged actions: the learned rollout and the stacked predictor of WMSTACK (`wm/wmstack.py`, unchanged by import).

## Decoders (probes)
* One convolutional decoder per source (Z seed 0, Z seed 1, G): input mapped to a 4x10 map with 256 channels, four 2x upsampling blocks, RGB output at 64x160, L1 loss, AdamW 2e-3, at most 20 epochs with early stopping on an inner split by episode. 5 folds by episode (fold seed 0): every evaluated or shown frame is decoded by a decoder that never saw its episode.

## Reads
* V1 (what is retained): held-out skill = 1 - MSE / MSE of the training-fold mean image, for Z (both seeds) and G, with the paired episode-cluster bootstrap (B 2,000, seed 0) for G minus Z. Outcome Z-BELOW-G if G minus Z is positive with an interval excluding 0 at both seeds, else Z-AT-G.
* V2 (what is imagined), at 0.8 s and 2.0 s, per seed and per predictor: error against the TRUE future frame of (a) the decoded predicted latent, (b) the decoded start latent (copy), (c) the decoded true future latent (ceiling). Headroom = (b) minus (c). Outcome at 2.0 s, needing both seeds: IMAGINE-PIXELS if (a) is below (b) with an interval excluding 0; IMAGINE-BELOW-COPY if above with an interval excluding 0; else IMAGINE-FLAT. If the headroom is not positive with an interval excluding 0, the outcome is NO-HEADROOM and (a) is printed but decides nothing.
* Pictures: 12 held-out windows fixed by rule before any decoding: the 6 windows with the largest absolute speed change over 2.0 s (at most one per episode) and the 6 windows with the smallest lead gap at the window start (at most one per episode). Columns: frame t, frame t + 2.0 s, decoded G(t), decoded Z(t), decoded predicted Z(t + 2.0 s), decoded true Z(t + 2.0 s). The panels hold camera content: they stay on the dev box and in the private page and are never staged for the public repo.

## Controls with known values
* v1: a decoder trained on latents permuted across frames inside the training folds reads held-out skill at most 0.02.
* v2: the ceiling must be real: (c) is below (b) at 2.0 s with an interval excluding 0, otherwise NO-HEADROOM as above.
* v3: decoding the training frames reads a skill at least as high as decoding held-out frames (direction check of the cross-fit).

## Next levers, committed
* Z-BELOW-G with IMAGINE-FLAT or NO-HEADROOM: the state the world model predicts cannot show a change of the scene; the lever is a spatial world-model state (the token grid, not its mean), with this read as its acceptance test (training: pod, launch gate).
* IMAGINE-PIXELS: decoded rollouts become the visual debug channel, and the same decoders are fitted for levels 2 and 3.

## Limits stated in advance
* A decoder shows what is linearly or shallowly recoverable from 128 numbers; a blurred picture is expected from a mean-pooled state. 139 episodes. One checkpoint per seed. L1 pixel error rewards the mean image; the skill is read against that floor.
