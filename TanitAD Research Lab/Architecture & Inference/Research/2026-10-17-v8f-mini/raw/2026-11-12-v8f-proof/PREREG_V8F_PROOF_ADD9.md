# PREREG_V8F_PROOF addendum 9: FULLRES, un-pooled and higher-resolution frozen tokens (frozen 2026-11-13 ~03:00, before its data)

Card: PREREG_V8F_PROOF.md (sha256 38d33abd...c3a7), hypotheses H-V8F-PROOF-4 (theorem T5) and H-V8F-PROOF-8 (stop cue). This addendum executes the next lever that addenda 1 (RESPROBE) and 5 (CUEPROBE) committed, and predictions P4a, P4b, P4c of the main card. Change of place only: the main card said "on the pod"; the frames and the encoder weights were found on the dev box on 13 Nov (`E:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl`, 139 episodes, 201 PNG frames each at 10 Hz, 416x1024 cylindrical, f_ref 488.92 px/rad; DINOv3 ViT-L/16 in the local Hugging Face cache), so it runs on the dev-box GPU behind the shared GPU lock. Tier T0 (frozen-feature probe on logged frames): a perception diagnostic, never driving performance. Zero training of any model; only probes are fitted.

## Encodes (frozen DINOv3 ViT-L/16, recipe of `dinov3_fp8_encode_ship.py`: bf16, ImageNet mean and std, last hidden state, special tokens dropped)
* Frames: every grid frame g = 0..100 (raw frame 2g) of the 139 episodes.
* LO: the frame resized to 260x640 (bilinear, antialias) and centre-cropped to 256x640, giving 16x40 tokens. HI: the 416x1024 frame as it is, giving 26x64 tokens.
* Channel reduction: one PCA basis 1024 -> 128, fitted without labels on the tokens of 3 frames per episode (both encodes pooled), stored as int8 at 6 sigma per component. Declared limit: the basis sees held-out frames (no labels), and it may hide low-variance detail; control f3 bounds this at the pooled level and the committed repair is r = 256.

## Feature sets (all from the reduced tokens; each is read with the ego block E of addendum 1)
* L40: LO mean-pooled 4x4 (the 4x10 grid). L160: LO mean-pooled 2x2 (8x20). U640: LO un-pooled (16x40).
* C640: the central 16x40 block of the HI token map (rows 5 to 20, columns 12 to 51): a centre crop at the higher resolution with the token count of U640. H1664: HI un-pooled (26x64).

## Probes
* P1: `res/resprobe.py` by import, unchanged (kernel PCA rank 256 fitted on the training folds, kernel ridge, linear kernel on the flattened tokens).
* P3 (attentive probe, new, translation tolerant): per-token MLP 128 -> 64, then per target a score and a value per token; prediction = sum over tokens of softmax(score) * value, plus a linear term in E. One multi-task network per (set, fold); AdamW, learning rate 1e-3, weight decay 1e-2, at most 40 epochs, early stopping on an inner split by episode (20 % of the training episodes); 3 initialisation seeds, predictions averaged.
* Folds: the 5 episode folds of addendum 1 (fold seed 0). Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, on the cross-fitted predictions. The encoder is frozen, so there is no training-run variance of a model; probe initialisation variance is handled by the 3 seeds.
* Targets: GAP, NEAR, COUNT, LEADP on the bank frames (labels of addendum 1); STOPFLAG and STOPDIST on the first-tick windows of addendum 5 (windows of episodes without frames are dropped and n is reported).
* Decision rule for every pair (fine, coarse): P3 decides. A "fine beats coarse" claim needs the P3 difference separated from 0 and the P1 difference not separated in the opposite direction. P1 is always printed beside.

## Predictions (main card, theorem T5; horizon-row bound f*h/16 = 25 m for LO and 40 m for HI with h = 1.31 m)
* P4a-full: GAP R2, U640+E minus L160+E, is positive with an interval excluding 0 (pooling isolated: same frames, same encode).
* P4a-res: GAP R2, H1664+E minus U640+E, is positive with an interval excluding 0 (resolution).
* P4b: STOPFLAG AUROC, C640+E minus E, is positive with an interval excluding 0, while L40+E minus E is not.
* P4c: the STOPFLAG gain of C640+E over E is larger on stop windows with a rest distance under 40 m than on the others (reported with n; no outcome name depends on it).

## Outcomes (names fixed now)
* GAP, pooling: UNPOOL-HELPS if P4a-full holds, else UNPOOL-FLAT. GAP, resolution: RES-HELPS if P4a-res holds, else RES-FLAT.
* GAP, the limit: INTERFACE-LIMITED if the better of U640+E and H1664+E beats L40+E by at least 0.10 R2 with an interval excluding 0. ENCODER-LIMITED (at these probe classes) if neither fine set is separated from L40+E. Otherwise INTERFACE-PARTIAL.
* Stop cue: CUE-FOUND if any of U640+E, C640+E, H1664+E beats E on STOPFLAG AUROC with an interval excluding 0; DIST-FOUND if the best of them beats E on the STOPDIST absolute error (paired interval excluding 0); else CUE-ABSENT-FULLRES.
* NEAR, COUNT, LEADP are printed beside for every pair and decide nothing.

## Controls with known values
* f1 (harness): P1 on the cached V40 kernel reproduces the banked RESPROBE GAP R2 of V40+E to 1e-6.
* f2 (recipe): on 2 frames per episode, the mean cosine between the 4x4 mean pool of the full-width LO tokens and the cached `pool4x10` row. Reported; at or above 0.85 the recipe counts as close. Below it, every comparison against a cached set is flagged; comparisons inside FULLRES are not affected.
* f3 (reduction and recipe at the pooled level): P1 on L40+E within 0.05 R2 (GAP) and 0.03 AUROC (LEADP) of the banked V40+E values. If f3 fails the read is INVALID as designed and is rerun with r = 256.
* f4 (shuffle): labels permuted within fold, P3 on L40 and U640: R2 at most 0.02 and the mean per-fold AUROC within 0.06 of 0.5 (the per-fold form, the lesson of addendum 1).
* f5 (planted object): a fixed random vector with the median token norm is added to one random token position in a random half of the frames; the label is "planted". P3 on U640 must read AUROC at least 0.95. P1 on U640 and P3 on L40 are printed beside (theorem T5, VC-H3, predicts the pooled set loses a one-token object).
* No late-token control: the one of addendum 5 (k4) missed its bar for a measured reason and is not reused.

## Next levers, committed for every outcome
* INTERFACE-LIMITED or INTERFACE-PARTIAL: a reader arm that gets a fine token set beside the 4x10 grid (pod, launch gate, needs the training-side frames), with this P3 probe on the training clips as its ceiling.
* ENCODER-LIMITED or UNPOOL-FLAT: TEMPORAL, run at once in the same chain with zero training: P3 on U640 of frame g together with frame g-5 (1.0 s earlier; 1,280 tokens with a frame bit), same targets, same folds: does a second frame (scale change, parallax) supply the range that one frame does not? Then a depth-aware head or a tuned last block on the pod.
* CUE-ABSENT-FULLRES: the stop cue needs the stop line or signal head above the pixel floor of T5: a trained cue head on the HI tokens of the training clips (pod), or map and rule input for the stop position.

## Limits stated in advance
* 139 episodes; GAP has 56 episodes with a lead; 131 stop windows before dropping. A null under P1 and P3 is not a null about a trained reader on the training clips. The LO frame is a resize of the 416x1024 frame, not the original 256x640 render (f2 measures the difference).
