# PREREG_V8F_PROOF_ADD38_N1 : LATVIS-2, what the build fixed, before any token-level number

Note to card `PREREG_V8F_PROOF_ADD38.md`. Frozen after the tool `latvis2/` was built and tested on synthetic worlds and a tiny random model (28 tests pass, run by the builder and again by the lead), and before the tool has read any token of the real cache through a real checkpoint: no D, no headroom, no skill of the token map exists. The card, its names and its controls stay as written. Tier T0. STRATEGIC: PAUSED (PI item 25).

## 1. Tool (sha256)

* `latvis2/lv2_pod_tokens.py` d5aa2c6d3b824805109b87b44f1e995d85dbbd6e6630cc5188bc7c9940b2b266 (pod side)
* `latvis2/lv2_read.py` ab3a4ed4367f3c61bb605ced06346b8ef0090c05e71bcc7ecc4d3762304617ad (read and pair)
* `latvis2/lv2_select.py` 7255e051de4da507b71da07ad20adfc5f6053c313ad1a4422ed92e0c792191b7, `lv2_frames.py` dcdf5d196c6637a0c6d4079318b0322474c7f59c20c27deb744a639d367edd6f, `lv2_sheet.py` 8db035033b460fb46cbe252cf9a3515c532c12ec721314af2b5c27f44fb17637, `test_latvis2.py` cec41eb0f23244e17dae01a042bcbd336bf2e4dbd55befa72a1b7c5d6bbb0af9
* Definitions: `latvis2/LATVIS2_DEFS.md`. Each control k1 to k6 has a deliberate regression in the tests that fails the control.

## 2. What the card left open, fixed here before the data

1. Ridge penalty of the token map: 0.01 times the mean diagonal of X'X (X = the centred 640-wide states of the training frames). One value, set here, not tuned on any outcome.
2. Token targets: the 1,024-wide pooled tokens of the cache (the 4x10 grid before the model's adapter), per cell a reduction to 64 principal components fitted on the frames of the four training folds; every error and the token change are in these coordinates. Frames: the unique frames covered by the 16 frames of the 1,948 windows.
3. Skill = 1 minus the squared error over the squared deviation from the mean of the training targets.
4. Moving cells of a window: the 10 of 40 cells with the largest squared distance between the true token coordinates of frame t and of frame t + 2.0 s; ties by the lower cell index. They come from true tokens only.
5. "The two-level interval lies above 0" = its lower bound is above 0 and the value of each seed is above 0.
6. Rows of the grid: cell c is row c // 10, column c % 10, row 0 at the top of the picture (`v8f_stopaux.py:10`). The names sky, far, near, hood of the card are these rows from top to bottom.

## 3. Which banked files the controls read

* k5: the banked file is `LATENTS_CTX_<ARM>_R3g.npz`. Three rollouts of each checkpoint are banked from the r-b work (R1: predictor context 1; R3h: context 3 seeded with the true states of earlier frames; R3g: context 3 from the state of frame t alone, the context filled by the rollout's own outputs). The card reads "the rollout of the stack from the state of frame t (the rollout of card ADD36)", which is R3g. MEASURED on the dev box, checkpoint G_s0, 200 windows, latent states only (no token): the tool's rollout equals R3g to 1.9e-05 (max abs), differs from R1 by 1.2 and from R3h by 0.49. The bar of k5 stays 1e-4, per arm on all 1,948 windows.
* k6: the reference is the list of 1,948 windows of `LATENTS_CTX_G_s0_R1.npz` (ids, order, count) with the folds of the pack `eval_pack.npz` (139 episodes, each in one fold). The floor file of the r-b read holds 1,809 of these windows and is not the reference, because the card reads on 1,948.

## 4. Windows of the sheets (MEASURED from labels and logged signals only; `latvis2/WINDOWS_LV2.json` sha256 a6c909a2fab48f269246deea34cd202c98658f50aa8c44eded5bd2a6cb458d4e)

* The 12 windows of the earlier panel are rows 1454, 830, 1876, 234, 1016, 1190, 1238, 1725, 926, 987, 1529, 272 of the window list. The earlier panel counted from the first history frame, so its pictures are 4 steps (0.8 s) earlier than t of the sheets; the sheets show frame t and frame t + 2.0 s of the window as every read of this line does.
* A window that is already one of the 12 is not chosen again among the lead windows. The six lead windows: 1411, 863, 1074 (label gap change -27.2, -23.4, -17.1 m over 2.0 s) and 603, 299, 1814 (+26.3, +17.5, +13.9 m).
* Gap changes of this size within 2.0 s can also be a change of which vehicle the label calls the lead. The sheets show the label as it is; this is a limit of the illustration, and no number of section 4 of the card depends on these windows.

## 5. Levels of the sheet as built

* L1: an empty tile with the reason, in this version (the earlier decoded pictures exist as composite panels at a time base 4 steps earlier). If decoded tiles are added later they are illustrations only.
* L2: one colour basis for all cells (three leading directions of the pooled token covariance of the training folds); tiles show the grids read from the present, the imagined and the true future state through the out-of-fold map, and the change tiles of the truth and of the imagination; the lead may add tiles of the true tokens of t and t + 2.0 s in the same colours (drawing only).
* L4 and L5 read the banked dumps of card ADD36 (arm LS, row H-V8F-PROOF-68); they carry no new number. The six L4 and L5 sheets in `latvis2/first_L4L5/` were drawn from `DUMP_LS_s0.npz` to check the layout.
* The camera frames of L0 are placed by the identity "cache step = step of the window" that the earlier panel used; it is taken over, not checked again against the pictures.

## 6. Run

Pod CPU, 4 threads, one arm after the other (G_s0, G_s1, RR_s0, RR_s1), loader tree `tree_leadsup`, states from the banked `probe/states/<ARM>_eval.npz`, checkpoints `runs/spatial/G_s0`, `runs/spatial/G_s1`, `runs/motion_a/RR_s0`, `runs/motion_d/RR_s1`. No GPU job is added. If k5 fails for an arm its names are not given and the cause is found before anything else is read.

## 7. Limits

As the card. In addition: the token map is linear and fitted on about 7,500 frames per fold for 2,560 targets; the colour tiles show three directions of 1,024.
