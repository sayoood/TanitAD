# PREREG_V8F_PROOF_ADD38_N2 : LATVIS-2, control k5 as written is not met; its cause, and k5 read in parts, before any D is read

Note to card `PREREG_V8F_PROOF_ADD38.md` and note N1. Frozen after the pod run of the first arm (G_s0) printed its k5 value and the per-fold skill of the token map (held-out 0.55 to 0.60, permuted below 0), and before any value of D, of the headroom or of a name has been looked at for any arm: the dev-box reader has not been opened on a dump. Tier T0. STRATEGIC: PAUSED (PI item 25).

## 1. k5 as written: FAIL (MEASURED, pod, `runs/lv2_z/DUMP_LV2_G_s0.npz.json`)

The rollout of the tool differs from `pred_2s` of `LATENTS_CTX_G_s0_R3g.npz` by 0.2009 (max abs) against the bar 1e-4; the start states differ from the banked `z0` by 0.2416 and the future states from `true_2s` by 0.2583. The same is expected for the other three arms and is reported as measured for each. By the card, with k5 as written failed, no name is given.

## 2. The cause (MEASURED on the pod, G_s0, 1,948 windows, latent states only; `latvis2/ops/lv2_k5_probe.py`, output `latvis2/ops/k5_probe_G_s0.out`)

* The rollout function is the same: applied to the banked start states and banked actions, the tool's rollout equals the banked `pred_2s` to 2.0e-05.
* The window list and the actions are the same (ids equal, actions differ by 0.0).
* The start states are two computations of the same states from two token files. The banked files were written on the dev box from its state file (`ctx_roll.py --states-pt`, the state file of the pixel job); the tool reads the pod state file, computed from the pod token cache the arms trained on. The two agree to a root mean square difference of 0.0237 against a state size of 0.949 (2.5 %), mean cosine 0.99967 (lowest window 0.99896), median correlation per dimension 0.99966; 0.15 % of the values differ by more than 0.1. For comparison the true change of the state over 2.0 s has a root mean square of 0.663.
* The frames are aligned: for every window the pod frame nearest to the banked start state is frame 4 and the one nearest to the banked future state is frame 14 (share 1.0 and 1.0); the squared distance at the right frame is 0.9 % of that at the nearer neighbour frame (median).
* Why the two token files differ by this much is a HYPOTHESIS, not measured here: the pod cache stores the tokens in an 8-bit float format, the dev-box features in another precision.

So k5 compared a rollout from pod states with a rollout from dev-box states. What k5 is for (the tool's rollout is the model's rollout of the r-b work, on the right frames) can be checked without that mismatch.

## 3. k5 in parts (fixed here; the bars are round values set before the other three arms are probed)

For each arm and seed, from the probe on the pod:
* k5a: the tool's rollout function on the banked start states and actions equals the banked `pred_2s` to 1e-4 (the card's bar).
* k5b: the window list equals that of the banked file and the actions differ by exactly 0.0.
* k5c: the share of windows whose nearest pod frame to the banked start state is frame 4 is at least 0.99, and the same for the banked future state and frame 14.
* k5d: the mean cosine between the pod states and the banked states is at least 0.999, for the start and for the future states.
* k5e: the probe's rollout from the pod states is the dump's rollout (its max abs difference to the banked rollout equals the dump's k5 value to 1e-6), and the probe ran on the dump's checkpoint.

## 4. How names are given from here

* Every report states: k5 as written, FAIL, with the value per arm and seed.
* A name of the card is given for an arm only if k1 to k4 and k6 pass as written and k5a to k5e pass at both seeds, and it is reported as "named under k5 in parts (note N2)". The reader takes the probe files by `--k5-parts`; without them it withholds the name as before.
* If any part fails for an arm, no name is given for it and the cause is found first.

## 5. Limits

The token-level read is on the pod states and the pod tokens (the inputs the arms trained on); the r-b reads of the earlier rows are on the dev-box states. The two differ by 2.5 % of the state size, so numbers of the two reads are not compared at a finer grain than that.
