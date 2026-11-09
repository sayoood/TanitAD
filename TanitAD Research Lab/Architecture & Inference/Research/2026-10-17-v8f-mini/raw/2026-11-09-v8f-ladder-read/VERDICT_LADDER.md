# VERDICT_LADDER (Master Mind, 2026-11-09; all nine runs)
Card `PREREG_V8F_LADDER.md`. Tier **T0**, a world-model diagnostic in latent space, never a driving result. MEASURED: `RESULT_LADDER_READ.md`, `results/arm_*.json`, `DIAG_SHRINK.md`, `R35B_CKPT_IDENTITY.json`. Interval: episode-cluster bootstrap over the 141 eval clips, B 2,000, with the 199-draw derangement reference; it answers "another draw of episodes". Both training seeds are required by the card. LONGITUDINAL and LATERAL appear through the IA-2 targets and the IA-3 gains; TACTICAL: n = 0, no tactical target is read here; STRATEGIC: PAUSED (PI item 25).

## The card's label, as committed
**BAND-MET on C0, S35, R35 and R35H at both seeds, and on R35b:** all three levels beat Copy-Last at their band end and are healthy. Skill at L1 2.0 s / L2 6.0 s / L3 12.0 s, seeds 0 / 1:

| arm | L1 2.0 s | L2 6.0 s | L3 12.0 s |
|---|---|---|---|
| C0 (control) | 0.064 / 0.080 | 0.278 / 0.267 | 0.359 / 0.401 |
| S35 | 0.069 / 0.096 | 0.294 / 0.339 | 0.472 / 0.480 |
| R35 | 0.178 / 0.190 | 0.419 / 0.430 | 0.535 / 0.541 |
| R35H | 0.259 / 0.267 | 0.546 / 0.555 | 0.672 / 0.687 |
| R35b (R35's flags, seed 0) | 0.178 | 0.419 | 0.535 |

## The replicate floor reads 0.0000, and that is what it is
R35b is a separate 3 h 16 min training run, and it reproduced R35_s0 **bit for bit**: 276 of 276 model tensors and 399 of 399 optimizer tensors are identical, maximum absolute difference 0.0. The control holds: R35_s0 against R35_s1 differs on 252 of 276 model tensors (`R35B_CKPT_IDENTITY.json`). The training log agrees to the last digit; only wall-clock fields differ. The checkpoint files differ in sha256 because each stores its own argv hash.

So on this rig (cached latents, one GPU, fixed seed) **training is deterministic, and a same-seed replicate cannot estimate run-to-run variance.** The card's rule, as written, gives a floor of 0.0 at every level, and every difference clears it. That is reported as committed, and it says little.

**Beside, written after reading it, decides nothing:** the seed-to-seed spread, the largest |s0 − s1| over the four arms at each level: **L1 0.027, L2 0.045, L3 0.042**. It is two seeds per arm, so it is a rough floor.

| difference (same seed; s0 / s1) | L1 | L2 | L3 | against the seed spread |
|---|---|---|---|---|
| S35 − C0 (step only) | +0.005 / +0.016 | +0.017 / +0.072 | +0.113 / +0.079 | L1 inside; L2 only at s1; L3 above at both |
| R35 − S35 (k-step reach) | +0.109 / +0.094 | +0.125 / +0.091 | +0.063 / +0.061 | above at every level and both seeds |
| R35H − R35 (multi-rate history) | +0.082 / +0.077 | +0.127 / +0.125 | +0.137 / +0.146 | above at every level and both seeds |

## World model of record: open, for the PI
The card says "the arm with the smaller reading cost" and defines the phrase nowhere (neither does `SPEC_HWM.md`). The reading logged in the card before R35b's read existed reports two forms, and names an arm only if it is smaller on both at both seeds:

* **(a) Latent prediction cost, 1 − skill at the band end:** R35H is smaller at every level at both seeds.
* **(b) Reader score on the rolled-out latent (IA-2, speed, lead gap, yaw rate at the three band ends):** R35H is higher on 15 of 18 rows. It is lower on speed at L3 12 s at both seeds (0.690 / 0.705 against 0.730 / 0.787, n = 141 clips) and on speed at L1 2 s at s0 (0.651 against 0.711). At L2 6 s yaw rate it reads 0.466 / 0.443 against 0.144 / 0.206.

**The two forms disagree on one target, so by the logged rule the choice goes to the PI, and both arms stay in the planner gates** (`ladder_plantest`, already running). My recommendation is R35H: the speed deficit is on one target and the yaw-rate and lead-gap gains are large. The cost to weigh is that R35H reads more past frames at inference (0.4 s at L1, 1.2 s at L2, 6.0 s at L3).

## Beside (decide nothing)
* Trivial baselines: R35 and R35H beat a no-action SHRINK and LINEAR baseline at every level at both seeds in the built context. C0 and S35 at L1 do not clearly (`DIAG_SHRINK.md`; R35b's weights equal R35_s0's, so its row would equal R35_s0's and was not recomputed).
* **Planning form (context 1): L1 does not hold.** No arm beats Copy-Last at L1 2.0 s (−0.059 to −0.012, R35b −0.059), and SHRINK beats the world model on all eight runs. L2 and L3 hold.
* IA-3 action response: lateral gain at L1 2.0 s stays near zero on every arm (−0.074 to −0.010).

## What this says
Training on the PI's bands produces prediction inside the 6 s and 12 s bands that a no-action baseline does not explain, and the k-step reach and the multi-rate history each add a measured amount above the seed spread. The operative level is the weak part: at 2 s the gain over Copy-Last depends on three past states, and its lateral response to actions is near zero.

## Next (Rule Zero)
1. **Planner gates on the ladder models** are running (`ladder_plantest`; the one-frame and three-frame FLAT1 waves on R35 and C0 first). They are the first test of whether this prediction gain reaches planning.
2. **Band retrain** on box 2: its replicate `OB-BAND_rep` already uses a different seed (2), so it does measure training variance.
3. **Measured-speed world model:** the card's replicate moves to a different seed for the reason above; its control arm is the world model of record once named.
4. **Rule for the programme:** on a deterministic rig, a replicate changes the seed. A same-seed replicate reading 0.0 is a statement about determinism.
