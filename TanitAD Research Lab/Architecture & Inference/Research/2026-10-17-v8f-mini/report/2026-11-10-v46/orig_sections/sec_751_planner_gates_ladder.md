<!-- source: Paper/TANITAD_PAPER.md v1.8 at 5bb44006, section 7.23, renumbered for the report -->
### 7.51 Planner gates, the readout ceiling and the band-trained ladder (2026-11-09)

**Tier T0 throughout. These are world-model and planner diagnostics, never driving results.** Evidence class MEASURED; records under `raw/2026-11-09-v8f-plantest3/` and `raw/2026-11-09-v8f-ladder-read/`, with register rows H-V8F-PLANTEST2, -DECN, -FULLN, -PT3N, -PT3M, -PT3F and -LADDER. TACTICAL has n = 0 (no decision output in these reads); STRATEGIC is PAUSED (PI item 25).

**The planner on the M2b latents.** Each gate gives the planner the encoded true future as its goal and scores the 12 logged actions of the next 2.4 s. Estimator: paired episode-cluster bootstrap over 71 SCORE episodes (994 windows), B = 2,000; both training seeds. HOLD (the first logged action held) reads 0.3348 m/s^2 and 0.0256 rad/s.

| gate | what changed | outcome, both seeds | numbers (G-TRUE, s0 / s1) |
|---|---|---|---|
| PLANTEST2 | solver seeded at HOLD, penalty on leaving it | NO-GAIN | FLAT1 acceleration MAE 1.0430 / 1.0033 m/s^2 |
| PT3N | decoded end-point cost, reader fitted on 1,053 episodes | NO-GAIN | 0.3308 to 0.3396 m/s^2 |
| PT3M | nonlinear four-frame reader in the cost | NO-GAIN | G-TRUE minus HOLD +0.0065 to +0.0287 m/s^2 |

The true goal beats a random goal on every row of PT3N and PT3M, so the goal does reach the plan. The plan still does not beat HOLD.

**Why: the ceiling experiment.** ORACLE-END builds the 2.4 s plan from the true end speed and yaw rate alone (constant acceleration, yaw ramp). It beats HOLD: 0.2161 against 0.3348 m/s^2, difference -0.1187 [-0.1364, -0.1014], and 0.0142 against 0.0256 rad/s, -0.0114 [-0.0151, -0.0082], ADE at 2.4 s 0.2066 m against 0.2284 m. So an end-point goal holds enough information, about 35 % of the longitudinal and 45 % of the lateral error. READ-END takes the same two values from the encoded true future with readers fitted on all 4,565 trained clips, and it does not beat HOLD: longitudinal 1.005 / 1.099 m/s^2, +0.6701 [+0.5138, +0.8426] and +0.7640 [+0.6121, +0.9311] worse; lateral 0.0228 / 0.0228 rad/s, -0.0028 [-0.0061, +0.0002] and -0.0028 [-0.0058, +0.0001], not separated. The mean speed read error at 2.4 s is 2.30 / 2.57 m/s. The latent carries speed at held-out R^2 up to 0.778 / 0.702 (best of four readers, optimistic), and that is not enough: the planning cost needs a far smaller absolute error than an R^2 bar suggests.

**The band-trained ladder.** Nine world-model runs, adapter and planner frozen at M2b. C0 keeps strides (2, 2) and one-step training; S35 moves to strides (3, 5), so levels step 0.2, 0.6 and 3.0 s; R35 adds k-step training (k = 10, 10, 6); R35H adds multi-rate history at each level's own step; R35b repeats R35 at seed 0. The quantity is skill against copying the present latent at each level's band end, `1 - sum|pred - target|^2 / sum|copy - target|^2`, on 141 eval clips, by a two-part rule (episode-cluster bootstrap lower bound above 0, B = 2,000, and point above the 97.5th percentile of 199 within-clip derangements), seeds 0 / 1.

| arm | L1 at 2 s | L2 at 6 s | L3 at 12 s |
|---|---|---|---|
| C0 | 0.064 / 0.080 | 0.278 / 0.267 | 0.359 / 0.401 |
| S35 | 0.069 / 0.096 | 0.294 / 0.339 | 0.472 / 0.480 |
| R35 | 0.178 / 0.190 | 0.419 / 0.430 | 0.535 / 0.541 |
| R35H | 0.259 / 0.267 | 0.546 / 0.555 | 0.672 / 0.687 |

Every arm reads BAND-MET under the card. Copy-Last is a lenient bar: a predictor that only shrinks toward the mean scores 0.05 to 0.08 at L1, 0.14 to 0.20 at L2 and 0.18 to 0.26 at L3 with no knowledge of the future. R35 and R35H beat that no-action baseline and a ridge baseline at every level at both seeds; C0 and S35 at L1 do not clearly do so (a post-hoc read, decides nothing).

**A replicate that measured determinism.** R35b reproduced R35_s0 bit for bit: 276 of 276 model tensors and 399 of 399 optimizer tensors are identical, while R35_s0 and R35_s1 differ on 252 of 276. Training on cached latents at a fixed seed is deterministic on this machine, so the same-seed floor is exactly 0, and the card's lever rule passes every difference. This agrees with what the paired bootstrap cannot see (§7, `H-ESTIM-SEED-1`): only a changed seed can estimate training variance. Against the largest seed-to-seed spread over the four arms (0.027, 0.045 and 0.042 at L1, L2 and L3; two seeds, so a rough floor, and post-hoc), k-step training adds +0.091 to +0.125 at L1 and L2 and +0.061 to +0.063 at L3, multi-rate history adds +0.077 to +0.146, both above the spread at every level; the step change alone is inside it at L1.

**What this section does not establish.** (i) Nothing here is driving performance. (ii) L1 gains need three context states: with one state, the form every planner gate above used, no arm beats Copy-Last at 2 s (-0.059 to -0.012). (iii) The lateral response of the level-1 prediction to actions stays near zero on every arm (gain -0.074 to -0.010, 1.0 being correct), so the operative level is the open defect. (iv) No planner has yet been read on these models; those gates are running. (v) The prediction gain is in latent space, and whether it reaches planning is exactly what is untested.
