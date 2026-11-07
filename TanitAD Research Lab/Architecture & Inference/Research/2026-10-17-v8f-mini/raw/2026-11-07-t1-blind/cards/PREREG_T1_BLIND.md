# PREREG_T1_BLIND: does the planner use its scene tokens on REAL frames? (Master Mind, 2026-11-07 04:44 dev-box clock, from `date`; written before any BLIND rollout at T1-kin exists)

## 1. Why this read, and why now
* At T2 the crawl card's instrument BLIND (C0 with every frame's scene tokens replaced by one fixed mean) did **not** move the crawl. BLIND − C0 on M1 is +0.0111 [−0.0704, +0.0917] at s0 and −0.0757 [−0.1599, +0.0143] at s1 (`D-V8F-T2-CRAWL-READ-1`). Rendering at the logged pose did not move it either (X2 ≈ 0).
* The render gap is large. The pooled-token cosine of a frame rendered at the logged pose against the real frame has a median of 0.84 and 0.74 on the two stage-1 scenes (`stage1/pull_box2/pull/STAGE1_RESULT.json`, steps.gap).
* SPEEDCAL read INFORMATION-LIMITED, and that sent the speed lever to perception: the LoRA arm L, about 241 h per seed on the pod.
* Two readings fit the T2 facts, and they send L in different directions:
  * **(a)** the planner uses its tokens on real frames, but renders put them out of reach. Then the T2 tier cannot credit any vision lever until the render gap is closed.
  * **(b)** the planner ignores its tokens even on real frames. Then BLIND ≈ C0 at T2 says nothing about renders, and the LoRA must create the token-to-motion path.
* T1-kin on eval-141 uses real frames. One pair of CPU rollouts tells these apart before any L checkpoint is read.

## 2. The arm
* **BLIND1_s** = C0_s (`v8f-recover/M2b_s{s}_recft_C0/ckpt_last.pt`). Every frame's scene tokens are replaced by the BLIND mean: the eval-141 cache mean, fp16 [40, 1024], 14,237 frames, 141 clips, sha256 `cd6a7ab5e3c5450ec3528889b04e06e5605fea30ea9ff02a0e36d9ba47a2fcb9` (PINS_E4). This is the same tensor the T2 BLIND arm used.
* Everything else is RIC0's: the T1-kin tool `v8f_t1_eval_ri.py` with `--smooth sg72 --scene reindexed --replan-steps 10 --nav deadreckon --schedule periodic --limit causal`. Measured ego history, including yaw rate, is unchanged. Only the scene tokens change.
* **Comparator:** the banked RIC0_s (sha256-checked), plus the P10 arrays inside each array file.

## 3. Gates (each must pass before any row is read)
* **G-ID:** the same path with the replacement switched off reproduces RIC0_s's arrays bit for bit at both seeds. Its delta reads exactly 0.
* **G-MEAN:** the tokens the policy receives equal the pinned mean exactly, on at least 1,000 frames. A mutant that leaves the tokens unchanged must fail G-MEAN.
* **G-EGO:** the ego inputs the policy receives equal RIC0's on the same frames.

## 4. Statistic and estimator
* B = BLIND1 − RIC0 per row: the 19 rows of the T1 reader in four families (STRATEGIC PAUSED, PI item 25).
* The estimator is a paired episode-cluster bootstrap, B 2,000, seed 0, over 141 episodes and 2,399 windows. It answers *another draw of episodes* only. The arm is a deterministic function of C0, so no training seed enters B. The seed question is answered by reading both seeds.
* F = 0.12074 m, the training-run spread of M2b on ADE6 (`D-V8F-LORA-FPRIME-1`).

## 5. Outcomes, committed now
1. **VISION-USED.** At both seeds, ADE6 B is separated-worse with a point above F, **or** M1 B is separated in either direction.
   * Read with T2's BLIND ≈ C0, this is **RENDER-GAP-BINDS**: on real frames the planner's driving depends on its tokens, and on renders it does not.
   * Next, committed: the LoRA dominance card gains a gate before LC's T2 read counts: L-BLIND at T2 must be separated from L on ADE6 or M1. A render-robustness lever gets its own card next. It either adapts the encoder to NuRec renders or trains with render-domain augmentation, and it is aimed at T2.
2. **VISION-UNUSED.** At both seeds, neither ADE6 B nor M1 B is separated, and |ADE6 B| < F.
   * The planner's driving does not depend on its scene tokens even on real frames. So T2's BLIND ≈ C0 is not a render effect.
   * Next, committed: L stays the route. Its card gains a T1-kin BLIND gate, because a perception lever the planner ignores cannot move the crawl: L-BLIND − L must be separated on ADE6 or M1 before any L number counts. The theory note records that the shrunken a0 gain is not offset by vision.
3. **MIXED.** Anything else is reported per seed and per row. Next: the BLIND gate enters the LoRA card at both tiers, and the render-robustness card is drafted, not run.

## 6. Beside, deciding nothing
* B − RIC0 on every row, and BLIND1 − P10.
* Distance keeping (dk141) for BLIND1, if the rollouts dump what dk141 reads.

## 7. Log (dated, append-only after the freeze)
* 2026-11-07 05:03 (dev-box clock, from `date`), Master Mind. **Scoping of outcome 1, entered before any BLIND1 row exists** (stages C to G still running; only G-ID s0 has been read, PASS). "At both seeds, A or M" binds per seed: at each seed, the ADE6 clause (separated-worse with a point above F) or the M1 clause (separated in either direction) must hold. `t1blind_read.py` applies that reading and prints beside it the reading in which each clause must hold at both seeds, deciding nothing. G-EGO compares the tick-0 ego inputs. At later ticks the ego history is the executed motion, which differs whenever the plan differs, by design.
* 2026-11-07 05:17 (dev-box clock, from `date`), Master Mind. **RESULT: VISION-USED** (`t1_blind/RESULT_T1BLIND.md`; `raw/READ_T1BLIND.json` sha256 `96049e35...`). All gates PASS at both seeds: G-ID bit for bit against RIC0; G-MEAN on 115,152 / 115,152 received frames; G-EGO on 2,399 / 2,399 tick-0 windows.
  * B = BLIND1 - RIC0, T1-kin, eval-141, paired episode-cluster bootstrap B 2,000. ADE6 +0.8313 [+0.5576, +1.1530] at s0 and +0.8158 [+0.5309, +1.1454] at s1: separated-worse, both points above F = 0.12074 m. M1 -0.0246 [-0.0428, -0.0085] and -0.0415 [-0.0654, -0.0208]: separated-better. Without its tokens the planner crawls LESS.
  * Also separated-worse at both seeds: cross-track MAE (+0.78 / +0.80 m), heading MAE, along-track MAE, DK-HW and DK-TG. The stricter reading (each clause at both seeds) also gives VISION-USED.
  * **By section 5, outcome 1, read with T2's BLIND ~ C0, this is RENDER-GAP-BINDS.** Committed next: (i) the LoRA dominance card gains the gate "L-BLIND at T2 separated from L on ADE6 or M1" before LC's T2 read counts; (ii) a render-robustness lever gets its own card, aimed at T2.
  * Beside, deciding nothing:
    * Distance keeping at 6 s, tick-0 plan (`raw/DK141_TABLE_T1BLIND.md`): LEAD-VIOL at-risk BLIND1 9.24 % / 10.59 % against RIC0 5.71 % / 5.38 %, and progress 0.944 / 0.957 against 0.909 / 0.891.
    * The T2 stage-2 means (`stage2/SCORE2.md`, unpaired): BLIND ADE6 7.729 (launch) / 6.794 (moving) against C0_s0 7.980 / 7.336. So on renders the tokens do not help ADE6, where on real frames they help by about 0.82 m.
