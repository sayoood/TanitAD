# PREREG_V8F_RHHW3 (FROZEN 2026-11-24, before any three-seed pooled number): is the adopted recipe's headway cost real across three seeds? Zero training, dev-box CPU

**Parents:**
* `PREREG_V8F_ROLLHIST.md` (781eea87): section 3.2 gives the cells and the cell rule; Amendment 2 gives the straddle definition.
* `PREREG_V8F_SEED3.md` (75f3b191) with Amendment 1.
* MODEL_REGISTRY section 20: RH ADOPTED as the stage B of the recipe of record under readout K (PI, 2026-11-24).

**PI go:** "Yes, run it", 2026-11-24, for this read. CPU only, no GPU, no spend.

## 1. Why, and what is already known
* The PI adopted RH with two stated costs. This card reads the second one: the headway unsafe rate U against B0.
* **Seen before this card:** the per-seed U values of RH − B0 were printed in earlier reads.
  * The two-seed RH read printed s0 and s1. The MM has not quoted them in this session.
  * The SEED3 read printed s2: +0.0125 [+0.0000, +0.0267] under C\*, and +0.0071 [+0.0000, +0.0161] under K.
* **Not seen:** no three-seed pooled value and no three-seed floor has been computed by anyone.
* This card fixes the pooling, the floor and the outcomes before that number exists. Because the per-seed points are partly seen, its evidence is weaker than a fresh arm's. The verdict says so.

## 2. Material
* Arms RH_s0 / RH_s1 / RH_s2 against B0_s0 / B0_s1 / B0_s2; checkpoints in MODEL_REGISTRY section 20.
* T1-kin/hb0.2 dumps, as already rolled by the frozen executor and read by the two-seed and SEED3 reads:
  * s0 and s1 under `F:/TanitAD-scratch/rollhist/read/out/t1/`;
  * s2 under the SEED3 read's t1 root.
* No new rollout. If a dump is missing, the read is BLOCKED; it is never re-rolled under this card.
* Forms: **K** (`R0KHb20`), the adopted readout, **decides**. C\* (`R0C10XHb20`) is printed beside.

## 3. Read, estimator, floor (committed now)
**Reader:** the frozen headway reader `code/headway/headway_read_r2.py`, unchanged.
* Its `--seeds 0,1,2` is used if it takes three seeds unchanged. Otherwise it runs per seed, and a new tool pools them in the reader's own form.
* **Control R0, which must pass before any three-seed number is printed:** the pooling path reproduces the frozen two-seed RH read's U value for RH − B0 (s0 + s1) exactly, D and interval, under both forms.

**Cell:** headway unsafe rate U, LONGITUDINAL distance-keeping. Contact C and falling-behind F are printed beside.

**Estimator:** the reader's paired episode-cluster bootstrap, B 2,000, seed 0.
* The same episode draw is applied to every seed, with D pooled over the three seeds' windows, as the reader pools two seeds.
* It answers "another draw of EPISODES"; the three seeds answer "another TRAINING run".

**Floor F3 per form:** max over seed pairs of |RH_si − RH_sj| and of |B0_si − B0_sj|, on the U level. It is the ROLLHIST two-seed F_c rule extended to three seeds.

**Cell verdict:**
* **WORSE:** the pooled interval excludes 0 on the worse side, all three seed points are on the worse side, and |D| > F3.
* **BETTER:** the mirror of WORSE.
* **NOT-SEPARATED:** otherwise.
* **Straddle** (Amendment 2): a NOT-SEPARATED cell whose interval excludes 0.

## 4. Outcomes (U under K decides), committed in advance
| name | condition | reading | next lever |
|---|---|---|---|
| **HW3-COST-REAL** | U under K reads WORSE | the adopted recipe keeps the lead less safely than B0 beyond the three-seed training spread | a longitudinal lever on RH, under a new card, back to the PI with its cost (candidates named there, not here) |
| **HW3-COST-BELOW-FLOOR** | U under K is NOT-SEPARATED but straddles on the worse side, or all three seed points sit on the worse side | a consistent small cost inside the training spread | the recipe stands. The stated cost becomes "consistent, below the three-seed floor" with its D; the coordinator carries it into the candidate's longitudinal check |
| **HW3-NOT-SEPARATED** | otherwise | no headway cost distinguishable at three seeds | the recipe stands, and the stated cost reads "not separated at three seeds" |
| **HW3-BLOCKED** | R0 fails, or a dump is missing | not read | fix the instrument; a missing dump is not re-rolled under this card |

* **Precedence:** BLOCKED, then COST-REAL, then COST-BELOW-FLOOR, then NOT-SEPARATED.
* **Other families:** LATERAL, TACTICAL and STRATEGIC are not recomputed. The verdict quotes them from the SEED3 and two-seed reads, with their files.

## 5. Cost
Zero training and zero GPU. Dev-box CPU, minutes per reader pass, behind the commit-headroom gate.

No three-seed pooled number exists.
