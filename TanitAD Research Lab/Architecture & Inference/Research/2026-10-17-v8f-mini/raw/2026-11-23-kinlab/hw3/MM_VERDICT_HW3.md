# MM verdict: three-seed headway read of the adopted RH recipe (HW3), 2026-11-24

**Outcome: HW3-NOT-SEPARATED.** Under readout K, the adopted readout, the headway unsafe rate of RH is not distinguishable from B0's at three seeds. The recipe stands.

* **Card:** `PREREG_V8F_RHHW3.md`, sha256 `3c85a42b…`, frozen 2026-11-24T05:09:51Z, before any three-seed pooled number.
* **PI go:** "Yes, run it", 2026-11-24.
* **Evidence class:** MEASURED (ours). Zero training: dev-box CPU on the existing T1-kin/hb0.2 dumps.
* **Caveat, stated in the card:** the per-seed points were partly printed in earlier reads, so this is weaker evidence than a fresh arm.

## Instrument and control
* **Reader:** the frozen headway reader `headway_read_r2.py`, unchanged. It takes `--seeds 0,1,2` as it is, so no new pooling code was needed.
* **Control R0 PASS:** with `--seeds 0,1` it reproduces the frozen two-seed RH read exactly: D, interval and per-seed points of U, C and F under K and under C\* (`R0_CHECK.json`).
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0, over the **53 episodes with a lead** and **562 lead windows per seed**. It answers "another draw of EPISODES"; the three seeds answer "another TRAINING run". Inference is deterministic.
* **Tier:** T1-kin/hb0.2, the executor of record, with no re-perception.

## U, RH − B0, pooled over seeds 0, 1, 2
| form | D | interval | per seed (s0 / s1 / s2) | F3 | verdict |
|---|---|---|---|---|---|
| **K (`R0KHb20`), decides** | **+0.0042** | [−0.0017, +0.0107] | +0.0071 / −0.0018 / +0.0071 | 0.0089 | **NOT-SEPARATED** |
| C\* (`R0C10XHb20`), beside | +0.0160 | [+0.0070, +0.0266] | +0.0178 / +0.0178 / +0.0125 | 0.0249 | NOT-SEPARATED, straddles on the worse side |

* **In window counts:** under K the per-seed differences are +4, −1 and +4 unsafe windows of 562. Under C\* they are +10, +10 and +7.
* **The name:**
  * Under K the interval includes 0, so it is neither WORSE nor a straddle.
  * One seed (s1) is on the better side, so it is not COST-BELOW-FLOOR.
  * Therefore HW3-NOT-SEPARATED.
* **The name tool passed its planted check**, through all four names: +0.03 reads COST-REAL, and 0.0 reads NOT-SEPARATED.
* **The two-seed value the stated cost rested on:** under K with s0 + s1, D was +0.0027 [−0.0035, +0.0106], so it was never separated there either.

## The reading
* **Under readout K the headway cost is not separated at three seeds** and is about one lead window in 240. The stated cost of the adoption becomes "not separated at three seeds under K".
* **Under the causal readout C\*** RH is consistently worse on U (all three seeds), but inside the training-seed floor.
  * That is one more reason readout K stays, beside P4's tactical cost.
  * It is a beside value and decides nothing.

## Four families
* **LONGITUDINAL distance-keeping:** this read.
* **LONGITUDINAL speed / along, LATERAL, TACTICAL and STRATEGIC:** not recomputed. They are as in MODEL_REGISTRY section 20 and `seed3/MM_VERDICT_SEED3.md`. Under K at s2: LATERAL BETTER; LONGITUDINAL speed / along, TACTICAL and STRATEGIC NOT-SEPARATED.

## Next lever
* **This card names none:** the recipe stands, and no longitudinal lever is opened on this evidence.
* **The remaining open cost of the adoption** is the T0 shift: +0.02 to +0.05 m against B0, inside the four-seed floor.
* **The candidate's integration** (spec, A90 margin under K on RH) is the coordinator's. Any GPU run on it needs its own PI go.

## Files
* `hw3/PREREG_V8F_RHHW3.md` and `.sha256`
* `hw3/MM_VERDICT_HW3.md`
* `hw3/RESULT_HW3.json`, `R0_CHECK.json`, `hw3_levels.json`, `hw3_name.py`, `FACTS_HW3.md`
* `hw3/out_r0/`, `hw3/out_3/`: the reader outputs
* `hw3/BRIEF_HW3.md`
