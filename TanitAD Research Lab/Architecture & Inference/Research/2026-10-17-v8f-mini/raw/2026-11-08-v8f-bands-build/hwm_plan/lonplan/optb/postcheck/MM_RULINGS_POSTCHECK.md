# MM rulings on the Option B post check (2026-11-18 ~05:40Z, after FTRSP_s0's post check failed, before any outcome read)

FTRSP_s0 trained its 4,000 steps on box 2 (commit 86da79cd). The chain then stopped at the post check (`C:/Users/Admin/optb_state/STOP.txt`) on two controls of `AMEND_OPTB.md` section 4. Both are **specification defects of the checker**, MEASURED below. Neither is a defect of the run.

These rulings change no outcome bar and read no outcome metric: the planner reads, MECH-MOVED and the outcome table are untouched. They correct two controls so that they test what the card says they test, and each keeps a regression arm that must go RED.

## 1. C5, the Adam-state count: 129 of 129, not 141 of 141

* **The card says:** "Adam state on every `wm.*` tensor except `wm.proj` (141 of 145)". The 145 is the number of `wm.*` STATE-DICT entries.
* **MEASURED on R35's built model** (optb tree = 86da79cd's stack): the 145 entries are **133 parameters + 12 persistent buffers** (`wm.stack.prior_{mean,std,lo,hi}_{1,2,3}`). The optimiser holds parameters only.
* So "every trainable `wm.*` tensor has Adam state" means **129 of 129** (133 − 4 `wm.proj`). The run read exactly 129 of 129. Every trainable tensor received a gradient.
* `wm.proj` bit-equal to the checkpoint and the non-WM tensors bit-equal both passed.
* **Ruling:** the checker counts parameter tensors (`N_WM_PARAMS = 133`). The meaning of C5 is unchanged. Regression arm: 128 of 129 is refused.

## 2. C2, the log identity: tested on the absolute residual in float32 ulps

* **The card says:** the recovered weight equals 1.0 within 1e-4 relative, at every logged step.
* **MEASURED on FTRSP_s0's `train_log.jsonl`** (pulled from box 2, `postfail/`):
  * the recovered weight lies in [0.998588, 1.000933];
  * the ABSOLUTE identity residual at weight 1.0 is at most **1.67 float32 ulps** of `level/L1` (median 0.44);
  * the worst steps are the ones where `rsp/L1` has trained down to 3e-4 to 6e-4, against a level of about 6.
  * The trainer sums the terms in float32. Rounding alone therefore moves the RELATIVE weight by up to ~1e-3 once the term is ~5e-5 of the level.
  * The levels 2 and 3 identities (gamma 100) hold to 1.5e-7 relative.
* A relative tolerance of 1e-4 on the weight is unattainable for a term that trains down. It would test the dtype, not the identity. This is the same class as the AMEND_7 C-WM7 ruling, where float32 rounding alone exceeded a numerical-identity bar.
* **Ruling:** a logged step passes C2 if the weight is within 1e-4 relative, OR the absolute residual is within **4 float32 ulps** of the level.
* **Regression power, all tested and RED** (`tests/test_optb_check.py`, UNIT ok=127 bad=0):
  * a term logged but not added (weight 0) is refused;
  * a weight of 0.5 is refused;
  * a weight of 0.98 is refused with the term at 3e-4 (2 % of the term is more than 4 ulps).
* The real FTRSP_s0 log passes the fixed C2 and fails the old one.

## 3. What happens to FTRSP_s0

* The run is moved aside on box 2 (`D:\tanitad\runs\v8f_optb\FTRSP_s0.postfail_c2c5`, kept), and FTRSP_s0 retrains under the chain.
* Its record of token → training → post check → bank is not rebuilt by hand. A token is never hand-written.
* Cost: about 80 min of box 2 (ESTIMATED from the pilot's MEASURED parts).
* The failed run stays beside as evidence. It is not read as an arm.
