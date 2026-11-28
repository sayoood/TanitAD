# STOPPERC Amendment 5: how the interval enters S0-L (frozen before any L data)

Card: `PREREG_V8F_STOPPERC.md` (sha256 `70c75176…8877`), section 4, the S0-L line:

> "STOPCUE on L vs B-prime s0, both dumped T1-kin through their own caches. A gain counts only if it exceeds the AUROC floor of 0.037."

## Why an amendment

The S0-L line names the floor but not the interval. The card's sibling AUROC rule, HEAD-UNDERUSES (section 4 table), reads *"exceeds T's own (same windows) by ≥ 0.037, CI lower bound above T"*. Amendment 2 (`AMENDMENT_2_S0P.md`) made that rule paired: the difference is taken on the same episode resample, and the point must be ≥ 0.037 AND the CI lower bound must be > 0. S0-L is the same kind of read: two arms, the same windows, an AUROC difference against the same floor. The programme's estimator rules also require the paired bootstrap for two arms on the same windows, and they make a separated interval necessary for any claim that a lever moved a metric. A point-only rule would let a difference whose interval covers 0 count as a gain.

## The rule (binding for S0-L)

* **GAIN** iff the paired dAUC = AUROC(L) − AUROC(B-prime s0) of STOPCUE's statistic meets both conditions:
  * the point **exceeds 0.037** (strictly, as the card writes it), **and**
  * the **CI lower bound is above 0**.

  The statistic is p(STOP_POINT) = `goal_prob[:, 6]` against `stop_hit` on the `has` windows. The dAUC is computed by `taniteval.ci.paired_stat_cluster_bootstrap` over episodes, with the same resample for both arms, B 2,000 and seed 0.
* Otherwise **NO-GAIN**. The point-only reading is reported beside and decides nothing.
* The window counts must equal STOPCUE's: 2,365 windows, 131 positives, 139 episodes. Otherwise the read is refused.

## What a GAIN does and does not license

* L and B-prime s0 are **one training seed each**. The interval answers "would another draw of EPISODES say this?" and is blind to "would another training run say this?" (`H-ESTIM-SEED-1`). The 0.037 floor is STOPBAL's own replicate-scale figure (`raw/2026-11-10-v8f-stopbal/read/VERDICT_STOPBAL.md:35`, "a floor of about 0.037"). For that reason a GAIN licenses **the next experiment** (L-DAG, "conditional on S0-L", card section 4 table). It does **not** license the claim "the LoRA encoder improves stop perception", which needs a second L seed against the replicate floor.
* NO-GAIN leaves the card's PERC-NULL route unchanged ("otherwise the 416x1024 source (Q5)").

## Data state at freezing (2026-11-13, about 10:45 local)

* No L data has been read. L_s0 is still training on the LoRA pod; its last reading was step 12,000 of 19,164 at 05:47 UTC on 2026-11-12 (LOOP_STATE). The dev box holds no L checkpoint and no L dump.
* What exists: the S0-L read code and its controls (`s0l/`). The controls ran on B-prime s0's mini-eval scores (B-prime against itself), on BAL-NULL_s0 T1-kin npz (BAL-NULL against itself), and on planted arms. None of them involves L.
* Tier T1-kin (first tick, eval-141), a decision-head diagnostic. TACTICAL family only; LONGITUDINAL and LATERAL are not covered by S0-L; STRATEGIC is PAUSED (PI item 25).

Code: `s0l/s0l_read.py::decide` implements this rule. Its test `test_decide_amendment5_needs_point_above_floor_and_ci_lower_above_zero` has a mutant (ignore the CI) that must go RED.
