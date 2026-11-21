# MM rulings on the DSCALE10K build (2026-11-16 ~19:45Z)

These rule on the four questions in `BUILD_REPORT_DSCALE10K.md`. The card is `../card/CARD_DATASCALE_TRAIN_DRAFT.md`, frozen, sha256 `48a74901c1f4e307d1d573062694055732d68a0fcff99a1ea8afb09d9cf5d0ec` (commit `b65f9272`). Nothing has trained. Every ruling below is made before any real-data check has run.

1. **Singleton `(n_rows, k0)` groups in the R_s0 derangement.**
   * Allowed up to **100 clips (1 % of the 10k)**, with `--max-singletons 100`. The default stays refuse.
   * A singleton keeps its own labels, so it is a fixed point. The gate token records the real count, and it is reported beside the R read.
   * Why: R must show that the new clips' labels reach the model. A fixed-point fraction up to 1 % can only weaken R's degradation, which makes NO-REACH more likely, never less. It cannot fake a reach.
   * Above 100 the merge refuses. The derangement is then redesigned (pad-aware groups) under an amendment.
2. **C-MASK class set: every goal class whose provenance CONTAINS `vlm`** (the tool's default, not `exact=True`). This includes `LANE_CHANGE_L/R` (`sam3|vlm`).
   * The card rule is "masks, never imputation". Over-masking loses some signal but never imputes a label the new clips do not have.
   * The goal F1 split by provenance (card section 8) shows what the mask costs.
3. **md5 acceptance per arm role:** ACCEPTED as built.
   * B takes the corpus md5 `83e30bd3…`; U and UE take the union md5; R takes the R md5.
   * The union and R md5s enter the set only after C-MERGE passes.
4. **Token seal tamper-evident, not tamper-proof:** ACCEPTED. It is the same property as `strat1_check.py`. The supervisor-side refusal plus the land-time audit cover it.

**Unverified inputs, to fill at the pod step:**
* The card's full sha256 is above.
* The corpus join md5 (prefix `9ebc7a48`) and the C-DISJ digest-json shapes are read on the pod before `mint`.
