# AMENDMENT 1 to PREREG_P6_SCORER_SPEED.md (sha256 8e8aa3ca..., H-P6-SCORER-SPEED-1)

**Date:** 2026-11-08 ~16:10Z. **Written before** P6's final checkpoint exists and before any P6 output was read on the C3 tokens.
**Disclosed:** the Master Mind has seen P6's training-log lines while checking the job is alive (step, losses, the labeller's on-policy pick/random/best on training sets). Those are training telemetry, not the read's tokens, and no value below was chosen from them.
**No bar changes.** Thresholds of (A), (B), (C) and the verdict table are as frozen. The rulings answer `p6_read/AMBIGUITIES.md`; the read's tool is `p6_read/p6_read.py` (`p6_read/CODE_PIN.json`).

1. **Code pins.** `selc_dump.py`, `selc_score.py` and the confirm `CODE_PIN.json` changed under `refe_selection_confirm_20261107/AMENDMENT_1_FRAME_CONTROL.md` (2026-11-08 ~08:15Z), because the frame control crashed on tokens with a missing OpenScene frame. The amended hashes are the pins: `1de68675...`, `0e76e98b...`, `a4b84011...`. The read's own files are pinned in `p6_read/CODE_PIN.json`.

2. **The frozen u400 reference is the content with sha256 `1ad5776a...`,** held at `p6_read/u400_reference.pinned.json` and produced by re-running the pinned `u400_reference.py`. The file at the card's path hashes to `4ba8043b...` and differs only in `R2_spearman_legacy`, the argsort-of-argsort estimator already retracted because its sign depends on the environment (RETRACTION_LOG 2026-11-07 ~20:50). The verdict does not use that block. The banked file is left as it is.

3. **Label correction, section 2.** TTC "-0.546 (n = 154)" is the paired-subset value (n = 52). The all-154 value is -0.479. Section 4's S_h over all CRUISE tokens where defined is what (B) uses, and both are printed.

4. **Verdict table, naming only.** N2 and N3 may both apply; all matching sub-cases are listed. A new label **N4** covers (A) failing because a gap lies above +0.15 with (C) failing: the heads now favour speed beyond the truth and the picks still slow down. It stays NOT PROVEN under the card's "anything else" clause. HOLDS and REFUTED are unaffected.

5. **Added gate G6-FP32.** The 64-proposal dump runs in fp32. The read fails as INSTRUMENT FAILURE if more than 1 % of x[19] values in [32, 64) m lie on the 0.25 m grid. Reason: under bf16 autocast, the 0.2 s chord speed v4 moves in 1.25 m/s steps at cruise, as measured on the sim's server on 2026-11-08. u400's reference reads 0 of 16,712.

6. **G6-DISJOINT.** `splits/navtrain.yaml` and the pod's metric-cache log list are both required to have empty overlap with the read's logs. The metric cache is accepted as the proxy for the logs of `--targets`.

7. **P6 navtest artifacts.** Passed as `--p6-seam` and `--p6-csv`. A missing or mismatched file is a red gate, never skipped. The result records their sha256.

8. **UNVERIFIED and stated as such:** `selc6_dump.py` and `selc6_score.py` have compiled but have not yet run a forward or a harness call. RUNBOOK step (b), a 3-token GPU smoke, runs first.
