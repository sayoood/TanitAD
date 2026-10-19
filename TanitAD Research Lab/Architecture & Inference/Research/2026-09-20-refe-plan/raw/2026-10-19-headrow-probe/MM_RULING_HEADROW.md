# MM rulings on HEADROW-PROBE (Master Mind, 2026-10-19)

**1. The calls stand as registered.** OUTCOME MIXED.
- NC, DAC and TTC are FEATURE-LIMITED. M is significantly WORSE than L on all three (overfits), and L is within 0.01 of P2.
- EP is INCONCLUSIVE: M - L is positive (+0.0054 / +0.0063), but FEATURE clause (b) cannot hold, because P2's EP row is unconditional (L - P2 +0.266).
- No row is CAPACITY-LIMITED. ⇒ The head-row line CLOSES for NC, DAC and TTC; their lever is the trunk (P6, read about 25 Oct).
- This agrees with `D-REFE-FASTNEG-CENSUS-1` and `D-REFE-PICKRULE-NC-1`.

**2. The EP f = 25 % re-run (the committed arm) is BLOCKED, and DEFERRED jointly with FASTNEG-F25-RERUN.**
- The probe's cost estimate (about 30 min of fit) assumed an f = 25 % bank. S0 exists only at f = 10 % (`D-REFE-FASTNEG-CENSUS-1`); f = 25 % needs about 6.7 dev-box GPU-h (EST).
- Both re-runs share that one build, and run together if it is ever made.
- The expected value of the EP re-run is low (ESTIMATED): a feature-free SLOT lookup scores EP concordance 0.979, above both probes (L 0.962, M 0.967). EP-R1 is already closed at the fit.

**3. The registered C2 is mis-specified: the Master Mind's design error, logged as `RETRACTION_LOG` R52.**
- It read RED in 8 of 8 cells. A probe trained on noise labels over P2's structured features gives a per-replicate RED of up to 0.375. The log-cluster bootstrap holds the trained probe fixed, so the control's own training noise is invisible to its CI, and "any of 5 replicates" multiplies the false-RED rate.
- C2 RED voids only a CAPACITY call (prereg); none was made, so no call changes.
- ⛔ Do not reuse this C2. Read a shuffled-label control against the empirical distribution of noise-trained replicates, with a multiplicity correction.

**4. The slot prior is a finding that qualifies every head-row number.**
- A feature-free lookup by proposal index scores concordance NC 0.829, DAC 0.682, EP 0.979, TTC 0.8185. Every head-row concordance includes this scene-independent component.
- For EP it is nearly all of it: the EP row has learned the proposal vocabulary's progress ORDER, not scene-specific progress.
- ⇒ An EP-led argmax therefore moves toward fast SLOTS irrespective of the scene, which is where the safety rows are blind (census, pick-rule audit).
- For the census: slot may confound speed. Its strata are speed-defined, and its DAC finding (a fast-band deficit) is read as "the scene-dependent DAC information is missing for fast proposals", whether the fast band is slot-defined or not. A slot-stratified re-read is a cheap descriptive check (BACKLOG).

**5. Storage.** `raw/f10/oof_proposals_f10.npz` (86.7 MB) and the 422 MB control tables are NOT in git. They are on the artifacts drive, `D:/Projects/TanitAD-artifacts/headrow_probe_2026-10-19/` (sha256-verified copies; `ARTIFACT_LOCATION.md`). They are regenerable by the landed code from S0's bank.
