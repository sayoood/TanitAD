# MM reading: why Option B's response term lowered the gain (2026-11-18). WORKS-AS-WRITTEN-BUT-WRONG-OBJECTIVE

- Companion to `../decide/MM_READING_OPTB.md` (OPTB-HARM at both chassis).
- Evidence class: MEASURED from banked logs and reads. CPU; zero training; no new model compute. File:line sources are in `DIAG_OPTB_GAIN.json`.

## Finding
1. **The term trained, and is wired as written.**
   - `_response_loss_fixed_k` (hjepa.py at 86da79cd, lines 897-918) minimises mean (g - 1)^2.
   - g is the K = 10 rollout contrast read through the frozen train-fit R35 readout.
   - `rsp/L1` fell from about 0.025-0.030 to about 0.0004 in all four FTRSP arms, and train g ends at 0.99-1.005.
   - Ruled out: a sign error, a self-target, readout drift, and an untrained term.
2. **Its target sits below where R35 already was.**
   - Through the term's own frozen readout on the eval clips: R35 reads 1.255 / 1.246 (this matches AMEND-1 section 1.5) and FTNULL reads 1.19-1.41.
   - FTRSP reads 0.98-1.01, exactly its target.
   - Minimising (g - 1)^2 from about 1.25 therefore **shrinks** the action response by about 20 %.
   - The eval gain in `mech.json` uses a ridge refit on the eval clips, a different readout. It shows the same shrink: FTNULL 0.910 to FTRSP 0.749 on the s0 pair.
3. **MECH-MOVED could not be met by this objective**, and lambda 0.3 is a smaller dose of the same shrink. It can only reduce the harm.

## Zero-cost check A: acceleration error against response gain, per arm (`A_GAIN_VS_ACC.json`)
- Inputs: frozen-probe gain from each arm's `ia3pk_<tag>.json`.
- Per-arm `acc_mae` (band 0-2 s) is computed two ways:
  - FT-NULL: the floor points in `DECIDE_OPTB.json`;
  - FTRSP: its FT-NULL partner plus the paired delta.

| chassis | arms | slope d acc_mae / d gain | r |
|---|---|---|---|
| 0 | all 5 | -1.00 | -0.99 |
| 0 | FT-NULL only (3) | -1.05 | -0.17 |
| 1 | all 5 | -0.75 | -0.96 |
| 1 | FT-NULL only (3) | -0.26 | -0.71 |

**Reading.**
- The direction is the same everywhere: a higher response gain goes with lower acceleration error.
- The all-arm rows are driven by the term-on against term-off contrast, so they are confounded with the term itself.
- The term-off rows have n = 3 per chassis and a narrow gain range. This is descriptive and **HYPOTHESIS-grade, not decision-grade**.

## Next lever
- **Proposal:** replace the card's lambda 0.3 arm with a **target arm**: g* = 1.5 (raise the response), with a neutral control at g* = 1.25 (hold R35's gain; it must read inside the FT-NULL floor).
- Same pairs and floors, four runs, about 5.5 box-2 h (1.36 h per run MEASURED on this chain).
- What this needs:
  - a new constant in LevelConfig with its G-DVB entry, and the gate;
  - a card amendment written before any run;
  - the PI's go, because it changes an arm.
