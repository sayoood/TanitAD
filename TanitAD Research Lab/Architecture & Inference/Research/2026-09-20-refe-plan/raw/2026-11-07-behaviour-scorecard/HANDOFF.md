# HANDOFF -- stream BEHAV (REFe behaviour scorecard + P6 scorer-speed prereg), 2026-11-07

**State: DONE.** Nothing is running. Nothing staged, committed or pushed (per the brief). Folder is outside the repo; the MM lands it.
`$P` = `E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/`. Python: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`.

## What to do, in order
1. **MM lands `PREREG_P6_SCORER_SPEED.md` + `.sha256` BEFORE P6 ends (about 2026-11-08 19Z) and before anyone opens the P6 checkpoint on the C3 tokens.** Verify: `sha256sum -c PREREG_P6_SCORER_SPEED.sha256`.
2. **MM rules on the tier label** (BEHAVIOUR_SCORECARD section 0): the sim's "T1-sim" is a closed loop = T2 under EVAL_DOCTRINE/VOCABULARY. The scorecard stamps "T2-class closed-loop sim" meanwhile.
3. **First agent after P6's end** writes `selc6_dump.py` + `selc6_analyze.py` from PREREG section 3 (differences table), pins them in `CODE_PIN6.json`, then runs dump (GPU, YAX lock) -> 128 harness calls (CPU) -> analyse. ETA ESTIMATED 0.7 h GPU + 1.3-1.8 h CPU.
4. **Sim session** writes `select_suite_tokens.py` from SCORECARD section 3.1, freezes `SUITE_TOKENS.json` + `SUITE_PIN.json`, confirms the extra per-tick fields (section 3.2), then runs tier 1 (10 scenarios, about 10 h) on u400 and the baseline replicate (seed 20261108).
5. `G-METRIC` and `G-H` (SCORECARD section 6) are CPU-only and can be built now.

## Findings that change earlier claims
- `P1_DIAG.md` "true" within-token Spearman (TTC -0.049, NC -0.039) is **not tie-safe**: legacy `argsort(argsort)` breaks ties by proposal index and the true NC/TTC are constant within most tokens. Tie-correct: true defined on 52 (TTC) / 38 (NC) of 154 tokens, about -0.09. Predicted -0.55 / -0.49. P1's direction stands; its "true" numbers do not.
- P1's "41 % is inflated by the zeroed stratum" is a TRAIN-logs statement; on all 154 CRUISE tokens the strata agree (0.407 / 0.413), weighted 0.412.
- P6 has **no lane factor** (6 outputs, rule `navsim_v1`). u400 without lane: 0.366 [0.229, 0.521], paired -0.046 [-0.104, -0.0005]. The P6 test controls for it.

## Re-run (CPU)
`cd $P && python u400_reference.py` -> `u400_reference.json` (must be bit-identical; sha in MANIFEST).
