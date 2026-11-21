# RESULT — v8F rung β: the world model scores the 117-candidate fan at eval time, zero training

**Status: gate built and tested; NOT RUN.** No trained M2 checkpoint exists, so β has **no accuracy number**. Nothing was trained; no GPU was used.

**What exists.** `v8f_mini_eval.py run --plan beta` ranks the plan's 117 candidates (13 × 9 constant `(a_lon, a_lat)`; `refc.py:2246` ff., pinned `test_anchor_flyability.py:11-22`) with M2's own world model (`HJEPAStack.score_candidates`, level 2, +6 s) against a **predicted** goal latent: S's route timed by P10's pace, rolled through the WM. M2's plan is a 118th candidate, replaced only on strict improvement. Zero tuned parameters. `compare-beta` applies the pre-registered verdict.

**Evidence (MEASURED, CPU).** `test_v8f_beta.py`, 35 tests, green. A constant cost reproduces M2's output bit for bit (planner and end to end); a closed-form world model fixes the argmin; five source mutants and a caller mutant go RED. `--plan beta` reaches `score_candidates` on `[B,118,30,2]` blocks; the default runs it zero times. Perturbing T's heads moves M2's plan, not the goal. Every test touching the changed modules passes; the full `stack` suite cannot run green in this sparse worktree (fixtures outside the cone).

**Pre-registered** (`PREREG_V8F_BETA.md`): PASS needs β < M2 at 6 s at both seeds, |Δ| > 2F, the shuffled-goal control losing, and none of 13 family components separated-worse. Each FAIL names its next lever. **Predicted risk:** RA-1's negative L2 − L1 monotonicity (INHERITED, not M2's WM) points to FAIL-RANKING-DEAD; `signal.recovery` reads it first.

**Needs the PI.** (1) The 60-clip holdout is **in-sample for M2**. (2) `anchors_117_alat_declared.pt` is pod-side: the fan is unchecked against it. (3) The strategic *decision* metric stays PAUSED.

**Manifest (staged, not committed).** `stack/tanitad/models/v8f_beta.py`, `taniteval/taniteval/v8f_beta_eval.py`, `stack/scripts/v8f_mini_eval.py`, `stack/tanitad/models/v8f_mini.py` (WMAux refactor), `stack/tests/test_v8f_beta.py`, `Project Steering/PREREG_V8F_BETA.md`, `Project Steering/GOALS_AND_CLAIMS.md` (row H-V8F-BETA-1), this package. Commands: `HANDOFF.md`.

**Note at landing (2026-10-18, the Master Mind session, not this package's author).** PI item (1) above is RESOLVED for the runs that will produce β's checkpoint: `PREREG_V8F_MINI` Amendment 4 trains M1 / M2b on box 2's full-corpus cache with `--exclude 540:600`, so the 60-clip holdout is OUT of sample for M2b. The `v8f_mini.py` WMAux refactor was checked against the tip before landing: on a synthetic batch, 0 of 131 (M1) and 0 of 276 (M2b) state-dict tensors differ and every loss term is bit-identical, so checkpoints trained before or after this landing load the same. Item (3)'s "PAUSED" is NOT the PI's strategic-layer pause: v8F-mini defines no labelled strategic decision, and S's checkpoint error IS read.
