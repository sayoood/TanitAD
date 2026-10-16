# PREREG_SAFEHEAD addendum A1 — the objective lever, queued behind the safety-head arms

Written 2026-10-16 after the identity controls passed (`controls.json` ALL_PASS) and BEFORE any safety-head J or proxy number was read. sha256 in `PREREG_SAFEHEAD_A1_TTCPEN.sha256`.
Why now: PREREG_SAFEHEAD.md s.8 names the objective as SH-NO's next lever, and Rule Zero says a failed arm must leave its next arm behind, run. The T-horizon arms finish after this agent's budget, so the lever is queued now (`run_queue2.sh`, starts after `ZZSH_QUEUE_DONE`), with its rows fixed here, whatever the SH rows turn out to be.

- **Arm FULL_TTCPEN**: FULL (no separation), EPO's TRAINING reward `R' = R - 0.25 * (1 - ttc)` (`sh_ttcpen.py`; one value, pre-set, never tuned). 0.25 takes TTC's effective weight in the reward from NC*DAC*5/12 = 0.42 to about 0.67.
- **Measured with the ORIGINAL R**: J, f and every proxy (the Evaluator's R is built from `SF.epo_reward`, which the lever never touches). Control: FULL lam 0 (the same as FULL's). f = Delta J / Delta J_FULL.
- **Horizons**: P K = 50 (f only; FULL shows no TTC loss at P) and T K = 250 (the decision).
- **Rows** (same metrics, bar and estimator as PREREG_SAFEHEAD.md s.5; under V-OK only, else OBJ-UNTESTABLE-TTC):
  - **OBJ-GO**: f_min >= 0.5 at both horizons AND TTC, NC, DAC conditions hold at T K = 250. Next: add `--epo-lam-ttc` to `rl_reward.RewardCfg` / `scorer_finetune.py` (default 0) and queue one A40 arm (PI decision). If an SH design also reads GO, the cheaper change (the reward flag, no inference cost) is proposed first.
  - **OBJ-PARTIAL**: safety holds at T K = 250 and 0.2 <= f_min < 0.5.
  - **OBJ-NO**: a safety condition fails at T K = 250 or f_min < 0.2. Next: constrained (Lagrangian) EPO, maximise J s.t. E_pi[TTC] >= E_piA[TTC] - 0.005, dual weight updated per update.
