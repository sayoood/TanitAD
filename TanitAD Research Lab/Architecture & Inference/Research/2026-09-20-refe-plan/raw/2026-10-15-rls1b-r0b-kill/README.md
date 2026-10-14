# RL-S1b arm R0b: KILLED by K3 at the LIVE monitor (outcome O2 of PREREG_RL_S1 section 10)

Banked 2026-10-15 ~01:35 Berlin by the Master Mind from the A40 pod (`/workspace/data/rls1/arms/R0b/`).

**Arm.** R0b = R0 with `--epo-lam 0.25`, seed 0, read against C0. Launch-gate PASS: `token_verify.json`, `launch_gate.log`.

**What happened (MEASURED).**
* **The fail-closed monitor worked live.** `monitor.log` shows it reached `ZZRLS1_MONITOR_READY rss_mb=57` and peaked at 93 MB. It checked step0,
  u250 and u500 live, then `ZZRLS1_KILL R0b K3 g3/g5 bound breached at update250 and update500`. KILLED.json is stamped 2026-10-14 23:17:44Z.
  This is the first live K3 kill in the programme. R0's kill was applied after the fact, because its monitor had died (aa49ff9 fixed that).
* **Same pattern as R0.** g5 failed at u250 AND u500 on TTC; NC held; g3 held. `kill_checks.jsonl` stores g5 as booleans. The TTC deltas
  themselves are recomputed by the calibration read (next lever).
* **The quarter dose bought nearly R0's full effect.** RL-val PDMS against C0 at the same update:

| update | R0b (`--epo-lam 0.25`) | R0 (full dose) |
|---|---|---|
| u250 | +0.532 | +0.60 |
| u500 | +0.899 | +0.93 |

**Hypothesis.** AdamW normalises each parameter's step, so scaling a gradient-dominant term by 0.25 barely changes the update. That is the risk
PREREG_RL_S1 section 10.2 named.

**Pre-registered outcome: O2.** "The linear-response dose model is REFUTED on this rig: the trade is not dose-separable." Next levers:
* (i) zero-A40: a TTC / NC head calibration read on the banked C0 / R0 / R0b dumps, dev box;
* (ii) a separate safety head EPO cannot move. That is an architecture change: PI decision.

As registered, the chain stopped after R0b's kill, so C1 and R1b did not run (about 7 A40-h saved).

**Data.** HF private `Sayood/tanitad-refe-rls1-r0b`: the arm in 20 files, plus u250 and the exit checkpoints. `chain_excerpt.log` holds the
chain's lines for this arm.
