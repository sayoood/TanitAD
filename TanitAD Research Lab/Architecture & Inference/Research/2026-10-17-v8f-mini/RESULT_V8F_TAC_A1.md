# RESULT: PREREG_V8F_TAC Amendment 1, the per-group oracle (2026-10-21)

**Verdict, as committed: MIXED** (`raw/2026-10-21-tacgrp/tac_groups_read.json`, read by `code/v8f_tac_groups_read.py`).
* Neither committed lever carries half the headroom on its own at both M2b seeds:
  * the categorical **decisions** (the v8F-HJ classifier lever);
  * the **speed constraints** alone (the speed-head lever).
* The committed next read is the joint oracle of the two largest single groups, `lonspeed + londec`. It is in Amendment 2 below.

**The finding inside the verdict, which the committed rule does not decide on:** the constraints carry the headroom **as a set**.
* **Evidence:** MEASURED, `tac_groups_read.json`.
* **Tier:** T0, every arm reads GT.
* **Estimator:** paired episode-cluster bootstrap, 141 episodes, 2,399 windows, B = 2,000.
* **Seed floor:** 2F = 0.4444 m (M1) and 0.2415 m (M2b).

6 s ADE, the oracle arm minus T's own code (Δ m, with its share of the full oracle):

| O reads the GT in | M1 s0 | M1 s1 | M2b s0 | M2b s1 |
|---|---|---|---|---|
| everything (PREREG_V8F_TAC) | −1.296 | −1.063 | −1.204 | −1.051 |
| **`constraints`** (26 dims) | **−1.153 (0.89)** | **−0.955 (0.90)** | **−1.076 (0.89)** | **−0.923 (0.88)** |
| `decisions` (34 dims) | −0.144 (0.11) | −0.076 (0.07) | −0.144 (0.12) | −0.076 (0.07) |
| `lonspeed` | −0.334 (0.26) | −0.209 (0.20) | −0.290 (0.24) | −0.217 (0.21) |
| `londec` | −0.069 (0.05) | −0.052 (0.05) | −0.082 (0.07) | −0.054 (0.05) |
| `goal` | −0.076 (0.06) | −0.027 (0.03) | −0.069 (0.06) | −0.028 (0.03) |
| `latdec` | −0.004 ns | +0.001 ns | −0.005 (0.00) | −0.000 ns |
| `latcons` | **+0.490 (−0.38)** | +0.232 (−0.22) | +0.316 (−0.26) | +0.322 (−0.31) |
| `stoplead` | **+0.255 (−0.20)** | +0.255 (−0.24) | +0.279 (−0.23) | +0.180 (−0.17) |

Every row is separated unless marked `ns`.

**What this settles:**
1. ⛔ **The decision-quality lever is ELIMINATED as the main lever.**
   * A perfect categorical tactical layer (goals + lateral + longitudinal decisions) buys 0.08–0.14 m, 7–12 % of the headroom.
   * That is below 2F at M2b.
   * So a v8F-HJ arm built to improve the tactical *classifier* is not where the 1 m is.
2. ⭐ **The headroom lives in T's continuous constraints, taken together:** 88–90 % at all four checkpoints, well beyond 2F.
3. **The single-constraint rows cannot be read cleanly**, and the reason is in source.
   * `v8f_mini.py:601-602` teacher-forces the **whole** code: GT or T's own, per sample, never a mix.
   * So "GT turn geometry beside T's own predicted speed" is an input O never saw.
   * That is why `latcons` and `stoplead` **hurt** alone (+0.18 to +0.49 m), although the set they belong to carries 0.89.
   * The pre-registration anticipated non-additivity: the composites are read directly, never as sums.

**The four families**, on the deciding composites at M2b s0 / s1 (Δ vs T's own code, same windows and estimator):

| | `constraints` | `decisions` |
|---|---|---|
| LON, speed MAE (m/s) | −0.509 / −0.389 | −0.092 / −0.045 |
| LON, along-track MAE (m) | −1.068 / −0.882 | −0.139 / −0.065 |
| LON, accel MAE (m/s²) | −0.248 / −0.056 | −0.044 / −0.014 |
| LAT, heading MAE (°) | −0.836 / −0.601 | +0.015 ns / +0.024 ns |
| LAT, yaw-rate MAE (°/s) | −0.412 / +0.207 ns | −0.086 ns / +0.012 ns |
| LAT, curvature MAE (1/m) | −0.006 ns / +0.002 ns | −0.002 ns / +0.000 ns |
| LAT, cross-track MAE (m) | −0.166 / −0.197 | −0.022 / −0.027 |

* **LON distance keeping** (headway / time gap / TTC) is **not instrumented in this reader**.
  * It is the open work item of the distance-keeping session card.
  * Stated per family, not dropped.
* **TACTICAL** is unchanged by construction. Only O's input changes, and T's outputs are bit-identical across the arms. T's tactical quality is reported in `RESULT_V8F_MINI.md`.
* **STRATEGIC:** PAUSED (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25).

**The run itself:**
* Box 2, SAFE MODE, the temperature guard: max USB 36 °C, boot unchanged.
* 32 / 32 evals `[ok]`, 18:08–18:15. The code was the remote tip `742fca0e`, with six key files sha-checked; no v8F file differs from `c7134f8`.
* ⚠️ **The chain stopped at its PULL step on a bash trap.**
  * It held its group list in `GROUPS=...`, a bash builtin that silently ignores assignment and expands to the user's gid (197121).
  * So it asked for `eval_M1_s0-tacorc-197121.npz`.
  * No eval was re-run: `box2_ops/finish_v8f_tacgrp.sh` did the pull and the read.
  * Fixed in the chain, pinned by `stack/tests/test_bash_specials.py` (RED on the unfixed chain), and filed in `TRAPS.md`.

**Next (Rule Zero: run in the same push, on the PI's box-2 go):** `PREREG_V8F_TAC` Amendment 2.
* **(A)** the committed `lonspeed + londec` joint oracle.
* **(B)** leave-one-out: GT everywhere except one group. Which constraint group is NECESSARY inside the consistent set?
* **(C)** a dose-response: the constraints moved 25 / 50 / 75 % of the way from T's code to the GT. Does partial accuracy pay?
* (C) is the read that decides whether a **training** arm on T's constraint loss can move ADE at all, before a GPU-hour is spent on one.
