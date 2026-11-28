# RESULT: PREREG_V8F_PGT (scheduled sampling 0.5 → 0.0), seed 0

**VERDICT, as committed: FAIL-NO-MECHANISM. STOP at seed 0; seed 1 was not trained.**

* **Evidence class:** MEASURED.
* **Tier:** T1-class open loop (eval-141 grid, 2,399 windows, 141 episodes). The arm is conditioned on its own held upper-layer outputs. The mechanism reads use T0 oracle evals, which read the GT code.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, over episodes. It answers "would another draw of EPISODES say this?" only (`H-ESTIM-SEED-1`). F = |M2b_s0 − M2b_s1| = 0.1207 m is the training-seed floor; 2F = 0.2415 m.
* **Raw:** `raw/2026-10-22-pgt/run/pgt_read_s0.json` (the committed read), `run/eval/`, `run/runs/M2b-pgt_s0/`, `run/box2_logs/`.
* **Run:** box 2, chain launched 18:02 at tip `14a27de`, gate PASS (4,512 clips, 19,164 steps, built p_gt 0.5 → 0.0). Training 18:07–22:07, the four evals 22:07–22:11, the read 22:13. Boot unchanged; no guard stop (max USB temperature 45 °C).
* **Dose, from the train logs:** M2b-pgt p_gt 0.499 → 0.00003; M2b 0.998 → 0.200. The lever reached training as declared.

## 1. The committed reads

| read | M2b_s0 | M2b-pgt_s0 | Δ (pgt − M2b) [95 %] | committed bar | met |
|---|---|---|---|---|---|
| **PRIMARY** ADE6 (m), 2,399 windows | 2.175 | 2.205 | **+0.029 [−0.063, +0.149], ns** | < −2F (−0.2415), separated, at both seeds | **no** |
| **M-LEAK** Λ (m), 743 lead windows / 77 episodes | 0.268 | 0.046 | **−0.222 [−0.335, −0.123], separated** | Λ halves (a fall ≥ 0.134) | **yes** |
| **M-DK** S5 tail share (TTC < 5 s at 2 s), 682 windows / 73 episodes | 17.30 % | 18.04 % | **+0.73 pp [−0.60, +2.13], ns** | the +4.42 pp excess over P10 halves (a fall ≥ 2.21 pp) | **no** |

The mechanism needs both arms, so it is NOT met. With the primary also missed, the committed label is FAIL-NO-MECHANISM.

## 2. The four families (M2b-pgt − M2b, paired, same windows)

* ⚠️ **One label quirk in the instrument:** the families block names its second arm `p10`. Its values are M2b's: speed MAE 1.0674 m/s, the value in M2b's own eval json. So every Δ below is M2b-pgt − M2b.
* **LONGITUDINAL**
  * Speed MAE +0.033 m/s [−0.013, +0.086], ns.
  * Along-track MAE +0.057 m [−0.034, +0.172], ns.
  * Accel MAE −0.019 m/s² [−0.066, +0.027], ns.
  * Distance keeping (K = 10, lead windows):
    * min time-gap **+0.0126 s [+0.0012, +0.0286], separated** (slightly more gap);
    * min headway +0.046 m, ns;
    * min TTC −0.23 s [−0.67, +0.18], ns.
* **LATERAL**
  * Heading MAE +0.014°, ns.
  * **Yaw-rate MAE +0.357 °/s [+0.023, +0.741], separated, WORSE.**
  * Curvature MAE +0.0049 1/m, ns.
  * Cross-track −0.028 m, ns.
* **TACTICAL**
  * Per arm, descriptive, not paired:
    * lateral manoeuvre macro-F1 0.816 → 0.837;
    * longitudinal macro-F1 0.499 → 0.486.
  * T's present-slot code RMSE 0.536 → 0.563 SD: MSE Δ **+0.0297 [+0.0141, +0.0474], separated, WORSE**.
    * By group: latcons 0.686 → 0.699, lonspeed 0.452 → 0.486, stoplead 0.815 → 0.838.
    * The PREREG anticipated this: under this arm T trains on S's own outputs more often. It decides nothing.
* **STRATEGIC** (un-paused for v8F-mini, PI ruling 2026-10-17). Per-arm episode-cluster CIs, not paired:
  * position error @30 7.02 → 6.98 m;
  * heading error @30 4.20° → 4.07°;
  * both well inside each other's intervals.

## 3. What the run shows (EXPLORATORY, post hoc, one training seed; it guides the next lever and decides nothing)

1. **The leak was exposure-driven, and PGT removed it.** Λ fell 83 %.
2. **It removed it by making O lean less on T's code, not by making O robust while it uses the code.**
   * The GT-code oracle ADE6 ROSE from 0.971 to 1.293 m: **+0.322 [+0.271, +0.374], separated** (`pgt_read_s0.json` `beside.oracle_ade6_delta`).
   * Own-code ADE6 did not move.
   * So the headroom reachable through T shrank from 1.20 to 0.91 m.
   * ⚠️ M2b's oracle row is the all-goal oracle eval (`-tacorc-all-goal`); M2b's own `-tacorc` predates `tac_code` (WC Amendment 1). M2b-pgt's row is its `-tacorc`.
3. **The tail did not follow the leak.** With the latcons leak gone, the S5 share did not fall, so the leak alone was not producing the tail.
4. **The code still moves the tail.** `raw/2026-10-22-pgt/tail_oracle/pgt_tail_oracle.py`, with its rule written before the run. **Its verdict: `CODE-STILL-MOVES`.**

| S5 at 2 s (lead windows) | M2b_s0 | M2b-pgt_s0 |
|---|---|---|
| own code − P10 | +4.42 pp [+1.23, +8.21], sep | +5.15 pp [+2.27, +8.53], sep |
| GT code − own code | −2.18 pp [−4.68, +0.00], ns (borderline) | **−3.09 pp [−5.59, −0.97], sep** |
| GT code − P10 | +2.30 pp [+0.15, +4.86], sep | +1.99 pp [+0.65, +3.42], sep |
| GT except latcons − GT code (latcons' share) | **+3.31 pp, sep; TTC −1.49 s, sep** | +1.16 pp, sep; TTC −0.09 s, ns |

* The reproduction holds. With the oracle `RESULT_V8F_DK.md` 9.4 used (`v8f_cz/eval/eval_M2b_s0-tacorc.npz`), GT − own = **−2.59 pp [−5.10, −0.41]**, against the banked −2.6 pp [−5.1, −0.4].
* ⇒ After PGT, **the whole GT code still cuts the tail, but latcons no longer carries most of that cut.** The fix moved to the other groups.
* In both arms O still closes early with the GT code: GT − P10 is separated-positive. **Part of the tail is O's own**, as DK 9.4 found.

## 4. Rule Zero: the next levers

* **Already running: `PREREG_V8F_GRID`**, which gives T the 8×20 input (PI go 2026-10-22). Its committed base rule read this result and chose **B = M2b**.
  * That base keeps M2b's code reliance (oracle ADE6 0.971 m) and its latcons-carried tail fix: the right base for a lever that makes T more accurate.
  * Point 4 says the code still has a route to the tail, so GRID is not aimed at a dead channel.
* **This PREREG's committed next is to separate O's longitudinal head from the lateral code slots.** That is an architecture change, so a PI decision.
  * ⚠️ Point 3 is evidence AGAINST spending it on the tail: PGT already removed the latcons coupling's effect on ADE, and the tail stayed.
  * It remains the committed next, and is reported as written.
* **The O-own part of the tail** (GT code − P10 still separated) is not reached by any running lever.
  * GRID's result will show how much of it a better T closes, before an O-side arm is designed.

## 5. Provenance

| artifact | where |
|---|---|
| committed read, check-inputs, gate selfcheck | `raw/2026-10-22-pgt/run/{pgt_read_s0.json, check_inputs.json, gate_M2b-pgt/selfcheck.json}` |
| the four evals (npz / json / log) | `raw/2026-10-22-pgt/run/eval/eval_M2b-pgt_s0*` |
| the run (summary, selfcheck, argv, train log) | `raw/2026-10-22-pgt/run/runs/M2b-pgt_s0/` |
| box-2 logs, guard, telemetry, wrapper | `raw/2026-10-22-pgt/run/box2_logs/`, `run/telemetry_v8fpgt.csv` |
| the tail-oracle read (EXPLORATORY) | `raw/2026-10-22-pgt/tail_oracle/pgt_tail_oracle.{py,json,out}` |
| checkpoint (87 MB, not in git) | dev box `D:/Projects/TanitAD-artifacts/v8f_pgt/M2b-pgt_s0/ckpt_last.pt`, sha256 `982b3ff395c7ec39aac8663f0498b796f3a0f8a8a2697e226d2f0af769262df0` (copied from the chain's pull, cmp-equal); box 2 `D:/tanitad/runs/v8f_mini_full/M2b-pgt_s0/` |
