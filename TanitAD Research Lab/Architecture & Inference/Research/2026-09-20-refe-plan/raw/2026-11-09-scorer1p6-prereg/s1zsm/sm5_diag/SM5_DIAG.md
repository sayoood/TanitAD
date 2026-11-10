# SM5 diagnostic: which scorer head makes the extra slow picks?

Descriptive, decides nothing, adds no arm and no K-ledger key. Follows SM5's committed next step (PREREG_S1ZSM_P6 section 5, zero GPU).

- Evidence class: MEASURED (ours). Artifacts: `sm5_diag.json`, `run.log`; inputs and sha256 are listed in the json (`inputs`).
- Tier: E2 selection tier. 2,000 E2 tokens, CRUISE subset n = 258 (reader's `cruise()` definition). Banked navtest logits of arms already read; no other token set read, nothing fitted.
- Metric: CRUISE slow-pick share = share of CRUISE tokens whose pick has plan v4/v0 < 0.85 (read_s1z.py line 312). Reported in points (x100).
- Attribution: `read_scorer1p6.shapley` called unchanged, with the slow indicator in place of PDMS. Coalition heads' probabilities are replaced by the proposal's PDM truth sub-score and the pick is re-taken by the reader's rule. Value of a Shapley entry = change in slow share when that head becomes truth, so **negative = the head produces slow picks**. "Single swap" = only that head replaced.
- Estimator: paired log-cluster bootstrap over E2's logs, 10,000 draws, seed 20260927, 95 % percentile; the draws are the same for every arm. Per-token Shapley values (linear) feed the bootstrap.

## Controls

| control | result |
|---|---|
| C1 slow shares reproduce the read | PASS: S1ZSM 0.2132, P6 0.1783 (0.21318 / 0.17829) |
| C2 empty coalition = own share | PASS: max abs error 0.0 (all arms) |
| C2 full coalition | NOT identical across arms: 15.50 (P6), 14.34 (S1ZSM), 14.73 (S1Z), 15.12 (S1Z1); spread 1.16 points. Heads C and DDC still come from the arm's logits in the pick rule, so the arm still matters. Reported, not a failure. |
| C3 efficiency | PASS: max abs error 1.8e-15 (< 1e-9) |
| C4 mutation (P6 logits in an arm slot) | PASS: every arm - P6 field (share, 4 Shapley, 4 single swaps, v_empty, v_all; point, lo, hi) is exactly 0.0 |
| P6_control logits | bit-equal to P6's logits (banked file `P6_control/e3_logits.npz`) |
| Proposals | bit-identical to P6's on E2 for S1ZSM, S1Z, S1Z1 |
| Item 4: which arm is the read's `shapley_split_E2` (PDMS)? | **S1ZSM.** Recomputed S1ZSM: v_empty 92.2682, v_all 98.1228, NC 0.666, DAC 1.159, EP 1.709, TTC 2.320 (matches the read to 1e-9). P6 gives 92.2357 / NC 0.633, DAC 0.918, EP 1.774, TTC 2.562; S1Z and S1Z1 differ too. |

## Per arm (points x100, CRUISE n = 258)

Slow share (95 % CI), and Shapley value per head (negative = head causes slow picks).

| arm | slow share | NC | DAC | EP | TTC | v_all (all four truth) |
|---|---|---|---|---|---|---|
| P6 | 17.83 [11.36, 25.19] | -1.32 | +0.48 | +1.91 | -3.39 | 15.50 |
| S1ZSM | 21.32 [14.74, 28.67] | -2.75 | -0.94 | +1.07 | -4.36 | 14.34 |
| S1Z | 22.09 [15.41, 29.54] | -3.07 | -1.00 | +1.07 | -4.36 | 14.73 |
| S1Z1 | 19.77 [13.50, 26.62] | -2.10 | +0.16 | +1.32 | -4.04 | 15.12 |

Single-head swap effect (change in slow share when only that head is truth):

| arm | NC | DAC | EP | TTC |
|---|---|---|---|---|
| P6 | -0.78 | +0.39 | 0.00 | -4.26 |
| S1ZSM | -3.10 | -0.39 | +0.39 | -5.81 |
| S1Z | -4.26 | 0.00 | -0.39 | -6.20 |
| S1Z1 | -2.33 | 0.00 | -0.78 | -5.04 |

## Arm minus P6, per head (paired, same draws; 95 % CI)

Shapley value difference (negative = the arm's head produces more slow picks than P6's head):

| arm | slow share diff | NC | DAC | EP | TTC |
|---|---|---|---|---|---|
| S1ZSM | +3.49 [+0.69, +7.02] | **-1.42 [-2.74, -0.26]** | -1.42 [-3.46, +0.11] | -0.84 [-2.36, +0.56] | -0.97 [-2.43, +0.24] |
| S1Z | +4.26 [+1.10, +8.05] | **-1.74 [-3.13, -0.52]** | -1.49 [-3.52, +0.04] | -0.84 [-2.54, +0.73] | -0.97 [-2.49, +0.42] |
| S1Z1 | +1.94 [-1.65, +5.90] | -0.78 [-1.78, +0.14] | -0.32 [-1.31, +0.62] | -0.58 [-2.59, +1.26] | -0.65 [-2.77, +1.30] |

Single-swap difference:

| arm | NC | DAC | EP | TTC |
|---|---|---|---|---|
| S1ZSM | **-2.33 [-4.60, -0.44]** | -0.78 [-4.17, +2.24] | +0.39 [-3.10, +3.79] | -1.55 [-4.85, +1.15] |
| S1Z | **-3.49 [-6.35, -1.12]** | -0.39 [-3.76, +2.62] | -0.39 [-4.17, +3.32] | -1.94 [-5.17, +1.19] |
| S1Z1 | -1.55 [-3.54, 0.00] | -0.39 [-2.14, +1.31] | -0.78 [-4.26, +2.62] | -0.78 [-4.96, +3.11] |

## Reading (descriptive)

- The extra slow share of S1ZSM and S1Z over P6 (+3.5 and +4.3 points) is led by **NC**, the only head whose arm - P6 CI excludes 0 (Shapley and single swap, for both arms). DAC is second (point -1.4, CI touches 0); EP and TTC are smaller and their CIs include 0.
- **TTC is not the head that differs from P6.** TTC is the largest slow-pick producer in absolute terms for every arm including P6 (-3.4 to -4.4), but its arm - P6 difference is -0.97 [-2.43, +0.24] for S1ZSM. Removing TTC from the twin loss is therefore not supported by this split as the lever for the extra slow picks; the data point to NC (and weakly DAC).
- Caveat: n = 258 CRUISE tokens (one token = 0.39 points), and the four Shapley differences sum to -4.65 against a +3.49 share change because the full-truth value also differs (v_all 14.34 vs 15.50, heads C and DDC). The CIs of the three non-NC heads are wide. One seed per arm; the bootstrap is blind to training variance (H-ESTIM-SEED-1). S1Z1 does not separate from P6 on the share itself.
- Ordering among heads is by point estimate; no head-versus-head test was run.
