# K-HEAD-D2 -- NO-GO (sealed rule). D2 0.7357 vs LH 0.7305: delta +0.0052 [-0.0522, +0.0631]; the bar was +0.030 with the CI excluding 0. Gap to the point bar 0.0248.

**Tier:** T0 label-level diagnostic on logged eval139 frames (heads read the frozen refcv8 EMA-48,852 trunk's t0 z_tac / the measured t0 ego state; no planner, no roll).
**Estimator:** Mann-Whitney AUC; paired episode-cluster bootstrap over the eval EPISODES (K-HEAD's loop, B 2000, seed 0, every arm on the same draws). Answers "another draw of eval EPISODES" only; blind to training
variance (one refcv8 seed, one head fit) and inference variance (heads deterministic). **Function class:** a 3-feature ridge logistic head -- a NO-GO is "this linear head does not clear LH by the bar", not "a0 carries no decel information".
**Registration:** `PREREG_LHA0_D2.md` sha256 `3f37a441d0df0cd2b2e287f56fb3db7250d63d7f285d3003f6c767a788d2abc8`, sealed 2026-10-26 21:35:44 Berlin (before any eval139 file was read by this arm); script pins verified at run time.
**n:** 2,495 windows (639 decel, 1,856 cruise), 136 episodes; d = 3 (D2) vs 513 (LH). Without the 3 no-sidecar clips: 2,424 windows (632 decel), 133 episodes.

## Instrument gates (all PASS; the run would have STOPped before any D2 score otherwise)
* **LH reproduced first:** population s6_valid 3,668 / compliant 3,251 / GT-decel 639 / GT-cruise 1,856 / union 2,495 / 136 episodes; LH point AUC 0.730459 (rounds to 0.7305); LH's CI [0.6749, 0.7855] also reproduces K_HEAD.json to 1e-4.
* constant arm reads exactly 0.5000; unit-weight WAuc equals the rank AUC for all 9 scored arms; nominal-dt fallback rows 102 windows / 3 clips / 71 union windows (as documented).

## Primary and controls (full population; the bar reads this)
| arm | AUC | 95% CI (episode bootstrap) | paired delta vs LH | 95% CI of the delta | clears bar? |
|---|---|---|---|---|---|
| **LH** (K-HEAD score) | 0.7305 | [0.6749, 0.7855] | - | - | - |
| **D2** (primary, lambda 10) | **0.7357** | [0.6930, 0.7808] | **+0.0052** | [-0.0522, +0.0631] | **NO** (point 0.0248 short; CI includes 0) |
| constant | 0.5000 | [0.5000, 0.5000] | -0.2305 | [-0.2855, -0.1749] | control reads 0.5 exactly |
| D2 permuted within episode (a0, psi_dot0; seed 0) | 0.6372 | [0.5960, 0.6803] | -0.0933 | [-0.1544, -0.0289] | NO -> the control does NOT clear: **not VOID** |

## Beside (never gating)
| arm | AUC | 95% CI | delta vs LH [95% CI] |
|---|---|---|---|
| raw -a0 (no fit) | 0.7254 | [0.6829, 0.7671] | -0.0051 [-0.0644, +0.0526] |
| fusion z(D2) + z(LH), equal weight, no fitted weight | **0.7755** | [0.7280, 0.8217] | **+0.0450 [+0.0185, +0.0716]** |
| D2NOACC (accel rows dropped from the fit) | 0.7225 | [0.6723, 0.7753] | -0.0080 [-0.0653, +0.0524] |
| D2NOACC permuted | 0.6485 | [0.6004, 0.6984] | -0.0819 [-0.1435, -0.0146] |

Contrasts: D2 - permuted D2 = **+0.0985 [+0.0625, +0.1364]** (the per-window value matters; the episode-level a0 / psi distribution alone does not carry it); D2 - (-a0) = +0.0103 [-0.0097, +0.0335]; fusion - D2 = +0.0398 [+0.0059, +0.0739]; D2NOACC - D2 = -0.0132 [-0.0322, +0.0058].

## Clock coverage (descriptive; the bar reads the full population)
Without the 71 union windows of the 3 no-sidecar clips: LH 0.7376, D2 0.7417, delta +0.0042 [-0.0547, +0.0591] (gap to bar 0.0258); permuted D2 delta -0.0959. The fallback is not what decides it.

## Reading (human-written after the machine verdict; the verdict word is the sealed rule's, unchanged)
1. **NO-GO, and not close on the CI clause.** The paired delta's bootstrap SE is 0.029 (the two heads are only weakly correlated: Spearman 0.33 on eval), so the CI half-width is ~0.058. The "CI excludes 0" clause therefore binds harder than the +0.030 point bar:
   a GO needed a point delta of roughly +0.057, not +0.030. D2 missed both.
2. **The fresh-split advantage of a0 did not carry over.** Fresh OOF: raw -a0 0.7699, LH-features 0.6699, D2 0.7614 (+0.0915 over LH's OOF). Eval139: raw -a0 **0.7254**, LH **0.7305**, D2 0.7357. a0's AUC fell by 0.045 and LH's rose by 0.061; the "a0 dominates z_tac"
   reading was a property of the fresh split's LH OOF being weak, not of a0 being strong. (POST-HOC by v0 band, `LHA0_D2_POSTHOC.json`: -a0 on eval139 0.838 / 0.699 / 0.716 / 0.705 in v0 bands [0,3) / [3,8) / [8,15) / [15,99) vs fresh 0.885 / 0.744 / 0.766 / 0.797;
   LH is the stronger arm in the middle bands (0.717 / 0.771 at [3,8) / [8,15)) and weaker below 3 m/s and above 15 m/s, where a0 / D2 lead.)
3. **The per-window a0 value is real.** The permuted control collapses to 0.637 (D2 - permuted +0.0985 [+0.0625, +0.1364]) -- the opposite of the fresh diagnostic's C4, where 72 % of the gain survived the shuffle.
4. **The one positive beside row is a COMPLEMENTARITY signal (the fusion arm was pre-registered as a non-gating beside control, so this number is a registered read; promoting it to a claim is post-hoc):** equal-weight fusion of D2 and LH reads 0.7755 = **+0.0450 [+0.0185, +0.0716] over LH** and +0.0398 [+0.0059, +0.0739] over D2.
   Fresh-OOF consistency (fresh-only compute, POST-HOC): fusion 0.7569 vs LH's OOF 0.6699, +0.087 [+0.060, +0.116]; but on fresh it adds nothing over a0 alone (fusion - D2 -0.0045 [-0.034, +0.023]), so the direction (fusion >= LH) holds on both splits while "fusion > a0" holds on eval139 only.
   It is NOT a result: it was not the gating arm, it uses eval-population z-scoring (a deployable fusion needs FROZEN fit-side z-statistics), and a confirmation needs windows not already seen.
5. Deployment caveat carried from the diagnostic: a0 here is the backward difference of the DATASET's smooth offline egomotion speed, not an IMU.

## Committed outcome applied
NO-GO: gap reported above; **stage 2 (`PREREG_LHA0_D2_ROLL_DRAFT.md`) is NOT drafted**; the fresh-split a0 finding stays an unconfirmed post-hoc observation that eval139 did not confirm at the bar.
