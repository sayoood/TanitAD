# RESULT_V8F_KINDX: which kind of T's code is worth fixing, on MIXO's checkpoint X (seed 0, T0)

Run 2026-10-25 on the dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`, 6 torch threads). Pre-registration: `PREREG_V8F_KINDX.md` at tip `eeb2043e`. Nothing staged, committed or pushed. `box2_ship` read only.

**Tier T0** (GT-conditioned oracles): a diagnostic of how O consumes T's code, never a driving claim.
**Evidence class:** MEASURED (ours). Raw data: `eval/eval_M2b-mixo_s0-kindx-*.npz|json`; reader output `READ.json`; controls `controls.json`.
**Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`), B = 2,000, 141 episodes, 2,399 windows. It answers "would another draw of EPISODES say this?" only, not "another training run" (`H-ESTIM-SEED-1`). **One seed (s0) decides only whether a second seed is worth training. No claim about a lever is made.**

## OUTCOME: CANDIDATE (speed)

One kind clears the committed bar 2F = 0.2415 m with a separated paired CI: **SPEED, gain +0.716 m [+0.623, +0.818]**. Every other kind is below the bar (angle +0.095, tte +0.110, geom +0.108; timegap and leadgap are zero within noise).

Committed next step, quoted from the PREREG: *"train X at seed 1 on box 2 (MIXO's argv, `--seed 1`, behind its launch gate) and re-read gain_k at s1. k is a CARRIER only if it clears 2F at s1 too (KIND's both-seed rule). A carrier gets a T-side arm aimed at it, under its own PREREG, along KIND's routing: SPEED → T's speed variance; ANGLE, GEOM or TTE → the encoder (LoRA); TIMEGAP or LEADGAP → lead perception. Several candidates are ranked by gain_k(X) and all go to the s1 read."*
Only `speed` is a candidate, so one candidate goes to the s1 read. Training X at s1 needs box-2 compute: that is the PI's / Master Mind's go, not taken here.

## Controls (all PASS before any gain was read)

| control | value | bar |
|---|---|---|
| X checkpoint sha256 | `c6a62876...126250d`, equal to the committed value | exact |
| structure | 2,399 windows / 141 episodes; eid identical to the banked evals; each json names `M2b_s0_mixo/ckpt_last.pt`; declared manipulations: none 0 dims, union 26, all 60, alpha 1 | exact |
| 1. own code vs banked `eval_M2b-mixo_s0` (and `-tacpred`), per-window ADE6 | max abs dev **2.23e-05 m** (pred array 7.2e-05) | <= 1e-4 m |
| 2. union oracle vs banked `-tacorc-constraints` | **1.06e-05 m** | <= 1e-4 m |
| 2. full GT code vs banked `-tacorc` | **1.13e-05 m** | <= 1e-4 m |
| 3. identity oracle (GT in no kind) vs own code | max per-window gain **0.0**, 0 of 2,399 windows nonzero, `pred` array bit-equal | == 0.0 |
| phase 2: the 34 windows with no v9 row, every oracle arm | gain exactly 0.0 on all 34, in all 8 oracle arms; fallback mask identical in all | == 0.0 |
| reader self-test on B s0's banked KIND evals | reproduces `kind_read.json` s0 gains: max point dev **0.0**, CIs identical | < 1e-9 |

Disclosure: the per-eval log tail printed each arm's mean ADE6 as it finished, so the mean ADE6 of the single-kind arms was visible before `kindx_controls.py 2` and the self-test ran. The controls were then run and passed before the paired read (`kindx_read.py`) was invoked, and the outcome rule has no free parameter.

## gain_k(X) = ADE6_X(own) - ADE6_X(GT in kind k only), the deciding read

ADE6 (m): own **2.1977**, union 1.3967, full GT 1.3710.

| kind | gain_k(X) m | paired 95 % CI | separated | clears 2F = 0.2415 |
|---|---|---|---|---|
| **speed** | **+0.7164** | [+0.6234, +0.8182] | yes | **yes** |
| angle | +0.0952 | [+0.0560, +0.1483] | yes | no |
| tte | +0.1101 | [+0.0748, +0.1456] | yes | no |
| timegap | -0.0036 | [-0.0140, +0.0061] | no | no |
| geom | +0.1078 | [+0.0801, +0.1391] | yes | no |
| leadgap | +0.0007 | [-0.0082, +0.0087] | no | no |

## Beside (decides nothing)

| read | value |
|---|---|
| gain_union(X) | +0.8010 [+0.6912, +0.9221] |
| gain_full GT (X) | +0.8266 [+0.7058, +0.9604] |
| sum of single-kind gains | 1.0266 |
| **A_X** = sum_k gain_k / gain_union | **1.282** (descriptive interval over the bootstrap draws [1.226, 1.342]). B s0 was -0.30. |
| speed share of the union gain | 0.716 / 0.801 = 89 % |

Reading A_X: on B the kinds interfered (A < 0: tte and geom alone made O worse). On X they no longer do (all four non-null kinds are positive), but they are not additive either: A_X > 1 means the single-kind gains overlap, since the six together buy less than their sum.

### gain_k(X) - gain_k(B s0), paired (does O on X use kind k more than B does?)

| kind | gain on B s0 | gain on X | shift X - B (paired 95 % CI) |
|---|---|---|---|
| speed | +0.2704 | +0.7164 | **+0.4459** [+0.3662, +0.5309] |
| angle | +0.0746 | +0.0952 | +0.0207 [-0.0066, +0.0469] ns |
| tte | -0.2375 | +0.1101 | **+0.3476** [+0.2534, +0.4474] |
| timegap | +0.0011 | -0.0036 | -0.0047 [-0.0136, +0.0036] ns |
| geom | -0.4345 | +0.1078 | **+0.5423** [+0.4164, +0.6803] |
| leadgap | +0.0047 | +0.0007 | -0.0040 [-0.0133, +0.0049] ns |
| union | +1.0756 | +0.8010 | **-0.2746** [-0.3584, -0.1899] |

The largest shift is **geom (+0.542 m)**, from the most harmful single kind on B (-0.43) to a small positive one (+0.11). speed gains the most in absolute terms on X. The union gain is lower on X than on B because X's own code leaves less on the table (B own 2.175, union 1.100; X own 2.198, union 1.397), so the pooled ceiling is not the same quantity across the two checkpoints.

### Four families (paired vs X's own code; per family, never pooled)

| family | result |
|---|---|
| **LONGITUDINAL** (speed-only oracle) | speed MAE -0.354 m/s [-0.406, -0.306]; along-track MAE -0.764 m [-0.869, -0.669]; accel MAE -0.054 m/s^2 [-0.097, -0.014]. Union: -0.371 m/s, -0.772 m. Other single kinds: tte speed MAE -0.054, geom -0.030, angle -0.013 (all separated, all small); timegap / leadgap not separated. |
| **LONGITUDINAL, distance keeping (K = 10, 743 lead windows / 77 episodes)** | lambda (mean lead m vs full oracle): own -0.239, speed-only -0.054 (paired CI vs the full-GT oracle [-0.097, -0.014]), union -0.024, full GT 0.000. Min headway / time gap / TTC means: own 27.66 m / 4.14 s / 20.74 s; speed-only 27.41 / 4.11 / 20.93. Lead-violation share 0.67 % (5 of 743) for own, 0.54 % speed-only, 0.40 % union. TTC is censored at 30 s (n_closing 384 of 699). |
| **LATERAL** (speed-only oracle) | cross-track -0.038 m [-0.060, -0.018]; heading, yaw rate and curvature not separated. angle-only: heading MAE -0.391 deg [-0.631, -0.214], cross-track -0.074 m [-0.118, -0.040]. geom-only: cross-track -0.053 m [-0.077, -0.033]. Union: heading -0.437 deg [-0.889, -0.061], cross-track -0.133 m [-0.188, -0.081]; yaw rate and curvature not separated in any arm. |
| **TACTICAL** | identical across all 9 arms by construction (bit-equal blocks, checked): lat macro-F1 0.827 (n = 2,107), lon macro-F1 0.502, goal macro-F1 0.271. |
| **STRATEGIC** | **PAUSED** under the PI's ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` ITEM 25), not a pass. The block is identical across all 9 arms by construction (position error 7.0 / 8.0 / 11.4 m at 30 / 50 / 80, descriptive only). |

### Strata (Amendment 1, descriptive): where each kind's pooled gain sits

Contribution to the pooled gain (n_stratum / N x stratum gain; the three sum to the pooled gain):

| kind | JUNCTION (298 w / 34 ep) | CURVE (140 w / 17 ep) | NO-TURN (1,961 w / 140 ep) |
|---|---|---|---|
| speed | 0.083 | 0.030 | **0.604** |
| angle | 0.016 | 0.045 | 0.034 |
| tte | 0.017 | 0.006 | 0.087 |
| geom | 0.020 | 0.023 | 0.065 |

Within the CURVE stratum alone the angle gain is +0.773 m and geom +0.390 m (17 episodes), but they are weighted by 140 of 2,399 windows, so the pooled gains stay far below 2F. The strata decide nothing; per KIND's own read rule this points to perception of bends (the encoder), not a route input, and rests on 17 episodes.

## What this does and does not say

* It does say: on X, with one seed and T0 oracles, GT speed-cell content is worth about 0.72 m ADE6 (a third of X's 2.198 m, 89 % of the union gain) and no other kind is; O on X now uses partial codes (tte and geom flipped from harmful to helpful), so single-kind gains on X are no longer forced lower bounds.
* It does not say: that speed is a carrier (needs s1 to clear 2F as well), that fixing T's speed code would yield this gain (an oracle is an upper bound on a perfect code, T0), or anything about training variance (`H-ESTIM-SEED-1`: the paired bootstrap is blind to it).

## Deviations from the written plan (none change a kind, cell, bar, tolerance, estimator or routing)

* Identity oracle: `v8f_mini_eval` has no empty group spec, so the driver patches `code_group_index('none') -> []` (declared in the handoff before any data; the eval's own `where(sel, code, code0)` then returns code0). Control 3 confirms it is bit-exact.
* The PREREG's beside table names necessity_k(X) but its 10-eval list has no leave-one-out arm: see the addendum below (decided before data; run after this outcome was written; `READ_outcome_frozen_before_loo.json` is the copy of READ.json at that moment).

## ADDENDUM (run AFTER the outcome above was written; decided before data; does not change the outcome)

`necessity_k(X) = ADE6(GT everywhere except kind k) - ADE6(full GT)`. Positive = O misses kind k when it is the one left wrong. Six leave-one-out evals (`--tac-groups all-<k>`, alpha 1; dims replaced 55 / 57 / 55 / 58 / 51 / 58 of 60; checkpoint X; same 2,399 windows; the 34 no-v9-row windows give gain exactly 0.0 in each). The deciding block of READ.json was byte-identical in value before and after (hash `d31488d7...3ea4ea` of gain / OUTCOME / candidates / union / all / A_X).

| kind | necessity_k(X) m | paired 95 % CI | separated | B s0 (KIND) |
|---|---|---|---|---|
| speed | **+0.412** | [+0.334, +0.497] | yes | +1.985 |
| angle | +0.066 | [+0.034, +0.104] | yes | +0.124 |
| tte | -0.017 | [-0.043, +0.008] | no | +0.457 |
| timegap | -0.004 | [-0.013, +0.004] | no | +0.010 |
| geom | -0.006 | [-0.031, +0.021] | no | +0.444 |
| leadgap | -0.005 | [-0.009, -0.001] | yes (tiny, negative) | +0.008 |

Reading (descriptive, one seed): on X only speed and angle are missed by O when left wrong. tte and geom, necessary on B (+0.46 / +0.44), are not necessary on X, so on X their information is redundant with the rest of the code. That fits A_X > 1 above. Speed is both the only sufficient kind (gain +0.716) and the only large necessary one (+0.412), and its necessity is smaller than its gain, so part of what speed carries is also reachable through other kinds. Nothing here changes the outcome or the bar.
