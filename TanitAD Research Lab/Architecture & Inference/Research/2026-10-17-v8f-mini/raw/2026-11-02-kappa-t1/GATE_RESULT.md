# G-KAPPA result (H-KAPPA-T1-1 Part 1): lever (a), the curvature-bounded decode

Card `PREREG_FROZEN.md` sha256 `9564d568...046f72c`, frozen 2026-11-02T15:44:58+0100 before any gate number. Output `raw/GATE_KAPPA.json` (run log `logs/gate_kappa.out`), script `code/gate_kappa.py`. Tier **T1-kin**. Evidence class **MEASURED (ours)**. Estimator: **none**. This is an exact point decomposition of the window-mean excess: it picks a lever and makes no claim. Inputs are the matched-thread (16) S runs `F:/TanitAD-scratch/smooth_t1/out_m16/SMOOTH_*`, read only.

**Known answers: all PASS.** KG1: the row here, applied to the banked DAG-OACC s0 Y and to P10, equals the landed arrays `gapdecomp/raw/arrays/CTRL_s0.npz` (max abs dev 8.9e-16 / 4.4e-16 1/m). KG2: the partition sums to E at every run (<= 1e-12). KG3: X against itself gives exactly 0.

**Decision: (a) s_HIGHK >= 0.5 at both DAG-OACC seeds.** So the arm is S + `project_feasible` box (A_MAX 4.0, KAPPA_MAX 0.2, mu=None, dt 0.2).

| run | E = curvature MAE X - P10 (1/m) | s_DIP | s_HIGHK | s_REST | f_LF | dip windows | HIGHK steps | HIGHK share from steps < 2 m/s | rho_a (beside) |
|---|---|---|---|---|---|---|---|---|---|
| DAG-OACC s0 | +0.0578 | 0.337 | **0.838** | -0.175 | -0.343 | 46 | 4,613 | 0.555 | 1.71 |
| DAG-OACC s1 | +0.1212 | 0.276 | **0.805** | -0.081 | -0.025 | 63 | 4,962 | 0.585 | 1.33 |
| SOUP s0 | +0.0919 | 0.401 | 0.687 | -0.088 | -0.111 | 74 | 5,468 | 0.534 | 1.41 |
| SOUP s1 | +0.1067 | 0.252 | 0.836 | -0.087 | -0.080 | 55 | 5,282 | 0.618 | 1.37 |

N = 2,364 / 2,365 windows with a finite row on both sides (of 2,399). E matches SMOOTH-T1's 1-thread X - P10 points (+0.0578 / +0.1210) to the 4th decimal.

**Beside (decides nothing).**
* About 55-62 % of the HIGHK part sits on steps with mean speed under 2 m/s. The excess is mostly heading noise over short arcs, where curvature = dheading / ds blows up.
* f_LF < 0 at every run: low-passing the executed path across the seams would put X *below* P10. So the excess is high-frequency on the executed path, though S already smooths inside each plan. (c) is ruled out on this evidence.
* rho_a > 1: clipping kappa_X to +-0.2 on the executed path, with no re-integration, removes more than the whole excess. This is the upper bound a per-knot curvature cap could reach on this row. It is not a prediction for the arm, because the projection re-integrates positions.
* The banked no-S Y of DAG-OACC has the same composition (s_HIGHK 0.90 / 0.82, s_DIP 0.19 / 0.25), so S did not change where the excess lives.
* Yaw-rate MAE splits differently (E -1.38 / +4.14 deg/s; REST dominant at s0, HIGHK at s1). This is descriptive only.
* The v0 3-8 m/s windows (586) carry 0.023 / 0.039 of E (40 % / 33 %).

**Tool-smoke observation (32 windows, not a gate number).** On real plans the box moves executed knots (1-10) by a mean of 0.079 m, p99 1.3 m, max 10.2 m. The worst case is a plan that launches at 3.4 m/s from v0 1.19 m/s and then accelerates at more than 4 m/s^2: the accel clamp holds its speed down. That is the mechanism behind the card's HARMS prediction (20 %). The committed HARMS next arm is the kappa-only projection.
