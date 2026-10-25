# PREREG_V8F_HACC: a per-step blend weight fitted with an acceleration term and no monotone constraint (written 2026-10-25, before any HACC number)

**Status: pre-registered.**
* Zero training; dev-box CPU. Under `PI-V8F-2026-10-25b` / `-25c` / `-25d` (box 2 and the dev-box CPU only).
* It is an additional zero-training arm toward HSPEC's dominance row. It does **not** displace HSPEC's committed next step, OACC, which is queued on box 2 after FLAT.

## Why (MEASURED, descriptive; no bar was registered on it)

HSPEC failed dominance over P10 on exactly two rows, accel MAE and curvature MAE (`RESULT_V8F_HSPEC.md`). A per-step profile of H's accel error, `raw/2026-10-25-hacc/accel_profile.{py,log,json}`, shows where that gap sits:
* **It is not spread over the horizon.** It is T1-class open loop, eval-141, both seeds.
* **The instrument is tied to the family metric.** Its per-window mean reproduces the banked `fw_lon_accel_mae` of H and P10 to 9.5e-7 / 4.8e-7, which is float32 epsilon.

| band | H − P10 per accel cell, s0 (m/s²) | s1 | reading |
|---|---|---|---|
| 2.2–2.6 s (cells 11–13) | +0.59 / +0.63 / +0.10 | +0.70 / +0.60 / +0.29 | **H is worse than B AND P10 here.** HBLEND's along weight ramps 0.12 → 0.50 over steps 10–14 (`hblend/weights_M2b_s*.json`). A steep w(t) puts curvature into the blend, since (w r)″ = w″r + 2w′r′ + w r″ |
| 5.2–5.8 s (cells 26–29) | +0.31 / +0.65 / +0.23 / +0.91 | +0.18 / +0.34 / +0.16 / +0.32 | B's own tail is rough. HBLEND's isotonic w is at its maximum there, so H keeps the roughness |
| 0.8–1.6 s (cells 4–8) | +0.24 / +0.13 / … / +0.08 | +0.18 / +0.09 / … / +0.03 | B is very rough (2–3.6 m/s²); H's small w shrinks it only partly |
| 2.8–5.0 s (cells 14–25) | −0.03 … −0.17 | −0.13 … +0.04 | **H is already better than P10 on acceleration** |

* HBLEND's weight is non-decreasing **by construction**, so it cannot dip where B is rough or ramp gently where B turns.
* HSPEC's map D3 smoothed w with an ad-hoc moving average and moved the s0 gap only from +0.104 to +0.080.
* This arm fits w with the acceleration error **in the objective**, and without the monotone constraint.

## The arm (one lever: the per-step weight's objective)

For each seed s (X = `M2b_s0` / `M2b_s1`, HBLEND's checkpoints) and each axis a (0 = along, 1 = cross), with r = τ − P, d = y − P and the plan H_w = P + w ⊙ r:

$$w_{s,a} = \arg\min_{w\in[0,1]^{30}} \sum_{n\in\mathcal W_{fit}} \Big[\sum_{t=1}^{30}(w_t r_{n,t,a} - d_{n,t,a})^2 + \lambda_s \sum_{k=1}^{29}\big(D_2[0;H_w]_{n,k,a} - D_2[0;y]_{n,k,a}\big)^2\Big]$$

* τ is the model's own held-output plan, P the checkpoint's P10 prior and y the GT, all on the 0.2 s knots, exactly HBLEND's arrays.
* $D_2[0; p]_k = p_{k+1} - 2p_k + p_{k-1}$ on the knots with the origin prepended (p_0 = the ego at NOW), so k = 1..29. These are the same 29 cells as the accel family.
* $\mathcal W_{fit}$ is HBLEND's training fit set (`raw/2026-10-25-hblend/`: 141 training clips, 9,582 windows). **No eval window enters any fit or any selection.**
* The fit is a bounded linear least squares (`scipy.optimize.lsq_linear`, `bvls`): 30 parameters per axis and seed, KKT-checked. There is **no isotonic step**.
* **At inference** on eval-141: HACC = P + w ⊙ (τ − P). Every other output is H's own. The weight is a fixed function of the step index; P is the deployed v8F prior (measured ego history, as in HBLEND), so inference stays vision-only with no situation-classifier output in the goal path.

**λ rule (train only, fixed now).**
* λ0 = (position term) / (D2 term), both evaluated at HBLEND's banked w on the fit set: the unit-ratio scale.
* The grid is λ = m · λ0, m ∈ {0, 1, 4, 16, 64, 256}.
* **λ\*** is the smallest grid λ at which, on the TRAIN fit windows:
  * HACC's accel MAE and curvature MAE are both ≤ 0.98 × P10's, AND
  * HACC's ADE2 and ADE6 are both < P10's.
* These train metrics are computed with the eval's own per-window family functions, the same `arm_pw` route as HBLEND / HSPEC.
* If no grid λ qualifies, λ\* is the grid λ with the lowest train accel MAE among those whose train ADE2 < P10's. If none of those exists, λ\* = 0.

## The read (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000, 141 episodes; F = 0.1207 m)

| read | definition | bar |
|---|---|---|
| **PRIMARY-DOM** | HACC − P10 on HSPEC's dominance row, with HSPEC's conventions | **separated-better on ADE2, ADE6, accel MAE AND curvature MAE, at BOTH seeds, and not separated-worse on any other row of it** |
| **PRIMARY-KEEP** | ADE6, HACC − H | upper CI bound ≤ +F/2 = +0.0604 m at both seeds |
| CEILING (T0 diagnostic, never a result) | the same objective at λ\*, fitted on the EVAL GT | beside |
| beside | the per-step accel profile of HACC / H / P10 / B (`accel_profile.py`'s instrument); w; λ\*; the four families, with the paired tactical / strategic tool `v8f_paired_families.py` for HACC vs H (T's decisions and S are H's own, identical by construction) | descriptive |

| outcome (read in this order) | condition | next (committed) |
|---|---|---|
| **DOMINATES P10** | PRIMARY-DOM and PRIMARY-KEEP at both seeds | HACC becomes the deployed post-map: 30 × 2 buffers per model, zero training, folded into `forward_infer` under its own landing with tests. The evidence videos are re-rendered on HACC. The T1-kin closed-loop read on HACC (`D-V8F-T1KIN-1`) is the next tier and needs a PI go for the pooled eval cache |
| **SMOOTHER, NOT DOMINANT** | not the above, and HACC − H accel MAE AND curvature MAE are both separated-negative at both seeds | OACC (box 2) runs as queued, and its HX read adds HACC's fit on OACC's checkpoint |
| **NO SMOOTHING** | HACC − H accel MAE not separated-negative at either seed | the roughness is not a weighting artefact; OACC is the lever |
| **MIXED** | anything else | reported per seed; seeds before windows |

## Controls (each must read its known value before any number is read)

1. **Machinery tie:** at λ = 0, adding HBLEND's isotonic step reproduces HBLEND's banked w at both seeds to 1e-12.
2. **λ = 0 without the isotonic step** equals the clipped closed form exactly (1e-12).
3. **Endpoints:**
   * w ≡ 1 reproduces B's banked eval per window (≤ 1e-4 m).
   * w ≡ 0 reproduces the P10 file (≤ 1e-4 m) and its floors' ADE6 2.4870 ± 1e-3.
4. **Optimality:** the KKT residual is ≤ 1e-6 at every fitted λ.
5. **Deliberate regressions:**
   * GT := P10 in the fit ⇒ w = 0 (≤ 1e-12), and the read is P10's (ADE6 difference ≤ 1e-4 m).
   * r permuted across training windows ⇒ mean w ≤ 0.10.
6. **The λ selection is train-only:** its code path receives training arrays alone, and a test that passes an eval array to it must raise.
7. **The accel instrument** reproduces the banked `fw_lon_accel_mae` in float32 to 2e-6 (MEASURED before this file: 9.5e-7 / 4.8e-7).

## Compute and order

Dev-box CPU; minutes (ESTIMATED). It touches nothing on box 2. ⛔ No grid, λ rule, bar or routing changes after any HACC number is seen.
