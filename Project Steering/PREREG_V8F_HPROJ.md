# PREREG_V8F_HPROJ: HBLEND followed by the deployable hard projection, so the 2 s / 6 s gain is kept without the consistency cost (written 2026-10-25, before any HPROJ data)

**Status: pre-registered.** Zero training. Eval-only, offline, from banked outputs, on the dev box's CPU, under `PI-V8F-2026-10-25b`.

## Why (MEASURED, `raw/2026-10-25-hblend/`, `H-V8F-HBLEND-1`)

HBLEND PASSED at both seeds:
* ADE2 H − B = −0.103 / −0.078 m;
* ADE6 H − B = −0.207 / −0.070 m;
* ADE6 H − P10 = −0.519 / −0.502 m;
* all separated.

Beside the bars, it COSTS consistency. CONSISTENCY(a) is the share of plans that violate a GT-derived constraint. Under H it worsens by +0.101 / +0.075 on 'any' and by +0.100 / +0.098 on SPEED, separated at both seeds. A plan that breaks more constraints is not a deployable improvement, whatever its ADE.

**The lever.** `stack/tanitad/models/v8f_mini.py::project_speed` is the programme's HARD inference projection. It applies speed ≤ the R1 cap from the max-speed INPUT and the Kamm circle |v²κ| ≤ μg, it preserves geometry and it re-times only. It reads model outputs and the max-speed input: vision-only and deployable. B's headline numbers are computed WITHOUT it (the eval reports it in a separate `projected_r1_kamm` block), and so are H's.

## The arm (ONE lever on top of H: the projection)

$$\tau_{HP} = \mathrm{project\_speed}\big(\tau_H^{(60)},\ \mathrm{cap}\big)[:, 1{::}2]$$

* **$\tau_H^{(60)}$ is H at the 0.1 s grid.**
  * B's $\tau^{(60)} = \mathrm{up}(P10) + \mathbf{B}c$ is recovered EXACTLY from the banked 30-step plan. The 12 residual coefficients $c$ are identifiable from 30 rows: least squares on the basis sampled at the 30 eval steps.
  * Then $\tau_H^{(60)} = \mathrm{up}(P10) + w^{(60)} \odot (\tau^{(60)} - \mathrm{up}(P10))$ in the ego frame at NOW.
  * $w^{(60)}$ is HBLEND's banked per-seed weights, linearly interpolated to the 0.1 s steps, with $w(0.1\,\mathrm{s}) = w(0.2\,\mathrm{s})$.
  * ⛔ No weight is re-fitted.
* `cap` is computed exactly as `v8f_mini_eval.py` computes it (`limit_ms_from_onehot(limit, limit_ok)`; 1e9 where there is no limit).
* **Reconstruction control:** $\mathrm{up}(P10) + \mathbf{B}c$ subsampled must reproduce B's banked 30-step plan (max|Δ| ≤ 1e-4 m), and `project_speed` applied to B's reconstructed $\tau^{(60)}$ must reproduce B's banked `projected_r1_kamm` ADE6 to 1e-3 m. If either fails, the run stops: rc 2, no read.

## The reading (T1-class open loop, eval-141, paired episode-cluster bootstrap, B = 2,000; F = 0.1207 m)

Every metric comes from the eval's OWN `evaluate()`, families included.

| read | definition | bar |
|---|---|---|
| **PRIMARY-C (no consistency cost)** | CONSISTENCY(a) 'any': HP − B | **upper 95 % bound ≤ +0.02, at BOTH seeds** |
| **PRIMARY-K (the gain is kept)** | ADE6: HP − B, and ADE2: HP − B | **both < 0 and separated, at BOTH seeds** |
| beside | CONSISTENCY(a) per constraint and (b); accel MAE; distance keeping; the four families; HP against P10 on ADE2, ADE6 and accel (the dominance row); HP against B's own projected plan | descriptive |

| outcome | condition | next (committed) |
|---|---|---|
| **PASS** | PRIMARY-C and PRIMARY-K at both seeds | HP is the deployed v8F output. Its weights become a model buffer, and the projection becomes part of `forward_infer` in the next trained arm. MIXO and FLAT are re-scored with their own fitted HP (descriptive) |
| **CONSISTENCY NOT RESTORED** | PRIMARY-C fails | H's violations are not of the posted limit: they are stops, leads or turns, which the cap does not see. Next: a constraint-aware fit under its own PREREG. It is the same closed form, fitted only on training windows with no active GT constraint, plus O's predicted stop / lead constraints in the projection |
| **GAIN LOST** | PRIMARY-C passes, PRIMARY-K fails | the projection and the shrinkage compete. Next: the constraint-aware fit as above |

⛔ No weight, cap or bar changes after any HPROJ number is seen.
