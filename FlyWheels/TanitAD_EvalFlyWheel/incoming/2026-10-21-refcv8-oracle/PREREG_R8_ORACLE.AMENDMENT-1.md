# PREREG R8-ORACLE, AMENDMENT 1: two BESIDE arms that separate class IDENTITY from one-hot SHARPENING

**Written 2026-10-21, clock read `2026-10-21T20:18:40+0200`, AFTER the 2-episode CPU smoke and BEFORE any GPU roll or any eval-scale number.** Sealed by `PREREG_R8_ORACLE.AMENDMENT-1.sha256`. The parent prereg (`PREREG_R8_ORACLE.md`, sha256 cd7ba96d8b1f16f1b8aaa5388ba7ad2257280351218eb750dc575b454176d118) is UNCHANGED: its arms, metrics, estimator, controls and the PRIMARY reading rule (section 5, S6 ADE, both seeds) stand exactly as sealed. This amendment ADDS two beside arms and one beside decomposition; it moves no threshold and decides nothing the parent decides.

## What the smoke showed (6 windows, 2 episodes, CPU fp32 trunk, seed 1: NOT a result, stated so the amendment's motive is on the record)

* Every control passed (C1-smoke vs the unwrapped child on the same CPU: max |diff| 2.9e-6 m, run-to-run float noise already present on the first window; C2, C3, C4, C6, C7 exact).
* GT_CONS and GT_SPEED_ONLY left the plan bit-identical to OWN on all 6 windows (the selected candidate was a base-fan candidate and did not flip; `v_end` is not in the selector's satisfaction terms, only in the allocated candidates' phi).
* GT_HYP changed the lon class on 5/6 windows and the selected candidate on 5/6; OWN_ARGMAX (own class, one-hot) ALSO changed the selected candidate on 5/6.
* Mechanism read from source and the MEASURED trained scalars (`CKPT_SELECTOR_WEIGHTS.json`): `beta_lat = -0.0430`, `beta_lon = -0.0210` are NEGATIVE, and the seam's class force is ONE-HOT (+-20 logits), which puts every non-forced hypothesis at `LOGP_FLOOR` (-9.21): candidates tagged with a NON-forced hypothesis then gain up to `+0.40 + 0.19` on the factorised score, and the allocation's floors move. So the native GT_HYP mixes two things: the class IDENTITY (what the oracle is meant to supply) and a confidence profile the trained selector never saw.

## The two beside arms

| arm | intervention | why |
|---|---|---|
| **GT_HYP_SWAP** | in the tactical decoder's 8-way `lat_logits` / `lon_logits`, SWAP the column of the model's own argmax hypothesis with the column of the GT hypothesis (`LAT3_V7_IDS` / `LON6_V7_IDS` rows), per axis, before `_r8_apply_force` runs; no native force | the GT class with the model's OWN confidence profile: identity changes, sharpness does not. Where own argmax == GT it is the identity (bit-exact with OWN). |
| **GT_BOTH_SWAP** | GT_HYP_SWAP + `set_r8_force(cons = GT, min_alloc = 2)` | the full tactical oracle without one-hot sharpening (the argmax is then GT, so the GT constraints land on the GT rows) |

Mechanism (wrapper only, no shared code edited): the read-only spy that already wraps the model instance's `_r8_apply_force` permutes the two logit columns in a COPY of the decoder output dict for that one forced call, then calls the original bound method unchanged. The swap is applied to every row of the window's batch.

## Beside decomposition (reported, NOT deciding)

`H_sw = OWN - GT_BOTH_SWAP`, `d_sw = OWN - GT_HYP_SWAP`, `c = OWN - GT_CONS` (the parent's arm), `I_sw = H_sw - c - d_sw`; shares on the same draws; the parent's reading function applied to these inputs and printed as **"BESIDE (amendment 1)"**. The headline verdict remains the parent's section-5 reading on the native arms. If the two disagree, both are printed and the disagreement is named as the seam's one-hot sharpening, quantified by OWN_ARGMAX.

## Added control

* **C3b**: on windows where the own argmax equals the GT tag on BOTH axes, GT_HYP_SWAP == OWN bit-exactly (trajectory and `sel_idx`); reported; a failure voids the two swap arms only.

## Tier

T0, unchanged.
