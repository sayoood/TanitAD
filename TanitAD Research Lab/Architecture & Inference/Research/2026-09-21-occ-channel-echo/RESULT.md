# `occluded` is worse than free — a 2-parameter read of the head's own output beats the channel

**Evidence class:** MEASURED (ours). **Tier:** not a driving tier — this is a perception-head
readout comparison on banked eval windows. **Compute:** CPU only, no GPU, A8 `ckpt_5000`.
**Date:** 2026-09-21.

## Why this exists

`c920f15` closed `l`/`w` as genuine defects and left **one** field of the quiet-field spectrum
unexamined: `occluded`, alignment **0.284**, with the note *"no target channel identified"*. This
answers it, and the answer moves twice — once away from the spectrum's prediction, and once away
from my own first verdict.

## 1. The field IS supervised here — the sentinel failure mode is refuted

The loss masks this field: `agent_slots.py:602` computes `om = ot >= 0.0`, and
`targets_from_join` initialises `occ` to **-1.0** (`:730`), overwriting it only from column 5 of the
join (`:739`). The trainer's own default is the same sentinel (`refc_v3_train.py:3922`, `:4345`).
So a corpus whose join carries no `occ` column would train this field on **zero** examples while the
head kept emitting it. That was the first hypothesis and it is **REFUTED**:

| | |
|---|---|
| valid targets | 1,482 |
| carrying an `occ` flag | **1,482 (100.0000 %)** |
| positives | 872 → base rate **0.5884** |

## 2. The target is an exact geometric identity — re-derived, not quoted

`agent_slots.py:82-87` states `occ` IS `bev_raster.fov_mask`'s predicate. Re-derived from the data
rather than inherited: sweeping the half-angle over 20–90° in 0.25° steps and choosing it on the
**fit half only** gives **60.0°**, with agreement

* **1.0000 on the fit half**, and
* **1.0000 on the scored half** (episode-disjoint).

⇒ `occ == |atan2(cy, cx)| > 60.0°` on the box centre, exactly. `occ` means **out of the front
camera's field while the track continues** — never object-object occlusion, and nothing here may be
reported as occlusion reasoning.

## 3. The head beats its base rate — and that is not the finding

logloss **0.288** vs **0.675** for the base-rate constant (fit-side rate 0.550), gain 0.386,
CI [0.352, 0.431]. Read alone this says the head's strongest demonstrated field is the one with the
**second lowest** alignment. The emission is not degenerate either: p spans 4e-05 → 0.99988,
sd 0.332 — so this is not the presence pattern (`286e0d3`).

## 4. ⛔ The finding: a two-parameter read of its own output beats it

848 scored pairs / 18 scored episodes, episode-disjoint, every hyper-parameter fit-side only.

| arm | logloss |
|---|---|
| base-rate control | 0.67462 |
| H1 HARD predicate on the head's own box | 0.44896 |
| **O — the head's `occ_logit`** | **0.28818** |
| H2 logistic on the head's own azimuth (d = 2) | **0.16055** |
| H3 logistic on the head's whole own box (d = 5) | 0.16211 |

`O − H2` = **−0.1276**, CI **[−0.1844, −0.0376]**, separated. `O − H3` = −0.1261,
CI [−0.1811, −0.0432].

**H3 adds nothing over H2** (0.162 vs 0.161) — exactly what §2's identity predicts, since azimuth
is the whole story and the rest of the box is noise for this target.

### The check that makes it admissible

H2 is fit on eval-side FIT episodes while the head was trained on the training corpus, so H2 has an
in-domain fitting advantage that has to be bounded. Its coefficients (azimuth **6.6902**, bias
**−7.1085**) imply a threshold at 7.1085 / 6.6902 = 1.0625 rad = **60.88°**, against the identity's
true **60.0°** — **0.88° off**. A 2-parameter fit on 634 pairs recovered the **physics**, not the
corpus. The advantage exists but it is not what produced the win.

## 5. ⛔ It survives the matcher-selection confound, which runs entirely against it

`match_slots`'s Hungarian cost is centre, cls and presence (`agent_slots.py:475-477`), so the scored
pairs are by construction ones where the predicted centre is close — **home ground for any arm that
reads the centre**. The confounder is observed (`|az_pred − az_gt|`), so stratify on it: cuts at the
FIT half's 1/3 and 2/3 quantiles, **one** logistic fit on the whole fit half so the confounder is
never handed to H2 as a feature.

| stratum | mean az err | n | eps | H2 | O | CI95 (H2 − O) | reading |
|---|---|---|---|---|---|---|---|
| LOW | 1.79° | 281 | 18 | **0.0307** | 0.2014 | [−0.1942, −0.1401] | ⛔ channel loses |
| MID | 6.65° | 301 | 17 | **0.0677** | 0.2667 | [−0.2300, −0.1584] | ⛔ channel loses |
| HIGH | 24.01° | 266 | 16 | 0.4028 | 0.4042 | [−0.1444, 0.2060] | — indistinguishable |

**The channel never wins.** Its best case is a TIE where the box is badly wrong, and it is worst
**precisely where the box is accurate** — 6.5× worse in the low-error stratum, where getting the
flag right is a threshold comparison the head has already performed.

## 6. ⭐⭐ The method lesson — my own first verdict was the opposite

`code/occ_echo.py` scored the channel against a **hard 0/1** predicate on the head's box and printed:

> ⭐ REAL — the occ channel beats a free geometric read of the head's OWN box, so it carries
> information the box does not express

A hard predicate eats `−log(clip) = 6.9` on every error while a logit **hedges**. That control was
**structurally handicapped relative to the thing it was controlling for** — it could not express
what the measured arm expresses. Give the box arm the same right to hedge and the sign flips.

⇒ **A control must be allowed to express what the measured thing expresses.** Same family as the
four estimator bugs of 2026-08-22, with the handicap living in the control's **output form** rather
than in its tuning. It never landed, so nothing is retracted — and the reversed arm is banked
beside the corrected one on purpose, because the failure is the reusable part.

## 7. Scope, stated

* **Function class.** H2/H3 are LINEAR (logistic). Beating them is evidence against a linear soft
  read of the head's own box, not against every possible read.
* **One checkpoint, one arm — and that is sufficient here.** This is **not** a lever claim: both
  arms are readouts of the **same forward pass**, so `H-ESTIM-SEED-1`'s training-variance floor does
  not bind and no replicate arm is owed. Say which question a CI answers; this one answers *would
  another draw of episodes say this?*, which is the right question for a fixed-model readout.
* **Semantics.** `occ` is the FOV predicate. The P4 question ("does the latent carry agents the
  camera cannot see") is **not** settled by a channel this weak.
* **n.** 60 windows → 1,482 supervised pairs over 35 episodes; 848 pairs / 18 episodes scored.

## 8. Consequence for `D-S1-DEP-BOX`

The head-change SPEC now carries three MEASURED items, and they are **not the same kind of work**:

| item | evidence | needs |
|---|---|---|
| size `l`/`w` | 75–78 % predictable, head does not use it (`c920f15`) | training ⇒ **PI's GPU call** |
| `presence` | no per-slot signal (`286e0d3`) | training ⇒ **PI's GPU call** |
| **`occluded`** | **worse than a 2-parameter read of its own output (this)** | **a DECODE change — zero training** |

`occluded` can be computed from the predicted centre's azimuth at zero training cost and be
strictly better than what the channel emits today. That takes one third of the blocker off the GPU
queue entirely. ⚠️ Not applied here: changing a live decode contract is a code change to
`agent_slots.py` that belongs in the SPEC and its own mutation-audited landing, not in a probe.

## Manifest

| file | what |
|---|---|
| `code/occ_supervision.py` → `raw/occ_supervision.json` | coverage + base rate + the head's emission spread |
| `code/occ_echo.py` → `raw/occ_echo.json` | the identity sweep, and **the reversed verdict** (§6) |
| `code/occ_calib.py` → `raw/occ_calib.json` | the H1/H2/H3 ladder; builds `occ_rows.npz` |
| `code/occ_strata.py` → `raw/occ_strata.json` | the stratified read against the matcher confound |
