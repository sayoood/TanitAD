# RL Stage A — arm 1 of 3, banked incrementally. ⛔ NO LEVER CLAIM IS POSSIBLE YET.

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: T0** (deployed sampler, recorded
future, proxy reward) · **INCOMPLETE — 1 of 3 Stage A arms has run.**

⛔⛔ **Read this first.** The length-matched control **`L1-NORL-s0`** and the replicate
**`L1-RL-s1`** have **not run**. Every number below is an **absolute reading of one arm**. ⛔ Nothing
here supports any statement about whether the RL lever helped, hurt, or did nothing — that requires
the control, and on this rig a replicate *at the same seed* has separately been measured to clear
"separated" on noise alone. This is banked now only because the checkpoint is **1.3 GB on a single
disk** and the numbers should not live there alone.

## The run, and what the pre-registered instrument says

`train --arm rl --seed 0 --steps 600 --batch 4 --il-form matched --grad-clip 100`, 600/600 steps in
**1,883 s (31.4 min)** on the RTX 4060. `check_arm_l1.py` exits **0** with **`problems: []`** —
I1/I3/I7–I10 all hold.

| instrument | reading |
|---|---|
| rows · form · clip · release | 600 · `matched_anchor` · 100.0 · `is_release: false` |
| reward first → last | 0.5935 → **0.8884** |
| IL distance first → last | 0.6532 → **0.4476** m |
| chain spread first → last | 32.89 → **24.24** m |
| `grad_norm_max` · `steps_clipped` | **216.78** · **3 / 600** |
| `human_nc_eq_1_frac_pooled` | 1.0 |

⭐ **The clip behaved as Amendment A-1 predicted.** At 100 it bound **3 of 600** — a spike guard.
A-1 was made because the original 1.0 would have bound **600/600**, an every-step 17–62× rescale
rather than a divergence guard. This is the first arm run since that amendment, and it is consistent
with the reason for it.

## Held-out T0 read (n = 493 windows, stride 10)

| metric | mean |
|---|---|
| `sel_pdms` — what selection picks | **0.8712** |
| `human_pdms` | 0.9860 |
| ⭐ `fan_pdms_best` — the best candidate **present in the fan** | **0.9949** |
| `fan_pdms_mean` | 0.6992 |
| `sel_ade_m` | **2.5399** |
| ⭐ `fan_minade_m` | **0.8175** |
| `fan_endpoint_spread_m` | 28.53 |
| `fan_nc_fail_frac` | 0.1221 |
| `sel_nc` · `sel_ttc` · `sel_ep` · `sel_comfort` | 0.9260 · 0.8418 · 0.8842 · 0.9980 |
| `traj_matches_fan_sel` FALSE | **0 of 493** |

## ⭐ The one observation that does not need the control

**The fan contains a near-human plan and selection does not pick it.** `fan_pdms_best` **0.9949**
against `sel_pdms` **0.8712** — a gap of **0.1237** — and `sel_ade` **2.54 m** against `fan_minade`
**0.82 m**, a **3.11×** ratio. ⭐ This is a property of **one arm read against itself**, so it needs
no cross-arm comparison and no control: whatever the RL lever did or did not do, on this checkpoint
the generator is producing a much better trajectory than the selector is choosing.

⚠️ **It is still T0** — deployed sampler, **recorded future**, **proxy reward** — so it is a
diagnostic and **never a driving number**. And `fan_pdms_best` is a **max over the fan**, which is
optimistic by construction: an oracle that picks the best of N is not a selector.

⭐ It is also consistent, from a different direction, with what the D3 package measured on the same
model family the same night: the planner **attends** to the lead (T-G 1.92×, separated) yet masking
the lead moves nothing. Both readings point at **information that is present and not used**.

## Where the artifacts live

⚠️ **ONE PLACE:** `devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/l1-rl-s0/` — `ckpt.pt`
**1.3 GB**, `metrics.jsonl` (600 rows), `config.json`, plus `heldout_l1-rl-s0.json` (493 rows,
211 KB). Too large for the repo; the summary here and `raw/` carry every number that has been
quoted.

## Next

`L1-NORL-s0` (running) → `L1-RL-s1` → the Stage A verdict. **Stage B runs only if** cumulative GPU
time after Stage A is ≤ 2.45 h, otherwise it is recorded **NOT RUN**, never "unnecessary".
