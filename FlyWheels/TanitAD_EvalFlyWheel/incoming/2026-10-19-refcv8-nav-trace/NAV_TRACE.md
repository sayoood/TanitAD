# refcv8 nav trace — R8-2 (per-frame nav) at train vs at the battery roll (2026-10-19)

**VERDICT: EVAL-FEEDS-CLIP-LEVEL-ONLY.** R8-2 is violated at the battery eval, and there is a **train/eval mismatch**.
Training fed the per-frame v9 token on every window; the battery roll feeds the clip-level v7 token (constant per episode) and no nav args.
Launch tree `C:/lgs/Lfb47a58376` (fb47a58). Source facts MEASURED by reading the files; counts MEASURED on banked dumps.

## 1. TRAINING — what refcv8's forward consumes (per WINDOW)
- `nav_cmd` [B] long. `--nav-from-v7` first writes the clip token (`refc_v3_train.py:4591-4599`, from `_nav_by_sid`, built per clip at `:3278-3341`).
  When `r8_nav_from_v9` is on, `:4665-4668` **overwrites** it with the v9 `nav_token` at the window's NOW row (`r8_join.item(sid, t+w-1+raw_offset)`, `:4657-4658`; reader `refcv8_train.py:895-898`).
  The token is never dropped (`refcv8_train.py:564`, "the TOKEN is untouched"). `r8_rc`: not fed (`--r8-no-rc`, `:568`).
- `r8_nav` [B,6] = `scale_nav(nav_args[5], known)` (`refcv8_train.py:367-378`). It is per window from `window_inputs` (`v9_labels.py:189-207`).
  In training, `known` is dropped with p=0.5 (`refcv8_train.py:560-567`, `--r8-nav-args-dropout 0.5`).
  It is consumed in the tactical condition through `_gated` (`refc_v3.py:1968-1973`). A `None` input becomes zeros(6), the same as an unknown row.
- Train wiring: `ds.r8_nav_from_v9 = True` + `enable_r8_v9` joins every window or refuses (`refc_v3_train.py:9544-9550`, `:4317-4332`); batch read `:5262`.

## 2. BATTERY EVAL — what the roll feeds (`nav_true` = arm `os`)
- The eval dataset IS built per-frame: `refcv7_loader.py:817-818` sets `r8_nav_from_v9` and the v9 eval join, so `item["nav_cmd"]` is per-frame.
- **But the roll ignores the item.** `refcv3_arm.py:2064-2071` builds `nav_true[i] = ds._nav_by_sid[sid]`, which is the **clip-level v7 token** from `--eval-labels` s2_labels_v8.
  That token is fed at `:2262` and `:2358`, and `os` is row 0 (`:2406`). The roll log prints `[v3] nav_from_v7: follow 88 / left 13 / right 38 (of 139 clips)` (`step40000ema_unc/roll_s0.log`).
- **`r8_nav` is never passed.** The model call at `refcv3_arm.py:2358` takes `nav_cmd, v0, steps, ego_state, geom`.
  The proxy `r7_roll.py:152-163` (`_inputs`) adds only `ego_poses / v_max_ms / v_max_valid`. So at every battery window `r8_nav=None`, which becomes the "args unknown" row.
  v9 args are valid on **2,802/4,754 (58.9 %)** of the banked windows and are never fed.
- G0 / the in-run eval use `compute_losses_v3` on dataset batches (`g0_refcv7.py:907`, trainer `:5262`). That path is per-frame, so **G0 and the battery feed different nav inputs** in the same package.

## 3. Banked data — `decisions.nav_cmd` IS the driving token (`refcv3_arm.py:2502` banks `nav_true[i]`, the same value fed at `:2262`)
- Control (v9 label file, md5 `6b5c7f20…` = launch `--r8-v9-eval-md5`): **47/139** clips change token within the clip (27,664 rows: FOLLOW 22,822 / L 1,282 / R 3,560).
- Banked feed: constant in **139/139** episodes in **all 12 refcv8 trees × 2 seeds** (`raw/nav_constancy_all_trees.json`); on those windows v9 changes in **46/139**.
  Join key: v9 row `(sid, k = ws+2)`, with `row__k == k` asserted per window. `ws + 2` is independently stated in the dump manifest `grid.ws_is`.
- On step40000ema_unc s0 (4,754 windows), the fed token ≠ the v9 per-frame token on **1,157 (24.3 %)**.
  Of the 1,743 fed-L/R windows **1,055 (60.5 %)** are wrong (1,019 FOLLOW per-frame, 36 the **opposite turn**); of 826 v9-L/R windows 138 are wrong.
  Confusion matrix: `raw/nav_trace_step40000ema.json`. Column sums reproduce the video agent's 3,011/445/1,298 (same channel, a consistency check only).
  Example: episode 13 (sha12 `d75f38babcda`) is fed {R} while v9 reads {F, L, R}.

## 4. Effect — measured at zero compute (natural experiment on banked s0 rows)
Within one batched forward the rows differ only in the token (same frames, v0, ego, `r8_nav=None`); where the banked shuffled or flipped token happens to equal the v9 per-frame token, that row is the per-frame-correct roll.
Estimator: `paired_episode_cluster_bootstrap` (`taniteval/ci.py:275`), 2,000 draws. Arm − os, m.
| subset | n win/ep | ADE 2 s | ADE 6 s | |lat| err 6 s | |lon| err 6 s |
|---|---|---|---|---|---|
| FIX shuf=v9≠fed | 662/58 | −0.021 [−0.042, +0.001] | **−0.524 [−0.796, −0.237]** (2.103→1.579) | **−0.769 [−1.304, −0.348]** | **−2.127 [−3.462, −0.828]** |
| FIX flip=v9 (opposite turn) | 36/7 | **−0.285 [−0.727, −0.017]** | −1.853 [−4.163, +0.030] | **−7.525 [−16.58, −0.24]** | −0.797 [−2.16, +1.19] |
| CTRL shuf=fed (must read 0) | 2302/138 | −0.000 [−0.005, +0.005] | +0.003 [−0.024, +0.030] | +0.015 [−0.05, +0.08] | +0.020 [−0.08, +0.11] |
| BREAK fed=v9≠shuf | 1596/126 | +0.044 [+0.031, +0.058] | **+0.924 [+0.718, +1.149]** | +0.949 [+0.49, +1.47] | +3.977 [+3.02, +5.06] |
- Inference-seed floor on the FIX windows (`os` s1 − s0): ADE 6 s −0.017 [−0.062, +0.032]. The FIX effect is about 30× that.
  On the CTRL windows the seed floor is +0.029 [+0.006, +0.054], so it reads "separated". That is a reminder of the H-ESTIM-SEED rule.
- FIX split by (fed→v9), ADE 6 s: R→F −0.297, L→F −1.152 (both separated), F→R −0.732 (n 22, not separated): `raw/nav_natexp_step40000ema_split.json`.
- Scope: this measures the token half of the defect only, with args absent in both rows, on a non-random 698-window subset of the 1,157; a T1 diagnostic and not a battery re-score. TACTICAL and STRATEGIC families were not computed: that is a work item.

## 5. Numbers this touches
- Every `os`-derived refcv8 battery number at every milestone (12 trees): the four families, PAIRED_READ, acceptance, criteria, the E1/E3/compliant re-reads, refcv8-vs-refcv7 pairs, and the nav-dependence pairs (`os_minus_navshuf / navzero`, navflip).
  The effect is concentrated at 3–6 s and laterally. At the 2 s grid it is small, except on opposite-turn windows.
  ESTIMATED full-set ADE 6 s bias ≈ 1,157/4,754 × 0.52 ≈ 0.13 m pessimistic. Do not quote this as a correction.
- INFERRED (not verified): the brake_locus / force_lon / kdraw re-rolls and the videos reuse this path. Unaffected: ha, ha0, ha0_ext, stop, `os_navzero`, G0 / in-run eval.

## 6. Cheapest next experiment (not run; needs GPU authorisation)
- Patch only the roll: in `refcv3_arm.py:2068-2071`, set `nav_true[i] = int(ds[wi]["nav_cmd"])`, or do the equivalent in `r7_roll.Refcv6Proxy.__call__` from `ctx.item`.
- In `Refcv6Proxy._inputs`, add `r8_nav = scale_nav(item["r8_nav_raw"], item["r8_nav_known"])`.
- Roll `os`-only, two arms (token-only; token+args), 2 inference seeds, on the 1,157 mismatched windows + 300 matched windows (must read 0).
  ESTIMATED ≈ 1 GPU-h (from the 4.83 s 4-row first forward in `roll_s0.log`). The current clip-level roll is the deliberate-regression arm.
- Then re-score the full 4,754-window battery per-frame before quoting any refcv8 bar.
