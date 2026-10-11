# PATCH PROPOSAL - refc-base-30k figures filed as refcv7's fan ceiling

Package: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-refc-base-30k-mislabel/` - EvalFlyWheel, zero GPU, read-only on `8522e49`.
This agent edits no steering file and ran no `git add`; landing is the Master Mind's. Mechanical build: `code/build_patch.py` -> this file + `raw/verify_mislabel.json`.
Every figure below is re-read from the raw JSON by that script, none typed by hand; every guard in section 5 must read PASS.

## 0. Verdict: the escalation is CONFIRMED (one correction: its line numbers)

* `D-REFCV7-FANCEIL` (GOALS_AND_CLAIMS) and MODEL_REGISTRY section 15.3 label oracle-in-fan 0.1914 / selected 0.4728 / delta +0.2813 [+0.2127, +0.3543] / >2x on 41.09 % as **refcv7 at 50.4k**.
  All five are **`refc-base-30k`** (step 29,999, 128 anchors, 881 windows / 40 val episodes). Guards A1-A3.
* The escalation's line numbers (GOALS 12229, registry about 5300) are the **live worktree** (`D:/Projects/TanitAD`: GOALS 12229, registry header 5298). At the requested rev `8522e49` the same text sits at
  **GOALS 15762** and **registry 5330-5333** (the file at `8522e49` is longer). The text is byte-identical in both trees (guards B1-B4), so the patch anchors on the verbatim line, not the number.
* The same GOALS sentence also says `sampler_ranks_the_fan` = false. That clause **is true of refcv7** (`D:/refcv7_eval_kit/ckpt/config.json` `selection.sampler_ranks_the_fan` = False, MEASURED, guard A8),
  so the row is a hybrid: an arm-specific clause that is right, and every number and the "at 50.4k" scope that are wrong.
* **Defective lines. GOALS_AND_CLAIMS: 1** (the row, one physical line). **MODEL_REGISTRY: 2** (the two physical lines under 15.3) **+ 1 optional** (the 15.3 header's tier label "T1-family open loop"; the raw stamp says the oracle quantity is T0 diagnostic).
  **No other file in `Project Steering/` repeats the conflation** (section 3).

## 1. Evidence - which artifact settles each line (4 dp; delta bounds are the paired episode-cluster bootstrap; frac2x in %)

| quantity | the doc row says (as refcv7) | base-30k banked raw (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` + `driving_refc-base-30k.json`, git `8522e49`) | base-30k re-derived by X8 (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_repro_refc-base-30k.json`) | refcv7 seed 0 (`ctl_refcv7_50400_fan_s0.json`) | refcv7 seed 1 (`_s1`) |
|---|---|---|---|---|---|
| oracle | 0.1914 | 0.1914 | 0.1914 | 0.1143 | 0.1145 |
| selected | 0.4728 | 0.4728 | 0.4728 | 0.2772 | 0.2790 |
| gap | 0.2813 | 0.2813 | 0.2813 | 0.1630 | 0.1645 |
| gap_lo | 0.2127 | 0.2127 | 0.2127 | 0.1446 | 0.1454 |
| gap_hi | 0.3543 | 0.3543 | 0.3543 | 0.1825 | 0.1847 |
| frac2x | 41.09 | 41.09 | 41.09 | 44.17 | 44.95 |

Windows / episodes / candidates: base-30k 881 / 40 / 128; refcv7 4754 / 139 / 117.

* **Settling artifacts.** (1) Arm identity of the doc numbers: the banked taniteval JSON at `8522e49` (881 windows / 40 episodes, `arm` refc-base-30k); the registry's own REF-C-base block (lines 2515-2516, 2595-2597) prints the same figures under base.
  (2) Independent re-derivation: X8's `fan_control_x8.py oldpt` on `fan_refc-base-30k.pt` (pt sha256 f7edc65b...) gives them to four digits. (3) True refcv7 values: X8's panel from the refcv7 fan roll (ckpt md5 b418d0fc4a92a6848c246a6a7c50207b -
  I re-hashed `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` and it equals registry section 15's model-only md5), cross-checked against the independently banked `os` dump: fan-selected 0.2772 vs banked 0.2784, paired -0.0012 [-0.0036, +0.0012], not separated.
* **Why a figure-level check could not catch it.** Every number in the row is a correct number for base. Re-deriving 0.1914 from the base fan matches to four digits because it IS base's. The defect is in the arm label and the scope clause ("at 50.4k"), not in any digit.
* **Size of the error (MEASURED, `raw/verify_mislabel.json`).** The row overstated refcv7's selected ADE 1.71x, oracle 1.67x, gap 1.73x. The qualitative claim (the fan holds much better plans than the pick; >2x on about 40 % of windows) **survives** on refcv7 (44.17 % / 44.95 %).
* **Evidence-class note (INFERRED, not proven).** On the dev box no refcv7 fan artifact exists before `D:/refcv7_eval_kit/fanctl/` was created (2026-10-11 16:17; contents `smoke_dump`, `s0`, `s1`, `ctl` only), while the row was landed by commit `235156d` at 15:39:45
  (its message: "re-land of dfd97373", which is not readable in the git dir I was given - UNVERIFIED). So the row was stamped MEASURED before any refcv7 fan existed to measure. Thor and other disks were not searched: absence at one location only (C2).
* **Also stale in the same lines:** "Random-pick control not yet run" - it has run (X8, `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL). Folded into the patch as an additive note; separable from the mislabel (see hunk notes).
* **Immutable:** the message of commit `235156d` repeats the conflated sentence ("Fan ceiling: oracle 0.1914 m vs selected 0.4728 m, 41.09 % of windows"). The RETRACTION_LOG entry (section 4) is the correction of record for it.

## 2. The patch (Master Mind applies)

Two applyable variants carrying the same content.
**Option A** replaces the defective line(s) in place and keeps every old figure inside a "(corrected 2026-10-11: was ...)" note. It passes a numeric-token-superset guard (C1-C3, C8, C9) and **fails a strict line-superset guard by construction** (M4 proves that guard can go red).
**Option B** is insert-only: the old lines stay byte-identical (guard C6), at the price of leaving the mislabelled sentence standing with a correction beside it. Use B if the lander refuses any removed line.
Anchor on the verbatim OLD text (unique in both trees), not on a line number.

### 2.1 GOALS_AND_CLAIMS.md - hunk G1 (Option A) - line 15762 at `8522e49` (live worktree line 12229)

OLD (verbatim):
```
| **D-REFCV7-FANCEIL** | **refcv7's proposal fan contains much better plans than the one it picks.** MEASURED at 50.4k (PhysicalAI held-out, T1-family open loop): ADE of the oracle-in-fan candidate **0.1914 m** vs the selected **0.4728 m** — Δ **+0.2813 m [+0.2127, +0.3543]** (episode-cluster bootstrap); a candidate more than 2× better than the selected one exists on **41.09 %** of windows; `sampler_ranks_the_fan` = **false**. | ✅ **SUPPORTED as a measurement; the lever is UNEXPLOITED.** ⚠️ Not yet a selection claim: an oracle gap needs a **random-pick control** (best − random is the headroom a selector can claim; best − actual alone is large for any wide fan). |
```
NEW:
```
| **D-REFCV7-FANCEIL** | **refcv7's proposal fan contains much better plans than the one it picks.** **CORRECTED 2026-10-11 (EvalFlyWheel X8, raw-reproduced): the figures this row carried were `refc-base-30k`'s, not refcv7's.** *(corrected 2026-10-11: was "MEASURED at 50.4k (PhysicalAI held-out, T1-family open loop): ADE of the oracle-in-fan candidate **0.1914 m** vs the selected **0.4728 m** — Δ **+0.2813 m [+0.2127, +0.3543]** (episode-cluster bootstrap); a candidate more than 2× better than the selected one exists on **41.09 %** of windows; `sampler_ranks_the_fan` = **false**." - those figures are REF-C-base at step 29,999: 128 anchors, 881 windows / 40 val episodes (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` `gap["refc-base-30k"]` and `comparisons.oracle_in_fan_full.a`; ADE@2s 0.4728 = `taniteval/results/driving_refc-base-30k.json` `headline.ade_0_2s.mean`, MODEL_REGISTRY REF-C-base results).)* MEASURED for refcv7-r101-s0 at step 50,400 (ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b; PhysicalAI held-out eval-139: 4,754 windows / 139 episodes / 117 candidates; **T0 DIAGNOSTIC** - the oracle and every difference containing it use GT, `selected` alone is the T1 open-loop pick; FULL-SET mean, episode-cluster bootstrap, paired for the delta, n_boot 2000 seed 0, which answers *another draw of EPISODES?* only): oracle-in-fan **0.1143 m** [0.1057, 0.1257] vs selected **0.2772 m** [0.2527, 0.3051] - delta **+0.1630 m [+0.1446, +0.1825]** (paired); a candidate more than 2x better than the selected one exists on **44.17 %** of windows; inference-seed replicate (seed 1, same checkpoint): 0.1145 / 0.2790 / +0.1645 [+0.1454, +0.1847] / 44.95 %; one training seed. `sampler_ranks_the_fan` = **false** (this clause WAS refcv7's: `config.json` `selection.sampler_ranks_the_fan` False). Raw: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json` (+ `_s1`); narrative `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL. Not comparable with the base-30k panel (different windows, 117 vs 128 anchors): do not pool or difference them. | ✅ **SUPPORTED as a measurement; the lever is UNEXPLOITED.** ⚠️ Not yet a selection claim: an oracle gap needs a **random-pick control** (best − random is the headroom a selector can claim; best − actual alone is large for any wide fan). *(update 2026-10-11: random-pick control now RUN on refcv7: uniform-random pick 1.5501 m [1.5081, 1.5905], selected better by +1.2728 [+1.2201, +1.3232], skill fraction 0.8865 (point estimate); but the plain argmax of refined_logits is 0.2732 (-0.0040 [-0.0069, -0.0016] vs selected, separated) and a random pick among the model's own top-2 by refined_logits is 0.2994, so re-ranking by the model's own logits recovers only 0.0040 m of the 0.1630 m gap - the rest is a GT-peeking ceiling; see `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL.)* |
```
The trailing `(update 2026-10-11: ...)` sentence in the status cell (random-pick control) is separable from the mislabel; drop that one sentence if only the attribution fix is wanted - the first cell stands alone.

### 2.2 GOALS_AND_CLAIMS.md - hunk G1-B (Option B): insert this one line directly after line 15762; the old row is unchanged
```
| **D-REFCV7-FANCEIL-CORRECTION** | **CORRECTION 2026-10-11 to the row immediately above: its five figures are `refc-base-30k`'s, not refcv7's** - those figures are REF-C-base at step 29,999: 128 anchors, 881 windows / 40 val episodes (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` `gap["refc-base-30k"]` and `comparisons.oracle_in_fan_full.a`; ADE@2s 0.4728 = `taniteval/results/driving_refc-base-30k.json` `headline.ade_0_2s.mean`, MODEL_REGISTRY REF-C-base results). MEASURED for refcv7-r101-s0 at step 50,400 (ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b; PhysicalAI held-out eval-139: 4,754 windows / 139 episodes / 117 candidates; **T0 DIAGNOSTIC** - the oracle and every difference containing it use GT, `selected` alone is the T1 open-loop pick; FULL-SET mean, episode-cluster bootstrap, paired for the delta, n_boot 2000 seed 0, which answers *another draw of EPISODES?* only): oracle-in-fan **0.1143 m** [0.1057, 0.1257] vs selected **0.2772 m** [0.2527, 0.3051] - delta **+0.1630 m [+0.1446, +0.1825]** (paired); a candidate more than 2x better than the selected one exists on **44.17 %** of windows; inference-seed replicate (seed 1, same checkpoint): 0.1145 / 0.2790 / +0.1645 [+0.1454, +0.1847] / 44.95 %; one training seed. Raw: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json` (+ `_s1`); narrative `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL. Not comparable with the base-30k panel (different windows, 117 vs 128 anchors): do not pool or difference them. | The row above keeps its original text for audit; its `sampler_ranks_the_fan` = false clause is refcv7's and stands. *(random-pick control now RUN on refcv7: uniform-random pick 1.5501 m [1.5081, 1.5905], selected better by +1.2728 [+1.2201, +1.3232], skill fraction 0.8865 (point estimate); but the plain argmax of refined_logits is 0.2732 (-0.0040 [-0.0069, -0.0016] vs selected, separated) and a random pick among the model's own top-2 by refined_logits is 0.2994, so re-ranking by the model's own logits recovers only 0.0040 m of the 0.1630 m gap - the rest is a GT-peeking ceiling.)* |
```

### 2.3 MODEL_REGISTRY.md - hunk R1 (Option A) - replace lines 5332-5333 at `8522e49` (live worktree 5300-5301)

OLD (verbatim, two physical lines):
```
Oracle-in-fan ADE **0.1914 m** vs selected **0.4728 m** (Δ +0.2813 [+0.2127, +0.3543]); a >2× better candidate on
**41.09 %** of windows; `sampler_ranks_the_fan` false. Random-pick control **not yet run** — `D-REFCV7-FANCEIL`.
```
NEW (three physical lines):
```
**CORRECTED 2026-10-11 - the figures this subsection carried belong to `refc-base-30k`, not to refcv7.** *(corrected 2026-10-11: was "Oracle-in-fan ADE **0.1914 m** vs selected **0.4728 m** (Δ +0.2813 [+0.2127, +0.3543]); a >2× better candidate on **41.09 %** of windows; `sampler_ranks_the_fan` false. Random-pick control **not yet run** — `D-REFCV7-FANCEIL`." - those figures are REF-C-base at step 29,999: 128 anchors, 881 windows / 40 val episodes (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` `gap["refc-base-30k"]` and `comparisons.oracle_in_fan_full.a`; ADE@2s 0.4728 = `taniteval/results/driving_refc-base-30k.json` `headline.ade_0_2s.mean`, MODEL_REGISTRY REF-C-base results).)*
**refcv7-r101-s0 @ 50,400** (ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b; 4,754 windows / 139 episodes / 117 candidates; T0 DIAGNOSTIC, FULL-SET mean, episode-cluster bootstrap n_boot 2000 - *another draw of EPISODES?* only): oracle-in-fan **0.1143 m** [0.1057, 0.1257] vs selected **0.2772 m** [0.2527, 0.3051] - delta **+0.1630 m [+0.1446, +0.1825]** (paired); a candidate more than 2x better than the selected one exists on **44.17 %** of windows; inference-seed replicate (seed 1, same checkpoint): 0.1145 / 0.2790 / +0.1645 [+0.1454, +0.1847] / 44.95 %; one training seed; `sampler_ranks_the_fan` false (`ckpt/config.json` `selection`).
Random-pick control now RUN on refcv7 (was: not yet run): uniform-random pick 1.5501 m [1.5081, 1.5905], selected better by +1.2728 [+1.2201, +1.3232], skill fraction 0.8865 (point estimate); but the plain argmax of refined_logits is 0.2732 (-0.0040 [-0.0069, -0.0016] vs selected, separated) and a random pick among the model's own top-2 by refined_logits is 0.2994, so re-ranking by the model's own logits recovers only 0.0040 m of the 0.1630 m gap - the rest is a GT-peeking ceiling. Raw: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json` (+ `_s1`); narrative `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL. Not comparable with the base-30k panel (different windows, 117 vs 128 anchors): do not pool or difference them. `D-REFCV7-FANCEIL`.
```

### 2.4 MODEL_REGISTRY.md - hunk R0 (OPTIONAL tier label) - line 5330 at `8522e49`
OLD: `### 15.3 Fan ceiling (MEASURED, T1-family open loop, PhysicalAI held-out)`
NEW: `### 15.3 Fan ceiling (MEASURED, T0 diagnostic - oracle uses GT, `selected` alone is the T1 open-loop pick; PhysicalAI held-out) *(corrected 2026-10-11: was "MEASURED, T1-family open loop, PhysicalAI held-out")*`
Raw stamp (all three panel JSONs): "T0 diagnostic (oracle and every difference containing it use GT); selected alone is the T1 open-loop pick".

### 2.5 MODEL_REGISTRY.md - hunk R1-B (Option B): insert block 1 directly above old line 5332, blocks 2 and 3 directly after old line 5333
```
**CORRECTION 2026-10-11 (inserted above the two lines that follow; they are retained verbatim for audit): the figures in those two lines belong to `refc-base-30k`, not to refcv7** - those figures are REF-C-base at step 29,999: 128 anchors, 881 windows / 40 val episodes (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` `gap["refc-base-30k"]` and `comparisons.oracle_in_fan_full.a`; ADE@2s 0.4728 = `taniteval/results/driving_refc-base-30k.json` `headline.ade_0_2s.mean`, MODEL_REGISTRY REF-C-base results).
```
```
**refcv7-r101-s0 @ 50,400** (ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b; 4,754 windows / 139 episodes / 117 candidates; T0 DIAGNOSTIC, FULL-SET mean, episode-cluster bootstrap n_boot 2000 - *another draw of EPISODES?* only): oracle-in-fan **0.1143 m** [0.1057, 0.1257] vs selected **0.2772 m** [0.2527, 0.3051] - delta **+0.1630 m [+0.1446, +0.1825]** (paired); a candidate more than 2x better than the selected one exists on **44.17 %** of windows; inference-seed replicate (seed 1, same checkpoint): 0.1145 / 0.2790 / +0.1645 [+0.1454, +0.1847] / 44.95 %; one training seed; `sampler_ranks_the_fan` false (`ckpt/config.json` `selection`).
```
```
Random-pick control now RUN on refcv7 (was: not yet run): uniform-random pick 1.5501 m [1.5081, 1.5905], selected better by +1.2728 [+1.2201, +1.3232], skill fraction 0.8865 (point estimate); but the plain argmax of refined_logits is 0.2732 (-0.0040 [-0.0069, -0.0016] vs selected, separated) and a random pick among the model's own top-2 by refined_logits is 0.2994, so re-ranking by the model's own logits recovers only 0.0040 m of the 0.1630 m gap - the rest is a GT-peeking ceiling. Raw: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8/ctl_refcv7_50400_fan_s0.json` (+ `_s1`); narrative `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md` section D-REFCV7-FANCEIL. Not comparable with the base-30k panel (different windows, 117 vs 128 anchors): do not pool or difference them. `D-REFCV7-FANCEIL`.
```

## 3. Other files - grep of `Project Steering/` (at `8522e49` and in the live tree)

Patterns: `FANCEIL`, `0.1914`, `0.2813`, `41.09`, `0.2127`, `0.3543`, `0.4728`, `0.4109`, `oracle-in-fan`, `fan ceiling`, prose variants; both `41.09 %` and `41.09%` forms. **No further file attributes these figures to refcv7.**
* CORRECT (arm named base-30k / REF-C-base) - leave alone: `GOALS_AND_CLAIMS.md` 9131 (D-REFCV5-RANKS-THE-UNSAMPLED-FAN, "41.09 % (base) / 45.4 % (REF-C-XL)") and 10080-10083 (D-P14-RANK-1, "base-30k ADE 0.4728 -> 0.1914 m"),
  `MODEL_REGISTRY.md` 726-730, 2515-2517, 2595-2597, 3747, `REFCV5_MISSING_PIECES_PLAN.md` 68/237/460-461 (P14), `PREREG_D-SEL_REFC_SELECTION_SURFACE.md` 31/48/319/340/554 (line 319 even states "The arm travels with the number").
* ARM-UNNAMED, not a conflation (optional hygiene): `PI_VIDEO_REVIEW_2026-09-06.md` line 30 quotes 41.09 % and "0.4728 -> 0.1914 m" under P14 without naming REF-C-base; it predates any refcv7 fan by weeks, so it cannot be a refcv7 claim.
* Clean of the figures: `PLAN_REFCV8.md`, `SPEC_REFCV8_DRAFT_SOURCES.md` (its "+0.2127" is an unrelated D0 dose-response CI), `PI_DECISION_QUEUE.md`, `LOOP_STATE.md`, `PROGRAM_OVERVIEW.md`, `Benchmarks & Eval/LEADERBOARD*.md`.
* `Mission Plan.md`: not touched, no edit proposed.
* A hit for the bare digits in a raw CSV/JSON elsewhere in the repo is not a steering claim and was not counted.

## 4. RETRACTION_LOG entry (draft; append-only; class number assigned at landing - the highest class referenced at `8522e49` is C166)

| date | retracted claim | class | what it cost / would have cost |
|---|---|---|---|
| 10-11 | *"refcv7's proposal fan: oracle-in-fan 0.1914 m vs selected 0.4728 m, delta +0.2813 [+0.2127, +0.3543], a >2x-better candidate on 41.09 % of windows, MEASURED at 50.4k"* (GOALS_AND_CLAIMS `D-REFCV7-FANCEIL`, MODEL_REGISTRY section 15.3, and the message of commit `235156d`) | **C<NEXT> - A number's ARM travelled by adjacency, not by artifact: a MEASURED row for arm B was filled with arm A's banked figures, correct as numbers and wrong as a claim.** Recognition signal: a MEASURED row that names no raw artifact path and no `n_windows` / anchor count, whose figures equal another arm's registered figures at 4 dp. Nearest existing classes: C4 (inherited without re-verification) and C6 (the contrast varied more than the label admits - here arm, step 29,999 vs 50,400, window set 881/40 vs 4,754/139 and anchor count 128 vs 117 all changed under one name). | All five figures are `refc-base-30k`'s (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json`); refcv7's own (X8 raw `ctl_refcv7_50400_fan_s0.json`, 4,754 windows / 139 episodes / 117 candidates) are 0.1143 / 0.2772 / +0.1630 [+0.1446, +0.1825] / 44.17 %, so the row overstated refcv7's selected ADE 1.71x and its gap 1.73x while the qualitative finding survived. **Why every figure-level check passed:** the digits were right, so only the arm label, the scope clause ("at 50.4k") and the tier label ("T1-family open loop"; the raw stamp says T0 diagnostic) were wrong, and a hybrid sentence (`sampler_ranks_the_fan` false IS refcv7's) made the row read as verified. **Evidence-class error:** stamped MEASURED although I found no refcv7 fan artifact on the dev box until the X8 roll of 2026-10-11 16:19-16:52, after the 15:39 landing (Thor not searched). Caught by the EvalFlyWheel X8 agent while producing the missing random-pick control, before the figure sized any experiment that I can find. **Durable fix proposed, not built:** a claims/registry lint that, for every row carrying figures, requires an artifact path and flags any quoted figure equal at 4 dp to a different arm's registered figure; its positive control is exactly this row. |

## 5. Mechanical guards (`raw/verify_mislabel.json`)

| guard | result |
|---|---|
| `A1_doc_figures_equal_base30k_banked_raw_git` | PASS |
| `A2_X8_reproduction_of_base30k_equals_banked` | PASS |
| `A3_no_doc_figure_equals_refcv7_seed0_or_seed1` | PASS |
| `A4_base30k_881_windows_40_eps_128_anchors` | PASS |
| `A5_refcv7_4754_windows_139_eps_117_cands_both_seeds` | PASS |
| `A6_base30k_ckpt_step_29999` | PASS |
| `A7_refcv7_ckpt_md5_declared_b418d0fc` | PASS |
| `A8_refcv7_config_sampler_ranks_the_fan_is_False` | PASS |
| `B1_old_goals_row_verbatim_exactly_once_at_8522e49` | PASS |
| `B2_old_goals_row_verbatim_exactly_once_in_live_worktree` | PASS |
| `B3_old_registry_block_verbatim_exactly_once_at_8522e49` | PASS |
| `B4_old_registry_block_verbatim_exactly_once_in_live_worktree` | PASS |
| `C1_numeric_token_superset_GOALS_row` | PASS |
| `C2_numeric_token_superset_REGISTRY_15_3_block` | PASS |
| `C3_numeric_token_superset_REGISTRY_15_3_header` | PASS |
| `C4_optionA_removes_exactly_1_GOALS_line_adds_1` | PASS |
| `C5_optionA_removes_exactly_2_REGISTRY_lines_adds_3` | PASS |
| `C6_optionB_removes_zero_lines_in_both_files` | PASS |
| `C7_goals_row_keeps_3_cell_shape` | PASS |
| `C8_file_level_numeric_tokens_lost_GOALS_is_zero` | PASS |
| `C9_file_level_numeric_tokens_lost_REGISTRY_is_zero` | PASS |
| `C10_new_text_is_ascii_outside_carried_old_text` | PASS |
| `M1_superset_guard_goes_red_when_an_old_figure_is_dropped` | PASS |
| `M2_arm_guard_goes_red_when_a_doc_figure_equals_refcv7` | PASS |
| `M3_oldtext_uniqueness_guard_goes_red_on_a_mutated_line` | PASS |
| `M4_line_superset_guard_goes_red_for_option_A_in_GOALS` | PASS |

A* tie the doc figures to the base-30k raw and rule out refcv7's; B* prove the OLD text is verbatim and unique in both trees; C* simulate the apply in memory (nothing written to the repo); M* are deliberate-regression controls - each guard must go RED on a knowingly broken input, and does.
Interval scope: the CI bounds quoted answer *another draw of EPISODES?* only (FULL-SET mean, episode-cluster bootstrap n_boot 2000, paired for deltas). The refcv7 panel is one training seed with an inference-seed replicate; the base-30k panel is one seed.

## 6. Unverified / not done
* `dfd97373` (the row's original commit) is not in the git dir I may read - the row's authoring time and source are UNVERIFIED beyond the landing commit `235156d`.
* Metric-horizon equality of the base-30k and refcv7 panels is UNVERIFIED (both use `_de_all` over 4 waypoints; refcv7's waypoint spacing was not checked), hence "do not pool or difference".
* Whether Thor holds an earlier refcv7 fan artifact was not checked (dev box only).
