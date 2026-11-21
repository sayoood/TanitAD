# PATCH PROPOSAL - refc-base-30k figures filed as refcv7's fan ceiling

Package: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-11-refc-base-30k-mislabel/` - EvalFlyWheel, zero GPU, read-only on `@@REV@@`.
This agent edits no steering file and ran no `git add`; landing is the Master Mind's. Mechanical build: `code/build_patch.py` -> this file + `raw/verify_mislabel.json`.
Every figure below is re-read from the raw JSON by that script, none typed by hand; every guard in section 5 must read PASS.

## 0. Verdict: the escalation is CONFIRMED (one correction: its line numbers)

* `D-REFCV7-FANCEIL` (GOALS_AND_CLAIMS) and MODEL_REGISTRY section 15.3 label oracle-in-fan 0.1914 / selected 0.4728 / delta +0.2813 [+0.2127, +0.3543] / >2x on 41.09 % as **refcv7 at 50.4k**.
  All five are **`refc-base-30k`** (step 29,999, 128 anchors, 881 windows / 40 val episodes). Guards A1-A3.
* The escalation's line numbers (GOALS 12229, registry about 5300) are the **live worktree** (`D:/Projects/TanitAD`: GOALS @@LG@@, registry header @@LR@@). At the requested rev `@@REV@@` the same text sits at
  **GOALS @@G1@@** and **registry @@R0@@-@@R4@@** (the file at `@@REV@@` is longer). The text is byte-identical in both trees (guards B1-B4), so the patch anchors on the verbatim line, not the number.
* The same GOALS sentence also says `sampler_ranks_the_fan` = false. That clause **is true of refcv7** (`D:/refcv7_eval_kit/ckpt/config.json` `selection.sampler_ranks_the_fan` = @@SRAF@@, MEASURED, guard A8),
  so the row is a hybrid: an arm-specific clause that is right, and every number and the "at 50.4k" scope that are wrong.
* **Defective lines. GOALS_AND_CLAIMS: 1** (the row, one physical line). **MODEL_REGISTRY: 2** (the two physical lines under 15.3) **+ 1 optional** (the 15.3 header's tier label "T1-family open loop"; the raw stamp says the oracle quantity is T0 diagnostic).
  **No other file in `Project Steering/` repeats the conflation** (section 3).

## 1. Evidence - which artifact settles each line (4 dp; delta bounds are the paired episode-cluster bootstrap; frac2x in %)

| quantity | the doc row says (as refcv7) | base-30k banked raw (`taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` + `driving_refc-base-30k.json`, git `@@REV@@`) | base-30k re-derived by X8 (`@@RELRAW@@/ctl_repro_refc-base-30k.json`) | refcv7 seed 0 (`ctl_refcv7_50400_fan_s0.json`) | refcv7 seed 1 (`_s1`) |
|---|---|---|---|---|---|
@@TBL@@

Windows / episodes / candidates: base-30k @@BW@@; refcv7 @@CW@@.

* **Settling artifacts.** (1) Arm identity of the doc numbers: the banked taniteval JSON at `@@REV@@` (881 windows / 40 episodes, `arm` refc-base-30k); the registry's own REF-C-base block (lines 2515-2516, 2595-2597) prints the same figures under base.
  (2) Independent re-derivation: X8's `fan_control_x8.py oldpt` on `fan_refc-base-30k.pt` (pt sha256 f7edc65b...) gives them to four digits. (3) True refcv7 values: X8's panel from the refcv7 fan roll (ckpt md5 b418d0fc4a92a6848c246a6a7c50207b -
  I re-hashed `D:/refcv7_eval_kit/ckpt/ckpt_50400.pt` and it equals registry section 15's model-only md5), cross-checked against the independently banked `os` dump: fan-selected 0.2772 vs banked 0.2784, paired -0.0012 [-0.0036, +0.0012], not separated.
* **Why a figure-level check could not catch it.** Every number in the row is a correct number for base. Re-deriving 0.1914 from the base fan matches to four digits because it IS base's. The defect is in the arm label and the scope clause ("at 50.4k"), not in any digit.
* **Size of the error (MEASURED, `raw/verify_mislabel.json`).** The row overstated refcv7's selected ADE @@RSEL@@x, oracle @@RORA@@x, gap @@RGAP@@x. The qualitative claim (the fan holds much better plans than the pick; >2x on about 40 % of windows) **survives** on refcv7 (@@F0@@ % / @@F1@@ %).
* **Evidence-class note (INFERRED, not proven).** On the dev box no refcv7 fan artifact exists before `D:/refcv7_eval_kit/fanctl/` was created (2026-10-11 16:17; contents `smoke_dump`, `s0`, `s1`, `ctl` only), while the row was landed by commit `235156d` at 15:39:45
  (its message: "re-land of dfd97373", which is not readable in the git dir I was given - UNVERIFIED). So the row was stamped MEASURED before any refcv7 fan existed to measure. Thor and other disks were not searched: absence at one location only (C2).
* **Also stale in the same lines:** "Random-pick control not yet run" - it has run (X8, `@@RELRES@@` section D-REFCV7-FANCEIL). Folded into the patch as an additive note; separable from the mislabel (see hunk notes).
* **Immutable:** the message of commit `235156d` repeats the conflated sentence ("Fan ceiling: oracle 0.1914 m vs selected 0.4728 m, 41.09 % of windows"). The RETRACTION_LOG entry (section 4) is the correction of record for it.
