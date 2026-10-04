# X10 — pose-to-image timing: RESULT (Data FlyWheel, 2026-10-04)

**Escalation (integration + decision, in the headline per the operating standard):**
1. **LAND the fix tree** `code/fix/` (3 new files, 3 minimal modifications; list + blob ids in `LANDING_READY_X10.txt`).
   Built on tip `acc6180`; the base blobs of the three modified files (and of the 7 source files DESIGN cites) were re-verified UNCHANGED at the current tip `c51d1cb`.
2. **SHIP the sidecar** `raw/refcv6_pose_sync_sidecar.jsonl` (5.4 MB, no clip ids) to Thor next to the clock sidecar
   (`/home/nvidia/data/`) — it covers **4,508 / 4,508** corpus clips (4,369 train + 139 eval). It is regenerable
   (`build_pose_sync_sidecar.py`; needs only the camera `timestamps.parquet`s, all on the dev box).
3. **DECIDE: enable `--pose-sync-sidecar` for refcv8?** My recommendation and its size are in §4. Short form: **enable —
   it is cheap, bit-identical when off, and turns SPEC X10's POSE-SYNC row from 17 ms to 0.07 ms — but expect NO movement
   in ADE or any planner metric from it**; the targets move by ~1 cm (0–2 s) and ~6 cm (6 s).
4. **DECIDE (minor):** whether the standalone eval kit (`tanitad/eval/refcv7_loader.py:787`) should also call
   `enable_pose_sync` (not changed here; §5).

## 1. The defect, measured (item 3 of the brief)

| | value | evidence |
|---|---|---|
| image-minus-pose time, eval139 (27,664 provider rows) | mean **16.505 ms**, p05 1.4, p50 16.6, p95 31.3, p99 32.7, max **34.02** | MEASURED `raw/x10_measure.json` `corpus_delta` |
| same, all 4,369 train clips (869,278 rows) | mean **16.502 ms**, max 35.43 | same |
| mechanism | `delta` falls 0.667 ms per row and wraps every ~50 rows (5 s): camera period 33.333 ms vs grid step 100.667 ms | `DESIGN.md` §1; first sidecar row `[0, 32667, 32000, 31332, …, 16, 32682]` µs |
| NOW-pose position offset (eval139 trainer windows, n 23,772) | mean **0.193 m**, p50 0.147, p95 **0.561**, p99 0.835, max 1.185 | MEASURED; 100 Hz-log version 0.186 m (n 27,664) |
| D3 cross-check | D3: "0–34 ms, mean 16.5 ms, 0.19 m mean" → **reproduced** (D3's "~0.9 m at p95 clip speed" is δ_max × p95 speed, a product; the joint p95 is 0.56 m) | D3's derivation was not banked; mine is independent (DESIGN §2) |

Independent derivation (DESIGN §2): the cached poses are reproduced **bit-exactly** from raw camera timestamps + raw
egomotion over **539 clips** (max |Δ| 0.0); the independently-derived clock sidecar agrees on `grid_start_s` to ≤ 4.3e-4 ms
and `dt_s` to ≤ 3.8e-9 s (n 535); a closed form from four numbers agrees with the measured sawtooth to a median of 53 µs.

## 2. What it does to the training targets (items 1 + 3)

Window shift ("B", the implemented correction) vs the uncorrected path ("U", what refcv7 trained on), through the **real
`V3Dataset` hook and the trainer's own `waypoint_targets`**, eval139, 23,772 windows (`raw/x10_dataset_e2e.json`; it
reproduces the independent numpy derivation in `raw/x10_measure.json` to the printed digits):

| horizon | n | mean m | p50 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| 0.5 s (tick 5) | 23,772 | 0.0042 | 0.0021 | 0.017 | 0.028 | 0.057 |
| 1.0 s | 23,772 | 0.0086 | 0.0044 | 0.033 | 0.054 | 0.114 |
| 2.0 s (tick 20) | 23,772 | 0.0182 | 0.0096 | 0.069 | 0.112 | 0.280 |
| **pooled 0–2 s (ticks 1–20)** | 475,440 | **0.0093** [0.0083, 0.0101] | 0.0036 | 0.038 | 0.074 | 0.280 |
| 4.0 s (tick 40) | 21,131 | 0.0388 | 0.0193 | 0.140 | 0.256 | 0.629 |
| **6.0 s (tick 60)** | 18,351 | **0.0597** [0.0517, 0.0669] | 0.0292 | **0.210** | 0.389 | 0.754 |

(brackets: 95 % episode-cluster bootstrap over the 139 clips, 1,000 resamples.) A seeded **train sample** (400 of 4,369
clips, seed 20261004, 68,373 windows) is the same size: 0–2 s mean **0.0104** [0.0098, 0.0110]; 6 s mean **0.0681**, p95 0.236,
p99 0.437, max 1.165.

* **Zero-mean:** the signed longitudinal mean at 6 s is −0.0035 m (eval) / +0.0013 m (train sample) against an absolute mean of
  0.044 / 0.048 m — it is label NOISE, not a bias, and carries no scene information (the sawtooth is a function of the row index).
* **v0** `= pose_last[:, 3]`: mean **0.0084 m/s** (0.03 km/h), p95 0.033, max 0.122 (eval139). Ego-history encoder inputs:
  |Δspeed| mean 0.0085 m/s, |Δyaw-rate| mean 0.0006 rad/s.
* **Tactical kinematic factored labels** flip on lat 28 / 23,772 (0.12 %) and lon 66 / 23,772 (0.28 %) windows.
* **Interpolation error** of the 10 Hz linear re-sampling vs the 100 Hz log: pose at NOW mean 0.5 mm, p95 1.8 mm, max 4.9 mm;
  targets pooled 0–2 s mean 0.5 mm / p95 1.5 mm and 0–6 s mean 1.0 mm / p95 3.6 mm — 20–30× below the shift it corrects (9.3 mm and 27.7 mm mean).
* **Why the targets move so little although the position offset is 0.19 m:** a trajectory target is a displacement in the NOW
  frame; a constant-velocity displacement is invariant to the shift and an accelerating one changes by `a·δ·H`. The 0.19 m is an
  *absolute registration* error; it is invisible to ego-relative targets.
* **The literal "per-row" reading is the wrong correction (MEASURED):** re-sampling every row at its own camera timestamp moves
  the 0–2 s targets by **0.109 m** mean (p95 0.361 m) — **~11× the 0.0093 m the defect is worth** — because the waypoint's elapsed
  time becomes `h·dt + delta_{r+h} − delta_r`. A cache rebuild that "samples poses at the image time" would bake exactly that in.

## 3. SPEC X10 row (SPEC_REFCV8_DRAFT §8.10) with the correction built

POSE-SYNC (mean |t_pose − t_image|, expressed as ‖pose_last − pose_100 Hz(t_image)‖ / speed, windows with v > 2 m/s, n 21,453):

| arm | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| OFF (refcv7 path) | **17.133 ms** | 17.4 | 31.4 | 32.9 | 34.1 |
| ON (`--pose-sync-sidecar`) | **0.067 ms** | 0.027 | 0.274 | 0.468 | 1.246 |

Bar **≤ 1 ms (mean): PASS** (0.067 ms; the residual is the 10 Hz interpolation, so it is an upper bound of the timing error).
The spec's control (a constant-velocity track shifted by 16.5 ms moves by v × 0.0165 exactly) and its deliberate regression
(correction off reads ≈ 16.5 ms) are both implemented as tests (§6). NIGHT-LUMA / LEFT-N are report-only and D3's; unchanged.

## 4. What the Master Mind must decide — enable for refcv8? (item 4)

**Recommendation: ENABLE, and expect nothing from it.**

| for | against / honest limits |
|---|---|
| Closes SPEC R8-7's X10 row (otherwise "NOT BUILT ⇒ R8-7 FAILED on X10"): 17.1 → 0.07 ms | Effect on the targets is **1 cm (0–2 s) / 6 cm (6 s) mean**, p95 4 / 21 cm — far below the rig's ≈ 0.30 m seed floor (H-ESTIM-SEED-1, `D-REFAV1-SEED-GOAL-MISMATCH`). **It cannot be seen in ADE; do not spend a replicate arm on it, and do not report an ADE change as its effect.** |
| `v0` becomes the speed at the image instant — the PI-admissible "velocity at cycle time" in its most faithful form | It is a **second changed lever** versus refcv7 (cm-scale target noise removed + a 0.03 km/h `v0` shift). Under one-variable discipline, state it in the launch record: `config.json` `pose_sync` is non-null exactly when it is on. |
| Opt-in; OFF is bit-identical (tests); no row, clock or `(sid, k)` join moves; adds **no** inference input | The 0.19 m **absolute** offset is NOT removed from anything keyed on the pose clock — the 2-D/3-D box join, v9 labels (immaterial at ±2 s), map GT (time base UNVERIFIED). See "next lever". |
| Warm start safe: targets do not enter the forward pass; the step-0 forward sees only the `v0` / ego-history speed perturbation above | The checkpoint was not forward-tested here (CPU box, no refcv7 weights in reach): the "negligible input perturbation" is arithmetic, UNVERIFIED by a forward. |

Cost: one 5.4 MB file on Thor and one launch flag; no cache rebuild, no GPU; the hook costs 0.67 ms per window on CPU (MEASURED, 3,000 windows, stub frames; a real window also decodes images).

**Next lever, ranked by measured size, if the PI wants the 0.19 m gone where it IS first-order:** the box/agent join is
keyed on the pose clock (`build_b1_agent_join_3d.py:253`), so every box label is stale against its image by `v_rel · δ`
(a static object by `v_ego · δ`, mean 0.19 m, p95 0.56 m, max 1.2 m) on top of its own nearest-label-sample rounding. The
same sidecar re-times that join (look the boxes up at `t_query + δ`) — it needs the join artifact rebuilt (a CPU job over
`b1_train_plus_eval_agents_3d.jsonl.xz`), which is shared with the warm start, so it is a PI decision, not mine. The cheapest
discriminating experiment first: project the banked 3-D boxes of ~20 static-object clips into the image at `t_query` and at
`t_query + δ` and compare box-edge NCC; if the shifted one wins, the join rebuild is justified. Not run here (needs the
cylindrical remap + the box join's per-frame lookup; ~half a day of CPU).

## 5. Eval-side note

`train()` calls `enable_pose_sync` on the in-training eval split too (one flag, one clock). The **standalone** refcv7 eval kit
(`refcv7_loader.py:787`) is unchanged: scoring refcv8 there keeps its GT on the uncorrected clock (≈ 1 cm mean ADE-level
difference at 2 s) and keeps every number comparable to banked refcv7 results. Adding the same call there is a 3-line,
Master-Mind-owned change.

## 6. What was built and how it is guarded (item 2)

* `stack/tanitad/data/pose_sync.py` (new): grid replica, closed form, fractional-row re-sampling (shortest-arc yaw, bit-exact at
  integer rows), sidecar reader with 8 refusals, window view.
* `stack/scripts/build_pose_sync_sidecar.py` (new): the builder (K1–K4 above).
* `stack/scripts/refc_v3_train.py` (+109 / −0): `pose_sync` class attr (default `None`), `V3Dataset.enable_pose_sync`,
  `_pose_sync_apply`, one `if self.pose_sync is not None:` in `__getitem__`, `--pose-sync-sidecar` (default off), train + eval
  enable, `config.json` `pose_sync`. `declared_vs_built.py` (+1): registers the new dest; `tests/test_declared_vs_built.py` (+2): that test PINS the registry size
  (256 -> 257) - it was the single regression the neighbour run found, fixed here.
* `stack/tests/test_pose_sync.py` (27 tests): **known values** (all literals, hand-worked in the file): constant velocity
  shifts by exactly 0.198 m (= 12 × 0.0165) and its targets do not move; constant acceleration moves the 10-tick waypoint by
  0.033220 m; the NOW row's own delta is used (0.024160 m at one window, 0.038253 m at another); the grid replica gives
  19497.4874 / 38994.9749 / 18492.4623 µs for a 25 fps camera. **Zero-offset control:** all-zero sidecar is bit-identical
  to the OFF path (reader level and every key of five dataset items; random yaw so the seam is exercised). **Deliberate
  regressions that must go RED:** sign flipped (the known-value check fails; the dataset target moves the opposite way, −0.033 instead of +0.033 m);
  the per-row shift injects the 0.24 m jitter. **Refusals:** negative / > 50 ms delta, wrong `n_rows`, `dt_s` out of band, no
  `sid`, missing / empty file, conflicting duplicates, a sidecar covering nothing or another timeline, > 1 % uncovered clips; a
  refusal leaves the dataset OFF. **Builder:** hand-worked deltas end to end, refuses a foreign timeline, refuses a disagreeing
  independent clock, writes no clip id.
* **Mutation check** (`code/mutation_check.py`, `raw/mutation_check.json`): 7 re-introduced defects (sign, per-row shift,
  lost bit-exactness, dropped future poses, wrong NOW row, no shortest-arc yaw, negative deltas accepted) → **7 / 7 caught**.
* **Regression of the neighbours:** the 92 test files that import `refc_v3_train` / `declared_vs_built` were run in the
  overlaid tree and in a pure tip tree — result below.

**Neighbour regression (MEASURED, `raw/suite_regression.json`, logs `raw/suite_overlay_final.log` / `raw/suite_base.log`):** 92 test files
(every file that imports `refc_v3_train` or `declared_vs_built`; 1,819 test items incl. skips and errors) run on (a) a pure `git archive` of the tip and (b) the same
tree with `code/fix/` overlaid: **both read 14 failed / 1,776 passed / 10 skipped / 19 errors, with the identical failing set (33 ids:
`test_openloop_suite` 19 errors, `test_launch_gate` 8, `test_refcv6_diffusion` 3, `test_g_box_overfit` 2,
`test_refcv6_loader_stamped_queries` 1) - zero regressions, zero differences.** Those 33 are the tip's own (this is a partial `git archive`
tree on a Windows dev box; not investigated). The FIRST overlay run read 15 failed: `test_declared_vs_built.py::test_GDVB_every_trainer_flag_...`
pins the registry size at 256 and the new dest made it 257 - fixed by a +2-line edit of that test (now in the fix tree), after which the
overlay run matches the tip exactly. Any other refcv8 stream adding a trainer flag edits the same pinned line.

### How to use it

```
# launch (refc_v3_train.py): add the one flag; omit it and the run is byte-identical to the tip
--pose-sync-sidecar /home/nvidia/data/refcv6_pose_sync_sidecar.jsonl
# the run record then carries config.json  pose_sync = {train: {...}, eval: {...}}   (None = uncorrected)

# regenerate the sidecar (CPU, ~1 min; needs only the camera timestamps.parquet files)
python stack/scripts/build_pose_sync_sidecar.py --manifest <eval _v2manifest.pt> --manifest <train _v2manifest.pt> --camera-dir <dir holding <clip_id>.timestamps.parquet> --clock-sidecar <refcv6_clip_clock_sidecar.jsonl> --out <sidecar.jsonl>
```

Environment of every number above: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `PYTHONPATH` = a clean `git archive` of the tip
(`stack/{tanitad,scripts,tests}` + `taniteval/{taniteval,tools}`) with `code/fix/` overlaid, in the session scratchpad; `tanitad.__file__`,
`pose_sync.__file__`, `physicalai.__file__` and `refb_labels.__file__` were printed and are inside that tree (the editable install that
points at G: was NOT used). CPU only; OMP_NUM_THREADS 1-4; no Thor, no GPU, no G:.

## 7. UNVERIFIED / not done

* Pixels of row `k` = camera frame `frame_idx[k]`: MEASURED on 4 rows / 2 clips only (all best-match offset 0, not bit-exact; `raw/x10_pixel_check.json`), the other rows rest on the builder code; what instant the camera timestamp stamps; camera ↔
  egomotion clock sync below 1 ms in the dataset itself (DESIGN §2).
* No forward pass of refcv7-50,400 under the shifted inputs.
* The box join / SAM3 map time bases were read from code (box join) or not at all (map); the "next lever" is a hypothesis with a
  named cheap test, not a finding.
* Eval139 + a 400-clip seeded sample, not all 4,369 train clips, for the TARGET shift (the OFFSET itself is over all 4,508).
