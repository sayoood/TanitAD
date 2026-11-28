# LABEL_SPEC: per-step strategic labels for STRAT1 (Architecture agent, 2026-11-16)

Status: accepts the Data-Eng table as built. One addition is requested (section 5, a sidecar, the table itself is unchanged). Evidence classes: MEASURED (M, artifact named), INHERITED (I, not re-verified), ASSUMED (A), UNVERIFIED (U).

## 1. The input table (accepted as is)

* File: `F:/TanitAD-scratch/strat1/labels/strat_labels_table.npz`, sha256 `8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff` (I, from `LABELS_REPORT.md`; the trainer re-hashes it, section 4).
* Source layer: the **augmented label release** `Sayood/tanitad-context-labels-v1` rev `100f03da` (private HF), not the published PhysicalAI-AV feature set.
* Row alignment: identical `train_off`, `train_n`, `eval_off`, `eval_n` and clip arrays to `navtok_table.npz` (I, MM-checked) and therefore to the trainer's join `J` (`clip`, `off`, `n`, built by `build_label_join`, `v8f_mini_data.py:166` at `cdbff971`; the trainer loads the prebuilt `--join-train` at `v8f_mini_train.py:3082`). Step j of clip c is row `off[c] + j`.
* Arrays used: `<split>_<var>_id` (int8, -1 invalid), `<split>_<var>_valid` (bool), `<split>_flag` (0 ok, 1 clip absent, 2 out of tolerance), `vocab_<var>`.

## 2. Variables, classes and what the trainer does with each

Variable order is FROZEN as `CTX_VARS` (a module constant, pinned by a test). Class ids are the table's ids, with no remap.

| # | var | classes (table order) | logits | loss target | read |
|---|---|---|---|---|---|
| 0 | country | 25 (Austria ... United States) | 25 | hard CE | accuracy + balanced accuracy vs constant 0.092 and vs the rig control (section 5) |
| 1 | weather | clear, fog, rain, snow | 4 | hard CE | accuracy + per-class recall vs constant 0.846 |
| 2 | surface | dry, snow_ice, wet | 3 | hard CE, **half weight** (borderline label) | accuracy + per-class recall vs constant 0.785 |
| 3 | illumination | dark, day, lit_night, twilight | 4 | hard CE, **twilight target masked** (card: 0/9 recalled) | accuracy on dark/day/lit_night vs constant 0.595; twilight reported as n/a with n (eval 240 steps, 1 clip) |
| 4 | period_of_day | afternoon, evening, midday, morning, night | 5 | hard CE | accuracy vs constant 0.241; the label is the clip clock hour (metadata), so this reads "can vision recover the clock" |
| - | visibility | good, reduced | **not built** | none | not read. Card audit 25/40 below the always-good 31/40 (I, CONTEXT_LABELS_CARD); the column is never loaded |

Constants above are the eval majority accuracies (I, `LABELS_REPORT.md`; the reader recomputes them from the table).

Total logits 41. The 4th illumination logit is never a positive target; it stays so that ids equal the table's and no remap can drift.

## 3. Which step's label a window uses (the alignment rule)

Line numbers are at the base commit `cdbff971` (justified in `BUILD_SPEC.md` section 0).

* The head reads the **current (NOW) frame**: model index `last_now = Fn - 1` (`v8f_mini.py:1029` at `cdbff971`), exactly as the level and lead heads do. It reads `feats` and `frame_ok` only. (Reading S's tick frame instead would make the head depend on the hold age `a_s`, a schedule input; NOW keeps it a function of the image alone.)
* **Label row = `J["off"][ci] + t`**, the NOW cache step of the window. It is staged in the trainer by a new `ctx_stage(batch, CL)` from `batch["ci"]`, `batch["t"]`, the same lookup `lead_stage` uses (`v8f_mini_train.py:2155-2161` at `cdbff971`). No data-item or collate change is needed.
* `ctx_stage` runs BEFORE `ego.apply` / `dag.apply` / `comp.apply` / `rcv.apply`, as `lead_stage` does: a swapped sample is still the logged frame and is asked for the logged frame's label.
* A target is -1 (masked) when any of: `flag != 0`; `<var>_valid` False; the clip is not in the table; `frame_ok[:, -1]` False (the NOW frame is not a real frame); (illumination only) the id is twilight (3).
* DAgger rows: the DAgger swap rewrites kinematic tensors only (`rows_changed` lists `p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav`, never `feats`; M, `t2v8f_pulls/obd_OBD-CTL_s0/selfcheck.json` `dagger_probe`). The ego-perturbation path (`EGO_PERTURB_REBUILT`, `v8f_mini_data.py:760` at `cdbff971`) does not list `feats` either. The build spec pins both by test.
* Eval: the T0 eval writes `win_ci`, `win_t` per window; the reader looks the label up at `eval_off[ci] + t` (fresh ages, so NOW is the tick).

## 4. Coverage the trainer must reproduce (and refuse on mismatch)

* MEASURED here (`strat_labels_table.npz`, this session): the 60 train clips the release lacks (flag 1 on every step) are **exactly cache indices 540-599**, which is the trainer's own `--exclude 540:600` (argv of every DAG-OACC-family run, e.g. `t2v8f_pulls/obd_OBD-CTL_s0/argv.txt`). So on the trained 4,512 clips the only masked steps are the 120 out-of-tolerance ones plus unknowns (43 weather, 13 surface, 13 illumination train steps; I, `LABELS_REPORT.md`).
* The trainer logs, per variable, the count of valid targets over the trained windows and refuses to start if any variable's valid share over trained windows is below 0.98 (a stated guard, ESTIMATED from the 98.66 % all-clip coverage and 99.97 % on present clips).
* The trainer asserts `table.train_clip == J["clip"]`, `train_off == J["off"]`, `train_n == J["n"]` and the sha256 (`--ctx-labels-sha256`) before any step. Any mismatch is a launch refusal, never a warning.

## 5. Requested addition: a platform sidecar (reader only)

The country rig control needs `platform_class` per clip, which the table does not carry. Please add a **separate** file so the table's sha256 stays `8055a321...`:

`F:/TanitAD-scratch/strat1/labels/strat_platform_by_clip.npz` with `train_clip`, `eval_clip` (same order as the table), `train_platform_id`, `eval_platform_id` (int8), `vocab_platform` = `["hyperion_8", "hyperion_8.1"]`, from the same `data_collection.parquet` the QA used, plus its sha256 in a sidecar json.

* It is read by the **reader only** (rig control, section 4 of `READ_PLAN.md`). The trainer never loads it, and the build spec adds a test that the trainer's source does not mention it.

## 6. Time base

ASSUMED by the table builder: cache row 0 = video frame 0 at 30 fps. A 0.4 s shift changes at most 1.6 % of eval steps for weather and 0 % for country and period_of_day (I, `LABELS_REPORT.md`). Accepted for this arm: the deciding reads are clip-dominated variables and the shift bound is below every decision margin in the card. The reader reports the 0.2 s shifted-label accuracy as a sensitivity row for weather and illumination.

## 7. What is NOT a label here

`month`, `hour_of_day`, `sun_el_deg_est` and `platform_class` are never model inputs and never trainer targets in this arm. `period_of_day` is a target only (its source is the clock hour), never an input.
