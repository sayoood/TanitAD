# refcv8 v9 TRAIN label release over all 4,572 clips (4,369 old + 203 ADD) -- 2026-10-12

Eval tier: n/a (label release, no model). Evidence MEASURED unless tagged; artifacts under `work/` (dev box) and the staged `v9_train4572/` dir.

## 0. Headline
**CONTROL PASS.** New release `v9_labels_train.npz` md5 **`83e30bd3791f0767707eade5f4671759`**, **4,572 clips / 909,651 rows** (old 4,369 / 869,278 + 203 ADD / 40,373 rows).
All 4,369 old clips are **bitwise identical** to the old release (`f63ece41...`), compared by clip (sid), every array: **4,369 clips, 869,278 rows, 118 row fields + 13 clip fields, 0 differing** (`row0` excluded -- the new manifest interleaves the ADD clips, so row offsets move, values do not).
Trainer-side check PASS: `load_v9_join` (md5 + integrity census, 0 violations) and `row()` over **all 781,635 dataset windows** (0 errors), `item()` x 396 pulls on 11 clips.
Thor (NEW path, md5-verified both ends): `/home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz` (+ `.manifest.json` md5 `7faeec88733c2aa9a29608e3ef5b9c0c`, same dir -- the reader derives the manifest path from the npz path).
**Launch gate must pin:** `--r8-v9-labels /home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz --r8-v9-md5 83e30bd3791f0767707eade5f4671759`.
**One script patch is REQUIRED to reproduce** (section 3): `stack/scripts/build_v9_labels.py` crashes on a parked recording; patched + test staged.

## 1. What was built (flags match the old release; only manifest + clock sidecar changed)
`build_v9_labels.py build --split train --manifest in/train4572_v2manifest.pt (77dafb50...) --sidecar refcv8_clip_clock_sidecar_ts33.jsonl (c81c2739...) --labels s2_labels_v8_train.jsonl.gz (b45377a1...) --lead lead_train4572.npz --lc lc_train.npz --lc-labels off --base-commit 58d413f` (old: same labels / `--lc-labels off`; manifest 3c9f8bc8 -> 77dafb50, sidecar 78466f99 -> c81c2739). Wall 459 s. Manifest JSON: `literals`, `vocab`, `policies`, `fields` all **equal** the old manifest's.
| input | value | how obtained |
|---|---|---|
| lead (`lead_train4572.npz`, md5 `d4336a44...`) | 4,427 clips, 849,263 frames | **MERGE** of the old `lead_train.npz` (4,369 clips, `67451737...`) + a lead-pass over the 203 only (`lead_add203.npz`, 58 clips, 11,133 frames, `4c9ade62...`). Per-clip keyed `(sha12, k)`, same schema/dtypes -> clean merge. |
| lead control | a **full** lead-pass over all 4,572 (`lead_full4572.npz`, `d2728533...`) is **record-for-record bitwise identical** to the merge (849,263 records, 0 differing in 8 fields) -- so the OLD lead reproduces under the current builder + ts33 sidecar. |
| agent join used | the **3-D** join `join3d/b1_train_plus_eval_agents_3d.jsonl.xz`, md5 `401c9c3119a50632dafc35c595d9e051` -- this is what the old `lead_train.npz` meta records. **The brief's dev copy (`b1train_agents.jsonl.xz`, `1c985e6d...`) is the 2-D train join and is NOT the file the old lead-pass used.** The 3-D join was copied Thor -> dev box. |
| lc (`lc_train.npz`, `3e4f0336...`) | REUSED, no lc-pass | `sam3_gt_v3` holds **0 of the 203** (file test 0/203; `MANIFEST.jsonl` 0/203; positive control: 50/50 old sha12 found). An lc-pass over the 4,572 list would select exactly the old 4,369 files. |
Clips absent from lead/lc are **MASKED, not an error**: `lead_by.get(s12)` / `lc_by.get(s12)` -> None -> `has_lead` / `has_lc` = 0 (`cmd_build`).

## 2. Controls
| control | result |
|---|---|
| old 4,369 clips vs old release, by sid, bitwise | **4,369 compared, 0 differing**; 132 fields in both, identical dtypes; per-clip `n_rows` equal (`logs/control.log`, `out/control.json`) |
| new clip set | sid set == the 4,572 manifest sids, sha12 set equal, clip order == manifest order, no duplicates |
| eval139 | **0** of 139 eval139 clip ids in the release (by sid AND by sha12) |
| ADD | 203 sids in new, 0 in old; 203/203 have a v8 record; `now_s` finite on all 909,651 rows |
| clocks | ADD: 203/203 `clock_src` = sidecar (the ts33 route for the 33 stationary ones); old clips: 4,347 sidecar + 13 pose_dt + 9 nominal (the 22 unclocked, unchanged) |
| lead provenance | merge == full pass (section 1) |
| trainer side (`trainer_check.py`) | `load_v9_join(..., expect_md5)` OK, census violations `{}` over 909,651 rows; `row()` (= `row_for_now`, now_s to 1e-6 s, trainer clock restated from `clip_clock`) **781,635/781,635 windows OK** (k = t+9, windows/clip = T_out-28; clock sidecar/pose_dt/nominal = 4,550/13/9); `item(speed_mode in {None,n2,n3}, cons=True)` 396 pulls on 5 ADD + the parked ADD clip + 5 old clips, no raise |
| existing tests | `test_v9_labels_builder.py` 20/20 green on the patched script |

## 3. Script patch (REQUIRED) -- `stack/scripts/build_v9_labels.py`
`lead-pass` AND `build` died with `ValueError: Shape of array too small to calculate a numerical gradient` on **`12c90362-a57f-4c3d-bf0d-ebe47a687186`** (sha12 `8f9a89b157ed`; ego x/y range exactly 0.0, v == 0 for 20.35 s; arc 0.0 m). Cause: `smooth_path` builds a ONE-point arc grid when the whole 10 Hz track is shorter than `ds` = 0.5 m, and `np.gradient` needs >= 2. Exactly 1 of the 203 ADD clips (next shortest arc 1.586 m); none of the old 4,369 reaches that branch.
Change (6 lines, `smooth_path`, after `grid = np.arange(...)`): `if len(grid) < 2: grid = np.array([0.0, ds])` (then the existing interp clamps to the end position). It executes ONLY where the old code raised, so no previously built clip can change -- confirmed by the control (0 differing). Resulting labels for that clip are the literal ones for a parked car: LANE_KEEP, HOLD, nav FOLLOW, no route checkpoint, no lead.
Staged: `stack/scripts/build_v9_labels.py` (md5 `e6b2a7d4f125961faae9a178754f2ca6`, LF, as the repo blob; base = origin 58d413f content, LF md5 `128846cb...` == the old release's recorded builder md5) and `stack/tests/test_v9_labels_parked_clip.py` (4 tests, literal expectations): **patched 4/4 green, original builder 3/4 RED** (deliberate-regression arm). The release manifest records the PATCHED builder md5 (`e6b2a7d4...`) and `s2_geom_emit_v7.py` `9a537ee2...` (== old release).

## 4. ADD coverage (203 clips, 40,373 rows) -- what the trainer will see (masking, not "no objects")
| field | ADD | old 4,369 |
|---|---|---|
| clips with a lead record (`has_lead`) | **58/203** (the 145 have no agent join -- unchanged, cannot be joined, see add203 RESULT sec. 4) | 4,369/4,369 |
| clips with lc / map GT (`has_lc`) | **0/203** | 4,369/4,369 |
| `lc_measurable` / `lc_frac` non-NaN | 0 % NaN-valid (`lc_frac` 0.0) | 97.9 % |
| `lead_valid` = 1 rows | 11.2 % | 40.3 % |
| `lat_cls_a` IGNORE (partial or masked) | 20.3 % | 16.1 % |
| **`lon_cls` IGNORE** | **78.1 %** (partial-label 68.8 %, fully masked 9.3 %) | 28.7 % (24.3 % / 4.4 %) |
| `goal_w` rows with any supervised goal | 93.1 % | 95.1 % |
| `lon_v_target_ms` non-NaN | 90.7 % | 95.7 % |
| `h_obs_s` mean | 7.67 s | 7.78 s |
| turns announced (clips mean) | 1.39 / clip (283 over 203) | 1.64 / clip |
**Caveat to carry (a data property, not a build defect):** the ADD set is heavily low-speed/stationary -- 132 of 203 ADD clips have >= 90 % of rows with `lon_cls` = IGNORE (mostly the PARTIAL-label branch, allowed-set non-empty), against 28.7 % overall on the old clips. The longitudinal class loss on these windows is therefore mostly the `-log sum_{c in allowed}` form; their share of the 781,635 windows is 34,689 (4.4 %). Full table: `logs/control.log` / `out/control.json` (`coverage`).

## 5. UNVERIFIED / not done
- **Windows-per-clip rule** in the trainer check is `T_out - 28` (window 8 + max_horizon 20), the formula the add203 run validated against 746,946; it reproduced 781,635 here, but the real `V3Dataset.index` was not instantiated (no image cache on the dev box). The check used the repo export 58d413f's `refcv8_train.py` / `v9_labels.py`, not Thor's `/home/nvidia/refcv8_ladder/tree` (not touched).
- lc-pass was reused, not re-run (justified by 0/203 GT files); no lc-pass over the 4,572 list was executed.
- The brief's join statement (dev copy `1c985e6d`) was found wrong; the 3-D join `401c9c31` is what the old lead-pass recorded -- the equality "merge == full pass" is the evidence that it is the right one.
- VIS-1 sidecar (`launch_gate.py:572`) is a separate blocker not in this scope.
- `base_commit` in the manifest is `58d413f` (old: `50efa52`); metadata only.
