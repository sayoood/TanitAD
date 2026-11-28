# FACTS_KINPUT: `--dagger-base own|logged` (BRIEF_BUILD_KINPUT, 2026-11-23)

Evidence class: MEASURED-FROM-SOURCE at fde8b716, tree `F:/TanitAD-scratch/kinlab/tree4` (`git archive fde8b716 stack taniteval`). DD = `stack/tanitad/data/v8f_dagger.py`, MD = `stack/tanitad/data/v8f_mini_data.py`, TR = `stack/scripts/v8f_mini_train.py`, T1 = `stack/scripts/v8f_t1_eval.py`, M = `stack/tanitad/models/v8f_mini.py`. Nothing staged, committed or pushed; no GPU; nothing on the pod; nothing on C:.

## PHASE 1: FACTS (banked before any code)

### 1. What a swapped own sample takes from the own state and what from the logged record
The batch row of a swapped sample is the LOGGED window at (clip, t) with t = r' (MD:`item` 660-690); `DaggerSwap.apply` (DD:785-800) selects only rows whose (ci, t) has a bank record with r' == t (`DaggerBank.candidates`, DD:323; `pick`, DD:333; the probe asserts `rec["r"] == batch["t"]`, TR:2948) and calls `swap_rows` (DD:582-636).

| model input / batch key | swapped sample | where |
|---|---|---|
| `p10f` [F, 8] (kin-feature history) | OWN (record `p10f`); a frame with no own row (padded, row 0, before the clip) keeps the logged value | DD:598 |
| `p10_t` (= p10f at NOW) | OWN (`P10[:, -1]`) | DD:599 |
| `kin3` (v0, a0, kappa0) | OWN (record `kin3`) | DD:600 |
| `base` [30, 2] | OWN: `kdx_base(own kin3, yaw)` | DD:601 |
| `pose` [F, 3] history | OWN world poses re-anchored onto the LOGGED NOW (`pose_now`) at r' | DD:602-607 |
| `pose_t`, `pose_s` | OWN (taken from the re-anchored `pose` at ages `a_t`, `a_s`) | DD:609-610 |
| `nav` | channel 3 (progress `d_next/100`) rebuilt from the own travelled arc on real frames sharing NOW's token with a valid route; channels 0-2 and 4 LOGGED | DD:611-618 |
| `feats` (the image-token grid), `frame_ok`, `limit`, `limit_ok`, `a_t`, `a_s`, `pose_now` | LOGGED at r' (KEPT, the same objects) | `D.EGO_PERTURB_KEPT` (MD:760) |
| `act` (world-model transition record) | LOGGED (RECORD) | MD `EGO_PERTURB_RECORD` |
| targets `gt`, `gt_ok`, `s_h`, `lead_gap_h`, `lab`, `lab_t`, `lab_s` | LOGGED (the human's at r'); `lab_s.rc` re-expressed in the own S frame (DD:619-622); `gt` / `s_h` replaced only under `--dagger-label speed_v1 / kin_v1` (DD:623-634) | |
`DG.REBUILT` = p10f, base, kin3, p10_t, pose, pose_t, pose_s, nav (DD:`REBUILT`, MD:760). The model's inference path reads `feats, frame_ok, p10f, act, nav, pose, limit, limit_ok, base, a_t, a_s, pose_now, pose_t, pose_s` (M:176); `kin3` and `p10_t` are NOT in that list.

### 2. How `base` is computed
* Logged sample (clip store yaw mode): `d["base"] = kdx_base(d["kin3"], KIN_UNITS["yaw"])` at MD:590, with `d["kin3"] = yaw_kin_from_track(track)` = (v0, a0, kappa0) at cache step j (MD:351-366, float64, row 0 NaN -> base 0); the item takes `d["base"][t]` (MD:681), float32, so it is `kdx_base` of the FLOAT64 logged kin3 at row t.
* Own sample: `kdx_base(rec["kin3"], yaw)` (DD:601), `rec["kin3"]` = the own kin3 at the record's clock row j, written by the bank generator through the same `yaw_kin_from_track` (T1:206-211 spelling `D.kdx_base(kin[s:s+1], yaw)`).
* The only reader of `base` in the model: `M:994`, `p30 = self.prior(p10f_at_i, b["base"])` = `base + MLP(standardised p10f)` (M:`P10Prior.forward`, 327-360), the prior path that O's residual is added to.

### 3. The matched row r', and the logged kinematics at r'
r' = `rec["r"]` = the row of the logged 0.2 s polyline nearest (by cumulative arc) to the foot of the own (x, y) (`match_route`, DD:129; PREREG section 2); records with r' cut (off_lat, off_head, end_edge) are not eligible. A batch row is swapped only when its own window row t equals a record's r', so **the logged window at r' IS the batch row**: `batch["kin3"][rows]` is the logged kin3 at r' (float32 of `kin3[r']`) and `batch["base"][rows]` is the logged `d["base"][r']`. The clip store holds (`ClipStore.get_kin`, MD:542-569) the float64 `kin3 [T, 3]` = (v, a, kappa) at every cache step (the logged kinematics at r'), and the bank record holds its own `kin3` at j (the clock row, j = t0 + 10 or t0 + 20; r' may differ from j).
**Consequence for the design:** the logged base at r' needs no lookup: it is the batch's own unswapped `base[rows]`. It is bit-identical to `kdx_base(store kin3[r'], yaw)`; `kdx_base(batch["kin3"])` would not be (float32-rounded input), so the build keeps the batch tensor rather than recomputing from `batch["kin3"]`.

### 4. What else derives from `kin3` / the own kinematics; what "the own pose history kept" leaves
* `base` is a function of `kin3` only. `p10f[:, -1]` / `p10_t` are `kin_features(kin3)` (MD:587), so under `logged` the own `p10f` (the other prior input) and the logged `base` belong to DIFFERENT kinematic states: `prior(p10f_own, base_logged)`. That is exactly the pair the card names ("`base` of an own sample rebuilt from the logged kinematics at r', the own pose history kept"); it is not a defect of the build, it IS the lever, and the facts say it plainly: **with `logged`, O's prior sees the own 8 kin features and the human's kd_x path.**
* `kin3` itself is not read by the model (`M` has no reader; M:176). In the trainer it is read as a LABEL only before the swap (`launch_stage`, TR:2164; `level_stage`, TR:2267, both "BEFORE dag.apply"), and by the probes. Under `logged` I leave `kin3` and `p10_t` OWN (the brief: only what phase 1 shows must move with base; nothing the model reads is derived from `kin3` except through `base` and `p10f`, and `p10f` is the history the card keeps). So **exactly one rebuilt tensor changes: `base`.**
* "Own pose history kept" leaves in place: `p10f` (own, 16 frames), `p10_t`, `kin3`, `pose` / `pose_t` / `pose_s` (own, re-anchored to the logged NOW), `nav[..., 3]`.
* The rest of the pair stays consistent where it matters for the lever: the label (`gt`, `s_h`, `lab_s.rc`) is untouched; the G-LIVE `dagger_probe` (TR:2889) required `base` to change on every selected row (TR:2958-2961, apart from rows at rest); under `logged` that clause must instead require `base` UNCHANGED on the selected rows (a probe edit in TR, with the loss still required to move through `p10f`).

### 5. Inference
* T1 executor (T1:206-211, `_finish`): each rollout step writes `d["base"][j] = kdx_base(kin[s:s+1], yaw)` from the OWN rolled kinematics, and the executor reads `model.prior(batch["p10f"][:, -1], batch["base"])` (T1:307). T0 eval reads the store's logged base. Neither reads any trainer flag; `--dagger-base` lives in `DaggerSwap.apply` / `swap_rows`, called only at TR:3796 (training loop) and in the gate's probes. **The change is training-only** (`v8f_t1_eval.py` is sha-pinned and untouched).

### Verdict of phase 1
Facts 1-4 are unambiguous: "rebuild `base` from the logged kinematics at r'" = keep the logged window's own `base[r']` in the swapped rows, everything else as today. No design choice outside the card was needed. Proceeding to phase 2.

## PHASE 2: WHAT WAS BUILT (tree4 = fde8b716 + 6 files, `tree4/CHANGED_FILES_KINPUT.txt`: sha256 and base blobs)
| piece | where |
|---|---|
| `DAGGER_BASES = ("own", "logged")`, `DEFAULT_BASE = "own"` (kept out of `constants()`); `swap_rows(..., base="own")` (unknown value -> ValueError; `own` = the DD:601 line unchanged; `logged` = `new["base"][rows] = batch["base"][rows]`, nothing else differs); `swap_item(..., base=)`; `DaggerSwap(..., base=)`; `built_levers()["dagger_base"]`; `law_problems` (unknown value, `logged` with a mode other than swap), `problems` (`logged` at p 0); `apply` passes `base` only when it is not the default (the gate's probe hooks wrap the three-argument `swap_rows`; the first run with a keyword broke 9 existing `test_v8f_dagger_train` tests, fixed in the library, not in the tests) | `v8f_dagger.py` |
| flag `--dagger-base own|logged` (default own); FLAG_CHECKS `"dagger_base": "dagger:dagger_base"` (wired as `dagger_kin_tb`); `dagger_base_refusal(a)` (p <= 0 or mode != swap) called at the bank load (-> `ZZDVB-REFUSEDZZ`) and in `bad_dagger`; `dfl` gains the key; declared-vs-built message `lever dagger_base: declared X but built Y`; `make_dagger` passes `base=`; G-LIVE `dagger_probe`: under `logged` it requires `base` UNCHANGED on every row (`rows_changed.base == 0`, `base_source: logged`) and skips the "base must change" clause; p10f / kin3 / p10_t are still required to change and the loss must still move | `v8f_mini_train.py` |
| test pins adjusted by one line each (a golden older than fde8b716 lacks the new levers key): `lv_n.pop("dagger_base")` | `test_v8f_kintb.py`, `test_v8f_kincache.py` |
| new `test_v8f_kinput.py` (13 tests), new golden `golden_v8f_dagger_fde8b716.py` (= `git show fde8b716:...v8f_dagger.py`, sha256 9976718b) | tests |

## The label cache (brief item 3)
Base is an input, not a label: `kin_label_bank` / `label_bank` read the bank records and the logged grid only, never `base`, so the cache ARRAYS are unchanged and `LABEL_CACHE_FIELDS` / the header fields need no new entry (`--dagger-base` is not in the identity; a cache is valid under both values). **But the header's `dagger_sha256` is the sha256 of `v8f_dagger.py`, which this change edits, so a cache written at fde8b716 is refused by this tree naming `[dagger_sha256]`** (KINCACHE's code binding, exactly as in KINTB; the labeller file is untouched, `labeller_sha256` does not move). Rebuild the caches from the landed commit (about 14 min CPU per bank) or let the MM decide on a predecessor-hash allowance; not built.

## Results (logs in `kinlab/`; on the FINAL files, whole file each, CPU, `CUDA_VISIBLE_DEVICES=-1`, one process behind `wait_commit.py 5.5 60`)
* `test_v8f_kinput.py` **13 passed** (`kif_test_v8f_kinput.log`): default gold (human / speed_v1 / kin_v1: batch, labels, built levers equal fde8b716's golden apart from the key); `logged` moves only base (planted bank; `base` = `kdx_base` of the logged float64 kin3 at r' read from the clip store, exact; all other keys equal the `own` batch; unselected rows the logged batch); refusals (p 0, add, unknown value; swap level and trainer level); reachability through the trainer argv (base differs on a swapped row, plan loss differs, levers read logged, gate passes, cross-declared gate refuses both ways); G-LIVE `dagger_probe` passes under `logged`; `main` summary carries it.
* Neighbours, final files: `test_v8f_kintb` 34, `test_v8f_kinlab` 35, `test_v8f_kincache` 24, `test_v8f_dagger` 57, `test_v8f_dagger_train` 95 (+2 skipped), `test_v8f_recovery` 22, `test_v8f_recovery_wiring` 4, `test_v8f_rolladd` 25: **all passed** (`kif_*.log`).
* **Pins:** all 21 launch-gate files (`test_launch_gate*.py`, `test_profile_testsuite.py`), whole file each, **all passed with the KINCACHE / KINTB counts** (150 + 2 skipped / 46 / 24 / 99 / 43 / 58 / 12 / 49 / 41 / 12 / 68 / 3 / 26 / 59 / 17 / 11 / 4 / 2 / 10 / 56 / 9; `ki_lg_*.log`). **No test-pinned fingerprint moved.** (The LG run started after the last edit of `v8f_dagger.py` and `v8f_mini_train.py`: file mtimes 09:23:57 vs LG logs from 09:25:06.) What moves on landing: the bytes of `v8f_dagger.py` and `v8f_mini_train.py`, hence the gate token `tree_sha256`, any trainer pin the MM keeps, the G-SUITE, and the label-cache `dagger_sha256`. Pod pin lists not checked.
* **Mutants** (`mutants_kinput/MUTANTS_RESULT.json`, runner `run_mutants_kinput.py`, on the final files after the last edit): control M0 GREEN; **14 of 14 RED**: P1 flag parsed not passed, P2 `logged` still rebuilds from the own kinematics, P3 logged base from another row, P4 built-levers key missing, P5 a non-swapped sample changed, P6 gate does not compare declared and built, P7 add not refused by the swap, P8 add not refused by the trainer, P9 p 0 not refused by the swap, P10 unknown value not refused, P11 default flipped to logged, P12 `own` keeps the logged base, P13 probe still requires base to change, P14 no FLAG_CHECKS entry. **The brief's "logged row taken at r instead of r'" mutant is not applicable as worded:** the logged base is the batch row itself (t = r' by construction, FACTS 3), no row is looked up; P3 (base from another row) stands in for it.

## CENSUS: NOT RUN (blocker, named)
The census needs shard 0 of RH bank s0 from the pod. The pull (the same read-only `pscp` as FACTS_KINTB) was REFUSED by the session's permission classifier when I started it; I did not retry or work around it, and no copy of the shard or of `native_tracks_union.pt` exists on F: (searched four levels under `F:/TanitAD-scratch`). Nothing was pulled, so nothing was deleted for it. The script is ready and UNTESTED on real data: `census_kinput.py <shard.npz> <native_tracks_union.pt> <out.json> F:/TanitAD-scratch/kinlab/tree4 60000` (needs the shard's `kin3` member; reports the share of swapped records whose base changes and max / p50 / p95 of the 30-point position difference). What unblocks it: the MM (or the PI) allowing the pull, or placing the two files on F:.
On the test fixture the planted own kin3 (+0.8 m/s, +0.5 m/s2, +0.02 1/m) changes the base on swapped rows; that is not a census of real records.

## Not run
Census on real records (above); the whole suite (`pytest stack/`) and G-SUITE at the new commit; anything on the pod or GPU. `v8f_t1_eval.py`, `launch_gate.py`, `stack/tanitad/models/__init__.py`, `E:/Projects/TanitAD/stack` untouched.
