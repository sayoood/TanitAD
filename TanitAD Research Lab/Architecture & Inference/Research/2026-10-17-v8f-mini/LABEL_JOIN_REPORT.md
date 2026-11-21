# Label-join report -- v9 per-frame labels onto the 0.2 s grid of the refav1 cache

All MEASURED; numbers in `raw/label_join_report.json`, `raw/join_check.json`, scripts in `code/`.

## The key
Cache step `j` = v2ep raw frame `k = 2j` = v9 release row `k` (`row_index(rel, sid, k)`, `sid = blake2b-8(clip_id) >> 1`). The release clock is `now_s = grid_start_s + k * dt_s` with `dt_s` ~ 0.1007 s per clip; the cache grid treats 2 frames as 0.2 s (nominal), as every refav1 number does.
A v2 MANIFEST row `i` is v2ep frame `i + 2` (bit-identical on eval139): the manifests hold no frames 0 and 1; the data module rebuilds frame-indexed arrays with two NaN rows (`full_from_manifest`).

## Files (found locally first)
| item | path | md5 / note |
|---|---|---|
| v9 train (4,369 clips) | `C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_train.npz` (also `D:/Projects/TanitAD-artifacts/v9labels/`) | `f63ece410b725febb8a5242cf2b01d3c` (the task's file) |
| v9 train (4,572 clips) | `C:/Users/Admin/qland/hold_v9train4572/work/release_train4572/v9_labels_train.npz` | `83e30bd3791f0767707eade5f4671759`; the 4,369 old clips bitwise identical |
| v9 eval139 | `C:/Users/Admin/qland/work/refcv8_hf_v9/release/v9_labels_eval139.npz` | `6b5c7f207cffc3b7eb3cd527fd433599` |
| lead (per-frame) | `D:/Projects/TanitAD-artifacts/v9labels/inputs/lead_{train,eval139}.npz`; `qland/hold_v9train4572/work/out/lead_train4572.npz` | input of the LEAD rule only |
| max-speed sidecar | `C:/Users/Admin/qland/landed_hold_datacov/raw/refcv6_speed_max_v8_{train,eval}.jsonl` (also `D:/refcv6_eval_kit/data/`) | train 4,572 rows (all valid), eval 147 rows (141 grid clips covered); **one value per clip** |
| poses/actions (train) | `C:/Users/Admin/tanitad-data/add203/in/train4572_v2manifest.pt` | used for the P10 refit; not on Thor-only |
| poses/actions (eval) | `raw/eval141_kin.npz` (from the local v2ep files; kin verified = banked `kin`, max abs diff 0.0, all 2,399 windows) | |

## Coverage
| | clips | windows (t in [3, T-31], 6 s complete) |
|---|---|---|
| refav1 train cache (= B1 - eval141) | 4,572 | 310,642 |
| with `f63ece41…` rows | 4,369 | 296,859 (95.56 %) |
| with `83e30bd3…` rows | 4,572 | 310,642 (100 %) |
| refav1 eval grid | 141 | 2,399 |
| with v9 eval139 rows | **139** | **2,365** |
| without | `38fe80c1-caf1-45ea-810b-2681ea488c6d`, `204b15b4-bd0a-4155-b01e-572c55197d0e` | 34 |

* The v9 eval release holds no episode outside the grid. The 2 unlabelled episodes have egomotion parquets, VLM records (`s2_labels_v8_eval.jsonl.gz`) and v2ep poses locally, but **no row in the clock sidecar**; `build_v9_labels.py build` falls back to `tanitad.data.clip_clock.pose_dt`, and that module is **missing from this checkout** (import error when tried; it exists in commit `82c2331a`). Lead fields additionally need the 3-D agent join. So the build is feasible in principle from local egomotion, not runnable here; nothing was invented.
* TRAIN/EVAL disjoint by clip id AND stable sid: 4,572 vs 141, overlap 0 (asserted at join time and in the test). Recording-session disjointness is enforced upstream by the S4 cache builder's parity gate (the clip id does not carry the session).
* RC-A valid rates (train, 4,572 release): 97.5 % / 96.7 % / 95.4 % at 30 / 50 / 80 m. Invalid RC rows are zero-filled and masked.
* Nav: 192,562 announced-turn frames; |dyaw| median 78.2 deg (p10 31.9, p90 109.3) -- the nav-only baseline's 90 deg turn is a fair nominal.
* Lead frames joined: 170,187 (4,572 release). Goal base rates are in the json (TURN_L 2.3 %, TURN_R 6.3 %, STOP_POINT 4.2 % of labelled eval steps; FOLLOW_LANE 75 %).

## 5 hand-checked windows (eval, `code/join_check.py`)
For each: (a) release `now_s` = clock `grid_start_s + k*dt_s` (abs diff 0.0); (b) `build_v9_labels.rc_fields` re-run from the ego-log parquet at that `now_s` reproduces the release's RC-A {30, 50, 80} x, y, psi (max abs diff **0.0**); (c) the ego-frame point of the unsmoothed human path (v2ep poses) at the same arc lies 0.03-2.31 m from RC-A (sigma 8 m smoothing); (d) `v_now_ms` vs pose speed. Whole grid (2,365 labelled windows): `v_now_ms - pose speed` mean 0.0050, p99 0.058, max 0.32 m/s. Values are pinned in `stack/tests/test_v8f_mini.py::test_five_hand_checked_real_windows`.

| clip | t | now_s | RC-A30 (x, y, psi deg) | RC-A50 | v_now |
|---|---|---|---|---|---|
| 10804db8 | 35 | 7.081525 | 29.9699, 0.3773, -0.748 | 49.9225, -0.2784, 0.1574 | 4.9823 |
| 480d7910 | 47 | 9.627721 | 30.0002, -0.0252, -0.0543 | 50.0002, -0.052, -0.1078 | 31.4377 |
| 91929cea | 59 | 11.994035 | 29.9743, -1.118, -3.4838 | 49.9086, -2.6974, -5.4043 | 7.6617 |
| b99058dc | 23 | 4.808529 | 29.9978, -0.3473, -0.9017 | 49.995, -0.5664, 0.1245 | 8.17 |
| f1a94fee | 39 | 7.977197 | 29.8471, 2.7474, 7.1904 | 49.7253, 4.8528, 4.6743 | 3.4963 |
