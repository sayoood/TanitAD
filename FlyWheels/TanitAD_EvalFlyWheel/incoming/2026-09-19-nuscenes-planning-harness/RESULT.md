# RESULT — W6 / E5: nuScenes open-loop planning harness + input adapter

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/`
**Briefs** `BRIEF.md` (E5) + `…/2026-09-19-eval-suite-build/BUILD_PLAN.md` row **W6** and §1.
**Dates** 2026-09-19 → 2026-09-20. **CPU only** (A8 training untouched). **No nuScenes byte was
downloaded; nothing was registered and no terms were accepted** — those are human acts.

> ⛔ **The number this harness produces is NOT a TanitAD criterion and cannot become one.**
> `H-EVAL-6` SUPPORTED · `D-BENCH-PORT` SKIP claim-bearing. Every artifact carries
> `claim_bearing: false`; the API has no parameter that sets it true, and W1's schema rejects a
> `nuscenes_ol` run that claims anything.

---

## PI ACTION LIST — nuScenes

The PI answered *"Yes, I'll register"*. Everything below is what only a human may do, then exactly
what to download, where it lands, and the one command that fetches it.

### 1. The three human-only steps

| # | step | link | note |
|---|---|---|---|
| 1 | Create a free account | `https://www.nuscenes.org/sign-up` | account creation is outside an agent's boundary |
| 2 | **Read and accept the Terms of Use** | `https://www.nuscenes.org/terms-of-use` | ⭐ the operative instrument. It carries documented **modifications on top of CC BY-NC-SA 4.0**, and the page returns EMPTY to every automated fetch (4 attempts on 2026-07-26, re-confirmed 2026-09-19) — so it has to be read by a person |
| 3 | Open the Download page and start the files in §2 | `https://www.nuscenes.org/nuscenes#download` | "Full dataset (v1.0)" → **Trainval: Metadata** + **Keyframe blobs parts 1–10**; **Map expansion (v1.3)**. ⛔ **Do NOT take the CAN bus expansion** — see §3 |

**Licence, stated once so it travels with the data** (PUBLISHED — arXiv 1903.11027, verbatim):
*"The nuScenes data is published under CC BY-NC-SA 4.0 license, which means that anyone can use
this dataset for non-commercial research purposes."* ⇒ research-only, **share-alike, and
derivatives inherit NC+SA**: anything built on it routes to the segregated copyleft shard and
**never** enters TanitDataSet-C (`stack/tanitad/data/nuscenes.py:12-40`). ⚠️ The AWS Open Data
registry lists the licence field as *"Commercial"* — a **CONFLICT** already recorded in
`CRITERIA_REGISTRY.json → benchmarks.nuscenes.licence_status`; treat as research-only until the PI
holds written terms.

### 2. The minimal file set (sizes MEASURED 2026-09-19 by bucket LIST + HEAD — `raw/bucket_listing.md`)

| tier | what | objects | bytes | GB |
|---|---|---|---:|---:|
| **A — start here** | the whole value question, and everything metadata-only | `v1.0-trainval_meta.tgz` | 461,678,030 | **0.46** |
| **B — the planning eval** | CAM_FRONT + LIDAR_TOP **keyframes** of all 850 trainval scenes (the 150 val scenes are spread across the parts) | `v1.0-trainval{01..10}_keyframes.tgz` | 44,902,690,772 | **44.90** |
| **C — SAM3 paint reference** (`H-SAM3-FUSION-1`) | the vector map layers (lane divider, road divider, ped crossing, stop line, drivable area, walkway) | `nuScenes-map-expansion-v1.3.zip` | 398,535,531 | **0.40** |
| **D — optional pilot** | 10 scenes with *all* sensors and sweeps: end-to-end smoke of the harness (mini_val = scene-0103, scene-0916) and a small SAM3 pilot | `v1.0-mini.tgz` | 4,168,148,189 | **4.17** |
| — | **A + B + C (one download serves the planning eval AND the SAM3 reference)** | | **45,762,904,333** | **45.76** |
| ⚠️ E — only if a 10 Hz-history arm is scored | camera **sweeps** (12 Hz) instead of keyframes | `…_blobs_camera.tgz` × 10 | 177,288,110,316 | 177.29 |
| ⛔ not needed | CAN bus expansion | `can_bus.zip` | 780,974,697 | 0.78 |

* **Overlap:** A + B are shared by both jobs; the SAM3 paint reference adds only **C (0.40 GB)** and
  reads the **LIDAR_TOP keyframes that are already inside B**. Nothing else is duplicated.
* ⛔ **v1.2 of the map expansion is useless**: the current devkit raises on any map older than v1.3
  (`map_api.py:99-101`). The acquisition message inside `stack/tanitad/data/nuscenes.py` still names
  v1.2 — an integration ask below.
* ⚠️ **Which keyframe part holds which val scene is not derivable** from the listing or the metadata
  (no part id in `sample_data.filename`), so the conservative ask is all ten parts. After the first
  part lands, `tar -tzf` settles it and the rest may be skippable.
* ⚠️ **Tier E is a REAL decision, not a detail.** Our models read a history at **0.1 s** spacing
  (`window` rows of a 3-frame D-015 stack — `taniteval/taniteval/bench/navsim/bridge.py:52-53`),
  while nuScenes keyframes are **0.5 s** apart. With tier B the harness can only feed the E2
  constructions `ST` (static history) or `NT` (nearest-in-time, ≤ 0.25 s error, declared); it
  **REFUSES** the `EXACT` construction with a message naming the sweeps. Choosing B means the
  nuScenes row is an ST/NT row and says so.

### 3. Why the CAN bus is deliberately excluded

It is the **ego-status channel** (50 Hz pose+velocity+acceleration, steering, wheel speeds) — the
very shortcut that makes the benchmark unreadable: AD-MLP with **no camera and no LiDAR** scores
L2 avg **0.29** against UniAD's 1.03, and ego status alone is worth **55–70 %** of L2 (BEV-Planner
Table 1). Our harness needs none of it: the GT trajectory comes from `ego_pose` in the metadata,
and the one ego scalar the PI ruled admissible (`v0` at t0) is a backward difference of those same
poses. If a *label* study ever needs CAN, that is a separate, declared download.

### 4. Where it lands (D:, as the PI ruled for NavSim v1)

```
D:/Archive/devbox-C/nuscenes/
    archives/      <- the .tgz / .zip stay PACKED here (11 files; the harness never reads them)
    data/          <- the devkit layout the harness reads (TANITAD_NUSCENES_ROOT)
        v1.0-trainval/*.json          13 tables from the metadata archive
        samples/CAM_FRONT/*.jpg       only the val keyframes we extract
        samples/LIDAR_TOP/*.pcd.bin   only if the SAM3 reference needs LiDAR
        maps/{expansion,basemap,prediction}/…
```

⚠️ **D: is exFAT with 1 MiB clusters** (MEASURED: `AllocationUnitSize 1048576`, 1,310.7 GB free;
C: is NTFS at 4 KiB with 164.5 GB free). Every extracted file occupies **at least 1 MiB**, so
extracting all ~410 k keyframes would allocate **≈ 430 GB for 45 GB of data**. Therefore:
**keep the archives packed and extract only what is needed** —

```
python -m adapters.nuscenes_planning extract-list --nuscenes-root D:/Archive/devbox-C/nuscenes/data \
       --split val --channels CAM_FRONT --out D:/Archive/devbox-C/nuscenes/val_cam_front.txt
tar -xzf archives/v1.0-trainval01_keyframes.tgz -C data -T val_cam_front.txt     # per part
```
≈ 6,019 CAM_FRONT val keyframes ≈ **6 GB allocated** on D: (≈ 1 GB of real bytes). ⚠️ A7 trains
overnight and reads D: — run the download when that is idle, or accept the I/O contention.

### 5. The one command, after acceptance

`code/fetch_nuscenes_after_tou.sh` — it **REFUSES to run** without
`--i-accepted-the-nuscenes-terms-of-use "<name>"`, verifies every file against the MEASURED byte
count, pulls the bucket's own `md5.checksum`, and writes a receipt naming who accepted and when.

```
bash code/fetch_nuscenes_after_tou.sh --i-accepted-the-nuscenes-terms-of-use "Sayed" planning
bash code/fetch_nuscenes_after_tou.sh --i-accepted-the-nuscenes-terms-of-use "Sayed" maps
```
⛔ **No agent runs it.** It was never executed here.

### 6. What happens the moment the bytes exist

```
setx TANITAD_NUSCENES_ROOT D:/Archive/devbox-C/nuscenes/data
set  TANITAD_NUSCENES_PROTOCOL=nuScenes_OL_L2_uniad
python -m taniteval.bench nuscenes_ol --ckpt none --split val --device cpu      # GT + STOP + CV
set  TANITAD_NUSCENES_PROTOCOL=nuScenes_OL_L2_stp3
python -m taniteval.bench nuscenes_ol --ckpt none --split val --device cpu      # the other convention
```
Metadata alone (tier A, 0.46 GB) already produces: both conventions' **GT-collision floor**
(the number that decides whether any collision column is quotable at all), the STOP and CV floors,
the sample counts (which must read **6,019 / 5,119 / 4,819** — see §"first contact" below), and the
VAD category-index audit. Images are needed only for a model arm.

---

## Headline — what now works

1. **`taniteval/adapters/nuscenes_planning.py`** (new, ~1,900 lines): L2 and collision @ 1/2/3 s in
   **both** published conventions, verbatim to the pinned reference code
   (ST-P3 `@69aabef`, UniAD `@609ee08`, VAD `@1688c4b`, AD-MLP `@4b93ba0`), with the sample sets,
   the GT, the occupancy builders, a port of `cv2.fillPoly`, the trivial arms, the four-family
   supplements and a §1 run-directory writer. Every number carries its `(reduction, pipeline)` tag;
   an untagged, string-tagged, mis-piped or mixed number raises.
2. **`taniteval/taniteval/bench/plugins/nuscenes_ol.py`** (W6's exclusive slot in W1's suite):
   `python -m taniteval.bench nuscenes_ol …` now runs, **one convention per run**, refusing when the
   operator did not choose one. Both run directories validate against W1's schemas.
3. **The input adapter**: nuScenes CAM_FRONT (a rectified pinhole) → the cylindrical training frame
   by RAY RESAMPLE (`calib.pinhole_rectify`), **never a resize**, with the observed mask stamped on
   every model arm: **70,737 / 163,840 px = 0.4317** in `PHYSICALAI_WIDE120_256x640`
   (columns 145–488), **0.5500** in the rig-clean 176×624 — a ~64.6° camera inside a 120° frame.
4. **Tests**: `taniteval/tests/test_nuscenes_planning.py` — **57 passed, 1 skipped (NOT RUN: the
   cv2-parity arm; no OpenCV in the venv), 0 failed** (`pytest -q`, CPU, 8.8 s), including two
   source-level **mutation arms that must go RED** and the PARA-Drive cross-check.
5. **Integration asks** (they are asks, not edits — see `COMMS.md`): W2 takes
   `code/criteria_guard.patch` (`git apply --check` clean; patched suite 171 passed / 4
   skipped, and the four guard tests FAIL against the unpatched checker); W4 takes
   `code/published_results_nuscenes.patch.json` — **39 published rows in W4's own row
   schema**, on W4's two protocol keys, with the three existing `implementation` strings
   reproduced byte-for-byte so the rows JOIN rather than forming a second table, 0 id
   collisions, every cited PDF re-hashed against `library.json` (9/9) and every
   `page_tokens` string re-found on its cited page — ⚠️ **with pypdf, the same extractor that
   CHOSE the tokens, so it is a transcription check and not an independent re-read** (F12);
   W4's PyMuPDF verifier is the admissible one and is still outstanding on these rows. W1 is
   asked for two CLI flags; the DataFlyWheel/loader owner for one stale line.

## Findings, each with its evidence class

| # | finding | class |
|---|---|---|
| F1 | **The two conventions are one line apart, and UniAD ships both.** `nuscenes_e2e_dataset.py:1041-1044` is the switch: `value[:i+1].mean()` (stp3) vs `value[i]` (uniad). At UniAD's FIRST public commit (`4e91222`, 2023-03-29) only `value[i]` existed; the `stp3` option arrived 2024-03-19 (`33cd8ec`, *"Add option for legacy planning metric definition"*). ⇒ `NUSCENES_PROTOCOL.md` §10 item 6 (*"UNVERIFIED whether the switch existed at publication"*) is **RESOLVED: it did not**; UniAD's published table is the at-t convention. | MEASURED-from-source (pinned, blob-verified) |
| F2 | **The reduction applies to collision too, and PARA-Drive's own table proves it.** Feeding PARA-Drive's at-t vector (L2 `0.2788…1.9821`; Col `0.10…1.11`) through our reducers reproduces its VAD-protocol row to 4 dp (L2 `0.4084 / 0.6980 / 1.0511`, Ave `0.7192`) and 2 dp (Col `0.07 / 0.17 / 0.40`, Ave `0.21`). An independently authored cross-check, not our own arithmetic replayed. | PUBLISHED (banked `paradrive-cvpr2024`, Table 1) + MEASURED |
| F3 | **VAD's point-collision check reads the wrong cell.** `metric_stp3.py:272-273` computes `(-bx[0]/2 − y)/dx` and `(-bx[1]/2 + x)/dx` = `49.75 − 2y`, `49.75 + 2x`, where the grid centre is 100 — a waypoint at (0, 10 m) queries cell **(29, 49)**, ~25 m from the ego. Reproduced verbatim and pinned by a literal test. It does **not** touch VAD's published column (`obj_box_col`, the box check), but any use of `plan_obj_col` from that codebase is measuring a different cell. | MEASURED-from-source |
| F4 | **The three pipelines disagree about which agents exist.** UniAD: 7 vehicle classes, pedestrians EXCLUDED, invisible and 0-lidar-point boxes INCLUDED, invalid future frames filled with 255. VAD: vehicles **and** pedestrians, but only boxes with ≥ 1 lidar point whose centre is inside ±15 m lateral / ±30 m longitudinal (`CustomObjectRangeFilter`, `point_cloud_range`). ST-P3: `vehicle.*` + `human.*` with visibility bin 1 dropped. Same scene, three different "other traffic". | MEASURED-from-source |
| F5 | **The published sample counts fall straight out of the three rules**: 6,019 − 6·150 = **5,119** (BEV-Planner's own valid count) and 6,019 − 8·150 = **4,819** (AD-MLP: *"all 4819 ones"*). The harness asserts these on first contact, so a wrong sample set cannot hide. | PUBLISHED + derived |
| F6 | **Our camera sees 43 % of our own frame on nuScenes.** CAM_FRONT's nominal matrix gives HFOV **64.56°** (the paper's nominal spec says 70°) against the training frame's 120°: 70,737 of 163,840 pixels observed, columns 145–488, rows 9–225 at the centre column. A nuScenes row from a 120°-trained model is a **partially-blind** row and must say so. | MEASURED (closed form, independently reproduced by `calib.pinhole_rectify_grid`) |
| F7 | **The GT control is the only way a collision number is readable, and in-protocol it is 0 by construction.** Every reference EXCLUDES steps where the GT box collides, so the GT arm scores 0; the floor is the RAW GT rate, which PARA-Drive measured at **0.36 %** (UniAD protocol) and **0.96 %** (VAD protocol) — larger than the spread between methods. The harness emits it per protocol with that warning attached. | PUBLISHED + MEASURED (harness) |
| F8 | **`stack/tanitad/data/nuscenes.py`'s acquisition message is stale on the map version**: it tells a human to fetch `nuScenes-map-expansion-v1.2.zip` (16 MiB), which the current devkit **refuses** (`map_api.py:99-101`: *"You are using an outdated map version"*). The live object is v1.3, 398,535,531 B. | MEASURED-from-source + bucket LIST |
| F9 | **A sub-pixel and a quantisation observation about `calib.py`** (shared code, not ours to edit): `pinhole_rectify_grid` normalises with `u/(w−1)·2−1` but samples with `align_corners=False`, a ≤ 0.5 native-pixel scale effect at the image edge (≈ 0.11 px at canonical scale, zero at the centre); and `pinhole_rectify` quantises with `.to(torch.uint8)` — truncation — so a constant 200 image comes back with some 199s. Both are shared with the PhysicalAI training frames, so they do not bias eval-vs-train; the NavSim frame builder (E2/W1) uses `align_corners=True`, so the two external-benchmark input paths differ by that ≤ 0.5 px. | MEASURED (pinned by a test) |
| F10 | **VAD's category sets are literal indices, not names** (`{2..8}` human, `{14..23}` vehicle). Under the lidarseg-ordered 32-entry `category.json` those are exactly the human and vehicle categories; index 23 exists only in that ordering. Because the mapping is ORDER-dependent and we have no metadata yet, the harness ships `vad_category_index_audit()` and prints it on first contact. | MEASURED-from-source + INFERRED (audit on arrival) |
| F11 | **For nuScenes the "harness version" IS the method — there is no devkit metric, and the codebases disagree by more than the methods do.** Four mutually incompatible collision implementations are in print (ST-P3/UniAD/VAD: 0.5 m grid, axis-aligned box · BEV-Planner: 0.1 m union, estimated yaw · SparseDrive: box-vs-box polygons · PARA-Drive: oriented box, first frame removed), and **re-scoring the SAME two checkpoints under the two legacy harnesses FLIPS the ranking** — UniAD 1.03 < VAD 1.07 at-t, VAD 0.72 < UniAD 0.76 averaged (`paradrive-cvpr2024` Tab. 8). The 39 rows handed to W4 therefore carry a reduction pinned to a COMMIT and a named collision implementation, never just a table number; the harness version is what moves them. | PUBLISHED + MEASURED-from-source |
| F12 | ⛔ **RETRACTED — my own "independent" cross-check was a consistency check, and the phrasing overstated it.** I reported *"15 of my rows share a (paper, table) with rows W4 extracted, and the page agrees 15/15, 0 disagreements"* and offered it as the independent control my pypdf token check could not be. W4 could not reproduce it; **re-derived here, W4 is right**: 15 of my rows sit on a paper W4 also cites, but only **5** share an exact `(library_key, table)` — my join key stripped the sub-table parenthetical (`'Tab. 1 (ID-3)'` → `'Tab. 1'`), which merged distinct sub-tables and inflated 5 → 15 — and **0** share a `(paper, page, system)` cell, **0** carry a value-set W4 also published. ⇒ **no number in the 39 rows has been read twice by two readers.** What agreed is *which page a table sits on*, which is fixed by the paper's layout and constrains nothing about a cell's value. ⚠️ **A third trap found while re-deriving it:** W4 has since MERGED the rows, so the reference file now contains them — run unfiltered, the same script reads **39/39** and looks perfect, because it is comparing my rows with themselves. ⭐ Root-cause class: *a cross-check must be derived independently of the value it checks* (`CLAUDE.md`) — here the check was run by another agent with another library but on a DIFFERENT QUANTITY than the one at risk, which is the `df`/`step_s` scope family: the measurement was true, the claim hung on it was not. Instrument: `code/recount_row_overlap.py` (prints all four keys and both the filtered and unfiltered counts). | MEASURED (re-derived; supersedes the earlier claim) |

## What is verbatim, and what is not (named, not hidden)

**Verbatim** (ported line-by-line from the pinned sources, with the file:line in the docstring):
the three kernels (box + point collision, the GT-exclusion rule, the x flips, VAD's row flip and
its point-check offset), both reductions, the masked-L2, the three sample sets, the GT trajectory,
UniAD's and VAD's occupancy builders, the ±2 m command rule (as a LABEL only).

**Not verbatim, and why:**
1. **ST-P3's occupancy builder is NOT implemented** (its per-frame labels are warped into t0 with
   `grid_sample`). The ST-P3 *pipeline* therefore scores L2 only and returns
   `collision: {status: UNAVAILABLE, reason, n}`. The two shipped protocols do not use it.
2. **`cv2.fillPoly` is a PORT** (OpenCV 4.5.4 `drawing.cpp`: `clipLine`, `LineIterator`,
   `CollectPolyEdges`, `FillEdgeCollection`, all pinned). It reproduces three literal masks, but
   **parity against a real `cv2` is NOT RUN** — the venv has no OpenCV and installing one is out of
   scope. The parity test exists and skips with that reason.
3. **Accumulation is float64** where the references accumulate float32 (differences far below the
   published 2-dp precision).
4. **Box-library yaw conventions inside mmdet3d are not replicated**: the builders rasterise the
   physical footprint (centre, size, heading). A reference whose box library mis-orients a box would
   differ — UNVERIFIED until a parity run against the reference label pipelines on real data.
5. **The model arm is plumbed but never run end-to-end** (no images exist here): frames →
   `pinhole_rectify` → the trainer's own `stack_frames` → `forward` → plan → LiDAR frame is tested
   with a FAKE model; `refc_forward` (W1's promoted E2 bridge) is untested against a checkpoint.

## First contact with real data — the checks that run before any number is quoted

1. sample counts **6,019 / 5,119 / 4,819** (`expected_sample_counts`);
2. `vad_category_index_audit()` — do VAD's literal index sets really select human/vehicle here;
3. the **GT-collision floor** per protocol (if it is not ≈ 0.36 % / 0.96 %, our occupancy differs
   from the references');
4. the non-straight subset size against PARA-Drive's **686** val key-frames;
5. the per-sample geometry stamp (per-sample intrinsics, not the nominal matrix).

## Registry / doctrine compliance

`claim_bearing:false` + `H-EVAL-6` reason on every artifact · registry keys
`benchmark.nuscenes.planning.{protocol_tag, gt_control, command_source, nonstraight}` all emitted ·
the GT-derived command is **refused by the API** as an input (`command_source ∈ {none,
predicted_goal}`) · four families reported per family with reason and n (longitudinal/lateral/
tactical PARTIAL with the metrics that ARE computable, strategic UNAVAILABLE) · interval
`UNAVAILABLE` with its reason (no pre-registered cluster unit for nuScenes) ·
`overlapping_holdout_se` named only to disavow it · tier `OPEN-LOOP-L2`, loop `OPEN`.

## Deliverable manifest — every artifact, and where it lives

⚠️ Everything is in the repo working tree on **D:** and **staged**; ⛔ nothing is committed or
pushed (agents stage, the orchestrator commits). Staging is verified by BLOB COMPARISON, not by
`git add`'s exit code: `git ls-files --stage <p>` against `git hash-object <p>`, with both sides
asserted to be 40 characters first — an empty string on each side compares EQUAL and would
otherwise print a success (`CLAUDE.md`, 2026-09-04). The end-of-turn run of that check is
reported to the orchestrator with this package.

| path (repo-relative, D:\Projects\TanitAD) | what | bytes |
|---|---|---|
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/RESULT.md` | this file — the PI action list, the headline, F1–F11, the manifest | 17,937 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/SPEC.md` | the pre-registration: literal expected values, committed before the code | 11,358 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/PLAN.md` | the plan and its priority order | 2,795 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/BRIEF.md` | the brief as received | 6,903 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/COMMS.md` | 7 integration asks, each with an owner and what breaks if unread | 9,138 |
| `taniteval/adapters/nuscenes_planning.py` | THE HARNESS: both conventions, the input adapter, the arms, the run writer | 106,689 |
| `taniteval/tests/test_nuscenes_planning.py` | 57 passed / 1 skipped (NOT RUN: cv2 parity) / 0 failed — incl. 2 mutation arms | 44,086 |
| `taniteval/taniteval/bench/plugins/nuscenes_ol.py` | W6's exclusive slot in W1's suite: `run_benchmark(ctx)`, one convention per run | 6,108 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/code/criteria_guard.patch` | FOR W2 — EXTERNAL_ONLY scope + registry gate + 5 tests (2 misuse arms) | 16,264 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/code/published_results_nuscenes.patch.json` | FOR W4 — 39 published rows in W4's row schema | 43,921 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/code/make_w4_rows.py` | the builder of those rows: re-hashes the PDFs, re-finds every token, refuses on a miss, carries a mutation arm | 27,937 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/code/recount_row_overlap.py` | the F12 instrument: how much of the patch W4 actually re-read (4 join keys, filtered and unfiltered) | 4,265 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/code/fetch_nuscenes_after_tou.sh` | FOR THE PI — refuses to run without the Terms acknowledgement; NEVER run by an agent | 5,482 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/nuscenes_external_rows.md` | the published rows in prose (the human-readable twin of the W4 patch) | 11,718 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/bucket_listing.md` | the download sizing: bucket LIST + HEAD only, 2026-09-19 | 2,698 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/bucket_listing_2026-09-19.json` | the raw listing behind it | 19,988 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/bucket_head_crosscheck_2026-09-19.txt` | HEAD re-check of every object quoted in the PI action list | 1,525 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/devkit_splits_b40adc4.py.txt` | the devkit split file (blob 6988b5c8), the source of the 150-scene val set | 20,272 |
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/reference_extracts/` | 33 pinned reference extracts + INDEX.json — every port's source, blob-verified MATCH | 34 files, 164,776 |

**Nothing is stranded**: there is no pod, no worktree and no scratch copy holding a W6
deliverable. The only W6 artifacts NOT in the repo are the ones that cannot exist yet — the two
run directories, which need nuScenes bytes the PI has not yet downloaded.
