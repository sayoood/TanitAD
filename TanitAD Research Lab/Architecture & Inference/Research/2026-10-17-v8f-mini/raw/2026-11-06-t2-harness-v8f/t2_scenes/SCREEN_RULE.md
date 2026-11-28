# T2 scene-screening rule (FROZEN)

Frozen before any scene log was read. Source: `BRIEF_T2_SCENES.md` lines 39-54, copied verbatim below the line. Only the
section "Concrete bindings" is added; it names fields, the repo id and the revision. It changes no threshold and no order.

---

## Rule (verbatim from the brief, step 1)

* **Universe.**
  * Every scene of the NuRec release, at the revision pinned in step 1.
  * Minus every scene whose PhysicalAI clip id is in v8F-mini's TRAINING set: the 4,512 clips. Find the train clip list that built
    box 2's `cache_full` (v9 release 4,572 clips minus the `--exclude 540:600` holdout); cite where you found it.
  * Minus scenes the renderer cannot render (missing members). Record the exclusion counts.
* **Order.** Sort the remaining scene ids, then apply `numpy.random.default_rng(0).permutation`. Screen in that order.
* **Speed.** v(t) is the logged ego speed from the scene's own ego track: path length over a centred 1.0 s window.
* **Launch event (an M1-bearing scene).** An upward crossing of 5.2 m/s by v(t) at time t_c, with three conditions:
  * v ≤ 3.0 m/s at some time in [t_c − 5 s, t_c);
  * t_c ≥ the first renderable time + 2.0 s;
  * t_c ≤ the last renderable time − 6.0 s.
* **Lead stratum (recorded, never used to select).** At the last time with v ≤ 3.0 m/s before t_c, is there a vehicle-class agent
  0 < x ≤ 30 m ahead with |y| ≤ 2.0 m in the ego frame?
* **Guard scene.** No launch event, and v ≥ 5.2 m/s on at least 50 % of the renderable span.
* **Quotas.** The first 42 launch scenes and the first 20 guard scenes in permutation order. Stop screening when both are met, or after
  800 scenes. If 800 are screened without meeting a quota, report the counts: do not loosen the rule.

---

## Concrete bindings (added at freeze; no threshold, count, order or quota above is altered)

**Release.**
* Repo `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec` (dataset, gated "auto"; this account's access was verified before the freeze by a HEAD on a scene file: 302, not 403).
* Revision: branch `26.04`, commit `5fe7a149b40ee83a2d3e09f7fe5dc2f217cabe11`. Release directory `sample_set/26.04_release/<scene>/`, 1,607 scenes. Source for the pin: `ALPASIM_STATE.md` ("rev 26.04", "scene release must be 26.04 to match the NRE 26.04 image"). `main` has since moved to 26.10 (commit `9a27fbd9...`) and is NOT used.
* Scene files per scene: `<scene>.usdz` (a zip, members STORED) and `camera_front_wide_120fov.mp4`. Listing: `raw/RELEASE_LISTING.json`.
* Scene id equals the PhysicalAI clip id (check: 16 of the 4,572 train clips are NuRec scene ids, the documented overlap of 16).

**Universe, in this order.**
1. All 1,607 scene ids of the pinned release.
2. Minus the ids in v8F-mini's TRAINING set: box 2 `D:\tanitad\cache_full\train\manifest.jsonl` (4,572 clips, copied to `raw/box2_cache_full_train_manifest.jsonl`), minus `--exclude 540:600` of the sha12 order (`sha12 = sha256(clip)[:12]`), giving 4,512 clips. Source of the recipe: `Project Steering/PREREG_V8F_MINI.md` Amendment 4. The reconstruction must reproduce `train_sha256 37f4f9ee468e0872134074a82d63c2915458596fdbcf3d0fcf5b09f1fe0d8cc1` (newline-joined sorted list) and the holdout sha `f6a1dee02af0b96f...`.
3. Minus scenes the renderer cannot render: a scene is renderable only if its usdz holds all of `volume.nurec`, `rig_trajectories.json` and `sequence_tracks.json`. This step reads ONLY each usdz's zip central directory (member names, sizes, CRCs, offsets), never a member's content. It is done for every scene remaining after (2), BEFORE the permutation is drawn, so the permutation is over the final universe.

**Order.** `ids = sorted(universe)` (Python string order); `perm = numpy.random.default_rng(0).permutation(len(ids))`; screening order = `[ids[i] for i in perm]`. The numpy version and a sha256 of the order are recorded in `raw/SCREEN_TABLE.json`.

**Tracks read in screening (nothing else: no model, no pixels, no map, no volume).**
* Ego track and time span: member `rig_trajectories.json`; entry `rig_trajectories[0]`; key `K = "camera_front_wide_120fov@" + sequence_id`; poses `cameras_frame_T_rig_worlds[K][f][1]` (4x4, rig to world; position = `[:2, 3]`); times `cameras_frame_timestamps_us[K][f][1]`. 10 Hz ticks are every `stride`-th camera frame, `stride = round(1e5 / (ts[1] - ts[0]))`. The code is the `Ego` class of `stack/experiments/alpasim-gsplat/scene_geometry.py`.
* Agent tracks: member `sequence_tracks.json`; `d[first key]["tracks_data"]` with `tracks_id`, `tracks_label_class`, `tracks_poses` (x, y, z, qx, qy, qz, qw; same world frame as the rig), `tracks_timestamps_us`. The code is the `Tracks` class of the same file.

**Definitions.**
* Renderable span: from the first to the last 10 Hz tick of the front-wide camera (`t_first`, `t_last`); span length in ticks N.
* `v(t_k)`: path length of the ego xy polyline over the 10 Hz ticks `k-5 ... k+5` (ten intervals), divided by 1.0 s. Defined only when all eleven ticks exist; undefined elsewhere (those ticks are never an event, never a "v <= 3.0" tick, and count as "not >= 5.2" in the guard share).
* Upward crossing at tick `k`: `v(t_{k-1}) < 5.2 <= v(t_k)`, both defined; `t_c = t_k`. A scene may have several crossings; each is tested against the three conditions; every passing crossing is a launch event. A scene is a launch scene if it has at least one launch event.
* Condition `v <= 3.0` in `[t_c - 5 s, t_c)`: some tick `t` with `t_c - 5.0 <= t < t_c` and defined `v(t) <= 3.0`.
* Lead stratum, per launch event: `t_s` = the last tick before `t_c` with defined `v <= 3.0`. Ego frame = full 4x4 inverse of the ego pose at `t_s` (x forward, y left). A track counts at `t_s` if it has an annotation within 0.15 s of `t_s` (as `Tracks.at`, `tol_us = 1.5e5`); its position is the pose centre `tracks_poses[:3]`. Vehicle classes = `automobile`, `heavy_truck`, `bus`, `other_vehicle`, `trailer`. Lead present iff some vehicle track has `0 < x <= 30` and `|y| <= 2.0`. A launch scene's stratum is that of its earliest launch event; all events are listed. The stratum is never used to select.
* Guard share: (ticks with defined `v >= 5.2`) / N, over all N ticks of the span. A guard scene has zero launch events and share >= 0.50.
* Quota bookkeeping: walk the order; a launch scene takes the next launch slot if fewer than 42 are filled, a guard scene takes the next guard slot if fewer than 20 are filled; any other scene is screened and not selected. Stop at the first scene after which both quotas are full, or after 800 scenes screened. A scene whose members cannot be parsed is screened, recorded with its reason, and not selected.
* Selected scenes' renderer members to download: `volume.nurec`, `rig_trajectories.json`, `sequence_tracks.json` (the three members of step 3 of the universe). Not downloaded: the reference mp4, `checkpoint.ckpt`, meshes, `clipgt/`, frames.
