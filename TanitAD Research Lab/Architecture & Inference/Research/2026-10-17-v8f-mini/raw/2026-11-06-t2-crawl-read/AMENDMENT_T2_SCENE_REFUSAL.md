# AMENDMENT_T2_SCENE_REFUSAL: scenes the renderer refuses before any planner runs (Master Mind, 2026-11-06 ~19:27; dated amendment beside PREREG_T2_CRAWL.md, frozen `2c4c6557`, and PREREG_T2_DOMINANCE.md, frozen `38aa9df3`)

## 0. What was seen before this amendment was written (disclosure)
* **Stage 2, run 2** finished at 19:18 with `overall` INCOMPLETE: preflight PASS, gate PASS.
  * Deciding cells: 340 DONE, 10 skipped as DONE from run 1, 70 FAIL.
  * Beside cells: 136 DONE, 4 skipped as DONE, 28 FAIL.
  * The 98 failed cells are every cell of 7 launch scenes and no cell of any other scene: `dffd4c04`, `00169207`, `1aa48b8a`,
    `c67bdfef`, `90a3d349`, `aa094af8`, `42a44314`. All 14 cells of each failed (10 deciding, 4 beside), every arm, both conditions.
* **What the cell log says** (`rollouts_deciding.log` from its first 16:48 line): 60 lines read `actor placement REFUSED by its own
  falsifier`, and 10 tracebacks end in `KeyError: no timestamps_us_ranges for layer 'dynamic_rigids'`. Both fire inside
  `closedloop_drive.main` at `attach_actors_verified`, BEFORE the first planner forward.
* **Stage 3** (from 19:22): its first guard scene `829c55af` fails the same way, at its first cells.
* **Not seen:** no stage-2 row of any arm has been computed. The stage-2 scorer has not been run on run 2. No pooled contrast exists.

## 1. Why the frozen reads cannot decide as written
* `read_stage2.py` (pin `251bb71e`) requires 420 of 420 deciding cells DONE. The dominance read (pin `6345d4c7`) requires `overall`
  COMPLETE in every stage (G3).
* A relaunch skips DONE cells and re-runs the FAIL cells. The refusal is a deterministic function of the reconstruction and the logged
  frames, so a relaunch reproduces it and neither read can ever be satisfied.
* Neither card says what happens to a scene the renderer refuses. This amendment says it, before any row is computed.

## 2. The rule (R-SCN)
1. **A scene leaves a stage's deciding set if and only if both of these hold:**
   * every one of its cells in that stage is FAIL with 0 records: all arms, both conditions, deciding and beside;
   * each of those cells ended on a scene-level renderer refusal raised before any planner forward. Two refusals count:
     * **ACTOR_REFUSED:** `actor_map.attach_actors_verified` returned verdict REFUSED (fewer than 2 of its 3 falsifier frames pass);
     * **NO_DYNAMIC_LAYER:** `actor_map.layer_track_ranges` raised KeyError because the reconstruction carries no `dynamic_rigids`
       time ranges.
2. **Attribution is mechanical, and it is code:** `t2v8f/code/scene_exclusions.py`, pinned with the reads.
   * The stage's failed cells, in run order, are matched one to one with the terminal events of the cell log. A terminal event is a
     refusal line or a traceback's final exception line. The counts must be equal.
   * Every excluded scene carries exactly one reason, across all its cells.
   * An ACTOR_REFUSED scene's `n_tracks_json` must equal the track count of that scene's own `sequence_tracks.json`, sha-checked
     against the stage's scene sources.
   * Any failure that does not satisfy rule 1 leaves the read NOT-READ: the instrument is fixed first, as before.
3. **Included scenes must be complete.** Every cell of every scene that stays in must be DONE, or the read is NOT-READ. The deciding
   count required becomes 10 × (included launch scenes) for stage 2.
4. **Pairing.** A scene excluded in ANY stage is out of every contrast that would pair across stages. For the dominance read, C0C
   (stage 4) and P10 (stages 2 and 3) are read on the union of the included scenes only.
5. **Disclosure, every time.** Every report lists the excluded scenes per scene, with the reason, the cell counts, and the number of
   starts lost. The excluded scenes' starts are never re-drawn and never replaced.
6. **Nothing else changes.** The thresholds (h = 0.0267 and 0.0298; D1's −0.2415 m), the outcome order, the estimator, the bootstrap
   seed, and every other gate stand as frozen.

## 3. Why this cannot select on the outcome
* Both refusals are computed from the reconstruction and the logged frames, before the first planner forward. P10, ZERO, BLIND and C0
  are refused identically: every cell of each scene failed.
* The rule names its two reasons in advance. Any other failure blocks the read; it is never excluded.
* It is fixed before any row is computed, and the reads are re-pinned with it before they run on data.

## 4. What it costs, named
* The deciding scene set shrinks: by 7 of 42 launch scenes in stage 2, and by whatever stage 3 refuses among the 20 guard scenes.
* The bootstrap answers another draw of the REFUSAL-SURVIVING scenes. A scene whose replayed agents the renderer cannot place is, on
  that evidence, outside what T2 `objects` can test. The result is a result on the surviving scenes, never on the 62.
* The power loss falls under the cards' own UNDERPOWERED outcome. No new floor is added.

## 5. Pins (filled in a dated entry below, before any read runs on data)

## 6. Log (append-only; every entry dated)
* 2026-11-06 ~19:27, Master Mind: written and frozen. Its sha256 over the bytes above the line `## 5. Pins` is in `FROZEN_SHA_SCENE_REFUSAL.txt`.
* 2026-11-06 19:31, Master Mind: pins, before any stage-2 row of run 2 was computed and before the stage-2 scorer ran on run 2.
  * `t2v8f/code/scene_exclusions.py` sha256 `83a5f64716a02182a2a880bf8cec23c6e71589c98c841c6c4ebd1e609eceb373`.
  * `t2v8f/code/read_stage2.py` re-pinned at sha256 `f2136b39442f6df18819465baa37113385770f6ca940dd019acdf2d9bb945810`; its first pin was `251bb71e1e3502078361c70f22e8e5eb23f2bb96d68f08a6680f9c504fd22b32`. Without `--exclusions` it reads exactly as the first pin.
  * Tests `t2v8f/tests/test_scene_exclusions.py` (sha256 `f834281a9184ab74...`): 16 passed. The RED arms: an extra traceback in the log, a DONE cell in a refused scene, a track-count mismatch, an
    unnamed reason, a changed sources sha, and seven mutations of the read's inputs. Mutation runs: deleting any one of the builder's checks turns a test RED; the read has two
    independent guards (the problem list and the completeness count), and deleting both turns 7 of 16 RED.
  * The builder run on the stage-2 run-2 pull (`t2v8f/stage2/EXCLUSIONS2.json`), as the rule's first application: ATTRIBUTED, 7 scenes. 6 ACTOR_REFUSED (`dffd4c04`,
    `00169207`, `1aa48b8a`, `c67bdfef`, `aa094af8`, `42a44314`), each refusal's `n_tracks_json` equal to its own `sequence_tracks.json` (62, 78, 6, 245, 56, 161); 1
    NO_DYNAMIC_LAYER (`90a3d349`). 70 of 70 deciding and 28 of 28 beside failed cells matched to log events from lines 15539 and 6217, the first lines of run 2.
