# HANDOFF E1 + E2 (T2 re-perception rollout, v8F-mini) -- 2026-11-06, tip 24b97ba9

Evidence class: MEASURED (ours; `logs/pytest.log`, `logs/mutations.log`, `raw/G2_result.json`; hashes in `PINS.txt`). CPU only, no network, nothing committed or staged.

## Built (`code/`)
* `t2a_v8f_policy.py` `V8FMiniT2Policy`: kit interface (`window`=17, `set_t0`, `set_history`, `plan`), plus `seed_track`, `set_nav`, `push_tokens`, `build_batch`, `forward_batch`.
  Items follow the T1-kin tool: the 10 Hz track (logged rows <= NOW, then executed rows) -> `D.yaw_kin_from_track` / `fk.kin_features` / `D.kdx_base`; nav dead-reckoned from own arc; causal bucket (21 rows); ages = the tool's `ages_for`.
  Arms: `model`, `p10` (same forward's `out["p10"]`). HACC via `hacc_weights=` (acts inside `forward_infer`). Guard: batch keys must equal MODEL_KEYS, keys read must lie in `INFERENCE_INPUT_KEYS` (`KeyGuardError`).
* `t2a_v8f_encoder.py` `InLoopEncoder`: f-theta frame + per-clip intrinsics -> 256x640 cylinder -> bf16 DINOv3-L -> drop 5 tokens -> 4x4 fp32 pool -> fp8. Weights: offline HF snapshot `ea8dc286` (found on C:, `local_files_only`); G: paths refused.
* Canon route (answers the sizing's U): box 2's `cache_full` and the eval-141 cache were built by `pull_encode_pooled.py --source local --src-geom 256`, bf16, GPU (`ROUTE.md`; cache `meta.json`: `source=local256`). Not thor416.

## Gates (`tests/test_t2v8f_gates.py`, 6 passed; `code/run_mutations.py`: MUTATIONS ALL RED)
| gate | result | mutation (all RED) |
|---|---|---|
| G1 tick 0, 6 eval-141 windows, C0 s0, banked tokens | all 12 model keys bit-identical to the reindexed tool's batch; plan bit-identical (float32) | stale history, tokens off by one step, yaw-rate feature zeroed, causal bucket dropped |
| G1 ticks 1-2 (m=10, 3 windows), executed track | batch and plan bit-identical | nav held at t0 |
| G1 HACC | same table, same plan as the model's own post-map; moves the plan | post-map no effect |
| G2 (clip 01be5919, step 40, CPU bf16, 1 frame) | rel-L2 **0.0264**, cosine **0.99965** vs banked; fp32 CPU 0.0323 / 0.99948; sizing residue 2.5-2.8 %. Stage A: mp4 frame -> in-loop rectify equals the cache's PNG in 100 % of values (max 0) | wrong mean/std (0.184), 1 special token (grid shape), neighbour frame (0.200), transposed grid (0.861) |
| G3 zero control | P10 arm == the tool's floor prior, max abs diff exactly 0.0, 4 windows; model arm differs (> 1e-3) | returns model plan; prior + 1e-6 |
| G4 key guard | extra batch key and a forward that reads `gt` both raise | guard removed: 0 of 2 caught |

G2 thresholds were fixed before measuring: rel-L2 <= 0.05, cosine >= 0.995, pixels within 3 levels. One frame, one clip: the residue is the CPU-vs-GPU device floor (ROUTE.md: 2.9-3.5 %).

## Open items
* E3 playback executor: call `set_history(h, n_new=1)` on EVERY raw tick (the 16-row window overlaps only then) and replan on even ticks only (`plan` refuses a half-step). Stationary history is ambiguous without `n_new` and is refused.
* E4 kit on the tip tree: `window` must come from the policy (17 raw -> `need`=19 frames); `set_t0` builds yaw from `gt_T` rows <= f0 and is NOT yet tested against a real NuRec scene. `NAV_FROM_HARNESS` (follow/left/right/straight -> v8F token) is UNVERIFIED; both scenes need a nav fill. Register the arm in `closedloop_drive.EXTRA_ARMS` (not done).
* E5 scorer: the policy returns 30 knots at 0.2 s; the kit's `wp_to_control` expects 4 knots at 0.5-2.0 s.
* E6: still to run: camera-blind mutation must move M1 through the live loop, render-replay cosine (the render gap on frozen tokens), zero control through the harness. The live path (`plan(frames, intr)` encoding frames into the deque) is built but not covered by a gate; test it with a rendered frame.
* Deque depth is `hmax` (16) rows, not 4: the model gathers 4 rows per head but reads up to `hist + max_age_s` = 10 rows back when the held ages are non-zero. `plan` needs tokens for those rows (the encoder fills them at the first call).
* Not loaded: C0 s1 and the DAG-OACC base (`v8f-dagoacc/M2b_s0_dagoacc`), hashes in PINS. Which base is an arm: U.

## Blockers
None for E1/E2. Box 2's venv and DINOv3 weights are not checked here (U).
