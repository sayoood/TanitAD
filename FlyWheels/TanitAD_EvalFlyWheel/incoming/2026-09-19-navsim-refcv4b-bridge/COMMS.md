# COMMS — E2 (TanitAD → NavSim bridge, refcv4b)

## Decisions MADE in this stream (each reversible; the PI / orchestrator may overrule)

| # | decision | why (evidence) |
|---|---|---|
| D1 | **Stage-2-only scoring for every camera arm; no number is labelled a two-stage EPDMS.** Primary statistic = S2-EPDMS-u (the devkit's stage-2 aggregation with uniform within-group weights = its own zero-mass fallback). | The 16 stage-1 scenes have **0 / 192** camera jpgs on this box (`raw/navsim_agent_inputs.json`); every warmup jpg is a 17-hex synthetic render. Decided and pre-registered in `SPEC.md` §0/§2 **before any scoring** (blob `cb1d11de…`, amended once before scoring → `3ca367ca…`, `raw/SPEC_PREREG_HASH.txt`). |
| D2 | Stage-1 tokens of camera arms answered by the **devkit's own `ConstantVelocityAgent`** as a DECLARED stand-in, so the unmodified official script runs end to end. Its summary rows are HYBRID and never reported as an arm's number. | READ FROM SOURCE (not executed): with no stage-1 row, `SceneAggregator._compute_two_frame_comfort` has no `ego_simulated_states` for the stage-1 pair, the `except` at `run_pdm_score.py:381` skips `compute_final_scores`, and `:415` then reads a `score_stage_one` column that was never created. |
| D3 | Seam agent keyed by the **scorer token** (`requires_scene=True`, reads `scene_metadata.initial_token` only) with an AgentInput **fingerprint consistency check**. | MEASURED: 8 groups / 19 tokens of DISTINCT synthetic renders carry a byte-identical 4-frame ego history → an AgentInput-only key is ambiguous (export rc=3). |
| D4 | Frames = the `wide` 256×640 variant (the model's own frame), time construction **ST** primary, **NT** declared sensitivity, **BLIND** diagnostic. | refcv4b reads 8 rows × 3-frame 10 Hz stacks (`refc_v3.py` window 8, `in_channels 9`, D-015); NavSim has 4 frames at 2 Hz. |
| D5 | Curvature from `ay / max(v0, 4)^2` clipped ±0.12 (the decoder's own caps). | KC control: sign agreement 0.969, Pearson r 0.958, median ratio 1.024 against the realised path curvature (n = 712 log windows, v0 > 4 m/s) — `raw/K2_K3_K9_roundtrip.json`. |
| D6 | Scoring runs through **E1's `navsim_win.py --patch-loader`** and E1's C: mirror (same interpreter, devkit copy, data, hash seed, thread caps as E1's CV/human numbers). | E1 MEASURED a Windows defect in `MetricCacheLoader` (backslash paths); D: is I/O-saturated by the navhard download (my first export sat in `PageIn` for 20 min). |

## ⭐ Integration asks (escalated — also in the report headline)

1. **PI decision — the warmup camera two-stage EPDMS is blocked on bytes, not code.** Either
   (a) download the 7 warmup logs' OpenScene-test camera frames (only 16 scenes × 4 frames × 3 cams
   = 192 jpgs are needed, but they ship inside `openscene_sensor_test_camera_{0..31}.tgz`; which
   parts hold these logs is unknown without a listing), or (b) accept stage-2-only warmup numbers
   and take the official column on **navhard_two_stage**, whose `curr_sensors` / `hist_sensors`
   archives are expected (NOT verified — they are still downloading) to carry stage-1 originals.
   `code/build_frames.py` builds stage-1 frames from the logs once the jpgs exist, and is proven
   bit-exact against the DataFlyWheel bank (6/6, `raw/frame_builder_reproduction.json`).
2. **DataFlyWheel correction to land:** `2026-08-28-navsim-corpus-adaptation/RESULT.md` says
   "204 / 204 scenes" for the warmup corpus. That is the STAGE-2 count; the split's 16 stage-1
   scenes have no frames and cannot have them from the warmup download. A coverage claim about "the
   split" measured on ONE of its two stages — the same root-cause class as that package's own
   one-axis-coverage retraction. Proposed RETRACTION_LOG class: *coverage stated for a whole from
   one of its parts*.
3. **Tier / loop wording to correct in briefs:** background vehicles in the two-stage scorer are
   **IDM-REACTIVE in both stages** (`run_pdm_score.py:77-79, 132-134` instantiate
   `traffic_agents_policy.reactive`), not log replay. `NAVSIM_PROTOCOL.md` §4 already says so; the
   E1/E2 briefs say "non-reactive logged agents".
4. **Promote E1's `navsim_win.py` loader patch** to a shared tool: every Windows NavSim scoring run
   needs it; two streams now depend on a file inside one stream's incoming package.
5. **⭐ A STOP floor for every NavSim row (E1 / E4 / leaderboard).** MEASURED on the official
   harness: the all-zero plan scores **official two-stage EPDMS 0.3009** on warmup vs CV's
   **0.1854** (the HF warmup LB's 18.54) — standing still beats constant velocity by 11.6 points,
   because EP ≡ 1 when the best compliant progress ≤ 5 m (`pdm_scorer.py:231-236`) and warmup's
   starts are slow. Any warmup (and, until measured, navhard) row quoted without its STOP floor is
   uninterpretable. `raw/score_STOP_zero.csv`, `raw/seam_STOP_zero.npz` (reusable as-is).
6. **E3: `tools/criteria_check.py` (registry v2.9.0) does NOT evaluate the four blocking
   `benchmarks.navsim` gates** — 0 occurrences of `navsim` in its report. This stream self-checked
   them from the registry's own keys (`code/build_artifacts.py::navsim_gate_selfcheck`, all PASS);
   the checker should own that.
7. **Register rows** (proposed text in `RESULT.md` §7 — this stream did not edit
   `GOALS_AND_CLAIMS.md`, which other agents have staged).
8. FYI box hygiene: two STOP scoring attempts were aborted by E1's RAM guard (available 2,726 MB,
   then 1,870 MB) while an UNRELATED user job ran (`earnings-swing` venv, `b2_sweep_stocks.py`,
   12 workers × ~1.4 GB); not touched; attempt 3 passed. And `git add` on D: failed twice with
   `unable to write file .git/objects/…: Permission denied` and succeeded on retry (transient).

## Coordination notes

* E1 owns the metric cache; this stream only polled `CACHE_DONE.json` and never built a cache.
* `taniteval/adapters/navsim.py`, `CRITERIA_REGISTRY.json`, `tools/criteria_check.py` were READ and
  CALLED, never edited (E3 owns them); the gate keys they lack are added by `code/build_artifacts.py`.
* navhard (next split, per the orchestrator 2026-09-19): the C: working copy is verified
  (`C:/Users/Admin/navsim-crun/data/openscene/navhard_two_stage/EXTRACT_DONE.json` ok — 5,462
  synthetic pickles, 2,731 `openscene_meta_datas`, 13,715 images each for CAM_F0/L0/R0, 76 log
  dirs); its metric cache is being built by an agent-free chain at
  `C:/Users/Admin/navsim/exp/metric_cache_navhard_two_stage` (this stream will not build another).
  Ready on this side: the export is split-parameterised (`E2_SPLIT`), `score_arm.py` carries a
  `navhard_two_stage` profile pointing at the C: copy, `build_frames.py` builds BOTH stages and is
  bit-exact on warmup. ⚠️ Check before launching: (i) that navhard's stage-1 history jpgs are among
  the 13,715 (the warmup lesson), (ii) that its 76 logs are under the mirror's
  `navsim_logs/test` (the C: mirror currently holds only warmup's 7), (iii) RAM: ~0.5 GB per
  scorer, CPU inference ~2 s / scene → ~3.3 h for 5,912 scenes per arm on this box.
