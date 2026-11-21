# 2026-09-02 — B1 EVAL lead block: LONGITUDINAL distance-keeping is PRESENT on refav1's 147-clip grid (backlog R1)

**Owner:** Benchmarks & Evals FlyWheel · **status:** built, wired, tested, banked — **UNVERIFIED on a
real refav1 checkpoint** (this box may not contact Thor; the wiring is validated on a synthetic roll
and on a MODEL-FREE dump over 20 REAL eval clips) · **integration needed:** the Master Mind passes
`--lead-block` (or ships the 2.5 MB npz beside the checkout) at the step-1000 read — §7.

**Headline.** refav1's T1 adapter reported the distance-keeping half of the LONGITUDINAL family
UNAVAILABLE *("no lead block on the 141-clip grid")*. It now has one: a per-frame lead block for
**all 147 v7.2 EVAL clips** (29,556 rows = every RAW 10 Hz frame, 2.49 MB, `raw/b1_eval_lead_block.npz`),
built from `obstacle.offline` (range-pulled from the hub, 145/147 clips), the corpus release's
egomotion and camera timestamps, through the programme's existing instrument
(`lead_source` → `lead_metrics` → `four_families`), and `refav1_arm.py` joins it by
**(clip_id, RAW frame 2t)** at analysis time (`--lead-block`, 0 GPU, `--analyze-only` works on old dumps).
Coverage (MEASURED, all frames): **LEAD 28.2 % · NO_LEAD 21.9 % · NOT_STRAIGHT 35.9 % · NO_LABEL 13.9 %**;
88 clips carry a lead on at least one frame; the free-flow share among labelled-and-straight frames is
**43.7 %** (per-clip median 51.1 %). On the 20-clip real slice the join is proven label-free
(max |dump v0 − block speed| = **2.1e-05 m/s**) and the kinematic-contract control `ol` reproduces the
block's GT headway to a **median 0.9 cm** (mean 0.14 m, p95 0.40 m).

Evidence classes: **MEASURED** = ours, artifact path given · **INHERITED** = another module/doc, not
re-verified here · **ESTIMATED** · **HYPOTHESIS**.

---

## 1. The container contract the adapter loads (read from the instrument, not invented)

Sources read end to end: `taniteval/taniteval/lead_source.py`, `lead_metrics.py`, `four_families.py`
(`longitudinal`, `_distance_keeping`, `_longitudinal_ci`), `dump_lead_join.py`,
`taniteval/tools/build_lead_block.py`, `eval_four_families.py::load_lead_block`,
`taniteval/tests/test_lead_strata.py`, `test_eval_four_families_tool.py::test_lead_block_loads_from_npz_the_banked_container`,
`stack/scripts/build_obstacle_join.py`.

**1a. What `four_families` consumes (`win["lead"]`, via `t1_eval.analyze(lead=)`)** — INHERITED from
the code, MEASURED by the tests here:

| key | shape / dtype | units / frame | consumed by |
|---|---|---|---|
| `leads` | `[W, K, 2]` float64 | lead **centre** at `t0 + ts_rel[k]`, **window-origin ego frame at t0, x forward / y left, metres, clip-local**; **NaN = no lead** | `lead_metrics.per_step_gap` (gap = along − `size_x`/2, evaluated in the *predicted path's* local heading, |lat| < 2 m corridor) |
| `lead_lens` | `[W]` float64 | lead `size_x` (m); **NaN where absent** | gap convention (rig origin → lead REAR face) |
| `speeds` | `[W]` float64 | ego speed at t0 (m/s) — the **time-gap denominator** AND the **speed-band stratifier** (`SPEED_BANDS` 0–1/1–3/3–6/6–10/10–15/15+) | `distance_keeping`, `distance_keeping_by_speed`, `v0_antiecho` |
| `state` | `[W]` str | one of `LEAD` / `NO_LEAD` / `NOT_STRAIGHT` / `NO_LABEL` (`lead_source`) — the per-stratum denominators (`window_states_total`, `window_states` per band) that keep NO_LABEL out of the free-flow count | `distance_keeping_by_speed(states=)` |
| `eid` | `[W]` | the **episode-cluster unit** of the bootstrap (`test_lead_strata.py`: the by_speed block is UNAVAILABLE without it; `_longitudinal_ci` needs it aligned to the scored windows) | `taniteval.ci.episode_cluster_bootstrap` |
| `n_boot`, `seed` | scalars | optional | by_speed |
| `path_steps`, `dt_s` | optional | the TIME join for a lead track coarser than the path (`_distance_keeping`); **not needed here** — the block is on the dump's own 0.2 s grid, shapes match | `_distance_keeping` |

"No lead" encoding: `leads[i] = NaN`, `lead_lens[i] = NaN`, `state[i] ≠ "LEAD"`, `has_lead[i] = False`,
`gap0_m[i] = NaN` — `distance_keeping` skips the window (`n` counts only finite headways); the state
says *why*. ⛔ NO_LABEL is never NO_LEAD (`lead_source` docstring).

**1b. The banked POSITIONAL container** (`…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`,
`np.savez`; MEASURED by loading it): `leads f8 (881,4,2)`, `lead_lens f8 (881,)`, `speeds f8 (881,)`,
`state <U8 (881,)`, `gap0_m f8`, `eid <U8 ("ep_00000"…)`, `ts_rel_s f8 (4,) = [0.5,1,1.5,2]`,
`episodes_json u1`. Rows are `rollout.collect`'s window order — a join by POSITION, which refav1's
grid (episode, cache step t) cannot use.

**1c. The PER-FRAME container this package banks** (`np.savez_compressed`; loads through
`eval_four_families.load_lead_block`, which requires `leads/lead_lens/speeds/state/eid`):

| key | dtype/shape | meaning |
|---|---|---|
| `clip_id` `<U36 [R]`, `frame` `i8 [R]`, `t0_s` `f8 [R]` | join key | RAW v2ep 10 Hz frame index `i ∈ [0, n_target)` (the refav1 loader's timeline: cache step t ↔ frame 2t; NOT the post-n_stack-trim provider index), and its clip time |
| `eid` `<U36 [R]` | = `clip_id` | the cluster unit at build time; the adapter re-maps to the dump's `ep{fi:03d}` |
| `leads f8 [R,10,2]`, `lead_lens f8 [R]`, `speeds f8 [R]`, `state <U12 [R]`, `gap0_m f8 [R]`, `has_lead b1 [R]` | as 1a | `<U12` because `NOT_STRAIGHT` is 12 chars (`<U8` truncates it) |
| `ts_rel_s f8 [10] = 0.2·(1..10)`, `dt_s f8 [1] = 0.2` | horizon grid | = refav1's plan horizon (`cfg.plan_steps` 10 × `op_dt` 0.2 s); the adapter refuses any other grid |
| `lead_track_id <U64`, `rel_speed_mps f8`, `headway_time_s f8` | per-frame descriptors | lead along-speed − ego speed at t0 (+ = pulling away; least squares over the lead's own samples in the last 0.6 s, NaN < 2 samples / < 0.15 s); gap0 / v_ego (NaN below `MIN_SPEED_MPS` 0.5) |
| `gt_headway_min_m`, `gt_time_gap_min_s`, `gt_min_ttc_s`, `gt_n_steps_in_corridor` | the GT arm | `lead_metrics.distance_keeping` of the GT ego path — the D-LEAD-1-style known value the kinematic-contract control must reproduce |
| `straight_dep_m`, `straight_checked_m` | the straight gate's own numbers | `lead_source.ego_lateral_departure` |
| `coverage_json u1`, `meta_json u1` | JSON bytes | per-clip coverage (counts, spans, grid, registration, cross-check, obstacle stats) and provenance (inputs + sha256, conventions, states, refusals) |

Conventions (INHERITED from `lead_source`/`lead_metrics`, unchanged): strictly causal selection
(cuboids ≤ t0, staleness ≤ 0.5 s, |lat| < 2 m, gap ≤ 80 m, vehicle classes
`automobile/heavy_truck/bus/other_vehicle/trailer`); gap = along − size_x/2; the straight-driving gate
(PI 2026-08-26) = the corridor's own half-width over the claimed range; the lead's future track is
**GT environment** — a scoring input, never a model input.

## 2. The time base: reconstructed from the corpus release, then cross-checked (MEASURED)

The B1 v2ep poses are `physicalai.signals_at(egomotion, t_query)` with
`t_query = linspace(t_cam[0], t_cam[-1], int(span_s·TARGET_HZ))` — the builder's own formula
(`stack/scripts/v2_compressed.py:115-124`, `stack/tanitad/data/physicalai.py:711-735`). Both inputs
ship with the corpus release (`_s2build/release/tanitad-v7-training-corpus/egomotion/egomotion_alpamayo.tar`,
4,719 parquets; `timestamps/timestamps.tar`, 4,719 parquets), so the grid AND the poses of every one of the
147 clips are reconstructed exactly (`build_lead_block_b1.episode_grid` / `episode_poses` — `TARGET_HZ` and
`signals_at` are IMPORTED from the stack).

### Ego cross-check on the 20 local v2ep clips (MEASURED, `raw/grid_crosscheck.json` + the build report)

| quantity | min | median | max |
|---|---|---|---|
| max \|Δxy\| reconstructed vs v2ep poses (m) | 0 | 0 | 0 |
| max \|Δyaw\| (deg) | 0 | 0 | 0 |
| max \|Δv\| (m/s) | 0 | 0 | 0 |
| \|grid formula − content registration (`lead_source.register_poses_to_time`)\| max Δt (s) | 8.51e-05 | 2.86e-04 | 1.82e-03 |
| registration median residual (m) | 0.00015 | 0.0017 | 0.0074 |
| grid a = clip time of frame 0 (s) | −0.1556 | −0.0754 | −0.0002 |
| grid b = frame spacing (s) | 0.100498 | 0.100667 | 0.100843 |
| egomotion rate (Hz) | 25 | 99.7 | 100 |
| `T_v2ep == n_target` (all 20) | True | | |

The reconstructed poses are **float32-identical** to the banked v2ep poses on 20/20 clips (the brief asked
for centimetres and < 1°; the builder refuses above 1 cm / 1° / 1 cm/s and the test pins it). The
content registration agrees to ≤ 1.8 ms. ⚠️ The grid is **0.1005–0.1010 s**, not 0.1 s, and frame 0 sits
at **−0.33 … +0.03 s** clip time (all 147 clips: `n_frames` 199–207) — `t = 0.1·i` would drift ~0.13 s by
frame 200 (~1.8 m of lead at 13.6 m/s). Egomotion is 100 Hz on 129 clips and 20–33 Hz on 18 (median
99.8 Hz); `reference_frame_timestamp_us == timestamp_us` on every cuboid (max skew 0.0 s), so the
rig→world→t0 chain's assumption is checked, not asserted.

## 3. The pull (MEASURED, `raw/pull_manifest.json`, `raw/pull.log`, `raw/hf_tree_obstacle.json`)

`labels/obstacle.offline/` on `nvidia/PhysicalAI-Autonomous-Vehicles` holds 3,146 chunk zips
(158.0 GB; 2.7–110.9 MB each). The 147 clips span **122 chunks** (`index/clip_to_chunk.parquet` of the
corpus release). Whole-chunk downloads were never needed: each zip's central directory is range-read
from its final 256 KB and ONLY `{clip}.obstacle.offline.parquet` is fetched (the release's
`tools/pull_egomotion_range.py` recipe, re-implemented in `build_lead_block_b1.pull_obstacle_offline`,
token read in place from `Keys.txt`). **145 members, 77 MB, 59 s**, every member parse-verified
(schema columns) before banking, sha256 per file in the manifest. Local store (ONE place, re-pullable):
`C:\Users\Admin\tanitad-data\physicalai\labels\obstacle_offline_b1eval\`.

Two clips have **no member in their chunk**: `204b15b4-bd0a-4155-b01e-572c55197d0e` (chunk 1894) and
`38fe80c1-caf1-45ea-810b-2681ea488c6d` (chunk 2309). Second probe, different mechanism: the hub's own
`clip_index.parquet` (306,152 rows) lists both as `clip_is_valid=True` in exactly those chunks, and
`features.csv` confirms the member template `{clip_id}.obstacle.offline.parquet` — so the pull looked in
the right zip for the right name; the absence is real (the dataset card's 97.44 % coverage; 2/147 =
1.4 % here). Their rows are **NO_LABEL with the reason in `coverage`**, never free flow.

## 4. Coverage (MEASURED, `raw/b1_eval_lead_block.npz.report.json`)

| quantity | all RAW frames (10 Hz) | even frames = the 0.2 s grid (frame 2t) |
|---|---|---|
| rows | 29,556 | 14,846 |
| LEAD | 8,341 (28.22 %) | 4,175 (28.12 %) |
| NO_LEAD | 6,480 (21.92 %) | 3,255 (21.93 %) |
| NOT_STRAIGHT | 10,622 (35.94 %) | 5,327 (35.88 %) |
| NO_LABEL | 4,113 (13.92 %) | 2,089 (14.07 %) |
| labelled AND straight (LEAD+NO_LEAD) | 14,821 | 7,430 |
| free-flow share = NO_LEAD / (LEAD+NO_LEAD) | **43.72 %** pooled; per-clip median 51.1 % over 135 clips with a labelled-straight frame | 43.81 % pooled |
| clips | 147 built / 147 requested; **88 with any LEAD frame** (84 with a GT distance-keeping value, i.e. a lead track reaching into the horizon); **2 without obstacle.offline** | |

⚠️ Reading the shares: NO_LABEL is dominated by the **span tail** — the labels stop at ~20.0 s while the
horizon reaches 2.0 s past t0, so the last ~15–17 frames of every clip are NO_LABEL by the span guard
(`OBS_SPAN_GUARD_S` 0.5 s) — plus the 2 unlabelled clips (402 rows) and a handful of short-span clips
(`5716299c` 8.5–8.9 s, `c0142275` 0.1–1.6 s, `a459164a` 11.2–13.0 s, `fb8e8200` 0.1–6.6 s, …).
NOT_STRAIGHT is the straight-driving gate (a bend within the claimed range — the lead gap, or 80 m with
no lead); it is its own state. The refav1 windows at the step-1000 read sit at cache steps
`t ∈ [W−1, T_c − reach − 2]` with `reach = max(horizon_k, wm_k) = 30` at defaults, i.e. frames
≤ ~138 (t0 ≤ ~13.8 s), so the span tail mostly falls OUTSIDE the scored windows and the effective
NO_LABEL share there will be LOWER than 14 % (ESTIMATED; the adapter prints the real counts).

### Per-clip coverage (MEASURED; RAW 10 Hz frames; free-flow = NO_LEAD/(LEAD+NO_LEAD); "—" = no denominator)

| clip (8) | frames | LEAD | NO_LEAD | NOT_STRAIGHT | NO_LABEL | free-flow | label span (s) | obstacle.offline |
|---|---|---|---|---|---|---|---|---|
| 01acc9de | 201 | 0 | 30 | 155 | 16 | 100 % | 0.0–20.0 | yes |
| 01be5919 | 201 | 0 | 6 | 178 | 17 | 100 % | 0.0–20.0 | yes |
| 026ef99a | 207 | 184 | 0 | 3 | 20 | 0 % | 0.0–20.0 | yes |
| 030011f7 | 201 | 0 | 185 | 0 | 16 | 100 % | 0.0–20.0 | yes |
| 060531e1 | 201 | 183 | 2 | 0 | 16 | 1 % | 0.0–20.0 | yes |
| 07f7b41f | 201 | 0 | 1 | 185 | 15 | 100 % | 0.0–20.0 | yes |
| 09759d8c | 201 | 0 | 34 | 74 | 93 | 100 % | 0.0–12.2 | yes |
| 0c99c815 | 201 | 184 | 0 | 1 | 16 | 0 % | 0.0–20.0 | yes |
| 0fa2b231 | 201 | 171 | 2 | 12 | 16 | 1 % | 0.0–20.0 | yes |
| 10804db8 | 201 | 0 | 42 | 144 | 15 | 100 % | 0.0–20.0 | yes |
| 11e40da1 | 201 | 22 | 30 | 70 | 79 | 58 % | 6.7–20.0 | yes |
| 126b3a97 | 201 | 67 | 64 | 54 | 16 | 49 % | 0.0–20.0 | yes |
| 13c95b6a | 201 | 0 | 0 | 186 | 15 | — | 0.0–20.0 | yes |
| 14157960 | 201 | 0 | 185 | 0 | 16 | 100 % | 0.1–20.0 | yes |
| 142a3a72 | 201 | 0 | 104 | 80 | 17 | 100 % | 0.0–19.9 | yes |
| 14bc9fcf | 201 | 0 | 80 | 104 | 17 | 100 % | 0.4–19.9 | yes |
| 16d89725 | 201 | 0 | 172 | 13 | 16 | 100 % | 0.1–20.0 | yes |
| 17775e74 | 202 | 184 | 0 | 1 | 17 | 0 % | 0.0–20.0 | yes |
| 18b72291 | 201 | 1 | 0 | 184 | 16 | 0 % | 0.0–20.0 | yes |
| 1bd040a8 | 201 | 2 | 58 | 126 | 15 | 97 % | 0.0–20.0 | yes |
| 1c3a2c7c | 199 | 172 | 0 | 12 | 15 | 0 % | 0.0–20.0 | yes |
| 204b15b4 | 201 | 0 | 0 | 0 | 201 | — | — | **NO** |
| 2172a5a3 | 201 | 183 | 0 | 2 | 16 | 0 % | 0.0–20.0 | yes |
| 24fee8a5 | 201 | 0 | 22 | 163 | 16 | 100 % | 0.0–20.0 | yes |
| 2602baaa | 201 | 183 | 2 | 0 | 16 | 1 % | 0.0–20.0 | yes |
| 262b900f | 201 | 0 | 20 | 165 | 16 | 100 % | 0.0–20.0 | yes |
| 269478eb | 207 | 146 | 3 | 38 | 20 | 2 % | 0.0–20.0 | yes |
| 27f44280 | 201 | 65 | 68 | 52 | 16 | 51 % | 0.0–20.0 | yes |
| 2cd0f4e4 | 201 | 121 | 8 | 55 | 17 | 6 % | 0.0–20.0 | yes |
| 2e8a2df2 | 201 | 0 | 28 | 158 | 15 | 100 % | 0.0–20.0 | yes |
| 316f3c56 | 201 | 0 | 0 | 185 | 16 | — | 0.0–20.0 | yes |
| 31d0adc4 | 201 | 183 | 0 | 2 | 16 | 0 % | 0.0–20.0 | yes |
| 32f9b6b2 | 201 | 29 | 9 | 146 | 17 | 24 % | 0.0–19.9 | yes |
| 330b4626 | 201 | 39 | 130 | 16 | 16 | 77 % | 0.0–20.0 | yes |
| 34a1e12e | 201 | 7 | 91 | 87 | 16 | 93 % | 0.0–20.0 | yes |
| 37e61793 | 201 | 40 | 0 | 145 | 16 | 0 % | 0.0–20.0 | yes |
| 38fe80c1 | 201 | 0 | 0 | 0 | 201 | — | — | **NO** |
| 3971a8c6 | 201 | 0 | 0 | 186 | 15 | — | 0.0–20.0 | yes |
| 39bf584e | 200 | 95 | 88 | 0 | 17 | 48 % | 0.0–20.0 | yes |
| 3ef3dcc8 | 201 | 5 | 99 | 81 | 16 | 95 % | 0.0–20.0 | yes |
| 3ff11411 | 201 | 0 | 142 | 42 | 17 | 100 % | 0.0–20.0 | yes |
| 42745b48 | 201 | 135 | 49 | 1 | 16 | 27 % | 0.0–20.0 | yes |
| 45358841 | 201 | 136 | 7 | 43 | 15 | 5 % | 0.1–20.0 | yes |
| 45a9f7ec | 201 | 0 | 106 | 78 | 17 | 100 % | 0.0–20.0 | yes |
| 45aea78c | 201 | 0 | 68 | 117 | 16 | 100 % | 0.0–19.9 | yes |
| 47241da8 | 201 | 0 | 89 | 96 | 16 | 100 % | 0.0–20.0 | yes |
| 480d7910 | 201 | 0 | 128 | 0 | 73 | 100 % | 6.0–20.0 | yes |
| 4c5264dd | 201 | 89 | 0 | 95 | 17 | 0 % | 0.0–20.0 | yes |
| 4ee70f88 | 201 | 120 | 1 | 64 | 16 | 1 % | 0.0–20.0 | yes |
| 50caa2df | 201 | 0 | 46 | 138 | 17 | 100 % | 0.0–19.9 | yes |
| 51c01ba4 | 201 | 183 | 1 | 0 | 17 | 1 % | 0.0–20.0 | yes |
| 5271d125 | 201 | 143 | 0 | 41 | 17 | 0 % | 0.0–20.0 | yes |
| 52ddcaaf | 201 | 0 | 90 | 94 | 17 | 100 % | 0.0–20.0 | yes |
| 565b6ff0 | 201 | 109 | 8 | 69 | 15 | 7 % | 0.1–20.0 | yes |
| 5716299c | 201 | 0 | 0 | 0 | 201 | — | 8.5–8.9 | yes |
| 59c25a0f | 201 | 0 | 186 | 0 | 15 | 100 % | 0.0–20.0 | yes |
| 5b4df66c | 201 | 0 | 18 | 167 | 16 | 100 % | 0.0–19.9 | yes |
| 5d39a1de | 201 | 0 | 0 | 185 | 16 | — | 0.0–20.0 | yes |
| 5d94cbbb | 201 | 6 | 28 | 152 | 15 | 82 % | 0.0–20.0 | yes |
| 5f86e111 | 201 | 0 | 75 | 54 | 72 | 100 % | 6.0–20.0 | yes |
| 62f3222e | 201 | 183 | 0 | 3 | 15 | 0 % | 0.0–20.0 | yes |
| 63434e0b | 201 | 1 | 0 | 184 | 16 | 0 % | 0.1–19.9 | yes |
| 66e4d574 | 201 | 183 | 0 | 2 | 16 | 0 % | 0.0–20.0 | yes |
| 672d70d8 | 200 | 0 | 14 | 170 | 16 | 100 % | 0.0–20.0 | yes |
| 6908302e | 201 | 0 | 55 | 131 | 15 | 100 % | 0.0–20.0 | yes |
| 695081db | 201 | 131 | 29 | 24 | 17 | 18 % | 0.0–20.0 | yes |
| 6a89cca3 | 201 | 0 | 70 | 0 | 131 | 100 % | 10.3–18.4 | yes |
| 6b3e802e | 201 | 0 | 50 | 135 | 16 | 100 % | 0.0–20.0 | yes |
| 6ed4ef7a | 201 | 1 | 0 | 106 | 94 | 0 % | 0.1–12.2 | yes |
| 73e750eb | 201 | 18 | 0 | 168 | 15 | 0 % | 0.0–20.0 | yes |
| 7526e299 | 201 | 0 | 53 | 132 | 16 | 100 % | 0.1–20.0 | yes |
| 77a247be | 201 | 49 | 0 | 136 | 16 | 0 % | 0.0–20.0 | yes |
| 79e30ac2 | 201 | 0 | 12 | 174 | 15 | 100 % | 0.0–20.0 | yes |
| 80b890fc | 201 | 160 | 24 | 1 | 16 | 13 % | 0.0–20.0 | yes |
| 80fcd92f | 201 | 184 | 2 | 0 | 15 | 1 % | 0.0–20.0 | yes |
| 826cd25c | 201 | 79 | 2 | 104 | 16 | 2 % | 0.0–20.0 | yes |
| 829320e2 | 201 | 88 | 2 | 95 | 16 | 2 % | 0.0–20.0 | yes |
| 84952a7e | 201 | 0 | 43 | 142 | 16 | 100 % | 0.0–20.0 | yes |
| 85a47aa1 | 201 | 158 | 4 | 22 | 17 | 2 % | 0.0–20.0 | yes |
| 8d610d87 | 201 | 0 | 180 | 0 | 21 | 100 % | 0.0–19.5 | yes |
| 8df0cb04 | 201 | 27 | 0 | 158 | 16 | 0 % | 0.0–20.0 | yes |
| 8f1baf35 | 201 | 105 | 0 | 80 | 16 | 0 % | 0.0–20.0 | yes |
| 8fe27d99 | 200 | 63 | 18 | 103 | 16 | 22 % | 0.0–20.0 | yes |
| 91929cea | 200 | 89 | 1 | 95 | 15 | 1 % | 0.0–20.0 | yes |
| 91e90a95 | 201 | 106 | 27 | 53 | 15 | 20 % | 0.0–20.0 | yes |
| 9370d445 | 201 | 26 | 1 | 157 | 17 | 4 % | 0.0–20.0 | yes |
| 93854321 | 201 | 55 | 27 | 103 | 16 | 33 % | 0.0–20.0 | yes |
| 998310a5 | 200 | 17 | 42 | 125 | 16 | 71 % | 0.0–20.0 | yes |
| 9b4347d2 | 201 | 0 | 55 | 130 | 16 | 100 % | 0.0–20.0 | yes |
| 9c50f803 | 201 | 129 | 0 | 55 | 17 | 0 % | 0.0–20.0 | yes |
| 9d3d1cd1 | 201 | 12 | 19 | 153 | 17 | 61 % | 0.0–20.0 | yes |
| 9fc24b6e | 201 | 64 | 121 | 0 | 16 | 65 % | 0.0–20.0 | yes |
| a0d95df6 | 201 | 164 | 2 | 19 | 16 | 1 % | 0.0–20.0 | yes |
| a459164a | 201 | 0 | 0 | 9 | 192 | — | 11.2–13.0 | yes |
| a6b2719b | 201 | 0 | 186 | 0 | 15 | 100 % | 0.0–20.0 | yes |
| a7e15877 | 201 | 0 | 185 | 0 | 16 | 100 % | 0.0–20.0 | yes |
| aa00f477 | 201 | 23 | 86 | 76 | 16 | 79 % | 0.0–20.0 | yes |
| aa594aa8 | 201 | 0 | 0 | 78 | 123 | — | 11.1–20.0 | yes |
| ac81f868 | 201 | 0 | 105 | 0 | 96 | 100 % | 5.5–17.0 | yes |
| acf8e77f | 201 | 0 | 167 | 18 | 16 | 100 % | 0.0–20.0 | yes |
| af6f5964 | 201 | 91 | 84 | 10 | 16 | 48 % | 0.0–20.0 | yes |
| afe44776 | 199 | 0 | 184 | 0 | 15 | 100 % | 0.0–20.0 | yes |
| b0252ad5 | 201 | 168 | 0 | 16 | 17 | 0 % | 0.0–20.0 | yes |
| b05195bf | 201 | 72 | 73 | 40 | 16 | 50 % | 0.0–20.0 | yes |
| b19eb35f | 201 | 126 | 2 | 56 | 17 | 2 % | 0.0–20.0 | yes |
| b62be3fa | 201 | 0 | 0 | 180 | 21 | — | 0.8–19.9 | yes |
| b99058dc | 202 | 183 | 0 | 1 | 18 | 0 % | 0.0–20.0 | yes |
| c0142275 | 201 | 0 | 0 | 2 | 199 | — | 0.1–1.6 | yes |
| c08bebe5 | 201 | 104 | 0 | 80 | 17 | 0 % | 0.0–20.0 | yes |
| c1cb24d2 | 201 | 19 | 70 | 96 | 16 | 79 % | 0.0–20.0 | yes |
| c84ea182 | 201 | 0 | 163 | 22 | 16 | 100 % | 0.0–20.0 | yes |
| c8a39711 | 201 | 5 | 181 | 0 | 15 | 97 % | 0.0–20.0 | yes |
| ca11a2a2 | 207 | 107 | 20 | 60 | 20 | 16 % | 0.0–20.0 | yes |
| caf90ce8 | 201 | 0 | 59 | 83 | 59 | 100 % | 4.6–19.9 | yes |
| d1d382aa | 200 | 20 | 120 | 44 | 16 | 86 % | 0.0–20.0 | yes |
| d217b78a | 201 | 0 | 0 | 184 | 17 | — | 0.0–20.0 | yes |
| d4729053 | 201 | 45 | 0 | 139 | 17 | 0 % | 0.0–20.0 | yes |
| d571e998 | 201 | 163 | 2 | 21 | 15 | 1 % | 0.0–20.0 | yes |
| d85682b8 | 201 | 18 | 49 | 118 | 16 | 73 % | 0.0–20.0 | yes |
| dbad28c2 | 200 | 0 | 14 | 155 | 31 | 100 % | 1.9–19.9 | yes |
| dca157a0 | 201 | 135 | 49 | 0 | 17 | 27 % | 0.0–20.0 | yes |
| e0e6b875 | 200 | 183 | 1 | 0 | 16 | 1 % | 0.0–20.0 | yes |
| e319062b | 201 | 0 | 186 | 0 | 15 | 100 % | 0.0–20.0 | yes |
| e66979e0 | 201 | 183 | 2 | 0 | 16 | 1 % | 0.0–20.0 | yes |
| e6cb7cd4 | 201 | 183 | 0 | 2 | 16 | 0 % | 0.0–20.0 | yes |
| e7ada9bb | 201 | 84 | 0 | 101 | 16 | 0 % | 0.0–20.0 | yes |
| e8ace736 | 201 | 0 | 51 | 133 | 17 | 100 % | 0.0–19.9 | yes |
| ea019ab7 | 201 | 183 | 1 | 0 | 17 | 1 % | 0.0–19.9 | yes |
| eabb9abf | 201 | 97 | 78 | 10 | 16 | 45 % | 0.0–20.0 | yes |
| eb96a48b | 201 | 12 | 27 | 146 | 16 | 69 % | 0.0–20.0 | yes |
| ebd51f52 | 201 | 0 | 21 | 164 | 16 | 100 % | 0.1–20.0 | yes |
| ed09262b | 201 | 183 | 2 | 0 | 16 | 1 % | 0.0–20.0 | yes |
| ed87040c | 201 | 1 | 17 | 168 | 15 | 94 % | 0.0–20.0 | yes |
| edd5d17b | 201 | 0 | 177 | 9 | 15 | 100 % | 0.0–20.0 | yes |
| f14ae2d2 | 202 | 3 | 182 | 0 | 17 | 98 % | 0.0–20.0 | yes |
| f19cffbf | 201 | 49 | 57 | 78 | 17 | 54 % | 0.0–20.0 | yes |
| f1a94fee | 201 | 18 | 101 | 66 | 16 | 85 % | 0.0–20.0 | yes |
| f1e1536b | 201 | 31 | 74 | 81 | 15 | 70 % | 0.0–20.0 | yes |
| f26af403 | 201 | 94 | 0 | 91 | 16 | 0 % | 0.0–20.0 | yes |
| f4226f59 | 201 | 0 | 186 | 0 | 15 | 100 % | 0.0–20.0 | yes |
| f4740120 | 201 | 55 | 2 | 129 | 15 | 4 % | 0.0–20.0 | yes |
| f6e7827e | 201 | 0 | 3 | 182 | 16 | 100 % | 0.0–19.9 | yes |
| f901e9e6 | 201 | 182 | 3 | 0 | 16 | 2 % | 0.0–20.0 | yes |
| fb8e8200 | 201 | 0 | 52 | 0 | 149 | 100 % | 0.1–6.6 | yes |
| fe764229 | 201 | 63 | 0 | 123 | 15 | 0 % | 0.0–20.0 | yes |
| fe83e06e | 201 | 6 | 0 | 179 | 16 | 0 % | 0.1–20.0 | yes |
| ffcd075e | 201 | 173 | 0 | 13 | 15 | 0 % | 0.0–20.0 | yes |

## 5. The wiring (`taniteval/tools/refav1_arm.py`) and its validation

**What changed.** `analyze_refav1(..., lead_block=)` loads the block through
`eval_four_families.load_lead_block` (the one two-container reader), refuses a POSITIONAL block by name
(no `clip_id`/`frame`), joins **(clip_id from the dump manifest, RAW frame 2t from `ws`)** in dump order,
checks the horizon grid (`ts_rel_s[:K] == 0.2·(1..K)`, else refused — never truncated/resampled) and the
**label-free speed proof** (dump `v0 = poses[2t,3]` vs block `speeds` at the joined row; both are the same
egomotion interpolation, tolerance 1e-3 m/s; a mismatch refuses THAT episode), then passes `lead=` into the
untouched `t1_eval.analyze` → `four_families.longitudinal` → `distance_keeping` (+ `by_speed`, + the
`distance_keeping.*` components of the longitudinal `ci` block). Windows the block does not cover stay
NO_LABEL and are **counted** (`n_windows_no_row`, per-episode status/reason); a block that covers zero
labelled windows is REFUSED with the counts rather than handed to `lead_metrics`, whose all-NaN reason
says "free-flow". `rec["refav1"]["distance_keeping"]` adds: coverage + block provenance (path, sha256,
build meta), a per-arm summary (status, n, means, CIs, per-band n), the **GT reference** on the same
windows, the **kinematic-contract check** (`ol` vs GT headway), and the **paired deltas** per arm pair
(`lead_metrics.paired_distance_keeping`, jointly-finite windows) — the same per-window values
`four_families` computed, asserted equal before use. CLI: `--lead-block PATH` (default = the banked
npz when the checkout carries it) / `--no-lead-block`. The join runs at analysis time, so
`--analyze-only <old dump> --lead-block …` gains the family with **zero GPU**.

**Tests (MEASURED in the off-Drive mirror, `C:\Users\Admin\tanitad-wt`, venv `tanitad`):**

| suite | result |
|---|---|
| `stack/tests/test_refav1_lead_block.py` (new: container round-trip, builder formula, PRESENT / REFUSED-with-count / NO_LABEL-never-free-flow / wrong-grid / positional / speed-mismatch / no-block, the real-slice pose reconstruction, the banked block on the real slice) | **11 passed** (24.8 s) |
| `stack/tests/test_refav1_arm.py` + `stack/tests/test_refa_v1_plan_goal.py` (adapter + planner-goal, incl. the scope-addition updates, §8) | **22 passed** (15 + 7, 29.8 s) |

**Model-free validation on the 20 REAL slice clips** (`raw/kinematic_dump_20clips_analysis.json`,
`raw/kin.log`; a refav1-shaped dump with `g` / `ol` (recorded actions from v0, T0) / `ha` (hold-action, T1)
built from the real fp8 cache + v2ep poses through the adapter's own loader and helpers — NO model, so
not a model result; then `analyze_refav1(lead_block=banked)`):

Windows 1,739 / episodes 20 (all OK) · counts LEAD 562 · NO_LEAD 367 · NOT_STRAIGHT 694 · NO_LABEL 116
(the no-obstacle clip `204b15b4` is among the 20 → its 87 windows are NO_LABEL) · no-row 0 ·
**speed-check max |dump v0 − block speed| = 2.1e-05 m/s** (tol 1e-3).

| arm (tier) | n with lead | mean min-headway (m) [CI] | mean min time-gap (s) [CI] (n) | mean min-TTC (s) [CI] (n_closing) |
|---|---|---|---|---|
| ol (T0) | 558 | 34.53 [22.37, 49.36] | 3.33 [2.23, 4.60] (558) | 23.17 [18.08, 27.73] (293) |
| ha (T1) | 556 | 34.44 [22.22, 49.23] | 3.32 [2.23, 4.59] (556) | 23.12 [18.00, 27.77] (287) |
| GT reference (block `gt_*`) | 556 | 34.42 [22.29, 49.67] | 3.32 [2.22, 4.55] (556) | 22.94 [17.56, 27.84] |

Episode-cluster bootstrap over the 20 clips, n_boot 500 (a 20-clip interval — wide by construction; the
numbers are the wiring proof, not a claim). Kinematic-contract check (`ol` vs GT headway): n_both 555,
mean |Δ| 0.145 m, **median 0.009 m**, p95 0.40 m, max 6.38 m; ol-only 3 / GT-only 1 (corridor-edge
windows). `ol` ADE over 2 s on this slice: 0.465 m [0.311, 0.635] — the real (a, κ) contract is looser
than the synthetic fixture's 0.08 m, and the headway tail (max) is where `ol` drifts across the corridor
edge so the min-over-steps jumps. Per band (`ol`): 0–1 m/s NOT-APPLICABLE 0/59 · 1–3 OK 48/85 ·
3–6 OK 42/288 · 6–10 OK 138/430 · 10–15 OK 148/188 · 15+ OK 182/689. The `--no-lead-block` CLI path
reproduces the previous UNAVAILABLE refusal with its reason (`raw/cli_smoke_nolead.json`).

## 6. Caveats (each a stated limit, none absorbed)

1. **UNVERIFIED on a real refav1 checkpoint.** No `cl` arm exists on this box; the paired `cl − ol` /
   `cl − ha` distance-keeping deltas are exercised on the synthetic roll only.
2. **NOT_STRAIGHT is 36 % of frames.** The straight gate (|lat| < 2 m over the claimed range, up to 80 m
   with no lead) is strict on urban clips. It is the instrument's rule (PI 2026-08-26), reported as its
   own state; the Master Mind should quote LEAD/NO_LEAD/NOT_STRAIGHT/NO_LABEL counts beside every number.
3. **Staleness convention.** A lead whose track ends inside the horizon is held at its last sample for
   ≤ 0.5 s (`MAX_STALE_S`), which shrinks the gap for GT and every arm alike; the lead's own motion within
   the ≤ 0.1 s sample staleness is uncompensated (`obstacle.offline` carries no velocity) — ≤ v_lead·0.1 s.
   Inherited from `lead_source`; paired deltas are on the same windows, so the direction is unaffected.
4. **`rel_speed_mps` is NaN** where the lead has < 0.15 s of history before t0 (first frames of a track).
5. **Coverage per window at the real read** differs from the all-frames shares (§4: the span tail is
   mostly outside the scored `t` range) — the adapter prints and records the real counts.
6. **Egomotion at 20–33 Hz on 18 clips** → poses interpolated over ≤ 0.05 s; registration residual ≤ 7 mm.
7. **The block is on refav1's default grid** (K = 10 × 0.2 s = `cfg.plan_steps × op_dt`). A `--horizon-k`
   < 10 uses the prefix; any other `dt`/grid is refused (rebuild with `--dt/--k`). refav1's loader refuses
   `op_dt ≠ 0.2` anyway.
8. **Two clips have no `obstacle.offline`** (§3) and several have short label spans (§4) — NO_LABEL.
9. The pulled parquets live in ONE place (local dev box); they are re-pullable deterministically and the
   per-file sha256s are in `raw/pull_manifest.json` — the block itself is in the repo.

## 7. What the Master Mind must do to integrate (the step-1000 read)

```bash
# on the eval box, from the repo checkout (needs the banked npz; 2.5 MB, sha256 below)
PYTHONPATH=<repo>/stack:<repo>/taniteval python3 taniteval/tools/refav1_arm.py \
  --ckpt … --cache … --episodes … --labels … --nav … --device cuda --window-stride 10 \
  --dump-dir <dump> --out <out.json> --arm refav1-<step> \
  --lead-block "<repo>/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"
# an EXISTING dump (any adapter version that wrote manifest.episodes[].clip_id — all of them):
python3 taniteval/tools/refav1_arm.py --arm refav1-<step> --analyze-only <dump> --out <out.json> \
  --tiers cl_navshuf=T1,cl_oraclegoal=T0 --lead-block <the npz>
```

* The flag defaults to the banked path when the checkout carries it; a checkout without the Research Lab
  tree (pod-style file-ship) must receive the npz explicitly — verify by sha256
  `c0525943605cfd4ab55258bed04046b0e58257bb888544c385095708a0091898` (2,489,644 bytes).
* Read the family at `rec["arms"][arm]["four_families"]["longitudinal"]["distance_keeping"]`
  (status/n/means/`by_speed`) and its intervals under `…["longitudinal"]["ci"]["components"]["distance_keeping.*"]`;
  coverage, GT reference, the `ol`-vs-GT check and the **paired** deltas at `rec["refav1"]["distance_keeping"]`.
  The console prints `[lead] … LEAD n NO_LEAD n NOT_STRAIGHT n NO_LABEL n · speed-check max … · PRESENT`.
* ⛔ Quote per state and per band, never pooled; `min_ttc` is censored at 30 s (`n_closing` beside it).
* A speed-check above 1e-3 m/s on any episode means the cache/v2ep on that box are not the ones this block
  was built against — the episode is refused, the record says so; do not "fix" by widening the tolerance.

## 8. Scope addition (coordinator, same turn): goal provenance in the adapter

`refa_v1.plan` (e609a98) stamps `goal_source ∈ {supplied, tactical_imagined, none}`, `goal_space`
(`tactical_query_field`) and `goal_action` (`{lat, lon, controls}`) on its result. The adapter now banks
them per window in the decisions sidecar (`goal_{source,lat,lon}_<arm>`), reports per planning arm
`goal_source_fractions`, `goal_space`, the selected-manoeuvre histograms `goal_action_{lat,lon}` (the
TACTICAL family's "selected" half) and `goal_vs_declared_head_agreement` (1.0 on the fixture — one decode
path), and DERIVES the `_protocol.goal_source` string from those fractions instead of the old constant
(an older model file without the attributes reads "none", never inferred).
`test_no_goal_plan_IS_a_floor_baseline_and_the_record_names_it` — which asserted the pre-fix defect
(`{"baseline:decel_1.5": 1.0}` + the −1.5 m/s² profile) and now fails by design — is replaced by
`test_default_goal_plan_is_no_longer_a_brake_by_tie_and_the_record_names_its_goal` (history kept in its
docstring): `decel_1.5` absent from the sources, chord speeds equal the held v0 (MEASURED on the fixture:
`baseline:hold_v0` on every window — an init property, heads ×1e-3), `goal_source == tactical_imagined`.
Both suites green in the mirror (§5). The mirror's `refa_v1.py` / `refa_v1_plan.py` are byte-identical to
the G: worktree, which equals HEAD (`e609a98`, "plan() imagines its default goal") — checked by blob hash.

## 9. Deliverable manifest

| artifact | where | state |
|---|---|---|
| `taniteval/tools/build_lead_block_b1.py` — the builder (pull + grid + block + cross-check) | repo (G:) + mirror copy | staged |
| `taniteval/tools/refav1_arm.py` — `--lead-block` join, `rec["refav1"]["distance_keeping"]`, goal provenance | repo (G:) + mirror copy | staged (modified) |
| `stack/tests/test_refav1_lead_block.py` — 11 tests | repo (G:) + mirror copy | staged |
| `stack/tests/test_refav1_arm.py` — scope-addition updates | repo (G:) + mirror copy | staged (modified) |
| this `RESULT.md` | repo (G:) | staged |
| `raw/b1_eval_lead_block.npz` — THE BLOCK (147 clips, 29,556 rows; sha256 `c0525943…0091898`) | repo (G:) + `C:\Users\Admin\tanitad-wt\_lead_b1\` | staged |
| `raw/b1_eval_lead_block.npz.report.json` — per-clip coverage, grid, registration, cross-check, obstacle stats, provenance | repo (G:) | staged |
| `raw/build.log`, `raw/pull.log`, `raw/pull_manifest.json` (per-clip sha256 of the pulled parquets), `raw/hf_tree_obstacle.json` (the 3,146-chunk listing with sizes + LFS sha256) | repo (G:) | staged |
| `raw/grid_crosscheck.json` — the 20-clip time-base cross-check | repo (G:) | staged |
| `raw/kinematic_dump_20clips_analysis.json`, `raw/kin.log`, `raw/cli_smoke.json`, `raw/cli_smoke_nolead.json` — the model-free real-data validation + CLI smoke records | repo (G:) | staged |
| the 145 pulled `obstacle.offline` parquets (77 MB) | **ONE place:** `C:\Users\Admin\tanitad-data\physicalai\labels\obstacle_offline_b1eval\` — re-pullable (`--pull`), sha256s banked | not in repo (size; derived from a public dataset) |
| the model-free 20-clip dump (`_lead_b1/kin_dump`) | **ONE place:** `C:\Users\Admin\tanitad-wt\_lead_b1\` — regenerable in 3 s from the local slice | not in repo (derived) |

Nothing was committed or pushed. No pod, no Thor, no GPU was touched.
