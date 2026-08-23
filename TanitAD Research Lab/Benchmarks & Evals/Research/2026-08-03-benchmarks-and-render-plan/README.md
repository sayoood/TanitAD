# 2026-08-03 — Community closed-loop benchmarks + the render improve-and-validate plan

Two PI questions answered together because both are about **making our numbers comparable to the outside
world**: which external closed-loop benchmark we adopt, and how we raise *and prove* render fidelity.

| file | what it is |
|---|---|
| **`Q1_CLOSED_LOOP_BENCHMARKS.md`** | the benchmark survey, the per-model geometry verdict, the recommendation (**AlpaSim first, Bench2Drive second**), the two-layer combination with our four-family panel, and 5 pre-registered falsifiers |
| **`Q2_RENDER_FIDELITY_PLAN.md`** | the render plan ordered by gain/effort (R0…R5), the ceiling decomposition, the standing validation test, and the "still improved but the video did not" hypotheses |
| **`bench_geometry_check.py`** | ⚙️ **executable** — computes whether a benchmark's camera rig can feed v5f (120° cylindrical 176×624) and v1/REF-C (256² at `F_REF=266.0`): azimuth coverage, angular sampling ratio, cylindrical-stitch parallax. `--selftest` → **6/6** |
| **`bench_geometry.json`** | its output (the matrix quoted in Q1 §3) |

## The four things a reader should take away

1. **Adopt AlpaSim / the AlpaSim E2E Closed-Loop Challenge 2026 first** — the only candidate that is
   genuinely closed-loop, on **our own camera rig and data distribution**, with a live 2026 leaderboard and
   a published external reference **that carries an interval** (0.73 ± 0.01 over 910 scenarios, PUBLISHED).
   We are already ~80 % integrated (MEASURED: n=12 and n=37 suites have run).
2. **Its score cannot diagnose anything** — MEASURED from our own artifact,
   `pass = collision_at_fault==0 ∧ offroad==0`, `score = min(clamp(progress_clipped_rel,0,1)/0.8, 1.0)`.
   No heading, curvature, yaw-rate, manoeuvre or route term. ⇒ It could **not** have detected the case where
   two of our four lateral metrics separated and two did not. **The community score buys comparability; the
   four families buy diagnosis. Publish both or neither.**
3. **Geometry is computed, not asserted.** v5f on the PhysicalAI/NuRec rig = **1 camera, 3.00× down-sample,
   0 px stitch error**; on CARLA = exact too (we may declare co-located cameras); on NAVSIM/nuScenes = a
   3-camera stitch with **~18–21 target-px parallax error at 5 m**, worst in the near field. The v5f blocker
   is **not** geometry — it is that v5f has no usable checkpoint (step 2,300, degrading).
4. **Render: the reference video is 6 frames off on `00040136` (5 on `7c72937c`), worth +0.1797 grad-NCC
   for free**, and we have been rendering **training views all along** (the scenes are reconstructed from
   6 views including the very camera we score against). So the ceiling question is not "render a training
   view" — it is *"what does a second renderer get on the same pose"* and *"what does grad-NCC pay for two
   consecutive real frames"*. Both are specified in Q2 §R1.

## DONE

- ✅ Surveyed and characterised **NAVSIM v2, nuPlan (+nuPlan-R), Bench2Drive, CARLA Leaderboard 2.0,
  AlpaSim/Alpamayo, HUGSIM, NeuroNCAP** plus the 2026 arrivals (HiDrive, Bench2Drive-Robust, DriveE2E,
  MDrive, Fail2Drive, Safe2Drive) — closed-loop vs proxy, scoring function, interval or not, sensor
  contract, cost.
- ✅ **Recovered AlpaSim's actual scoring function from our own run artifacts** rather than from prose —
  `score_criteria` + the 33-key metric bundle (`min_ade@{0.5,1,2.5,5}s(gt)`, `min_distance_to_obstacle_m`,
  `min_distance_to_lane_boundary_m`, `wrong_lane`, `plan_deviation`, …).
- ✅ Established that the **AlpaSim E2E Closed-Loop Challenge 2026 is live** (2026-06-15 → leaderboard
  closes 2026-10-31, rules freeze 2026-09-15; PAI-AV and nuPlan tracks; Docker submissions run by organiser
  workers on private scenes) — and that this means **the NGC-gated renderer does NOT block a submission**.
- ✅ **Built and ran an instrument** (`bench_geometry_check.py`, self-test 6/6) turning "can this benchmark
  feed our models?" into three computed tests; produced the matrix for 4 rigs × 2 models.
- ✅ Judged every benchmark against the PI's own criterion — *could its score have seen what our four
  families saw* — citing the MEASURED open-loop panel where four lateral metrics split two–two.
- ✅ Wrote the two-layer combination design (external scalar + our panel, same rollouts, shared estimator)
  and the per-model integration plan with the geometry obstacle stated per model.
- ✅ Render: assembled the MEASURED state, **flagged that every published absolute needs re-baselining**,
  and produced an ordered plan with the ceiling decomposition (metric floor / second renderer /
  cross-camera control), the residual-attribution design, the lever ranking, and the standing test.
- ✅ Identified a **new, file:line-grounded lever**: production renders with appearance basis `f0` at
  `tau = 0.0` — the DC term, frozen for the whole clip — while the asset ships a time embedding and
  per-layer Fourier features. On a **night** clip this is likely the largest untested renderer-side lever.
- ✅ Proposed the answer to *"the still looked better but the video did not"*: full-clip per-frame scoring,
  a **differential flicker** instrument, and **the side-by-side video (with the per-frame grad-NCC burned
  in) as the standard artifact** — with `git add -f` called out because `.gitignore:24` excludes `*.mp4`.
- ✅ 5 falsifiers pre-registered for Q1, one per plan item in Q2, both outcomes committed in advance.

## NOT DONE (and why)

- ⛔ **No benchmark was actually run, and no render was executed.** This stream is research + one
  instrument; the render files belong to a sibling stream and I did not edit them (§ownership note in Q2).
- ⛔ **The challenge's exact metric spec and container requirements were not retrieved** — the public
  overview names a "capability score" and a "safety metric" and points at a metrics doc I could not fetch.
  That our `score_criteria` equals the challenge's is **HYPOTHESIS** (falsifier F1).
- ⛔ **NAVSIM / nuScenes per-camera HFOV, yaw and baseline are ESTIMATED**, flagged `estimated_inputs=true`
  in the JSON. The 0 px vs ~20 px ordering is robust; the exact pixel counts are not (falsifier F5).
- ⛔ **`camera_names()` was not run** on a scene's `rig_trajectories.json` — the 6-view reconstruction claim
  stands on the dataset card alone (one probe). One command on Thor; it is item 0 of Q2.
- ⛔ **No re-baselining was performed.** R0 is specified, not executed — it is the renderer stream's file.
- ⛔ **v5f's driver adapter was not written.** It is 0-GPU work and belongs to whoever owns the AlpaSim
  adapters; the exact resample (916.7 → 305.58 px/rad, then the 176×624 crop) and its self-check are
  specified in Q1 §5.5.
- ⛔ **NVIDIA's 0.73 ± 0.01 is not yet a comparable number** — the resampling unit behind the ± is unread.

## Manifest

| artifact | where it lives | state |
|---|---|---|
| `README.md` (this file) | `TanitAD Research Hub/Benchmarks & Eval/Research/2026-08-03-benchmarks-and-render-plan/` | repo, **staged, not committed** |
| `Q1_CLOSED_LOOP_BENCHMARKS.md` | same directory | repo, **staged, not committed** |
| `Q2_RENDER_FIDELITY_PLAN.md` | same directory | repo, **staged, not committed** |
| `bench_geometry_check.py` | same directory | repo, **staged, not committed** — runs anywhere with python3, no deps |
| `bench_geometry.json` | same directory | repo, **staged, not committed** — regenerable via `python bench_geometry_check.py` |

Nothing in this stream lives on a pod, in a worktree, or only in an agent's context.

## Escalations (integration requests, not doc-notes)

1. **Renderer stream** — R0: apply the per-scene offset rule in `render_quality.py:load_refs` and add the
   ±3 neighbour scan to the negative control (`MIN_WRONG_GAP = 40` makes the current control blind to this
   defect *by construction*), then re-baseline panels 1–6. **Do not hard-code +6** — scene `7c72937c` is +5.
2. **Renderer / video stream** — make the side-by-side **video with the per-frame grad-NCC trace** the
   standard artifact, staged with `git add -f`.
3. **AlpaSim adapter owner** — write the v5f cylindrical-resample driver + `px/rad` self-check now; it is
   0-GPU and it is the only thing between a v5f checkpoint and an external number.
4. **Orchestrator / PI** — the challenge's **rules freeze is 2026-09-15** and the public leaderboard closes
   **2026-10-31**. A submission decision has a calendar, not just a priority.
