# The B1 EVAL agent join is 3-D now — and `center_z` is the CENTRE, measured against LiDAR, not assumed

**2026-09-17 · Architecture & Inference · CPU/IO only (the GPU was another agent's)**

## Headline

The refcv6 §6 box head could not be supervised or scored on height because the banked eval
join carried `{cx, cy, yaw, l, w, occ, track_id, cls}` and nothing else. **It can now.**
`b1eval_agents_3d.jsonl.xz` carries `cz` and `h` on **905,512 / 905,512 agents (100 %)**
across **26,394 lines / 139 clips**, on the **same key, the same frame index space and the
same bytes** — stripping the two new fields reproduces the 2-D join byte for byte on
**every** line, not a sample.

**The convention question the brief flagged is settled, MEASURED, not assumed:**

> `center_z` is the cuboid **CENTRE**. The base is `center_z − size_z/2`.
> Ground-standing vehicle bases sit at **−0.015 m** (n = 713,543) against a road plane an
> independent LiDAR pipeline puts at rig z = 0. The rival reading ("`center_z` is already
> the base") would float the same vehicles **+0.789 m** in the air.

⚠️ **ESCALATION — nothing blocks, two things need a decision:**

1. **`agent_cuboid_gt.py` is co-owned this session.** It was written by the refcv6-perception
   agent and is *staged but uncommitted* in its worktree. I copied it in verbatim
   (md5 `d13007c70fa8ea726a77c4f9a32af4a4` of the copy taken) and my change is a **pure
   append** plus 3 added lines in `__all__` — no existing line touched — so the two stagings
   should merge. **Someone must land them together, and the perception branch's copy of
   `semantic_map_gt.py` (an added `open_path`) is a separate, untouched-by-me change.**
2. **`box3d_head.py` does not exist on this branch** (it is the perception agent's). The
   end-to-end test "3-D join → `zh_targets` → `n_z > 0`" therefore **could not be run here**
   and is marked UNVERIFIED below. Everything up to the loader seam is tested.

---

## 1. What was built

| artefact | where | md5 |
|---|---|---|
| `b1eval_agents_3d.jsonl.xz` (12,628,320 B) | `TanitAD-artifacts/b1-agent-join-3d-20260917/` **(data, ONE place)** | `ff0e68fd41e86f6a30180ff6173b0685` |
| its sidecar `…jsonl.xz.meta.json` | same dir **and** sanitised copy at `raw/b1eval_agents_3d.meta.json` (this dir) | — |
| builder | `stack/scripts/build_b1_agent_join_3d.py` | — |
| loader seam | `stack/tanitad/data/agent_cuboid_gt.py` (appended section) | — |
| tests | `stack/tests/test_b1_agent_join_3d.py` (28 passed, 0 skipped with the env gates set) | — |

The md5 is **identical across three independent full builds** (evidence class: MEASURED).

**Line schema** — the 2-D schema with two keys appended per agent, nothing else changed:

```
{"clip_id": str, "frame": int, "frame_idx": int, "t_s": float,
 "agents": [{"cx","cy","yaw","l","w","occ","track_id","cls", "cz": f, "h": f}]}
```

`cz` = cuboid centre height, rig frame, +z up, z = 0 the road plane, 4 decimals.
`h` = `size_z`, 3 decimals (the join's own rounding for a coordinate and for a size).

### It is an ANNOTATION, not a rebuild

The builder reads the banked 2-D join line by line and re-emits each line with `cz`/`h`
appended. It never computes a frame, a key or a geometry. That is why the correspondence
control is an **identity** rather than a similarity, and why a consumer joining on the old
`(clip_id, frame)` key keeps working unchanged.

Why z is recoverable *exactly*: the join's `rig@sample → world → ego@frame` composition is
planar SE(2) — `bev_raster.ego_frame_agents` writes columns 0/1/2 and *"sizes and the occ
column pass through"* (`bev_raster.py:202-224`). `center_z`/`size_z` are invariant across
every step between parquet and join line.

---

## 2. The frame convention — the trap, and the measurement that settles it

`RETRACTION_LOG.md:15278` records: *PhysicalAI's tracked boxes carry z about 1 m under the
LiDAR ground; a display mask built from `z − h/2` blanked the road in front of every
vehicle.* Taking that at face value would have produced a wrong join. So it was measured.

**Reference (independent, VERIFICATION ONLY — the PI has ruled LiDAR is not a training
target):** `TanitAD-artifacts/bev-lidar-gt-b1eval-20260913`, a different sensor and a
different pipeline, whose own sidecar states
`frame_convention: "+x forward, +y LEFT, +z UP; origin rear axle on the road plane"` and
`ground_plane: "rig z = 0 … reproduced from LiDAR returns at −0.035 m"`, with a per-clip
measured `ground_peak_z_m` (median **−0.025 m** over 134 clips).

Corroborating, **not** used as the reference: the camera extrinsics in
`calibration/sensor_extrinsics` place `camera_cross_left_120fov` at rig z = +0.918 m and
`camera_front_tele_30fov` at +1.444 m — positive sensor heights, i.e. a rig origin at road
level, not at a sensor.

**Verdict (MEASURED, n = 713,543 ground-standing vehicle boxes in the built join):**

| reading | statistic | value |
|---|---|---|
| **`center_z` is the CENTRE** (adopted) | median(`cz − h/2`) | **−0.0146 m** |
| `center_z` is the BASE (rejected) | median(`cz`) | +0.7888 m |
| separation | — | 0.774 m |

**The retraction's number is reproduced, and explains itself.** A consumer that treats an
already-centre `cz` as a base and subtracts `h/2` again lands at
**−0.773 m** — "about 1 m under the LiDAR ground", the reported symptom. The defect was the
double subtraction, not the label frame.

### ⚠️ The honest caveat: rig z = 0 is a FLAT plane and a road is not

Ground-standing vehicle base by ego range (MEASURED, built join):

| range | n | base median | p10 | p90 | within ±0.2 m |
|---|---|---|---|---|---|
| 0–20 m | 86,068 | −0.006 m | −0.278 | +0.220 | **74.3 %** |
| 20–40 m | 145,795 | +0.004 m | −0.644 | +0.686 | 38.9 % |
| 40–60 m | 151,154 | −0.033 m | −1.076 | +1.075 | 24.6 % |
| 60+ m | 330,526 | −0.043 m | −1.610 | +1.664 | 15.8 % |

The median is unbiased at **every** range; the **scatter** grows with range because the road
climbs and falls away from the ego's flat rig plane. ⛔ **A consumer must not treat
`base ≈ 0` as a per-agent identity at range** — it is a corpus-level statement.

The worst clip in the LiDAR cross-check (sha12 `c16543c621dc`, Δ = −2.92 m) is exactly this:
it has **0** vehicle boxes inside 20 m and 20 of them at 20–60 m on a descent
(`abs_yaw_rate_p95` 0.347 rad/s, `ego_v_ms_median` 12.5 m/s), so the statistic was comparing
a near-field LiDAR ground estimate with labels 40 m down a hill. Its own LiDAR content
checks pass with no failures — the gap is road geometry, not a broken reference.

---

## 3. The controls

Nine controls; the builder **refuses to write** if a gated one fails (it did, twice, during
development — see §4). Full detail in `raw/b1eval_agents_3d.meta.json`.

| id | what it reads | scope | result |
|---|---|---|---|
| **C1** | the serializer round-trips the **banked bytes** | 26,394 lines | 0 fail |
| **C2** | `strip_zh(3-D line)` is **byte-identical** to the 2-D line | **ALL** 26,394 lines | 0 fail |
| **C3** | ground-standing base on the road plane, tol **0.20 m** | n = 713,543 | median **−0.0146 m** |
| **C4** | per-clip base vs **LiDAR-measured** ground | 133 paired clips | median \|Δ\| **0.167 m** |
| **C4n** | same, **near field only** (≤ 30 m, ≥ 5 boxes/clip) | 127 clips | median \|Δ\| **0.077 m**, 74.8 % within 0.2 m |
| **C5** | picked `track_id` **sequence** == the line's (membership **and** order) | **ALL** lines | 0 fail |
| **C6** | picked parquet row's `size_x/size_y` == the line's `l`/`w` | 905,512 agents | 0 fail |
| **C7** | *diagnostic, not a gate* — what the rounded `t_s` could have cost | 905,512 agents | see §4 |
| **C8** | the rival convention, computed rather than dismissed | n = 713,543 | verdict CENTRE, sep. 0.774 m |
| **C9** | the **recovered** frame time == the banked builder's | **ALL** lines | 0 fail, `max|Δt| = 0.0` |

**C3's tolerance is adopted, not invented**: 0.20 m is the LiDAR artifact's own
`content_checks.bands.ground_peak_z_m = [−0.2, +0.2]`, the programme's standing statement of
"this is the road plane" on this corpus (the `LEAD_SPEED_TOL_MPS` precedent). It cannot be
tuned to pass because it is quoted from elsewhere.

**C3 excludes `protruding_object`** — airborne by definition (base median **+0.867 m**,
n = 3,534) — and excludes `person`/`rider`/`stroller`/`animal`, which are not vehicle bases.
Pooling them would move the statistic the control is about.

⛔ **C4 is verification only.** No number derived from LiDAR reaches the artifact's agents.

---

## 4. Two things that were nearly wrong, and the controls that caught them

### (a) The rounded `t_s` could have moved a height by 1.22 m

The banked 2-D line carries `t_s` rounded to 4 decimals. The first build used it. **C7 then
refused the corpus**: 19 of ~13,600 picks in 400 lines sat within 1e-4 s of a tie — and the
worst tie was **two label samples 101 ms apart with 0.714 m between their z**. A rounding
that can flip which sample a height comes from is not a detail.

Fix: the frame time is now **recovered exactly** (`register_poses_to_time` re-run over the
raw poses; 138 clips, plus 1 clip via the lead block's `t0_s`, matching the 2-D build's own
1 registration-impossible clip). **C9 proves the recovery is the banked builder's own**:
`round(t_recovered, 4)` equals the banked `t_s` on all 26,394 lines with `max|Δt| = 0.0`.

C7 is kept as a **diagnostic** and now prices the route not taken: over the full corpus,
**924 of 905,512 picks (0.102 %)** lie inside the rounding's reach, worst-case candidate
z gap **1.223 m**. That is what exact-time mode bought.

⚠️ C7's `passed` field is `false` in the sidecar **on purpose** — it is the verdict on that
hypothetical, and the sidecar marks every control with `gated: true|false` so the reader
cannot mistake it for a failing gate.

### (b) A mutation table that never mutated

The first mutation sweep omitted the `--mutate` flag. All five runs reported
`mutate: "none"` and **identical md5s** — which is the only reason it was caught. Recorded
here because "guards need mutation, not inspection" fails the same way one level up: a
mutation table can itself be a control-shaped hole.

---

## 5. ⛔ The mutation table (re-run on the final code, 4,000 lines / 21 clips each)

Each defect must redden **exactly** the control(s) it is declared to redden — and leave the
others green, or the table proves nothing about which control is load-bearing. The test
`test_the_mutation_table` asserts both halves on a synthetic corpus built by the **2-D
builder itself**.

| mutation | C1 | C2 | C3 | C4 | C5 | C6 | C9 | C3 median | C4 median \|Δ\| |
|---|---|---|---|---|---|---|---|---|---|
| none | OK | OK | OK | OK | OK | OK | OK | −0.0097 m | 0.152 m |
| **emit-base-as-centre** (flip centre↔base) | OK | OK | **RED** | **RED** | OK | OK | OK | **−0.8703 m** | **0.925 m** |
| **shift-frame** (+1 frame of labels) | OK | OK | OK | OK | **RED** | **RED** | OK | −0.0007 m | 0.177 m |
| **misattach** (right set, wrong pairing) | OK | OK | OK | OK | OK | **RED** | OK | +0.0044 m | 0.165 m |
| **perturb-cx** (touch a carried field) | OK | **RED** | OK | OK | OK | OK | OK | −0.0097 m | 0.152 m |

Readings:

* **The centre/base flip goes RED by roughly `h/2`**, not by an arbitrary amount: C3 moves
  **0.861 m** — **4.3×** the 0.20 m tolerance — against a median `h/2` of 0.765 m on that
  subset. ⚠️ Not *exactly* `h/2`: the statistic is a median and the height mixture is
  bimodal (car 1.5 m, truck 3.5 m), so `median(cz − h) ≠ median(cz − h/2) − median(h)/2`.
  The test pins the shift into `[0.40, 0.75] × h_median`, which a control that merely
  failed would not satisfy, and which does not silently assume a unimodal corpus.
* **`misattach` is the failure C5 structurally cannot see.** It rotates the picks *after*
  C5 has compared the sequences: the agent set is right, the pairing is wrong, C5 stays
  green and only C6 sees it (138,933 / 139,128 agents). That is why C6 exists.
* **C3/C4 stay green under `shift-frame` and `misattach`** — correctly. They are statements
  about the *convention*, not about correspondence; claiming otherwise would be a control
  cited for an answer it cannot give.
* **`perturb-cx` reddens C2 on 3,950 of 4,000 lines**, proving the strip-identity is not
  vacuous.

The session-path guard in the test file is mutation-proven too
(`test_the_guard_would_catch_each_pattern` feeds it each forbidden pattern, assembled from
fragments so the guard cannot flag itself — it did, on the first run).

---

## 6. Census

**Coverage.** 139 clips / 26,394 lines / 905,512 agents, **zh_coverage 1.000000** —
0 agents lack `cz`/`h`. All 145 `obstacle_offline_b1eval` parquets carry `center_z` and
`size_z`, all `reference_frame = rig`, and **0** rows have a non-positive or non-finite
`size_z` (MEASURED over all **945,091** rows of the 145 parquets). The join's 905,512
annotated agents are a *different* population from that row count — one agent is one
(track, frame) pick across 139 clips, and no subset relation between the two numbers was
measured, so none is claimed.

**No clip lacks the columns entirely.** The 2 clips missing from the join
(139 of 141 offered) were already absent in the 2-D build for want of an obstacle parquet —
this builder adds no exclusions, and refuses to run if a 2-D line has no parquet.

**Heights and bases by class** (MEASURED, n as given, from the built join):

| class | n | `cz` median | **h median** | base median | base p10 | base p90 | in C3 |
|---|---|---|---|---|---|---|---|
| automobile | 676,084 | +0.767 | **1.511 m** | −0.013 | −1.160 | +1.189 | yes |
| person | 164,275 | +0.839 | 1.661 | +0.009 | −0.837 | +0.755 | no |
| rider | 22,977 | +0.885 | 1.704 | +0.047 | −1.116 | +0.745 | no |
| heavy_truck | 18,300 | +1.745 | **3.545 m** | −0.074 | −1.242 | +1.058 | yes |
| bus | 8,440 | +1.602 | **3.322 m** | −0.086 | −1.211 | +0.923 | yes |
| trailer | 7,514 | +1.613 | 3.231 | +0.030 | −1.176 | +1.405 | yes |
| protruding_object | 3,534 | +1.793 | 1.599 | **+0.867** | −0.178 | +1.755 | no (airborne) |
| other_vehicle | 3,136 | +1.404 | 2.701 | −0.038 | −0.800 | +0.879 | yes |
| stroller | 723 | +0.724 | 1.109 | +0.156 | −0.333 | +0.463 | no |
| animal | 460 | +0.805 | 1.333 | +0.209 | −0.634 | +2.219 | no |
| train_or_tram_car | 69 | +1.655 | 3.527 | −0.108 | −1.778 | +1.616 | yes |

**The sanity check the brief asked for passes**: a car is **1.511 m** (expected ≈1.5), a
heavy truck **3.545 m** and a bus **3.322 m** (expected ≈3), a person **1.661 m**. Nothing
disagrees, so there is nothing to report as wrong. `test_real_heights_are_the_physical_ones`
pins these bands so a future rebuild that drifts fails loudly.

This also **supersedes with 18× more data** the note in `box3d_head.ground_standing_z`
("MEASURED 2026-09-16 on 51,844 cuboids, bottom-face median 0.039 m automobile / 0.118 m
person / 0.051 m heavy_truck"): same conclusion, n = 945,091 parquet rows.

---

## 7. The loader seam — both paths, both tested

`stack/tanitad/data/agent_cuboid_gt.py` gains `AgentJoin3D`, `open_join3d`, `zh_for_frame`
and `base_from_centre` (appended; no existing line changed).

```python
join3d = open_join3d()                 # env TANITAD_AGENT_JOIN3D, or None
cz, h, mask = zh_for_frame(clip_id, frame, track_ids, join3d=join3d)
```

* **3-D join present** → the join's `cz`/`h`, `mask` True per track found.
* **absent** (no join, or that `(clip_id, frame)` not carried, or that track not on the
  line) → zeros with an **all-False mask** — exactly today's masked behaviour.

⛔ **Absence is a state, not a zero.** The values are zero only because an array needs a
fill; the mask is what a consumer honours. The two paths differ *only* in the mask, so
`box3d_head.zh_targets` / `box3d_set_loss` need no branch of their own and cannot train on
the fallback's zeros.

`AgentJoin3D.open` **refuses** a 2-D join handed to it under a 3-D name — silently indexing
it would mask every target while reporting that a 3-D join was found.

Indexing the full artifact costs **4.1 s**; track-id strings are interned, so the cost is
one string per track rather than one per agent, and `clips=` restricts the index.

**UNVERIFIED (cannot be run on this branch):** the end-to-end `zh_targets(tgt, cz, h,
mask=mask) → n_z > 0` step. `box3d_head.py` is the refcv6-perception agent's uncommitted
file and does not exist here. Everything up to and including `zh_for_frame`'s contract is
tested; the final hop is a one-line call whose input shape is asserted here.

---

## 8. Reproduce

```
set PYTHONPATH=<repo>/stack
python stack/scripts/build_b1_agent_join_3d.py ^
  --join-2d      "<repo>/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz" ^
  --obstacle-dir <data>/physicalai/labels/obstacle_offline_b1eval ^
  --eps-dir      <data>/refav1-eval141/eps ^
  --ego-dir      <data>/physicalai/labels/egomotion_alpamayo ^
  --lead-block   "<repo>/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz" ^
  --bevgt-dir    <artifacts>/bev-lidar-gt-b1eval-20260913 ^
  --out          <artifacts>/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz
```

82 s wall, CPU/IO only. Add `--mutate <name>` for the table in §5.

Tests, including the real-data gates:

```
set TANITAD_JOIN3D=<artifacts>/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz
set TANITAD_JOIN2D=<repo>/.../2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz
python -m pytest stack/tests/test_b1_agent_join_3d.py -q     # 28 passed
```

(The real-data suite includes the strip-identity across **all 26,394 lines** against the
banked 2-D join — asserted, not sampled.) `stack/tests/test_b1_agent_join.py` and
`test_obstacle_join.py` still pass unchanged (32 passed).

---

## 9. What this does NOT do

* **B1 TRAIN is untouched.** Only the 139-clip EVAL join was annotated. The same builder
  would serve `obstacle_offline_b1train` given a 2-D TRAIN join to annotate; there is none
  on this box.
* **No new geometry, no new corpus, no new key.** If a consumer's results change after
  switching to this artifact for anything other than z/h, that is a bug in the consumer.
* **The 2-D join is not overwritten.** `2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz`
  is byte-identical to what it was (md5 `3ddb42ecbd3926066795a94587af2aed`, its own sidecar's
  recorded digest) and is read-only input here.
* **The base is not a label.** `base_from_centre` is arithmetic on the labels; the head
  predicts `cz` and `h`.
