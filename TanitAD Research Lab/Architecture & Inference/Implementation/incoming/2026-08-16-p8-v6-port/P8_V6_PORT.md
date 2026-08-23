# P8 (BEV occupancy readout) ported to v6 — and the geometry mismatch it exposed

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Tier** T0-diagnostic
**Status** implemented + CPU-tested; **no GPU pass run** (Thor is training v6F S-W on the only GPU)

---

## 0. Why now, and the headline

`stack/scripts/train_v6_staged.py:259` declares `STAGE_GATE_SPEC["S-W"]["reported"] =
("P2", "P5", "P8", "O6_spectrum")` and names `scripts/train_p8_occupancy.py` as P8's owner
(`:265`). v6F S-W is training now and will hit that gate. P8 could not read a v6 checkpoint,
so the gate would have recorded P8 as *not-run, ImportError* — honest, and worth nothing.

**Two findings, in priority order.**

1. ⭐ **The geometry mismatch is real, is QUANTIFIED, and it is NOT a v6 problem — v6 makes it
   ~3.6× smaller.** P8's target is a Cartesian ego-frame raster (60 m × ±16 m, 0.5 m cells,
   `[120, 64]`); the thing that must fill it is a monocular forward camera, which sees an
   azimuth **wedge**. At v6's 120° cylindrical field **590 of 7 680 target cells (7.682 %) are
   outside the camera entirely**. At the **legacy 256×256 pinhole frame P8 was originally
   written against**, it is **2 126 cells (27.682 %)**, and *nothing below ~32.75 m is fully
   covered*. Every one of those cells is unanswerable from a vision-only latent, and scoring
   them measures the grid's corners.
2. **The port itself is thin, because `tanitad/eval/v6_probe_trunk.py` already existed** — its
   docstring names `p8_latents` explicitly. What was missing was the *call*, the *window*, the
   *frame*, and everything geometric. All four are now in the single entry point.

**P8 has a prior.** `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt2.json` is a complete
v5 run — gate (a) **PASS**, `ratio 0.93191` at k=10, over the canonical 881-window grid, on
`flagship-v5f-w120-30k`. Its absolute `iou_enc` is **0.020052**, and the JSON records **no frame**
(see escalations 2–3). The port's job is to make the v6 row exist *and* to make that provenance
gap impossible next time.

**Evidence class.** Every number in §1 is **MEASURED (ours)** — pure geometry, no corpus, no
checkpoint, no GPU. Artifact: `raw/p8_v6_geometry.json`, reproduced by
`code/p8_geometry_census.py`, and pinned in `stack/tests/test_p8_v6.py`.

---

## 1. The geometry mismatch, stated precisely

### 1.1 What P8 consumes and emits (established from source)

| | |
|---|---|
| **latent in** | `z [B, S]`, `S = world.state_dim`. The head is `Linear(S → ch0·nx/8·ny/8)` then two `ConvTranspose2d` — `train_p8_occupancy.py:133-136`. **It flattens the state; it reads no cell layout.** |
| **target out** | occupancy logits `[B, 120, 64]` — `BEVGrid(x_fwd_m=60, y_half_m=16, cell_m=0.5)`, `bev_raster.py:85-99`. Ego frame, **+x forward, +y LEFT**, row 0 at the ego origin (`bev_raster.py:50-56`). |
| **GT** | `obstacle.offline` cuboids via the pod-built join file, rig-frame at their own timestamp — `train_p8_occupancy.py:189-203`. |
| **latent source (v5)** | `eval_flagship_v4.load_v1_from_ck`, rolled by `metric_dynamics.rollout_transitions`; `trans[k-1][1]` is `ẑ_{t+k}` — `train_p8_occupancy.py:698-700`. |

### 1.2 What v6 actually exposes

| | |
|---|---|
| `V6Stack.encode_window(frames)` | `[B, W, d_op]` — `v6.py:1420-1424`. Same name, same shape as v5. |
| `d_op` | **derived, never set**: `n_cells × d_readout` = `grid × grid_w × d_readout` — `v6.py:999-1003`. Catalog `ReadoutConfig(grid=4, d_readout=128)`, `grid_w=None` (`v6.py:908-909`, `train_v6_staged.py:1740-1742`) ⇒ **4×4 cells × 128 = 2048**. |
| `SpatialGridReadout` | pools the ViT token grid to `grid × grid_w` — `readout.py:114-125`. MEASURED: 256×640 / patch 16 → **16×40 tokens**; 16 % 4 = 0 and 40 % 4 = 0 ⇒ `exact_pool=True`, `AvgPool2d((4, 10))`. Each readout cell = **64 px × 160 px**. |
| `predictor_op` | `OperativePredictor.forward(states, actions, intent=None) -> dict[int, Tensor]`, `action_dim=3` — so `rollout_transitions`' `predictor(s, a)[1]` works unchanged. |
| `readout_grid_ranges` | per-row nominal range, **row 0 = image TOP = FAR**, geometric spacing over **3 → 80 m**, and it is **ESTIMATED — a declared monotone image-row prior, NOT calibrated depth** (`v6.py:247-273`). |

⇒ `d_op = 2048` is *the flagship width*. `BEVOccupancyHead(2048)` lands at **0.985 M** params,
inside the pre-registered `PARAM_BAND (500k, 2M)`. **The port needs no head re-tuning.**

### 1.3 The mismatch, MEASURED (`raw/p8_v6_geometry.json`)

| arm | out-of-field cells | % of target | first fully-visible row | readout-column map |
|---|---|---|---|---|
| **v6F 256×640 cyl 120°** | 590 / 7 680 | **7.682 %** | row 18 (x = 9.25 m) | exact (16×40 → 4×4) |
| v5 sub-frame 176×624 cyl 117° | 626 / 7 680 | 8.151 % | row 19 | **REFUSED** (39 token cols ∤ 4) |
| legacy 256×256 pinhole (`CANONICAL_256`) | **2 126 / 7 680** | **27.682 %** | **row 65 (x = 32.75 m)** | exact |

Four consequences, each independently checked:

1. **The out-of-field cells are ALL near.** Max x of an out-of-field cell is **8.75 m**; the
   threshold for a fully-visible row is `15.75 / tan 60° = 9.093 m`. Within the band x < 9 m
   they are **51.2 %** of the cells. That band is exactly where headway / distance-keeping
   lives — the family CLAUDE.md makes binding and where 88.7 % of the oracle gap sits.
2. **Half of v6's readout WIDTH answers 15 % of the target.** The four readout columns are
   exactly 30° azimuth wedges (cylindrical ⇒ image column linear in azimuth, `calib.py:90`).
   The two OUTER wedges hold **1 186 cells (15.44 %)** and reach only **27.25 m**; the target's
   far half (x ≥ 30 m ⇒ |az| ≤ 27.5°) lives **entirely** in the two inner wedges.
3. **The row orders are OPPOSITE.** v6 row 0 = far (80 m); P8 row 0 = ego origin. A cell-aware
   port that aligned them row-for-row would map far onto near.
4. **v6's own row prior overhangs the target on both ends**: **5.00 %** of the target (384 cells,
   x < 3 m) is nearer than v6's declared `near_m`, and row 0's 80 m is off the 60 m grid — and
   that prior is ESTIMATED, so it cannot be used to justify a cell alignment either way.

### 1.4 What I did NOT do, and why

**I did not change the head to be cell-aware.** The flat `Linear(state_dim, …)` discards v6's
cell layout — which is precisely the DINO-WM lesson v6 encodes structurally ("pooling is where
geometry goes to die", `v6.py:1426-1434`). A cell-aware head is the natural follow-on. It is
**not** in this port because adopting it silently would confound *"P8 now runs on v6"* with
*"P8 has a different decoder"*, and every banked P8 number was produced with the flat head. It
is a declared arm, listed in §6.

**I did not switch the gate to the masked metric.** Both are always computed and reported;
`--fov-gate` selects the gated one and **defaults to `all`**, so no banked verdict moves. See
§3 for which one to quote.

---

## 2. The port

**One entry point, two trunk generations.** `scripts/train_p8_occupancy.py` now routes through
`tanitad.eval.v6_probe_trunk.load_trunk_auto` — the seam `probe_latent_state.py:818-821`
already uses. A second P8 would have drifted within a week.

Four things now come off the trunk instead of from a v5 default:

| | was | now |
|---|---|---|
| construction | `load_v1_from_ck` (v5 `WorldModel`) | `load_trunk_auto` → `V6ProbeTrunk` for `{"stack": …}` ckpts, v5 path otherwise |
| causal window | `cfg.predictor.window` = **8** | `getattr(world, "window", …)` = **6** for v6 |
| model frame | resolved from the CLI only | resolved from the **run's own args** (`run_frame_of`); a CLI that contradicts it is **REFUSED** with the exact flags to pass |
| input channels | assumed | cross-checked against the trunk; a mismatch is refused |

### 2.1 Files changed

| file | change |
|---|---|
| `stack/scripts/train_p8_occupancy.py` | v6 trunk load; window/frame/channel guards; FOV census + mask; `iou_*_infov` twins; `--fov-gate`; `--hold-action-control`; `assert_raster_shape` on every metric; `p8_latents_ex` (`p8_latents` unchanged) |
| `stack/tanitad/data/bev_raster.py` | **new, pure numpy**: `cell_centers_xy`, `cell_azimuth_rad`, `fov_mask`, `readout_column_index`, `readout_row_ranges_m`, `fov_census` |
| `stack/tanitad/eval/v6_probe_trunk.py` | `run_frame_of`; `V6ProbeTrunk` gains `grid_shape` / `d_readout` / `n_cells` / `token_grid` / `in_channels` / `frame` / `is_v6` / `cells()` |
| `stack/tests/test_p8_v6.py` | **new**, 41 CPU tests |

### 2.2 The refusals (the task's "must not silently reshape")

| refusal | trigger | why |
|---|---|---|
| `assert_raster_shape` | logits `[B,120,64]` vs target `[B,120,1]` / `[B,1,64]` / `[B,60,64]` | torch would **broadcast** the first two and return an IoU against a grid nobody specified |
| `readout_column_index` | `token_w % n_cols != 0` | non-tiling ⇒ `AdaptiveAvgPool2d` ⇒ **overlapping** bins ⇒ a BEV cell belongs to two readout columns; an index would be fiction. **Fires on the real 176×624 sub-frame** (39 ∤ 4) |
| geometry contradiction | ckpt frame ≠ CLI-resolved frame | feeding an encoder a field it never saw |
| channel contradiction | trunk `in_channels` ≠ dataset channels | wrong stack depth |
| `BEVOccupancyHead.forward` | `z.shape[-1] != state_dim` | pre-existing; now also covered for the v6 `d_op` path |
| `fov_census` | column mapping refused | records `exact: False` + the reason **instead of dying** — the FOV mask does not depend on tiling and stays valid |

### 2.3 Backwards compatibility

- `p8_latents` keeps its **3-tuple** return. Four live callers unpack three values
  (`probe_latent_state.py:594`, `lf0_bev_lead.py:232`, `p8_bev_reel.py:220`, and this file);
  widening the arity would have been a silent break for a convenience. `p8_latents_ex` is the
  4-tuple with the control arm; `p8_latents` is a one-line view onto it — one implementation.
- With no `fov` and `--fov-gate all` (the defaults), the JSON's incumbent keys and the gate
  verdict are **unchanged**.
- `p8_gate_dict(per_k)` still defaults to the all-cells set.

---

## 3. ⚠️ ADMISSIBILITY — where every probe-time input comes from

The PI's binding rule (2026-08-03): **labels may use anything; inference is VISION-ONLY**, and
for any head, *ask whether its inputs at inference include something the label was derived from*.

| input at probe time | provenance | derives the target? |
|---|---|---|
| `frames` | camera | no |
| `future_frames` | camera (future) | no — and it feeds only the **reference** (encoded) arm, not the arm under test |
| `actions`, `future_actions` | **PRIVILEGED ego channel** | not the target's **content** |
| `pose_last[:,3]` (v0 ego speed) | **PRIVILEGED ego channel** | not the target's **content** |
| GT raster | `obstacle.offline` cuboids, **rig-frame at their own timestamp** — computed with **no ego state** | — |

**The trunk stays vision-only.** `encode_window` is fed frames and nothing else; the privileged
channels reach only the **predictor's action conditioning**, which is what a world model is
defined to consume.

⚠️ **The one residual dependence, stated rather than buried.** The target at *t+k* is expressed
in the **ego frame at t+k**, and the true future actions determine that pose. So the privileged
inputs supply the target's *coordinate frame*, though never its *content*. **That is the
defining property of a T0 probe, not a defect** — and it is exactly why `EVAL_DOCTRINE.md`
forbids reading a T0 number as driving performance.

**Verdict: admissible as T0. NOT admissible as evidence about a deployed vision-only inference
path.** This is now written into `p8_gate.json` as `input_provenance`, so the stamp travels
with the number instead of living in this document.

### 3.1 The control that makes it measurable — `--hold-action-control`

Rather than argue about the size of that dependence, roll the same latents under **held**
actions (`hold_future_actions`) and report `iou_hold*` and `hold_over_pred`. This is the
instrument §1.12 already used (S-curve reproduction **97.9 % open-loop vs 0.0 % hold-action**).
Cost: **one extra predictor roll, no extra encoder pass.** Default OFF so the incumbent output
is unchanged; **recommended ON for the S-W gate.**

⚠️ **The control is VACUOUS on an untrained trunk** — MEASURED and pinned
(`test_the_hold_control_is_vacuous_on_an_UNTRAINED_trunk`): the predictor's action conditioning
is FiLM with **zero-init** (`predictor.py:39`), so at random init actions have *exactly* no
effect and the two arms coincide bit-for-bit. `hold_over_pred == 1.0` on an untrained trunk
means *"the conditioning has not trained"*, **not** *"the scene is action-independent"*.

### 3.2 Which number to quote

The retention **ratio** is the robust half — out-of-field cells depress numerator and
denominator alike, so gate (a) is fairly insensitive to the mask. The **absolute** readout
quality is not: quote `iou_*_infov`. Both verdicts are written to `p8_gate.json`
(`gate_a` and `gate_a_other_cell_set`), so a verdict that would flip on the other cell set is
visible as a fact rather than discovered later.

---

## 4. Tests

`stack/tests/test_p8_v6.py` — **42 tests, CPU, no corpus / checkpoint / GPU / join file.**

- field mask vs an **independent analytic loop** (not the vectorised function under test);
- the 590 / 7 680 census, the near-band concentration (51.2 %), the legacy-frame 2 126 / 7 680;
- cylindrical columns are exactly 30° wedges; pinhole columns are **not** the same cells;
- the non-tiling **refusal** on the real 176×624 sub-frame, and `fov_census` recording it
  instead of dying;
- `readout_row_ranges_m` pinned numerically to v6's torch `readout_grid_ranges`, plus the
  **inverted row order** and the 3–80 m overhang;
- every metric refuses a broadcastable-but-wrong raster shape;
- masking **changes the answer** (occupancy only outside the field: unmasked IoU 1.0, masked
  NaN) and **does not** change a fully-in-field case;
- gate cell-set selection: default `all`, `in-fov` branch, not-computable branch, unknown-metric
  refusal;
- **`p8_latents_ex` run end to end against a real (tiny) `V6Stack` through `V6ProbeTrunk`** —
  shapes, finiteness, the hold arm, and the two arms coinciding when the future actions are
  already held;
- `BEVOccupancyHead` in-band at v6's `d_op = 2048`, refusing another width.

**Suite:** `PYTHONUTF8=1 …/python.exe -m pytest` from `stack/` →
**2961 passed · 0 failed · 17 skipped · 2 xfailed** (382 s). Brief baseline was 2810/0/17/2;
the delta is this file's 42 plus a concurrent stream's additions (`test_v6_ladder_edges.py` and
siblings). **Zero failures, zero regressions.**

---

## 5. ⛔ THE GPU PASS THIS STILL NEEDS (schedule at a deliberate training pause)

Nothing below has been run. **P8 has no number yet on any v6 checkpoint.**

### 5.1 Prerequisite: the join file — it EXISTS; verify it is on the v6 pod

`--raster-source join-file` needs a jsonl of `{clip_id, frame_idx, agents[]}` in **episode index
space**, built from `obstacle.offline` (`train_p8_occupancy.py:25-44`). The `episode` source
refuses by design (no provider carries `.agents`).

**MEASURED** (`…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt2.json`, `raster_source`):
that file was built and used — `/workspace/data/p8_join/combined140.jsonl`, **137 clips,
26 084 records, `occlusion_flags: true`**. So the visible/occluded cell-recall split is
available too, and P8's full path has run end to end before.

⚠️ **Two things to verify before scheduling, not assume** (this is a pod-side artifact from
2026-08-07 and pods drift):
1. the file is still present on the pod that will run v6F — if not, rebuild via `clip_id` +
   registered frame time → `taniteval.lead_source.register_poses_to_time` →
   `bev_raster.agents_at_time` (**ESTIMATED 1–3 h**, CPU);
2. its 137 clips still cover the v6 corpus's episodes. The run's own coverage census prints
   `covered/total` and **refuses to train on zero**, so a mismatch fails in seconds rather than
   after the expensive part.

### 5.2 The run

```
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 scripts/train_p8_occupancy.py \
    --ckpt /workspace/experiments/v6f-sw/ckpt_stepNNNNN.pt \
    --v2-cache     /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
    --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
    --raster-source join-file --join-file /workspace/data/p8_join/agents.jsonl \
    --hold-action-control --steps 3000 \
    --out /workspace/experiments/p8-occupancy-v6f
```

⚠️ Pass **no `--v2-subframe`** for a v6F trunk trained at the full 256×640 — the frame guard
will refuse the mismatch and print the exact flags.

### 5.3 Cost — anchored on a MEASURED prior P8 run, not on an extrapolation

⭐ **This script has been timed.** `p8_gate_attempt2.json` (3 000 steps, bs 32, on
`flagship-v5f-w120-30k`): **`wall_s` 23 508.4 s = 6.53 h**, of which the **mini-eval was
`wallclock_s` 255.9 s = 4.3 min**. Attempt 1: 21 099.5 s = 5.86 h, mini-eval 233.7 s.
**MEASURED (ours; artifacts = those two JSONs).**

| phase | v5 MEASURED | v6 ESTIMATED | note |
|---|---|---|---|
| head training, 3 000 steps × bs 32 | **5.9–6.5 h** | **5–7 h** | ⚠️ the bottleneck is **cold MooseFS payload loads**, not FLOPs — the trunk is frozen, so there is no ViT backward. `--eps-per-batch` exists precisely to cut those loads ~8×. A "forward-only so it must be fast" intuition is **refuted by this measurement.** |
| mini-eval (881-window grid) | **4.3 min** | **5–7 min** | +1 predictor roll with `--hold-action-control`; **no** extra encoder pass |
| **total** | **≈ 6.5 h** | **≈ 5–7 h** | v6's window is 6 vs v5's 8 ⇒ ~25 % fewer frame decodes per window, pushing the estimate down; a different pod's MooseFS pushes it anywhere |

⚠️ The v6 column is **ESTIMATED** — same script, same grid, same batch, a trunk of the same
size class, on a *different host*. The MooseFS dependence is exactly why it is a range and not
a number.

⭐ **Cheap re-runs.** The mini-eval is 4 min against 6 h of head training, and the head is saved
every `--save-every` steps to `p8_head.pt`. If a v6 head is ever trained, re-scoring it (a new
`--fov-gate`, the hold control, a different `--ks`) is minutes — so **train the head once, at
the pause, and bank `p8_head.pt`.**

**Scheduling.** ⛔ Do **not** co-schedule with S-W: CLAUDE.md forbids adding GPU/RAM load to a
training pod, and this is a full eval-weight job. Run it at a **deliberate pause** — e.g. the
S-W → S-T stage boundary, where the trunk is frozen anyway and P8 is exactly a
frozen-trunk-stage report. Use `OMP_NUM_THREADS=6` (7 concurrent arms once sat at 0–6 % `sm`
for 50 minutes without it), small batch, few workers on Thor.

**If it must run before the pause:** the *geometry* half of this port already ran — §1's census
is complete, CPU-only, banked, and is the part that decides whether P8's number will mean
anything. It does not wait on a GPU.

---

## 6. Declared follow-on arms (NOT done here, deliberately)

1. **Cell-aware head** (`--head-layout cellwise`): read v6's `[C, d_readout]` cells and decode
   each 30° wedge into its own BEV sector. Directly tests RC1 ("lead geometry lives in the
   cells and dies in aggregation"). Must be run *against* the flat head, or it is a confound.
2. **Wedge-native target**: score a polar (azimuth × range) raster instead of a Cartesian one,
   which removes the out-of-field cells *by construction* rather than by masking. Breaks
   comparability with every banked P8 number — hence an arm, not a change.
3. **Calibrated visibility mask**: the corpus **does** carry `camera_intrinsics` and
   `sensor_extrinsics` (`physicalai.py:153-154`), so the vertical field / hood occlusion could
   replace today's horizontal-only *necessary* condition with a real one. Pod-side.
4. **Calibrated cell→metre table** to replace `readout_grid_ranges`' ESTIMATED prior. v6's own
   docstring says a calibrated table can be dropped in without touching O2.

---

## 7. Deliverable manifest

| artifact | path |
|---|---|
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-p8-v6-port/P8_V6_PORT.md` |
| geometry census (MEASURED) | `…/2026-08-16-p8-v6-port/raw/p8_v6_geometry.json` |
| census reproducer | `…/2026-08-16-p8-v6-port/code/p8_geometry_census.py` |
| P8 entry point (v6 path) | `stack/scripts/train_p8_occupancy.py` |
| BEV camera-field geometry | `stack/tanitad/data/bev_raster.py` |
| v6 trunk adapter (extended) | `stack/tanitad/eval/v6_probe_trunk.py` |
| tests | `stack/tests/test_p8_v6.py` |

All staged in the working tree (`git add`), **never committed and never pushed by this agent** —
per `AGENT_OPERATING_STANDARD.md`. Nothing lives only on a pod or a worktree.

⚠️ **A concurrent sweep committed the index mid-session.** HEAD moved to `41057e5` while this
work was in progress and now contains six of the seven artifacts; the seventh (a docstring edit
to `train_p8_occupancy.py`) is staged against it. Verified file-by-file that nothing was lost:
`git ls-tree -r HEAD` lists the new files and HEAD's `bev_raster.py` carries `fov_census`.

⛔ **A verification note worth keeping.** `git ls-files --cached <path>` — the check CLAUDE.md
prescribes for the silent-`add`-no-op trap — is **not sufficient for a MODIFIED tracked file**:
it proves the path is in the index, not that *your* version is. It reported "staged" for a file
whose index blob was the pre-edit one. The check that actually settles it is
`git ls-files --stage <path>` vs `git hash-object <path>` — **compare the blob hashes.**

---

## 8. Escalations (do not leave these in a doc — they need an owner)

1. ⚠️ **Verify the join file is on the v6 pod before scheduling.** §5.1. It exists
   (`combined140.jsonl`, 137 clips, MEASURED) but is a 2026-08-07 pod-side artifact, and pods
   drift. The run refuses on zero coverage, so this fails fast rather than expensively.
2. ⚠️ **THE BANKED P8 RESULT DOES NOT RECORD THE FRAME IT WAS SCORED ON — and the port fixes
   that.** `p8_gate_attempt2.json` PASSES gate (a) (`ratio 0.93191`, `iou_enc 0.020052`,
   `iou_pred 0.018686`, `n_enc 797` at k=10, 881-window grid, `tau* 0.7`) on
   `flagship-v5f-w120-30k` — but the JSON carries **no frame, no HFOV, no projection**, so the
   out-of-field fraction that number was computed under is **not recoverable from the
   artifact**. From the script's own documented v5f invocation it is almost certainly the
   **176×624 cyl 117° sub-frame ⇒ 8.151 % out-of-field** (INHERITED, not verified), and
   *certainly not* the legacy square frame (27.68 %) — that arm's `w120` naming rules it out.
   The v6 path now writes the full `geometry` census into `p8_gate.json`, so this ambiguity
   cannot recur. **Someone should confirm the v5f attempt-2 frame from its `train_log.jsonl`
   (which does carry `args`) and annotate the banked JSON.**
3. ⚠️ **Masking is a correctness fix, NOT an explanation for that 0.0201.** Removing ~8 % of
   cells cannot turn 0.02 into a healthy IoU. The banked readout is weak on its own terms and
   the retention ratio (0.93) is a ratio of two small numbers — the same shape as the attempt-1
   trap the `TAU_GRID` sweep was added to catch. Whatever the v6 run returns, **quote
   `iou_*_infov` for absolute quality and treat a high ratio over a tiny `iou_enc` as
   uninformative**, exactly as `p8_gate_dict`'s `iou_enc <= 0` branch already refuses.
4. ⚠️ **`bev_raster` is also consumed by `lf0_bev_lead.py` and `p8_bev_reel.py`, which I did
   not audit.** Both now inherit the field-mask helpers but neither applies them. If either has
   banked a number on a square pinhole frame, 27.68 % of its grid was unobservable.
5. ⚠️ **`--fov-gate` defaults to `all` — a decision the PI may want to reverse.** I chose the
   conservative default (nothing banked moves). The in-FOV metric is the more honest absolute
   number and the JSON reports both, but the *pre-registered* gate reads whichever flag says.
6. **No file owned by another live stream was touched.** `taniteval/taniteval/hierarchy.py`,
   `four_families.py`, `eval_four_families.py`, `e_wc2_sigma_star.py`, `physicalai*.py` are all
   untouched. The port needed no change in any of them.
