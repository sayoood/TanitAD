# The `bev_raster` consumer audit — finishing what the P8 v6 port started

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Tier** T0-diagnostic (geometry only)
**Status** implemented + CPU-tested; **no GPU used** (Thor is training v6F on the only GPU)

---

## 0. The headline, in priority order

The P8 v6 port MEASURED the whole-grid mismatch and escalated (#4) that
*"`bev_raster` is also consumed by `lf0_bev_lead.py` and `p8_bev_reel.py`, which I did not
audit. If either has banked a number on a square pinhole frame, 27.68 % of its grid was
unobservable."* This closes that, and the two-file scope turned out to be one file short.

1. ⭐ **LF0's banked verdict CANNOT MOVE, and that is MEASURED, not assumed.** LF0 never
   scores the grid — it walks a ±1.5 m corridor — and at the frame it actually ran on
   **0 of the 708 cells it scanned lie outside the camera field**. Masked and unmasked
   reads are identical *by construction*. **RC1 stays REFUTED.**
2. ⭐ **The audit's own scope was too small — the third consumer is a PUBLISHED PAPER
   FIGURE.** `Paper/figures/make_lf0_bev_panels.py` consumes the geometry by *restating it
   inline* (`NX, NY = 120, 64`), so it matched no import search, and it draws all 7 680
   cells unmasked. Its cell counts **are** affected and are **UNRECOVERABLE** (§3.2).
3. ⭐ **The banked P8 frame is RECOVERED** (port escalation #2) — not from the run's
   `train_log.jsonl` (that host is gone, 3 probes) but from the **launch chains banked in
   this repo**. It is the **176×624 cyl 117 ° sub-frame ⇒ 8.151 % out-of-field**, which
   confirms the port's INHERITED guess and promotes it to MEASURED. `p8_gate_attempt2.json`
   is annotated in place.
4. ⭐ **The P4 "object permanence" result and the field mask are THE SAME PREDICATE.** The
   join's `occ` flag is `|atan2(cy, cx)| <= hfov/2` (`build_obstacle_join.py:210-223`) —
   *character-for-character the test `bev_raster.fov_mask` applies*, at agent-center rather
   than cell-center granularity. So "the latent carries agents the camera cannot see" is a
   claim **about** the masked-out set. Adding an `_infov` twin there would empty it. §4.2.
5. `p8_bev_reel.py` is **affected as a rendering**: 626 cells (8.151 %) of every pane were
   drawn as scene, and its captioned IoU was all-cells. Fixed.

**Evidence class.** Every number in §2–§3 is **MEASURED (ours)** — pure geometry, no corpus,
no checkpoint, no GPU. Artifact: `raw/bev_consumer_geometry.json`, reproduced by
`code/bev_consumer_census.py`, pinned in `stack/tests/test_bev_consumer_fov.py` (35 tests).

---

## 1. The full consumer list — and how it was established

⚠️ **Unbounded searches only.** `RETRACTION_LOG` **C69** is an absence claim made today from
`find -maxdepth 4` on files that sat at depth 6. Three independent sweeps, none depth-limited:

| sweep | query | what it caught that the others did not |
|---|---|---|
| 1 | `grep -rIn "bev_raster"` repo-wide | the import sites |
| 2 | `grep` on **every exported symbol** (`GRID_DEFAULT|BEVGrid|fov_mask|fov_census|agents_at_time|ego_frame_agents|agents_to_array|cell_centers_xy|readout_column_index|readout_row_ranges_m|cell_azimuth_rad`) | **`probe_latent_state.py`** (sweep 1 missed it) |
| 3 | `find` for `*lf0* *p8* *bev*` + `git ls-files` | the **banked artifacts** — `p8_gate_attempt*.json`, `Paper/figures/lf0_bev_panels.*`, the ops-bundle launch chains |

⭐ **Sweep 2 is why the list is complete and sweep 1 alone would not have been.** But note
what *no* import search can find: **consumer #8 restates the geometry inline and imports
nothing.** It was found only by sweep 3, through its output artifact.

### 1.1 Consumers that IMPORT the module

| # | consumer | frame assumption | hardcodes square-frame numbers? | published number | verdict |
|---|---|---|---|---|---|
| 1 | `stack/scripts/train_p8_occupancy.py` | resolved from the **run's own args** | no | P8 gate (§3.3) | **audited + fixed by the port** |
| 2 | `stack/scripts/lf0_bev_lead.py` | `resolve_eval_frames` → `model_frame` | **no** | LF0 (§3.1) | ✅ **UNAFFECTED** — measured, §2.1 |
| 3 | `stack/scripts/p8_bev_reel.py` | `resolve_eval_frames` → `model_frame` | **no** | the reel deliverable | ⚠️ **AFFECTED (rendering)** — §3.4 |
| 4 | `stack/scripts/build_obstacle_join.py` | **none — no camera model in its domain** | n/a | join stats (137 clips, 26 084 records) | ✅ **UNAFFECTED, structurally** — it emits agent records in metres, never rasterises, never scores. But see §4.2: it *does* carry the same field test under another name. |
| 5 | `stack/scripts/lf0_chain.sh` | **pins the frame explicitly** | no | none — a maths smoke (`20.25 m`) | ✅ **UNAFFECTED** — and it is the artifact that **recovered LF0's run frame** |
| 6 | `stack/tests/{test_p8,test_p8_v6,test_obstacle_join,test_lf0_bev_lead,test_p8_reel,test_physicalai_feature_readset}.py` | synthetic | n/a | none | ✅ unaffected |
| 7 | `…/2026-08-16-p8-v6-port/code/p8_geometry_census.py` | all three arms, explicitly | no | the port's census | the instrument |

### 1.2 Consumers that RESTATE the geometry inline — the class an import search cannot see

| # | consumer | what it restates | published number | verdict |
|---|---|---|---|---|
| 8 | `Paper/figures/make_lf0_bev_panels.py` | **`NX, NY = 120, 64` as a literal**, beside a live `GRID_DEFAULT`; draws every cell **unmasked** | ⭐ **YES** — `Paper/figures/lf0_bev_panels.{svg,png}`, both git-tracked, quoted in `MODEL_REGISTRY.md:1763-1768` | ⛔ **AFFECTED + UNRECOVERABLE** — §3.2 |
| 9 | `stack/scripts/ph0_v2_overlay.py:93` | the `+y is LEFT` convention, in a comment | PH0 overlay MP4 | ✅ **UNAFFECTED** — it draws the **ego's own realised path** (`engine_a["polyline_xy"]`), an ego-state fact. No decoder output lands on any cell, so there is nothing for a visibility mask to qualify. |

**Not consumers** (checked, so the list is falsifiable rather than merely short):
`stack/scripts/probe_latent_state.py:73` mentions `agents_at_time(classes=…)` in a docstring
only; `…/2026-08-11-ops-bundle/render_v5f_bev.py` (which produced
`Evaluation/Videos/v5f-bev-2026-08-10/v5f_bev_compact.mp4`) draws a **trajectory fan**, not an
occupancy raster, and imports nothing from this module.

---

## 2. The measurement

### 2.1 ⭐ LF0 — the corridor, not the grid

⛔ **THE DENOMINATOR IS THE WHOLE FINDING.** Quoting the grid-wide 8.151 % against LF0 would
have overstated its exposure by **more than 9×** — the mirror of the error this audit exists
to catch. Out-of-field cells are all *near and off-axis*: a cell at lateral `|y|` leaves a
half-angle `th` field only below `x = |y| / tan(th)`, and LF0's corridor is `|y| ≤ 1.75 m` at
its widest.

MEASURED (`raw/bev_consumer_geometry.json`), corridor cells LF0 actually scans:

| frame | grid-wide | ±1.0 m | **±1.5 m (headline)** | ±2.0 m |
|---|---|---|---|---|
| **v5 sub-frame 176×624 cyl 117 °** — *its run frame* | 626 / 7 680 (8.151 %) | **0 / 472** | **0 / 708** | **0 / 944** |
| v6F 256×640 cyl 120 ° | 590 / 7 680 (7.682 %) | **0 / 472** | **0 / 708** | **0 / 944** |
| legacy 256×256 pinhole | 2 126 / 7 680 (27.682 %) | 2 / 472 | **8 / 708 (1.130 %)** | 18 / 944 |

⇒ **At the frame LF0 ran on, every cell it scanned was inside the camera's horizontal field.
The masked and unmasked reads are identical by construction, so the banked verdict cannot
move.**

⚠️ **This is a property of the CONFIGURATION, not of the geometry alone — and it was luck.**
`--min-row 2` exists to skip *the ego's own footprint*, an unrelated reason. Without it the
same corridor on the same frame is **6 / 720**:

| | `--min-row 0` | `--min-row 2` (default) |
|---|---|---|
| v5 sub-frame, ±1.5 m | 6 / 720 | **0 / 708** |
| legacy pinhole, ±1.5 m | 18 / 720 | 8 / 708 |

⛔ **And on a legacy pinhole frame the same default is NOT enough** — 8 cells out to
**x = 2.25 m** stay unanswerable. That matters more than the count suggests: `read_lead_range`
returns the **nearest** hit, so an out-of-field false positive does not add noise, it
**shortens** the read — *"lead at 1.25 m"*. A forward-looking hazard, not a retraction: LF0
has run exactly once, on the sub-frame.

### 2.2 The reel — the whole grid, as pixels

`p8_bev_reel.py` renders every cell of the raster in two of its three panes and captions an
all-cells IoU. Its exposure therefore *is* the whole-grid number — **626 cells (8.151 %) per
pane** at its run frame, **2 126 (27.682 %)** had it ever run on a square frame — but as a
*rendering* fact: pixels a viewer reads as *"the world model's belief about the scene"*, all
of them at x < 9.25 m, i.e. exactly the band read as *"the vehicle in front of me"*.

---

## 3. Which published numbers are affected — enumerated, with n

### 3.1 ✅ UNAFFECTED — the LF0 headline (`MODEL_REGISTRY.md:1729-1796`)

Also in `Reports/2026-08-12-0554-program-report.md:90-104` and
`Reports/2026-08-15-2200-campaign-science-addendum.md:110`. **None of these move**, per §2.1:

| quantity | value | n |
|---|---|---|
| censoring on labelled, `enc@1.5` / `pred@1.5` | **81.40 % / 92.25 %** | **129** |
| corridor sweep `enc` censoring ±1.0/±1.5/±2.0 | 82.17 / 81.40 / 68.22 % | 23 / 24 / 41 |
| reader sanity `gt@1.0` ρ / `gt@2.0` ρ | 1.0 / 0.9596 | 129 |
| R² `enc` / `pred` | −21.00 / −16.12 | 24 / 10 |
| MAE `enc` / `pred` (m) | 26.85 / 42.65 | 24 / 10 |
| **verdict — RC1 REFUTED** | stands | — |

### 3.2 ⛔ AFFECTED and UNRECOVERABLE — the paper figure's cell counts

`MODEL_REGISTRY.md:1763-1768` and `Paper/figures/lf0_bev_panels.{svg,png}`:

> *"it puts **40 / 43 / 45** (encoded) and **68 / 43 / 35** (predicted) cells above τ —
> comparable to the ground truth's **33 / 31 / 31**"* — **n = 3 windows.**

These are **whole-grid** counts over all 7 680 cells, of which **626 (8.151 %) are outside the
camera field**, and the figure draws them unshaded. An unknown share of each count sits on
cells no camera observed.

⚠️ **The recount is NOT RECOVERABLE.** Three probes: `lf0_panels.npz` (pod4) and
`panels_compact.json` (a `/tmp/claude-0/…` scratchpad, per `make_lf0_bev_panels.py:88-89`) are
in neither the working tree, nor `git ls-files`, nor any banked bundle. It needs a re-run.

⭐ **But the CLAIM these counts support SURVIVES, and this is a measured argument, not a
hope.** The load-bearing sentence is *"essentially none of them land in the ego band"* — a
statement about the **corridor**, which §2.1 shows is **entirely in-field**. Masking can only
*remove* decoded cells from **outside** the band; it can never move one **into** it. So
**"confident MISLOCATION, not absence of output" stands**, and masking would, if anything,
sharpen it. ⇒ **No retraction. A caveat, now rendered onto the figure itself.**

### 3.3 ⚠️ RECOVERED frame; absolute IoU UNRECOVERABLE-as-in-field

`p8_gate_attempt2.json` / `MODEL_REGISTRY.md:1476`.

**The frame is recovered** — port escalation #2 discharged. Not from `train_log.jsonl` (no
`p8-occupancy-c` artifact exists in the repo; 3 probes: `find` for the run dir, `git ls-files`
for `p8_head.pt`/`p8_gate.json`, `find` for `train_log*`) but from **two independent in-repo
launch chains** that name the exact flags and bind to this JSON by `base_ckpt`, `out` dir and
`join-file`:

- `…/2026-08-11-ops-bundle/p8c_chain.sh` — header: *"P8 attempt-2 (p8c)"*
- `…/2026-08-11-ops-bundle/p4p8c_chain.sh` — *"Same command as pod5's queued p8c"*

Both: `--frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical --v2-subframe
176x624` ⇒ **176×624 cyl 117.0 °, 626 / 7 680 = 8.151 % out-of-field**. This **confirms the
port's INHERITED guess and promotes it to MEASURED**, and rules out the legacy square frame by
the flags rather than by the `w120` naming. Written into the banked JSON as
`_geometry_recovered_2026_08_16`.

| number | affected? |
|---|---|
| `ratio` **0.93191** (gate a, PASS) | ✅ **robust** — out-of-field cells depress numerator and denominator alike |
| `iou_enc` **0.020052**, `iou_pred` **0.018686** (n_enc 797 / n_pred 823, k=10) | ⚠️ **all-cells; the in-field twin cannot be recomputed** — the per-window IoUs were not banked, so it needs a re-run |

⛔ **AND THE WARNING THAT TRAVELS WITH IT.** A high ratio over a tiny denominator is
**uninformative**. `iou_enc` 0.0201 means masking is a **correctness** fix, **not an
explanation of quality** — removing ~8 % of cells cannot turn 0.02 into a healthy IoU. Quote
`iou_*_infov` for any absolute claim; `train_p8_occupancy.py` now emits it.

### 3.4 ⚠️ AFFECTED as a rendering — the P8 reel

`p8-occupancy-c/reel/` (`MODEL_REGISTRY.md:1487`) — pod-side, not in the repo. 626 cells
(8.151 %) of every belief pane were drawn as scene, and the caption IoU was all-cells. **Fixed
for future renders** (§4.1); the banked stills would need a re-render, which is minutes given
a `p8_head.pt`.

⚠️ **A separate viz defect, stated rather than fixed.** `compose_frame` resizes `[120, 64]` to
`(pane_w, PANE_H)`, so at `width=1280` a 32 m × 60 m grid is drawn at ~13.0 px/m across vs
5.3 px/m forward — a **2.44× horizontal stretch**. Ranges read off the pane by eye are wrong by
that factor. Not changed (it would move every banked still); recorded in the docstring.

---

## 4. The fixes

Every one follows the port's rule: **the incumbent metric is reported unchanged, PLUS an
in-field variant, so no banked verdict moves without a decision.**

### 4.1 Files changed

| file | change |
|---|---|
| `stack/tanitad/data/bev_raster.py` | `fov_row_floor(grid, half_angle, cols=None)` — the first row from which a **column subset** is fully in-field, which is the fact a corridor consumer needs and the grid-wide census cannot give. **Plus the stale-count docstring fix** (§4.3). |
| `stack/scripts/lf0_bev_lead.py` | `read_lead_range(..., fov=None)` (incumbent **byte-identical**, refuses a wrong-shaped mask); `corridor_fov_census()`; `_infov` twins on all 9 arms; **one truth per cell set**; a `geometry` block in `lf0_gate.json`; `--fov-read` (default `all`); `verdict_other_cell_set`; the field mask into `lf0_panels.npz`; UTF-8 pinned on the JSON write |
| `stack/scripts/p8_bev_reel.py` | `fov=` shading on both BEV panes (`None` ⇒ byte-identical); `NOFOV` / `NOFOV_HIT` colours; `iou_pair()` + an **in-field IoU in the caption**; `--no-fov-shading` opt-out; the 2.44× aspect note |
| `Paper/figures/make_lf0_bev_panels.py` | imports `GRID_DEFAULT` instead of typing `120, 64`; `nofov_spans()` which **refuses to guess a frame it was not given**; a footer caveat when none is recorded; **`LF0_FIG_OUT`** (§4.4) |
| `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt2.json` | `_geometry_recovered_2026_08_16` — annotation only, no existing key touched |
| `stack/tests/test_bev_consumer_fov.py` | **new, 35 CPU tests** |

### 4.2 ⛔ What I deliberately did NOT fix — and why the obvious fix is wrong

**The P4 permanence split has no `_infov` twin, and it should not get one.**
`train_p8_occupancy.py:920-935` accumulates `cell_recall` with no mask and reads `tau_star[""]`
(the all-cells operating point) at `:982`, so `MODEL_REGISTRY.md:1483-1486`'s *"occluded-agent
recall is NOT worse than visible — enc **0.2178** occluded vs **0.1881** visible; pred 0.1743
vs 0.1717 at k=10 (n 194/548)"* is an all-cells number.

⭐ **But the join's `occ` flag and `fov_mask` are the SAME PREDICATE.** `build_obstacle_join.py:210-223`:

```
az = np.arctan2(ag[:, 1], ag[:, 0])
return np.where(np.abs(az) <= math.radians(float(hfov_deg)) / 2.0, 0, 1)
```

— *the identical test* `bev_raster.fov_mask` applies, at **agent-center** instead of
**cell-center** granularity (the two differ only for an agent whose center is outside the
field while part of its footprint is inside, and vice versa). Its own docstring is explicit:
*"NOT object-object occlusion; no vertical bound"* (`:739-741`).

⇒ **"The latent carries agents the camera cannot see" is a claim ABOUT the masked-out set.**
Masking it to the field would empty the occluded arm by construction. The correct fix is to
**say the two instruments are not independent**, not to twin them — and the registry's own
stamp (*"must be re-checked against a diffuseness control before it is quoted as permanence"*)
already points the same way. **ESCALATED, not edited** (§6.1).

⚠️ **A second-order mismatch found while establishing this, MEASURED:** the join flags
visibility at `HFOV_DEG_DEFAULT = 120.0` (`:153`) while the encoder saw the **117 °**
sub-frame. **36 cells (626 − 590 = 0.469 % of the grid)** are flagged *"visible"* by the join
but were outside the field the model was actually fed.

### 4.3 The coordinator's handover — the stale count in `bev_raster.py:12`

Fixed, and **the layer is named rather than a bare number given**, because the undefined
subject *"our ingest"* is what let it rot four times (2 → 4 → 5 → 6): the docstring now states
**episode build = 5**, **program-wide = 6** (the sixth being `obstacle.offline`, joined
pod-side — which is why `grep obstacle physicalai.py` returns zero), names
`scripts/physicalai_r0.py` = 2 as the third read-set, and **points at
`stack/tests/test_physicalai_feature_readset.py` as the authority** rather than restating the
numbers. `grep -rIn "4 of 36"` over `stack/**/*.py` is now clean.

### 4.4 ⛔ A mistake I made, and the guard it earned

While exercising the figure generator I called `main()` with synthetic data and it
**overwrote the published, git-tracked `Paper/figures/lf0_bev_panels.svg`** — `main()` wrote
straight to its own directory. Recovered with `git checkout` and verified byte-identical
(md5 `cbe9b5b1…`, `git diff HEAD` empty). It was only recoverable **because it happened to be
committed**.

⇒ `LF0_FIG_OUT` now redirects the output dir, and
`test_figure_main_never_writes_outside_its_output_dir` both exercises the renderer end to end
and asserts the published file is untouched. *Same class as the traps CLAUDE.md collects: a
routine that writes to a fixed path is a destructive operation wearing a render's costume.*

---

## 5. Tests

`stack/tests/test_bev_consumer_fov.py` — **35 tests, CPU, no corpus / checkpoint / GPU / join.**

- the 0/708 corridor result at **both** cylindrical frames × all three widths, and the
  legacy frame's 8/708 with its 2.25 m depth;
- that **`--min-row 2` is what buys the zero** (6/720 without it) — so the result is pinned as
  configuration-dependent, not as a geometric inevitability;
- the **denominator rule**: corridor exposure < grid exposure ÷ 9;
- `fov_row_floor` agrees with the census, reproduces `first_fully_visible_row` (18 / 65), and
  returns **`None`** rather than a usable-looking index when no row is ever clean;
- masking is **inert in-field** and **does change** an out-of-field read — including the case
  that matters: a near false positive *shortening* the read, recovered to the true 20.25 m;
- a wrong-shaped mask is **refused**;
- reel panes are **byte-identical without a mask**, shade 626 cells with one, colour
  belief-on-unobservable distinctly, and **keep GT visible outside the field** (a labelled
  agent there is a real agent);
- in-field IoU can differ from all-cells IoU (else reporting both is theatre); empty union is
  **NaN, not 0**;
- the figure imports the grid, **refuses to guess a frame**, shades exactly 626 / 590 / 2 126
  cells when given one, caveats when not, and cannot overwrite the published SVG;
- the **launch-chain provenance** of the recovered P8 frame.

**Suite:** `PYTHONUTF8=1 …/python.exe -m pytest -q -p no:cacheprovider` from `stack/` →
**3 036 passed · 0 failed · 17 skipped · 2 xfailed** (356 s), banked at
`raw/stack_pytest.txt`. Brief baseline was **2 969 / 0 / 17 / 2**; the +67 is this file's 35
plus 32 from a concurrent stream (`test_e_wc2_sigma_star.py`, staged in the same index).
**Zero failures, zero regressions.**

---

## 6. Escalations — these need an owner

1. ⛔ **The P4 permanence number and the FOV mask are not independent instruments** (§4.2).
   `MODEL_REGISTRY.md:1483-1486`'s occluded-vs-visible recall is scored on the set the field
   mask removes, by the *same formula*. It is not wrong, but it must not be read as
   *"permanence, independently of the visibility question"*. Owner: whoever holds
   `train_p8_occupancy.py`. **The fix is a stamp, not an `_infov` twin** — a twin would empty
   the arm.
2. ⚠️ **The join flags visibility at 120 ° while the v5f encoder saw 117 °** — 36 cells
   (0.469 %) disagree. Cheap fix: pass the model frame's HFOV to `--hfov-deg` on the next join
   build, instead of the default.
3. ⛔ **`Paper/figures/lf0_bev_panels.{svg,png}` is published with unshaded out-of-field cells
   and its data source is not banked** (§3.2). Either re-run LF0 (the mask is now written into
   `lf0_panels.npz`) and re-render, or accept the caveat. **The load-bearing claim does not
   change either way** — this is about the figure being self-describing.
4. ⚠️ **`p8_gate_attempt1.json` is NOT annotated.** Its frame is not established: it predates
   `p8c_chain.sh` (which is explicitly *"attempt-2"*), and no attempt-1 chain is banked. Its
   `base_ckpt` and `raster_source` match attempt-2 exactly, which makes the same sub-frame
   likely — but that is **INHERITED, and I did not write a guess into a banked artifact.**
5. ⚠️ **`--fov-read` defaults to `all`, as `--fov-gate` does.** Conservative by choice — nothing
   banked moves. For LF0 the choice is currently **vacuous** (§2.1: the two cell sets are the
   same cells), and `verdict_other_cell_set` makes that visible in the JSON.
6. **No file owned by another live stream was touched.** `train_v6_staged.py`, `v6.py`,
   `e_wc2_sigma_star.py`, `refc_dump_latents.py`, `CLAUDE.md`, `RETRACTION_LOG.md` — all
   untouched.

---

## 7. Deliverable manifest

| artifact | path |
|---|---|
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-bev-consumer-audit/BEV_CONSUMER_AUDIT.md` |
| per-consumer census (MEASURED) | `…/2026-08-16-bev-consumer-audit/raw/bev_consumer_geometry.json` |
| census reproducer | `…/2026-08-16-bev-consumer-audit/code/bev_consumer_census.py` |
| corridor-aware field helper + docstring fix | `stack/tanitad/data/bev_raster.py` |
| LF0 in-field twins + geometry block | `stack/scripts/lf0_bev_lead.py` |
| reel field shading + in-field IoU | `stack/scripts/p8_bev_reel.py` |
| figure: imported grid, frame refusal, output guard | `Paper/figures/make_lf0_bev_panels.py` |
| recovered frame, annotated in place | `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt2.json` |
| tests (35, CPU) | `stack/tests/test_bev_consumer_fov.py` |

All **staged in the working tree (`git add`), never committed and never pushed** — per
`AGENT_OPERATING_STANDARD.md`. Nothing lives only on a pod or a worktree.

---

## 8. Staging verification

⛔ **Verified by BLOB, not by `--cached`.** `git ls-files --cached <path>` answers *"is this
path in the index?"* — which is **yes for any tracked file, including one whose index blob is
the pre-edit version**. For all nine artifacts, `git ls-files --stage` was compared against
`git hash-object`; all nine matched. Re-verified at end of turn, because a concurrent
orchestrator sweep moves the index underneath a running agent.

**Two integrity checks worth recording:**

- `Paper/figures/lf0_bev_panels.{svg,png}` are **byte-identical to HEAD** after the incident in
  §4.4 — `git diff HEAD` empty, md5 `cbe9b5b10f7db1cc48b73e52049d1d2a` /
  `713b70c3ea6a1f4c0e38b3f6cdae0727`. **Only the generator is modified.**
- `raw/bev_consumer_geometry.json` **reproduces bit-identically** from
  `code/bev_consumer_census.py` (md5 `8780681b053883ba5fd928499d3d56b3` before and after a
  re-run), and its three whole-grid figures reproduce the port's independently — 590 / 626 /
  2 126 — a cross-check that this audit's `centred_subframe`-derived half-angle agrees with the
  port's hand-entered 117 °.
