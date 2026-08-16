# P4's "object permanence" and the FOV mask are the same predicate — so the fix is a STAMP, not a twin

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Tier** T0-diagnostic (geometry only)
**Status** implemented + CPU-tested; **no GPU used** (Thor is training on the only GPU)

---

## 0. The headline

⭐ **This is a case where applying the obvious fix would have destroyed the finding it was meant
to protect.** On 2026-08-16 the `bev_raster` consumer audit added an `_infov` twin to every metric
that scores BEV cells, because scoring cells no camera could observe is a correctness defect.
**The P4 visible/occluded split is the one place that fix is wrong**, and the reason is exact
rather than rhetorical.

1. ⭐ **VERIFIED AT SOURCE, AND IT IS BIT-EXACT.** `build_obstacle_join.visibility_occ`
   (`stack/scripts/build_obstacle_join.py`, the `az = np.arctan2(ag[:,1], ag[:,0])` /
   `np.abs(az) <= math.radians(hfov_deg)/2` pair) and `bev_raster.fov_mask`
   (`stack/tanitad/data/bev_raster.py`, `np.abs(cell_azimuth_rad(grid)) <= half_angle_rad` over
   `arctan2(y, x)`) select **identical sets**: **0 of 7 680 cells disagree** at hfov
   30/60/90/117/120/150/179°, and the two **defaults are the same IEEE double** —
   `math.radians(60.0)` and `math.radians(120.0)/2.0` both hex to `0x1.0c152382d7365p+0`,
   **ULP gap 0.0**. The only difference is granularity: `occ` grades the **agent centre**,
   `fov_mask` grades the **cell centre**.
2. ⛔ **⇒ P4's `occluded` arm IS the masked-out set, and an `_infov` twin would DELETE it.**
   MEASURED on the real rasteriser over 10 000 occluded agents per extent: a sub-cell agent keeps
   **0.0–1.5 %** of its cells (98.5–100 % emptied outright), an automobile (4.5 × 2.0 m) **10.8 %**
   (60.4 % emptied), a heavy truck (12 × 2.6 m) **26.7 %** (24.2 % emptied). The survival fraction
   **rises monotonically with vehicle length** — the twin does not correct the population, it
   **re-selects it by extent**. `cell_recall` returns NaN on an emptied subset and `_mean_n` drops
   it, so the arm would lose its `n` *silently*.
3. ⭐ **The stamp is now machine-readable in five places and guarded by a test that fails if a twin
   is ever added** — including a source-level guard on the accumulation block itself, verified
   against three realistic "tidy" mutations (§4.3).
4. ⚠️ **A published sentence is wrong, and it is in the paper.** `TANITAD_PAPER.md` §7.18 said the
   occluded ≥ visible ordering *"holds at k = 5/15/20"*. Re-reading the banked artifact: true for
   the **encoded** arm at all four k; **false for the predicted arm**, which reverses at k = 5/15/20
   (−0.0035 / −0.0034 / −0.0057) and is positive at **k = 10 alone**. Corrected in place (§3.2).
5. ⚠️ **The 120°/117° mismatch is real but its published "0.469 %" is the wrong denominator**, and
   the direction of the error is **conservative** for P4 (§5).
6. ⛔ **The sharper question — permanence or geometry? — cannot be answered here.** It needs the
   pod-side `p8_head.pt`. What I could do is make it **cheap**: the discriminator is implemented
   and pre-registered with both outcomes, and a `--head-ckpt` re-score path turns it from a
   **6.5 h retrain into a ~5 min mini-eval** (§6).

**Evidence class.** Every number in §1–§5 is **MEASURED (ours)** — pure geometry and a re-read of
banked JSON. No corpus, no checkpoint, no GPU, no join file. Artifact:
`raw/p4_predicate_identity.json`, reproduced by `code/p4_predicate_census.py`, pinned in
`stack/tests/test_p4_fov_predicate.py` (31 tests).

---

## 1. The predicate identity, verified at source

Both sides quoted from the files, not from the writeup that reported them.

**`stack/scripts/build_obstacle_join.py`** — `visibility_occ` (search for
`def visibility_occ`; the docstring begins *"P4 flag per EGO-FRAME agent row"*):

```python
az = np.arctan2(ag[:, 1], ag[:, 0])
return np.where(np.abs(az) <= math.radians(float(hfov_deg)) / 2.0, 0, 1).astype(np.int64)
```

**`stack/tanitad/data/bev_raster.py`** — `fov_mask` + `cell_azimuth_rad` (search for
`def fov_mask` / `def cell_azimuth_rad`):

```python
def cell_azimuth_rad(grid=GRID_DEFAULT):
    x, y = cell_centers_xy(grid)
    return np.arctan2(y, x)

def fov_mask(grid=GRID_DEFAULT, half_angle_rad=math.radians(60.0)):
    ...
    return np.abs(cell_azimuth_rad(grid)) <= float(half_angle_rad)
```

⚠️ **Citations here are by CONTENT, not by line number, deliberately.** The bev-consumer audit
cited `MODEL_REGISTRY.md:1483-1486` for the P4 claim; when I went to read it the text was at
**:1480-1487** — three lines of drift, in a document nobody had edited for that purpose. The same
audit cited `build_obstacle_join.py:210-223` for `occ`, which was correct at the time and is now
wrong **because of my own edit**. Line numbers in this repo have a half-life measured in hours.

### 1.1 The census (MEASURED — `raw/p4_predicate_identity.json`)

| half-angle (hfov) | cells | disagreeing | in-field |
|---|---|---|---|
| 30 / 60 / 90 / **117** / **120** / 150 / 179° | 7 680 | **0** at every one | 3 856 / 5 904 / 6 688 / **7 054** / **7 090** / 7 408 / 7 680 |

And the defaults are not merely close:

| | value | hex |
|---|---|---|
| `fov_mask`'s default `half_angle_rad` | 1.0471975511965976 | `0x1.0c152382d7365p+0` |
| `visibility_occ` at `HFOV_DEG_DEFAULT` | 1.0471975511965976 | `0x1.0c152382d7365p+0` |
| **ULP gap** | **0.0** | — |

⇒ **These are not two instruments that agree. They are one predicate written twice.**

### 1.2 The only real difference, and it is small

`occ` grades the agent **centre**; `fov_mask` grades the **cell** centre. The discrepancy is
therefore the footprint sliver of an agent whose centre is outside the field while part of its
body is inside (and the mirror case). It is demonstrable — pinned by
`test_the_only_difference_is_granularity` — and it is exactly why §2's "masking empties it" is
**98–100 %** for a sub-cell agent rather than a tautological 100 %.

---

## 2. What an `_infov` twin would actually do — measured, not argued

⚠️ **This section corrected an over-claim of my own, which is why it is stated with its sampling
rule.** My first sweep drew agent centres uniformly over the whole grid, yielded **45** occluded
agents in 4 000 draws, saw 0 survivors, and I nearly banked *"a point agent loses 100 % of its
cells"*. The guard test caught it at **8/1 500**. Every occluded agent on this grid lives at
`x < y_half / tan(θ) = 9.2376 m`, so the corrected sweep samples that wedge directly, 20 000 draws.

MEASURED on `bev_raster.rasterize`, same agent centres for every extent:

| agent extent | occluded agents | cells kept | **cell survival** | **agents emptied outright** |
|---|---|---|---|---|
| sub-cell 0.05 × 0.05 m | 100 | 0 / 100 | **0.00 %** | **100 %** |
| sub-cell 0.45 × 0.45 m | 8 020 | 123 / 8 214 | **1.50 %** | **98.47 %** |
| automobile 4.5 × 2.0 m | 10 000 | 33 911 / 313 545 | **10.82 %** | **60.38 %** |
| heavy truck 12 × 2.6 m | 10 000 | 240 797 / 900 769 | **26.73 %** | **24.15 %** |

⭐ **The load-bearing row is the monotonicity, not any single number.** 0.0 → 1.5 → 10.8 → 26.7 %
is survival ordered by **vehicle length**. A masked "occluded" arm is therefore not the occluded
arm corrected — it is a **truck-weighted boundary-sliver population** with no interpretation as
*"agents the camera cannot see"*. And because `cell_recall` returns NaN on an empty subset and
`_mean_n` drops NaN, the deleted 60 % would vanish from `n` without an error.

⚠️ **Population caveat, stated rather than buried.** The agent-position prior above is
**uniform-on-wedge (synthetic)**. The corpus prior needs the pod-side join file
(`/workspace/data/p8_join/combined140.jsonl`). The *mechanism* is exact geometry; the *weighting*
is not corpus-calibrated, and I have not pretended otherwise.

---

## 3. Where P4's claim is published — located by content

Six sites, found by grepping the numbers and the phrase, not by following a citation:

| # | site | text (content anchor) | action |
|---|---|---|---|
| 1 | **`Project Steering/MODEL_REGISTRY.md`** — the P8 attempt-2 entry | *"occluded-agent recall is NOT worse than visible — enc 0.2178 occluded vs 0.1881 visible; pred 0.1743 vs 0.1717 at k=10"* (n 194/548) | ⭐ **STAMPED** (§3.1) |
| 2 | **`Paper/TANITAD_PAPER.md` §7.18** | *"encoded 0.2178 occluded vs 0.1881 visible … and the ordering holds at k = 5/15/20"* | ⭐ **CORRECTED + STAMPED** (§3.2) |
| 3 | `Paper/TANITAD_PAPER.md` §143, §2341-2342, §2934-2936, §3267 | summary restatements; already hedged as *"consistent with permanence pending a diffuseness control"* | left — §7.18 is the definition site and now carries the stamp |
| 4 | `…/2026-08-07-hierarchical-wm-redesign/WM_PHYSICS_PROOF.md` (the *"P8 RUN (attempt 2)"* checklist item) | *"occluded recall ≥ visible … the latent carries agents the camera cannot see"* | ⭐ **STAMPED** |
| 5 | `Project Steering/Reports/2026-08-12-0554-program-report.md`, `…/2026-08-15-2200-campaign-science-addendum.md` | dated snapshots | **deliberately NOT edited** — dated reports are historical records and CLAUDE.md makes the registry the only quotable source. Enumerated here so the set is falsifiable rather than merely short. |
| 6 | `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt{1,2}.json` | `visible_occluded_split` — the raw numbers | ⭐ **ANNOTATED IN PLACE** (§3.3) |

### 3.1 The registry stamp

Added directly beneath the P4 sentence: the identity (0/7 680, bit-identical defaults), the
⛔ do-not-twin rule with the measured deletion, the region-disjointness that gives the required
diffuseness control a concrete form, and the k-fragility of the `pred` half.

### 3.2 ⚠️ The paper correction

> *"…**predicted 0.1743 vs 0.1717** — and the ordering holds at k = 5/15/20."*

MEASURED from `p8_gate_attempt2.json`, all four banked k:

| k | enc occ − vis | pred occ − vis |
|---|---|---|
| 5 | **+0.0128** | −0.0035 |
| **10** | **+0.0296** | **+0.0026** |
| 15 | **+0.0143** | −0.0034 |
| 20 | **+0.0168** | −0.0057 |

⇒ the **encoded** ordering holds at all four k; the **predicted** ordering holds at **k = 10
alone**. The sentence sits immediately after the predicted numbers and reads as covering both.
**Corrected in place**, with the fairness point kept: **k = 10 is the pre-registered gate k**
(`GATE_K`), so this is a *principled* quotation, not selection after the fact — but it is
k-specific and must not be stated as holding across k.

⚠️ **Two further things the banked numbers do not carry, worth saying once.** There is **no
interval on any of these gaps** — no estimator, no CI, and the per-window values were not banked,
so a paired episode-cluster bootstrap is not computable from the artifact. And
`recall_occluded_enc` is **flat and non-monotone in k** (0.1991 / 0.2178 / 0.2017 / 0.2001, spread
0.0186), which is what a fixed regional prior looks like and not what decaying memory looks like.
Both feed §6.

### 3.3 The banked artifacts

`code/annotate_banked_p8.py` adds one top-level key, `_p4_predicate_identity_2026_08_16`, to both
`p8_gate_attempt1.json` and `p8_gate_attempt2.json`. **Annotation only** — the script re-parses and
asserts every pre-existing key is byte-identical afterwards (15 and 16 keys respectively), and is
idempotent. Same pattern the bev-consumer audit used for `_geometry_recovered_2026_08_16`.

---

## 4. The stamp, and the guard that keeps it

### 4.1 Machine-readable, at both ends of the pipe

| where | what |
|---|---|
| `build_obstacle_join.P4_PREDICATE_IDENTITY` | the producer's stamp: `occ_is_fov_mask`, both sources, the granularity, the measured identity, `DO_NOT_ADD_AN_INFOV_TWIN`, `encoder_frame_rule` |
| `build_obstacle_join.assert_occ_matches_fov_mask()` | **measures** the identity at the run's own half-angle and **raises** if a single cell disagrees — the stamp is a measurement of *this* run, not a quotation |
| the join's `<out>.meta.json` | `p4_predicate_identity` block carrying the stamp + `hfov_deg_used` + the self-check result, so it travels **with the data** |
| `train_p8_occupancy.P4_SPLIT_STAMP` / `P4_SPLIT_CELL_SET` | the consumer's stamp, same `stamp` token so one grep finds both |
| `p8_gate.json` → `visible_occluded_split.predicate_identity` | written in **both** branches (flags present or absent) |
| `p8_gate_attempt{1,2}.json` | annotated in place |

Both ends share the token `P4_OCCLUDED_IS_THE_FOV_MASK_COMPLEMENT`, asserted equal by test.

### 4.2 A runtime refusal is not possible here — so the guard is a test

There is nothing to refuse at runtime: adding an `_infov` twin is an *edit*, not an input. So the
guard is `stack/tests/test_p4_fov_predicate.py`, **31 CPU tests**, in three layers:

1. **API surface** — `cell_recall`'s signature must be exactly `{logits, subset_target, tau}`. A
   `mask=`/`fov=` kwarg fails the suite with the reason.
2. **Source of the accumulation block** — the block between `if occ_acc is not None:` and
   `head.train()` is scanned for `_infov`, `fov`, `mask`. This catches the routes a signature check
   cannot (pre-multiplying the subset raster, adding an `_infov` subset name).
3. **Emitted output** — no key of the stamp or the split may end in `_infov`, and
   `FOV_GATE_SUFFIX` must not appear anywhere in the split's construction.

### 4.3 The guard was verified against real mutations, not assumed

| mutation | caught by |
|---|---|
| `cell_recall(log_enc[sub], rs[rows]*fov, …)` | `fov` |
| `for s in p4_sets + ("occluded_infov",):` | `_infov`, `fov` |
| `rs = rs.to(device); rs = rs * mask` | `mask` |

Clean source → no hits. *(A guard that has never been shown to fire is not a guard.)*

---

## 5. ⚠️ The second defect: the join flags at 120°, the encoder saw 117°

### 5.1 Which is correct — and it is not "the sensor"

- The join's `HFOV_DEG_DEFAULT = 120.0` is the **sensor** field (`camera_front_wide_120fov`).
- The banked P4/P8 run encoded the **176 × 624 centred sub-frame = 117.0°**
  (`p8_gate_attempt2.json`'s `_geometry_recovered_2026_08_16`, itself recovered from two in-repo
  launch chains).

P4's claim is *"the latent carries agents **the camera** cannot see"* — a claim about what
information reached the latent. **"The camera" must therefore mean the frame the encoder was fed.**
⇒ **117° is correct for that run; 120° is correct only as a statement about the raw sensor.**

⚠️ **The default is not wrong per se** — v6F trains at the full 256×640 = 120°, where it is exactly
right. What was wrong is that the coupling was **unrecorded**. So: **no default change** (that
would silently re-define a flag other artifacts were built with), and instead a loud runtime notice
when the sensor default is used, the `encoder_frame_rule` in the stamp, and the actual `hfov_deg`
written into the sidecar.

### 5.2 The n — and why the published 0.469 % is the wrong denominator

| quantity | value |
|---|---|
| out-of-field cells at 120° | 590 / 7 680 (7.682 %) |
| out-of-field cells at 117° | 626 / 7 680 (8.151 %) |
| **cells disagreeing** | **36 (0.469 % of the grid)**, azimuth band (58.5°, 60.0°], x 0.75–9.25 m, |y| 1.25–15.75 m |

⛔ **But `fov_mask` grades CELLS and `visibility_occ` grades AGENT CENTRES.** The join-side n — the
one that decides how much the P4 numbers move — is *the number of agent rows whose centre azimuth
falls in the 1.5° annulus*. That is a **property of the corpus**, is **not derivable from the
grid**, and needs the pod-side join file. **The 36-cell figure answers a different question than
the one it is being quoted for.** I have not converted one into the other.

### 5.3 What the 0.469 % touches — enumerated with n, not waved away

| number | affected? |
|---|---|
| **P4 split** (0.2178 / 0.1881 / 0.1743 / 0.1717, n 194/548) | ⚠️ **YES, and CONSERVATIVELY.** Agents in the annulus are labelled `visible` though the encoder never saw them — occluded-like rows in the *visible* bucket. Since the published result is occluded > visible, that contamination can only **raise** the visible arm and **shrink** the gap. Correcting to 117° can only make the gap **larger**. Magnitude unmeasurable in-repo (§5.2). |
| **LF0's banked verdict** (RC1 REFUTED; 81.40 / 92.25 % censoring, n 129) | ✅ **NO — MEASURED.** **0 of the 36 cells** lie in LF0's scanned corridor at its run configuration (±1.0/1.5/2.0 m with `--min-row 2`). At `--min-row 0` it would be 2. Pinned by test. |
| P8 gate (a) `ratio 0.93191` | ✅ no — it is a ratio, and out-of-field cells depress numerator and denominator alike |
| `iou_enc` / `iou_pred` absolute (0.020052 / 0.018686) | ⚠️ all-cells; the in-field twins were never computed. Unchanged by this work — it is the bev-consumer audit's open item, not a new one. |
| the 8.151 % / 7.682 % frame figures in the audit and the annotated JSON | ✅ correct as stated — they **are** the 117/120 pair |

---

## 6. ⭐ The sharper question: permanence, or the geometry of the mask?

**If `occ` and `fov_mask` are the same predicate, what is P4 measuring?**

### 6.1 The confound, stated precisely

`recall_occluded` is scored over cells that lie (up to the granularity sliver) **entirely inside a
590-cell wedge** — 7.68 % of the grid, **all of it at x < 9.2376 m**, on the ego's near left and
right shoulders. `recall_visible` is scored over the other 92 %, out to 60 m. The two arms are
therefore **disjoint regions of different size and different range**, and `cell_recall` is **not
region-normalised**.

⇒ **A decoder whose per-cell firing rate is higher on the near shoulders scores
occluded > visible while carrying no particular agent at all.** With `pos_weight` 79.7 on rasters
that are 1.239 % positive, a strongly non-uniform firing rate is exactly what training produces.
This is the registry's already-stamped *"diffuseness control"* requirement — but **the identity
tells us what shape that control must have**: not "is the decoder diffuse?" but **"is the gap
regional?"**

Two supporting reads from the banked artifact, both weak on their own and both pointing the same
way: `recall_occluded_enc` is **flat and non-monotone in k** (0.1991/0.2178/0.2017/0.2001) where
memory predicts decay; and the **predicted** arm's ordering survives at only 1 of 4 k.

### 6.2 The discriminator, pre-registered with both outcomes

**Compare the occluded arm against SAME-REGION, SAME-VISIBILITY agents.** Split the *visible*
agents at the same range boundary that bounds the occluded wedge (`x < 9.2376 m`) and score
`visible_near` beside `occluded`:

| outcome | reading |
|---|---|
| `occluded_over_visible_near ≈ 1.0` (and both > `visible_far`) | ⛔ **the gap is REGIONAL.** P4 is measuring the geometry of the mask, not permanence. The claim retracts to *"the readout fires more densely near the ego"*. |
| `occluded_over_visible_near > 1.0` | ⭐ **it survives the regional explanation** and is genuine evidence for permanence — the first version of this claim that the identity does not undermine. |

**Committed before the number exists**, in `select_subset`'s docstring, in the emitted
`region_control.read` string, and pinned by `test_the_control_is_pre_registered_with_both_outcomes`.

**Implemented, default OFF** (`--p4-region-control`), so no banked output moves. It costs **no
forward pass** — it re-scores the already-computed `log_enc`/`log_pred` against two extra CPU
rasterisations; asserted by `test_the_control_costs_no_extra_forward_pass`.

### 6.3 ⛔ It CANNOT be answered here, and this is the cost

It needs the trained readout. `p8_head.pt` is **not in this repo** — three probes: `git ls-files`
(no `.pt`), `find -maxdepth 6 -name "p8_head*.pt"` (nothing), and a repo-wide grep that finds the
name only in text. It lives at `/workspace/experiments/p8-occupancy-c/p8_head.pt`, pod-side.

⭐ **But I found the reason it was expensive, and removed it.** `train_p8_occupancy.py` *saved*
`p8_head.pt` every `--save-every` steps and **had no way to load it back** — while
`lf0_bev_lead.py` has always loaded that exact file. So every P8 re-scoring question cost a
**6.5 h retrain** (MEASURED: `wall_s` 23 508 s) rather than the **4.3 min** mini-eval
(MEASURED: `wallclock_s` 255.9 s). Added `--head-ckpt` (loader copied from `lf0_bev_lead.py`'s
precedent, refusing a mismatched `state_dict` rather than scoring random weights) and made
`--steps 0` a valid re-score configuration.

**Cost to answer, with the head in hand:**

| | |
|---|---|
| GPU | **~5–7 min** (one mini-eval; ESTIMATED from the MEASURED 4.3 min v5 mini-eval + 2 extra CPU rasterisations per window, no extra forward pass) |
| prerequisites | `p8_head.pt` **and** `combined140.jsonl` **and** the v5f checkpoint co-located on one pod |
| without `--head-ckpt` (before this change) | **~6.5 h** |

```
python3 scripts/train_p8_occupancy.py --ckpt <v5f ckpt> --steps 0 \
    --head-ckpt /workspace/experiments/p8-occupancy-c/p8_head.pt \
    --raster-source join-file --join-file /workspace/data/p8_join/combined140.jsonl \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
    --v2-subframe 176x624 --p4-region-control --out /workspace/experiments/p8-p4-region
```

**I have not guessed the outcome.** Both branches are written down and neither is favoured.

⚠️ **One honest limit of the control.** `visible_near` and `occluded` share a *range* band but not
an *azimuth* band (the occluded set is off-axis by construction). A residual azimuth confound
therefore survives; a second control splitting `visible_near` by |azimuth| would close it and is
the natural follow-on, not something I have silently folded in.

---

## 7. Files changed

| file | change |
|---|---|
| `stack/scripts/build_obstacle_join.py` (**owned**) | `P4_PREDICATE_IDENTITY`; `assert_occ_matches_fov_mask()` run at build start; the stamp into `<out>.meta.json`; docstring + `visibility_occ` docstring + `--hfov-deg` help carry the do-not-twin and encoder-frame rules; a loud notice when the sensor default is used; **`encoding="utf-8"` pinned on the meta write** (it was unpinned) |
| `stack/scripts/train_p8_occupancy.py` | `P4_SPLIT_STAMP` / `P4_SPLIT_CELL_SET`; the stamp into `visible_occluded_split` in **both** branches; `select_subset` + `out_of_field_x_ceiling_m` + `--p4-region-control` (the discriminator, default OFF); `load_head_ckpt` + `--head-ckpt` + `--steps 0` re-score path; `--steps 0` skips the unused auto pos-weight sampling; module docstring + `mini_eval` docstring state why `fov` must not reach the P4 split |
| `Project Steering/MODEL_REGISTRY.md` | the P4 predicate stamp beneath the P8 attempt-2 entry (additive) |
| `Paper/TANITAD_PAPER.md` | §7.18 k-ordering **correction** + the predicate stamp |
| `…/2026-08-07-hierarchical-wm-redesign/WM_PHYSICS_PROOF.md` | the stamp on P4's own definition site |
| `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt{1,2}.json` | `_p4_predicate_identity_2026_08_16` — annotation only, every existing key byte-verified |
| `stack/tests/test_p4_fov_predicate.py` | **new, 31 CPU tests** |

**Not touched** (owned by live streams): `train_v6_staged.py`, `v6.py`, `planner_p2.py`, `ci.py`,
**`bev_raster.py`**, `CLAUDE.md`, `RETRACTION_LOG.md`, `V6F_PLANNER_DESIGN.md`. ⭐ Note that
`bev_raster.py` needed **no** change — the identity is a fact *about* `fov_mask`, not a defect *in*
it.

---

## 8. Tests

`stack/tests/test_p4_fov_predicate.py` — **31 tests, CPU, no corpus / checkpoint / GPU / join.**

- the identity at 7 half-angles; the bit-identical defaults by `float.hex()`; the granularity gap
  demonstrated rather than asserted;
- the shipped self-check both **agrees** and **fires when the predicate is broken** (monkeypatched
  to "everything visible" → 590/7 680 disagree → `AssertionError`);
- the twin's damage: a sub-cell agent all but emptied (with the residual pinned as **non-zero** —
  the over-claim §2 records), survival **rising with vehicle length**, and `cell_recall` → NaN on
  an emptied subset;
- **the guard**, all three layers, plus the mutation check;
- the wedge ceiling 9.2376 m and the 590 cells; `select_subset`'s four subsets and its refusal;
  that every on-grid occluded agent is in the near band (3 000 draws);
- the control is pre-registered with both outcomes and **costs no forward pass**;
- `--head-ckpt` round-trips **both** saved shapes and **refuses** a mismatched `state_dict`;
  `--steps 0` is a valid configuration;
- the stamps exist at both ends with the same token, the banked JSONs are annotated and their
  numbers untouched, and the census artifact's numbers are pinned;
- the 120/117 pair (590/626/36), the annulus bounds, and **0 disagreeing cells in LF0's corridor**.

**Suite:** `PYTHONUTF8=1 …/python.exe -m pytest -q -p no:cacheprovider` from `stack/` →
**3 154 passed · 0 failed · 17 skipped · 2 xfailed** (365.7 s), banked at `raw/stack_pytest.txt`.
Brief baseline was **3 036 / 0 / 17 / 2** at `b65b3ab`; HEAD has since moved to `efd49f5` and
concurrent streams added ~87 tests, so the arithmetic that closes is **3 123 (same tree, this file
ignored) + 31 = 3 154**. **Zero failures, zero regressions.**

⚠️ **One transient failure, chased to ground rather than waved off — and it was not mine.** An
earlier full run showed `test_e_ag1_anchor_floor.py::test_no_situation_classifier_path` FAILED on
`assert "tanitad.data.situations" not in sys.modules`. That file was `AM` in `git status` — staged
*and* being edited by a live concurrent stream mid-run. Attribution took four measurements
(isolation: pass; my file + that test: pass; suite without my file: pass; suite with: fail), and
the answer arrived from the owner's own repair: they replaced the in-process assertion with a
**subprocess** check whose new comment states the diagnosis verbatim — *"`sys.modules` in a
full-suite run is polluted by every other test that legitimately imports the situation detectors.
An in-process check would pass alone and fail in the suite."* ⇒ **a process-global assertion in
someone else's in-flight test, since fixed.** Nothing in this work touches
`tanitad.data.situations`; MEASURED, importing `build_obstacle_join` + `train_p8_occupancy` leaves
it out of `sys.modules`. *(Recording it because the trap generalises: a suite-wide `sys.modules`
assertion reports the SESSION's imports, not the module's — the same class as CLAUDE.md's
"a probe that reports the wrong scope is worse than no probe".)*

---

## 9. Escalations — these need an owner

1. ⛔ **The P4 permanence/geometry question is OPEN and now costs ~5–7 min of GPU.** The
   discriminator is implemented and pre-registered; it needs `p8_head.pt` + `combined140.jsonl` +
   the v5f checkpoint on one pod. **Owner: whoever next has a pod at a deliberate pause.** Until it
   runs, the registry's *"must be re-checked against a diffuseness control"* stands and P4 must not
   be quoted as permanence standing alone.
2. ⚠️ **`train_p8_occupancy.py` could not load the head it saves.** Fixed here — but the same shape
   is worth auditing elsewhere: a script that writes a checkpoint nothing reads turns every
   re-scoring question into a retrain. *(The P8 v6 port's "re-scoring it is minutes" was INHERITED
   and, as written, was not achievable.)*
3. ⚠️ **The next join build must pass the ENCODER's `--hfov-deg`, not the default.** For a v6F
   trunk at the full 256×640 the default 120° is correct; for any sub-frame arm it is not. The
   build now prints a notice and records the value, but nothing can enforce it from inside.
4. ⚠️ **No interval exists on any P4 number, and none is computable from the artifact** — the
   per-window `cell_recall` values were not banked. The next run should bank them so the paired
   episode-cluster bootstrap (`taniteval/ci.py`) can be applied. ⛔ Never `overlapping_holdout_se`.
5. ⚠️ **The dated reports** (`2026-08-12-0554-program-report.md`,
   `2026-08-15-2200-campaign-science-addendum.md`) still carry the unstamped P4 sentence. I left
   them as historical records. If the programme's convention is that reports get corrected in
   place, that is a decision for the PI, not for me.
6. ⚠️ **A residual azimuth confound survives the region control** (§6.3). A `visible_near` split by
   |azimuth| would close it — declared, not silently folded in.
7. **No file owned by another live stream was touched** — `train_v6_staged.py`, `v6.py`,
   `planner_p2.py`, `ci.py`, `bev_raster.py`, `CLAUDE.md`, `RETRACTION_LOG.md`,
   `V6F_PLANNER_DESIGN.md` all untouched.

---

## 10. Deliverable manifest

| artifact | path |
|---|---|
| this writeup | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-p4-fov-predicate/P4_FOV_PREDICATE.md` |
| predicate census (MEASURED) | `…/2026-08-16-p4-fov-predicate/raw/p4_predicate_identity.json` |
| census reproducer | `…/2026-08-16-p4-fov-predicate/code/p4_predicate_census.py` |
| banked-artifact annotator (idempotent) | `…/2026-08-16-p4-fov-predicate/code/annotate_banked_p8.py` |
| suite output | `…/2026-08-16-p4-fov-predicate/raw/stack_pytest.txt` |
| producer stamp + self-check | `stack/scripts/build_obstacle_join.py` |
| consumer stamp + region control + re-score path | `stack/scripts/train_p8_occupancy.py` |
| registry stamp | `Project Steering/MODEL_REGISTRY.md` |
| paper correction + stamp | `Paper/TANITAD_PAPER.md` |
| P4 definition-site stamp | `…/2026-08-07-hierarchical-wm-redesign/WM_PHYSICS_PROOF.md` |
| annotated banked results | `…/2026-08-07-hierarchical-wm-redesign/p8_gate_attempt{1,2}.json` |
| guard tests (31, CPU) | `stack/tests/test_p4_fov_predicate.py` |

All **staged in the working tree (`git add`), never committed and never pushed** — per
`AGENT_OPERATING_STANDARD.md`. Nothing lives only on a pod or a worktree.

**Staging verified by BLOB, not by `--cached`** — `git ls-files --stage <path>` compared against
`git hash-object <path>` for every artifact, and **re-verified at end of turn**, because
`--cached` answers *"is this path in the index?"* and says yes for a tracked file whose index blob
is the pre-edit version.

⚠️ **HEAD moved twice under this work** — `b65b3ab` → `efd49f5` → `eca7106`, concurrent
orchestrator sweeps. The second sweep **committed 11 of the 13 artifacts**; verified file-by-file
that nothing was lost or altered (`git rev-parse HEAD:<path>` == `git hash-object <path>` for all
11, and the stamp tokens grep out of HEAD's copies: 1 hit in `build_obstacle_join.py`, 15 in
`train_p8_occupancy.py`). The remaining two — this writeup and `raw/stack_pytest.txt`, both edited
after the sweep — are **staged and blob-verified**. **This agent committed nothing and pushed
nothing.** Three of the files moving in the same window (`test_e_ag1_anchor_floor.py`,
`test_anchor_goal_labels.py`, `anchor_goal.py`) belong to a sibling stream, not to me.

**Integrity check worth recording:** `raw/p4_predicate_identity.json` **reproduces
bit-identically** from `code/p4_predicate_census.py` — md5 `8dcf419dc19af734b2dceae8e0a309d6`
before and after a re-run — and its 590 / 626 whole-grid figures reproduce the bev-consumer
audit's and the P8 v6 port's independently, from a different code path.
