# refcv6 adversarial review — PERCEPTION HEADS (map + 3-D box) and dataset alignment

**Reviewer:** independent adversarial reviewer, Architecture & Inference, 2026-09-22.
**Scope:** SPEC_REFCV6_V2.md §6 (MAP + BOX heads), §7 (data), §12 (geometry); the
`ADVISORY_FROZEN_TRUNK_DEFECT_CLASSES.md` **class G** (a filter whose semantics encode its own
author's requirement) and **class E** sweeps, applied to the map/box GT paths.
**Out of scope, covered by siblings:** trunk/input, diffusion, tactical/nav, training/guards.

**Repo:** `D:/Projects/TanitAD` @ `37645fc` (branch `agent/arch-inf-20260803`).
⚠️ `stack/scripts/refc_v3_train.py` had a *worktree ≠ HEAD* blob at review time
(HEAD `2b0770a3…`, worktree `38999cef…`) — siblings are live. Every trainer line cited below was
re-asserted against **`HEAD`**, not only the worktree.

⛔ **LINE DISCIPLINE.** Every corpus number carries its line. **parity** =
`physicalai-train-e438721ae894` (2,308-clip join). **v7-B1** = `physicalai-b1-w120-256x640cyl`
(4,719 clips; 4,572 train / 147 eval in the v7.2 release). **eval-139** = the 139 B1 eval clips with
pixels. refcv6 trains the **v7-B1** line. Clip ids appear only as `sha256(clip_id)[:12]`.

---

## 0. Verdict in one table

| # | finding | class | evidence |
|---|---|---|---|
| **D-1** | The refcv6 §6 **BOX** loss applies **no visibility filter at all** — the v6 seam's does. **59.8 % of B1 boxes are out of field, 83.6 % outside the decode box, 50.0 % behind the ego.** | **G** | MEASURED · `raw/p1_box_path_filter.json`, `raw/p2_b1_join_filter_census.json` |
| **D-2** | The 100-query budget then keeps the **nearest**, so **49.1 % of query slots go to boxes behind the ego** and **27.2 % of available in-field supervision is destroyed by ordering alone** — biased by class (bus **56.0 %**, heavy_truck **43.9 %**, animal **0 %**). | **G** | MEASURED · `raw/p6_query_budget_bias.json` |
| **D-3** | The MAP head's `seen` is a **clip-lifetime** mask, not a camera-field one (`non_causal`, 135/135). **90.1 % of cells the rig can never see at any instant are labelled `seen`**; **11.05 % of map supervision sits on cells `BEVLift` has already zeroed as unobserved.** Fix is one line and its input already exists. | **G/E** | MEASURED · `raw/p4_map_gt_noncausal.json`, `raw/p7_map_seen_vs_lift_valid.json` |
| **D-4** | The `--agent-cls-weight` **corpus-line guard cannot go red on the operator error it names** (expectation and file are selected by the same key). And fixing D-1 makes the banked B1 vector wrong by up to **1.746×**. | **F** | MEASURED · `raw/p5_cls_weight_guard.json`, `raw/p6_…json` |
| **D-5** | **z and h ARE emitted and ARE supervised — the registered open question is RESOLVED.** But the 3-D join is **eval-139 only**; on the **4,427-clip B1 train split `n_z == 0` by construction.** The source parquets are on this box: **unblocked, ~45 min CPU.** | — | MEASURED · source + tests + `…b1eval_agents_3d.jsonl.xz.meta.json` |
| **D-6** | `visible_frac` / the join's `occ` are an **azimuth test on the box centre** and nothing else — no occlusion, no range, no image-edge truncation. B1 **0.4018 / 0.4078 / 0.40195**, and `refc_agents.py`'s quoted 41.06 % is the **parity** line. | E | MEASURED · join sidecars + `raw/p2_…json` |

---

## 1. Q1 — does the BOX head emit z and h, and are they supervised?

### 1.1 Emitted: YES.

`stack/tanitad/models/box3d_head.py:90-92`

```
N_EXTRA_3D   = 2
SLOT3D_FIELDS = SLOT_FIELDS + (("cz", 1), ("h", 1))
SLOT3D_WIDTH  = SLOT_WIDTH + N_EXTRA_3D          # 21 + 2 = 23
```

`SLOT_WIDTH = 21` (`agent_slots.py:176` over `SLOT_FIELDS:160-174`). The widening is **appended,
never interleaved**, and the contract is asserted at import (`box3d_head.py:105-111`): every 2-D
slice offset must survive, or the module refuses. `decode` (`:263-283`) emits `box3d [B,N,7] =
(x, y, z, l, w, h, yaw)` — exactly §6's order — with `h` softplus-decoded and `z` linear.
Decode constants `Z_RANGE_M 4.0` / `H_RANGE_M 2.0` are MEASURED (`:119-124`, 51,844 eval cuboids).
**MEASURED** · source.

### 1.2 Supervised: YES on eval-139, and the registered "n_z = n_h = 0 over 900 windows" is CLOSED.

The defect was one missing keyword on the **second** of two `JoinFileReader` constructions
(`stack/tests/test_eval_join_track_ids.py:1-21`, commit `b43697c`). Positive assertion against
**HEAD**, not the worktree:

```
git show HEAD:stack/scripts/refc_v3_train.py | grep -c "with_track_ids="   -> 2
git show HEAD:stack/scripts/refc_v3_train.py | grep -c "with_rates="       -> 2   (control)
git merge-base --is-ancestor b43697c HEAD                                  -> YES
```

train reader `refc_v3_train.py:6731-6738`, eval reader `:6933-6937` — both carry
`with_track_ids=bool(getattr(args, "join3d", None))`. The hidden coupling that would have
re-opened it (the widening at `:2757` reads `self.map_clip_of_ep`, which only `enable_map_gt`
sets) is closed by the fallbacks at `:6808-6809` (train) and `:6979-6980` (eval).

**END TO END on the real join:** `stack/tests/test_refcv6_perception_realdata.py::
test_the_3d_join_reaches_the_height_targets` exercises join → `zh_for_frame` → `zh_targets` →
`box3d_set_loss` with `n["z"] > 0` and a no-label control beside it.

⚠️ **It is SKIPPED in the default suite.** Run as-is: `18 passed, 2 skipped`, reason
`set $TANITAD_AGENT_JOIN3D to the b1eval 3-D join`. With the env var pointed at the banked join:
**`16 passed`**, no skips. ⇒ *the proof exists and does not run unless someone remembers to point
it at the artifact.* **MEASURED** (ours; both invocations above).

### 1.3 ⛔ But the 3-D join is EVAL-ONLY, so on the TRAIN split z/h are inert.

Two probes for the absence, each with a control:

1. **Filesystem.** `find … -iname "*agents_3d*"` returns exactly two paths, both
   `b1eval_agents_3d.jsonl.xz{,.meta.json}` (control: the same sweep returns 135 `*.sam3mapgt.npz`,
   so the search reads this disk).
2. **The builder's own scope.** `stack/scripts/build_b1_agent_join_3d.py:1` — *"B1 **EVAL** agent
   join -> 3-D"*; `:862` stamps `task: "B1 EVAL agent join -> 3-D …"`.

And the B1 **train+eval 2-D** join's agent schema, read from the file (line 0):
`{cx, cy, yaw, l, w, occ, track_id, cls}` — **no `cz`, no `h`**.

⇒ With `--join3d` pointed at the only existing artifact, the **4,427 train clips** miss the
`(clip_id, frame)` key, `zh_targets(t)` takes its all-False branch (`refc_v3_train.py:2759`) and
`box3d_set_loss` reports `n["z"] == 0` — **correctly, loudly, and with the term at 0.0**. The
architecture is right; the label is absent.

**MEASURED coverage of what does exist** (`b1eval_agents_3d.jsonl.xz.meta.json`): 139 clips,
26,394 lines, **905,512 agents, `n_agents_with_zh` 905,512, `zh_coverage` 1.0**, md5
`ff0e68fd41e86f6a30180ff6173b0685`. Its controls C1/C2/C5/C6/C3/C4 all pass on **all** lines;
C7 is explicitly a **diagnostic, not a gate** (`passed: false`, `gated: false`) and prices a route
not taken — correctly declared, not a failure.

### 1.4 ⭐ THE NEXT LEVER, AND IT IS NOT BLOCKED (Rule Zero)

`ls /c/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1train | wc -l` → **4,440
files**. The 3-D builder is fully parameterised (`--join-2d`, `--obstacle-dir`, `--out`;
`build_b1_agent_join_3d.py:995-1020`) and is **not** eval-hardcoded. The eval half took
**78.8 s for 139 clips** ⇒ the train half is **≈ 42 min of CPU** on this box.

⚠️ One honest degradation to declare in its meta: control **C4** cross-checks the label base face
against `bev-lidar-gt-b1eval`, which exists **only for eval**. On the train split C4 reads
`available: false` and **C3** (the rig `z = 0` road-plane assertion, `tol 0.2 m`) carries the
verification alone.

**I did not run it**: it writes a new multi-hundred-MB corpus artifact, which is a production
action outside a read-only review's remit and inside the DataFlyWheel's lane. It is named here as
an unblocked, priced lever, not as a blocker.

---

## 2. Q2 — the MAP head's ground truth

### 2.1 SAM3 only, no LiDAR: CONFIRMED.

`grep -ci lidar stack/tanitad/models/bev_encoder.py` → **2**; both are the *prohibition*
(`:18` the PI ruling, `:20` "may still be quoted as an INDEPENDENT evaluation reference; nothing
in this module reads it"). Control: `grep -ci sam3` → **11**, so the file was read.
`grep -ni lidar stack/tanitad/data/semantic_map_gt.py` → **0 hits, and the file's 23,786 bytes were
read in the same breath** (its `CHANNELS`/`CART_SPEC` are quoted below). There is no parameter on
either module through which a LiDAR array could arrive. **MEASURED** · source.

### 2.2 Soft CE, 9 classes, seen cells only: CONFIRMED, and all three decisions are declared.

`bev_encoder.map_soft_ce:269-325`. `seen` **must be bool** (`:295-297` — *"a float mask would
silently weight cells instead of selecting them"*); the per-cell term is multiplied by
`m.unsqueeze(1)` at `:305`, so an unseen cell contributes **exactly zero**; `n_cells` travels with
the loss (`:323`); a target whose per-cell fractions leave `1 ± 9/255` on a **seen** cell is
**refused** (`:303-311`). The 9 classes (`semantic_map_gt.py:73-77`) are
`seen-no-map-class · drivable · lane/road line · crosswalk · arrow/text · non-drivable edge ·
hatched area · sidewalk/verge · **not seen**` — i.e. **8 semantic + `not seen`**, kept as a class
rather than renormalised away, which §6's bare *"9 classes"* does not say.

### 2.3 The seen/unseen mask is REAL — verified, not assumed.

Rule (`semantic_map_gt.py:330-331`): `seen = 2*(255 − cart_frac[:,8]) >= 255`, exact integer
arithmetic. The brief's *"`0` = seen-no-class, `255` = unseen"* is the **`fine_codes`** array's
encoding (`meta_json.fine.codes = "0-7 class, 255 not seen"`); the **consumer** reads `cart_frac`,
so both statements are true of different arrays and only the second one binds.

MEASURED over **135 clips / 5,527 frames** (`raw/p3_map_seen_mask_census.json`):

| | |
|---|---|
| `frac_seen_overall` | **0.91267** — strictly between 0 and 1, so the mask selects |
| integer-rule vs float-share-rule mismatches | **0** (discriminating control) |
| frames with zero seen cells | **0** |
| per-clip `frac_seen` | min 0.5639 · p50 0.9435 · max 0.9690 |

⇒ **the loss does not silently train on unseen cells.** What it *does* train on is the subject of
§3.3, and that is the real finding.

### 2.4 ⚠️ Not asked, and load-bearing: the 9 classes are as imbalanced as the boxes.

Label mass **on seen cells** (same probe): `sidewalk/verge` **37.14 %**, `drivable` **34.00 %**,
`seen-no-map-class` **25.90 %** ⇒ **97.05 % in three classes**; `arrow/text` **0.067 %**,
`hatched area` **0.063 %**, `crosswalk` 0.52 %, `lane/road line` 1.66 %. `map_soft_ce` takes **no
class weight** — there is no `cls_class_weight` analogue on this head. A map head can reach a good
soft CE while never predicting five of its nine classes, and only `map_metrics`' **per-class** IoU
(`bev_encoder.py:327+`, correctly never pooled) can see it. **MEASURED.**

---

## 3. Q4 — every filter on the map and box GT paths (class G)

The table is the deliverable; the two rows in bold are where the author's requirement and ours
diverge.

| filter | where | what it protected ITS author against | is that OUR requirement? | how much it removes (v7-B1) | correlated with? |
|---|---|---|---|---|---|
| **F1 FOV / azimuth** | `refc_agents.filter_targets_to_visible:385` | a monocular head trained to hallucinate outside the camera | **YES** — and **the refcv6 box path never calls it** (D-1) | **59.805 %** of 28,958,699 boxes | **class** (0.360 animal … 0.459 rider, 1.27× spread); **not range** (0.364–0.418 flat across 6 bins) |
| **F2 decode box** (0≤cx≤60, \|cy\|≤16) | `refc_agents.visible_target_filter:416` ← `SlotDecodeRanges` ← `GRID_DEFAULT` | a target the head cannot *express* | **YES** — same non-call | **83.626 %** (in-field ∩ box = 16.374 %) | **class, strongly**: `other_vehicle` 0.1149 / `heavy_truck` 0.1160 vs `stroller` 0.3027 — **2.64× spread** |
| **F3 query budget** (keep nearest 100) | `agent_slots.match_slots:181-186` | crowded frames flattering the head; the near field is what the plan acts on | **YES, but only after F1** | 5.21 % of lines over budget; **27.18 %** of available in-field supervision lost | **class**: bus 55.98 %, heavy_truck 43.85 %, automobile 30.13 %, person 27.99 %, animal 0 % |
| **F4 vocabulary** | `targets_from_join:307-311` → `cls = −1` | a class with no stated weight taking an implicit one | YES | `train_or_tram_car` **29,206** = 0.101 % | by definition one class |
| **F5 `--agent-pad`** | `refc_v3_train.py:2564, 2704-2713` | a caller-supplied pad smaller than the data | YES | **none by default** — `pad or reader.max_agents_per_frame` | — |
| **F6 map seen-cells** | `semantic_map_gt.py:330-331` → `map_soft_ce` | supervising cells no sensor covered | **PARTLY — see D-3** | **8.73 %** of grid cells | range, mildly: near 0.9733 / far 0.8520 (**1.14×**); rows 110-119 fall 0.763→0.731 |
| **F7 map coverage floor** | `perception_targets.require_map_coverage:204-275`, `MIN_MAP_COVERAGE 0.90` | a map head trained on a biased subset of windows | YES | **refuses**, never deletes; uses `frac_ok` (the **lower** bound) | n/a — this one is built right |
| **F8 join skips** | `build_obstacle_join.py:773-823` | absent `obstacle.offline` / egomotion / failed registration | YES | **145 train + 2 eval clips** (below) | untestable here (§3.4) |
| **F9 `clip_is_valid`** | `physicalai_r0.py:57` | upstream corpus selection | YES, r0 only | outside refcv6's path (r0 selected the corpus long ago) | — |

### 3.1 ⛔ D-1 — the refcv6 BOX loss has no visibility filter. Two probes, one control each.

**Probe A (static, with control).** `grep -n "visible_target_filter\|filter_targets_to_visible\|
filter_visible" stack/scripts/refc_v3_train.py` → **0 hits**; the same command for
`box3d_loss_row` → **1 hit at `:4372`**, so the file was read and the absence is about the
content. `refcv6_perception_branch.py` likewise: **0 hits** for any filter name, **3** for
`box3d_set_loss`.

The two paths, side by side:

```
refc_v3_train.py:3969   ag  = refc_agents.agent_losses(...)        # v6 seam
                              -> agent_losses:833  filter_visible=True -> visible_target_filter
refc_v3_train.py:4372   _brow = _perc.box3d_loss_row(_s3, _t3, ...) # refcv6 §6 BOX head
                              -> refcv6_perception_branch.py:503 -> box3d_set_loss:327
                                 (no field cut, no range cut, anywhere on this path)
```

**Probe B (data-flow).** `_t3["valid"]` at `:4345` is `batch["agent_valid"]`, which is
`t["valid"][0]` straight out of `targets_from_join` at `:2727`; the only thing between them is a
row `index_select` and `collate_box_targets` (which pads, `perception_targets.py:385`). The
dataset's own docstring states the intent — *"Targets are emitted RAW (no visibility filter here).
The filter lives in `refc_agents.agent_losses` where it can be turned OFF by name for the
deliberate-regression arm"* (`refc_v3_train.py:2687-2691`). **The refcv6 head reads the raw block
and never reaches the place the filter lives.**

**Probe C (by construction — the analytic target).** `raw/p1_box_path_filter.json`: one frame,
three boxes whose visibility is known before any code runs — A `(+20, 0)` visible, B `(−20, 0)`
behind the ego, C `(+120, 0)` at 2× the decode box.

| arm | refcv6 `n_target` | v6 `prefilter → visible` |
|---|---|---|
| all three | **3** | 3 → **1** |
| only A (discriminating control) | 1 | 1 → 1 |
| **Δ** | **+2** | **0** |

⇒ the refcv6 path supervises the behind-ego box and the 120 m box; the v6 path drops both, by
design. `filter_targets_to_visible`'s own docstring is the indictment: *"Without this filter a
monocular head is trained to hallucinate on ~62 % of its supervision, and the resulting AP would
be a measure of how well it guesses at the unobservable."*

**Magnitude, on the right line** (`raw/p2_b1_join_filter_census.json`, full pass over
`b1_train_plus_eval_agents.jsonl.xz`, 875,657 lines / **4,566 clips** / **28,958,699 boxes**,
608.5 s):

| | v7-B1 (MEASURED here) | parity (published at `refc_agents.py:361-367`) |
|---|---|---|
| in field | **40.195 %** (11,639,984) | 41.06 % |
| in field ∩ decode box | **16.374 %** (4,741,807) | 15.67 % |
| **behind the ego (cx < 0)** | **50.038 %** (14,490,476) | not published |
| in-vocabulary boxes | **28,929,493** | 12,122,129 |
| imbalance (automobile : animal) | **1,825.7 : 1** | 1,071.1 : 1 |

⚠️ **The 41.06 % / 15.67 % in `refc_agents.py` are PARITY numbers used to justify a default on an
arm that trains the B1 line.** The conclusion survives — it is *worse* on B1 for the FOV cut —
but the figures are not interchangeable and this file has already produced one retraction from
exactly that substitution.

### 3.2 ⛔ D-2 — and the query budget makes it concrete.

`match_slots:181-186` drops the **farthest by `√(cx²+cy²)`** when targets exceed queries, and
declares the policy: *"the near field is the one the plan acts on"*. That is right **after** F1.
Without it the ordering key runs over a set that is **half behind the ego**, so a car 5 m *behind*
outranks a car 40 m *ahead*.

MEASURED over the whole B1 join at `N_QUERIES_DEFAULT = 100`
(`raw/p6_query_budget_bias.json`, 102.6 s):

| | |
|---|---|
| lines over budget | **45,592 / 875,657 = 5.21 %** |
| query slots spent on those lines | 4,559,200 |
| **spent on boxes with cx < 0** | **2,240,627 = 49.145 %** |
| in-field boxes present | 2,326,106 |
| in-field boxes kept by nearest-N of the **raw** set | 1,675,369 |
| in-field boxes available if **filtered first** | 2,300,748 |
| **in-field supervision destroyed by ordering alone** | **625,379 = 27.18 %** |

Controls: **1,377** lines carry ≥100 in-field boxes and `available == N` on **all** of them
(analytic, non-vacuous — the smoke run of this probe hit `n = 0` and reported a *vacuous* pass,
the advisory's class-F item 3, and the probe now returns `None` on zero items instead).

**The loss is class-biased, and against the classes that matter:** lost ÷ (lost + kept) reads
**bus 55.98 %**, **heavy_truck 43.85 %**, rider 30.40 %, automobile 30.13 %, person 27.99 %,
other_vehicle 13.30 %, **animal 0 %**. Large vehicles are visible far away, so they sit exactly
where a behind-ego box out-ranks them.

### 3.3 ⛔ D-3 — the MAP head's `seen` is a clip-lifetime mask, not a camera-field one.

The artifact says so itself. `meta_json.non_causal` = *"labels use every frame of the clip;
inference must never read this file"* — present on **135 of 135** files
(`raw/p4_map_gt_noncausal.json`). Under the PI's 2026-08-03 ruling that is **legal for a label**.
The defect is that `map_soft_ce` then uses it as the *supervision* mask for a vision-only head.

**The near-analytic proof** (same probe, 135 clips / 2,825 frames). A cell with azimuth > 60° is
outside the rig's only camera at *every* instant, so a `seen` label there can only have come from a
different frame. The count is exact before any file is opened:

| | |
|---|---|
| cells per frame | 7,680 |
| out-of-field cells, **ANALYTIC** (\|y\| > x·tan 60°) | **590** |
| out-of-field cells, **MEASURED** | **590** ✅ identical |
| **fraction of those 590 that are labelled `seen`** | **90.088 %** |
| fraction of the 7,090 in-field cells labelled `seen` | 90.597 % |

⇒ the mask treats never-observable and observable cells **the same to within 0.5 pp**. Supervision
falling outside the instantaneous field is **7.642 %**, against an all-true-mask control of
**7.682 %** — the mask removes **essentially none** of it.

**The sharper number, against the mask the pipeline already has** (`raw/p7_map_seen_vs_lift_valid.json`,
135 clips, real per-clip extrinsics from `extrinsics141.json`, 0 clips unmatched):

| | |
|---|---|
| supervised (`seen`) cells | 19,647,460 |
| `seen` **and** lift-valid | 17,476,890 |
| **`seen` but lift-INVALID** | **2,170,570 = 11.048 %** |
| per clip | min 6.10 % · p50 11.11 % · p90 12.12 % · max 15.51 % |
| lift-valid cells per clip | mean **89.00 %** of 7,680 |
| ANALYTIC control: valid cells outside 60° | **0** (expected 0) ✅ |
| discriminating arms: all-true / all-false `valid` would lose | 0 / 19,647,460 — the measured value sits strictly between |

**`BEVLift.forward` has already zeroed those samples and substituted an `unobserved` embedding**
(`bev_lift.py:229-232, 262-263`), and the loss then asks the head to name a class there. That is
the D-1 defect one head over, and it is worse in kind: on the box head the head *could* in
principle be wrong, here the input is a constant.

⭐ **The fix is one line and its input already exists.**
`refcv6_perception_branch.LiftGeometryBank.geometry:271` already returns `valid [Z,X,Y]`; pass
`seen & valid.any(dim=1)` to `map_soft_ce` instead of `seen`. `n_map_cells` already travels with
the loss row (`:483-488`), so the change is visible in the log the moment it lands, and the
un-masked arm is the deliberate regression.

⚠️ **Honest bound:** 11.05 % is a **lower** bound on non-instantaneously-observable supervision. It
prices only the *geometric* component (azimuth, elevation, range). A cell 55 m ahead behind a truck
is lift-valid and still unobserved, and nothing here can see that without the images.

### 3.4 F8 — the skip reasons, counted.

Not in the file the trainer reads. The combined
`…/2026-09-09-wpc-oracle-gate/raw/b1_train_plus_eval_agents.jsonl.xz.meta.json` carries **only**
`{bytes, md5, n_lines}` and a provenance note — **no skip block**. The counts live in the two
halves' own sidecars:

| split | joined / offered | `no_obstacle` | `no_egomotion` | `error` | `alignment_failed` |
|---|---|---|---|---|---|
| B1 **TRAIN** (`…/2026-09-06-b1-train-join/raw/…meta.json`) | **4,427 / 4,572** | **132** | 0 | **6** | **7** |
| B1 **EVAL** (`…/2026-09-06-b1-agent-join/raw/…meta.json`) | **139 / 141** | **2** | 0 | 0 | — |

145 + 2 = **147 clips skipped**; 4,427 + 139 = **4,566 clips**, 849,263 + 26,394 = **875,657 lines**
— both reproduced **exactly** by my independent pass (§3.1). That agreement is the control that
the census read the same artifact the trainer does.

**Class-G reading:** every one of these four reasons *is* our requirement (no cuboids, no ego
track, a registration that did not converge ⇒ no admissible label). **The correlation test could
not be run**: it needs the 147 skipped clip ids joined against the corpus catalogue
(`clip_index.parquet` / `clip_is_valid` / chunk / city / duration), and I deliberately did not pull
raw clip ids into this package. It is **2.89 %** of clips, so the ceiling on any bias is small —
but "small" is not "measured", and this is named as unresolved in §6.

### 3.5 F4 — the out-of-vocabulary class is a silent half-target.

`ALL_CLASSES` (`bev_raster.py:93-94`) has **10** entries and `train_or_tram_car` is not one;
`AGENT_CLASSES = tuple(ALL_CLASSES)` (`agent_slots.py:156`). `targets_from_join:307-311` maps an
unknown class to **−1**, and `slot_set_loss` masks `cls >= 0`. ⇒ the **29,206** B1
`train_or_tram_car` boxes remain **valid detection targets** carrying presence / centre / size /
yaw / rates supervision, with **no class label**. The banked B1 weight artifact documents exactly
this (`_out_of_vocabulary._handling`), so it is a *declared* behaviour, not a bug — but a reader of
§6's *"+ class"* would not expect 0.101 % of positives to be class-free, and `bus` at 0.71 % of the
corpus is only 7× more common than this non-class.

---

## 4. Q3 — class imbalance and `--agent-cls-weight b1`

### 4.1 The B1 vector is the one a refcv6 arm loads. CONFIRMED.

`refc_v3_train.CLS_WEIGHT_CHOICES:4966-4970` maps `b1 → (agent_cls_weights_b1.json,
CORPUS_LINE_B1)`, used at `:4986` (the stamp) and `:6250` (the model attach). Loaded live
(`raw/p5_cls_weight_guard.json`): `corpus_line` **`v7-b1-physicalai-b1-w120-256x640cyl`**,
10 classes, imbalance **1825.7 : 1**, counts summing to **28,929,493** — identical to my
independent census (§3.1). The digest is **verified, not copied** (`agent_slots.py:352-358`), and
`cls_weight_digest:283-306` is recomputable from the tensor the *model* carries, which is the
correction that made the earlier hand-written digest more than decoration.

### 4.2 ⛔ D-4 — but the `corpus_line` guard cannot go red on the error it names.

The expectation and the file are selected **by the same key**. `CORPUS_LINE_PARITY` /
`CORPUS_LINE_B1` appear in exactly one place outside their definition —
`CLS_WEIGHT_CHOICES` — and **nothing compares either against the arm's actual corpus**
(`--agent-join`, `--v2-cache`, the v2 manifest). Control: `grep -c "agent_cls_weight"
refc_v3_train.py` → 11, so the sweep read the file.

Three arms, `raw/p5_cls_weight_guard.json`:

| arm | outcome |
|---|---|
| A `b1` artifact, `b1` expectation | **LOADED** (`v7-b1-…`) |
| **B `train2400` artifact, its own `parity` expectation — the operator error** | **LOADED** (`parity-…`) |
| C **MUTATION**: `b1` artifact, `parity` expectation | **REFUSED** |

⇒ `guard_can_go_red: true`, `guard_blocks_the_operator_error: **false**`. A refcv6 B1 arm launched
with `--agent-cls-weight train2400` trains on the **parity** vector and nothing refuses. The two
vectors differ by up to **1.857×** (`rider`; also person 1.573×, stroller 1.511×, animal 0.751×) —
independently reproduced here, matching the claim at `refc_v3_train.py:4961-4963`.

**Cheapest fix, in the guard's own idiom:** derive the expected line from the arm — the join path
or the v2 manifest already in `config.json` — and pass *that* as `expect_corpus_line`, so the
comparison has two independent sources. Regression arm: launch a B1 arm with `train2400` and
require a refusal.

### 4.3 ⛔ And fixing D-1 makes the banked B1 vector wrong by up to 1.746×.

The vector is inverse-frequency over the **raw** join. That is the correct scope **today**, because
D-1 means the loss sees every box. The moment the visibility filter is added, the frequencies the
`cls` term actually meets change — and not uniformly (`raw/p6_…json::class_weight_scope_check`):

| class | raw n | post-filter n | banked w | w if counted post-filter | ratio |
|---|---|---|---|---|---|
| other_vehicle | 78,231 | 8,988 | 0.8772 | 1.5317 | **1.746** |
| heavy_truck | 553,977 | 64,278 | 0.1239 | 0.2142 | **1.729** |
| bus | 205,566 | 28,692 | 0.3338 | 0.4798 | 1.437 |
| trailer | 282,911 | 40,451 | 0.2426 | 0.3403 | 1.403 |
| automobile | 21,515,941 | 3,219,256 | 0.0032 | 0.0043 | 1.341 |
| animal | 11,785 | 2,479 | 5.8233 | 5.5535 | 0.954 |
| person | 5,487,965 | 1,193,623 | 0.0125 | 0.0115 | 0.922 |
| rider | 641,740 | 141,927 | 0.1069 | 0.0970 | 0.907 |
| protruding_object | 114,855 | 26,366 | 0.5975 | 0.5222 | 0.874 |
| **stroller** | 36,522 | 11,054 | 1.8791 | 1.2454 | **0.663** |

**Reconstruction control: max \|reconstructed − banked\| = 4.96 × 10⁻⁷** — my probe reproduces the
banked vector's own recipe from the raw counts, so the right-hand column is like-for-like and not a
different normalisation. End-to-end spread **2.63×** (1.746 / 0.663).

⇒ **D-1 and D-4 must be fixed in the same change.** Adding the filter without recounting the vector
swaps one scope error for another — the `anchors.pt` units lesson, in a frequency costume, exactly
as `agent_slots.py:268-277` warns.

### 4.4 The collapse claim itself.

*"2,000/2,000 and 320/320 slots emit ONE class of ten"* is **INHERITED** — it lives in the flag's
help text (`refc_v3_train.py:8145-8150`) and in `H-BOXCLS-1`. I did not re-run it; it is not
contradicted by anything here, and §4.3 says the weighting that answers it needs its scope decided
first.

---

## 5. Q5 — `visible_frac`, and Q6 — geometry alignment

### 5.1 What `visible_frac` MEANS, stated by the artifact.

`…/2026-09-06-b1-train-join/raw/b1train_agents.jsonl.xz.meta.json::conventions.occ`:

> `0` = agent centre inside the 120 deg front-camera field, `1` = outside while the track continues;
> **IS `bev_raster.fov_mask` at agent-centre granularity (P4_PREDICATE_IDENTITY), not an
> independent occlusion label**

⇒ it is an **azimuth test on the box centre**. Not occlusion, not range, not image-edge truncation,
not a size threshold. A 16 m truck whose centre is at 59.9° counts as fully visible.

| line | `visible_frac` | source |
|---|---|---|
| **v7-B1 TRAIN** | **0.4018** (11,270,675 / 28,053,187) | its meta `content_assertion` |
| **eval-139** | **0.4078** (369,310 / 905,512) | its meta `content_assertion` |
| **v7-B1 TRAIN+EVAL** | **0.40195** (11,639,984 / 28,958,699) | **MEASURED here**, `raw/p2_…json` |
| parity | 0.4106 | `refc_agents.py:361` |

**Independent confirmation that the flag is the predicate:** recomputing `atan2(|cy|, cx) ≤ 60°`
from the coordinates and comparing with the stored `occ` over all 28,958,699 boxes gives
**28,958,692 agreements, 7 disagreements** (`frac_agree` 0.9999997582). 7 is a rounding boundary,
not a defect.

### 5.2 Does the box loss train only on visible boxes? **NO — and the concern lands the other way.**

The brief's worry was that the head's effective `n` is far below the headline 28.9 M. It is not:
D-1 means the loss consumes **all** of it. The headline is right and **59.8 % of it is supervision
for something the camera cannot see**, with a further 27.2 % of what *is* visible destroyed by the
query budget on crowded frames (D-2). The honest effective-`n` table for a *correctly filtered*
head is:

| supervision set | n (v7-B1) | share |
|---|---|---|
| every box in the join | 28,958,699 | 100 % |
| in field (F1) | 11,639,984 | 40.20 % |
| **in field ∩ decode box (F1 ∩ F2)** | **4,741,807** | **16.37 %** |

### 5.3 Q6 — do the three extents agree? **YES, and it is asserted at import.**

| extent | value | source |
|---|---|---|
| BEV grid | `x_fwd 60.0 · y_half 16.0 · cell 0.5` ⇒ `(120, 64)` | `bev_raster.BEVGrid:98-111` |
| SAM3 map GT | `{"x_max_m": 60.0, "y_half_m": 16.0, "cell_m": 0.5, "shape": [120, 64]}` | `semantic_map_gt.CART_SPEC:80`, and the same block in every `meta_json` read (135/135) |
| box decode box | `x_fwd_m = GRID_DEFAULT.x_fwd_m`, `y_half_m = GRID_DEFAULT.y_half_m` | `agent_slots.SlotDecodeRanges` |

`bev_encoder.py:87-91` **raises at import** if `CART_SHAPE != GRID_DEFAULT.shape`, and the module
is stride-1 with `padding == (kernel−1)//2 · dilation` throughout so nothing resamples between the
prediction and the label. `build_lift_geometry` samples the trunk at those same cell centres
(`bev_lift.py:146-185`).

⚠️ **The fourth extent is the one that does not agree: the BOX GT has no extent.** The join is
unbounded — `mean_abs_cx_m` **49.76 m** (train meta) and the corpus carries boxes past 120 m and
behind the ego. Nothing truncates it to the 60 × 32 m box on the refcv6 path. That is D-1 restated
as a geometry statement.

### 5.4 Geometry alignment against §12's 416 × 1024 — a scope note, not a defect.

`refcv6_perception_branch.LiftGeometryBank.geometry:271` passes `frame=self.frame` explicitly, so
the §12.1 `h % 32` refusal and the `FRAME_416x1024` declaration are respected and no default bites.
⚠️ `build_lift_geometry:146` still **defaults** to `PHYSICALAI_WIDE120_256x640`; it is unused on
this path but it is a pinned geometry sitting in a signature, which is the class the box head's
`image_hw` refusal (`box3d_head.py:170-174`) was written to remove. ⚠️ `observed=` is never passed,
so lift validity is the projection test only, not the cylindrical cache's own observed-pixel mask
— which makes §3.3's 11.05 % an optimistic (lower) figure on that axis too.

ⓘ All map/box GT figures here are on the **256 × 640** corpus that exists
(`physicalai-b1-w120-256x640cyl`, 4,713 episodes on Thor), not SPEC §12's **416 × 1024**. Nothing
in the map/box GT is image-geometry dependent — both live in **metres** on the rig grid — so the
numbers carry across. What does not carry is the **lift's `valid` mask**, which is a projection
through the frame and must be recomputed at 416 × 1024 before §3.3's fix is quoted there.

---

## 6. What I could NOT answer

1. **Is the F8 skip biased?** 147 skipped clips (2.89 %), reasons counted, **correlation untested**
   — it needs the skipped ids joined to the corpus catalogue, and I did not pull raw clip ids into
   this package. Cheapest closure: a sha12-keyed join of the two sidecars' skip lists against
   `clip_index.parquet` on duration / chunk / city.
2. **Occlusion.** Every "visible" number in this review, ours and the artifacts', is **azimuth
   only**. No probe here can price inter-agent or hood occlusion, so §3.3's 11.05 % and §5.2's
   40.20 % are both **upper bounds on what the camera actually sees**.
3. **The `cls` collapse itself** (2,000/2,000 slots, one class) is INHERITED, not re-run.
4. **The map head on the train corpus.** Only **135** SAM3 GT clips are on this box (the eval-139
   set). Every map number here is **eval-139** and none of it is a claim about the 4,572-clip train
   release, whose production the SPEC dates to *"~22 Sep"* — today.
5. **Whether any of this moves a metric.** This is a static + GT-side review; no arm was trained and
   no AP was measured. D-1/D-2/D-3 are defects in the *supervision*, and their effect size on
   `box3d_ap` / map IoU is unmeasured.

---

## 7. DELIVERABLE MANIFEST

All paths under `D:/Projects/TanitAD/` (repo). **Staged, never committed, never pushed.**

| artifact | path | what it is |
|---|---|---|
| this review | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-refcv6-review/PERCEPTION_DATA_REVIEW.md` | the deliverable |
| P1 instrument | `…/2026-09-22-refcv6-review/code/p1_box_path_filter_probe.py` | analytic 3-box proof that the refcv6 box path does not filter |
| P1 result | `…/raw/p1_box_path_filter.json` | MEASURED |
| P2 instrument | `…/code/p2_b1_join_filter_census.py` | full v7-B1 join filter census, by class / range / clip |
| P2 result | `…/raw/p2_b1_join_filter_census.json` (+ `p2_SMOKE.json`, `p2_run.log`) | MEASURED, 608.5 s, 28,958,699 boxes |
| P3 instrument | `…/code/p3_map_seen_mask_census.py` | SAM3 `seen` mask reality + per-row bias |
| P3 result | `…/raw/p3_map_seen_mask_census.json` (+ `p3_SMOKE.json`, `p3_run.log`) | MEASURED, 135 clips / 5,527 frames |
| P4 instrument | `…/code/p4_map_gt_is_noncausal.py` | analytic 590-cell proof of temporal accumulation |
| P4 result | `…/raw/p4_map_gt_noncausal.json` (+ `p4_SMOKE.json`, `p4_run.log`) | MEASURED, 135 clips / 2,825 frames |
| P5 instrument | `…/code/p5_cls_weight_guard_regression.py` | 3-arm mutation test of the `corpus_line` guard |
| P5 result | `…/raw/p5_cls_weight_guard.json` | MEASURED |
| P6 instrument | `…/code/p6_query_budget_bias.py` | query-budget bias + class-weight scope arithmetic |
| P6 result | `…/raw/p6_query_budget_bias.json` (+ `p6_SMOKE.json`, `p6_run.log`) | MEASURED, 102.6 s |
| P7 instrument | `…/code/p7_map_seen_vs_lift_valid.py` | `seen` vs the lift's own `valid`, real extrinsics |
| P7 result | `…/raw/p7_map_seen_vs_lift_valid.json` (+ `p7_SMOKE.json`) | MEASURED, 135 clips |

**Read-only inputs, not modified, not copied into the repo:**
`D:/Projects/TanitAD-artifacts/a40-rescue/b1_train_plus_eval_agents.jsonl.xz` ·
`D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/` ·
`D:/Projects/TanitAD-artifacts/sam3-maps-eval/` (135 `*.sam3mapgt.npz`) ·
`D:/Projects/TanitAD-artifacts/refcv5v2_final/extrinsics141.json` ·
`C:/Users/Admin/tanitad-data/physicalai/labels/obstacle_offline_b1train/` (4,440 files, listed only).

⚠️ `raw/` is **shared with four sibling reviewers** (their files use `q*`/`bev_*`/`f1_f9_*`
prefixes). Mine are `p1`–`p7`. No sibling file was read, written or overwritten.

**Escalations — these need a decision, not a doc:**

1. **D-1 + D-4 are one change.** Add `visible_target_filter` to the refcv6 box path **and** recount
   the B1 class-weight vector on the filtered scope, in the same commit. Doing either alone
   replaces one scope error with another (§4.3).
2. **D-3 is one line** (`seen & valid.any(dim=1)`, §3.3) and its input already exists in the branch.
3. **D-5's train-split 3-D join is unblocked** — parquets local, builder parameterised, ≈42 min CPU,
   with C4 declared unavailable on that split (§1.4). It is the DataFlyWheel's to run.
4. **The eval z/h proof does not run by default** — `TANITAD_AGENT_JOIN3D` gates it (§1.2). Point
   it at the banked artifact in CI or the regression is decorative.

<!-- REFCV6-REVIEW-PERCEPTION-2026-09-22 -->
