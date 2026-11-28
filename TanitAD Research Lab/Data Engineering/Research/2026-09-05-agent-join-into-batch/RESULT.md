# `E-AGT-HEAD` TRAINS — the join was there all along, and **32 queries is refuted on the train corpus**

**Stream:** DataFlyWheel · **2026-09-05** · branch `agent/arch-inf-20260803`
**Evidence class:** MEASURED (ours). Artifacts named per claim; raw JSON in `raw/`.

---

## The headline, in four numbers

| | |
|---|---|
| **P1** the banked train join | md5 **`24cbdca8c3b23aafc2fb17e6bf99cf76` MATCH**, and CONTENT-verified through its real consumer: `n_records 433040 · n_clips 2308 · has_occlusion_flags True · has_classes True`, with `n_agent_boxes` **12,122,129** and `visible_frac` **0.4106** re-derived, not copied |
| **P2** the batch contract | every key `compute_losses_v3` reads is emitted at its dtype/shape. Pre-filter **n = 898** targets → in-field **428 (47.7 %)** → in-field ∩ decode box **158 (17.6 %)** on a 64-window sample |
| **P3** the train density | in-field ∩ decode box: mean **4.39**, p99 **30**, **max 94** (val40: mean 3.16, p99 19, max 24). ⛔ **`--agent-queries 32` drops 41,362 boxes on 3,250 frames and the nearest sacrificed target sits at 13.1 m** |
| **P4** `E-AGT-HEAD` | **reached a checkpoint with a real, moving detector loss**: over 600 steps `agent_centre` **19.31 → 3.98 m**, `agent_presence` **0.426 → 0.141**, `agent_cls` **2.034 → 0.631**, `agent_size` **4.03 → 1.18 m**. Three deliberate-regression controls all REFUSE |

---

## P1 — the join existed, and the absence claim was the `CLAUDE.md` class

refcv5's escalation #7 read *"Needs a train-corpus join (only val40 exists on this box)"*. The
join had been built **2026-08-17** and banked to HF. Pulled in **11.5 s** (136,689,648 B) and
verified two ways:

```
md5(joins/train2400_agents.jsonl.xz) = 24cbdca8c3b23aafc2fb17e6bf99cf76   MATCH
JoinFileReader -> {n_records: 433040, n_clips: 2308, has_occlusion_flags: True}   == meta.reader_verify
re-derived independently:  n_agent_boxes 12,122,129 (exact)   visible_frac 0.4106 (exact)
```

⚠️ **Framed as the class, not as a criticism.** Escalating was the right action — the stream
probed its own box, found only val40, and said so. The record had a second home it could not see
from where it stood. ⇒ The durable fix is applied **in the code that needs it**: `--agent-join`'s
help text now carries the HF path, the clip/frame counts and the md5, so the next reader does not
have to already know where to look.

⚠️ **Two things the second location did not advertise, both found by looking rather than assuming.**

1. The meta sidecar is `joins/train2400_agents.jsonl.xz**.meta.json**`, not the
   `joins/train2400_agents.meta.json` the package doc names. Listing the repo found it; a direct
   fetch of the documented name 404s.
2. ⛔ **The two joins' metas record `md5` over DIFFERENT ARTIFACTS.** val40's `summary.md5` is the
   md5 of the **decompressed** `.jsonl`; train's is the md5 of the **compressed** `.xz`
   (decompressed = `b7c6b5a7da2d942e8be9dc9fc316e9a9`). The refcv5 stream's C5 check, inherited
   from val40, therefore **REFUSES the train join** — a correct-looking checker rejecting a good
   file. This package checks both by name and records which artifact the digest covers. *Same
   family as the units trap: a true digest quoted about the wrong artifact.*

---

## P2 — the batch contract, matched not invented

The consumer is `refc_v3_train.compute_losses_v3`, which builds `tgt_ag` and hands it to
`refc_agents.agent_losses`. Every key/dtype/shape below was **read off that call site** before a
line was written. MEASURED on a 64-window collated batch (`raw/batch_contract.json`):

| key | shape | dtype | source |
|---|---|---|---|
| `agent_box` | `[B, N, 4]` | float32 | `(cx, cy, l, w)` metres, ego frame `+x` fwd `+y` LEFT |
| `agent_yaw` | `[B, N]` | float32 | rad |
| `agent_cls` | `[B, N]` | int64 | index into `bev_raster.ALL_CLASSES`, **−1 = unknown** |
| `agent_valid` | `[B, N]` | bool | padding mask |
| `agent_occ` | `[B, N]` | float32 | the join's P4 flag (−1 = no flag) |
| `agent_rates` | `[B, N, 3]` | float32 | `(v_rel_x, v_rel_y, yaw_rate_rel)` |
| `agent_rates_mask` | `[B, N]` | bool | ⛔ a rate that was not observed is MASKED, never zero-filled |
| **`agent_label`** | `[B]` | bool | **NEW** — NO_LABEL vs LABELLED-CLEAR |
| `agent_n_raw` / `agent_n_truncated` | `[B]` | int64 | **NEW** — so a drop is countable |

### The pre/post-filter n, and the control that says the wiring is right

```
pre-filter                            n = 898
in-field only (|az| <= 60 deg)        n = 428   0.4766     corpus visible_frac  0.4106
the join's OWN occ == 0 flag          n = 428   0.4766     <- IDENTICAL, independently
in-field AND decode box (60 x +-16 m) n = 158   0.1759     corpus                0.1567
```

⭐ **The strongest single line in this package:** my `filter_targets_to_visible` predicate and the
join builder's own `occ` flag select **exactly the same 428 rows**. Two implementations written
months apart, on different sides of the label boundary, agreeing box for box — that is what makes
the geometry claim admissible rather than plausible. The 0.4766-vs-0.4106 gap is a 12-clip
subsample of a 2,308-clip corpus, and the corpus-level figure was re-derived at 0.4106 exactly.

### Two design decisions, stated because they are load-bearing

1. **The filter stays in the LOSS, not the dataset.** `agent_losses(filter_visible=…)` is the
   deliberate-regression arm; filtering at the dataset would delete that control. The dataset
   emits raw rows and reports `n` both sides.
2. **`n_pad` is the join's own MEASURED `max_agents_per_frame`**, so the dataset never truncates
   and `match_slots` keeps its counted nearest-N policy as the only drop. A fixed pad would hide
   drops the loss exists to report. (Train corpus raw max = **312**; the 16-clip smoke = 161.)

### ⛔ The id-space trap this wiring now refuses

`JoinFileReader` accepts the 63-bit **stable** id *and* a **legacy 16-bit** key that is the first
4 BYTES of the clip-id string. MEASURED: that key collides on **34 of the 2,308** train clips, so
an episode **absent** from the join can match a **different** clip that is present — a label
corruption no downstream metric could attribute. `enable_agent_join` now REFUSES a legacy-only
match unless `--agent-join-allow-legacy-ids` is passed **by name**, and stamps
`id_space: "legacy-16bit-COLLIDING"` into `config.json`.

---

## P3 — ⛔ **`--agent-queries 32` IS REFUTED ON THE TRAIN CORPUS**

Escalation #3 was right that this was unmeasured. It is now measured, over all **433,040 frames /
12,122,129 boxes / 2,308 clips** (`raw/train_agent_density.json`).

**Cut = in-field ∩ decode box — the only honestly-trainable target set:**

| corpus | n_frames | n_boxes | mean | p95 | p99 | p99.9 | **max** | zero-frames |
|---|---|---|---|---|---|---|---|---|
| **val40** (the basis for 32) | 7,400 | 23,389 | 3.16 | 13 | 19 | 24 | **24** | 34.5 % |
| **train2400** | 433,040 | 1,899,481 | **4.39** | 17 | 30 | 68 | **94** | 30.8 % |

**What `match_slots` would drop, on train:**

| N | frames with a drop | boxes dropped | **nearest sacrificed target** |
|---|---|---|---|
| 16 | 23,103 (5.34 %) | 212,224 (11.17 %) | **7.3 m** |
| 24 | 8,614 (1.99 %) | 88,864 (4.68 %) | **9.8 m** |
| **32** | **3,250 (0.75 %)** | **41,362 (2.18 %)** | **13.1 m** |
| 48 | 729 (0.17 %) | 16,155 (0.85 %) | 19.4 m |
| 64 | 480 (0.11 %) | 6,218 (0.33 %) | 33.9 m |
| **94** | **0** | **0** | — (C4 reads exactly 0.0) |

⇒ **The verdict, against the stream's OWN stated rule.** `AgentSeamConfig.queries`'s docstring
adopts *"a covering N with ZERO drop on the FILTERED set"* and explicitly rejects "use the p99"
because *"that 1 % is not random — it is exactly the crowded frames"*. On val40, 32 satisfies that
rule. **On train it does not**: the zero-drop floor is **94**, and the reasoning that refuted 16 on
val40 (nearest sacrificed target at 38.5 m) applies to **32 on train with far more force**, because
**13.1 m is inside the braking envelope**, not clutter.

**This is not a val40 error.** Both numbers are correct about their corpus. It is a **scope**
error of exactly the family `CLAUDE.md` catalogues — a measurement quoted outside the corpus it
was taken on — and it is why the docstring's own "⚠️ measured on val40, the train distribution is
UNMEASURED" caveat was the right thing to write and the wrong thing to leave standing.

**What I recommend, and what I refuse to decide alone.** The zero-drop floor is 94; the query
table costs 64 params per query (94 → 6,016, still 0.2 % of the 2–4 M band), but decoder
cross-attention scales with N. The honest options are **94 (zero drop, the stream's own rule)** or
**64 (0.33 % of boxes, nearest sacrifice 33.9 m — clutter, not the braking envelope)**. **32 is
not among them.** ⇒ **Escalated to the Master Mind** rather than silently re-defaulted, because
`--agent-queries` is a registered arm knob.

**Controls, all green except one box:**

* **C0 re-statement control: PASS** — the re-stated cuts reproduce the banked val40
  `infield_bevbox` table **exactly** (`mismatches {}`), so the train table is produced by the same
  predicate as the val40 one. Upstream module and JSON verified sha256-identical to their G:
  originals (`9515504070c24b6e…`, `c1681cc115f2d107…`).
* **C2**: `n_frames 433040 ✓ · n_clips 2308 ✓ · n_boxes 12122129 ✓ · visible_frac 0.4106 ✓`.
* **C3 nesting violations: 0.** **C4 zero-drop at N = max: exactly 0.0.**
* ⚠️ **C1: 1 disagreement in 12,122,129 boxes** (val40: 0). Reported, not hidden: it is a single
  box on the ±60° boundary where the builder's and my `arctan2` comparisons land on opposite sides
  of the tie. Rate 8.2e-8; it changes no table above. Named so nobody later "discovers" it.
* Class histogram over the trainable set: `automobile 1,326,130 · person 434,044 · rider 44,334 ·
  heavy_truck 39,358 · trailer 21,924 · bus 12,470 · protruding_object 12,098 · stroller 4,409 ·
  other_vehicle 3,180 · animal 610` (+ 924 unknown). **All 10 corpus classes present** — the
  enum in `bev_raster` is the right one and the hand-written `bicycle/motorcycle/train` tuple the
  refcv5 docstring warns about would indeed have trained three impossible classes.

### ⭐ The load-bearing number, reproduced by a SECOND implementation

`--agent-queries` is a registered arm knob, so the number that refutes 32 must not rest on one
script. `indep_max.py` re-derives it with **no numpy, no imported cut predicate and no shared code**
with `measure_train_agent_density.py` — a plain Python loop over the raw jsonl
(`raw/indep_max.out.txt`):

```
n_frames             433040    expect 433040    True
n_boxes            12122129    expect 12122129  True
visible_frac           0.4106  expect 0.4106
infield_bevbox boxes  1899481  expect 1899481   True
MAX per frame              94  expect 94        True
frames > 32              3250  expect 3250      True
frames > 24              8614  expect 8614      True
nearest sacrificed target at N=32: 13.0897 m
```

⚠️ This is the check `CLAUDE.md` insists on and that repeated runs of ONE script cannot give: *"a
second probe means a different mechanism, not the same command run again."* Two independent
mechanisms agree to the last box, so **max 94** and **3,250 frames dropping at N = 32** are facts
about the corpus, not about a script.

---

## P4 — `E-AGT-HEAD` reaches a checkpoint with a **real, moving** detector loss

**Run:** `raw/smoke_agt_head_600.{config,metrics,summary}.json` · CPU, off-Drive mirror, 600 steps.

```
[v3] agent join loaded: 3081 records / 16 clips (filtered out 429959) in 19.6 s
[v3] agent join: 16/16 episodes (legacy-16bit-COLLIDING), 2728/2797 windows labelled (97.5 %),
     100475 prefilter target boxes, pad 161
[v3:hier] ckpt step 600 -> ckpt.pt        [v3:hier] DONE — summary.json written
```

| step | presence | cls | centre (m) | size (m) | yaw | n_labelled/20 | n_target | n_dropped |
|---|---|---|---|---|---|---|---|---|
| 50 | 0.4261 | 2.0341 | 19.3122 | 4.0286 | 0.7370 | 20 | 103 | 0 |
| 150 | 0.2643 | 1.3070 | 6.6474 | 3.0785 | 0.6592 | 20 | 74 | 0 |
| 300 | 0.1850 | 0.8306 | 6.1281 | 0.9856 | 0.7184 | 19 | 78 | 0 |
| 450 | 0.1724 | 0.6977 | 4.6668 | 1.1679 | 0.7589 | 19 | 102 | 0 |
| **600** | **0.1409** | **0.6313** | **3.9832** | **1.1754** | **0.5580** | 19 | 72 | 0 |

Every detection term is **non-zero and falling**; total loss 63.70 → 32.21. `agent_n_labelled`
varying 18–20 of 20 is the **NO_LABEL mask firing**, visible in the log rather than silent.

### The three deliberate-regression controls — all REFUSE, none scores

| control | result |
|---|---|
| `--agents head --w-agent 1.0` **without `--agent-join`** | ⛔ refuses at **preflight, before the GPU**: *"has NO LABELS … a REFUTATION manufactured by missing supervision"* |
| `--agent-join` pointed at the **val40** join (wrong corpus) | ⛔ `ValueError: … no usable records for the 16 requested episode ids (7400 filtered out) — that is the WRONG JOIN for this corpus, not an empty file` (exit 1) |
| legacy-id join **without** `--agent-join-allow-legacy-ids` | ⛔ refuses and names the collision risk (exit 1) |

⭐ Guard **ORDER** matters and is now pinned: the `--w-agent 0` weight guard fires **before** the
label guard, because with `--w-agent 0` no join could help. Getting this backwards broke the
refcv5 stream's own `test_agent_head_without_its_LOSS_REFUSES`, which is exactly what that test is
for.

### ⚠️ What this run is NOT

The only real-episode source on this box is a **non-parity 400-episode epcache** with **legacy
ids**, 9-channel 256 px. The tiny rig wants 1-channel 64 px, so `make_tiny_joined_cache.py`
reduces the **real** frames (latest RGB triplet → grey → 64 px) while preserving `episode_id`,
`poses`, frame order and count. ⇒ **This is a WIRING claim — the loss is real, supervised and
falls. It is NOT a detection result**, it is not tier-stamped, and a 64 px grey frame cannot
support monocular 3D detection. A real arm needs a **v2 cache with stable ids** on a pod.

---

## What changed in the repo

| file | change |
|---|---|
| `stack/scripts/train_p8_occupancy.py` | `JoinFileReader` gains `episode_ids=` (memory: the full train join is ~2.1 GB RSS), `with_rates=` (streams a 3-deep per-clip window through the EXISTING `agent_slots.track_rates_from_join`, and REFUSES out-of-order records), `.xz` input, `max_agents_per_frame`, `lookup_rates()`. **Every default preserves the historical behaviour** — pinned by `test_reader_defaults_are_unchanged`. |
| `stack/scripts/refc_v3_train.py` | `V3Dataset.enable_agent_join()` + `_agent_item()`; the target block emitted at the window's NOW frame `t + w - 1` (the SAME instant the v7.2 tactical labels are read at); `--agent-join / --agent-pad / --agent-join-no-rates / --agent-join-allow-legacy-ids`; the preflight label guard; NO_LABEL row selection in `compute_losses_v3`; `agent_join_stats` in `config.json`. |
| `stack/tests/test_refc_v3_agent_join.py` | **19 new tests** (all green), pinning the contract, the NO_LABEL/LABELLED-CLEAR distinction, rate masking, the `t_s`-based denominator, `.xz`, the three refusals, and that the loss falls under optimisation. |

`pytest -q` over `stack/` is green (see the manifest for the one pre-existing mirror-staleness
collection error and its fix).

---

## ⚠️ ESCALATIONS — for the Master Mind, not for a reader to find later

1. ⛔ **`--agent-queries 32` is REFUTED on train** (P3). The zero-drop floor is **94**; at 32 the
   nearest sacrificed target is **13.1 m**. `AgentSeamConfig.queries`'s docstring and
   `--agent-queries`' help both still say "32 drops ZERO targets" — **true on val40, false on
   train**. Needs a ruling (94 or 64), then a docstring correction. **I did not silently change
   the default**: it is a registered arm knob.
2. ⛔ **`--agent-w-project` / `--agent-w-ground` are SILENTLY NO-OPS on every current arm.**
   `refc_v3_train.py` sets `model._rig_camera = None` and never assigns it; `agent_losses` guards
   both terms on `cam is not None`, so a run can pass `--agent-w-project 0.2`, stamp it in
   `config.json`, and compute nothing. **This is precisely the failure the stream just fixed one
   level up** (a weight set while the thing it weights is absent), in the camera's costume. The
   fix is either to build the `RigCamera` from `rig_projection` at model setup, or to REFUSE the
   flags while `_rig_camera` is None.
3. ⚠️ **`w_agent` and `w_u0` are not in `config.json`.** They travel as plain model attributes
   (`model._w_agent`) and `AgentSeamConfig.as_dict()` does not carry them, so a finished run's own
   record cannot say what weight its detector was trained at — the `SEAM_STATE.md` failure again.
   One line in `_seam_stamp`.
4. ⚠️ **The two joins' `summary.md5` cover different artifacts** (P1). The refcv5 density script's
   C5 check refuses the train join for this reason alone. Either restate the metas or make the
   checker try both.
5. ⚠️ **No parity-corpus episode source on this box.** The join is keyed to the parity clips; the
   only local real episodes are a non-parity epcache with **legacy colliding ids**. A real
   `E-AGT-HEAD` arm needs a **v2 cache with stable ids** on a pod. The refusal in (P2) makes that
   requirement loud instead of silent.
6. ⚠️ **C1 = 1 boundary disagreement in 12.1 M boxes** — recorded, harmless, named so it is not
   rediscovered as a defect.
