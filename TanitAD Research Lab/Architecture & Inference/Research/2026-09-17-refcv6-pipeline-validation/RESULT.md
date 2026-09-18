# SPEC §10.6 pipeline validation — the refcv6 chain carries END TO END on the GPU, and it found two defects doing it

**Date:** 2026-09-17 · **Evidence class: MEASURED (ours)** · ⛔ **No tier stamp and no metric
family: this is a WIRING validation, not an eval.** 40 steps, batch 2, `n = 2` windows per step.
Nothing here is a capability claim and nothing here may be quoted as driving performance.

## What §10.6 asks for, verbatim

> *"The 139 B1 eval clips are rebuilt at 256 × 1024 and carry the whole chain end to end — trunk →
> lift → map and box heads → planner — **before any pod hour is spent**. Their SAM3 maps (135 of
> 139) are already on the dev box."*

§8's two prerequisites are complete: the **D3 proof package** (48 files landed) and the **D9 RL
lever L1** (Stage A closed — `H-DDV2RL-2` = **FAIL-HARM**). This was the remaining dev-box gate.

## ⭐ The chain carries, and every stage is learning

`resnet34.a1_in1k`, ImageNet-init, **256 × 1024**, CUDA, 40 steps, seed 0. Real 256×1024 v2 cache
(141 episodes), **real SAM3 map GT**, real 3-D agent join, real per-clip extrinsics, v8 labels
joined **139/139 episodes = 100 %**.

| stage | reading | first → last |
|---|---|---|
| **planner** | `traj` | **53.4272 → 16.1725** (3.3×) |
| | `loss` | 321.4581 → 182.0350 |
| **lift → map head** | `map` | 2.0906 → 1.8882 |
| | `n_map_cells` · `map_n_labelled` | 15,264 → 14,324 · **2 / 2 windows** |
| **3-D box head** | `box3d_yaw` · `box3d_z` · `box3d_h` | 1.7912 → **0.6235** · 1.3039 → **0.6975** · 0.2392 → **0.1133** |
| | `box3d_rates` · `box3d_occ` | 38.4188 → 17.7008 · 0.9350 → 0.4813 |
| | `box3d_presence` · `box3d_n_matched` | 0.0958 → **0.6466** · 5 → **34** |
| | `box3d_n_dropped` | **0 → 0** |
| **tactical v6** | `tacv6_goal_bce` · `tacv6_lat_ce` · `tacv6_lon_ce` | 0.7343 · 2.0635 · 2.0492 (all present) |
| | ⭐ **`tacv6_n_scene_mean`** | **496.0 on every step** |

⭐⭐ **`tacv6_n_scene_mean = 496.0` is the PI's R2 ruling, validated live on the GPU: 16 agent
tokens + 480 BEV tokens.** The BEV tokens reach the behaviour decoder on the real corpus, not only
on a hand-built geometry.

⭐ **`box3d_z` and `box3d_h` are real supervised terms now and both fall.** The standing rule was
that *no refcv6 arm may report a z or h number until the 3-D join exists*. It exists, it is joined,
and they are being learned.

⚠️ **`box3d` TOTAL rises (68.68 → 70.66) while every sub-term falls.** That is not a regression:
`box3d_presence` climbs 0.0958 → 0.6466, so matched boxes go **5 → 34**, and `box3d_centre` is a
sum over matches (37.56 → 52.12). More boxes found, more centre error summed. Reporting the total
alone would invert the reading.

⚠️ **`tac_v6` barely moves (0.1383 → 0.1402).** 40 steps is far too few, and the supervision is
sparse: `tacv6_n_supervised_lat` / `_lon` read **1–2 per step**. Consistent with the wiring agent's
*"4 of 10 live steps gave `ga_tac_decoder` exactly 0.0 — label sparsity, not wiring; n = 10, do not
quote a rate."* Same caution here: **n = 40, do not quote a rate.**

## ⛔ Two defects this gate exists to catch, and caught

### 1. The gradient-conflict detector had never met a GPU — FIXED (`213d81a`)

`cosine_stats` created its three float64 accumulators with `torch.zeros((), dtype=dtype)` and **no
`device=`**, so they landed on the CPU and `saq += av.dot(av)` raised *"Expected all tensors to be
on the same device, but found at least two devices, cuda:0 and cpu!"* — **at step 0, inside the
detector's own self-check**.

⭐ **Why it shipped is the part worth keeping.** The detector was mutation-proven, benchmarked and
landed **entirely on CPU**: `test_refcv6_grad_conflict.py` exercises `cosine_stats` **twelve times
with zero CUDA references**, and its own banked overhead note says *"the GPU figure is
UNVERIFIED."* ⇒ **a guard exercised only on one device is a guard that has never met the other
one**, and no amount of CPU mutation testing could have found it — only running it could.

Fixed by deriving the device from the first non-`None` gradient across both lists. Accumulation
stays **float64**, because the module's contract is that the `+1` / `−1` controls are **identities,
not tolerances** — a fix that removed the error while turning `1.0` into `0.9999997` would have
broken the thing the module exists to report. Four new CUDA arms, each `skipif`-labelled
**INCONCLUSIVE, never passing**; **mutation proof: 3 of 4 go red on the shipped defect, baseline
green at 34.**

### 2. ⛔ `resnet101` — §10.2's PRIMARY trunk — does NOT fit the dev box

At 256 × 1024, batch 2, it dies with `torch.OutOfMemoryError` inside the trunk's `conv3` on the
**8 GB** card. `resnet34` (§10.2's comparison trunk) trains in the same configuration.

⇒ **This is a pod-sizing fact, not a defect**, and it is exactly what §10.6 exists to surface
*before* a pod hour is committed: the primary arm cannot be smoke-tested on this box, so its first
run will be its first run at scale unless a larger card is used for a rehearsal.

## ⛔ Geometry: 256 × 1024 and not 408 × 1024

`408 × 1024` is **refused by `timm_trunk.py:216`** (`h % 32`; **408 % 32 = 24**) — verified by
constructing each candidate: `256×640` ✅ · `256×1024` ✅ · **`408×1024` ⛔** · `416×1024` ✅.
See `PI_DECISION_QUEUE.md` **item 20**, default **416 × 1024**. §10.1's standing amendment is
256 × 1024 and its cache is already built, so this validation used it.

## What is NOT established

* ⛔ **No capability claim.** No tier stamp, no metric family, no held-out read. 40 steps.
* ⛔ **`resnet101` is unvalidated end to end** — it never completed a step here.
* ⭐ **Defect 1's fix IS proven on the real chain, not only in a unit test.** The detector-on
  re-run reaches its own self-check on the GPU and prints
  `conflict controls OK: cos(g,g)=1.0 cos(g,-g)=-1.0 detached cos=nan conflict=0.0` — the `±1`
  controls read **exactly**, which is what the float64 accumulation was preserved for, and the
  degenerate side reads **NaN** rather than a tolerated epsilon, as the module's contract requires.
  `grep -c "same device"` on that log reads **0**. ⚠️ That arm is still running its 40 steps; only
  the crash-clearing and the identities are claimed here.
* ⭐ **Map coverage is now MEASURED and fully explained — see the section below.** An earlier
  draft of this file said the run *"did not count how many"* clips fell below
  `--map-min-coverage 0.90`. It had counted them all along: the gate writes its whole report into
  `config.json`, and it reconciles exactly.

## ⭐ Map coverage: the gate had already counted, and two independent instruments agree

`--map-min-coverage 0.90` is **a refusal, not a filter**, and it is evaluated over **windows, not
clips** — over this dataset's own window index, before the GPU. Both arms' `config.json` carry the
identical report at `refcv6_perception.map_gt_stats.train`:

| | |
|---|---|
| windows · clips | **23,772** · 139 |
| `frac_ok` · `frac_ok_upper` | **0.9712** · **0.9712** — identical |
| verdict vs the 0.90 floor | **PASS** |
| states | `ok` **23,088** · `no_file` **684** · `frame_out_of_range` **0** · `inconclusive` **0** |
| `reasons` | exactly **4** clips, each `FileNotFoundError errno 2` |

⭐ **The deficit is fully explained and nothing is unaccounted for.** `no_file` is the *only*
non-`ok` state, so every uncovered window belongs to one of those four clips. Independently, from
the cache manifest and a directory listing (no trainer involved): **139 clips, 135 map files, 4
missing, 0 orphans** — and **the four sha12s are the same four**, set-identical to the trainer's
`reasons` keys. The arithmetic closes too: windows/clip = `n_frames − 30`, the four clips carry
201 frames each → **171 windows each → 4 × 171 = 684**, exactly the `no_file` count.

⭐ **`frac_ok == frac_ok_upper` means the bound is tight, not merely satisfied.** The gate's
comparison deliberately uses the LOWER bound, so an inconclusive window would count *against* the
run — *"we could not tell"* is not coverage. There are **zero** inconclusive windows here, so the
2.877 % is exact rather than a worst case.

⚠️ **Correction to a claim made earlier in this session: those four clips are NOT "skipped by the
chain".** `_map_item` emits their windows with `map_label = False` and an all-false `map_seen`, so
`map_soft_ce` scores **zero cells** on them — deliberately, so the loss cannot read `0.0` as
*"supervised, and perfect"*. The trunk still sees those frames and still takes planner and 3-D-box
gradient from them; only the **map term abstains**. "Skipped" would have implied 4 clips of lost
training signal. The real cost is 2.877 % of the map head's supervision, and nothing else.

## Artifacts

`raw/r34nocd_metrics.jsonl` (8 logged rows, steps 5→40) · `raw/r34nocd_config.json` ·
`code/pipeline_val.sh` · `code/boxstat.py`. Clip identifiers redacted to **sha12**; the scan found
**0** in both files.

## ⭐⭐ CLOSURE 2026-09-18 — all 139 clips carry the chain end to end, at the PI's 416 × 1024

§10.6 asks that *"the 139 B1 eval clips … carry the whole chain end to end"*. The 40-step
arms above drew **80** windows of 23,772. This closes it properly, on the geometry the PI
ruled on 2026-09-17.

### How, and why the coverage is a FACT rather than a probability

⛔ `refc_v3_train` **refuses** a run whose train and eval caches share episodes
(*"a held-out split that is not held out measures memorisation"*). That refusal is correct
and was **not worked around**: the 139 clips were split into two **disjoint** halves
(sorted-order alternating, so neither half is biased toward one end of an id-sorted
corpus) and each pass **evaluates one half while training on the other**.

⭐ The eval subset is built as
`perm = randperm(len(e_ds), generator=Generator().manual_seed(12345))[:nb*batch]` —
**deterministic** — so which clips it touches was **replayed offline against the cache
manifest before either pass ran**: split A needs **150** eval-batches to touch all 70
clips, split B needs **130** for all 69. The passes used **250** and **200**.

### Result — both halves, every head

| | **split A** (70 clips) | **split B** (69 clips) |
|---|---|---|
| windows through the chain | **500** | **400** |
| planner `eval_traj` | 24.373 | 24.848 |
| map `eval_map` · `n_map_cells` | 2.350 · **14,267.2** | 2.354 · **13,847.8** |
| 3-D box `n_matched` · **`n_dropped`** | 40.19 · **0.0** | 35.96 · **0.0** |
| agent `n_matched` · **`rows_no_cam`** | 8.58 · **0.0** | 7.18 · **0.0** |
| tactical **`tacv6_n_scene_mean`** | **496.0** | **496.0** |
| `nav_injected` · `ego_injected` | 1.0 · 0.0 | 1.0 · 0.0 |

⭐ **900 windows over all 139 clips, every head live, zero dropped boxes, zero rows
without a camera.** The two halves agree closely on every reading, which is itself
evidence that nothing clip-specific breaks: had one half contained a pathological clip,
the halves would not track each other this well.

⭐ **The map gate cross-checks the earlier count from a third direction:** split A reports
`no_file` **171** (= 1 clip × 171 windows) and split B **513** (= 3 × 171). **1 + 3 = 4**,
exactly the four map-less clips measured independently from the manifest and the directory
listing. `frame_out_of_range` and `inconclusive` are **0** in both.

### ⚠️ A finding this pass surfaced, and it is NOT a sampling accident

`eval_box3d_z` and `eval_box3d_h` read **0.0** — with **`eval_box3d_n_z = 0`** and
**`eval_box3d_n_h = 0`**, on **BOTH** halves, over 900 windows. ⇒ **the eval path
supervises neither z nor h at all**, while the training path does (`box3d_z` 1.3039 →
0.6975 in the 40-step arm).

⛔ **Quoting `eval_box3d_z = 0.0` without its `n` would read as "perfect"** — the exact
`tac_goal` inversion `_map_item`'s own docstring was written to prevent. The instrument
reports the count beside the value, so it is readable; **the work item is why the eval
join does not build the z/h targets**, and until that is answered **no refcv6 arm may
quote an eval-side z or h number**.

### ⛔ What this does NOT establish

* ⛔ **Still no capability claim**: 1 training step, ImageNet-init trunk, eval cache == the
  object under test by design. **No tier stamp, no metric family.** This establishes
  **reachability** — that every clip's data flows through trunk → lift → map + box3d heads
  → planner without error — and nothing else.
* ⚠️ `resnet34`, not `resnet101`: §10.2's PRIMARY trunk OOMs on the 8 GB card at 416 × 1024
  **batch 1**. Its shapes are proven separately on CPU (`wallclock_s` 382.9, every head
  live, rig coverage 139/139).
* The split halves are **copies, not links** — D: is exFAT and both `os.link` and
  `os.symlink` fail there with `WinError 1`.

`cover_evalA` wallclock **3,221.9 s**, `cover_evalB` **2,534.3 s**; both `summary.json`
carry `"done": true`.

<!-- PIPEVAL-139-CLIP-COVERAGE-CLOSED-2026-09-18 -->

## ⭐ WORK ITEM CLOSED 2026-09-18 — the eval z/h term was DEAD, and the cause was one missing keyword

The closure above flagged an open item: `eval_box3d_z` / `_h` read **0.0 with `n_z` =
`n_h` = 0** over 900 windows on both halves, while the TRAIN side of the same runs read
`box3d_n_z == box3d_n_matched` **exactly**. It is found, fixed and **re-measured on the
same 900 windows**.

### Cause

`refc_v3_train` builds a `JoinFileReader` **twice** — train at `:6111`, eval at `:6310` —
and **only the train one passed `with_track_ids`**. `lookup_track_ids`'s own docstring
says it returns **None** without that flag, so `__getitem__` fell through to
`zh_targets(t)`'s **all-False** mask and every eval z/h term was masked out.
⇒ **two construction sites, one missing a keyword.** A default that silently disables
supervision is worse than a refusal, because the run record says nothing.

### Re-measured, all 139 clips, the SAME 900 windows

| | before A | **after A** | before B | **after B** |
|---|---|---|---|---|
| `eval_box3d_n_matched` | 40.192 | **40.192** | 35.955 | **35.955** |
| `eval_box3d_n_z` | **0.0** | **40.192** | **0.0** | **35.955** |
| `eval_box3d_n_h` | **0.0** | **40.192** | **0.0** | **35.955** |
| `eval_box3d_z` | 0.0 | **1.4905** | 0.0 | **1.4744** |
| `eval_box3d_h` | 0.0 | **0.2489** | 0.0 | **0.2608** |
| `eval_box3d` (total) | 69.320 | 71.060 | 76.472 | 78.207 |

⭐ **`n_matched` is IDENTICAL before and after** — the fix changed **only** the z/h
supervision and nothing about matching. That control is what makes the rest readable.
⭐ **`n_z == n_matched` now holds on both halves**, the same identity the train side always
had. ⚠️ The `box3d` TOTAL rises because two previously-masked terms now contribute —
**correct, not a regression**, the same aggregation caveat this package already carries.

⇒ **The prohibition is lifted: refcv6 arms may quote eval-side z and h numbers again**,
provided they are quoted with their `n` as every term in this package is.

<!-- PIPEVAL-EVAL-ZH-FIXED-2026-09-18 -->
