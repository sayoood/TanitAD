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
