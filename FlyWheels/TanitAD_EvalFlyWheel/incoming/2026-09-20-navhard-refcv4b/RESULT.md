# W7 · refcv4b on NAVSIM v2 `navhard_two_stage` (EPDMS) — RESULT

**Package** `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b/`
**Agent** W7 (EvalFlyWheel, P7 TanitEval) · **Dates** 2026-09-20 → 2026-09-21 (Europe/Berlin) · **Box** dev box
**Status** ✅ **COMPLETE.** Final run `taniteval/results/bench/navsim_v2/navhard_two_stage/20260921T122714Z-navsim_v2-refcv4b_b1_v72_40k-859e25` — judge SUCCESS (4/4 arms OK), contract 0 errors, **criteria 0 violations on all four arms**, W5 report + failure gallery rendered.

---

## ⭐ THE RESULT

**Stamp on every number below:** T1-family · stage 1 loop **OPEN** · stage 2 **UNRULED** (a PI ruling
is pending, ⛔ not guessed) · IDM-reactive background traffic, ego non-reactive · ⛔ never closed loop ·
protocol `EPDMS_v2_navhard_two_stage`, devkit `autonomousvision/navsim@0a380a9` · refcv4b ckpt md5
`99b573e8277d94a5e3bfbf630cb4d751` (`refcv4b-b1-v72-40k`, step 40,284) · **zero-shot** (no NavSim
training). All MEASURED from the final run's `summary.json`.

| arm | inputs | **official two-stage EPDMS** (`score`) | **S2-EPDMS-u** (stage-2 statistic) | stage-1 mean | stage-2 mean |
|---|---|---|---|---|---|
| **A1 · refcv4b** | frames (3-cam stitch) + t0 ego + NavSim command | ⛔ **UNDEFINED** — 450 stage-1 rows are the declared CV stand-in (n = 450) | **0.3841** | 0.2896 *(= CV: it IS the stand-in)* | **0.3742** |
| STOP | nothing | **0.2985** [0.2737, 0.3253] | **0.4828** | 0.5632 | 0.4702 |
| ECHO | t0 ego only | **0.1429** [0.1168, 0.1698] | 0.3413 | 0.3866 | 0.3268 |
| CV | t0 speed | **0.1148** [0.0825, 0.1450] | 0.3389 | 0.2896 | 0.3294 |

Intervals on the official EPDMS: `navsim_log_cluster_bootstrap`, 76 log clusters, B = 2000, seed 0.
✅ **CV reproduces the official leaderboard exactly: 0.11481608441648.** ⚠️ The HYBRID combined row for
A1 (CV stage 1 × refcv4b stage 2) is **0.1113** and is ⛔ **not a refcv4b number** — kept under
`statistics.official_combined_row_HYBRID`, never the headline, and (after the fix below) never the
interval either.

### BAR-W7-1 (pre-registered 2026-09-20 16:53, before any score existed): ⛔ **FAIL — sub-case FAIL-A**

**`A1 > max(CV, STOP, ECHO)` on S2-EPDMS-u → 0.3841 vs STOP 0.4828.** refcv4b beats CV and ECHO and
**does not beat a stopped car on navhard.** This is the outcome the pre-registration named as
expected, from E2's warmup result — reported plainly, as committed.

| paired on the identical 450 stage-2 groups / 76 logs | Δ S2-EPDMS-u | diagnostic 95 % interval | separated |
|---|---|---|---|
| **A1 − STOP** | **−0.0987** | [−0.1110, −0.0856] | ✅ yes — A1 worse |
| A1 − CV | +0.0452 | [+0.0279, +0.0628] | ✅ yes — A1 better |
| A1 − ECHO | +0.0429 | [+0.0315, +0.0550] | ✅ yes — A1 better |

⛔ **Interval, estimator and question — stated exactly.** The settled `navsim_log_cluster_bootstrap`
rules on two aggregates only (`navsim_ci.AGGREGATIONS`: the two-stage mapping-key mean, the
single-stage token mean). S2-EPDMS-u is neither, so **the FORMAL interval for refcv4b's quotable
statistic is `{status: UNAVAILABLE}`**, as the mission requires. The column above is a **DIAGNOSTIC**
(`raw/s2_paired_interval_DIAGNOSTIC.json`): the settled estimator's own resampling machine
(`taniteval.ci.paired_episode_cluster_bootstrap`, eid = `log_name`, B 2000, seed 0, RG-14 floor 8)
applied to the S2-EPDMS-u units, whose full-sample points reproduce the suite's `summary.json` to
**0.0** for all four arms. It answers *"would another draw of LOGS say this?"* only. It is blind to
training variance (`H-ESTIM-SEED-1`); inference variance is closed by construction for refcv4b
(deterministic decoder at inference, `MODEL_REGISTRY.md` §4.6). ⛔ `overlapping_holdout_se`: not used.

### RULE ZERO — where A1 loses, and the largest measured lever (`raw/decomposition_A1.json`)

The local EPDMS reconstruction reproduces the devkit `score` column to **1.11e-16** before any
counterfactual is admitted.

* **A1 zeroes 2,493 of 5,462 stage-2 scenes (45.6 %) vs STOP's 1,017.** 1,504 scenes are A1-zero /
  STOP-positive; only 28 the other way. W/T/L vs STOP 2,176 / 1,434 / 1,852.
* **A1 loses every multiplier** — **DAC −0.2127**, DDC −0.1678, NC −0.1032, TLC −0.0088 — and the
  safety-weighted TTC (−0.1203); it **wins** progress and comfort (EP +0.2442, HC +0.2981, EC +0.1402).
  It drives; the drive costs it the multipliers.
* **Zero attribution, uniquely attributable:** **DAC 1,084** ≫ NC 316 > DDC 171 > TLC 27.
* ⭐ **Single-term lever ceiling (one term perfect, the rest frozen): DAC → 0.4875 (+0.1133) — the
  ONLY single term whose repair lifts A1 above STOP (0.4702).** Next: EP +0.0904, DDC +0.0351, NC
  +0.0286. ⚠️ Ceilings, not predictions; terms trade against each other.
* **Speed bands:** A1 loses to STOP in every band but ≥ 12 m/s (+0.0182, n 144); worst in the middle
  (2–5 m/s **−0.1367**, 5–8 m/s **−0.1181**). **A1 leads STOP on only 4 of 76 logs** (vs CV: 58 of 76).

⇒ **The lever with the largest measured effect is drivable-area compliance: refcv4b's plans leave
the drivable area.** The cheapest next experiment was run in the same turn (0 GPU, 0 scoring):

### ⭐ The next experiment, RUN — why the plans leave the road (`raw/dac_diagnosis*.json`)

Measured on the plans the **official scorer executed** (`A1_hooks.json` `agent_poses`, ego frame at t0).
The **attributable** set is where **STOP passes DAC and A1 fails it** — there A1's *motion* caused the
failure (**1,192 scenes**; STOP's own 748 DAC failures are the *scene's*, a parked car starting off the
polygon). Compared against plans where both pass (3,522).

⚠️ **Speed is a confound, so the comparison is made WITHIN t0-speed bands:**

| v0 band | final heading (fail / pass) | lateral per metre of path (fail / pass) |
|---|---|---|
| 0–2 m/s | 2.4° / 0.0° | 0.029 / 0.033 |
| 2–4 m/s | 7.6° / 2.6° | 0.056 / 0.035 |
| **4–6 m/s** | **28.3° / 3.2°** | **0.181 / 0.027** |
| **6–8 m/s** | **29.6° / 2.6°** | **0.123 / 0.023** |
| 8–12 m/s | 2.2° / 1.6° | 0.021 / 0.016 |

**At junction speeds (4–8 m/s) the failing plans TURN ~28–30° in 4 s — 5–7× more lateral per metre
than passing plans at the same speed.** Above 8 m/s the difference vanishes.

**And against the NavSim command** (4–8 m/s, n = 534 failures): turn commands are **enriched** among
failures (LEFT + RIGHT **46 %** vs **25 %** of passes); when refcv4b turns on a LEFT/RIGHT command it
turns the **right way in 179 of 190 (94 %)**; and **99 failing plans turn ≥ 15° under a STRAIGHT
command** (173 at all speeds).

⇒ **The tactical DECISION is right; the turn GEOMETRY leaves the drivable area** — plus a second,
smaller population of **turns nobody commanded**. That matches the one independent fact on record:
refcv4b's **curvature MAE sits ABOVE the straight-line floor** on its own corpus (`MODEL_REGISTRY.md`
§4.6, *"tracks the road WORSE than a plan that never steers"*) — a lateral **SHAPE** defect, the same
one, now seen on a second benchmark. ⚠️ **Evidence class:** the medians and counts are MEASURED; *"the
turn geometry is the cause"* is a **HYPOTHESIS** they support — a description of the failing plans,
not a mechanism, and it cannot separate the model's shape defect from the zero-shot camera transfer
(PhysicalAI rig → nuPlan 3-cam cylindrical stitch). A human reference would, and it exists only on
stage 1 — whose frames are the extraction below.

### What would move it, ranked — the named next arms (none runnable inside this turn without a decision)

| # | lever | why | blocked on |
|---|---|---|---|
| 1 | **extract the 76 navhard logs' stage-1 frames** | gives the FIRST official two-stage EPDMS for a camera arm, and the four families (the human-future comparison that separates shape error from camera transfer) | **PI scheduling** (~1.5–2 h streaming, < 1 GB written; W3's chain on D:) |
| 2 | **A4 (frames-blind)** on navhard | the registered deliberate regression; the only refcv4b-derived arm with a DEFINED official EPDMS today, and the vision attribution | nothing but box time: +~3 h CPU inference, +~90 min scoring, under the same RAM killer — `--reuse-floors/-echo` make it A4-only |
| 3 | **turn-geometry lever** (curvature / turn-timing) on refcv4b | the measured defect: right direction, wrong path at 4–8 m/s | a training change → `TanitAD_ValidateAIDesign` pre-registration, not a benchmark knob |

### Four metric families — refused per family, with reason and n

| family | A1 (refcv4b) | why |
|---|---|---|
| LONGITUDINAL | ⛔ UNAVAILABLE, **n = 450** | the families compare a plan with the logged human future, which exists only on the 450 stage-1 scenes — and A1's stage-1 rows are the CV stand-in (frames not unpacked). Refused, not dropped: **0 criteria violations** |
| LATERAL | ⛔ UNAVAILABLE, n = 450 | same |
| TACTICAL | ⛔ UNAVAILABLE, n = 450 | same |
| STRATEGIC | ⛔ UNAVAILABLE, n = 450 | same; and NavSim scores no route (the command is a ROUTE-LEVEL ORACLE) |

### Defects found and fixed while landing the number (each MEASURED, each with a control)

| # | defect (whose) | what it would have done | fix + control |
|---|---|---|---|
| 1 | ⛔⛔ **A1's `interval` was the HYBRID's** (W1 `benchmark.py`) | `summary.json` carried `A1.interval = {status: OK, [0.0834, 0.139]}`, point **0.1113 = the HYBRID combined row 0.111329** — CV's stage-1 uncertainty published under refcv4b's name, the exact trap the mission warned of, in the interval field | a stand-in arm's interval is now `UNAVAILABLE` with the reason; the hybrid's interval is kept under `statistics.official_combined_row_HYBRID.interval_NOT_this_arm` |
| 2 | **criteria violations on every arm** (W1 `artifacts.py` vs registry 2.10.3) | floors 1 each, A1 **15** — the same CV/STOP artifacts read 0 under 2.10.2 and 1 under 2.10.3. `estimator` still emitted the dead v2.9.0 shape (`cluster_unit = {UNAVAILABLE, reason: "log_name"}`, an admissible interval hidden under a refusal); the stand-in path emitted family-level refusals the checker's per-key lookups cannot see | settled unit as a string; the estimator's own interval block promoted; every criterion the checker would read ABSENT gets its own split-correct refusal. **Control:** rebuilt from the real banked scores → **0 / 0 / 0 / 0**, while the OLD artifacts still read 1 / 1 / 1 / 15 under the same registry (the checker is live) |
| 3 | **warmup text in a navhard refusal** (W1, `NO_GT` constant) | A1 told a navhard reader *"the 204 stage-2 scenes … the 16 stage-1 scenes … 0/192 jpgs"* — every number from a different split | `no_gt_reason(split, n1, n2, n_standin)`, derived from the run; `benchmark.py` carries **0** references to the constant |
| 4 | **my own guard: "byte-equal" compared VALUES** (W7 `floor_reuse.py`) | A1's 450 stand-in rows are all-NaN; `np.array_equal` said three files with the **same sha256** differed → a false refusal of the re-derivation | compares `dtype`, `shape` and **bytes**; regression test proves value-equality calls identical bytes different while byte-equality still catches a **one-ulp** change; the real-artifact probe still refuses all 6 mutations |

⇒ The final run was produced by **re-derivation, not re-scoring** (`--reuse-model-scores`, plan
byte-equal): 32.5 s, and **all four scores CSVs byte-identical** to the scored run. Six superseded run
directories are **tombstoned, not deleted** (`TOMBSTONE.json` in each; the results tree is append-only).

For scale, the floors on those 450 stage-1 scenes (vs the logged human future): along-track MAE
**CV 1.627 m · ECHO 1.218 m · STOP 12.08 m**; tactical κ lat/lon **ECHO 0.374 / 0.403**, CV and STOP 0/0.
⭐ **STOP has by far the worst driving families AND the best EPDMS** — the `D-EPDMS-FAM` point
(EPDMS measures compliance and outcomes, not driving) made concrete in one table.

---

## ⛔ HEADLINE 1 (MEASURED, and it decides what may be claimed)

**refcv4b's OFFICIAL two-stage EPDMS on `navhard_two_stage` is UNDEFINED FOR THIS RUN, and this is
proven by count, not assumed.** navhard's `sensor_blobs` carries the **synthetic stage-2** frames
only; the **stage-1** scorer tokens index the ORIGINAL OpenScene *test* camera archives, which are
⛔ **not UNPACKED anywhere this run could read them**.

⚠️ **SCOPE CORRECTED 2026-09-20, and the correction matters more than the original wording.** An
earlier draft of this line said the archives were *"not on this box"*. That was **too strong**: the
orchestrator measured that they ARE on this box, downloaded and sha256-verified, and I then
re-established it myself on the archive side. **What my census measured is the UNPACKED disk, and
that measurement stands unchanged.** ⇒ the two-stage EPDMS is recoverable by **EXTRACTION**, not by
a new download — see "The unlock" below. ⭐ This is the *"absence found at ONE location is not
absence"* rule catching me in the exact shape it was written for: eight roots is a thorough probe of
*unpacked* files and says nothing about a tarball.

| stage | scorer tokens | camera jpgs EXPECTED | camera jpgs PRESENT (8 roots probed) | fraction |
|---|---|---|---|---|
| **1** | 450 | **5,400** | ⛔ **0** | **0.000000** |
| **2** | 5,462 | **65,544** | ✅ **65,544** | **1.000000** |

*MEASURED 2026-09-20, `raw/stage_frame_census.json`, instrument `code/probe_stage_frames.py`.*

⭐ **The discriminating control is why this is a claim about the CORPUS and not about my probe.**
CLAUDE.md: *a count of 0 from a file that could not be READ is indistinguishable from a genuine
absence.* Stage 2 is the same-breath control — it read **65,544/65,544 in the same pass**, plus a
byte-level JPEG-magic read on a sample, so the probe demonstrably sees files. Had BOTH stages read
zero the instrument returns **INCONCLUSIVE**, never "absent". The 8 roots include both data trees on
the box (`navsim-crun`, `navsim`), both splits' `sensor_blobs`, the `openscene-v1.1` tree and the
D: archive — *absence found at ONE location is not absence*.

⇒ **A camera arm's stage 1 can only be a DECLARED devkit-CV stand-in.** The official
`extended_pdm_score_combined` row mixes a CV stage 1 with a refcv4b stage 2 — it is a **HYBRID**, not
a refcv4b number, and the suite refuses it as a headline (`headline = {status: UNAVAILABLE, …}`,
`statistics.official_combined_row_HYBRID` keeps it labelled). ⛔ **Nothing in this package lets a CV
stand-in be scored as refcv4b.**

⇒ The quotable model statistic is the **stage-2-only S2-EPDMS-u** on the 5,462 synthetic scenes, and
it is ⛔ **not** a two-stage EPDMS. The one refcv4b-derived arm that *can* answer both stages is the
frames-blind **A4** (it reads no pixels) — included precisely so the run leaves behind at least one
DEFINED official EPDMS row, clearly labelled as the registered deliberate regression.

⚠️ **This is the same finding E2 measured on warmup (0/192), re-established on navhard rather than
inherited.** The bridge's stage-1 stand-in is *hard-coded* from the warmup result, so without this
census a hard-coded claim about warmup would have scored silently on a different split.

### ⭐ THE UNLOCK — the frames are in the verified archives, and I measured that myself

**From the orchestrator (2026-09-20), attributed, evidence classes as stated by them:**
navhard's 450 stage-1 tokens span **76 logs, all 76 inside NAVSIM v1 navtest's 136-log universe**
(`D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz`); **3,198/3,375
(94.76 %)** of the exact (cam, basename) pairs are also in navtest's scorer index, and the remaining
**177 are missing from that INDEX, not from the DATA** — positively tested on one log, 3/3 files
found inside shard 6. Those are exhaustive counts except the last, which they flagged honestly as
*a mechanism plus ONE positive log*, not proof.

**MEASURED BY ME, independently, on the archive side** (`raw/stage1_in_archives_shard6.json`,
instrument `code/probe_stage1_in_archives.py`; a tar **LISTING**, no extraction, `rc` read from a
FILE never through a pipe):

| probe | result |
|---|---|
| 32 shards present at `D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera/` | ✅ **127,882,665,618 B**, byte-exact to `receipt_navtest_camera.json` (`all_verified: true`) |
| navhard logs inside shard 6 | 2 of 76 |
| their **stage-1** files | ✅ **120 / 120 — both logs COMPLETE, per-log all-or-nothing** |
| their **stage-2** files (discriminating control) | ⛔ **0 / 1,185**, as it must be |

⭐ **I tested WHOLE LOGS, which is the claim, rather than sampling files.** *"The archives store whole
logs"* is falsified by a single partial log and confirmed by none being partial. And the stage-2
control reading **exactly 0** is what proves the probe discriminates: the two stages index genuinely
different corpora, so a probe that "found everything" would have been finding the wrong thing.

⚠️ **Still NOT proven, and I am not going to write it as if it were:** that *every* one of the 5,400
files is present across *all* 32 shards. Two shards' worth of logs is two instances of a mechanism.
⇒ **STRONGLY SUPPORTED, not proven.** The closure is cheap and must be a **REQUIREMENT, not an
assumption**: the extraction asserts **5,400 / 5,400 by name** before any stage-1 arm is scored. A
silent partial extraction is the poisoned-bank class — an arm scored on fewer frames, reported as
complete.

⚠️ **The archive layout is `openscene-v1.1/sensor_blobs/test/<log>/CAM_F0/<file>.jpg`** (MEASURED in
the shard-6 listing) while the loader probes
`…/navhard_two_stage/sensor_blobs/<log>/CAM_L0/…`. A path mapping is needed, and ⛔ **D: is exFAT, so
symlinks are unavailable** (`WinError 1`) — it must be a copy or a loader-root override.

**Cost of the follow-on (ESTIMATED from the shard-6 pass, stated as ESTIMATED):** `.tgz` is not
seekable, so a targeted extraction still **streams all 32 shards = 119.1 GB**; shard 6 (6.21 GB)
listed in ~5 min under this box's current load ⇒ **~1.5–2 h wall**, but it **writes < 1 GB** (5,400
jpgs at ~150 KB). ⭐ **Only the 76 navhard logs are needed, not all 136** — the expensive half is the
decompression pass, not the disk. ⛔ **Not run now**: W3's chain is on D: (shard 14/32) and this
box's A1 inference is live; this is a scheduling call for the PI, not a thing to start beside two
other jobs.

---

## ⛔ HEADLINE 2 (MEASURED) — a 10.83 GB frame-cache leak that would have aborted the run at hour ~4

`taniteval/taniteval/bench/navsim/model_arms.py` cached decoded frames in an **unbounded dict**
keyed by `scene_token`.

| quantity | navhard | warmup (where it was written) |
|---|---|---|
| stage-2 tokens | 5,462 | 204 |
| **DISTINCT** `scene_token`s | **5,462** (max **1** token per scene) | 204 |
| consecutive repeats in the loop's own order | **0** | 0 |
| cache bytes at end | **10.83 GB** | 0.41 GB |
| host RAM AVAILABLE on this box | **7.9 GB** | — |

*MEASURED 2026-09-20 (`raw/frame_cache_arithmetic.json`); per entry = `4·256·640·3` frame bytes +
`256·640` mask = 2.03 MB.*

⇒ The cache had a **0 % hit rate by construction** and would have climbed past available RAM. The
failure would NOT have looked like a leak: the wrapper's RAM guard aborts an arm below
`--ram-floor-mb`, surfacing hours later as `RAM_GUARD_ABORT` / *retryable*. ⭐ Same family as the
`e_trunk_pooling.py` dense-window trap in `CLAUDE.md` — **price the host tensor against FREE RAM, not
total** — and it was invisible on the smaller split, which is how it survived E2.

**Fix (staged, W1's file — ESCALATED for integration):** bounded LRU `FRAME_CACHE_MAX = 64`
(130 MB), plus `rss_gb` + hit/miss/eviction counters in the progress log and the per-arm manifest, so
the bound is a *number in the artifact* rather than a claim in a comment.

**Control that the bound is INERT:** the smoke runs the identical arms twice, with the bound at 64
and at 2 (below the working set, forcing evictions), and requires **bit-identical poses**. See
`raw/smoke_cache_control.json`.

---

## Preconditions established before any scoring (all MEASURED this turn)

| # | check | result | artifact |
|---|---|---|---|
| P1 | per-stage camera census of the **UNPACKED** disk + same-breath control | stage 1 **0/5,400**, stage 2 **65,544/65,544** | `raw/stage_frame_census.json` |
| P1b | second, mechanistically different probe (match by BASENAME over a full tree walk) | stage 1 **0/3,375**, stage 2 **40,965/40,965**, the two sets **disjoint** | `raw/stage1_second_probe_basename.json` |
| P1c | ⭐ the same frames **inside the verified archives** (tar listing, no extraction) | shard 6: **120/120** stage-1 files for both navhard logs, **complete per log**; stage-2 control **0/1,185** | `raw/stage1_in_archives_shard6.json` |
| P2 | frame builder is faithful on a NEW split | **40/40 bit-exact** vs the DataFlyWheel warmup corpus | `raw/frame_builder_reproduction.json` |
| P3 | navhard stage-2 frame bank | 5,462 scenes, content-asserted (u8 `[4,256,640,3]`, sha == provenance, mean > 1) | `raw/BUILD.json` |
| P4 | checkpoint identity | md5 `99b573e8277d94a5e3bfbf630cb4d751` = `MODEL_REGISTRY.md` §4.6 | run `bench_run.json.ckpt` |
| P5 | model path REACHABLE from the real caller | smoke PASS; A1 vs A4-blind differ on **6/6** scenes (max 20.63 m) | `raw/smoke_*.json` |
| P6 | frame-cache arithmetic | 10.83 GB vs 7.9 GB free ⇒ bounded | `raw/frame_cache_arithmetic.json` |

⭐ **P5's control is the load-bearing one.** A blind arm that matched A1 would prove the **pixels
never reached the model** — the `CLAUDE.md` "built, tested, and unreachable from its caller" class,
which has now surfaced five times in this programme, three of them *after* an arm was trained. The
smoke drives the REAL caller (`taniteval.bench.navsim.model_arms.run_model_arms`), not a
re-implementation, so what passes here is what the run executes.

---

## Tier + loop stamp (binding, on every number in this package)

**T1-family** · stage 1 loop **OPEN** (PI ruling 2026-09-02) · stage 2 **UNRULED** (a PI ruling is
pending — ⛔ not guessed here) · background traffic **IDM-reactive in both stages, ego non-reactive**
· ⛔ **never closed loop.** Protocol `EPDMS_v2_navhard_two_stage`, devkit
`autonomousvision/navsim@0a380a9`.

---

## The run (LIVE)

```
python -m taniteval.bench navsim_v2 \
  --ckpt D:/Projects/TanitAD-artifacts/refcv4b_final/ckpt_40284_FINAL.pt \
  --split navhard_two_stage --arms A1 --device auto \
  --registry-key refcv4b-b1-v72-40k --ckpt-md5 99b573e8277d94a5e3bfbf630cb4d751 \
  --frame-bank C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920 --cpu-threads 16
```

**Run dir** `taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T144140Z-navsim_v2-refcv4b_b1_v72_40k-b11027`
**Arms** `A1` requested; **CV + STOP added by rule** (mandatory NavSim floors, not removable) and
**ECHO added by rule** (a checkpoint is scored) ⇒ **4 arms on IDENTICAL tokens**, which is what makes
the paired comparison legitimate. Launcher `code/launch_run.sh`, log `raw/bench_launch.log`.

### ⛔ The first launch died in 4.3 s — and that was the correct outcome

```
StackShadowError: `tanitad` would be imported from D:\Projects\TanitAD\stack\tanitad
but the pinned stack root is G:\Meine Ablage\...\TanitAD\stack   (intent: installed:find_spec)
```

The venv's **editable `tanitad` install still points at the dead G: mount** — the trap `CLAUDE.md`
records verbatim. ⭐ **A guard that fails loud is worth more than a run that finishes**: had it not
fired, the arm would have imported a pre-move stack and published a plausible wrong number.
Fixed by naming the tree (`TANITEVAL_STACK_OVERRIDE=D:/Projects/TanitAD/stack`), ⛔ **never** by
downgrading the guard to `warn`. The 4.3 s FAILED run dir (`…-b74f95`) is left in place — the
results tree is append-only and a failure is evidence.

### Device — CPU, and why that is the compliant choice (MEASURED)

`[gpu-gap] NO_GAP_CPU {'reasons': ['GPU memory used 1175 MiB >= limit 1024 MiB']}`

⚠️ **The gap gate is, on this box in its ordinary idle state, UNSATISFIABLE — and that is a finding,
not a nuisance.** With **no training process alive at all**, Windows/WDDM desktop clients (explorer,
Chrome, VS Code, the NVIDIA overlay, a running screensaver) hold **1,175–1,512 MiB**, above the gate's
**1,024 MiB** limit. Sampled at 15 s over 3 min with the trainer gone: **0 gaps in 12 samples**
(`raw/gpu_watch.json`). The brief's own earlier reading of *1,018 MiB* is the same baseline sitting a
few MiB under the line — this is a knife's edge, not a margin.

⇒ Inference runs on **CPU**, which **cannot collide with training at all** and is therefore *stricter*
than the ruling, not a relaxation of it. ⛔ The gate's threshold was **not** moved, and no GPU work was
done outside the launcher. **Work item for W1 / the PI:** the limit needs an idle-baseline offset, or
a desktop box can never satisfy it — see "Escalations".

**Inference pace (MEASURED, live):** 1.42–1.61 s/scene at `--cpu-threads 16` ⇒ ~2.4 h for A1's 5,462
scenes. `--cpu-threads` is new: `run_model_arms` always took a `threads` argument and **nothing ever
reached it**, so every CPU run used the default 6 on a 24-core box — the `CLAUDE.md` "built, tested,
and unreachable from its caller" class again. Measured on 12 navhard scenes: **6 threads 4.34 s/scene
vs 16 threads 2.49 s/scene (1.74×)**, i.e. 6.6 h vs 3.8 h for one arm.

**The cache fix, verified in the LIVE run, not in a comment:**
`[A1] 200/5462 stage-2 scenes, 1.61 s/scene, device=cpu, rss=2.08 GB, cache=64/64 (hit 0 / miss 200)`
— RSS **flat at 2.08–2.09 GB** across 200 scenes, and the hit count is **0**, exactly the 0 % rate the
arithmetic predicted. Unbounded, this line would climb 2.03 MB per scene toward 10.83 GB.

---

## The bar, PRE-REGISTERED before any score existed

`PREREG.md` — **BAR-W7-1: `A1 > max(CV, STOP, ECHO)`** on the stage-2 statistic **S2-EPDMS-u**, over
the identical 5,462 stage-2 tokens, with the paired `navsim_log_cluster_bootstrap` interval. Written
at **16:53** while A1 was at scene ~300/5,462 of *inference*; the run's `scores/` directory did not
yet exist, which is asserted in the same command that staged the file. ⭐ **FAIL-A (`A1 > CV` but
`A1 <= STOP`) is the outcome the pre-registration EXPECTS**, because that is what E2 measured on
warmup — and saying so in advance is the whole point of writing it down.

---

## Context the run inherits: WHY a stopped car is hard to beat here (MEASURED on the banked floors)

`raw/decomposition_floors_STOP_vs_CV.json` — my own instrument, validated against the suite's
independently computed `statistics.*_scene_mean` for **both** arms and **both** stages (agreement
< 1e-6; I derive from the raw CSV, `summarize.py` derives its own way).

**navhard stage 2, STOP vs CV, 5,462 identical tokens: STOP +0.1409** (scene mean 0.4702 vs 0.3294),
**W/T/L 2,381 / 1,392 / 1,689**, STOP ahead on **72 of 76 logs**. Zero-score scenes: **CV 2,968 vs
STOP 1,017**.

| term | STOP | CV | Δ (STOP−CV) | role in EPDMS |
|---|---|---|---|---|
| NC | 0.9564 | 0.8338 | **+0.1227** | **multiplier** |
| DAC | 0.8631 | 0.5692 | **+0.2938** | **multiplier** |
| DDC | 0.9508 | 0.7345 | **+0.2162** | **multiplier** |
| TLC | 0.9954 | 0.9861 | +0.0093 | **multiplier** |
| EP | 0.3843 | 0.7006 | −0.3163 | weighted, w = 5.0 |
| TTC | 0.9438 | 0.8155 | **+0.1283** | weighted, w = 5.0 |
| LK | 0.4960 | 0.4345 | +0.0615 | weighted, w = 2.0 |
| HC | 0.6430 | 0.9634 | −0.3204 | weighted, w = 2.0 |
| EC | 0.2483 | 0.5910 | −0.3427 | weighted, w = 2.0 |

⭐ **The structure is the thing any model arm has to beat: STOP wins all four multipliers and every
safety-shaped weighted term, and loses the progress/comfort terms.** EPDMS is
`prod(NC, DAC, DDC, TLC) × Σ(w·m)/Σw`, so a single zeroed multiplier sinks a scene outright while
progress and comfort only move an average. Every speed band agrees, and the penalty for moving peaks
in the MIDDLE of the range (2–5 m/s **+0.2087**, 5–8 m/s **+0.1640**), not at speed.

⚠️ **TTC IS NOT A MULTIPLIER — I had it wrong for one draft, and the DATA caught it.** My first pass
listed five multipliers from memory. The zero-attribution then reported TTC zeroing 242 STOP scenes
while being the UNIQUE zero in **0** of them — *impossible for a real multiplier*. Reading the
promoted `summarize.py::epdms_formula` settled it: four multipliers, TTC weighted at 5.0. Same family
as the units trap in `CLAUDE.md` — a correct formula with a term in the wrong role reads exactly like
an answer. The weights are now read from the promoted source, never recalled.

**Zero-attribution for STOP (co-occurrence upper bound → uniquely attributable):** DAC 748 → **682**,
NC 221 → **176**, DDC 97 → **69**, TLC 25 → **19** (1,017 zeroed scenes of 5,462). ⚠️ Read the second
column: a stopped car's zeros are overwhelmingly **drivable-area compliance** — what it means to be
parked where the scene begins — not a driving error.

**Lever ranking for STOP, stage 2** (`lever_ranking_counterfactual`): if one term alone were perfect
with everything else frozen, the scene mean 0.4702 would become EP **0.6434 (+0.1731)** ≫ DAC 0.5408
(+0.0705) > EC 0.5389 (+0.0687) > LK (+0.0443) > HC (+0.0299) > DDC (+0.0207) > NC (+0.0132) > TTC
(+0.0032) > TLC (+0.0028). ⭐ **The counterfactual is only admissible because the local formula is
asserted against the devkit's own `score` column first: `max_abs_diff = 1.11e-16` over all 5,462
stage-2 rows.** A ceiling computed from a formula that cannot reproduce the real number is fiction.
⚠️ These are ceilings under a single-term repair, they TRADE against each other (braking buys NC and
loses EP), and they do not add up.

---

## Timeline, and how this run finishes

The suite scores arms in a fixed order — **CV → STOP → ECHO → A1** — and runs *all* model inference
first. ⚠️ **The mission arm's number therefore arrives LAST.** Measured pace (this box, CPU,
`--cpu-threads 16`): inference ~1.85 s/scene; scoring ~92 min/arm (the banked floors run measured CV
5,338 s and STOP 5,510 s on the same 5,912 tokens).

| stage | ETA (Europe/Berlin) |
|---|---|
| A1 inference, 5,462 stage-2 scenes | ~19:35 |
| CV scored | ~21:10 |
| STOP scored | ~22:40 |
| ECHO scored | ~00:15 |
| **A1 scored + `summary.json` + W5 report** | **~01:45 (2026-09-21)** |

⛔ **The floors are re-scored rather than reused, and that is deliberate, not an oversight.** CV and
STOP are MANDATORY and **not removable** (`benchmark.py::resolve_arms`), and the paired comparison
must come from arms scored *in the same run on the same tokens*. Reusing the banked CSVs would save
~4.6 h and produce a cross-run pairing that nothing in the contract certifies. The banked run is
instead used as a **reference check** (`controls.reference_check`): our CV and STOP must reproduce it.

**To finish it (one command each), once `summary.json` exists:**

```bash
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
RUN=taniteval/results/bench/navsim_v2/navhard_two_stage/20260920T144140Z-navsim_v2-refcv4b_b1_v72_40k-b11027
cd FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-navhard-refcv4b
$PY code/read_summary.py --run ../../../../$RUN --arm A1 --floors CV,STOP,ECHO --out raw/bar_verdict.json
$PY code/decompose.py    --run ../../../../$RUN --arm A1 --vs STOP,CV,ECHO \
     --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json \
     --out raw/decomposition_A1.json
$PY code/manifest.py --run ../../../../$RUN --bank C:/Users/Admin/tanitad-caches/navsim-frames-navhard-20260920 \
     --pkg . --out MANIFEST.json
```

⭐ **A DETACHED FINISHER IS ALREADY ARMED AND RUNNING** (`code/finish_when_done.sh`, log
`raw/finisher.log`, started 18:11:45). It waits for `summary.json` to EXIST — asserting on the
artifact, never on an exit code, and never chaining a step behind a pipe — then runs all three
instruments, writes `FINISH_REPORT.md`, and **stages its outputs by explicit path** (⛔ never a
directory, on a shared branch). It distinguishes *"still running"* from *"ended without a
summary"* (a `bench_run.json` with no `summary.json` is a real outcome and exits 4, rather than
waiting forever). ⛔ It deliberately does **not** rewrite `RESULT.md`: the verdict prose, the
evidence classes and the reading of a failed bar are a judgement, not a template fill. ⇒ even with
nobody watching, the run's numbers land in the repo instead of on one disk — which is the exact
failure the operating standard exists to make structurally impossible.

`read_summary.py` prints the **pre-registered bar verdict as a literal comparison**, not as prose, so
it cannot drift from the arithmetic; `decompose.py` produces the per-sub-metric / per-stage /
per-speed-band / per-log decomposition **and** the single-term lever ceiling — which is the
RULE-ZERO continuation the pre-registration commits to if the bar fails. Both instruments are
already validated against the banked floors run (`read_summary.py` reproduces CV 0.3389 / STOP
0.4828; `decompose.py` matches the suite's stage means < 1e-6 and the devkit `score` column to
1.11e-16).

---

## ⛔ ESCALATIONS — decisions and merges that will otherwise sit unread

*(The operating standard: escalate integration in the report's headline; a "please merge" written
into a README went unread for 10 days.)*

| # | what | who decides | why it cannot wait |
|---|---|---|---|
| **E1** | **Three edits to W1's suite are staged, not committed** — `navsim/model_arms.py` (bounded frame cache + navhard in `DEFAULT_BANKS` + RSS/cache counters), `navsim/benchmark.py` (pass `threads` through), `bench/cli.py` (`--cpu-threads`). 277 bench tests pass. | **W1 / Master Mind** | Without the cache bound, **every future navhard model arm climbs to 10.83 GB and is aborted by the RAM guard hours in.** The `threads` fix is the "unreachable from its caller" class: the knob existed and nothing reached it. |
| **E2** | **The GPU-gap gate is unsatisfiable on this box** (`D-NAVSIM-GPUGATE-1`): idle Windows desktop GPU memory is **1,175–1,512 MiB** against a **1,024 MiB** limit, 0 gaps in 12 samples with no trainer alive. | **PI + W1** | `--device cuda` silently costs its full 8 h wait and lands on CPU anyway. The fix is an **idle-baseline offset** (measure the floor at launch, compare FOREIGN memory to it) — ⛔ not a bigger constant, which would stop detecting a real neighbour. |
| **E3** | ⭐ **NOT a download — an EXTRACTION.** The OpenScene test camera archives are already on this box, sha256-verified 32/32 (**127,882,665,618 B**, byte-exact to the receipt), and navhard's stage-1 frames are inside them (**120/120 files complete for both navhard logs in shard 6**, stage-2 control 0/1,185). What is needed is a **targeted extraction of the 76 navhard logs**: ~**1.5–2 h** of streaming (all 32 shards must be read — `.tgz` is not seekable), writing **< 1 GB**, plus a path mapping (archive `openscene-v1.1/sensor_blobs/test/<log>/CAM_X/` vs the loader's `…/navhard_two_stage/sensor_blobs/<log>/CAM_X/`; ⛔ **exFAT has no symlinks**, so copy or override the loader root). ⛔ It **must assert 5,400/5,400 by name** — a silent partial extraction is the poisoned-bank class. | **PI** (scheduling: W3's chain is on D:, this box's A1 inference is live) | Until then **no camera arm can have an official navhard EPDMS**, and the four metric families are structurally UNAVAILABLE for one (they need the logged human future, which exists only on stage 1). ⭐ Afterwards, `code/build_frames.py --stage 1` + a re-run yields the programme's first genuine two-stage EPDMS for a camera arm. |
| **E4** | ⚠️ **`products/P7-TanitEval/benchmarks/published_results.json` is currently STAGED by a sibling (W4/W5) and my run's report hook REWRITES it on disk when it finishes.** The sibling's staged blob is not lost (the index keeps it), but the worktree and index will then disagree. | **Master Mind** | This is the two-correct-rules collision `CLAUDE.md` describes: a sibling stages as it goes, and a later run rewrites the same file. Re-stage it deliberately after the run, ⛔ never with a pathspec-free commit that sweeps whichever version happens to be there. |
| **E5** | **A4 (frames-blind) was NOT run** and is the named next arm. | W7 / EvalFlyWheel | It is the **only refcv4b-derived arm that can answer stage 1** (it needs no pixels), so it is the only route to a DEFINED official two-stage EPDMS from this checkpoint **and** the only one for which the four families can be computed on navhard at all. Cost: one extra arm in the same run (~92 min scoring + ~2.5 h CPU inference). |

| **E7** | ⭐ **`…859e25` should become navhard's REPORT RUN OF RECORD.** `taniteval/results/bench/.gitignore` keeps ONE rendered report per split in git (reports re-render byte-identically, so the rest regenerate); navhard's is the floors-only run `…06e257`, whose failure **gallery was UNAVAILABLE**. `…859e25` is the programme's first model row on navhard, with the gallery **RENDERED** (verify PASS, 953 numbers checked). Its 22 report files are therefore on disk and **not staged — correctly, per that policy**. | **W1** (it is W1's policy file) | swap the negation `!…06e257/report/**` → `!…859e25/report/**`; until then the report regenerates with `python -m taniteval.benchreport <run_dir>`. ⛔ I did not force-add or edit the policy on my own authority |
| **E8** | **Four suite defects fixed in W1's files are staged, not committed:** the hybrid interval published as the model's (`benchmark.py`), the dead v2.9.0 estimator shape + invisible family refusals (`artifacts.py`), the warmup refusal text (`artifacts.py`, `benchmark.py`), and three new reuse paths (`--reuse-floors/-echo/-model-scores`, `floor_reuse.py`). 169 bench tests green. | **W1 / Master Mind** | every future NavSim model arm with a stand-in stage 1 would otherwise re-publish the hybrid interval under its own name and read 15 criteria violations |

⚠️ One failed 4.3 s run directory, `…/20260920T144053Z-navsim_v2-refcv4b_b1_v72_40k-b74f95`
(`BENCH_STATUS=FAILED`, the stack-shadow refusal), is left in place: the results tree is append-only
and a refusal is evidence. It carries no scores and no summary, so no consumer can read it as a row.

⚠️ **E6 — the RUN DIRECTORY is not yet staged, because it does not yet exist in full.** The suite's
own results tree is tracked in git (the banked floors run is 235 tracked files, with >20 MB artifacts
handled by the suite's `.offrepo.json` sidecars). When the run finishes, `taniteval/results/bench/
navsim_v2/navhard_two_stage/20260920T144140Z-navsim_v2-refcv4b_b1_v72_40k-b11027/` must be staged by
explicit FILE paths in batches — ⛔ never `git add <directory>` on this shared branch (auto-memory
"stage by path on shared branch"; and a bulk directory add across large trees has died with
`0xC0000006 STATUS_IN_PAGE_ERROR`).

---

## ✅ A1 INFERENCE COMPLETE — 19:50:24 (MEASURED, `<run>/raw/model/model_run.json`)

**5,462 model rows + 450 declared CV stand-ins in 11,232.6 s (3.12 h), device `cpu` throughout.**

| fact | value | why it is quotable |
|---|---|---|
| checkpoint | md5 **`99b573e8277d94a5e3bfbf630cb4d751`**, `step` 40284, 117 anchors | the run REFUSES a mismatch (`--ckpt-md5`); md5 = `MODEL_REGISTRY.md` §4.6 |
| state-dict load | `missing_keys` **[]**, `unexpected_keys` **[]**, `tolerated_inert_buffers` **[]** | a strict load — no head was silently dropped |
| frame tag | model `256x640f305.5775cyl` **== bank** `(256, 640, 305.577491, cylindrical)`, `match: true` | ⛔ the bridge refuses a geometry that is not the model's |
| device | `cpu`, `NO_GAP_CPU` at launch, **no back-off event** | no GPU was touched at any point |
| **frame cache** | **hits 0 · misses 5,462 · evictions 5,398 · max 64** | ⭐ **the 0 % hit rate was PREDICTED from the token→scene mapping before the run and came back EXACT.** RSS ended **1.284 GB**; unbounded this is the 10.83 GB that would have aborted the arm |

**Stand-in count read from the SEAM ARTIFACT, not from the producer's report** (`raw/model/A1.npz`,
re-opened independently): **5,912 rows = 450 `cv_standin` + 5,462 `refcv4b`**; every model row's
poses **finite (0 non-finite)**; every stand-in row **all-NaN, exactly as declared**. ⇒ A1's official
two-stage headline will be `UNAVAILABLE` with **n = 450**, and the suite will emit the HYBRID
combined row separately under `statistics.official_combined_row_HYBRID` — labelled, never as the
headline.

---

## ⛔ THE RUN FAILED IN SCORING — all four arms `RAM_GUARD_ABORT` (21:32:04)

**The guard's own words, from `raw/STOP/STOP.score.log`:**

```
E1_RAM_GUARD_ABORT: system available memory 1035 MB (floor 3000 MB for 1 consecutive samples;
                    hard floor 2000 MB); own RSS 768 MB; killed 0 child process(es)
```

⭐ **Read the second number: the scorer's own RSS was 768 MB. It was not the consumer.** The box's
available memory fell to **1,035 MB** because sibling agents' jobs took it (at diagnosis time: 84.1 %
used, 5.05 GB available, 8.8 GB swap, with `diag_aimed.py`, `build_scorer_targets.py` and four more
uv-python processes started 21:20–21:33). The suite classifies this exactly right —
`retryable: true`, *"the run did NOT fail on its own terms — it was never allowed to finish"*.

| arm | rc | status | wall | how far it got |
|---|---|---|---|---|
| CV | 3 | `RAM_GUARD_ABORT` | 910 s | no CSV |
| STOP | 3 | `RAM_GUARD_ABORT` | 5,166 s | **4,915 / 5,912 agent calls (83 %)** |
| ECHO | 3 | `RAM_GUARD_ABORT` | 1 s | 0 — RAM was already under the floor at launch |
| A1 | 3 | `RAM_GUARD_ABORT` | 0 s | 0 — same |

⛔ **Nothing from this run is quotable. There is no EPDMS number, and I am not reporting one.**
⚠️ **A `summary.json` EXISTS for it and is NOT a result** — the suite writes one for a failed run
too, with every arm `status: FAILED` and every headline `UNAVAILABLE`.

### What this is NOT

⛔ **It is not the frame-cache defect.** That fix held all the way through: `hits 0 · misses 5,462 ·
evictions 5,398`, RSS **1.284 GB** at the end of a 3.12 h arm. Had the cache still been unbounded it
would have been holding **10.83 GB** of the memory that ran out — i.e. the defect I fixed this
morning is precisely the one that just killed the run *from the outside*, and the fix is why A1's
inference survived to be reusable at all.

### ⭐ THE NEXT LEVER, EXECUTED IN THIS TURN — the 3.12 h of inference is NOT redone

`CLAUDE.md`: *"an analysis-time import that fails AFTER the rollout destroys the run's output while
the compute is already paid for … check for the banked dump BEFORE re-running anything."* The A1
seam is complete and intact. So:

**`--reuse-seams <dir>` is implemented, guarded and mutation-tested** (`model_arms.reuse_seam`,
`cli.py`, `benchmark.py`; `taniteval/tests/test_bench_navsim_reuse_seams.py`, **10 tests**). It
adopts a banked seam only after an IDENTITY check — ckpt md5, arm spec, frame tag, seam row count,
and the seam's own source counts **re-read from the artifact, never trusted from the manifest** — and
**RAISES on any mismatch**; there is no silent fall-back to recomputing, because a silent fall-back
turns a refusal into a three-hour surprise.

**Mutation arms, all required to go RED:** a seam from another checkpoint · from another arm
(blind vs camera) · at another frame geometry · over a different token set · **truncated by one row**
· with a manifest whose counts lie about the file. Plus a **positive control** (an identical seam must
be adopted and copied byte-for-byte) — without it a guard that refuses everything would pass.

**Verified on the REAL banked artifact, not only on fixtures:** 5,462 model rows + 450 stand-ins
adopted, copy byte-identical (sha256 `dd7814b7e74e8e85…`), wrong-md5 control **REFUSED**.

⚠️ **And the real artifact caught a bug the fixtures could not.** `frame_tag_check` returns
`bank_frame` as a **tuple** in-process; JSON round-trips it to a **list**, so a raw `!=` refused an
*identical* geometry. My unit fixture used a list on both sides and was blind to it. ⇒ normalised,
and pinned by a regression test that deliberately passes a tuple against a JSON list. ⭐ **A
comparison whose two operands come from ONE serialisation path is not testing the comparison
production performs** — the units trap with types swapped in.

### The retry is armed and RAM-gated (`code/retry_scoring.sh`, log `raw/retry.log`)

⛔ Retrying blindly is what cost 86 min last time. Each attempt therefore waits for **5 consecutive
60 s samples with ≥ 6.0 GB available** — sustained headroom, not one lucky reading — then launches
with `--reuse-seams` (inference: **0 s**). On success it writes `raw/SUCCESSFUL_RUN_DIR.txt` and
**tombstones** the aborted run (the results tree is append-only; a tombstone marks it superseded and
records that its model seam was kept and adopted). Up to 6 attempts, 20 min backoff. Armed 21:44:44;
first sample read **4.32 GB**, so it is correctly waiting.

⛔ **The `--ram-floor-mb 3000` guard was NOT lowered.** It exists to prevent a kernel OOM (the
"silent trainer death = kernel OOM" class), and `CLAUDE.md` is explicit: if a gate fires, fix the
run, not the gate.

### ⚠️ MY OWN FINISHER HAD THE DEFECT CLASS I SPENT THE DAY AUDITING FOR

It waited for `summary.json` to **exist** and treated that as success. The suite wrote one for the
FAILED run, so the finisher produced a `FINISH_REPORT.md` that **looked like a result** — an arm
table, correct headers, `UNAVAILABLE` in every cell. That is a success criterion disconnected from
the thing being checked, committed by the very guard I wrote to prevent stranded work. ⇒ **fixed to
gate on CONTENT** (at least one arm with `status: OK`, counted and printed to the log so the gate is
auditable); otherwise it writes a `FAILURE_REPORT.md` naming the per-arm records and what survived,
and exits 5. The misleading file has been deleted and unstaged.
⚠️ Second, smaller bug in the same script: one multi-path `git add` hit
`fatal: pathspec '…/bar_verdict.json' did not match any files` and **aborted the whole call**, so
files that *did* exist went unstaged and the blob check then read INCONCLUSIVE for all of them —
which looks like a mount problem rather than one missing sibling path. ⇒ one `git add` per file,
each guarded by `[ -f ]`.

---

## ⛔⛔ ATTEMPT 1 (d3c2b2) ALSO DIED — AND MY RETRY SCRIPT REPORTED IT AS A SUCCESS

**Caught by the orchestrator, verified by me from the artifacts (2026-09-21).**

Run `…/20260920T201038Z-navsim_v2-refcv4b_b1_v72_40k-d3c2b2`: **CV, STOP, ECHO, A1 all
`status=FAILED`**, headline `UNAVAILABLE`, n = 0. Cause = the RAM guard again, at the HARD floor:
`E1_RAM_GUARD_ABORT` at **1,175 / 1,750 / 1,077 / 948 MB** available (hard floor 2,000); CV's own RSS
774 MB; CV killed at stage-two scenario **4,294 / 5,462 (79 %)** — the same shape as attempt 0 (STOP at
83 %). Other sessions' jobs, not ours. ✅ `--reuse-seams` DID work: `REUSED banked seam — 5462 model
rows, 450 CV stand-ins (identity verified)` at 22:10:48.

⛔ **The defect is mine.** `retry_scoring.sh:73` was `[ -f "$RD/summary.json" ]` — **existence**. The
suite writes `summary.json` for a failed run too — *which I had established myself the evening
before* — so the script logged `BENCH_STATUS=FAILED` and then `SUCCESS` **on the very next line**,
wrote `raw/SUCCESSFUL_RUN_DIR.txt` pointing at the all-FAILED run, and **exited with 5 of 6 attempts
unused**. It is the finisher's self-trap, fixed in `finish_when_done.sh` and **not propagated to its
sibling**. ⭐ The class is not "a bad test", it is **a fix applied to one copy of a check that lives in
two** — the second copy is where it survives.

Two more defects surfaced in the same failure chain, both mine:
* the tombstone step ran `python -m taniteval.bench` from the repo root → `No module named
  taniteval.bench` (the namespace-shadow trap in `CLAUDE.md`'s memory index);
* the content-gated finisher correctly judged `arms OK: 0` and wrote a `FAILURE_REPORT.md` — but with
  a **relative path before its `cd`**, so it landed in the **repo root** and its `git add` failed.

### What was done, in the orchestrator's priority order

**A — success is judged on CONTENT, by ONE shared judge** (`code/judge_run.py`): SUCCESS iff every
required arm (CV, STOP, ECHO, A1) reads `status: OK`; counts printed; a literal `JUDGE=` line is
grepped rather than an exit code read through a shell. Both scripts now call it — no copy of the logic
exists to drift. **Mutation-tested (`raw/judge_mutation_test.json`, 4/4):** the REAL d3c2b2 summary →
**FAILED** (`n_ok 0/4`) · a fabricated all-OK copy → **SUCCESS** · 3 OK + A1 FAILED → **FAILED** · no
summary → **FAILED**. The false marker is **renamed, not deleted**, to `raw/FALSE_SUCCESS_d3c2b2.txt`
(anything reading the old name would inherit a total failure as the good run); the stray report is
moved to `raw/FAILURE_REPORT_d3c2b2.md`; the tombstone step runs from `taniteval/`; the finisher `cd`s
first.

**B — ⭐ the floors are no longer re-scored** (`--reuse-floors`, `navsim/floor_reuse.py`). Both
attempts died at ~80 % of their FIRST scored arm; CV + STOP had already been scored completely on the
identical tokens in `…06e257` (CV == official 0.11481608441648 exactly). Adoption REFUSES unless: the
source arm completed (`PASS`, `rc 0`, 5,912 successful, 0 failed, 5,912 valid rows) · same split and
protocol · same devkit sha · **same patch set by name AND raw-bytes blob** · same metric cache path and
identity · **identical token set BY VALUE with stage labels** (stage read from which columns the devkit
populated) · same agent-input export sha256 · and for STOP, **its seam byte-equal to the source's**.
**Probed on the REAL artifacts (`raw/floor_reuse_probe.json`): both floors adopt, scores CSV
byte-identical; 6/6 mutations REFUSED** — wrong devkit sha · one token dropped · one stage flipped ·
the IDM patch removed · another export · one STOP pose moved by **1 mm**. ⛔ The artifact states that
the mandatory-floors rule is **satisfied by floors scored on the same tokens** and names the source run
(`controls.floors_provenance`, a per-arm `FLOOR_REUSED_FROM_SOURCE_RUN` caveat, and
`raw/<arm>/<arm>.reused.json`) — never a silent skip. **The exposure window is halved: only ECHO + A1
are scored.**

**C — not needed:** B landed clean. It remains the named fallback (per-chunk CSV + re-aggregation, so a
kill costs ≤ 1 chunk) if this attempt dies too.

**D — launched while the box was quiet.** Retry v2 (`code/retry_scoring.sh`) waited for **5
consecutive 60 s samples ≥ 6.0 GB** (sustained, 11:13:29 → 11:17:30), launched at **12.45 GB**. ⛔ The
guard floor stays 3,000 / 2,000 MB.

**Verified from the live log, not assumed:**
```
11:17:42 [navsim] model arm A1 (A1_ego_cmd): REUSED banked seam — 5462 model rows, 450 CV stand-ins (identity verified, nothing recomputed)
11:17:42 [navsim] CV: REUSED from 20260920T082848Z-navsim_v2-none-06e257 — 5912 tokens (450 stage-1) identical by value, devkit 0a380a9, 6 patches, cache + export identical (not re-scored)
11:17:43 [navsim] STOP: REUSED from … — …, STOP seam byte-equal=True (not re-scored)
11:17:43 [navsim] scoring ECHO (seam) on navhard_two_stage …
```
Run **`…/20260921T091732Z-navsim_v2-refcv4b_b1_v72_40k-f99b29`**. ETA ECHO ~12:48, A1 + summary ~14:20.

⚠️ **A fourth latent defect, found while waiting and fixed before it could fire:** `read_summary.py`
silently dropped a MISSING floor from the pre-registered `max(CV, STOP, ECHO)` — so an ECHO killed by
the RAM guard would have let the bar be decided over CV + STOP only. That is moving the goalpost after
seeing which arms survived, and ECHO is exactly the floor this failure mode takes out. ⇒ a missing
floor now makes BAR-W7-1 **UNDEFINED**, verified with a discriminating control on the floors run
(ECHO absent → UNDEFINED; only present floors named → decided).

---

*(The results sections are appended once an arm is actually scored.)*
