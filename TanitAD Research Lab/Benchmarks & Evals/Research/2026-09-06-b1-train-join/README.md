# B1 TRAIN agent join — the corpus `--agents head` actually trains on

**Date** 2026-09-06 · **Agent** B1 TRAIN join · **Evidence class** MEASURED (ours) ·
**Tier** N/A (a label artifact, not an eval result) · **Compute** dev-box CPU + network,
**zero GPU** — the A40 was left untouched for `refcv5-cap-b1-v72-40k`

The B1 **EVAL** join (`../2026-09-06-b1-agent-join/`) closed WP-6's evaluation side and
named its own next lever: *"Agent conditioning during TRAINING needs the B1 TRAIN split —
4,572 clips — and that join does not exist."* This package is that join.

⭐ It does double duty. The consumer derives **`v_rel_x`** — the **closing rate** — from
this file, and a frozen-trunk probe measured closing rate as a **clean null on every arm**
(+0.0061; an explicit temporal difference recovered nothing) and named it a
**representation** defect. So agent conditioning is simultaneously the environment link
DiffusionDrive has and refcv5 lacked, **and** rank-1's missing representation.

---

## 0. ⛔ Quota first — read before any byte moved

**Account `Sayood`, `isPro: true`, `billingMode: prepaid`, period ends 2026-10-01.**

| metered thing | reading | what this pull adds |
|---|---|---|
| HF **storage** (owned repos) | **≥ 761.625 GiB** over the 12 largest owned repos (817,788,428,526 B) | **0 B** |
| HF **ZeroGPU** compute | Space `Sayood/TanitAD` **SLEEPING**, `hardware.current = None` | **0 s** |

⭐ The pull **touches neither**. It is an anonymous-shaped **READ** of
`nvidia/PhysicalAI-Autonomous-Vehicles` (`author: nvidia`, `private: false`, `gated: auto`,
246 TB, 208,205 downloads) writing to **local disk** — not an upload to our storage and not
an HF GPU job. Explicit quota endpoints (`/users/Sayood/storage`, `/billing`,
`zerogpu-quota`) all return **404**, so no numeric download quota is exposed; the two
quantities the PI's rule actually names are the two above, and the delta on both is zero.

⚠️ **The listing endpoint lies about storage.** `GET /api/models?author=Sayood&full=true`
returns `usedStorage: 0` for **all 46** owned repos — a field it does not populate, not a
real zero. The per-repo endpoint returns the truth (424.102 GiB for
`tanitad-physicalai-w120-256x640cyl` alone), and `nvidia/…` reading 246 TB on that same
endpoint is the positive control that separates "the field is empty" from "the value is 0".

## 1. ⭐ The cost was RE-PRICED before the pull, and the published figure was the wrong quantity

The plan of record carried **2.3 GB**, ESTIMATED as `505 KB × 4,572 clips` from **one**
datum (the eval obstacle parquets on disk) with no receipt.

⛔ **That anchor measures EXTRACTED PARQUET BYTES ON DISK. The pull's cost is HTTP
range-request WIRE volume**, and the zip tail (central directory) is read once per chunk —
**1,387 times** — while the anchor prices it zero.

**Method (stated, not implied):** a counter wrapped around the only place bytes enter the
process, with the per-chunk FIXED phase separated from the per-clip MEMBER phase, measured
first on a **40-chunk random sample** (seed 20260906) and then on the whole pull.

| | MEASURED |
|---|---|
| fixed cost per chunk | **262,146 B** = 256 KiB tail + 2 × 1-byte range probes |
| member wire per clip (sample, n=121) | 619,875 B — extracted + **91 B** local file header |
| **sample-based prediction** | **3.107 GB**, 95 % CI **[2.665, 3.549]** |
| **ACTUAL, whole pull** | **2.673 GB** over **17,391 HTTP range requests**, 18.5 min |
| extracted on disk | **2.311 GB** across **4,440** parquets, **0** zero-byte files |

⚠️ **Report the miss, not just the hit.** The prediction landed at the bottom of its own
interval: the 40-chunk sample drew heavier-than-average members (**619,875** B/clip against
the corpus's **517,098 B**), so the point estimate was **1.16× high** while the
interval covered the truth. The sample was honest and under-powered, and saying so is the
difference between a measurement and a lucky number.

⚠️ **And the published 2.300 GB looks nearly right for the wrong reason.** Extracted bytes
came in at **2.311 GB** — within ~0.5 % — because **two errors cancelled**: the
estimate counted 4,572 clips when only **4,440** have a member, and under-counted
bytes per clip. The quantity that is actually spent is the **wire** figure, **2.673 GB**,
**15.7 % above** the extracted bytes.

⭐ Coverage: **4,440 / 4,572 = 97.11 %** of B1 TRAIN clips have an
`obstacle.offline` member; **132** have none. The dataset card's own **97.44 %**
is the independent cross-check — this is a corpus fact, **NO_LABEL, never "road clear"**.

⚠️ **"Has a member" and "has labels" are two different things**, and the sidecar counts
them separately: **6** of the pulled parquets exist but hold **0 rows**. They are refused
with the reason recorded (`skipped.error`), not silently joined as empty — an empty
`agents` list IS a label meaning clear road, so admitting a 0-row parquet would have
taught the head that 200 unlabelled frames were 200 empty roads.
⇒ the clip ledger reconciles exactly: **4,572 offered − 132 no member − 6 zero-row
− 7 gate-excluded = 4,427 joined**.

## 2. The corpus, as a fraction with both numbers and the corpus NAMED

**B1 TRAIN = `r0_selection.parquet` (4,719) − the 147 EVAL clips = exactly 4,572.**

⛔ A train-side join *does* exist — for a **different corpus**.
`train2400_agents.jsonl.xz` covers **2,308 clips of the parity TRAIN corpus**, of which
**182 are also in B1 TRAIN**: **182 / 4,572 = 3.98 %**. Saying "the train join does not
exist" without naming the corpus is the exact failure the EVAL package documented; saying
"we have a train join" without naming it is the same failure with the sign flipped.

## 3. ⭐ The 161 GB blocker that turned out not to exist

`build_b1_agent_join.py` read poses from `*.v2ep.pt`. The B1 TRAIN epcache is
**~34 MB/episode × 4,713 ≈ 161 GB** and lives on **Thor**, not this box — and the join reads
exactly **one `[T, 4]` array (~6 KB)** out of each 34 MB record.

⛔ **Do not price an artifact by the file you found; price what the CONSUMER reads.**
`build_lead_block_b1` already reconstructs the poses from `{clip}.timestamps.parquet` +
egomotion (`episode_grid` → `episode_poses` — the same two functions
`physicalai.build_episode` itself uses), and the banked
`grid_crosscheck.json` measures that reconstruction **`max_dxy_m 0.0`, `max_dyaw_deg 0.0`,
`max_dv_mps 0.0` on 20/20 clips** — float32-**IDENTICAL**, not merely close.

⭐⭐ **Proven end to end on the real product, not on a proxy statistic:** re-running the
**141 EVAL clips** through the new `--pose-source reconstruct` path reproduces the banked
v2ep-built artifact **byte for byte** — md5 **`3ddb42ecbd3926066795a94587af2aed`**,
10,012,564 B, 139 clips, 26,394 lines, 905,512 boxes. The pose source changed and the
output did not.

⇒ every non-obstacle input is local: **egomotion 4,572/4,572** and
**timestamps 4,572/4,572**.

## 4. ⛔ The gate had to change, and the first replacement was WRONG

The EVAL gate compares **two independently produced speed arrays** indexed by the same
frame (v2ep `poses[:,3]` vs the banked block's `speeds`). **B1 TRAIN has neither**: no lead
block, and the poses are *built on* the camera grid, so `poses[:,3]` **is** the reference's
own speeds — comparing them would read exactly 0.0 with infinite separation and measure
nothing.

⚠️ **The first fix manufactured a difference and was refuted by its own control.**
Re-sampling one side at the *registered* times and keeping the speed statistic gives, on
the 141 EVAL clips, **true 1.191e-01 m/s vs mis-join(+1) 2.134e-02 m/s** — **the control is
SMALLER than the true value**, separation **0×**. It had stopped measuring key correctness
and started measuring registration PRECISION: the same confusion the EVAL package diagnosed
for its invented 5e-3 s time gate, re-introduced in a new costume one package later.

⭐ **The gate asks the key-correctness question in its own units — FRAMES.** A frame index
is an integer, so the decision boundary is **0.5 frames**; the thresholds sit a 2× margin
either side (`≤ 0.25` true, `≥ 0.75` control), and the mis-join(+1) control must read a
**known value ≈ 1.0**, not merely a larger one. The two sides are genuinely different time
bases: `register_poses_to_time` recovers time from **POSITION**, the reference from the
**CAMERA TIMESTAMP GRID**.

| statistic (clips KEPT) | value |
|---|---|
| TRUE `max` frame error | **0.1872** frames (allowed ≤ 0.25) |
| mis-join(+1) control `min` | **0.8193** frames (required ≥ 0.75, expected ≈ 1.0) |
| separation | **4.4×** |
| median / p95 / p99 across the corpus | **0.0032 / 0.0103 / 0.0317** frames |

### 4.1 ⭐ The gate is PER CLIP — and that is stricter, not looser

The first full build **REFUSED all 4,433 good clips** because a global max is dominated by
the single worst. Refusing the corpus protects nothing — the offender would simply be
dropped — so a failing clip is **EXCLUDED and NAMED** in `skipped.alignment_failed`,
exactly as a clip with no obstacle member is. This is **stricter** than the global form,
which would have passed a corpus containing a clip at 0.24 frames.

⚠️ Because *"exclude the failures"* is one edit away from *"loosen the gate"*, a
corpus-level bound stays: **> 0.5 % excluded refuses the build**, because a systematic
one-frame defect would hit many clips at once. MEASURED here: **7 of 4,434 =
0.158 %**, ~3× inside the bound.

### 4.2 ⭐ Why those clips fail — a class, with a control, not an anecdote

`raw/registration_underdetermination.json`

| group | n moving poses | `n_probe` | mean speed | frame error |
|---|---|---|---|---|
| **7 offenders** | 9–22 of ~200 | **9–22** | 0.148–0.524 m/s | **0.2506–1.0506** |
| **6 controls** | 152–201 | **96** (the cap) | 2.52–29.04 m/s | **0.0020–0.0075** |

**No overlap on either axis**, and the error is monotone in the probe count. The mechanism
is exact: probes are restricted to poses whose **central-difference** neighbour distance
exceeds `MIN_PROBE_MOVE_M = 0.30 m`, the floor to attempt a fit is
`MIN_MOVING_PROBES = 8`, and the cap is `n_probe = 96`. A near-stationary clip therefore
fits `t = a + b·i` from a 9-point lever arm and **extrapolates the intercept across ~200
frames**.

⛔ **The cliff edge, exactly:** `065482b3` has **9** moving poses — **one above the floor of
8** — so it did *not* raise `RegistrationError`; it returned a confident fit that is
**1.0506 frames wrong**. *A guard that refuses at 8 returns a wrong answer at 9.* Its
sibling `af6f5964` in the EVAL build had **0** and was correctly rescued.

⭐ **Named lever, not run.** The **camera timestamp grid is constitutive** —
`physicalai.build_episode` *defines* frame time as
`linspace(t_cam[0], t_cam[-1], n_target)`, so the registration only *recovers* what the grid
*is*. Falling back to the grid when **`n_probe < 32`** would recover every excluded clip.
Not run: it costs a full ~40 min rebuild to recover **0.158 %** of clips whose ego
mean speed is **< 0.53 m/s** — i.e. clips that contribute essentially nothing to a
**closing-rate** representation. The criterion is pre-fit (`n_probe`), never selected on the
resulting error.

## 5. ⭐ INDEPENDENT cross-check: 182 clips against a join built by a different builder

The frame gate proves the time base; it does not prove the **agent geometry**.
`train2400_agents.jsonl` was built by **`build_obstacle_join.py`** from the **parity
corpus's v2ep epcache** — a different builder, a different pose source, a different corpus —
and **182** of its clips are in B1 TRAIN. Both key `frame_idx` in the same post-trim space
(its `n_frames` 199 = 201 − 2 confirms n_stack 3), so the comparison is direct.

| | keys | tracks | `|Δcx|` median / p95 / max | `|Δcy|` median | `|Δyaw|` median |
|---|---|---|---|---|---|
| **TRUE** (same `frame_idx`) | 34,764 | 1,063,192 | 0.0000 / 0.0000 / 4.0407 | 0.0000 | 0.0000 |
| **CONTROL** mis-join(+1) | 34,732 | 1,036,188 | 0.6954 / 2.2729 / 1300.8881 | 0.0460 | 0.1513 |

⭐ **The statistic that actually means something: EXACT agreement.**

| | shared boxes | `|Δcx|` **exactly 0** | `|Δcx| > 0.05 m` |
|---|---|---|---|
| **TRUE** | 1,063,192 | **1,061,921 = 99.8805 %** | 495 = 0.0466 % |
| **CONTROL** (+1) | 1,036,188 | 691 = **0.0667 %** | 938,935 = 90.6143 % |

**99.88 % of over a million agent boxes are BIT-EQUAL** (to the join's 4-decimal rounding)
to a join built by a different builder, from a different pose source, for a different
corpus — and one frame of shift collapses that to 0.0667 %. **Separation 1,497.5×** on a
statistic bounded in [0, 1].

⚠️ **A ratio that was NOT quoted.** The obvious summary — median `|Δcx|` control / true —
computes to **695,400,000×**, because the TRUE median is **exactly 0.0**. That is a division
by zero wearing the shape of a spectacular result, and it is retracted in the banked JSON
rather than deleted. The exact-agreement fraction replaces it.


## 6. The artifact

`b1train_agents.jsonl.xz` — md5 **`1c985e6d6ad34e605c4ebd30cb353558`** (scope: **compressed**), **317,028,572 B**.

| | |
|---|---|
| clips joined | **4,427 / 4,572** offered (B1 TRAIN) |
| lines (labelled frames) | **849,263** — unique keys 849,263, no duplicates |
| agent boxes | **28,053,187** |
| visible (in-field) boxes | 11,270,675 — `visible_frac` **0.4018** |
| `max_agents_per_frame` | **397** |
| mean \|cx\|, \|cy\| | **49.7568 m, 22.1761 m** (non-degenerate — the zeros-file assertion) |
| build wall | 30.1 min, dev-box CPU |

⭐ **A third independent cross-check, free:** `visible_frac` reads **0.4018** here,
**0.4078** on the B1 EVAL join and **0.4106** on `train2400` — three joins built at
different times over **disjoint clip sets** agreeing to within **0.009**. That is a real
consistency check on the geometry and the field mask, not a restatement of one number.

⛔ **Not in git — deliberately, and by precedent.** At **317 MB** this follows
`train2400_agents.jsonl.xz` (136 MB), of which git holds **only the sidecar**. The sidecar,
this README and the reproduce command are banked; **the artifact lives at
`C:\Users\Admin\tanitad-caches\b1-train-join-20260906\`**. Pushing it to HF is a
**storage-quota action and therefore a PI decision** — it is the one thing here that would
consume the quota §0 measured, so it was not done autonomously.

### Content assertions (on the WRITTEN bytes, never the builder's counters)

A decode that raises into a pre-allocated buffer leaves a full-size file of zeros and can
still exit 0, and this mount has produced correct-size all-NUL files. So the build re-reads
what it wrote and REFUSES on: 0 lines, 0 boxes, a line count disagreeing with the counter,
duplicate `(clip_id, frame)` keys, or degenerate coordinates (`mean|cx| == 0`).

### Both index spaces are emitted

| key | index space | consumer |
|---|---|---|
| `frame` | RAW v2ep index `i` | refav1 adapter / lead-block |
| `frame_idx` | post-`n_stack` trim (`i − 2`) | `train_p8_occupancy.JoinFileReader`, which `refc_v3_train.py --agent-join` uses |

At n_stack 3 the two differ by **2 frames ≈ 0.2 s ≈ 2.7 m of lead displacement at
13.6 m/s**. A reader ignores keys it does not know, so one artifact serves both.
⚠️ The builder's module docstring claimed it emits *"`frame`, never `frame_idx`"* and
**contradicted its own code** since the first build; corrected here.

## 7. Verified through the REAL consumer

Not by inspecting the file — by loading it with the class `--agents head` opens.

| check | result |
|---|---|
| `train_p8_occupancy.JoinFileReader` | `n_records=849,263  n_clips=4,427  max_agents_per_frame=397  has_classes=True  has_occlusion_flags=True` |
| `lookup(uid, 10)` | **HIT** — 19 agents, `[A, 6]` |
| `lookup(uid, 100000)` | **None** — the must-miss control |

⛔ **The reader must be given STABLE ids.** MEASURED on these 4,572 clips: the **63-bit
`episode_uid_of_clip` is unique — 4,572/4,572, zero collisions**, but the **16-bit legacy id
puts 258 clips (5.6 %) into colliding groups**, which the reader detects at load and
REFUSES at lookup. ⇒ **do NOT pass `--agent-join-allow-legacy-ids`** on this corpus; the v2
providers' default `stable_ids=True` is required, not optional.

### ⛔ The join was UNLOADABLE by the trainer, and the sidecar is why

MEASURED here, by reading the consumer rather than assuming it:
`refc_v3_train.py --agent-join-verify auto` calls
`join_meta.read_digest_scope`, which requires a **top-level `digest_scope`
block** and **REFUSES** a sidecar that declares nothing — deliberately, with no
fallback. `build_b1_agent_join.py` wrote only **`summary.digest_scope`, a STRING
nested inside `summary`**, which that reader never looks at. ⇒ a perfectly good
join would have killed the train run at startup with `JoinDigestScopeMissing`.

⚠️ **And the value was invalid too:** the builder emitted `"plain"` for a
non-`.xz` output while `ARTIFACT_SCOPES` is `("compressed", "decompressed")`, so
even a reader that found the string would have raised on it. Two independent
defects in one field, neither reachable by looking at the builder alone.

⭐ **Fixed at the source and migrated:** the builder now calls
`join_meta.attach(...)` and **REFUSES to publish** if the declaration cannot be
made; the two already-built sidecars were migrated with `join_meta.backfill`,
which **MEASURES** which artifact the recorded digest covers rather than
guessing — B1 EVAL resolved unambiguously to `compressed`
(`3ddb42ecbd3926066795a94587af2aed` vs the decompressed
`51551ca060d95946f8a25ad59466924a`), and the consumer's own `join_meta.verify`
now passes on both. Pinned by two mutation tests.

### `v_rel_x` — present, populated, and in stated units

⚠️ **It is NOT a field in the join.** It is derived by the consumer:
`JoinFileReader(with_rates=True)` → `agent_slots.track_rates_from_join`.

> **`v_rel_x = d(cx)/dt` [m/s]**, with `cx` the agent's along-track coordinate in the
> **per-frame EGO frame** (+x forward) and `dt` **read from the records' own `t_s`** (grid
> ≈ 0.1007 s, never assumed 0.1). Central difference when both neighbours carry the
> `track_id`, one-sided when one does, **MASKED when neither** — zero is a legitimate value
> (a stationary car), so a missing rate is never zero-filled.
> `v_rel_y = d(cy)/dt` [m/s]; `yaw_rate_rel = d(yaw)/dt` [rad/s], wrapped before dividing.

| | value |
|---|---|
| coverage | **2,527,336 / 2,530,548 = 0.9987** of boxes carry a rate (400 clips, 72,360 labelled frames) |
| `v_rel_x` mean / median | **-6.558 / -5.914 m/s** |
| p05 / p95 | **-19.59 / 3.30 m/s** |
| min / max | **-4990.7 / 2589.6 m/s** |
| exactly zero / non-finite | **0.0334% / 0.0000%** |

⭐ **The sign is physically right, which is the cheapest available sanity check:** a median
of **-5.914 m/s** is negative because static roadside structure streams *backwards*
in the ego frame at −v_ego. A median near 0 would have meant the ego-frame composition was
not applied.

⚠️ **The tails are wide and the head will need to cope**: min/max reach
**-4990.7 / 2589.6 m/s**, far outside anything physical. These are track-association
artefacts — a `track_id` re-appearing at a distant position across one frame — not a unit
error, since the same records give a physically correct median. **Flagged, not resolved**:
a rates loss on this corpus wants clipping or a robust penalty, and that is a head-design
decision, not a join defect.

## 8. Reproduce

```
set PYTHONPATH=<repo>/stack;<repo>
python <cache>/pull_b1_train_obstacle.py ^
    --clips b1_train_clips.json --chunk-map <b1>/r0/r0_selection.parquet ^
    --out-dir <labels>/obstacle_offline_b1train --keys <repo>/Keys.txt ^
    --workers 8 --report wire_cost_full.json
python stack/scripts/build_b1_agent_join.py --pose-source reconstruct ^
    --ts-dir      <physicalai>/camera/camera_front_wide_120fov ^
    --clips       b1_train_clips.json --n-stack 3 ^
    --ego-dir     <physicalai>/labels/egomotion_alpamayo ^
    --obstacle-dir <physicalai>/labels/obstacle_offline_b1train ^
    --out         <cache>/b1train_agents.jsonl.xz
```

## 9. Is `--agents head` runnable on B1 TRAIN now?

****YES, on the dev box, today.** The join loads through the real consumer (`JoinFileReader`: 849,263 records / 4,427 clips), its sidecar now declares the digest scope `refc_v3_train.py --agent-join-verify auto` requires, and `v_rel_x` is populated on 99.87 % of boxes in stated units. Three operational conditions, all named and satisfied or explicit: **(1)** the trainer must use STABLE episode ids -- never `--agent-join-allow-legacy-ids`, because the 16-bit id collides on 258 clips (5.6 %); **(2)** `--agent-pad 0` will size the padded block to **397** targets per window, so pass a smaller value deliberately if that is too wide -- it truncates VISIBLY (counted in `agent_n_truncated`), never silently; **(3)** the 317 MB artifact lives on the **dev box only**, so a pod or Thor run needs it shipped, and pushing it to HF is a storage-quota action and therefore a PI decision.**

---

### Deliverable manifest

| artifact | where it lives |
|---|---|
| builder (extended: `--pose-source reconstruct`, the frame gate, the per-clip exclusion) | `stack/scripts/build_b1_agent_join.py` (repo) |
| guard tests (17, all by MUTATION) | `stack/tests/test_b1_agent_join.py` (repo) |
| **the join** | `C:\Users\Admin\tanitad-caches\b1-train-join-20260906\b1train_agents.jsonl.xz` (dev box — **not git, see §6**) |
| sidecar (per-clip stats, gate proof, exclusions, provenance) | `raw/b1train_agents.jsonl.xz.meta.json` (repo) |
| wire-cost measurements | `raw/wire_cost_sample40.json`, `raw/wire_cost_full.json` (repo) |
| under-determination class | `raw/registration_underdetermination.json` (repo) |
| cross-check vs `train2400` | `raw/xcheck_vs_train2400.json` (repo) |
| consumer + `v_rel_x` verification | `raw/verify_consumer_b1train.json` (repo) |
| obstacle parquets (4,440, 2.311 GB) | `C:\Users\Admin\tanitad-data\physicalai\labels\obstacle_offline_b1train\` (dev box) |
| pull / cross-check / verify tools | `raw/pull_b1_train_obstacle.py`, `raw/xcheck_vs_train2400.py`, `raw/verify_consumer.py` (repo) |
| this doc | `README.md` (repo) |
