# RESULT — E-ARCH-EGOZERO-1: the ego-zero collision is REAL, it is LARGE, and it sits on the trajectory path

*Architecture & Inference FlyWheel, 2026-09-03. **0 GPU** — every number below was computed
on the dev box (CPU) or read from the pod; **nothing was run on `tanitad-refcv3`, which is
training** (step 37,450/40,284 at the time of the reads). Pre-registration in `SPEC.md`;
raw counts under `raw/`.*

**Tier stamp.** §1–§3 and §5 are **structural / corpus facts** — they are properties of the
data and the source, not of a model, so no eval tier applies. §4 reads the live run's
in-training eval, which is **T0** (train-side instrument) and is **never** driving
performance. **No claim here is T1.**

---

## 0. The verdict in five lines

1. **The collision is real and it is the ONLY unguarded consumer of the ego speed.**
   `keep` is computed on every forward and is *already* passed to the decoder as `ego_keep`
   for the S2 reachability band; it is withheld from the measurement encoder — the primary
   conditioning route — **only** by `ego_valid_channel = False`. *(MEASURED, source.)*
2. **Base rate: 4.4531 % of the trainer's own train windows carry v0 EXACTLY 0.0**
   (34,807 of 781,635), rising to 6.548 % at ≤ 0.5 m/s. *(MEASURED, consumer-exact.)*
3. **Collision size: 52.227 % of training samples present a zero speed to the model, and
   only 4.263 % of those are a genuine standstill — 22.5 withheld for every 1 real.**
   At eval, dropout is off, so **100 %** of the zeros are genuine: the same input token
   means something **23.5× different** between train and eval. *(MEASURED + arithmetic.)*
4. **The longitudinal signature IS present and large (5.3×) — but it CANNOT be attributed
   to this defect**, because the heads that carry it never read the speed channel. The
   columns that *do* read it are not decomposed into longitudinal and lateral, so
   `metrics.jsonl` **cannot test this hypothesis at all.** *(MEASURED; a NEGATIVE on
   observability, and itself a work item.)*
5. **refav1 does not have this defect.** Its speed channel has no dropout and no zero-fill,
   and a missing `v0` **raises** instead of masking to zero. *(MEASURED, source.)*

---

## 1. The setup, re-verified at source (MEASURED)

Every line below was read on the pod, whose `refc.py` is **md5-identical to the repo**
(`b33656099fd3ace0e4ce17986aac16e1`), as is `refc_v3_train.py`
(`bf18716780036269092e17df76be756c`).

| fact | site |
|---|---|
| `ego_dropout: float = 0.5` — "per-sample Bernoulli zero of v0 (training)" | `stack/tanitad/refs/refc.py:429` |
| the dropout is **training-only**: `if self.training and self.cfg.ego_dropout > 0:` then `v = v * keep` | `refc.py:2032-2034` |
| `ego_valid_channel: bool = False` (X15) — its own comment: *"0.0 m/s is in-distribution 'stationary'; masking to it is a confident lie"* | `refc.py:585` |
| `meas_in = ([v, nav] + ([keep] if self.cfg.ego_valid_channel else []) + ...)` — with the gate off, `keep` is computed but **NOT fed** | `refc.py:2060` |
| v0 is **ALWAYS** supplied by the trainer (`v0 = pose_last[:, 3]`), so `keep` starts at 1.0 on every sample and only dropout zeroes it | `scripts/refc_v3_train.py:445`; docstring `scripts/refc_train.py:38` |

### 1.1 The live run has the gate OFF — three probes, three different path bindings

1. **`config.json`** on `tanitad-refcv3` at `/workspace/experiments/refcv3-b1-v72-30k/`
   contains **no ego key at all** (top-level keys enumerated programmatically; no
   `ego_dropout`, no `ego_valid_channel`).
2. **The running process** (PID 2311954) carries no ego flag — and
   `grep 'ego_valid\|ego_dropout' refc_v3_train.py` is **EMPTY on the pod AND in the repo**:
   the trainer has no CLI flag for either, so neither can be overridden from the launch line.
3. **The pod's `refc.py` is byte-identical to the repo**, so the dataclass defaults *are*
   the live values.

⇒ **`ego_valid_channel = False` and `ego_dropout = 0.5` in the model training right now.**
CONFIRMED, not assumed.

### 1.2 ⭐ The companion bit already exists — it is just not wired to the encoder

`refc.py:2157` passes `ego_keep=keep.squeeze(-1) > 0.5` into the decoder, and the decoder
honours it in **both** reachability sites (`:1354` anchor prefilter, `:1494` S2 clamp),
with a docstring that states the principle exactly: *"using it to filter candidates on a
sample whose speed was withheld … would leak the channel back in through the ranking."*

⇒ **The programme has already accepted this principle for the speed channel and
implemented the bit.** The defect is that the same bit is withheld from the **measurement
encoder**, which is the route by which the speed reaches the trajectory decoder at all.
The fix is a gate flip on a tensor the model already computes — not a new signal.

### 1.3 What actually consumes the ego-dropped `v` — the complete list

| consumer | reads | guarded? |
|---|---|---|
| `self.measurement(cat(meas_in))` → `m` → **the trajectory decoder** | the **dropped** `v` | ⛔ **NO** — this is the defect |
| core tactical heads (`lat_head`/`lon_head`) via `tac_in` | `pooled` **only** — `tac_in = ... if cfg.tactical_speed_input else pooled`, and `tactical_speed_input = False` (`refc_v3.py:213`, *"goal path stays vision-pure (E11)"*) | n/a — **never sees `v`** |
| S2 reachability band / anchor prefilter | the **raw** `v0` (`v_ms`), gated by `ego_keep` | ✅ yes |

**This table is the reason §4 reads the way it does.**

---

## 2. The base rate (MEASURED, consumer-exact)

**Artifact:** `/root/data/train/_v2manifest.pt`, md5 `00d7dcaefa50370b69a82257f0a9fe65`
(24,881,873 B, 4,572 clips) — verified identical after transfer. This is **the artifact the
consumer opens**: `build_v2_providers` hands the dataset `man["poses"][i]`
(`v2_dataset.py:543`), and `ep.poses` **is** that tensor.

**Enumeration, matched to the trainer exactly:** `pose_last = ep.poses[t + window - 1]`
(`_contract.py:137`) for `t in range(T - window - max_horizon)` (`refb_train.py:117`), with
`window = 8` (`refc.py:419`) and `max_horizon = 20` (`refc_v3_train.py:798`). ⇒ the fed
speed is `poses[t+7][3]` for t in `range(T-28)`.

**Speed provenance:** `v = ||(vx, vy)||` from the egomotion parquet, `physicalai.py:618-619`.

### 2.1 TRAIN — n = **781,635 windows** over **4,572 clips**

| band | windows | share |
|---|---|---|
| **v0 == 0.0 exactly** | **34,807** | **4.4531 %** |
| v0 ≤ 0.1 m/s | 40,712 | 5.209 % |
| v0 ≤ 0.5 m/s | 51,179 | 6.548 % |
| v0 ≤ 1.0 m/s | 62,094 | 7.944 % |
| v0 ≤ 2.0 m/s | 85,464 | 10.934 % |
| v0 ≤ 5.0 m/s | 183,282 | 23.449 % |

mean **10.570** m/s · median 9.132 · std 7.735 · p05 **0.0646** · p10 1.691 · p99 35.556 ·
max 39.960. Non-finite 0, negative 0. `T_out` min 189 / median 199 / max 207.

⭐ **The distribution is bimodal with a hard ATOM at exactly zero.** The 0.1 m/s-wide
histogram over [0, 2] reads `[40712, 3087, 2447, 2426, 2507, 2229, ...]` — the first bin is
**13× the second**, and 34,807 of its 40,712 are exactly `0.0`. This is a real standstill
mode (traffic lights), not interpolation noise. **1,011 of 4,572 clips** contain at least
one window at ≤ 0.5 m/s; **138 clips spend more than half their windows** there.

### 2.2 EVAL — n = **24,114 windows** over **141 clips**

exact 0.0 **740 = 3.069 %** · ≤ 0.1 **940 = 3.898 %** · ≤ 0.5 **1,260 = 5.225 %** ·
≤ 1.0 **1,553 = 6.440 %** · mean 11.361 · median 10.099.
*(md5 `4c146bbbd982ce79ee5ee6954d186438`.)*

### 2.3 Independent cross-check — a DIFFERENT path binding

The manifest is a built cache; a defect in the build would make §2.1 an artifact. Probed
the **raw egomotion parquet** instead (`egomotion_alpamayo.tar`, dev-box local), all
**4,719 clips**, all **14,094,549 native rows**, recomputing `hypot(vx, vy)` from source:

**exact 0.0 = 462,546 = 3.282 %** · ≤ 0.5 = 4.703 % · ≤ 1.0 = 5.653 %.

⇒ The zero atom is **real in the source** and not a cache artifact. The two numbers are
**not identical** (4.453 % vs 3.282 %) and should not be quoted interchangeably: the raw
probe is over *native* rows of *whole* clips, while the manifest measure is over the *10 Hz
resampled* grid restricted to the *window range* (which trims the first 0.7 s and last
2.1 s of every clip) on the *4,572* clips actually cached. **The consumer-exact §2.1 number
is the authoritative one**; §2.3 corroborates the phenomenon, it does not replicate the
statistic.

---

## 3. The collision arithmetic (MEASURED base rate + exact arithmetic)

`keep = 1` on every sample before dropout (v0 is always supplied), so the model's input
speed is `0` **iff** the sample was withheld **or** the car was genuinely stopped. With
`p = ego_dropout = 0.5` and `q = P(v0 == 0) = 0.044531`:

```
P(input reads 0)      = p + (1-p)·q = 0.5 + 0.5 × 0.044531 = 0.5222655   ->  52.227 %
share WITHHELD        = p / (p+(1-p)q) = 0.5      / 0.5222655 = 0.957367 ->  95.737 %
share GENUINE stopped = (1-p)q / (p+(1-p)q) = 0.0222655 / 0.5222655      ->   4.263 %
odds                  = 22.46 withheld : 1 genuine
```

**More than half of every training batch — 52.2 % of samples — shows the measurement
encoder a zero speed, and roughly 1 in 23.5 of those zeros is a car that is actually
stopped.**

| band used for "zero-ish" | P(input reads in band) | genuine share |
|---|---|---|
| exactly 0.0 (byte-identical) | **52.227 %** | **4.263 %** |
| ≤ 0.1 m/s | 52.604 % | 4.951 % |
| ≤ 0.5 m/s | 53.274 % | 6.145 % |
| ≤ 1.0 m/s | 53.972 % | 7.359 % |

⭐ **The train/eval asymmetry is the sharp part.** `ego_dropout` is training-only
(`refc.py:2032`), so at eval `input == 0 ⟺ v0 == 0` and the genuine share is **100 %**.
The same input token carries a **23.5× different** meaning between the two regimes
(4.263 % → 100 %). That is the nav-collapse defect's exact shape, on a second channel.

### 3.1 ⭐⭐ The consequence is STRUCTURAL, not merely statistical — and it is MEASURED

`refc_tactical.py:202-242` (md5 `a0b28c1296949f2f64ec61d9e2beaf33`, local clone ==
pod) defines the two label axes asymmetrically:

```
lat = f(dyaw)                                        <- does NOT read v0
lon = f(dv, v0, v1),  dv = v1 - v0                   <- READS v0
      brake_stop <= dv < -1.0  OR  (v1 < 0.3 AND v0 >= 1.0)
```

The stop-branch is gated on **`v0 >= MOVING_V_MS = 1.0`**, so `brake_stop` is
**unreachable at v0 == 0**; and from v0 = 0 with v1 ≥ 0, `dv ≥ 0`, so the dv-branch is
unreachable too. **Prediction: zero brake_stop windows at v0 == 0. MEASURED: 0 of 34,807
(train) and 0 of 740 (eval).** The control reads its known value exactly.

Conditional longitudinal label distributions over the trainer's own windows
(`raw/label_shift.json`), classes `(brake_stop, steady, accelerate)`:

| population | train | eval |
|---|---|---|
| corpus marginal | (0.1671, 0.6467, 0.1862), H = 0.8939 nats | (0.1604, 0.6841, 0.1555), H = 0.8427 |
| **given v0 == 0** | **(0.0000, 0.9084, 0.0916)** | **(0.0000, 0.8932, 0.1068)** |
| what TRAIN presents behind "input == 0" | (0.160, 0.658, 0.182) | — |
| what EVAL presents behind "input == 0" | — | (0.0000, 0.8932, 0.1068) |

⇒ At training the token "input speed = 0" carries **16.0 % `brake_stop` mass**; at eval the
same token carries **0.0 %**. Total variation between the two meanings: **0.2506** for the
longitudinal head against **0.1542** for the lateral one — **1.63× larger on the axis whose
label is a function of the dropped value.**

⚠️ **Two honest caveats.** (a) The lateral distribution is *also* shifted (0.1542), because
a stationary car is also not turning — so the asymmetry is 1.63×, not infinite; the
*structural* difference is that `lon` reads v0 as an **argument** while `lat`'s shift is a
mere correlation. (b) These are **input-channel** ambiguities; the model also has vision,
which is **not** subject to them and can partly identify a stopped car. **The measured
shift is therefore an UPPER BOUND on the damage, not the damage.** Quantifying the realised
damage is exactly what `SPEC.md` exists to do.

---

## 4. Does it show in the live run? — the honest answer is **this file cannot tell us**

**Artifact:** `/workspace/experiments/refcv3-b1-v72-30k/metrics.jsonl`, snapshot md5
`b3cbd3d39f7344be45583d0ff0313401` (424,536 B, **873 rows**, 78 eval rows, steps
550–37,450). The run is live and the file is still growing; this is a snapshot, not a final
read. **Tier T0.**

⛔ **No confidence interval is computable from this file by any estimator** — every value is
already a pooled mean over 160 windows, with no per-window rows and no episode ids
(independently recorded in `GOALS_AND_CLAIMS.md`, D-REFCV3-EPOCH-READ point 5). Everything
below is a **DIRECTION**, never an interval.

**Eras** (from the 2026-09-03 10:05 programme report; they may never be pooled): A ≤ 16,500
(pre-nav-switch); 16,500–17,500 the in-training eval leaked held-out label marginals
(`C-REFCV3-EVAL-PRIOR-LEAK`); **C ≥ 18,000 is the only clean era**.

### 4.1 The columns are NOT comparable raw — and in comparable units the gap is 5.3×

`lat`/`lon` are the core aux over the **kin3 3-class** vocabulary (chance ln 3 = 1.0986);
`lat_tac`/`lon_tac` are the z_tac heads over the **v7.2 EIGHT-class** vocabulary
(`HEADS['tac_lat'] = HEADS['tac_lon'] = 8`, chance ln 8 = 2.0794). Comparing them raw is
invalid. The admissible statistic is the share of the no-information loss removed,

**skill = 1 − CE / H(label marginal)**,

which reads **exactly 0** for a prior-only predictor and is in the same units for any
vocabulary. `H` is **MEASURED** from the eval corpus' own labels (§3.1), not assumed:
`H_lat = 0.5195`, `H_lon = 0.8427` nats.

| era | n eval rows | lat skill (last 10) | lon skill (last 10) | lat − lon | lat beats lon |
|---|---|---|---|---|---|
| A ≤ 16,500 | 37 | **+0.7362** | **+0.0967** | +0.6395 | 36 / 37 |
| B leaked | 3 | +0.7463 | +0.1079 | +0.6383 | 3 / 3 |
| **C clean ≥ 18,000** | 38 | **+0.7627** | **+0.1437** | **+0.6190** | **38 / 38** |

The z_tac 8-class heads agree directionally: raw `lon_tac`/`lat_tac` CE ratio **1.108 →
1.179 → 1.305** across the eras, with `lon_tac` worse on **38/38** clean-era rows.

⇒ **The longitudinal signature is present, consistent and large: the lateral head has
removed 76.3 % of its no-information loss, the longitudinal head 14.4 % — a 5.3× gap, on
every one of the 38 clean-era rows.**

### 4.2 ⛔ …and it CANNOT be attributed to the ego-zero collision

Per §1.3, **`tactical_speed_input = False`**, so `tac_in = pooled` and the heads producing
`lat` / `lon` / `lat_tac` / `lon_tac` **never receive the speed channel at all**. The
ego-dropout cannot reach them. The 5.3× gap is real, but it is **evidence about something
else** — most plausibly the already-documented 5-way manoeuvre collapse and the intrinsic
difficulty of predicting longitudinal intent from vision.

The columns that *do* consume the collided `v` are the trajectory path — `eval_traj`,
`eval_goal2s_err_m`, `eval_anchor_acc` — and **none of them is decomposed into a
longitudinal and a lateral component.** *(For reference, era C: `goal2s_err_m` mean 2.1172,
last-10 2.0038, best 1.8720 — a single scalar with no axis split.)*

⇒ **VERDICT: `metrics.jsonl` cannot test H-ARCH-EGOZERO-1.** The hypothesis is neither
supported nor refuted by the live run. Reporting the 5.3 % / 5.3× gap as support would be a
**"true but wrong for the reader"** error — a correct number implying a wrong next action.

⚠️ **This is itself a finding and a work item.** The binding four-families rule requires a
longitudinal family on every eval; the live run's trajectory instrument has **no
longitudinal/lateral decomposition**, so the largest single family cannot be read from it at
all. `SPEC.md` §2.1.3 and §4 carry the fix forward.

---

## 5. What refav1 does instead — it AVOIDS the collision (MEASURED, source)

`stack/tanitad/refs/refa_v1.py`, md5 `3a539ea2996f1337e627a1e64c5fddda`:

| fact | site |
|---|---|
| `speed_channel: bool = False` (default off) — when on, the predictor input is `(a, kappa, v_k / SPEED_SCALE_MPS)` | `refa_v1.py:400`, `:386-399` |
| the channel is **DERIVED**, never supplied: `v_k = v0 + Σ_{j<k} a_j·op_dt` | `refa_v1.py:1311-1355` |
| **a missing `v0` RAISES** — `"speed_channel=True needs v0 [B]"` — it is never masked to zero | `refa_v1.py:1343-1346` |
| a pre-widened action tensor is **refused**, so no caller can inject a speed the rule cannot audit | `refa_v1.py:1338-1341` |
| **no ego dropout and no zero-fill of v0 exists anywhere in the file** (grep for dropout/mask/keep on v0 returns nothing) | — |

**The live refav1 run has it ON**: `/home/nvidia/experiments/refav1-b1-v72-ep3-speed/config.json`
stamps `args.speed_channel = True` and `cfg.speed_channel = True` (read on Thor).

⇒ **refav1 has NO ego-zero collision.** There is no withheld state to confuse with a
standstill: the speed is always present or the call fails loudly. A `0` in refav1's third
channel means exactly one thing — the car is stopped.

⚠️ **A provenance gap worth closing regardless of this hypothesis's outcome:** refav1's
`config.json` **stamps** `speed_channel`; refcv3's `config.json` stamps **neither**
`ego_dropout` nor `ego_valid_channel`. The live refcv3 launch record cannot answer "was the
speed masked, and how often?" — it had to be reconstructed from three source probes (§1.1).
`SPEC.md` §2.1.1 requires the stamp.

---

## 6. Evidence-class ledger

| claim | class |
|---|---|
| the gate is off in the live run; `keep` computed but unfed; `ego_keep` already guards S2 | **MEASURED** (source, pod == repo by md5; three probes) |
| base rate 4.4531 % exact-zero over 781,635 windows | **MEASURED** (consumer-exact artifact, md5-verified) |
| 3.282 % exact-zero over 14,094,549 raw egomotion rows | **MEASURED** (independent path binding) |
| 52.227 % of samples present a zero; 4.263 % genuine; 23.5× train/eval meaning shift | **MEASURED** base rate + exact arithmetic |
| `brake_stop` unreachable at v0 == 0; 0 of 34,807 | **MEASURED** (prediction from source, confirmed on data) |
| lat/lon skill 0.7627 / 0.1437 in era C | **MEASURED**, tier **T0**, **no CI computable** |
| the 5.3× gap is *not* attributable to this defect | **MEASURED** (source: `tactical_speed_input = False`) |
| refav1 avoids the collision | **MEASURED** (source + live config) |
| closing the gate will IMPROVE driving | ⛔ **HYPOTHESIS** — `SPEC.md`, unlaunched |

⛔ **Nothing here is a T1 capability claim, and no arm has been spent.**

---

## 7. Instruments (all re-runnable, 0 GPU, ~4 min total on the dev box)

| file | what it does |
|---|---|
| `ego_zero_base_rate.py` | §2 — consumer-exact base rate + collision arithmetic from the v2 manifest |
| `label_shift.py` | §3.1 — conditional label distributions and the train/eval meaning shift |
| `metrics_lat_lon.py` | §4 — era-split lat/lon in skill units, with the marginal entropies measured |

Inputs are the two `_v2manifest.pt` sidecars and `metrics.jsonl`, all md5-verified against
the pod after transfer. Outputs are in `raw/`.
