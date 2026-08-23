# The comma-yaw anchor, settled — the disqualification-lifting claim is WITHDRAWN on its headline evidence and survives only on the other half, at half the size

**Date:** 2026-07-27 · **Agent:** `anchor-settlement` · **HEAD at start:** `84028f4`
**Hosts:** dev box (RTX 4060) + `tanitad-eval` (A40, idle). ⛔ **pod1 (training) and pod2 (small
validation) were NOT touched** — no SSH, no process, no read. pod3 was not needed.
**Closes:** `…/2026-07-27-heading-default/HEADING_DEFAULT.md` §7 escalations **#1**, **#2**, **#3**.
**Reads with:** `Project Steering/RETRACTION_LOG.md` classes **C29**, **C42** (and the new **C43**
this pass adds).

**Evidence class + tier on every number. Label protocol stated beside every number. Per corpus,
never pooled.** **Estimator:** `taniteval.ci.episode_cluster_bootstrap` (callable `r2` reducer) and
`paired_episode_cluster_bootstrap`, B = 2000, unit = the episode. ⛔ **`overlapping_holdout_se` is
not called anywhere in this work.** 🔒 Counts only, no clip UUIDs. 🔒 Parity untouched — comma2k19
is a NON-PARITY corpus and no PhysicalAI path, key or selection was modified; **no PhysicalAI number
is re-issued** (`n_pai_changed = 0`, re-measured here, not inherited). ⛔ Nothing pushed to HF.

---

## 0. Headline

> **1. 🔴 THE ANCHOR IS CONTAMINATED. Overlap = 2, by content.** Two of the anchor's **22** comma
> evaluation episodes are **bit-identical** — sha256 of the raw pose bytes **and** of the raw
> `frames_u8` sensor bytes — to two of `idm_head_v1`'s own **40 comma TRAINING** clips. Not similar:
> the same bytes. `76b:ep_00018 ≡ 61c:ep_00008` and `76b:ep_00039 ≡ 61c:ep_00020`.
>
> **2. ⭐ Those 2 episodes ARE the anchor.** The published anchor reproduces exactly here
> (**+0.011430 → +0.330822**, and its CI to 3 decimals). Remove the two contaminated episodes — 272
> of 2,992 windows, **9.1 %** — and the deployed head's repaired comma yaw R² goes
> **+0.3308 → −0.746** (CI **[−1.574, −0.177]**, separated, and *negative*). The two alone read
> **+0.856**, *identically under all three label protocols* — the repair does nothing to them; the
> head simply memorised them.
>
> ⇒ ⛔ **`+0.3308` is WITHDRAWN.** It is not "partially valid" and it is not a smaller positive
> number. On content-disjoint comma clips the deployed head reads comma yaw **negative**.
>
> **3. ⚠️ It was never separated from zero anyway, and the artifact said so.** The published
> `compare_v3.json` records the anchor's own interval as **[−1.2982, +0.7047]** and this pass
> measures the OFF→ON contrast at **+0.3194, CI [−1.262, +0.6425] — NOT separated**. The
> disqualification was lifted on a point estimate whose interval spanned zero *before* anyone knew
> about the leak.
>
> **4. ⭐ The OTHER half of the claim is a different problem and must not be lumped in.** `+0.679` is
> `R0` — and **`R0` IS the shipped `V3F`'s rotation head** (`ship_v3.json → composite.rotation_from`;
> the two comma yaw values are bit-identical). `R0` trained on the v3 TRAIN split, which is
> content-disjoint from its val split, so **`R0` is NOT contaminated and `+0.6791` is not withdrawn.**
> What it *is* is **composition-fragile**: on the same 20 content-clean episodes `R0` reads
> **+0.3038 (CI [+0.054, +0.479], separated)** — still real, **less than half** the quoted level.
> Every one of the 18 persisted v3 arms loses **0.36–0.58** R² the same way.
>
> **5. ⇒ What the disqualification-lifting claim is now worth.** The *conclusion* — **comma CAN test
> yaw** — survives, but **only on the retrained-head evidence and at +0.30, not +0.68**, and with
> **none** of the deployed-head evidence it was announced with. On its own held-out comma clips the
> deployed head does not recover yaw under **any** protocol (`61c`: legacy +0.000048, repaired
> −0.000421, strict-admissible −0.288; `76b` clean-20: legacy −0.001, repaired −0.746). **Testable ≠
> working**, and that is the cleaner statement.
>
> **6. ⭐ Admissibility is now expressed in code, and it is the lesson that outranks the fix.**
> `hold_heading_through_standstill`'s `observable` mask has a consumer: a sanctioned label
> derivation whose **default cannot produce a silent number**. Measured consequence, per corpus:
> on `61c` it removes **every** impossible label and collapses the label's own std
> **0.938 → 0.046 rad/s** (258 of 4,140 windows); on `76b` it drops **50 of 2,992** and moves R² by
> **+0.007**. ⚠️ **The consequence is corpus-dependent; the correctness is not.**
> PhysicalAI: **0 windows changed, 0 dropped, R² bit-identical** — measured here, not inherited.

---

## 1. Pre-registration — fixed before the probe ran

Quoted verbatim from the brief, which set both outcomes before any measurement:

> **overlap 0** ⇒ the anchor is clean and the +0.3308 stands *on its substrate*.
> **overlap > 0** ⇒ **the anchor is contaminated and every claim resting on it — including the
> lifted disqualification — must be withdrawn or re-measured.**
> ⛔ **Do not soften a contaminated result into "partially valid."**

It is recorded machine-readably in `raw/anchor_overlap.json → /pre_registration`, written by the
same run that produced the answer. **Outcome 2 fired.** This document does not soften it: §0.2 says
*withdrawn*, not *reduced*.

⚠️ **One thing the pre-registration did not anticipate, and it matters:** it treats "the anchor" as
one object. The re-issue's sentence actually rests on **two** numbers with **two different
provenances** — a contaminated one (`+0.3308`, the deployed head) and an uncontaminated but fragile
one (`+0.679`, a retrained head). Applying "withdraw everything" to both would have been *wrong in
the other direction*. §4 separates them.

---

## 2. 🔴 Job 1 — the overlap, BY CONTENT

**Raw:** `raw/anchor_overlap.json` · **producers:** `code/fingerprint_comma_cache.py` (run on both
hosts) → `raw/fp_61c46fca8f7f.json`, `raw/fp_76b6e94a97a1.json` → `code/intersect_by_content.py`
**Evidence class:** `MEASURED (ours; sha256 of raw bytes on both hosts)`.
**Tier:** decision-grade — the comparison has **no dependence on any naming convention**.

### 2.1 Why names could not be used

`comma2k19.episode_id_of` is `int(sha1(f"{route_folder}/{segment_name}")[:8], 16)` — **a hash of a
PATH**. So is `ep_00018.pt`, and so is the `cm_00018` tag. The brief's rule stands on measured
history: **600/600 filename overlap with 0/600 real overlap**, and separately **4 of 36 val
episodes that would have leaked (11 %)** in a cache carrying no `episode_id` at all.

### 2.2 What was hashed

Per episode, on the host that owns the cache: sha256 of the raw **`poses` [T,4] float32** bytes (the
brief's named primary), of the **x/y** columns alone, of the **yaw** column alone, of the **speed**
column alone, of **`actions` [T,2]**, and of the entire **`frames_u8` [T,9,256,256] uint8** tensor —
**the raw sensor bytes, which no label protocol can touch.** Frames are hashed incrementally
frame-by-frame (peak RSS = one frame, and per-frame digests are kept for a shift-tolerant fallback
that was not needed).

### 2.3 The sets, and the exact paths compared

| | set A | set B |
|---|---|---|
| cache | `comma2k19-val-61c46fca8f7f` | `comma2k19-val-76b6e94a97a1` |
| path | `C:\Users\Admin\tanitad-data\eval\comma2k19-val-61c46fca8f7f\ep_*.pt` | `tanitad-eval:/root/valdata/comma2k19-val-76b6e94a97a1/ep_*.pt` |
| host | dev box `FREEDOM2035` | `tanitad-eval` (`1e0bac0df88a`) |
| episodes fingerprinted | **90** (all) | **64** (all) |
| roles | `A0_TRAIN cm_[0:40]` = 40 · `A0_HELDOUT cm_[40:70]` = 30 · `A0_UNUSED cm_[70:90]` = 20 | `V3_VAL` (every 3rd) = **22** · `V3_TRAIN` = 42 |

`V3_VAL` is `cm_00000, cm_00003, …, cm_00063`, read from the run's own record
(`…/2026-07-27-idm-v3/results/arms_v3.json → /split/val_eps`), not reconstructed from prose.

### 2.4 ⭐ THE ANSWER

| cross-tab (by `poses_sha256`; `frames_sha256` agrees on every cell) | count |
|---|---:|
| **`A0_TRAIN` × `V3_VAL` — the leak** | 🔴 **2** |
| `A0_TRAIN` × `V3_TRAIN` | 9 |
| `A0_HELDOUT` × `V3_VAL` | **0** |
| `A0_HELDOUT` × `V3_TRAIN` | **0** |
| `A0_UNUSED` × `V3_VAL` / `V3_TRAIN` | **0** / **0** |

**The two contaminated pairs, with the exact files compared:**

| the anchor evaluated on | is bit-identical to | which A0 used for | `poses_sha256` / `frames_sha256` (first 16) | `episode_id` |
|---|---|---|---|---|
| `tanitad-eval:/root/valdata/comma2k19-val-76b6e94a97a1/ep_00018.pt` (tag `cm_00018`) | `C:\Users\Admin\tanitad-data\eval\comma2k19-val-61c46fca8f7f\ep_00008.pt` | **TRAINING** | `94aba1c14a2bf1c7…` / `86eca5f5f6668d5c…` | `2831231267` |
| `…/comma2k19-val-76b6e94a97a1/ep_00039.pt` (tag `cm_00039`) | `…\comma2k19-val-61c46fca8f7f\ep_00020.pt` | **TRAINING** | `8e6a09ae4d371355…` / `d738a24373370ba5…` | `2160020170` |

*(full hashes in `raw/anchor_overlap.json`. `poses_bitwise_equal` is asserted `true` on the decoded
float32 arrays as well, so the match is not a hash artefact. Both `episode_id`s appear in the 40
A0-comma-TRAIN ids the `heading-default` pass staged — the name-derived cross-check agrees.)*

### 2.5 The checks that could have falsified it — all run, all reported

| check | result |
|---|---|
| do all **6** content hash families give the same global intersection? | ✅ **11 / 11 / 11 / 11 / 11 / 11** |
| does the **name-derived** `episode_id` agree? | ✅ **11** — it *could* have disagreed; it did not. Reported as corroboration, **never as the evidence** |
| **within-cache duplicates** (one segment cached twice under two names) | **0** in both caches (90 distinct of 90; 64 of 64) |
| ⭐ **is the C42 substrate itself clean?** `A0_TRAIN` × `A0_HELDOUT` *inside* `61c`, by content | **0** by poses **and** 0 by frames — the C42 re-score's substrate is genuinely disjoint. Its `episode_id`-based claim was a NAME claim; this is the content one |

⚠️ **Note the shape of the finding.** All **11** cross-cache content matches involve `A0_TRAIN`;
**none** involves A0's heldout or unused clips. The two caches were built from the same 21 routes of
`raw_data/Chunk_1` by different samplers, and the collision landed entirely inside the 40 clips A0
trained on. That is not a coincidence to wave at — it is why a *name* check could never have caught
this, and why the same probe should run before any future cross-cache comparison.

---

## 3. ⭐ The consequence — the anchor re-measured on its own substrate

**Raw:** `raw/anchor_resettlement.json` · **producer:** `code/resettle_anchor.py` (on `tanitad-eval`)
**Evidence class:** `MEASURED (ours)`. **Tier:** decision-grade.
⛔ **Nothing retrained, nothing re-encoded** — the persisted `a0_preds.npy` (the exact array the
anchor was computed from) is re-used, so every row differs only by *which windows and which labels*
enter the statistic.

**Pins, all asserted at run time:** deployed head md5 `fa4462f0b898b036be729c790278b823` = the card's
`/weights_md5`; an **independent reimplementation** of the repair reproduces the labels the anchor
was scored against to `2.07e-06` (the float32 storage precision of `val_gt_v3.npy`); the anchor's own
published values reproduce **exactly**.

### 3.1 The reproduction pin

| | published (`compare_v3.json`) | this pass |
|---|---:|---:|
| comma yaw R², legacy | `+0.011430149647448706`, CI [−0.866, +0.6719] | **+0.011430**, CI [−0.866, +0.672] |
| comma yaw R², repaired | `+0.3308220914810539`, CI [−1.2982, **+0.7047**] | **+0.330822**, CI [−1.298, **+0.705**] |

### 3.2 🔴 Leave-the-contaminated-two-out — the deployed head, `heading_repair` protocol as stated

| subset | n win / eps | **legacy** | **repaired** (`v_min` 0.5) | **strict-admissible** |
|---|---:|---:|---:|---:|
| **`cm_ALL22`** — *the published anchor* | 2 992 / 22 | +0.011430 | **+0.330822** | +0.337404 |
| 🔴 **`cm_CLEAN20`** — *content-verified disjoint* | 2 720 / 20 | −0.001230 | **−0.745999** · CI **[−1.574, −0.177]** | −0.727306 |
| **`cm_INTRAIN2`** — *the leak, alone* | 272 / 2 | +0.856185 | **+0.856185** | +0.856185 |
| `pai_ALL14` (control) | 1 203 / 14 | +0.903482 | +0.903482 | +0.903482 |

**Read the `INTRAIN2` row carefully — it is the mechanism.** The three protocols give the *same*
number to 6 decimals, i.e. **the repair changes nothing on those two clips** (they carry 0 of the 9
impossible legacy labels). They are simply the two clips whose yaw label is **4.29× wider** than the
rest (`gt_std` 0.1081 vs 0.0252 ⇒ **18.4×** the variance), and the head has seen them. R² is
variance-weighted, so **9.1 % of the windows carry 61 % of the total sum-of-squares**
(3.181 of 5.188 rad²/s²) — enough to move the headline by **1.08**.

**And the honesty conditions did not warn us.** MAE moves −0.0187 on ALL22 and −0.0205 on CLEAN20 —
essentially the same. ρ moves 0.3438→0.3448 on ALL22 and 0.3201→0.3211 on CLEAN20 — essentially the
same. **Every caveat behaves identically on the contaminated and the clean set.** That is C42's
treachery repeating one level up.

### 3.3 PhysicalAI — measured here, not inherited

`n_pai_windows_changed_by_repair` = **0**; legacy, repaired and strict-admissible R² all
**+0.903482, bit-identical**; strict admissibility drops **0** windows (the heading is
quaternion-derived, so it is observable by construction). ⛔ **No PhysicalAI number is re-issued.**
*Diagnostic only:* applying the same `v ≥ 0.5` gate to PhysicalAI as a blunt speed filter would drop
32 of 1,203 windows and move R² +0.9035 → +0.9044 — which shows the admissibility rule is **not**
"drop slow windows".

---

## 4. ⭐ The other half of the claim — `+0.679` is a DIFFERENT problem

**Raw:** `raw/arms_resettlement.json` · **producer:** `code/resettle_arms.py`

**Identification, not inference:** `ship_v3.json → composite.rotation_from = "R0"`, and
`ship_v3.json → V3F/yaw_rate/cm/r2` is **bit-identical** to
`compare_v3.json → arms/R0/yaw_rate/cm/r2` = `0.6791059738542922`. **V3F's rotation head IS R0**, so
measuring R0 measures the `+0.679` that the re-issue quoted.

⚠️ **R0 is NOT contaminated.** It trained on the v3 **TRAIN** split of `76b`, which is disjoint from
the v3 val split by content (the cache has **no duplicate episodes at all**, §2.5). The two leaked
clips are in *A0's* training set — A0 is a different model. **`+0.6791` is therefore NOT withdrawn.**

What it is, is **composition-fragile** — and so is every arm on this substrate:

| arm | comma yaw R², `ALL22` | comma yaw R², **`CLEAN20`** | Δ | ρ ALL22 → CLEAN20 |
|---|---:|---:|---:|---|
| `A0_deployed` (the anchor) | +0.3308 | 🔴 **−0.7460** [−1.574, −0.177] | **−1.077** | 0.345 → 0.321 |
| **`R0` ≡ `V3F` rotation** | +0.6791 [+0.131, +0.812] | **+0.3038 [+0.054, +0.479]** ✅ separated | −0.375 | 0.602 → **0.598** |
| `V2R` (v3 translation arm) | +0.6608 [+0.010, +0.821] | +0.2023 [−0.112, +0.417] — not separated | −0.459 | 0.562 → 0.543 |
| `R0LEG` (legacy-trained) | +0.5894 | +0.0639 [−0.374, +0.334] — not separated | −0.526 | 0.522 → 0.514 |
| best of all 18 arms (`G1n`) | +0.6978 | **+0.3381** | −0.360 | — |

*(all 18 persisted arms are in the raw JSON; every one loses **0.36–0.58**.)*

⭐ **The instructive contrast between the two halves.** For the deployed head the two clips flip the
sign; for the retrained heads they roughly **halve** a real effect while **ρ barely moves**
(0.602 → 0.598). A retrained head genuinely ranks comma yaw; what it does not do is explain 68 % of a
variance that 2 of 22 episodes supply. ⇒ **`R0`'s honest clean-episode number is `+0.3038`**, and
that — not `+0.679` — is what "comma can test yaw" is worth.

---

## 5. ⭐ Job 2 — admissibility: the decision, the implementation, the consequence

### 5.1 The decision

> **Admissibility belongs to the LABEL DERIVATION, expressed as a sanctioned function whose DEFAULT
> cannot produce a silent number — not to the episode contract, and not to each scorer's discretion.**

Both shapes HEADING_DEFAULT §7 offered were rejected, for measured reasons:

| candidate | ⛔ why not |
|---|---|
| **(a) widen the episode contract** (`poses` `[T,4]` → carry admissibility) | `ToyEpisode.poses` is `[T,4]` for **four** corpora. Widening it for a defect that exists in **one** changes every corpus's cached bytes under keys that already mean something — the exact "same key, different content" failure C29 is about. And it is **unnecessary**: `observable` is a pure function of `poses[:, 3]`, which every consumer already holds. **Nothing needed storing.** |
| **(b) "every yaw scorer must call the mask"** | That is the status quo restated. The mask *was* returned, documented, and tested — and **no caller consumed it for five days**. An obligation with a permissive default is the defect, not the fix. |

⇒ What was actually missing was **a derivation whose default refuses silence.** Implemented as
`admissibility="nan"`: inadmissible windows come back **NaN**, `n` is unchanged (so window sets stay
comparable), and **any metric over them becomes NaN** — a number that cannot be quoted by accident.
`"drop"` is available when `n` may legitimately change. `"keep"` — the pre-2026-07-27 behaviour —
requires `allow_inadmissible=True` **and a non-empty written reason**, the
`resolve_heading_mode` / `resolve_vision_rank` discipline: *a boolean can be flipped
absent-mindedly, a sentence cannot.*

### 5.2 What shipped

| file | change |
|---|---|
| `stack/tanitad/data/comma2k19.py` | `admissible_from_poses`, `heading_admissible_centers` (**the one definition** of the t−1/t/t+1 rule), `yaw_rate_from_heading`, `assert_yaw_rate_admissible`, `InadmissibleYawLabel`, `KEEP_INADMISSIBLE_YAW_REASON`, `YAW_RATE_ADMISSIBILITY` / `DEFAULT_YAW_RATE_ADMISSIBILITY` |
| `stack/tests/test_yaw_admissibility.py` | **new, 19 tests** — every refusal exercised on the input that must trigger it, plus the two non-over-reach directions |

⛔ **Nothing existing changed meaning.** No cache key, no `label_params`, no existing label, no
signature. The additions are purely new surface; `stack/` went 1557 → **1576 passed** (+19 = exactly
the new file) with **12 skipped, zero new skips**, and `taniteval/` **663 passed**.

**The guard is demonstrated firing**, not asserted (class **C13**): `"keep"` without acknowledgement
raises; with the flag and no reason raises; with a blank reason raises; with a reason and no flag
raises; a typo in the policy raises `ValueError` rather than being treated as `keep`; and the refusal
message carries the measured mechanism so whoever hits it need not go find out why. The two
directions that matter are both pinned — the wholly-stationary clip loses **every** window, and a
fully observable segment loses **none** and reproduces the naive derivation exactly.

**The API's first consumer is real, not a fixture:** `code/admissibility_consequence_61c.py` derives
every label in §5.3's `61c` row through it. On that substrate the **default** policy returns NaN for
**258** windows across **6 of 30** clips, and `mean(|yaw_rate|)` over the wholly-stationary clip is
**NaN** — the fail-loud behaviour, demonstrated on corpus data.
⚠️ **That check itself had to be fixed to be able to fail:** its first version looked only at the
first clip, `cm_00040`, which is fully observable — it reported *"no NaN"* while the prose claimed
otherwise. It now sweeps all 30. **A check that cannot see the condition it is cited for is class
C13**, and it nearly shipped inside the pass whose subject is exactly that.

### 5.3 ⭐ The measured consequence, per corpus, never pooled

**A. `comma2k19-val-61c46fca8f7f`, `cm_[40:70]`** — `idm_head_v1`'s own held-out comma clips,
**content-verified disjoint** from its training clips (§2.5). 30 clips, 4,140 windows, `k=4`,
`stride=2`. Deployed head, nothing retrained.
**Raw:** `raw/admissibility_consequence_61c.json` · `MEASURED (ours; dev box)`

| protocol | n | **yaw R²** | CI | ρ | nMedAE | label `gt_std` | impossible labels |
|---|---:|---:|---|---:|---:|---:|---:|
| legacy | 4 140 | +0.000048 | [−0.004, +0.003] | +0.1980 | 2.262 | 1.2270 | **132** |
| repaired (`v_min` 0.5) | 4 140 | −0.000421 | [−2.288, +0.148] | +0.1987 | 2.490 | 0.9382 | **84** |
| **strict-admissible** | **3 882** (−258, 6.2 %) | **−0.288336** | [−3.170, +0.286] | +0.2107 | 2.365 | **0.0460** | **0** |

**B. `comma2k19-val-76b6e94a97a1`, the v3 val split** — the anchor's own corpus. 22 clips, 2,992
windows, `k=8`, `stride=2`. **Raw:** `raw/anchor_resettlement.json`

| subset | strict-admissible drops | impossible: legacy → repaired → strict | R² repaired → strict |
|---|---:|---|---:|
| `cm_ALL22` | **50 / 2 992 (1.7 %)** | 9 → 0 → 0 | +0.3308 → **+0.3374** |
| `cm_CLEAN20` | 50 / 2 720 | 9 → 0 → 0 | −0.7460 → **−0.7273** |

**C. PhysicalAI** (`physicalai-val-0c5f7dac3b11`, 14 val eps, 1,203 windows): **0 windows changed by
the repair, 0 dropped by admissibility, R² +0.903482 bit-identical** across all three protocols.

⭐ **The statement that has to travel with these two rows:** on `61c` admissibility is the difference
between an *undefined* statistic and a defined one — it removes **84 physically impossible labels up
to 15.28 rad/s** contributed by a single wholly-stationary clip and collapses the label's own std by
**20×**. On `76b` it moves almost nothing, because that split contains **no wholly-stationary clip at
all** — its least-observable val clip (`cm_00060`) still has **194 of 300** observable frames, and
**0 of 22** have zero; on `61c` `cm_[40:70]`, `cm_00045` has **0 of 300** (v_max **0.039 m/s**) and
supplies all 138 of its windows to the statistic. ⚠️ **The consequence is
corpus-dependent; the correctness is not.** A number computed over windows with no defined label is
not slightly wrong on `76b` — it is undefined and happened to land near the defined answer.
⛔ And raising `v_min` still cannot substitute: 84–85 survivors at 1.0 / 2.0 / 4.0 m/s.

### 5.4 ⇒ "Testable ≠ working" — stated plainly, because it is the cleanest thing here

**comma2k19 yaw is measurable.** The label defect is real, its mechanism is understood, the repair is
correct, admissibility now makes the label *defined*, and a **retrained** head reads
**+0.3038 (CI [+0.054, +0.479], separated)** on content-clean episodes.

**And the deployed head does not do it.** On its own held-out comma clips with a clean label it reads
**R² −0.288** (interval spans zero, so *"negative"* is not itself decision-grade), **ρ 0.211**,
**nMedAE 2.36 — worse than predicting the median**; on the anchor's content-clean episodes,
**−0.746**. Both substrates, one answer.

That is a different statement from *"the label was broken"* (the 2026-07-27 re-issue) and from
*"comma is disqualified for yaw"* (the pre-2026-07-27 conclusion). **Both were wrong in opposite
directions.** The channel is testable; this model does not perform on it.

---

## 6. What is now true about the disqualification-lifting claim

| the claim as reported to the PI | status now |
|---|---|
| *"the deployed head reads comma yaw **+0.3308** on repaired labels"* | ⛔ **WITHDRAWN.** Contaminated: 2 of its 22 eval episodes are, by content, in its own training set. Content-clean value: **−0.746**. It was also never separated from zero (published CI [−1.2982, +0.7047]) |
| *"a retrained head reads **+0.679**"* | ✅ **stands as published** (no leak) — but it is **+0.3038 [+0.054, +0.479]** on the 20 content-clean episodes, and the number's dependence on 2 of 22 episodes must travel with it |
| *"three documents concluding **comma is disqualified for yaw** were wrong"* | ✅ **still true**, and for the reason they gave — the mechanism was a label artefact (**C29**). But the *evidence* for the lift is now **one** retrained-head number at **+0.30**, not two numbers at +0.33/+0.68 |
| *"this gates the YouTube-IDM rotation channel"* | ⚠️ **the gate should be re-opened on the corrected size.** A rotation channel justified by +0.68 and one justified by +0.30-with-a-fragility-caveat are different decisions. **Escalated (§8.1)** — I did not re-take that decision |

---

## 7. Environment — what was verified, and one thing `stack_check` structurally cannot see

| | |
|---|---|
| `tanitad-eval` stack | `python3 -m taniteval.stack_check --require v5` ⇒ **`ok: true`** — **only** with `TANITEVAL_STACK_OVERRIDE=/workspace/TanitAD-head/stack`. With no override it **REFUSES**, resolving `/root/TanitAD/stack` (a pre-v5 tree) |
| `tanitad-eval` git | `8ab5327` — **2 commits behind** the repo's `84028f4`. Irrelevant to these measurements (they read persisted artifacts and hash raw bytes) but stated rather than assumed |
| 🔴 **the trap the brief named, CONFIRMED** | `idm2_lib.py:19` and `idm3_a0.py` run an unconditional `sys.path.insert(0, "/root/taniteval")`. On this host that ci.py is **`ef925f06febd20a99f5901491fcf75cb`** while HEAD's is **`c92618a02b36f8191a581fb74a491a8d`** — **a different estimator implementation**, and `stack_check` cannot see it because it checks the *tanitad* stack, not which *taniteval* was imported |
| what I did about it | imported **neither** module; pinned `sys.path` to `/workspace/TanitAD-head/taniteval` and **asserted the md5 of the ci.py actually loaded** before computing anything. `code/admissibility_consequence_61c.py` asserts the repo-relative path locally. Both assertions are in the raw JSON (`ci_py_md5`) |
| suites | `stack/` **1576 passed, 12 skipped** (baseline 1557/12 → +19 = exactly the new test file, **zero new skips**); `taniteval/` **663 passed** (baseline 663, unchanged, zero new skips) |

⚠️ **A small correction to a sibling claim, found on the way.** HEADING_DEFAULT §2.1 records the
repair's moving-part heading as *"bit-identical"*. It is not, in general: the repair rebuilds every
heading as `arctan2(sin, cos)`, so an **observable** frame moves by up to **1 ULP (1.11e-16 rad)**,
reaching the yaw-rate label at **5.6e-16 rad/s** — measured over 10,000 random headings, ~4 % of
frames affected. That held for its fixture (whose moving part had yaw exactly `0.0`). Physically
irrelevant; *"bit-identical"* is a claim with a definite meaning, and it is pinned now by
`test_the_repair_is_a_float_ROUND_TRIP_on_observable_frames_not_a_noop`.

---

## 8. 🔴 Escalations — these need an owner, not a note in a file

1. ⭐ **The YouTube-IDM rotation gate was re-opened on `+0.3308 / +0.679`; `+0.3308` is withdrawn and
   `+0.679` is `+0.3038` on clean episodes.** Its owner should re-take the go/no-go on the corrected
   size. **I did not re-take it** — that is a program decision, not a measurement.
2. ⭐ **The highest-value place to run this probe next is `physicalai-val-0c5f7dac3b11` ×
   `physicalai-train-e438721ae894`.** `MODEL_REGISTRY.md:97` states that val is *"episode-disjoint
   from train ✅"* — that is an **`episode_id` claim, not a content claim**, and this program has
   **already measured a 78 % leak once** on a sibling val cache (`physicalai-val-f1b378f295ae`,
   **62 of 79 episodes** into parity train, `…/2026-07-23-planner-frozen-wm/STATUS.md`). That pair
   decides the credibility of **every** parity-val headline, and it has never been checked by
   content as far as I could find.
   ⚠️ **Stated carefully, because absence at one location is not absence:** the discipline *does*
   exist in places — `idm4_steer.py:210` and `idm5_ensemble.py:248` both assert *"pool/val CONTENT
   overlap"*. What I found no instance of is a **cross-CACHE** content check, which is precisely the
   case that defeated names here. `code/fingerprint_comma_cache.py` +
   `code/intersect_by_content.py` run in **~25 s per cache** and need only torch.
3. **`idm2_lib.py` / `idm3_a0.py` still insert `/root/taniteval` unconditionally**, and that file is
   a *different* ci.py from HEAD on `tanitad-eval`. Every v3 interval was produced through it. The
   point estimates are unaffected (they are not computed by `ci.py`), but **the published v3 CIs were
   produced by an unpinned estimator** — worth one re-run, and worth deleting the two `sys.path`
   lines.
4. **`ADM`-style admissibility exists now for comma only.** The same question is open for any corpus
   whose heading is velocity-derived (YouTube pseudo-labels in particular). Nobody has checked.
5. **Wholly-stationary clips remain a corpus-selection question** (HEADING_DEFAULT §7 #4, unchanged):
   `61c:cm_00045` is 30 s of a parked car. Admissibility now makes it *harmless*; it does not decide
   whether it belongs in a driving corpus.

---

## 9. Deliverable manifest

**STAGED, NEVER PUSHED.** No commit, no push, no branch switch. **Nothing lives in only one place.**

| # | artifact | where it lives | note |
|---|---|---|---|
| 1 | `ANCHOR_SETTLEMENT.md` | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-anchor-settlement/` | this file |
| 2 | `code/fingerprint_comma_cache.py` | same dir | content fingerprints; run on **both** hosts |
| 3 | `code/intersect_by_content.py` | same dir | the overlap, the hash-family agreement, the duplicate and self-overlap checks |
| 4 | `code/resettle_anchor.py` | same dir | the anchor re-measured; also `/root/resettle_anchor.py` on `tanitad-eval` |
| 5 | `code/resettle_arms.py` | same dir | all 18 persisted v3 arms; also `tanitad-eval:/root/resettle_arms.py` |
| 6 | `code/admissibility_consequence_61c.py` | same dir | the second corpus, **through the shipped API** |
| 7 | `raw/fp_61c46fca8f7f.json` | same dir | 90 episodes × 6 content hashes + per-frame digests |
| 8 | `raw/fp_76b6e94a97a1.json` | same dir | 64 episodes, ditto; produced on `tanitad-eval` |
| 9 | `raw/anchor_overlap.json` | same dir | ⭐ **the answer**, the pre-registration, every path and count |
| 10 | `raw/anchor_resettlement.json` | same dir | §3 + §5.3B + the PhysicalAI control |
| 11 | `raw/arms_resettlement.json` | same dir | §4, all 18 arms |
| 12 | `raw/admissibility_consequence_61c.json` | same dir | §5.3A |
| 13 | `stack/tanitad/data/comma2k19.py` | `repo:` | ⭐ the admissibility API + its own header/refusal-message anchor quotes amended |
| 14 | `stack/tests/test_yaw_admissibility.py` | `repo:` | 19 tests |

### 9.1 Documents corrected — **every** location that carried the anchor

⛔ **Every superseded value is preserved with its date.** No number was deleted, no block rewritten;
each correction is an appended, dated, class-tagged amendment. ⛔ `PRE_REGISTRATION_IDMV2.md` and
`Mission Plan.md` were **not opened**. ⛔ **No PhysicalAI number was re-issued**, and no "repaired
ceiling" was derived — it stays unknown until measured.

| # | document | what it carried | what it says now |
|---|---|---|---|
| 15 | ⭐ `…/2026-07-24-idm-pipeline-derisk/RESULTS_idm_pipeline_derisk.md` | **the 1st of the three** — *"comma cannot test yaw pseudo-labels"*, overturned on `+0.3308`/`+0.679` | overturn **stands**, on `+0.30` from a retrained head; `+0.3308` withdrawn |
| 16 | ⭐ `…/2026-07-24-branchb-transfer-eval/v1-encoder-char/RESULTS_v1_encoder_char.md` | **the 2nd** — caveat **C6**, disqualification lifted | lift **stands**, resized; do not read `+0.33/+0.68` as a v1 rotation capability |
| 17 | ⭐ `…/2026-07-24-branchb-transfer-eval/MANIFEST.md` | **the 3rd** — *"untestable on comma"* lifted | lift **stands**, resized; `yaw +0.504` and all rig numbers unaffected |
| 18 | `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` | the re-issue itself | new **§0.1** amendment block; §3.3 and §8 #3 annotated |
| 19 | `…/2026-07-27-heading-default/HEADING_DEFAULT.md` | §5.5's open probe, §7 #1–#3 | probe **resolved (non-empty)**; #1/#2/#3 marked actioned, #4/#5 left open |
| 20 | `…/2026-07-27-idm-v3/IDM_V3.md` | §4.2 — **the anchor's producer** | comma column withdrawn; PhysicalAI column and E1's ordering explicitly **not**; the `/root/taniteval` CI provenance escalated to its author |
| 21 | `…/2026-07-27-idm-v3/MODEL_CARD_IDM_V3.md` | the shipped card's `comma2k19 yaw 0.679` | **not withdrawn**; composition caveat + the leave-2-out figure |
| 22 | `…/2026-07-26-idm-v2/IDM_DIAGNOSIS.md` | the `0.3521` ceiling retraction, argued from the anchor | retraction **stands on the mechanism**; the number-vs-number comparison marked inadmissible; ⛔ still **no repaired ceiling** |
| 23 | `…/2026-07-26-idm-v2/IDM_V2_RESULTS.md` | §3.2's three-protocol table | comma column withdrawn, ordering kept; `R0LEG +0.5894` flagged not-separated on clean-20 |
| 24 | `…/2026-07-26-idm-youtube-db-retry/DB_RETRY.md` | defect #2's closure + `0.0719 → +0.3308` | closure **stands**; the *size* of the recovery withdrawn |
| 25 | `…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` | the deployed head's own card | **additive** block `anchor_settlement_2026_07_27`; every existing value programmatically re-read and asserted identical |
| 26 | `Project Steering/MODEL_REGISTRY.md` | §8.1 #6's `LABEL-PROTOCOL CORRECTION` block | ⭐ the registry's **quotable statement** restated; `yaw R² 0.000` cell still STALE-PENDING |
| 27 | `Project Steering/RETRACTION_LOG.md` | C5 / C29 / C42 | **new class C43**; three forward pointers, **no row rewritten** |
| 28 | `Benchmarks & Eval/LEADERBOARD.md` | §7.3's annotation | amended; **FAIL verdict unchanged** (speed 0.657 + ADE 2.40) |
| 29 | `Project Steering/LOOP_STATE.md` | the pod3 IDM entry | amended; **NO-GO left standing** |
| 30 | `Project Steering/PROGRAM_OVERVIEW.md` | the `yaw ≈ 0` inline note | amended; cell stays STALE-PENDING |
| 31 | `Paper/TANITAD_PAPER.md` | §H7 **and** the §7.9 summary line | both amended; the three-measurement H7 argument never rested on yaw and is unchanged |
| 32 | `stack/tests/test_comma2k19.py`, `stack/tests/test_comma_heading_regime.py` | two docstring copies of the anchor | amended; **the guards and the 26.27 % defect table are untouched** — they never depended on it |

**On the eval pod only (regenerable, deliberately not staged):** `/root/fp_76b6e94a97a1.json`
(identical to #8), `/root/anchor_resettlement.json`, `/root/arms_resettlement.json` — all three are
byte-identical to the staged copies (md5 verified on both sides).
**Not staged:** the 30 re-encoded latents in the scratchpad (37 MB, regenerable in ~90 s from
md5-pinned artifacts) — the `heading-default` pass's decision, unchanged.
