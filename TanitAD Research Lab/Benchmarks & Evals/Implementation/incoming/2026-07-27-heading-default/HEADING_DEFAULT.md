# comma2k19 heading — the default fixed, and the re-score that says the opposite of the anchor

**Date:** 2026-07-27 · **Agent:** `heading-default` · **HEAD at start:** `8ab5327`
**Host:** dev box only (RTX 4060). **pod1 (training ~21,650/30,000), pod2 (small validation,
cgroup 53.9/55.0 GB) and pod3 were NOT touched** — no SSH, no pod GPU, no eval on a training host.
**Closes:** `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` §8 escalations **#1** and **#2**.
**Reads with:** `Project Steering/RETRACTION_LOG.md` class **C29**, `…/2026-07-27-idm-v3/IDM_V3.md` §4.

**Evidence class + tier on every number. Label protocol stated beside every number**
(`heading_repair` on/off, `v_min`) — two numbers differing only by protocol are otherwise
indistinguishable on the page, which is the whole reason C29 stood for five days.
**Estimator:** `taniteval.ci` episode-cluster bootstrap, B = 2000, unit = the episode; contrasts use
the **paired** version on identical windows. ⛔ **`overlapping_holdout_se` is not used anywhere.**
🔒 Counts only, no clip UUIDs. 🔒 Parity untouched: comma2k19 is a NON-PARITY corpus and no
PhysicalAI path, key or selection was modified.

---

## 0. Headline

> **1. The defect is no longer the default.** `DEFAULT_HEADING_MODE` moved
> `enu_velocity` → **`enu_velocity_hold_v1`**, and legacy now **raises** unless the caller passes
> `allow_legacy=True` **and a written reason**. Every accident path — `None`, a missing config key,
> an unset default — resolves to the **repair**. **No existing cache changed meaning**: legacy
> contributes no cache-key fragment (`4bce7a330c31` before = after), the repair contributes one
> (`79a902d541f6`), and acknowledged-legacy output is **bit-identical** (`max_abs_delta = 0.0`) to an
> independent reimplementation of the pre-flip formula. **23 new tests**, every refusal
> **demonstrated firing**.
>
> **2. ⭐ The re-score was DONE — on the dev box, no pod, no training — and it does NOT reproduce
> the anchor.** On `idm_head_v1`'s **own held-out comma clips** (the card's own split, 4,140 windows
> / 30 clips), the repair moves comma yaw **R² +0.000048 → −0.000421** — i.e. **it does not move at
> all.** MAE falls **−33.3 %** (paired, separated), medAE **−2.6 %**, **nMedAE +10.1 % WORSE**,
> Spearman ρ **flat** (+0.198 → +0.199).
>
> **⚠️ The honesty condition the brief carried is REPRODUCED IN SHAPE AND BROKEN IN MAGNITUDE.**
> The anchor's *pattern* (MAE down hard, medAE barely, nMedAE worse, ρ flat) appears exactly. The
> anchor's **R² lift (+0.0114 → +0.3308) does not.** Anyone who had pasted the anchor into the
> card would have published a **+0.33 that is actually ≈ 0.**
>
> **3. ⭐ WHY — measured, after one hypothesis was tested and refuted: ONE WHOLLY-STATIONARY CLIP.**
> `cm_00045` is 300 frames with **zero observable frames** (v_max **0.039 m/s**). The repair
> deliberately returns the heading **unchanged** for such a segment — inventing one would be worse —
> so all **138** of its windows keep a garbage label, **84** of them physically impossible (up to
> **15.28 rad/s**). Those 84 windows (**2.0 %** of the split) pin R² at zero. **Raising `v_min`
> cannot help**: 84–85 survivors at 1.0 / 2.0 / 4.0 m/s.
>
> **4. 🔴 The fix already exists and NO CALLER USES IT.**
> `hold_heading_through_standstill` returns an `observable` mask precisely for this
> (*"callers that need a strict admissibility mask should keep the second return value"*). Applying
> it (**a third protocol, a diagnostic, not a re-issue**) drops 6.2 % of windows, removes **every**
> impossible label, and collapses the label's own `gt_std` **0.938 → 0.046 rad/s**. **Escalated.**
>
> **5. ⛔ And the answer is still not "comma yaw works".** Under strict admissibility the deployed
> head reads comma yaw **R² −0.288** (CI **[−3.17, +0.29]**, spans zero), ρ **+0.211**,
> **nMedAE 2.36** (worse than predicting the median). **On its own held-out comma clips this head
> does not recover yaw — with a clean label.** That is a different statement from *"the label was
> broken"*, and it is the one the evidence supports here.

---

## 1. The choice, and why it is BOTH shapes rather than one

The brief offered two defensible shapes. **Neither is sufficient alone**, symmetrically — so the
implementation is both.

| shape | what it fixes | ⛔ what it leaves open |
|---|---|---|
| **(a) flip the default to the repair** | new builds stop getting broken labels — the live defect | a script whose job is to **reproduce a committed cache** keeps running, **silently**, on *repaired* labels. The two results differ **only by label protocol**, so nothing on the page distinguishes them |
| **(b) keep LEGACY reachable but never silent** | the deliberate case becomes loud and self-documenting | if LEGACY stays the **default**, a caller who passes nothing **still gets broken labels**. Making the default *raise* is not "reachable but not silent" — it is *unreachable*, and it breaks every existing caller |

**(a)'s residual risk is the program's most-logged failure class**, not a hypothetical: C29 itself; the
`heldout`/`full_set` blast radius (**27 arms, −6.67 % to +11.69 %, bidirectional**, one sign flip);
the `observed_frac` correction. Each is *the same number under a different protocol, with the
protocol invisible*. **This document is itself the proof that (a) alone is not enough** — §5's
number differs from the anchor by **protocol and substrate only**, and would have been silently
mistaken for it.

**⇒ Implemented: (a) AND (b).**

1. `DEFAULT_HEADING_MODE = HEADING_MODE_HOLD`. Silence is now **safe**.
2. `HEADING_MODE_LEGACY` requires `allow_legacy=True` **and** a non-empty written `reason` — the
   `models.vision_rank.resolve_vision_rank` discipline, copied deliberately: *a boolean can be
   flipped absent-mindedly, a sentence cannot.* The module ships `LEGACY_HEADING_REASON` so a
   legitimate reproduction need not invent wording, but passing it is still an **explicit act at the
   call site** and it lands in the run log.
3. The **cache key** separates the regimes (`label_params`, the `physicalai.label_params`
   construction). Without this, (a) actively *fails*: a repaired build would load — or overwrite —
   the existing legacy-keyed dir.

**Why the opposite default from `physicalai.WHEELBASE_MODE` is admissible.** PhysicalAI's legacy
regime must stay the default because it *is* the parity key (`physicalai-train-e438721ae894`, 2376
eps, skip-hash `f09e44db`). **comma2k19 is an explicitly NON-PARITY corpus** — `data/parity.py`
classifies it as *"unregistered corpus (comma / cosmos / OOD)"* and returns `parity=False`.
`test_wheelbase_regime.py` still pins `label_params() == {}` on the PhysicalAI side and is green.

### 1.1 What changed, by file

| file | change |
|---|---|
| `stack/tanitad/data/comma2k19.py` | `DEFAULT_HEADING_MODE` → `HEADING_MODE_HOLD`; new `LegacyHeadingRefused`, `LEGACY_HEADING_REASON`, `resolve_heading_mode()`, `label_params()`, `cache_build_params()`; `actions_and_poses` / `build_episode` / `Comma2k19Dataset` take `heading_mode=None` + `allow_legacy_heading` + `legacy_heading_reason` |
| `stack/tanitad/train/train_worldmodel.py` | the comma branch builds cache params through `cache_build_params` (not a bare dict), passes the resolved mode to `build_episode`, **prints the protocol into the run log**, and exposes `--comma-heading-mode` / `--comma-legacy-heading-reason`. `_build_datasets` / `train()` forward both; `mix` and `realmix` recursions forward them too |
| `stack/tanitad/lake/ingest.py` | 🔴 **found on a second probe.** `Comma2k19Ingestor.build_params` had **no label fragment**, and `build_params_hash` is written into **every lake record and exported to HF** — so a lake built after the flip would have carried the **same hash** as one built before while its comma yaw labels differed. Now goes through `cache_build_params`, with `heading_mode` / `legacy_heading_reason` fields |
| `stack/tests/test_comma_heading_regime.py` | **new, 23 tests** — the flip, the guard firing, the cache-key separation, the reproducibility pin, the trainer end-to-end |
| `stack/tests/test_comma2k19.py` | `test_heading_mode_default_is_byte_identical` → `test_heading_mode_default_is_the_repair` (its old assertion — *"the default equals legacy"* — is now false by design) |

`Comma2k19Dataset` resolves **before decoding anything**: a refused legacy request dies in the first
millisecond, not after 40 minutes of video decode
(`test_dataset_refuses_legacy_BEFORE_decoding_anything`: `assert calls == []`).

---

## 2. ⭐ The guard, DEMONSTRATED FIRING

⛔ **A guard that cannot fire is worse than none** (class **C13**; several have shipped here). Every
refusal is exercised **on the input that must trigger it**, and read out of a JSON artifact produced
by *running the shipped code*, not from a test name.

**Raw:** `raw/heading_default_guard.json` · **producer:** `code/heading_default_demo.py`
**Evidence class:** `MEASURED (ours; dev box)`. **Tier:** instrument-grade — a synthetic fixture that
**reproduces** the measured corpus defect; it is not itself a corpus measurement.

### 2.1 The failing direction — the defect reproduced

Fixture: 6 standstill frames (`|v| = 0.01 m/s`, random direction — GNSS noise, not motion) then 24
frames at 10 m/s, `dt = 0.05 s`.

| quantity | `heading_repair` **OFF** | `heading_repair` **ON** (the new default) |
|---|---:|---:|
| max \|yaw_rate\| over the standstill run | **61.78 rad/s** (3 540 °/s) | **0.0** |
| physically possible? (\|ω\| ≤ 1.5 rad/s) | ⛔ **no** | ✅ yes |
| moving-part heading | — | **bit-identical** |
| speed channel | — | **bit-identical** |
| frames changed | — | **6**, of which **0** above `v_min` 0.5 |

**The repair repairs; it does not smooth.** Not one frame at or above the measured threshold moves.

### 2.2 Every accident mode and every half-acknowledgement

| what a caller does | result |
|---|---|
| `resolve_heading_mode("enu_velocity")` | ⛔ `LegacyHeadingRefused` |
| `…, allow_legacy=True` (flag, no reason) | ⛔ `LegacyHeadingRefused` |
| `…, allow_legacy=True, reason="   "` (blank) | ⛔ `LegacyHeadingRefused` |
| `…, reason="I want the old labels"` (reason, no flag) | ⛔ `LegacyHeadingRefused` |
| `cache_build_params(base, "enu_velocity")` — **the trainer's and the lake's own call** | ⛔ `LegacyHeadingRefused` |
| `actions_and_poses(…, heading_mode="enu_velocity")` | ⛔ `LegacyHeadingRefused` |
| a typo — `"enu_velocty"` | ⛔ `ValueError` (**not** quietly treated as legacy) |
| `None` / missing config key / unset default | ✅ resolves to **`enu_velocity_hold_v1`** |
| `allow_legacy=True` **and** a written reason | ✅ returns `enu_velocity` |

The refusal message carries the measured mechanism with it, so whoever hits it need not go find out
why — the `zeros_naive` priced-trap construction.

### 2.3 The trainer, end-to-end

`test_trainer_default_build_does_NOT_reuse_the_legacy_cache_dir` drives the **real**
`train_worldmodel._build_datasets` comma branch (decode mocked) twice against the **same** root and
asserts the two occupy **different** cache dirs, with the legacy dirs carrying the **unchanged
pre-flip key**. `test_trainer_refuses_legacy_without_a_written_reason` asserts the refusal.

---

## 3. ⭐ The deployed path still reproduces — pinned

| pin | assertion | result |
|---|---|---|
| **the label** | acknowledged legacy == an **INDEPENDENT reimplementation** of the pre-flip formula, at strides 1 and 2 | **bit-identical**, `max_abs_delta = 0.0` |
| **the cache dir** | `label_params(LEGACY) == {}` ⇒ params unperturbed ⇒ key unchanged | `4bce7a330c31` **before = after** |
| **the separation** | the repaired regime cannot collide | `79a902d541f6` **≠** `4bce7a330c31` |

The reimplementation is deliberate: calling the same code path on both sides would let one bug
satisfy the pin twice. `test_build_episode_legacy_path_reproduces_bit_identically` extends it through
`build_episode`, i.e. including the `n_stack` alignment.

⚠️ **What the pin does NOT claim.** It pins the *loader*. It does not re-verify any committed comma
number end-to-end — those came from pod-side pipelines that build their own windows. **UNVERIFIED.**

---

## 4. Suite

| suite | result | vs the brief's baseline |
|---|---|---|
| `stack/` `pytest -q` | **1557 passed, 12 skipped**, 2 warnings | baseline 1534/12 → **+23 = exactly the new file**, **zero new skips** |
| `taniteval/` `pytest -q` | **663 passed, 0 skipped** | ⚠️ the brief's baseline says 661. `taniteval/` is **byte-identical to HEAD** (`git status` clean) and this pass changed nothing in it — **HEAD `8ab5327` itself added `taniteval/tests/test_stack_guard.py` (+53 lines, +2 tests)**. Not attributable here; flagged rather than papered over |

Run with the project venv (`C:\Users\Admin\venvs\tanitad`); system `python` 3.14 has no pytest.

---

## 5. ⭐ Job 2 — `idm_head_v1` re-scored, per corpus

**Raw:** `raw/idm_head_v1_comma_rescore.json` · **producer:** `code/rescore_idm_head_v1_comma.py`
**Evidence class:** `MEASURED (ours; dev box RTX 4060)`. **Tier:** decision-grade for the comma
channel on this substrate. **Nothing retrained** — the persisted head run forward once, scored
against two label sets.

### 5.1 What was scored, and what was NOT

| | |
|---|---|
| **substrate** | the card's own `val_heldout_traindomain`, **comma part**: `cm_[40:70]` ≡ `ep_00040…ep_00069` of `comma2k19-val-61c46fca8f7f`, **30 clips / 4,140 windows** (`k=4`, `stride=2`, 9-frame windows) — **43.9 %** of the 9,420 |
| **artifacts, md5-pinned and verified at run time** | head `fa4462f0b898b036be729c790278b823` (card `/weights_md5`), encoder `b5f07d9e3dd2ca643949bc86832e6585` (card `/config/encoder/ckpt_md5`), 2,899,724 params = card |
| **disjointness** | A0 trained on `cm_[0:40]`; this is `cm_[40:70]`. `episode_id` sets are **disjoint** (asserted in the artifact) |
| ⛔ **NOT scored** | the rig-A/rig-B half (5,280 windows). `physicalai-train-e438721ae894` is not on this box — **and PhysicalAI is provably unaffected** (`n_pai_changed = 0`, R² bit-identical), so recomputing it would buy re-pooling, not knowledge |
| ⛔ **NOT re-issued** | the **pooled** `0.010433` / `0.11814`. Pooling across corpora of different label quality is what created C5 in the first place |

### 5.2 The measurement — comma2k19 only, per corpus, never pooled

**Protocol OFF:** `heading_repair` off, no `v_min` — **the card's own protocol**.
**Protocol ON:** `heading_repair` on, `v_min` **0.5 m/s** (`HEADING_OBSERVABLE_V_MPS`, MEASURED).
`v_min` is the repair's *observability* threshold, **not** a scoring gate — **0 windows dropped**,
both arms score the identical 4,140.

| statistic | **OFF** | **ON** | change | paired CI (B=2000, 30 eps) |
|---|---:|---:|---:|---|
| **R²** | **+0.000048** | **−0.000421** | −0.00047 | **[−2.277, +0.130] — NOT separated** |
| MAE | 0.228804 | 0.152695 | **−33.3 %** | **[−0.150, −0.012] — separated** |
| **medAE** | 0.0250254 | 0.0243787 | **−2.6 %** | [−0.0016, 0.0000] — not separated |
| **nMedAE** | 2.26228 | 2.49000 | **+10.1 % WORSE** | — |
| Spearman ρ | +0.198030 | +0.198744 | **+0.0007 (flat)** | — |

**⚠️ The honesty condition, restated against my own result.** The brief's condition (MAE −42.5 %,
medAE −1.1 %, nMedAE +8.0 % worse, ρ flat) **reproduces in shape** here — MAE −33.3 %, medAE −2.6 %,
nMedAE +10.1 % worse, ρ flat. What does **not** reproduce is the **R² lift**. A re-issue quoting only
R² would have overstated the anchor; a re-issue *pasting* the anchor would have published **+0.33
where the measurement is ≈ 0.**

**Controls** (all in the raw JSON): speed / steer / long_accel labels **bit-identical** between
protocols; **4,020 of 4,140** windows bit-identical (`max_abs_label_delta = 0.0`); on the **120**
changed windows the legacy label reaches **14.95 rad/s** while the head predicted **0.051** and GT
speed averages **0.104 m/s** — **C29 confirmed again on fresh data: the model was right and the
label was wrong.**

### 5.3 ⭐ Why R² does not move — one hypothesis tested and refuted, then the measured cause

**Refuted first (stated because a refuted hypothesis is evidence too):** *the survivors are
observability-BOUNDARY windows.* **0 of 84** survivors cross an anchor boundary. Dead.

**MEASURED cause: a wholly-stationary clip.**

| | |
|---|---:|
| clip | `cm_00045` |
| frames | 300 |
| **observable frames (v ≥ 0.5 m/s)** | **0** |
| v_max over the whole clip | **0.039 m/s** |
| its windows in the split | 138 |
| impossible labels (\|ω\| > 1.5) surviving the repair | **84 of 84 program-wide** |
| worst surviving label | **15.28 rad/s** |

`hold_heading_through_standstill` returns the heading **unchanged** when a segment has **no
observable frame at all** — deliberately, because inventing a heading would be worse
(`test_hold_heading_no_observable_frames_is_a_noop` has always pinned this). The consequence was
never followed through: **such a clip has no admissible heading label anywhere, and it was scored
anyway.** 84 impossible rows out of 4,140 hold the label's own `gt_std` at **0.938 rad/s** — against
**0.142** for PhysicalAI's clean yaw channel — and R² has an unbounded left tail.

**Raising `v_min` cannot fix it** (a stationary clip is stationary at every threshold), and the
sweep confirms it: **84–85 survivors at `v_min` 1.0 / 2.0 / 4.0 m/s**, including comma.ai's own
published `calib_challenge` threshold of 4 m/s. ⛔ *That sweep is a DIAGNOSTIC — each row is a
different protocol and none of them re-issues anything.*

### 5.4 ⚠️ DIAGNOSTIC — what the channel reads once the label is DEFINED

⛔ **A third protocol. NOT a re-issued number, NOT a "repaired ceiling", and it must never be pasted
over the card's figure** — it drops 6.2 % of the split and so is not comparable to it.

Protocol: repair ON (`v_min` 0.5) **and** the repair's own `observable` mask required at `t−1`, `t`,
`t+1` — i.e. score only windows whose centred heading difference is *defined*.

| | value |
|---|---:|
| windows kept / dropped | **3,882 / 258 (6.2 %)** |
| impossible labels remaining | **0** |
| label `gt_std` | **0.938 → 0.0460 rad/s** |
| **yaw R²** | **−0.288** · CI **[−3.170, +0.286]** — spans zero |
| MAE / medAE | 0.0338 / 0.0229 |
| **nMedAE** | **2.365** (⛔ worse than predicting the median) |
| Spearman ρ | **+0.211** |

**Read it honestly in both directions.** The label defect is real and now fully removed on this
split. And with a clean label the deployed head **still does not recover comma yaw** on its own
held-out clips: R² negative (though its interval spans zero, so *"negative"* is not itself
decision-grade), nMedAE above 1, ρ ≈ 0.21. ⚠️ **This is a statement about `idm_head_v1` on these 30
clips. It is NOT a repaired ceiling** — the label's own achievable ceiling was not recomputed and
**must stay unknown until measured**.

### 5.5 🔴 Why this does not simply contradict the anchor — and the open probe

The anchor (`+0.0114 → +0.3308`) is **not on this substrate**, and the difference is larger than
"different windows":

| | the anchor (v3) | **this re-score** |
|---|---|---|
| comma cache | `comma2k19-val-**76b6e94a97a1**` — **64 segments / 21 routes** | `comma2k19-val-**61c46fca8f7f**` — **90 episodes** |
| val selection | every 3rd tag, `cm_00000 … cm_00063` (22 eps) | `cm_[40:70]` (30 eps) |
| relation to A0's training clips | ⚠️ **UNKNOWN** | ✅ **disjoint** (`episode_id`-verified) |
| windows | 2,992 | 4,140 |

**⚠️ Two different comma caches with different segment counts ⇒ the `cm_*` tag indices are NOT
comparable, so I could not determine whether the anchor's val set overlaps A0's own 40 comma
TRAINING segments.** `HYPOTHESIS`, not a claim — but a consequential one, because A0's comma
training clips and the anchor's val clips are drawn from the **same 21 routes of `raw_data/Chunk_1`**.
**The discriminating probe is one command** and I have prepared my half of it: the artifact carries
the `episode_id` of all 40 A0-train and all 30 held-out comma clips (a **stable sha1 of
route/segment**, therefore cache-independent). Intersect them with the `episode_id` of the 22 v3 val
tags recorded in `/root/idm2/manifest.json` (eval pod). **Non-empty ⇒ the anchor is partly in-train.**
**ESCALATED — see §8.**

> ✅ 🔴 **RESOLVED 2026-07-27 by the `anchor-settlement` pass — NON-EMPTY. THE ANCHOR IS PARTLY
> IN-TRAIN.** *(This section keeps its text and its date.)* It was settled **by content**, not by
> `episode_id`: sha256 of the raw `poses` float32 bytes **and** of the raw
> `frames_u8 [300,9,256,256]` sensor bytes, on both hosts, with 6 hash families agreeing.
> **Overlap = 2 of 22.** `76b:ep_00018 ≡ 61c:ep_00008` and `76b:ep_00039 ≡ 61c:ep_00020`, both in
> A0's `cm_[0:40]` TRAINING set. Without them the deployed head's repaired comma yaw R² is
> **−0.746** (CI [−1.574, −0.177]) instead of **+0.3308** — ⛔ **the anchor is WITHDRAWN.**
> ⭐ **And this section's own substrate is CLEAN, verified the same way:** A0-train × A0-heldout
> inside `61c` = **0** by pose bytes *and* frame bytes. The `episode_id`-disjointness asserted in
> §5.1 was a NAME claim; it now has a content one behind it.
> ⚠️ §5.6's *"the gap is in the direction §5.5 predicts"* (comma speed R² +0.554 here vs +0.759 for
> the anchor's set) is consistent with the confirmed leak, but it remains a plausibility band —
> `UNVERIFIED`, not evidence.
> Record: `…/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md` §2, raw
> `raw/anchor_overlap.json`.

### 5.6 Fidelity of the re-encode

The latents were re-encoded here rather than read from the pod's `lat_flagshipv1`.

- **Same encoder, byte-verified** — md5 `b5f07d9e…` matches the card, checked at run time.
- **Same code path** — `run_idm_proof.load_encoder` / `encode_frames` verbatim, the functions
  `idm2_encode.py` also uses, whose docstring records `idmval_zcmp` **cosine 1.0000** against the
  latents `idm_head_v1` was trained on (`INHERITED`, not re-verified here).
- **Behavioural check:** comma-only speed R² reads **+0.554** here vs **+0.759** for the anchor's
  comma set. Different clips *and* a possibly-in-train comparator (§5.5), so this is a plausibility
  band, **not** an equality check — the gap is in the direction §5.5 predicts. `UNVERIFIED`.
- **The contrast is immune regardless:** OFF and ON use the **identical predictions**, so any
  latent-reproduction error cancels exactly in the OFF→ON delta, which is the load-bearing number.

---

## 6. Stale-pending: what this resolves, and what stays

**Resolved — the comma component of the highest-priority stale-pending substrate:**

| entry | status now |
|---|---|
| `idm_head_v1_card.json → /val/val_heldout_traindomain/r2/yaw_rate` = **0.010433** | ⛔ **still STALE-PENDING as a POOLED number** — the rig half was not recomputed and, per the per-corpus rule, the pooled figure should not be quoted at all. **Its comma component is now MEASURED**: +0.000048 (OFF) → −0.000421 (ON) on the same clips |
| same → `/mae/yaw_rate` = **0.11814** | same; comma component **0.228804 (OFF) → 0.152695 (ON)** |
| `RETRACTION_LOG.md` C5, *"R² 0.010 at scale (n=9,420)"* | pointer added; the **row is not rewritten** and the superseded value keeps its date |
| `COMMA_YAW_REISSUE.md` §4 row 1 (⭐ highest priority) + §8 escalations #1, #2 | annotated as **actioned**, with the outcome and this document's path |

**Explicitly NOT resolved — left stale-pending, not inferred** (the other 6 substrate groups of the
46, unchanged): `2026-07-22-idm-proof` (14 values + 3 run logs), `2026-07-22-own-dynamics-encoder`,
`2026-07-24-branchb-transfer-eval/v1-encoder-char`, `2026-07-24-idm-pipeline-derisk`,
`2026-07-26-idm-v2` arms B1/V2/V3/V3wB, and the `IDM_DIAGNOSIS` ceiling family.
⛔ **The `0.352` ceiling was NOT recomputed and no repaired ceiling was derived.** ⛔ **No PhysicalAI
number was re-issued.** ⛔ **`PRE_REGISTRATION_IDMV2.md` was not opened.** ⛔ **`Mission Plan.md` was
not opened. Nothing was pushed to HF.**

⚠️ **And my own number is now a stale-pending risk for someone else.** It is comma-only, on
`61c46fca8f7f` `cm_[40:70]`, `k=4`, `stride=2`, protocol as stated in §5.2. **It is not a substitute
for any other substrate**, which is the exact rule that produced the 46-entry list.

---

## 7. 🔴 Escalations — these need an owner, not a note in a file

> ✅ **#1, #2 and #3 were ACTIONED 2026-07-27 by the `anchor-settlement` pass**
> (`…/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`). In one line each:
> **#1** admissibility is now expressed in code — `yaw_rate_from_heading` /
> `heading_admissible_centers` / `assert_yaw_rate_admissible` in `comma2k19.py`, default `"nan"` so
> a silent number is impossible, `"keep"` needs a flag **and a written reason**, **19 tests**; the
> episode contract was **not** widened, because `observable` is a pure function of `poses[:,3]` and
> nothing needed storing.
> **#2** answered, **NON-EMPTY** — 2 of 22, by content; `+0.3308` **WITHDRAWN** (§5.5 above).
> **#3** qualified as this section asked, and *quantified*: the lift stands at **+0.3038
> [+0.054, +0.479]** from a retrained head, not +0.3308/+0.679.
> **#4 and #5 are left OPEN** — neither was in that pass's scope.

1. **⭐ `hold_heading_through_standstill`'s `observable` mask is returned and NEVER USED.** It is the
   documented admissibility signal, and its absence is why 84 impossible labels survived into a
   published-grade score. **Decide whether the episode contract should carry admissibility**
   (`poses` is `[T,4]` today, with nowhere to put it) or whether every yaw-channel scorer must call
   the mask itself. Until then **every** comma yaw number, repaired or not, silently includes
   windows with no defined label. *I did not extend the contract — that is a design decision above
   this brief's scope, and half-building it would be worse than escalating it.*
2. **⭐ Is the v3 anchor partly IN-TRAIN?** §5.5. One command on the eval pod
   (`/root/idm2/manifest.json`), my half already staged. If non-empty, `+0.3308` needs a caveat and
   the `R0`/`V3F` arms trained on that split need re-reading.
3. **The "comma can test yaw" conclusion needs qualification.** `COMMA_YAW_REISSUE.md` §3.3 lifted
   the disqualification — correctly, the *mechanism* was right. But on A0's own held-out comma clips
   with a clean label the channel reads **R² −0.288 / ρ 0.211 / nMedAE 2.36**. **Testable ≠ working.**
4. **Wholly-stationary clips are a corpus-selection question, not only a label question.**
   `cm_00045` is 30 s of a parked car. Whether such clips belong in a driving corpus at all is a
   data decision nobody has taken.
5. **`taniteval` baseline drift** — the brief's 661 vs the measured 663 at HEAD (§4). Harmless, but
   the next brief will inherit the stale figure unless someone re-baselines.

---

## 8. Deliverable manifest

**STAGED, NEVER PUSHED.** No commit, no push, no branch switch by me.
⚠️ **The orchestrator committed mid-session** (`2cc2526`), sweeping the then-staged Job-1 files in
alongside a sibling's work; the remaining edits are staged in the working tree.
**Nothing lives in only one place.**

| # | artifact | where it lives | note |
|---|---|---|---|
| 1 | `HEADING_DEFAULT.md` | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-27-heading-default/` | this file |
| 2 | `code/heading_default_demo.py` | same dir | produces (3) by RUNNING the shipped guard |
| 3 | `raw/heading_default_guard.json` | same dir | every §2 number: the defect reproduced, every refusal, the cache keys, the reproducibility pin |
| 4 | `code/rescore_idm_head_v1_comma.py` | same dir | the re-score; md5-pinned, resumable, ~10 min on the dev box |
| 5 | `raw/idm_head_v1_comma_rescore.json` | same dir | every §5 number + controls + the mechanism + the leak-probe episode ids |
| 6 | `stack/tanitad/data/comma2k19.py` | `repo:` | ⭐ the default flip + the guard |
| 7 | `stack/tanitad/train/train_worldmodel.py` | `repo:` | cache-key wiring + CLI + run-log protocol line |
| 8 | `stack/tanitad/lake/ingest.py` | `repo:` | 🔴 second-probe find: `build_params_hash` carried no label regime |
| 9 | `stack/tests/test_comma_heading_regime.py` | `repo:` | 23 tests |
| 10 | `stack/tests/test_comma2k19.py` | `repo:` | the one assertion the flip invalidated |
| 11 | `…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json` | `repo:` | §6 per-number verdicts updated; **no existing value altered** |
| 12 | `…/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md` | `repo:` | §4 row 1 + §8 #1/#2 annotated as actioned |
| 13 | `Project Steering/RETRACTION_LOG.md` | `repo:` | C5 forward pointer; row not rewritten |

**Scratchpad, deliberately NOT staged:** `…/scratchpad/lat_cm40_69/` — 30 re-encoded latent files
(37 MB). Regenerable in ~90 s by (4) from artifacts that are themselves pinned by md5; staging
binary latents would add weight without adding reproducibility.
