# E1's held-out split is CONTENT-CLEAN at the pixel level — the closed-loop chain stands

**MEASURED 2026-07-28**, pod3 (`1682186f1e9b`). Evidence class: **MEASURED (ours; sha256 of raw
bytes)**. Artifact: `pod3:/workspace/leakcheck/E1_heldout44_x_train.json`, summary committed here.

## 1. Why this was worth GPU-free pod time

**Every closed-loop number in the program** — E1a's horizon result (which produced retraction **C6**),
E1b, **E1c's BOUND verdict**, and **E1d's α-frontier** — is scored on
`physicalai-val-heldout-79d4e3d2d4c6`. That split had been verified **at the `poses` level only**;
**`frames_u8` was never checked.**

That is exactly the gap that has already bitten this program twice:
- **C50** — `idm_head_v1`'s card said *"held-out"*; content hashing found **80.0 % leaked**, 40 %
  literal train-on-test, and the clean counterpart moved ADE@2s **2.703 → 3.856 (+42.7 %)** with a
  `long_accel` **sign flip**.
- The sibling val cache `physicalai-val-f1b378f295ae` is **77.5 % leaked** and was for months
  described as *"episode-disjoint"*.

If this split were leaked, E1c's departure reduction and E1d's barrier finding would both be
unquotable. **A claim that would change a decision must be measured.**

## 2. Method — the same audited tool, not a new one

`fingerprint_pai_cache.py --mode full` (md5 **e2efccf5c460c79531adf0f8d779adb6**, byte-identical to
the committed copy that produced the existing train-side fingerprint) then `intersect_pai.py`.
Seven independent hash families over raw bytes, plus per-frame `sha1[:16]` digests supporting a
**shift-tolerant / partial** match — a val episode that is a *sub-window* of a train clip, which an
all-or-nothing whole-tensor hash cannot see.

- VAL: `physicalai-val-heldout-79d4e3d2d4c6`, **44 episodes**, mode `full`.
- TRAIN: `physicalai-train-e438721ae894`, **2,376 episodes**, mode `full` (pre-existing fingerprint
  from the same tool; **not** re-hashed — 260 GB avoided, and the numbers stay comparable).

**Pre-registered before the probe ran** (carried inside the artifact): overlap = 0 ⇒ CLEAN;
overlap > 0 ⇒ LEAKED, reporting count, identities and the fraction of each metric's mass — *not*
softened into "minor".

## 3. THE ANSWER — CLEAN

| family | overlap with train |
|---|---|
| `poses_sha256` | **0** |
| `poses_xy_sha256` / `poses_yaw_sha256` / `poses_v_sha256` | 0 / 0 / 0 |
| `actions_sha256` | 0 |
| **`frames_sha256`** (raw sensor bytes) | **0** |
| `maneuvers_sha256` | 10 ← see §5 |

**Partial containment: 0 of 44 val episodes share EVEN ONE FRAME with any train episode**
(`n_val_with_any_shared_frame: 0`, `max_frac_frames_in_train: 0.0`). So the split is not merely
free of whole-episode duplicates — no val episode is a sub-window of a train clip either.

## 4. The controls — the zero is a real zero, not a broken matcher

- **SELF** (val × val through the identical matcher, must return n=44): **44/44 on every family.** ✅
- ⭐ **SPIKE** (3 real val episodes injected into an in-memory copy of train; the matcher must find
  exactly 3 with the right identities): **found 3 by `poses_sha256` AND 3 by `frames_sha256`,
  identities recovered exactly.** ✅ **The matcher demonstrably detects a leak of this shape**, which
  is what makes the 0 above evidence rather than absence of evidence.
- **Within-cache duplicates:** 44 distinct on every family, 0 duplicate groups.

## 5. Two things NOT to gloss over

**(a) `maneuvers_sha256` = 10, and `all_content_families_agree` is `false`.** Ten val episodes share
a `maneuvers` hash with some train episode. This is **not** sensor contamination: `maneuvers` is a
`[T] int64` sequence over a handful of discrete classes, so two unrelated 199-step episodes of
mostly "keep lane" collide by construction. It carries **no** pixel or trajectory information. The
high-entropy families — `poses` (4 float32 × 199) and `frames` (9×256×256 uint8 × 199) — are both
**0**. Reported because the tool flagged a disagreement and a flag that is explained is not a flag
that is hidden.

**(b) `filename_overlap` = 44 — ALL 44 filenames also exist in train, while content overlap is 0.**
This is the program's known pattern (600/600 filename overlap with 0/600 real overlap). ⇒ **Names
are not evidence, in either direction.** Had we checked names we would have concluded a 100 % leak;
had we trusted `episode_id` (overlap 0) we would have been right by luck, since `episode_id` is a
name-derived integer written by the cache builder, not a function of the pixels.

## 6. Consequence

✅ **E1a (C6), E1b, E1c and E1d rest on a content-verified-clean held-out split.** The E1c departure
reduction (0.5877 → 0.147) and the E1d barrier finding are **not** leak artifacts.
✅ The registry's claim for this split is upgraded from an `episode_id` claim to **content-verified,
sensor-level, including partial containment**.

⚠️ **Scope, stated rather than implied:** this verifies `heldout-79d4e3d2d4c6` against
`physicalai-train-e438721ae894` **only**. It says nothing about any other split.

### 6.1 ⭐ The C49-class gap is CLOSED, not merely flagged *(added 2026-07-28, same day)*

The paragraph above originally ended *"…and nothing about REF-C base's own pre-training corpus,
whose overlap with this split remains **unmeasured** — the same gap that made C49 a retraction."*
**That caveat was correct to raise and is now discharged by measurement rather than by argument.**

`refc-diffusion-base-v21-30k/config.json` records its training data as:

```
data: {'cache_dir': '/workspace/pai_epcache/physicalai-train-e438721ae894',
       'n_episodes': 2376, 'n_windows': 406099}
```

That is **the identical path and the identical 2,376 episodes** this probe intersected against —
`sets.TRAIN.path` in the artifact reads `/workspace/pai_epcache/physicalai-train-e438721ae894`,
character-for-character the same string, with the same `n = 2376`.

⇒ **REF-C base's own pre-training corpus IS the corpus measured here, and its content overlap with
the held-out 44 is 0 — including 0 shared frames.** So the E1 chain's guarantee is the *strong* one
C49 demands: not "clean against some parity split", but **clean against the base arm's actual
training data**, verified at the sensor level.

⚠️ **What C49 warned about still applies elsewhere.** C49's lesson is that symmetry of a confound is
an empirical claim; here the claim was checked and the confound is absent *for this arm on this
split*. It does **not** transfer to any other arm or split — REF-A, REF-B and the flagship arms were
trained on the same parity corpus but are evaluated on **different** splits, each of which needs its
own content check before its "held-out" label is trusted.
