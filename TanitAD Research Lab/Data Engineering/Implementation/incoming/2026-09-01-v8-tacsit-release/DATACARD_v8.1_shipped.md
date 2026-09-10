# tanitad-v7-training-corpus

**4,719 clips / 26.2 h** of front-camera driving with hierarchical labels
(strategic goal+action · tactical multi-label goals · lateral/longitudinal
actions · nav-command input) plus the FULL Alpamayo2-Super augmentation
(meta_action, CoT, auto-labeling, VQA, grounding boxes) and provider egomotion.
Corpus id `a48251e89c7a8603`. PRIVATE — research use within the TanitAD
programme; sources: nvidia/PhysicalAI-Autonomous-Vehicles (research license) +
Sayood/tanitad-alpamayo2-augmentation.

Released 2026-08-29 on PI authorization as the training baseline for
**v7f, refcv3, refav1, refd**. Ground-truth register row: `D-LABEL-GT`.

⛔ **PARITY.** This is a NEW, separately identified corpus. It does NOT
re-select the canonical parity corpus `physicalai-train-e438721ae894` (2,376
episodes, skip-hash `f09e44db`), which remains untouched and authoritative for
all cross-arm parity comparisons.

## Label releases in this repo

| release | files | what it adds |
|---|---|---|
| v7 | `labels/s2_labels_v7.jsonl.gz` | the original hierarchical release |
| v7.1 | `labels/s2_labels_v7.1.jsonl.gz` | see `V71_MANIFEST.json` |
| v7.2 | `labels/s2_labels_v7.2_{train,eval}.jsonl.gz` | see `V72_MANIFEST.json` |
| **v8.1** ⭐ **current** | `labels/s2_labels_v8_{train,eval}.jsonl.gz` | `tac_SIT` · `nav_30s` · `lane_change_text` · `speed_max_input` (+ `v_max_bucket`) — see `V8_MANIFEST.json` |

`schema_version` stays **`s2-geom-v7`** across all of them — the schema is the
contract and it did not change. The release rides in the `release` key.
**4,572 train + 147 eval = 4,719 records**, one per clip.

## v8 — the four added blocks, and how each may be used

⛔ **Read this section before training on any of them.** Each block carries its
own epistemics inside every record; nothing here is a silent default.

### `tac_SIT` — causally-cited situation

Three-valued, from `alpamayo.chain_of_causation`, available on **100 %** of
records. **1,585 clips carry at least one citation.** Top terms:
intersection 444 · parked_car 350 · pedestrian 255 · oncoming_vehicle 212 ·
crosswalk 193 · pedestrian_in_crosswalk 179 · roundabout 163 · cyclist 85 ·
roadwork 50.

⛔ **The values are CITED / NOT_CITED — never present/absent.**
`chain_of_causation` states the JUSTIFICATION for the action, so it names only
what caused the behaviour. A clip with pedestrians who did not affect the plan
carries no pedestrian citation. Absence is not evidence of absence.

The false-negative rate is **UNMEASURED and not boundable from this field**;
for scale, `highway` is cited on 2 clips here and appears on 210 clips in the
other Alpamayo fields. `emergency_vehicle`, `highway` and `blocking_obstacles`
have **insufficient support**. Yield demand is emitted under `demand`, not
`cited` — it is the ACTION those situations demand, so do not one-hot it
against them. Every citation ships the sentence that produced it, so a label
can be audited after the fact. **Precision is UNMEASURED** — spot-check ~50
clips per class before training.

### `nav_30s` — every nav manoeuvre within 30 s

**ADDITIVE, and it does NOT replace `nav_command`**, which is a live model
input that `nav_conditioning.py` refuses a batch without. The vocabulary is
**FROZEN and identical** to the scalar's: `NAV_FOLLOW_ROAD`, `NAV_TURN_L`,
`NAV_TURN_R` — richer structure, same alphabet. 5,418 entries:
2,926 FOLLOW_ROAD / 1,200 TURN_L / 1,292 TURN_R. The horizon edge is
`t_start_s <= 30.0`, INCLUSIVE.

* **Times** (`t_start_s`, `t_end_s`) are **ANCHOR-RELATIVE**: 0 is the s2
  anchor, which sits at `RAW_T0_S = 8.0 s` on the raw recording timeline. This
  matches every other time field in `s2-geom-v7`.
* **Distances** are **ARC LENGTH along the driven path from the anchor**.
  ⛔ NOT Euclidean — over 30 s round a bend they differ by tens of metres.
  ⛔ NOT from the recording start; that was the v8.0 defect corrected in v8.1
  below. `distance_m` is populated on **5,418 / 5,418** entries;
  `distance_end_m` is **null on 535**, honestly, where the ego track does not
  extend to a manoeuvre ending near the horizon.
* ⛔ **Provenance is `ego-future` on EVERY entry, unconditionally** — this is
  an ORACLE input. `allow_oracle_nav` is BINARY and cannot distinguish one
  token from this list, so a consumer MUST record
  `oracle_nav_payload: nav_30s` plus `n_entries`, or two arms with very
  different amounts of oracle at the input become indistinguishable in their
  own artifacts.
* A contested turn (obstacle pass) commands NO turn (PI 2026-08-29), so such
  clips fall through to `NAV_FOLLOW_ROAD`, consistent with the scalar.

### `lane_change_text` — text-only, no geometry

The source is `alpamayo.chain_of_causation` **ONLY**. `geometry_used` is
`false` on every record, by PI direction 2026-09-07, after three geometry
detectors scored 53.2 / 62.1 / 61.0 % side-agreement against the text
(chance = 50 %) — a disagreement whose root cause was the same `RAW_T0_S`
offset described below.

| status | n | meaning |
|---|---|---|
| EXECUTED | **105** (L 74 / R 31) | a lane change happened; `side` is the label |
| ANTICIPATED | 57 | text describes preparing for a later one |
| NEGATED | 15 | text says it did NOT / could not happen |
| MULTI | 1 | two manoeuvres in one sentence; `n_sides` = 2, do not halve |
| NO_SIDE | 1 | asserted but no side word near the phrase |
| NOT_CITED | 4,540 | not mentioned — ⛔ NOT evidence of absence |

⛔ **There is no timestamp, by construction.** `chain_of_causation` carries a
time expression on 6 of 4,729 clips (0.13 %). **Precision is UNMEASURED** —
spot-check against camera frames before training.

### `speed_max_input` — an INPUT CHANNEL, not a label

On **100 %** of records. PI 2026-09-01: *"at inference this data will be
provided as input by the user like the nav command, thus these are not training
labels, just input data."* Units are **m/s**, declared; median 10.07, mean
11.46. Required controls: **`hold`** (freeze at its t0 value — does the arm
track a moving input?) and **`shuffled`** (serve another clip's value — does
the model USE the channel at all?).

⚠️ **TRAIN/DEPLOY MISMATCH, and it is not a leak.** The TRAINING value is
`max(ego REALISED speed)` over `[anchor+2s, anchor+6s]`. At deployment the user
supplies a LIMIT — a bound the driver may not reach, versus what the ego did
reach. `provenance: ego-future` and `oracle: true` are stamped for the reason
they exist on `nav_command`: an arm trained on the ego-derived value stays
distinguishable from a later arm trained on a genuine speed-limit channel.

**`v_max_bucket`** snaps UP to the lowest posted limit ≥ `v_max_ms`, on a
ladder pinned from road law (24 EU countries + US) and **not fitted**:
20 · 30 · 50 · 70 · 80 · 100 · 120 · 130 km/h.
Histogram: 832 / 958 / 1,645 / 645 / 170 / 238 / 144 / 87.
`v_max_bucket_kmh` is the CANONICAL exact integer; `v_max_bucket_ms` is DERIVED
and kept full-precision so that re-snapping the shipped value is
**idempotent**, asserted at build. A 4-decimal rounding once moved 57.5 % of
the corpus one step up.

⛔ **Bucketing is COSMETIC as a leak fix and REAL as a semantics fix.**
MEASURED 5-fold out-of-fold: `v_hi ← v0` R² 0.8789; `v_hi ← (bin, v0)` R²
0.9702. The bin still carries **75.4 %** of the future information `v0` lacks.
What it does fix: the raw value's `frac_over` is 0.000000 on 0 clips — it IS
the max of the speed being scored, so it can never fire — while the quantized
value's is 0.005564 on 28 clips.

⚠️ **INTERSECTION ARTEFACT, and it runs opposite to the mismatch above.**
Snapping UP from a stopped ego reports the LOWEST posted limit: **75 % of
intersection clips get ≤ 30 km/h where a real map would say 50.** The channel
therefore risks teaching *"slow ego ⇒ low limit"*. This is not fixable inside
the design; it needs a real map channel.

### Traffic lights — teacher signals, not ground truth

`g_tac.goals` carries **805** traffic-light emissions with the colour attached:
RED 384 · GREEN 378 · YELLOW 25 · colourless 18.

⛔ **These are ALPAMAYO-DERIVED TEACHER SIGNALS. Downstream text must never
call them "GT traffic light".** PI 2026-09-07: *"no vlm, we stick to the
alpamayo labels as teacher signals"* — so the per-instance colour check is
CLOSED, not pending, and no second model will grade them. What IS established
is MECHANISM, not correctness: 182 of 805 emissions carry an independent 2D box
and 0 are contradicted, but 601 were never checked; grounding boxes are
label-only and **0 carry a colour attribute**, so colour exists ONLY in the
model's text.

⛔ **The consequence is load-bearing. Because these labels are accepted
UNVERIFIED, the EGO-ONLY CONTROL on any traffic-light head is no longer
optional — it is the only remaining guard.** Score any such head against a head
given `v0` and the speed trace alone. If they match, the head learned the ego's
deceleration, not the lamp.

## ⛔ v8.1 — a correction, and the mechanism that hid it

The v8.0 build shipped 2026-09-07 measured `nav_30s` arc distances on the **RAW
recording timeline** while `t_start_s` is **ANCHOR-RELATIVE** — an 8.0 s
offset. MEASURED over 2,485 turn entries: **81.8 % were wrong by more than 1 m,
54.6 % by more than 10 m, worst case 169.2 m.** `NAV_FOLLOW_ROAD` was
unaffected (exact 0.0). Times were always correct and are untouched.

⭐ **Why it survived its own build gate, which is the part worth carrying
forward:** the corpus-wide median error is **+0.0 m**, because
`NAV_FOLLOW_ROAD` is 54 % of entries and carries a hardcoded zero. The error
hid behind a pile of correct zeros — a summary statistic reporting health for a
field that was wrong on four fifths of the rows that actually use it.

Superseded v8.0 hashes: train `bb54dfa0…`, eval `920a9fcb…`.

## Read this before training

Every consumer MUST apply the five `trainer_conditions_mandatory` in
`MANIFEST.json`. The labels carry their own epistemics per token —
`provenance`, `disputed`, `grounded`, `corroboration`, `time_basis`,
`t_nominal_s`, and per-clip `turn_suppression` — so a trainer chooses what to
trust explicitly rather than inheriting silent defaults.

## Camera — SHIPPED IN THIS DATASET

⚠️ **Corrected 2026-09-09: an earlier version of this card said the camera was
"not shipped (~47 GB)" and described range-reads against the upstream NVIDIA
repo as the only route. That has been false since the PI directed the frames to
ship inside the private dataset** — `MANIFEST.json:camera.status` has read
`SHIPPED_IN_DATASET` throughout, and the card disagreed with it.

Frames are at **`camera/<clip_id>.mp4`**, 4,719 files, feature
`camera_front_wide_120fov`. `tools/pull_camera.py` and
`tools/pull_egomotion_range.py` are RETAINED for provenance and reproduction —
they range-read the member `{clip_id}.camera_front_wide_120fov.mp4` out of the
2.05 GB upstream chunk zips, fetching the zip central directory from the final
256 KB — but they are no longer the primary path. Per-file sha256 lives in
`camera/camera_sha256.json`.

## Cropping — per-clip cy is MANDATORY

`index/front_wide_cy.parquet` carries `clip_id, width, height, cx, cy, rig`.
TWO rigs exist (cy ~543 rig A / cy ~755 rig B) and **B is the majority, 2,723
of 4,719**; a geometric-centre crop is ~215 px wrong for rig B. Every epcache
build from these mp4s MUST crop around the clip's own cy, never the frame
centre.

## Projection — state it before using any camera formula

The corpus is **cylindrical** `256×640` at `f_ref` 305.577, where the column is
**LINEAR IN AZIMUTH**: `az_max = (W/2)/f_ref` gives the rig's true **120°**,
matching its own name `camera_front_wide_120fov`. ⛔ The pinhole formula
`2·atan((W/2)/f)` yields **92.6°** here and looks entirely plausible.

## Verify

Every file carries a sha256 in `MANIFEST.json`; the v8 files carry theirs in
`V8_MANIFEST.json:files`. ⛔ `V8_MANIFEST.json` also carries a
`files.build_chain` listing per-stage INTERMEDIATE hashes — those describe
files that no longer exist and must not be used to verify a download. The
`files.{train,eval}` block is authoritative. Labels: 4,719 rows, 0 exclusion
violations, schema `s2-geom-v7`.
