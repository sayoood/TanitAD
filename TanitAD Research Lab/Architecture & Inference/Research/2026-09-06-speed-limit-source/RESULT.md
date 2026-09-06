# SL-SRC-1 — a posted speed limit EXISTS in the Alpamayo augmentation on 57/4,729 clips, carries a NUMBER, and is EGO-COUPLED, UNGROUNDABLE and UNIT-LESS on a third of it

**Date** 2026-09-06 (Europe/Berlin) · **Stream** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **Evidence class MEASURED unless stamped
otherwise** · **GPU-days spent: 0. No model was trained, no arm was scored, the
A40 was not approached, and NOTHING was pulled from HuggingFace** (the source
parquet was already on local disk, so no HF quota was consumed — quota not read
because no pull was made).

**Answers the PI's MAX SPEED instruction: P1 (the untested VLM-CoT source),
P2 (extract vs supply), P3 (the honest fallback).**
**Builds on, and does not contradict, `…/Research/2026-09-06-constraints/`
(commit `a96f48d`).**

⛔ **NO EVAL TIER IS STAMPED ON ANY NUMBER BELOW, AND NO FOUR-FAMILY TABLE IS
REPORTED — DELIBERATELY.** This is a corpus/label-source audit. No model
produced a trajectory, so T0/T1 does not apply and ADE/longitudinal/lateral/
tactical/strategic have nothing to be computed over. Stamping a tier here would
be a category error. The four-family rule binds the arm that consumes this
label, and it is named in §5.

---

## 0. THE ANSWER, in seven lines

1. ⭐ **A POSTED SPEED LIMIT DOES EXIST IN THE CoT CHANNEL, WITH A VALUE.**
   **57 / 4,729** clips of `Sayood/tanitad-alpamayo2-augmentation` carry a
   speed-limit value the VLM asserts it read off a sign, unhedged and
   un-negated — e.g. *"A 70 km/h speed limit sign appears ahead within the
   first 2 seconds and sets the target speed for the ego vehicle."*
2. ⛔ **THE PIPELINE'S OWN DOCUMENTED YIELD IS A 5.7× UNDER-COUNT.**
   `cot_tokens_v7.py`'s docstring publishes `speed limit  41  0.9 %`. That is
   the `meta_action` task **alone** (measured here: `meta_action=41`, exactly).
   Across all five tasks the phrase appears on **235 / 4,729** clips.
3. ⛔ **IT IS A DEAD BOOLEAN.** `CotTokens.speed_limit` is `bool`, set at
   `cot_tokens_v7.py:238` and **read nowhere** — `goals_from_cot` emits no
   token for it. The *value* is discarded at extraction; only presence survives.
4. ⛔⛔ **GROUNDED SHARE IS 0 / 57.** The `grounding_via_vqa` box census over
   4,728 rows contains **no sign class of any kind** (Car, Pedestrian, traffic
   light, Truck, Bike-with-rider, Bus, lead vehicle, Motorcycle-with-rider) and
   **0 of its questions mention a sign**. Not one reading can be corroborated in
   image space. *(Precedent: traffic lights got 175 grounded / 604 disputed.
   Here it is 0 grounded / 57 disputed.)*
5. ⛔ **31.6 % OF THE READINGS CARRY NO UNIT** — 18/57 state a bare number
   (*"a 70 speed limit sign"*). 70 km/h = 19.44 m/s, 70 mph = 31.29 m/s, a
   **1.61×** spread. Under the units rule those 18 are inadmissible as they
   stand.
6. ⛔⛔ **AND THE READINGS ARE EGO-COUPLED, WITH THE COUPLING SHARPEST AT THE
   ANCHOR FRAME.** Median `|limit − ego|` is **2.10 m/s** against **6.03**
   (constant control) and **8.12** (shuffled control); median ratio
   `limit/ego` = **1.01**. A temporal sweep puts the global minimum at
   **t = 5.0 s** — the Alpamayo anchor is **5.1 s**. This is C87's
   dashboard-roundel failure mode's signature, at corpus scale.
7. ⇒ **VERDICT: there is a real signal, and it is NOT an admissible regulatory
   prior today.** It is 1.21 % coverage, ungroundable, partly unit-less, and
   statistically entangled with the ego's own speed. §5 puts extract-vs-supply
   to the PI; §6 costs the three routes that could make it admissible.

---

## 1. P1.1 — THE CoT's ACTUAL ENUMERATED FIELDS (⛔ not assumed — read from the artifact)

**Corpus:** `C:/Users/Admin/tanitad-data/alpamayo/records.parquet`,
25,970,018 B, md5 **`9f13474723b880eec7fcc09a7be478d8`** (independently
confirmed by a second agent against the HF snapshot
`cedbf57c124149310a9ecd661ef42bb73cec34d5`, byte-identical).
**23,644 rows over 4,729 distinct clips.** Hard-coded at
`alpamayo_records.py:54`.

### 1.1 Parquet columns — 12, measured

`clip_id` · `t0_us` · `task` · `model_id` · `quantisation` · `seed` ·
`wall_s` · `error` (non-null **0**) · `vqa_qid` · `vqa_category` ·
`question` · `raw_json`

⚠️ `model_id` = **`nvidia/Alpamayo2-Super`**, `quantisation` =
**`NF4-backbone-4bit-UNVALIDATED`**, `seed` = 42, `t0_us` = **5,100,000 on
every row** (one anchor per clip, and it is **not** our 8.0 s s2 anchor).

### 1.2 The five tasks

| task | rows | what it carries |
|---|---|---|
| `meta_action` | 4,729 | 3-axis action + the CoT sentence |
| `trajectory` | 4,729 | Alpamayo's own predicted path + a rendered CoC string |
| `auto_labeling` | 4,729 | `chain_of_causation`, `critical_components_analysis`, `ego_vehicle_motion_analysis`, `trajectory_analysis` |
| `vqa` | 4,729 | **one** sampled Q/A per clip, from a **506-question bank over 21 categories** |
| `grounding_via_vqa` | 4,728 | one 2D-box question per clip |

### 1.3 `raw_json` inner keys

`raw_outputs` · `cot` · `meta_action` · `answer` · `box` · `cot_auto_labeling`
(the `trajectory` task additionally carries `figure_style`, `clip_id`,
`t0_us`, `camera_tmin`, `camera_indices`, `input_profile`).

### 1.4 What our extractor keeps — `CotTokens`, 11 fields

`traffic_light` (RED|YELLOW|GREEN|UNKNOWN) · `oncoming` · `yield_` (SIGN|HAZARD) ·
`merge` · `lane_change` (L|R) · `gap` · `exit_side` (LEFT|RIGHT|UNKNOWN) ·
`overtake` · `evade_obj` (PARKED|CYCLIST|PEDESTRIAN|DOOR|ONCOMING) ·
**`speed_limit` (bool)** · `evidence` (the verbatim sentence).

⛔ **`speed_limit` occurs at exactly two lines of `cot_tokens_v7.py` — 183
(declaration) and 238 (assignment). It is never read.** `goals_from_cot`
emits tokens for eight of the other nine fields and **none** for this one.
⭐ **The value was available and was thrown away at the regex**:
`_SPEED_LIMIT = re.compile(r"\bspeed limit\b")` captures no group.

### 1.5 ⭐ The VQA bank ASKS about speed limits — four dedicated questions

| question | clips asked | answered with a digit |
|---|---|---|
| *"Is there a speed limit sign applicable to the ego lane within 100 metres?"* | 16 | 0 (**yes=0, no=16**) |
| *"What does the nearest speed limit sign say or indicate?"* | 9 | **0** |
| *"What is the state or condition of the speed limit sign nearest the ego…?"* | 9 | 0 |
| *"Are there speed limit signs visible in the scene, and where?"* | 6 | 0 (yes=1, no=5) |

⛔⛔ **ASKED DIRECTLY WHAT THE SIGN SAYS, ALPAMAYO RETURNS A NUMBER 0 TIMES OUT
OF 9.** Every numeric reading in this report comes from the **unprompted** CoT
(`meta_action` / `auto_labeling` / `trajectory`), never from the question that
was designed to elicit it. Any plan to "just ask the VLM for the limit" must
confront that asymmetry first — the direct-question arm is a **measured 0/9**,
not an untried idea.

⚠️ A neighbouring family, `"What is a safe maximum speed for this <road-type>
under current conditions?"`, is asked on **107 clips**. ⛔ Its answers are the
**road-type language prior** (*"residential areas typically have 25–30 mph"*) —
the `road_class` circularity the sibling retracted, in VLM costume. **Excluded
from every count below.**

---

## 2. P1.2 — COVERAGE, AS A FRACTION, WITH THE CORPUS NAMED

**Denominator throughout: 4,729 distinct clips in `records.parquet`.**
⚠️ **This is NOT the parity corpus** (`physicalai-train-e438721ae894`, 2,376
episodes) and **not** B1's 4,572. Three different denominators are live in this
programme; every fraction here is against 4,729 and must be re-derived before
being quoted against another.

| class | clips | of 4,729 |
|---|---|---|
| phrase *"speed limit"* anywhere in any task | **235** | 4.97 % |
| ⭐ **READ off a sign, unhedged, un-negated, WITH A VALUE** | **57** | **1.21 %** |
| HEDGED — language prior (*"typically"*, *"would likely"*) | 66 | 1.40 % |
| NEGATED (*"the speed limit sign is not visible"*) | 26 | 0.55 % |
| *"posted"* | 61 | 1.29 % |
| a number with explicit `km/h`/`kph` | 73 | 1.54 % |
| a number with explicit `mph` | 14 | 0.30 % |
| school-zone mentions | 52 | 1.10 % |

**Rows contributing a READ, by task:** `auto_labeling` 24, `vqa` 23,
`trajectory` 15, `meta_action` 15 (77 clip-rows over 57 distinct clips —
the tasks corroborate each other on the same clip).

**Units on the 57:** `km/h` **37** · `mph` **2** · ⛔ **none stated 18 (31.6 %)**.

⚠️ **CORRECTION, mine, same turn (2026-09-06).** This figure first read **35.1 %**
in the first committed version of this document. **35.1 % is 20/57 — the *suspicious*
count (value ≤ 15 **OR** no unit) — not the no-unit count.** The two extra rows
(`42f52617` 10 **mph**, `74808e88` 15 **km/h**) both state a unit, so the sentence
contradicted its own numerator; **18 was correct everywhere it appeared.**
⭐ **ROOT-CAUSE CLASS: a percentage carried over from a DIFFERENT query than its
numerator** — the *"a number carries its ARM and its ARTIFACT PATH, and PERCENTAGES
are the worst offenders"* rule in its narrowest form.
⛔ **It did NOT propagate**: a sibling (`…/Research/2026-09-06-cot-loader/`)
independently recomputed **31.6 %** from `raw/limits_classified.json` rather than
copying my prose. Re-verified against the artifact: unit census
`{NONE: 18, km/h: 37, mph: 2}`, 18/57 = 0.3158.

**Values (57/57 are legal posted-limit round numbers):**
10, 15, 20, 30, 35, 40, 50, 60, 70, 80, 100, 120 — with **30 the mode**.

### 2.1 ⛔⛔ GROUNDED SHARE = 0 / 57

The brief asks for it because supervising on disputed labels teaches noise.
**Measured over all 4,728 `grounding_via_vqa` rows:**

* box-label census: `Car` 9,172 · `Pedestrian` 4,316+200 · `traffic light`
  3,559+28 · `Truck` 712 · `Bike with rider` 588 · `Bus` 164 ·
  `lead vehicle` 48+8 · `Motorcycle with rider` 44 · `car` 8.
  ⛔ **There is no sign class at all.**
* **questions mentioning "sign": 0 of 4,728.**

⇒ **The grounding task can never corroborate a sign reading, because it was
never asked to.** Per `alpamayo_records.py`'s own rule — *"grounding can CONFIRM
a token, it can NEVER REFUTE one"* — the honest statement is not that the
readings are wrong, but that **0 of 57 are corroborable, and all 57 stay
`disputed` permanently under the current export.**

---

## 3. ⭐ THE ADMISSIBILITY MEASUREMENT — the readings are EGO-COUPLED

C87 released the SAM3 sign channel as a **presence flag only** and forbade
KIND and TEXT, because its two highest-scoring false positives were a
**dashboard `30` roundel (0.927)** — the ego speedometer — and a hoarding.
`30` is the **modal value** of Alpamayo's readings (21 of 77 clip-rows). So the
same failure mode was tested here directly, with controls.

**Setup:** the 57 READ clips, ego speed from
`labels/egomotion_alpamayo/*.parquet` at the Alpamayo anchor **5.1 s**
(`ego_echo_test.py`). Ego track present on **57/57**, 0 missing.

| arm | median `\|limit − ego\|` | median `limit/ego` | frac(limit < ego) |
|---|---|---|---|
| **readings, stated unit (km/h default)** | **2.10 m/s** | **1.01** | 0.439 |
| readings, all forced to mph | 7.15 m/s | 1.63 | 0.105 |
| ⛔ **CONTROL — constant 50 km/h for every clip** | **6.03 m/s** | 1.03 | 0.439 |
| ⛔ **CONTROL — the same values shuffled across clips** | **8.12 m/s** | 0.88 | 0.509 |

⇒ the readings sit **2.9×** closer to the ego's own speed than a constant, and
**3.9×** closer than a shuffle of themselves. The link is clip-specific and real.

### 3.1 The discriminating test — TEMPORAL SPECIFICITY

⚠️ **The coupling alone is confounded: drivers obey limits, so a CORRECT sign
reading would also sit near ego speed.** The two hypotheses were separated by
sweeping the comparison time across the whole clip. A speedometer read off one
frame must be sharpest **at that frame**; a posted limit governs the whole
stretch and has no reason to be.

| t (s) | 0.0 | 2.0 | 4.0 | **5.0** | 7.0 | 10.0 | 15.0 | 19.0 |
|---|---|---|---|---|---|---|---|---|
| median `\|limit − ego(t)\|` | 3.78 | 2.89 | 2.60 | **2.32** | 2.64 | 2.75 | 3.09 | 3.05 |

⭐ **The global minimum is at t = 5.0 s. The Alpamayo anchor is 5.1 s.** The
error rises monotonically away from it in both directions.

⚠️ **TWO HONEST QUALIFICATIONS, both binding on how this is quoted.**
1. **The minimum is SHALLOW** — the profile's whole range is **1.46 m/s**
   (3.78 → 2.32). This is a real minimum, not a cliff.
2. ⛔ **This does NOT prove the speedometer mechanism.** Alpamayo was shown
   frames *centred* on the anchor, so **any** correctly-described quantity
   would be sharpest there. The test discriminates
   *"anchored to the ego state at the observed frame"* from *"a road-wide
   constant"* — and the readings are the former. It does not discriminate
   *reading the dashboard* from *describing the scene accurately at t0*.

⇒ **What survives under BOTH readings, and is the admissibility-relevant
conclusion: the label is not an INDEPENDENT regulatory prior.** It co-varies
with the ego state at the frame it was generated from. That is enough to bind
§5, and it is as far as the evidence goes.

### 3.2 The arithmetic that is hardest to explain benignly

**25 of 57** clips claim a limit **below** the ego's speed at the anchor, i.e.
the ego would be speeding on 43.9 % of the clips where a limit was read. Two
are not survivable as sign readings:

* `09cf59f8` — claims **60 km/h** (16.67 m/s); ego is at **36.19 m/s ≈ 130 km/h**.
* `3d459173` — claims **70** (19.44 m/s as km/h); ego is at **34.88 m/s ≈ 126 km/h**.

⚠️ ⛔ **Both are ALSO consistent with a unit error rather than a hallucination**
(`3d459173` states no unit; at 70 mph = 31.3 m/s it is nearly right). **Which
one it is cannot be settled without the frames**, and it is exactly why §1.5's
unit gap is load-bearing rather than cosmetic. **I did not open the frames and
I do not claim to know.**

---

## 4. ⛔ WHAT IS *NOT* A SOURCE — checked so it is not re-proposed

* **SAM3 sign detections** — `presence` only; **KIND and TEXT forbidden by
  C87**, and a threshold *keeps* the harmful false positives (dashboard
  roundel 0.927 > true signs). The G1 sign-text gate is **CLOSED at 0/31**.
  `stack/tests/test_speed_band_derivation_blocker.py` pins this; I did not
  touch it.
* **`vlm_route_labels.py` / `vlm_semantic_labels.py`** — both already ship a
  `sign_type` enum containing `speed_limit` and a `sign_reads` slot with
  `{"value": <number you READ or null>, "unit": "kph|mph|none"}`. ⭐ **The
  instrument for option (a) is already WRITTEN.** Its own docstring records the
  48-clip pilot measuring the model **fabricating band edges on 48 % of clips**,
  which is why VTARGET was left kinematic. ⚠️ I found **no banked output** of
  either script carrying populated `sign_reads`. ⛔ **That absence is
  INCONCLUSIVE, not established.** A repo-wide `find` by name completed
  (exit 0) and returned no output directory — but that is **not evidence**:
  the script's own usage writes to `--out /workspace/vlm_route` **on pod3**,
  i.e. **off-repo by design**. The only in-repo trace is
  `Project Steering/G1_SIGN_OCR_GRADING_SHEET.md`. Settling it needs a pod
  probe, which I did not run.
* **`g_tac.goals.SPEED_BAND` / `a_tac.lon_args.v_target_ms`** — ego-future,
  already refuted by the sibling. Not revisited.
* **`strata.road_class`** — circular. Not revisited.
* **OSM / map-matching** — impossible, no lat/lon. Not re-probed.

---

## 5. ⭐⭐ P2 — THE DECISION FOR THE PI: EXTRACT vs SUPPLY

⛔ **I am not deciding this.** Both designs are laid out with their
consequences, as instructed.

**The two PI rulings that frame it, and they point opposite ways:**
* *"The Alpamayo labels and all vocabulary are GT labels **to train the models
  to extract them from vision combined with ego data**."* → (a).
* *"The nav command is NOT a training signal — it is an **INPUT** simulating the
  nav system of the vehicle."* → a posted limit is arguably the same kind of
  thing: a real signal a production car receives from its map/nav stack → (b).

### Option (a) — EXTRACT the limit from vision (a supervised sign-reading head)

| | |
|---|---|
| **What it is** | a vision head predicting `(limit_value, unit, applies_to_ego)`, supervised by the 57 CoT readings. Deployable: reads its own signs. |
| **Buys** | no inference-time input ⇒ **no leak is possible by construction**. Matches the "train the model to extract them from vision" ruling exactly. Directly answers *"reach max speed when the situation allows"* with a model-produced ceiling. |
| **Costs** | ⛔ **57 positive labels is not a training set** — 1.21 % coverage, and 0/57 grounded. ⛔ **The labels are ego-coupled (§3)**, so a head fit to them can score well by predicting ego speed — the `SPEED_BAND` defect reappearing one level up. ⛔ **18/57 carry no unit**, so a third of the regression targets are ambiguous by 1.61×. ⚠️ MEASURED precedent: the *linear* vision→target-speed readout is separated **worse** than repeating `v0` (0.2379 vs 0.4078). |
| **Mandatory control if chosen** | an **ego-speed-only arm** must be beaten. If a head given only `v0` predicts the "limit" as well as the vision head, the channel is an echo and the arm is dead — that control is not optional, it is the whole experiment. |

### Option (b) — SUPPLY the limit like the nav command

| | |
|---|---|
| **What it is** | `max_speed` becomes an input channel alongside `nav_cmd`, simulating a map/nav speed service. |
| **Buys** | realistic — production stacks do receive posted limits from the map. Sidesteps the 57-label problem entirely: the *input* need not be learned. Makes the under-driving side scoreable **immediately**. |
| **Costs** | ⛔⛔ **On THIS corpus the only available value is the ego-coupled CoT reading (ratio 1.01 to ego speed).** Supplying it is supplying a number that is, to within 1 %, the speed the car was already doing — **the nav-echo defect with no horizon guard**, exactly what killed `SPEED_BAND`. ⛔ **Coverage 1.21 % means 98.79 % of clips would need a fill value**, and whatever that value is becomes the actual signal. ⚠️ The route-input precedent: a supplied route is *optimistic by construction* here because our only supplier is the ego's own future. |
| **What would make it clean** | an **external limit source independent of the ego trace** (§6). With one, (b) becomes defensible; without one, (b) is an echo channel. |

⭐ **The one principle that survives either choice, and it is the sibling's:
the constraint is SCORED ON THE OUTPUT.** *"An input that does not exist cannot
echo."* Whichever way the PI rules, `frac_over` / `frac_under_when_allowed` are
computed from the model's own proposed trajectory against the ceiling —
`stack/tanitad/eval/constraints.py` already implements this and fires in both
directions.

⚠️ **My recommendation, offered as input and not as a decision:** neither (a)
nor (b) is fundable on 57 ego-coupled labels. **The cheapest thing that changes
the answer is §6.1 — a targeted re-ask that costs 10.8 h on hardware we own** —
and it also settles the 0/9 direct-question anomaly, which is itself the most
suspicious fact in this report.

---

## 6. P3 — THE COSTED ROUTES (⛔ no proxy was invented)

⭐ **Every cost below is priced from the CONSUMER'S OWN RECORD** — the
`wall_s` column of the artifact the pipeline actually produced — not from a
file found on disk. **MEASURED medians, s/clip:** `vqa` **8.2** ·
`trajectory` 9.5 · `meta_action` 10.0 · `grounding_via_vqa` 10.1 ·
`auto_labeling` 18.1. The full 23,644-row build cost **282,090 s = 78.4 h =
3.26 GPU-days**.

### 6.1 ⭐ CHEAPEST — re-ask Alpamayo with a speed-limit question on every clip

The `vqa` task already samples **one** question per clip from a 506-item bank.
Re-running it **pinned** to the speed-limit questions covers all 4,729.

* **Cost: 4,729 × 8.2 s = 38,778 s = 10.8 h** on one GPU (mean-based: 11.3 h).
  **No HF pull, no spend** — the model is `nvidia/Alpamayo2-Super`, weights
  already used for this corpus.
* **Also delivers the answer to the 0/9 anomaly**: is the empty direct-question
  answer a *prompt* failure or a *capability* failure? 9 samples cannot say;
  4,729 can.
* ⚠️ **Known risks, stated in advance:** the export is
  `NF4-backbone-4bit-UNVALIDATED`, and 4-bit quantisation is exactly what
  degrades small-glyph OCR. The 48-clip pilot already measured this model
  fabricating numbers on 48 % of clips.
* ⛔ **It cannot fix grounding** — that needs 6.2.

### 6.2 Add a sign class to the grounding pass

`grounding_via_vqa` asks **one** box question per clip and **none** is about a
sign. Adding *"Provide the 2D bounding box of the speed limit sign"* would make
readings corroborable in image space for the first time.

* **Cost: 4,729 × 10.1 s = 47,763 s = 13.3 h.** No spend.
* ⛔ **Buys CONFIRMATION only, never refutation** (`alpamayo_records.py`'s own
  sampled-bank rule). A missing box means the question failed, not that no sign
  exists.
* ⭐ **6.1 + 6.2 together = 24.1 h ≈ 1.0 GPU-day** and yield a
  *value + unit + box* triple — the first speed-limit label in this programme
  that could clear the `disputed` bar.

### 6.3 External corpus carrying posted limits

Mapillary Traffic Signs, BDD100K, and the German/Swedish/Belgian TSR sets carry
posted-limit annotations. ⛔ **Transfer to PhysicalAI cannot be verified**: our
clips have **no lat/lon** (settled at five probes), so a sign detector trained
elsewhere could be *applied* to our frames but its output could never be
checked against a map. **This is a detector-transfer route, not a label route**,
and it inherits every C87 failure mode — including the dashboard roundel.
⚠️ **Not costed.** Costing it honestly needs a licence read and a pilot, and
inventing a number here would be the estimate-without-a-consumer error.

### 6.4 Human annotation pass over our clips

* **Scope for a usable set:** the 235 phrase-bearing clips are the efficient
  stratum (4.97 %); a random draw would be ~1 % positive.
* ⛔ **Unpriced — it is a PI spend decision, not an engineering estimate**, and
  a rate invented here would be exactly the class of number this programme
  retracts. **What it uniquely buys** is the only route in this list producing
  labels that are *not* generated by a model with a documented fabrication rate.

### 6.5 ⭐ What each route unblocks — the under-driving side

MEASURED by the sibling: GT's own under-rate **never falls below 0.51** across
the whole threshold grid, and reaches **0.9987** on no-lead steps, which are
**54.4 %** of all steps. ⇒ **Today only the OVER-driving half of the PI's
instruction is scoreable.** 6.1/6.2 make the under-driving half scoreable on
the covered stratum; 6.3/6.4 make it scoreable corpus-wide. **No modelling
change can substitute — it is a dataset extension, as the sibling concluded.**

---

## 7. 🔴 ESCALATIONS

1. **→ PI (decision).** §5: extract (a) vs supply (b). ⭐ **And the prior
   question: authorise the 10.8 h re-ask (6.1), or 24.1 h for 6.1+6.2?** It
   runs on hardware we own, needs no HF pull and no spend, and it is the only
   cheap experiment that can move this from 57 ungrounded labels to a real one.
   ⛔ I did not start it — it needs GPU the PI has reserved.
2. **→ v7 label owners (DataFlyWheel).** `cot_tokens_v7.py`'s docstring
   publishes `speed limit 41 0.9 %` as a corpus-wide yield; it is the
   `meta_action` task alone and the corpus figure is **235/4,729**. Same class
   as the stale feature-count the programme pinned a test for. ⛔ I did not edit
   the file — a sibling may own it this session.
3. **→ v7 label owners.** `CotTokens.speed_limit` discards the value at the
   regex. A capturing group would have banked all 57 numbers at zero cost.
4. **→ whoever owns `alpamayo_records.py`.** `_load()` returns `{}` **silently**
   when `C:/Users/Admin/tanitad-data/alpamayo/records.parquet` is absent
   (`if not os.path.exists(RECORDS): return {}`, ~line 207). On any machine
   without that local path **every CoT token vanishes with no error** and the
   labels read as legitimately empty. This is the silent-absence class; it wants
   a loud failure.
5. **→ eval/tools.** `stack/tanitad/eval/constraints.py` (the sibling's) is the
   right consumer for whichever design wins. Not wired by me.

---

## 8. What I did NOT do

* ⛔ No model trained, no arm scored, no GPU job launched, no pod touched, no
  A40 approached, **no HuggingFace pull** (source was local; quota untouched).
* ⛔ **I did not invent a max-speed proxy** and scored nothing against one.
* ⛔ I did not open the camera frames, so I cannot say whether any individual
  reading is a real sign, a dashboard, or a hallucination — only what the
  statistics of the set are.
* ⛔ I did not edit `refc.py`, `refc_v3*.py`, `refcv3_arm.py`, `taniteval/ci.py`,
  `train_v6_staged.py`, `v6.py`, `predictor.py`, `goal_point.py`,
  `v7_labels.py`, `stack/tanitad/rl/`, `refav1_lon_cost.py`, the tactical label
  reader, `constraints.py`, `CLAUDE.md`, or the paper. **Nothing outside my own
  Research directory was modified.**
* I did not re-derive the sibling's `SPEED_BAND` / `road_class` / OSM results;
  they are cited INHERITED with their path.
* ⚠️ I did not establish that `vlm_route_labels.py` has never been run — my
  search was scoped to `stack/` and is **INCONCLUSIVE**, not a proven absence.

---

## 9. Deliverable manifest

| artifact | path | state |
|---|---|---|
| this document | `…/Research/2026-09-06-speed-limit-source/RESULT.md` | repo, **staged** |
| schema enumeration | `code/enum_schema.py` | repo, **staged** |
| speed-limit search, all 5 tasks | `code/probe_speedlimit.py` · `raw/speedlimit_probe.json` | repo, **staged** |
| READ vs HEDGED classifier | `code/classify_limits.py` · `raw/limits_classified.json` | repo, **staged** |
| ⭐ ego-echo test + controls | `code/ego_echo_test.py` · `raw/ego_echo.json` | repo, **staged** |
| ⭐ temporal-specificity discriminator | `code/temporal_specificity.py` · `raw/temporal.json` | repo, **staged** |

**Source data (NOT banked — 24.8 MB, already on local disk and on HF):**
`C:/Users/Admin/tanitad-data/alpamayo/records.parquet`, md5
`9f13474723b880eec7fcc09a7be478d8`, mirrored at
`Sayood/tanitad-alpamayo2-augmentation` snapshot `cedbf57c…`. Every script
above regenerates its JSON from it. Nothing lives only on a pod or only in a
worktree.
