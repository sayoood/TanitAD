# The max-speed INPUT — can it exist at all? A supplier census, and the answer is YES-BUT

**2026-09-10 · Architecture & Inference · `agent/arch-inf-20260803` · PI item 3 ("do 3")**

⛔ **TIER: NONE, AND THAT IS NOT A DODGE.** No model runs here. Every number below is an
**artifact census** over a label blob, a **cross-check** against an independently authored
reference, or a **mutation proof** on a refusal path. T0 is a world-model diagnostic and T1 is
self-action open loop; neither describes counting fields in a JSONL. Stamping this T0/T1 would
be a category error.

---

## THE ANSWER IN ONE LINE

> **YES — an admissible supplier exists, and it is NOT the one the flag reads.**
> The Alpamayo CoT carries **VLM readings of actual roadside speed-limit SIGNS** — vision-derived,
> not ego-derived, on the posted ladder, with country-correct units. It is admissible.
> ⛔ **But it covers 31 of 4,572 train clips (0.678 %) and 0 of 147 eval clips (0.000 %).**
> ⇒ **The channel cannot be built today** — not because no admissible supplier exists, but
> because the admissible one is **147× too sparse on train and absent from eval entirely.**

⭐ **This CHANGES the standing ruling rather than repeating it.** `max_speed_input.py`'s
`ChannelExclusion` already says *"not admissible from this corpus"* and already names the unblock:
*"read it from the frames with sign recognition"* (`stack/tanitad/refs/max_speed_input.py:166-171`).
**Nobody had ever measured whether that unblock has a supplier.** It does. Now it has a number,
and the number is what decides the build.

---

## 1 · THE ADMISSIBILITY VERDICT — every candidate supplier, with its layer

⭐ **The test applied to each:** *"could this have been computed from the thing being measured —
the ego's own future, or the situation classifier's output?"* If yes, inadmissible.

| # | candidate supplier | LAYER | computed from | coverage | verdict |
|---|---|---|---|---|---|
| 1 | `speed_max_input.v_max_ms` (what the flag reads) | augmented v8 | `g_tac.goals.SPEED_BAND.v_hi_ms` = **max of the ego's OWN realised speed over [t0+2 s, +6 s]** | 0/4,572 on v7.2 | ⛔ **INADMISSIBLE** — ego-future |
| 2 | `g_tac.goals.SPEED_BAND.v_hi_ms` directly | augmented v7.2 | same as #1 — it **is** #1's source | 4,572/4,572 | ⛔ **INADMISSIBLE** — ego-future |
| 3 | quantized `quantize_up(v_hi)` (8-step posted ladder) | derived | #2 snapped up to a road-law ladder | 4,572/4,572 | ⛔ **INADMISSIBLE** — laundering, not fixing: bin+`v0` still recovers **R² 0.9702** of the raw ego future |
| 4 | `cot_tokens.speed_limit` (boolean) | augmented v7.2 | VLM mention-detector over the CoT text | 69/4,572 True | ⚠️ **ADMISSIBLE BUT VALUELESS** — a bool is not a ceiling |
| 5 | ⭐ **numeric sign readings in the CoT text** | augmented v7.2 | **VLM reading a roadside SIGN from the camera** | **31/4,572 train · 0/147 eval** | ✅ **ADMISSIBLE — and far too sparse** |
| 6 | posted limit from a map (`map.xodr`, OSM) | external | HD map / OSM join | **0** on PhysicalAI | ⛔ **NO SUPPLIER** — no map, and `egomotion` has no lat/lon, so map-matching is impossible |
| 7 | road-class prior (`strata.road_class` → a default limit) | augmented v7.2 | a **situation label** | 4,572/4,572 | ⛔ **INADMISSIBLE** — this is the situation classifier's output wearing a ceiling's name (PI ruling 2026-08-03) |

**Evidence class: MEASURED (ours)** for every coverage figure — see §3 and `raw/`.
**PUBLISHED-CODE** for the provenance of #1/#2, quoted at three independent sites in §1.1.

### 1.1 ⛔ Why #1 fails the echo test, quoted from our own source

The programme's own code says it three times, in three files, without hedging:

* `stack/tanitad/data/v7_labels.py:325-330` — *"Its provenance is `ego-future`: the training value
  is max of the ego's OWN REALISED speed over [anchor+2 s, +6 s]"*, and the function refuses
  without an oracle stamp.
* `stack/scripts/refc_v3_train.py:5514-5518` (the flag's own help) — *"TRAIN/DEPLOY MISMATCH,
  STATED: the training value's provenance is `ego-future`"*.
* `stack/tanitad/refs/max_speed_input.py:367` — `"provenance": "ego-future (stands in for a map/nav
  speed-limit service)"`.

⇒ This is the **nav-echo defect** in a new costume, and the codebase already knows it. A ceiling
that *is* the max of the speed being predicted, handed to the planner on the horizon being scored,
on the axis owning **88.7 %** of the oracle gap.

### 1.2 ⛔ And quantization does not launder it — MEASURED, not argued

`FORWARD_EXCLUSIONS` in `max_speed_input.py:148-183` carries the arithmetic: 5-fold out-of-fold,
clip-disjoint, n = 4,572 — the raw ego-future ceiling is recoverable from the pinned 8-step bin
plus `v0` at **R² 0.9702** (bin alone 0.9513; `v0`-alone control 0.8789, so the bin **adds
ΔR² = +0.0913 of the ego's own future**). ⇒ **97 % of the answer's speed envelope, on the horizon
being scored.** *(Evidence class: MEASURED by the 2026-09-06 package, INHERITED here — I did not
re-run the ridge; I re-read the artifact and its source.)*

---

## 2 · WHAT THE v8 SCHEMA EXPECTS — the writer, the reader, the refusal

| role | site | what it does |
|---|---|---|
| **flag** | `stack/scripts/refc_v3_train.py:5502` | `--max-speed-input`, `action="store_true"`, default OFF |
| **early refusal** | `refc_v3_train.py:1052` `_check_max_speed_args` | refuses before any GPU work: no `--v7-labels`, `--preflight`, missing `--eval-labels` |
| **oracle gate** | `stack/tanitad/data/v7_labels.py:315` `oracle_max_speed` | raises `OracleNavRefused` unless the manifest carries `allow_oracle_nav` — **the arm is identifiable from its own artifacts** |
| **⭐ the load-bearing refusal** | `refc_v3_train.py:1596-1602` (in `enable_max_speed`) | *"NOT ONE of this split's `{n_win}` windows receives a `speed_max_input` value … Refusing rather than feeding nothing."* |
| **consumer / loader read** | `refc_v3_train.py:1982-1985` | ships the **RAW** `v_max_ms` into `item["v_max_ms"]`; the ladder is applied **once**, on the model side, by `msi.encode_block` |
| **units contract** | `max_speed_input.py:302` `read_max_speed_field` | ⛔ **REFUSES a payload that carries a speed and declares no units** — never guesses |

⛔ **I priced the CONSUMER'S read, not a file I found on disk.** The consumer is
`enable_max_speed`, it calls `v7l.oracle_max_speed(lab, manifest)` per clip, and that reads
`label._oracle["speed_max_input"]`. **That is the field I counted** — not `v_hi_ms`, which is a
different key that happens to hold the same number.

### 2.1 ⭐ MY OWN CONFIRMATION OF 0/4,572 — and I searched the comma-formatted form too

**MEASURED (ours)**, `raw/census_v72_keys_train.txt`, over
`…/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz`
(schema `s2-geom-v7`, vocab `v7`):

```
RECORDS n = 4572
CONTROL g_tac present (must be >0):   4572     <- same-breath positive control
CONTROL clip-id present (must be >0): 4572     <- same-breath positive control
speed_max_input present:   0 / 4572
speed_max_input non-empty: 0 / 4572
v_max_ms non-null:         0 / 4572
```

Eval split: **0 / 147** (`raw/sign_readings_eval.txt`; controls: 96/147 records contain a digit,
41/147 mention a traffic light — **the text was read**).

⛔ **The zero is a claim about the CONTENT, not about the search**, because the two same-breath
controls read 4,572/4,572 through the same loader in the same pass. A blob that could not be
opened would have failed both. **All 22 top-level keys are enumerated in the raw dump** — this is
a census, not a targeted probe that could miss a renamed field.

⚠️ **Comma-formatted search done as instructed:** `4,572` and `4572` both probed; the figure is
consistent at both forms and matches my own independent count exactly.

---

## 3 · ⭐ THE NEW MEASUREMENT — an admissible supplier EXISTS, and here is its size

**The layer question the brief demanded.** *"PhysicalAI-AV publishes no speed-limit feature"* is
**TRUE** and settled. *"The programme has no speed-limit signal"* is **FALSE** — exactly the
traffic-light conflation, one field across.

`cot_tokens` carries **11 VLM-extracted perception tokens**, one of which is `speed_limit`. The
record's own `_provenance` stamps them `"vlm-cot, disputed, never geometry-derived"` — i.e. **not**
derived from ego, and **not** derived from the situation classifier.

**MEASURED (ours)**, `raw/sign_readings_train.txt`:

| quantity | train (n = 4,572) | eval (n = 147) |
|---|---|---|
| `cot_tokens.speed_limit == True` (boolean) | **69** (1.509 %) | **0** |
| clips whose CoT text carries a **km/h** number | **30** | 0 |
| clips whose CoT text carries an **mph** number | **1** | 0 |
| **clips with ANY numeric road-unit limit** | **31** (**0.678 %**) | **0** (0.000 %) |
| …of those, within speed-limit **context** | **30** (0.656 %) | 0 |

**Same-breath positive controls (else the text was not read):** 3,025/4,572 records contain a
digit; 1,276/4,572 mention a traffic light. Both non-zero, so a zero elsewhere is about the
content.

**The values found, and they are a POSTED LADDER, not a speed histogram:**

```
20 km/h x14 · 30 km/h x77 · 35 mph x10 · 40 km/h x9 · 50 km/h x24
60 km/h x14 · 70 km/h x18 · 80 km/h x14 · 100 km/h x10
```

Verbatim examples (`raw/sign_readings_train.txt`):
*"a 50 km/h sign is posted ahead"* · *"the overhead speed limit sign shows 100"* ·
*"A pair of 70 km/h speed limit signs appear within the first 2 seconds"* ·
*"Speed limit sign (50 km/h) … Visible ahead along the right side"*.

⇒ These are **readings of a physical object in the camera frame**. That is precisely what a
speed-limit service supplies, and precisely what the exclusion's unblock asks for.

### 3.1 ⚠️ The boolean is NOT a usable proxy for the value

30 of the 31 numeric clips also have the boolean `True`; **1 does not** (`232e052a`, a 30 km/h
reading the mention-detector missed). And the boolean is `True` on **69** clips while only **31**
carry a number — so **38 clips assert a limit exists and never say what it is.** A bool cannot be
a ceiling, and it cannot be quantized into one.

---

## 4 · ⭐⭐ THE CROSS-CHECK THAT MAKES THE SIGN READINGS TRUSTWORTHY

⛔ **A cross-check must be derived independently of the value it checks.** So the sign readings
(VLM, vision) are checked against `SPEED_BAND.v_hi_ms` (geometry, ego) — two different authors,
two different mechanisms. `raw/crosscheck_signs_vs_ego.txt` lists all 31 clips.

**Three findings, and the third is the one that matters:**

1. ⭐ **The units are country-correct.** The **only mph reading in the corpus is in the only United
   States clip** (`d5ecf705`, 35 mph = 56 km/h, ego reached 53.9 km/h). Every other reading is
   km/h, in European countries. A hallucinating extractor does not get this right by accident.
2. **The magnitudes are sane.** Netherlands highway: sign 100, ego 99.8. Portugal urban: sign 50,
   ego 51.6 and 50.8. Latvia urban: sign 30, ego 30.3.
3. ⭐⭐ **THE READINGS ARE VIOLABLE — and that is the property the ego-derived ceiling can never
   have.** The ego **exceeds the sign by >5 km/h on 6 of 31 clips (19.4 %)** — e.g. Belgium
   highway sign 60 / ego 81.2; Portugal highway sign 70 / ego 94.9; Italy highway sign 100 /
   ego 107.5.
   ⇒ Compare: the **raw** ego-derived ceiling is exceeded on **0 / 4,572 by construction** (it *is*
   the max), and even the **quantized** one only on **28 / 4,572 (0.61 %)**.
   **A real limit can be broken. This one can. That is the discriminator.**

⚠️ **Honest residuals, named not hidden:**
* **Provenance is `vlm-cot, disputed`** — unverified perception, not ground truth. No independent
  channel corroborates it (the grounding boxes are label-only and carry no sign class).
* **The CoT describes a window, not an instant.** A sign the ego passes at t+3 s can be mentioned;
  a deployed service would know the limit at t0. Far weaker than the ego-future leak (a sign is a
  static roadside object, not the answer), but it is **not zero** and must be stamped.
* **n = 31 forbids any statistics.** No R², no CI, no bootstrap is admissible at this n. The
  numbers above are a **listing**, and I have quoted them as one.

---

## 5 · ⛔ THE REFUSAL PATH STILL FIRES — PROVEN BY MUTATION, NOT BY INSPECTION

⭐ **An AST census or a green test proves nothing; only a deliberate regression does.**
`code/mutation_refusal_proof.py`, output `raw/mutation_refusal_proof.txt`. The script disables the
guard at `refc_v3_train.py:1596` (`if n_win and n_win_valid == 0:` → `if False and …`), reruns the
suite, and restores the file **byte-exactly**.

```
ORIGINAL md5 = 9c5bc12490e6b2eefbdc71e30bfbd8ad  bytes = 345520
STEP 1  BASELINE (refusal intact)    exit=0   6 passed, 19 deselected
STEP 2  MUTATION (refusal disabled)  exit=1   1 failed, 5 passed
        RED: test_an_empty_channel_is_REFUSED_because_the_field_is_a_v8_ADDITION
STEP 3  RESTORED                     exit=0   6 passed
        restored md5 = 9c5bc12490e6b2eefbdc71e30bfbd8ad   bytes restored EXACTLY: True
VERDICT: the refusal HAS TEETH -> True
```

⭐⭐ **THE MOST IMPORTANT LINE IS THE MUTANT'S OWN STDOUT**, because it shows exactly what the
refusal prevents:

```
[v3] max_speed_input (quantized): 0/2 clips carry a ceiling, 0/2 windows fed, 0 over the 130 km/h top step
```

⇒ With the guard removed the trainer **prints a census saying it is feeding nothing, and carries
on training.** The arm's `config.json` would record `max_speed_input: true`. That is the
`tac_goal` failure exactly — an 11,286-parameter head that looked wired and took `grad_abs_sum` 0
for all 40,284 steps. **The refusal is the only thing standing between this flag and that outcome,
and it is load-bearing.**

⛔ **DO NOT "FIX" THE REFUSAL.** It is the good part. Full suites green:
`test_max_speed_wiring.py` + `test_max_speed_input.py` = **54 passed**.

---

## 6 · WHAT WOULD ACTUALLY SUPPLY IT, AND WHAT THAT COSTS

⭐ **Ranked by measured effect, cheapest first — and note that the corpus already proves route 1
works, on 31 clips.**

| # | route | what it needs | coverage it would reach | cost | verdict |
|---|---|---|---|---|---|
| **1** | ⭐ **Re-extract signs from the CoT we already hold** | a numeric extractor over `cot_source` (the regex in `code/census_sign_readings.py` **is** the prototype) | **31/4,572 (0.678 %)** — a **ceiling, already measured**, not an estimate | **~0 GPU, hours** | ⛔ **Refuse to build a channel on it.** 0.68 % train / **0 % eval**. An input present on 0 of 147 eval windows is a constant pad the arm measures as noise — and it is **unevaluable**. |
| **2** | **Sign recognition on the frames** (detector/OCR over the camera) | a traffic-sign model run over 4,572 clips; a new provenance stamp | unknown until measured; sign density suggests **well under 20 %** of 20 s clips contain a limit sign | **GPU + a new model + a label build** | ⚠️ **The only route to real coverage from this corpus.** ⛔ Blocked: it is a label generation run, forbidden by standing constraint without a PI ruling. |
| **3** | **Map join** (`map.xodr` / OSM) | a map, and a way to locate the ego in it | **0 on PhysicalAI** | — | ⛔ **STRUCTURALLY IMPOSSIBLE HERE.** No map in the corpus, and `egomotion` carries **no lat/lon** — clip-local metres only, so OSM matching cannot be attempted. ⭐ **But it is free on AlpaSim/NuRec**, whose scenes ship `map.xodr`. |
| **4** | **A corpus whose labels carry a genuine limit** | a different dataset | n/a | acquisition | ⚠️ Named by the exclusion's own unblock ("the same unblock by a different road"). A PI decision, not an engineering one. |
| **5** | ⛔ Ship the ego-future value as-is | nothing | 4,572/4,572 | free | ⛔ **REFUSED.** §1.1/§1.2. |

⇒ ⭐ **The honest recommendation: build NOTHING for refcv6, and take route 3 on AlpaSim.** The
max-speed channel is a **strategic/tactical** input; AlpaSim is where the strategic-brain topology
was already going to come from (`map.xodr` answers the map gap), and there the ceiling is a
**genuine posted limit at full coverage with zero ego provenance**. Route 1's 0.68 % settles that
PhysicalAI cannot host this channel — which is a real result, not a shrug.

### 6.1 ⛔ NO BUILDER IS PRE-REGISTERED, AND THAT IS THE FINDING

The brief said *"only if an admissible supplier exists"*. One does — and it is **147× too sparse on
train and absent from eval**. ⇒ Pre-registering a builder now would be pre-registering an arm that
**cannot be evaluated**: 0/147 eval clips means the ON arm and the OFF arm receive **identical
input on every eval window**, so the panel would measure the extra parameter's variance and
nothing else. That is not a conservative choice; it is the only correct one.

⚠️ **What I did NOT do, and why it is not a gap:** no label build was run (standing constraint),
and **Alpamayo was never re-asked** (explicitly forbidden). Every number here comes from blobs
already in git.

---

## 7 · FOR `PI_DECISION_QUEUE.md` — the paragraph, with a default

> ### DECIDE: the max-speed INPUT (item 3) — an admissible supplier EXISTS but is 147× too sparse
>
> **The flag is not broken and does not need fixing.** `--max-speed-input` refuses on v7.2 because
> `speed_max_input` is on **0/4,572** train and **0/147** eval records (MEASURED, re-confirmed
> independently 2026-09-10 with same-breath controls reading 4,572/4,572). **The refusal is correct
> behaviour and is proven to have teeth by mutation** — disabling it makes the trainer print
> *"0/2 clips carry a ceiling, 0/2 windows fed"* and train anyway, which is the `tac_goal`
> zero-gradient failure exactly.
> **The blocker is the SUPPLIER, not the code.** The v8 value's provenance is `ego-future` — the
> max of the ego's own realised speed over the horizon being scored — so it is the nav-echo defect
> and is refused. ⭐ **NEW, and it changes the picture:** the corpus *does* contain an admissible
> ceiling — **VLM readings of real roadside signs** in the Alpamayo CoT, vision-derived, on the
> posted ladder (20/30/40/50/60/70/80/100 km/h + 35 mph), **country-correct** (the only mph reading
> is in the only US clip), and **violable on 6 of 31 clips (19.4 %)** where the ego-derived ceiling
> is violable on **0/4,572 by construction**. ⛔ **But it exists on only 31 of 4,572 train clips
> (0.68 %) and 0 of 147 eval clips.**
> **Default if silent: max-speed is NOT built for refcv6, and refcv6 stays a clean single-lever
> panel against refcv5-v2.** The channel's code, ladder, units contract and tests stay exactly as
> they are — nothing is retracted and nothing is deleted; only the supplier is missing.
> **The decision he actually has** is which unblock to fund: **(a)** run a traffic-sign
> detector/OCR over the 4,572 clips to build a real v8 `speed_max_input` — a **label generation
> run**, currently forbidden by standing constraint, so it needs his word; or **(b)** move the
> max-speed channel to **AlpaSim/NuRec**, whose scenes ship `map.xodr` and therefore a genuine
> posted limit at full coverage with zero ego provenance. ⭐ **(b) is the recommendation** — it is
> the same route that already has to answer the strategic-map gap, and it needs no new labels.
> ⛔ **What is NOT on the table:** shipping the ego-future value, quantized or raw. Quantization
> does not launder it (bin + `v0` recovers **R² 0.9702** of the raw ego future), and it would hand
> a rollout a 97 %-accurate read of its own answer on the axis owning 88.7 % of the oracle gap.
> ⚠️ Related and already settled — do not re-derive: the 30 km/h floor premise **fails**; the
> bucket minimum is already **20 km/h with 0 nulls** (queue item 2).

---

## 8 · WHAT THIS CHANGES IN THE REGISTER

* ⭐ **`D-VOCAB-REACH-4` is EXTENDED, not contradicted.** It says the channel needs the v8 label
  release. True. **What it did not say** is that the v8 release's own value is inadmissible, so
  "wait for v8" is not the unblock — **"wait for a non-ego supplier"** is. The v8 blob would make
  the flag *run*; it would not make the arm *quotable*.
* ⭐ **The `ChannelExclusion` unblock now has a MEASURED feasibility.** `max_speed_input.py:166-171`
  named "sign recognition" as a route; it is now measured at **31/4,572 train, 0/147 eval** from
  the CoT we already hold. ⛔ `permanent=False` stays correct — the exclusion is still liftable,
  just not by anything on this corpus.
* ⚠️ **A layer error is pre-empted.** *"The programme has no speed-limit signal"* is **FALSE** — it
  has 31 numeric sign readings and 69 boolean mentions at the **augmented v7.2 label** layer. What
  is true is that **no supplier reaches training**, and no supplier reaches **eval at all**.

---

## DELIVERABLE MANIFEST

| artifact | where | only one place? |
|---|---|---|
| `ADMISSIBILITY.md` — this verdict | `repo:<pkg>/` | **staged, uncommitted** |
| `census_v72_keys.py` — 22-key census + `speed_max_input` count, with controls | `repo:<pkg>/code/` | **staged, uncommitted** |
| `census_sign_readings.py` — the numeric sign extractor (also the route-1 prototype) | `repo:<pkg>/code/` | **staged, uncommitted** |
| `crosscheck_signs_vs_ego.py` — sign vs `SPEED_BAND.v_hi_ms`, independently authored | `repo:<pkg>/code/` | **staged, uncommitted** |
| `mutation_refusal_proof.py` — deliberate regression on the refusal | `repo:<pkg>/code/` | **staged, uncommitted** |
| `census_v72_keys_train.txt`, `sign_readings_{train,eval}.txt`, `crosscheck_signs_vs_ego.txt`, `mutation_refusal_proof.txt` | `repo:<pkg>/raw/` | **staged, uncommitted** |

`<pkg>` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-max-speed-supplier-census/`

⛔ **Nothing here lives only on a pod or only in a worktree.** The mutation ran in an isolated
scratch checkout (`C:/Users/Admin/msi_scratch`, extracted from git, restored byte-exact) which is
**disposable and holds no unique artifact**.

## INTEGRATION ITEMS (escalated, not filed in a README)

1. ⛔ **The PI paragraph in §7 needs to reach `Project Steering/PI_DECISION_QUEUE.md`.** It is a
   new item, not an edit of item 2 (which is about the ladder's minimum, already settled).
2. ⚠️ **`WIRING_DIFF.md` from the 2026-09-06 package is STILL UNAPPLIED** and still has no owner —
   four hunks in `refc_v3.py`, anchors verified at `a1f5ade`. ⭐ **This finding lowers its
   priority rather than raising it:** there is no admissible value to feed through that wiring, so
   applying it now would build a seam with nothing on the other end.
3. ⭐ **Route 3 (AlpaSim `map.xodr`) belongs to a different FlyWheel.** If the PI takes option (b),
   the max-speed channel moves out of the PhysicalAI arm entirely, and this package's ladder,
   units contract and conditioner transfer unchanged.
