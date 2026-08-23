# REF-C retrain — the corpus and the labels

**Stream:** REF-C improvement DIRECTION 3 of 3 — data and labels · **Date:** 2026-08-03
**Owns:** `stack/tanitad/data/`, `stack/scripts/refb_labels.py`, label QA instruments.
**Does not own:** `stack/tanitad/refs/` (architecture stream), `stack/experiments/*-gsplat/`
(planner/vision stream). Nothing in either was edited.

Machine-readable companion: **`CORPUS_LABEL_MANIFEST.json`** (every corpus and every label
with its size, parity verdict, valid fraction, degenerate fraction, class balance,
denominator and evidence class).

---

## 0. The one-paragraph answer

**Retrain REF-C on the SAME parity episodes it already has, with BETTER TARGETS ON THEM.**
Every richer corpus in reach is a different episode selection, and the label defects that
actually bound REF-C are *representation* defects, not *quantity* defects: the 5-way tactical
target destroys a live longitudinal decision on **17.0 %** of windows (n = 9,203, stride 8;
9.68 % on the canonical 40-ep val at stride 5), the strategic **input** hands the model a
meaningless `follow` on **62.4 %** of the windows where it says `follow`, and the ego channels
that claim to be strictly causal read 0.1 s into the future on **91.3 %** of frames. Fixing
those costs zero new frames and keeps parity intact. The one corpus change worth making is
also parity-preserving: **join `obstacle.offline` into the episode build** so the LONGITUDINAL
distance-keeping family the PI made binding can be computed at all — today its n is **0**.

---

## 1. What was measured, and on what

**⚠️ THE EVIDENCE BASE AND ITS LIMIT, STATED FIRST.** Every percentage in this document that
is marked MEASURED was computed on the only PhysicalAI episode data reachable without a pod:

| split | path | episodes | windows (stride 8) | `corpus_key_of` |
|---|---|---:|---:|---|
| local train | `C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74` | 400 | 9,203 | **`None`** |
| local val | `.../physicalai-val-bb543bdf7836` | 100 | 2,301 | **`None`** |

Both resolve to **`None`** — they are **NOT** the canonical `physicalai-train-e438721ae894`
set. They are admissible as evidence **about a labeler** (which is a pure function of `poses`)
and about a **mechanism**; they are **not** cross-arm comparable and no percentage here may be
quoted as a parity number. §4 shows exactly why the percentages travel badly and the mechanisms
do not.

Median clip 19.9 s (min 19.7, max 20.5), 99,477 frames, 0 skip markers. Episode record:
`{frames_u8 [199, 9, 256, 256] uint8, actions [199, 2], poses [199, 4], episode_id, maneuvers [199]}`.

Run directory: `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-03-refc-corpus-and-labels/`
· raw results in `results/`.

---

## 2. Label quality — every family, with its denominator

### 2.1 TACTICAL — the 5-way target destroys a live longitudinal decision on 17.0 % of windows

MEASURED, `results/labelqa_pai_train_400ep.json` → `TACTICAL.LOSSY_RATE`, **n = 9,203 windows /
400 episodes / stride 8**, v2 curvature gate (v1 endpoint gate in brackets):

| quantity | n | % of windows |
|---|---:|---:|
| lateral class is a TURN | 2,958 | 32.14 % (35.18 %) |
| longitudinal class is LIVE (accelerate or brake_stop) | 4,591 | **49.89 %** |
| **LOSSY — turn AND live longitudinal** | **1,562** | **16.97 %** (18.49 %) |
| share of live-longitudinal windows destroyed | — | **34.02 %** (37.07 %) |

Val (n = 2,301 / 100 episodes): **16.60 %** lossy, 32.96 % of live-longitudinal destroyed.

Class balance, v2 gate, train (n = 9,203):

| man5 | % | | lat | % | | lon | % |
|---|---:|---|---|---:|---|---|---:|
| lane_keep | 34.95 | | lane_keep | 67.86 | | brake_stop | 21.20 |
| turn_left | 17.72 | | turn_left | 17.72 | | steady | 50.11 |
| turn_right | 14.42 | | turn_right | 14.42 | | accelerate | 28.69 |
| accelerate | 16.74 | | | | | | |
| brake_stop | 16.17 | | | | | | |

**The projection identity holds exactly.** `collapse(lat, lon)` reproduced the shipped 5-way
label on **11,504 / 11,504** windows, **0 mismatches**, for both the v1 and the v2 gate. This is
the fact the retrain recommendation rests on: switching REF-C to the factored target is a
**refinement at identical thresholds**, not a relabel — nothing that trains today changes
meaning, and `derive_man5_logprobs` keeps publishing the same 5-wide vector downstream.

**⚠️ RECONCILING 17.0 % WITH THE PUBLISHED 9.68 %.** They are the same quantity under the same
definition (`refc_tactical_probe.py:360-367` uses `man5 in {turn_left, turn_right}`, which
`COLLAPSE_TABLE` makes identical to `lat != lane_keep`). They differ because the **populations**
differ: the canonical 40-episode val has a turn rate of 16.1 % (142/881, `taniteval/results/
planfan_clips_tactical_head_val.json`) while this 400-episode sample has 32.1 %, and the lossy
rate is very nearly proportional to the turn rate. **Neither number corrects the other. 9.68 %
(n = 1,364, stride 5, canonical val) remains the parity figure; 17.0 % is what the same defect
looks like on a turn-heavy sample.** A third figure, 7.15 % (n = 881), is reconstructed from fan
waypoints rather than raw poses and its own artifact says so.

### 2.2 STRATEGIC — the nav INPUT is degenerate on 62.4 % of the windows where it says `follow`

MEASURED, `results/labelqa_pai_train_400ep.json` → `STRATEGIC`, n = 9,203:

| route reason | n | % |
|---|---:|---:|
| `tight_transient` | 6,024 | 65.46 % |
| `no_arc` | 1,200 | 13.04 % |
| `road_following` | 1,194 | 12.97 % |
| `gray_zone` | 785 | 8.53 % |

Route class: left 36.09 %, right 29.37 %, **unknown 21.57 %**, straight 12.97 %.
`route_valid_rate` **78.43 %**.

**The defect is on the INPUT side, and it is the v2.1 D2 bug wearing a new costume.**
`route_target_v21` refuses to say `straight` when it does not know — it returns
`ROUTE_UNKNOWN`, deliberately outside CE range. But `nav_command_v21` maps the answer back
through `_ROUTE_TO_NAV.get(route, NAV_FOLLOW)` (`refb_labels.py`, v2.1 section), and
`ROUTE_UNKNOWN` is not a key. So:

| what the model is fed | n | |
|---|---:|---|
| `nav = follow` total | 3,179 | |
| …because the road really does go straight | 1,194 | 37.56 % |
| …because the labeler had **no idea** | **1,985** | **62.44 %** |

The model input cannot tell those apart. The `valid` flag carries the distinction and every
consumer that drops it — the label exports do, they carry the class — re-creates the bug.

**⚠️ THE MECHANISM IN THE BRIEF IS THE v2 MECHANISM, NOT v2.1's.** The brief says
`nav_command_v21` "needs a 25 s lookahead (15 s min)". It does not: v2.1 **removed** the
`min_steps` guard and replaced it with an arc gate, which is why coverage went 26.0 % → 81.9 %.
The 15 s/25 s collapse belongs to `nav_command` / `nav_command_v2` (measured 560/800 = 70 %
`no_future`). v2.1 reaches `nav = 0` by the different route above. Same symptom, two mechanisms
— and they need different fixes, which is why this matters.

**⭐ A SECOND, UNKNOWN BUG FOUND WHILE CHECKING THAT.** Two live callers pass `min_steps=10` to
`route_from_future_v21` to build a scene-length-adapted short-horizon nav, and swallow the
result in a bare `except`. v2.1 has no such parameter, so **the fallback has never once run**.
MEASURED, every rollout row in
`stack/experiments/alpasim-gsplat/results/openloop-thor-2026-08-03/rollouts/*.json`:

```
"nav_short": 0, "nav_short_valid": false,
"nav_short_err": "TypeError(\"route_from_future_v21() got an unexpected keyword argument 'min_steps'\")"
```

Callers: `closedloop_drive.py:368`, `score_t1_strategic.py:392`. **Every strategic number that
relied on `nav_short` was reading a swallowed exception, not a short-horizon label.**

### 2.3 STRATEGIC — the label has no distance, and the turn share is a SPEED artefact

MEASURED, `results/route_within_clip_probe.json`, n = 7,507 turn-labelled windows / 500 episodes:

| distance to the turn | share |
|---|---:|
| `d_now` (< 10 m) | 32.92 % |
| `d_10_25` | 21.53 % |
| `d_25_50` | 21.05 % |
| `d_50_100` | 18.24 % |
| `d_100_200` | 6.27 % |

Median 22.1 m, p90 85.4 m, **24.5 % of turn windows are more than 50 m from the turn** — and all
of them carry the identical 3-class token. `refb_labels.DIST_BAND_TOKENS` (width 8) is already
minted by `route_from_future_v3`; `lake/vocab.py` lists `ROUTEDIST` as a **v1.1 candidate, not
enrolled**. Enrolment is a vocabulary version bump and is the PI's call.

Within-clip behaviour (500 episodes, ~23 windows each): **5.4 %** of episodes carry a constant
route, mean 2.36 distinct classes, **1.49 switches per episode** — against 4.57 switches for the
tactical label. The strategic label is *coarse and slowly varying*, **not** the pure clip tag I
expected to find; that hypothesis is refuted here rather than asserted.

### 2.4 The turn label is overwhelmingly a function of ego speed

MEASURED, `results/route_gate_speed_probe.json`, n = 11,504 windows / 500 episodes:

| ego speed | share of corpus | turn rate | **LOSSY rate** | median \|κ\| (1/m) |
|---|---:|---:|---:|---:|
| < 1 m/s | 16.79 % | 23.15 % | 10.82 % | 0.00075 |
| 1–3 | 8.98 % | 56.63 % | 38.24 % | 0.02804 |
| 3–6 | 22.28 % | **60.24 %** | **34.02 %** | 0.02728 |
| 6–10 | 38.88 % | 24.26 % | 9.88 % | 0.00480 |
| 10–15 | 12.52 % | **2.50 %** | **1.81 %** | 0.00132 |
| 15+ | 0.56 % | 0.00 % | 0.00 % | 0.00153 |

The turn gate is a fixed `|κ| ≥ 1/60 m⁻¹` at every speed, so this spread is not the gate moving
— it is what the corpus contains at each speed *plus* estimator variance (at 3 m/s the 2 s
window spans 6 m of road, so a 0.1 rad heading change already clears the gate). **Whatever the
mix of the two, the operational consequence is the same and it is the strongest parity argument
in this document: any episode re-selection that moves the speed distribution moves every
tactical and strategic label distribution with it.** Adding highway-heavy comma2k19 would not
"add data" — it would silently re-scale the class priors REF-C is compared against.

### 2.5 The transience gate is inert — and that is NOT why the turn share is high

MEASURED: **96.0 %** of windows have less observed arc than `TRANSIENCE_MIN_ARC_M` (150 m;
median arc 66.9 m), so `route_from_future_v21` forces `transient = True` and the turn rule
degenerates to the tightness test alone — exactly the false-turn mode the concentration gate was
added to prevent.

I pre-registered both readings and built the fix: `route_from_future_v22` re-enables the gate at
**60 m** of road (`conc_arc = clamp(arc/3, 20 m, 60 m)`; above 180 m it is identical to v2.1 by
construction — no new fitted threshold).

**⛔ OUTCOME B FIRED. The A/B refutes the hypothesis.** `results/route_v21_vs_v22_ab.json`,
same 11,504 windows: **115 windows change (1.00 %)**, all of them turn → unknown; turn rate
65.3 % → 64.3 %; valid rate 77.86 % → 76.86 %. Re-enabling the gate does essentially nothing.

**⇒ `route_from_future_v22` is implemented, tested and NOT recommended as a default.** The turn
share is a property of this slow, junction-dense corpus (and of the estimator at low speed), not
of the disabled gate. Per the pre-registration, the corpus recommendation must address the
**speed mix**, which is what §5 does. Recording a refuted hypothesis with its instrument intact
is the point of pre-registering it.

### 2.6 SITUATION — the causality break, sized

MEASURED, `results/labelqa_pai_*.json` → `SITUATION.CAUSALITY_AUDIT`, **79,577 train frames /
19,900 val frames**, `|centred − strictly-causal backward difference|`:

| channel | frames differing > 1e-6 | mean abs diff | mean abs value | relative | max abs diff |
|---|---:|---:|---:|---:|---:|
| `alon_pre` | **91.31 %** | 0.0307 m/s² | 0.6774 m/s² | **4.53 %** | 0.611 m/s² |
| `omega_pre` | **91.19 %** | 0.00306 rad/s | 0.0938 rad/s | **3.26 %** | 0.0750 rad/s |

Val is the same to within 0.2 pp. This is a real, everywhere-present leak in a channel the
module docstring calls *"the only ego channels a head may receive"* — not a boundary effect.

Situation coverage (a labeling-density fact for the retrain): **14.0 %** of train episodes
contain a lane change, **50.5 %** contain an intersection event (59 and 222 events over 400
episodes); val 12.0 % / 45.0 %.

### 2.7 LONGITUDINAL distance-keeping — n = 0, and that is a work item

The PI's binding four-family rule requires headway / time-gap / TTC to the lead agent. The
episode record contains **no lead-agent track at all**. MEASURED by enumerating an episode's
keys, not inferred. Reporting per the rule: **LONGITUDINAL distance-keeping — NOT COMPUTABLE,
reason: no lead-agent channel in any existing cache, n = 0.** Target speed IS computable
(`v(t+2s)`, n = 9,203). `obstacle.offline` carries 3D agent tracks on 97.44 % of the source
corpus and the episode builder reads 4 of 36 features. See §5.

---

## 3. What was fixed (code, in the repo, staged)

| # | defect | fix | file |
|---|---|---|---|
| 1 | `alon_pre` / `omega_pre` built on `np.gradient` (centred) under a `STRICTLY CAUSAL` comment | `backward_diff` + `_trailing_mean`; `kinematics(causal_pre=True)` default, both variants always returned, `pre_mode` stamped | `stack/tanitad/data/situations.py` |
| 2 | `route_from_future_v21` rejects `min_steps`; two callers swallow the TypeError | `min_steps: int \| None = None`, accepted and **honoured**; `None` keeps v2.1 semantics byte-for-byte | `stack/scripts/refb_labels.py` |
| 3 | `nav_command_v21` collapses `ROUTE_UNKNOWN` and `ROUTE_STRAIGHT` onto the same `NAV_FOLLOW` | `nav_command_v21_ex` (returns `unknown_sentinel`) and `nav_input_v22` (returns the `(command, known)` pair a trainer should feed) | `stack/scripts/refb_labels.py` |
| 4 | transience gate inert below 150 m of arc | `route_from_future_v22` / `route_target_v22` — opt-in, measured, **and not recommended** (§2.5) | `stack/scripts/refb_labels.py` |

**Scope discipline on fix 1.** Only the two `*_pre` channels changed. `omega` / `kappa` / `alon`
— the detector channels whose thresholds are FROZEN by `PRE_REGISTRATION.md §2`, and which the
PI's 2026-08-03 ruling explicitly permits to use the future — are byte-identical, and a test
pins that.

**⚠️ ESCALATION, not a README note.** Fix 1 changes every `[v, alon_pre, omega_pre] / EGO_SCALE`
block that has ever been banked: `sc_build_labels.py:164`, `build_substrate.py:111`,
`sc_train.py:38`, `sc_train_v2.py:38`, `gen1_sc_train.py:38`, `run_sitclf_opt.py:78`,
`tanitad/eval/sitclf_deploy.py:264`. Those substrates were built on the leaky channels and their
numbers do not carry over. `causal_pre=False` reproduces them bit-for-bit so nothing is stranded.
**The sitclf stream must rebuild before quoting an ego-arm number as causal.** A byte-identical
copy of the defect also survives at
`…/incoming/2026-07-26-situation-classifier/scripts/sc_situations.py:87-92` and a separate
`alon_pre` lineage at `…/2026-07-26-h2-label-v2/scripts/l2_build.py:280` — **neither was touched
by me** (different owners) and both need the same treatment.

**Tests:** `stack/tests/test_label_causality_and_nav.py`, **18 tests, all passing**. They pin the
causal property by perturbing the future and asserting the past does not move; pin that the
legacy channel *still fails* that test (so the fixture keeps exercising the defect); pin that the
detector channels did not move; pin `min_steps` acceptance, honouring and default-identity; pin
that `nav_command_v21` really does collapse the sentinel and that the new API exposes it while
reproducing the shipped 2-tuple exactly; pin that v2.2 can never turn a decided window into
no-data; and pin the `collapse(lat, lon) == man5` identity.

---

## 4. Corpus inventory — the short form

Full table with sizes, paths and evidence classes: `CORPUS_LABEL_MANIFEST.json`.

| corpus | size | parity | usable for a REF-C retrain? |
|---|---|---|---|
| parity raw epcache train `e438721ae894` | 260 GB / 2,376 eps / 24 skips | **CANONICAL** | ✅ **yes — this is the arm** |
| parity raw epcache val `0c5f7dac3b11` | 65.56 GiB / 600 eps (40-ep deployment → 881 windows) | canonical, **digest UNRECORDED — count-only** | ✅ yes |
| 256 px raw epcache on HF `Sayood/tanitad-physicalai-w120-256x640cyl` | 349.5 GB / 3,053 files | asserted parity | ✅ **the recovery path now pod2 is terminated** |
| w120 v2 siblings (same HF repo) | 106 GB, 2,400 + 600 clips | parity-preserving **re-cache** | ⛔ 256×640 cylindrical; REF-C is 256 px **square** |
| v2bal 50 h balanced `4b7eeeac222d` | 9,000 clips / 49.7 h | **DIFFERENT SELECTION by design** | ⛔ not a drop-in — a new arm on a new axis |
| comma2k19 | ~88 GB | non-parity | ⛔ highway-dominated → moves the speed mix (§2.4) |
| NuRec + `map.xodr` | 2.89 TB gated, 1,607 scenes, T1 = 141 | different corpus | ✅ **as a strategic-label validator, not as training data** |
| l2d / argoverse2 / nuscenes / cosmos_drive | — | — | ⛔ none is wired to any trainer — unbuilt capability, not available corpus |

Two claims in the brief did not survive probing and should be corrected upstream:
`epcache-256px-phase0/` (**UNCONFIRMED**, four probes at different locations; the real shape is
`/workspace/data/physicalai_phase0/_epcache/<corpus_key>/`) and `sequence_tracks.json`
(**UNCONFIRMED**, not in `build_t1_labels.py`'s member list nor anywhere in the repo).
`clipgt/obstacle.parquet` is named in `README.md:229` but **no parser reads it**.

The map-derived NuRec strategic GT is real and already banked: 141 T1 candidates → 141 labelled
→ 110 admissible (31 refused by the map-vs-realised-heading control) → 78 with a scoreable
decision → **77 scored, 116–117 events, 4,745 poses**. It carries an option set and a distance
per event — i.e. exactly the two things the kinematic route label lacks.

---

## 5. Recommendation for the REF-C retrain corpus

### 5.1 The corpus: unchanged. The targets: changed.

> **Train the retrain arm on `physicalai-train-e438721ae894` (2,376 episodes, skip-hash
> `f09e44db`), evaluate on the 40-episode `physicalai-val-0c5f7dac3b11` deployment (881 stride-8
> windows). Do not re-select a single episode.**

**Parity implication, plainly:** there is none. This keeps strict parity, so the retrain is
directly comparable to `refc-base-30k` and `refc-xl-30k` and to every other parity arm. Every
richer corpus in reach (v2bal, comma2k19, realmix, NuRec) is a **different episode selection**,
and §2.4 shows what that costs: the tactical and strategic class priors are strong functions of
the speed mix, so a corpus swap changes the *label distribution* at the same time as the *data*,
and any ADE delta is then unattributable. That is precisely how the v2corpus arm was falsified.

### 5.2 The one corpus change worth making — and it is parity-preserving

**Join `obstacle.offline` into the episode build.** It is a **RE-CACHE, not a re-selection**:
the ordered clip list and `split_clips` take no agent-track argument, so the episode uid set,
count and skip indices are unchanged and `parity.register_geometry_sibling` can prove it (that
function refuses unless the uid digest, count and skip indices match exactly).

Why it is the highest-value data work available:

1. It is the **only** way to compute the LONGITUDINAL distance-keeping family the PI made
   binding. Today headway / time-gap / TTC have **n = 0** and every REF-C eval must report them
   as absent.
2. It converts the **TACTICAL** family from "which of 5 classes" to a decision with a *reason*
   (a brake with a lead agent at 1.2 s time-gap is a different label from a brake at a stop
   line), which is the `HEADWAY` slot `lake/vocab.py` already froze at width 5 and nothing has
   ever minted.
3. `obstacle.offline` covers **97.44 %** of the corpus and the episode builder currently reads
   **4 of 36** features, so the data is already paid for.

### 5.3 The targets to change, in priority order

| # | change | why | cost |
|---|---|---|---|
| 1 | **factored `lat × lon` tactical target** — flip `RefCConfig.factored_maneuver` | recovers the 9.68 % (parity val) / 17.0 % (this sample) of windows the 5-way target cannot represent. `collapse(lat, lon) == man5` on 11,504/11,504 windows, so it is a refinement, not a relabel | **zero** — labels and trainer wiring already exist, gated off |
| 2 | **feed `(nav_cmd, nav_known)`** via `nav_input_v22` instead of the bare command | 62.4 % of the `follow` windows the model is fed today carry no information and it cannot tell | one input channel |
| 3 | **enrol `ROUTEDIST` (width 8)** from `route_from_future_v3` | 24.5 % of turn-labelled windows are > 50 m from the turn and all carry the same token; a planner needs the distance | a vocabulary version bump — **PI's call** |
| 4 | **rebuild the sitclf ego substrate** on the causal channels | 91.3 % of frames were leaking 0.1 s | one rebuild, `causal_pre=True` is now the default |
| 5 | **score the strategic head against the NuRec map-derived GT** (77 scenes, 116 events) | it is the only non-circular strategic label we have; every kinematic route label is a function of the ego's own future | already banked |

### 5.4 The comparison design that keeps the result attributable to ONE axis

The failure to avoid is the v2corpus one: a single arm that changed the data *and* ten other
things. So:

* **Baseline** — `refc-base-30k`, already published, parity corpus, 5-way target. Nothing to run.
* **Arm A (the retrain)** — identical corpus, identical schedule, identical seed, **one change:
  `factored_maneuver=True`**. `refc.refc_f1only_config()` already exists as the *input*-only arm
  on the unchanged 5-way head, so the structure change is separable from the input change and
  each is independently ablatable. Run A first; it is the change with the measured label
  evidence behind it.
* **Arm B** — Arm A **plus** `nav_input_v22`. One further axis, run only after A reports.
* **Never** combine a target change with a corpus change in the same arm. If `obstacle.offline`
  lands, it produces **Arm C on the re-cached corpus with Arm A's targets**, so the corpus delta
  is read against A and not against the baseline.
* **Read all four families per arm** (LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC), each with
  the **paired episode-cluster bootstrap** over the 40 val episodes (`taniteval/ci.py`), on the
  same windows as the ADE — never `overlapping_holdout_se`, and never pooled into one score.
  Where distance-keeping is still absent, say so **per family with the reason and n = 0**.

### 5.5 Pre-registered readings for Arm A

* **A-positive** — TACTICAL per-class recall for `accelerate` / `brake_stop` rises above the
  0.0000 / 0.0256 floor at unchanged lateral recall, and ADE does not regress beyond its paired
  CI. Reading: the label was the binding constraint; keep the factored target and proceed to B.
* **A-null** — longitudinal recall stays at the floor. Reading: the defect is **not** in the
  target representation, and the remaining candidates are the head's *input* (F1:
  `tactical_speed_input`) and the *decision rule* (F3: `logit_adjust`), both of which are already
  gated and separately ablatable. That would be a real finding, not a failure.
* Both readings are committed here, before the run.

---

## 6. DONE vs NOT DONE

### DONE

1. **Corpus + label manifest** — `CORPUS_LABEL_MANIFEST.json` (9 corpora, 8 label families, each
   with size/path/parity verdict/valid fraction/degenerate fraction/class balance/denominator/
   evidence class) and §4 here.
2. **Label defects fixed, with tests** — the centred-difference causality break; the dropped
   `min_steps` (which also uncovered a never-run fallback in two live callers); the nav-input
   sentinel collapse; plus an opt-in scene-adaptive route labeler. 18 new tests.
3. **Label quality quantified, not asserted** — every number in §2 carries its denominator and
   its artifact path. The 9.68 % vs 17.0 % discrepancy is explained rather than papered over.
4. **Recommendation with the parity implication stated and a one-axis comparison design** — §5.
5. **A refuted hypothesis recorded with its instrument intact** — §2.5, pre-registered, Outcome
   B, `route_from_future_v22` NOT recommended as a default.
6. **`pytest -q` green** on the full suite.

### NOT DONE (and why)

1. **No measurement on the canonical parity corpus.** Every episode cache with the parity key
   lives on a pod, and pod2 is terminated while `tanitad-new` and `tanitad-pod4` are training and
   must not be touched. The instruments are pod-ready: `python labelqa_scan.py --cache-dir
   /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 --out ...` will stamp
   `is_registered_parity_corpus: true` and produce the parity version of every table in §2.
   **This is the single highest-value follow-up and it needs one idle pod-hour, CPU only.**
2. **`obstacle.offline` is not joined.** §5.2 argues for it; implementing it needs the source
   corpus, which is pod-side. Not started.
3. **`ROUTEDIST` not enrolled.** `lake/vocab.py` is a frozen vocabulary; enrolment is a version
   bump and the PI's call, not mine.
4. **The duplicate causality defect is not fixed everywhere.**
   `…/2026-07-26-situation-classifier/scripts/sc_situations.py:87-92` is a byte-identical copy
   and `…/2026-07-26-h2-label-v2/scripts/l2_build.py:280` is a separate `alon_pre` lineage. Both
   belong to other streams; escalated in §3 rather than edited.
5. **No banked sitclf substrate was rebuilt.** That is the sitclf stream's artifact; the fix
   makes the rebuild correct and `causal_pre=False` makes the old one reproducible.
6. **`clipgt/obstacle.parquet` / `sequence_tracks.json` remain UNCONFIRMED.** Two probes each at
   different locations found nothing; they are reported as unconfirmed rather than as absent.
