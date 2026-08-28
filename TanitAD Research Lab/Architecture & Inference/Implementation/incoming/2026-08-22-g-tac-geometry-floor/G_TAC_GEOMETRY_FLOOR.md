# `g_tac` had ZERO producers — the LON axis now has a 92 % geometry floor on parity; the LAT axis does not discriminate

**Owner:** tactical/strategic label-augmentation stream · 2026-08-22 · branch
`claude/exciting-kepler-4d4449` (worktree `exciting-kepler-4d4449`, fast-forwarded
to `37c90cd`)

---

## 0. TL;DR

| | before | after |
|---|---|---|
| `g_tac` producers, anywhere in the programme | **0** | 1 (`tanitad.data.g_tac_geom`) |
| `g_tac_lon` coverage | **0.00 %** | **92.28 % ON PARITY** (`STOP_POINT` 3.23 % + `LON_UNCONSTRAINED` 89.05 %), n = 69 447, d = 2 400 |
| `g_tac_lat` coverage | 0.00 % | **0.00 % — abstains; the estimator does not discriminate (§3)** |
| ego `args` tier | dropped on 100 % of clips (silently) | cause identified (§1.2) |
| `g_tac` in the label census | **absent — the 0 % was invisible** | counted, with per-axis abstention reasons |

⭐ **The headline is not the coverage number. It is that a whole layer of the
approved vocabulary had no producer and nothing was counting it.**

⚠️ **This package contains a RETRACTION OF ITS OWN FIRST DRAFT (§3.2b).** The LAT
conclusion was first measured on a non-parity dev-box cache and overstated; the
parity numbers reverse the mechanism from *model class* to *estimator*. Both
censuses are banked so the delta is auditable.

---

## 1. The gap, MEASURED at three independent sites

### 1.1 `g_tac` is never produced

| site | evidence |
|---|---|
| `stack/scripts/vlm_tac_compose.py:112-126` | the single `compose(...)` call passes **neither `vlm_goal` nor `ego_args`** |
| `stack/tanitad/lake/tac_str_labels.py:669-674` | ⇒ the goal branch takes `vlm_goal is None` ⇒ `ABSTAIN`, **every clip** |
| `stack/scripts/ph1_fuse.py:794-795` | emits `g_tac_lat`/`g_tac_lon` as `{"token": None, …}`; the reason string says verbatim *"NOTHING in this fuse derives them"* |
| `…/2026-08-19-alpamayo-screening/LABEL_PIPELINE_CONFIRMATION.md` §4 | *"the tactical-goal layer (`g_tac`) … is currently never produced at all"* |

Three probes, one answer. `PROVENANCE`: MEASURED (ours), file:line above.

### 1.2 ⭐ The second-order consequence nobody had recorded

`tac_str_labels.compose` gates the **ego argument tier** on the goal:

```python
if goal.value in (ABSTAIN, None) or not vlm_referent:
    flags.append("EGO_ARGS_DROPPED_NO_REFERENT")      # tac_str_labels.py:679-681
```

Since `goal.value` is `ABSTAIN` on 100 % of clips, **`ego_args` is dropped on
100 % of clips** — the arg slots of the whole vocabulary are structurally empty,
independent of whether the ego tier ever fires. The gate is correct; its input was
never supplied.

### 1.3 And it violates a binding spec rule

`HIERARCHY_VOCABULARY.md` §2: *"Every token must be **HINDSIGHT-DERIVABLE from
Engine A geometry alone** (VLM enriches; geometry guarantees) — so the vocabulary
works even where the VLM abstains."*

`g_tac` was reachable **only** through the VLM leg — the expensive one
(**25.6 T4-days** for the 4 522-clip target), the partially-run one, and the one
with **0/42 human-reviewed labels**. That is the opposite of a guarantee.

---

## 2. What now ships — the LON axis

`stack/tanitad/data/g_tac_geom.py` (+ `stack/tests/test_g_tac_geom.py`, 60 tests).

**Coverage — ON THE PARITY CORPUS** (`physicalai-train-e438721ae894-w120-256x640cyl`,
2 376-episode parity key; the cache holds 2 400 clip files), n = 69 447 windows,
d = 2 400, stride 5, run on Thor:

| `g_tac_lon` | n | share |
|---|---|---|
| `LON_UNCONSTRAINED` | 61 846 | **89.05 %** |
| `STOP_POINT(position_arc_m)` | 2 245 | **3.23 %** |
| `ABSTAIN` (each with a reason) | 5 356 | 7.71 % |

⇒ **92.28 % coverage.** The w120 val split (`physicalai-val-0c5f7dac3b11`, d = 35)
reproduces: `LON_UNCONSTRAINED` 91.03 %, `STOP_POINT` 2.56 %.

⚠️ The dev-box cache `physicalai-train-14231cd29c74` (d = 400, a DIFFERENT and
non-parity selection) gives 76.10 % / 6.01 % / 17.89 % — see §3.5 for why the two
corpora differ and why that matters more than the coverage delta.

**Controls, all passing:**

* **Analytic stopping distance.** `v0=10, a=-2` ⇒ `v0²/(2|a|) = 25.00 m`; the
  label reads **24.48 m**. Also `15/-3` → 36.75 vs 37.50 and `8/-1.5` → 20.87 vs
  21.33. Residual is the 0.1 s sampling of the stop instant, bounded by `v·dt`.
* **Ego-echo refusal.** An already-stopped ego **abstains** rather than emitting a
  `STOP_POINT` predictable from `v0` (the model is handed `v0` as the integrator
  constant). Pinned by `test_an_already_stopped_ego_abstains_rather_than_emitting_an_echo`.
* **Constant-only control.** `lon_majority_share = 0.761`, `degenerate_lon =
  False` — the deriver is not a constant.
* **Four caches.** parity-train (d = 2 400), parity-val w120 (d = 35), dev-box
  train400, dev-box val100 — all four non-degenerate, all four with the same
  token set.

⛔ `STOP_POINT`'s `reason` slot (`sign|light|queue|hazard`) is **left unset with
mask 0** — geometry sees the stop, never its cause. The `IGNORE` discipline from
`anchor_goal.py`, so the gap stays visible instead of being papered over.

---

## 3. ⛔ The LAT axis does not discriminate — and the mechanism is the ESTIMATOR

### 3.1 What was built

`CORRIDOR_OFFSET` against a **constant-curvature continuation**: the circular arc
leaving `(x_t, y_t, yaw_t)` with the curvature the ego carried over the preceding
1 s. The point was to cancel road curvature, which raw lateral displacement cannot.

**On synthetic paths it works exactly as intended:**

| control | result |
|---|---|
| constant-radius bend, R ∈ {50,100,200,500} m × v ∈ {5,10,30} m/s, both directions | offset **≤ 0.014 m**, 12/12 `LAT_UNCONSTRAINED` |
| the REFUTED `LANE_TARGET` gate on the **same** bends | fires **12/12** — up to **122.7 m** of "lateral displacement" on a gentle bend at 30 m/s |
| positive control: a real 3.5 m lateral step | recovered as **3.5000 m** |

### 3.2 ⛔ On real poses it does NOT discriminate

**The control caught it.** On the **parity** corpus (n = 69 447, d = 2 400) the
curvature-relative deriver fires at **65.63 %** and the refuted naive gate at
**63.07 %** — the new deriver fires *more* than the gate it was built to replace.
Realised `|lat_offset|` p50 = **2.012 m**, still ~2.7× the 0.75 m lane bar.
Reproduced on the w120 val split (d = 35): **64.40 %** vs **65.88 %**, p50
**1.915 m**.

⇒ **`CORRIDOR_OFFSET` as built is not admissible for supervision**, and the LAT
axis abstains.

### 3.2b ⚠️ RETRACTION, SAME PACKAGE — the mechanism I first published was WRONG

An earlier draft of this section concluded *"a single-arc ego-geometric corridor
reference is inadmissible at a 6 s horizon on PhysicalAI-AV, on every band"* and
attributed the failure to the **MODEL CLASS**. **That conclusion was measured on
the wrong corpus** — the dev-box cache `physicalai-train-14231cd29c74` (d = 400),
which is a different and NON-PARITY selection. On the parity corpus the oracle
ceiling is **more than 3× lower** and the model class is adequate at the median.

| the ORACLE circle (curvature fitted to the very path it measures) | dev-box train400 | **PARITY train2400** | w120 val35 |
|---|---|---|---|
| overall p50 residual | 1.891 m | **0.625 m** ✅ *below the lane bar* | 0.681 m |
| overall p90 | 10.534 m | **6.002 m** | 8.636 m |
| share with θ = \|κ\|·s < 0.05 | 25.6 % | **49.9 %** | 48.9 % |
| bands whose ceiling clears 0.75 m | 1 (25.6 % of corpus) | **2 (62.6 % of corpus)** | 1 (48.9 %) |
| n | 11 003 | **68 491** | 1 009 |

⇒ **The parity corpus is materially straighter driving**, and there the circular
model class is NOT the blocker.

⭐ **The corrected finding, and it is a better one.** The failure is the
**ESTIMATOR, not the model class**: on parity the oracle reaches **0.625 m** while
the deployable past-referenced deriver realises **2.012 m** — a **3.2× gap** that a
better curvature estimate could in principle close, where a model-class failure
could not. What is refuted is *"read the curvature off the preceding 1 s and
extrapolate 6 s"*, not *"a circular corridor"*.

`ROOT-CAUSE CLASS`: **a corpus-level conclusion drawn from a non-parity cache.**
The same family as the `df`/`tegrastats`/`memory.usage_in_bytes` traps — a probe
answering a different question than the one asked, read as the answer. The
CLAUDE.md rule *"absence found at ONE location is not absence"* has a twin:
**a distribution measured on ONE corpus is not the corpus.** Both censuses stay
banked so the delta is auditable.

⚠️ What survives from the first draft, unchanged and still measured:
* the **synthetic** controls (§3.1) — the reference does cancel a constant-radius
  bend to ≤ 0.014 m, and the naive gate false-positives on 12/12 of the same bends;
* the **past-window sweep** is flat over a 10× range — so the estimator gap is not
  closed by simply lengthening or shortening the window;
* the **curvature-change mechanism is FALSIFIED** (R² = 0.0025, shuffled control
  R² = 0.0003) — that first hypothesis was wrong on both corpora.

### 3.3 What this does and does not say about `LANE_TARGET`

`LANE_TARGET` was ruled out by PI adjudication on 2026-08-16 — *"a pure road curve
clears the gate at any highway speed"* — with **n = 18** and no mechanism.

✅ **Confirmed and strengthened:** the *raw lateral-displacement* gate is bad. On
synthetic constant-radius bends it false-positives **12/12** (up to 122.7 m of
"displacement" on a gentle bend at 30 m/s), and on the parity corpus it fires on
**63.07 %** of all windows.

⛔ **NOT established:** that no ego-geometric lateral derivation can work. The
parity oracle (0.625 m) says a circular corridor is adequate at the median; what
failed here is one specific ESTIMATOR of it. Do not cite this package as
*"lateral goals from ego geometry are impossible"* — it does not support that.

### 3.4 ⭐ E-LAT-1 — the road IS a clothoid; its parameters are NOT in the ego's past

**Pre-registered, both outcomes committed** before running
(`code/clothoid_vs_circle.py`, `WIN_RATIO = 0.75`). MEASURED on **parity**,
n = 63 016, d = 2 400, both controls passing:

| arm | p50 | p75 | p90 | |
|---|---|---|---|---|
| `circle_past` | 2.012 m | 6.077 | 14.269 | **DEPLOYABLE** |
| `clothoid_past` | **3.307 m** | 8.112 | 15.817 | **DEPLOYABLE — 1.64× WORSE** |
| `circle_oracle` | 0.687 m | 2.330 | 6.268 | ceiling |
| ⭐ **`clothoid_oracle`** | **0.085 m** | **0.193** | **0.380** | **ceiling** |

⛔ **Option (a) is REFUTED.** The deployable clothoid is **1.64×** the circle
against a pre-registered win bar of **≤ 0.75×**. The second parameter buys pure
variance when fitted to one second of past.

⭐ **AND THE ORACLE IS THE REAL FINDING. The road is a clothoid to 8.5 cm at the
median and 38 cm at p90** — a *two-number* model that essentially nails the
corridor. The deployable clothoid is **39× worse than its own oracle**.

⇒ **The information the LAT axis needs EXISTS and is TINY (two floats), and it is
NOT in the ego's past. It is in the world ahead.** That is the whole result, and
it reframes the fix: this is not a corridor-model problem, it is a *"where do the
two numbers come from"* problem.

**Controls** (both PASS): a constant-radius bend reads **0.00011 m** on all four
arms (a clothoid represents a circle exactly); a genuinely-clothoid path reads
**0.00014 m** on the clothoid oracle vs **4.46 m** on the circle oracle (the arms
are genuinely different, so the comparison is not vacuous).

⚠️ **The controls earned their keep twice.** `control_synthetic_bend` FAILED at
**0.4237 m on all four arms** through two attempted fixes. It was neither the
estimators nor the integrator: the synthetic generator derived yaw as `atan2` of
consecutive vertices, which is the tangent at the **segment midpoint**, a
half-step ahead of the vertex. That rotated the ego frame by `κ·ds/2` and, over a
~90 m band, manufactured a residual the size of the lane bar. Real corpus poses
carry a **measured** yaw channel and never had the artifact — but an uncontrolled
run would have reported a plausible, wrong number.
`ROOT-CAUSE CLASS`: **a synthetic control whose own discretisation is the size of
the effect it certifies.**

### 3.5 ⇒ Revised recommendation

| option | status after E-LAT-1 |
|---|---|
| (a) clothoid instead of circle | ⛔ **REFUTED** — measured 1.64× worse |
| ⭐ **(b) predict (κ₀, κ₁) from vision** | **now the strongest candidate** — E-LAT-1 gives it a **measured ceiling of 0.085 m**, the output is **two floats**, and curvature readout is already **P1-proven at R² 0.84** (`KIN-READOUT`) |
| (c) agent tracks as a road observation | still live, free, `obstacle.offline` on 97.44 % of corpus — and it unblocks four other tokens via `agent_slot` |
| (d) Engine C / SAM3 drivable surface | still live; most robust, needs the 32-clip residual + Colab budget |

⚠️ **The TL;DR's earlier line *"SAM3 is on the LAT axis's critical path"* is
withdrawn.** It was written under the model-class conclusion. SAM3 is one of four
routes and no longer the only one.

SAM3 state (MEASURED, `…/2026-08-16-sam3-dtype-fix/SAM3_DTYPE_FIX.md` §0):
**83 of 115 clips re-run**, 2 496 detections, 0 error strings; **32 clips still
carry the C77 payload**, residual named in `raw/residual_32_clips.json`, blocked on
free-Colab T4 budget.

⚠️ **Correction to the brief I was given:** it stated *"77/115 content-complete"*.
The primary source says **83**. `PROVENANCE`: the brief was INHERITED; the package
is MEASURED.

## 4. Corrections to the brief this stream was given

| brief said | primary source says |
|---|---|
| `SPEED_BAND` derives from "OCR speed-limit + corridor speed stats" | ⛔ **F-14 BLOCKER** (`v6.py:183-217`): sign **kind and text are FORBIDDEN**, not merely missing (`RETRACTION_LOG` C87; G1 closed 0/31), and **no corridor exists** (dataset card: *"we do not include open maps data"*). Worse, the two highest-confidence false positives are a **dashboard `30` roundel (0.927)** and a hoarding — a sign-derived target speed here would be **the ego speedometer arriving through the vision channel**. |
| "Alpamayo meta-action mapping table" is to be built | ✅ **already exists in code**: `stack/tanitad/lake/tac_str_labels.py` (`ALPAMAYO_LANE_TO_LAT`, `LON_ADMISSIBLE`, `lon_from_alpamayo`), PI-approved 2026-08-18 |
| SAM3 backfill 77/115 | **83/115** (§3.4) |
| `refc_v3_train.py --preflight` shows the ego-geometric subset | ✅ confirmed — and that subset is `refb_labels.goal_tac_targets`, a **continuous 4-vector (x, y, heading, speed) regression**, not a token at all |

---

## 5. ⚠️ Limits of this package — what it does NOT license

* **Parity — now MEASURED on parity.** The headline census ran on Thor against
  `physicalai-train-e438721ae894-w120-256x640cyl` (the parity key; the cache holds
  2 400 clip files against the 2 376-episode key — the 24-file delta is NOT
  reconciled here and is flagged for whoever quotes an episode count). It
  **selects no episodes** and produces a distribution, so it cannot break parity.
  The dev-box censuses are retained as the retraction's evidence, not as corpus
  statistics.
* **Thor's `stack/` is at `30d6d60`, drifted from this branch.** The two modules
  were copied in and verified with a **real import** plus a **vocabulary pin
  check** (`TACTICAL_GOAL_TOKENS`, LAT/LON partitions, `TAC_BAND_S`, `DT` all
  identical). The census is read-only; nothing on Thor was trained or modified
  beyond those two files.
* **No correctness claim.** §2 measures **coverage and provenance**, not accuracy.
  No human has reviewed a `STOP_POINT` label.
* **The echo control is owed.** Any head trained on these labels must clear the
  **v0-shuffle echo control** before its number is quotable — the same requirement
  `tac_str_labels` places on `LON_FROM_EGO_KINEMATICS`. The already-stopped
  abstention removes the worst case, not the whole family.
* **`LAT_OFFSET_MIN_M = 0.75` is still DECLARED, not calibrated.** It is now moot
  for production (the LAT axis abstains), but it sets the bar in §3.2's table.

---

## 6. ⛔ ESCALATION — three decisions for the PI

1. **The LAT axis needs the road's two clothoid parameters from SOMEWHERE
   OTHER THAN THE EGO'S PAST (§3.4).** The free extrapolation option is now
   refuted by measurement. **My recommendation: (b) predict (κ₀, κ₁) from vision**
   — E-LAT-1 hands it a measured ceiling of **0.085 m**, the target is two floats,
   and `KIN-READOUT` already reaches R² 0.84 on curvature. Second choice (c) agent
   tracks, because it is free and unblocks four other tokens.
   ⚠️ **This blocks the tokenizer freeze**, which `HIERARCHY_VOCABULARY.md` §7
   requires *before any v6 head is built*.
2. **`vlm_goal`/`ego_args` are never passed** (§1.1-1.2). Wiring
   `g_tac_geom` into `vlm_tac_compose` is a small change but it changes the label
   stream's schema. Confirm before I wire it.
3. **Thor is IDLE** — GPU 0 %, no `thor_sweep.sh` and no trainer process running
   (probed 2026-08-23 00:0x local). My brief said a 4-arm sweep was in flight; it
   is not. **That is a fleet fact for the orchestrator, not my stream** — but a
   GPU has been sitting free.

---

## 7. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `g_tac_geom.py` — the deriver (LON ships, LAT refuted+preserved) | `repo:stack/tanitad/data/g_tac_geom.py` | no — staged |
| `test_g_tac_geom.py` — 60 tests incl. the bend + oracle controls | `repo:stack/tests/test_g_tac_geom.py` | no — staged |
| `g_tac_geom_census.py` — re-runnable census w/ 4 mandatory controls | `repo:stack/scripts/g_tac_geom_census.py` | no — staged |
| ⭐ **E-LAT-1** clothoid-vs-circle on parity (n = 63 016 — the §3.4 decision) | `repo:…/raw/E_LAT_1_clothoid_vs_circle_PARITY.json` **and** `thor:~/gtac/` | no — staged |
| ⭐ **E-LAT-1** probe source (pre-registered, portable, 2 controls) | `repo:…/code/clothoid_vs_circle.py` **and** `thor:~/gtac/` | no — staged |
| ⭐ census, **PARITY 2400**, refuted arm (n = 69 447 — the headline evidence) | `repo:…/raw/census_PARITY2400_refuted_arm.json` **and** `thor:~/gtac/` | no — staged |
| ⭐ census, **PARITY 2400**, production arm (the shipping LON distribution) | `repo:…/raw/census_PARITY2400_production.json` **and** `thor:~/gtac/` | no — staged |
| census, **w120 val35** (parity-side second probe) | `repo:…/raw/census_VAL35_refuted_arm.json` **and** `thor:~/gtac/` | no — staged |
| census, train400, refuted arm (dev-box; the RETRACTION's evidence) | `repo:…/raw/census_train400_refuted_arm.json` | no — staged |
| census, train400, production arm (dev-box) | `repo:…/raw/census_train400_production.json` | no — staged |
| census, val100 (dev-box second cache) | `repo:…/raw/census_val100_refuted_arm.json` | no — staged |
| the two modules, copied onto Thor for the parity run | `thor:~/TanitAD/stack/tanitad/data/g_tac_geom.py`, `thor:~/TanitAD/stack/scripts/g_tac_geom_census.py`, `thor:~/gtac/` | no — identical to the staged repo copies |
| census run log | `repo:…/raw/census_train400_refuted_arm.log` | no — staged |
| this document | `repo:…/2026-08-22-g-tac-geometry-floor/G_TAC_GEOMETRY_FLOOR.md` | no — staged |

**Nothing in this package lives on only one disk.** Everything is in the repo
working tree and staged. Per the operating standard: **staged, never committed,
never pushed.**
