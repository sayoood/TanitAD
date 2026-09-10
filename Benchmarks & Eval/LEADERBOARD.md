# TanitAD Leaderboard

> ⭐ **AMENDED 2026-09-03 — three additions, on the PI's direct instruction to *"document the results
> in the leaderboard even if the old includes partial criteria."*** Nothing below §1d was
> re-measured and **no old row was retro-filled**:
> * **§0.8 — the `loop` column (open / closed)**, from the PI ruling of **2026-09-02**. ⛔ **Every row
>   on this page except §3 / §5.5 is `OPEN`**, including every T1 row, and **13 sites on this page
>   plus `EVAL_DOCTRINE.md:11` call a T1 quantity *"closed loop"***. They are enumerated with
>   file:line and **deliberately NOT edited here** — a separate stream owns that correction.
> * **§0.9 — CRITERIA COVERAGE**: what this schema records, the **six criteria it has no column for**
>   (`loop`, `ha0`, STRATEGIC, TACTICAL, distance-keeping, the LAT per-window reducer), and a
>   row-by-row statement of what is quotable **as-is** versus what needs a re-measurement.
> * **§1d — the B1-corpus line (`refcv3`, `refav1`)**. ⛔ refcv3's four families are **PENDING with
>   the filename that fills them**, because `taniteval/results/refcv3-30k-openloop-*.json` **does not
>   exist yet** (two differently-bound probes). ⛔ **Nothing was invented to fill a cell**; §1d.3
>   states, from source, the three reasons the numbers that *do* exist may not be promoted.
> * New work items **W-18 … W-22** in §12.1.
>
> ⭐ **Corroboration, arrived mid-write:** a sibling stream minted **`MODEL_REGISTRY.md` §4.5
> `refcv3-b1-v72-30k`** while §1d was being written, and it reaches the **same pending verdict
> independently** — ⛔ *"NO EVAL RESULT EXISTS"*, all four families **NOT MEASURED**, and the **same**
> artifact filename named as the thing that closes them. It also **settles the parameter question**:
> **107,032,901 parameters**; 107,082,365 is the **all-tensor** count (544 tensors, of which 201
> buffers = 49,464 elements). ⚠️ §1d.5 records that this page's own registry-absence claim went
> **half stale within the hour** — the repo advances mid-session, and that correction is kept visible
> rather than quietly applied.

*Rebuilt **2026-08-23** by the EvalFlyWheel from `Project Steering/MODEL_REGISTRY.md` and the raw
eval artifacts. **Only two sources are quotable here: the registry and raw eval JSON.** No number on
this page is transcribed from a summary, changelog, weekly report or `PROJECT_STATE.md`; where a
prose doc and the registry disagree the registry wins and the conflict is reported in §11 / §12.*

> ⚠️ **Staleness this rebuild closes.** The page header said *"Rewritten 2026-07-21"* while the body
> carried un-headlined patches dated 2026-08-02, 2026-08-16 and 2026-08-17 — so the banner
> **understated** how much had moved and **overstated** how coherent it was. The content gap was
> larger still: **the entire v5f / v5.8f / v1arch / unicycle-readout / T1 campaign (2026-08-05 →
> 2026-08-12) had no row anywhere on this page.** Full audit: §11.
> ⛔ **The structural defect this rebuild fixes** is not staleness: §1–§4 were headed *"Driving
> capability … the standard read"* while every number in them is **TIER T0, teacher-forced** —
> which `Project Steering/EVAL_DOCTRINE.md` forbids being quoted as driving performance. See §0.1.

**Regenerate the T0 canonical driving tables — CPU-only, no GPU, no pod, seconds:**
```
python -m taniteval.runner driving-all          # recompute every arm with a windows_*.pt
python -m taniteval.driving --leaderboard       # emit §1's markdown table
```
✅ **Both commands VERIFIED BY CONTENT 2026-08-23** (they were not checked at the last rebuild):
`driving-all` is registered at `taniteval/taniteval/runner.py:496`; `--leaderboard` at
`taniteval/taniteval/driving.py:1113`, emitting `leaderboard_md()` (`driving.py:1063`).
⛔ **Neither command can regenerate §1a (T1), §3 (T2) or the w120 / OOD-val blocks** — those come
from `taniteval/tools/t1_eval.py`, `taniteval/tools/ff_rescore.py`,
`taniteval/tools/eval_four_families.py` and per-campaign gate emitters, and are banked as raw JSON
under `TanitAD Research Lab/…/Implementation/incoming/`. A "regenerate the leaderboard" that runs
only the two lines above rebuilds **one tier of three**.

---

## 0. How to read this page — tiers, units, estimator, floors (binding)

### 0.1 ⛔ TIERS — the first thing to read on any row (`Project Steering/EVAL_DOCTRINE.md`, PI 2026-08-07)

| tier | condition | may be quoted as |
|---|---|---|
| **T0** | **teacher-forced** — the predictor consumes the **recorded future actions** | *"prediction quality"* — ⛔ **NEVER "driving performance"** |
| **T1** | **action-closed loop** — the predictor consumes the **decoder/planner's own actions**; perception context fixed at t0 | *"closed-loop (imagination) driving"* — **the PRIMARY offline eval** |
| **T2** | **perception-closed loop** — AlpaSim / NuRec re-render, or real-footage log-replay | *"closed-loop driving"* |

**A capability claim ("drives", "handles", "improves driving") requires T1 or better.** T0 supports
attribution and prediction claims only. **Every row below carries its tier.**

⛔ **THE CANONICAL 881-WINDOW LEADERBOARD (§1–§2, §4–§5) IS TIER T0.** Verified from source, not
from prose: `taniteval/taniteval/rollout.py:184` feeds `ep.actions[t+window : t+window+fwd_k]` — the
**expert's future actions** — into `rollout_decode`, and the module's own docstring
(`rollout.py:151-160`) says the result is *"a world-model fidelity decode of a known control
sequence … it just may not be quoted as driving or as hierarchy"*, stamping
`actions_source="expert_future"` and `pc2_pass = False` **by construction** (`rollout.py:237`).
⇒ **Every ADE in §1, §2, §4 and §5 is a T0 number**, and the previous heading of §2 — *"Driving
capability … the standard read"* — was a doctrine violation. It is restamped here, the numbers are
unchanged.

⚠️ **NAME COLLISION, and it caused the violation.** `taniteval.driving` emits
`BLOCK = "taniteval.driving/tier0"` and `surface: "4 waypoints 0.5 s apart (tier-0)"`. That
**"tier-0" is the METRIC-SUITE tier** (the sparse 4-knot surface vs the dense tier-1 surface) and is
a **different axis** from EVAL_DOCTRINE's T0/T1/T2. Both happen to read "tier 0" for these rows, but
for unrelated reasons. **Never infer an EVAL_DOCTRINE tier from the `BLOCK` string.** On this page,
`T0/T1/T2` always means the EVAL_DOCTRINE tier; the metric surface is written out in words.

### 0.2 UNITS

§1–§5 rows are **metric-BEV ego-frame `ade_0_2s`, metres**, averaged over the waypoints at
0.5/1/1.5/2 s. They are **not** the camera-frame `ADE@1s` of the 2026-07-12 D1 gate (§8) and the two
must never be compared. §1a's `ade_dense_m` is a **dense 20-step 10 Hz** mean and is a **different
reducer on a different grid** — do not difference it against `ade_0_2s`. Speeds m/s, accelerations
m/s², headings degrees, yaw-rates °/s, curvature 1/m, latency ms.

### 0.3 CORPORA — there are now FOUR, and they are never mixed

| # | corpus | grid | used by |
|---|---|---|---|
| **C1** | `physicalai-val-0c5f7dac3b11` (256×256 pinhole) | **881 windows / 40 episodes**, window 8, stride 8, K=20 @10 Hz, `nav=follow`, operative step intent-free | §1, §1b, §1c, §2, §4, §5 (**all T0**) |
| **C2** | `physicalai-val-0c5f7dac3b11-w120-256x640cyl` (cylindrical, HFOV 120°, subframe 176×624) | **881 windows / 600 episodes**, stride 8 | §2.5 (v5f / v5.8f, **T0**) |
| **C3** | same w120 cache, **stride 1** | **6,844 windows / 40 episodes**, dense 20-step | §1a (**T1** + its T0 control) |
| **C4** | `physicalai-oodval-6f4b94e4c7ce-q90` — PhysicalAI-AV's **own official eval split**, 290 clips, zero training overlap | **6,382 windows / 290 clusters** (v1arch), **6,834 windows / 40 eps stride-1** (unicycle line) | §2.6, §2.7, §1a.4 |

⛔ **C1's 881 and C2's 881 are DIFFERENT 881s** (40 episodes vs 600). The coincidence has already
misled once; always carry the corpus letter.

### 0.4 ESTIMATOR

The decision-grade interval is the **episode-cluster bootstrap** over the val episodes
(`taniteval/ci.py`, B = 2000); for two arms or an arm-vs-floor on the same windows it is the
**paired** form. The legacy `heldout ± ci95` — historically mislabelled *"8-split episode-disjoint
jackknife"* — is `overlapping_holdout_se`; it is **1.107–3.100× too narrow, median 1.499×** over
**27 dumps = 25 distinct arms**, and its central value is a **mean-of-split-means** that shifts the
point estimate **−6.67 % to +11.69 %, bidirectionally** (MODEL_REGISTRY §6; blast radius
`…/incoming/2026-07-25-jack-blast-radius/`). It appears on this page **only** in the column
explicitly marked DEPRECATED, so published figures stay traceable;
`taniteval/driving.py` *refuses* to emit it
(`estimator.deprecated_and_refused: "overlapping_holdout_se"`, present in every `driving_*.json`).
⛔ **No verdict, ranking or gate on this page rests on a `heldout` value.** The 2026-08-16 gate
re-drive (`…/incoming/2026-08-16-jack-in-gates/`) re-decided G1 and G4 under the correct estimator
and **neither flipped**; the one banned verdict still standing is `planner_beats_cv` (§1, P2 row),
carried as **UNDECIDED**, not as ✗.

⚠️ **The 1.107–3.100× band supersedes the "1.28–2.06× across 10 arms" that §0 of this page used to
print** — the older band was never wrong, only under-sampled. Corrected here 2026-08-23.

### 0.5 win / tie / LOST is three-way on purpose

A paired interval that excludes zero while favouring the **floor** means the trivial baseline beat
the model. ⛔ **This rule was being violated by §1's own `beats CV` column**, which rendered
two-way ✅/✗ and printed **✗ for two arms the raw JSON calls `tie`** (`refb-10k` and `refb`; see
§11). Fixed below: the column is now three-way and reads `favours` straight out of
`driving_<key>.json → verdict.ade_vs_cv`.

### 0.6 FLOORS on C1 (881 windows)

MEASURED. Rows 1–2 from `taniteval/results/driving_flagship-30k.json → floor_values` (re-read
2026-08-23, exact); rows 3–4 recomputed 2026-08-02 on these exact windows —
`TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-08-02-ctrv-floor/raw/ctrv_readjudication.json`.

| floor | ADE@2s m | FDE@2s m | miss@2m | speed MAE m/s | \|along\| m | \|cross\| m | heading° | κ-sign | wins/881 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant velocity (CV) | 0.8377 | 1.7406 | 0.3042 | 0.4678 | 1.0955 | 1.0089 | 6.623 | 0.6103 | 156 |
| hold-v0 (straight at entry speed) | 0.7876 | 1.6521 | 0.2917 | 0.4818 | 1.1040 | 0.9137 | 6.344 | 0.5119 | 302 |
| ⭐ **CTRV, speed-gated 2 m/s** | **0.5265** | **1.1272** | **0.1896** | 0.4682 | 0.9382 | **0.3741** | **3.679** | — | **423** |
| CTRV, ungated | 0.5230 | 1.1194 | 0.1896 | 0.4683 | 0.9347 | 0.3667 | 3.333 | — | — |
| *best-of-3 **ORACLE*** (privileged per-window min) | *0.4820* | — | — | — | — | — | — | — | — |
| no-vision ego-status ceiling (AD-MLP repro) | *0.5735* | — | — | — | — | — | — | — | — |

⭐ **CTRV IS THE FLOOR, and it is still missing from the gate.** `taniteval/driving.py:304` scores
every arm against `FLOORS = ("cv", "holdv0")` — **both straight lines**. CTRV is admissible under
the identical information budget (`poses[last]`, `poses[last-1]`, no future), is already computed on
every window by `driving_diagnostic.baseline_waypoints` and discarded by `rollout.collect`, and it
beats both incumbents: paired **CTRV − CV = +0.3113 m [0.1674, 0.4844] separated**,
**CTRV − hold-v0 = +0.2611 m [0.1419, 0.4061] separated**. It wins **423 of 881** windows outright
(CV 156, hold-v0 302). ⇒ **every lateral / turn / curvature verdict published against the two-floor
family carries a CV-derived magnitude that is ~5× too generous** — see §1b. Patch + 11 tests,
validated end-to-end, still **UNMERGED** at `…/incoming/2026-08-02-ctrv-floor/` — §12 W-2.

*hold-v0 is the strongest trivial **longitudinal** floor and the one VTARGET provably loses to at
2 s (MAE 1.65 vs 0.475, MODEL_REGISTRY §4.1). **best-of-N is an ORACLE** — it picks the winner per
window using ground truth, like the nuScenes `PhysicsOracle`. A reference, never a competitor.*

⚠️ **The T1 floor is a DIFFERENT floor and it is `ha` (hold-action), not CV.** See §1a — a control
that merely holds the last action beats both closed-loop arms by 10–25×. **Clearing a T0 floor is
not evidence of anything driving-related** (EVAL_DOCTRINE rule 2).

### 0.7 THE FOUR METRIC FAMILIES (binding, PI 2026-08-02)

Every eval reports **LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC**, per family, never pooled,
never ADE alone. Each family carries its estimator and CI on the same windows as the ADE it
accompanies. **A family that cannot be computed is reported `NOT MEASURED` with its reason and its
n — a WORK ITEM, never a silent omission.** Family coverage on this page:

| block | tier | LONG | LAT | TACTICAL | STRATEGIC |
|---|:--:|:--:|:--:|:--:|:--:|
| §1a T1 (C3) | T1 | ✅ (distance-keeping ❌) | ✅ | ✅ | ❌ n/a-corpus |
| §1c T0 vs floors (C1) | T0 | ✅ (distance-keeping ❌) | ✅ | ❌ | ❌ |
| §2.5 v5f / v5.8f (C2) | T0 | ✅ (distance-keeping ❌) | ✅ | ❌ | ❌ |
| §2.6 v1arch OOD-val (C4) | T0 | ✅ **incl. distance-keeping** | ✅ | ✅ | ⚠️ measured **and it is a constant predictor** |
| §1 / §2 / §3 / §4 canonical (C1) | T0 | partial | partial | ❌ | ❌ |
| §3 T2 AlpaSim / low-OOD | T2 | ❌ | ⚠️ corridor only | ❌ | ❌ |
| **§1d refcv3** (B1 corpus) | T1 **UNRULED** | ⏳ PENDING (distance-keeping ⚠️ partial by construction) | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING |
| **§1d refav1** (B1 corpus, step-1000) | T1 | ✅ **incl. distance-keeping** | ✅ | ✅ | ❌ `families_unavailable` |
| **§1d refcv4b** (B1 corpus) | T1 | ✅ **incl. distance-keeping** (n 1,224) | ✅ | ✅ | ✅ **route head, n 3,622** |
| **§1d refcv5-v2** (B1 corpus) | T1 | ✅ **incl. distance-keeping** (n 1,308) | ✅ | ✅ | ✅ **route head, n 3,622** |

Every ❌ above is enumerated as a work item in §12. ⏳ **PENDING** is a cell whose eval has not run —
distinct from ❌ (measured as absent) and from **NOT MEASURED** (no instrument). §1d names the exact
file that fills each pending cell.

### 0.8 ⛔ THE LOOP COLUMN — open vs closed (PI ruling **2026-09-02**, BINDING) · *and it makes this page's own T1 headers wrong*

**The ruling:** a model consuming **its own planner's output** is **STILL OPEN LOOP**, because its
trajectory **does not affect the incoming ego data**. **TRUE closed loop** means the trajectory
**drives the vehicle** — in simulation (AlpaSim / NuRec) or in a real vehicle.

⇒ **Loop is a SECOND AXIS, orthogonal to the T0/T1/T2 tier**, and every row must carry it as its own
column. The tier says *what the predictor consumes*; the loop says *whether the prediction moves the
world it is then measured against*.

| tier | what the predictor consumes | **loop (2026-09-02 vocabulary)** |
|---|---|---|
| **T0** | the recorded future actions (teacher-forced) | **OPEN** |
| **T1** | the decoder/planner's **own** actions; perception context fixed at t0 | ⚠️ **OPEN** — the plan never reaches the ego data |
| **T2** | AlpaSim / NuRec re-render, or a real vehicle | **CLOSED** |

⛔ **EVERY ROW IN §1, §1a, §1b, §1c, §1d, §2, §4 AND §5 OF THIS PAGE IS `OPEN`.** Only §3 and §5.5
(T2, AlpaSim NuRec) are `CLOSED`. §5.5's *real-footage log-replay* line is ⚠️ **UNRULED** under the
new vocabulary — a replayed log does not respond to the plan either; that specific call is escalated,
not decided here.

⚠️ **THE NUMBERS ARE UNAFFECTED. THIS IS A NAMING DEFECT, NOT A MEASUREMENT DEFECT.** ⛔ **A separate
stream owns the correction — the mislabelled headers below are NOT edited by this pass**, they are
enumerated so the correction cannot be missed. Every site is a place where a **T1** quantity is
called *closed loop*:

⚠️ **Line numbers are AS OF THIS AMENDMENT (2026-09-03), after §0.8/§0.9/§1d were inserted, and they
were re-derived AFTER the last structural edit.** Inserting those sections shifted every citation
below its own insertion point — the first draft of this table carried **pre-edit** numbers and was
wrong by +16 / +100 / +278 depending on the region, and a second draft was wrong again by the size of
this very paragraph. ⇒ **the quoted PHRASE, not the integer, is the durable identifier**; re-derive
the integer before acting on it.

| file:line | the mislabel |
|---|---|
| `Project Steering/EVAL_DOCTRINE.md:11` | ⛔ **the doctrine source itself** — T1 defined as *"action-closed loop"*, may be quoted as *"closed-loop (imagination) driving"* |
| `Benchmarks & Eval/LEADERBOARD.md:65` | §0.1's tier table, T1 row — same two phrases, inherited from the doctrine |
| `…LEADERBOARD.md:166` | *"beats both closed-loop arms by 10–25×"* (T1 arms) |
| `…LEADERBOARD.md:282` | §1a's **section header** — *"TIER T1 — ACTION-CLOSED LOOP"* |
| `…LEADERBOARD.md:294` | *"`cl` = action-closed loop (**T1**)"* |
| `…LEADERBOARD.md:308` | *"THE PROGRAMME HAS NO CLOSED-LOOP DRIVING COMPETENCE AT T1"* |
| `…LEADERBOARD.md:396` | *"the closed-loop harness once published `route_head_eq_logged`"* (a T1 harness) |
| `…LEADERBOARD.md:411` | §1a.6's **table header** — *"v1.6 closed (T1)"*, *"v1.7 closed (T1)"* |
| `…LEADERBOARD.md:419`, `:421` | *"the closed loop ~5 %"*, *"closed-loop, the stack drives near-straight"* |
| `…LEADERBOARD.md:457` | `planner_p2` row — *"the *closed-loop* windows ARE banked"* |
| `…LEADERBOARD.md:525` | *"S-curve reproduction 97.9 % → 5 % closed-loop"* |
| `…LEADERBOARD.md:575`, `:577`, `:584` | the standing G-B1 footnote — *"Open-loop ⊥ closed-loop"*, *"imagination-closed-loop **1.7318**"*, *"the closed-loop failure was UNDERSTATED"* |
| `…LEADERBOARD.md:1413`–`:1414` | §11 — *"the §1.12 decoder-conditioned closed loop"*, *"the T1 pseudo-closed-loop campaign"* |

✅ **NOT mislabels, listed so the correction stream does not over-reach:** `:66` (the T2 row of §0.1),
`:975` (§3's header) and `:1121`–`:1197` (**genuinely T2**, AlpaSim NuRec / real-footage §5.5), and
`:1354`+ (**external** CARLA/Bench2Drive §9.3); `:409` and `:583` are **artifact FILENAMES**
(`closed_loop_analysis.json`, `closedloop_flagship-30k.CORRECTED.json`) and renaming them would break
provenance; *closed* is used as a **verb** at `:558`, `:1084`, `:1160`, `:1301`, `:1438`, `:1440`
("closed at source", "R12 closed 2026-07-21", "K-step closed the ratio"); `:1344` is the external
method name **PDM-Closed**.

### 0.9 ⛔ CRITERIA COVERAGE — what this page's schema RECORDS and what it OMITS

*Added 2026-09-03 on the PI's direct instruction — **"document the results in the leaderboard even if
the old includes partial criteria."** The gap is written down rather than papered over. ⛔ **No old row
is retro-filled: an unmeasured cell stays unmeasured.***

**Column-level gaps — criteria the schema has NO COLUMN for, on ANY row:**

| # | omitted criterion | binding since | status on this page |
|---|---|---|---|
| **G-1** | ⛔ **the `loop` column (open / closed)** | PI 2026-09-02 | **ABSENT from every table.** §0.8 supplies the mapping; §1d is the first block to carry the column inline. Retro-fitting §1–§5 is **zero-GPU** — it is a lookup on the tier, not a re-measurement (**W-18**) |
| **G-2** | ⭐ **the `ha0` trivial-baseline column** (constant velocity at the **measured** `v0`; `a = 0, κ = 0`) | `D-REFAV1-HA0-ARM`, `t1_eval.py:148-153` | **ABSENT from every row on this page.** The page's floors are CV / hold-v0 / CTRV (**T0**, §0.6) and `ha` (**T1**, §1a). `ha0` is neither: `ha` holds the last *observed* `(a, κ)`, which drifts ~0.12 m even where the human drives straight, so **an arm can beat `ha` by doing nothing at all**. ⛔ Until `ha0` is beside each number, "beats the trivial baseline" is unproven at T1 (**W-19**) |
| **G-3** | **STRATEGIC** | PI 2026-08-02 | **omitted on every PhysicalAI block** — §1a.5, §1c, §2.5. §2.6 is the sole measured one, *and it measured a constant predictor*. Blocked on a **CORPUS** fact, settled at 5 probes (§12 W-3) |
| **G-4** | **TACTICAL** | PI 2026-08-02 | **omitted on C1 and C2** (§1c, §2.5) — the scored pass is teacher-forced, so no manoeuvre decision is decoded. Present on C3 (§1a.4) and C4 (§2.6) (§12 W-4) |
| **G-5** | **LONGITUDINAL distance-keeping** (headway / time-gap / TTC) | PI 2026-08-02 | **omitted on C1, C2 and C3** (§1a.2, §1c, §2.5) — a **JOIN** gap, not a data gap. Closed on C4 (§12 W-1) |
| **G-6** | **per-window reducer for the three LAT metrics** | §1c | heading / yaw-rate / curvature carry **point estimates with the interval REFUSED** on C1, and the same two-reducer hazard is **unflagged at T1** (§12 W-5) |

**Row-level status — which existing rows are quotable AS-IS, and which need a re-measurement:**

| row family | quotable as-is? | what a re-measurement would have to add |
|---|---|---|
| §1 canonical T0 ADE (21 arms) | ✅ **YES, within T0** — `full_set` point + episode-cluster CI + 3-way paired verdict, all re-verified by content 2026-08-23 | `loop = OPEN` (lookup, **W-18**); `ha0` is **not applicable** at T0 (its floors are CV/hold-v0/CTRV, and CTRV is the correct one — **W-2, still unmerged**) |
| §1b CTRV re-adjudication (25 arms) | ✅ **YES** — paired, bit-exact alignment verified | nothing; it is the correct T0 floor read |
| §1a T1 (`stage-a-repaired`, `v5f-30k`) | ⚠️ **PARTLY** — the four families are present and CI'd, but **no `ha0` arm exists in those dumps** | ⛔ **a re-run with `ha0`** — the whole 22–25× headline is stated against `ha`, and `ha − ha0` was MEASURED **not separated** on a fixture, so the margin over the *true* floor is unknown. GPU |
| §1a.6 T1 (v1.6 / v1.7, C4) | ⚠️ **PARTLY** — same `ha0` gap, plus the table header mislabels T1 as *closed* | `ha0` + the loop column |
| §2 split read (21 arms) | ⚠️ **PARTLY** — 6 rows have ADE + verdict but **blank along/cross/speed/heading/κ** | a **zero-GPU** re-render; `driving-all` already emits them (§12 W-6) |
| §2.6 `flagship-v1arch-v2bal-30k` (C4) | ✅ **YES — the only COMPLETE four-family block on the page** | `ha0` + the loop column |
| §2.5 v5f / v5.8f (C2) | ⚠️ TACTICAL, STRATEGIC and distance-keeping all omitted | three families |
| §3 / §5.5 T2 AlpaSim | ✅ **YES, and they are the page's only genuinely `CLOSED` rows** | LONG / TACTICAL / STRATEGIC all omitted; §5.5 is a **2-second** number and must never be quoted unqualified |
| §8 camera-frame gate ladder | ⛔ **NO — SUPERSEDED, different unit.** Never compare to §1 | historical only |
| **§1d refcv3 / refav1 (B1 corpus)** | ⛔ **NO — the four families are PENDING.** See the row's own cells | the eval itself; §1d states the exact filename that fills each cell |
| **§1d.6 refcv4b / refcv5-v2 (B1 corpus)** | ✅ **YES — all four families, paired episode-cluster CIs, on the SAME 4,823 windows / 141 episodes, with `ha` / `ha0` / `ha0_ext` floors and 5/5 constant-arm checks** | a **training-seed** replicate (`H-ESTIM-SEED-1`); `anchor_selection` needs a fan+selector surface; the lead block wants rebuilding on the arm's own grid (`--dt 0.5 --k 4`) |

---

## 1a. TIER **T1** — ACTION-CLOSED LOOP · ⭐ **THE PRIMARY EVAL** ⭐

*MEASURED 2026-08-11 ~23:27Z (rollout) / 2026-08-12 ~00:55Z (four-family rescore). Corpus **C3** —
6,844 windows / 40 val episodes, stride 1, dense 20 steps @ 10 Hz. Instruments
`taniteval/tools/t1_eval.py` (`--v2-val-cache --grounding-readout`) and
`taniteval/tools/ff_rescore.py`. Point estimates are `full_set` pooled means; intervals are the
**episode-cluster bootstrap** (B = 2000, 40 clusters); cross-arm deltas are the **paired** form.
`overlapping_holdout_se` is used nowhere in this block. **Evidence class: MEASURED (ours).**
Artifacts, all banked in-repo and md5-verified against the release manifest:
`TanitAD Research Lab/Benchmarks & Evals/Implementation/incoming/2026-08-18-v58f-artifact-banking/gates/four_families/ff_{stageA,v5f30k}_{cl,ol,ha}.json`
(+ `ff_comparison.full.json`); registry anchor **MODEL_REGISTRY §1.14**.*

**Three surfaces per arm.** `cl` = action-closed loop (**T1**) · `ol` = teacher-forced (**T0**, a WM
diagnostic, never driving) · `ha` = hold-action control (**T1**). All three on identical windows.

### 1a.1 T1 headline

| arm | surface | **tier** | ADE dense m [ep-cluster CI95] | FDE last m | evidence | artifact |
|---|:--:|:--:|---|---:|---|---|
| `stage-a-repaired` | **`cl`** | **T1** | ⛔ **9.3697** [6.6822, 12.2576] | 19.5256 | MEASURED 2026-08-11 | `ff_stageA_cl.json` |
| `stage-a-repaired` | `ha` | **T1** *(control)* | ⭐ **0.4246** [0.3500, 0.5132] | 1.2242 | MEASURED 2026-08-11 | `ff_stageA_ha.json` |
| `stage-a-repaired` | `ol` | T0 *(diagnostic)* | *0.3659* [0.2926, 0.4521] | 1.0231 | MEASURED 2026-08-11 | `ff_stageA_ol.json` |
| `v5f-30k` | **`cl`** | **T1** | ⛔ **23.9837** [21.4420, 26.3470] | 53.4756 | MEASURED 2026-08-11 | `ff_v5f30k_cl.json` |
| `v5f-30k` | `ha` | **T1** *(control)* | **0.9597** [0.8361, 1.0879] | 2.8631 | MEASURED 2026-08-11 | `ff_v5f30k_ha.json` |
| `v5f-30k` | `ol` | T0 *(diagnostic)* | *0.9397* [0.8162, 1.0679] | 2.8003 | MEASURED 2026-08-11 | `ff_v5f30k_ol.json` |

⛔⛔ **THE PROGRAMME HAS NO CLOSED-LOOP DRIVING COMPETENCE AT T1, AND THIS IS THE PAGE'S HEADLINE.**
The hold-action control beats the repaired arm **22×** (0.4246 vs 9.3697) and the v5f arm **25×**
(0.9597 vs 23.9837). Within-arm paired `cl − ol`: **+9.0039 [6.3659, 11.8487]** (repaired) and
**+23.0439 [20.5613, 25.3884]** (v5f), both separated at `p_delta_gt0 = 1.0`. **The same checkpoint
on the same windows reads 0.3659 at T0 and 9.3697 at T1 — a 25× gap.** That single row is why the
tier doctrine exists, and why no T0 number anywhere on this page may be read as driving.

**The stage-A repair wins on every surface, separated.** Paired `stage-a-repaired − v5f-30k`, same
windows: `cl` ADE **−14.6139 [−16.9319, −12.2010]** (`p_delta_gt0` 0.0), `ol` **−0.5739 [−0.7002,
−0.4570]**, `ha` **−0.5351 [−0.6644, −0.4181]**; `cl` LON speed MAE **−17.2064 [−19.7815,
−14.4927]**. The repair targeted action-response gain (0.27 → 0.971/0.966, longitudinal sign 1.0 —
MODEL_REGISTRY §1.13c) and improves **exactly the axis it targeted**.

### 1a.2 T1 — LONGITUDINAL

| metric | `stageA·cl` | `stageA·ha` | `v5f·cl` | `v5f·ha` |
|---|---:|---:|---:|---:|
| speed MAE m/s [CI95] | **9.7291** [7.1431, 12.5948] | 0.5671 [0.4815, 0.6615] | **26.9356** [24.8280, 28.8900] | 1.4531 [1.2837, 1.6287] |
| speed bias m/s *(+ = too fast)* | ⛔ **+9.3892** | −0.0435 | ⛔ **+26.5931** | −1.2149 |
| along-track MAE m [CI95] | **9.2655** [6.5869, 12.1362] | 0.3487 [0.2865, 0.4210] | **23.8965** [21.3555, 26.2788] | 0.8901 [0.7758, 1.0098] |
| along final bias m | **+18.5801** | — | — | — |
| accel MAE m/s² | ⛔ **19.0948** *(> 1.9 g — a true blow-up)* | 1.6187 | ⛔ **51.148** | 2.5658 |
| ego progress ratio (mean / median) | **1.7279** / 1.0994 | — | — | — |
| target-speed acc @0.5 / 1.0 / 2.0 m/s | 0.3398 / 0.5069 / 0.6564 | — | — | — |
| **distance-keeping (headway / time-gap / TTC)** | ⛔ **NOT MEASURED — WORK ITEM** | ⛔ | ⛔ | ⛔ |

**distance-keeping reason (verbatim from the artifact, `status: UNAVAILABLE`, `n: 0`):** *no
lead-agent track was supplied to the scorer.* PhysicalAI-AV **does** ship `obstacle.offline` (3D
agent tracks on 97.44 % of the corpus) and the instrument **exists** at
`taniteval/tools/build_lead_block.py` — it has simply not been built **for this dense C3 grid**.
⚠️ The registry twice cited it as `tools/build_lead_block.py`, which does not exist, making a built
instrument look unbuilt. **Not supplying it is a WORK ITEM, not a pass** (§12 W-1), and it is the
half of LONGITUDINAL where 88.7 % of the T0 oracle gap was measured to live.

⭐ **THE DIVERGENCE IS ~99 % LONGITUDINAL — visible ONLY because the families are reported.** Of the
repaired arm's `cl` ADE 9.3697, along-track MAE is **9.2655** and cross-track is **0.7446**; for
v5f, 23.8965 against 0.9993. The car holds its lane and its **speed integrates away**. A scalar ADE
would have shown the 25× gap and **not** shown that it is one axis.

### 1a.3 T1 — LATERAL *(healthy, and NOT the problem)*

| metric | `stageA·cl` | `stageA·ha` | `v5f·cl` | `v5f·ha` |
|---|---:|---:|---:|---:|
| heading MAE ° — **pooled-over-steps reducer** | 3.8776 | 2.4189 | 2.7171 | 2.9279 |
| heading MAE ° — **per-window reducer** [CI95] | 5.3945 [3.6203, 7.5348] | 3.9859 [2.2402, 6.0477] | 3.6204 [2.2465, 5.5110] | 4.5954 [2.7428, 6.7966] |
| yaw-rate MAE °/s | 4.9188 | 3.0260 | 4.5151 | 4.3442 |
| curvature MAE 1/m (bias) | 0.018586 (−0.0024) | 0.011293 | 0.017753 | 0.016452 |
| cross-track MAE m [CI95] | 0.7446 [0.5436, 0.9794] | 0.1689 [0.1281, 0.2192] | 0.9993 [0.7595, 1.2875] | 0.2072 [0.1444, 0.2856] |

⚠️ **TWO HEADING NUMBERS, SAME FILE, SAME ARM — reported, not smoothed over.** `four_families.
lateral.heading_mae_deg` pools over valid steps (3.8776 for `stageA·cl`); `intervals.metrics.
LAT_heading_mae_deg` reduces per window then means (5.3945). **MODEL_REGISTRY §1.14 publishes
5.3945 in its table and 3.8776 in its prose without stating that they are different reducers.**
Same defect class as §1c's three refused intervals. **Quote the reducer with the number.** All four
LATERAL members the binding rule names — heading, curvature, yaw-rate, cross-track — are present,
and none of them is where the failure lives.

### 1a.4 T1 — TACTICAL *(factored; the collapsed 5-way reports neither axis)*

| arm · surface | tier | lateral acc / κ | **longitudinal acc / κ** | 5-way acc / κ | goal bearing MAE ° | goal range ratio |
|---|:--:|---|---|---|---:|---:|
| `stageA·cl` | **T1** | 0.7515 [0.6844, 0.8159] / **0.3795** [0.2351, 0.5073] | 0.3327 [0.2608, 0.4062] / ⛔ **0.0405** | 0.3036 / 0.1404 | 4.8098 | **1.7584** |
| `stageA·ha` | **T1** | 0.8675 / 0.6427 | 0.5586 / 0.2072 | 0.5776 / 0.3904 | 3.7017 | 0.9722 |
| `stageA·ol` | T0 | 0.9109 / 0.7614 | 0.6034 / 0.2665 | 0.6518 / 0.4915 | 2.0754 | 0.9813 |
| `v5f·cl` | **T1** | 0.8378 / 0.5083 | 0.2181 / ⛔ **0.0102** | 0.2224 / 0.1173 | 3.7124 | **4.8141** |
| `v5f·ha` | **T1** | 0.8336 / 0.5493 | 0.2092 / **−0.0223** | 0.2696 / 0.1465 | 4.3538 | 0.8751 |
| `v5f·ol` | T0 | 0.8608 / 0.6186 | 0.2145 / −0.0198 | 0.2849 / 0.1625 | 3.3482 | 0.8751 |

⛔ **κ 0.0405 is chance agreement: longitudinal decision-making at T1 is AT CHANCE.** The collapsed
5-way (κ 0.1404) sits *between* the two axes and reports neither — **the direct measurement of the
lat/lon-mixing 5-way softmax that `CLAUDE.md` names as our largest known architectural defect**, and
it is visible only because the family is reported **factored**. Per-class lateral (`stageA·cl`):
`lane_keep` recall 0.8092 / precision 0.8747; `turn_left` 0.4994 / 0.6627; `turn_right` recall
0.6003 / **precision 0.2879** (1,195 predicted against 573 true — right turns over-predicted 2.1×).
⚠️ **The hold-action control beats the model on BOTH axes** — the tactical restatement of the
headline. ⭐ **Goal-setting: the DIRECTION is right and the DISTANCE is wrong** — bearing MAE 4.81°
against a range ratio of 1.7584 and a long-bias of +18.58 m vs a lat-bias of −1.21 m. *(The
artifact labels `goal_point_error_m` as FDE under another name; it is not sold as a new metric.)*

### 1a.5 T1 — STRATEGIC

⛔ **NOT MEASURED — `status: UNAVAILABLE`, n = 6,844.** Per clause 5 of the binding rule, the reason
and the n are given rather than the family dropped. **Reason (from the artifact):** PhysicalAI-AV
carries no map, no lane graph, no junction/roundabout label, no traffic-light feature and no
route/goal signal — the dataset card says verbatim *"we do not include open maps data"*, and
`obstacle.offline`'s enum over 87,481 cuboids is 10 classes, all dynamic agents; `egomotion` carries
no lat/lon/GNSS (clip-local metres), so OSM map-matching is impossible. Both label sources that do
exist are **inadmissible**: (a) a route class read off the ego's own future yaw cannot tell whether
the map admitted a choice at all — that is how the closed-loop harness once published
`route_head_eq_logged = 1.0000`, and `GATE_PROTOCOL` §0.7 declares `nonav_route_beats_majority`
VOID BY CONSTRUCTION; (b) a supplied route is optimistic by construction, because our only route
supplier here is the ego's own future path.
⇒ **No rescore of these windows can close it.** The instrument that would is the VLM strategic
pipeline **PH0 → PH1 → PH2** (PH0 v2 gate PASSED 8/8 on an n = 8 smoke, MODEL_REGISTRY §1.14).
**Settled at five independent probes — do not re-ask** (`CLAUDE.md` operating standard rule 2).
**This is a WORK ITEM blocked on a CORPUS fact, not a pass** — §12 W-3.

### 1a.6 Other T1 measurements *(different corpus — do not mix with 1a.1)*

*MEASURED 2026-08-06, corpus **C4** stride-1 (6,834 windows / 40 eps), predictor rolled on the
**decoder's own** actions. Registry anchor **MODEL_REGISTRY §1.12**; artifact
`…/incoming/2026-08-06-v1-defect-triage/results/closed_loop_analysis.json`. Evidence: MEASURED.*

| metric | v1.6 open (T0) | **v1.6 closed (T1)** | v1.7 open (T0) | **v1.7 closed (T1)** | CV floor |
|---|---:|---:|---:|---:|---:|
| ADE m | 0.3398 | **0.4714** (+0.132 [0.112, 0.152]) | 0.2849 | **0.4616** (+0.177 [0.148, 0.208]) | 0.5352 |
| net-yaw err rad | 0.0108 | **0.0725** (×6.7) | 0.0109 | 0.0761 | — |
| speed MAE m/s | 0.441 | 0.606 | 0.369 | 0.598 | — |
| **S-curve reproduction** | **0.9785** | ⛔ **0.0538** | 0.9785 | 0.0430 | 0 |

⛔ **The finding that re-frames every open-loop lateral number on this page:** the **hold-action arm
reproduces 0.0 % of S-reversals** and the closed loop ~5 %, against 97.9 % teacher-forced. **The
counter-steer in a T0 eval comes from the TRUE-action conditioning, not from vision.** Open-loop
LATERAL skill is largely an **action echo**; closed-loop, the stack drives near-straight with speed
control and retains **~33 %** of its T0 ADE advantage over CV.
⚠️ These two T1 blocks (§1a.1 on C3, §1a.6 on C4) are **not comparable** — different corpora,
different arms, different action interfaces. The C3 arms diverge catastrophically; the C4 arms
degrade gracefully. Reconciling that is itself an open question (§12 W-9).

---

## 1. TIER **T0** — canonical open-loop ADE on C1 (`ade_0_2s`, m) ⛔ *prediction quality, NOT driving*

*Every value below **re-verified BY CONTENT 2026-08-23** against
`taniteval/results/driving_<key>.json → headline.ade_0_2s` and `→ verdict.ade_vs_cv`; all agree with
MODEL_REGISTRY §6 to 4 dp. Estimator `episode_cluster_bootstrap`, B = 2000, 40 clusters, in every
file. Latency is the arm's own `taniteval/results/eff_<key>.json`, fp32, batch 1, A40 —
re-verified 2026-08-23. Evidence class: **MEASURED (ours)** for every row.*

| Rank | Arm | key | Step | Params | **ADE@2s m, full-set [ep-cluster CI95]** | FDE@2s | miss@2m | **vs CV (paired, 3-way)** | tick p50 fp32 | 10 Hz @p99 | *heldout ± ci95 (DEPRECATED)* |
|---:|---|---|---:|---:|---|---:|---:|:--|---:|:--:|---|
| **1=** | Flagship v1 (speed+jerk) FINAL | `flagship-30k` | 29 999 | 263.4 M | **0.4271** [0.3675, 0.4871] | 0.9075 | 0.045 | ✅ **model** +0.4106 [0.2050, 0.6240] | 97.32 ms | ❌ | *0.4522 ± 0.0312* |
| **1=** | REF-C-XL (anchored diffusion) FINAL | `refc-xl-30k` | 29 999 | 251.9 M | **0.4714** [0.3896, 0.5556] | 1.0061 | 0.142 | ✅ **model** +0.3663 [0.2029, 0.5521] | 44.06 ms | ✅ | *0.4577 ± 0.0572* |
| **1=** | REF-C-base (anchored diffusion) FINAL | `refc-base-30k` | 29 999 | **104.2 M** | **0.4728** [0.3835, 0.5699] | 1.0031 | 0.142 | ✅ **model** +0.3649 [0.2008, 0.5558] | **21.78 ms** | ✅ | *0.4523 ± 0.0497* |
| — ‡ | Flagship v1.6 (LP-FT, `ab` head) | `flagship-v16-ab-ft` | 5 999 | ~263 M | **0.4375** [0.3423, 0.5501] | 0.9297 | 0.106 | ✅ **model** +0.4003 [0.2731, 0.5533] | — | — | *0.4886 ± 0.0800* ⚠ |
| — ‡ | REF-C v1.2, k16-reg rescorer | `refc-v12-k16reg` | 29 999 | 251.9 M | **0.4576** [0.3742, 0.5438] | 0.9750 | 0.129 | ✅ **model** +0.3801 [0.2204, 0.5654] | — | — | — |
| — ‡ | REF-C v1.2, learned rescorer | `refc-v12` | 29 999 | 251.9 M | **0.4625** [0.3781, 0.5486] | 0.9819 | 0.137 | ✅ **model** +0.3752 [0.2146, 0.5589] | — | — | — |
| — ‡ | REF-C-XL live-decode snapshot | `refc-xl-live` | 29 999 | 251.9 M | **0.4788** [0.3977, 0.5638] | 1.0193 | 0.148 | ✅ **model** +0.3590 [0.1959, 0.5451] | 44.0 ms | ✅ | — |
| — ‡ | REF-C-**small** FINAL | `refc-small-30k` | 29 999 | **54.7 M** | **0.5261** [0.4295, 0.6262] | 1.1115 | 0.171 | ✅ **model** +0.3116 | **11.50 ms** | ✅ | *0.5007 ± 0.0671* |
| — | *hold-v0 (trivial floor)* | — | — | 0 | *0.7876* | 1.6521 | 0.292 | — | — | — | — |
| — | *constant velocity (trivial floor)* | — | — | 0 | *0.8377* [0.6234, 1.0716] | 1.7406 | 0.304 | — | — | — | *0.8248* |
| 5 | REF-B v2 (arch-v2) FINAL | `refb-v2-30k` | 29 999 | 271.6 M | 0.5913 [0.4766, 0.7131] | 1.2434 | 0.207 | ✅ **model** +0.2464 [0.0969, 0.4216] | — | — | *0.5921 ± 0.0685* |
| — ‡ | REF-C-XL snapshot | `refc-xl` | ~28 000 | 251.9 M | 0.6048 [0.5170, 0.7009] | 1.1873 | 0.167 | ✅ **model** +0.2329 [0.0642, 0.4260] | 44.0 ms | ✅ | — |
| 6 | Flagship v1, 19 k relay | `flagship-speed` | 19 000 | 263.4 M | 0.6152 [0.5422, 0.6951] | 1.3168 | 0.167 | ✅ **model** +0.2225 [0.0218, 0.4302] | 99.6 ms | ❌ | *0.6277 ± 0.0551* |
| 7 | REF-B v2 @20 k milestone | `refb-v2-20k` | 20 000 | 271.6 M | 0.6435 [0.5410, 0.7516] | 1.3218 | 0.216 | ✅ **model** +0.1942 [0.0541, 0.3652] | — | — | *0.6462 ± 0.0548* |
| 8 | REF-B speed | `refb-10k` | 10 000 | 262.8 M | 0.8372 [0.6753, 1.0218] | 1.6964 | 0.268 | ⚖️ **TIE** +0.0005 [−0.0982, 0.0951] | 60.47 ms | ✅ | *0.8255 ± 0.0992* |
| — ‡ | Flagship v4.1 @10 k | `flagship-v4.1-10k` | 10 000 | ~273 M | 0.8522 [0.7468, 0.9800] | 1.5176 | 0.249 | ⚖️ **TIE** −0.0145 [−0.1508, 0.1448] | — | — | *0.8707* |
| 9 | REF-B v1 | `refb` | 6 000 | 262.5 M | 0.8629 [0.6928, 1.0385] | 1.7351 | 0.318 | ⚖️ **TIE** −0.0252 [−0.1007, 0.0496] | 59.80 ms | ✅ | *0.8682 ± 0.0817* |
| — ‡ | Flagship v4.2 @4 k | `flagship-v4.2-step4000` | 4 000 | ~273 M | 0.9869 [0.8795, 1.1088] | 1.7487 | 0.294 | ⚖️ **TIE** −0.1492 [−0.2980, 0.0256] | — | — | *1.0490* |
| 10 | P2 CEM planner over frozen v1 | `planner_p2` | (n/a) | 0 trained | — ⚠️ **the open-loop CEM arm (`plan_wp`) was never dumped per-window**; the *closed-loop* windows ARE banked and both gates were re-decided 2026-08-16 — **neither flips** | — | — | ⛔ **UNDECIDED** *(banned verdict, flip reachable)* | — | — | *0.893 ± 0.114* |
| 11 | Flagship **v3enc** (RESTART, reg. §1.4) | `flagship-v3enc-10k` | 10 000 | 272.9 M | **1.9654** [1.6556, 2.2859] | 3.6084 | 0.690 | ❌ **floor** −1.1277 [−1.4134, −0.8741] | — | — | *2.1072 ± 0.2020* |
| 12 | REF-A DINOv2 4B | `refa-dinov2` | 29 999 | 156.6 M† | 2.1675 [1.9081, 2.4212] | 3.2803 | 0.613 | ❌ **floor** −1.3298 [−1.5415, −1.1233] | 88.58 ms | ❌ | *2.1322 ± 0.1821* |
| 13 | Flagship **no-speed** (ablation control) | `flagship-nospeed` | ~22 000 | 263.4 M | 3.0175 [2.5450, 3.5444] | 5.0282 | 0.742 | ❌ **floor** −2.1798 [−2.7714, −1.6714] | 101.58 ms | ❌ | *2.9176 ± 0.3558* |
| 14 | REF-A dyn-in 4B | `refa-dynin-30k` | 29 999 | 156.6 M† | 3.0471 [2.4984, 3.6878] | 4.7642 | 0.741 | ❌ **floor** −2.2094 [−2.8164, −1.7232] | 84.52 ms | ❌ | *2.9196 ± 0.3937* |
| 15 | Flagship v2 (killed) | `flagship-v2-6k` | 6 000 | 272.9 M | 5.9396 [4.3273, 7.6249] | 12.4011 | 0.852 | ❌ **floor** −5.1019 [−6.9322, −3.4332] | — | — | *6.179 ± 1.2845* |
| — | Flagship v1 **tactical head** (not rollout) | `plan_flagship-30k` | 29 999 | — | **3.3839** [2.8336, 3.9722] | — | — | ❌ **floor** | — | — | *3.150 ± 0.347* |

**Ranks are MODEL_REGISTRY §6's, unchanged** (they run 1=…15 after the 2026-07-25 re-emission
renumbered the table). ‡ marks arms with a window dump and a scored block but **no §6 rank**; they
are placed in ADE order and never renumbered — the registry is the source of truth for ranks.
`vs CV` is the **paired** episode-cluster delta (CV − model, m) with its `favours` label read
straight from the JSON.

† REF-A's params **exclude the external frozen DINOv2/I-JEPA encoder**, and so does its latency —
never compare a features-in row to a pixels-in row unadjusted.
⚠ v1.6's `heldout` comes from a **different eid family** (`eval_flagship_v16.py` clusters on real
`episode_id`, `bench.py` on file indices 0–39); the two `heldout` means are **not comparable**. The
full-set and episode-cluster columns are unaffected. v1.6 vs v1 paired: **Δ +0.0104 [−0.0888,
+0.1147], NOT separated** (registry §1.4b) → an ADE **tie**, which is why it carries no rank.
⚠️ **`refc-v12-identity` is deliberately absent.** It is a designed control that is
**bit-equivalent to `refc-xl-30k`** (same decode; `max |Δpred| = 7.6e-06`) and listing it would
double-count one model. `taniteval/results/dump_exclusions.json` is the machine-readable truth;
`taniteval.dump_census` must be imported for any census — **a bare glob is a defect**
(EVAL_DOCTRINE rule 6, C126). Same for `windows_overfit_refa-dynin-30k.pt` ≡ `refa-dynin-30k`.
⚠️ **The REF-A overfit ladder** (`overfit_refa-dynin-{5k,15k,20k,30k}` = 3.8307 / 3.7818 / 3.1138 /
3.0471, all ❌ floor-separated) is a **milestone curve of one arm**, not four arms. Recorded here so
the dumps are accounted for; never rank them.

**Ranks 1= are a three-way ADE tie no paired test can order** (§6: base-vs-XL Δ +0.0013 [−0.0281,
+0.0316]; flagship-vs-XL +0.0443 [−0.0544, +0.1465]). **Latency is the only separator among them —
and §2 adds a second one.**

### 1b. Floor re-adjudication — 25 banked arms rescored against CTRV (2026-08-02, MEASURED) [T0]

Paired episode-cluster bootstrap, B = 2000, same C1 windows, orientation `floor − model`. Alignment
verified **bit-exact** (`max_abs_diff_cv = max_abs_diff_gt = 0.0`) on 25 of 27 dumps; the two
88-window `refc-v12-smoke-*` partials are refused, not approximated.

| verdict on `ade_0_2s`: vs CV → vs CTRV | n | arms |
|---|---:|---|
| beats floor → **beats floor** | 6 | `flagship-v16-ab-ft` (+0.0890), `refc-v12-k16reg` (+0.0688), `refc-v12` (+0.0639), `refc-v12-identity` / `refc-xl-30k` (+0.0550, **marginal, lo = +0.0001**), `refc-base-30k` (+0.0537) |
| beats floor → **TIE** | 3 | ⭐ **`flagship-30k` (v1, deployed)**, `flagship-speed`, `refc-xl-live` |
| beats floor → **LOSES to floor** | 3 | `refb-v2-30k`, `refb-v2-20k`, `refc-xl` |
| tie → **LOSES to floor** | 4 | `refb`, `refb-10k`, `flagship-v4.1-10k`, `flagship-v4.2-step4000` |
| loses → loses | 9 | REF-A family, `flagship-nospeed`, `flagship-v2-6k`, `flagship-v3enc-10k` |

**16 of 25 verdicts move. 12 arms beat the trivial floor under CV; 6 do under CTRV (11 / 5 over
DISTINCT arms after `dump_exclusions.json`), and the best surviving margin in the whole fleet is
+0.0890 m.**

⭐ **flagship-v1 @30k** — ADE 0.4271, vs CV **+0.4106 separated**, vs CTRV **+0.0993 [−0.0258,
+0.2204] NOT separated**. ⛔ *"the FIRST arm below EVERY trivial bar"* is a **point-estimate
statement** (0.4271 < 0.5265 is true); under the programme's own decision-grade paired estimator it
is a **TIE** against a constant-turn-rate extrapolation. **And it is a T0 tie**, so it was never a
driving claim in the first place.

**Where the win survives (flagship-v1, pre-registered criteria — all three HELD, ~5× smaller):**

| stratum / metric | n | model | CV | CTRV-g | vs CV | **vs CTRV** | shrink |
|---|---:|---:|---:|---:|---|---|---:|
| `sustained_turn` ADE | 142 | 0.5061 | 2.3124 | 0.8460 | +1.8063 model | **+0.3398 [0.153, 0.550] model** | 5.3× |
| `sustained_turn` \|cross\| | 142 | 0.3462 | 3.9126 | 0.9156 | +3.5664 model | **+0.5694 [0.334, 0.804] model** | 6.3× |
| `curv_sharp` heading° | 144 | 3.81 | 28.74 | 11.50 | +24.93 model | **+7.69 [4.75, 11.09] model** | 3.2× |
| overall \|cross\| | 881 | 0.2369 | 1.0089 | 0.3741 | +0.7720 model | **+0.1372 [0.026, 0.252] model** | 5.6× |

⇒ `verdict.where_the_win_lives = "lateral only"` **survives as a direction** and must never again be
quoted with a CV-derived magnitude. ⚠️ **And §1a.6 bounds what it means:** a T0 lateral win is
largely an **action echo** (S-curve reproduction 97.9 % → 5 % closed-loop, 0 % hold-action).

**Where CTRV reveals a loss CV structurally could not** (flagship-v1; exploratory, not pre-registered):

| scope | metric | n | vs CV | vs CTRV |
|---|---|---:|---|---|
| overall | `fde_2s` | 881 | +0.8330 model | **+0.2196 tie** |
| overall | heading median° | 881 | tie | **floor** (−0.6456) |
| `speed_high` | `ade_0_2s` | 294 | tie | **floor −0.2154 [−0.386, −0.030]** |
| `speed_top10pct` | `ade_0_2s` | 89 | floor −0.4156 | **floor −0.6173 [−0.782, −0.459]** |
| `speed_top10pct` | \|cross\| / heading° / crosstrack | 89 | tie / tie / tie | **floor / floor / floor** |

⭐ At the top speed decile **CTRV's ADE is 0.0986 m and the model's is 0.7159 m — 7.3× worse — and
the model loses LATERALLY too.** The high-speed weakness has been framed as purely *longitudinal*
because a straight-line floor cannot expose a lateral one on a road that is locally an arc.

### 1c. THE FOUR METRIC FAMILIES vs the floors on C1 [T0]

Same C1 windows; point estimates from `taniteval/four_families.py` itself; paired episode-cluster
bootstrap where a per-window form reproduces the module (refusals below).
⚠️ Sparse 4-waypoint cadence (`DT_S = 0.5`) — **not** comparable to a dense-10 Hz run.
Artifact: `…/incoming/2026-08-02-ctrv-floor/raw/four_families_vs_floors.json`. MEASURED 2026-08-02.

| family / metric | flagship-30k | REF-C-XL-30k | CV | hold-v0 | **CTRV-g** | flagship vs CTRV |
|---|---:|---:|---:|---:|---:|---|
| **LONG** `speed_mae_mps` | 0.4710 | 0.4545 | 0.4678 | 0.4818 | 0.4682 | tie |
| **LONG** `speed_bias_mps` *(+ = too fast)* | **+0.1911** | +0.0209 | −0.0545 | −0.1340 | −0.0557 | **floor** −0.2468 [−0.347, −0.144] |
| **LONG** `along_final_bias_m` | **+0.3375** | +0.0511 | +0.0347 | −0.1232 | −0.1107 | **floor** −0.4482 [−0.652, −0.238] |
| **LONG** distance-keeping (headway / time-gap / TTC) | ⛔ **NOT MEASURED — WORK ITEM.** No lead-agent track is read from the episode cache on this grid. `obstacle.offline` exists on 97.44 % of the corpus and the reader exists (`taniteval/tools/build_lead_block.py`); it has not been joined to C1. **n = 0.** §12 W-1 |
| ⭐ **LAT** `curvature_mae_1pm` | **0.026969** | 0.012138 | 0.012221 | 0.012221 | **0.008967** | *point only — interval refused* |
| **LAT** `yaw_rate_mae_degps` | 1.8581 | 1.8216 | 3.6951 | 3.6951 | 2.2333 | *point only — interval refused* |
| **LAT** `heading_mae_deg` | 1.5032 | 1.1484 | 3.8265 | 3.5322 | 1.6213 | *point only — interval refused* |
| **LAT** `cross_mae_m` | **0.1152** | 0.1310 | 0.5259 | 0.4662 | 0.1604 | **model** +0.0452 [+0.008, +0.084] |
| **TACTICAL** | ⛔ **NOT MEASURED — WORK ITEM.** The scored pass is a teacher-forced WM-fidelity rollout (`pc2_pass = False`, `actions_source = "expert_future"`), so **no manoeuvre decision is decoded at all**. **n = 0.** ⇒ needs a hierarchy-traversing eval, or read it at T1 where the driven path *is* the decision (§1a.4, now closed at source: `t1_eval.py` passes `tactical_from_traj=True, tier=t`). §12 W-4 |
| **STRATEGIC** | ⛔ **NOT MEASURED — WORK ITEM.** Same teacher-forced reason, **and** the corpus reason of §1a.5. **n = 0.** §12 W-3 |

⭐⭐ **CROSS-TRACK AND CURVATURE DISAGREE, AND CURVATURE IS THE ONE THAT MATTERS HERE.** The flagship
**beats** the CTRV floor on cross-track (+0.0452 separated) while its **curvature error is 3.0×
worse than CTRV's and 2.2× worse than a straight line's**. The path goes through roughly the right
points with the wrong *shape* — the "smooth but wrong" failure ADE and cross-track both hide.
REF-C-XL's curvature is **2.2× better** than the flagship's while its ADE is *worse* (0.4714 vs
0.4271): **ADE inverts this ordering.**

⚠️ **Three intervals REFUSED on purpose.** `heading` / `yaw_rate` / `curvature` are published as
point estimates only: the module reduces them as a pooled mean over valid steps, a per-window form
is a mean-of-per-window-means, and the two differ (7.6e-01 / 5.9e-01 / 3.6e-02 on flagship-30k). The
driver measures the disagreement per metric and refuses the interval above 1e-3 rather than
bootstrap a statistic that is not the published one. **The same two-reducer hazard is live at T1 and
is NOT refused there** — see §1a.3. Work item: a per-window reducer inside `four_families` (§12 W-5).

> ⚠ **Open-loop ⊥ closed-loop (standing footnote, G-B1).** arXiv 2605.00066 (Apr-2026, 15 methods):
> ADE/FDE have **no reliable correlation** with closed-loop Driving Score. Our own evidence, now
> three-fold: flagship v1 open-loop **0.4271** [0.3675, 0.4871] → imagination-closed-loop **1.7318**
> [1.5707, 1.9070] (**4.05×**), divergence > 5 m on **23.50 %** [16.80 %, 30.27 %] of windows
> (registry §1.2); v1.6 T0 0.3398 → T1 0.4714 (§1a.6); and the decisive one — **stage-A T0 0.3659 →
> T1 9.3697, 25×** (§1a.1).
> ⛔ **ESTIMATOR CORRECTED 2026-08-17 — this footnote once published three BANNED
> `overlapping_holdout_se` split-means** (*0.4522 → 1.685, 22.2 %*), all corrected above from
> `…/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json`; **all
> three moved the same way — the closed-loop failure was UNDERSTATED**, so the footnote's own point
> *strengthens*. **Never rank a TanitAD checkpoint on a T0 number alone.**

---

## 1d. THE **B1-CORPUS** LINE — `refcv3` (one-shot anchor) and `refav1` · ⚠️ **loop = OPEN** · *added 2026-09-03*

*Added on the PI's direct instruction to record these results **even though the page's older rows
carry only partial criteria** (§0.9 documents that gap). ⛔ **CORPUS BOUNDARY, and it is the first
thing to read:** both arms train on the **B1 Alpamayo-labelled corpus** (4,572 train clips / 141 eval
clips, `D-CORPUS-B1`), **NOT** the parity corpus `physicalai-train-e438721ae894`. They therefore form
a **NEW DOMAIN** and **may never be same-data-compared to any row in §1, §1a, §1b, §1c, §2, §4 or
§5** — those are C1–C4. Cross-domain, the only admissible statement is each arm's **margin over the
shared `ha0` floor**, per family, paired.*

### 1d.1 Identity and admissibility

| | **refcv3** | **refav1** |
|---|---|---|
| what it is | supervised **ONE-SHOT anchor** trajectory model — the whole 6 s path from **one** forward pass | supervised **autoregressive** planner (`plan()` + iCEM search) |
| anchors / horizon | **128 anchors × 8 slots** at `(5,10,15,20,30,40,50,60)` × 0.1 s = **0.5–6.0 s** (`refc_v3.py:196`, `:197`, `:106`) | per-step decode, `K = 10` @ 0.2 s = 2.0 s on the step-1000 read |
| **action input** | ⛔ **NONE** — `forward(frames, nav_cmd, v0, steps, lan, nav_known)` has no action argument (`refc_v3.py:480`) | yes — feeds `(steer = arctan(L·κ), a_j)` back per step |
| rollout | ⛔ **none** — there is no action loop to close | yes |
| inference inputs | observed frames + v7.2 nav token + **ONE measured ego scalar** `v0` at t0 (`refc_v3_train.py:445`) **and nothing else** | frames + nav + ego state |
| **future ego data** | ⛔ **none** — a future-only perturbation moved the arm by **exactly 0.0** (instrument control, MEASURED) | none in `cl`; `ol` / `cl_oraclegoal` are **T0** by construction |
| params | ⭐ **107,032,901 — quote this one.** MEASURED twice in exact agreement (`MODEL_REGISTRY.md` §4.5): `config.json → param_breakdown.total`, and `ckpt_30000.pt`'s state_dict as **544 tensors / 107,082,365 elements − 201 BUFFER tensors / 49,464 elements**. ✅ **The apparent 49,464 conflict is RESOLVED, not open: 107,082,365 is the ALL-TENSOR count and 107,032,901 is the PARAMETER count** — both figures are correct and they count different things. Breakdown: core 104,879,522 · phi_tac 1,757,440 · tac_latent_proj 262,656 · gstr_cond 66,816 · nav_inject 50,176 · tac_heads 14,364 · scorer 1,156 · str_goal_head 771 | — |
| checkpoint identity | `ckpt_30000.pt` **428,519,790 B**, md5 **`00da81c6efcd91e7b618a1fbddb3b78f`** (MEASURED on the pod and again after transfer — both agree) | md5 `45b9f4d82a3a7f15bda6eecf11e4fb71` (step-1000, **retired**) |
| geometry / arm | `image_hw [256, 640]`, `arm hier`, `size base`, `seed 0`, `tac_vocab_version v7.0` | — |
| checkpoint | `ckpt_30000.pt` **INTERIM**; ⏳ **final run is 40,284 steps, ETA ≈22:40Z 2026-09-03** | 🟢 **TRAINING on Thor**, step ~15,250 / 21,109, ETA ≈00:50Z |
| **arm names** | ⭐ **`os`, NEVER `cl`** (`D-REFCV3-EPOCH-READ`) — a shared name is how two different procedures end up read as levels of one quantity. **`ol` is ABSENT**, with its structural reason: refcv3 consumes no recorded actions | `cl` / `ha` / `ol` / `cl_navshuf` / `cl_oraclegoal` |
| tier | ⚠️ **stamped `T1` with `status: UNRULED`** — whether the doctrine admits at T1 a model that consumes **no actions at all** is an **open PI ruling** (BACKLOG R30). The margin-over-`ha0` framing keeps every number correct either way | T1 (`cl`, `ha`, `cl_navshuf`) · T0 (`ol`, `cl_oraclegoal`) |
| **loop (§0.8)** | ⚠️ **OPEN** | ⚠️ **OPEN** |

⛔ **ADMISSIBILITY WARNING — refcv3's nav token is an ORACLE.** `config.json` states it verbatim:
`nav_cmd_derivation = "v7.2 nav_command token (oracle, provenance ego-future; allow_oracle_nav=True)"`.
Under the binding rule *"a supplied route is optimistic by construction — our only route supplier is
the ego's own future path"*, **every refcv3 number carrying nav is an optimistic bound**, and the
`os_navshuf` / `os_navzero` arms are what bound the size of that optimism. ✅ The goal path is
**clean** on the other axis: `config.json → goal_provenance` records
`contains_situation_classifier_output: false` and `situation_classifier_in_graph: false`, so the
2026-08-03 goal-input rule is satisfied.

### 1d.2 ⛔ THE FOUR BINDING FAMILIES — refcv3 · **ALL PENDING, none invented**

*⛔ **There is no admissible refcv3 family number yet.** Two differently-bound probes (`ls` + `grep`,
and .NET `Directory.GetFiles`) agree that `taniteval/results/refcv3-30k-openloop-*.json` **does not
exist**; `taniteval/results/` holds 105 files and none matches `refcv3|refav1|openloop`. The cells
below name the file that fills each one.*

| family | metric | `os` | `os_navshuf` | `os_navzero` | `oracle_sel` (T0 ceiling) | ⭐ **`ha0`** *(trivial floor)* | fills from |
|---|---|---|---|---|---|---|---|
| **ADE** | ADE / FDE m, paired ep-cluster CI | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | `taniteval/results/refcv3-30k-openloop-*.json` |
| **LONGITUDINAL** | speed MAE + bias, along-track, accel, target-speed acc | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | same |
| **LONGITUDINAL** | **distance-keeping** (headway / time-gap / TTC) | ⚠️ **PARTIAL BY CONSTRUCTION** — `join_lead_block` maps `t → 2t`, so refcv3's `--grid 2s` joins only at **{1.0, 2.0} s**; an origin on an **odd** raw frame joins to `frame − 1` (count emitted as `_odd_raw_frames`). **W-4/W-5 of the arm package** | — | — | — | — | rebuild the B1 lead block on refcv3's own grid, `--dt 0.5 --k 4` |
| **LATERAL** | cross-track, heading, curvature, yaw-rate | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | same |
| **TACTICAL** | trajectory-derived lat/lon manoeuvre + **`anchor_acc` vs chance 1/128** | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | ⏳ PENDING | same |
| **STRATEGIC** | route head under **three** nav conditionings + echo index | ⏳ PENDING — ⛔ **inadmissible without `os_navshuf` beside it**; the trajectory-only row is **UNAVAILABLE BY DESIGN** (a route class cannot be read off a 2 s path) | ⏳ | ⏳ | — | — | same |

⭐ **THE `ha0` COLUMN IS NOT DECORATION.** MEASURED on the arm package's own fixture: paired
**`ha − ha0` = +0.1330 [−0.0502, 0.2326], NOT separated** — holding the last *observed* steer is **not
better than doing nothing**. ⇒ **a win over `ha` alone is not skill**, and any refcv3 or refav1 number
that does not beat `ha0` must be visible as such at a glance. That is why the column sits beside the
model's, not in a footnote.

### 1d.3 What refcv3 numbers DO exist today — and why NONE of them may enter the table above

*MEASURED 2026-09-03, Benchmarks & Evals FlyWheel, 0 GPU. Artifact
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-03-refcv3-epoch-conclusions/raw/refcv3_epoch_read.json`;
instrument `taniteval/tools/refcv3_metrics_read.py`. Registry anchor: ⛔ **NONE — `MODEL_REGISTRY.md`
has no refcv3 row at all** (§1d.5).*

**Tier T0 · loop OPEN · in-training loss monitor over 160 FIXED held-out windows**, provably constant
across all 55 evals and all 10 launches (`fixed_window_set_invariants.ALL_CONSTANT = true`). Era C
(≥ 18,000 — the only clean era; era A is contaminated by a superseded nav source **and** an
eval-prior leak), n = 15 evals over steps 18,000–25,500:

| column | first | **last** | min | slope / 1k steps | R² | quotable as a rate? |
|---|---:|---:|---:|---:|---:|:--|
| `eval_loss` | 8.54336 | **7.81994** | 7.81994 | −0.0791 | **0.247** | ⛔ **NO** — R² < 0.80 |
| `eval_traj` ⚠️ | 1.15026 | **0.96580** | 0.93653 | −0.0224 | **0.514** | ⛔ **NO** |
| `eval_anchor_acc` | 0.575 | **0.600** | 0.50625 | +0.0026 | **0.039** | ⛔ **NO** |
| `eval_goal2s_err_m` | 2.16344 | **1.87202** | 1.87202 | −0.0128 | **0.029** | ⛔ **NO** |
| `eval_lat_tac` | 0.98060 | **0.92121** | 0.91660 | −0.0032 | **0.043** | ⛔ **NO** |
| `eval_lon_tac` | 1.23716 | **1.14682** | 1.14682 | −0.0074 | **0.281** | ⛔ **NO** |
| ⭐ `eval_goal_gate` | 0.07328 | **0.13072** | 0.07328 | +0.0072 | **0.987** | ✅ **yes** — the strategic gate is **OPENING** |
| `eval_goal_score_absmean` | 8.73380 | **4.75025** | 4.75025 | −0.4602 | **0.876** | ✅ yes |
| `eval_law` | 0.05461 | **0.04010** | 0.04010 | −0.0022 | **0.829** | ✅ yes |

⛔⛔ **THREE REASONS NO NUMBER ABOVE MAY BE PROMOTED INTO §1d.2 — each measured, not asserted:**

1. ⛔ **`eval_traj` IS NOT AN ADE.** `loss_traj` is `|Δ|.sum(-1) / (sv.sum()·2)`
   (`refc_v3_train.py:462–465`) — a **mean absolute error PER COORDINATE**. An ADE is an **L2 norm**.
   They are different statistics and **must never share a column** with the flagship's or refav1's ADE.
2. ⛔ **`eval_traj` IS ORACLE-ANCHOR-SELECTED.** `a_star = dist.argmin(dim=1)` is the anchor nearest
   the **GROUND TRUTH** (`refc_v3_train.py:457–459`) and `recon = out["anchor_traj"][ar, a_star]`
   (`:463`) is what is scored — while the model's **own** selection agrees on only **57 %** of windows
   (`eval_anchor_acc` 0.5692 mean). ⇒ it is a **loose LOWER BOUND** on what refcv3 would drive. The
   deployed path is `out["traj"]`, ranked by `sel_score_v3` (`refc_v3.py:518–527`), **never `a_star`**.
3. ⛔ **NO CONFIDENCE INTERVAL IS COMPUTABLE BY ANY ESTIMATOR** — `refcv3_epoch_read.json →
   estimator_refusal.ci_available = **false**`. Every `eval_*` value is **already** the pooled mean
   over the 160 windows; the file carries **no per-window values and no episode index**, both of which
   `taniteval/ci.py` requires. ⇒ every statement in this block is a **DIRECTION, not a verdict**, and
   the binding paired-episode-cluster-bootstrap rule **cannot be satisfied by this artifact at all**.
   The fix is forward-looking only: the in-training eval must dump per-window values with episode ids.

⭐ **THE STOPPING CONCLUSION, and it is the operationally important one:** in era C the headline
columns move by **less than 1× the series' own step-to-step scatter** — `eval_lat_tac` 0.36×,
`eval_lon_tac` 0.41×, `eval_goal2s_err_m` 0.16×, `eval_anchor_acc` 0.22×, `eval_loss` 0.87× — while
the **train–eval gap FLIPS SIGN** across the boundary (loss −0.886 → **+0.238**; `goal2s_err_m`
−0.020 → **+0.311**), a flip that **survives** the pre-registered robustness pass that drops
post-resume rows. **Train keeps improving; held-out does not.** ⇒ **read the epoch-end checkpoint at
T1; do not extend the run.**
⚠️ **The strategic path is being USED but is not yet shown to HELP**: the goal gate opens 0.0733 →
0.1307 (R² 0.987), yet the two flattest columns in the whole panel are the goal **error** and the goal
**loss**.

### 1d.4 refav1 — the first real T1 read *(⚠️ NOT decision-grade; final read PENDING)*

*MEASURED 2026-09-03 22:22–00:54Z, dev-box RTX 4060, EXIT 0. Artifact
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refav1-step1000-read/raw/refav1_t1_step1000.json`
(388,352 B). Instrument `taniteval/tools/refav1_arm.py` @ blob `de4660cb`, families via
`t1_eval.analyze` (**imported**, not copied). Estimator: point = `full_set` pooled mean; intervals =
**paired episode-cluster bootstrap** (`taniteval/ci.py`, B = 2000). ⛔ `overlapping_holdout_se` used
nowhere. **n = 20 episodes / 140 windows**, stride 10, K = 10 @ 0.2 s. Evidence class: **MEASURED
(ours)**. **loop = OPEN.***

⚠️ **NOT DECISION-GRADE, stated before the numbers:** a **20-clip subset** of the 141-clip eval split,
at **step 1,000 of 21,109 (4.7 % of one epoch)**, on a checkpoint that has **since been retired**
(`refav1-b1-v72-1ep-21109`, fp32, no EMA, md5 `45b9f4d82a3a7f15bda6eecf11e4fb71`). It is an instrument
check and a first read. The **pre-registered** read is the end-of-epoch checkpoint on the **full**
split, on Thor — ⏳ **PENDING, ETA ≈00:50Z**.

| arm | tier | **loop** | ADE m [ep-cluster CI95] | distance-keeping (43 lead windows): headway m · time-gap s · min TTC s |
|---|:--:|:--:|---|---|
| `cl` (plan(), true nav, own actions) | **T1** | **OPEN** | **0.547** [0.405, 0.714] | 33.40 · 3.55 · 24.46 |
| `ha` (hold observed `(a, κ)`) | **T1** | **OPEN** | 0.610 [0.410, 0.831] | 33.23 · 3.52 · 24.02 |
| ⭐ **`ha0`** *(constant velocity at measured `v0`)* | **T1** | **OPEN** | ⛔ **ABSENT FROM THIS DUMP** — `arm_keys = ['cl','ha','ol','cl_navshuf','cl_oraclegoal']`. The read predates `D-REFAV1-HA0-ARM`; `refav1_arm.py` carries `ha0` **now** (10 occurrences) and `t1_eval.DEFAULT_TIERS` stamps it `T1` (`t1_eval.py:148–153`) | — |
| `ol` (recorded future actions) | T0 | OPEN | *0.499* [0.326, 0.688] | 33.28 · 3.53 · 23.93 |
| `cl_navshuf` (nav permuted) | **T1** | **OPEN** | 0.642 [0.492, 0.815] | 33.47 · 3.55 · 24.50 |
| `cl_oraclegoal` (true future field as goal) | T0 | OPEN | *0.718* [0.608, 0.839] | 33.61 · 3.58 · 25.01 |

**Paired deltas, `cl` minus control** (bold = interval excludes zero):

| family / metric | `cl − ha` (T1−T1) | `cl − navshuf` (T1−T1) | `cl − ol` (T1−T0) |
|---|---|---|---|
| ADE m | −0.064 [−0.165, +0.035] | **−0.095** [−0.158, −0.040] | +0.047 [−0.063, +0.159] |
| FDE m | **−0.283** [−0.571, −0.001] | **−0.254** [−0.423, −0.107] | +0.079 [−0.221, +0.372] |
| **LON** speed MAE m/s | **+0.237** [+0.117, +0.375] | **−0.141** [−0.234, −0.058] | **+0.376** [+0.232, +0.546] |
| **LON** along MAE m | +0.059 [−0.027, +0.144] | **−0.107** [−0.177, −0.045] | **+0.137** [+0.044, +0.229] |
| **LON** accel MAE m/s² | **+0.189** [+0.086, +0.313] | **−0.143** [−0.238, −0.059] | **+0.405** [+0.263, +0.573] |
| **LAT** cross-track MAE m | **−0.179** [−0.281, −0.094] | 0.000 exactly | **−0.129** [−0.209, −0.064] |
| **LAT** heading MAE ° | **−1.99** [−3.03, −1.13] | 0.000 exactly | **−1.59** [−2.36, −0.91] |
| **LAT** yaw-rate MAE rad/s | **−0.048** [−0.073, −0.027] | 0.000 exactly | **−0.036** [−0.054, −0.021] |
| **TAC** lon manoeuvre correct | **−0.136** [−0.257, −0.036] | **+0.093** [+0.036, +0.164] | **−0.257** [−0.371, −0.150] |
| **TAC** lat manoeuvre correct | +0.064 [−0.021, +0.164] | 0.000 exactly | +0.021 [−0.057, +0.107] |
| **STRATEGIC** | ⛔ `families_unavailable = ['strategic']` **on every arm** — the route-accuracy block needs the strategic head's declared route (40 nav-labelled windows exist, `nav_valid_frac` 1.0) |

⛔ **THE ECHO TEST IS NOT PASSED ON ADE AT STEP 1,000.** `cl − ha` is **−6 cm with an interval
crossing zero**, and the anti-echo block reads `holdv0 = NOT_SEPARATED`, `copy_detector = **ECHO**`
(echo index **0.9929** against the human's 0.1286) — because the deployed plan **IS** hold-v0 on
**75.7 %** of windows (`baseline:hold_v0` 75.7 %, `cem` 24.3 %). ⇒ **this is the init-floor property
of a 4.7 %-epoch checkpoint, not a verdict on the architecture** — and it is exactly the reading that
`ha0` exists to make unambiguous, which is why its absence from this dump matters.
⭐ **Lateral planning already beats holding, longitudinal loses to it** — both separated. The
planner's weak side is the **speed channel**, the same side the `v0`-as-input repair targeted.
⭐ **Nav reaches the plan through its LONGITUDINAL channel only**: the shuffle costs 9.5 cm ADE and
every LON metric (separated) while **every LAT delta is exactly 0.000 on all 140 windows**.
⭐ **The ORACLE GOAL HURTS** (T0 vs T1): the true future field as the goal is **17 cm** worse on ADE
than the imagined tactical goal — attribution is *"cannot search toward a goal field"*, not *"no goal
information"*.

### 1d.5 Registry status — ⭐ refcv3 **LANDED MID-WRITE**; refav1 still absent

⚠️ **This section was written as a two-arm absence claim and was HALF STALE within the hour — the
correction is recorded rather than quietly applied.** At **21:2xZ** three differently-bound probes
(Bash `grep -n -E`, PowerShell `Select-String -LiteralPath`, .NET `File.ReadAllText` +
`Regex.Matches`) agreed on **0** hits for `refcv3`/`refc_v3`/`refc-v3`/`refav1`/`refa_v1` across all
**381,371** characters of `MODEL_REGISTRY.md`. At **21:5xZ** a re-probe read **385,121** characters
and **7 / 10 / 3** hits: a sibling stream had minted the row while this section was being written.
*(The lesson is the standing one — **the repo advances mid-session**; re-probe before shipping an
absence claim, especially one written inside a page about stale absence claims.)*

| arm | `MODEL_REGISTRY.md` | status |
|---|---|---|
| **refcv3** | ✅ **§4.5 `refcv3-b1-v72-30k`** — 🟢 LIVE, step **37,400 / 40,284** @ 2026-09-03T19:24Z | **row exists and it AGREES with §1d above**, independently: same one-shot/no-action reading from `refc_v3.py:480`, same B1 non-parity corpus, same **ORACLE** nav caveat, same `ha0`-margin framing, same UNRULED tier — and it carries its own ⛔ **"NO EVAL RESULT EXISTS — do not quote an accuracy number"** header with all four families **NOT MEASURED**, naming `taniteval/results/refcv3-30k-openloop-*.json` as the artifact that will close it. ⇒ **two independently-written sources reached the same pending verdict** |
| **refav1** | ⛔ **STILL ABSENT** — **0** hits for `refav1` / `refa_v1` / `REF-A v1` on the 21:5xZ re-probe | ⇒ **W-20 stands, narrowed to refav1 alone.** Until a row is minted, §1d.4's numbers are admissible **only with their raw-JSON path attached**. Same class as W-12 (v7-tiny) |

⭐ **The registry also SETTLES the parameter question** (§4.5, MEASURED twice): **107,032,901
parameters**; **107,082,365** is the **all-tensor** count over 544 tensors, of which **201 are buffers
totalling 49,464 elements**. Neither figure was wrong — they count different things, and the registry
directs that **107,032,901** be quoted.

### 1d.6 ⭐ THE B1 LINE IS NO LONGER PENDING — `refcv4b` and `refcv5-v2`, FOUR FAMILIES, MEASURED

*MEASURED 2026-09-09, re-read from the raw JSON 2026-09-11. Artifacts banked and byte-verified in
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/raw/`
(`refcv5-v2_vs_refcv4b.json` sha256 `8acbcc17…`, `refcv5-v2-seed1_vs_refcv4b.json`
`11e31e95…`, plus both per-arm JSONs and both panel logs). Registry anchors:
`MODEL_REGISTRY.md` **§4.6** (refcv4b) and **§4.7** (refcv5-v2). ⛔ Same B1 corpus
boundary as §1d above — these may never be same-data-compared to §1/§1a/§1b/§1c/§2/§4/§5.*

⛔⛔ **READ THE SIGN BEFORE THE WORD `separated`.** The metric is `ade_m`, **lower is
better**, so a **positive** delta means the arm is **WORSE** than what it is differenced against.
Three rows below are separated *against* the model. A separated interval says only *"another draw of
episodes would say the same"*; it never says who won.

#### 1d.6.1 The floors, and the bar — metric `ade_m`, tier **T1**, paired episode-cluster bootstrap

| row | **refcv5-v2** (seed 0) | **refcv5-v2** (seed 1) | **refcv4b** | reading |
|---|---|---|---|---|
| **`os − ha0_ext`** — the committed bar `BAR-REFCV5V2-1` | **+0.0205** [+0.0043, +0.0390] · sep · **fragile** | **+0.0204** [+0.0047, +0.0395] · sep · **fragile** | **+0.0091** [−0.0055, +0.0254] · **NOT sep** | ⛔ **refcv5-v2 FAILS — separated WORSE than the echo control. refcv4b merely TIES it.** |
| **`os − ha`** — `BAR-REFCV5V2-2` | +0.0082 [−0.0082, +0.0267] · **NOT sep** | +0.0082 [−0.0081, +0.0274] · **NOT sep** | −0.0032 [−0.0179, +0.0133] · **NOT sep** | ⛔ **FAILS** — the predicate is `delta < 0 and hi < 0` |
| `os − ha0` | −0.3644 [−0.4219, −0.3128] · sep | −0.3645 [−0.4216, −0.3129] · sep | −0.3758 [−0.4336, −0.3233] · sep | both clear the do-nothing floor; **refcv4b by more** |
| **arm delta `refcv5-v2 − refcv4b`** (informational) | **+0.0114** [+0.0059, +0.0172] · sep · p 1.0 | **+0.0114** [+0.0052, +0.0174] · sep | — | ⛔ **refcv5-v2 is WORSE, separated** |
| `os − refcv3` | −0.1341 [−0.1536, −0.1128] · sep | — | −0.1455 [−0.1655, −0.1240] · sep | both beat refcv3; **refcv4b by more** |
| **absolute `ade_m`** | **0.3079** [0.2795, 0.3390] | **0.3078** | **0.2965** [0.2697, 0.3280] | — |
| ⭐ CONTROL `ha` / `ha0` / `ha0_ext`, cross-arm | **0.0000** [0, 0] | **0.0000** [0, 0] | — | all three read **exactly 0**, as they must |

⭐ **The tool's own verdict on the bar row, carried VERBATIM:** *"⛔ NOT YET MEASURED —
separated but FRAGILE (the interval reaches back to within 25 % of the point estimate). At ONE seed
this is not a direction."* ⇒ **both readings stand: the arm MISSED its bar, and the SIZE of the
miss is not yet a quotable direction.**

**Grid:** 4,823 windows / 141 episodes, `dt_s` 0.5, `horizon_steps` 4 (2 s), `n_boot` 2000,
`PAIRING OK`, `constant_arm_checks` 5/5. ⛔ `overlapping_holdout_se` is not used; point
estimates are `full_set`. ⚠️ **ORACLE nav on BOTH arms** — all 4,719 v7.2 nav records
are `ego-future`; fair between the arms, optimistic in absolute terms, and never a production command.

#### 1d.6.2 ⭐ The inference-seed replicate — the third variance, MEASURED

`land.log` records `--infer-seed 0` and `--infer-seed 1` on the **same checkpoint**. The planner
samples, so this is the replicate the third-variance rule demands.

| quantity | seed 0 | seed 1 | Δ |
|---|---|---|---|
| `ade_m` (`os`) | 0.3079 | 0.3078 | **0.0001** |
| the bar `os − ha0_ext` | +0.0205 | +0.0204 | **0.0001** |
| distinct anchors selected | 62 | 64 | 2 — *the sampler really did re-roll* |

⇒ **Inference-seed floor on this rig ≈ 0.0001 m, ~200× below the +0.0205 bar miss —
so the FAILURE is robust to inference seed.** refav1's ~0.30 m floor does **not** transfer here.
⛔ **Training variance is UNMEASURED** — one training seed per arm, so `H-ESTIM-SEED-1`
still binds on the `+0.0114` arm delta.

#### 1d.6.3 ⛔ THE FOUR BINDING FAMILIES — per family, with n. None pending, none pooled.

| family | metric | **refcv5-v2** | **refcv4b** | n | better |
|---|---|---|---|---|---|
| **LONGITUDINAL** | `speed_mae_mps` | 0.2919 | **0.2900** | 4,823 / 141 ep | refcv4b |
| **LONGITUDINAL** | `speed_bias_mps` | +0.0399 | **+0.0331** | 4,823 | refcv4b |
| **LONGITUDINAL** | `along_mae_m` ⭐ *carries the ADE result* | 0.2655 | **0.2544** | 4,823 | **refcv4b** |
| **LONGITUDINAL** | `accel_mae_mps2` | **0.3596** | 0.4346 | 4,823 | **refcv5-v2 (−17.3 %)** |
| **LONGITUDINAL** | target-speed acc @ 0.5 / 1.0 / 2.0 m/s | **0.8346** / **0.9437** / 0.9881 | 0.8329 / 0.9422 / **0.9892** | 4,823 | split |
| **LONGITUDINAL** | **distance-keeping** `headway_min_m` | 27.8285 [23.9407, 32.4796] | 28.4517 [24.4003, 32.7611] | **1,308 / 68 ep** vs **1,224 / 67 ep** | overlapping |
| **LONGITUDINAL** | `time_gap_min_s` | 4.0885 [3.2646, 5.0234] | 4.1118 [3.2775, 5.0945] | 1,167 / 1,154 | overlapping |
| **LONGITUDINAL** | `min_ttc_s` ⚠️ **censored** | 24.6454 [23.2247, 26.0281] · `n_closing` **524** | 24.8434 [23.3648, 26.1655] · `n_closing` **470** | 1,308 / 1,224 | overlapping |
| **LATERAL** | `heading_mae_deg` | **1.2121** | 1.2964 | 4,823 | refcv5-v2 |
| **LATERAL** | `yaw_rate_mae_degps` | **1.0534** | 1.7368 | 4,823 | **refcv5-v2 (−39.4 %)** |
| **LATERAL** | **masked** `curvature_mae_1pm` | **0.003485** | 0.008150 | **13,553** / **13,558** valid step-pairs | **refcv5-v2 (−57.2 %)** |
| **LATERAL** | ⭐ **ratio to the straight-line floor** (0.006802) | **0.5123 — BELOW** | 1.1982 — **ABOVE** | same | **refcv5-v2** |
| **LATERAL** | `cross_mae_m` | 0.0994 | **0.0978** | 4,823 | refcv4b (marginal) |
| **TACTICAL** | lateral decision acc / kappa | 0.9563 [0.9430, 0.9676] / 0.8193 | **0.9579** [0.9455, 0.9687] / **0.8277** | 4,823 | refcv4b (CIs overlap) |
| **TACTICAL** | longitudinal decision acc / kappa | **0.8354** [0.8121, 0.8580] / **0.5463** | 0.8260 [0.8019, 0.8500] / 0.5186 | 4,823 | refcv5-v2 (overlap) |
| **TACTICAL** | `accelerate` recall | **0.5967** | 0.5056 | 538 true | **refcv5-v2** |
| **TACTICAL** | `brake_stop` recall | 0.5008 | **0.5388** | 631 true | refcv4b |
| **TACTICAL** | goal setting: FDE m / bearing MAE° | 0.6607 [0.6026, 0.7239] / **1.3730** | **0.6345** [0.5781, 0.6964] / 1.5484 | 4,823 / 4,614 bearing | split |
| **TACTICAL** | `anchor_acc` vs chance **1/117** = 0.008547 | 0.5271 [0.4858, 0.5685] — **61.67×** | 0.5275 [0.4883, 0.5664] — **61.72×** | 4,823 | **indistinguishable** |
| **TACTICAL** | `n_distinct_selected` of 117 | **62** (seed 0) / **64** (seed 1) | **51** | 4,823 | refcv5-v2 — *diversity, not accuracy* |
| **TACTICAL** | `anchor_selection` (fan scoring) | ⛔ **UNAVAILABLE** | ⛔ **UNAVAILABLE** | 4,823 each | *one path per window — no fan to score. A WORK ITEM, not a pass* |
| **STRATEGIC** | `route_acc` vs chance **1/3** | 0.7708 [0.7146, 0.8254] | **0.7791** [0.7248, 0.8318] | **3,622 / 128 ep** | refcv4b (CIs overlap) |
| **STRATEGIC** | `route_kappa` | 0.4614 | **0.4864** | 3,622 | refcv4b |
| **STRATEGIC** | ⭐ `route_pred_identical_under_nav_shuffle` | **1.000000** | **1.000000** | 3,622 | *structural — **NOT an echo of its input*** |

⭐⭐ **refcv5-v2's ONE unambiguous structural win is LATERAL SHAPE.** The straight-line floor
is the curvature error of a plan that **never steers**. refcv4b reads **1.1982× it** — the
instrument calls that *"a SHAPE defect: a plan that never steers tracks the road better"*. refcv5-v2
reads **0.5123×** — *"tracks the road better than a plan that never steers"*. ⇒ WP-4's
control-space anchored Gaussian **fixed the path shape it was pre-registered to fix**, and still
failed the bar, because ADE is dominated by the **along-track** term where it lost.
⛔ A curvature number is inadmissible without its validity mask: `min_ds_m` 0.25, with
**1,140** / **1,146** steps excluded and counted.

⭐ **The route head is not an echo.** Shuffling the nav token changes **nothing** in the route
prediction (identity 1.000000, both arms) because the head reads the observed window only and never
the nav token. ⚠️ `nav_echo_index` (0.6458 / 0.6422) must be computed through
`_ROUTE_TO_NAV {0:1, 1:0, 2:2}`; comparing the raw indices is a **TYPE ERROR** (3-wide
`ROUTE_CLASSES` vs 4-wide `NAV_COMMANDS`) and published a wrong number on 2026-09-06.

⛔⛔ **CORRECTION carried here so it cannot rot: "refcv5-v2 is the only arm with a strategic
output; refcv4b reads n = 0" is FALSE.** `STRATEGIC` has two children and mixing them manufactures
the difference: `as_declared_by_refcv3_arm` reads `UNAVAILABLE, n 0` for **BOTH** arms, and
`computed_here` reads `OK, n 3,622` for **BOTH**. Reading one child for one arm and the other child
for the other arm is a **sub-key scope error** — the `df` / cgroup / `step_s` family, with the
scope being a JSON key.

#### 1d.6.4 ⛔ What these numbers may NOT be credited to

1. ⛔ **The 22-token tactical vocabulary was NEVER SUPERVISED.** `--tac-goal-tok-head` is passed
   (head built, **11,286** params) but **`--w-tac-goal` is absent from `argv`** ⇒ default
   **0.0**. MEASURED from the checkpoint's own tensors: `max|w| = 0.0441898` against the `nn.Linear`
   init bound `1/sqrt(512) = 0.0441942` and `E[max of 11,264 draws] = 0.0441903` — agreement to
   six significant figures, i.e. **still at initialisation**.
2. ⛔ **Three heads, 18,472 params, took no gradient:** `core.decoder.offset_head` (6,160, at its
   init bound), `tac_goal_tok_head` (11,286, at its init bound), `scorer.goal_point` (1,026,
   **exactly zero** — 0 of 1,024 weights non-zero, so it emits the constant origin).
   ⭐ The discriminating control: the **fourth** goal head, `tac_goal_head` (6,156, the
   *geometric* one), reads `max|w| = 0.209529` = **4.74×** its bound — it **did** train, so
   "nothing trained" is wrong. ⚠️ `tac_goal_head` ≠ `tac_goal_tok_head`
   (`refc_v3.py:995` warns the name is taken).
3. ✅ **`goal_setting` is nonetheless ADMISSIBLE** — `four_families.tactical_from_trajectory`
   computes every one of its numbers from `pred[:, -1]`, the **planned path's last waypoint**, and
   never touches a goal head. ⇒ valid as a trajectory readout; **not** evidence any goal head works.
4. ⛔ **The DiffusionDrive agent coupling was ABSENT, not tested** — `--agents off`
   throughout (`agent_join = None`, `agent_join_digest = None`; `--w-agent` effective 0.0,
   `builds_graph false`). ⇒ nothing here supports or refutes WP-6.
5. ⛔⛔ **THE COMPARISON IS NOT SINGLE-LEVER, AND THE EVALUATED ARM IS NOT THE RUN THE
   REGISTRY LAUNCH TABLE DESCRIBES.** MEASURED by argv diff: refcv5-v2
   (`refcv5-v2-noagents-b1-v72-40k`) moves **TWO live mechanism groups** against refcv4b —
   **WP-4** (`--sampler ddim --w-u0 0.5`) **and P14** (`--sel-refined --sel-score-emitted`,
   a selection change that carries no loss term and is therefore invisible to `effective_weights`).
   `--tac-goal-tok-head` is inert (never supervised) and `--agents off` is inert. ⚠️
   **`--sel-refined` ALONE is the 0.0259 m separated-WORSE lever (`D-REFCV4B-SELREF-1`); this
   arm carries the PAIRED form, which `refc_v3_train.py:386-401` refuses to let be split, so
   that number is NOT this arm’s handicap.** ⇒ **the +0.0114 m loss is NOT attributable
   to WP-4.** ⭐ Cheapest discriminating next step, **zero training**: re-score the banked
   dump with `--ablate sel_refined` (`raw/paired_ablate.py`) to separate WP-4 from P14.
   See `MODEL_REGISTRY.md` §4.7.0 / §4.7.0b.

---

## 1e. THE **B1-CORPUS** LINE, CONTINUED — `refcv4b` and `refcv5-v2` · ⚠️ **loop = OPEN** · *added 2026-09-10*

⭐ **§1d's cells were PENDING because the eval did not exist. It exists now** — for the two arms
*after* refcv3, on the same B1 surface, with all four families and a paired estimator. This section
fills them. ⛔ refcv3's own §1d cells stay PENDING: this panel scores refcv4b and refcv5-v2, and
carries refcv3 only through the two paired deltas at the bottom.

**Tier `T1` · loop = OPEN** (self-action open loop — one forward pass at t0, the path is the model's
own selection; a planner feeding its own predictor is STILL open loop, PI ruling 2026-09-02).
**Surface:** 4,823 windows / 141 episodes, `dt = 0.5 s`, `K = 4`.
**Estimator:** paired episode-cluster bootstrap (`taniteval/ci.py`), `n_boot 2000`, `seed 0`,
cluster = **episode**. ⛔ `overlapping_holdout_se` is used nowhere — it biases the POINT ESTIMATE,
not merely the interval.
**Artifact:** `C:\Users\Admin\refcv5v2_final\panel.log` (seed 0) and `panel_seed1.log` (seed 1);
package `TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/`.
**Registry:** `MODEL_REGISTRY.md` §4.6 (refcv4b) and §4.8 (refcv5-v2).

⛔ **ADMISSIBILITY — the nav command is ORACLE-DERIVED.** Both arms consume the v7.2 nav command
and all 4,719 v7.2 nav records are `ego-future`, derived from the ego's own future path. It is a
**first-class route input** under the PI ruling of 2026-09-04 and both arms share it identically, so
the comparison is FAIR — but it is noiseless and perfectly timed where a real router is coarse.
⛔ **Neither arm's nav may be read as a production command.**

### 1e.1 THE FOUR FAMILIES

| family | metric | **refcv5-v2** | **refcv4b** | `ha` (hold-action) | `ha0_ext` (echo) | `ha0` (straight-line) |
|---|---|---|---|---|---|---|
| **ADE** | `ade_0_2s` m [CI95] | 0.3079 [0.2795, 0.3390] | **0.2965** [0.2697, 0.3280] | 0.2996 [0.2755, 0.3278] | **0.2874** [0.2649, 0.3137] | 0.6723 [0.6007, 0.7469] |
| **LONGITUDINAL** | speed MAE m/s | 0.2919 | 0.2900 | **0.2540** | **0.2540** | 0.4880 |
| | target-speed acc@0.5 | 0.8346 | 0.8329 | **0.8662** | **0.8662** | 0.7047 |
| | along-track MAE m | 0.2655 | 0.2544 | 0.2348 | **0.2341** | 0.4705 |
| | min headway m / time-gap s / min TTC s | 27.83 / 4.089 / 24.65 | 28.45 / 4.112 / 24.84 | 28.05 / 4.207 / 25.59 | 28.02 / 4.197 / 25.55 | 27.46 / 4.118 / 24.11 |
| **LATERAL** | heading ° | **1.2121** | 1.2964 | 1.5489 | 1.4322 | 2.7799 |
| | yaw-rate °/s | **1.0534** | 1.7368 | 1.4542 | 1.3395 | 2.3714 |
| | cross-track m | 0.0994 | **0.0978** | 0.1226 | 0.1070 | 0.3132 |
| | ⭐ curvature MAE 1/m (× floor) | **0.003485 (0.512×)** | ⛔ 0.008150 (**1.198×**) | 0.004030 (0.593×) | 0.003712 (0.546×) | 0.006802 (**1.000× = floor**) |
| **TACTICAL** | lateral acc / κ | 0.9563 / 0.8193 | **0.9579 / 0.8277** | 0.9382 / 0.7374 | 0.9419 / 0.7548 | 0.8659 / **0.0000** |
| | longitudinal acc / κ | 0.8354 / 0.5463 | 0.8260 / 0.5186 | 0.8443 / **0.6071** | 0.8443 / **0.6071** | 0.7576 / **0.0000** |
| | anchor selection acc (lift vs 1/117) | 0.5271 (**61.67×**) | 0.5275 (**61.72×**) | — | — | — |
| | goal FDE m / bearing MAE ° | 0.6607 / **1.3730** | **0.6345** / 1.5484 | 0.6588 / 1.6673 | 0.6323 / 1.5529 | 1.4029 / 2.8832 |
| **STRATEGIC** | route acc [CI95] | 0.7708 [0.7146, 0.8254] | **0.7791** [0.7248, 0.8318] | ⛔ n = 0 | ⛔ n = 0 | ⛔ n = 0 |
| | route κ · n | 0.4614 · 3,622 / 128 eps | **0.4864** · 3,622 / 128 eps | — | — | — |

⚠️ **`n = 0` on STRATEGIC for the three controls is STRUCTURAL, stated per family with its reason
rather than dropped:** `ha` / `ha0_ext` / `ha0` are model-free trajectory constructions that emit no
`route_pred` / `route_gt`. No strategic number exists for them by construction.
⚠️ **Distance-keeping is censored:** 784 of 1,308 refcv5-v2 windows never close on the lead and are
censored at `TTC_CAP_S = 30 s`. The means above are over censored data.
⚠️ **Curvature is the MASKED estimator** (refcv5-v2 `n_steps` 13,553, 1,140 excluded below
`min_ds = 0.25 m`). `ha0`'s curvature is exactly 0, so its MAE **is** the floor — a reference.

### 1e.2 THE PAIRED MARGINS — and both bars FAIL

| margin | delta m [CI95] | separated | verdict |
|---|---|---|---|
| **refcv5-v2 `os − ha0_ext`** (BAR-1) | **+0.0205 [+0.0043, +0.0390]** | **yes, WRONG WAY** | ⛔ **FAIL** |
| **refcv5-v2 `os − ha`** (BAR-2) | **+0.0082 [−0.0082, +0.0267]** | no | ⛔ **FAIL** |
| refcv4b `os − ha0_ext` | +0.0091 [−0.0055, +0.0254] | no | ⛔ **tie, bar not cleared** |
| refcv4b `os − ha` | −0.0032 [−0.0179, +0.0133] | no | ⛔ **tie** |
| **refcv5-v2 − refcv4b** (`os`) | **+0.0114 [+0.0059, +0.0172]** | yes | ⛔ **v2 separably WORSE** |
| refcv5-v2 − refcv3 (`os`) | −0.1341 [−0.1536, −0.1128] | yes | **win** vs refcv3 |
| refcv4b − refcv3 (`os`) | −0.1455 [−0.1655, −0.1240] | yes | **win** vs refcv3, by more |
| **CONTROL** v2 − refcv4b on `ha` / `ha0` / `ha0_ext` | **0.0000 [0, 0]** ×3 | no | ⭐ **ONE surface** — the comparison is admissible |

⛔⛔ **THE THING THIS PAGE EXISTS TO SAY: NEITHER ARM BEATS DOING NOTHING.** `ha0_ext` — an echo
control — scores **0.2874 m**, better than refcv4b's 0.2965 and refcv5-v2's 0.3079. Per §0.5 both
arms are **LOST** against `ha0_ext` and **tie** against `ha`. A win over refcv3 is a win over a
*worse arm*, not evidence of driving skill.
⭐ **The one family where a trained arm separably beats every control is LATERAL SHAPE:**
refcv5-v2's curvature MAE **0.003485** is **0.512×** the straight-line floor, against refcv4b's
**0.008150 = 1.198×** — refcv4b tracks the road *worse than a plan that never steers*, and v2 fixes
exactly that. That is the real, non-ADE finding on this row.

### 1e.3 Variance — which question the intervals answered

| question | status |
|---|---|
| *would another draw of EPISODES say this?* | ✅ **ANSWERED** — paired episode-cluster bootstrap, cluster = episode, `n_boot 2000` |
| *would another INFERENCE run say this?* | ✅ **ANSWERED** — refcv5-v2 samples (`--sampler ddim`), so the whole panel was re-run at a second inference seed. BAR-1 **+0.0204 [+0.0047, +0.0395]**, BAR-2 **+0.0082 [−0.0081, +0.0274]**, arm delta **+0.0114 [+0.0052, +0.0174]** — the verdict does not move |
| *would another TRAINING run say this?* | ⛔ **NOT MEASURED** — one training seed per arm (`H-ESTIM-SEED-1`) |

⚠️ **Panel multiplicity:** 6 of 9 cells read separated = **66.7 %**, against a pure-replicate
false-positive rate of **14.3 % (6/42)**. The panel rate exceeds the replicate rate; individual
cells still answer only the episode-draw question.

### 1e.4 ⛔ Two dead-weight facts that travel with the refcv5-v2 row

1. ⛔ **`tac_goal_tok_head`: 11,286 parameters, `grad_abs_sum` EXACTLY 0 for all 40,284 steps.**
   Wired, never trained ⇒ **no refcv5-v2 number may be credited to the 22-token tactical
   vocabulary**, including the TACTICAL row above.
2. ⛔ **`--agents off` throughout** ⇒ the DiffusionDrive agent coupling was **ABSENT, NOT TESTED**.
3. ⛔ **refcv5-v2 moved FOUR levers vs refcv4b** (`--sampler ddim --w-u0 0.5`, `--sel-refined
   --sel-score-emitted`, `--tac-goal-tok-head`, `--agents off`), so **no per-lever attribution is
   available from this row** — and one of them, `--sel-refined`, was MEASURED **0.0259 m separated
   WORSE** on refcv4b.

⭐ **Cheap and real:** `str_goal_head` is **771 parameters** delivering **0.7708** route accuracy
against **0.3333** chance, beside a `core` of **106,067,312**.

---

## 2. TIER **T0** — the split read on C1 · *what a single ADE column hides*

*MEASURED 2026-07-21, `python -m taniteval.runner driving-all`, CPU-only over the committed
`windows_<key>.pt`. Every interval is an **episode-cluster bootstrap** (B = 2000, 40 episodes);
every win/tie/LOST is a **paired** episode-cluster test against a trivial floor. Spec:
`TanitAD Research Lab/Benchmarks & Evals/TANITEVAL_V2_METRIC_SUITE.md`. Artifacts:
`taniteval/results/driving_<key>.json`. ⛔ **T0 — prediction quality, not driving.** The section
heading that stood here until 2026-08-23 was "Driving capability … the standard read"; that framing
is retracted, the numbers are unchanged.*

**ADE is one column, not the verdict.** The same 0.43-vs-0.47 that reads as a tie in §1 decomposes
into three different competencies here, and the arms rank differently on each.

| arm | ADE@2s m [ep-cluster CI95] | along / cross @2s m (vs CV) | speed MAE m/s: model vs hold-v0 (vs CV) | cruise Δ m/s vs hold-v0 | heading on straights ° | κ-sign | tick p50 | where the win lives |
|---|---|---|---|---|---|---|---|---|
| flagship-30k | **0.4271** [0.3675, 0.4871] | 0.841 **tie** / 0.237 win | 0.471 vs 0.482 **tie** | −0.212 **LOST** | 7.98 vs CV 1.399 | 0.954 | 97.3 ms | **lateral only** |
| refc-xl-30k | **0.4714** [0.3896, 0.5556] | 0.878 win / 0.280 win | 0.455 vs 0.482 **tie** | −0.069 **LOST** | 3.863 vs CV 1.399 | 0.919 | 44.1 ms | both axes |
| refc-base-30k | **0.4728** [0.3835, 0.5699] | 0.866 win / 0.292 win | 0.446 vs 0.482 **tie** | −0.054 **LOST** | 5.834 vs CV 1.399 | 0.916 | 21.8 ms | both axes |
| flagship-v16-ab-ft | **0.4375** [0.3423, 0.5501] | 0.683 win / 0.423 win | **0.389 vs 0.482 win** | −0.058 **LOST** | 7.687 vs CV 1.399 | 0.865 | — | both axes |
| refc-v12-k16reg | **0.4576** [0.3742, 0.5438] | — | — | — | — | — | — | both axes |
| refc-v12 | **0.4625** [0.3781, 0.5486] | — | — | — | — | — | — | both axes |
| refc-xl-live | **0.4788** [0.3977, 0.5638] | — | — | — | — | — | — | both axes |
| refc-small-30k | **0.5261** [0.4295, 0.6262] | 0.970 win / 0.314 win | 0.506 vs 0.482 **tie** | −0.091 **LOST** | 3.531 vs CV 1.399 | 0.921 | 11.5 ms | both axes |
| refb-v2-30k | **0.5913** [0.4766, 0.7131] | 1.029 **tie** / 0.408 win | 0.530 vs 0.482 **LOST** | −0.097 **LOST** | 5.75 vs CV 1.399 | 0.907 | — | lateral only |
| refc-xl | **0.6048** [0.5170, 0.7009] | 1.033 **tie** / 0.340 win | 0.572 vs 0.482 **LOST** | −0.186 **LOST** | 8.974 vs CV 1.399 | 0.869 | 44.0 ms | lateral only |
| flagship-speed | **0.6152** [0.5422, 0.6951] | 1.178 **tie** / 0.407 win | 0.663 vs 0.482 **LOST** | −0.406 **LOST** | 8.992 vs CV 1.399 | 0.927 | 99.6 ms | lateral only |
| refb-v2-20k | **0.6435** [0.5410, 0.7516] | 1.109 **tie** / 0.434 win | 0.579 vs 0.482 **LOST** | −0.128 **LOST** | 6.222 vs CV 1.399 | 0.889 | — | lateral only |
| refb-10k | **0.8372** [0.6753, 1.0218] | 1.157 **tie** / 0.894 win | 0.546 vs 0.482 **LOST** | −0.115 **LOST** | 2.181 vs CV 1.399 | 0.803 | 60.5 ms | lateral only |
| flagship-v4.1-10k | **0.8522** [0.7468, 0.9800] | — | — | — | — | — | — | neither axis separated |
| refb | **0.8629** [0.6928, 1.0385] | 1.153 **tie** / 0.960 **tie** | 0.541 vs 0.482 **LOST** | −0.106 **LOST** | 1.854 vs CV 1.399 | 0.781 | 59.8 ms | neither axis separated |
| flagship-v4.2-step4000 | **0.9869** [0.8795, 1.1088] | — | — | — | — | — | — | neither axis separated |
| flagship-v3enc-10k | **1.9654** [1.6556, 2.2859] | — | — | — | — | — | — | neither axis separated |
| refa-dinov2 | **2.1675** [1.9081, 2.4212] | 3.099 **LOST** / 0.578 win | 1.775 vs 0.482 **LOST** | −1.472 **LOST** | 1.925 vs CV 1.399 | 0.866 | 88.6 ms | lateral only |
| flagship-nospeed | **3.0175** [2.5450, 3.5444] | 4.968 **LOST** / 0.448 win | 2.521 vs 0.482 **LOST** | −2.250 **LOST** | 5.986 vs CV 1.399 | 0.948 | 101.6 ms | lateral only |
| refa-dynin-30k | **3.0471** [2.4984, 3.6878] | 4.536 **LOST** / 0.782 **tie** | 2.379 vs 0.482 **LOST** | −1.963 **LOST** | 4.781 vs CV 1.399 | 0.838 | 84.5 ms | neither axis separated |
| flagship-v2-6k | **5.9396** [4.3273, 7.6249] | 11.659 **LOST** / 3.066 **LOST** | 6.353 vs 0.482 **LOST** | −7.282 **LOST** | 16.867 vs CV 1.399 | 0.751 | — | neither axis separated |

*Blank cells are **NOT MEASURED**, not zero: the six arms added to this table on 2026-08-23 have a
`driving_<key>.json` with a full headline + verdict block, but were never carried into the
2026-07-21 panel narrative. Their `where the win lives` is read from their own
`verdict.where_the_win_lives`. Filling the remaining columns is a **zero-GPU** pass — §12 W-6.*

**Column definitions.** `along/cross` = the Frenet split of the 2 s residual on the GT path tangent
(orthonormal, so `along² + cross² = ‖err‖²` exactly); the tag is the paired test **vs CV**.
`speed MAE` compares the planned speed profile to the realised one, floor = **hold-v0**; the tag is
the paired test **vs CV**. `cruise Δ` = **L1 CRUISE-QUALITY**, speed MAE on the 639 longitudinally
steady windows, paired vs hold-v0. `heading on straights` = **T3** on the 634 windows with
\|net heading\| < 5°. `κ-sign` = **T4** curvature *sign* agreement (the curvature *magnitude* is
refused at this resolution — MEASURED 24× the signal). `tick p50` = panel 04b, fp32, batch 1, A40.

### What the split changes — five readings a single ADE column hid

1. **The rank-1= three-way tie is not a tie on prediction quality.** All three beat CV on ADE, but
   only the two REF-C arms beat CV **along-track** (XL +0.2170 [+0.0584, +0.3783]; base +0.2300
   [+0.0773, +0.3816], both separated). **flagship v1's along-track win is +0.2543 [−0.0278,
   +0.5304] — not separated.** The programme's flagship is the *only* member of its own rank tier
   with no CI-separated longitudinal competency; its entire separated advantage is lateral
   (cross-track +0.7720 [+0.4166, +1.1914]) — and §1a.6 shows a T0 lateral advantage is largely an
   action echo.
2. **flagship-v1.6 is an ADE tie and a longitudinal win.** The **only arm in the programme** whose
   speed MAE beats CV with a separated interval (+0.0785 [+0.0066, +0.1516]); best along-track
   anywhere (0.683 m), best progress error (0.697 vs v1's 0.837), longitudinal share of squared
   error **0.8933 → 0.5638**. It pays laterally: cross 0.423 vs 0.237, path geometry 0.204 vs 0.111,
   κ-sign 0.865 vs 0.954. *"Unfreezing changed nothing measurable"* is exact **on ADE** and wrong in
   both directions on the split: unfreezing **traded lateral geometry for longitudinal tracking.**
3. **No arm in the programme can hold a steady speed as well as doing nothing.** Every one of the 14
   panelled rows is CI-separated **against** hold-v0 on the 639 steady windows — −0.054 (refc-base)
   to −7.28 (flagship v2). A programme-level finding, not a flagship quirk (§3.0).
4. **On going straight, the ADE ranking inverts.** CV scores 1.399° mean heading error on the 634
   straight windows; the best arms there are the two *worst* on ADE among the trained set (`refb`
   1.854°, `refb-10k` 2.181°) while flagship v1 scores 7.98° and REF-C-XL 3.863°. On sharp curves it
   flips back (flagship v1 3.811° vs `refb` 26.559°, §4). Neither ordering is visible in ADE.
5. **A catastrophic ADE can hide an intact competency.** `flagship-nospeed` (3.0175, the no-speed
   ablation control) still beats CV on cross-track (+0.5611 [+0.1934, +0.9886], separated) and posts
   κ-sign 0.948; **98.97 %** of its squared error is along-track. Its failure is *purely*
   longitudinal — the cleanest confirmation of the speed-channel result we have, invisible in the scalar.

### 2.5 TIER **T0** — the v5f / v5.8f line on corpus **C2** *(w120 cylindrical — cross-frame, never comparable to §1)*

⚠️ **NOT COMPARABLE TO ANY 256×256-PINHOLE NUMBER.** Different geometry, different corpus, different
val episode set (600 vs 40). ⛔ **Every selected-ADE row below is measured under an ORACLE GOAL** —
the artifacts stamp `goal_provenance.goal_source = "oracle_gt_future"`, `is_oracle: true`,
`deployable: false`. **These are upper bounds on a non-deployable configuration.**

| arm | metric | value | tier | estimator / interval | evidence | artifact |
|---|---|---:|:--:|---|---|---|
| `flagship-v5f-w120-30k` | selected ADE@2s | **0.4011** | T0 | point on the fixed 881 grid — **no interval** | MEASURED 2026-08-09 | `…/2026-08-18-v58f-artifact-banking/gates/i4a_none.json` |
| `flagship-v5f-w120-30k` | **oracle (best-in-fan) ADE@2s** | **0.1975** | T0 | point | MEASURED 2026-08-09 | same |
| `flagship-v5f-w120-30k` | `sel_gap` | **0.2036** | T0 | point | MEASURED | same |
| `flagship-v5f-w120-30k` | miss@2m | 0.1487 | T0 | point | MEASURED | same |
| `flagship-v5f-w120-30k` | `wm_canary_ade@2s` | 1.2450 | T0 | point *(inert-controller regime — benign by construction)* | MEASURED | same |
| **v5.8f** `rescorer-top8-kincost` | selected ADE@2s | **0.4815** [0.3928, 0.5771] | T0 | **episode-cluster bootstrap**, B = 2000, 40 clusters | MEASURED 2026-08-10 | `…/2026-08-07-hierarchical-wm-redesign/v58f_rescore_ci.json` |
| **v5.8f** frozen-argmax *(control)* | selected ADE@2s | 0.7933 [0.6414, 0.9757] | T0 | ep-cluster bootstrap | MEASURED 2026-08-10 | same |
| **W4** UnicycleEmission fan | oracle ADE | **0.1077** | T0 | point | MEASURED 2026-08-10 | `…/2026-08-07-hierarchical-wm-redesign/w4_gate.json` |
| **W4** | selected-candidate accel MAE | **0.515** m/s² *(vs v5f's 8.10 — 16×)* | T0 | point | MEASURED | `v58f_rescore_ci.json` / `w4_gate.json` |
| **W7-w4r** K=32 (repaired trunk) | selected ADE | **3.6142** *(gate FAIL vs 0.4505)* | T0 | point; ρ across-window **0.439** [gate ρ≥0.3, CI excludes 0] | MEASURED 2026-08-11 | `…/2026-08-18-v58f-artifact-banking/gates/w7_w4r_k32_gate.json` |

**I4a imagination ablation — the imagination channel is LOAD-BEARING** (same ckpt, same grid, only
the imagination input changed; MEASURED 2026-08-11):
`intact` **0.4011** (oracle 0.1975, miss 0.1487) · `zeroed` **7.6493** — **19× collapse** (oracle
1.4573, miss 0.8048) · `shuffled` **1.2492** — 3.1× (oracle 0.4259, miss 0.2974).
Artifacts `gates/i4a_{none,zero,shuffle}.json`. The ordering **zero ≫ shuffle ≫ intact** is the
discriminating result: shuffling preserves the marginals and destroys only the window↔consequence
correspondence, so the planner reads imagination as **content**, not as a bias term.
⚠️ Stamped: the head was **trained** with imagination present, so this measures the dependence of
*this* architecture, not the value of retraining without it.

**Four families for `flagship-v5f-w120-30k` on C2** — artifact
`…/2026-08-07-hierarchical-wm-redesign/v5f_four_families_30k.json`, MEASURED, **point estimates, no
intervals in the artifact**:

| family | value |
|---|---|
| **LONG** | speed MAE **0.7024** m/s · speed bias **+0.1009** · along MAE **0.3031** m · along final bias +0.1193 · **accel MAE 8.1075 m/s²** · ego progress ratio 0.9961 (median 1.0007, n 850) |
| **LONG** distance-keeping | ⛔ **NOT MEASURED — WORK ITEM**, `status: UNAVAILABLE`, **n = 0**, no lead track supplied. §12 W-1 |
| **LAT** | heading MAE **4.0779°** · yaw-rate MAE **49.6867 °/s** · curvature MAE **0.297514 1/m** (bias −0.047794) · cross MAE **0.1819** m (final 0.5311) |
| **TACTICAL** | ⛔ **NOT MEASURED — WORK ITEM**, `status: UNAVAILABLE` — tactical decisions are not present in the scored pass. §12 W-4 |
| **STRATEGIC** | ⛔ **NOT MEASURED — WORK ITEM**, corpus reason as §1a.5. §12 W-3 |

⭐ **The accel MAE 8.1075 and yaw-rate 49.69 °/s are the point of this block.** The waypoint fan is
kinematically infeasible: **97.6 %** of all 256×20×881 fan steps violate \|a\| ≤ 4 ∨
\|yr\| ≤ 0.33 v + 0.05, and **100 %** of selected / oracle / all candidates are infeasible at the
>5 %-bad-steps threshold (`x0_lite_f32.json`). W4's unicycle re-parameterisation fixes it by
construction (accel MAE 0.774, violations 0.0) and **nearly halves the oracle** (0.1975 → 0.1077) —
the waypoint jitter was hiding coverage, not providing it. ⚠️ **The whole remaining deficit is
SELECTION:** three independent fast-scoring surfaces (pooled query / +kinematics / spatial
cross-attention) all failed the same held-out gate, and W7 (WM-roll re-rank) fails at every K with a
**winner's-curse** signature. Fast per-candidate scoring on this trunk is **RETIRED**
(MODEL_REGISTRY §1.13/§1.14).

### 2.6 TIER **T0** — `flagship-v1arch-v2bal-30k` on corpus **C4** *(the first COMPLETE four-family block)*

⛔ **NOT COMPARABLE to §1** (different corpus). ⛔ **This arm's CANONICAL-val numbers are
INADMISSIBLE**: **21 of the 40 canonical val episodes are inside its 9,000-clip training pool**
(`Project Steering/LEAK_v1arch_val_2026-08-05.md`). C4 (`physicalai-oodval-6f4b94e4c7ce-q90`, 290
clips, **zero** overlap) is the admissible corpus and is the only one quoted here.
*MEASURED 2026-08-05, episode-cluster bootstrap over 290 clips, B = 2000, 6,382 windows. Registry
anchor **MODEL_REGISTRY §1.9**; artifacts
`…/incoming/2026-08-05-v1arch-oodval-four-families/` (RESULT.md + raw JSON); protocol
`Project Steering/EVAL_PROTOCOL_OODVAL_2026-08-05.md`.*

| family | headline | reading |
|---|---|---|
| — (ADE) | `ade_mean_4wp` **0.5752** [0.5370, 0.6142] · `fde_2s` **1.4018** [1.3040, 1.5010] | |
| **LONGITUDINAL** | speed bias **+0.484 m/s** · along final bias **+0.943 m** · ego progress **1.0795×** · **time-gap at 15+ m/s 1.43 s** | ⛔ systematic **over-speed** — **71.95 %** of windows ahead at 2 s, **75.51 %** faster than the human: a **prior**, not a tail |
| **LONGITUDINAL** distance-keeping ✅ | **n = 2,846 lead windows**: headway **25.53 m** · time-gap **5.76 s** · min-TTC **14.73 s** (632 censored at the 30 s cap). Window states LEAD 3,002 / NO_LEAD 2,752 / NO_LABEL 628 | ⭐ **the ONLY block on this page where distance-keeping is MEASURED** |
| **LATERAL** | cross MAE **0.0552 m** [0.0500, 0.0611] · heading MAE 0.806° · curvature bias −0.000126 | tight; not where effort belongs |
| **TACTICAL** | κ **0.6033** (SUBSTANTIAL), agreement 0.8881 [0.8740, 0.9021] · **`seams_beneficial_of_3` = 0** | the manoeuvre label is honest; **the hierarchy seam is FALSIFIED at this checkpoint** |
| **STRATEGIC** ⚠️ | `route_acc_follow` **0.8031** == `majority_straight_rate` **0.8031** · `follow_pred_distribution` **{left 0, straight 1737, right 0}** · `route_acc_nav` **1.0000** | ⛔ **no vision-only route skill at all — a constant predictor.** `route_acc_nav = 1.0000` is an **ECHO of its own input**, not skill. Confirmed off-leak |

⚠️ **Two harnesses disagree 0.8 % on this corpus** and it is recorded, not smoothed over:
`eval_flagship_v4.py`'s MODE-A canary gives **0.5705**, `eval_four_families.py` gives **0.5752**.
Unresolved — §12 W-7. **JPEG-format control:** raw uint8 vs the q90 round-trip on the same 6,382
windows moves every metric **< 0.03** (largest: heading 0.0236°); q90 is the headline because it is
format-faithful.

### 2.7 TIER **T0** — the frozen-trunk unicycle readout line on corpus **C4** *(decoder-only contrast)*

*MEASURED 2026-08-06, paired episode-cluster bootstrap over 40 episodes, 2,000 draws, **6,834
stride-1 windows**, same frozen-trunk latent rolls for both arms. Registry anchor **§1.10 / §1.11**;
artifacts `…/incoming/2026-08-06-v1-defect-triage/results/{v16_full_eval.json.xz,
v16_distance_keeping.json, v16_tactical_executed.json}`. ⛔ **T0 — teacher-forced; the T1 reading of
these same arms is §1a.6, and it is much worse.***

| metric | v1arch | **v1.6 `flagship-v16-unicycle`** | Δ [CI95] | separated | **v1.7 `flagship-v17-speedloss`** |
|---|---:|---:|---|:--:|---:|
| ADE 2 s (m) | 0.3584 | **0.3398** | −0.0186 [−0.0706, +0.0293] | ✗ (parity) | **0.2849** (−0.0549 [−0.0713, −0.0412] vs v1.6, ✅) |
| speed bias (m/s) | +0.3793 | **−0.0265** | −0.4058 [−0.5162, −0.3022] | ✅ | — |
| along-final bias (m) | +0.7524 | **−0.0511** | −0.8036 [−1.0197, −0.5851] | ✅ | — |
| accel RMS (m/s²) | 2.9465 | **0.7172** | −2.2293 [−3.0886, −1.4641] | ✅ *(human ≈ 0.91)* | — |
| jerk RMS (m/s³) | 36.1682 | **1.1334** | −35.0348 [−46.6907, −24.8134] | ✅ *(human ≈ 1.71)* | 1.567 |
| net-yaw err (rad) | 0.0307 | **0.0108** | −0.0199 [−0.0248, −0.0153] | ✅ (−65 %) | Δ +0.0001, ✗ |
| heading MAE (rad/step) | 0.0027 | **0.0015** | −0.0012 [−0.0015, −0.0009] | ✅ | — |
| cross-track MAE (m) | 0.0502 | **0.0363** | −0.0139 [−0.0193, −0.0091] | ✅ | — |
| replan accel jump (m/s²) | 1.1310 | **0.1016** | *point est., 11× lower* | — | 0.125 |

**LONGITUDINAL distance-keeping ✅ MEASURED** (30 lead-bearing episodes, 880 stride-8 windows, LEAD
419 / NO_LEAD 329 / NO_LABEL 132; sign = *v1arch − v1.6* and *GT − v1.6*): min headway 25.23 /
**25.52** / GT 25.39 (−0.238 [−0.385, −0.091] ✅ · −0.026 [−0.225, +0.153] ✗); min time-gap 7.71 /
**7.98** / GT 7.98 (−0.138 [−0.252, −0.049] ✅ · −0.010 ✗); min TTC 14.97 / **17.82** / GT 17.10
(**−2.818 [−3.892, −1.784] ✅** · −0.597 ✗). ⇒ **v1arch is CI-separated more aggressive on all three;
v1.6 is statistically indistinguishable from GT on every distance-keeping metric at this n.**

**TACTICAL (executed) ✅ MEASURED**: agreement with GT-executed **v1arch 0.5016 → v1.6 0.7694**;
executed toggle rate 0.0620 → **0.0309** vs GT 0.0318 (paired Δ vs v1arch −0.0311 [−0.0462, −0.0177]
separated; Δ vs GT −0.0009 [−0.0066, +0.0055] NOT separated); executed dwell 2.61 → 4.38 s (GT 3.52).
v1arch over-calls `accelerate` 2,668 vs GT 975 — the longitudinal defect in decision space.
⚠️ TACTICAL **declared-head** metrics are identical to v1arch's (policies untouched; the declared
0.55 s dwell-toggling defect is NOT fixed by v1.6). **STRATEGIC: NOT MEASURED** — no map (§1a.5).

**v1.7 pre-registered gates (both outcomes committed before launch): the PRIMARY gates FAILED.**
P1 decel response ratio **0.1547** vs ≥ 0.40 ❌ · P2 accel lag **+0.173 s** vs ≤ +0.15 ❌ · N1–N5 all
✅. ⇒ **Outcome B binds: the position-loss hypothesis for the decel ramp is REFUTED for the speed-L1
lever.** v1.7 is registered as the best open-loop head in the lineage, **not** as the lag fix.

---

## 3. TIER **T2** — perception-closed loop *(AlpaSim NuRec n = 12 · real-footage low-OOD n = 40)*

*A DIFFERENT AXIS from §1–§2 — never mixed with a T0 ADE. See §5.5 for the full block, which is
retained unchanged below with its retractions and its mandatory "@ 2 s" qualifier. Standing T2
ordering, **triple-confirmed** across three independent instruments (n = 1 scene-dependent → n = 12
NuRec → n = 40 real-footage): **REF-C base > flagship v1**.*

### 3.0 Longitudinal regime on C1 — L1 CRUISE-QUALITY vs L2 TRANSIENT-RESPONSE [T0]

*(kept at its historical position in the reading order; **tier T0**.) Speed MAE (m/s) by realised
longitudinal regime (\|mean accel over the window\| vs ±0.5 m/s², a PROPOSED threshold). Floor =
hold-v0. Paired episode-cluster, orientation floor − model, so **positive = the model wins**.
`steady` n=639, `brake` n=95, `accel` n=147.*

| arm | steady: model / hold-v0 | paired Δ [CI95] | brake Δ | accel Δ |
|---|---|---|---|---|
| flagship-30k | 0.4231 / 0.2109 | **−0.2122** [−0.2778, −0.1443] **LOST** | +0.6433 win | +0.5716 win |
| refc-xl-30k | 0.2796 / 0.2109 | **−0.0687** [−0.1205, −0.0227] **LOST** | +0.0958 **tie** | +0.3998 win |
| refc-base-30k | 0.2646 / 0.2109 | **−0.0537** [−0.0981, −0.0119] **LOST** | +0.0810 **tie** | +0.3955 win |
| flagship-v16-ab-ft | 0.2684 / 0.2109 | **−0.0575** [−0.0954, −0.0228] **LOST** | +0.6404 win | +0.3906 win |
| refc-small-30k | 0.3016 / 0.2109 | **−0.0908** [−0.1544, −0.0344] **LOST** | −0.1304 **LOST** | +0.3316 win |
| refb-v2-30k | 0.3080 / 0.2109 | **−0.0971** [−0.1554, −0.0464] **LOST** | −0.0921 **tie** | +0.1949 win |
| refc-xl | 0.3972 / 0.2109 | **−0.1863** [−0.2306, −0.1451] **LOST** | −0.0621 **tie** | +0.3090 win |
| flagship-speed | 0.6167 / 0.2109 | **−0.4059** [−0.4791, −0.3266] **LOST** | +0.4780 win | +0.3690 win |
| refb-v2-20k | 0.3385 / 0.2109 | **−0.1277** [−0.1794, −0.0810] **LOST** | −0.1234 **LOST** | +0.0542 **tie** |
| refb-10k | 0.3262 / 0.2109 | **−0.1154** [−0.1768, −0.0627] **LOST** | −0.1361 **LOST** | +0.2026 win |
| refb | 0.3166 / 0.2109 | **−0.1058** [−0.1529, −0.0609] **LOST** | −0.1972 **LOST** | +0.2312 win |
| refa-dinov2 | 1.6831 / 0.2109 | **−1.4722** [−1.6692, −1.2879] **LOST** | −1.0228 **LOST** | −0.6922 **LOST** |
| flagship-nospeed | 2.4611 / 0.2109 | **−2.2503** [−2.6697, −1.8783] **LOST** | −1.5438 **LOST** | −1.4437 **LOST** |
| refa-dynin-30k | 2.1740 / 0.2109 | **−1.9632** [−2.4065, −1.6136] **LOST** | −1.9921 **LOST** | −1.5498 **LOST** |
| flagship-v2-6k | 7.4926 / 0.2109 | **−7.2817** [−9.5500, −5.0368] **LOST** | −1.9020 **LOST** | −2.3065 **LOST** |

**Cruise quality and transient response point in opposite directions for the same checkpoint, and
ADE averages them away.** flagship v1 is **2.0× worse than hold-v0** on the 639 steady windows while
winning brake (+0.6433) and accel (+0.5716) decisively — a model that disturbs a speed that needed
no disturbing. **72.5 % of the corpus is steady**, so the failure dominates the corpus and is still
invisible in the scalar because *geometry* carries ADE there (flagship steady ADE 0.3834 vs
hold-v0's 0.5430). `flagship-v16-ab-ft` cuts the cruise loss **3.7×** versus v1 (−0.0575 vs −0.2122)
*while keeping* v1's braking response (+0.6404 vs +0.6433) — the only arm that does both.

---

## 4. TIER **T0** — heading and curvature by GT curvature bucket on C1 (T3 / T4)

*Mean heading error @2 s, degrees. Buckets: straight <5° (n=634) / gentle 5–15° (n=103) / sharp ≥15°
(n=144) on \|net heading change\|. **R5 applies:** heading is heavy-tailed, so the median is shown
beside the mean and is the honest reducer — flagship v1's corpus mean of 6.61° carries a bootstrap
CI of [2.34, 12.02].*

| arm | straight: mean / median | gentle | sharp | κ-sign straight / gentle / sharp |
|---|---|---|---|---|
| flagship-30k | 7.980 / **1.105** | 2.060 | 3.811 | 0.947 / 0.932 / 0.998 |
| refc-xl-30k | 3.863 / **0.520** | 4.901 | 7.704 | 0.925 / 0.861 / 0.935 |
| refc-base-30k | 5.834 / **0.509** | 6.136 | 8.022 | 0.918 / 0.858 / 0.947 |
| flagship-v16-ab-ft | 7.687 / **0.757** | 7.504 | 9.653 | 0.876 / 0.786 / 0.875 |
| refc-small-30k | 3.531 / **0.534** | 4.293 | 8.319 | 0.927 / 0.877 / 0.926 |
| refb-v2-30k | 5.750 / **0.608** | 5.843 | 10.855 | 0.908 / 0.845 / 0.944 |
| refc-xl | 8.974 / **1.682** | 6.798 | 8.891 | 0.868 / 0.806 / 0.919 |
| flagship-speed | 8.992 / **1.470** | 2.100 | 4.487 | 0.914 / 0.916 / 0.991 |
| refb-v2-20k | 6.222 / **0.678** | 8.328 | 11.298 | 0.888 / 0.819 / 0.942 |
| refb-10k | 2.181 / **0.933** | 6.923 | 21.666 | 0.850 / 0.647 / 0.708 |
| refb | 1.854 / **0.700** | 7.372 | 26.559 | 0.862 / 0.537 / 0.597 |
| refa-dinov2 | 1.925 / **0.895** | 6.168 | 17.915 | 0.891 / 0.754 / 0.840 |
| flagship-nospeed | 5.986 / **1.184** | 21.305 | 41.720 | 0.952 / 0.922 / 0.951 |
| refa-dynin-30k | 4.781 / **0.976** | 6.658 | 15.319 | 0.860 / 0.712 / 0.836 |
| flagship-v2-6k | 16.867 / **5.177** | 27.059 | 52.596 | 0.839 / 0.534 / 0.521 |
| ***CV floor*** | *1.399 / **0.451*** | *7.852* | *28.743* | *0.764 / 0.233 / 0.204* |

**The mean/median gap is the point.** On the mean, flagship v1 is **5.7×** worse than a straight line
at going straight (7.980 vs 1.399). On the **median** — the reducer R5 mandates — it is **2.45×**
(1.105 vs 0.451). Both are real; the mean says a tail of windows is badly wrong, the median says the
typical straight window is only moderately wrong. **Quote the median as the headline and the mean as
the tail evidence; never quote one alone.**

Every trained arm beats CV decisively on **gentle and sharp** curves and on **curvature sign** (CV
scores 0.233 / 0.204 there — a straight line has no sign to agree with). That is where the vision is
doing work — **at T0**; §1a.6 shows how much of it survives closing the loop.

---

## 5. Deployment axis — inference efficiency (panel 04b) *(tier-independent)*

*MEASURED on one A40, batch 1, ≥200 warmed iterations, per-iteration CUDA events,
`torch.cuda.synchronize()` bracketed, precision applied identically to every arm and recorded.
Source: `taniteval/results/eff_<key>.json` → `<precision>.plan_step.{p50_ms,p99_ms}` (canonical
files only; the `*.CONTAMINATED-*` runs are excluded). **Re-verified BY CONTENT 2026-08-23** — every
cell below is exact to the artifact.*

| arm | p50 fp32 | p99 fp32 | p50 tf32 | p50 amp16 | params | meets 10 Hz @p99 |
|---|---:|---:|---:|---:|---:|:--:|
| refc-small-30k | **11.50** | 11.56 | — | — | 54.7 M | ✅ (fp32; tf32/amp16 not measured) |
| refc-base-30k | **21.78** | 22.33 | 15.81 | 15.88 | 104.2 M | ✅ all three |
| refc-xl-30k | 44.06 | 44.44 | 27.78 | 21.00 | 251.9 M | ✅ all three |
| refb / refb-10k | 59.80 / 60.47 | 60.31 / 61.12 | — | — | 262.8 M | ✅ |
| refa-dynin-30k † | 84.52 | 128.36 | — | — | 156.6 M | ❌ |
| refa-dinov2 † | 88.58 | 107.67 | — | — | 156.6 M | ❌ |
| flagship-30k | 97.32 | 122.77 | 97.70 | 123.83 | 263.4 M | ❌ all three |
| flagship-nospeed | 101.58 | 127.91 | — | — | 263.4 M | ❌ |

† excludes the external frozen encoder — not comparable to a pixels-in arm.

**Admissibility is a gate, ranking is a frontier, and there is no scalar composite.** An arm is
admissible only if it (a) meets the 10 Hz budget at **p99** in its declared deploy precision **and**
(b) beats every trivial floor on the headline capability metric with a CI-separated paired
bootstrap — **at T1**, per EVAL_DOCTRINE rule 2. ⛔ **On that reading NO arm in the programme is
admissible today**: §1a.1 shows both T1 arms losing to a hold-action control. Any single number
trading metres against milliseconds embeds an unmeasured exchange rate, and our arms rank
*oppositely* on the two axes (REF-C wins latency 2.2–4.6×, the flagship wins batched throughput 34.8
vs 29.9 windows/s @ batch 32). Report the Pareto frontier; do not collapse it.
R12 (closed 2026-07-21): the composed inference levers put flagship v1 at **18.75 ms p50 / 18.76
p99 = 53.3 Hz**, which *does* clear (a).

> ⚠ **Conflict on record, reported not resolved (registry R14).** MODEL_REGISTRY §6 reading 3 quotes
> the flagship tick as **103.42 / 93.76 / 104.49 ms** (fp32/tf32/amp16) and REF-C-XL amp16 as
> **26.12 ms**. The committed artifacts say **97.32 / 97.70 / 123.83** and **21.00** — re-confirmed
> here 2026-08-23. The registry itself flags these six figures as **UNRESOLVED SOURCE (2026-08-03)**
> and says *"do not re-cite"*. The flagship's own repeatability record (`eff_repeatability.json`, 5
> clean reps) is **99.03–100.05 ms p50**, bracketing neither. **This page quotes the committed
> artifact.** The conclusion is unchanged in every version. Needs one reconciliation pass — §12 W-8.

### 5.1 Deployment path — ONNX + TensorRT-FP16 + CUDA-graph (Orin / Thor) — MEASURED on an A40 proxy

*The deploy object is flagship v1's **planning tick** (`encode 1 new 9-ch frame → slide 8-state
window → 20 sequential operative steps → SE(2) accumulate`). MEASURED 2026-07-22 on an **A40
(SM 8.6) proxy**; raw under `TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-07-22-orin-thor-deployment/artifacts/`.*

| measurement | value | source JSON |
|---|---|---|
| composed planning tick (L1+L2+L3+L7) | **18.75 ms p50 / 18.76 p99 = 53.3 Hz** | `eff_levers_flagship-30k.json` (registry §1.2, R12) |
| CUDA-graph rollout, K=20 (predictor-only proxy) | eager **96.40 → graph 27.87 ms** p50 (**3.46×**) | `bench_latency_report.json` |
| predictor 1-call, fp32 / fp16 | 4.96 / 4.12 ms p50 | `bench_latency_report.json` |
| TensorRT-FP16 engine (A40 proxy) | encoder **1.205 ms** · predictor **0.666 ms** p50; **MHA fuses** | `trt_fp16_report.json` |
| static-shape ONNX export | encoder + predictor build clean, torch-vs-ORT parity ≤ **1.9e-6** | `export_report.json` |

**Per-chip precision map (PUBLISHED, vendor specs — a plan, not a measured tick):** Orin (Ampere
SM 8.7) → **FP16 baseline; INT8 only behind a per-layer benchmark; NO FP8, NO FP4**. Thor
(Blackwell) → **FP16/FP8 + NVFP4**. INT8 on an Orin ViT can run **~2.7× slower than FP16** on
non-optimal kernels — a per-layer hypothesis to disprove, never a default.

**⚠️ Hardware-blocked, stated honestly.** Every latency above is an **A40** number; the A40 TRT
engine is a **proxy — TRT engines are NOT portable across GPU architectures**. Real Orin/Thor
throughput, the on-device engine build and any NVFP4 number **need the target silicon** and are
**not fabricated** here. What the A40 build establishes and *does* transfer: the ONNX→TRT path
builds with no plugin, MHA fusion is achievable for our ViT (retires NVIDIA #4537 on SM 8.6),
CUDA-graph capture is exact, and the 20-step rollout is the binding term.

## 5.5 TIER **T2** — AlpaSim NuRec reconstructions (n = 12) · ⚠️ RECONSTRUCTION-OOD CONFOUNDED

*MEASURED 2026-07-22 on the AlpaSim closed-loop harness (NuRec photoreal reconstructions, **480×854**,
20 s rollouts) — the programme's **first external-simulator** closed-loop numbers. Raw
(`…/incoming/2026-07-22-alpasim-closedloop-evalpod/`): `REFC_suite_results.json`
(+ `REFC_suite_{base,xl}_results.json`), open-loop control `REFC_openloop_diagnostic.json`, flagship
`Flagship_v1_results-summary.json`. A **"pass" = no at-fault collision AND no off-road**; `mean
score` folds in progress-to-GT. **A DIFFERENT AXIS from §1–§5 — never mixed with a T0 ADE.***

> ⚠️⚠️ **HEADLINE — these numbers are ENV-CONFOUNDED, not a clean model result (`RETRACTION_LOG.md`
> C6).** The open-loop control settles it: **REF-C's open-loop ADE *on the AlpaSim reconstructions*
> is 1.52 m (de@2s 2.58), 3.21× its taniteval real-footage 0.4728** — consistent across **4 scenes /
> 288 predictions** (per-scene 1.40–1.77 m). REF-C is fed NuRec input **~3× off its training
> distribution**, so the pass rates below measure **model × reconstruction-fidelity, NOT the
> model.** The base-vs-XL *ordering* survives; *"REF-C collides closed-loop"* does **not** survive as
> a model indictment.

| arm | params | **at-fault collision** | off-road | **pass rate** | **mean score** | **dist-to-GT (m)** | progress-rel |
|---|---:|:--:|:--:|:--:|:--:|---:|---:|
| **REF-C-base** | 104.2 M | **33.3 % (4/12)** | 16.7 % (2/12) | **6/12** | **0.345** | **1.642** | 0.877 |
| **REF-C-XL** | 251.9 M | **33.3 % (4/12)** | 25.0 % (3/12) | **5/12** | **0.246** | 1.973 | 0.885 |

**⚠️ n = 12 — one scene = 8.3 pp**, and the raw JSON's own caveat is *"wide binomial CIs at n = 12"*.
Further caveats carried verbatim: **n = 12 subset** of the 916-scene public suite; **480×854** render;
**NuRec reconstructions, not real-world**. **base ≥ XL ORDERING holds under the shared OOD** — base
mean score 0.345 > XL 0.246, passes 6/12 vs 5/12, closer to GT (1.64 vs 1.97 m) at the same 33 %
at-fault rate. **Scale bought no closed-loop advantage.**

**Flagship v1 DOES drive closed-loop (via its `tactical_policy` head) — but a PAIRED n = 12 suite
REVERSES the n = 1 "beats REF-C" read.** MEASURED 2026-07-23, same 12 scenes, identical NuRec
renders, f-theta verified live: **REF-C base statistically beats flagship v1** — pass **8/12 vs
2/12**, mean score **0.496 vs 0.066**, paired Δ **−0.430 [−0.646, −0.215]** (scene-cluster boot95
excludes 0), score sign-test **8-0** (p = 0.008), pass-McNemar **6-0** (p = 0.031); **at-fault
collisions TIED** (1-1, p = 1.0). Mechanism: flagship's tactical head is a **high-deviation
planner** — plan_dev **1.12 vs REF-C 0.34** (3.3× wider) — so its failure mode is **off-road, not
collision** (8/12 offroad). **Resolution confound RESOLVED (2026-07-23):** a native-1080×1920 paired
re-run **holds the delta** (**−0.295 [−0.494, −0.117]**, sign-test 7-0) → the **model is the dominant
axis, resolution is second-order**.

**⭐ sim2real reconstruction-OOD axis CLOSED — REF-C base still beats flagship v1 on a REAL-FOOTAGE
low-OOD harness (n = 40, 2026-07-23).** Real-footage log-replay (drive the recorded frames, integrate
the ego kinematically, arc-length re-index + homography-warp for on-policy deviation — both arms held
at **1.02–1.20× OOD**, ≪ NuRec's 3.75×;
`…/incoming/2026-07-23-lowood-lanekeeping-refc/lowood_lanekeep_40ep.json`, episode-cluster bootstrap,
paired). Because a map/agent-free source **cannot** emit off-road/collision, it carries
**`corridor_departure_rate`** (on-policy |XTE| > 1.75 m lane-half-width).

> 🔴 **MANDATORY QUALIFIER — every number in this block is a *2-SECOND* closed-loop number.**
> Source-read (`lowood_closedloop.py:59`): the instrument rolls out **K = max(WP_STEPS) = 20 at
> DT = 0.1 → a 2.0 s horizon**. The failure mode these numbers are used to reason about — a junction
> crossing — is a **~20 s** event, and imitation compounding error scales ~**T²ε**. **Quote these as
> "closed-loop @ 2 s", never as "closed-loop" unqualified.** This does NOT overturn the *ordering*;
> it bounds what the ABSOLUTE rates mean.
> ⚠️ **Unreconciled:** the closed-loop research doc reports a junction **window**-departure of
> **0.368** against the 0.0134 all-strata / 0.064 junction figures below — a metric-definition
> mismatch (window-level vs episode-level) that must be reconciled BEFORE either is quoted (§12 W-10).

| n = 40 / 881 win, paired | flagship v1 | REF-C base | Δ (flag − refc) | separated |
|---|:--:|:--:|:--:|:--:|
| closed-loop ADE@2s (m) | 1.488 [1.329, 1.647] | **0.564** [0.452, 0.676] | +0.924 [+0.781, +1.065] | **yes** |
| `corridor_departure_rate`@1.75m | 0.0318 | **0.0134** | +0.0184 [+0.0077, +0.0328] | **yes** |
| peak XTE (m) | 0.764 | **0.442** | +0.321 [+0.193, +0.495] | yes |

**REF-C base wins in EVERY stratum → the ordering is TRIPLE-confirmed** (n = 1 → n = 12 NuRec →
n = 40 real-footage), so it is **not a reconstruction artifact**. It also **decomposes flagship's
deficit**: in longitudinal scenes both arms keep the lane near-perfectly (departure 0.4 % / 0.04 %)
yet flagship's ADE is 4× REF-C's (1.455 vs 0.354) → flagship's gap is **longitudinal, not
lane-keeping**; in junctions flagship departs 2.3× more (14.6 % vs 6.4 %, peak XTE 2.37 vs 1.46 m).
⚠️ **Bound:** lane-keeping / on-policy drift only, NOT off-road/collision; within-source relative.

> ⚠️ **Retractions on record (`RETRACTION_LOG.md`, 07-22/07-23).** **C5** — the n=1 *"REF-C collides
> at-fault"* over-read the worst-case scene `01d503d4`. **C6** — the n=12 *"REF-C fails ~half
> closed-loop"* is **reconstruction-OOD confounded**: run the open-loop-on-reconstructions control
> **before** attributing a closed-loop failure to the model.
> <!-- lint-ok: the next line QUOTES the C7-retracted claim in the act of reversing it. -->
> **C7 (07-23)** — the n=1 *"flagship v1 **beats** REF-C closed-loop"* is **reversed** by the paired
> n=12 suite: a closed-loop win from n=1 is scene-dependent — never headline it until n ≥ ~12.

---

## 6. What TanitEval deliberately does **not** measure

Refusals are part of the contract and are recorded in every `driving_<key>.json → refused`. Each is
a data limitation, not an oversight:

| refused | why |
|---|---|
| headway / distance-keeping / **TTC** *on C1 and C2* | no lead-agent state is joined to those grids (`lead_state` is a shape-fixed `None` stub). ⚠️ **This is a JOIN gap, not a data gap** — `obstacle.offline` exists on 97.44 % of the corpus and the family **is** measured on C4 (§2.6, §2.7). ⇒ a WORK ITEM (§12 W-1), not a permanent refusal |
| any **VTARGET**-referenced target-speed metric at 2 s | refuted with numbers — it sits +1.42 m/s above v0 and loses to holding v0 (MAE 1.65 vs 0.475); the right quantity at the wrong timescale |
| **intersection / roundabout / merge capability** | the events are 5–20 s, the horizon is 2 s, the clips ~20 s. A 2 s window inside a roundabout is kinematically indistinguishable from a constant-radius curve. The S1 strata are **kinematic signatures** (`launch_from_stop`, `stop_approach`, `sustained_turn`) and must never be renamed |
| **lane-centre deviation / lane-keeping** | no lane geometry exists; the only lane number in the codebase is a hard-coded `LANE_HALF_M = 1.75` proxy |
| naive **curvature MAE** at the C1 4-knot resolution | MEASURED 1.2015 vs a signal of 0.0495 1/m — 24× the signal. It measures knot jitter. Sign agreement survives and discriminates. *(At the dense 10 Hz surface — C2, C3 — curvature IS reported; see §1a.3, §2.5.)* |
| **collision rate / drivable-area / NAVSIM PDMS / nuPlan CLS / CARLA DS** | need agent boxes, an HD map or a simulator. See §9 |
| **STRATEGIC route/goal quality** on every PhysicalAI grid | no map, lane graph, junction label, traffic-light feature or route signal exists in the corpus; both available label sources are inadmissible (§1a.5). **Settled at five probes** |
| a scalar **capability × efficiency** composite | embeds an unmeasured exchange rate; the arms rank oppositely on the two axes |

---

## 7. Different-corpus measurements — do not mix with §1–§5

*These are **not** on the C1 val windows and are not comparable to the tables above. All **T0**.*

### 7.1 Trivial-baseline floor, comma2k19 + Cosmos-DD (2026-07-15) — camera/BEV mixed, 26 132 anchors

CTRV wins 55–58 % of anchors, CV 20–30 %. **Report `skill_score` = model_ADE ÷ best-of-3 floor.**

| stratum (comma-hwy, v≈25 m/s) | n | best-of-3 floor ADE@1s | CV@1s | CTRV@1s | floor@2s |
|---|---:|---:|---:|---:|---:|
| straight | 18 785 | **0.056 m** | 0.088 | 0.062 | 0.206 |
| gentle | 3 008 | **0.059 m** | 0.275 | 0.060 | 0.228 |
| sharp (speed-gated ≥2 m/s) | 212 | **0.164 m** | 0.404 | 0.167 | 0.608 |

Baselines use privileged GT ego-state → a *denominator*, not a competitor. Curvature strata are
speed-gated (κ = yaw_rate/v is singular at v→0; 12.4 % of comma anchors are near-standstill).
Source: `Implementation/incoming/2026-07-15-baseline-floor/`.

### 7.2 Ego-status shortcut ceiling, comma-hwy (2026-07-17) — metric-BEV, 7 920 val anchors

| predictor | L2@1s | L2@2s | L2@3s | avg |
|---|---:|---:|---:|---:|
| stop (null) | 24.88 | 49.85 | 74.89 | 49.87 |
| best-of-3 kinematic floor | 0.122 | 0.479 | 1.102 | **0.571** |
| **ego-status shortcut** (no vision, learned, held-out) | 0.144 | 0.552 | 1.256 | **0.658** |

comma highway is **73.9 % straight** — identical to nuScenes → our open-loop val inherits the same
ego-status-shortcut pathology (AD-MLP, arXiv 2312.03031). Convention note: `pointwise` = UniAD,
`cumulative` = ST-P3/VAD; they differ ~2×. The **in-corpus** version of this ceiling is the
**0.5735 m** ego-status ridge on our own C1 windows (§0.6).

> ⚠ **The warning this page must carry.** The set of metrics computable from ego logs alone is
> *precisely* the set the critique literature showed is gameable by an ego-status MLP with no
> perception. **A map-free suite cannot, on its own, discriminate perception quality.** The two
> antidotes are first-class members of the suite: the **ego-status ceiling** (0.5735 — flagship v1's
> 0.4271 clears it) and the **vision-ablation on high-divergence windows** (vision effect
> **+1.325 m, CI [+1.04, +1.64]**, CI-separated). ⚠️ Both are **T0** statements.

### 7.3 Supervised-IDM cross-domain probe (2026-07-22) — a FINDING, not a leaderboard model

*A ~2.9 M supervised inverse-dynamics head (latent window → speed / yaw-rate / steer / accel), the
pre-registered gate for a YouTube-scale IDM data pipeline. Raw:
`…/incoming/2026-07-22-idm-proof/results.json`. Gate: cross-domain speed R² > 0.9 AND yaw R² > 0.9
AND ADE@2s < 1.5× the in-domain held-out ADE.*

| split | in-distribution | cross-domain | verdict |
|---|---|---|---|
| PhysicalAI → **comma2k19** (primary go/no-go) | held-out speed R² **0.930**, yaw R² 0.924, ADE@2s 2.73 | comma speed R² **0.657**, yaw R² **0.000** ⚠️STALE-PENDING, ADE@2s 6.56 | **FAIL** (ADE ratio 2.40) |
| rig-A → **rig-B** (same corpus, other camera rig) | held-out speed R² 0.786, ADE@2s 4.36 | rig-B speed R² **−2.465**, yaw R² −0.109, ADE@2s 17.47 | **FAIL** (ADE ratio 4.01) |

> 🔴 **LABEL-PROTOCOL CORRECTION 2026-07-27 (C29) — read before quoting the comma `yaw R² 0.000`.**
> That cell was scored with **`heading_repair` OFF** and no `v_min` gate. comma2k19's heading is
> `arctan2` of the ENU velocity and is **undefined at standstill**: **26.27 % of comma frames below
> 0.5 m/s are physically impossible and 0.000 % above it** (PhysicalAI: zero in every bin, so the
> `0.924` / `−0.109` / rig-B cells are UNAFFECTED). **`0.000` measures the label, not the transfer.**
> Left in place for audit, marked **STALE-PENDING**: no repaired measurement exists on *this*
> substrate, so nothing may be substituted into the cell.
> 🔴 **AMENDED 2026-07-27 (class C43) — `+0.3308` is WITHDRAWN.** Settled **BY CONTENT** (sha256 of
> raw `poses` and `frames_u8` bytes, never filenames): **2 of the 22 comma val episodes are
> bit-identical to 2 of the deployed head's own 40 comma TRAINING clips**. Without them the same head
> reads comma yaw **R² −0.746 (CI [−1.574, −0.177])**. ✅ **`+0.679` is NOT withdrawn** (trained on a
> content-disjoint split) but reads **+0.3038 (CI [+0.054, +0.479])** on the 20 clean episodes.
> ⇒ **comma yaw is TESTABLE; the DEPLOYED head does not do it.** PhysicalAI unaffected.
> ⚠️ **The FAIL verdict is not overturned** — it rests on speed (0.657) and the ADE ratio 2.40,
> neither of which any heading label touches.

**The supervised-IDM paradigm works in-distribution and does NOT transfer.** It fails even the
*same-corpus, other-rig* split (rig-B speed R² −2.465 — worse than predicting the mean), so the
failure is **domain shift, not dataset**. Recorded as a finding; **no model row.**

---

## 8. Historical — camera-frame gate ladder (SUPERSEDED, different unit)

*Retained for traceability only. **Unit: camera-frame `ADE@1s`**, not metric-BEV `ade_0_2s`.*

### FLAGSHIP — step 27 000, route-resampled protocol, exact training val (comma+pai), 2026-07-12

| Gate | Verdict | Value | Note |
|---|---|---|---|
| **D1** (probe ADE@1s) | FAIL | **6.44 ± 0.55 m** (8 route splits; range 4.96–7.41) | **camera-frame unit** — superseded by §1 |
| **D2** (imagination ranking) | ✅ PASS | dir-acc 0.864, P4 fwd-dyn 0.971, fit-R² 0.98 | the world-model-usable-for-selection claim holds |
| **D3** (imagined vs oracle @2s) | FAIL, K-step-improved | imagined 1.97 m vs oracle 1.52 m, ratio 1.30 | K-step closed the ratio from ~4× |

> **D1/D3 statistical-power footnote (G-B1).** Those gates came from a **single fixed seed=0** split
> at 4–9 val episodes; a measured power audit shows the ADE@1s estimator swings **5–7 m across split
> seeds** on the *same* checkpoint (95 % CI half-width ±4.5 m at n=4). Single-seed D1 values are
> descriptive, not decision-grade. Superseded by §1's 881-window, 40-episode protocol.

### Live scenario metrics — SC-01 Work-Zone Phantom (2026-07-08, scripted policies, single seed)

| Policy (scripted, NOT our checkpoint) | OKRI ↓ | LOPS ↑ | TMS ↑ | CNCE ↑ | LAL-v1 |
|---|---|---|---|---|---|
| reactive (E2E-like) | 32.37 | 0.00 | 0.006 | 8.68e5 | −0.7 |
| world_model (anticipatory) | **12.83** | **0.834** | 0.023 | 1.06e6 | −0.7 |

Weak rows: scripted archetypes, single seed, and **LAL-v1 is non-discriminative here** — superseded
by **LAL-v2** (+0.3…+3.1 s anticipation lead vs −0.3 s reactive). LOPS's 0.0 is structural. Not an
edge claim.

---

## 9. EXTERNAL BENCHMARKS — ⏳ **PENDING, table skeletons reserved for the merge**

⛔ **DO NOT FILL THESE FROM MEMORY OR FROM PROSE.** Two sister agents own the published numbers;
this section is a **merge target with fixed column headers** so their rows drop in mechanically.
The EvalFlyWheel internal-rows agent (this rebuild) deliberately researched **no** external
benchmark. Every external row must arrive with **benchmark · split · metric · value · source
(arXiv/leaderboard + date) · access date**, and must state whether TanitAD can compute the metric at
all.

**Programme goal this section serves: `G3 — Beat published SOTA on community benchmarks (NavSim …)
via TanitEval` — status OPEN** (`Project Steering/GOALS_AND_CLAIMS.md`).

### 9.1 NavSim (PDMS / EPDMS) — ⏳ `PENDING — NavSim research agent`

| System | Benchmark | Split | Metric | Value | Source / date | Note |
|---|---|---|---|---:|---|---|
| *(rows pending)* | NavSim v1 / v2 | navtest / navhard | PDMS / EPDMS | — | — | — |
| **TanitAD** | NAVSIM v2 | — | EPDMS | **— NOT COMPUTABLE TODAY** | — | EPDMS needs agent boxes, drivable-area polygons and a route centerline; PhysicalAI-AV has none (§1a.5). We adopt only its **Extended Comfort** idea and the human-log filter. **A partial PDMS is never published.** |

*Carried forward from the 2026-07-21 page as ORIENTATION ONLY, pending the agent's re-verification —
⚠️ these are INHERITED, not re-checked in this rebuild, and must be replaced or confirmed:*
SOTA claim (survey) 89.3 EPDMS navtest, arXiv 2606.19641 · HAD 88.6 navtest, arXiv 2604.03581 ·
Drive-JEPA 93.3 PDMS **NAVSIM v1** (not comparable to EPDMS), arXiv 2601.22032 · DrivoR 56.3 EPDMS
**navhard**, arXiv 2606.07170 · DriveFuture 55.5 EPDMS **navhard**, arXiv 2605.09701 · PDM-Closed
51.3 EPDMS navhard, arXiv 2506.04218.

### 9.2 nuScenes — ⏳ `PENDING — nuScenes research agent`

| System | Benchmark | Split | Metric | Value | Source / date | Note |
|---|---|---|---|---:|---|---|
| *(rows pending)* | nuScenes | val | L2 @1/2/3 s · collision rate | — | — | ⚠️ the agent must state the **convention** (`pointwise` = UniAD vs `cumulative` = ST-P3/VAD — they differ ~2×) and whether the ego-status-shortcut control was run |
| **TanitAD** | nuScenes | — | L2 / collision | **— NOT RUN** | — | no nuScenes ingest exists in `taniteval/registry.py` (three eval corpora, none nuScenes). A cross-corpus claim needs an ingest first — §12 W-11 |

### 9.3 Closed-loop (Bench2Drive, CARLA) — the arbiter block *(kept, INHERITED)*

| System | Driving Score | Success Rate | Source / date |
|---|---|---|---|
| TF++ (VLAAD-MIL) | 86.97 | 71.97 % | arXiv 2603.25946, 2026 |
| ADT | 77.90 | 55.0 % | Bench2Drive leaderboard, 2026 |
| **TanitAD** | — | — | Phase 1; MetaDrive closed-loop first (G0.5), then CARLA/Bench2Drive |

CARLA seed variance ≈ 5 DS same-model → our closed-loop rows will report mean ± CI over ≥3 seeds; a
"beats baseline" claim requires separated CIs.

### 9.4 Competitor parameter envelope (W-05 / CNCE)

| System | Params | Deployment class | Source |
|---|---|---|---|
| NVIDIA Alpamayo-2 | **32 B** | on-car VLA policy | Opponent profiles, 2026-07 |
| Wayve GAIA-3 | **15 B** | offline generative world model | Opponent profiles, 2026-07 |
| **TanitAD flagship v1** | **263.4 M** | on-car hierarchical latent WM + tactical | MODEL_REGISTRY §1.2 |
| **TanitAD REF-C-base** | **104.2 M** | on-car anchored-diffusion planner, 21.8 ms fp32 | MODEL_REGISTRY §4.3 |

*Not an apples-to-apples score* — a parameter/compute-envelope comparison only. The efficiency wedge
is credible only *at matched safe-progress*, which requires a T1 or T2 number we do not yet have.

---

## 10. Provenance and regeneration

**Every number in §0–§5 traces to** `Project Steering/MODEL_REGISTRY.md` or to a raw eval artifact:
`taniteval/results/driving_<key>.json` (§1, §1b, §2, §3.0, §4, §6), `eff_<key>.json` (§5),
`windows_<key>.pt` (the substrate), and the banked campaign JSONs listed inline in §1a, §2.5, §2.6,
§2.7, §3, §5.1. **Nothing is transcribed from a summary, changelog, weekly report or
`PROJECT_STATE.md`.**

**Regenerate — T0/C1 only:** `python -m taniteval.runner driving-all` then
`python -m taniteval.driving --leaderboard`. CPU-only, offline, ~1 minute.
⛔ **A census over `windows_*.pt` MUST import `taniteval.dump_census` and honour
`taniteval/results/dump_exclusions.json`; a bare glob is a defect** (EVAL_DOCTRINE rule 6, C126) —
27 dumps are **25 distinct arms**.

---

## 11. What changed since the 2026-07-21 rebuild *(so the staleness is auditable)*

**A. Errors found and fixed in the page itself** (each verified against raw JSON, cited in
`LEADERBOARD_RECONCILIATION.md`):

| # | what was wrong | what it is now |
|---|---|---|
| F-1 ⛔ | §1–§4 headed *"Driving capability … the standard read"* with **no tier stamp**; the numbers are **teacher-forced** | restamped **T0**; §0.1 added; T1 promoted to §1a as the primary read |
| F-2 ⛔ | `flagship-v3enc` row read *"running · 🟥 not evaluated"* | **1.9654** [1.6556, 2.2859], from `driving_flagship-v3enc-10k.json` (registry §6 rank 11 already carried it) |
| F-3 ⛔ | `beats CV` printed **✗** for `refb-10k` (+0.0005) and `refb` (−0.0252) | both are **`favours: "tie"`, `separated: false`** in their own JSON → rendered **TIE**, as §0.5 requires |
| F-4 | §0 quoted the interval-narrowing band as **1.28–2.06× over 10 arms** | **1.107–3.100×, median 1.499×, over 27 dumps = 25 arms** (registry §6) |
| F-5 | six scored arms had a `driving_<key>.json` and **no row**: `flagship-v4.1-10k`, `flagship-v4.2-step4000`, `refc-v12`, `refc-v12-k16reg`, `refc-xl-live`, plus the REF-A overfit ladder | all added to §1 / §2 with their verdicts |
| F-6 | header said *"Rewritten 2026-07-21"* while carrying 08-02 / 08-16 / 08-17 patches | rebuilt and dated 2026-08-23; the patch dates are kept inline |
| F-7 | §5's ⚠️-conflict box implied the registry figure might stand | the registry itself marks those six latency figures **UNRESOLVED SOURCE / do not re-cite**; recorded as such |

**B. Whole campaigns that had NO row on this page and now do:**
v5f-w120-30k (§2.5) · v5.8f W1/W2/W4/W4b/W4c/W7 ladder (§2.5) · I4a imagination ablation (§2.5) ·
`flagship-v1arch-v2bal-30k` OOD-val four families (§2.6) · `flagship-v16-unicycle` and
`flagship-v17-speedloss` (§2.7) · the §1.12 decoder-conditioned closed loop (§1a.6) · **the T1
pseudo-closed-loop campaign and its four-family rescore (§1a) — the programme's PRIMARY tier, absent
for 11 days.**

**C. Doctrine that landed after 2026-07-21 and is now enforced here:**
`EVAL_DOCTRINE.md` T0/T1/T2 (2026-08-09) · the four-family binding rule (PI 2026-08-02) ·
`dump_exclusions.json` / `dump_census` (2026-08-18, C126) · the jack-in-gates re-drive (2026-08-16) ·
the goal-input / vision-only inference rules (PI 2026-08-03).

**D. Still NOT on this page, deliberately:** the **v6 / v7-tiny** line. `v6F-SW-30k` appears in
MODEL_REGISTRY **§12 only, as a frozen-trunk readout diagnostic explicitly stamped
"may NEVER be quoted as driving performance"**, and **v7-tiny has no MODEL_REGISTRY row at all** —
its results live in `Project Steering/GOALS_AND_CLAIMS.md` (H-RANK-*, H-INIT-1) and under
`TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/`. Per the
quotable-source rule, **no v7-tiny number is admissible on the leaderboard until it has a registry
row** — §12 W-12.

---

## 12. Gaps and work items — every model with no current number, every missing family

### 12.1 Missing metric families *(a missing family is a WORK ITEM, never a pass)*

| id | gap | where | why it is missing | cost |
|---|---|---|---|---|
| **W-1** | **LONGITUDINAL distance-keeping** (headway / time-gap / TTC) NOT MEASURED on C1, C2, C3 | §1a.2, §1c, §2.5 | no lead-agent track joined to those grids — a **JOIN** gap, not a data gap. `obstacle.offline` covers 97.44 % of the corpus; the instrument exists at **`taniteval/tools/build_lead_block.py`** (⚠️ the registry twice cites the non-existent `tools/build_lead_block.py`) | CPU + a join; it is already **closed on C4** (§2.6) — copy that recipe |
| **W-3** | **STRATEGIC** NOT MEASURED anywhere on PhysicalAI | §1a.5, §1c, §2.5 | corpus fact: no map / lane graph / junction label / route signal, and both available label sources are inadmissible. **Settled at 5 probes** | blocked on the VLM PH0→PH1→PH2 pipeline (PH0 v2 gate PASSED at n = 8) or an external corpus |
| **W-4** | **TACTICAL** NOT MEASURED on C1 and C2 | §1c, §2.5 | the scored pass is teacher-forced (`pc2_pass = False`), so no manoeuvre decision is decoded | closed at source for **future T1 runs** (`t1_eval.py` passes `tactical_from_traj=True`); a C1/C2 backfill needs a hierarchy-traversing rescore |
| **W-5** | three LAT intervals **refused** on C1, and the same two-reducer hazard is **unflagged at T1** | §1c, §1a.3 | `four_families` pools over steps; a per-window form is a mean-of-per-window-means; they differ | implement a per-window reducer inside `four_families`; zero GPU |
| **W-6** | six §2 rows have ADE + verdict but **blank along/cross/speed/heading/κ columns** | §2 | never carried into the 2026-07-21 panel narrative | `driving-all` already emits them — a **zero-GPU** re-render |
| **W-18** | ⛔ **NO `loop` COLUMN (open / closed) on any row** — and the page's T1 headers call an OPEN-loop quantity *closed loop* | §0.8, and 13 enumerated sites | the PI ruling of **2026-09-02** postdates every table on this page: a model consuming its own planner's output is **still OPEN LOOP** | **zero GPU** — retro-fitting the column is a lookup on the tier, not a re-measurement. The **header** corrections are owned by a **separate stream** and are listed with file:line in §0.8 |
| **W-19** | ⭐ ⛔ **NO `ha0` COLUMN anywhere** — the constant-velocity-at-measured-`v0` trivial floor | §0.9 G-2; §1a, §1a.6, §1d.4 | `ha0` postdates those dumps (`D-REFAV1-HA0-ARM`, `t1_eval.py:148–153`). The page's T1 floor is `ha`, which holds the last **observed** `(a, κ)` and drifts ~0.12 m even on straight driving | ⛔ **GPU — a re-run, not a rescore.** MEASURED on the arm fixture: `ha − ha0` **+0.1330 [−0.0502, 0.2326] NOT separated** ⇒ **a win over `ha` alone is not skill**, so §1a's 22–25× headline has an **unknown margin over the true floor** |
| **W-20** | ⛔ **`MODEL_REGISTRY.md` has no row for `refav1`** — 0 hits for `refav1`/`refa_v1`/`REF-A v1` on the 21:5xZ re-probe. ⚠️ **NARROWED mid-write: refcv3's row (§4.5) LANDED from a sibling stream between two probes of this page** — the original claim named both arms and went stale within the hour | §1d.5 | the B1 line was minted after the last registry refresh | mint a §1.x row for refav1. Until then §1d.4's numbers are admissible **only with their raw-JSON path attached**. Same class as **W-12** (v7-tiny) |
| **W-21** | **refcv3's in-training eval can never carry an interval** — `ci_available: false`; every value is already the pooled mean over 160 windows, with no per-window values and no `eid` | §1d.3 | `taniteval/ci.py` requires per-window values + episode ids | **forward-looking only** — the in-training eval must **dump per-window values with episode ids**. No rescore can recover it for the run already banked |
| **W-22** | **refcv3 distance-keeping is PARTIAL BY CONSTRUCTION** — `join_lead_block` maps `t → 2t`, so a `--grid 2s` run joins only at **{1.0, 2.0} s**, and an odd-raw-frame origin joins to `frame − 1` | §1d.2 | the banked B1 lead block is a 0.2 s / K = 10 grid; refcv3's grid is not a subset of it | rebuild the B1 lead block on refcv3's own grid (`build_lead_block_b1.py --dt 0.5 --k 4`), and add a **frame-identity** mode to `join_lead_block` |

### 12.2 Models in MODEL_REGISTRY with no current leaderboard number

| registry § | key | status | what is missing |
|---|---|---|---|
| §1.7 | `flagship-v2corpus-30k` | ⚠️ **status UNVERIFIED** — registry reads 🟢 RUNNING with an ETA **26 days in the past**, no completion / final-step / final-eval row anywhere, and its host pod1 shows `/dev/nvidia*` empty | **re-probe the run before anything quotes or waits on it.** No eval JSON exists |
| §1.5 | `flagship-v4-fromscratch`, v4.1/v4.2 milestone ladder | v4.1-10k and v4.2-step4000 now ranked (§1); the rest single-disk, not HF-backed | milestone evals; checkpoint banking |
| §2.2 | `refa-ijepa-4brain-speed-15k` | ⛔ **val is ~80 % leaked into train — the number is UNUSABLE** | a clean split, or permanent exclusion |
| §3.4 | `refb-refbpatch-30k` | crashed | nothing to score |
| §5 | `planner_p2` (open-loop CEM arm) | **no per-window dump** for `plan_wp` | ~400 s of GPU closes the last arm and settles the standing `planner_beats_cv` UNDECIDED |
| §10 | `dynenc-branchB` | side IDM model, 🟥 FAIL on held-out-rig transfer | not a driving arm — no leaderboard row by design |
| §12 | `v6F-SW-30k` | T0-diagnostic readout line only | ⛔ **may never be quoted as driving.** A driving number needs a T1 eval |
| *(absent)* | **v7-tiny line** | ⛔ **NO REGISTRY ROW AT ALL** | **W-12: mint a MODEL_REGISTRY §1.x row** before any v7-tiny number is admissible here (see §11-D) |

### 12.3 Instrument and reconciliation work items

| id | item |
|---|---|
| **W-2** | **CTRV is still not in the gate.** `driving.py:304` scores against `FLOORS = ("cv","holdv0")`, both straight lines. The patch + 11 tests are validated end-to-end and **UNMERGED** at `…/incoming/2026-08-02-ctrv-floor/`. ⚠️ Until merged, every new arm is auto-scored against a floor that is ~5× too generous laterally |
| **W-7** | **Two harnesses disagree 0.8 %** on C4 for `v1arch` (`eval_flagship_v4.py` MODE-A 0.5705 vs `eval_four_families.py` 0.5752). Recorded, unresolved |
| **W-8** | **Registry R14 latency conflict** — registry §6 reading 3's six figures are marked UNRESOLVED SOURCE and are not in any committed artifact. One reconciliation pass on an idle A40, or restate from the JSONs |
| **W-9** | **The two T1 blocks disagree in character** — §1a.1 (C3) diverges 22–25× against `ha`; §1a.6 (C4) degrades ~1.4×. Different corpora, arms and action interfaces; the reconciliation is unwritten |
| **W-10** | **Junction departure-rate definition mismatch** — 0.368 (window-level, research doc) vs 0.064 (episode-level, §5.5). Reconcile BEFORE either is quoted |
| **W-11** | **No nuScenes ingest exists** (`taniteval/registry.py` lists three eval corpora, none nuScenes). G3 needs one before any cross-benchmark claim |
| **W-13** | **Tier-1 metric surface is blocked on one line** — `rollout.collect` computes the dense 20-step path and discards 16 of 20 steps at `rollout.py:94`. Persisting it unlocks jerk, the adopted nuPlan/NAVSIM comfort bounds, the curvature *profile*, decel-onset lead time (implemented and unmerged since 2026-07-09) and plan-stability / Extended Comfort. **~1 MB per arm** |
| **W-14** | **Regime thresholds are PROPOSED** (±0.5 m/s², \|κ\| < 1e-3, the 5°/15° split). Effects are large and unlikely to be threshold-driven, but a sensitivity sweep is owed |
| **W-15** | **`driving.py` buckets curvature at 5°/15°; `driving_diagnostic.curvature_bucket` at 5°/20°.** Both are recorded in every block; the panels are **not interchangeable** until reconciled |
| **W-16** | **`flagship-v16-ab-ft`, `refb-v2-*`, `flagship-v2-6k`, the v4 line and the REF-C v1.2 family have no `eff_<key>.json`** → no latency column. Nothing blocks it but a run on an idle GPU |
| **W-17** | The C1 4-waypoint speed is a **0.5 s box-average**, not an instantaneous speed, so §3.0's cruise numbers are **conservative** — the dense path makes them stricter, not looser |
