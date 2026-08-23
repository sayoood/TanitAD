# JACK IN GATES — the banned estimator was deciding `G1_pass` / `G4_pass`

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start:** `4a0f57e`
**Compute:** CPU-only, no GPU touched, no model loaded, no pod contacted (Thor is training).

---

## 0. THE ANSWER — does either verdict FLIP?

> ## **NO. Neither `G1_pass` nor `G4_pass` flips.**
> **G4 is fully re-decided on banked per-window data under both estimators. G1 is
> re-decided on 3 of its 4 arms; the 4th (the open-loop CEM planner) was never banked
> per-window, and the flip requirement for it is quantified below and is unreachable.**

| gate | BANNED `overlapping_holdout_se` | decision-grade episode-cluster bootstrap | flips? |
|---|---|---|---|
| **G1** head − planner ADE@2s | **+2.2572 ± 0.3292**, separated → **PASS** | **PASS** (3/4 arms exact; see §3) | ❌ **no** |
| **G4** planner closed-loop vs head | **1.0375 < 1.6852** → **PASS** | **0.9799 [0.7456, 1.2312] < 1.7318** → **PASS**, and the whole CI clears the bar | ❌ **no** |
| **G4 — PAIRED** *(new, first ever)* | −0.6873 ± 0.2191, separated → PASS | **−0.7375 [−0.9362, −0.5295]**, separated, p(δ>0) = 0.0000 → **PASS** | ❌ **no** |

**A null, but not a cheap one.** The point estimates move **−6.9 % to +6.8 %** (one row **+20.3 %**),
the intervals were **1.17×–2.17× too narrow**, the G4 *threshold itself* was **2.69 % low**, and the
banned estimator was **giving 7 of the 40 val episodes exactly zero weight**. The verdicts survive
because the effects are large (G1: 2.26 m on a 0.33 m interval; G4: 0.74 m separation), not because
the estimator was harmless.

**Everything deciding is now migrated. Repo-wide AST scan after the change: 0 deciding sites.**

---

## 1. THE FACTS, VERIFIED AT `file:line`

Pre-migration `taniteval/taniteval/planner_p2.py` (the state at `4a0f57e`):

| line | code | class |
|---|---|---|
| `:389` | `def _jack_scalar(vals, eids, splits)` | definition of the banned estimator |
| `:397` | `def _jack_paired(a, b, eids, splits)` | ditto, **paired form** |
| `:415` | `heldout = {k: _jack_scalar(ade[k], eids, splits) for k in ade}` | open-loop arms |
| **`:458`** | **`g1_delta = _jack_paired(ade["head"], ade["plan"], eids, splits)`** | ⛔ **DECIDES** |
| `:468` | `"G1_pass": bool(g1_delta["mean"] > 0 and g1_delta["separated"])` | ⛔ **the verdict** |
| `:586` | `heldout = {k: _jack_scalar(ade[k], eids, splits) for k in ade}` | closed-loop arms |
| `:594` | `"G4_head_baseline_ade2s": 1.6852` | ⛔ threshold, itself a legacy `heldout` mean |
| `:595` | `"G4_pass": bool(heldout["closed_bike"]["mean"] < 1.6852)` | ⛔ **the verdict** |

Confirmed at line level:

* `_jack_paired` **is** `overlapping_holdout_se` arithmetic — `1.96 * np.std(sm) / sqrt(len(sm))`
  over 8 **overlapping** random 20 % episode holdouts (`taniteval/taniteval/ci.py:121-139`
  documents the identical arithmetic under its honest name; `ci.py:5-27` documents why it is
  neither a jackknife nor a valid SE).
* Its central value is a **mean-of-split-means**, not the `full_set` mean — so it biases the
  **point estimate** before any interval is drawn.
* `g1_delta` **is exactly a paired delta**, the statistic on which the 2026-07-25 blast radius
  measured errors up to **×−4.15 including a sign flip**.
* The file's own docstring at `:399-401` already said *"DEPRECATED estimator (not a jackknife).
  Prefer `ci.paired_episode_cluster_bootstrap`"*. **The correct instruction was written into the
  artifact it applied to, and nothing read it for 21 days.**

---

## 2. HOW THIS WAS RE-DECIDED WITHOUT A GPU

`Project Steering/MODEL_REGISTRY.md:2599` says of P2: *"🟥 **NOT RECOMPUTABLE** — no raw JSON and
no `windows_*.pt` in the repo"*.

> ⚠️ **That absence claim is REFUTED.** Both exist, at depth 6–8 in the tree.
> (This is CLAUDE.md operating-rule 2 — *absence found at ONE location is not absence* — and it is
> the reason this gate sat un-re-decided for three weeks: it was believed to need GPU.)

| artifact | repo path | what it is |
|---|---|---|
| published P2 JSON | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-closedloop-artifact-rerun/_pod_pulled/planner_p2_flagship-30k.json` | the artifact behind the registry's P2 row |
| planner closed-loop windows | `…/2026-07-26-closedloop-artifact-rerun/raw_windows/p2win_flagship-30k.pt` | 221 win / 20 ep, stride 16 |
| head-baseline closed-loop windows | `…/2026-07-26-closedloop-artifact-rerun/raw_windows/clwin_flagship-30k.pt` | 881 win / 40 ep, stride 8 — **and its `plan_direct` is the G1 tactical-head arm** |
| open-loop operative + CV windows | `taniteval/results/windows_flagship-30k.pt` | 881 win / 40 ep — **the identical window set** (`gt` and `eid` bit-equal to `clwin`) |

### The reproduction gate — run FIRST, because otherwise the comparison proves nothing

Recomputing the **banned** estimator from these banked windows reproduces the published numbers:

| arm | published (2026-07-19) | recomputed from banked windows | |
|---|---|---|---|
| G1 `tactical_head` | 3.1501 ± 0.3472 | **3.1501 ± 0.3472** | ✅ bit-exact, 4 dp, mean **and** ci95 |
| G1 `operative_rollout_trueA` | 0.4522 ± 0.0312 | **0.4522 ± 0.0312** | ✅ bit-exact |
| G1 `constant_velocity` | 0.8248 ± 0.1035 | **0.8248 ± 0.1035** | ✅ bit-exact |
| G4 `open_grnd` / `cv` / `divergence` | 0.4244 / 0.7704 / 0.0871 | identical | ✅ bit-exact |
| G4 `closed_bike_ade2s` | 1.0377 ± 0.2022 | 1.0375 ± 0.2023 | ✅ to 2e-4 — residual is the **unseeded CEM** (`planner_p2.py` draws `torch.randn` with no seed; drift measured at **0.019 %** in `planner_p2_G4.CORRECTED.json`) |

The CV baseline is a **model-free** quantity, so its exact reproduction (0.8248 ± 0.1035) is a proof
that the banked window set and the 8-split structure are the same objects the 2026-07-19 gate ran on.

---

## 3. G1 — open loop, 881 windows / 40 episodes

| arm | **BANNED** mean ± ci95 | **CORRECTED** mean [lo, hi] ± ci95 | point-estimate error | CI widening |
|---|---|---|---|---|
| `tactical_head` | 3.1501 ± 0.3472 | **3.3839** [2.8336, 3.9722] ± 0.5693 | **−6.909 %** | **1.640×** |
| `operative_rollout_trueA` | 0.4522 ± 0.0312 | **0.4271** [0.3675, 0.4871] ± 0.0598 | **+5.877 %** | **1.917×** |
| `constant_velocity` | 0.8248 ± 0.1035 | **0.8377** [0.6234, 1.0716] ± 0.2241 | **−1.540 %** | **2.165×** |
| `planner` (CEM, open loop) | 0.8929 ± 0.1143 | ⚠️ **NOT BANKED per-window** | — | — |
| **G1 delta** head − plan | **+2.2572 ± 0.3292**, separated | ⚠️ partial (below) | | |

Note the bias is **bidirectional within a single artifact** — head −6.9 %, operative +5.9 % — which
is exactly why "how big is the correction" has no single answer and must be measured per arm.

### 3.1 Why G1's recomputation is PARTIAL, and why it is still decisive

`collect_openloop`'s `plan_wp` (`planner_p2.py:395`) — the open-loop CEM search — was never dumped
per-window. Probed at three locations before the claim was made: `taniteval/results/*.pt` (`pred`,
`cv`, `gt` only), `…/2026-07-26-closedloop-artifact-rerun/raw_windows/*.pt` (closed-loop only), and
`…/2026-07-23-frozen-wm-learned-planner/artifacts/perwin*.pt` (a different protocol, 265/881 windows,
arms `oracle`/`cv`/`holdv0`/`search_cold`/`search_warm`). Closing it needs **~400 s of GPU**.

**What can be said without it, rigorously:**

* The corrected `tactical_head` full-set mean is **3.3839**.
* For G1 to flip, the planner's corrected full-set mean would have to reach **≥ 3.3839**, i.e. the
  banned estimator would have to have been wrong on that one arm by **−73.6 %**.
* The MEASURED envelope of the banned estimator's point-estimate error **on this exact window set,
  under this exact 8-split structure** is **−6.9 % to +5.9 %** (three arms). The programme-wide
  27-arm envelope is **−6.67 % to +11.69 %**.
* ⇒ **A flip needs an error ~11× larger than anything ever measured for this estimator, on the one
  arm that was not banked.** G1 does not flip.

**A same-shape paired control, with BOTH arms banked** — `head − cv` on the identical 881 windows,
which is the closest measurable analogue of what `_jack_paired` did to G1:

| | delta | interval | separated |
|---|---|---|---|
| BANNED | +2.3253 | ± 0.3067 | ✅ |
| CORRECTED (paired episode-cluster bootstrap) | **+2.5462** | [2.0115, 3.0901] ± 0.5393 | ✅ |
| | **−8.676 %** point error | **1.758×** widening | **verdict unchanged** |

---

## 4. G4 — closed loop, 221 windows / 20 episodes — FULLY re-decided

| row | **BANNED** | **CORRECTED** [lo, hi] | point error | CI widening |
|---|---|---|---|---|
| `closed_bike_ade2s` | 1.0375 ± 0.2023 | **0.9799** [0.7456, 1.2312] | **+5.878 %** | 1.200× |
| `closed_bike_fde2s` | 2.1940 ± 0.4552 | **2.0583** [1.5463, 2.6134] | **+6.593 %** | 1.172× |
| `open_grnd_ade2s` | 0.4244 ± 0.0573 | **0.4063** [0.3293, 0.4907] | **+4.455 %** | 1.408× |
| `cv_ade2s` | 0.7704 ± 0.1704 | **0.7214** [0.4680, 1.0360] | **+6.792 %** | 1.667× |
| `divergence_rate_gt5m` | 0.0871 ± 0.0460 | **0.0724** [0.0225, 0.1409] | **+20.304 %** | 1.287× |

⚠️ **The divergence rate — the safety-shaped number — was overstated by 20.3 %.** It reports
**8.7 %** under the banned estimator and **7.2 %** correctly. Rates are the metric class where a
mean-of-split-means bites hardest, because a split's mean of a 0/1 indicator is dominated by which
episodes were drawn.

### 4.1 The threshold was itself the banned estimator

`G4_pass` compared against **1.6852**, sourced from `closedloop_flagship-30k.json` — a legacy
`heldout` mean. The `full_set` value of the *same banked windows* is **1.7318** (reproduced here
bit-exactly from `clwin_flagship-30k.pt`, and matching
`closedloop_flagship-30k.CORRECTED.json`). **The bar was 2.69 % lower than it should have been** —
i.e. the pre-migration gate was *harder* than the honest one, so the correction only strengthens the
PASS.

| | planner | threshold | verdict |
|---|---|---|---|
| BANNED (both sides) | 1.0375 | 1.6852 | **PASS** |
| CORRECTED (both sides) | **0.9799** [0.7456, **1.2312**] | **1.7318** | **PASS**, and `hi` **1.2312 < 1.7318** ⇒ CI-separated |

### 4.2 NEW — the first PAIRED G4 test

The published G4 was **unpaired**: planner 221 win / 20 ep / stride 16 vs head baseline 881 win /
40 ep / stride 8. Its "CI-separated" claim rested on two independent intervals, both banned.

The planner's windows are the **stride-16 subset** of the baseline's stride-8 windows on the same 20
episodes: within episode *e*, planner window *j* is baseline window *2j*. Verified by **GT waypoint
equality to `atol=1e-5`, window-for-window, for all 221** before use. That makes a paired test
possible for the first time:

| | delta (planner − head) | interval | separated |
|---|---|---|---|
| BANNED paired | −0.6873 | ± 0.2191 | ✅ |
| **CORRECTED paired** | **−0.7375** | **[−0.9362, −0.5295]** ± 0.2034, p(δ>0) = **0.0000** | ✅ |

planner **0.9799** vs head **1.7174** on the *same* windows ⇒ **42.9 % less closed-loop drift**,
paired, decision-grade, `n = 221 / 20 episodes`. **The strongest form of the G4 claim the programme
has ever had — and it agrees with the verdict it replaces.**

---

## 5. A NEW DEFECT, MEASURED HERE: the banned estimator ZEROES OUT PART OF THE VAL SET

`heldout = (1/S) Σ_s mean_{i ∈ V_s} v_i` is a weighted mean `Σ_i w_i v_i` with
`w_i = (1/S) Σ_s 1[i ∈ V_s] / |V_s|`. Drawing 8 holdouts of 8 episodes from 40 leaves episodes that
appear in **none** of them — and those get weight **exactly 0**.

| window set | episodes | **zero-weight episodes** | max episode weight | uniform |
|---|---|---|---|---|
| G1 open loop | 40 | **7 (17.5 %)** — ids 1, 9, 22, 23, 27, 28, 34 | 0.064972 (**2.60× uniform**) | 0.025 |
| G4 closed loop | 20 | **2 (10.0 %)** — ids 16, 18 | 0.100000 (**2.00× uniform**) | 0.050 |

⇒ **The G1 gate was decided on a statistic that never looked at 7 of the 40 validation episodes, and
over-weighted its most-drawn episode by 2.6×.** This is a *sharper* statement than "the interval is
too narrow": it is not a variance problem, it is the wrong population. It is also why no finite
bound on the planner arm can be derived from the published `heldout` alone — the unweighted windows
are unconstrained by it — and hence why the flip argument in §3.1 is stated as a measured envelope
rather than as an algebraic bound.

---

## 6. WHAT CHANGED IN THE CODE

### 6.1 `taniteval/taniteval/planner_p2.py` — the deciding sites, migrated

Both gates now read the decision-grade estimator; **the legacy values are kept beside them**, never
deleted and never silently re-published.

| new line | change |
|---|---|
| `:81-94` | estimator policy **imported** from `driving.py` (`N_BOOT`, `DECISION_ESTIMATORS`, `DEPRECATED_ESTIMATOR`, `ESTIMATOR_NOTE`, `LEGACY_BLOCK`) — one policy in one place, so a third copy cannot drift from the first two |
| `:116-130` | `G4_HEAD_BASELINE_ADE2S = 1.7318` (+ `…_LEGACY = 1.6852` + its provenance string) |
| `:446-484` | `_interval` / `_paired` / `_width_ratio` / `_point_shift_pct` |
| `:487-516` | `_jack_scalar` / `_jack_paired` **kept**, re-documented as REPRODUCTION ONLY, now self-labelling (`estimator`, `deprecated`) and routed through `ci.overlapping_holdout_se` so the arithmetic has exactly one home |
| `:519-522` | `assert_no_deprecated_estimator` — refuses to return a block whose deciding numbers are deprecated; delegates to `driving`'s single implementation |
| `:536`, `:584`, `:595` | **G1**: arms from `episode_cluster_bootstrap`; delta from `paired_episode_cluster_bootstrap`; `G1_pass` reads `delta`/`separated` |
| `:761`, `:774-775` | **G4**: arms from `episode_cluster_bootstrap`; `G4_pass` vs the corrected threshold, **plus a stricter `G4_pass_ci_separated`** (`hi < baseline`) |
| `:625-…`, `:793-…` | `legacy_overlapping_holdout_se` quarantine block, carrying the old numbers, the old verdict as `G1_pass_LEGACY` / `G4_pass_LEGACY`, an explicit `…_verdict_flip_vs_decision_grade_LEGACY`, and — **re-measured per run, not cited from a doc** — `ci_width_ratio_new_over_legacy` and `point_estimate_shift_pct_legacy_vs_corrected` |

Verified end-to-end on the real banked windows: `analyze_closedloop(p2win_flagship-30k.pt)` returns
`G4_pass True`, `G4_pass_ci_separated True`, planner `0.9799 [0.7456, 1.2312]`, threshold `1.7318`,
and `G4_verdict_flip_vs_decision_grade_LEGACY False`.

### 6.2 `taniteval/taniteval/gate_guard.py` + `taniteval/tests/test_no_jack_in_gates.py` — the AST guard

⚠️ **AST walk, not a regex — and the reason is MEASURED in §7.**

* **Taint sources**: `_jack_*` (regex over the family, so a *new* sibling is caught the day it is
  written), `overlapping_holdout_se`, and **import aliases** — `from … import _jack_paired as agg`
  is resolved, because a rename is not a fix.
* **Taint propagation to a fixpoint**: laundering through any number of intermediate variables,
  dict comprehensions, subscripts and tuple targets still reaches the verdict.
* **Deciding expressions**: dict keys, assignments *and* subscript stores whose name carries a
  verdict (`*_pass`, `verdict`, `gate_ok`, `admissible`, `G<n>_pass`).
* **Exactly one exemption**, and it is explicit and greppable: a name marked `_LEGACY` / `legacy_`.
  Nothing else — not a comment, not a docstring, not a `# noqa`.

16 tests. Six **negative controls** keep it from being vacuous — it must fire on the pre-migration
G1 shape, the pre-migration G4 shape, the inlined `bool(_jack_paired(...)["mean"] >= 0.2)`, a
3-variable laundering chain, an attribute call `ci.overlapping_holdout_se(...)`, a not-yet-invented
`_jack_*` sibling, an import alias, and a subscript store. Three **false-positive controls** pin
that it does *not* fire on a docstring/comment naming the family, or on the migrated shape.

### 6.3 REPORTS-only sites — marked, not rewritten

12 files call the banned estimator without deciding anything. Nine already emit the decision-grade
estimator beside it (`taniteval/taniteval/{bench,closedloop,hierarchy,planner_p2}.py`,
`taniteval/recompute_ci.py`, the three `taniteval/tests/*`, and two re-drive scripts). **One had no
decision-grade estimator anywhere** and is now marked in place:

* `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/closedloop.py`
  — the frozen 2026-07-19 predecessor of `taniteval/closedloop.py`. Given a header banner:
  *"SUPERSEDED ARTIFACT — EVERY NUMBER THIS FILE EMITS IS `overlapping_holdout_se`… do NOT promote
  any number from here into a decision"*, pointing at
  `closedloop_flagship-30k.CORRECTED.json` as the quotable replacement, and naming the
  1.6852 → 1.7318 gap because **1.6852 is exactly the number that became the G4 threshold**.

---

## 7. THE FULL SITE INVENTORY (unbounded — C69)

`rglob` from the repo root, **no depth limit**; `4 696` files scanned, `996` with a textual hit,
**max path depth reached: 8** (the C69 failure was a `find -maxdepth 4` on files at depth 6).
Raw JSON: `raw/jack_site_inventory.json`. Regenerate: `code/inventory_jack_sites.py`.

| class | n (`.py`) | meaning |
|---|---|---|
| **DECIDES** | **0** (was **1**, `planner_p2.py`, 2 violations) | a verdict is data-dependent on the banned estimator |
| **REPORTS** | 13 | the banned estimator is called; nothing decides on it |
| **DEFINES** | 1 (`taniteval/taniteval/ci.py`) | the canonical home of the deprecated arithmetic |
| **DEAD** | 228 | no banned call at all — the name appears only in text |
| *prose / artifacts* | 754 | `.md` / `.json` / logs |

Totals: **37 banned call sites**, **8 banned definition sites**, **0 deciding violations**.

### ⭐ The measurement that justifies the AST design

The 228 DEAD Python files break down as:

| reason | n |
|---|---|
| **the estimator is named in prose ONLY — usually to declare the file does NOT use it** | **176** |
| `heldout` used as an ordinary English word (e.g. `stack/tanitad/train/heldout_gate.py`, a *training* held-out gate) | 49 |
| the word `jackknife` only | 17 |

> ⛔ **A regex guard would have raised ≥ 176 false positives, every one of them on documentation
> that says the estimator is banned.** Sampled verbatim: `h2c_stats.py:1` *"episode-cluster only,
> never overlapping_holdout_se"*; `s3_labels.py:34` *"`overlapping_holdout_se` is never used here"*;
> `e1a_horizon.py:395` *"NEVER overlapping_holdout_se"*.
> **This is the `pgrep -f` self-match trap, and the polling-monitor self-match trap, in a third
> costume — quantified: 176 : 0.** An AST walk cannot see a comment or a docstring, so it is immune
> by construction rather than by a cleverer pattern.

---

## 8. ⚠️ ESCALATIONS — integration the orchestrator must sequence, not me

**None of these was applied unilaterally: the brief says escalate rather than re-publish.**

1. ⛔ **`MODEL_REGISTRY.md:2599` carries a REFUTED absence claim.**
   It reads *"🟥 **NOT RECOMPUTABLE** — no raw JSON and no `windows_*.pt` in the repo"* for P2.
   Both are in the repo (§2). **Consequence: the belief that re-deciding P2 needed GPU is what kept
   this gate on the banned estimator for 21 days.** Same class at `:2605` for the tactical head
   (*"no windows dump — legacy only"*): `clwin_flagship-30k.pt`'s `plan_direct` **is** that arm and
   reproduces 3.1501 ± 0.3472 bit-exactly, `full_set` **3.3839**.
2. **`MODEL_REGISTRY.md:2482` and `:2716` (D-033) quote the banned numbers without the stamp.**
   Suggested replacements, all MEASURED here, all with artifact paths in §9:
   * G1 open-loop: planner *0.893 ± 0.114* → **planner arm pending a GPU re-drive**; head
     **3.3839 [2.8336, 3.9722]** (was 3.150 ± 0.347); delta *+2.257 ± 0.329* → **stands, sign and
     separation confirmed, magnitude pending the planner arm**.
   * G4 closed-loop: *1.038 ± 0.202 vs 1.685 ± 0.098* → **0.9799 [0.7456, 1.2312] vs 1.7318**, and
     add the **paired** form **−0.7375 [−0.9362, −0.5295]**, `n=221/20 ep`, which is strictly
     stronger than the two-interval claim it replaces.
   * divergence *8.7 %* → **7.2 % [2.25 %, 14.09 %]**.
   * `:283` already flags *"estimator NOT STATED"* on `1.685 ± 0.098`: it is
     `overlapping_holdout_se`, and the corrected value is **1.7318**.
3. **The one open measurement: G1's planner arm.** ~400 s GPU on the 40 val episodes, re-running
   `collect_openloop` **and dumping `plan_wp`/`head_wp` per-window this time** (the omission that
   made this recomputation partial). Blocked only by Thor's 336 M training run (~7.4 d).
   `code/recompute_g1_g4.py` will close it with no changes once the dump exists.
4. **`planner_p2.py`'s CEM is UNSEEDED** (`torch.randn`, no seed) — a re-drive is not
   bit-reproducible. Measured drift is small (0.019 %) but it means every P2 number carries an
   unquantified sampling component. Cheap fix, not made here because it changes the physics path and
   the brief scoped me to the estimator.
5. **`RETRACTION_LOG.md` entry owed** (not written — another agent holds that file):
   *root-cause class: **a deciding path left on a banned estimator because the correct instruction
   lived in the artifact's own docstring, where nothing had to read it**; plus a **stale absence
   claim** (`NOT RECOMPUTABLE`) that made the fix look GPU-gated when it was CPU-only.* The durable
   fix is mechanical and is now in place: `taniteval/tests/test_no_jack_in_gates.py`.

---

## 9. DELIVERABLES

| artifact | repo path |
|---|---|
| this writeup | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-16-jack-in-gates/JACK_IN_GATES.md` |
| dual-estimator re-decision (raw) | `…/2026-08-16-jack-in-gates/raw/g1_g4_both_estimators.json` |
| unbounded site inventory (raw) | `…/2026-08-16-jack-in-gates/raw/jack_site_inventory.json` |
| re-decision script (CPU) | `…/2026-08-16-jack-in-gates/code/recompute_g1_g4.py` |
| inventory script | `…/2026-08-16-jack-in-gates/code/inventory_jack_sites.py` |
| **migrated deciding sites** | `taniteval/taniteval/planner_p2.py` |
| **AST gate guard** | `taniteval/taniteval/gate_guard.py` |
| **guard test (16 tests, 6 negative + 3 false-positive controls)** | `taniteval/tests/test_no_jack_in_gates.py` |
| REPORTS-only marking | `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-19-alpasim-closedloop-v1/closedloop.py` |

**Suites (MEASURED, this task, CPU):**

| suite | baseline at `4a0f57e` | after this work |
|---|---|---|
| `taniteval` | 1042 passed | **1058 passed / 0 failed** (= 1042 + the 16 new guard tests) |
| `stack` | 2996 passed / 0 failed / 17 skipped / 2 xfailed | **3036 passed / 0 failed / 17 skipped / 2 xfailed** |

The `stack` surplus is concurrent agents' new tests (`test_bev_consumer_fov.py` et al.), not mine —
this task touched no file under `stack/`. **No test was loosened, skipped or deleted.**

**Evidence class:** every number in §§0–7 is **MEASURED** (this task, CPU, from the banked artifacts
named in §2 and §9). The 27-arm blast-radius envelope and the ×−4.15 sign-flip figure are
**INHERITED** from `CLAUDE.md` / `…/incoming/2026-07-25-jack-blast-radius/JACK_BLAST_RADIUS.md` and
are used only as context, never as a decision input.
