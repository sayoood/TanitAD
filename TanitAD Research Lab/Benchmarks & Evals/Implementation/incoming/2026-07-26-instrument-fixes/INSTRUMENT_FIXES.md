# Three instrument defects, fixed — gate renderer · OOD guard · closed-loop harness

**Date:** 2026-07-26 (Europe/Berlin; pods and logs are UTC) · **Surface:** CPU, dev box, `OMP_NUM_THREADS=8`
**Suites:** `stack` **1053 passed / 3 skipped** (baseline 1020/3 + 33 new) · `taniteval` **401 passed** (baseline 360 + 41 new)
**Nothing was `git add`ed.**

Evidence class on every claim: `MEASURED (ours + artifact)` · `PUBLISHED` · `INHERITED` · `ESTIMATED` · `HYPOTHESIS`.

All three defects were found by the flagship-v4 30 k gate this morning
(`…/incoming/2026-07-26-v4-30k-gate/GATE_30K_RESULTS.md` §3, §6.4). Each blocked or corrupted
future work: (1) no gate of that card's shape could be rendered at all, (2) every long-horizon OOD
number in the program is a lower bound presented as a measurement, (3) the registered co-primary
lived only in `incoming/`.

---

## 0. HEADLINE

| | before | after |
|---|---|---|
| **D1** `run_gate.py check` on the registered 30 k card | `TypeError` on load; then `NOT_YET` at 29999 | **reproduces the hand adjudication exactly** — RESTART / NOT-CONTINUE, budget 0/2 |
| **D2** OOD verdict at K=185 | *"within the measured envelope on average"* (ratio 1.2741) | **EXTRAPOLATION — NOT a measurement**; 54.63 % of steps, 90.24 % of windows outside |
| **D3** v4 co-primary at K=185 | reachable only via `incoming/v4_corridor_cl.py` | `taniteval.clhorizon`, **bit-identical** to that driver; per-window paths persisted by default |

**Sweep result (D2):** **14 committed artifacts** carry a defective OOD reading — **3** with a
factually **false verdict string** (14 nodes, `MEASUREMENT` → `EXTRAPOLATION`), **2** more with an
undeclared saturation, **6** quoting a saturated ratio with no verdict at all, **3** with no envelope
evidence whatsoever. Plus **5 governance-layer documents** and **2 live guard constants** asserting
"genuine in-distribution failure, not extrapolation" from the ratio alone. Details in §2.4.

---

## 1. DEFECT 1 — `run_gate.py` could not render a modern gate card

### 1.1 What was broken

| # | file:line (pre-fix) | what | MEASURED symptom |
|---|---|---|---|
| D1a | `stack/scripts/run_gate.py:889` — `card = GateCard(**json.loads(Path(a.card).read_text()))` | `GateCard` is a plain dataclass; **11 registered keys have no slot** | `TypeError: GateCard.__init__() got an unexpected keyword argument 'registered_before_checkpoint_exists'` |
| D1b | same line + `run_gate.py:757-768` (flat `co_primary_*` fields) | the card supplies `co_primary` as a **nested dict** | `has_co_primary` reads `False` ⇒ the **DEMOTED** `ade_0_2s` illegally re-enters the kill conjunction |
| D1c | `run_gate.py:1075` `kill_inputs = [co_ok] if card.has_co_primary else [bool(passed)]` | no state for `role: REPORT_ONLY_THIS_GATE`; **no reference to `secondary_void` anywhere in the file** | mapped ⇒ an unmeasured, deliberately unthresholded co-primary forces `INCOMPLETE` |
| D1d | ⚠️ `run_gate.py:964` `if cur < card.gate_step:` | a **0-indexed** trainer counter compared to a **1-indexed** count | `VERDICT: NOT_YET — step 29999 < pre-registered gate step 30000` on a **complete** 59-hour run. **This alone refuses every completed gate.** |

The 11 keys: `co_primary`, `goal_provenance`, `goal_provenance_note`, `preflight_checks`,
`primary_role_note`, `reference_ade_0_2s`, `reference_note`,
`registered_before_checkpoint_exists`, `registration_note`, `required_reporting`, `secondary_void`.

### 1.2 The fix

`stack/scripts/run_gate.py`:

* **`GateCard.from_dict()`** replaces `GateCard(**…)` everywhere. Unknown keys are **preserved** in
  `card_extras` and printed (a card is a pre-registration document; dropping its text is how a card
  stops binding the tool that renders it). The nested `co_primary` dict is mapped onto the flat
  fields through an explicit keymap — **nothing is invented**: the 30 k card registers no
  `threshold`, so `co_primary_threshold` stays `None`.
* **`co_primary_role`** ∈ {`kill`, `REPORT_ONLY_THIS_GATE`}. A report-only co-primary is measured,
  printed in full (overall + interval + n + junction stratum + OOD), and **excluded from the kill
  conjunction**. An **unknown role raises** rather than defaulting to `kill` — silently treating an
  unimplemented role as `kill` is precisely how the 30 k gate mis-adjudicated.
* **`secondary_void`** is a first-class list. Every entry is **excluded from the conjunction AND
  printed** with its `status` / `adjudication` / `authority` / `reason` / `re_arms_when` and an
  explicit `IN KILL SET : NO`. A metric listed in **both** `secondary` and `secondary_void` is
  **void** — `flagship-v4.card.json` is exactly that shape, and 0.7 is an adjudication, not a
  suggestion.
* **`step_reached(cur, gate_step)`** accepts either convention and **names the one it used** in the
  verdict (`out["step_indexing"]`). `0-indexed` is claimed **only** on the exact boundary
  `cur == gate_step - 1`; `gate_step - 2` is still refused, so the fix is not a licence to gate an
  unfinished run.
* Consequential, and it is what the hand adjudication had to reason out by hand: a conjunction
  containing a **hard FAIL is unsatisfiable**, so an unmeasured secondary can no longer downgrade a
  determined NOT-CONTINUE to `INCOMPLETE`. `INCOMPLETE` is now reserved for *"nothing measured has
  failed and something is missing."*
* `_corridor_strata_node` additionally accepts the **horizon-sweep** artifact shape
  (`all_windows[<K>]`) that the co-primary emitter actually produces — but **only at the card's
  registered K**, and it names the horizons present when the registered one is absent.
* Also fixed a **kill co-primary with no bar**: refused loudly (0.3), never silently passed.

⚠️ **`heldout` was NOT renamed.** `_deprecated_present` still keys on the literal
`("heldout", "model")` / `("driving", "heldout", "model")` paths, and
`TestDeprecatedTripwireIntact` asserts the literal strings are present in the source.

### 1.3 Verification — the 30 k gate, re-rendered

Inputs, all provenance-stamped:

* card: `Project Steering/Gates/flagship-v4-30k.card.json` (the binding one, unmodified)
* train log: pulled from `Sayood/flagship-v4-fromscratch`, **md5 `7d8bdeb458e064ab36216f9d03214c61`**
  — MEASURED here, and identical to `GATE_30K_RESULTS.md` §1's table
* eval: `…/2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-oracle.json`
* corridor: `…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json`
* secondary values: the five MEASURED ones from §4.3; the two with no emitter **deliberately not
  supplied**

Machine output: `GATE_30K_verdict_FIXEDTOOL.json` + `GATE_30K_rerender.log` (this directory).

> **✅ IT REPRODUCES THE HAND ADJUDICATION. No discrepancy.**

| criterion | bar | MEASURED | tool | hand (`GATE_30K_RESULTS.md` §4.3) |
|---|---|---|---|---|
| `wm_canary_ade_2s` | ≤ 0.55 | 1.1409 | **FAIL** | **FAIL** |
| `speed_benefit_recovered_frac` | ≥ 0.70 | *null* | NOT SUPPLIED | NOT MEASURED |
| `oracle_in_fan` | ≤ 0.30 | 0.2330 | PASS | PASS |
| `miss_at_2m` | ≤ 0.10 | 0.2123 | **FAIL** | **FAIL** |
| `seam_norm_ratio_max` | ≤ 1.0 | 0.1208 | PASS | PASS |
| `encoder_touching_levers` | ≤ 2 | 2 | PASS | PASS |
| `deploy_tick_p99_ms` | ≤ 50 | *null* | NOT SUPPLIED | NOT MEASURED |

`kill_conjunction: {n_pass: 3, n_fail: 2, n_unmeasured: 2, n_void: 1}` — **3 PASS · 2 FAIL · 2
unbuilt**, exactly as hand-adjudicated.

```
VERDICT: RESTART (NOT-CONTINUE) — pre-registered kill criteria FAILED on MEASURED values:
wm_canary_ade_2s = 1.1409… <= 0.55, miss_at_2m = 0.2123 <= 0.1. 2 pre-registered secondary(ies)
were NOT MEASURED (speed_benefit_recovered_frac, deploy_tick_p99_ms) — the gate is formally
INCOMPLETE, but a conjunction containing a hard FAIL cannot be rescued by measuring anything
else, so NOT-CONTINUE is already determined
```

with `restart_budget 0/2` for lever family `joint-planner-wm` ⇒ **RESTART**, not
`REFUTE_LEVER_FAMILY`. Also reproduced: the demoted `ade_0_2s` = **0.6423 FAIL, recorded, does not
adjudicate**; `verdict_adjudicated_by = ["secondary(7)"]`; the co-primary **0.6388 [0.5565, 0.7128]**
overall / **0.8432 [0.7874, 0.8919]** junction, n = 41/40, `pass: null`, `adjudicates: false`; and

```
[VOID]  nonav_route_beats_majority    original bar: >=1
        STATUS       : VOID_BY_CONSTRUCTION
        ADJUDICATION : INSTRUMENT-FAIL, NEVER MODEL-FAIL
        AUTHORITY    : GATE_PROTOCOL 0.7
        IN KILL SET  : NO — structurally excluded …; it did NOT contribute to the verdict
```

**One difference from the hand adjudication, and it is a difference in the WORD, not the outcome.**
The hand call was *"`INCOMPLETE` formally — but NOT-CONTINUE is already determined … resolves to
RESTART."* The tool now emits `RESTART` **directly**, carrying `not_continue: true` **and**
`formally_incomplete: true` and naming the two unmeasured secondaries in the reason. That is the
hand reasoning implemented, not a stronger claim: `INCOMPLETE` for a conjunction that can never be
satisfied is a category error, and leaving it there is what forced the hand adjudication.

⚠️ **NOT horizon-honest, and the tool says so.** With the co-primary report-only, the kill set is
K=20 (2.0 s) secondaries. The verdict is stamped `horizon_honest: false` with the reason. That is
the card's deliberate choice and it is the gap the next v4-line gate closes.

### 1.4 Tests — `stack/tests/test_run_gate_card_render.py` (33)

Load-bearing: `test_reproduces_the_hand_adjudicated_30k_verdict` renders the **real** card against
the **committed** artifacts and asserts every number above.
`test_the_old_code_path_would_have_refused_this_gate` pins the **defect** (`pytest.raises(TypeError)`
on `GateCard(**raw)`) so a regression is loud. Plus: the 0/1-indexing boundary table; unknown-key
preservation; nested-`co_primary` mapping; unknown-role refusal; report-only exclusion + full
printing + `pass is None`; void exclusion + printing + the `secondary`∩`secondary_void` case;
unsatisfiable-conjunction vs genuine `INCOMPLETE`; and the `heldout` tripwire.

---

## 2. DEFECT 2 — the OOD guard was structurally unable to fire

### 2.1 What was broken

`OODMap.ratio_arr` (`…/2026-07-26-v4-30k-gate/coprimary/v4_corridor_cl.py:188-192`; identical at
`…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py:176-181`):

```python
al = np.interp(lat_abs, self.lat_x, self.lat_y)      # CLAMPS beyond 3.0 m
ay = np.interp(yaw_abs_deg, self.yaw_x, self.yaw_y)  # CLAMPS beyond 12 deg
```

`np.interp` **clamps** at the envelope edge, so the ratio **saturates**. Two consequences:

1. **Every long-horizon OOD ratio this program has quoted is a LOWER BOUND, not a measurement.**
2. The `ratio > ~1.5x` criterion **structurally cannot fire out of envelope** — uninformative exactly
   where it matters most. MEASURED here (`test_np_interp_clamps_so_the_ratio_stops_growing`): on a
   P1-shaped map, `|dlat| = 3 m` and `|dlat| = 300 m` return the **same** ratio, and it never reaches
   1.5.

The verdict emitter (`v4_corridor_cl.py:427-431`) tested only that half:

```python
"EXTRAPOLATION_VERDICT": ("EXTRAPOLATION — peak OOD ratio exceeds ~1.5x …"
                          if float(ratio.max(1).mean()) > 1.5 else
                          "within the measured envelope on average"),
```

E1a's rule was always a **DISJUNCTION** (`e1a_horizon.py:28-30`): ratio > ~1.5x **OR whose steps
leave the measured envelope**.

MEASURED at the 30 k gate: ratio **1.2741** ("under 1.5") with **54.63 %** of steps beyond
|dlat| = 3 m, **35.48 %** beyond |dψ| = 12°, **90.24 %** of windows out of envelope — emitted as
*"within the measured envelope on average."*

### 2.2 The fix — `taniteval/taniteval/ood.py` (new, canonical)

* `envelope_fractions()` — the out-of-envelope fractions are **first class**, each naming its own
  denominator (`frac_steps_lat_over_3m`, `frac_steps_yaw_over_12deg`, `frac_steps_any`,
  `frac_windows_any_step_out_of_envelope`). `OODMap.ratio_and_fractions()` is the intended entry
  point: **the ratio never travels alone.**
* `verdict()` — E1a's **full disjunction**, reporting **which clause fired and why the other could
  not** (`criterion_1_ratio_over_1p5.informative: false` when saturated), and stamping
  `ratio_is_lower_bound`.
* `assert_envelope_verdict_consistent()` — runs **inside** `verdict()`, so it cannot be forgotten.
  It **raises** (`EnvelopeVerdictError`) on: a `MEASUREMENT` verdict with **any** step outside; any
  non-`EXTRAPOLATION` verdict with a **majority** outside; and a ratio reported with steps outside
  but **not stamped** as a lower bound. **A saturating estimator must declare its own saturation.**
* `verdict_class()` — verdicts are compared by **class**, never by wording, so a re-wording is not
  mistaken for a retraction (or vice versa). The legacy string classifies as `MEASUREMENT`, which is
  exactly why it was wrong.
* **Numerics unchanged.** `ratio_arr` is byte-equivalent to E1a's: the ratio was never the error.
  What changed is that it can no longer be quoted, or adjudicated on, without its saturation.

**Wired into the gate.** `run_gate.read_corridor` now adjudicates the full rule on every corridor
block it reads (`_ood_verdict` / `_corridor_ood`) and **prints it**:

```
  OOD      = EXTRAPOLATION — NOT a measurement at this horizon
             ratio 1.2741 (fires=False, informative=False) OR steps-outside-envelope
             (fires=True, steps=0.54634, windows=0.9024)
             !! the OOD ratio is a LOWER BOUND here (np.interp CLAMPS at |dlat|=3.0 m /
                |dyaw|=12.0 deg) — it may NOT be quoted as an in-distribution certificate
```

A block carrying **no** `EXTRAPOLATION_*` fraction is reported **UNKNOWN**, never in-envelope.
`stack` cannot import `taniteval`, so `run_gate` **mirrors** the rule; `test_run_gate_mirror_agrees`
pins the two against each other — the exact drift `closedloop.py` warns about.

### 2.3 Verification — `taniteval/tests/test_ood_guard.py` (26)

`test_a_majority_outside_cannot_read_as_in_envelope` feeds the **exact MEASURED 30 k numbers** and
requires the refusal. `test_the_30k_numbers_now_read_EXTRAPOLATION` re-adjudicates them:
`MEASUREMENT → EXTRAPOLATION`, `criterion_1.fires: false`, `criterion_2.fires: true`. Guarded the
other way too: `test_a_clean_rollout_is_still_allowed_to_say_MEASUREMENT` — the fix must not make
every verdict pessimistic.

### 2.4 THE SWEEP — which committed artifacts are wrong

Tool: `sweep_ood_verdicts.py` (this directory) → `ood_sweep.json` / `ood_sweep.log`. It walks every
**git-tracked** JSON, re-adjudicates every OOD node from the `EXTRAPOLATION_*` fields the emitters
**already wrote** — pure arithmetic, **no tensors, no GPU, no re-run**. Re-emission where per-window
tensors survive is `…/2026-07-26-v4-30k-gate/coprimary/fix_ood_verdict.py`, **reused, not rewritten**.

**194 OOD nodes across 14 committed artifacts.**

#### A. FALSE VERDICT STRING — the emitted claim is factually wrong (14 nodes, 3 files)

| artifact | nodes | MEASURED | class flip |
|---|---|---|---|
| `…/2026-07-26-v4-30k-gate/raw/corridor_v4_30k_K185.json` | 4 (`all_windows.185.ood.*`) | ratio 1.2741, steps 0.5463, **windows 0.9024** | `MEASUREMENT` → **`EXTRAPOLATION`** |
| `…/2026-07-26-v4-30k-gate/raw/corridor_refcbase_30k_K185.json` | 4 (`all_windows.185.ood.*`) | ratio 1.2761, steps 0.5043, **windows 0.9268** | `MEASUREMENT` → **`EXTRAPOLATION`** |
| `…/2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json` | 6 (`paired_common_start.185.ood.*` + `.20.`) | ratio 1.2741, steps 0.5463, **windows 0.9024** | `MEASUREMENT` → **`EXTRAPOLATION`** |

⚠️ **New finding, not in the gate report.** `fix_ood_verdict.py` corrected the **`all_windows`**
nodes of `coprimary/corridor_v4_30k_K185.json` but **not its `paired_common_start` nodes** — its
`main()` writes only `d["all_windows"][K]["ood"]`. **The same file therefore carries a corrected
verdict and a false one at K=185.** Pinned by
`test_the_committed_v4_artifact_still_carries_a_false_string`.

#### B. SATURATION UNDECLARED — verdict right, but the ratio is quoted without its lower-bound status (17 nodes, 3 files)

`…/2026-07-26-v4-30k-gate/coprimary/corridor_refcbase_30k_K185.json` (6) ·
`…/coprimary/corridor_v4_30k_K185.json` (7) · `…/coprimary/corridor_v4_30k_K185_produced.json` (4).

#### C. RATIO QUOTED WITH NO VERDICT AT ALL, at long horizon (118 nodes, 8 files)

The E1a family and its descendants emit the fractions but **no** verdict string — so the *prose*
supplied the conclusion, and the prose used the ratio alone.

| artifact | nodes (K≥100) | worst |
|---|---|---|
| `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44_K185.json` | 12 (8) | `all_windows.185.junction` ratio 1.2989, **windows 1.0** |
| `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44.json` | 36 (16) | `all_windows.120.junction` ratio 1.2957, **windows 1.0** |
| `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_clean17.json` | 34 (16) | `all_windows.80.junction` ratio 1.2865, **windows 1.0** |
| `…/2026-07-26-gate-primary-change/regate/corridor_refc-base-30k_K185_from_E1a.json` | 4 (4) | `junction` ratio 1.2989, **windows 1.0** |
| `…/2026-07-26-gate-primary-change/regate/refc-base-30k-INSTRUMENT-TEST-gate.json` | 1 (1) | `co_primary` steps 0.5281, **windows 0.907** |
| `…/2026-07-26-v4-30k-gate/{raw,coprimary}/corridor_*_K185.json` | 31 | see A/B |

#### D. NO ENVELOPE EVIDENCE AT ALL — UNKNOWN, not in-envelope (15 nodes, 3 files)

`…/2026-07-23-lower-ood-closedloop-source/lowood_closedloop.json` ·
`…/2026-07-23-lowood-lanekeeping-refc/lowood_lanekeep_40ep.json` · `…_smoke.json`.
These are **short-horizon** (K=20) and their ratios (~1.02–1.05) are plausibly valid, but they carry
no `EXTRAPOLATION_*` field, so the envelope clause **cannot be evaluated**. Recorded as UNKNOWN.

#### E. PROSE + LIVE GUARD CONSTANTS — the governance layer (MEASURED by direct read)

The claim *"the OOD-envelope ratio stays ≤ 1.30, so this is genuine **in-distribution** failure, not
extrapolation"* stands, **uncaveated**, in:

| file:line | status |
|---|---|
| `Project Steering/GATE_PROTOCOL.md:28-29` | **WRONG** — §0.1, the canonical protocol |
| `Project Steering/RETRACTION_LOG.md:65` | **WRONG** — the C6 entry |
| `Project Steering/LOOP_STATE.md:7, :11` | **WRONG** |
| `stack/scripts/run_gate.py:44-45` | **FIXED HERE** — replaced with an explicit retraction paragraph |
| `…/2026-07-25-closedloop-horizon-and-shift/E1a_E2a_RESULTS.md:173` | **WRONG at origin** — says the out-of-envelope fraction "is small"; it is **0.907**. Contradicts its own §1 honest bound at :34-36 |
| `…/2026-07-26-alpasim-consolidation/TANITSIM_FORK_RECOMMENDATION.md:212-214` · `Project Steering/Reports/2026-07-26-0757-program-report.md:34-35` · `Project Steering/Reviews/…/R4_measurement_results_rigor.md:29-31` | **WRONG** (inherited) |

⚠️ **Two LIVE guard constants gate real decisions on the saturated ratio:**

* `…/2026-07-25-e1b-failure-gated-clsft/scripts/e1b_eval.py:403` — `"c_ood_in_band": bool(ood_ft <= 1.30 + 1e-9)`
* `…/2026-07-26-e1c-heldout-gated-clsft/scripts/e1c_common.py:34` — `OOD_BAND = 1.30  # Gc (E1a measured band)`

**`1.30` has no provenance as an envelope test**: it is the observed K=185 ratio, i.e. the saturated
lower bound. E1b's guardrail (c) and E1c's Gc are therefore **not envelope checks**. E1b/E1c are
outside this brief's scope — **escalated, not silently patched**; both should adopt
`taniteval.ood.verdict`.

**What is NOT retracted.** The horizon finding itself stands: the paired K=20 vs K=185 delta is
measured on **identical windows** and does not depend on the envelope. What is retracted is the
*in-distribution certificate* attached to it — the long-horizon numbers are real closed-loop
measurements taken **mostly outside the validated warp envelope**.

---

## 3. DEFECT 3 — `taniteval/closedloop.py` could not serve a v4 arm

### 3.1 What was broken

| file:line (pre-fix) | what |
|---|---|
| `taniteval/taniteval/closedloop.py:101` — `K_MAX = max(WP_STEPS)` | 20 ticks = 2.0 s, treated as a **cap**. That is `ade_0_2s`' own **BLIND** horizon — the one the co-primary exists to replace (0.0035 at K=20 vs 0.5877 at K=185 on identical windows) |
| `closedloop.py:897` — `if not L["traj_capable"] or getattr(model, "tactical_policy", None) is None: SKIP` | a v4 `FlagshipV4Head` checkpoint has **neither**, so `run_and_save` **rejected every v4 arm** |
| `run_and_save` throughout | **no per-window persistence** |

Consequence: **the registered co-primary was reachable only through a one-off driver in
`incoming/`** — the exact stranding the operating standard forbids. And the missing per-window dumps
are why **no closed-loop artifact was correctable by arithmetic and all five had to be re-driven on
GPU** when the OOD rule was fixed.

### 3.2 The fix

**`taniteval/taniteval/clhorizon.py` (new)** — the port:

* `corridor_rollout(planner, episodes, goals, device, K, …)` — `v4_corridor_cl.rollout`
  (= `e1a_horizon.rollout`) with the **plan call injected**. Loop body, reference-index geometry,
  window/stratum bookkeeping reproduced verbatim. **Nothing caps K**; the only limit is the
  structural `T − W − K ≥ 1` (`HORIZON_CEILING_K = 190`, refused above).
* `V4Planner` — the v4 plan step: `world.encode_window` → `goal_modes.resolve_goal` →
  `head(st, v0, lambda_plan=1.0, **goal_kw)`, byte-for-byte `eval_flagship_v4.collect_planner`'s
  forward pass, with `traj[:, LOOKAHEAD_STEP-1]` to the same pure-pursuit controller. `goal_modes` is
  **injected**, not imported, so `taniteval` stays free of the stack layout.
* `emit()` / `corridor_from_perwindow()` — `taniteval.corridor.stratified` (THE registered emitter,
  unchanged) plus the **fixed** OOD block from `taniteval.ood`. `corridor_from_perwindow` is the
  arithmetic-only path that did not exist: a corridor block can be recomputed at a new half-width, a
  new stratification or a corrected rule **without a GPU**.
* `run_v4(...)` — a runnable end-to-end entry point (lazy stack imports) plus the ported `GoalCache`,
  so the co-primary needs no bespoke driver at all. `horizon_windows()` exposes the n-collapse.

**`taniteval/taniteval/closedloop.py`:**

* `K_MAX` is documented as a **default, not a cap**; `K_ADE2S`, `HORIZON_CEILING_K`,
  `OPEN_PLAN_MAX_K` added. `collect` refuses only the **structural** ceiling.
* `closed_loop_rollout` / `collect` / `run_and_save` take `plan_fn` — an arm without the
  `strategic_policy`/`tactical_policy` hierarchy can now supply its own plan step; `run_and_save`
  also accepts a pre-built `model`. It refuses **only** when there is genuinely no way to plan, and
  the message names `clhorizon.V4Planner`.
* Arm (A), the single-shot open-loop plan, needs the tactical head's knots. When an injected planner
  replaces that head — or `k` runs past the last knot — it is filled with **NaN, never zeros** (a
  zero path is a *plausible* path and would quietly produce a real-looking imagination A/B) and
  `analyze` emits an explicit `measured: false` NOT-MEASURED node.
* **`save_per_window=True` by default.** `run_and_save` writes
  `closedloop_<key>_perwindow_K<k>.pt` beside the JSON and records the path; the JSON says in words
  what the dump is for. With `save_per_window=False` the artifact is stamped *"NOT correctable by
  arithmetic."* CLI: `--k`, `--out-dir`, `--no-per-window`.

### 3.3 Verification — the reproduction

**(a) Bit-identity with the `incoming/` driver** (`test_port_is_tensor_identical_to_the_driver`):
the ported rollout and `v4_corridor_cl.rollout` are run on the **same** synthetic 3-episode corpus
with the **same** stub planner at K=30, and every tensor is asserted with `torch.equal`:
`lat`, `yaw`, `ade2s`, `hd2s`, `hdK`, `speed`, `t0`, `epi`, `de_fixed`, plus `eid`, `fixed_steps`
and `_rollout_steps_executed`. **All identical.** "Keeping the driver's measured behaviour" is
asserted, not asserted-about.

**(b) The gate numbers, on the COMMITTED per-window tensors**
(`test_reproduces_the_gate_coprimary_numbers`, MEASURED):

| arm | source tensor | overall CDR@1.75 m, K=185 | junction | n |
|---|---|---|---|---|
| **flagship-v4 30 k (oracle)** | `coprimary/corridor_v4_30k_K185_perwindow_K185.pt` | **0.6388 [0.5565, 0.7128]** ✅ | **0.8432 [0.7874, 0.8919]** ✅ | 41 win / 40 ep |
| **REF-C base 30 k** | `coprimary/corridor_refcbase_30k_K185_perwindow_K185.pt` | **0.5833 [0.5024, 0.6561]** ✅ | 0.7027 [0.4099, 0.8856] | 41 win / 40 ep |

Point estimates **and** intervals reproduce the `GATE_30K_RESULTS.md` §6.3 table exactly, estimator
`episode_cluster_bootstrap`, B = 2000. The 1.00 m threshold reproduces too (**0.7048**), which is the
re-aggregation-is-arithmetic proof. **No GPU was used** — the reproduction runs on the persisted
tensors, which is the capability D3's per-window fix exists to create.

The ported emitter's OOD block reads **EXTRAPOLATION** on all four strata
(`test_the_emitted_OOD_block_is_the_FIXED_one`): the string the original artifact carried **cannot be
produced** here.

### 3.4 Tests — `taniteval/tests/test_clhorizon.py` (15)

Beyond the two above: K far past 20 accepted; the structural ceiling refused; `horizon_windows`
n-collapse (200-frame episode → 1 window at K=185, 0 at T=190); no surviving window is a
NOT-MEASURED, not a pass; `K_MAX` is a default and `k` is a parameter of all three entry points;
`run_and_save` exposes `plan_fn` / `model` and defaults `save_per_window=True`; the injected plan
path **never touches** `strategic_policy` / `tactical_policy` (asserted with a model that raises on
access); and arm A NaN → NOT-MEASURED rather than a zero-filled A/B.

---

## 4. WHAT IS ESCALATED, NOT FIXED HERE

1. **The governance layer still carries the wrong OOD claim** — `GATE_PROTOCOL.md:28-29`,
   `RETRACTION_LOG.md:65`, `LOOP_STATE.md:7,:11`, the 07-26 program report, `E1a_E2a_RESULTS.md:173`
   and `TANITSIM_FORK_RECOMMENDATION.md:212-214`. Not edited: `Project Steering` is not this brief's
   surface and a protocol amendment is Sayed's call. **`GATE_PROTOCOL.md` §0.1 is the one that
   matters most** — it is the document every future card cites.
2. **Two live guard constants adjudicate on a saturated ratio** — `e1b_eval.py:403` and
   `e1c_common.py:34`. E1b/E1c decisions that turned on guardrail (c) / Gc rest on a criterion that
   **structurally cannot fire**. They should adopt `taniteval.ood.verdict`.
3. **`fix_ood_verdict.py` corrects only `all_windows`** — `paired_common_start` nodes in
   `coprimary/corridor_v4_30k_K185.json` are still false. One-line fix in its `main()`, left to the
   artifact's owner so the re-emission provenance note stays accurate.
4. **The next v4-line gate needs a corridor threshold** — and per §6.4 + this report, it should
   either register at a horizon where the envelope holds or re-validate P1 out to 18.5 s **on v4**,
   not on v1.

---

## 5. DELIVERABLE MANIFEST

**Repo — NOT `git add`ed, per the brief.**

| path | what |
|---|---|
| `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-instrument-fixes/INSTRUMENT_FIXES.md` | this report |
| `…/2026-07-26-instrument-fixes/sweep_ood_verdicts.py` | the D2 sweep (reuses `fix_ood_verdict.py`'s re-emission) |
| `…/2026-07-26-instrument-fixes/ood_sweep.json`, `ood_sweep.log` | the sweep result, 194 nodes / 14 artifacts |
| `…/2026-07-26-instrument-fixes/GATE_30K_verdict_FIXEDTOOL.json`, `GATE_30K_rerender.log` | the 30 k gate re-rendered through the FIXED tool |
| `stack/scripts/run_gate.py` | **modified** — D1 (all four), the OOD guard wiring, the docstring retraction |
| `stack/tests/test_run_gate_card_render.py` | **new**, 33 tests |
| `taniteval/taniteval/ood.py` | **new** — the canonical OOD guard (D2) |
| `taniteval/taniteval/clhorizon.py` | **new** — the horizon-capable rollout + v4 plan step (D3) |
| `taniteval/taniteval/closedloop.py` | **modified** — K cap, v4 refusal, per-window persistence |
| `taniteval/tests/test_ood_guard.py` | **new**, 26 tests |
| `taniteval/tests/test_clhorizon.py` | **new**, 15 tests |

**Scratchpad (not repo):** `…/scratchpad/v4fs/train_log.jsonl` — the v4-fromscratch train log pulled
from HF for the re-render, md5 `7d8bdeb458e064ab36216f9d03214c61`.

**Pods:** none touched. pod1 is TRAINING and was not contacted; no GPU was used — the whole
reproduction runs on the committed per-window tensors.

### Reproduction

```bash
export OMP_NUM_THREADS=8
cd stack      && python -m pytest -q      # 1053 passed, 3 skipped
cd ../taniteval && python -m pytest -q    # 401 passed

# the OOD sweep (arithmetic only, no GPU)
python "TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-instrument-fixes/sweep_ood_verdicts.py" \
    --repo . --json ood_sweep.json

# the 30 k gate through the fixed tool (train_log.jsonl from HF Sayood/flagship-v4-fromscratch)
python stack/scripts/run_gate.py check \
  --card "Project Steering/Gates/flagship-v4-30k.card.json" \
  --log <train_log.jsonl> \
  --reference-log taniteval/results/trainlogs/v1-speedjerk_train_log.jsonl \
  --eval-json     ".../2026-07-26-v4-30k-gate/raw/flagship-v4-fromscratch-30k-oracle.json" \
  --corridor-json ".../2026-07-26-v4-30k-gate/coprimary/corridor_v4_30k_K185.json" \
  --secondary-value wm_canary_ade_2s=1.1409059762954712 oracle_in_fan=0.23301841780357274 \
                    miss_at_2m=0.2123 seam_norm_ratio_max=0.1208 encoder_touching_levers=2
```
