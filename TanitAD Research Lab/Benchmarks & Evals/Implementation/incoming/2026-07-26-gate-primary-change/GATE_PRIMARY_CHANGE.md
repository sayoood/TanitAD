# Gate primary change — `corridor_departure_rate @ K` becomes co-primary, `ade_0_2s` is demoted

**Tier-1 #1** of the independent chief-scientist review (`01_EXECUTION_PLAN.md` B.2 **T1-1**,
*"the single highest-leverage correction in the review"*) plus **T1-1b** (re-gate v3enc and v4.1).
Date **2026-07-26**. Dev box only — **zero GPU, zero pod contact**.

**Nothing here is committed or pushed.** Files are edited/created in the working tree for the
orchestrator to stage.

---

## 0. The one-paragraph version

`run_gate.py`'s primary was `ade_0_2s`, measured at **K=20 (2.0 s)**, and E1a proved that instrument
blind to the dominant failure mode: on **43 identical held-out windows** corridor departure runs
**0.0035 at K=20 → 0.5877 at K=185**, junction **0.025 → 0.8414**, peak cross-track **0.35 m →
38.94 m**, paired Δ **+0.5842 [0.5071, 0.6565]**, separated, `p_delta_gt0 = 1.0` — while on those same
windows the paired **ADE@2s** delta is `0.0109 [−0.0, 0.0312]`, **not separated**. The gate now carries
`corridor_departure_rate` at a **pre-registered, explicit K** as **co-primary**, reports the **junction
stratum separately**, and demotes `ade_0_2s` to a reported diagnostic. **Both standing verdicts
(v3enc `RESTART`, v4.1 `FAIL`/`INCOMPLETE`) survive the change but neither is now admissible as
horizon-honest** — the horizon-honest re-gate is **data-blocked**: `0 of 30` committed `windows_*.pt`
carry a dense path, and the open-loop dense surface caps at **K=20** anyway, so a long-horizon corridor
read needs a **closed-loop rollout on GPU**.

---

## 1. Evidence table — every number in this document

| Claim | Class | Artifact |
|---|---|---|
| CDR@1.75 m: 0.0035 (K=20) → 0.5877 (K=185); junction 0.025 → 0.8414; peak XTE 0.35 → 38.94 m; paired Δ +0.5842 [0.5071, 0.6565] separated, p=1.0; ADE@2s Δ 0.0109 [−0.0, 0.0312] **not** separated | **MEASURED** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44_K185.json` (`paired_common_start`, 43 windows, `episode_cluster_bootstrap` B=2000) |
| Structural horizon ceiling K=190 (19.0 s); clips 190–199 frames; K=200 impossible | **MEASURED** | same file, `_horizon_ceiling_note`, `episode_T_min/max`; `taniteval/taniteval/corridor.py::horizon_ceiling` |
| **0 of 30 committed `windows_*.pt` carry `pred_dense`/`gt_dense`; all have `wp_steps=[5,10,15,20]`** | **MEASURED (ours, this task)** | sweep over `taniteval/results/windows_*.pt` + `TanitAD Research Hub/**/windows_*.pt`; reproduced by `regate/…` and by `gate_emitters.py corridor --windows` |
| REF-C base-30k ADE@2s = 0.4728 [0.3835, 0.5699], `episode_cluster_bootstrap`, 881 win / 40 eps | **MEASURED** | `taniteval/results/driving_refc-base-30k.json::headline.ade_0_2s` |
| v3enc@10k ADE@2s 1.9654 [1.6556, 2.2859]; probe R² 0.393; overshoot 2.195 | MEASURED / **INHERITED** (secondaries re-supplied from the 2026-07-21 gate JSON, not re-measured) | `taniteval/results/flagship-v3enc-10k.json`; `Project Steering/Gates/flagship-v3enc-gate-10k-2026-07-21.json` |
| v4.1@10k ADE@2s 0.8522 [0.7468, 0.98]; oracle_in_fan 0.4838; miss@2m 0.2486 | MEASURED / **INHERITED** (secondaries re-supplied from the 2026-07-23 gate JSON) | `taniteval/results/flagship-v4.1-10k.json`; `Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json` |
| v3enc's `decorr` was NEVER ON for the whole 0–9,999 gate window (`decorr_w = 0.0 if step < 10000`) | **INHERITED** (registry; `train_flagship4b.py:92` not re-read this task) | `Project Steering/MODEL_REGISTRY.md` §"the finding that reframes the failure" |
| The deprecated estimator is 1.28–2.06× too narrow **and** point-biased up to ×4.29 with sign flips | **INHERITED** | `CLAUDE.md`; `Project Steering/CI_RECOMPUTE_2026-07-20.json`; the 2026-07-25 blast-radius sweep |
| The 8 KILL / 5 report-only / CONTINUE dry-run split | **MEASURED** | `taniteval/results/v1_g1_dryrun_gate_FIXED.json::fixes_verified.split_8_KILL_5_REPORT` |

⚠️ **Not re-derived here:** E1a's numbers are read from its committed JSON, not recomputed. The
library that reproduces them (`taniteval/taniteval/corridor.py`, landed today, lifted verbatim from
`e1a_horizon.py`) is pinned against those artifacts by `taniteval/tests/test_corridor.py` — which is
green in the 334-test run below.

---

## 2. TASK 1 — what changed, file:line

### 2.1 `stack/scripts/run_gate.py`

| Where | What |
|---|---|
| `:1-…` docstring | New section **"THE HORIZON CORRECTION"** with E1a's table, the two hard rules (verdict names its horizon; K is pre-registered), and the bounded back-compat contract. Enforcement item **9** added. |
| `:147-148` | `PAIRED_CLUSTER_BOOTSTRAP_ESTIMATOR`, `DECISION_ESTIMATORS` — the admissible set, named. |
| `:152-156` | `DEPRECATED_ESTIMATOR` comment extended: **also** biases the point estimate (×4.29, sign flips), not only the width. |
| `:157-175` | `CORRIDOR_METRIC`, `CORRIDOR_PRIMARY_M`, `CORRIDOR_STRATA`, `JUNCTION_STRATUM`, **`ADE2S_K = 20`** (the demoted primary's own horizon, named not implied), **`HORIZON_CEILING_K = 190`**, `HORIZON_HONEST_MIN_K = 100`. |
| `:488` `horizon_seconds()` | K ↔ seconds at the MEASURED 10 Hz. |
| `:492` **`validate_horizon_K()`** | Refuses **K ≤ 20** (that *is* the blind horizon — message carries 0.0035 vs 0.5877) and **K > 190** (structurally impossible on 190–199-frame clips). Flags `K < 100` as not horizon-honest. Returns `frac_of_ceiling`. |
| `:536` `_corridor_strata_node()` | Locates the `taniteval.corridor` stratified block at top level or under `corridor` / `co_primary` / `driving.corridor`. Nothing else is guessed at. |
| `:553` **`_corridor_stratum_value()`** | **Raises** on `overlapping_holdout_se`, on an estimator outside `DECISION_ESTIMATORS`, on a missing estimator, and on a bare number with no interval. No fallback — unlike `ade_0_2s`, a corridor number has only ever come from one emitter, so an interval-free one means something re-implemented the metric. |
| `:593` **`read_corridor()`** | The co-primary reader. Records value + interval + estimator, **the junction stratum separately and always**, `horizon_K`/`horizon_s`, `n_windows`/`n_episodes`, `surface`, the full threshold grid, and the `EXTRAPOLATION_*` flags. **Refuses** a K ≠ registered K and a half-width ≠ registered half-width. |
| `:696` `read_corridor_paired()` | Arm-vs-arm delta; accepts **only** `paired_episode_cluster_bootstrap`. |
| `:757-773` `GateCard` | 9 new fields, **all defaulted** so every pre-existing card still loads: `co_primary_{metric,threshold,direction,horizon_K,corridor_m,stratum,junction_threshold,source}`, `primary_role`, `no_co_primary_reason`; property `has_co_primary`. |
| `:820-…` `cmd_register` | **Refuses to write a horizon-blind card** unless `--no-co-primary "<reason>"` is given on the record. Validates K. Sets `primary_role = "diagnostic"` automatically whenever a co-primary exists. |
| `:990-1016` `cmd_check` primary | `out["primary"]` gains `role`, `horizon_K`, `horizon_s` and a `demotion_note` carrying the measured reason. Printed as `[primary/diagnostic] … (recorded, does NOT adjudicate)`. |
| `:1018-1020` | `out["co_primary"]` and **`out["horizon"]`** — the mandatory horizon record. |
| `:1075-1105` verdict | Kill conjunction is **`co_primary ∧ secondaries`** when a co-primary is registered, **`ade_0_2s ∧ secondaries`** otherwise (back-compat). A registered-but-unmeasured co-primary ⇒ **`INCOMPLETE`**, never a verdict on `ade_0_2s` alone. |
| `:1107-1113` | `out["qualifier"]` — a demoted primary that FAILED while the co-primary passed is surfaced on the verdict, never swallowed. |
| `:1115-1120` | `out["horizon_honest"]` + the printed warning. |
| `:1128` `_co_primary_block()` | Builds the record; a missing/absent/**skipped** corridor artifact is `measured: False`, never a pass and never a crash. Reads the optional paired delta. |
| `:1219` `_horizon_block()` | The `horizon` block, including the legacy branch whose `warning` states the K=20 blindness in full. |
| `:1261` `_print_co_primary()` | Console rendering of overall + junction + paired. |
| `:907-948` comparative | **Bug fixed:** `matched_step_ratio` raises when two logs share no step carrying `compare_metric` — the normal state for v4.1, whose trainer logs `plan_ade`/`wm`/`oracle_ade` and has **no** `g_op_fwd_ade_m`. That raise propagated and **no v4.1 verdict could be rendered at all** on this box. A **diagnostic may not abort a verdict**: it is now recorded as `matched_step_ratio.available = false` with the arm's actual metric names. The standalone `run_gate.py ratio` subcommand still refuses hard. |
| CLI | `register`: `--co-primary-{metric,threshold,direction,horizon-K,corridor-m,stratum,junction-threshold,source}`, `--primary-role`, `--no-co-primary REASON`. `check`: `--corridor-json`, `--corridor-paired-json`. |

### 2.2 `stack/scripts/gate_emitters.py`

| Where | What |
|---|---|
| docstring | New section on the co-primary emitter **and the measured blocker** (0/30 dumps dense; open-loop caps at K=20; K≥100 needs a closed-loop rollout, i.e. GPU). |
| `:261-265` | `CORRIDOR_METRIC`, `CORRIDOR_HALFWIDTH_M`, `HORIZON_CEILING_K`, `ADE2S_K`. |
| `:267` `corridor_from_corridor_json()` / `:283` `…_dict()` | Torch-free read of a `taniteval.corridor` panel → gate value + horizon + **junction reported separately** + estimator + `evidence_class` + the exact `--corridor-json` argument. Emits `WARNING_blind_horizon` when `K ≤ 20`; emits `NOT MEASURED` for a `skipped` panel. |
| `:344` `corridor_from_windows()` | Computes the panel from `windows_<arm>.pt` (lazy torch import, same pattern as `speed_benefit_emit`). Returns the honest `skipped` node + a `blocker` string when the dump has no dense path. |
| `:394` `gate_values(corridor_json=…)` | Emits the co-primary under `co_primary` / `co_primary_arg`. **Deliberately NOT in `GATE_NAMES`** — routing the co-primary through `--secondary-value` would make it *off-card*, i.e. **report-only**, which silently disarms it (§9 split card). Pinned by a test. |
| CLI | new `corridor` subcommand (`--corridor-json` \| `--windows`); `gate-values --corridor-json`. |

### 2.3 New tests — `stack/tests/test_run_gate_corridor.py` (41 tests)

The load-bearing one is **`test_the_inversion_the_change_exists_for`**: two numbers MEASURED on the
**same arm** (`refc-diffusion-base-v21-30k` @ 29999) fed to the gate —

* `ade_0_2s` **0.4728** [0.3835, 0.5699] (open-loop 4wp) → **PASSES** a 0.60 bar
* `corridor_departure_rate` **0.5877** [0.5107, 0.6622] @ K=185 (closed loop) → **FAILS** a 0.35 bar

**old gate: `CONTINUE`. corrected gate: `RESTART`.** Other families pin: the horizon is explicit and
named with n; K≤20 and K>190 are refused (by `register` *and* by `check`, so a hand-forged blind card
cannot be smuggled through); a wrong-K or wrong-half-width block is refused; the junction stratum is
always reported and can adjudicate; a `None` stratum is NOT-MEASURED not a pass; the deprecated /
unnamed / interval-free estimator can never adjudicate; **the `heldout` tripwire is re-pinned**; a
missing/skipped corridor artifact is `INCOMPLETE` not a crash; `register` refuses new blind cards; the
committed dry-run split survives; a diagnostic cannot abort a verdict; and the E1a / REF-C numbers this
module quotes are re-read from their artifacts so doc-drift goes red.

### 2.4 Design decisions worth challenging

1. **"Demoted" means it does not adjudicate.** On a card with a co-primary, `ade_0_2s` is removed from
   the kill conjunction (`primary_role: "diagnostic"`) and a failure surfaces as `qualifier`. A
   diagnostic that still kills is not a diagnostic. `--primary-role kill` restores the conjunction for
   anyone who wants belt-and-braces, and the JSON always says which was used.
2. **Back-compat is bounded, not free.** A pre-2026-07-26 card renders **exactly** its old verdict
   (the dry-run pin reproduces), but is stamped `horizon_honest: false` + reason, and `register` will
   not write a *new* blind card without a written exception. Making old cards `INCOMPLETE`
   retroactively would have broken the committed dry-run, which the brief explicitly protects.
3. **No program-wide corridor threshold is invented here.** `register` requires one per arm. There is
   no standing bar — E1a's 0.5877 is the incumbent's *measured level*, not a target. Setting the real
   bars is a PI decision.

---

## 3. TASK 3 — `Project Steering/GATE_PROTOCOL.md`

New **§0 "THE HORIZON RULE"** (why, MEASURED, with E1a's table; what changed; K pre-registered and
bounded; junction separate; estimator; the honest limits), the co-primary threaded through **§1**
(register/check commands) and **§2** (evidence table, with the co-primary and the horizon as rows and
`ade_0_2s` relabelled *demoted*), new **§4b** (re-gate status, below), **§5** amended (persisting
`windows_<key>.pt` is necessary but **not sufficient** — the co-primary needs the dense path, and
K>20 needs a closed loop), new **§6** (back-compat boundary), new **§7** (the three standing rules).

**§7 rule 3 is the requested addition**, mirroring the exponent rule:

> **Never quote a gate verdict without its horizon and n** — K, seconds, windows, episodes, junction
> windows, surface. A verdict that does not name its horizon is not admissible, for the same reason a
> bare exponent is not: *the number is a function of a choice that was not disclosed.*

with the reason it is written down rather than merely implemented: **this file has already carried an
inadmissible number in a binding protocol** — `reached_at_step: 450`, declared VOID by the registry,
sat in §4 for four days and was struck only on 2026-07-25.

---

## 4. TASK 2 — re-gate of v3enc and v4.1

### 4.1 Results

| arm | historical verdict | re-render on its **original** card | re-gate on a **horizon-honest** card |
|---|---|---|---|
| **flagship-v3enc** @ 10k | `RESTART` (2026-07-21) | **`RESTART`** — identical, now `horizon_honest: false` | **`INCOMPLETE`** — co-primary not measured |
| **flagship-v4.1** @ 10k | `INCOMPLETE`, substantively FAIL (2026-07-23) | **`INCOMPLETE`** — identical, now `horizon_honest: false` | **`INCOMPLETE`** — co-primary not measured |

**Both verdicts survive the horizon correction in the sense that neither is overturned — and neither is
now admissible as horizon-honest.** They were rendered on an instrument MEASURED blind to the dominant
failure mode; the corrected gate refuses to re-issue them as decision-grade until the co-primary is
measured. The prior blast-radius sweep found no flips from the *estimator* correction; the *horizon*
correction produces no flip either, but for a different reason — it produces **no verdict at all**.

Artifacts (all in `regate/`):

* `flagship-v3enc-ORIGINALCARD-rerender-2026-07-26.json` · `flagship-v4.1-ORIGINALCARD-rerender-2026-07-26.json`
* `flagship-v3enc-REGATE.card.json` + `flagship-v3enc-regate-2026-07-26.json`
* `flagship-v4.1-REGATE.card.json` + `flagship-v4.1-regate-2026-07-26.json`
* `refc-base-30k-INSTRUMENT-TEST.card.json` + `-gate.json` + `corridor_refc-base-30k_K185_from_E1a.json` + `INSTRUMENT_TEST_log_stub.jsonl`

⚠️ The two `*-REGATE.card.json` are **post-hoc re-gate cards, not pre-registrations** — their `note`
says so. Their co-primary thresholds are E1a's measured incumbent levels used as a no-worse-than bar so
the card is well-formed; **no verdict here depends on their value**, because the co-primary is
unmeasured at every threshold.

### 4.2 The blocker — MEASURED, and it is data, not code

**`0 of 30`** committed `windows_*.pt` dumps carry `pred_dense`/`gt_dense`. All 30 are the 4-waypoint
sparse view with `wp_steps = [5, 10, 15, 20]`, including both arms under re-gate
(`windows_flagship-v3enc-10k.pt`, `windows_flagship-v4.1-10k.pt`). `taniteval.corridor.from_windows`
therefore returns its self-describing `skipped` node for every one of them, and the gate correctly
renders `INCOMPLETE`.

**And landing the dense keys is not sufficient.** `rollout.collect`'s dense path runs to `fwd_k = 20`,
i.e. **K = 20 (2.0 s)** — the blind horizon itself. A co-primary at K ≥ 100 is an inherently
**closed-loop** quantity (E1a's surface: the ego is simulated forward and control error accumulates).

**What would settle it, in order of cost:**

1. **Closed-loop rollout at K=185 on the two 10k checkpoints**, `e1a_horizon.py` protocol, ≥ 40 val
   episodes, `episode_cluster_bootstrap`, junction stratum separate. **Needs GPU.** Both checkpoints
   exist (`tanitad-eval:/root/models/flagship-v3enc-10k/`, `…/flagship-v4.1-10k/`); v4.1's is on a
   **single pod disk, not HF-backed** (registry §1.5.2) — a transfer risk that should be closed first.
2. **Paired vs the incumbent** — `corridor.paired_stratum_delta` on shared windows, never a quadrature
   combination. `run_gate check --corridor-paired-json` already reads it.
3. **Cheap partial credit, no GPU:** re-running `rollout.collect` needs the model, so even the K=20
   open-loop dense number is unavailable from disk. There is **no zero-GPU path** to any corridor
   number for these arms.

### 4.3 v3enc's second, independent defect — a corridor number will not repair it

v3enc's entire 0–9,999 gate window ran with **`decorr` NEVER ON** (`decorr_w = 0.0 if step < 10000`).
The gate measured the arm *before* the staged lever under test was applied, so the 10k gate does not
test D-A7's hypothesis **at any horizon**.

**Re-running the corridor metric on `ckpt_step10000.pt` would settle only "is this checkpoint
corridor-safe" — never "is the `encoder-grounding` lever family refuted".** Settling the latter needs a
gate step **after** decorr engages (≥ ~12–15k) or a restart with the lever on from step 0. That is a PI
decision about spending GPU-days, not an eval, and it is **outside this task's remit**. I have not
manufactured a verdict for it.

*(Provenance note: the brief cited `postmortem_b_egodropout_v3enc10k.json` for decorr-never-on. That
file measures **ego-dropout**, not decorr — the string `decorr` does not occur in it. The
decorr-never-on finding is in `MODEL_REGISTRY.md` §"the finding that reframes the failure", sourcing
`train_flagship4b.py:92`. Same conclusion, different artifact; flagged so the citation does not
propagate.)*

---

## 5. Test evidence

| Suite | Baseline (before) | After |
|---|---|---|
| `stack` | **924 passed / 3 skipped** | **965 passed / 3 skipped** (+41 new) |
| `taniteval` | **334 passed** | **334 passed** (untouched) |

Both re-run from a clean invocation with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`.
No pre-existing test was modified; `test_gate_emitters.py`, `test_run_gate_eval_metric.py` and
`test_run_gate_reached_at.py` are byte-unchanged and green.

**Instrument test** (the `v1_g1_dryrun_gate_FIXED.json` pattern — proves the corrected gate *sees* what
`ade_0_2s` cannot), `regate/refc-base-30k-INSTRUMENT-TEST-gate.json`:

```
[primary/diagnostic] ade_0_2s = 0.4728  CI [0.3835, 0.5699]  <= 0.6   -> PASS  (does NOT adjudicate)
[co-primary] corridor_departure_rate <= 0.35 @ K=185 (18.5 s)
  overall  = 0.5877 [0.5107, 0.6622] (episode_cluster_bootstrap, n=43 windows / 43 episodes) -> FAIL
  junction = 0.8414 [0.8144, 0.8667] (n=6 windows / 6 episodes)                              -> FAIL
VERDICT: RESTART
  horizon: corridor_departure_rate@1.75m K=185 (18.5 s, 97% of the K=190 corpus ceiling),
           n=43 windows / 43 episodes, surface=closed_loop
```

⚠️ **Two surfaces, declared:** the ADE is open-loop 4wp, the corridor is E1a's closed loop. That
pairing is intended (the co-primary is the closed-loop safety axis, the demoted primary the open-loop
trajectory-quality diagnostic) and both are stamped; they are never pooled. The 0.35 / 0.60 bars are
**illustrative**, chosen to exhibit the inversion, and decide nothing in the program.

---

## 6. What the sibling-owned docs must change

I did **not** edit `MODEL_REGISTRY.md`, `HYPOTHESIS_LEDGER.md`, or `Paper/`. Requests, escalated here
rather than left in a README:

1. **`MODEL_REGISTRY.md` §1.5.2 (flagship-v4.1)** — the gate verdict block should carry
   `horizon_honest: false` and a pointer to `…/2026-07-26-gate-primary-change/regate/`. The wording
   *"formal `INCOMPLETE`, substantively FAIL"* stays true, but "substantively FAIL" now rests on a
   metric MEASURED blind at K=20 and should say so.
2. **`MODEL_REGISTRY.md` v3enc §** — same stamp on the 2026-07-21 `RESTART`, plus the note that its
   re-gate is **doubly blocked** (no corridor artifact **and** decorr-never-on), so no amount of
   re-evaluating `ckpt_step10000.pt` can settle the lever family.
3. **`HYPOTHESIS_LEDGER.md`** — any hypothesis whose status was decided on `ade_0_2s` at 2 s needs a
   horizon column or an explicit "decided at K=20" marker. H25 (decorr-never-on) should carry the
   corollary that v3enc's 10k gate is void **for the lever family**, not merely for its estimator.
4. **`Paper/`** — the gate-methodology section must state the co-primary, the horizon rule, and the
   junction stratum. Any published gate verdict rendered before 2026-07-26 is horizon-blind and must be
   labelled.
5. **`RETRACTION_LOG.md`** (whoever owns it) — a new **root-cause class** is warranted:
   *"metric measured at a horizon shorter than the failure it is supposed to detect"*. It is distinct
   from the estimator class and from the prose-copying class, and it invalidated **every** gate verdict
   in the program's history as a decision-grade read.
6. **`taniteval/rollout.py` owner** — `collect` persists the dense path only to `fwd_k = 20`. The
   co-primary needs a closed-loop surface; if a long-horizon open-loop dump is ever wanted, `fwd_k`
   must be raised and the K recorded in the dump.

---

## 7. Deliverable manifest

**Nothing is `git add`ed, committed or pushed.** All paths repo-relative.

| Path | State |
|---|---|
| `stack/scripts/run_gate.py` | **MODIFIED** — co-primary, horizon rule, demotion, diagnostic-cannot-abort fix |
| `stack/scripts/gate_emitters.py` | **MODIFIED** — corridor co-primary emitter + `corridor` subcommand |
| `stack/tests/test_run_gate_corridor.py` | **NEW** — 41 tests |
| `Project Steering/GATE_PROTOCOL.md` | **MODIFIED** — §0, §1, §2, §4b, §5, §6, §7 |
| `TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-26-gate-primary-change/GATE_PRIMARY_CHANGE.md` | **NEW** — this file |
| `…/2026-07-26-gate-primary-change/regate/flagship-v3enc-REGATE.card.json` | **NEW** — post-hoc re-gate card |
| `…/regate/flagship-v3enc-regate-2026-07-26.json` | **NEW** — `INCOMPLETE` |
| `…/regate/flagship-v3enc-ORIGINALCARD-rerender-2026-07-26.json` | **NEW** — `RESTART`, `horizon_honest: false` |
| `…/regate/flagship-v4.1-REGATE.card.json` | **NEW** — post-hoc re-gate card |
| `…/regate/flagship-v4.1-regate-2026-07-26.json` | **NEW** — `INCOMPLETE` |
| `…/regate/flagship-v4.1-ORIGINALCARD-rerender-2026-07-26.json` | **NEW** — `INCOMPLETE`, `horizon_honest: false` |
| `…/regate/corridor_refc-base-30k_K185_from_E1a.json` | **NEW** — E1a's K=185 strata, verbatim, in the gate-readable shape |
| `…/regate/refc-base-30k-INSTRUMENT-TEST.card.json` + `-gate.json` | **NEW** — the ADE-passes / corridor-fails inversion |
| `…/regate/INSTRUMENT_TEST_log_stub.jsonl` | **NEW** — labelled stub, not a measurement |

Untouched, as instructed: `Project Steering/MODEL_REGISTRY.md`, `HYPOTHESIS_LEDGER.md`, `Paper/`,
`…/incoming/2026-07-25-h2-*`, `…/incoming/2026-07-25-e1b-*`, `taniteval/taniteval/corridor.py`,
`taniteval/tests/test_corridor.py`, and every pod.
