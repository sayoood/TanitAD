# INTAKE — CTRV: the third trivial floor the driving block is missing

**From:** Benchmarks & Eval agent · **Date:** 2026-08-02 · **Readiness: VALIDATED** (D-029)
**Branch:** `agent/benchmarks-eval-20260802`

> ⏹ **CLOSED 2026-08-16 — THE TITLE IS NOW STALE. The driving block is NOT missing CTRV; it landed
> 2026-08-03 and is in the tip.** Read the title as history, not as a to-do.
> Evidence (MEASURED, tip working tree — all four pieces of the proposal are present):
> - `taniteval/taniteval/driving.py:313` → `FLOORS = ("cv", "holdv0", "ctrv")` (was the two-member
>   family), with a `FLOOR_DESC` table naming each floor and a block comment recording the defect and
>   its size (*"adding CTRV moves 16 of 25 banked arms' headline verdicts and shrinks flagship-v1's
>   sustained-turn margin 5.3x"*).
> - `taniteval/taniteval/rollout.py:233` → persists `"ctrv": torch.cat(CT).float()` beside `cv`
>   (hunk 1 applied).
> - `taniteval/taniteval/ctrv_backfill.py` → present, as proposed.
> - `taniteval/tests/test_ctrv_backfill.py` → present, i.e. the backfill is inside the suite that must
>   stay green rather than only in this package.
> This matches the ORCHESTRATOR VERDICT below (**integrate-with-changes**, 2026-08-03) exactly, so the
> verdict is corroborated by the code and not merely asserted.
> ⚠️ **Escalation #1 in the block below is NOT closed by this.** `Project Steering/MODEL_REGISTRY.md:219`
> still reads *"Clears every trivial bar on the same 881 windows: … CTRV oracle 0.523"* — a
> point-estimate claim, while the package's paired estimator gives **+0.0993 [−0.026, +0.220] NOT
> separated**. ⛔ Registry text is Project-Steering-owned and was deliberately not edited by this
> sweep; the restatement is still owed. Swept by the 2026-08-16 stale-blocker sweep.

## What

Add **CTRV** (constant turn rate + constant velocity) to `taniteval/driving.py`'s trivial-floor
family, which today is `FLOORS = ("cv", "holdv0")` — **two straight lines**.

Two-file change, shipped as `proposed_ctrv_floor.patch` (verified: `git apply --check` clean against
repo tip `4978a82`):

| file | change |
|---|---|
| `taniteval/taniteval/rollout.py` | persist `baseline_waypoints(...)["constant_yaw_rate"]` as `ctrv` beside `cv`. **Zero extra compute** — the tensor is already computed on every window and discarded. |
| `taniteval/taniteval/driving.py` | `FLOORS = ("cv", "holdv0", "ctrv")`; `tier0` derives a per-call `floors` tuple from what the dump actually carries, so the 25 pre-2026-08-02 dumps keep scoring, and emits `floors_missing` so a two-floor row can never be mistaken for a three-floor one. |

## Why

`stack/scripts/driving_diagnostic.baseline_waypoints` returns **three** trivial predictors;
`rollout.collect` keeps one. So every lateral / turn / curvature verdict the canonical block has ever
emitted — `sustained_turn`, `by_curvature`, `lat_abs_2s_m`, `pathgeom_crosstrack_m`,
`verdict.where_the_win_lives` — was measured against a family **structurally unable to turn**.

MEASURED this run on the canonical 881-window / 40-episode val (`raw/ctrv_readjudication.json`):

- **CTRV is the dominant floor member**: ADE **0.5265** (gated) / **0.5230** (ungated) vs CV **0.8377**
  and hold-v0 **0.7876**; paired **+0.3113 m [0.167, 0.484] separated**; wins **423/881** windows
  (CV 156, hold-v0 302).
- **16 of 25 banked arms' headline verdicts move.** 12 arms "beat the floor" under CV; **6** do under
  CTRV. Seven arms flip to *losing* to the trivial floor.
- ⭐ **flagship-v1 @30k** (deployed): vs CV **+0.4106 separated**; vs CTRV **+0.0993 [−0.026, +0.220]
  NOT separated**.
- The turn wins are **real but ~5× smaller**: `sustained_turn` ADE +1.8063 → **+0.3398** (still
  separated, favours model); sharp-curvature heading +24.93° → **+7.69°**.

Community reference: the nuScenes `PhysicsOracle` is a best-of-**four** including **two yaw-rate
models**; ours was a best-of-two with both removed.

## Evidence / tests run

| check | result |
|---|---|
| `pytest tests/` (standalone, no stack) | **11 passed / 1.6 s** — analytic ground truth: independent-reference agreement, pure-arc CTRV vs CV order of magnitude, straight-line CTRV ≡ CV, the documented forward-Euler half-step bias, the speed gate, `hold_v0`, the window enumeration, `last=0` refusal, and both alignment-gate branches (G-B2) |
| alignment precondition on real dumps | **bit-exact** — `max_abs_diff_cv = max_abs_diff_gt = 0.0` on **25 of 27** banked dumps; the two 88-window smoke dumps correctly refused |
| CV floor reproduces the published gate | **0.8377 = `raw_v4fs-30k-produced.json` `floor_values.cv.ade_0_2s`** |
| model ADE reproduces the registry | **0.4271 = `MODEL_REGISTRY` §6 flagship-v1 full_set** |
| **patched block, end-to-end on the eval pod** | legacy dump → `floors=['cv','holdv0']`, `floors_missing=['ctrv']`, ADE **unchanged** 0.4271; backfilled dump → `floors=['cv','holdv0','ctrv']`, `ctrv_ade=0.523`, `vs_ctrv delta=+0.0959 [−0.0283, 0.2177] separated=False`. **The model's own numbers do not move** — asserted, not eyeballed. (`verify_patched_block.py`) |

Resource: **eval pod `tanitad-eval` (A40), CPU-only, 372.9 s** for the 27-arm sweep + ~90 s for the
patch validation. GPU untouched, no checkpoint loaded, no training pod touched, $0 marginal.

## Proposed target locations

- `taniteval/taniteval/rollout.py` — apply hunk 1.
- `taniteval/taniteval/driving.py` — apply hunk 2.
- `ctrv_floor.py` → **`taniteval/taniteval/ctrv_backfill.py`** (or `stack/tanitad/eval/`): keeps the
  25 legacy dumps scorable with three floors without re-running any model, and carries
  `verify_alignment`, which is the precondition gate any backfill needs.

## Risk

- **Low, and additive by construction.** No model number changes; only a floor is added. The
  regression goldens' *model* rows are untouched; any golden that pins `floors` or `floor_values` will
  need a one-time refresh (expected, and visible via `floors_missing`).
- **The patch persists the UNGATED CTRV**, identical to `driving_diagnostic.baseline_waypoints`. The
  speed gate (zero `omega` below 2 m/s, the 2026-07-15 standstill-yaw-noise fix) is a **separate
  decision**: MEASURED difference on val40 is **0.0034 m** (gated is marginally *worse*), so the
  artifact is small here. Shipping ungated keeps one definition in the codebase; `ctrv_floor.py`
  carries `v_gate=` for anyone who needs the gated form.
- **Known conservatism:** the shipped arc is forward-Euler with a measured **0.3044 m half-step bias**
  on a perfect 1 rad arc. A midpoint-corrected CTRV would be a *stronger* floor, so every margin above
  is a lower bound on the artifact. Deliberately **not** changed here — the floor must stay identical
  to the program's existing CTRV so it is comparable to the 2026-07-15 numbers.

## Rollback

`git apply -R proposed_ctrv_floor.patch`. Legacy dumps are unaffected either way; nothing is written
to any dump on disk by this change (the backfill is in-memory in the driver).

---

## ORCHESTRATOR VERDICT

- **Verdict:** **integrate-with-changes**
- **Date / by:** 2026-08-03 · orchestrator daily sweep
- **Reason & notes:**
  - **Re-verified here, not inherited.** Package tests re-run on this box against repo tip:
    **11 passed / 1.50 s** (`venvs/tanitad`). `git apply --check` clean against tip `dfddd4e`
    (the INTAKE cited `4978a82`; the patch still applies unchanged).
  - Integrated as the **priority override**: while `FLOORS = ("cv","holdv0")` every gate run
    publishes a CV-only verdict, and the package MEASURED that 12 arms "beat the floor" under
    CV where only **6** do under CTRV. That is a correctness defect in the block that decides
    restarts, so it outranks the age queue.
  - **The one change** (hence -with-changes, not plain integrate): `tests/test_driving.py::
    test_every_emitted_interval_names_its_estimator` iterated the **global** `D.FLOORS` and
    KeyError'd on `ctrv` for the pinned legacy 2-floor dump. This is exactly the golden refresh
    the package's own Risk section predicted ("any golden that pins `floors` … will need a
    one-time refresh"). Fixed to iterate the block's **emitted** `out["floors"]` and to assert
    `floors ⊎ floors_missing == FLOORS` — so the guard now also proves the partition, and adding
    a fourth floor later cannot silently unverify every legacy dump.
  - `driving.py`'s module docstring said "against BOTH trivial floors (**CV** and **hold-v0**)";
    corrected to name the family and the `floors_missing` mechanism.
  - Verified green AFTER integration: **taniteval 785 passed / 75.7 s** (was 784+1 failed),
    `test_ctrv_backfill.py + test_driving.py` **44 passed**.
- **Integrated as:**
  - `taniteval/taniteval/rollout.py` — persists `ctrv` beside `cv` (patch hunk 1).
  - `taniteval/taniteval/driving.py` — `FLOORS = ("cv","holdv0","ctrv")`, per-call `floors`,
    `floors_missing` (patch hunk 2) + docstring correction.
  - `ctrv_floor.py` → **`taniteval/taniteval/ctrv_backfill.py`** (as proposed), with its analytic
    ground-truth suite as **`taniteval/tests/test_ctrv_backfill.py`** — the backfill is now inside
    the suite that must stay green, not only in the incoming package.
  - `taniteval/tests/test_driving.py` — the golden refresh above.

### Escalations — orchestrator disposition

1. **`PROJECT_STATE.md` / `MODEL_REGISTRY.md` §0.3 "first arm below EVERY trivial bar"** —
   **UPHELD, restatement owed.** The escalation is correct: on the canonical windows with the
   programme's own paired estimator flagship-v1 vs CTRV is **+0.0993 [−0.026, +0.220] NOT
   separated**. Point estimate still below the floor; the *separation* claim is not. Logged to
   `PROJECT_STATE.md` this sweep; the registry edit is queued behind it (registry is the quotable
   source, so it gets a dedicated edit, not a drive-by).
2. **Three dumps carry a packed-string `eid`** (`windows_flagship-v4.1-10k.pt`,
   `-v4.2-step4000.pt`, `-v16-ab-ft.pt`) — **ACCEPTED as a work item**, not fixed here: it is a
   write-time normalisation in the dump producer, owned by Benchmarks & Eval, and touching three
   banked dumps mid-sweep would break the alignment precondition this package depends on.
   Independently corroborated by the banked-window memo, so it is not a single-source claim.
3. **The v4 30k gate's `sustained_turn +1.2416` / `where_the_win_lives = "lateral only"`** —
   **QUARANTINED.** Both were measured against the two-floor family and must not be re-quoted
   until re-run with three floors. Recorded in `PROJECT_STATE.md`.

## Escalation (⛔ not editable by this agent — Project Steering)

1. **`PROJECT_STATE.md` / `MODEL_REGISTRY.md` §0.3** state flagship-v1 is *"the FIRST arm below EVERY
   trivial bar"*, citing an INHERITED `CTRV 0.523` from another corpus with no interval. On the
   canonical windows with the program's own paired estimator that comparison is **a tie, not a
   separated win**. The point estimate is still below the floor — the claim needs restating, not
   deleting.
2. **`windows_flagship-v4.1-10k.pt`, `windows_flagship-v4.2-step4000.pt`,
   `windows_flagship-v16-ab-ft.pt` carry a DIFFERENT `eid` encoding** (packed string uid,
   e.g. `808464434`) from every other dump (`0..39`), with bit-identical tensors. Any cross-arm join
   keyed on `eid` will mis-join these three. One-line normalisation at write time.
3. **The v4 30k gate** (`incoming/2026-07-28-v4-30k-gate`) published `sustained_turn +1.2416 favours
   model` and `where_the_win_lives = "lateral only"` against the two-floor family. Re-run with the
   third floor before either is quoted again.
