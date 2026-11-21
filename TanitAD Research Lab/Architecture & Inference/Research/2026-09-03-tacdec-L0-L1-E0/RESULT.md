# RESULT — L0 (seed = goal), L1 (the invisible loss term), E0 (the refutation arm)

**Date:** 2026-09-03 (Europe/Berlin) · **FlyWheel:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **SPEC:** `./SPEC.md`, written before any edit.
**Governing pre-registration:** `Project Steering/PREREG_TACTICAL_DECODER.md` (L0 §5/§7 S1;
L1 §5/§3 preflight; E0 §8).
**Compute:** dev-box RTX 4060, forward-only, **40.8 s** of GPU for the whole of E0.
⛔ **Thor was NOT contacted** (`refav1-b1-v72-ep3-speed` runs to ≈ 2026-09-04 02:00Z); no pod was
contacted; nothing here is shipped to a live run.

---

## ⭐ HEADLINE — and E0's verdict against its committed branch, verbatim

> ### ⛔ **E0 DID NOT REFUTE THE LINE — and the reason it did not is itself the finding.**
>
> **The committed refutation branch, verbatim** (`PREREG_TACTICAL_DECODER.md` §8.2 Q1):
> *"If the goal-term advantage stays ≤ 0 on a majority of the 25 even at seed/goal identity and in
> f64, then the goal SPACE — not the decoder, not the metric, not the weights — is what cannot
> represent a manoeuvre, and the whole decoder-then-cost line is REFUTED."*
>
> **MEASURED:** at seed/goal identity, in f64, the goal-term advantage is **POSITIVE on 25 / 25
> windows** under **both** conventions. The majority threshold is 13. **The branch did not fire.
> `H-REFAV1-COST-SEED-1` and `H-REFAV1-TAC-DECODER-1` both survive E0.**
>
> **But §8.4's OTHER committed branch DID fire**, verbatim: *"If the oracle-goal cost also fails to
> prefer the turn, the defect is the cost; if it succeeds, the defect is the imagined goal."*
> With the **ground-truth 6 s field** as the goal, the canonical turn is **FURTHER** from it than
> constant velocity on **22/25** (conv A) and **24/25** (conv B) — and on **6/9** / **8/9** of the
> windows where the car *actually* turns ≥ 5°. ⇒ **the defect is the COST.**

**The three numbers that carry the whole panel:**

| | MEASURED | reading |
|---|---|---|
| goal-term advantage at identity (f64, conv A, median) | **+1.49 × 10⁻⁹** | the goal space **does** prefer the turn |
| the `0.05·κ²` charge the same turn pays | **2.24 × 10⁻⁴** | …by a factor **1.5 × 10⁵** too little |
| turn total-cost wins under the shipped `1−cos`, ALL arms, both conventions | **0 / 25** | the preference is unusable as shipped |

⭐ **AND ONE CONFIGURATION DOES RANK THE TURN, AT THE SHIPPED WEIGHTS:** L0 identity **+ chord
(`√(2(1−cos))`) + the REPAIRED units crossing** (`model_action_units="steer"`, convention B) wins
**25 / 25**. Under the legacy convention A the same arm wins **5 / 25**, so `S3`'s *"both
conventions"* clause **FAILS as written** — my registered prediction that S3 would fail is upheld,
but for a reason the prediction did not name. ⚠️ **Read the n before quoting the 25/25:** those 25
windows are **4 episodes** (7/7/7/4) and **all 25 carry the identical token
`TURN_R × ADAPT_SPEED_FOR_CURVE`**; the margin is a thin **1.19–1.23×** on three of the four
episodes. It is a 4-cluster result on one token, not a 25-sample result.

**⛔ ESCALATION — INTEGRATION.** Three items need the Master Mind, not a note in a README:
1. **`plan_horizon_s` 2.0 → 6.0 is a DESIGN decision and it is now the load-bearing one.** L0 as
   implemented buys *identity*, not *horizon* (§2.3). Only R1 makes the seed span the manoeuvre the
   token names, and it changes the optimised window, the search dimensionality, the proposal head's
   output shape (`refa_v1.py:1093`) and every banked comparison. **It is not mine to take.**
2. **The chord (R29 / L3) is no longer optional and is no longer weight-neutral** — it is the only
   metric under which any arm ranks a turn at the shipped weights, and it must land **jointly with
   L4** as §5 requires.
3. **`model_action_units="steer"` moved from "removes a trap, wins nothing"
   (`D-REFAV1-BOUNDARY-NULL`) to LOAD-BEARING**: the 25/25 exists only under it. That registry row
   needs a qualifier, not a retraction.

---

## 0. Test baseline — before and after, with every pre-existing failure named

Selection `-k "refa or refav1 or tac or seed_goal"`, run mirror `C:\Users\Admin\tanitad-wt`,
venv `C:\Users\Admin\venvs\tanitad`.

| | result | artifact |
|---|---|---|
| **BEFORE any edit** | **1 failed, 711 passed**, 0 skipped, 5053 deselected, 91.07 s | `raw/pytest_before.txt` |
| **AFTER** | **1 failed, 736 passed, 1 skipped**, 5053 deselected, 112.56 s | `raw/pytest_after.txt` |

**Reconciliation: 711 + 26 new = 737 = 736 passed + 1 skipped.** No pre-existing test changed state.

**The one pre-existing failure, named:**
`stack/tests/test_decision_check.py::test_refa_surfaces_the_measured_arm` — **fails identically
before and after**. Its own stdout says `DECISIONS_2026-07-20.md NOT FOUND`,
`MODEL_REGISTRY.md NOT FOUND`, `RETRACTION_LOG.md NOT FOUND`: the **run mirror carries only
`Project Steering/BACKLOG.md`**. A mirror-environment artifact, not a code failure, unrelated to
every file this package touches.

**The one skip, named** (it is new only because the box changed, not because the code did):
`stack/tests/test_refa_v1_precision.py:330` — *"GPU busy — util 47 %, 810/8188 MiB, other python
compute apps []: another agent's job must not be disturbed"*. It is that file's own yield-the-GPU
guard firing on desktop compositing load; **no other python compute app was present**. Forced
re-run `TANITAD_REFAV1_CUDA_TEST=1 pytest stack/tests/test_refa_v1_precision.py` → **8 passed in
3.94 s**. *(MEASURED; appended to `raw/pytest_after.txt`.)*

New tests added by this package: `stack/tests/test_seed_goal_agreement.py` (**18**) and
`stack/tests/test_tac_loss_logging.py` (**8**) — **26**, all passing.

### 0b. The FULL suite — because the selection is not the suite

`refa_v1.py` is imported widely, so the `-k` selection is not sufficient evidence that nothing else
moved. **`pytest stack/tests -q` (5,790 tests): 20 failed, 5,653 passed, 109 skipped, 2 xfailed,
7 errors, 480.06 s.**

⚠️ **I did not assume those 20 were pre-existing — I measured it.** The two modified files were
replaced in the **run mirror only** by `git show HEAD:…` (the repo and the index were never
touched) and the failing files re-run:

| check | result |
|---|---|
| the 8 files carrying 18 FAILED + all 7 ERRORs, HEAD vs mine | **identical failure sets — `comm` diff empty in both directions** (25 lines each) |
| the 2 remaining failures (`test_bev_consumer_fov`, `test_build_parity_guard`), re-run at HEAD | **both reproduce** |

⇒ **every full-suite failure is pre-existing; this package introduces zero regressions across the
whole suite.** All of them are mirror-environment artifacts: missing `Project Steering/*` records,
a missing `…/2026-08-11-ops-bundle/p8c_chain.sh`, an ungated `scripts/dinov3_fp8_encode_ship.py`,
and an uninstalled git hook. Complete lists and the HEAD baseline: **`raw/pytest_full_suite.txt`**.
⛔ Whether `pytest -q` is green **in the repo checkout** is the committer's gate, not this
FlyWheel's — and nothing here is committed.

---

## 1. L1 — the loss term that had never been logged (BACKLOG R37)

### 1.1 The absence, re-verified here with two differently-bound probes

| probe | binding | result |
|---|---|---|
| POSIX `grep -c "loss_lat_label\|loss_lon_label\|loss_route_label" stack/scripts/refa_v1_train.py` | repo-relative path, ripgrep/grep | **0** (exit 1) |
| Python `io.open(<absolute path>).read().count(k)` for each key | absolute path, CPython file read | **0, 0, 0** (and `loss_feat_str_ext` **0**) |

*(MEASURED. Two probes with different path-binding, as the G:-mount rule requires.)*

### 1.2 What the term is worth — the number that made it a preflight condition

| checkpoint | tactical label share of the label+feature sum | source |
|---|---|---|
| incumbent (step 1,000) | **13.5924 %** | `…/2026-09-03-tactical-decoder/raw/intent_probe_incumbent.json` → `loss_share.nav_true` |
| ep2 (step 1,000) | **20.5106 %** | `…/raw/intent_probe_ep2.json` → same key |

*(MEASURED — re-read from the sibling package's raw JSON, not from its prose. Tier T0.)*

### 1.3 The change, and the artifact behind it

`stack/scripts/refa_v1_train.py` — four keys in the log row, built from a block that folds them
**exactly as `refa_v1.py` folds them into the objective**:

```
w_tac_label * mean(present of {loss_lat_label, loss_lon_label})   # refa_v1.py:1646-1649
  + w_str_label * loss_route_label                               # refa_v1.py:1663-1664
--------------------------------------------------------------   / loss
```

**Banked evidence: `raw/l1_train_log_rows.json`** — two real runs of the shipped trainer.

| | labelled step | unlabelled step |
|---|---|---|
| `loss_lat_label` | 2.899582624435425 | **`null`** |
| `loss_lon_label` | 1.7412283420562744 | **`null`** |
| `loss_route_label` | 0.6842218637466431 | **`null`** |
| `loss_label_weighted_share` | **0.18101985041464227** | **`null`** |
| hand computation of the share | 0.18101985041464227 (`_hand_matches: true`) | n/a |

⛔ **`None`, never `0.0`.** A `0.0` in this column reads as *"supervised, and perfect"*, so an
unlabelled step and a perfectly-classified step would print the same character. `refa_v1.py:1643`
**skips** an all-ignored family rather than averaging a NaN, and **8.13 %** of batch-8 steps carry
no lateral label at all (MEASURED, `…/2026-09-03-tactical-decoder/raw/window_band_census.json`), so
this is the common case, not an edge one. `test_tac_loss_logging.py::test_c_*` rejects `0.0` **by
identity**, because `not 0.0` is `True` and a laxer assertion would pass on it.

### 1.4 The load-bearing regression

`test_e_every_pre_existing_key_survives_unchanged` (both labelled and unlabelled) asserts that the
**21 pre-change keys** are all still present, that `set(new) − PRE_CHANGE_KEYS` is **exactly** the
four added keys (nothing renamed, nothing dropped), and that every pre-existing key is
**bit-identical across two identically-seeded runs**. `test_f_*` additionally closes the accounting
identity — `w_feat_op·L_op + w_feat_tac·L_tac + w_feat_str·L_str + (share · loss) == loss` to
`1e-6` relative — which is what makes the share's **denominator** meaningful.

⇒ **`PREREG_TACTICAL_DECODER.md` §3's preflight condition *"`loss_lat_label` / `loss_lon_label` are
still absent from the trainer's log row"* is DISCHARGED.**

---

## 2. L0 — the seed and the goal (BACKLOG R38)

### 2.1 The diagnosis, exactly, with file:line for both sides

*(Line numbers here are POST-edit and were each read out of the file with `grep -n`; the pre-edit numbers were `:1757`/`:1761` and `:1932`. ⚠️ `SPEC.md`'s citations are PRE-edit BY CONSTRUCTION — it was written before the edit and is deliberately not back-dated.)*

| side | site | operative indices it consumes |
|---|---|---|
| **the GOAL's control** | `stack/tanitad/refs/refa_v1.py:1804` builds `ctrl = canonical_controls(lat, lon, v0, cfg.op_steps=30, cfg.op_dt=0.2)`; **`:1814`** feeds the tactical predictor `self._model_actions(ctrl, v0, units)[:, ::stride][:, :cfg.tac_steps]`, `stride = 3` | **`[0, 3, 6, 9, 12, 15, 18, 21, 24, 27]`** — 6.0 s |
| **the SEED** | `stack/tanitad/refs/refa_v1.py:2014` `seed = goal_action["controls"][:cfg.plan_steps][None]`, `plan_steps = round(plan_horizon_s / op_dt) = round(2.0/0.2) = 10` (`:328`, `:628-629`) | **`[0..9]`** — 2.0 s |
| the cost's re-grid | `:1980-1985` (`tac_idx`) and `:2065-2066` | `"dense"` (DEFAULT, `:1885`) → `[0..9]`; `"tactical"` → `[0,3,6,9,9,9,9,9,9,9]` |

⇒ **the goal's actions at operative indices 12, 15, 18, 21, 24 and 27 are NOT IN THE SEED AT ALL.**
No `cost_time_grid` can address them. **The cause is the TRUNCATION, not the regrid** — which is why
`4139203`'s repair does not close it, exactly as `seed_goal_mismatch.py` reported.

### 2.2 Which repair I took, and why — stated plainly

| # | repair | verdict |
|---|---|---|
| R1 | `plan_horizon_s` 2.0 → 6.0 | ⛔ **NOT MINE.** A DESIGN decision: it changes the optimised window, the search dimensionality, the proposal head's output shape (`:1093`, `proposal_k · plan_steps · a_dim`), the length of the returned plan and every banked comparison. **Escalated, not taken.** |
| R2 | pack the goal's stride-3 actions into the 10 seed slots (`ctrl[::3][:10]`) | ⛔ **REJECTED, and recorded in the source so it is not re-proposed.** It gives 64/64 under `"dense"` ONLY, and it makes the seed a **3× time-compressed control**: the plan is executed on the operative grid at `op_dt` (`PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt)`, `:1938`) and the coarse→fine re-score rolls `self.operative` over the same tensor with no regrid, so it would drive a 6 s manoeuvre in 2 s. Under `"tactical"` it cannot work at all — `tac_idx` takes 4 distinct values and cannot address 10 distinct goal actions. |
| **R3** | **the honest alternative the brief names: build the goal over the plan's OWN horizon, flag-gated, default OFF** | ⭐ **TAKEN.** |

**⇒ I DID THE SECOND THING, NOT THE FIRST.** Making the seed reproduce the shipped 6 s goal is
**impossible** without changing `plan_horizon_s`, for the reason in §2.1: the actions simply are not
in the seed. So the goal is instead re-rolled from the seed's own feed.

**R3 as implemented** — new `plan()` keyword `goal_time_grid`, mirroring the existing
`cost_time_grid` machinery (`GOAL_TIME_GRIDS = ("full", "plan")` at `refa_v1.py:202`,
`_check_goal_time_grid` at `:205`):

* **`"full"` — THE DEFAULT, byte-identical to the shipped code.** Verified two ways in §2.4.
* **`"plan"`** — at the seed-construction site (`:2014-2041`) the goal is re-rolled from the seed's
  **own** action feed: the same `_model_actions(...)` call, the same `tac_idx` re-grid, the same
  predictor (`search_pred`) and the same `z0` (`search_z`) a candidate gets in `_cost_chunk`. The
  canonical seed's cost rollout and the goal rollout are then **the same forward pass**.

⚠️ **THE HONEST COST, stated because it must not be discovered later: `"plan"` BUYS IDENTITY, NOT
HORIZON.** Under it the goal is the manoeuvre's first `plan_horizon_s` (`"dense"`) or its first
`plan_horizon_s` held to 6 s (`"tactical"`), **NOT** the 6 s manoeuvre the token names.
`test_h_the_repair_buys_identity_not_horizon` asserts the two goals genuinely differ, so no reader
can quietly upgrade the claim. Restoring the token's manoeuvre needs **R1**.

⚠️ Two further honest costs: the flag adds **one extra tactical rollout per plan tick** (the
original `_imagine_tactical_goal` must still run — its `ctrl` *is* the seed), and it is a **`plan()`
flag**, so the standalone `imagined_goal()` API is unaffected. `goal_source` deliberately does not
change; the provenance is the call-site argument, stamped on the result as `res.goal_time_grid`,
exactly as for `cost_time_grid`.

### 2.3 BEFORE / AFTER on the 64 token pairs — `seed_goal_mismatch.py`'s own metric

| | agreeing pairs | κ-mismatching pairs | worst heading excess | source |
|---|---|---|---|---|
| **BEFORE (banked, pre-edit)** | **2 / 64** | 48 / 64 | **82.506°** on `TURN_L` | `…/2026-09-03-tactical-decoder/raw/seed_goal_mismatch.json` |
| **DEFAULT, re-run on the PATCHED code with the tool UNMODIFIED** | **2 / 64** | 48 / 64 | **82.506°** on `TURN_L` | `raw/seed_goal_mismatch_AFTER_L0_default.json` |
| **REPAIR `goal_time_grid="plan"`, `dense`** | **64 / 64** | 0 | **0.000°, exactly** | `raw/seed_goal_agreement.json` |
| **REPAIR, `tactical`** | **64 / 64** | 0 | **0.000°, exactly** | same |

The agreeing pairs under the default are exactly `(LANE_KEEP, CRUISE)` and `(ABORT_LC, CRUISE)` —
the all-zero control — in both grids, unchanged. **The middle row is the deliberate-regression
evidence:** the READ-ONLY tool, re-run unmodified against the patched source, reproduces the defect
number to the digit, so the default path did not move.

**Tolerance on the `TURN` rotation error: EXACTLY 0.0°, not "small".** Both sides consume the
identical `list` of floats, so this is an identity, not an approximation, and
`test_b_*` asserts `exc == 0.0` for all 64 pairs rather than a bound.

**And the same metric on RUNNING code** (`raw/seed_goal_agreement.json`,
`tools/seed_goal_agreement_report.py`): a real `RefAV1.plan()` is driven once per token pair per
`cost_time_grid`, the feed the goal rollout **actually receives** is captured, and compared
element-for-element with the feed `_cost_chunk` gives the seed:

| arm | feeds equal | worst `1 − cos(z_seed, goal)` (f32) |
|---|---|---|
| `full` × `dense` (**the default**) | **2 / 64** | — |
| `full` × `tactical` | **2 / 64** | — |
| `plan` × `dense` | **64 / 64** | **1.788 × 10⁻⁷** |
| `plan` × `tactical` | **64 / 64** | **−2.384 × 10⁻⁷** |

against a float32 cosine resolution of **4.768 × 10⁻⁷** (`4 × spacing(1f)`) — i.e. **identity, to
the precision the shipped cost is computed in.** *(MEASURED, tiny random-init rig; the identity is
a property of which ACTIONS are fed, not of what the weights know.)*

### 2.4 The default is unchanged — two independent checks

1. `test_e_the_default_flag_changes_nothing`: `plan(...)` and `plan(..., goal_time_grid="full")`
   return equal `controls`, `cost`, `source` and `baseline_costs`.
2. E0's `shipped__A` arm reproduces the **banked cost surface** window for window — see K6 in §3.2.

---

## 3. E0 — the cheapest discriminating arm (`PREREG_TACTICAL_DECODER.md` §8)

**Population:** ep2's curved-goal windows, `goal_kappa_max > 0`, selected exactly as
`turn_decomposition.py` selects them. **MEASURED: 25 of 140 windows.**
**Compute:** 40.8 s on the 4060, forward-only. **Artifacts:** `raw/e0_goalspace_ep2.json`
(analysis), `raw/e0_goalspace_ep2_rows.json` (every per-window number), `raw/e0_ep2.log`,
`raw/run_e0.sh`.

### 3.0 ⚠️ The n, printed before any conclusion (§6a gate 5)

| fact | MEASURED |
|---|---|
| windows | **25** of 140 |
| **episodes** | **4** — 7 / 7 / 7 / 4 windows |
| **distinct decoded tokens across all 25** | **1** — every window is `TURN_R × ADAPT_SPEED_FOR_CURVE` |
| of the 25, `gt_turn_deg ≥ 5°` (the car actually turns) | **9** |

⛔ **This is a 4-cluster panel on ONE token.** Any "x / 25" below is really "x windows across 4
episodes"; the episode-cluster estimator has **n = 4**. The 25 curved-goal windows and the
geometric turn stratum are **different denominators** and no count here mixes them.

### 3.1 Q1 — does the goal field prefer a turn at all, at seed/goal identity, in f64?

Convention A = `model_action_units="kappa"` (as shipped, the legacy path);
B = `"steer"` (the repaired crossing).

| arm | goal | candidate | turn `c_goal` f64 (max abs) | advantage f64 (median) | **advantage > 0** |
|---|---|---|---|---|---|
| `shipped` | 6 s imagined | seed `ctrl[:10]`, dense | 6.147e-06 | **−3.967e-09** | **5 / 25** (A), 5 / 25 (B) |
| **`identity_full`** | **6 s imagined** | the goal's own tactical actions | **3.331e-16** | **+1.491e-09** | **25 / 25** (A and B) |
| `identity_plan` (**L0**) | re-rolled over the plan window | seed `ctrl[:10]` | 2.220e-16 | +8.473e-09 (A) / +7.772e-08 (B) | **25 / 25** (A and B) |
| `C_REG` (deliberate regression) | 6 s imagined | seed with the **saturating** regrid | 1.428e-06 | −3.967e-09 | 5 / 25 |

⇒ **Q1 answers YES: the tactical goal field DOES prefer the turn once the seed is the goal's own
control — 25/25, f64, both conventions.** The refutation branch required *"≤ 0 on a majority"*; the
observed count is **0 windows ≤ 0** on both identity arms. **NOT REFUTED.**

⚠️ **And the `shipped` arm shows why this was invisible:** as shipped, the turn's goal term
(median **5.570e-09**) is **LARGER** than constant velocity's (**1.491e-09**) — the canonical seed
is *further from its own goal than doing nothing is*, on **20 of 25** windows. That is the
seed/goal mismatch, measured in the cost.

### 3.2 Controls — every one read its known value

| control | requirement | MEASURED | pass |
|---|---|---|---|
| **K1** identity arms | turn `c_goal` == 0 to ≤ 4·spacing(1f) = 4.768e-07 | `identity_full` **3.331e-16**, `identity_plan` **2.220e-16** (f64) | ✅ |
| **K2** zero-goal stratum | a window whose canonical control is all-zero has goal rollout ≡ cv rollout ⇒ `cv_c_goal` EXACTLY 0 | **4.441e-16** max abs over the stratum | ✅ |
| **K3** `C_REG` | must **NOT** read identity | max abs turn `c_goal` **1.428e-06** ≫ tol ⇒ does not | ✅ |
| **K4** ULP | every fp32 difference carries `resolvable_in_fp32` | banked per arm and per window | ✅ |
| **K5** n and d | printed | §3.0 | ✅ |
| **K6** banked agreement | `shipped__A` reproduces the banked cost surface | **25 / 25 matched on (ep_name, t)**, **25 / 25 decoded token agrees**, max abs diff `turn_c_goal` **1.788e-07**, `cv_c_goal` **2.384e-07** — both inside the fp32 cosine resolution the banked numbers were themselves quantised to | ✅ |

⚠️ **K6 FAILED ON THE FIRST PASS AND THE FAILURE WAS MINE, NOT THE CODE'S — recorded because it is
a reusable trap.** I keyed the join on the window index `t` alone. The 25 curved windows take only
**six** distinct `t` values across four episodes, so a `t`-keyed dict silently compared a window with
a **different episode's** row and reported a 6.2e-06 disagreement — the same order as the values
themselves, i.e. a manufactured refutation of my own instrument. Matching on **(ep_name, t)** made
it agree to 1.8e-07. *Root-cause class: a join key that is not a key.* The corrected matcher and the
warning are now in `tools/e0_goalspace_probe.py`, and the re-analysis cost **0 GPU** because the
tool has a `--from-rows` path (the `t1_eval --analyze-only` lesson, applied in advance).

### 3.3 Q2 — by how much, and what weight would let it win

The advantage is real and it is **1.5 × 10⁵ times too small.**

| quantity (conv A, `identity_full`) | MEASURED |
|---|---|
| goal-term advantage, f64, median | **1.491 × 10⁻⁹** |
| the `0.05·κ²` charge the same turn pays | **2.240 × 10⁻⁴** |
| the `0.02·jerk²` charge (mean / max) | 5.921e-04 / 1.124e-02 |
| **turn total-cost wins under `1 − cos` at shipped weights** | **0 / 25 — on EVERY arm, both conventions** |
| the multiplier on **both** explicit weights that would tip the median window (`1 − cos`) | **3.328 × 10⁻⁷** ⇒ `w_kappa` would have to be **1.66 × 10⁻⁸** |

**⇒ under the shipped metric the answer is not a weight, it is a deletion of the penalty.** That is
the sizing §8 asked for, and it rules L4-alone out.

**The chord changes the arithmetic, and it is where the panel turns:**

| arm | chord leverage gain (median) | **turn total-cost wins under the chord, at the SHIPPED `w_kappa` = 0.05** |
|---|---|---|
| `shipped` | 12,477 × | **0 / 25** (A) · **0 / 25** (B) |
| `identity_full` | 36,616 × | **3 / 25** (A) · **3 / 25** (B) |
| **`identity_plan` (L0)** | 15,362 × | **5 / 25** (A) · **⭐ 25 / 25 (B)** |
| `C_REG` | 12,477 × | 2 / 25 (A) · 0 / 25 (B) |

⭐ **L0 + chord + the repaired units crossing (B) is the first configuration in this programme on
which the canonical turn beats constant velocity on the TOTAL cost.** *(MEASURED, f64,
`raw/e0_goalspace_ep2_rows.json`; independently recomputed from the raw rows.)*

⚠️ **The caveats travel with it, in the same breath:**
* **4 episodes, one token** (§3.0). All four episodes are all-win, but n = 4 clusters.
* **The margin is thin:** chord ÷ charge is **min 1.187, median 1.232**, max 3.813. Three of the
  four episodes sit at ~1.19–1.23×.
* **`S3` FAILS AS WRITTEN.** §7 S3 requires *"≥ 13/25 … fp64, **both conventions**"*. Convention A
  gives **5 / 25**. ⇒ S3's failure branch fires: *"the cost line is insufficient even repaired;
  escalate to the goal-SPACE branch (§10)"* — **but only under convention A**, and A is the path
  `refa_v1.py` itself documents as *"the LEGACY path and it is wrong by `arctan(L·kappa)`"*. The
  honest statement is that **S3 is convention-dependent**, which the pre-registration did not
  anticipate; I am not re-writing S3 after seeing the data.
* **My registered prediction (§7) — "S1 and S2 will hold; S3 will FAIL" — is UPHELD**, and the
  mechanism I named (float32 saturation) is *also* confirmed: the `identity_full` advantage
  (1.49e-09) is **below** the fp32 cosine resolution, and only **13 / 25** windows are
  `advantage > 0` when the same cosine is read in fp32 rather than f64.

### 3.4 §8.4 — the RAW-INPUT FLOOR, and the branch it fires

The same 25 windows with the goal replaced by the **ground-truth 6 s field**
(`model.adapter(model.std(future_feats))[:, op_steps-1]`, pooled through the model's own
`_tac_field` — the same pooling `plan(goal_field=...)` applies).

| arm | turn `c_goal` f64 (median) | advantage f64 (median) | advantage > 0 | wins (chord, shipped weights) |
|---|---|---|---|---|
| `oracle_floor` (goal-action candidate), conv A | **9.140e-03** | **−4.133e-07** | **3 / 25** | 0 / 25 |
| `oracle_floor`, conv B | — | −1.858e-06 | **1 / 25** | 0 / 25 |
| `oracle_floor_shipped_seed`, A and B | 9.184e-03 | −4.089e-06 | **0 / 25** | 0 / 25 |
| …restricted to the **9** windows where `gt_turn_deg ≥ 5°` | — | −3.963e-07 (A) | **3 / 9** (A) · **1 / 9** (B) | — |

**⇒ §8.4's committed branch, verbatim: *"If the oracle-goal cost also fails to prefer the turn, the
defect is the cost."* IT DOES FAIL TO PREFER THE TURN. The defect is the COST.**

Two facts make this reading sharp rather than a formality:

1. **The oracle terms are NOT float32-saturated** — they are **O(10⁻²)**, four orders of magnitude
   above the imagined-goal terms, and `resolvable_in_fp32` is **25 / 25**. So this is not the
   precision defect; it is a *geometry* defect. The turn and cv rollouts sit **~10⁻⁴ apart in
   relative terms** against a base distance of ~10⁻², which is the §10 statement
   (*"a curvature change moves the field by a relative displacement of ~5 × 10⁻⁴"*) measured against
   the real future rather than against another imagination.
2. ⚠️ **The honest limit of this floor, named rather than discovered later:** it compares the model's
   **imagination** (a tactical rollout) with an **adapter-encoded real field**. A systematic offset
   between those two spaces makes both candidates far away and their difference small, so the floor
   bounds *"can this cost rank a turn against the truth"* and **not** *"is the WM's turn correct"*.
   It is nonetheless the arm §8.4 pre-registered, and the arm the banked dump already carries.

### 3.5 What E0 settles, and what it does not

**Settled:**
* the goal SPACE is **not** the blocker (Q1: 25/25 positive at identity) — §10's branch does **not**
  fire;
* the seed/goal mismatch is real **in the cost**, not only in the arithmetic: as shipped the turn is
  further from its own goal than cv on 20/25;
* `1 − cos` at the shipped weights cannot rank a turn under **any** arm (0/25 everywhere);
* the **cost** is the defect (§8.4), and the chord + L0 + convention B is a candidate repair that
  ranks the turn at unchanged weights.

**NOT settled, and not claimed:**
* whether the repair survives outside `TURN_R × ADAPT_SPEED_FOR_CURVE` and outside 4 episodes —
  **UNVERIFIED**;
* whether it survives at T1 (this is **T0**; `turn_frac` is a property of the COST, never of the
  car);
* whether the WM's imagined turn is *correct* (§3.4 caveat 2) — **UNVERIFIED**;
* the four metric families (longitudinal / lateral / tactical / strategic) are **not** reported
  here because **no arm reaches G-DRIVE in this document** (§6d); this is a cost/WM diagnostic and
  it is not presented as a driving result.

---

## 4. Threats to validity, and how each was discharged

| # | threat | discharge |
|---|---|---|
| 1 | **A join key that is not a key** | §3.2 — measured, corrected, and the correction re-ran with 0 GPU |
| 2 | **An instrument that cannot see the defect** | the READ-ONLY `seed_goal_mismatch.py` re-run unmodified still reads 62/64 on the patched code; `C_REG` still fails K1; `test_a_*` pins 2/64 under the default |
| 3 | **fp32 differences below their own ULP** | every cost difference carries its ULP and `resolvable_in_fp32`; the f32 and f64 counts are both reported and they **differ** (13/25 vs 25/25) |
| 4 | **Mixed denominators** | the 25 curved-goal windows are never mixed with the 27/140 geometric stratum; the `gt_turn_deg ≥ 5°` sub-table is labelled as a sub-table with n = 9 |
| 5 | **A tiny-rig identity read as a capability** | §2.3 states the rig is random-init and that the identity is a property of the ACTIONS fed, not of the weights |
| 6 | **Overclaiming L0** | `test_h_the_repair_buys_identity_not_horizon` asserts the two goals differ; the source comment and this document both say "identity, not horizon" |
| 7 | **A default that silently moved** | two independent checks, §2.4 |
| 8 | **Nav echo** | not applicable: E0 fixes nav at `nav_true` and asks a question about the COST, not about the decoder's skill. The decoder's nav-echo obligation is discharged in the sibling package and is untouched here. |

---

## 5. Refusals recorded

* ⛔ Thor and every pod **not contacted**; no trainer edit shipped to any live run.
* ⛔ `plan_horizon_s` **not** changed — escalated as a design decision.
* ⛔ The five tools of `2026-09-03-tactical-decoder/` **not modified**; `seed_goal_mismatch.py` was
  re-run unmodified for the BEFORE number.
* ⛔ `taniteval/tools/cost_surface_probe.py` **not modified** (read-only; its banked output is the
  K6 gate).
* ⛔ The retired `participation ≥ 8.56` floor is not used, quoted, or reintroduced.
* ⛔ The `w_kappa`-scaled win counts are reported as a **SIZING** number (Q2) and never as evidence:
  at the median tipping scale roughly half the windows win **by construction of the median**. The
  decision numbers in this document are all at the **shipped** weights.

---

## 6. Provenance — what ran, and how to re-run it

**The tool that produced each artifact is byte-identical to the one committed here** (md5,
2026-09-03):

| file | md5 | note |
|---|---|---|
| `tools/e0_goalspace_probe.py` | `d5c631bd0100fdbf4a7483e883eb0e3d` | repo copy == the copy that ran |
| `tools/seed_goal_agreement_report.py` | `81d3423b190584dd91cf253ff0a2a6bb` | repo copy == the copy that ran |
| `…/2026-09-03-tactical-decoder/tools/seed_goal_mismatch.py` | `224bfc18bccad4409bc0ad60f4e7a6d3` | ⛔ **READ-ONLY, re-run UNMODIFIED** — the md5 of the copy that produced `raw/seed_goal_mismatch_AFTER_L0_default.json` is identical to the shipped tool's |

**Environment.** The G: mount cannot import `tanitad` (`Errno 22` mid-import), so everything RAN
from the mirror `C:\Users\Admin\tanitad-wt` with
`PYTHONPATH=C:/Users/Admin/tanitad-wt/stack;…/colab;…/taniteval`, venv
`C:\Users\Admin\venvs\tanitad\Scripts\python.exe`, `PYTHONIOENCODING=utf-8`, `OMP_NUM_THREADS=6`.
**Every EDIT was made in the repo first and staged immediately**; the mirror was refreshed from the
repo before each run, and each edit was verified by MARKER in the STAGED BLOB (`git cat-file`) as
well as in the worktree.

**Re-run:**

```bash
# E0 (40.8 s GPU) — see raw/run_e0.sh for the exact invocation
python tools/e0_goalspace_probe.py --ckpt <ckpt_ep2/ckpt.pt> --config <config.json> \
  --cache <fp8> --episodes <eps> --labels <s2_labels_v7.2_eval.jsonl.gz> --nav <same> \
  --name ep2 --arm-tool-dir <taniteval/tools> --episodes-n 20 --window-stride 10 --device cuda \
  --banked-cost-surface <…/2026-09-03-refav1-cost-surface/raw/cost_surface_ep2.json> \
  --out raw/e0_goalspace_ep2.json --raw-out raw/e0_goalspace_ep2_rows.json

# ⭐ re-ANALYSE with ZERO GPU (the compute is already paid for)
python tools/e0_goalspace_probe.py --from-rows raw/e0_goalspace_ep2_rows.json --name ep2 \
  --banked-cost-surface <…> --out raw/e0_goalspace_ep2.json

# L0's before/after (0 GPU, seconds)
python <…/2026-09-03-tactical-decoder/tools/seed_goal_mismatch.py> \
  --out raw/seed_goal_mismatch_AFTER_L0_default.json          # must still read 62/64
python tools/seed_goal_agreement_report.py --out raw/seed_goal_agreement.json

# L1's banked rows (0 GPU, seconds)
python tools/make_row_evidence.py <out dir> [<path to refa_v1_train.py>]
```
