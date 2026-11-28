# SPEC — L0 (seed = goal), L1 (log the invisible loss term), E0 (the refutation arm)

**Date:** 2026-09-03 (Europe/Berlin) · **FlyWheel:** Architecture & Inference ·
**Branch:** `agent/arch-inf-20260803` · **Written BEFORE any edit and before any compute.**
**Governing pre-registration:** `Project Steering/PREREG_TACTICAL_DECODER.md` (L0 = §5 L0 / S1;
L1 = §5 L1, a preflight condition at §3; E0 = §8).
**Diagnosis this rests on:**
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tactical-decoder/RESULT.md`.

> ⛔ **Thor is running `refav1-b1-v72-ep3-speed` to ≈ 2026-09-04 02:00Z and is NOT contacted.**
> Every trainer/model edit here is for the NEXT launch. The live run keeps its launch line.
> The dev-box 4060 is forward-only for refav1 (`step-profile` §2: no batch fits for fwd+bwd).

---

## 0. Baseline, taken before the first edit

`pytest stack/tests -q -k "refa or refav1 or tac or seed_goal"` in the run mirror
(`C:\Users\Admin\tanitad-wt`, venv `C:\Users\Admin\venvs\tanitad`):
**1 failed, 711 passed, 5053 deselected, 91.07 s.**

The single pre-existing failure is named here so it cannot be confused with anything I do:
`stack/tests/test_decision_check.py::test_refa_surfaces_the_measured_arm` — it searches
`Project Steering/{DECISIONS_2026-07-20,MODEL_REGISTRY,RETRACTION_LOG}.md`, and the run mirror
carries only `Project Steering/BACKLOG.md`. It is a **mirror-environment artifact**, not a code
failure, and it is unrelated to every file this package touches.

---

## 1. L1 — the loss term that has never been logged

### What it will show

`refa_v1.py:1592` computes `out["loss_lat_label"]` / `out["loss_lon_label"]` and folds them in at
`:1595` with `w_tac_label`; `:1609` computes `out["loss_route_label"]` and folds it at `:1610` with
`w_str_label`. `refa_v1_train.py`'s log row (`:699-734`) carries **no** such key — verified here by
two differently-bound probes (POSIX `grep -c` over the repo path → 0 with exit 1; a Python
`open(...).read().count()` on the absolute path → 0 for each of `loss_lat_label`,
`loss_lon_label`, `loss_route_label`, `loss_feat_str_ext`). ⇒ the term is absent from all nine
banked `train_log.jsonl`.

**The change:** four keys in the trainer's row —

| key | value | when the model did not emit the term |
|---|---|---|
| `loss_lat_label` | `float(out["loss_lat_label"])` | **`None`** |
| `loss_lon_label` | `float(out["loss_lon_label"])` | **`None`** |
| `loss_route_label` | `float(out["loss_route_label"])` | **`None`** |
| `loss_label_weighted_share` | the **weighted** label supervision as a fraction of `loss` | **`None`** when no label term was emitted at all |

⛔ **`None`, never `0.0`.** A `0.0` in this column reads as *"supervised, and perfect"* — the exact
class of true-but-wrong-for-the-reader defect the operating standard names. An unlabelled step and
a perfectly-classified step must not print the same character.

**The share is defined to match the model's own folding EXACTLY, so it is hand-computable:**

```
w_tac_label * mean(present of {loss_lat_label, loss_lon_label})   # refa_v1.py:1593-1596
  + w_str_label * loss_route_label                               # refa_v1.py:1610-1611
--------------------------------------------------------------   / loss
```

### What would refute it

* any EXISTING key in the row changing its value or its type (the load-bearing regression);
* a `0.0` appearing where the term was not emitted;
* the share disagreeing with the hand computation by more than `1e-9` relative;
* the trainer failing to run at all with the labelled path off.

### Test — `stack/tests/test_tac_loss_logging.py`

1. **labelled path ON:** a smoke-shaped step emits all four keys, all floats, share in `(0, 1)`.
2. **labelled path OFF:** all four keys present and **exactly `None`** (asserted with `is None`,
   not `== 0`), and `0.0` explicitly rejected.
3. **the hand computation:** the share recomputed from the row's own three losses, `w_tac_label`,
   `w_str_label` and `loss` matches to `1e-9` relative.
4. **THE LOAD-BEARING REGRESSION:** every key the row carried before this change is still present,
   still the same type, and (for a fixed seed and a fixed step) still the same value — asserted by
   building the row twice, once through a reference construction that omits the new keys.
5. **all-ignored labels** (`-100` everywhere) → the model emits nothing → the row reads `None`,
   which is the case that distinguishes "no label" from "loss 0".

---

## 2. L0 — make the planner's canonical seed reproduce its own goal

### The diagnosis, to be quoted with file:line in RESULT.md

* **the goal's control:** `refa_v1.py:1757` builds `ctrl = canonical_controls(lat, lon, v0,
  cfg.op_steps=30, cfg.op_dt=0.2)`; `:1761` feeds the tactical predictor
  `self._model_actions(ctrl, v0, units)[:, ::stride][:, :cfg.tac_steps]` with `stride = 3`
  ⇒ **operative indices `[0,3,6,…,27]`** (6.0 s).
* **the seed:** `refa_v1.py:1932` `seed = goal_action["controls"][:cfg.plan_steps][None]` with
  `plan_steps = round(plan_horizon_s / op_dt) = round(2.0/0.2) = 10` (`:275`, `:575-576`)
  ⇒ the seed holds **operative indices `[0..9]` only** (2.0 s), and the cost then re-grids it at
  `:1963` / `:1969-1970`: `"dense"` (the DEFAULT, `:1846`) gives tactical step `j ← seed[j]`
  ⇒ `[0..9]`; `"tactical"` gives `j ← seed[min(3j, 9)]` ⇒ `[0,3,6,9,9,9,9,9,9,9]`.

**⇒ the goal needs operative actions at 12, 15, 18, 21, 24, 27 — all ≥ `plan_steps`. They are not
in the seed at all.** No re-gridding of a 10-element sequence can produce them.

### The three candidate repairs, and which one I will take

| # | repair | verdict |
|---|---|---|
| R1 | `plan_horizon_s` 2.0 → 6.0 so the seed spans the goal | ⛔ **NOT MINE.** It changes the optimised window, the search dimensionality, the proposal head's output shape (`refa_v1.py:1040`, `proposal_k * plan_steps * a_dim`), the length of the returned plan, and every banked comparison. A DESIGN decision for the PI / Master Mind. |
| R2 | **pack** the goal's stride-3 actions into the 10 seed slots (`seed = ctrl[::3][:10]`) | ⛔ **REJECTED, and the reason is recorded so nobody re-proposes it.** It yields 64/64 under `"dense"` only, and it makes the seed a **3× time-compressed control**: the plan is executed on the OPERATIVE grid at `op_dt` (`pc = PlanConfig(horizon=cfg.plan_steps, dt=cfg.op_dt)`, `:1872`) and the coarse→fine re-score rolls `self.operative` over the same tensor with no regrid, so a 6 s manoeuvre would be driven in 2 s. Under `"tactical"` it cannot work at all: `tac_idx` takes only 4 distinct values and cannot address 10 distinct goal actions. |
| R3 | **the honest alternative the brief names:** build the goal over the **plan's own horizon**, flag-gated, default OFF | ⭐ **TAKEN.** |

### R3 — exactly what is implemented

A new `plan()` keyword `goal_time_grid`, mirroring the existing `cost_time_grid` machinery
(module tuple `GOAL_TIME_GRIDS = ("full", "plan")` + a `_check_goal_time_grid`):

* **`"full"` — THE DEFAULT, and byte-identical to the shipped code.** No behavioural change; the
  goal stays the 6 s manoeuvre field, the mismatch stays at 62/64. Every banked number remains
  reproducible.
* **`"plan"` — the repair.** At the seed-construction site the goal is **re-rolled from the seed's
  OWN action feed** — the same `_model_actions(...)` call, the same `tac_idx` re-grid, the same
  predictor and the same `z0` the cost uses for a candidate — so the canonical seed's cost rollout
  and the goal rollout are **the same forward pass**. `1 − cos` is then `1 − cos(x, x)`, i.e. **0
  by construction**, on every token, in **both** `cost_time_grid` modes and at **both**
  `plan_level`s.

⚠️ **The honest cost of R3, stated before it is measured, not after.** Under `"plan"` the goal is
the manoeuvre's **first 2.0 s** (dense) or its first 2.0 s **held to 6 s** (tactical), NOT the 6 s
manoeuvre the token names. **R3 buys IDENTITY, it does not buy HORIZON.** Restoring the 6 s
manoeuvre requires R1. Any register row must say this, or it overclaims.

### What would refute L0

* the fix not reaching 64/64 on the token pairs, under either grid;
* the default path not still reading 62/64 (i.e. I changed the default by accident);
* the deliberate regression (`C-REG`: force the truncation back) **not** reproducing 62/64 — then
  the instrument, not the code, is what moved;
* the `TURN` heading excess not reading **0.0°** under the fix (tolerance: **exactly 0.0**, since
  both sides consume the identical `list` of floats — this is an identity, not an approximation;
  the tolerance on the model-level goal term is `≤ 4 × spacing(1f) = 2.4e-07`, the float32 cosine
  resolution near 1);
* any change to a value produced under the default flag.

### Test — `stack/tests/test_seed_goal_agreement.py`

Using **`seed_goal_mismatch.py`'s own metric** (the goal's tactical action sequence vs the seed's,
element for element, plus the integrated heading `Σ κ·v0·tac_dt`):

1. **BEFORE (default `goal_time_grid="full"`):** exactly **2/64** token pairs agree, under BOTH
   `cost_time_grid` modes; the agreeing pairs are exactly `(LANE_KEEP, CRUISE)` and
   `(ABORT_LC, CRUISE)`; worst heading excess **82.506°** on `TURN_L`.
2. **AFTER (`goal_time_grid="plan"`):** **64/64** agree, both grids; `TURN_L`/`TURN_R` heading
   excess **exactly 0.0°**.
3. **END TO END on a real `plan()` call** (the tiny-model fixture of
   `test_refa_v1_plan_goal.py`): with `goal_time_grid="plan"` the canonical seed's goal term is
   `≤ 2.4e-07`; with the default it is not forced to be, and for a curved token it is strictly
   larger. Both `cost_time_grid` modes, both `plan_level`s.
4. **DELIBERATE REGRESSION:** the default flag must still read 62/64 — the gate must be able to
   see the defect it exists to catch.

---

## 3. E0 — the cheapest discriminating arm (`PREREG_TACTICAL_DECODER.md` §8)

### The pre-registered specification, quoted

> **Population:** ep2's **25 windows whose imagined goal already carries curvature**.
> **Procedure:** re-score those 25 windows' named seeds, in **float64**, under the 2 × 2 × 2 cross
> of `{seed as shipped, seed = goal's control} × {1−cos, chord} × {w_kappa as shipped, w_kappa
> scaled}`, plus the `C-REG` regression arm.
> **Q1 — does the tactical 6 s goal field prefer a turn AT ALL, once the seed is the goal's own
> control?** If the goal-term advantage stays ≤ 0 on a majority of the 25 **even at seed/goal
> identity and in f64**, then the goal SPACE — not the decoder, not the metric, not the weights —
> is what cannot represent a manoeuvre, and **the whole decoder-then-cost line is REFUTED.**
> **Q2 — if it does prefer the turn, by how much,** and what `w_kappa` / `w_jerk` would let it win?
> **§8.4:** E0 must also report the RAW-INPUT FLOOR — the same 25 windows scored with the goal
> replaced by the *ground-truth* 6 s field. If the oracle-goal cost also fails to prefer the turn,
> the defect is the cost; if it succeeds, the defect is the imagined goal.

### How the identity arm is realised (two readings, both reported)

At exact seed/goal identity the turn's goal term is **0**, so the turn's goal-term advantage over
constant velocity **is** `cv_c_goal = 1 − cos(z_cv, z_goal)` in f64. Q1 is therefore the question
*"is the cv rollout distinguishable from the goal rollout at all?"* — which is precisely the
goal-space question, and it is not a tautology.

| arm | goal | candidate | what it answers |
|---|---|---|---|
| `shipped` | 6 s imagined | seed as shipped (`ctrl[:10]`) | reproduces the banked 8/25 (conv A) / 4/25 (conv B) — the C1-style agreement gate |
| `identity_full` | **6 s imagined (unchanged)** | the goal's OWN 10 tactical actions | **Q1 on the SHIPPED goal space** |
| `identity_plan` | rebuilt over the plan window (**L0 R3**) | seed as shipped | what L0 as implemented actually buys |
| `C-REG` | 6 s imagined | seed truncated **and** `tac_idx` saturating | the deliberate regression: must FAIL |
| `oracle_floor` | **ground-truth future field** (§8.4) | as `identity_full` | the raw-input floor: is the defect the cost, or the imagined goal? |

Every arm × `{conv A = "kappa", conv B = "steer"}` × `{1−cos, chord = √(2(1−cos))}` ×
`{w_kappa = 0.05 shipped, the scale that would tip it}`. All cosines evaluated in **float64** on
the float32 fields, with the fp32 ULP printed beside every difference.

### The committed branches (verbatim from the PREREG, reported beside the measurement)

* **REFUTED:** *"If the goal-term advantage stays ≤ 0 on a majority of the 25 even at seed/goal
  identity and in f64, then the goal SPACE — not the decoder, not the metric, not the weights — is
  what cannot represent a manoeuvre, and the whole decoder-then-cost line is REFUTED."*
  ⇒ §10 fires: both hypotheses become *irrelevant rather than wrong*, and the next
  pre-registration is about what the 6 s tactical query field encodes.
* **NOT REFUTED:** the advantage is positive on a majority ⇒ Q2 sizes L4 exactly, and S3
  (`≥ 13/25` total-cost wins under C3) becomes the next test.

**Majority** = `≥ 13` of the 25. **`≤ 0`** is read in f64 and reported alongside the fp32 ULP, so a
difference below its own resolution is named as unresolvable rather than counted as a sign.

### Controls (a failure VOIDS the panel)

| control | must read |
|---|---|
| **identity** | the identity arms' turn goal term is `0.0` to `≤ 2.4e-07` (fp32 cosine resolution) — asserted, not assumed |
| **C-REG** | reproduces the shipped mismatch and does **not** clear the identity assertion |
| **constant-only / cv** | the all-zero control's own goal term against a `LANE_KEEP × CRUISE` goal is `0.0` exactly (the goal rollout **is** the cv rollout — `cost_surface_probe.py`'s `zero_goal` stratum) |
| **banked agreement** | the `shipped` arm reproduces `turn_decomposition_A.json`'s per-window `turn_c_goal` / `cv_c_goal` on the same 25 windows |
| **n and d printed** | 25 windows of 140, 20 episodes, named per arm |
| **ULP** | every fp32 difference carries `resolvable_in_fp32` |
| **f32 vs f64** | both reported; a conclusion that exists only in f32 is not a conclusion |

### Splits

```yaml
splits:
  fit:  none — no parameter is fit
  val:  none
  test: ep2's 25 curved-goal windows of the 140 banked windows / 20 episodes, scored ONCE
```

---

## 4. Refusals recorded in advance

* ⛔ Thor and every pod are **not contacted**; nothing here is shipped to a live run.
* ⛔ `plan_horizon_s` is **not** changed — it is a design decision that is not this FlyWheel's.
* ⛔ The five tools of `2026-09-03-tactical-decoder/` are **read-only**; `seed_goal_mismatch.py` is
  re-run unmodified for the BEFORE number rather than reimplemented.
* ⛔ The retired `participation ≥ 8.56` floor is not used, quoted, or reintroduced.
* ⛔ No number is reported without its evidence class and its tier.
* Any quantity I cannot measure is marked **UNVERIFIED**, never estimated into a table.
