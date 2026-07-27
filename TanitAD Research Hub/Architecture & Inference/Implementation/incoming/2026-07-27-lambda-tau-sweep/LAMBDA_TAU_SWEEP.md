# E-H2 / E-H3 — the (λ, τ) prior-strength axis: instrument built and gated, **curve NOT measured**

**Stream:** Architecture & Inference implementation, 2026-07-27.
**Spec:** `…/Research/2026-07-27-hierarchy-prior-vs-constraint/HIERARCHY_PRIOR_RESEARCH.md` §5.3, §6.3, §6.4.
**Mode:** CPU only, dev box (`SAYED-PC`). **No pod was contacted. No GPU was used. No credential was read.**
**Author:** implementation subagent, under `Project Steering/AGENT_OPERATING_STANDARD.md`.

---

## HEADLINE — read the second line before the first

1. ⛔ **THE 42-CELL CURVE WAS NOT MEASURED, AND COULD NOT BE FROM THIS HOST.** The spec's
   *"one forward pass plus seconds of CPU per cell"* is right about the cost and wrong about the
   inputs: **that forward pass has never been run.** `refined_pre`, the three class logit vectors
   and `sel_score` exist in **no staged artifact** — §9.4 of the research file says so in advance,
   and I verified it key-by-key across every candidate dump in the repo. The dev box has neither the
   v4 checkpoint (it lives at `pod:/workspace/_v4gate/flagship-v4-fromscratch-30k/ckpt.pt`) nor the
   parity corpus, and pods were out of scope. **Tier: MEASURED-ABSENCE (two probes: tensor-key
   enumeration of all six candidate dumps, and a filesystem search for the checkpoint).**
2. ⭐ **What DID get done is the part that gates the curve, and it all passes.** The (λ, τ) knobs are
   exposed and **bit-identical at λ=1, τ=1 (`max_abs_diff = 0.0`)**; `λ=0` reproduces the pre-graft
   score exactly; all six committed bars recompute to the digit; the sweep harness is written and
   **validated in both directions on four fixtures**; the shadow branch is measured; the `base_rank`
   label is fixed at source. **One command on a box with the checkpoint produces the curve.**
3. ⭐⭐ **THE NAMED TRAP IS ANSWERED ON THE λ AXIS ALREADY, EXACTLY, WITHOUT THE FORWARD PASS — and
   the answer is that it does not bite there.** The v4 gate already emits
   `seam_norm_ratio_max`, and for the exact sweep checkpoint it is **0.1204** (produced) /
   **0.1208** (oracle). Below `seam_clamp = 1.0` that recorded value **is** the pre-clamp ratio, and
   the graft scales **exactly** linearly in λ. ⇒ the clamp does not bind until **λ > 8.31** and the
   shipped `seam_fail = 1.5` guard does not raise until **λ > 12.46**. **Zero of the 7 λ values in
   the pre-registered grid are clamp-bound at τ=1.** The trap lives on the **τ** axis instead.
   **Tier: MEASURED (ours), DECISION-GRADE for the λ axis at τ=1.**
4. ⛔ **A SECOND TRAP THE SPEC DID NOT NAME, AND IT IS SHARPER THAN THE CLAMP:** `seam_fail = 1.5` is
   checked **before** the clamp and **raises**. So on the sharpening branch a large prior does not
   saturate — **it crashes the head**. Any sweep run against the shipped config would have died at
   the first hard cell instead of returning a flat curve.
5. ⭐ **SHADOW BRANCH, MEASURED (881 windows / 40 clusters, paired episode-cluster bootstrap
   B=2000): there IS real headroom — 0.0737 m beyond "turn the goal off" — and none of the four
   deployable reliability signals we can compute reaches it.** Pre-registered verdict: **REFUTE**.
   But *"turning the goal off is free"* now needs a qualifier it did not have: **the produced goal is
   better than no goal on 43.1 % of windows**, so turning it off is free *on average* and lossy *per
   window*.
6. ⛔ **`base_rank` correction landed at source**, with a **second probe the first pass did not run**:
   881/881 rows have a tail that is **strictly increasing in anchor index**. New retraction class
   **C15** appended.

> **Headline tier:** the deliverable's headline is *"the instrument is built and gated; the curve is
> not measured"* — **MEASURED for every gate and for the shadow branch, MEASURED-ABSENCE for the
> blocker, and NOT MEASURED for the 42 cells.** Nothing in this file quotes a (λ, τ) cell.

---

## 0. PRE-REGISTRATION — written before any cell was scored

The grid is the spec's, unaltered: **λ ∈ {0, 0.25, 0.5, 1, 2, 4, 8} × τ ∈ {0.1, 0.25, 0.5, 1, 2, 4}**,
**q = 256 always**, **two clamp sheets** (`deployable` seam_clamp = 1.0, `diagnostic` clamp off).
Estimator for every arm: **paired episode-cluster bootstrap, B = 2000, unit = episode cluster**
(`taniteval/ci.py`). **`overlapping_holdout_se` is never used, anywhere in this stream.**

### 0.1 The optimum-locating rule, and what makes it return a FAILING value

Committed in code (`code/eh2_lambda_tau_sweep.py`, `locate_optimum`) before any cell was scored.
It has **four failing states and one passing state**:

| verdict | condition | class |
|---|---|---|
| **DEGENERATE** | the prior moves the pick nowhere on the grid (spread < 1e-9) | FAIL |
| **SATURATED** | the **argmin cell itself** is ≥99.9 % clamp-bound *and* numerically tied (1e-12) with a different λ at the same τ ⇒ the optimum is **not identifiable in λ** | FAIL |
| **UNPOWERED** | argmin ≠ the shipped cell but its paired CI vs the shipped cell **includes zero** | FAIL |
| **NO-INTERIOR** | the argmin sits on a grid **edge** (λ ∈ {0, 8} or τ ∈ {0.1, 4}) ⇒ the optimum is not bracketed | FAIL |
| **CONFIRM-INTERIOR** | argmin is interior **and** separated-better than the shipped λ=1/τ=1 cell | PASS |

The **interval on the optimum** is the **admissible set**: every cell whose paired Δ vs the argmin is
**not** separated. *A single argmin cell with no interval is not a located optimum* — the rule refuses
to report one.

⚠️ **Deliberate design choice, stated because it is the difference between a finding and an
artefact:** SATURATED fires on the **argmin cell**, not on the sheet. Saturation confined to the
high-λ tail is reported in the clamp audit and does **not** disqualify an optimum found below it. My
first implementation failed on this — it declared every realistic sheet SATURATED, because beyond the
clamp *all* λ cells at a given τ collapse to the same number. That is a true fact about the clamp and
a useless verdict rule, and the self-test caught it before it could be quoted.

### 0.2 Validation in BOTH directions — the fixtures, and what they prove

`code/eh2_lambda_tau_sweep.py --selftest` · raw: `raw/eh2_selftest.json` · **4/4 PASS**

| fixture | built to | verdict returned | argmin |
|---|---|---|---|
| `planted` | have an **interior** optimum by construction | ✅ **CONFIRM-INTERIOR** | λ=1.0, τ=0.25, admissible set of **10 cells** |
| `degenerate` | zero graft weights | ✅ **DEGENERATE** | — |
| `saturated` | grafts 8× with a 0.02 clamp | ✅ **SATURATED** *("argmin (0.25, 0.1) is 100 % clamp-bound and TIED with λ = [0.5, 1, 2, 4, 8]")* | — |
| `unpowered` | a prior scaled to 1.5e-4 | ✅ **UNPOWERED** *(CI [−0.0260, +0.0062] includes zero)* | — |

**What would make the rule return a FAILING value on the real data**, stated in advance:
**DEGENERATE** if the grafts were still at their ReZero zero-init (they are not — ‖W‖_F =
0.6457/0.6592/0.7260, INHERITED from `v5_hier.json::graft_mechanism_check`, and re-checked by the
cache builder before it will run); **SATURATED** if `seam_clamp` pins the argmin's λ row (§3 says
this cannot happen at τ=1 and is *expected* on the sharpening branch); **UNPOWERED** if the best
cell's paired CI vs λ=1/τ=1 includes zero at n=40 episodes; **NO-INTERIOR** if the argmin lands on
λ ∈ {0, 8} or τ ∈ {0.1, 4}. **Confirmed reachable: three of the four fired on fixtures.**

⚠️ **Two non-obvious things the fixture work taught, which apply to the real run:**

- **The graft consumes `log_softmax`, not `softmax`, so it carries a class-INDEPENDENT nuisance.**
  `Σ_j W[c,j]·L[j]` decomposes into a class term plus `ḡ·Σ_j W[c,j]`, and the row-sum term has
  **K× the variance** of the signal. Un-centred, my first fixture's prior was dominated by it and λ
  was monotonically harmful — a property of the fixture, not of prior strength. *(The real W was
  trained under this same parameterisation, so it has had the opportunity to absorb it; whether it
  did is a one-line check on the cache and is listed in §6.)*
- **λ and τ are NOT independent axes.** `‖log_softmax(z/τ)‖` grows asymptotically like `1/τ`, so τ
  carries a **gain** component on top of its **shape** component. The fixtures' response surface is a
  clean ridge along `λ/τ ≈ const`, which is exactly this. ⇒ **a τ effect on the real data cannot be
  read as "sharpness" without a norm-matched control**, and §6 adds one as a third sheet.

---

## 1. THE (λ, τ) EXPOSURE — 6 lines, and the gate that makes them safe

`stack/tanitad/models/flagship_v4.py`, exactly as §5.3 specified:

```python
tau, lam = self.cfg.graft_tau, self.cfg.graft_lambda       # E-H2 knobs
g_lat  = self.lat_to_anchor (lsm(lat_logits  / tau, dim=-1))
g_lon  = self.lon_to_anchor (lsm(lon_logits  / tau, dim=-1))
g_dist = self.dist_to_anchor(lsm(dist_logits / tau, dim=-1))
graft  = lam * (g_lat + g_lon + g_dist)
# ... norm clamp and fail-loud unchanged ...
```

plus two config fields (`graft_lambda: float = 1.0`, `graft_tau: float = 1.0`) and **three new
telemetry keys that exist solely to make the named trap visible**:
`seam_norm_ratio_preclamp_mean` and `seam_clamp_bound_frac` (the share of the batch for which λ has
already stopped mattering), beside the existing `seam_norm_ratio_preclamp_max`.

### 1.1 ⛔ GATE G1 — the λ=1, τ=1 bit-identity gate. **PASS.**

`raw/eh2_gate.json::G1_bit_identity` · also pinned as a unit test.

| check | result |
|---|---|
| λ=1, τ=1 reproduces the **shipped** graft arithmetic bit-for-bit | ✅ **True**, `max_abs_diff = 0.0` |
| λ=0 reproduces the **pre-graft** score bit-for-bit (`F_base_only` becomes a config, not a re-implementation) | ✅ **True** |

**The reference is not the new code compared to itself.** The pre-change expression
`W_m(log_softmax(logits_m))` summed and norm-clamped is **re-implemented inside the gate** and
compared with `torch.equal`. The knobs are provably no-ops at their defaults because `x / 1.0` and
`1.0 * x` are **exact in IEEE-754** — asserted, not assumed. The grafts are given deterministic
**non-zero** weights first; with zero-init grafts every λ is identical and a broken knob would pass.

Pinned in the suite as
`stack/tests/test_flagship_v4.py::test_lambda_one_tau_one_is_bit_identical_to_the_shipped_graft_path`,
alongside `…::test_graft_lambda_zero_removes_the_prior_exactly` and
`…::test_graft_tau_to_zero_makes_the_class_posterior_one_hot`.
**`cd stack && pytest -q` → 1123 passed, 7 skipped.**

⚠️ **One correction to the spec's τ→0 claim, found while writing its test.** §5.3 implies τ→0 gives
"a one-hot class posterior" and by extension the argmax class's own anchor column. **The posterior
does go one-hot; the graft does not converge to that column.** Because the map consumes
`log_softmax`, the non-argmax entries go to `−gap/τ`, so the graft's **direction converges while its
magnitude diverges like 1/τ**. The test pins the true property (cosine similarity between τ=1e-3 and
τ=1e-4 grafts > 0.999, norm ratio in [5, 20]) rather than the plausible-sounding one. **That
divergence is the mechanism by which the clamp — not λ — ends up deciding the τ axis.**

### 1.2 ⭐ `q`: verified ABSENT from the deployment path, then fenced. **GATE G3 PASS.**

The brief said *"delete `q`"*. **There was nothing to delete.** A scan of the selection path itself
(`FlagshipV4Head._factor_grafts` + `FlagshipV15Head.select`, extracted with `inspect.getsource` —
**not** a whole-file grep) finds **zero** occurrences of `topk` / `top_k` / `hierarchical_pick` /
`masked_fill` / `scatter_` / `-inf` / `argsort` / `[:q]`. `q` exists **only** in the E-V5-2
*measurement* harness (`v5_hierarchical_select.py::hierarchical_pick`). Executed at five
(λ, τ) settings including (8, 0.1), the selector scores **all** candidates, all finite, and
`sel_idx == sel_score.argmax(dim=1)` every time.

⚠️ **A whole-file scan gives a FALSE FAILURE here**, and did on my first run: `flagship_v15.py`
contains two `masked_fill` calls — both **goal-token dropout during training**
(`vt_band → VT_DROPPED`, `route → ROUTE_DROPPED`). They mask the *condition*, never the candidate
set. The gate now scans the right scope and records the two hits as accounted-for.

⇒ The deliverable is a **regression guard**, not a deletion:
`stack/tests/test_flagship_v4.py::test_the_selector_never_truncates_the_candidate_set`, which asserts
across five (λ, τ) settings that the full 256-wide score is finite and the pick is its flat argmax.
That is what stops `H_graft(q)` — a measurement arm worth **+0.21 … +5.82 m** — from creeping in.

---

## 2. GATE G2 — the committed bars, reproduced before anything else was allowed to run

`raw/eh2_gate.json::G2_committed_bars` · 881 windows / 40 episode clusters · **PASS**

| bar | committed | recomputed | |
|---|---:|---:|---|
| `produced\|F_flat` | 0.8563 | **0.8563** | ✅ |
| `produced\|O_oracle_in_fan` | 0.2505 | **0.2505** | ✅ |
| `produced\|F_base_only` | 0.8781 | **0.8781** | ✅ |
| `neutral\|F_flat` | 0.7620 | **0.7620** | ✅ |
| `oracle\|F_flat` | 0.6423 | **0.6423** | ✅ |
| `oracle\|F_base_only` | 0.6615 | **0.6615** | ✅ |

Paired `H_graft(64) − F_flat`, re-run end-to-end on the decision-grade estimator:
**+0.2059 [+0.1316, +0.2897]**, separated (committed: +0.2059 [+0.1290, +0.2975]).
Single-arm interval on `produced|F_flat`: **0.8563 [0.7220, 0.9984]**, episode-cluster bootstrap.
The `v5_hier_windows.pt` and `v5_v4_windows_reduced.pt` dumps are confirmed to be **the same 881
windows** (`allclose(produced|F_flat, ade_by_arm/A0_as_trained)`), which is what makes cross-artifact
arithmetic admissible at all.

⚠️ **Not restated as the hierarchical prior's value:** the **+0.0218 m** figure is
`F_flat − F_base_only` and bundles **two** priors. On the oracle surface it decomposes into
**graft ≈ 0.0092 m** and a **constant-velocity term ≈ 0.0100 m** (`0.6615 − 0.6523 − 0.6423`, both
endpoints of which are among the six bars re-verified above). **The hierarchy is worth about half.**
Tier: **DERIVED, PROVISIONAL** — cross-artifact arithmetic, no CI admissible, and the *produced*-
surface split is still **UNKNOWN**. E-H0b (one forward pass, same cache) settles it.

---

## 3. ⭐⭐ THE NAMED TRAP — answered exactly on the λ axis, and relocated to τ

`raw/eh2_gate.json::G5_clamp_reachability` · **PASS** · **MEASURED (ours), DECISION-GRADE**

The v4 gate **already emits the number this needed**, and it was sitting in a staged artifact for the
exact checkpoint the sweep targets (`flagship-v4-fromscratch-30k`, step 29999, ckpt_md5 `8771c1d9…`):

| surface | `seam_norm_ratio_max` | threshold | |
|---|---:|---:|---|
| produced | **0.1204** | 1.0 | pass |
| oracle | **0.1208** | 1.0 | pass |

The head records `min(pre_max, seam_clamp)`. **Because 0.1204 < 1.0, the recorded value *is* the
pre-clamp ratio** — there is no ambiguity to resolve. And the graft scales **exactly** linearly in λ.
Therefore, with no forward pass and no estimate:

| crossing (at τ = 1) | λ |
|---|---:|
| the clamp first binds (`λ·r > 1.0`) | **8.31** |
| the shipped `seam_fail = 1.5` would **RAISE** (`λ·r > 1.5`) | **12.46** |
| λ values in the pre-registered grid that are clamp-bound | **none (0 of 7)** |
| λ values in the grid the shipped guard would refuse | **none (0 of 7)** |

> ⭐ **The λ axis is clean across the entire pre-registered grid, and the two clamp sheets are
> expected to be IDENTICAL along τ = 1.** A flat λ axis in the eventual run would therefore be a
> finding about λ, **not** saturation — which is the exact discrimination the brief demanded and
> which is now available *before* the run rather than argued about after it.

**Where the trap does live, and it is not settled here.** `‖graft(τ)‖` grows asymptotically like
`1/τ`, so at τ = 0.1 the ratio is expected near **1.204** — **above `seam_clamp`, below
`seam_fail`** — i.e. the **sharpening branch** is where cells become clamp-bound. The exact factor
needs the class logits, which are not staged. **Tier: ESTIMATED**, and the sweep emits the true
per-cell value (`preclamp_ratio_{max,mean,p50,p95}`, `clamp_bound_frac`, `would_trip_seam_fail`,
`trip_frac_seam_fail`) so the estimate is checked rather than trusted.

### 3.1 ⛔ The trap the spec did not name: `seam_fail` **raises**, it does not saturate

`seam_fail = 1.5` is checked **before** the clamp is applied and throws `RuntimeError`. On the
sharpening branch — where §3 says the ratio *will* climb — a sweep run against the shipped config
does not produce a flat curve, it **dies**. Consequences, all now handled:

- the sweep harness runs with the guard **explicitly raised** and records
  `_seam_fail_raised_for_the_sweep: true` in every sheet;
- every cell reports `would_trip_seam_fail` / `trip_frac_seam_fail`, so **cells that are not
  reachable in the deployed head at all** are visibly separated from cells that merely clamp;
- the head's error message now names `graft_lambda` / `graft_tau` and says a prior-strength sweep
  must raise the guard deliberately.

---

## 4. ⭐ THE SHADOW BRANCH (E-H3) — measured, and it REFUTES at the pre-registered bar

`code/eh3_shadow_branch.py` · `raw/eh3_shadow_branch.json` · 881 windows / 40 clusters ·
paired episode-cluster bootstrap, **B = 2000** · **produced (deployable) surface**

### 4.1 The three rungs, and the ceiling on any fallback rule

| arm | `ade_0_2s` ↓ | paired vs neutral |
|---|---:|---|
| `R_produced_always` — the shipped model | **0.8563** [0.7220, 0.9984] | +0.0943 |
| `R_neutral_always` — *"turn the produced goal off"* | **0.7620** [0.6282, 0.9069] | — |
| ⭐ **`O_shadow` — per-window `min(produced, neutral)`** | **0.6883** | **−0.0737 [−0.0928, −0.0552]** ✅sep |

> ⭐ **A shadow branch has real headroom: 0.0737 m beyond the free win, separated.** Reading it
> against the goal channel as a whole: the produced→shadow move is worth **0.1680 m**, and the
> *oracle* goal is worth **0.2140 m**, so a perfect fallback rule would capture **78.5 %** of what a
> perfect goal is worth. **That is GoalFlow's league** (their predicted goal recovers 73 % of the
> oracle headroom) **and it is reachable without fixing the producer at all** — *if* a rule can find
> the right windows.

> ⚠️ **AND IT PUTS A QUALIFIER ON A CLAIM THE PROGRAM IS CURRENTLY ACTING ON.** *"The produced goal
> is worse than no goal, so turning it off is free"* is true **on average** and misleading **per
> window**: neutral wins on only **56.87 %** of windows, so **the produced goal is better on
> 43.13 %.** Turning it off is not free — it is a **−0.0943 m average paid for by discarding a
> 43 % win rate.** Tier: **MEASURED (ours), DECISION-GRADE.**

### 4.2 The deployable rules — four non-oracle signals, all evaluated leave-one-episode-out

Each rule is a threshold on a signal **available at inference with no ground truth**, with both
directions admissible. `canary_err` and `fan_err4` are **deliberately excluded**: they are scored
against ground truth and would make any rule an oracle in a deployable costume.
**LOEO**: the threshold is chosen on 39 episodes and applied to the 40th, so the number is deployable
rather than the in-sample ceiling (the failure Bar A's 0.4907 exists to warn about).

| signal | LOEO `ade` | paired vs **neutral** | paired vs **produced** | falls back on | threshold stability |
|---|---:|---|---|---:|---|
| **`v0` ego speed** | **0.7579** | **−0.0041 [−0.0137, +0.0042]** ✗not sep | −0.0984 ✅sep | 84.9 % | **40/40 folds identical** thr + direction |
| `imag_cost` of the deployed pick | 0.7692 | **+0.0072 [+0.0028, +0.0129]** ✅sep **WORSE** | −0.0871 ✅sep | 91.4 % | 3 thresholds, 33/7 direction split |
| `ctrv` inconsistency of the pick | 0.7646 | +0.0026 [−0.0010, +0.0071] ✗not sep | −0.0917 ✅sep | 93.5 % | 2 thresholds, 2/38 split |
| `imag_cost` spread over the fan | 0.7643 | +0.0023 [−0.0004, +0.0058] ✗not sep | −0.0920 ✅sep | 96.4 % | 2 thresholds, 37/3 split |

> ⛔ **VERDICT: REFUTE**, against the pre-registered bar (*"CONFIRM if any deployable rule beats the
> neutral 0.7620, paired and separated"*). The best rule captures **0.0041 of the 0.0737 m of
> available headroom — 5.6 %** — and is not separated. The world-model's own disagreement with the
> plan it chose is *separated **worse*** than always falling back.

**What this does and does not license.** It refutes **these four signals**, not the shadow branch:
the four that were reachable from staged artifacts are all **plan-quality** proxies, and none of them
is **goal-specific**. **GoalFlow's own rule is not among them** and is **not computable from anything
in the repo today** — it is a distance between the *conditioned* and *unconditioned* **trajectories**,
and no staged artifact holds the neutral branch's trajectory, only its per-window error.
`code/eh2_build_cache.py` dumps `pick_traj` for **both** goal modes; with that cache the literal
GoalFlow rule is **one entry in `load_staged()`** and nothing else changes.

⇒ **The honest recommendation to replace "turning it off is free":** keep the free win as the interim
default, and treat the shadow branch as **open with a measured 0.0737 m prize and one un-tested
candidate rule** — not as refuted. The stronger statement I was invited to make (*"the shadow
trajectory is strictly better than turning the goal off"*) is **PUBLISHED for GoalFlow and NOT
established for us**; what is established for us is that **the prize is real and our current
reliability signals do not find it.**

---

## 5. ⛔ THE BLOCKER, stated precisely enough to be removed in one command

### 5.1 What is missing, verified key-by-key

The sweep needs, per window: `refined_pre [W,256]`, `lat/lon/dist` logits, the three graft weight
matrices, the assembled longitudinal term, per-candidate `fan_err`, and `ep`.
**Enumerated across every candidate dump in the repo:**

| artifact | has | lacks |
|---|---|---|
| `…/v5-imagination-selection/raw/v5_hier_windows.pt` (100 kB) | per-window ADE for 26 arm × goal-mode combinations | **every score ingredient** |
| `…/v5-imagination-selection/raw/v5_v4_windows_reduced.pt` (14.5 MB) | `fan_err4`, `ep`, `t`, `v0`, per-candidate imagination costs, picks | `refined_pre`, logits, `sel_score`, graft weights |
| `…/bar-a-selector/raw/bar_a_{produced,oracle}_windows.pt` | `traj_ce`, `traj_regret`, `tgt`, per-window ADEs | same |
| `…/v4-restart-lever/rescued_perwindow/windows_…-30k-{produced,oracle}.pt` | `pred`, `gt`, `cv`, `pred_dense` | same |
| `…/v4-30k-gate/coprimary/corridor_*.pt` | closed-loop corridor tensors | same |

The graft **weights** exist only as summary statistics (`v5_hier.json::graft_mechanism_check`:
Frobenius norms and max-abs, not the matrices). **Tier: MEASURED-ABSENCE, two independent probes** —
tensor-key enumeration of all six dumps, and a filesystem search for any v4 checkpoint (none on this
host; the only local `.pt` models are two Phase-0 pipecheck artifacts).

### 5.2 Why it was not run here, and what to run instead

- **dev box** — no v4 checkpoint, and its episode cache is keyed `14231cd29c74`, **not** the parity
  key `e438721ae894`. Running the pass here would silently change the window set and **no committed
  bar would reproduce**, which is precisely the parity violation the program refuses.
- **pods** — explicitly out of scope for this stream (pod1 training, pod2 owed-controls, pod3
  classifier, eval pseudo-simulation). ⭐ **The build adds no training load: one `torch.no_grad` pass
  over 881 windows, no imagination** — E-V5-2 measured **165.9 s / 159.8 s** per goal mode for the
  same pass *with* imagination, so this is **~2–3 minutes per goal mode**. It must not run on a pod
  that is training; it is an eval-host job.

```bash
#  ONE forward pass, wherever the v4 ckpt and the PARITY corpus live (NOT the dev box):
PYTHONPATH=/workspace/TanitAD/stack \
python eh2_build_cache.py --out /workspace/_eh2/eh2_cache.pt      # produced + neutral

#  then, on ANY CPU — seconds per cell:
python eh2_lambda_tau_sweep.py --cache eh2_cache.pt --out eh2_sweep.json
python eh3_shadow_branch.py   --cache eh2_cache.pt --out eh3_shadow.json   # + GoalFlow's own rule
```

The sweep **refuses to quote a cell** unless two gates pass first: its **self-test** (§0.2) and a
**fidelity gate** asserting that the CPU re-scorer reproduces the forward pass's own `sel_score` and
`sel_idx` exactly. A cache from a non-parity corpus fails the second one loudly.

### 5.3 What the same cache unblocks for free

`eh2_build_cache.py` also dumps `prior [W,256]` and `sel_score [W,256]` — the two keys §9.4 of the
research file names as the reason `O_graft(q)` and `H_rand(q)` were undecidable. **E-H0b** (the
3-arm graft/gate decomposition that turns the PROVISIONAL 0.0092 m into a CI) is the same pass with
`graft_lambda ∈ {0,1}`, now that λ=0 is a config rather than a re-implementation.

---

## 6. THREE AMENDMENTS TO THE SPEC'S §6.3, each earned by something measured here

1. **Add a third sheet: `diagnostic_norm_matched`.** τ carries a **gain** component as well as a
   **shape** component (§0.2), so a τ effect on the raw grid **cannot** be read as "sharpness". The
   spec's `REFUTE-SHARPNESS` / `CONFIRM-SHARPNESS` discriminators (§6.3) are **not identified**
   without renormalising the graft to a fixed norm before the clamp. Without this, "τ → 0.1 at λ = 1
   degrades" is ambiguous between *commitment sharpness costs* and *a 10× louder prior costs*.
2. **Check whether the trained `W` is row-centred across the class axis.** The `log_softmax`
   parameterisation carries a class-independent `Σ_j W[c,j]` nuisance with **K×** the signal's
   variance. A one-line check on the cache (`W.mean(dim=1)` vs `W.std()`) says whether training
   absorbed it — and if it did not, part of what the sweep will attribute to "prior strength" is that
   nuisance getting louder.
3. **Expect the deployable and diagnostic sheets to coincide along τ = 1** (§3). If they do not, the
   cache is wrong, not the finding.

---

## 7. ⛔ THE `base_rank` CORRECTION — fixed at source, with a second probe

`raw/eh2_gate.json::G4_base_rank_semantics` · **PASS**

| check | result |
|---|---|
| rows matching `[as-trained pick] ++ [anchor index order]` | **881 / 881** |
| ⭐ rows whose tail is **strictly increasing anchor index** *(a probe the first pass did not run)* | **881 / 881** |
| column 0 reproduces `F_flat` | **0.8563** ✅ |

⇒ columns 1.. carry **no score information whatsoever**. **Fixed in three places, all at source:**

- `…/2026-07-26-v5-imagination-selection/code/v5_cost_curve.py` — the misleading comment rewritten,
  the variable renamed `nested_order`, the use-site annotated, and the dump given a `nested_order`
  key plus a `_base_rank_IS` docstring key. **The legacy `base_rank` key is kept** so already-staged
  `.pt` files stay readable and E-H1's harness is unaffected.
- `…/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` §5.2 — a correction block that
  states what the `n` axis actually varies **and explicitly preserves the conclusions**: the finding
  holds for *any nested family* of candidate sets, index order is still a nested family, so
  *"breadth costs −10.66 m"* and *"spend none of the imagination budget"* stand as written.
- `Project Steering/RETRACTION_LOG.md` — new class **C15: a tensor's semantics taken from its NAME
  rather than from its construction site**, with the standing consequence and the reason the wrong
  reading survived: **column 0 genuinely is the deployed pick**, so the one consistency check that
  was run (`gather(base_rank[:, :1]) → 0.8563`) passes for a reason unrelated to the claim.

---

## 8. WHAT I COULD NOT DO, plainly

- **The 42-cell (λ, τ) curve is NOT measured.** No interior optimum is located, no interval is
  reported, no clamp-bound count is measured (only the λ-axis reachability **derived exactly** from a
  measured anchor, §3). Nothing in this file quotes a cell.
- **GoalFlow's literal shadow rule is NOT measured** — the neutral branch's trajectory is not staged.
  What is measured is the **bound any such rule is scored against (0.6883)** and four alternative
  non-oracle signals, all of which fail.
- **E-H0b is NOT run** — the graft-alone term stays **DERIVED / PROVISIONAL** at ≈ 0.0092 m with no
  CI, and the produced-surface split stays **UNKNOWN**.
- ⚠️ **Every interval here is n = 40 episode clusters and is UNPOWERED, not refuted, at a null**
  (`MODEL_REGISTRY §1.2a`: half-widths shrink ×2.8–3.9 at n = 600). The shadow-branch REFUTE is a
  refutation of *those four signals at this power*, and any winning cell in the eventual sweep must
  be re-run at n = 600 before it steers a GPU-day.

---

## 9. DELIVERABLE MANIFEST

**STAGED, NEVER COMMITTED, NEVER PUSHED.** Paths relative to
`G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/`.

| artifact | location | exists elsewhere? | note |
|---|---|---|---|
| `LAMBDA_TAU_SWEEP.md` (this file) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-27-lambda-tau-sweep/` | **repo only** | |
| `code/eh2_lambda_tau_sweep.py` | same dir | **repo only** | the 42-cell sweep + the 4-fixture self-test; runs anywhere with a CPU |
| `code/eh2_build_cache.py` | same dir | **repo only** | ⛔ **THE BLOCKER'S REMEDY — never yet executed.** Needs the v4 ckpt + parity corpus |
| `code/eh2_gate.py` | same dir | **repo only** | G1–G5; reruns in ~40 s on the dev box |
| `code/eh3_shadow_branch.py` | same dir | **repo only** | the shadow branch, staged artifacts only |
| `raw/eh2_gate.json` | same dir | **repo only** | G1–G5 raw output, ALL_PASS |
| `raw/eh2_selftest.json` | same dir | **repo only** | the both-directions validation, 4/4 |
| `raw/eh3_shadow_branch.json` | same dir | **repo only** | the shadow-branch measurement |
| `stack/tanitad/models/flagship_v4.py` | `repo:stack/` | **repo only** | (λ, τ) + clamp telemetry; bit-identical at defaults |
| `stack/tests/test_flagship_v4.py` | `repo:stack/` | **repo only** | 4 new tests incl. the bit-identity gate and the no-truncation guard |
| `…/2026-07-26-v5-imagination-selection/code/v5_cost_curve.py` | `repo:` | **repo only** | ⚠️ **another stream's file** — label fix only, no behaviour change |
| `…/2026-07-26-v5-imagination-selection/V5_IMAGINATION_SELECTION.md` | `repo:` | **repo only** | ⚠️ **another stream's file** — §5.2 correction block |
| `Project Steering/RETRACTION_LOG.md` | `repo:` | **repo only** | append-only: class C15 + entry + standing consequence |

⚠️ **Everything this stream produced exists in exactly ONE place — the repo working tree — and is
staged.** Nothing lives on a pod: **no pod was contacted.**
**Reproducing everything measured here:** two staged tensors, two staged JSON diagnostics,
`taniteval/ci.py` and `stack/`. No GPU, no pod, no network.

### 9.1 ESCALATIONS — these must not sit in a file

1. ⛔ **The (λ, τ) curve is one forward pass away and that pass has an owner-shaped hole.**
   `eh2_build_cache.py` is written, gated and never executed. It needs **an eval host with the v4
   checkpoint and the parity corpus for ~5 minutes** — not a training pod. **This is the single
   action that converts this stream from an instrument into a result**, and it also discharges
   E-H0b and the deferred `O_graft(q)` / `H_rand(q)` arms in the same pass.
2. ⚠️ **"Turning the produced goal off is free" needs its qualifier to travel.** It costs a **43.1 %
   per-window win rate**, and a perfect fallback would be worth **another 0.0737 m [−0.0928,
   −0.0552]**, separated. Any brief or registry line quoting **−0.0943 m** as the whole answer is
   incomplete.
3. ⚠️ **`seam_fail = 1.5` RAISES on the sharpening branch.** Anyone sweeping prior strength — or
   raising `graft_lambda` in a training config — must raise it deliberately and record that they did.
   It is not a saturation, it is a crash.
4. ⚠️ **The +0.0218 m figure is still circulating as "the hierarchical prior".** It is
   **graft ≈ 0.0092 + constant-velocity ≈ 0.0100**, DERIVED/PROVISIONAL, oracle-surface only.
   `MODEL_REGISTRY.md` does **not** currently quote it (checked), so the fix is to keep it out.
5. ⭐ **New retraction class C15 is appended and applies program-wide**: every reduced `.pt` in the
   program should carry a `_<key>_IS` docstring key beside any non-obvious tensor.
