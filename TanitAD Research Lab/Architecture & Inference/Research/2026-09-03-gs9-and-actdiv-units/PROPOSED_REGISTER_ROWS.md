<title>Proposed register rows — GS-9 transition probe + actdiv units invariance (2026-09-03)</title>

# PROPOSED REGISTER ROWS for `Project Steering/GOALS_AND_CLAIMS.md`

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Proposed text, NOT applied. Status stays PROPOSED until the PI / Master Mind reads it —`
`the precedent is D-P2-LEAK-AUDIT and the sibling actdiv package.`

MARKER: `E-AI-GS9UNITS-0903-ROWS`

⚠️ **Ownership note.** `CLAUDE.md` binds *"any session that asserts, supports, or refutes a claim
updates `GOALS_AND_CLAIMS.md` IN THE SAME TURN"*, while this task's brief scopes me to read-only
outside my own package. The rows are therefore proposed here verbatim-ready and the conflict is
escalated, not resolved unilaterally.

**Probed before writing (two differently-bound mechanisms — `git grep -a` and
`Select-String -LiteralPath`, which agree):** `H-GS9-1` **0 hits** and
`D-ACTDIV-UNITS-INVARIANCE` **0 hits** in `GOALS_AND_CLAIMS.md` — both rows are NEW.
`H-GS8-1` (2), `D-ACTDIV-ANCHORED-REFAV1` (4) and `k8clip05p30k` (2) exist.
⚠️ `MODEL_REGISTRY.md` still has **NO row for `k8clip05p30k`** — reported again, not papered over;
its model facts here come from the k8 package's raw JSON and from the checkpoint's own md5.

---

## ROW 1 — `H-GS9-1` (NEW). Verdict: **L3-TRANSITION-FAIL + ENC-TRANSITION-ABSENT + ACTION-ECHO-ONLY**

🔬 **`H-GS9-1` — THE BANKED GS-9 TRANSITION PROBE IS READ AT LAST, AND IT IS THE FIRST ADMISSIBLE
READ OF THE QUESTION `H-LEAK-2` WITHDREW EVERY L3 t-STATISTIC FOR — IT READS **FAIL** (MEASURED
2026-09-03, dev-box CPU, 0 GPU, T0-DIAGNOSTIC; raw
`Research/2026-09-03-anchored-actdiv-and-transition-probe/raw/transition_probe_v7.json`, read-out
and audit `Research/2026-09-03-gs9-and-actdiv-units/RESULT.md` §§1–3, new control
`…/raw/gs9_vleak.json`).** The probe predicts the one-tick ego-state delta
(`dx_fwd, dy_left, dyaw, dv`, all from `poses`) from the predictor's latent displacement
`Δẑ = ẑ_{t+1} − z_t`, against the encoder-only `Δz`, a raw-pixel floor and the fed action, on
**129 held-out clips / n = 11,868 rows**, split **BY CLIP** (5-fold outer × 5-fold inner), skill
against the fit-split mean, clip-cluster bootstrap intervals. The readings R1–R4 were
**pre-registered by the sibling SPEC §7 before the run**; this row applies them unchanged.

* **R1 — ENC-TRANSITION-ABSENT (both arms).** `dz_enc` reads **−0.0000 … +0.0005** on all four
  targets. It does not merely fail to beat the pixel floor: **it reads the constant's own value.**
  Its Pearson r equals the constant's to three decimals, i.e. the ridge shrinks to the mean.
* **R2 — L3-TRANSITION-FAIL (both arms), 1 of 4 targets against a bar of 2.** `z_t+dzhat_true` −
  `z_t` = **+0.0748 [+0.018, +0.133]** (`postrain30k`) and **+0.0752 [+0.051, +0.102]**
  (`k8clip05p30k`) on `dx_fwd` only; `dyaw` fails the CI (`postrain30k`) or the 0.02 effect floor
  (`k8clip05p30k`, +0.0178); `dy_left` and `dv` are negative.
* **R3 — the action reaches the transition ON `dv` ONLY, and the anchored response is a LOSSY
  ECHO.** `dzhat_true` − `dzhat_zero` = **+0.0433 [+0.026, +0.062]** / **+0.1119 [+0.081, +0.141]**
  on `dv`; on `dx_fwd` it is **NEGATIVE with the CI excluding zero** (−0.0076 / **−0.0515**) — the
  true action makes the prediction worse than a zero-action one. `dzhat_anch` is **below the fed
  action `act2` on every target of both arms** (18–93 % of it): the predictor's anchored action
  response carries strictly LESS about the transition than the two numbers it was handed.
* **R4 — SPLIT, and it localises the defect.** Action-carried cells are genuinely
  transition-specific (`dzhat_anch`→`dv` 93 %, →`dyaw` 91 %; `act2`→`dv` 90 %). Prediction-carried
  cells are not: `dzhat_true`→`dx_fwd` keeps only **17 %** after a within-clip target shuffle — 83 %
  of its headline 0.5703 is a **clip-level cue**.

⛔ **AUDIT FINDINGS THAT TRAVEL WITH THE ROW (this package's own contribution).**
**(a) `v` IS an input, at the model boundary.** `clip_features:521` calls
`_lift3(a2, v0, "steer_accel_v")`, whose third channel is `poses[rows, 3] / SPEED_SCALE` = `v_t/10`,
and the zero-action arm zeroes **only** steer/accel — so `dzhat_true` and `dzhat_zero` are outputs
of a predictor that was handed `v_t`. **MEASURED on the identical 11,868 rows through the identical
estimator (`raw/gs9_vleak.json`): `v_t` ALONE, one dimension, scores `dx_fwd` skill
0.9986 [0.9970, 0.9994], raw r = 0.9997.** The banked `dzhat_true` scores **0.5703** on that column from 2,048 dimensions
(residuals `√(1−0.5703)=0.655` vs `√(1−0.9986)=0.037`, i.e. **≈ 18× the residual error of a 1-d
readout of the raw input**) — and `v_t`'s own within-clip-shuffle reads 0.9010,
which is the same clip-level cue R4 caught. ⇒ **R2's single passing target is inadmissible as a
capability claim** (R2 fails the count independently). The leak is **surgical**: `r(v_t, ·)` ≤ 0.104
on the other three targets and `act2+v_t` does not beat `act2` there, so R3's `dv` result and
`dzhat_anch`'s `dyaw`/`dv` scores are clean of `v` and stand. Control validation: `act2` reproduces
the banked panel to four decimals on all four targets; `const` reads exactly 0.0000; every
global-shuffle ≤ |0.0006|.
**(b) The LINEAR raw-input floor is DEGENERATE.** `pixdelta`'s λ sat at the **grid ceiling 1e4 on
3 of 5 folds** and it reads the constant's value, so it discriminates nothing; every "beats the
floor" statement is re-stated as the stronger "**beats the CONSTANT**". The **RFF** floor is not
degenerate and is model-independent (identical on both arms, a hidden control at its known value).
**(c) ⭐ On the transition-specific column the RFF read shows RAW PIXELS carrying 5–200× more
structure than the learned `Δz`** (pixels +0.0193 `dx_fwd` / +0.0204 `dyaw`; `dz_enc` +0.0001–0.0041
/ +0.0015–0.0017). **Point estimates, NO interval** — the banked `mlp_rff` cells carry none — so
this is the second function class required by the linear-probe rule, agreeing with the linear
negative, **not** a pre-registered beat.
**(d) R4's own fallback CANNOT be executed from the raw.** It asks for a CI on `real − within`; the
JSON stores aggregates only, no per-row predictions. The point estimates (+0.0216 `postrain30k`,
+0.0106 `k8clip05p30k`) straddle the 0.02 floor and are reported **without a verdict**.

✅ **WHY THIS IS ADMISSIBLE WHERE L3 WAS NOT (`H-LEAK-2`).** L3's t's came from `envpred.loeo`
scoring a pooled 12-clip second half per iteration — **24 scores at ~92 % row overlap treated as
independent** — and were withdrawn wholesale; in that form a *constant* beat `z_t` at "t 4.7".
GS-9 instead (i) scores **every clip exactly once out-of-fold** under a clip-grouped 5-fold split
(no reuse to pseudo-replicate), (ii) intervals are a **clip-cluster bootstrap over the 129 clips**
with the estimator named on every CI, (iii) marginals are **paired on the same clip draws**, never
combined in quadrature, (iv) λ is chosen by **clip-grouped inner CV on the fit split only** and the
shuffled controls are scored **through the identical fits**, and (v) the metric is skill against
the **fit-split mean**, so **a constant reads exactly 0.0000 by construction and cannot beat
anything**. ⇒ **L3 now has an admissible read on ego-transition targets, and it FAILS.**
⭐ **AND `H-LEAK-2`'s OWN CORRECTED READ AGREES ON BOTH POINTS, from a different target and a
different estimator:** it reported that under true leave-one-clip-out with within-clip r *no column
separates from `z_t`* on `n_agents` (|t| ≤ 1.5, every K, three arms) **and that raw pixels DO
separate where the `ẑ` columns do not**. GS-9 reaches the same two conclusions on ego `Δ` targets
under a clip-grouped K-fold + clip-cluster bootstrap. Two instruments, different targets, different
estimators, different corpora — same two answers. That independent agreement is what makes the row
quotable; a probe agreeing with itself would be the `ls-tree` trap.

**Scope and function class (both binding on any quotation of this row):** two arms
(`postrain30k` md5 `a58585883c27…`; `k8clip05p30k` md5 `2d744d6d2faa…`, ⚠️ no registry row), **one
tick (k = 1)**, one corpus, **T0-DIAGNOSTIC — never driving performance**. The negative is about a
**ridge-linear map**, cross-checked against one RFF non-linearity that agrees; it is **not** a claim
that the information is absent from `Δẑ`. `o11p30k`, `splitp30k` and `postrain30k_freeze` are
**Thor-only and were NOT run** (Thor trains refav1 until ≈ 2026-09-04 01:00Z) — one CLI invocation
each when it frees. Four metric families: **REFUSED with the reason stamped** (T0 world-model probe;
no driving metric exists here to family-ise).

**What it does to `P5` and `P2`.** The launch gate's *"the predictor is not transporting the scene,
it is restating it"* now has an admissible transition-level measurement — and a sharper form: on
`dx_fwd` the predictor is **not even restating faithfully**, it attenuates a handed scalar to 57 %.
`P2` narrows in the same motion: the action **does** reach the latent transition (`dzhat_anch`
0.22–0.53 on `dyaw`/`dv`, 91–93 % of it transition-specific) but arrives **strictly degraded
relative to the two numbers fed** — never above `act2` on any target of either arm. So `P2` at the
transition level is a **transmission-loss** problem, not an absence-of-wiring problem — the same
conclusion GS-8's anchored read reached from the geometry side (`SEPARATED-NONMONOTONE`, gain
~10⁻³ of the scene).

⛔ **INTEGRATION (escalated, not filed in a README).** `taniteval/tools/transition_probe.py` is
wired into **nothing** — `mm_e19_read.py`'s actdiv stage still calls the banked `actdiv_local.py`.
It is the only instrument in the programme that answers L3's question on an admissible estimator
and it must enter the standard read-out path. **And `run_panel()` iterates a FIXED `FEATURE_ORDER`
tuple and silently skips any cell not in it** — a false-negative generator for anyone extending the
panel (it would have dropped the `v_t` control above without a word); it needs an explicit
unknown-feature refusal (2 lines, not my file).

---

## ROW 2 — `D-ACTDIV-UNITS-INVARIANCE` (NEW). Verdict: **INVARIANCE-PARTIAL**

📐 **`D-ACTDIV-UNITS-INVARIANCE` — THE "SEPARATION IS INVARIANT UNDER A MONOTONE REPARAMETRISATION,
SO NO NUMBER CHANGES" ARGUMENT WAS TESTED INSTEAD OF INHERITED, AND IT IS **HALF TRUE**:
`INVARIANCE-PARTIAL` (MEASURED 2026-09-03, dev-box CPU, 0 GPU, T0-DIAGNOSTIC, ~22 min per arm per
convention; raw `Research/2026-09-03-gs9-and-actdiv-units/raw/actdiv_units_{kappa,steer}.json`,
decision table `…/raw/units_tables.txt`, reading `…/RESULT.md` §4).** `actdiv_anchored.py
--family refav1` was run on **both** banked step-1,000 checkpoints **twice**, differing in exactly
one thing — the number placed on channel 1: `--action-units kappa` feeds the swept level unchanged
(the banked path), `--action-units steer` reads it as a TRUE CURVATURE and feeds
`arctan(L_enc·level)` through `tanitad.models.kinematic.as_command` with `L_enc =
STEER_WHEELBASE_M = 2.9`, the **ENCODING** constant (fed values `0.02 → 0.0579351`,
`0.05 → 0.1439964`, `0.1 → 0.2822574` rad; the flag is refav1-only and the tool **refuses** it on
`--family v7`, because that constant does not travel to ZOD / l2d / alpasim). This discharges item
(iii) of `D-ACTDIV-ANCHORED-REFAV1`'s own next-probe list and the instrument
`D-UNITS-CHECKPOINT-TEST-VOID` named.

✅ **B0 GATE — the raw pass reproduces the banked run on 51/51 statistics per arm at 6 significant
figures** (all three spaces' `F_sep` / `F_over_null_p95` / `scene_spread` / `rel_units` /
`rel_mag_at_material_level` / `max_abs_displacement`, each axis's `F_sep` / `F/p95` / `null.p95`,
the full-field `F_sep`, every `norm_by_level` and `sign_cos_by_level`, both controls, both action σ).
⇒ the flag is a **true pass-through**, the read is deterministic, and **the banked refav1 anchored
result is REPRODUCIBLE.**

✅ **INVARIANT (the argument is right here).** The **verdict** `LAT-INSENSITIVE-REFUTED` holds in
all three spaces on both checkpoints under both conventions. The **κ-axis separation ratio**
`F_sep/null p95` moves only **×0.965** (2297.75 → 2217.60, incumbent) and **×1.003** (11.755 →
11.787, clean epoch), inside the pre-registered ×1.25 band. **Spearman ρ(|level|, ‖m‖) = +1.000**
in every pass. The **material-magnitude** reading is effectively unchanged (0.375 → 0.377 %;
1.251 → 1.307 % of the scene spread, against the 5.95 % bar), because that aggregate is dominated by
the **untouched accel axis** — whose permutation null and `max_abs_displacement` are **bit-identical**
across conventions, an internal control at its known value. Every control reads its known value in
all four passes: C0 identity **0.0**, zero-model **0.0**, zero anchor `[0.0, 0.0]`, scene spread
4.249215 / 0.089146 identical across conventions.

⛔ **NOT INVARIANT (the argument is wrong here), with the banked numbers that MOVE named exactly.**
**(1)** `‖m(κ = 0.1)‖ / ‖m(a = 1.5)‖` — the ratio this register's own
`D-ACTDIV-ANCHORED-REFAV1` point (4) calls *"the ratio that IS comparable"* — moves
**0.00329 → 0.00935 (×2.843)** on the incumbent and **0.02514 → 0.06966 (×2.770)** on the clean
epoch, **in every space** (`tac` ×2.834 / ×2.756, `pooled` ×2.843 / ×2.769). ⇒ the same row's
*"the lateral channel moves the imagination only **1/300th** to **1/40th** as far as the
longitudinal one"* becomes **1/107th to 1/14th** under the curvature reading. **(2)** The
**linearity digits**: `1 : 2.500 : 5.005` and `1 : 2.498 : 4.980` become `1 : 2.490 : **4.911**` and
`1 : 2.467 : **4.768**` — so *"proportional to κ to three significant figures"* is
**convention-dependent**, and necessarily so: the level→fed gain varies **2.6 %** across the grid
(×2.8968 / ×2.8799 / ×2.8226), so a response linear in the FED value cannot also be linear in the
NOMINAL level. **(3)** The **σ-matching sentence** — *"the two axes are compared at matched σ, which
is why this ratio is meaningful"* — is **TRUE as a steer statement and FALSE as a curvature one**:
σ measured in the channel as fed is **σ_steer = 0.047565 rad** (⇒ σ_κ = 0.0164140 rad/m), so
level 0.1 is **2.102 σ_steer** in the raw pass but **5.934 σ_steer** when fed as a curvature, and
**0.1 rad/m is 6.092 σ_κ** against `a = 1.5 = 2.020 σ_a`. **(4)** The **antisymmetry weakens with
stimulus**: cos(m(+L), m(−L)) at the top level falls **−0.9956 → −0.9615** (incumbent) and
−0.9980 → −0.9856 (clean epoch) — between the row's committed HOLD bar (≤ −0.99) and its committed
FAIL bar (> −0.95), and reported as such rather than rounded.

⭐ **AND THE CONVERTED PASS MEASURED SOMETHING THE RAW ONE COULD NOT: the predictor's lateral path
is essentially a LINEAR MAP OF THE FED STEER ANGLE out to 0.2823 rad (16.2°) — a 4.87× extension of
the banked range — to within +0.7 % (incumbent, mildly super-linear) and −1.9 % (clean epoch, mildly
sub-linear), WITH NO SATURATION.** Measured gain vs the arctan gain per level: 2.8975/2.8968,
2.8853/2.8799, 2.8435/2.8226 (incumbent); 2.8935, 2.8572, 2.7705 (clean epoch). **This calibrates
the queued h = 10 cost-surface probe for free**: a sweep may use a **single measured lateral gain
per checkpoint** instead of a per-level grid, and `C-STEER-CURVATURE-INTERFACE`'s ×2.9 boundary
defect propagates into the goal term as a **clean ×2.8 factor on the field displacement** while the
`0.05·κ²` penalty is charged at the full proposed κ. ⇒ the three-factor compounding argued in
`D-ACTDIV-ANCHORED-REFAV1` is **correct in direction and its multiplier is now MEASURED
(×2.77–2.84), not assumed from the wheelbase.**

⭐ **The precise form in which the argument's intuition survives** — worth recording because it is
the rule for every future units question here: **at ~2 σ the arctan map is within 0.23 % of the
identity** (the level that is exactly `2 σ_κ` = 0.0328279 rad/m is fed as 0.094915 rad =
**1.9955 σ_steer**, against `2 σ_steer` = 0.09513). ⇒ **a σ-MATCHED comparison IS
convention-invariant; only a comparison pinned to a fixed NOMINAL level is not.** The banked ratio
is pinned to `κ = 0.1`, which is 2.1 σ in one reading and 6.1 σ in the other — which is exactly why
it moves.

**Speed scale, stamped as required.** Both checkpoints carry `cfg.speed_channel = False` (verified
from the primary artefacts: `ckpt_ep2/config.json`, and the incumbent's `ckpt['cfg']` as loaded),
so `a_dim 2 → a_in_dim 2` and `RefAV1.augment_actions` (`refa_v1.py:1263`) returns `(a, κ)`
unchanged. `SPEED_SCALE_MPS = 30.0` and it is **INERT**; all four passes printed
`SPEED_SCALE_MPS=30 (UNUSED: speed_channel=False …)` and banked `speed_scale_effective: null`. The
`v/30`-vs-`v/10` defect that invalidated a banked instrument family this week **cannot occur here**,
structurally — the tool never widens the action tensor. Population: the banked **140 windows**
(20 eval-slice episodes, `k_loader = 30`, stride 10), `--n-perm 200 --seed 0`, `h = 1`, verdict space
`tac`; md5 `45b9f4d82a3a…` / `c26c7ad00e5b…`. Four metric families **REFUSED with the reason
stamped** (T0 world-model probe). Tests: `stack/tests/test_actdiv_anchored.py` **42 passed**
(8 new pin the pass-through, including that `kappa` returns the candidate list **object-identical**
and that `steer` is **refused on `--family v7`**).

---

## ROW 3 — CORRECTION to the standing `D-ACTDIV-ANCHORED-REFAV1` row (should not wait)

⚠️ **AMENDMENT TO `D-ACTDIV-ANCHORED-REFAV1` (2026-09-03, MEASURED, `D-ACTDIV-UNITS-INVARIANCE`
above): ITS MAGNITUDE NUMBERS ARE STEER-CONVENTION NUMBERS AND MUST CARRY THAT STAMP.** The row's
point (4) — *"the ratio that IS comparable, ‖m(κ)‖/‖m(a)‖, moves the RIGHT way (0.00329 → 0.02514,
the clean epoch 7.6× more lateral-relative-to-longitudinal)"* — and its three-factor paragraph's
*"1/300th to 1/40th"* are **correct as measured** and remain the numbers to quote **when the swept
level is read as a steer angle**, which is what the channel is. Under the curvature convention the
same quantities are **0.00935 → 0.06966** and **1/107th to 1/14th**. ⚠️ The *cross-checkpoint*
comparison is unaffected — the clean epoch is **7.6×** more lateral-relative-to-longitudinal in the
raw convention and **7.5×** in the converted one, so point (4)'s conclusion stands. What must change
is only that each magnitude carries its convention, and that the row's supporting sentence *"the two
axes are compared at matched σ"* is qualified: it is exact in steer units (2.10 σ vs 2.02 σ) and
false in curvature units (6.09 σ vs 2.02 σ). The row's **verdict, controls, separation ratios,
Spearman ρ and material-magnitude reading are confirmed unchanged** by an independent re-run that
reproduced 51/51 statistics per arm at 6 significant figures.
