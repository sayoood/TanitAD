<title>SPEC — GS-9 read-out of a banked transition probe, and the actdiv units-invariance test (2026-09-03)</title>

# SPEC — E-AI-GS9UNITS-0903

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Tier: T0-DIAGNOSTIC on every reading below — world-model probes, never driving performance.`
`Device: dev-box CPU throughout (⛔ Thor trains refav1 until ≈ 2026-09-04 01:00Z; the A40 pod runs`
`refcv3 until ≈ 22:15Z — neither is contacted). The RTX 4060 carries no python compute app`
`(probed: nvidia-smi --query-compute-apps → desktop/browser processes only, 822 MiB, no python).`

MARKER: `E-AI-GS9UNITS-0903-SPEC`

---

## 0. Two tasks, and what "pre-registered" means for each

| task | backlog | what is pre-registered, and by whom |
|---|---|---|
| **A — GS-9 read-out** | R12 | ⚠️ **The readings R1–R4 were pre-registered by the SIBLING package** (`…/2026-09-03-anchored-actdiv-and-transition-probe/SPEC.md` §7, written 04:03 Berlin; the probe ran 04:19–04:50 and wrote `raw/transition_probe_v7.json` at 02:19Z). I **adopt them verbatim and amend nothing.** ⛔ **Full disclosure: I inspected the raw JSON before writing this file.** The readings are therefore *not mine to pre-register* — my contribution is the **admissibility audit** in §2 and the read-out itself, and both are stated here before any verdict is written into RESULT.md. Pretending otherwise would be the retraction-worthy move. |
| **B — units invariance** | R22 | **Genuinely pre-registered**: §4 below is committed BEFORE any refav1 forward pass exists under the new flag. The tool change (§3) is written and tested first; the readings are frozen here; then the four passes run. |

---

## 1. Task A — the question

`D-ACTDIV-ANCHORED-REFAV1`'s sibling banked a GS-9 **transition probe** and never read it: the
package's own `RESULT.md` §4 still says *pending*. GS-9 probes **transitions**, not states: predict
the one-tick ego-state delta (`dx_fwd`, `dy_left`, `dyaw`, `dv`, all from `poses`) from the
predictor's latent displacement `Δẑ = ẑ_{t+1} − z_t`, against the encoder-only `Δz = z_{t+1} − z_t`,
a raw-pixel floor and the fed action.

**Adopted readings (sibling SPEC §7, verbatim).** "A beats B" = paired clip-bootstrap 95 % CI of
skill(A) − skill(B) excludes 0 **AND** |Δskill| ≥ 0.02.

| read | quantity | holds ⇒ | does not hold ⇒ |
|---|---|---|---|
| **R1** | `dz_enc` beats `pixdelta` on `dx_fwd` and `dyaw` | ENC-TRANSITION-CARRIED | **ENC-TRANSITION-ABSENT** |
| **R2** | `z_t+dzhat_true` beats `z_t` on ≥ 2 of 4 targets | L3-TRANSITION-PASS | **L3-TRANSITION-FAIL** |
| **R3** | `dzhat_true` beats `dzhat_zero`; then if skill(`dzhat_anch`) ≈ skill(`act2`) within 0.02 it is an ECHO | ACTION-REACHES-TRANSITION | **ACTION-ABSENT-AT-TRANSITION** |
| **R4** | transition-specific skill (real − within-clip-shuffle) ≥ 0.5 × real skill on the winning cells | the read is about TRANSITIONS | it is a CLIP-LEVEL cue; R1–R3 re-read on the transition-specific column |

VOID conditions (sibling SPEC §7): `const` ≠ 0.0000 exactly; any cell's global-shuffle outside
[−0.05, +0.05].

## 2. Task A — the admissibility audit (committed here, applied in RESULT §2)

This is the part of Task A that is mine. Every item is a **pass/fail on the banked artefact**, and
a fail is reported as a fail — never quietly narrowed. Each is checked from the **raw JSON and the
tool source**, never from the sibling's prose.

| # | rule (source) | how it is checked | if it fails |
|---|---|---|---|
| A1 | **λ selected on the FIT split only** (CLAUDE.md probe-trap #2/#3) | read `select_lambda` / `crossfit` in `taniteval/tools/transition_probe.py`; confirm the inner CV sees only fit-split rows and that shuffled variants reuse the real-target λ | VOID for every cell |
| A2 | **PCA basis fit on the FIT split only** | `panel.pca`; if 0, the trap cannot fire and that is stated, not assumed | VOID |
| A3 | **Constant-only control reads the no-information value EXACTLY** | `cells.const.real.skill == 0.0` on all 4 targets, all arms | VOID (sibling's own condition) |
| A4 | **A raw-input floor exists and is not degenerate** | `cells.pixdelta`; ⛔ **additionally check whether its λ hit the GRID EDGE** (`LAMBDA_GRID` max = 1e4) — a floor shrunk to the constant reads the same value as A3 and therefore **discriminates nothing** | the linear floor is declared DEGENERATE and the RFF floor becomes the operative floor; any R1/R3 verdict that rests on "beats the floor" is re-stated as "beats the CONSTANT" |
| A5 | **Time-shuffled control** | `within_shuffle` (within-clip) and `global_shuffle` (global) present on every cell; global in [−0.05, +0.05] | VOID |
| A6 | **`n` and `d` printed** | `n_score`, `n_fit_per_fold`, `d` per cell; report `n/d` for the widest cell | the panel is not quotable |
| A7 | ⛔ **`v` is NEVER an input** (realised motion; r 0.9988) | audit **both** paths: (a) the feature list, and (b) **the model boundary** — `clip_features` calls `_lift3(a2, v0, cond_param)` with `cond_param = "steer_accel_v"`, so `dzhat_true` / `dzhat_zero` are produced by a predictor that WAS handed `v_last/SPEED_SCALE` | every cell built from a v-fed predictor output is flagged **v-CONTAMINATED at the model boundary** and may not be read as "the predictor transports the transition" without a v-zeroed control |
| A8 | **Estimator named on every interval** | `estimator` field on each CI; paired for marginals | not quotable |
| A9 | **Speed scale matches the arm's own training scale** | `_lift3` divisor vs the arm's trainer constant (`flagship_v15.SPEED_SCALE`); this is the defect that invalidated the banked `actdiv` family this week | stamped as a caveat on every affected cell |
| A10 | **Linear-probe rule** (CLAUDE.md) | any negative is stated as *not linearly decodable*; the `--mlp` RFF column with its own within-shuffle is the second function class | the verdict must name the function class |

**Rescue rule (committed):** where an audit item fails, I re-score from the raw **only if the raw
contains what a re-score needs**. The banked JSON stores **aggregates only** (skills, r, CIs,
λ per fold) and **no per-row predictions**, so any re-score requiring row-level predictions —
notably a CI on the *transition-specific* column, which R4's fallback asks for — is
**NOT RESCUABLE FROM THE RAW** and will be reported as such rather than approximated.

## 3. Task B — the tool change (the only code I own here)

`taniteval/tools/actdiv_anchored.py` gains **one pass-through flag** and nothing else:

```
--action-units {kappa,steer}   default kappa   (refav1 family only)
--steer-wheelbase FLOAT        default tanitad.models.kinematic.STEER_WHEELBASE_M = 2.9
```

* `kappa` — the swept level is fed **unchanged**. This is the legacy path and MUST be
  byte-identical to the banked run (§4 B0).
* `steer` — the swept level is read as a **true curvature in rad/m** and the value fed at the model
  boundary is `arctan(L_enc · level)`, applied by **`tanitad.models.kinematic.as_command(controls,
  "steer", wheelbase)`** — the repo's own encoder-side inverse, never a hand-rolled `atan`.
  `level` / `abs_level` keep the **nominal curvature** so linearity is measured *against the swept
  action levels*; a new record field `fed_value` carries what the model actually received.

⛔ `STEER_WHEELBASE_M = 2.9` is the **ENCODING** constant (`physicalai.signals_at` wrote
`steer = arctan(2.9 · curvature)` for every shipped cache). It is **never** cross-applied to ZOD,
l2d or alpasim, which encode with different wheelbases. The flag is refav1-only and the tool
refuses `--action-units steer` for `--family v7`.

A unit test (`stack/tests/test_actdiv_anchored.py`) pins: (i) `kappa` returns the candidate list
**object-identical** to the unconverted one; (ii) `steer` maps 0.1 → 0.28225742 rad and leaves the
accel channel untouched; (iii) the conversion goes through `kinematic.as_command`; (iv) `steer` is
refused on `--family v7`.

## 4. Task B — the pre-registered readings (committed BEFORE any forward pass)

Population, both conventions, both arms: **the banked 140 windows** — 20 eval-slice episodes,
`k_loader = cfg.op_steps = 30`, `--window-stride 10`, `--n-perm 200 --seed 0`, `h = 1`, verdict
space `tac`, κ levels `0.02, 0.05, 0.1`, accel levels `0.5, 1.5`. **One variable between the two
conventions: the number fed on channel 1.** Everything else — windows, seed, permutations,
checkpoints, levels — is identical.

**Speed channel (committed, and printed on every reading).** Both step-1,000 configs carry
`cfg.speed_channel = False` (verified: `ckpt_ep2/config.json` `cfg.speed_channel = False`; the
incumbent's `ckpt['cfg']` as loaded, banked `a_in_dim = 2`), so `RefAV1.augment_actions`
(`refa_v1.py:1263`) returns the `(a, κ)` controls unchanged and **no speed value reaches the
predictor**. The model's own constant is `refa_v1.SPEED_SCALE_MPS = 30.0` and it is **INERT here**.
⇒ the `v/30`-vs-`v/10` defect that invalidated the banked `actdiv` family this week **cannot occur
on this read**, and that is a structural guarantee (the tool never widens the action tensor), not a
lucky configuration. Every reading prints `SPEED_SCALE_MPS=30 (UNUSED: speed_channel=False)`.
*(The v7 arms' 10.0 does not enter Task B at all — no v7 arm is run here.)*

### The arithmetic that is fixed in advance (L_enc = 2.9)

| nominal level | fed under `kappa` | fed under `steer` | gain | ‖·‖-ratio to 0.02 if the model is exactly linear in the FED value |
|---|---|---|---|---|
| 0.02 | 0.02 | 0.0579353 | 2.8968 | 1.0000 |
| 0.05 | 0.05 | 0.1439959 | 2.8799 | **2.4855** (raw: 2.500) |
| 0.10 | 0.10 | 0.2822574 | 2.8226 | **4.8720** (raw: 5.000) |

σ of the windows' own channel-1 actions, **measured in the fed (steer) channel** =
**0.047565 rad** (banked, identical across arms). Its curvature equivalent is
`tan(0.047565)/2.9 =` **0.0164141 rad/m**.

### The five committed readings

| id | reading | HOLDS if | FAILS if |
|---|---|---|---|
| **B0** | **Reproduction.** The `kappa` pass reproduces the banked run | every quoted `tac` statistic (κ and a `F_sep`, `F_over_null_p95`, full-field `norm_by_level`, `rel_units`, controls) matches `raw/actdiv_anchored_refav1_step1000.json` to **≥ 6 significant figures** on both arms | any mismatch ⇒ the flag is **not** a pass-through, or the read is not deterministic; **the whole of Task B is VOID** and reported as such |
| **B1** | **Verdict invariance.** | both conventions read `LAT-INSENSITIVE-REFUTED` on both arms in all three spaces | any verdict flips ⇒ the invariance argument fails at its strongest point |
| **B2** | **Separation-ratio invariance.** κ-axis `F_sep / null_p95` | `|log(conv/raw)| ≤ log(1.25)` on both arms ⇒ the *ratio* is invariant | outside that band ⇒ **a banked number moves**; a ratio that RISES is a bigger stimulus, one that FALLS is saturation |
| **B3** | **Linearity against the swept action levels.** full-field `‖m‖` ratios to level 0.02 | `kappa`: 2.500 / 5.000 (banked 2.500 / 5.005 and 2.498 / 4.980) **and** `steer`: **2.486 ± 0.02 / 4.872 ± 0.05** ⇒ the model is linear in the FED value and the apparent "linearity to three significant figures at fixed nominal level" is **convention-dependent** | `steer` ratios **below** 4.82 at the top level ⇒ the predictor **SATURATES** past ~0.28 rad, and the converted numbers move by more than arctan alone predicts — a fact no reparametrisation argument can supply. Spearman ρ(\|level\|, ‖m‖) is committed to stay **+1.000** in both conventions (rank is invariant under any monotone map) |
| **B4** | **Antisymmetry.** cos(m(+L), m(−L)) at every level | ≤ **−0.99** at all three levels in both conventions (arctan is odd) ⇒ the antisymmetry half of the invariance argument HOLDS | any level > −0.95 under conversion ⇒ sign consistency breaks at the extended stimulus range |
| **B5** | **The magnitude ratio, and whether "matched σ" survives.** full-field `‖m(κ=0.1)‖ / ‖m(a=1.5)‖` | committed prediction: it grows by **×2.75 – ×2.90** (0.00329 → ≈ 0.0093 incumbent; 0.02514 → ≈ 0.0709 clean epoch) if the model is linear. ⭐ The **verdict clause**: at nominal level 0.1 the converted read feeds **5.93 σ_steer**, i.e. **0.1 rad/m is 6.09 σ_κ**, against `a = 1.5 = 2.02 σ_a` — so under the curvature reading the banked sentence *"the two axes are compared at matched σ"* is **FALSE**, and §4.2's ratio is not a matched-σ ratio | if the ratio moves by ≤ ×1.25 the invariance argument holds for it and my prediction is wrong |

### The verdict this task must return (committed wording)

`D-ACTDIV-UNITS-INVARIANCE` is decided as **one of exactly three**:

* **INVARIANCE-HOLDS** — B1–B5 all invariant within their committed bands. No banked number moves.
* **INVARIANCE-PARTIAL** — the *verdict*, the *rank-monotonicity* and the *antisymmetry* are
  invariant (B1, B3-ρ, B4) but at least one **magnitude or linearity digit quoted at a fixed
  NOMINAL level** moves outside its band (B2, B3-ratios, B5). ⇒ the argument is right about what a
  separation test measures and wrong about "no number changes"; **every banked sentence indexed by
  a κ LEVEL must be re-quoted with its convention.**
* **INVARIANCE-FAILS** — a verdict flips, or the response saturates, so even the separation
  conclusion is convention-dependent.

⚠️ **Committed in advance, because it is the honest half of the argument:** a predictor is a
function of the number it is fed. Any two conventions that feed the **same number** must produce
**identical** `d(a)` — that part is true by construction and B0 verifies the tooling honours it.
The question this task can actually decide is whether **the mapping from a nominal level to the fed
number** leaves the banked *statements* intact. Those are different claims and the RESULT must not
merge them.

## 5. Method notes and admissibility

* Every number: `MEASURED (ours; dev-box CPU; raw path)`. Evidence class, **tier T0**, **device**
  and **speed scale** on every reading (deliverable requirement).
* Four metric families: **REFUSED with the reason stamped** — both tasks are T0 world-model probes
  and no driving metric exists here to family-ise (EVAL_DOCTRINE). A scope statement, not an
  omission.
* Registry facts cited from `MODEL_REGISTRY.md` rows or raw JSON only; a missing row is reported as
  a conflict.
* `tanitad.__file__` and every checkpoint md5 are stamped into every JSON.
* ⛔ Nothing is committed or pushed. Every deliverable is `git add`-ed and verified by
  **blob comparison** (`git ls-files --stage` vs `git hash-object`) plus **two differently-bound
  marker probes** (`Select-String -LiteralPath`, and a second mechanism), re-verified at the end of
  the turn.
