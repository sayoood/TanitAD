<title>SPEC — anchored action-divergence on refav1 (H-REFAV1-LAT-INSENSITIVE), 2026-09-03</title>

# SPEC — E-AI-ACTDIV-REFAV1-0903: the anchored displacement read on both banked refav1 step-1,000 checkpoints

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Class: INSTRUMENT + READ. Tier: T0-DIAGNOSTIC throughout — nothing here is a driving-performance claim.`
`Compute: dev-box CPU only. The RTX 4060 is NOT used. ⛔ No pod, no Thor contact (refav1 trains on Thor).`
`Hypothesis: H-REFAV1-LAT-INSENSITIVE. Instrument seed: GS-8 / H-GS8-1.`
`Primary sources: Delta-JEPA arXiv 2606.31232 Fig. 6; ActSWM 2607.26712 Table 5 (both banked in Library/).`

---

## 0. Provenance of the pre-registration — READ THIS FIRST

⭐ **The binding pre-registered verdict table for this read was NOT written by this session.**
It is **§10 of the sibling package's SPEC**, at

`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-anchored-actdiv-and-transition-probe/SPEC.md`

written at **04:10 Berlin, 2026-09-03**, *before any refav1 forward pass had been executed by
anyone*, by the agent that built `actdiv_anchored.py` and was then killed by a model limit. That
file is the pre-registration of record. It is reproduced verbatim in §4 below so this package is
self-contained, and it is also **encoded in code** — `verdict_refav1()` and the `THRESHOLDS` dict
in `taniteval/tools/actdiv_anchored.py`, both pinned by tests. **Neither the function nor the
thresholds were touched by this session** (§5 lists every change made, and none of them is a
statistic).

⚠️ **Honest sequencing record.** This session did run the instrument twice *before* writing this
file, as plumbing smoke tests: `n = 2` windows / 1 episode and `n = 4` windows / 2 episodes on the
`ckpt_ep2` checkpoint only, at ~05:45 Berlin. Both smoke runs printed a `LAT-INSENSITIVE-REFUTED`
line. **Those runs are NOT the read and are not quotable** — 2 and 4 windows against a
pre-registered population of 140, with `n_perm` 10 and 20 against 200. They are recorded here
because a SPEC that implied blindness this session did not have would be a false document. The
verdict criteria they were measured against were already fixed in writing and in code 95 minutes
earlier, so the criteria cannot have been chosen to fit them — which is the property
pre-registration exists to protect.

---

## 1. Why this is now the primary refav1 read

MEASURED 2026-09-03 (register rows `D-REFAV1-PAIRED-READ-VOID` and the amendment to
`D-REFAV1-STEP1000-READ`): refav1's step-1,000 **T1 planning read is blind to the world model**.
Two checkpoints a full recipe apart produce **bit-identical trajectories on 140/140 eval windows**
(max |Δ| exactly 0.0 m) for the arms `cl` / `ha` / `ol` / `cl_navshuf`, because the deployed plan
is the constant-velocity baseline — a function of the measured `v0` alone. Every closed-loop plan
has curvature ≡ 0; `baseline_won_frac` 0.75, and where the CEM search wins its optimum is the same
control. The planner's cost surface is minimised at `(a = 0, κ = 0)` on ~100 % of windows.

⇒ **The deployed trajectory cannot discriminate recipes. The imagination and the cost surface
can.** This instrument reads the imagination.

**Where the flat plan could come from (read from `refa_v1.py::plan`, `_cost_chunk`):** candidate
controls are rolled `last_only`, the terminal operative field is pooled into the tactical query
space by `_tac_field`, scored by `1 − cosine` against the goal, **plus `0.02·jerk²`, plus an
explicit `0.05·κ²` curvature penalty**, plus a target-speed term. If the world-model term is flat
in κ, the κ² penalty alone drives κ → 0 exactly. That is the mechanism the hypothesis names.

**H-REFAV1-LAT-INSENSITIVE.** *refav1's predictor is insensitive to the lateral action* — the same
defect family as v7's P2.

## 2. The instrument (Delta-JEPA Fig. 6, ActSWM Table 5)

Published form (`Research/2026-09-03-sota-action/RESULT.md` §F1): keep the history fixed, replace
the action with each candidate, and measure the displacement **against the predictor's own
zero-action prediction**:

    d_i(a) = ẑ_{t+1}(a) − ẑ_{t+1}(0)

Delta-JEPA's LeWorldModel reads *"means concentrated at the origin"* (insensitive) against
Delta-JEPA's own *"spread"*. ActSWM's comparable gap runs **0.002 → 0.760** across its fix;
AD-JEPA's post-hoc centring **0 → 0.19–0.52**. Those are the field's effect sizes; our MM-E10
scene-vs-action ratio (0.00408–0.00595) has no published counterpart, which is why GS-8 adopts the
anchored form.

Here, at one operative step (h = 1, 0.2 s), from the model's own state and its own brains:

    d_i(a) = step(last_i, augment_actions(a)_0, intent_i) − step(last_i, augment_actions(0)_0, intent_i)

with `intent_i` from the model's own `_run_brains` under the loader's nav, **held fixed while the
action varies** (so nav is not a confound).

**Population.** Exactly the 140 windows the T1 read used: 20 episodes of the eval slice
(`C:\Users\Admin\refav1_eval_slice\fp8` + `...\eps`), `k_loader = cfg.op_steps = 30`, stride 10 on
`t − (W−1)`. Built by **reusing** `refav1_arm.py`'s own loader path, never a copy:
`taniteval/tools/refav1_arm.py:483` (`build_loader`), `:477` (`episode_names`), `:271`
(`load_model`, the strict loader with the config cross-check). The reuse is by file-import,
`actdiv_anchored.py:_load_refav1_arm`.

**Candidates (physical units, the coordinator's grid).** κ ∈ ±{0.02, 0.05, 0.1} rad/m at a = 0,
and a ∈ ±{0.5, 1.5} m/s² at κ = 0 — **10 candidates + the zero anchor**. Channel mapping is pinned
by test: κ is action channel **1**, a is channel **0** (`refa_v1.py:231`, predictor input is
`(a, kappa, v/SPEED_SCALE_MPS)`).

**Spaces.** Read in three, all reported: **`tac`** — the tactical query pooling `_tac_field`, i.e.
*the planner's own cost space*, and therefore the **verdict space**; **`pooled`** — the token-mean
field (1,024-d); **`full_subsample`** — the full token field, exact per-candidate means and F from
running sums over all 640 × 1024 coordinates, with the permutation null on a fixed 8,192-coordinate
subsample. A verdict that differs across spaces is reported as SPACE-DEPENDENT with all three.

**Statistics.** Per candidate: mean displacement `m(a) = mean_i d_i(a)`, its norm, and its norm
relative to the scene spread (std across windows of the zero-action step — the MM-E10 denominator,
so magnitudes sit on the banked instrument's scale). Per axis: `F_sep` (one-way between/within
pseudo-F of the displacement vectors), its **measured** label-permutation null (200 permutations →
median and p95), the level norms, Spearman ρ of ‖m‖ against |level|, and cos(m(+L), m(−L)) — a
linear response reads ≈ −1.

## 3. ⛔ The speed-scale ruling for refav1 (D-P2-LEAK-AUDIT / H-LEAK-1)

A banked instrument family was invalidated this week for feeding `v/30` to models trained on
`v/10`. The rule adopted here is stronger than "use 30":

> **This tool never builds the speed channel at all.** It calls the model's own
> `RefAV1.augment_actions`, which normalises by `refa_v1.SPEED_SCALE_MPS`
> (`stack/tanitad/refs/refa_v1.py:90`, = **30.0**) and **refuses a pre-widened 3-wide action
> tensor** by construction. The scale is therefore whatever the *loaded checkpoint's own module*
> says it is and cannot drift from it. Pinned by
> `test_refav1_speed_channel_is_derived_by_the_model_never_by_this_tool` and by a source-level
> guard, `test_refav1_tool_never_writes_a_speed_scale_of_its_own`, which fails if any hand-rolled
> `/ 30` ever appears inside `refav1_read`.

⭐ **AND FOR THESE TWO CHECKPOINTS THE CONSTANT IS INERT.** Both step-1,000 configs carry
`speed_channel: false` — VERIFIED from the primary artefacts, not from prose:
`C:\Users\Admin\refav1_eval_slice\ckpt_ep2\config.json` (`"speed_channel": false`, in both the
`args` and `cfg` blocks) and the incumbent's `ckpt['cfg']` as loaded. So `augment_actions` returns
the `(a, κ)` controls unchanged, `a_dim 2 → a_in_dim 2`, and **no speed value enters the predictor
at all**. The leak this rule guards against **cannot occur on this read**. The scale is printed
beside every reading anyway — `SPEED_SCALE_MPS=30 (UNUSED: speed_channel=False, predictor input is
(a,kappa) only)` — and recorded in the JSON as `speed_scale_mps: 30.0` with
`speed_scale_effective: null`.

## 4. THE PRE-REGISTERED VERDICT — reproduced verbatim from the 04:10 SPEC §10

Read in the `tac` space at h = 1.

| outcome | criterion |
|---|---|
| **VOID** | any control off its known value |
| **LAT-INSENSITIVE-CONFIRMED** | `F_sep(a-axis) ≥ 5 × its null p95` **AND** `F_sep(κ-axis) < 5 × its null p95` — the a-bins separate, the κ-bins sit at the within-bin floor |
| **LAT-INSENSITIVE-REFUTED** | `F_sep(κ-axis) ≥ 5 × its null p95` — the predictor DOES respond to κ consistently; the flat-κ plan is then a COST property (the κ² penalty against a κ-response that does not move the goal cosine), not a WM insensitivity |
| **BOTH-INSENSITIVE** | neither axis clears 5× — the P2 picture; lateral is not special |

**What each says (committed in advance):**
- **CONFIRMED** ⇒ H-REFAV1-LAT-INSENSITIVE stands as a world-model property; the fix is on the
  predictor's κ path (P2 family), not in the planner.
- **REFUTED** ⇒ the fix is in the **cost** (the κ² penalty / the goal cosine's blindness to lateral
  change), and the register row must say the world model is not the cause. **The next probe is
  then the cost surface itself.**
- **BOTH-INSENSITIVE** ⇒ refav1 at step 1,000 is action-deaf like the v7 arms (P2), and the
  lateral finding is a special case of it.

Also reported, not gating: the magnitude asymmetry `‖m(κ = 0.1)‖ / ‖m(a = 1.5)‖` in every space.

### Controls — each must read a KNOWN value or the panel is VOID

| control | known value | how it is produced |
|---|---|---|
| **C0 identity** | max\|Δ\| **exactly 0.0** | the same zero-action inputs forwarded twice through the real predictor |
| **zero model** | max\|d\| **exactly 0.0** | the real predictor fed the **zero** action while the candidate label says otherwise — an action-blind predictor by construction |
| **C1 scene spread** | **> 1e-6** | std across the 140 windows of the zero-action step; if ≈ 0 the ratio is 0/0 (the MM-E10 "degenerate in both directions" case) |
| **shuffled labels** | `F_sep` inside the permutation band | 200 within-window label permutations; the null is MEASURED (median, p95), never assumed to be 1 |
| **shuffled actions** | between/within falls to the floor | realised mode: window *i* is fed window *i+j*'s action and labelled by *i*'s original bin |
| **output scale** | `F_sep`, ρ, cos **unchanged** | every prediction × c; the statistics are scale-homogeneous — asserted, not assumed |
| **n and d** | printed | `n_windows`, `state_dim` per space, `n_candidates`, per-level norms |

⚠️ **A known instrument property, pinned by test:** an **exactly dead axis reads `nan`, not 0** —
its F is 0/0 (no between variance because every candidate mean is the zero vector; no within
variance because every window's displacement is that same zero vector). `verdict_refav1` guards
with `np.isfinite(f) and f >= bar`, so `nan` is correctly *not separated*. A test that expected 0
would have failed on a working instrument; two did, and are now pinned to `not isfinite`.

## 5. Every change this session made to the instrument

Nothing statistical was touched. In full:

| file | change | why |
|---|---|---|
| `taniteval/tools/actdiv_anchored.py` | record `speed_scale_mps` + `speed_scale_source` in the result JSON | the brief's D-P2-LEAK-AUDIT requirement: the scale must be **on the artefact** |
| `taniteval/tools/actdiv_anchored.py` | print `SPEED_SCALE_MPS`, `a_dim → a_in_dim`, `speed_channel` on the per-arm line | the scale must be beside **every reading** |
| `taniteval/tools/actdiv_anchored.py` | print **total** parameters, not `prov['trainable_parameters']` | ⚠️ `refav1_arm.load_model` calls `requires_grad_(False)` on every parameter **before** counting, so the trainable count reads **0 by construction** and the log claimed an empty model |
| `taniteval/tools/actdiv_anchored.py` | per-arm JSON: `parameters_total`, `a_in_dim`, `speed_channel`, `speed_scale_effective` | same |
| `stack/tests/test_actdiv_anchored.py` | **+7 tests** (group 8): the refav1 read END-TO-END on a random-init tiny `RefAV1` (85k params, fitted standardizer) through the real predictor path | the brief's requirement that every control read its known value on a tiny model |

The 7 new tests: the speed-scale contract (the model's own 30.0, the integration formula, the
refusal of a pre-widened tensor, and `speed_channel=False → 2 channels`); the source-level
anti-leak guard; the end-to-end controls (C0 == 0.0, zero-model == 0.0, C1 > 0 in all three
spaces); the full-field statistics' well-formedness; an **action-blind** tiny model reading
**exactly** 0.0 with `ss_between == ss_within == 0`; a **planted κ-only** response reading
`LAT-INSENSITIVE-REFUTED`; and a **planted accel-only** response reading
`LAT-INSENSITIVE-CONFIRMED`. The last is the one that matters most for credibility: without it the
instrument could only ever refute the hypothesis, never confirm it.

**Suite: 34 passed** (27 inherited + 7 new), CPU, no checkpoint, no corpus, no GPU.

## 6. The run

```
python taniteval/tools/actdiv_anchored.py --family refav1 \
  --arm incumbent_fp32=C:\Users\Admin\refav1_eval_slice\ckpt\ckpt.pt \
  --arm clean_epoch_ema_bf16=C:\Users\Admin\refav1_eval_slice\ckpt_ep2\ckpt.pt \
  --episodes-n 20 --window-stride 10 --n-perm 200 --seed 0 --device cpu --batch 4 \
  --out .../raw/actdiv_anchored_refav1_step1000.json
```

Launched **detached** (`powershell.exe Start-Process`, PID 27116, 05:56 Berlin) per the > 55 min
rule. `PYTHONPATH` = the off-Drive mirror; `import tanitad` verified to resolve to
`C:\Users\Admin\tanitad-wt\stack\tanitad\__init__.py` by the tool itself, which **refuses to run**
if it resolves anywhere else.

**The two checkpoints.**

| arm | path | config source | step |
|---|---|---|---|
| `incumbent_fp32` | `refav1_eval_slice\ckpt\ckpt.pt` (retired fp32 incumbent) | `ckpt['cfg']` (no `config.json`) | 1,000 |
| `clean_epoch_ema_bf16` | `refav1_eval_slice\ckpt_ep2\ckpt.pt` (clean epoch, EMA + bf16 + TF32), md5 `c26c7ad00e5b408b54570cb2bd1bcfae` | `config.json` beside it | 1,000 |

No later checkpoint of the live run was pulled, and **none was assumed to exist** — the live run is
on Thor and this package must not contact it.

## 7. Admissibility

- **Tier T0-DIAGNOSTIC.** This is a world-model probe. It is not a driving-performance number and
  may not be quoted as one.
- **The four metric families are REFUSED, explicitly and in the JSON**
  (`metric_families: {longitudinal|lateral|tactical|strategic: "REFUSED"}`) with the reason
  recorded: a T0 latent-space diagnostic has no trajectory to score. Per `EVAL_DOCTRINE`, a refusal
  on the record is admissible; a silent omission is not.
- **Evidence class: MEASURED (ours; dev-box CPU).**
- The null band is **measured** by permutation, not assumed.
- No interval is quoted, so no estimator claim is made. `F_sep` is reported against its own
  measured permutation p95, which is the comparison the verdict uses.
