# Nav-COMPLIANCE metric + refcv4b post-training hierarchy prereg — RESULT

**status: DELIVERED (metric + tests + harness + registry + prereg + smoke, all staged and committed in this turn) — one item PENDING: a stride-5 re-roll of the same step-9,500 checkpoint WITH the nav-FLIP arm is rolling detached on the dev-box RTX 4060 (`C:\Users\Admin\navcomp\dump9500b\`, record → `C:\Users\Admin\navcomp\bank\raw\refcv4b-9500-openloop-s5flip.json`, ~70 min; if this session ends first it lives in ONE PLACE and the next turn banks it).**

*Benchmarks & Eval FlyWheel, 2026-09-05. Resumes the predecessor's partial work at `C:\Users\Admin\navcomp\`
(its md5-verified checkpoint pull and 141-clip eval cache, `raw/pull.log`) and harvests, from its mirror,
three drafts that never reached the repo — a `nav_compliance.py`, registry and test drafts — whose STRATA
and nav-FLIP ideas are folded in here; the drafts are preserved verbatim under `raw/predecessor_drafts/`.*

---

## 0. The verdict in one paragraph

**The instrument works and is falsifiable where the old one was not; on the step-9,500 checkpoint the
emitted path follows nav and the strategic goal does not.** MEASURED (ours), **EARLY-TRAINING DIAGNOSTIC**
(refcv4b at 9,500 of 40,284 steps = 23 %, **T1\*** self-action open loop, arm `os`), 1,860 windows / 141
eval clips (stride 10, grid 6 s), **141 imminent windows / 22 episodes**: plan-compliance with the TRUE
command **0.390 [0.303, 0.472]** drops to **0.291** under nav-SHUFFLE (paired Δ **+0.099 [+0.041,
+0.167]**, separated) and to **0.298** under nav-ZERO (Δ **+0.092 [+0.027, +0.169]**, separated) →
`FOLLOWS_NAV`; the SELECTED anchor reads the same (Δ_shuffle **+0.106 [+0.049, +0.170]**); **`g_str` is
bit-identical under all three conditionings** (0.792 / 0.792 / 0.792 on n = 485 / 43 eps, Δ **0.0000
[0.0000, 0.0000]**) → `NAV_BLIND` — the zero-init `nav_to_str` edge (E13 → strategic) has not moved at
9,500 steps. Seam reading: **PATH_FOLLOWS_WITHOUT_GOAL** — nav reaches the operative layer through the
measurement encoder, not the strategic decision: the inversion `D-REFCV4-NAV-WIRING` measured in refcv3,
now measured on refcv4b early in training. ⚠️ The plan's compliance does **not** clear the nav-blind
floor: `ha0_ext` (constant-a, constant-κ from the measured t0 state) complies on **0.418** of the same
windows and `ha` on **0.390** — nav-following exists and is weak. Every known-value control read its
value (§3). The standalone artifact passes registry v2.6.0: **0 violations, 13 work items** (the other
families declined at their keys, pointing at the companion record). ⛔ Nothing here decides a hypothesis
of the prereg; it decides that the instrument can.

---

## 1. Why the old metric could not fail, and why this one can

The strategic route metric compared `route_pred` (a head argmax) to `route_label`, and
`D-REFAV1-ROUTE-LABEL-IS-THE-NAV` MEASURED that label as an exact bijection of the nav token fed at
inference (141/141): a head copying its input scores **1.0000** and no behaviour of the model can lower
it. The existing echo test compared the prediction to the input and could not see that the LABEL was
the input.

This metric (`taniteval/taniteval/nav_compliance.py`):

1. **scores BEHAVIOUR** — the emitted 6 s path's terminal heading (with end-bearing and lateral end as
   secondary signals), the strategic goal bearing `g_str`, and the SELECTED vocabulary element's
   geometry (the selection surface, decided at the t=0 confidence: audit `7c67b3d`) — against the
   COMMANDED side, on INFORMATIVE windows only (true command left/right, token valid, the commanded turn
   overlapping the 6 s plan by ≥ 1 s; for `g_str`, the turn within 160 m of arc — LAN's farthest anchor),
   **n windows and n episodes always reported**;
2. **is decided by the intervention pair, never by the rate**: compliance with the TRUE command must DROP
   under nav-SHUFFLE and nav-ZERO on the same windows (paired episode-cluster bootstrap), and on the
   changed subset the behaviour must follow the FED command; the optional nav-FLIP (left↔right,
   `--with-navflip`) disagrees with the truth on every informative window. A rate that does not drop is
   coincidence with the scene — and the nav-blind controls measure exactly that floor on the same
   windows, per readout, with the matching signal and tolerance;
3. **cannot be satisfied by an echo**: a path is not a token. The deliberate-regression test
   (`test_DELIBERATE_REGRESSION_a_token_echo_is_invisible_to_the_behavioural_readout`) hands the metric an
   arm whose route head copies the token perfectly and whose path is scene-driven: the head's own
   "accuracy" is 1.0 and the behavioural metric reads Δ = 0.0 exactly;
4. **reports plan- and `g_str`-compliance separately on the same windows**, so *right goal + wrong path*
   (a seam failure, E4/E7/E9) and *right path + wrong goal* (nav bypassing the decision layers) are
   different readings (`seam_reading`), plus a strata census (imminent / deferred / stale / conflict /
   ambiguous — the predecessor's design) and a DEFERRED-window consistency block (turn beyond the plan:
   the plan must HOLD while `g_str` already points the way — the PI's "in consistency to the strategic
   goals");
5. **carries controls that must read known values**: `ha0` (straight line) compliance = **0.0000
   exactly**; every nav-blind control's intervention delta = **0.0000 exactly**; the GT future on the
   informative windows ≈ 1; the `always_commanded` synthetic policy gives the CEILING drop any follower
   can show on these windows; the label time base is decided ON THE GT YAW, not on a docstring.

The registry (v2.6.0) makes the rule mechanical: `strat.nav_compliance` (the rate) plus one criterion
PER CONTROL (`_ctrl_shuffle`, `_ctrl_zero`), all required — an artifact carrying the rate alone is two
NAMED violations, and `unavailable_block()` declines at every registered key so an un-rolled control is a
tracked work item, never a silent omission (7 new tests, incl. a deliberate-regression arm per control).

---

## 2. Deliverables

| # | artifact | where | state |
|---|---|---|---|
| 1 | `taniteval/taniteval/nav_compliance.py` — the metric (readouts, strata, per-readout nav-blind controls, verdicts, seam, deferred consistency, dump reader, `unavailable_block`) | `repo:taniteval/taniteval/nav_compliance.py` | STAGED + committed |
| 2 | `taniteval/tests/test_nav_compliance.py` — 28 tests incl. the token-echo regression arm | `repo:taniteval/tests/test_nav_compliance.py` | 28 pass (mirror) |
| 3 | `taniteval/tools/nav_compliance_report.py` — standalone artifact CLI (0 GPU; compliant `taniteval.driving` shell) | `repo:taniteval/tools/nav_compliance_report.py` | staged |
| 4 | `taniteval/tools/refcv3_arm.py` — ADDITIVE: ego-state fed on `ego_state_inject` builds (a regime fix), nav-compliance sidecar per conditioning, `--with-navflip` (`os_navflip`), the analysis block; every pre-existing arm and key byte-unchanged | `repo:taniteval/tools/refcv3_arm.py` | staged |
| 5 | `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0 — `strat.nav_compliance`, `strat.nav_compliance_ctrl_shuffle`, `strat.nav_compliance_ctrl_zero` (+ the `route_head_echo` guard accepts the pair) | `repo:products/P7-TanitEval/CRITERIA_REGISTRY.json` | staged |
| 6 | `tools/tests/test_criteria_check.py` — fixture + 7 tests | `repo:tools/tests/test_criteria_check.py` | 73 pass / 1 skip (mirror) |
| 7 | `Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` — the post-training ablation prereg, both outcomes committed, honest scope | `repo:Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` | staged |
| 8 | `Project Steering/GOALS_AND_CLAIMS.md` — section `D-NAVCOMP` (D-NAVCOMP-1..3, H-NAVC-1..3, H-SEAM-1/H-H19-1/H-CONS-1/H-SEL-1) | `repo:Project Steering/GOALS_AND_CLAIMS.md` | appended (insert-only), staged |
| 9 | smoke evidence: `raw/refcv4b-9500-openloop.json` (full record), `raw/navcomp_refcv4b-9500.json` (the artifact, 0 violations), logs, patch scripts, the predecessor's drafts | `repo:…/2026-09-05-nav-compliance-metric/raw/` | staged |
| 10 | the dumps (`dump9500/`, `dump9500b/`), the stride-5 record (pending) and `ckpt_step9500.pt` (md5 `ab4cd79e35b92bf70d6e2b80ac053e51`, verified against the pod in `raw/pull.log`) | `devbox:C:\Users\Admin\navcomp\` — **ONE PLACE** (1.3 GB; the pod holds the live run's own copy) | not banked (size) |

---

## 3. The EARLY-TRAINING DIAGNOSTIC — first pass (stride 10, grid 6 s, no flip)

`raw/refcv4b-9500-openloop.json` → `refcv3.strategic.nav_compliance`; `raw/navcomp_refcv4b-9500.json`.
ckpt step **9500** (verified by content), config rebuilt through the trainer's own parser
(`--nav-from-v7`, `--goal-str`, `--ego-state-inject`, 117 v0-conditioned `alat` anchors), STRICT load,
`ego_state_fed: true`. 1,860 windows / 141 episodes (552 skipped: 6 s horizon beyond the episode).
Estimator: episode-cluster bootstrap, paired for every delta, `n_boot` 2000, seed 0, cluster = episode.

**Controls (must read known values):**

| control | required | measured |
|---|---|---|
| label time base on the GT yaw | one hypothesis reproduces `dyaw_deg` on most clips | **relative** 22/23 (0.957) vs absolute 1/39 (0.026) → relative (the label's manoeuvre times are offsets from `t0_s` = 8.0 s) |
| `ha0` compliance | 0.0000 exactly | **0.0000** ✓ (`panel_ok: true`) |
| nav-blind deltas (`ha`, `ha0`, `ha0_ext`) | 0.0000 exactly | **0.0000** ✓ |
| GT future on imminent windows | ≈ 1 | **0.986** ✓ |
| `always_commanded` ceiling | the largest drop a follower can show | Δ_shuffle ceiling **0.837** (the model's +0.099 is 12 % of it) |
| tolerance derivation | GT + label only | τ_plan = **0.2608 rad (14.9°)**, Youden J 0.81 (TPR 0.986 / FPR 0.176; 141 informative vs 1,467 follow windows); τ_gstr = 0.055 rad (J 0.185 — the 6 s GT bearing separates deferred turns poorly; near the 5° fallback) |
| fan coverage | — | a reach-surviving candidate complying with the command exists on **1.000** of imminent windows under all three conditionings (mean 47 of 117) → a miss is selection, not vocabulary |

**Readouts (imminent windows n = 141 / 22 eps unless stated; rates = compliance with the TRUE command):**

| readout | nav_true | nav_shuffled | nav_zero | Δ true−shuffled | Δ true−zero | verdict |
|---|---|---|---|---|---|---|
| plan (terminal heading) | **0.390** [0.303, 0.472] | 0.291 [0.230, 0.351] | 0.298 [0.226, 0.372] | **+0.099 [+0.041, +0.167]** sep | **+0.092 [+0.027, +0.169]** sep | FOLLOWS_NAV; below the `ha0_ext` floor 0.418 |
| plan (end bearing) | 0.418 | 0.326 | 0.340 | +0.092 [+0.052, +0.141] sep | +0.078 [+0.019, +0.144] sep | FOLLOWS_NAV |
| plan (lateral end, τ 1 m) | 0.468 | 0.348 | 0.326 | +0.121 [+0.073, +0.168] sep | +0.142 [+0.092, +0.193] sep | FOLLOWS_NAV |
| selected anchor (bank path) | 0.383 [0.296, 0.469] | 0.277 | 0.284 | **+0.106 [+0.049, +0.170]** sep | +0.099 [+0.041, +0.164] sep | FOLLOWS_NAV |
| `g_str` bearing (n = 485 / 43 eps) | **0.792** [0.660, 0.914] | 0.792 | 0.792 | **0.0000 [0.0000, 0.0000]** n.s. | 0.0000 | NAV_BLIND (bit-identical; no kinematic floor exists at the strategic horizon) |
| `ha0_ext` (nav-blind floor) | **0.418** [0.361, 0.467] | = | = | 0 | 0 | the coincidence floor |
| `ha` | 0.390 [0.323, 0.447] | = | = | 0 | 0 | |

⚠️ The shuffle's changed subset "fed a different TURN" is only **n = 19** at stride 10 (the shuffle feeds
`follow` on most changed windows) — it is why the flip arm exists and why the re-roll is at stride 5.

**Seam:** `PATH_FOLLOWS_WITHOUT_GOAL`. **Selection profile:** 35 distinct anchors, modal **#67 = the
straight-ahead control on 60.2 %** of windows, entropy ratio 0.356, oracle agreement 0.093. **Four
families (ADE@6 s, T1\*):** `os` **3.517 m** [3.182, 3.872] vs `ha` 3.560 (paired −0.043, n.s.) vs `ha0`
4.955 (−1.438, separated); `os_navzero` 3.502 (Δ vs `os` +0.014, n.s.). ⚠️ At 23 % of training the model
ties hold-action on ADE and beats the straight line; not a result about the run.

**Reading, with its scope:** at step 9,500 the nav token changes what the model EMITS and SELECTS (a
~0.10 drop in compliance when the token is permuted or withheld, separated on 141 windows / 22 episodes)
but not what it DECIDES strategically (`g_str` identical to the bit). That is the `D-REFCV4-NAV-WIRING`
inversion and the H-NAVC-2 failure mode the prereg names — **early in training, where a zero-init edge is
expected to be closed**. Whether it opens by 40,284 is the post-training question the prereg asks.

---

## 4. Instrument findings worth escalating

1. ⚠️ **`refcv3_arm.py` evaluated v4 builds in a regime they never trained in.** `run_dump` passed no
   `ego_state`, so an `ego_state_inject` model (refcv4b) had its goal path starved while the core still
   saw `v0` (training draws keep with `ego_dropout` on the WHOLE block; the trainer's eval feeds it with
   keep = 1). FIXED additively (v3 builds byte-unchanged; the manifest records `ego_state_fed`). Any
   refcv4b number from the unpatched harness is inadmissible. (Register: D-NAVCOMP-2.)
2. ⚠️ The label's `manoeuvre_sequence` times are RELATIVE to `t0_s` — decided on the GT yaw (22/23), not
   from the emitter's docstring; the window's NOW is `(ws + provider_to_raw_frame_offset) × 0.1 s`. The
   trainer's tactical join ignores the raw offset (0.2 s at `n_stack` 3) — harmless for a ±2 s band.
3. **The old strategic block stays in every record as an echo diagnostic** (`strategic.conditionings.*`
   unchanged); the new block sits beside it at `strategic.nav_compliance`. The full refcv3-style record
   is `UNKNOWN_SCOPE` to the checker (pre-existing: no in-scope marker at its top level; the
   openloop suite promotes blocks) — the standalone artifact is the in-scope one.
4. **Not implemented, ESCALATED (0 GPU, must precede the post-training run):** the prereg's eval-time
   ablation switches (`g_str`-zero/shuffle, E7-off, E9-off, H19-off, ego-zero, SEL-refined) as
   `refcv3_arm.py --ablate …` flags — one function each on `RefCV3Model` / the decoder.
5. ⚠️ The core's `route_head` readout cannot be tested by any eval-time ablation (`graft_route` False);
   the prereg excludes it explicitly rather than reporting a vacuous "no effect".

---

## 5. Raw

| file | what |
|---|---|
| `raw/refcv4b-9500-openloop.json` | the full record (four families + strategic incl. `nav_compliance`), first pass, re-analysed with the final module |
| `raw/navcomp_refcv4b-9500.json` | the standalone nav-compliance artifact (registry v2.6.0: 0 violations, 13 work items) |
| `raw/dump9500.log`, `raw/analyze9500.log` | the roll / re-analysis logs (first forward 0.58 s per 3-row batch on the RTX 4060) |
| `raw/patch_*.py` | the exact additive edits applied to `refcv3_arm.py`, the registry and the criteria tests |
| `raw/predecessor_drafts/` | the predecessor's unstaged drafts, verbatim |
| `raw/pull.log` | the predecessor's md5-verified pull of the checkpoint + eval cache |
| *(pending)* `raw/refcv4b-9500-openloop-s5flip.json` | the stride-5 record with `os_navflip` — lands on the dev box first (§ status) |
