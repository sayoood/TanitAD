# 2026-08-03 — The LONGITUDINAL family can finally see distance-keeping, and the gauge is admitted

**Architecture & Inference · daily run (slot 3, 09:51 local) · 0 GPU · $0 · ~2 h wall-clock**

## 0. The one-paragraph verdict

Sayed made four metric families binding on 2026-08-02. **Half of the LONGITUDINAL family has been
uncomputable ever since** — `taniteval/taniteval/four_families.py` returned
`distance_keeping: UNAVAILABLE` because our episode ingest never read `obstacle.offline`. That is now
closed. The metric is implemented, the ingest that feeds it is implemented, and — before any arm was
scored — the **pre-registered GT-vs-CV discrimination control D-LEAD-1 PASSED on all three sub-metrics**:
Δ min-TTC **+1.7474 s** [1.5813, 1.9218], Δ headway **+0.9769 m** [0.8830, 1.0758], Δ time-gap
**+0.1641 s** [0.1499, 0.1786], paired episode-cluster bootstrap over **14,027 windows / 1,431 clip
clusters**, all separated, all with the correct sign. The gauge moves, so it is admissible. ⛔ It is
**not yet fed on the eval path** — that is the next work item and it is named below, not implied.

**Evidence class: MEASURED (ours).** Artifact:
`Implementation/incoming/2026-08-03-longitudinal-distance-keeping/raw/dlead1_discrimination.json`.

## 1. Why this was the top item

`Research/2026-08-03-sota-scan/SOTA_SCAN.md` §11 ranked it **#1 of 8**: 0-GPU, *binding*, currently
*absent*. §2 of the same scan is the reason it outranks everything else — on n=8 systems
([arXiv 2605.00066](https://arxiv.org/html/2605.00066)) **traditional L2 (ADE/FDE) correlates with
closed-loop driving score at ρ = −0.36, p = 0.43 — not significant**, while PDMS aggregates reach
ρ = 0.90 and Ego Progress alone 0.83. Every closed-loop score we benchmark against carries **TTC and
progress** as first-class sub-scores ([NAVSIM](https://arxiv.org/html/2406.15349v1), EPDMS per
[IDOL](https://arxiv.org/html/2605.31476v1)); the criticality family itself — TTC, Time Headway,
Time-to-Brake, DST — is the field's standard ([arXiv 2603.28029](https://arxiv.org/html/2603.28029)).

So the programme's binding rule and the published evidence point at the same missing instrument, and
88.7 % of our oracle gap is longitudinal. Building it is worth more than any further ADE horizon sweep.

## 2. What already existed — and why the new part is genuinely new

**Absence found at one location is not absence** (CLAUDE.md rule 2), so I looked before building.
`stack/scripts/lead_state_gate.py` already contains a **proven, strictly causal `obstacle.offline`
reader** (`ego_frame` / `lead_frame`), and E-GOAL-1
(`incoming/2026-07-27-egoal-1-lead-vehicle/code/eg_common.py`) already wraps it with clock-join and
span assertions. Re-implementing either would have re-derived their bugs.

**What they answer:** *where is the lead right now, relative to the ego's TRUE pose.* That is an
**input feature**.

**What scoring an arm asks:** *where would the lead have been relative to the trajectory THIS ARM
PREDICTED.* Different question, and it needs a coordinate step neither existing path performs.
`obstacle.offline` cuboids carry `reference_frame='rig'` — each row is in the ego frame **at its own
timestamp**, so two rows 1 s apart live in two different frames, while an arm's waypoints all live in
the frame at t0. The composition is exact and mandatory:

```
world:  L_w(t) = ego_xy(t) + R(yaw(t)) @ [center_x, center_y]
t0:     L_0(t) = R(-yaw(t0)) @ (L_w(t) - ego_xy(t0))
```

⚠️ **Skipping it would have invented tailgating everywhere** — it understates the gap by roughly the
distance the ego travels over the horizon, ≈27 m at 13.6 m/s over 2 s, which is larger than most real
headways in this corpus (mean 29.05 m). This is the kind of defect that produces a plausible,
publishable, wrong table.

## 3. The instrument

`taniteval/taniteval/lead_metrics.py` — pure, I/O-free, no torch.

| metric | definition | convention that must travel with it |
|---|---|---|
| `headway_min_m` | tightest longitudinal gap over the horizon | **rig-origin to lead REAR face** (`along − size_x/2`), identical to `lead_frame`. ⛔ NOT bumper-to-bumper — our origin is the rig, the ego's front overhang is not subtracted. Two gap conventions in one programme is a retraction waiting to happen. |
| `time_gap_min_s` | headway ÷ ego speed at t0 (time headway, THW) | **undefined at standstill** (NaN below 0.5 m/s), not clamped to a large number |
| `min_ttc_s` | gap ÷ closing rate, closing rate = −d(gap)/dt | **capped at 30 s when not closing** (`lead_frame`'s convention). A capped TTC is a **censored observation** — `n_closing` is emitted so a mean is never read as if every window closed. |

Two design points worth stating because they are where a naive version goes wrong:

1. **The corridor gate follows the PREDICTED path's own local heading**, not the t0 axes. An arm that
   drifts laterally must *lose* its lead — otherwise the metric credits it with distance-keeping on a
   vehicle it is no longer behind. `test_corridor_follows_the_predicted_heading_not_the_t0_axes` pins it.
2. **Lead SELECTION at t0 is strictly causal** (samples ≤ t0 only, `lead_frame`'s rule), enforced at
   runtime by `assert_selection_causal`. The lead's *future positions* are ground truth about the
   world — the same standing as the GT ego waypoints an ADE is measured against — and are a scoring
   input, never an arm input. Retraction class C23 (oracle-shaped-as-ego-state) is what that guard exists for.

## 4. D-LEAD-1 — the pre-registered discrimination control

`PRE_REGISTRATION.md` was written and committed **before** the runner executed. Three outcomes fixed
in advance, including the **INSTRUMENT-FAIL branch that C63's prereg lacked**: separation with the
wrong sign, n < 100 windows, < 10 clusters, or > 50 % censoring in *both* arms ⇒ **no verdict issued**.

**Arms.** GT = the human's true future path. CV = hold-`v0`, straight ahead at the t0 speed — a policy
that never brakes and never steers. Horizon 2.0 s, dt 0.5 s (the grid our ADE@2s uses), 1.0 s stride.

**Result (MEASURED, dev-box CPU, 125.3 s, $0):**

| metric (GT − CV) | Δ | CI95 | p(Δ>0) | separated |
|---|---|---|---|---|
| **min-TTC (s)** — primary | **+1.7474** | **[1.5813, 1.9218]** | 1.0 | ✅ |
| headway (m) | **+0.9769** | [0.8830, 1.0758] | 1.0 | ✅ |
| time-gap (s) | **+0.1641** | [0.1499, 0.1786] | 1.0 | ✅ |

Estimator: **paired episode-cluster bootstrap** (`taniteval.ci`), clusters = clips, B = 2000, seed 0.
⛔ never `overlapping_holdout_se`, which biases the point estimate as well as the interval.

| coverage | |
|---|---|
| clips scanned / with a lead | 2,417 / **1,548** |
| windows scanned / with a lead | 41,087 / **15,760** (38.4 %) |
| **paired windows / clusters** | **14,027 / 1,431** |
| GT-only / CV-only windows | 336 / 713 |
| dropped for span | 45 |
| censored at TTC_CAP | GT 51.7 %, CV 47.5 % |

⇒ **prereg branch 1 — PASS, ADMISSIBLE.** No INSTRUMENT-FAIL clause fired.

**Absolute levels, for context only:** GT mean headway 29.05 m, time-gap 4.29 s, min-TTC 23.56 s
(n_closing 6,936 / 14,363); CV 28.02 m, 4.22 s, 21.28 s (n_closing 7,738 / 14,740). The CV arm closes
on the lead **more often** and ends up nearer — which is precisely the behaviour that makes the family
discriminate, and precisely what an ADE-only report cannot see.

## 5. ⛔ Four things this does NOT establish

1. **It says nothing about any TanitAD arm.** GT-vs-CV measures the *gauge*, not a model. No arm has
   been scored on it yet.
2. **It does not close the 88.7 % longitudinal gap.** It builds the instrument that can finally *see*
   the distance-keeping half of it.
3. **`min_ttc_s` is censored.** ~50 % of windows never close and sit at the 30 s cap. The mean is a
   mean over censored data — `n_closing` must be quoted beside it, always. A future refinement should
   report a closing-only stratum or a survival-style summary.
4. **Coverage is the 26 label chunks on this disk**, not the canonical 2,376-episode corpus and **not
   the 40 val episodes**. These numbers are a *gauge property*, not a corpus statistic.

## 6. What is wired, and the honest gap

**Wired (suite green before and after — taniteval 810 ✓, stack 1719 ✓ / 12 skipped / 2 xfailed):**
`longitudinal(pred, gt, dt, lead=None)` and `all_families` reading `win["lead"]`. Strictly additive:
without a lead track the family still reports UNAVAILABLE with its reason and `_complete` stays False.

⛔ **The gap, stated rather than implied: arm evals will still report the family UNAVAILABLE.**
`build_lead_tracks.py` reads the PhysicalAI label zips **on the dev box**. Feeding the eval path needs
the `obstacle.offline` chunks for the **40 val episodes** on the eval host and a `win["lead"]` builder
in the runner. That is the next work item (backlog **P0 L1**), not a detail.

**Production-readiness (D-029):** the metric is **validated** (admitted by a pre-registered control on
14k real windows, 14 tests). The end-to-end eval path is **prototype** — the gap to *validated* is
exactly the val-40 lead-track build above.

## 7. Recommendations, each with its gate and falsifier (G-AI1)

1. **Build `win["lead"]` for the 40 val episodes and re-report every banked arm's LONGITUDINAL
   family.** Gate: the binding four-family rule. Falsifier: if < 20 % of val windows carry a causal
   lead, the val set is too free-flow and the family is reported NOT-APPLICABLE **with its n**, not
   silently dropped. *(Prior from this run: 38.4 % of windows carry one — so this is likely to bite.)*
2. **Report a closing-only stratum beside the censored mean.** Gate: same. Falsifier: if the
   closing-only Δ ranks arms identically to the pooled Δ, the stratum is redundant — drop it.
3. ⚠️ **Do not let ADE decide a longitudinal question.** MEASURED elsewhere, PUBLISHED here: ADE↔DS
   ρ = −0.36 (p = 0.43, n=8). Any arm ranking that rests on ADE alone is, on the field's own evidence,
   uninformative about closed-loop driving.

## 8. Deliverable manifest

| artifact | where it lives |
|---|---|
| this note | `TanitAD Research Hub/Architecture & Inference/Research/2026-08-03-longitudinal-distance-keeping.md` — repo, staged |
| pre-registration | `…/Implementation/incoming/2026-08-03-longitudinal-distance-keeping/PRE_REGISTRATION.md` — repo, staged |
| metric (pure) | `taniteval/taniteval/lead_metrics.py` — repo, **landed** (deviation declared in INTAKE.md) |
| tests (14) | `taniteval/tests/test_lead_metrics.py` — repo, **landed** |
| wiring | `taniteval/taniteval/four_families.py` — repo, **landed**, additive |
| ingest | `…/incoming/2026-08-03-longitudinal-distance-keeping/build_lead_tracks.py` — repo, staged |
| runner | `…/incoming/2026-08-03-longitudinal-distance-keeping/run_discrimination_control.py` — repo, staged |
| **result JSON** | `…/incoming/2026-08-03-longitudinal-distance-keeping/raw/dlead1_discrimination.json` — repo, staged |
| intake | `…/incoming/2026-08-03-longitudinal-distance-keeping/INTAKE.md` — repo, staged |

**Nothing is stranded on a pod or in a worktree.** Hardware: dev-box CPU only; 125.3 s for the control;
$0. Why not the eval pod: the job is 2 minutes of pandas/numpy over local label zips — a GPU would sit
idle, and `tanitad-eval` refused connection this run anyway (banner exchange / connection refused).
