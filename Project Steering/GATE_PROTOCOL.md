# TanitAD Restart/Continue Gate Protocol — STANDING

**Status:** binding default from 2026-07-20. Replaces the learning-curve power-law exponent gate
(D-031 / D-A7). Origin: 360° review W2 / P1. Tool: `stack/scripts/run_gate.py`.

---

## 1. The rule

**No run is killed or continued except at a pre-registered gate step, on a held-out metric, against
a threshold written down before launch.**

Concretely, before any multi-GPU-day launch:

```bash
python stack/scripts/run_gate.py register --run <arm> --gate-step <S> \
    --primary-metric ade_0_2s --primary-threshold <T> \
    --secondary "<mechanism>>=<v>" ... \
    --reference-run <ref> --reference-log <ref train_log.jsonl> \
    --compare-metric g_op_fwd_ade_m --tau 1.5 \
    --lever-family <family> --restarts-used <n> \
    --card "Project Steering/Gates/<arm>.card.json"
```

and at step `S`:

```bash
python stack/scripts/run_gate.py check --card "Project Steering/Gates/<arm>.card.json" \
    --log <run log> --eval-json <held-out taniteval result JSON> \
    --secondary-value <name>=<measured> ...
```

`check` refuses to return a verdict before the registered step (`NOT_YET`) and refuses to decide from
a train-log slope (`BLOCKED`). Both refusals are the point.

## 2. What counts as evidence

| Element | Prescription | Enforced by |
|---|---|---|
| **Primary** | **Held-out** val ADE@2s at an archived milestone (D-032 archives 5k/15k/20k/30k). Never a train-log slope. | `check` exits `BLOCKED` without `--eval-json` |
| **Comparative** | **Matched-step ratio** r(s) = M_new(s)/M_ref(s) at *identical* s, plus the assumption-free "the reference reached the new run's current value at step X". No power law, no extrapolation. | `run_gate.py ratio` |
| **Interval** | Bootstrap CI on every slope and every ratio. Decision-grade single-arm and paired intervals come from the **episode-cluster bootstrap** (`taniteval/ci.py`), not the deprecated overlapping-holdout SE. | `taniteval/ci.py`, `SlopeFit` |
| **Budget** | Compare at equal **GPU-hours**, not equal steps. `check` prints s/step and steps/GPU-hour for both arms. | `gpu_hours()` / `s_per_step()` |
| **Multiplicity** | **One** pre-registered gate step. Not "look at every milestone and decide". | card + `NOT_YET` |
| **Anti-regress** | **Two** restarts per lever family. A third failure **refutes the lever family**; it does not license more schedule tuning. | `restart_cap`, verdict `REFUTE_LEVER_FAMILY` |

## 3. The exponent is a diagnostic, and it cannot be quoted bare

An exponent may be logged. It may never decide a restart. Whenever one is printed it carries its
**fit window, R², n and bootstrap CI** — there is no code path in `run_gate.py` that returns a bare
float (`SlopeFit.exponent` raises below the R² floor; `SlopeFit.render()` always carries provenance).

- **R² ≥ 0.80** required before an exponent may be quoted at all. Below it: "power law unsupported",
  fall back to the ratio.
- **Extrapolation capped at 2×** the fitted range. `SlopeFit.project()` refuses beyond it.
- All arms compared must be **refit over identical step windows**.

### Why — measured on the actual logs, 2026-07-20

Same metric (`g_op_fwd_ade_m`), same two runs, different windows:

| fit window | flagship v1 | v3enc |
|---|---|---|
| 50–5350 | −0.421 (R² 0.566) [−0.507, −0.324] | −0.387 (R² 0.579) [−0.461, −0.332] |
| 1500–5350 | −0.663 (R² 0.375) [−0.851, −0.483] | −0.505 (R² 0.238) [−0.715, −0.316] |
| 2000–5350 | — | −0.738 (R² 0.299) [−1.022, −0.461] |
| 3000–5350 | — | −0.621 (R² 0.091) [−1.125, −0.138] |
| **1500–7500** | **−0.839** (R² 0.541) [−0.990, −0.689] | n/a |
| 50–29999 | −0.836 (R² **0.853**) | n/a |
| 1500–29999 | −1.021 (R² **0.877**) | n/a |

Three things follow, and they are the whole case:

1. **v1's famous −0.84 is the 1500–7500 window at R² 0.541.** It is below the floor and should never
   have been quotable. Its agreement with the full-run −0.836 is a coincidence of that window.
2. **On matched windows v1 and v3enc are statistically indistinguishable** (50–5350: −0.421 vs
   −0.387, CIs overlapping heavily). The claim "v3enc sits at the level that killed v2, far from
   v1's −0.84" is an artefact of comparing *unmatched windows*.
3. **The only R² ≥ 0.8 fits are full-run fits**, which by construction cannot gate an early decision.
   At every early window the power law does not describe the data — on *either* run.

## 4. What the corrected gate says about v3enc (2026-07-20, step 5350)

**VERDICT: `NOT_YET`.** The old gate had no basis to judge. The pre-registered 10k criterion stands
in its place, and it is `Project Steering/Gates/flagship-v3enc.card.json`:

- **primary** held-out ADE@2s ≤ 2.5 m at step 10 000
- **secondary** encoder speed-probe R² ≥ 0.55 · high-speed long overshoot ≤ 8 m
- (thresholds are the val-side gates already written down at
  `2026-07-19-flagshipv2-6k-diagnostic.md:196-199` — this protocol adopts them, it does not invent them)

Comparative diagnostics at step 5350, for information only:

- matched-step ratio v3enc/v1 = **1.834**, CI [1.759, 1.909], first 0.757 → last 2.224 (**widening**).
  v2's was 1.51 → **4.33**: v3enc is roughly **half as far behind** as v2 was.
- v1 reached v3enc's current `g_op_fwd_ade_m` (0.422) at step **450** → ~**12×** slower. v2's figure
  was ~30×.
- **Budget:** v3enc 10.22 s/step vs v1 10.89 s/step → 352 vs 331 steps/GPU-hour. Equal-step is
  ~equal-cost here (within 6 %); the concern that budget normalization would flatter the new arm does
  **not** bite for this pair. v3enc has spent **15.2 GPU-hours** to step 5350.
- restart budget: **1 / 2** for lever family `encoder-grounding`. One more failure exhausts it and
  refutes the family.

**Do not kill v3enc on its pre-5k exponent.** There is no admissible exponent to kill it on.

## 5. Relationship to the interval estimator

The comparative ratio's CI is a bootstrap over log points and is labelled `log_point_bootstrap` —
**diagnostic-grade**. Decision-grade intervals come from `taniteval/ci.py`
(`episode_cluster_bootstrap`, `paired_episode_cluster_bootstrap`) on held-out windows. Milestone
evals must therefore persist `windows_<key>.pt`, or the paired comparison the gate wants is not
computable afterwards.
