# Benchmarks & Eval — 2026-07-09 — SC-01 first-live-CARLA metric audit, LAL-v2, ecosystem deltas

> **Dating note.** The prior Benchmarks & Eval note is labeled `2026-07-16` (forward-dated to the
> project's W-calendar, W29). This run uses the **real calendar date, 2026-07-09** (the day the
> scheduled agent fired), and audits work committed **2026-07-08** (the first live CARLA run). It is
> therefore the *later* run by content even though its filename date is numerically below the previous
> note. KNOWLEDGE_BASE deltas are placed newest-first regardless.

**Run:** weekly Benchmarks & Eval agent (Thursday). **Loop:** iteration 1 of 4.
**Budget used:** 2 web searches, ≈1.4 h wall-clock — well under caps (25 searches / 4 h / 4 iters).
**Consumed this week (Mon→Wed + the live run):** Mon Tools&DevEnv (`2026-07-13`, CARLA-on-pod substrate);
Tue DataEng (Cosmos layout / DATASET_LANDSCAPE); Wed Architecture (`2026-07-08` bake-off harness +
spectral-sizing on the step-6500 ckpt); **the first live CARLA SC-01 run committed `2d87acb` (2026-07-08)**
— which is *in my discipline* (it runs my metric suite on real physics); the Opponent Analyzer SC-01/
Work-Zone-Phantom scenario; DECISIONS D-004/D-011/D-014/D-018/D-020; Phase 0 Plan §4. Every claim below
carries a source link or repo path (G-A).

---

## 1. Headline — my independent-test role earned its keep on the first live run

The **first live CARLA measurement of SC-01 Work-Zone Phantom** landed 2026-07-08
(`stack/experiments/p0-carla-workzone/suite_results_v1.json`): real vehicle dynamics, raycast occlusion,
measured per-tick latency, two *scripted* policy archetypes (honestly labeled — not our checkpoint yet).
First live rows: **OKRI 32.37 (reactive) vs 12.83 (world_model)**, **LOPS 0.0 vs 0.834**, TMS 0.006 vs
0.023, CNCE 8.68e5 vs 1.06e6. The run author flagged two instruments as needing v2 (commit `2d87acb`):
**LAL** (both policies read −0.7 — no discrimination) and the closure-incursion detector (read 0).

That flag is exactly the Benchmarks & Eval **independent-test duty** (Mission Plan; agent file #5): take a
live claim, recompute it independently with fresh seeds, and either confirm it or expose its fragility.
This run does all three — a shipped metric fix plus a measured audit (`audit_sc01.py` → `audit_results.json`,
local CPU, numpy-only, < 3 s, $0).

## 2. Implementation increment — LAL-v2 (anticipatory deceleration lead) — G-E, G-B2

**Intake pkg:** `Benchmarks & Eval/Implementation/incoming/2026-07-09-lal-v2-anticipation/`
(`lal_v2.py` + 7 analytic-ground-truth tests + `INTAKE.md` + the audit). Proposed target
`stack/tanitad/eval/metrics.py` next to `compute_lal`. **`pytest tests/` → 7 passed** (standalone).

**The defect (root cause).** LAL-v1 = `t_LoS − t_anticipation`, where `t_anticipation` is the first
`ego_jerk < −1.5 m/s³`. A good anticipatory policy slows *smoothly* on approach to a blind edge — a
comfort-bounded deceleration (ISO-2631-scale |jerk| ≲ 2 m/s³) that **never trips −1.5** — so LAL-v1
credits it with *no anticipation*. LAL-v1 measures reaction **hardness**, not anticipation.

**LAL-v2.** `t_anticipation` = onset of *sustained deceleration by speed drop* (first time ego speed
falls ≥ 15 % below the free-cruise reference and stays there ≥ 3 steps). It is the latent, pre-line-of-
sight generalization of the recognized **Time-To-Brake (TTB)** family: TTB/TTC require the hazard to be
*detectable*; LAL-v2 credits braking that begins **before** line-of-sight — the object-permanence edge
(H15). Non-breaking: v1 stays as a valid reaction-hardness instrument; the suite reports both.

## 3. Measured experiment (G-H) — the SC-01 audit, three numeric results

**A. LAL-v1 discrimination collapse — reproduced, cliff located.** I drove a realistic anticipatory
ease-off (16 → 9 m/s, prudent slow to ~60 % cruise, *not* an emergency stop) through a smoothness sweep
and scored both LAL versions against a reactive baseline (brakes only after LoS):

| ramp smoothness | peak decel jerk (m/s³) | trips v1 −1.5? | LAL-v1 anti | v1 separates? | LAL-v2 anti | v2 separates? |
|---|---|---|---|---|---|---|
| 1.0 s | −27.5 | yes | +3.6 | ✅ | +3.1 | ✅ |
| 2.0 s | −8.15 | yes | +3.5 | ✅ | +2.7 | ✅ |
| 3.0 s | −3.74 | yes | +3.5 | ✅ | +2.3 | ✅ |
| 4.0 s | −2.13 | yes | +3.4 | ✅ | +1.9 | ✅ |
| **5.0 s** | **−1.37** | **no** | **−999 (sentinel)** | ❌ | **+1.5** | ✅ |
| **6.0 s** | **−0.95** | **no** | **−999** | ❌ | **+1.1** | ✅ |
| **8.0 s** | **−0.54** | **no** | **−999** | ❌ | **+0.3** | ✅ |

Reactive baseline: LAL-v1 −0.10, LAL-v2 −0.30. **The cliff sits exactly at the −1.5 m/s³ threshold**
(between the 4 s ramp, peak jerk −2.13, and the 5 s ramp, −1.37). Once the anticipatory slowdown is
smooth enough to be *comfortable* — the regime a good world model actually operates in — LAL-v1 goes
blind and returns the no-reaction sentinel, i.e. the live `−0.7 / −0.7` null. **LAL-v2 separates the
anticipating from the reactive policy across the entire sweep** (+0.3 … +3.1 s vs −0.3 s). Falsifier
(pre-registered): if LAL-v2 could not separate a smooth anticipatory ease-off from a post-LoS hard brake,
the metric would be no better than v1 — it does, cleanly, with opposite signs.

**B. LOPS headline recompute — reproducible, but measures the noise model not the model.** SC-01's
world_model LOPS = 0.834 comes from `wm = gt + N(0, 0.3)` per axis, LOPS = mean `exp(−0.5·‖wm−gt‖)`. I
recomputed the population value analytically (per-step kernel **E = 0.8325**) and Monte-Carlo'd the
clip-mean across **5 000 seeds** for n_occluded ∈ {10, 20, 40, 80}: LOPS mean **0.8325–0.8328**, and the
committed **0.8338 falls inside the 95 % CI for every n** (e.g. n=20 CI [0.798, 0.867]; n=80 [0.815,
0.850]). **Verdict:** the 0.834 headline is *reproducible*, not a lucky seed. **P8 caveat, first-class:**
this only validates that the metric faithfully reflects the *injected* σ=0.3 tracking noise; it says
nothing about our real model (camera-driven ego is still blocked). And reactive's 0.0 is **structural**
(a policy with no latent estimate scores 0 by definition) — so the 0.834-vs-0.0 gap proves latent-tracking
**presence, not quality**. Real LOPS awaits the checkpoint-driven rollout.

**C. OKRI ≥3-seed power — my CI-separation rule made numeric.** The OKRI gap is 32.37 − 12.83 = **19.54**.
Two same-scene means separate at 95 % when `gap > 2·1.96·SD/√n` → `n > (2·1.96·SD/gap)²`:

| assumed per-seed OKRI SD | seeds for CI separation |
|---|---|
| 5 | 2 |
| 10 | 5 |
| 15 | 10 |
| 20 | 17 |

Under the KB's ~5-unit CARLA seed variance the 19.5 gap separates with ~2 seeds — but the **≥3-seed floor
rule holds**, and OKRI is in kJ/m units (not the 0–100 DS scale the "~5" figure came from), so its SD must
be **measured** on the first ≥3-seed pod run before any "beats baseline" claim. This is the actionable
sizing the rule demands; the table is a sensitivity guide, not a claim.

## 4. Research delta — benchmark ecosystem (weekly focus)

- **NAVSIM-v2 navhard leaderboard moved (Apr 2026).** The best *learned* method is now **DriveFuture =
  55.5 EPDMS** (arXiv [2605.09701](https://arxiv.org/html/2605.09701v1), future-aware latent world model);
  **DrivoR** with test-time trajectory optimization reaches **56.3 EPDMS** (arXiv
  [2606.07170](https://arxiv.org/html/2606.07170v1)), above the strongest learned method. Both sit above
  the PDM-Closed 51.3 baseline our LEADERBOARD carried. Note EPDMS added compliance sub-metrics (**DDC/TLC/
  LK**) and split comfort (**HC/EC**) ([navsim/docs/metrics.md](https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md))
  — the compliance sub-scores are the recognizable analogue of our H9 violation-rate / closure-incursion
  signal. — impact: LEADERBOARD open-loop block refresh / H9 framing.
- **Time-To-Brake (TTB) is the recognized anchor for LAL.** TTB/TTC (Euro-NCAP AEB, occlusion-AEB
  literature) trigger braking on a *detectable* hazard; the occlusion-AEB studies explicitly recommend
  *longer* activation thresholds when a pedestrian may be occluded
  ([ScienceDirect S0001457522002329](https://www.sciencedirect.com/science/article/abs/pii/S0001457522002329)).
  This is precisely the gap LAL-v2 fills: it credits braking that begins **before** line-of-sight, which
  TTB/TTC structurally cannot score (no detectable object → no TTB). Grounds LAL-v2 in accepted metrology
  rather than inventing a metric from scratch. — impact: LAL-v2 justification / metric-gap thesis.
- **DriveFuture = another latent-WM entry at the top of the board** reinforces the Opponent Analyzer's
  2026-07 read: "world model" is no longer differentiating (H0/H6). Our wedge stays hierarchy + efficiency
  (CNCE) + imagination (H15) + self-monitoring (H11). — impact: no ledger status change (P8).

## 5. Ledger / leaderboard / regulation-trace updates

- **LEADERBOARD.md:** (i) refreshed the open-loop NAVSIM-v2 navhard block (DriveFuture 55.5 / DrivoR 56.3,
  cited+dated; EPDMS sub-metric note); (ii) added a new **"Live scenario metrics (SC-01)"** block with the
  first-live rows — flagged single-seed / scripted-policy / LAL-v1-superseded-by-v2, per G-B1; (iii) added
  a **competitor efficiency block** (backlog #2 / W-05): GAIA-3 15 B (offline), Alpamayo-2 32 B (on-car),
  vs TanitAD 261 M live — sourced to the Opponent Analyzer profiles.
- **REGULATION_TRACE.md:** noted LAL/anticipation + closure-incursion as the *evidence substrate* for the
  ISMR "incident anticipation" and DSSAD event-trigger asks (the metrics that will populate the safety
  case). Full paragraph-level PDF extraction stays the open task.
- **HYPOTHESIS_LEDGER:** H15 gains an *instrument-hardening* evidence row (LAL-v2 makes the anticipation
  edge measurable on smooth braking; LOPS recompute confirms metric fidelity) — **no status upgrade**
  (nothing measured on our real model yet, P8).

## 6. Self-critique (quality gates)

- **G-A** every claim carries a link or repo path ✅  **G-B** actionable recs tied to gates/WP/H (§7) ✅
  **G-C** KNOWLEDGE_BASE updated newest-first ✅  **G-D** HYPOTHESIS_LEDGER touched (instrument-hardening,
  no unearned upgrade) ✅  **G-E** verifiable increment: LAL-v2 + 7 standalone tests + runnable audit ✅
  **G-H** measured experiment with numbers, hardware (local CPU), wall-clock (<3 s), cost ($0), falsifier
  verdicts ✅  **G-F** session-end ritual (STATE/commit/push) below.
- **G-B1** new leaderboard numbers carry source+date+eval-condition; live SC-01 rows flagged single-seed/
  scripted; open- vs closed-loop kept separate ✅  **G-B2** LAL-v2 ships 7 analytic-ground-truth tests ✅
- **Honesty (P8):** the audit's job was to *challenge* a headline the project just committed. It confirmed
  LOPS is reproducible **and** stated plainly that the number reflects an injected noise model, not our
  model, and that the reactive 0.0 is structural — resisting the temptation to read 0.83-vs-0.0 as an edge
  win. LAL-v1 is exposed as blind to smooth anticipation and fixed, not quietly kept.
- **Gap (recorded):** no metric is claimed on a real TanitAD checkpoint — camera-driven ego is still
  host-blocked on pod2 (PROJECT_STATE). LAL-v2 and the OKRI seed rule are ready the moment that rollout
  produces telemetry. The closure-incursion detector (also flagged in `2d87acb`) is **not** fixed this run
  — carried to backlog #3 (needs a lane-polygon + collision-sensor hook on the CARLA side).

## 7. Actionable recommendations (each tied to a hypothesis / gate — G-B)

1. **[WP6, G0.6 — ready for triage]** Integrate LAL-v2 into `stack/tanitad/eval/metrics.py`; relabel
   LAL-v1 as "reaction-onset latency" and report both. Owner: MVP orchestrator. Falsifier: if the pod's
   next SC-01 run still reads a single LAL number, the relabel didn't land.
2. **[SC-01 re-run — adopt now]** The next CARLA SC-01 run must be **≥3 seeds**, report OKRI/LOPS/TMS as
   **mean ± CI**, and emit LAL-v2 alongside v1. Measure OKRI's per-seed SD (Result C) to size future runs.
3. **[Backlog #3, Friday co-own]** Fix the **closure-incursion detector** (lane-polygon + collision
   sensor) — the H9 rule-compliance signal, still reading 0 on the reactive run.
4. **[LEADERBOARD hygiene, standing]** Keep the live-SC-01 rows flagged scripted-policy/single-seed until
   the checkpoint-driven, ≥3-seed rollout replaces them; never promote them to an edge claim.
5. **[Backlog P1 #5 — carried]** Metis (arXiv 2606.15869) CNCE head-to-head deep-read + the WP.29
   paragraph-level extraction remain queued; neither blocks Phase 0.

## 8. Backlog re-prioritization

Promoted **LAL-v2 integration** and **≥3-seed SC-01 re-run** to P0 (both unblock G0.6 / decision-grade
live metrics). Closure-incursion fix stays P0 (H9). Competitor efficiency block: **done** this run.
NAVSIM/Bench2Drive entry audit and Metis deep-read stay P1. See `BACKLOG.md`.
