# Opponent Analyzer — Standing Goals (D-029)

> 1–3 concrete, measurable objectives with a target number and a deadline. Each run: advance a goal
> with a **measured** step and update its status. A goal with no movement for two runs is escalated in
> STATE, not silently carried. Created **2026-07-15** (run #3) — this file was missing (D-029 gap).

| # | Goal (measurable) | Target + deadline | Status (2026-07-15) |
|---|---|---|---|
| **G1** | **Scenario-database coverage & lifecycle.** Every documented opponent-failure *class* has an SC-entry, and ≥ 1 entry advances one lifecycle step per run. | **≥ 12 SC-entries with ≥ 3 at `spec-drafted`-or-beyond by W33 (2026-08-16).** | **On track.** 14 SC-entries exist; SC-01 live-measured (partial), SC-04 spec-drafted, **SC-13 spec-drafted this run**. → 2/14 beyond spec + SC-01 = 3. Target met early on count; now push data-sourced. |
| **G2** | **≥ 1 weak-spot scenario feed into the eval set per month** (intake pkg w/ telemetry oracle + passing tests), mirroring `work_zone_phantom`. | **1 new intake pkg / month; all offline tests green.** | **Met for this window.** SC-13 intake shipped (16/16 tests). Prior: SC-04 (11 tests, 07-24), work-zone (9 tests, 07-17). Next: SC-14 red-light (near-free off SC-04). |
| **G3** | **CNCE head-to-head readiness.** Convert the "no opponent publishes a compute-normalized causal-efficacy number" gap into a concrete, sourced comparison table the moment a rival discloses params. | **A populated CNCE comparability table (params × a causal-efficacy proxy) for ≥ 3 opponents by W34 (2026-08-23), the instant Metis/others disclose a param count.** | **Blocked on external disclosure.** Metis (2606.15869) still reports no params; NVIDIA Alpamayo 10 B/32 B known → 1/3 rows fillable now. No movement 2 runs on the *rival-side* number → keep watching Metis github; not escalated (the block is external, not ours). |

## How goals map to the program's top risk
Program top risk = the **single-camera driving-capability gap** (`Benchmarks & Eval/
DRIVING_DIAGNOSTIC_FRAMEWORK.md`). Our contribution is **adversarial, opponent-grounded eval
scenarios** that stress exactly the capabilities their recalls expose — SC-13 (stationary/same-lane,
the cheapest and most damning), SC-01 (work-zone), SC-05 (degraded-visibility). Each SC-entry that
reaches `live-measured` is a direct, falsifiable check on that gap.
