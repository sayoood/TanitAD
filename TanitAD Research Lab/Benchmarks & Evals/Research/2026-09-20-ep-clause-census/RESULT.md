<title>EP clause census</title>

# The ≤ 5 m ego-progress clause fires on 18.1 % of warmup scenes — so it is *part* of why standing still wins, not all of it

`Benchmarks & Evals · 2026-09-20 · Master Mind · MEASURED, 0 GPU, no scoring — read from the per-scene CSVs the EvalFlyWheel banked in 3b02a41.`
`Instrument: code/ep_clause_census.py → raw/ep_clause_census.json. Source: …/2026-09-19-navsim-refcv4b-bridge/raw/score_*.csv, column ego_progress_stage_two.`

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **1** | ⭐ **The clause is REAL and its size is now measured: it fires on 37 of 204 scenes = 18.1 %.** `pdm_scorer.py:231-236` sets EP to 1 for *every* proposal when the best rule-compliant proposal's progress is ≤ 5 m. Detection: the all-zero STOP plan has zero displacement, so its EP can only be 1.0 where the clause fired. | MEASURED |
| **2** | ⭐ **The control passes exactly.** On those same 37 scenes **every other arm also reads EP = 1.0 on every scene** — CV, A1, ECHO, A2 and A4, `all == 1.0: true` for each. An arm-*independent* value is the signature of this clause and of nothing else the arms could have done. | MEASURED (discriminating control) |
| **3** | ⛔ **AND THE CLAUSE DOES NOT EXPLAIN THE REST.** On the other **167** scenes a **zero-displacement plan still earns EP mean 0.2174 / median 0.1949**, and reads exactly 0.0 on only **7** of them. A stationary car collects about a fifth of the progress credit on scenes where the clause is *not* firing. **Mechanism UNKNOWN — not asserted here.** | MEASURED; cause OPEN |
| **4** | ⚠️ **The frames-blind arm has the HIGHEST progress of all**, mean **0.8289** / median **0.9676** on the non-firing scenes, with 79 scenes at exactly 1.0 — while leaving the drivable area on 86.3 % of scenes. It drives fast and straight, off the road. EP and DAC are pulling in opposite directions, and that is the whole shape of this benchmark's failure mode. | MEASURED |
| **5** | **A2 "vision-pure" reads 0.2245 against STOP's 0.2174** on the same scenes — a third independent confirmation that A2 *is* a stop, alongside the 202/204 displacement count and the −0.0001 score gap. | MEASURED |

## 1 · The numbers

**n = 204 scenes** — which is also the n the bridge RESULT reports, an independent agreement.

| arm | EP on the 37 **firing** scenes | EP on the 167 **remaining** scenes | ==0.0 | ==1.0 |
|---|---|---|---|---|
| **STOP** (all-zero plan) | **1.0000** (by construction of the test) | **0.2174** mean · 0.1949 median | 7 | 0 |
| **A2** vision-pure | 1.0000 | 0.2245 · 0.1986 | 5 | 1 |
| **A1** bar arm | 1.0000 | 0.4696 · 0.4150 | 7 | 25 |
| **ECHO** `ha0_ext` | 1.0000 | 0.4923 · 0.4264 | 7 | 28 |
| **CV** devkit reference | 1.0000 | 0.6559 · 0.7081 | 4 | 47 |
| **A4** frames-blind | 1.0000 | **0.8289** · **0.9676** | 4 | 79 |

## 2 · What this does to `H-NAVHARD-STOP-1`

The registered prediction is that **on a split whose starts are not slow, STOP must LOSE to A1**. Finding 1 was the reason to expect that; finding 3 is a reason it might fail:

> If the ≤ 5 m clause accounted for STOP's win, removing slow starts would remove the win. It accounts for **18.1 %** of scenes. On the other 82 % a stopped car already scores 0.2174 against A1's 0.4696 — so A1 *is* ahead on progress there, and STOP's overall win must be coming from the **safety multipliers** (NC, DAC, DDC, TTC), exactly as the bridge RESULT's attribution said.

⇒ **The prediction stands unchanged and is now RISKIER, which is the point of having registered it before the data.** ⛔ It is not amended, and this package does not touch it — a hypothesis edited after seeing new evidence is not a pre-registration.

⇒ **The sharpened question for the navhard run**, worth more than the original: report the clause-firing fraction on **both** splits *and* the per-scene EP of a zero-displacement plan on the non-firing scenes. If that second number stays near 0.2 on a fast-start split, the ≤ 5 m clause is a minor term and the real question is why EP credits a stationary vehicle at all.

## 3 · ⚠️ An error in this census, caught and recorded

The first run reported **206 scenes**. The CSVs carry **three aggregate rows** (`extended_pdm_score_stage_one` / `_stage_two` / `_combined`) beside the per-scene rows, and an unfiltered read treats them as scenes. The instrument now filters to hex tokens and **prints the number of rows it dropped per arm (3, 3, 3, 3, 3, 3)** so the filter's own effect is visible rather than silent. Same family as the `df` / cgroup / `step_s` traps: a true quantity read at the wrong scope.

⚠️ A second reading was also discarded: an earlier probe joined on `ego_progress_stage_one`, which is populated for the 16 stage-1 scenes only, and collapsed the sample to one aggregate row while still printing means and a correlation. **A statistic over n = 1 that prints without complaint is why the row count is now asserted before any statistic is taken.**
