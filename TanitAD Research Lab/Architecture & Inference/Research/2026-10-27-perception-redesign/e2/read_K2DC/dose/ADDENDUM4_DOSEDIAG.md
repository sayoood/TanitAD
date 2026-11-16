# K2DC addendum 4: the dose validity rule barely discriminates under the measured reading noise (MM 4a3d05, 2026-11-15 23:50Z)

**Status.** Written while K2DC trains (step 2251 of 6,000), before any firing at step ≥ 3,000 and before the E2 read. It is a
**post-hoc DIAGNOSTIC**: it was not in the frozen pre-registration. It **adds** a statistic beside the pre-registered rule and changes
neither the rule nor the E2 read rule.

Instrument: `C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc/dose_diag.py`.
It is banked into the repo with K2DC's read.

## Measured (K2DC on Thor, `runs/r9e2-K2DC/metrics.jsonl`, fields `dc_share_read` / `dc_w_before` / `dc_w_eff`)

| step | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 2250 |
|---|---|---|---|---|---|---|---|---|---|
| share read | 0.1075 | 0.0518 | 0.2280 | 0.3220 | 0.0631 | 0.0612 | 0.2964 | 0.0695 | **0.6969** |
| w after | 23.83 | 25 (clamp) | 18.23 | 9.31 | 12.25 | 14.34 | 7.86 | 9.35 | **2.45** |

* **Noise.** The readings after the first correction have a log-sd of **0.989**: each one-batch reading is off by a factor of about
  2.7. The fixed-weight probe (addendum 3) had already shown ×4.8 at w 25, so this is mostly estimator noise, not dose drift.
* **Centre.** The geometric-mean share is **0.146**, firing-bootstrap 95 % CI **[0.078, 0.280]** (n = 8, MEASURED).
* **In band.** 4 of 8 readings fall inside [0.05, 0.15].

## What the noise does to the rule (ESTIMATED: lognormal reading, measured log-sd, independent firings)

The rule is: in band at ≥ 2 of the 13 firings at steps 3,000 … 6,000.

| true share centre | P(one reading in band) | P(rule passes) |
|---|---|---|
| 0.10 (on target) | 0.417 | 0.991 |
| 0.146 (measured) | 0.371 | 0.979 |
| 0.20 (2× target, outside the band) | 0.305 | **0.941** |
| 0.40 (4×) | 0.143 | 0.574 |
| 0.80 (8×, about where K2 sat: 0.39–0.81) | 0.043 | 0.104 |

⇒ **The rule refuses a K2-sized overdose but cannot tell "held at 0.10" from "held at 2× the target".** A DOSE-VALID verdict
from it is weak evidence that the dose was held in band.

## Committed now, both outcomes, before the read

1. **The rule is reported exactly as written**, as DOSE-VALID or DOSE-VOID.
2. **The diagnostic is reported beside it.** It is the geometric-mean share and firing-bootstrap 95 % CI of K2DC's readings at
   steps ≥ 3,000 (13 firings), plus the same over all readings after step 250.
   * **CI upper bound < 0.15:** "dose held at or under the band's top" (DIAGNOSTIC).
   * **CI lower bound > 0.15:** "rule VALID but the dose sat ABOVE the band". The dose-hypothesis read is flagged as weakened, never
     upgraded.
   * **Otherwise:** "not resolved by n = 13".
3. **This statistic decides nothing.** The E2 verdict follows its own rule (REFCV9_DESIGN 4.4) unchanged.

## For E3 (named, not built)

* **Same distribution, 4× less averaging.** At cadence 1000 each firing still sets w from ONE noisy reading: the solved k depends on
  that batch's |a|, ⟨a,b⟩, |b|. The draw of w is unchanged, but it is held 4× longer, so the run averages 73 draws instead of 293.
  MEASURED example: step 2250's reading of 0.70 cut w to 2.45 for the next 250 steps; at 1000 it would hold for 1000.
* **The next lever.** A **smoothed controller**: solve k on the geometric mean of the last N readings' implied k, or on summed
  |a|, ⟨a,b⟩, |b| over N firings. It costs nothing per firing, because it reuses the readings already taken.
  * It is a **design change**, so it needs its own pre-registration, the launch-gate entries and a PI decision.
  * **Not proposed for E3 now.** E3 must stay the K2DC that E2 reads. This lever is ready if K2DC reads DOSE-VOID, or reads NO-GO
    with the diagnostic above the band.
