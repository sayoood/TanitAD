# P2's zero tail is mainly a SELECTION failure (Master Mind, 2026-10-15, from the agent's RESULT.json)

**Why this was run.**
- The gap from u400 to 94.55 is mostly zeroed tokens: 4.15 of 6.30 PDMS (`D-REFE-GAP-1`). P2 still zeroes 509 of 12,146 navtest tokens.
- This package asks whether those zeros are a **SELECTION** failure (SEL: a non-zero proposal existed among the 64) or a **PROPOSAL** failure (PROP: all 64 are zero).

**Provenance.**
- `PREREG.md` was written before any count, including Amendments 0 and 1.
  - File order: `PREREG.md` 21:40:00, `zero_tail.py` 21:43:51, `RESULT.json` 21:44:04.
- Tests: `test_zero_tail.py`, **24 OK**. The Master Mind re-ran them: 24 OK.
- **Tier:** T1, NAVSIM v1.1 pseudo-simulation of an open-loop plan. Not closed loop.
- **Estimator:** log-cluster bootstrap over 136 logs, 10,000 resamples, seed 20260927. It answers "another draw of LOGS" only.
- The agent could not write this file (subagent write restriction). It is transcribed from `RESULT.json` and the agent's hand-back. Every number below is in `RESULT.json`.

**Dump read.**
- `D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv`, sha256 `61073a61…04b61e6`.
- This is the banked original. Its recomputed mean is **88.8184** against RESULT_P2's 88.82, and it has **509** zeros.

## Q1: selection or proposal?

**Committed reading: SELECTION.**

| arm | evidence | zeros | SEL / PROP | SEL share [95 % CI] | oracle-on-SEL ceiling, PDMS [95 % CI] |
|---|---|---|---|---|---|
| **u400** (own 64-proposal table) | **MEASURED** | 529 | **455 / 74** | 86.0 % [80.1, 91.7] | **+3.62 [+2.85, +4.55]** |
| final pre-SFT model (own census) | MEASURED | 983 | 795 / 188 | 80.9 % | +6.37 |
| **P2**, V1 bridge (headline) | **INFERRED** | 509 | **439 / 70** | 86.2 % [80.4, 91.8] | **+3.49 [+2.86, +4.18]** |
| P2, V3-lo floor (uncovered tokens counted PROP) | INFERRED | 509 | 273 / 236 | 53.6 % [47.4, 60.2] | +2.15 [+1.70, +2.66] |

**Why P2's row is inferred.**
- **P2 has no 64-proposal table**: its dump keeps one pose per token. Its SEL/PROP is bridged token by token from the u400 table, else from the final-model census.
- **The bridge is partial.** It directly covers 343 of the 509 zeros (67.4 %).
- **Bridge checks.** The SEL/PROP labels agree on 99.2 % of the 1,058 tokens both proxies cover. P2's pick lies within 0.5 m of a u400 proposal on 87 % of tokens.
- **Robustness.** The reading holds on all four bridge variants. On the V3-lo floor it holds on the point estimates only: both intervals straddle the bars.
- **Fan density.** On the SEL zeros the median fan holds **22 of 64 non-zero proposals**. It is not a needle in a haystack.
- ⚠️ **The oracle figure is a ceiling, not an expected gain.**

## Q2: which gate? (MEASURED on P2's own picks)

**Overall split of the 509 zeros.**

| gate | tokens | share |
|---|---|---|
| **DAC only** | **376** | **73.9 %** [66.7, 80.3] |
| NC only | 129 | 25.3 % |
| both | 4 | |
| neither | 0 | |

- **DDC does not gate.** 125 of 141 rows with DDC = 0 still score above 0.
- **Split by SEL / PROP** (via the bridge):
  - SEL = 309 DAC-only, 127 NC-only, 3 both.
  - PROP = 67 DAC-only, 2 NC-only, 1 both.
- **Collision zeros are almost all selection failures:** 98 % of NC-type zeros are SEL, against 82 % of DAC-type zeros.
- **PROP is mostly off-road:** 64 of the 70 PROP tokens have every one of the 64 proposals leaving the drivable area.
- **⇒ DAC-only SEL tokens carry +2.46 of the +3.49 ceiling.** The lever is a scorer that knows where the road is (P7.1, the drivable-area-aware scorer), with collision awareness second.

## Q3: concentration
- **Logs.** 16 logs carry half the zeros, against a permutation null of 26 [23, 29]. The top 10 logs hold 39.5 % (null 28.3 %), and 8 of them are also u400's top 10.
- **Commands.** Turning commands are 33.6 % of tokens but **54.8 % of zeros**. Zero rates: left 6.2 %, straight 2.9 %, right 7.8 %.
  - The tags come from the navtest input export, joined by token.

## Q4: TTC
- **394 non-zeroed tokens have TTC = 0** (MEASURED on P2).
- 96.7 % of them have a TTC-ok, non-zero proposal in the proxy fan (INFERRED). The ceiling is **+1.44 [+1.17, +1.72]**.

## Post-hoc (flagged in RESULT.json, not pre-registered)
- The zero set turns over between u400 and P2: 331 persist, 178 are new, 198 were rescued.
- **All 70 PROP tokens are persistent zeros.**

## What it means for the plan
1. **The zero tail is mainly a scorer problem.** Proposal generation is fine on about 86 % of the zeros. That confirms selection, not proposals, as the main gap, and ranks a **drivable-area-aware scorer** as the zero-tail lever.
2. **Why S1 matters here.** S1 tried this with a head on frozen visual features and failed K2 (`D-REFE-S1-READ`). P6 trains the trunk on PDM labels, which DAC is part of. P7.1 adds map supervision.
3. **The remaining PROP share** (about 14 %, almost all off-road fans) needs proposal-side work and is capped at about 1 PDMS.
4. **Turning tokens** carry a disproportionate share. Every P6/P7 read should stratify by command.

## To turn INFERRED into MEASURED for P2 (queued)
- Produce P2's own 64-proposal table on the 509 zero tokens. That is about 11–18 min of dev-box GPU plus about 15 min of 4-worker CPU scoring (agent's estimate from the MEASURED C3 timings).
- It needs `c3_run.py` run with P2's checkpoint and `RULE=navsim_v1`, behind the dev-box GPU lock.
