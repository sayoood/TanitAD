# PREREG: the anatomy of P2's zero tail on the full navtest (written 2026-10-15, BEFORE any count below was computed)

**Question.** Of P2's 509 zero-PDMS tokens (12,146-token navtest, NAVSIM v1.1, **T1**), is the zero a SELECTION failure
(a non-zero proposal existed among the 64, the scorer picked a zero one) or a PROPOSAL failure (all 64 are zero)?
Selection failure points to scorer-side levers (P7.1, P7.2). Proposal failure points to proposal-side levers.

## 0. What I had seen before writing this (so the reader can judge the pre-registration)

1. `eval/RESULT_P2.md`: P2 = 88.82 on 12,146 tokens / 136 logs, "zero-PDMS tokens fell from 529 to 509".
2. `eval/RESULT_C3_SELECTION_HEADROOM.md` (u400, **not P2**): 455 / 529 zeroed tokens rescuable (86.0 %). `REVIEW_7 §2.4`
   (the pre-SFT final model, **not P2**): 793 of 983 zero picks had a safe proposal in the fan; best of 64 = 97.07.
   ⇒ I EXPECT a SEL-heavy picture. The decision rule below does not depend on that expectation.
3. The P2 per-token CSV header, its first 20 rows, its byte layout (CRLF original vs LF repo copy) and its hashes.
4. File listings and array shapes of every input named in §2. **No P2 count, split, overlap or concentration was computed.**

## 1. Finding that shapes the whole design (MEASURED by inspection, 2026-10-15)

**P2 has NO all-64 per-proposal dump.** `stage2_p2_full.npz` holds `token, fingerprint, poses[12146, 8, 3], sampling, arm`:
ONE pose per token (the pick). Its `.report.json` holds summary statistics of the proposals (`ade_oracle_m` etc.) and no array.
`data/refe_navtest/proptable/` on E: and D: holds only `navtest_final` (the pre-SFT final model) and the `sub200_*` arms.
Therefore **SEL / PROP cannot be measured on P2 from banked data.** The rest of this file fixes how I answer anyway, and
how every number is labelled. The two arms that DO have an all-64 table:

| role | arm | table | coverage |
|---|---|---|---|
| **PRIMARY proxy** | SFT-4 arm B u400 (88.25; P2's own comparator, same inference settings: route fix on, heading repair on) | `raw/2026-10-11-c3-selection-headroom/table.npz` (PDMS + 6 sub-scores x 64) | 1,058 tokens: ALL 529 u400-zero tokens + 529 random non-zeroed |
| SECONDARY proxy | pre-SFT final model (83.24; no route fix) | `raw/2026-10-04-lane-discipline/lane_census.jsonl.xz` (NAVSIM GT per proposal, 64 + PDM-Closed + human) | all 12,146 tokens |

## 2. Inputs (fixed now)

| input | path | integrity check |
|---|---|---|
| P2 per-token csv (**the dump used**) | `D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv` (CRLF original; the repo copy `raw/p2_readout/p2_full/arm/` is content-equal after CRLF to LF) | sha256 `61073a61e44a3dcf41fad6dc8c5b261ce97eb8f8d8b49c1adc077dd7c04b61e6` = manifest `csv_sha256`; md5 `bed5687f...` = `MD5SUMS.txt` |
| u400 per-token csv | `raw/p2_r2_instrument/baseline/u400_navtest_full.csv` | mean 88.25 within 0.01 |
| token to log map | `raw/p2_r2_instrument/baseline/navtest_full_tokens.json` (`token_log`) | 12,146 tokens, 136 logs |
| P2 seam (pick poses) | `raw/p2_readout/p2_full/arm/stage2_p2_full.npz` | 12,146 rows |
| u400 all-64 table | `raw/2026-10-11-c3-selection-headroom/{table.npz,tokens.json,proposals.npz}` | `table.pdms[pick]` equals the u400 csv score on all 1,058 (C3 gate G2) |
| final all-64 census | `raw/2026-10-04-lane-discipline/lane_census.jsonl.xz` | 12,146 records; `pdms[pick+1] == csv_pdms` |
| situation tags | `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz` (`ego_statuses[-1].driving_command`, `ego_velocity`) | the export C3 used; **not a field of the P2 dump** (the dump has none), joined by token |

**Gates (any failure STOPS the run, no result is written):**
- G1: mean of the 12,146 P2 token scores x100 is within 0.01 of 88.82, and equals the csv's own `average` row.
- G2: P2 token set equals the `navtest_full_tokens.json` set; 136 logs; u400 mean within 0.01 of 88.25.
- G3: P2 zero-token count equals 509 (RESULT_P2); u400's equals 529.
- G4: u400 table at the pick equals the u400 csv score on 1,058 / 1,058 (max |d| <= 1e-9).
- G5: census pick score equals `csv_pdms` on every token.

## 3. Definitions (fixed now)

- **zero token**: `score == 0.0` exactly in that arm's per-token csv (the `average` row excluded).
- **SEL** (token): `max_k PDMS_k > 0` over the arm's 64 proposals. **PROP**: all 64 `PDMS_k == 0`. (The brief's definition.)
  Sensitivity, reported beside it and never in place of it: **SAFE** = some proposal has `NC == 1 and DAC == 1` (C3's "rescuable").
- **Cause of a zero (Q2)**: at the pick, `DAC == 0 only` / `NC == 0 only` / `both` / `neither` (PDMS = 0 with NC and DAC non-zero).
  Whether DDC gates is TESTED, not assumed: the count of rows with `DDC == 0 and score > 0` is reported (> 0 means DDC does not gate).
- **oracle-on-SEL gain** = `sum over SEL zero tokens of max_k PDMS_k / 12,146`, x100 = PDMS points. (A zero token contributes 0 now, so
  the gain is the sum of the best-of-64.)
- **TTC token (Q4)**: a token with `score > 0` and `TTC == 0` at the pick. (On a zeroed token the TTC term is moot; the count of
  zeroed tokens that also read TTC = 0 is reported.) **SEL_TTC**: some proposal has `TTC == 1 and PDMS > 0`; **PROP_TTC**: none.
  **TTC headroom** = `sum over SEL_TTC tokens of max(0, max over {k: TTC_k=1, PDMS_k>0} PDMS_k - PDMS_pick) / 12,146`, x100.
- **Population weights for u400's table**: its non-zeroed stratum is a 529-token random sample of 11,617. Any quantity that touches the
  non-zeroed stratum (all of Q4 on u400) is Horvitz-Thompson weighted by 11,617 / 529 and labelled ESTIMATED. The zero stratum is the
  complete population (weight 1) and its numbers are exact.

## 4. The questions and the bridge to P2

**Q1 SEL vs PROP.**
- **Q1a, MEASURED on u400** (its own 529 zero tokens, all in the table): n_SEL, n_PROP, share, oracle-on-SEL gain.
- **Q1b, MEASURED on the final model** (its own zero tokens from the census): same.
- **Q1c, INFERRED for P2.** Each of P2's 509 zero tokens is classified by a PROXY fan, in four pre-registered variants:
  - **V1 hierarchy:** the u400 table if the token is in it, else the final census. The tier of every token is reported.
  - **V2:** final census only.
  - **V3-lo / V3-hi:** the u400 table only; tokens it does not cover counted PROP (lo) or SEL (hi). For the gain, an uncovered token adds 0 (lo) or its census best-of-64 (hi).
  - The gain for a SEL token is the proxy fan's best-of-64 PDMS.
- **Bridge strength** (committed): tier-1 coverage = (P2 zeros present in the u400 table) / 509. `>= 75 %` strong, `50-75 %` partial, `< 50 %` weak.
- **Transfer checks (reported, not gating):** T1 token persistence `|P2z & u400z| / |P2z|` (also vs the final zeros); T1b agreement of the SEL/PROP label between the u400 table and the final census on the tokens both cover; T2 how close P2's pick is to u400's 64 proposals on the 1,058 C3 tokens (min over k of max-abs pose difference; share <= 1e-3 m and <= 0.5 m).

**Q2 which gate.** The cause split of P2's 509 zero tokens, overall and per SEL / PROP (V1 and V2). The same for u400 (MEASURED) and the final model (MEASURED).
Within PROP tokens, what the fan fails on: all 64 `DAC == 0`, all 64 `NC == 0`, or a mixture.

**Q3 concentration (P2, with u400 beside it).** k50 = the smallest number of logs whose zero counts sum to `>= ceil(509 / 2)`; the top 10 logs by zero count with their token count and zero rate; the overlap of the top-10 lists with u400's.
**Control (committed):** a permutation null that shuffles the zero labels over all 12,146 tokens preserving the total (10,000 draws, seed 20260927), reporting its k50 median and 95 % interval, so "half in k logs" has a reference.
Tags: zero count and zero rate by `driving_command` (left / straight / right / unknown) and by ego speed v0 bins [0, 0.5), [0.5, 3), [3, 8), [8, 14), [14, 99) m/s. Only tags that exist in the export are reported; none is invented.

**Q4 the TTC piece.** The Q1 / Q2 treatment for TTC tokens: P2's TTC-token count at the pick; its SEL_TTC / PROP_TTC split under V1 and V2; the TTC headroom. MEASURED on u400 (ESTIMATED, weighted) and on the final model (MEASURED, full census).

## 5. Estimator

Point counts are exact. Any share, rate or gain quoted as a population claim carries a **log-cluster bootstrap over the 136 logs**:
10,000 resamples, seed 20260927, percentile 95 %, ratio estimators `sum(a) / sum(b)` computed on the same log draw (`rng.integers(0, 136, (10000, 136))`).
It answers "would another draw of LOGS say this?" and nothing else. It does NOT answer "would another training run say this?" (H-ESTIM-SEED-1): P2 is one seed, and the proxy fans are other models' fans.

## 6. Committed reading (verbatim from the brief)

- **SEL >= 50 % of the zero tokens AND oracle-on-SEL adds >= 2.0 PDMS** ⇒ "the zero tail is mainly a SELECTION failure; scorer-side drivable / collision awareness (P7.1 / P7.2) is the ranked lever".
- **PROP >= 50 %** ⇒ "mainly a PROPOSAL failure; scorer-side levers are capped at the SEL share, and the proposal head is the lever".
- **Otherwise** ⇒ "mixed", with both shares. (This includes SEL >= 50 % with a gain below 2.0.)

**How it is applied (committed):**
1. To u400's own zero tokens (Q1a): a MEASURED reading about u400.
2. To P2 (Q1c) under each variant. The P2 reading is **INFERRED from proxy fans**, never MEASURED on P2. It is called ROBUST only if V1, V2, V3-lo and V3-hi land on the same branch; otherwise the report names V1's branch and states which variants disagree.
3. If the bridge is `weak`, the P2 reading is downgraded to "UNDETERMINED from banked data" and only the u400 and final readings stand.
4. The report states in its first line that P2's own all-64 table does not exist and what it would take to produce it.

## 7. What would invalidate the reading (stated in advance)

- A gate in §2 failing.
- T1b agreement below 80 %: the proxy fans disagree about which tokens have a safe plan, so a classification does not transfer between checkpoints of this family and the bridge is not supported.
- A deliberate-regression arm that fails to move the counts (the test file swaps the SEL / PROP rule and must go RED).

## 8. Cost of the direct P2 measurement (to be written into REPORT.md from the MEASURED C3 timings)

The proposals dump needs the P2 checkpoint on a GPU; the harness scoring is CPU. The estimate is linear scaling of the C3 run on u400 (`STATUS.json`: dump 2,275 s for 1,058 tokens, score 1,852 s for 64 columns at 4 workers) and of P2's own full seam (16,402 s for 12,146 tokens).

## Amendment 0 (2026-10-15, still before any count was computed): two provenance corrections to §1

Found by reading the banked seam reports, not by looking at any outcome. They change no definition, gate or rule; they change how far the proxies may be trusted.

1. **u400 and P2 do not share the selection rule.** `u400_navtest_full_seam.report.json` has `rule = navsim_v1_lane`; `stage2_p2_full.report.json` has `rule = navsim_v1`. They share `goal_fix = pdm_route` and `repair_last_heading = true`. §1's "same inference settings" therefore means those two, not the selector. The proposals are generated before the selector runs, but P2 is a different fine-tune, so its fan is not u400's fan.
2. **The final-model census is PRE-route-fix** (`lane_census.py` header: `refe_navtest_final.csv`; PREREG_LANE1 line 13: "pre-route-fix"). The route fix changed the goal on 3,386 of 12,146 tokens, so on those tokens the census fan was generated from a different goal than P2's. It is the weaker proxy and V2 must be read with that in mind.

## Amendment 1 (2026-10-15, before any SEL / PROP count was computed): gate G5 tolerance

The first run of `zero_tail.py` stopped at G5 (gates G1-G4 passed; nothing past the gates was computed). My G5 text said "equals", and the code enforced 1e-6. The census stores its per-proposal `pdms` rounded to 4 decimals (max distance from the 1e-4 grid 9.1e-13), while `csv_pdms` is full precision; the worst pick difference is 4.9998e-05, exactly the rounding quantum, on 8,080 tokens.
This is a property of the instrument, found by a diagnostic on the census file alone, not by looking at any anatomy outcome.

**G5 is restated:** `|pdms[pick] - csv_pdms| <= 5.0e-5 + 1e-9` on every token, AND `(csv_pdms == 0) == (pdms[pick] == 0)` on every token (zero status agrees), AND no census proposal has `NC x DAC > 0` with stored `pdms == 0` (so the rounding cannot turn a tiny positive PDMS into a false PROP).
Measured by the diagnostic: zero-status agrees on all 12,146; the count of proposals with `NC x DAC > 0` and `pdms == 0` is 0. The SEL / PROP definition is unchanged.
