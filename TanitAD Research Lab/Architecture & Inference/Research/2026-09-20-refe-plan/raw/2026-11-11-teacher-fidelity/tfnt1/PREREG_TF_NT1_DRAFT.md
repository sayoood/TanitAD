# PREREG TF-NT1 (DRAFT, NOT FROZEN): the N7-T3 cascade on the navtrain labels of the REFe student

Status: DRAFT by a REFe agent, 2026-11-15, for the Master Mind to edit and freeze. Nothing was run. **No navtest data is used anywhere in this card.** Design basis and evidence: `TFNT1_STUDY.md` (same folder). Evidence classes: M / P / I / E / H.

## 1. Question
Can the label table that REFe's P5-C successor imitates be made **paper-level at the label tier**: the RJ5 teacher + P5-R labels of the rank-0 rows of the P7 bank, with every tracked-builder zero-DAC label repaired by the N7-T3 cascade (tiers T0-T3, tolerance certificate, no human-log label, no T4), without harming NC or DDC on any row, and with the tracker-aware check passed on every edited row?
This is a **label-construction** question. It makes **no driving claim and no student claim**.

## 2. Population, scorer, tier, estimator
- **Population:** the **92,896 rank-0 rows** of `train_grow_cmd_3c_p7` (`targets_sha256 92af8d66…`), RJ5 rows (jerk 5.0) merged by `rj5_merge.py`, labels by the unchanged dual labeller (Tc20s, then P5-R lambda 3.0). 877 + 198 logs. Rank-1 rows (57,707) keep the identity label (`source: raw_rank1`); rank policy `rank0`.
- **Rows on `teacher_device: None` bank rows (189 local, more in A/B/C):** named in the report and reported separately; the R0 identity read uses `cpu` rows only.
- **Scorer:** the TanitAD builder tracked scorer (NAVSIM v1.1 PDM components, LQR tracker, scene = D9 parts or DB rebuild). **Tier T1, builder-side; not the pinned harness** (no navtrain harness cache exists).
- **Estimator:** none for the primary bars (counts over the full population, deterministic arms, C-DET). Where a rate is quoted with an interval it is a log-cluster bootstrap over logs (10,000 draws, seed 20260927) and answers "another draw of logs" only. An exact binomial interval is used for repair counts, as TF-N5.
- **Evidence class of results:** MEASURED, builder tier.

## 3. Arms
| arm | definition | role |
|---|---|---|
| **A0** | the RJ5 + P5-R table, no repair (identity cascade) | comparator; = `p5r` labels |
| **A1** | A0 + REPAIR-D + L1 (the N5-A construction applied to navtrain) | intermediate |
| **N7-T3** (PRIMARY) | A1 + cascade T0-T3, tolerance certificate, **no human label**, no T4 | the deliverable table |
| **N7-X** | deliberate regression: cascade with the certificate inverted (DAC = 0 required) | must change **0** rows, bit-identical to A0 |
| **N7-ID** | identity control: the cascade run with all levers disabled | must equal A0 bit for bit |
Cascade parameters are the frozen `n7_lib.LEVERS` / `n7_select.TIERS` of `TF-N7` (pins `n7_lib dfd5c7c4`, `n7_select bf2cbfa4`); **no lever is re-tuned on navtrain data**. A flag whose default disables a lever is named in argv or the arm did not test it (G-DVB rule).
Route: `n7_nt_build.py derive --route cache --metric-cache-root <D9>` on the pod (O2 must pass first); local 198 logs re-derived as C-LOCAL.

## 4. Bars (label side; committed before any NT1 number)
- **B1 residual (PRIMARY).** In N7-T3, tracked-builder **zero-DAC labels <= 0.10 % of rank-0 rows (<= 92 of 92,896)** (the same rate as the paper-level navtest bar, 12 of 12,110), **and** >= 70 % of the P5-R zero-DAC rows repaired. Stretch (reported, not a bar): <= 0.05 %.
- **B2 no harm.** 0 edited rows with builder NC or DDC below the unedited label's; 0 certificate violations on re-score; N7-T3 never touches a row that A1 already certified (cascade guarantee, C-ORDER).
- **B3 tracker-aware gates on every edited row** (G-T1 to G-T4 of `TFNT1_STUDY.md` s5). Bar: 100 % of rows in the final table pass, or revert to the unedited label (`p5r_unrepaired`); the reverted count is reported and **B1 is read after the reverts**.
- **B4 table gates.** TB-0 to TB-7 of `PREREG_P5C.md` s4.2 on the final table (TB-6 rank-0 <= 2x acc and heading rate; coverage 92,896 / 92,896 or declared `require_coverage`; fallbacks <= 1 %), and the `--rank-policy rank0` meta.
- **B5 report, no bar.** Tracked and transparent PDMS per edited row and per tier (G-T5), TD, TTC and C change (tolerance tolerated), per-tier counts T0/T1/T2/T3, blocking class per remaining zero-DAC row, and the four families of the edited rows' own labels vs unedited (STRATEGIC: PAUSED, PI 2026-09-19, never a pass, never dropped).
- **K:** none proposed (label-side, deterministic, no navtest, as `REPAIR_NT.md`). MM may open an entry.
- **Wording (binding):** every number tracked **and** transparent; "survives the benchmark's tracker"; banned: "drives better", "safer". Labels use privileged data (map, DB ego, agents); inference of any student stays vision-only.

## 5. Controls (all must pass; each needs a deliberate-regression arm that goes RED)
- **C-DERIV:** 200 smallest-sha1 tokens, `R(traj)` rebuilt from D9-parts ego == the banked P5-R label, bar **0.0 m** (mutation: cache ego forced to the wrong state must read 0 of 200, as the 6.36 mm void). Decides route O2.
- **C-LOCAL:** the pod reproduces the dev-box REPAIR-NT result on the 198 logs: **90 zero-DAC, 73 REPAIR-D repaired, 17 residual (same tokens)**, and the cascade picks of `sel17` (13 of 17 certified T0-T3), bit-identical.
- **C-CACHECOV:** D9 parts exist for 92,896 of 92,896 rank-0 keys; missing keys are counted and REFUSE the arm.
- **C-CACHEDB:** on 500 rows (log-spread, including all 90 local DAC-0), cache-route builder score == DB-route builder score on DAC, DDC, NC (the 1-15 cm object-position difference may move NC at the margin; any difference is named, bar: DAC and DDC 0 differences, NC <= 2 rows).
- **C-COV, C-KEY:** every label keyed to the bank row's own traj sha (re-key per O3), all rows resolve, `TableRefused` never raised.
- **C-DET:** the cascade run twice on every residual row, bit-identical.
- **C-ID (identity):** every row outside the edited set is bit-identical to A0; N7-ID == A0.
- **C-REG:** N7-X changes 0 rows and equals A0 bit for bit.
- **C-SELECT / C-CERT-B7 / C-ORDER / C-N6SAME:** the TF-N7 build controls, unchanged (`n7_nt_build.py` s6 of its card).
- **C-REACH:** every tier T0-T3 executes on a real navtrain row (the TF-N7 fixture `fixture_navtrain_tiers.json`); a mutated tier list drops its rows.
- **Launch gate:** if any training run reads this table, it needs a PASS token with the table sha in the data manifest, and a G-DVB entry `traj_label_table` (`PREREG_P5C.md` s10).
- **Disk/memory guards:** run tree on the pod volume; dev-box steps only when free virtual memory >= 10 GB per worker.

## 6. Outcomes, both directions (Rule Zero: no branch ends at a verdict)
| outcome | condition | what it means | next arm, same turn |
|---|---|---|---|
| **REACHED** | B1-B4 hold | the student's navtrain label table is paper-level at the builder tier: no zero-DAC label above 0.10 %, no harm, all edits tracker-checked | hand the table (sha) to the P5-C-RJ5 successor as its label; student read is that card's, with its own bar B = 0.7031 and replicate |
| **PARTIAL (`residual_over_bar`)** | B2-B4 hold, B1 residual > 0.10 % | the cascade's navtest rescue rate did not transfer | read the blocking class: (i) no DAC-1 candidate -> map-feasible path search (the N7 card's lever i); (ii) NC-blocked -> joint longitudinal + lateral search; (iii) DDC-blocked -> PI decision on the DDC clause; (iv) T4 only with PI consent; run lever (i) on the residual list in the same turn |
| **HARM (`harm`)** | any edited row with NC/DDC lower | a tier hurts | name the tier, drop it, re-run the build only (cheap: ~1-5 CPU-h), re-read |
| **TRACKER (`tracker_gate`)** | some edited rows fail G-T2..G-T4 | edits are not tracker-safe | revert those rows to unedited; report B1 after the revert; if B1 now fails, as PARTIAL |
| **INVALID** | any control fails, in particular C-DERIV or C-CACHEDB | the cache route is not the DB route | fall back to the DB route on the failing logs only (re-fetch only logs with DAC-0 rows, ~20-40 % of logs, E) |
| **NEGATIVE-TRANSFER** | C-LOCAL differs from REPAIR-NT | the pod and the dev box disagree (cross-env bit-identity bars fail, noted in memory) | calibrate the noise first; use quantile tolerances, not a new bar |
| **NO-OP** (a result, not a failure) | edited rows < 0.1 % and B1 already holds at A0 | nothing to repair | report the A0 rate and stop |

**What this card does NOT license:** any student-level or driving claim. The edited rows are ~0.5 % of rank-0 rows (E), too few to separate from the P5-C floor (B = 0.7031 PDMS); a student arm that differs from A0 only by these rows is not proposed.

## 7. Cost (ESTIMATED; see `TFNT1_STUDY.md` s3)
- Pod, cache route, after P7 and the P5-C label job: **16-31 CPU-h** (S1 labels 11-15, S2 scan 2-6.5, REPAIR-D 1.7-5.2, cascade 1.4-4.8); ~0.5-1 h wall at <= 32 cores. No GPU.
- Chain C roll (dev box, queued): ~11 CPU-h, behind W-A7.
- Dev box only (DB route): ~55-65 CPU-h plus a ~145 GB DB re-fetch; not recommended.
- Disk: run tree + tables < 1 GB; delete scratch at the end by literal path and log it.

## 8. Provenance pins (to fill at freeze)
bank `targets_sha256`; RJ5 merge sha (`3ca35564…` was the local-only merge; the 877-log merge is new); P5-R code blobs `16d2aaf0…`, `d0a4084b…`, lambda* sha `83988dda…`; dual labeller `07e84b1c…`; `n4_*`/`n5_*`/`n7_*` blobs (TF-N7 pins, `N7_PINS.template.json`); D9 cache listing digest; table module sha (`PREREG_P5C_A2.md` s44-45); K ledger tip.

## 9. Open points for the MM before freezing
O1 placement and window; O2 route validity (C-DERIV on the pod); O3 key convention (RJ5-keyed labels are refused by `Table.lookup` against the P7 bank); O4 chain C or a 0.85-coverage table; O5 no separate student arm; O6 T4 excluded; O7 K entry. Details in `TFNT1_STUDY.md` s8.
