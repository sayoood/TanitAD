# NOTES: LAUNCH-INTENT v2 package (PREREG draft v2 + its reader)

Built 2026-10-28 by a TanitAD sub-agent for the Master Mind, from `MM_REDESIGN_2026-10-28.md` (the spec) applied to the v1 draft in `dagland/launch_intent/`. Where the draft and the spec conflict, the spec wins.
Nothing here was committed, pushed or landed; box 2 was not contacted; no GPU, training or rollout ran. CPU only (`CUDA_VISIBLE_DEVICES=-1`, at most 4 threads), never more than two pytest runs at once.

## 1. What is proven (MEASURED; every number has its artifact)

| claim | evidence | artifact |
|---|---|---|
| The enumeration is unchanged: 524,288 cells, each matching exactly one row; per-row counts equal DAG-ORC's and the v1 draft's; the 16 guarded (R-CRAWL, R-MOVE) cells map as DAGGER2 A1; the per-seed AND over two pairs checked on all four combinations of every predicate and guard, and on 200,000 seeded pair-level draws (all 8 rows reached, each draw one row) | `enumerate_cells_li.py` rc 0, `FAILS: none`; its counts section is identical to v1's `enumerate_cells_li.out` (lines 1-12) | `enumerate_cells_li.out`, `tools/enumerate_cells_li.py`, `tools/enumerate_v1_vs_v2.diff` |
| The enumerator's own checks bite: 23 of 23 mutants exit 1 | `enumerate_mutation.py` | `enumerate_cells_li_mutation.txt` |
| The reader's pure layer and its wiring are tested: 87 tests pass (75 on the pure layer, 12 end to end through `main_read` on a scripted fake heavy layer). Every outcome row is reached; the bar, `clears`, `fires`, the K-SHARE band (+-0.02), the K-MASK band, R-T0 / R-KEEP (<= 0.0604), the slow-draw thresholds (strict 0.5) and every per-seed AND are tested on both sides of their boundary | `pytest -q test_launch_intent_read.py` | `evidence/pytest_final.txt` |
| Mutation pass: 245 of 245 mutants of the reader are killed (0 survivors). Each mutant is the reader with ONE change, run against the whole test file with `-x`; the unmutated reader passes first | `mutation_launch_intent_reader.py` | `evidence/mutation_reader.json` |
| Dry run on banked real dumps (the dag-orc dumps stand in for the pairs: CTRL = DAG-OACC, REP = REP): the reader reproduces DAG-ORC's banked `A - CTRL`, `A - P10`, `DAG-OACC - P10` and `REP - CTRL` tables at both seeds, 19 rows x every key, **0 differences, max |delta| of delta / lo / hi = 0.0**; RF equals `RF_REP.json` and `DAGORC_READ.json` floors.RF; v1's bars `2 x max(F, RF0, RF)` are reproduced; the composition, handed DAG-ORC's own bars, equals the banked per-seed predicates of the main pair (flags, guards, R-CRAWL / R-MOVE rows, domain, fires, violations, T0 and KEEP points); `DAG-OACC - P10` equals `DAGOACC_READ.json` (max |delta| 0.0, separated flags equal); every (X, X) contrast reads exactly +0 [0, 0] (10 K-ZERO controls PASS); the REP licence PASSES on the real REP runs at both seeds; the slow-draw method reproduces `rep_s0_speedcells.json` on 20 cells with 0 differences and labels REP s0 alone SLOW-T | `launch_intent_read.py --dry-run` (204 controls, 0 failed, 516.4 s) | `evidence/dry_run.json` |
| The v2 PREREG carries, verbatim, the 8 outcome rows, the 7 per-seed predicate rows and the bar row the reader codes | `launch_intent_read.py --static-check --prereg PREREG_V8F_LAUNCH_INTENT.md` -> STATIC-OK | `evidence/static_check.txt` |
| The v2 crawl bars (1.1070 / 0.0526 / 0.0136 / 0.0096) are each below the DAG-OACC - P10 gap at both seeds; the v1 bars are above it at >= 1 seed | arithmetic on `P0_RF0.json` (sha256 `1608d70b...`) and `DAGORC_READ.json` (`70a7a3a4...`) | reader test `test_the_crawl_bars_are_reachable_below_the_ctrl_p10_gap_at_both_seeds`; PREREG section 6 |

## 2. What is NOT proven

* **No LAUNCH-INTENT run exists.** The paths that only a real new arm exercises (a real `--launch-intent-w` argv through K-PAIR, a real `summary.json` with the `launch_*` counters through K-SHARE / K-MASK, a real K-BUILD record, the logged lane and fresh-age T0 of K4 for A and A_rep) run on synthetic fakes and on the banked CTRL / REP records only. The key names are the staged trainer's (`summary.launch_n`, `launch_eligible`, `launch_rows`, `launch_intent_w`, `launch_weight_share`; `v8f_mini_train.py:2036-2038`), read from the v1 stage, not from a run.
* **The replicate-pair stand-in in the dry run is not a one-lever pair.** `A_rep` is DAG-ORC's A (seed s) against REP (seed 10 / 11): it exercises the machinery (tables, composition, controls, K-ZERO) and has no banked table to be compared with. Only the main-pair stand-in (A, CTRL), `DAG-OACC - P10`, `REP - CTRL` and the RF are reproduced against banked numbers.
* **The fake heavy layer scripts the rows.** It proves the wiring from tables to guards to the outcome word, the per-seed AND across pairs and the refusal paths. It does not exercise the DAG-ORC `Rig`; the dry run does, on the stand-ins.
* **"All mutants killed" is a statement about these 245 one-change mutants, not about every fault.** The first run of the pass left 11 survivors (`sep_worse_zero`, `fire_whole_labels`, `nobase_xp10_dropped`, `decide_arm_swaps_keep_nobase`, `reada_family_dropped`, `reada_crawl_m1_only`, `boot_default_rng_seed`, `share_w_unchecked`, `share_w_ratio_loose`, `share_counts_unchecked`, `core_b_needs_both`); each pointed at a missing boundary or column test, the tests were added, and the pass was rerun from the start on the final reader and tests (a second run left 3: `main_rf0_files`, `result_strategic_text`, `trim_hidden_rows`; closed the same way; the third and final run reads 0 survivors). Survivors in this class are findings, not noise. The heavy layer (`Heavy.load_slot / prepare / contrast / guard_rows / speed / finish`) is exercised by the dry run only and is not mutation-tested.
* **The reader was not run on the K4 / logged-lane path of a new arm.** `load_slot` reuses the DAG-ORC reader's own functions for that path unchanged (as the SPEEDSUB-T1 reader does).
* **Evidence classes inherited from the v1 draft** (the f measure, the simulated shares, the REP argv comparison, the staged trainer and its tests) are carried as the v1 package left them; v2 re-measured only what it states in PREREG section 14.

## 3. Readings this build chose where the spec is silent (PREREG section 0 item 8; confirm or name another)

(a) The replicate pair's domain D, for R-NEW and R-NOBASE, is every row not separated-worse for REP - P10; the crawl set C stays DAG-OACC's for both pairs.
(b) M-LAUNCH counts as "moved" only when it moves in both pairs at both seeds. The reader does not compute M-LAUNCH (below), so `next_launch` prints UNDECIDED.
(c) LAUNCH-INTENT-HI follows the same 4-run pair design as B (spec item 5 names B only).
(d) The slow-draw label (SLOW-T) needs all five T0 speed cells (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms`) separated downward (CI upper end < 0) with each shift below -0.5 m/s, strictly. The reader also labels the mirror, FAST-T (all five separated upward by more than 0.5 m/s): this build's addition, beside, deciding nothing. The bootstrap is the banked one (`rep_s0_speedcells.py`'s `boot`): one shared `default_rng(0)`, consumed in the order s0: REP, A; s1: REP, A, in the T0 npz's own dtype (float32; a float64 cast moves the 8th digit and the banked table is not reproduced bit for bit; the first dry run showed it).
(e) The v1 K-SHARE clause "`levers.launch_intent_w == 0.0`" for CTRL and REP is read as "absent, or exactly 0.0, and no `launch*` key in `summary.json`": MEASURED, neither banked `summary.json` has such a key (both runs predate the flag), so the literal clause could not pass. The dry run's real-record check passes under this reading.
(f) PREREG pinning: the reader pins the tables it codes verbatim (the outcome rows, the per-seed predicate rows, the bar row). A whole-file sha256 is supplied at read time with `--prereg-sha` once the Master Mind promotes the file; this build cannot know the promoted bytes.

## 4. Spec items judged not buildable here, named exactly

* **Not built, and listed in the result's `not_built` block (never a silent omission):** M-LAUNCH (X - Y of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, per pair), the anti-echo block, along-track / speed / L2 on the GT-launch subset and M2, and the T0 four families (`orc_families.py` pointed at the arms and their controls). They are v1 beside rows (PREREG section 7), not part of the redesign; the DAG-ORC reader they would extend carries none of them. Each is a build item.
* **The four families at T1-kin** are reported per family from the Read-A rows per pair (LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC; STRATEGIC not paused), never pooled; the T-head F1 and head accuracy are NOT COMPUTABLE from the paired tool's eval dumps and say so.
* Nothing else in the spec was judged infeasible. The 2.8 h box-2 estimate and the doubled dev-box rollouts are carried in PREREG section 13 as ESTIMATED, from DAG-ORC's measured run times.

## 5. Defects found and fixed while building (each with its root-cause class)

* **A zero that was a claim about the probe, K-SHARE:** the v1 text asked `levers.launch_intent_w == 0.0` of two runs whose `summary.json` has no such key; read literally it could never pass. Fixed as reading (e), with the control reading the real banked records.
* **A short-circuited `and` hid a missing row:** `compose_pair` read `X - Y` only when `X - P10` was already separated-worse, so a NOT COMPUTED row could pass R-NEW silently. All three tables are now validated before any predicate is read (mutation `compose_validation_dropped`).
* **A failed K-P0 crashed instead of refusing:** with no RF0 the bars cannot be built; `main_read` now writes the REFUSED record with every control read so far (test `test_e2e_k_build_k_p0_and_static_are_controls`).
* **A dtype cast broke an exact reproduction:** see (d).
* **A dry-run stand-in failed its own SUMMARY control:** the replicate-pair stand-in is DAG-ORC's A at seed s, expected at seed s, not 10 + s.
* **A one-seed B was silently left out:** B present at seed 0 only passed the old half-present check and `read_core` ignored it. All four slots (B_0, B_1, B_rep,0, B_rep,1) or none are required now; some is rc 4, nothing written.
* **A fixture that made the suite 10x slower on exFAT:** 141-file fake dump directories, hundreds of times; the tests now set `R.N_EPISODES` to 3 for the fakes (the real constants, 141 and 2,399, are asserted once). The suite runs in about 6 s.
* **A property worth stating, not a defect:** inside the pair's domain D, R-NOBASE's violation condition equals R-NEW's fire condition (both need X - P10 separated-worse, X - Y separated-worse and the worsening >= the bar; D already requires Y - P10 not separated-worse). The two predicates share one contrast, as PREREG section 7 discloses; "FIXED / NOT ADOPTED through R-NOBASE alone" cannot occur on real rows, and the reader keeps the two columns apart in its record.

## 6. How it is used

```
python launch_intent_read.py --static-check --prereg PREREG_V8F_LAUNCH_INTENT.md [--prereg-sha <sha256>]
python launch_intent_read.py --dry-run --out evidence/dry_run.json           (CPU, about 8 min; needs the banked dag-orc dumps under F:/TanitAD-scratch/dagorc_t1_out and E:/Projects/TanitAD-artifacts)
python launch_intent_read.py --manifest runs.json --prereg PREREG_V8F_LAUNCH_INTENT.md --out LAUNCH_INTENT_READ.json
python -m pytest -q test_launch_intent_read.py                               (LAUNCH_INTENT_READER=<mutant> and LI_PREREG=<file> select another reader or PREREG text)
python mutation_launch_intent_reader.py --reader launch_intent_read.py --tests test_launch_intent_read.py --work <scratch> --out mutation_reader.json --prereg PREREG_V8F_LAUNCH_INTENT.md
```

The heavy layer imports `dag_orc_read.py` (sha256 `2e5f9f17...`, with its `ref/` pins) unchanged from `--orc-dir` (default `E:/Projects/TanitAD-artifacts/dagorc_rebase/read`); in the repo that directory is the DAG-ORC read package under `raw/2026-10-27-dag-orc-read/`. `ref/rep_s0_speedcells.json` (sha256 `9788df70...`) is the banked slow-draw reference the dry run compares with.
Exit codes of the read: 0 outcome written, 2 REFUSED (failed controls written, no row) or a PREREG / row error (nothing written), 3 UNPLACED, 4 required slots missing (nothing written).

## 7. Integration steps (for the Master Mind; none done here)

1. Land `PREREG_V8F_LAUNCH_INTENT.md` as a DRAFT under `Project Steering/` (the Master Mind promotes it) and register `H-V8F-LAUNCH-INTENT-1`.
2. Land the package under the proposed repo paths in `MANIFEST.txt`.
3. The box-2 chain for A_0, A_1, A_rep,0, A_rep,1 (its one-lever asserter must allow `--launch-intent-w` and nothing else beyond `--out`, and `--seed` for REP's reuse check) and the trainer landing are the integration steps the v1 package and its HANDOFF already name; the PI's go for box-2 time is separate.
4. Check the REP licence on DAG-ORC r2's REP run record before any row is read (`K-REP-LICENCE` does it, and passes on the banked records).
