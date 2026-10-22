# P7 session reserve: the seeded draw (D-SESSION-RESERVE, DECIDED 2026-10-22, option (a))

> Written 2026-10-22. **Design and CPU only.** No GPU, no pod, no network. No P7 data exists; the draw below is recorded before any.
> Evidence classes: **MEASURED** (ours, artifact named) · **INHERITED** · **ESTIMATED** · **UNVERIFIED**.
> `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Inputs were exported from the mirror at `f32ed6b` (shas in `raw/src/SOURCES.json`).

## 0 · Headline

| | R-A: **the registered draw** (session closure, as decided) | R-B: **alternative, needs a ruling** (vehicle-day closure, §5) |
|---|---|---|
| sessions / logs | **16** / 116 | 17 / 104 (5 vehicle-days) |
| scenes reserved | **10,141 = 9.84 %** of the 103,037 bank scenes (9.82 % of navtrain's 103,288 yaml tokens) | 9,974 = 9.68 % |
| scenes P7 trains on | **92,896** → 25 × ⌈92,896/256⌉ = **9,075 steps** (P6: 10,075) | 93,063 → 9,100 steps |
| seed / accepted attempt | **20261022** / attempt 2 (two rejected, §2) | 20261022 / attempt 521 (520 rejected; the pool is tiny) |
| `sessions_sha256` | `b16dc83386ad02ce95ed54826bce93ea9e3259ff8696168ffbf836940a711fab` | `8cd2c0478b48cd91…` (full in the file) |
| `logs_sha256` (the trainer's pin) | `c651d1f6080134c1d3922c336122c0f7c61bfc08017ecfd8a9b55f470cc8c2f0` | `76c64b85dc37b489…` |
| file sha256 | `53f3bcd4684513f0ee765fbcfbaf583b587f58c97e9c8915f4e4df3ae3e32f29` (`raw/reserve_sessions.json`) | `fb6646fb3e31c6e9…` (`raw/reserve_vehicleday.json`) |
| vehicles / cities | 9 of 16 / all 4 (LV 9, Pittsburgh 3, Boston 2, Singapore 2) | 4 / 2 (LV 12, Pittsburgh 5) |
| same-vehicle same-day sibling still in training | **16 of 16 sessions** | 0 of 17 |

**Open question #1 (MM / PI, before P7 data exists): R-A or R-B.** §5 shows navtest shares **no vehicle-day** with navtrain (0 of 20) and only 1 of 9 dates, while every R-A session has a same-day sibling in training. R-A is what was decided and is registered here; R-B is what matches navtest's kind of unseenness. Both are recorded; only one may be used, and the choice must precede the first P7 training input.

## 1 · The rule (`code/draw_reserve.py`; every constant is pinned by `code/test_draw_reserve.py`)

A **session** is the log name minus its last two `_` parts. A session's **token count** is the number of navtrain scenes attributed to its logs in `PK/raw/rl_pool/navtrain_token_table.tsv.gz` (the file `PK/raw/2026-10-22-search-pessimistic-feasibility/raw/session_probe.py` reads). Of the table's 103,288 rows, 103,037 carry a log; that equals the 3-C bank's rank-0 scenes (`launch_gate.P6_BANK_TUPLES` comment, INHERITED), so **the bank's scene count is the denominator**.

1. **Eligible:** not navtest; holds no SFT-held-out log (`PK/splits/sft_heldout_logs.txt`: 24 logs, **21 sessions**, which S1, Stage 0 and C3 already read, so reusing them would reuse a design-time look); 150 ≤ tokens ≤ 1,500 (neither trivial nor ≥ 14.6 % of the reserve); its vehicle has ≥ 3 navtrain sessions (so no vehicle vanishes from training). Pool: **109 sessions, 66,542 tokens.**
2. **Stratify by vehicle:** the 16 seats go to vehicles by systematic probability-proportional-to-size over eligible token mass (vehicles in name order, seeded start), so each vehicle gets ⌊quota⌋ or ⌈quota⌉. Each vehicle drives in one city (MEASURED, `raw/src/vehicle_city.json`, all 16), so this also stratifies by city.
3. **Stratify by token count:** inside a vehicle, sessions sorted by (tokens, name); k taken at equal rank spacing from a seeded start, so a vehicle's picks span its small and large sessions.
4. **Accept the first attempt** (0, 1, 2, …) with: 16 sessions; 9–11 % of the bank scenes; ≥ 3 sessions in the small tercile (< 390 tokens) and ≥ 3 in the large (≥ 734); ≤ 35 % of any vehicle's tokens; ≥ 6 vehicles; all 4 cities. Randomness is `sha256(label|seed|attempt|purpose)` only (no `random`, no numpy): identical on Python 3.12 and 3.14.5 (MEASURED, both `CHECK OK`).

**Rejected before acceptance (disclosed):** attempt 0 (token fraction 8.91 % < 9 %; veh-52 share > 35 %), attempt 1 (veh-39 share > 35 %).
**Order of events (disclosed):** the constants were fixed from the token table's marginals (sizes, vehicle shares) before the first run; the seed was not tried against alternatives (the tests' seed-sensitivity arm draws seeds +1..+3 and only compares them for inequality). The §5 closure measurement was made **after** R-A was drawn; it changes nothing in R-A. R-B's constants were set after seeing R-A and the pool size, so R-B is an alternative offered on its merits, not a second try at a better-looking draw.

## 2 · The registered draw R-A (`raw/reserve_sessions.json`)

| # | session | veh | city | logs | tokens | size |
|---|---|---|---|---|---|---|
| 1 | `2021.06.07.17.46.49_veh-35` | 35 | LV | 3 | 575 | mid |
| 2 | `2021.06.07.19.43.00_veh-35` | 35 | LV | 2 | 154 | small |
| 3 | `2021.06.09.12.06.35_veh-35` | 35 | LV | 4 | 338 | small |
| 4 | `2021.06.09.19.40.26_veh-12` | 12 | LV | 5 | 614 | mid |
| 5 | `2021.06.14.13.27.42_veh-35` | 35 | LV | 29 | 1,175 | large |
| 6 | `2021.06.23.16.52.00_veh-26` | 26 | LV | 6 | 930 | large |
| 7 | `2021.06.23.17.31.36_veh-16` | 16 | LV | 8 | 1,022 | large |
| 8 | `2021.06.23.21.56.29_veh-35` | 35 | LV | 5 | 889 | large |
| 9 | `2021.07.16.00.24.14_veh-38` | 38 | LV | 4 | 227 | small |
| 10 | `2021.08.17.18.11.12_veh-08` | 08 | Pittsburgh | 4 | 517 | mid |
| 11 | `2021.08.24.14.35.46_veh-45` | 45 | Pittsburgh | 3 | 176 | small |
| 12 | `2021.09.14.16.46.51_veh-45` | 45 | Pittsburgh | 2 | 355 | small |
| 13 | `2021.09.15.12.32.43_veh-28` | 28 | Boston | 11 | 399 | mid |
| 14 | `2021.10.01.19.16.42_veh-28` | 28 | Boston | 13 | 1,096 | large |
| 15 | `2021.10.05.06.24.06_veh-50` | 50 | Singapore | 6 | 729 | mid |
| 16 | `2021.10.11.02.57.41_veh-50` | 50 | Singapore | 11 | 945 | large |

Sizes: 5 small / 5 mid / 6 large. Per vehicle (reserved/total sessions; tokens): veh-35 5/34 (3,131 of 26,141 = 12 %); veh-28 2/20; veh-45 2/16; veh-50 2/11 (1,674 of 9,503 = 18 %); veh-08, 12, 16, 26, 38 one each; vehicles 14, 17, 39, 40, 42, 43, 52 contribute none. All logs of every reserved session are listed in the JSON (completeness tested against `navtrain.yaml`'s 1,192 logs).

## 3 · What P7 loses, and the leak paths the exclusion must close

* **Scenes:** −9.84 % (10,141 of 103,037) → 9,075 steps instead of 10,075 (−9.9 %). **Tuples (ESTIMATED, UNVERIFIED per session):** if rank-1 twins fall uniformly, ≈ 6,300 of the 64,263 rank-1 rows go too, leaving ≈ 150,800 of the 3-C bank's 167,300. The exact count is measured by the filter and registered before step 0 (`launch_gate` pins a derived constant per bank).
* **Not only the bank.** The exclusion must cover every training input keyed by log: bank rank-0 / rank-1 rows, the D9 metric cache (`<log>/<token>[__twin].pkl.xz`), the aux label store, the per-log image zips and calibration rows, and **every downstream arm that trains on navtrain** (RL post-training, scorer fine-tunes, ensemble members, hard-negative banks).
* **Logless tokens (MEASURED, `raw/src/logless251_probe.json`):** 251 table rows carry no log and sit in no bank row. 103 resolve to a nuPlan DB; **5 of those resolve into R-A's sessions** (listed in the JSON); 148 stay unattributed (UNVERIFIED which session). The aux label store holds 103,139 frames, 102 more than the bank, so a DB-resolved token can reach a label store: **exclude by session, not by bank row**, and gate that every key in every input has a resolved log.

## 4 · What the reserve is for, and never for

**For:** (1) fitting selection rules, calibration, pessimism and ensemble weights of S4 (`PLAN_BEYOND_PAPER.md`); (2) measuring generalisation: P7's PDMS, sub-scores and head IoU on unseen sessions, and the seen-vs-unseen split of any selection gain (compare P7 on the reserve with P7 on a seeded same-size subset of its own training sessions). Any fit-and-score inside the reserve is **session-grouped K-fold** (K = 4): never score a rule on sessions it was fitted on. The final judge stays one navtest look.

**Never for:** training in any form; architecture, `w_aux`, epoch, checkpoint or hyper-parameter choice inside P7 (the checkpoint is `model_final.pt`, fixed in advance); the lever claim "P7 beats P6" (**P6 and P2 trained on these sessions, so their reserve scores are seen-session scores**: a P6-vs-P7 delta on the reserve is exposure plus lever, never lever alone). A reserve read that influences a P7 training decision voids the reserve.

## 5 · The closure finding: how unseen is "unseen" (MEASURED from names, `test_session_closure_*`)

| | sessions | dates | vehicle-days | shared with navtrain |
|---|---|---|---|---|
| navtrain | 162 | 19 | 46 | n/a |
| navtest | 44 | 9 | 20 | sessions **0**; vehicle-days **0 of 20**; dates **1 of 9** (2021.10.06) |
| **R-A's 16 sessions** | 16 | 12 | 15 | **16 of 16 have a same-vehicle same-date sibling session in training** |

⇒ R-A is unseen at the **session** level, exactly as `D-REFE-SESSION-EXPOSURE-1` defines it, but its nearest training neighbours are the same car on the same day, which navtest never has. **Consequence (HYPOTHESIS):** the seen→unseen gap measured on R-A will understate the seen→navtest gap, so a pessimism fitted on R-A may be too weak for navtest. Cheap mitigation if R-A stays: report R-A's gap beside P7's navtest gap and treat R-A's fitted κ as a **lower bound**.

**R-B** closes whole vehicle-days (`--closure vehicle-day`, `raw/reserve_vehicleday.json`): 0 siblings left. Costs, from the pool (not a choice): excluding vehicle-days that hold a held-out session leaves **16 of 46 units (25,128 tokens)**, 13 of them Las Vegas, so R-B covers 4 vehicles and 2 cities, has 5 independent vehicle-days, and needed relaxed bounds (cities ≥ 2, vehicle share ≤ 50 %, units 400–3,000 tokens). If the PI wants navtest-like unseenness **and** wider coverage, the held-out exclusion (§1.1) is the lever to relax; that is a ruling, not a script change.

## 6 · Limits

* **Power.** 16 sessions is fewer independent units than Stage 0's 21, which read **NULL-UNDERPOWERED** (`BACKLOG.md` SEARCH-PESSIMISTIC, A1). The reserve's gain is that the **whole model** (trunk and head) is unseen and that it holds 10,141 scenes, not that it has more sessions. Level precision, ESTIMATED by scaling P2's navtest CI (±0.96 PDMS over 136 logs): ≈ ±1.0 over 116 logs, wider once clustered by session. It can resolve a large generalisation gap, not a 0.3-point lever. Open question #2: is ~16 sessions enough for S4, or 24–32 (15–20 %)?
* **Navtest's own yaml is not in the repo (UNVERIFIED).** The navtest log set (136 logs, 44 sessions, 12,146 tokens) is the intersection-checked content of two independent harness artifacts (`C2/navtest_plan.json`, `navsim-estimator-and-route-leak/cluster_maps/navtest.json`; sha256 in `SOURCES.json`); `extract_inputs.py` refuses if they disagree.
* The held-out exclusion (§1.1) is my rule, not the brief's. Open question #3: confirm or drop. Dropping it changes the draw, so it must precede data.

## 7 · Verification (all MEASURED, `raw/test_run.out`, `raw/mutation_check.out`)

* `python -I code/test_draw_reserve.py` → `ZZRESERVE_PASS 17`, also on Python 3.12. Covers: byte-identical reproduction of both recorded files; order-independence under shuffled inputs; 0 shared sessions, logs and **tokens** with navtest (12,146 navtest tokens vs all 103,288 table tokens; the table's `in_navtest` column also reads 0); every bound of the rule; completeness of each session's logs; the pins.
* **Mutations that must fail, and do:** navtest guard disabled → `test_guard_refuses_a_navtest_contaminated_pool` FAILS (control: clean inputs pass); held-out exclusion removed → 5 tests FAIL; seed changed → 6 FAIL; unit closure collapsed → the draw refuses.
* Recompute: `python -I code/draw_reserve.py --src raw/src --out raw/reserve_sessions.json --check` (and `--closure vehicle-day … reserve_vehicleday.json`). `code/extract_inputs.py` regenerates `raw/src/` from the repo artifacts.

## Deliverable manifest

| artifact | where |
|---|---|
| `SESSION_RESERVE.md`, `SPEC_P7.md` | `C:/Users/Admin/par18/p7spec/` (MM lands) |
| `code/draw_reserve.py`, `code/extract_inputs.py`, `code/test_draw_reserve.py` | same, `code/` |
| `raw/reserve_sessions.json` (R-A), `raw/reserve_vehicleday.json` (R-B), `raw/test_run.out`, `raw/mutation_check.out` | same, `raw/` |
| `raw/src/` (token table copy 1.9 MB, derived logs/navtest/vehicle-city files, probe, `SOURCES.json`) | same; the token table also lives at `PK/raw/rl_pool/` in the repo |
| suggested repo path | `PK/raw/2026-10-22-p7-session-reserve/` (reserve) and `PK/SPEC_P7.md` |

All of it exists **only** in `par18/p7spec/` until landed. Nothing was staged or committed.
