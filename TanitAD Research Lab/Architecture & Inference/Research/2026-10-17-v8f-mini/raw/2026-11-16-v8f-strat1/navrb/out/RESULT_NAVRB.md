# RESULT_NAVRB -- roundabout rows of the router text

Evidence class: MEASURED (ours; pinned `navtok_table.npz` sha256 2836369e..., frozen navtok grammar over `texts.json`, zero training, CPU). Tier: T1-kin tick 0 (NOW frame). Estimator: episode-cluster bootstrap, B 2,000, seed 0; 6 episodes, so intervals are descriptive. Artifact: `F:/TanitAD-scratch/strat1/navrb/out/RESULT_NAVRB.json`. Committed reading and rule: `PREREG_NAVRB.md` (written before any number).

## 1. What the rows carry
* Table: built by `navtok/build_table.py:42-71` (latest nav row with ts <= now_s x 1e6 - 195,000 us); columns `eval_cls_id` (44-class vocab), `eval_dist_m`, `eval_flag`, `eval_text_i` (index into the raw texts, which hold the exit number), `eval_age_s`. Classes from `navtok.py:22-34`, vocab `:43-58`. (Files sit in the landed hold `C:/Users/Admin/qland/landed_hold_navtok/stage/.../2026-11-15-v8f-datascale/navtok/`, not in the repo tree.)
* Roundabout classes over all 9,864 distinct texts (0 unparsed): RB_TAKE_EXIT (1,937 texts: exit number always, distance in 1,929), RB_ENTER (252, distance only), RB_EXIT (251, distance only), ROTARY_EXIT (251, distance only), RB_NEXT_EXIT (1, bare). Side field non-empty: 0; degree: 0; a left/right word in any of them: 0. No bearing field exists.
* 141-episode eval grid (2,399 windows): 235 windows in 16 episodes have a roundabout row at NOW: RB_TAKE_EXIT 202 (exit 1: 92, 2: 96, 3: 14; distance on 194), RB_EXIT 16, ROTARY_EXIT 17 (no exit number). 43 of them are labelled turn windows, all in the 6 audited episodes (31, 52, 69, 102, 126, 135). Eval steps with a roundabout row (flag 0): 1,334, exit numbers 1: 508, 2: 564, 3: 57, 4: 26.

## 2. Side from the route (the 43 windows, 6 episodes)
| rule | right on v9 side | n |
|---|---|---|
| R1 (exit 1 -> R, exit 3 -> L, else none; causal) | **0.465 [0.000, 0.833]** (20) | 43 |
| R1, current row only | 0.465 (20) | 43 |
| R1 with router rows up to 8 s ahead (not causal) | 0.465 (20) | 43 |
| constant R (context) | 0.814 (35) | 43 |
| constant L (context) | 0.186 (8) | 43 |
| roundabout state at NOW | **1.000 (43/43)** | 43 |

* Exit number vs label (latest numbered row at or before NOW): exit 1 -> TURN_R 20 / TURN_L 0; exit 2 -> L 8 / R 4; exit 3 -> R 7 / L 0 (the convention says L: wrong on all 7, episode 135); no numbered row -> R 4. 39 of 43 windows have a numbered row. The label side is R on 35 of 43, so exit 2 (convention: straight, no side) and exit 3 carry the misses.
* Future-read probe: router rows after NOW replaced by garbage, 0 of 43 predictions differ; the rule reads no track or ego array. A future-read mutant goes RED (`test_navrb.py`, 7 pass).
* Reading: **RB-STATE-ONLY** (R1 0.465 < 0.90; roundabout state at NOW 43/43 >= 0.90).
* Secondary: all 43 labelled turn windows with a roundabout row at NOW are these same 43.
