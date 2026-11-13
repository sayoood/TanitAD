# HANDOFF TF-N7 generator (developed on navtrain; NOT integrated, no navtest touched)

2026-11-13, engineering agent for the REFe MM. **Integration needed from the MM: write `n7_nt_build.py` (chain), freeze the card, ledger n = 50, launch.**

## State
- Result: `DEV_N7.md`. Navtrain 17 residual: TF-N6 generator 2 certified; cascade T0-T3 13; with T4 (deep slow-down s 0.5-0.6) 16; 1 unrescued (`229a59137a705430`). 73 repaired: 0 harmed (T0 certifies all 73; cascade leaves them untouched).
- Code: `n7_env.py n7_lib.py n7_run.py n7_select.py n7_analyze.py`; tests `test_n7_gen.py` (20 passed, `test_run.log`).
- Card draft: `PREREG_TF_N7_DRAFT.md` (arms N7-T, N7-T3, N7-X; K = 50; fourth read disclosed).
- Raw: `raw/candidates_navtrain.tar.xz` (every candidate + metrics), `raw/sel17_floor05.json`, `raw/sel73_floor05.json`, `raw/lever_table{17,73}.json`.

## Exact next step
1. MM decision: is the T4 slow-down tier admissible as teacher output (card runs N7-T with and N7-T3 without).
2. Build `n7_nt_build.py` from `../n6_tol/chain/n6_nt_build.py` with `n7_select.cascade` in place of `capture`; pin, freeze, launch.
3. Optional growth of the dev set: run `n7_run.py --set` on the 877 rolled logs' residual tokens for a held-out check of the tier order and the 0.5 floor.

## Blockers / UNVERIFIED
- n = 17 on 7 logs; tier order and floor set on the same 17. Navtest rescue rate is ESTIMATED only.
- Deliberate-regression arms are not zero-rescue (stretch inverted 1, D inverted 1, L1 inverted 1 of 5 test tokens); the offset grid is two-sided so its sign flip is vacuous. See DEV_N7.md section 5.
- `n7_run.py --set rep73` reads the extracted `derive_rows.tar.xz` from `C:/Users/Admin/l2dev/x` (deleted at the end): extract it somewhere and set `N7_ROWS`.

## Deletion log (rule 4)
See the report; scratch was `C:/Users/Admin/l2dev` (26 MB).
