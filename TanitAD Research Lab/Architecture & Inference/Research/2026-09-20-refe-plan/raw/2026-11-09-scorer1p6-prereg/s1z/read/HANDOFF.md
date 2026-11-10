# HANDOFF: read_s1z.py (CPU only; python venvs/tanitad)
Files here: `read_s1z.py`, `test_read_s1z.py` (28 pass), `test_results.json`. It imports the Batch 1 reader `../../eval/read_scorer1p6.py` unchanged (S1Z occupies its S1E slot; relabelled S1Z). Adds M2, G4, S1Z adoption (P1 and G1-G4, G3 vs P6 only), Z1-Z6/Unnamed, K ledger + Bonferroni interval. C0r is required.

## Real read (when S1Z navtest logits exist at DIR_S1Z)
1. Missing scores: `python read_s1z.py emit-changed --c0 F:/tanitad_scratch/p6arms/pass/C0 --c0r F:/tanitad_scratch/p6arms/pass/C0r --s1z DIR_S1Z --out OUT` -> seam/tokens/needs_scoring.json; score with the harness, copy csv to `DIR_S1Z/e3_changed_scores.csv`.
2. `python read_s1z.py read --c0 F:/tanitad_scratch/p6arms/pass/C0 --c0r F:/tanitad_scratch/p6arms/pass/C0r --s1z DIR_S1Z --out OUT` (defaults: P6 proposals/csv, E2 tokens/table under p6base/e2, export). Waits for 3 GB free RAM. Exit 0 READ, 2 = gate failed / INCOMPLETE (no verdict). Env CUDA_VISIBLE_DEVICES=-1.
A READ adds key `S1Z:<logits sha>:E3` to `p6base/selectors/ledger_K.json` (`--no-ledger` to skip) and prints K.

## Control (done)
Smoke with S1E as stand-in (`--smoke-standin --control-verdict F:/tanitad_scratch/p6arms/read/verdict/scorer1p6_read.json`): 63 leaves (E3 PDMS, vs P6, P1 vs C0, G1 sub-scores, G2, G3, goal, slow share) compared with the Batch 1 S1E numbers: 0 mismatches, EQUAL. Outputs `F:/tanitad_scratch/s1zread/smoke/` stamped SMOKE_NOT_A_RESULT (stand-in outcome Z4 is not a result). Read took 7 s.

## G4 definition (P6_RESULT.md)
`E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read/result/P6_RESULT.md:37` "Slow picks on cruise 0.224 [0.117, 0.348]" and `:43` "slow-pick share 0.224 against the bar 0.20". The file gives no numeric threshold; the rule is `refe_behaviour_20261107/p6_read/selc6_analyze.py:42` `SLOW_RATIO = 0.85` (used at :228: pick's v4/v0 < 0.85 on CRUISE tokens). Implemented as Batch 1's slow share (0.5 s grid plan v4, E2 uniform tokens).

## Unverified / choices
- P6_RESULT's 0.224 is C3 stratum-weighted on 20-step plans; E2 value will differ (definition shared, value not).
- K ledger file did not exist; Batch 1 reads are in it only if someone wrote them (K may undercount).
- Z6 end-to-end not built synthetically (M2 and P1 move together on the synthetic world); Z6 and Unnamed tested on the pure map only.
- Z3 for a failed G2/G3 alone: card names no lever; stated as such.
- Within-token AUC and near-twin AUC not computed (reasons in the json). Z overlaps resolved Z1,Z2,Z3,Z5,Z6,Z4 (Batch 1 A7 style).
