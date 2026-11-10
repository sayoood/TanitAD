# HANDOFF: read_s1zsm.py (reader of PREREG_S1ZSM_P6, card sha256 ac75b289...d6b8; CPU only)
Files (p6base/s1zsm/read/): read_s1zsm.py, test_read_s1zsm.py (33 pass, ZZTEST_PASS, test_results.json), control_out/ (S1Z-as-arm control: s1zsm_read.json/.md, s1z_read_ref.json = the banked S1Z read exported from mirror blob 0c19b8fd:...s1z_read.json). Nothing in git; ledger_K.json untouched (sha 94b59c0b... before and after).
Imports read_s1z.s1z_read unchanged (arm in its S1Z slot, relabelled S1ZSM); adds M2 (vs P6 only), M3, SM1..SM6/Unnamed, S1Z-referenced reports, pair shares, s_p, ledger.

## Real read (when S1ZSM navtest logits exist at DIR)
1. Missing scores: `python read_s1zsm.py emit-changed --c0 F:/tanitad_scratch/p6arms/pass/C0 --c0r F:/tanitad_scratch/p6arms/pass/C0r --s1z F:/tanitad_scratch/p6arms/pass_s1z/S1Z --s1zsm DIR --out OUT` -> harness; copy csv to DIR/e3_changed_scores.csv.
2. `python read_s1zsm.py read --c0 F:/tanitad_scratch/p6arms/pass/C0 --c0r F:/tanitad_scratch/p6arms/pass/C0r --s1z F:/tanitad_scratch/p6arms/pass_s1z/S1Z --s1zsm DIR --run-log <pair_weight.json or run log> --rep-read <rep_read.json> --out OUT`
   (venv C:/Users/Admin/venvs/tanitad/Scripts/python.exe, CUDA_VISIBLE_DEVICES=-1; ~10 s; waits for 3 GB RAM). Exit 0 READ; 2 = card/gate failed or INCOMPLETE (no verdict). A READ adds key S1ZSM:<logits sha>:E3 to selectors/ledger_K.json (`--no-ledger` skips; do that for dry runs).
Control re-run: same with `--s1zsm F:/tanitad_scratch/p6arms/pass_s1z/S1Z --control-s1z-read control_out/s1z_read_ref.json --out <tmp>` (never writes the ledger).

## Control result (recorded in control_out/)
S1Z as arm: 89 leaves (P1, M2-vs-P6 + shares, G1-G4, EP-vs-P6, goal) 0 mismatches vs s1z_read.json; M3 = 0.0 [0.0, 0.0]; internal pool checks 0.0 diff. Outcome shown there (SM3+SM5) is the control's, not a result.

## Ambiguities resolved (also in the json, T1-T9)
- M2 uses only the vs-P6 difference of s1z_read's M2 block (no C0 clause). M3: EP = harness sub-score ego_progress x100 of the pick, S1ZSM - S1Z, same draws as P1, CI-lo > 0.
- SM labels: card order SM1..SM6; SM5/SM6 are stated by the card with only M2/M3 conditions, so they can co-match SM1/2/3 (listed, primary by order). SM4 requires "not SM3". Unnamed also when guards fail with P1,M2,M3 holding (no label covers it; note names the failed guards).
- "Logits bit-identical to banked picks on unchanged tokens": read_s1z has no logit gate; the gates are G0-E3 (proposals bit-identical, applied to the arm AND the S1Z reference), G0-PICK, G0-E2, G0-P6CSV. RED tests use those.
- E3 scores of changed picks come from each arm's e3_changed_scores.csv (as read_s1z), not table_p6final_c.npz.
- --run-log: card names no file; accepts pair_weight.json or a JSON/JSONL log holding sets_with_pair_share (last one wins). --rep-read needs seed.{s_p,t_0.975} (read_rep.py). Neither is deciding.
## UNVERIFIED
- Real S1ZSM logits/run log do not exist yet: the real-arm path (new-arm scoring, pair-share file format of the actual run) is exercised only on synthetic data and the S1Z control.
- The rep_read.json key layout was read from read_rep.py source, not from a produced file.
Temp deleted: F:/tanitad_scratch/s1zsmread/ (~382 KB: pytest basetemps, ref copy, 0-byte ctl) and .pytest_cache.
