# HANDOFF: read_pess.py (reader of PREREG_PESS_P6, card sha256 185de406...f42f380; CPU only, no GPU)

Refuses to run if the card's sha256 differs. Imports `eval/read_scorer1p6.py` unchanged (pick_logits, gates, BootSet, family components) and `selectors/selcore.py` (ledger).
Scoring = lookup in `F:/tanitad_scratch/selq2read/table_p6final_c.npz` (64 proposals x 12,146 tokens); no csv files are written.

## The real read (NOT yet run; counts +3 in the K ledger; run it only when S1ZD's pass is done and G0-E3 passes, else use the eight)
```
set CUDA_VISIBLE_DEVICES=-1
C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/pess/read/read_pess.py read ^
 --p6 F:/tanitad_scratch/p6arms/pass/P6_control --out <OUT> ^
 --member C0=F:/tanitad_scratch/p6arms/pass/C0 --member C0r=F:/tanitad_scratch/p6arms/pass/C0r --member S1E=F:/tanitad_scratch/p6arms/pass/S1E ^
 --member C0r2=F:/tanitad_scratch/p6arms/pass_rep_c0r2_s1e1_s1z1/C0r2 --member S1E1=F:/tanitad_scratch/p6arms/pass_rep_c0r2_s1e1_s1z1/S1E1 --member S1Z1=F:/tanitad_scratch/p6arms/pass_rep_c0r2_s1e1_s1z1/S1Z1 ^
 --member S1Z=F:/tanitad_scratch/p6arms/pass_s1z/S1Z --member S1ZSM=F:/tanitad_scratch/p6arms/pass_s1zsm/S1ZSM --member S1ZD=F:/tanitad_scratch/p6arms/pass_dose/S1ZD
```
Add `--no-ledger` for a dry run (no K key). Default ledger: `selectors/ledger_K.json`; keys `ENS9|PESS|LCB1:<sha256 of combined logits>:E3`. Exit 0 READ, 2 = card or gate failed (no verdict).
Expected: ~10 s per arm set after loading, ~35 s total measured on 1 to 3 members (6.8 s for one); nine members load ~75 MB proposals each transiently; peak RAM about 2 to 3 GB (waits for `--min-free-ram-gb`, default 3).

## Outputs (in --out)
`pess_read.json` (gates, per-arm PDMS, minus P6, P1, EP x100, zero-pick share, G1 to G4, family vs P6, P2 and M vs ENS9, outcome PE1..PE4/Unnamed for PESS, LCB1 reported-not-adopted, K ledger block, Bonferroni intervals, combined-logit shas, ambiguities) and `pess_read.md` (the same, tables). Controls write `pess_read_control_<c>.json/.md` or `control_c3.json` and never the ledger.

## Controls (all passed; outputs in `control_out/`; `run_controls.sh` reruns them)
c1 (S1ZSM alone: every arm bit-equal to its logits, PDMS x100 91.29707529608928 = banked, table = S1ZSM's csv on all 6,447 changed picks), c2 (ENS9 of C0, C0r, S1E = banked ENS3 +0.1649806668275697 [-0.0113024, +0.3457336], diff 0.0), c3 (MAX vs MIN PESS on 8 members: 48.6 % of picks differ, shas differ).

## Ambiguities: see the list inside the output (T1..T10): P1 uses the printed 0.4971 on the point; M = zero-pick share (PDMS exactly 0), not the S1ZSM M2; E2 guards use E3 combined logits at the E2 tokens; PE labels overlap (card order), PE2 not conditioned on M or guards.
