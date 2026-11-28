# HANDOFF: read_rep.py (PREREG_REP_P6 reader, CPU only)
Files (E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/rep/read/): read_rep.py, test_read_rep.py (19 pass, test_results.json), control_out/ (control run), HANDOFF.md. Nothing in git, no ledger written.
Imports read_scorer1p6 (run_read per run) and read_s1z (s1z_read for S1Z runs: G1-G4, M2) unchanged. E3 scores of changed picks come from table_p6final_c.npz (sha 3bda2080...), not from arm csvs. Card sha checked (4f59ae7c...), refused if different.
CONTROL (C0, C0r, S1E vs scorer1p6_read.json): 224 leaves compared, 0 mismatches (C0 91.229 +0.0325, C0r 91.608 +0.4120 [0.1971,0.6364], S1E 91.271 +0.0744; G1-G3 equal).
SMOKE (C0r2:=C0r, S1E1:=S1E, --smoke-standin --no-ledger): READ in 15 s, SMOKE_NOT_A_RESULT, F:/tanitad_scratch/repread/smoke/.
## Real read
python read_rep.py read --run C0=F:/tanitad_scratch/p6arms/pass/C0 --run C0r=F:/tanitad_scratch/p6arms/pass/C0r --run C0r2=F:/tanitad_scratch/p6arms/pass_rep/C0r2 --run S1E=F:/tanitad_scratch/p6arms/pass/S1E --run S1E1=F:/tanitad_scratch/p6arms/pass_rep/S1E1 --out <OUT>
(add --run S1Z=DIR --run S1Z1=DIR when they exist; omit --no-ledger to write K keys to selectors/ledger_K.json; exit 0 READ, 2 otherwise.) Env CUDA_VISIBLE_DEVICES=-1, venv tanitad.
## Choices / UNVERIFIED
- Fewer runs than the card's (C0 3, S1E 2, S1Z 2 or 0) -> stamped PARTIAL_NOT_THE_CARD_READ; t for df outside 2-4 from scipy.
- R2 uses 1/n_C0 (equals the card's 1/3 for 3 runs). Floor threshold 2*0.3796 = 0.7592. G3 of S1E runs also vs the real C0 run (Batch 1 rule).
- ENS_R reads are reported, not keyed in the ledger (card keys runs and recipe means). Real read re-reads C0, C0r, S1E: new REP keys, separate from any Batch 1 key.
- Real C0r2/S1E1 logits dirs untested (do not exist); duplicate logits refused unless --smoke-standin.
