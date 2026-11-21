# HANDOFF - oaccflatq (DAG-OACC + FLAT-DAG box-2 build) - FINAL STATE, 2026-10-27 ~13:05

STATE: DONE on the dev box (CPU only). BUILT, PROVEN, NOT LAUNCHED. Nothing pushed, no git checkout touched, no box-2 contact, no GPU.
Read NOTES.md (headline, evidence, MEASURED/UNVERIFIED, install + launch commands) and MANIFEST.txt (sha256 of every file here).

## What the Master Mind still has to do (integration; nothing below is mine to do)
1. Land on the remote tip: stage/<queue> + stage/stack/tests/*.py (6 tests) + the F1 trainer fix (E:/Projects/TanitAD-artifacts/oaccflat_check/stage/stack/scripts/v8f_mini_train.py, sha256 30819a6f..., single copy).
2. Free several GB on E: (1.9 GB free at 12:55 and falling; FLAT-DAG alone publishes ~2 GB; every pull keeps a 1 GiB floor) - PI-only deletion.
3. Install box2_ship/* with cp -n and launch DAG-OACC while DAG2's chain is still running, then FLAT-DAG (NOTES.md sections 5 and 6). Needs the PI's go.
4. Re-run the Master Mind's pytest run that my earlier process kill took down (NOTES.md section 4, item 1).
5. Remove scratch the never-delete rule stopped me removing: F:/oaf_scratch (7.6 GB), the 132 MB runaway task output (NOTES.md section 4, item 2).

## If the proof must be re-run
OAF_SCRATCH=F:/some/dir CUDA_VISIBLE_DEVICES=-1 python mutate_oaf.py --workers 6   (needs C:/Users/Admin/.../scratchpad/oaccflatq_work/pytree, 24 MB of the 1d4f4fbb .py files, and the E: corpora named in NOTES.md section 4 item 6). ~20 min wall.
