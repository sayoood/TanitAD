# HANDOFF (read2 build agent, 2026-11-17 ~22:00)

State: reader, tests, launcher built (see BUILD_DSCALE10K_B2_READ.md). Remaining, in order, each after `C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 6 120`, CUDA_VISIBLE_DEVICES=-1, TMP/TEMP = read2/bt, one pytest at a time:
0. Status 22:20: tests 57 passed after final edits; a mutation re-run (after fixing the equivalent mutant) is running in the background; smoke not run.
1. A background `mutate_check_b2.py` (started 21:2x) may still be running or may have died; check `read2/out/MUTATIONS_B2.json`. If absent and no `mut/` growth, re-run `python mutate_check_b2.py` (15 mutations, ~8 min when commit is free). It started before the zero-pass port; re-run once if its files predate `ds_run.py` / `ds_read.py` edits.
2. `python -m pytest -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/datascale/read2/bt/t test_ds_read_b2.py test_ds_read.py` (expect all green).
3. Smoke: `PYTHONPATH=F:/TanitAD-scratch/otband_read/tree/stack;F:/TanitAD-scratch/otband_read/tree/taniteval OMP_NUM_THREADS=1 python smoke_ds_read_b2.py` (prints SMOKE OK).
4. Delete `read2/mut`, `read2/bt`, `read2/smoke`, `read2/__pycache__` by literal path; log them below. Update the PENDING lines of the BUILD note with the numbers.
5. MM decision open: block-1's zero-pass fix (other agent) vs the interim drop-zero port in read2.

Deletions logged so far: none by this agent beyond the mutation script's own mutant removals.
deleted scratchpad patch scripts p2,p3,p4,patch_zero (own temp)
