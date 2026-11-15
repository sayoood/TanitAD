# HANDOFF: IMAGPLAN tool (card PREREG_V8F_PROOF_ADD40 + note N3 decel family). Builder, dev-box CPU only. Last update 2026-11-15T22:23Z. Nothing staged, committed, pushed; no pod, box 2, Thor, network.

Tier T1-kin (kinematic rollout, logged frames); diagnostic only. STRATEGIC: PAUSED (PI item 25). No number here is a card result. Detail: `IMAGPLAN_DEFS.md` (section F = decel family).

## State
`imag_plan.py` (pure rule), `v8f_t1_eval_imagplan.py` (wrapper; flags as before plus `--cand-family {scale,decel}`), `pod_imagplan_run.sh` (not run; new env `IP_FAMILY`), `test_imag_plan.py` (37 passed = 26 old, unedited + 11 new, about 50 s), `run_mutations.py` (control green, 19 of 19 mutations red: 9 old + 10 new decel). Default family `scale` unchanged (26 old tests pass untouched). Wrapper reuses `dist_rule.py`, `v8f_t1_eval_rule.py` (../distrule), `imag_dump.py` (../imaghead); none edited.

## Read first
1. Action definition (A2): corpus acceleration = FORWARD difference of the cache-step speed; default `--action-def logged`.
2. p4 on 16 real windows (plumbing only): corr acc 0.982 (bar 0.99 missed), yaw 0.9956. Measure p4 on the full set before any run.
3. Card/note limits: p1 identity probe = `--cand-set 1.0 --fallback none --floor cap-only` (scale) or `--cand-family decel --cand-set 0 --fallback none` (decel); `--const-gap` applies after the gate. Note N3 cannot hold: a stopped candidate restarts if the plan speeds up again; fallback zero = largest d OF THE SET (see DEFS F).

## Rerun
```
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
cd F:/TanitAD-scratch/v8proof/imagplan
/c/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest test_imag_plan.py -q -p no:cacheprovider --basetemp=F:/TanitAD-scratch/v8proof/imagplan/work/pt/runN   # new runN each time
/c/Users/Admin/venvs/tanitad/Scripts/python.exe run_mutations.py        # work/mut/mutations.json, about 8 min
```
Pod: `bash pod_imagplan_run.sh <NEW> <QUEUE> <BASE> <cand-imag|cand-cv|cand-label|off> [TREE]`; env `IP_FAMILY IP_CAND_SET IP_CONST_GAP IP_SHIFT_M IP_FLOOR IP_FALLBACK IP_NOISE IP_P45=1`; `IP_DRY=1` prints the command.

## Open points
* No lead-head checkpoint on this box: cand-cv / cand-imag ran only on the tiny test model; decel never ran on real windows. p5b (note N3) is NOT measured.
* Stray scratch I could not delete: `work/dbg.py`, `work/dbg_test.py`, `work/mut/*` old dirs, `work/pt/*`, `work/smoke/label`, `work/mut_n3.out`.
* Wrapper threading of `family` is covered by CLI and e2e tests, not by a mutation.

## Manifest (sha256, LF)
- imag_plan.py cdeb04077104b6c53231e5e13318ab1fbe82ebe750e0101e3da057b378b95078
- v8f_t1_eval_imagplan.py a2b28eba17e3d846c8b673e86577f88efadc91b5c16a1d23967b4adbeb99f476
- pod_imagplan_run.sh dbb13cfaa426ea2f68651ea4f99bf8e544a29b053b66b3672e45da65238d25bb
- test_imag_plan.py 16df5bc385ec020a214938dfb4dded19123c4045e933f3e1ca39622fe113cb93
- run_mutations.py 0e9775053c4e0a661fefbe2c2417cf318006d2beafee6b59065707f1391692cf
- IMAGPLAN_DEFS.md abf31f1558586d52101f4562bfcaffdd2c72dfa346321e3271adbb0e1bf4d664
Pinned tool unchanged: 61e140edb9da01acbbba4eef81e74bbb68b57e2ff31842e05958759446d97397.
