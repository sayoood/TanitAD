# HANDOFF: HIERWHERE (PREREG_V8F_PROOF_ADD6), 2026-11-13
State: run complete, nothing staged, committed or pushed. Evidence class MEASURED, tier T1-kin (logged frames, a diagnostic), CPU only, zero training, zero rollouts.
Outcome (stated thresholds): R3 = MIXED, R1 = CRIT-CONCENTRATED. Controls: h1 PASS (both seeds, deviation 0.0 m), h2 PASS (residual 0.0), h3 flagged definition gap (Jaccard 0.314 < 0.90), h4 FAIL at s0 under swap seed 0 (exit code 2; HIER_DEFS R6), h5 printed (one named change: s0 x1.25 WITHIN interval now includes 0; no sign change).
By the card a failed control means the read is not valid until it reads its known value: the verdict is printed but is flagged, and the PI or Master Mind decides whether the h4 chance rejection (19 other swap seeds include 0 in 19/19 at s0 and 17/19 at s1) is accepted or the control is re-registered.
Largest share of D at both seeds sits in BOTH-MISS (s0 -0.2594 of -0.3451), a term the card names but its outcome rules do not read.

Re-run (about 15 s warm, about 2 min cold; check free commit >= 4 first):
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=-1
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/TanitAD-scratch/v8proof/hier/hierwhere.py run
  C:/Users/Admin/venvs/tanitad/Scripts/python.exe F:/TanitAD-scratch/v8proof/hier/test_hierwhere.py
Exit 0 all controls pass, 2 a control failed (h1 failing stops before any read). Smoke: --smoke N (first N episodes; h1 then not evaluated).

Files (all under F:/TanitAD-scratch/v8proof/hier/): HIER_DEFS.md, hierwhere.py, test_hierwhere.py, RESULT_HIERWHERE.json, VERDICT_HIERWHERE.md, run.log, tests.log, HANDOFF.md.
Inputs read-only: the four dump dirs named in the card, the pinned tool tree E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a, the stopdec npz.
Next lever if the PI wants the read valid: fix the h4 seed rule (a calibrated reading over many swap seeds), and trace why 66 label STOP windows still move at 6 s on the logged future (R5b).

Deletions: none (every file above is a deliverable; no scratch left; no __pycache__ written).
