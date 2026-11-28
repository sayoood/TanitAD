# HANDOFF GATEMIX (PREREG_V8F_PROOF_ADD8, card sha256 807bb2115013)
State: DONE. Zero training, zero rollouts, CPU only. Overall outcome GATE-COSTS (P1 s0 GATE-TIES, P1 s1 GATE-WORSE, P2 s0 GATE-WORSE, P2 s1 GATE-TIES). Controls g1, g2, g3, g4 all PASS (g3: 20/20, 20/20, 20/20, 19/20). Tier T1-kin, diagnostic only. STRATEGIC PAUSED (PI item 25).

Files (all in F:/TanitAD-scratch/v8proof/gate/):
* gatemix.py (one command `run`), test_gatemix.py (6 tests OK, tests.log), GATE_DEFS.md (definitions, deviations R1-R8)
* VERDICT_GATEMIX.md, RESULT_GATEMIX.json, stages/{g2,g1_g4,g3,read_P1_s0,read_P1_s1,read_P2_s0,read_P2_s1}.json, run.log

Re-run: `cd F:/TanitAD-scratch/v8proof/gate; CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe gatemix.py run` (exit 0 = controls pass, 2 = a control failed). Tests: `python test_gatemix.py`.
Runtime: 387 s on the shared dev box (g2 reproduction 348 s of it: it loads 8 dump sets and runs 6 reference bootstraps; everything after g2 is 40 s). One process, 4 threads, about 600 MB; before each run free commit read >= 4 GB (values 7 to 12).
History: the first run stopped after g2 on an exact-equality alignment test for `gt` (ticks store float32, diff 7.6e-6 m); relaxed to < 1e-4 m (deviation R3), rerun from the start. No composite was formed in the stopped run.
Read-only inputs used: E:/Projects/TanitAD-artifacts/v8f-dagger-t1 (dumps and ticks), v8f-flatdag-t1 (dumps), dagt1_tree_b43c7a9a (tools), F:/TanitAD-scratch/v8proof/hier (hierwhere.py, pair2/read JSON). Nothing staged, committed or pushed.
Deletions: none (no temporary files were created outside the deliverables; run.log and stages/ are outputs and were kept).
