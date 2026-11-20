# HANDOFF: OPTB3 read (Amendment 3, AMEND_OPTB3_TARGET.md sha256 cb4f943f). Written by the build agent; nothing committed or pushed.

## State
LAUNCHED 2026-11-19 11:45:35Z through WMI (CreateFlags 0x08000200). cmd pid 31396, driver bash pid 38216 (child 46372). Driver stdout `F:/TanitAD-scratch/optb3_read/optb3_driver.out`, chain log `out/chain.log`, sentinel `out/ZZ_OPTB3READ.txt`.
At launch the chain was WAITING in its first gate: free commit was 3.6 GB against the 5.5 GB gate, held by another session's job (F:/TanitAD_p5cread_tip). Waiting is silent (wait_commit.py prints only on success; each gate polls up to 120 min, then FAIL commit-low). Heartbeat to look for: `out/state/FTT15_s0.evidence.done` once the gate opens.
Expected end: about 3.3 h of work after the gates open, so about 15:10Z if memory frees at once (ESTIMATED from Amendment 2's MEASURED per-arm minutes); later by however long the gates wait.
SMOKE (honest account): one arm FTT15_s0, planner `--n 8` / 1 seed, run on the new tree through the route: evidence C2/C3/C5 hold; planner done; IA-3P-K read and delta done (12,827 anchors, gain point 1.10, CI [1.00, 1.25], MEASURED, not interpreted); `optb3_decide.py` ran end to end on aliased Amendment 2 plans (n_boot 20, labels meaningless) with all 16 window identities true. NOT smoked: `ia_read` (BAND-MET). `lr_lib.memory_guard` refuses under 6.0 GB free physical memory and the box held about 5 GB; the chain now waits for 6.5 GB (phys_gate) and runs BAND-MET FIRST for each arm, so a defect in that stage surfaces within minutes of the gate opening. Smoke outputs deleted (DELETIONS.log).

## What the chain does (all CPU, tier T0, one arm at a time, resumable: each stage writes out/state/<tag>.<stage>.done)
Order FTT15_s0, FTT15_s1, FTT125_s0, FTT125_s1 (banks `C:/Users/Admin/t2v8f_pulls/optb3_<tag>/`). Per arm:
1. evidence: sha256 vs RUN.done and BANK_MANIFEST; C2 and C3 from the banked files; C5 (129 of 129 Adam tensors, `wm.proj` bit-equal to the R35 start). Tool = the Amendment 2 `readchain_tools.py` unchanged; `shim/optb_check.py` makes it import the OPTB3 checker (FTT tags, RUN_ORDER).
2. planner (HWM oracle, inference seeds 0 and 1, lam 0, hold) through `route/optb_route.py`; about 45 min.
3. IA-3P-K read plus the beside read at 1.0 m/s^2; TARGET-READ (frozen-probe cycle gain) comes with the read.
4. BAND-MET (`ia_read.py`), about 5 min.
Then `optb3_decide.py` writes `out/DECIDE_OPTB3.json`: Floor_k = max(F_T,k, F_I) with F_I over every arm present; FTT15 and FTT125 against FTNULL of their seed; attribution FTT15 - FTT125; MECH-RAISED; TARGET-READ plus rsp/L1 trajectory; the four-family grid per band; STRATEGIC on 6 s reported in full; TACTICAL goal/anchor NOT COMPUTABLE with the reader's reason; per-chassis and combined section-4 label. The label computation is INSIDE the chain. Estimators: paired episode-cluster bootstrap B 2,000 seed 0 over 994 windows / 71 episodes = another draw of EPISODES only; F_T = another training run; F_I = another inference run.

Expected wall time: about 3.3 h for four arms plus the decide (MEASURED per-arm minutes of the Amendment 2 chain: about 50 min). The card's ESTIMATE was about 9 h.

## Differences from Amendment 2's read (and why)
* Model tree: `tree/` = `stack` + `taniteval` of commit 0f471d57 (the OPTB3 G-SUITE commit), 1,867 `.py` files, every sha256 equal to the git blob (`route/tree_manifest.sha256`, re-checked at each start). Reason: the Amendment 2 tree (86da79cd) raises TypeError on the checkpoint config field `wm_rsp_target`. 0f471d57 differs from 86da79cd only in the trainer flag, the response-loss target and the config field. The scorer, planner, reader, probes and windows are the Amendment 2 ones.
* FT-NULL and FT-RSP arms are NOT re-read: their plans and IA-3P-K reads are the banked Amendment 2 outputs under `F:/TanitAD-scratch/actrsp/optb/` (plans/, readchain/out/reads/).
* The pair is one lever seed per chassis (single paired delta), as the card says; BETTER needs CI wholly below 0 and |delta| > Floor_k (no second-seed sign test).

## Next step for the MM
1. Start the waiter: `bash C:/Users/Admin/actsens_launch/wait_optb3read.sh` (polls 600 s, cap 120 polls; prints one WAKE line).
2. On `WAKE done`: read `out/DECIDE_OPTB3.json` keys `outcome`, `chassis_labels`, `floors`, `decision_inputs`, `mech_rows`, `target_read`, `rsp_L1_trajectory`, `ftt125_line`, `controls`, `attribution_reading`. Check `window_identity_eid_now` is all true.
3. Interpretation flagged for the MM (the card is ambiguous, both are in the JSON): section 4 says attribution is read "on that quantity". The label uses the quantities where FTT15 is BETTER at that chassis; `attribution_reading.common_better_quantities_both_chassis` gives the stricter one-common-quantity reading.
4. On `WAKE fail`: the reason names the stage and the log under `out/logs/`. Resume by moving `out/ZZ_OPTB3READ.txt` aside and relaunching (finished stages are skipped). A `commit-low` fail means another session held the commit below 5.5 GB for 120 min.
5. Cleanup after the label is banked: `F:/TanitAD-scratch/optb3_read/smoke/` was deleted at build; `out/tmp` and `tree/` are re-creatable (tree from `git archive 0f471d57 stack taniteval`); the plans, reads and DECIDE json are the only copy until landed.

## Manifest (all under F:/TanitAD-scratch/optb3_read/ unless stated)
optb3_readchain.sh, optb3_readchain.cmd, optb3_decide.py, test_optb3_decide.py (4 tests), route/ (optb_route.py copy with tree default, tree_manifest.sha256, mk_manifest.py), shim/optb_check.py, tree/, DELETIONS.log, launch_job.ps1.txt (inline WMI body), HANDOFF_OPTB3READ.md; `C:/Users/Admin/actsens_launch/wait_optb3read.sh`. Outputs at run time: out/ (chain.log, state/, evidence/, plans/, reads/, reads_beside/, bandmet/, ia_read/, logs/, DECIDE_OPTB3.json, ZZ_OPTB3READ.txt).
