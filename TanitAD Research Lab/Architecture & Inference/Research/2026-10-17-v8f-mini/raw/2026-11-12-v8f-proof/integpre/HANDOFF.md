# HANDOFF integpre (build agent, card PREREG_V8F_PROOF_ADD44 INTEG-PRE, 2026-11-19)

State: tool chain built and tested on the dev box; the new pod folder `/workspace/v8proof/integpre` holds the pushed tools (sha256 equal) and one DRY=1 output (`dry_pod.out`). No GPU job, no extract, no fit, no rollout was started. Card not edited. No git.
Tiers: states, lead read, fits T0; rollouts T1-kin (kinematic rollout on the model's own actions; the lead does not react); never driving performance.

## Launch (lead)
`cd /f/TanitAD-scratch/v8proof/integpre && nohup bash chain_ip1.sh > chain_ip1.out 2>&1 < /dev/null &`   (DRY=1 prints the plan; last line `ZZ CHAIN-IP1 DONE` or `STOP`; refuses if IP_MAX_WINDOWS is set)
Then `bash read_ip1.sh` (numpy only, through ops/pycap.sh; writes `read/READ_IP1.json` and `READ_IP1.md`). Pod end markers: `/workspace/v8proof/integpre/run/INTEGPRE_DONE.txt` / `INTEGPRE_STOP.txt`; a second start is refused by `INTEGPRE_STARTED.txt`.

## Sizes (MEASURED 2026-11-19 on the pod)
One existing all-training-clips state dump (`qhead_dose/states/LSD_s0_dose_train.npz`, 4,572 clips, D 640) = 594,171,496 bytes; its lead read 6,001,012 + 331,734. Expected: 8 x 600,504,242 + 26 rollouts x 30 MB = 5,584,033,936 bytes (5.2 GB), printed by the queue; it refuses if the volume (`stat -f` on /workspace/v8proof, not df) has less than that plus 10 GB (now 46 GB container memory free, about 429 GB volume free).
DRY output (dev box and pod): 103 lines = 50 slot waits, 51 commands (8 x extract, lead read, fit; 24 rollouts; 2 control rollouts; 1 compare), the volume line, the end-marker line.

## Files (dev box `F:/TanitAD-scratch/v8proof/integpre/`; pod `/workspace/v8proof/integpre/` flat)
pod_integpre_queue.sh, integpre_arms.txt, tools/{pod_ruleattr_run.sh, dist_rule.py, cmp_dumps.py, v8f_t1_eval_rule.py, v8f_t1_eval_rule_mix.py, v8f_t1_eval_rule_qhead.py, qhead_model.py, qhead_fit_n1.py} (copies, sha256 equal to the banked pod files; all pinned in the queue), chain_ip1.sh, read_ip1.py, read_ip1.sh, ip_testlib.py, test_read_ip1.py, test_queue_ip.py, run_mutations_ip1.py, MUTATIONS_INTEGPRE.json.
Pod inputs by reference (pinned by sha256 in the queue): `probe/extract_states.py`, `probe/pod_lead_apply.py`, `qhead_dose/pack/train_pack_all.npz`, tree `tree_actcons`, checkpoints `runs/actcons_c0|c1|e2|e3/<RLAm|RLFT>_s<k>/ckpt_last.pt`.
Outputs: rollouts `runs/distrule/t1/IR0_|IRL_|IRQ_<arm>_s<k>` plus `IRH_RLAm_s0`, `IRQI_RLAm_s0` (the tool's fixed location, new names only); states, heads, run log under `integpre/`. Pulled: `pull_t1/`, `pull_runs/`, `pull_small/`.

## Tests and mutations
39 tests green (24 reader, 9 queue incl. params; unmutated suite green before and after). Mutations 32 of 32 RED (`MUTATIONS_INTEGPRE.json`, all_ok true; 21 reader, 11 queue; the three card mutations are `queue_wrong_checkpoint_for_a_head`, `queue_head_of_another_base`, `queue_rule_form_swapped_*` x3). One mutation (c3) first read NOT RED; a planted case was added, then the whole list was rerun. Tests: test_read_ip1.py (planted cases for every name in both directions, missing-base rule, a failed control for each of c1 to c6), test_queue_ip.py (DRY parse: each checkpoint uses its own files, rule forms, c3 placement, slot wait before every job, volume refusal, second start refused).

## REVIEW items
1. P: no P* was calibrated for RLAm / RLFT; the arms file uses 0.5 (the threshold of the imagined-rollout planner on them, and the value of control c3). Change in `integpre_arms.txt` before the launch if the lead wants another.
2. The fit uses all 4,572 train positions (fit 3048 / select 762 / check 762) as in the dose step; the RL fine-tunes excluded clips 540:600 from training (argv), those 60 sit in the fit set.
3. A failed control withholds every name, also a per-base c6 failure; only a head outside the band or a missing dump drops a base. The card says "a failed control of that base" counts as missing: read literally, I withhold instead (stricter).
4. Price name from the family reader's `cost_families` (`final` flag printed); `INTEG-PRE-COSTS-<family>` per family.
5. c6 is read as the headway pair `IR0_<arm> : IPox_<arm>` per-base difference <= 1e-9 (banked IPox dumps for bases 0,1 in `C:/Users/Admin/v8proof_c/pull`, 2,3 in `imagplan/pull_t1`).
6. Planner form (cand-cvq) is not built: note ADD40_N7 had not read at the freeze; the card says it names nothing.

## Not verified
Extract / lead read / fit / rollout on the RL checkpoints were never run (tree_actcons has `lead_head`, 36 hits; the first extract is the real test, fail closed). `read_ip1.sh` and the two reader tools were not run on real INTEG-PRE dumps; the JSON keys come from read_q1.py. The slot-wait functions are copies of the ADD42 queue, not mutation-tested here.

## Strays
Deleted (my own scratch, literal paths): `integpre/work/bt` (basetemp, about 4 MB), `work/mut` (about 1 MB), `work/t1` (about 1 MB), `work/t_*.tmp`, `work/pt.out`, `work/ptq.out`. Kept: `work/mut_run.out`. Pod: `integpre/` only (my own pushed files, `dry_pod.out`). Nothing deleted elsewhere. A Python `__pycache__` was not written (PYTHONDONTWRITEBYTECODE=1).
