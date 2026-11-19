# HANDOFF_REPS1 (builder agent, 2026-11-19): UE_s1 + UM_s1 on the LoRA pod. BUILT, NOT LAUNCHED.

## State
* Chain `/root/dscale10k/blockreps1/` on the pod (queue_reps1.sh -> blockreps1.sh). DS_FAKE dry run: 33 of 33 scenarios PASS, ends `ZZ DSCALE10K REPS1 DONE ZZ` (`dryrun_reps1.sh`).
* Prep (`blockreps1_prep.sh`) was run by hand under `/root/gpu_queue.lock`: see PREP RESULT below. Argv pairing proof: `blockreps1/PAIRING_PROOF.json` (all pairs pass, 8 of 8 regressions refused).
* Nothing under `runs/UE_s0`, `runs/UM_s0`, `block3/`, `blockum/` or any pre-existing argv file was modified. New files only: `argv/UE_s1.A.json`, `argv/UM_s1.A.json`, `dry/UE_s1|UM_s1.A.dry.json`, `gate/init_ue_s1.json`, `gate/init_um_s1.json`, `gate/probe2/UE_s1|UM_s1.A.json`, `gate/evidence_UE_s1|UM_s1.json`, `gate/tokens/UE_s1.A`, `gate/tokens/UM_s1.A`, `runs/UM_s1/selfcheck_pre*`.

## PREP RESULT
Prep ran 05:27-05:41Z under the lock (GPU part about 1 min: two reach probes and UM's trainer --selfcheck-only; CPU: two dry builds about 4 min each). `ZZ REPS1-PREP DONE ZZ`.
* UE_s1 token (17 required checks) and UM_s1 token (19, incl. C-ARGV-MIX with its 7 G-RM regressions all refused, C-TACSRC-LIST) are minted: `gate/tokens/UE_s1.A/`, `gate/tokens/UM_s1.A/`. `gate/verify_*.A.json` pass:true, tampered-copy regressions all refused.
* G-HYG: 7,214,495 trainable on both. C-INIT: UE_s1 step-0 state, params, buffers and cfg equal U_s1's; UM_s1's equal UE_s1's and U_s1's (`gate/init_ue_s1.json`, `gate/init_um_s1.json`).
* Reach probes: `gate/probe2/UE_s1.A.json`, `UM_s1.A.json`, problems []. UM pre-launch selfcheck carries tac_src_probe (`runs/UM_s1/selfcheck_pre/`).
* Reused unchanged from seed 0 (INHERITED, same trees): G-EVAL, G-CKPT (incl. resume_um), G-CLOCK, G-SUITE (suite.json for tree, suite_um.json for tree_um), C-CACHE, C-SESS.
* Pod at the end of prep: 148 GB free, GPU idle (2 MiB), `/root/gpu_queue.lock` free, no STOP file.

## Launch (the MM, as its OWN ssh command, not inside an && list)
`source C:/Users/Admin/lorapod/podenv.sh; pssh 'cd /root/dscale10k && nohup setsid bash /root/dscale10k/blockreps1/queue_reps1.sh < /dev/null > /root/dscale10k/blockreps1.log 2>&1 &'`
(the same command text, run from the pod shell: `nohup setsid bash /root/dscale10k/blockreps1/queue_reps1.sh < /dev/null > /root/dscale10k/blockreps1.log 2>&1 &`). It takes `/root/gpu_queue.lock` and runs UE_s1, then UM_s1.

## Expected run time
UE_s0 took 13,804 s and UM_s0 11,554 s of trainer time (`runs/*/A/summary.json` `seconds`, MEASURED) = 3.8 h + 3.2 h, about 7.1 h plus a few minutes of gates, if the pod speed is unchanged. The PI's figure (about 8.2 pod-h) is ESTIMATED and covers it.

## Where things are
* Chain log: `/root/dscale10k/blockreps1.log`; per-arm: `runs/<ARM>/A.run.log`, `runs/<ARM>/A/` (summary.json, post.json, ckpt_last.pt, ARM_A_DONE), `runs/UM_s1/um_pre.out`, `um_post.out`, `um_checks_<pre|post>.json`.
* Sentinels (last line of the chain log): `ZZ DSCALE10K REPS1 DONE ZZ` or `ZZ DSCALE10K REPS1 FAIL ZZ <why>`; per-arm `ZZ ARM <ARM> DONE ZZ`. Stop between steps: `touch /root/dscale10k/chains/STOP`.
* Banking: the banker (`banker.log`) uploads each checkpoint to `<account>/tanitad-v8f-dscale10k-runs` under `UE_s1/A/` and `UM_s1/A/` and writes `bank_ok/<ARM>.A.sha256`; the sweep deletes a run's scratch only against bank_ok.

## The MM must check BEFORE launching
1. **The empty-batch bar.** The replay of UM_s1's sampler (MEASURED, CPU, `blockreps1/replay/um_empty_replay_s1.json`) gives 7 empty batches. The bar as written is <= 5, so `um_checks post` will FAIL UM_s1 after about 3.2 h of training (as it did UM_s0 at 10). The checkpoint is banked before that check, and no ARM_A_DONE is written. Decide first: keep the bar (UM_s1 reads BLOCKED (reach), exactly as the s0 arm was) or adopt UM-A1's exact-replay test for both seeds. Changing it means editing `um_checks_reps1.py` `post()` (one line) before launch.
2. Freeze `AMENDMENT_UM_A2_DRAFT.md` and fill its section 4 (how the s0 and s1 readings combine; the card has no rule).
3. Tree choice (draft section 2): each arm is on its seed-0 twin's tree. If UE_s1 should share UM_s1's tree (648d4ca3), the token set must be re-minted.
4. Free disk >= 60 GB (pod had 148 GB free at build time; the chain refuses below 60 GB). GPU idle and the lock free; `cat /root/dscale10k/chains/STOP` absent.
5. `ls /root/dscale10k/gate/tokens/UE_s1.A/gate_token.json /root/dscale10k/gate/tokens/UM_s1.A/gate_token.json` exist and `gate/verify_UE_s1.A.json`, `gate/verify_UM_s1.A.json` read `"pass": true`. If the commit tip of the trees moved, the runner will refuse (rc 2) and nothing trains.
6. The pod's code of record must not change between now and launch: a land touching `stack/tanitad` or the trainer after the token mint voids the tokens (the runner refuses, rc 2).
7. After DONE: the reads (UM_s1 - UE_s1 and the combination rule) are for a fresh read agent; this build does not read any evaluation number.
