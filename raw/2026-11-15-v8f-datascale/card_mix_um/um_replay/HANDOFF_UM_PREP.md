# HANDOFF_UM: UM_s0 launch-readiness (2026-11-18, builder agent)

State: everything is built and measured on the pod except the stage-A token. The mint is blocked by ONE new-test failure in G-SUITE (below). UM_s0 training was NOT started.

## Blocker
G-SUITE on tree_um (cdbff971 + 3 files of 42ef66da): 0 regressions, base 14,851 tests (12,524 pass / 431 fail / 205 error / 1,691 skip) vs new 14,883 (12,555 / 436 / 205 / 1,687), 32 new-only tests, 0 base tests missing. BUT 1 new test fails on the pod:
`test_v8f_dscale_mix.py::test_model_mutation_turns_its_check_RED[c1_upper_identity_epsilon]` (DID NOT RAISE). The mutation `w * (rw + 1e-3)` is a uniform scaling of the goal weights, so it cancels in the weighted mean and is seen only through float rounding (RED on the dev box, not on the pod). Fix (tested on a scratch copy on the pod: the whole file 31 passed, 1 skipped): `F:/TanitAD-scratch/datascale/um_ready/test_mutation_fix.diff` (one line, non-uniform epsilon). It changes the code of record, so the MM lands it as a new child commit, then:
1. update the tree: overlay the new test blob on `/root/dscale10k/tree_um`, re-verify blobs (`blockum_vb.py`: edit the expected sha of the test file),
2. replace the commit id `42ef66da6ca9169b8db67c1c91669e80dc2dae59` in `blockum/gsuite_um.sh`, `blockum/chk_armum.sh`, `blockum/verify_token_um.py`, `blockum/um_checks.py`, and `CODE_COMMIT` in `blockum/study/dscale10k_check.py`,
3. rerun `gsuite_um.sh` (about 25 min CPU wall), then `blockum_prep.sh` (it re-runs the suite item and mints). The reach probe / selfcheck / smoke results do not depend on the test file.
Also: 5 base-skip -> new-fail tests (test_v8f_wc x3, test_refcv7_vis1 x2) are environmental: `/root/dscale10k/tree` (cdbff971, control) fails the same 5; `/root/strat1/base` lacks the partial `TanitAD Research Lab` dir so they skip there. Not a regression by the gate's rule (pass -> fail).

## Done on the pod (`/root/dscale10k`)
tree_um (3,401 blobs verified), mix/bset_clips.txt (4,512 ids, sha256 37f4f9ee..., equals the card), argv/UM_s0.A.json, dry/UM_s0.A.dry.json (14,512 clips, S_U 15,392, train_sha256 9844151f..., join_union.npz sha256 unchanged), gate/init_um.json PASS (state sha256 equals UE_s0 and U_s0), gate/probe2/UM_s0.A.json (fresh reach), runs/UM_s0/selfcheck_pre (trainer's tac_src_probe checks 1 to 5), smoke_um/run200 (200 real union steps), gate/resume_um.json PASS, blockum/ (chain, prep, queue, checker fork, dry run 31/31 PASS).
## Not done
Evidence assemble, mint, verify_token (wait for suite_um PASS); G-EVAL, G-CKPT (block-1 part) and G-CLOCK are the block-1 records applied by sha (INHERITED), as block 3 did.
Deletions: `/root/dscale10k/blockum/deletions_um.log`.
