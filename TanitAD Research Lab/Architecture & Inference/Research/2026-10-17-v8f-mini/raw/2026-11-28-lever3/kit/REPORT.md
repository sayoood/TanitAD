# LEVER3 kit report (card PREREG_V8F_LEVER3_V2, sha256 707c5225..), 2026-11-28

## P1 control b3: PASS (B3_CHECK.json, b3_check.py, b3_setup_slices.py)
* Slices (E:/TanitAD-scratch/lever3/b3/{orig,repack}): 20 clips, 10 corpus (dev-box encode) + 10 from s001_smoke20 (pod encode); copies, own manifests. Repack = pool8x20 dropped, every other key byte-equal (b1 re-run per clip).
* Trainer path (the trainer's own build_data -> V8FMiniWindows -> ChunkedSampler seed 0 -> first batch, batch 64, yaw kin, slice tracks): first-batch sha256 9fcc5cd4565c9c7d equal on orig and repack; every clip through ClipStore.get and get_kin equal; join npz equal; window list equal (20 clips, 1,357 windows).
* Bank path (v8f_dagger_bank.py gen, pod s2 flags, first 3 TRAIN clips, 204 windows): content_sha256 c87d483d32ef37ff equal on orig and repack, K-B controls equal.
* Mutation: pool4x10 deleted from one repacked clip: trainer KeyError 'pool4x10' (rc 1); bank generator KeyError 'pool4x10' (rc 1). Both fail as required.
* Extra: the 10 s001 repacked slice clips equal the MM's cache_union files by md5 (10 of 10).
* Limits: lead labels were absent for the slice (lead_train path unset, as lead head is not in the argv); labels used the real v9 union npz, speed from the records tar, tracks from the corpus and 10k files.

## P2 bank kit: built, smoke PASS (bank_gen.py, l3_bank_check.py, rh_pilot_check.py copy)
* Flags exactly as the pod s2 bank (bank/s2/PROVENANCE.json). Generator ckpt UE_sK/A; sha256 pinned (s0 2fd4b747..., s1 f13df8ce..., MEASURED). eval grid sha256 659c3b65... equals the pod bank's recorded grid_sha256 (copy in inputs/).
* Smoke, 4 shards x 4 threads, 20-clip slice, 1,357 windows: PARTIAL bank, BANK_CHECK checks a-d pass (e fails by design for partial), L3_BANK_CHECK pass, BANK_DONE written. Bank sha256 of the smoke = feb88080...
* Speed MEASURED: 0.080 s per window per shard (4 threads each), same as one shard alone (0.083). Four shards: about 50 windows/s, so 985,048 windows is ESTIMATED 5.5 h per bank, plus about 30 s start-up per shard. (Windows from a 20-clip slice; the full union has longer LRU/IO costs: UNVERIFIED.)
* Memory MEASURED: free commit fell 9.3 -> 3.0 GiB with 4 shards (about 1.6 GiB each on the slice). bank_gen refuses unless free commit >= shards x 1.7 + 1.5 GiB (it did refuse at 7.7 GiB free). On the full union the shards may grow: UNVERIFIED.
* Join: bank shards LOAD join_union.npz; `bank_gen.py join` builds it once with the trainer's build_data under --fit-join (loads the 2 GB v9 npz; the slice version took about 70 s). Stage-A argv for the bank's train list is UE_s0.A.json from the pod record with paths mapped (the pod used chainlib.py m2bargv, not available; seed is not read by build_data: UNVERIFIED for s1 beyond that reasoning).

## P3 gate kit: PARTIAL
* Built and tested: l3_argv.py (renders D3K_s0/s1, D9K_s0/s1 from the RH_sK templates; control b4 verify; selftest: 10 planted regressions per arm all refused, PASS), l3_tree_prep.py (tree_cdb blob-by-blob vs git ls-tree 97fe0a56 in the mirror: 3,407 of 3,407, no extras, PASS).
* NOT built: G-SUITE port, dry mint, gate record, token. FINDING: the RH arms on the pod did not use launch_gate.py. Their token comes from the study-side checker dscale10k_check_s3.py (94 KB, kit_s3/), wrapped by chk_arm_s3.sh, with inputs that exist only on the pod (stage-A evidence and dry records, G-SUITE base junit of cdbff971, clips141.txt, pod holdout manifest, corpus join) plus the real bank, cache manifest and selfcheck.json of the arm. launch_gate.py in the tree only has refcv7/refcv8/v8f(refcv8-family) profiles, none for the V8FMini trainer. The card's phrase "launch_gate.py PASS" therefore needs an MM decision: port dscale10k_check_s3.py (D3K/D9K arms added) or add a profile (not allowed for me: launch_gate.py is pinned). Not attempted for lack of the inputs and budget.

## P4: written, NOT run (arm_run.py, l3_chain.py)
Both refuse without a token and the checker; I did not execute them (they launch work). Syntax checked only.
