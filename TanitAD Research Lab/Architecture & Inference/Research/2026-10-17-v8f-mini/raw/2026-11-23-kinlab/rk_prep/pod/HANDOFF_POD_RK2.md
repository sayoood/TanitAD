# HANDOFF_POD_RK2 (RK rev 2, commit 1a8b11a1 + Amendment 1). Prep DONE on the pod; chain NOT launched; nothing staged/committed.
Kit `/root/rollhist/kit_rk/` (SHA256_RK2.txt verified; rev1 files kept as `*.rev1`). Results: PREP_RK2.md.
State: tree_rk 3420/3420 blobs OK; caches s0 c882733d..4ea (0.6398), s1 5a99afdc..4dd (0.6397), bar 0.5 met; G-SUITE 0/0/0 at 1a8b11a1; DRY walk PASS.
Before launch (MM): PI go with cost; ADD123 chain running or ended (the chain waits for its `ZZ ADD123 ` line).
Check (writes nothing):  pssh 'DRY=1 bash /root/rollhist/kit_rk/rh_rk_chain.sh'   -> `ZZ ROLLHIST-RK DRY PASS ZZ`
Launch:                  pssh 'bash /root/rollhist/kit_rk/launch_rk.sh'            -> `ZZ RK LAUNCHED ZZ`
Log /root/rollhist/rh_rk_chain.log, last line `ZZ ROLLHIST-RK DONE ZZ` or `... FAIL ZZ <why>`; abort before the lock: `touch /root/rollhist/chains/STOP`.
The chain re-validates tree, G-SUITE result, caches (sha256 of the files now) and argv in seconds BEFORE the lock; any refusal is before the lock. Do not edit v8f_dagger.py / v8f_recovery.py or the banks: a cache then refuses (rerun `bash kit_rk/prep_rk2_run.sh` after renaming the cache).
Lock time projection: precompute (2 x 1,645 s) and G-SUITE (37.5 min) are now outside the lock. Under the lock per arm: selfcheck with a cache hit + 3,000 steps; the ~1,640 s labeller x2 per arm of rev1 is gone. ESTIMATED lock 0.6-0.75 pod-h per the card (no RK training wall measured; hit read cost unmeasured).
Read after both ARM_DONE: as HANDOFF_POD_RK.md (rh_read_rk_retry.cmd; MECH3 -> out/read_rk/MECH3_KINLAB.json).
Open: (a) marker take-over also fires on ADD123's `marker removed (mine)` (unchanged from rev1); (b) first real cache HIT under the GPU selfcheck is unobserved; (c) checker patched to admit the cache pair (see PREP_RK2.md).
