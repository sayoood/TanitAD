# HANDOFF_POD_RK (KINLAB RK pod kit, build + dry mint). Nothing staged, committed or launched. Kit: F:/TanitAD-scratch/kinlab/pod/ (13 files + SHA256_RK.txt); also on the pod in /root/rollhist/kit_rk/ (verified).
State: built and tested on the dev box; read extension done and dry-checked; CPU dry mint on the pod passed at both seeds: census 0.6398 (s0) / 0.6397 (s1), bar 0.5 met; 21 of 21 checks (G-SUITE planted in the scratch root only); DRY_MINT_RK.md has the per-tick table. `dry_rk/` deleted (DELETIONS.log).
Before the launch (MM): (1) PI go with the cost; read DRY_MINT_RK.md "labeller wall" first: the kin labeller runs on the CPU at every load, twice per arm (selfcheck and train) under the lock; (2) G-SUITE runs inside the chain (~37 min, before the lock).
Ship (dev box, Git bash; `source C:/Users/Admin/lorapod/podenv.sh`):
  cd F:/TanitAD-scratch/kinlab/pod
  pscp rk_env.sh rk_tree_prep.sh rk_gsuite.sh rk_gate_record.py rh_rk_chain.sh rk_arm.sh chk_arm_rk.sh rk_argv.py rh_rk_marker_watch.sh launch_rk.sh all_blobs_rk.txt rk_delta.tar SHA256_RK.txt "$POD_TARGET:/root/rollhist/"
  pscp dscale10k_check_rk.py "$POD_TARGET:/root/rollhist/study/"
  pssh 'cd /root/rollhist && sha256sum -c SHA256_RK.txt 2>&1 | grep -v "dscale10k_check_rk.py: FAILED open"; sha256sum study/dscale10k_check_rk.py; for f in rk_env.sh rk_tree_prep.sh rk_gsuite.sh rh_rk_chain.sh rk_arm.sh chk_arm_rk.sh rh_rk_marker_watch.sh launch_rk.sh; do bash -n $f && echo ok $f; done'
  (the checker's line is checked by hand: 273befbc8cba48ae... in SHA256_RK.txt)
Launch (only after the ADD123 chain is running or ended; the chain waits for it): pssh 'bash /root/rollhist/launch_rk.sh'   -> `ZZ RK LAUNCHED ZZ`
  Log /root/rollhist/rh_rk_chain.log, last line `ZZ ROLLHIST-RK DONE ZZ` or `... FAIL ZZ <why>`; watcher log /root/rollhist/rk_marker_watch.log. Abort before the lock: `touch /root/rollhist/chains/STOP` (read between steps).
Read (dev box, after both ARM_DONE): WMI-launch rh_read_rk_retry.cmd (runs rh_read_rk_chain.sh, resumable); MECH3 is its step `mech3` -> out/read_rk/MECH3_KINLAB.json; result out/read_rk/RESULT_ROLLHIST_RK.json.
Facts, tests, limits: FACTS_POD_RK.md. Open for the MM: (a) the marker take-over also fires on ADD123's `marker removed (mine)` line (one pattern to drop for the literal form); (b) the real G-SUITE at 0ca6f49d has not run; (c) COST: the kin labeller takes ~1,640 s per load on the loaded pod (RL's speed_v1: ~195 s) and runs twice per arm under the lock: projected 1.5-2.3 pod-h vs the card's 0.6-0.75 and the 0.9 line: back to the PI before launch.
