# HANDOFF_B5 (ROLLHIST pod repoint to 43319227; nothing launched)
State: /root/rollhist/tree_cdb = 43319227, 3407/3407 blobs verified, 0 bad. Scripts repointed (commit, tree_cdb, blob list all_blobs_43319227.txt), kit installed (dscale10k_check_rh.py, rh_evidence.py new; chk_arm_rh.sh, rh_arm.sh already identical), `.b2bda` backups kept, bash -n ok. Old markers in done_2bda9e1a/, fresh done/. RH_READOUT.chosen=causal_g1x kept. G-SUITE base /root/strat1/junit/base (cdbff971). Pilot check (a) unpatched tree /root/dscale10k/tree = cdbff971 exactly (3400/3400 blobs).
Untested on pod: gate token path (chk_arm_rh.sh mint/run) first real run; expect a possible path fix.
LAUNCH (MM; chain GPU-gated by the flock on /root/gpu_queue.lock, trainer session holds it until ~01:25Z):
1. `source C:/Users/Admin/lorapod/podenv.sh; pssh 'nohup setsid bash /root/rollhist/rh_marker_watch.sh < /dev/null > /dev/null 2>&1 &'`
2. `pssh 'RH_READOUT=causal_g1x nohup setsid bash /root/rollhist/rh_chain.sh < /dev/null > /dev/null 2>&1 &'`
3. `bash C:/Users/Admin/actsens_launch/wait_rollhist.sh` (0 DONE, 1 FAIL, 3 timeout)
Resume after FAIL: same line. Deletable once verified: /root/rollhist/tree (2bda9e1a), done_2bda9e1a. Logged in /root/rollhist/deletions.log.
