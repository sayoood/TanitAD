rm tmp_dl/small_set.tar 372654080 B
P1: RH s0-2 copied from rollhist/read (sha ok), UE s0-2 + TK16 s0,s1 + qmap q50 s0,s1 extracted; small_set.tar deleted (372654080 B)
P2: tree_ov (867 files) + build helpers + pins pulled and sha-verified vs pod manifests (tree_pod_manifest.txt, kit_build/build_pod_manifest.txt, pins_pod_manifest.txt)
rm tree_ov.tgz 13355142 B
P3: box2 D:\tanitad\rhacc10k laid out; tree+build+inputs+pins sha-verified on box2 (D: free before 4.29 GB)
P4: kit/box2_rhacc_queue.py written (24 runs, amendment 1)
P5: kit/test_box2_rhacc_queue.py written; pytest waiting on the pycap mutex (another capped job) at time of writing
P6: box2_dry.out saved (24 argv, 0 checkpoints missing; no run folder created on box 2)
rm tmp_dl/MANIFEST_FILES.txt 8234017 B
P5: 39 tests pass (plain venv python, pycap mutex was held by another job for >10 min; tests are small); rm tmp_t/bt (pytest basetemp, few KB)
P6: box2_dry.out re-saved after final driver (24 CMD lines)
