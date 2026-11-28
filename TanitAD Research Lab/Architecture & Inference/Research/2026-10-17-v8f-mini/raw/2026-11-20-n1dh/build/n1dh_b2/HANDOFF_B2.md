# HANDOFF_B2 (N1DH build, phase B2 prep) 2026-11-21
State: pod tree /root/strat1/n1dh/tree = cc8c98a1, 3,439 of 3,439 blobs verified, 0 bad; chain staged in /root/strat1/n1dh/, NOT started. Nothing heavy ran on the pod (one 0.8 s CPU render/pairing dry run, output deleted).
Lineage: N1D trained at 0c3b374f; M2b at the box-2 tip 6e5d3768; both are ancestors of cc8c98a1 (merge-base --is-ancestor). G-SUITE base = junit/n1dc (bcf3ba3f, the parent, run on this pod); the step re-verifies the base tree and requires exactly 14 differing entries.
Chain (done/<step>.done, relaunch resumes): ROLLHIST last line must be `ZZ ROLLHIST DONE ZZ` (N1DH_SKIP_RH_CHECK=1 overrides) -> flock fd 9 on /root/gpu_queue.lock to the end -> tree + inputs -> G-SUITE (new-only: 3 test files) -> render + pair + 5 negative controls -> N1DH_s0, N1DH_s1 (mint+selfcheck, pair, train 3,000 steps, post, zero-grad gate with controls route_emb.weight AND route_head.weight, K-CLIP) -> Z-USE copies + 8 D1-HO dumps (hb0.2 follow/zero/zuse x2 seeds, old follow x2).
Log /root/strat1/n1dh/n1dh_chain.log; last line `ZZ N1DH DONE ZZ` or `ZZ N1DH FAIL ZZ <why>`. Arms in runs/N1DH_s*; dumps in out/.
LAUNCH (MM; one line, the gate waits for ROLLHIST DONE, then starts the marker watcher and the chain):
  source C:/Users/Admin/lorapod/podenv.sh; pssh 'nohup setsid bash /root/strat1/n1dh/n1dh_launch_after_rollhist.sh < /dev/null > /dev/null 2>&1 &'
Waiter: `bash C:/Users/Admin/actsens_launch/wait_n1dh.sh` (0 DONE, 1 FAIL, 3 timeout). Gate log: n1dh_launch_after_rollhist.log (exit 1 if ROLLHIST FAILs, nothing started).
Left for the MM/dev box: real K-LIM (n1dh_klim.py on a real probe batch), Z-POST (confirm the dumps carry route_logits), eval-141 dumps, S1-slot view, the paired reads.
Temp: pod delta tar and render dir, local tar/extract deleted (DELETIONS.log).
