# refcv8 EMA-48,852 videos on the PER-FRAME nav feed, RT layer ON (the shipped re-timed plan, table 2aacc9ff)

Tier: T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); inference seed 0; one training seed.
139/139 episodes rendered 2026-10-20 22:54-23:36 Berlin with map/box/tactical heads from `dump_heads_r8.py --feed perframe_full`
(heads feed parity PASS, render-side parity PASS). The long video (`LONG_REEL_EPISODES.txt`: every 10th episode index + the
three earlier rule picks 13 / 103 / 114, 17 episodes, 341.6 s) was built by `make_long_reel.sh` (stream-copy full quality +
a CRF-26 share copy). The mp4s stay on the dev box (`E:/refcv8_eval_kit/videos/final48852_pf/`, `C:/Users/Admin/refcv8_pf/reel48852/`);
`MP4_OFFBANK.sha256` pins all of them. No clip ids here (episode indices only; the heads log stays on the dev box).
