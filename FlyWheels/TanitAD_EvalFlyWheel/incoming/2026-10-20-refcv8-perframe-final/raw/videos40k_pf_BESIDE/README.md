# refcv8 EMA-40k videos on the PER-FRAME nav feed (BESIDE, not the 48,852 read)

*Clip ids: none in this folder (episode indices only; the heads dump log, which prints id prefixes, stays on the dev box).*
Tier: T1 self-action OPEN LOOP on logged eval139 frames, a diagnostic, not driving performance. Inference seed 0, single training seed.

* What: all 139 eval139 episodes rendered by the landed per-frame video code (1a323fd; `render_all_r8.py --tree step40000ema_pf`,
  heads from `dump_heads_r8.py --feed perframe_full`), 1920x1080, 10 Hz, 20.1 s each, CRF 18. Panels: front camera (cylindrical),
  metric BEV with GT agents and the box head, GT (SAM3) map vs the refcv8 map head, tactical decoder (lat/lon softmax, NAV FED this
  frame vs the clip nav not fed), ADE metrics vs refcv7-50,400, speed plot (plan vs logged human speed vs fed ceiling).
* Feed: `perframe_full` in every `RENDER_LOG.jsonl` line; heads feed parity vs the roll sidecar AND the independent v9 lookup:
  `FEED_PARITY_heads_40000_pf.json` status PASS. 139/139 episodes `ok: true`.
* RT layer OFF on purpose: the re-timed deploy npz for this tree lives only in the rehearsal package, and copying it into the live
  `pfpkg` would plant a foreign `PRIOR_PARITY.json` before tonight's 48,852 lane (see `../rehearsal40k_BESIDE/`). The 48,852 videos
  render after the binding lane, with RT ON if the lane wrote its deploy npz.
* Chain: `run_v40pf.sh` (heads under the dev-box GPU lock 13:21-14:00, check, render 14:00-14:40 Berlin), `run_v40pf_b.sh` (re-arm, SKIPPED
  because the first chain finished). Markers: `v40pf_markers.log`.
* The mp4s stay on the dev box (`E:/refcv8_eval_kit/videos/v40000_pf/`, 1.3 GB); `MP4_OFFBANK.sha256` pins all 139.
* Showcase picks (the existing rule-based picks from `2026-10-19-refcv8-videos/raw/PICKS_step40000ema.json`: R1 cruise false-brake
  ep 114, R2 sharp turn ep 13, R3 stop/slow-down ep 103; R2 and R3 depend only on the logged ego path; R1 was picked on the CLIP-fed plan).
  On the per-frame feed the fed token differs from the clip token on 34/34 windows of ep 114 and 29/34 of ep 13 (`RENDER_LOG.jsonl`).
