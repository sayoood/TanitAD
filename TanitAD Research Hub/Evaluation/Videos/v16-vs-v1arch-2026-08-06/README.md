# v1.6 (unicycle readout) vs v1arch — 7.3-minute open-loop reel

**2026-08-06** · 440 s · 165 windows (every 40th stride-1 window) over all 40 OOD-val q90
episodes · md5 `ef65fd27e14688e1448361454b1bc180`.

Overlays (established palette): **GT pink · v1arch orange · v1.6 green**, projected into the
model's actual 256×256 front-crop input (FlatProjector) + metric BEV with range rings +
per-window ADE / net-yaw / jerk HUD.

⛔ Banner caveats, burned into every frame: both arms are **action-conditioned WM rollouts
under TRUE future actions — not closed-loop planning**; v1.6 = the 2.11 M
`flagship-v16-unicycle` readout on the FROZEN v1arch trunk; **not** the 2026-07
`flagship-v16-ab-ft` "v1.6".

Numbers behind the reel: registry §1.10 / `…/2026-08-06-v1-defect-triage/results/v16_full_eval.json.xz`
(paired episode-cluster bootstrap, 6,834 windows).

## v16_continuous.mp4 — the continuous, real-time cut (requested by Sayed 2026-08-06)

**683 s (11.4 min) · one video frame per stride-1 window · 10 fps = real time** · all 6,834
windows of all 40 OOD-val q90 episodes, in order · md5 `a9a930dd0b03d5a45d46803d71c5dbb2`
(verified matching pod4 after transfer).

Differences from the reel above, per the request: **continuous playback, not held frames**, and
**only two overlays — GT pink · v1.6 green** (v1arch omitted). Because every frame is a new
stride-1 window, what you see at each instant is the **freshly re-predicted 2 s / 20-waypoint
plan** — the frame-to-frame stability of the green path IS v1.6's replan smoothness (the metric
that improved 11× over v1arch's step-readout).

Same burned-in caveat applies: **WM rollout under TRUE future actions — NOT closed-loop
planning.** Camera pane (FlatProjector into the model's 256×256 input) + metric BEV + per-frame
ADE / v0 HUD + episode/time banner.

Renderer: `…/2026-08-06-v1-defect-triage/tools/render_v16_cont.py` reading the banked eval dump
(`/workspace/v16_eval/dump/ep*.npz`) — no re-inference; the drawn paths are byte-identical to the
scored ones.
