# v1.7 (flagship-v17-speedloss) — continuous real-time video

**2026-08-06** · 683 s (11.4 min) · one frame per stride-1 window at 10 fps = real time ·
all 6,834 windows, 40 OOD-val q90 episodes · md5 `6fb443a9dd314a279162d7701ff72d58`
(verified matching pod4).

Overlays: **GT pink · v1.7 green** · camera pane (FlatProjector into the model's 256×256
input) + metric BEV + per-frame ADE/v0 HUD. Burned-in caveat: **WM rollout under TRUE
future actions — NOT closed-loop planning** (closed-loop numbers: registry §1.12).

v1.7 = v1.6 + speed-profile L1 (pre-registered run 6, outcome B — better ADE/speed, NOT
the lag fix). Numbers: registry §1.11; renderer `…/2026-08-06-v1-defect-triage/tools/render_v17_cont.py`
reading the banked closed-loop dump's `o6` arm — no re-inference.
