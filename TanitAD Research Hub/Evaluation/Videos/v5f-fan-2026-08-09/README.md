# v5f diffusion-planner fan video — 2026-08-09 (banked 2026-08-10)

`v5f_fan_compact.mp4` (19.4 MB, 624px, crf28; md5 c76f05273e558c7f12b25e7c6cd5f13f).
Full-resolution original (179 MB, 1248px): HF `Sayood/tanitad-flagship-v5f-w120` /
`v5f_fan_full.mp4`; compact also mirrored there as `v5f_fan_compact.mp4`.

Content: 57 val episodes (9,600 windows @10 fps, ~16 min), front 120° cylindrical
camera (176x624 sub-frame upscaled 2x). Overlays: top-24-of-256 diffusion fan
(weight-alpha green), deployed pick (bold green), fan oracle (dashed cyan), GT
(pink), HUD with sel/oracle ADE + v0; bottom strip = goal_head TACTICAL scalars
(target v(5s), curv3s, time-to-manoeuvre) and STRATEGIC route bar
(tanh(curv_5s/CURV_TURN_PER_M), class threshold ±0.762). TIER T0 caveat on
frame: fan generated from encoded context (lambda_plan path), no future actions
enter the head; selection is the head's own sel_score argmax.

Generator: pod5:/workspace/render_v5f.py (gpu dump + render stages; HWC
9-channel stacked-temporal frames, last RGB slice rendered). Model:
flagship-v5f-w120-30k (registry §1.8), ckpt_30k_final.pt, oracle goal mode.
