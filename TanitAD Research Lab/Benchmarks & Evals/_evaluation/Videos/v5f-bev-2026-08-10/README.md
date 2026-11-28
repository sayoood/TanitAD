# v5f video v2 — clean camera + BEV fan pane (PI request 2026-08-10)

`v5f_bev_compact.mp4` (19.8 MB, 624px, crf28; md5 d9151ad88a3a5e94b89bda9c7d78e2f3).
Full-res (1248px): HF `Sayood/tanitad-flagship-v5f-w120` / `v5f_bev_full.mp4`
(compact mirrored there too). Same 57-episode / 9,600-window grid as
v5f-fan-2026-08-09.

Camera pane: ONLY GT (pink) + the v5f selected plan (bold green) — fan removed
per PI remark. BEV pane below (ego frame, forward=up, 10 m range rings, 60 m
shown): full dumped fan top-24-of-256 (weight-alpha green), selected (bold
green), fan-oracle (dashed cyan), GT (pink), ego marker. Bottom strip: tactical
scalars (target v(5s), curv3s, ttm) + strategic route bar from goal_head.
TIER T0 caveat on-frame. Generator: pod5:/workspace/render_v5f_bev.py over the
existing /workspace/v5f_viz dump (render-only; frames+mp4 on container /tmp to
stay off MooseFS after the 05:00Z I/O incident).
