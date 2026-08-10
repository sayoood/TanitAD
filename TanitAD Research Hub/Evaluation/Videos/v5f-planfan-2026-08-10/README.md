# v5f video v3 — FULL 256-anchor plan-fan BEV (REF-C plan_fan conventions)

`v5f_planfan_compact.mp4` (20.1 MB, 624px; md5 656cdac72f8a09060a9dddf71a4c2945).
Full-res: HF `Sayood/tanitad-flagship-v5f-w120` / `v5f_planfan_full.mp4`.
Same 57-episode / 9,600-window grid as the two prior reels.

PI ask: "the whole anchor fan visualization like the refc visualization."
Layers per taniteval/plan_fan.py: (1) raw anchor VOCABULARY shadow (faint grey);
(2) ALL 256 refined proposals, viridis by softmax(sel_score) on the FIXED log
scale (P_FLOOR 1e-4), alpha+width scale with score, ascending draw order;
(3) top-8 emphasis + waypoint dots; (4) SELECTED plan halo+white core; (5) GT
dashed green + fan-oracle dashed cyan; (6) colorbar + 10 m rings + modes>1%
HUD. Camera pane (top): GT + selected ONLY (v2 request kept). Bottom strip:
tactical scalars + strategic route bar from goal_head. TIER T0 caveat on-frame.

Generators: pod5:/workspace/x_fanfull.py (full-fan GPU dump, FF_EXIT=0, joins
/workspace/v5f_viz frames by construction-identical window order) +
pod5:/workspace/render_v5f_bev... (render_v5f_fanfull.py; frames+mp4 on
container /tmp). Model: flagship-v5f-w120-30k, oracle goal mode.
