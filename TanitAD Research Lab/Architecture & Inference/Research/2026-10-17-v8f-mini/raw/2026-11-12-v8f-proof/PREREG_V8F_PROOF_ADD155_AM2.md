# ADD155 Amendment 2 — A5 centre-matched control (frozen before its data)

Written 2026-11-28 after the A5 primary read (USED at all three seeds; `C:/Users/Admin/wua_work/a5/RESULT_A5.json`) and BEFORE any planned speed of this control was computed. Amendment 1 (`wua/a5/IMPL_A5_AM1.md`) is the A5 history-length note; this is Amendment 2 of the card. It does not change the A5 primary label; it adds a control whose result is reported beside it and decides one added name.

**Why.** The card's CONTROL cells are mirrored about the image centre and shifted outward, so they sit at the periphery, while the LEAD cells sit near the centre, which also carries the road ahead. USED therefore does not separate "the planner reads the lead" from "the planner reads the centre of the image". Post hoc in origin (prompted by the control's placement, seen in the A5 hand-back), pre-registered before its own data.

**Control C-CENTRE.** For every kept A5 LEAD window w (323 windows, 48 clips) with LEAD mask S_w (all 16 history frames, as AM1): pick up to 3 NO-LEAD windows from the same eval grid windows (`rhacc10k/pins/eval_grid_windows.json`), from other clips, with join `lead_valid` = 0 at their frame t, no agent of the agent join with 0 < x <= 50 m and |y| <= 2.0 m at raw frame 2t, and cache ego speed at t within 1.0 m/s of w's; the 3 nearest in speed, ties by (clip id, t). Apply S_w (the same cells per history slot) with the same fill (each cell's mean token) to each and read the planned speed at 2.0 s exactly as A5 (`wua_a5.speed2s` of `forward_infer` traj30). dv_centre(w) = mean over its matched windows of |v(masked) - v(base)|.

**Metric:** |dv(LEAD)(w)| - dv_centre(w), per lead window, episode-cluster bootstrap over the lead windows' clips (B 2,000, rng 0); windows with no match are dropped and counted. **Name:** `A5-LEAD-SPECIFIC` if the lower bound > 0 at all three seeds; `A5-CENTRE-NOT-EXCLUDED` otherwise. Beside: the signed v(masked) - v(base) on no-lead windows (a centre effect that raises speed on empty road is not a lead effect). Control: an empty mask on the no-lead windows gives dv exactly 0.0.

Tier T0. Four families do not apply (card section 0).
