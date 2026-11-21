# SELECTION_V2: pre-declaration of the v8F-mini evidence video v2 (written BEFORE any rendering or any counterfactual run; not edited afterwards)

`OUT` = `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz2`. Dev box, CPU only. Arm M2b, training seeds 0 and 1 (checkpoints sha256 prefix 601cb435 / 6104fb15). Model code = tip `9a6551d5` (exported to `OUT/code`). Nothing here is staged, committed or pushed.

## 1. Windows (exactly the four of v1, `viz/out/selection.json`; no other window may appear on any panel)

| row | stratum | eval window | clip | t (cache step; NOW) | cycle ages of that window (`--ages cycle`: a_T = i % 3, a_S = a_T + (i // 3) % 5; formula, before any run) |
|---|---|---|---|---|---|
| JUNCTION | JUNCTION median | #1131 | 79e30ac2 | 35 (7.0 s) | a_T 0, a_S 2 |
| CURVE | CURVE median | #654 | 45aea78c | 31 (6.2 s) | a_T 0, a_S 3 |
| NO-TURN | NO-TURN median | #763 | 51c01ba4 | 59 (11.8 s) | a_T 1, a_S 5 |
| FAILURE (worst) | worst of 2,399 | #2158 | ed87040c | 59 (11.8 s) | a_T 1, a_S 5 |

* Every panel is **seed 0** (M2b_s0). **Seed 1** appears in a summary card for every number quoted from a panel. The failure case stays in the video as it is.
* All window-level model runs use the window's own banked cycle ages, the eval's own loader and the eval's own item (`EvalWorld.item(i)` of v1: `v8f_mini_eval.cmd_run` lines 233-300), so "B" below is the banked own-code plan of that window.

## 2. Nav counterfactual (segment 3)

* Command vectors in the model's own nav encoding `[FOLLOW, TURN_L, TURN_R one-hot, d_next / 100 clipped to [0, 3], args_valid]`: FOLLOW `[1,0,0,0.3,1]`, TURN_L `[0,1,0,0.3,1]`, TURN_R `[0,0,1,0.3,1]` (d_next = 30 m, args_valid = 1). The command replaces the nav input of the named layer(s) at every history position (the layer reads `nav` at its own tick); **everything else is the window's own** (ego / prior features, speed limit, scene tokens, held ages, the upstream outputs that the chain computes from them).
* Injection points: (a) all three layers S, T, O; (b) O only; (c) T only; (d) S only. The change propagates downstream through the model's own chain (S checkpoints -> T -> T's code -> O). Layers run through the model's own `run_s` / `run_t` / `run_o` (no re-implementation of the cond assembly).
* Shown per case: the plan (`traj30`), S's three checkpoints (NOW frame), T's lateral argmax (LANE_KEEP / TURN_L / TURN_R / LANE_CHG_L / LANE_CHG_R). Also drawn for reference: the window's own-nav plan B, and the GT path.
* **Headline metric (fixed now):** D6(config) = mean over c in {TURN_L, TURN_R} of the Euclidean distance at t = 6.0 s (last of the 30 plan steps) between the plan under command c and the plan under FOLLOW, at the same injection point. Reported beside it: the left-right lateral separation at 6 s, |y(TURN_L) - y(TURN_R)|, and each command's own D6. Per injection point, per window, both seeds.
* Caption wording fixed: "intervention on the model's input, open loop; not driving".
* Optional context card (declared now so that it is not a later addition; no window is drawn on it): the same counterfactual over all 2,399 eval windows, mean D6 per injection point per seed with episode-cluster bootstrap (B = 2,000, 141 episodes; answers "another draw of EPISODES" only). If it is not run, the card is absent and the report says so.

## 3. Plans compared (segment 4)

GT (the eval's own context), P10 prior (the checkpoint's own frozen P10, `p10`), B = own-code plan (`tau`, the KIND run's banked `pred`), H = HBLEND's shrinkage plan, HS = HSPEC stage-2 plan. Arrays: `scratchpad/hblend/score_M2b_s{0,1}_perwindow.npz` (`tau`, `P`, `H`) and `scratchpad/hspec/score_hs2_M2b_s{0,1}_perwindow.npz` (`HS`), indexed by eval window index; the index is confirmed against `selection.json` (banked ADE6) in the controls. Eval-wide per-horizon ADE table (B / H / P10, plus HS beside) quoted from `hblend/score_M2b_s*.json` and `hspec/score_hs2_M2b_s*.json` with their estimator. The remaining gaps (acceleration at seed 0; the consistency cost) are stated as the source documents state them.

## 4. Strategic layer at its own tick (segment 5)

S's three checkpoints at its held tick (cycle ages of the window) vs fresh (`--ages fresh`: a_T = a_S = 0) vs GT (v9 RC-A labels in the NOW frame, `X['rc']`), on **two** windows fixed now: **JUNCTION #1131 and NO-TURN #763** (the two windows segment 3 treats in full). Eval-wide numbers quoted from `sfresh/SFRESH_RESULT.md` (fresh S - nav-only baseline at 50 m: -2.4331 m s0, -2.5260 m s1; banked cycle: +3.2826 / +3.2088).

## 5. Controls (`OUT/CONTROLS_V2.json`)

1. The window's own nav injected at all three layers reproduces the model's own forward (B's CPU plan), max |diff| 0.0, 4 windows x 2 seeds; plus a structural-zero arm: a copy of the checkpoint whose nav embedding weight is zeroed must give identical plans for the three commands at every injection point (D6 exactly 0.0) while the real checkpoint does not.
2. H / HS / B / P10 as drawn equal the banked per-window arrays (max |diff| <= 1e-5 m); ADE6 recomputed from the drawn plan and the eval's GT equals the banked per-window ADE6 (<= 1e-5 m); the CPU own-code plan vs banked `tau` and the CPU fresh / held S checkpoints vs the banked eval's checkpoints are reported with their actual maxima.
3. Frame alignment as in v1: cache step j is native frame 2j of the v2ep episode of `refav1-eval141-256rb`; v1's `align_control` (404 / 404 frames, step j the argmax, PSNR >= 48.6 dB against the banked eval frame) applies unchanged, and is re-read for every frame used here.

## 6. Video

`OUT/v8f_evidence_v2.mp4`, 1920 x 1080, H.264, about 90-150 s: 1 title; 2 who listens to what (`listen/fig_listen_grid.png`, numbers from `LISTEN_RESULT.md`, estimator and tier stated); 3 nav counterfactual (JUNCTION and NO-TURN in full, the other two briefly, then a seeds-0-and-1 summary card); 4 better plans; 5 strategic layer at its own tick; 6 closing card (measured / tier / not shown / artifact paths). Every panel carries a legend. QA stills `OUT/qa_*.png` (at least 6). Every quoted number has its artifact path in `OUT/VIDEO_NOTES.md`.
