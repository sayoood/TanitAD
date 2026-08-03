# AlpaSim OPEN-LOOP videos — rendered on the Jetson Thor, 2026-08-03

⭐ **These are the programme's FIRST open-loop AlpaSim videos.** Every previous AlpaSim clip and number is closed loop.

**OPEN LOOP means the ego follows the LOGGED trajectory.** Each frame is rendered at the pose the rig actually had, the model consumes that frame stack and emits a plan, and the plan is scored against the log's own future motion. **The model never drives.** No controller step, no bicycle integration, no divergence.

**Why it exists.** In closed loop, perception error and control drift are confounded: a bad frame moves the car, which produces a worse frame. That is not a hypothetical here — MEASURED 2026-08-03, flagship v1's driven path moved a mean **9.05 m (max 37.78)** from a *render change alone*. Open loop pins the observation distribution to the log, so what is left is prediction.

## The eight videos

| file | scene | arm | traffic | frames | duration | decode |
|---|---|---|---|---|---|---|
| `./flagship-v1_openloop_empty_road.mp4` | 00040136 (night) | flagship v1 | empty road | 190 | 19.0 s | ✅ 0 errors |
| `./flagship-v1_openloop_with_objects.mp4` | 00040136 (night) | flagship v1 | with objects | 190 | 19.0 s | ✅ 0 errors |
| `./refc-base_openloop_empty_road.mp4` | 00040136 (night) | REF-C base | empty road | 190 | 19.0 s | ✅ 0 errors |
| `./refc-base_openloop_with_objects.mp4` | 00040136 (night) | REF-C base | with objects | 190 | 19.0 s | ✅ 0 errors |
| `junction-7c72937c/flagship-v1_openloop_empty_road.mp4` | 7c72937c (JUNCTION, day) | flagship v1 | empty road | 190 | 19.0 s | ✅ 0 errors |
| `junction-7c72937c/flagship-v1_openloop_with_objects.mp4` | 7c72937c (JUNCTION, day) | flagship v1 | with objects | 190 | 19.0 s | ✅ 0 errors |
| `junction-7c72937c/refc-base_openloop_empty_road.mp4` | 7c72937c (JUNCTION, day) | REF-C base | empty road | 190 | 19.0 s | ✅ 0 errors |
| `junction-7c72937c/refc-base_openloop_with_objects.mp4` | 7c72937c (JUNCTION, day) | REF-C base | with objects | 190 | 19.0 s | ✅ 0 errors |

Each is **1800×850 @ 10 fps**, front camera + metric BEV inset + decision HUD, with **OPEN-LOOP burned into the frame** and a legend naming ground truth vs prediction. Every file was verified by **decoding it back** and md5-matched against the Thor copy — `video_verification.json` holds both.

## The render — the 2026-08-03 chosen configuration

`layers=background,road` + all dynamic layers + **scale-cull 0.95** + **gated sky gain 0.3** (ramp 0.0–6.0° above horizon).

| | grad-NCC ↑ | neg-control margin ↑ | ms/frame |
|---|---|---|---|
| morning (`background+road`) | 0.2774 | +0.0873 | 23.3 |
| **this render** | **0.3424 (+23.4 %)** | **+0.1020** | 36.3 |


> ⛔ **SUPERSEDED 2026-08-03 — the reference video is offset from the rig by a per-scene
> constant** (`+6` on `00040136`, `+5` on `7c72937c`; rule: `video_idx = rig_idx +
> (n_mp4_decodable − n_rig_frames)`, measured by the renderer, unanimous over 12 frames each).
> Re-baselined against the **aligned** reference the improvement is **roughly half the size and
> does not replicate**: `00040136` n=5 **+13.5 %** (was +23.4 %), n=12 **+8.0 %**, and
> `7c72937c` n=12 **+4.4 % — NOT SEPARATED** [−0.0097, +0.0521]. Absolutes move too:
> BEFORE 0.2774 → **0.4228**, AFTER 0.3424 → **0.4800**. The render is still better; the
> magnitude quoted here is not. Corrected table + estimator:
> `TanitAD Research Hub/Evaluation/Implementation/incoming/2026-08-03-render-rebaseline/`;
> `RETRACTION_LOG.md` R-2026-08-03-align. ⚠️ No closed-loop conclusion moves — `cl_metrics.py`
> never opens the reference video.


`run_dir = thor:~/rq_out/panel6_chosen`; evidence in `stack/experiments/alpasim-gsplat/RENDER_QUALITY.md`. **Identical to the render the closed-loop videos use**, deliberately, so open and closed loop are on the same pixels.

MEASURED on this run: scene 00040136 **199 frames, 114.8 ms/frame mean (123.5 p50), 88.2 s wall for BOTH arms**; junction scene 199 frames at 186.0 ms.

⛔ **Rolling shutter is OFF.** It is measured-better (grad-NCC 0.3747, +35.1 %) and costs **161×** (3749 ms/frame). Nothing here uses it; if a later run does, it must say so and quote the cost.

## ⭐ Four invariants — CHECKED against the banked dumps, not asserted

One render pass drives every arm inside one process, and the per-tick **md5 of the rendered frame** is recorded in each rollout file. Every claim below would leave the panel looking fine if it were false, so `ol_verify_invariants.py` re-derives all four from the banked per-window dumps:

| scene | condition | ego pinned to log | arms share pixels | md5 consistent | windows aligned | n |
|---|---|---|---|---|---|---|
| junction7c72937c | objects | ✅ | ✅ | ✅ | ✅ | 190 |
| junction7c72937c | empty | ✅ | ✅ | ✅ | ✅ | 190 |
| scene00040136 | objects | ✅ | ✅ | ✅ | ✅ | 190 |
| scene00040136 | empty | ✅ | ✅ | ✅ | ✅ | 190 |

* **junction7c72937c** — the objects/empty A/B changes the pixels on **199/199** ticks. ✅ A silently no-op ablation would read as *the model ignores the agents*, the most tempting wrong conclusion here.
* **scene00040136** — the objects/empty A/B changes the pixels on **199/199** ticks. ✅ A silently no-op ablation would read as *the model ignores the agents*, the most tempting wrong conclusion here.

`ALL_INVARIANTS_HOLD = True` (`OL_INVARIANTS.json`). `ego_pinned_to_log` is checked at 1e-9 on x, y AND yaw — if it were false the run would not be open loop at all and this whole page would be mislabelled, and nothing else in the pipeline would notice.

This matters because the renderer is a **step function of pose**: a 0.1 px camera rotation has been measured to move the 2 s waypoint 6.65 m, and a gRPC float32 round-trip alone costs 4.59 m. "We rendered it the same way twice" is not good enough for a paired estimator — it has to be the same bytes.

## ⛔ Five metrics are SETUP, not RESULT — measured, then struck out

The ego IS the logged path, so these are pinned to zero by the experiment's own definition. They were **measured** rather than assumed, and every one confirmed at float tolerance (`all_confirmed=True`):

| metric | measured max\|value\| | why |
|---|---|---|
| `dist_to_gt_traj_m` | 0 | literally abs(cross_track) in cl_metrics — the same measurement under a second name |
| `executed_speed_err_ms` | 0 | the ego speed IS the logged speed; no controller runs |
| `cross_track_abs_m` | 0 | the ego pose IS the logged pose, so its distance to the logged polyline is zero by construction |
| `cross_track_signed_m` | 0 | same quantity, signed |
| `route_corridor_departure_rate` | 0 | departure is \|cross_track\| > 2 m, and cross_track is pinned at 0 |

⚠️ `manoeuvre_exec_eq_plan` additionally **collapses onto `manoeuvre_plan_eq_logged`** — the 'executed' manoeuvre is classified from the ego poses, which are the logged poses. *Does the arm execute what it selects* is only askable in CLOSED loop.

⚠️ **Headway/time-gap/TTC from the ego to the annotated lead are also properties of the LOG in open loop** and come out bit-identical across arms (all four `real_lead_*` paired deltas exactly +0.0000). That family is therefore measured by a **new instrument** — see below — not dropped.

## The four families + ADE — scene 00040136 (night)

#### with objects (the scene's own dynamic layers rendered)

`flagship-v1` (A) vs `refc-base` (B) · **170 paired windows · 9 clusters** · paired episode-cluster bootstrap.

| family | metric | A | B | paired Δ (A−B) | |
|---|---|---|---|---|---|
| ADE | ADE 0–2 s (m) | 4.7304 [3.2677, 6.2121] | 0.6238 [0.4144, 0.8403] | **+4.1066** [+2.6228, +5.6146] | **separated** |
| ADE | displacement @2 s (m) | 7.3963 [4.9236, 9.9378] | 1.3333 [0.8929, 1.7956] | - |  |
| LONGITUDINAL | target-speed err (abs, m/s) | 4.0316 [2.9663, 5.0732] | 0.0601 [0.0518, 0.0695] | **+3.9716** [+2.9051, +5.0106] | **separated** |
| LONGITUDINAL | target-speed err signed (m/s) | -4.0316 [-5.0732, -2.9663] | 0.0254 [0.0016, 0.0477] | - |  |
| LONGITUDINAL | along-track ADE (m) | 4.7119 [3.2407, 6.1994] | 0.5870 [0.3691, 0.8039] | **+4.1250** [+2.6492, +5.6323] | **separated** |
| LATERAL | heading err (rad) | 0.0434 [0.0331, 0.0545] | 0.0261 [0.0165, 0.0396] | **+0.0173** [+0.0067, +0.0289] | **separated** |
| LATERAL | curvature err (1/m) | 0.0018 [0.0014, 0.0024] | 0.0010 [0.0006, 0.0017] | **+0.0008** [+0.0005, +0.0012] | **separated** |
| LATERAL | yaw-rate err (rad/s) | 0.0293 [0.0208, 0.0386] | 0.0091 [0.0061, 0.0127] | **+0.0201** [+0.0102, +0.0304] | **separated** |
| LATERAL | lateral ADE (m) | 0.2394 [0.1781, 0.3159] | 0.1458 [0.1040, 0.1984] | **+0.0936** [+0.0374, +0.1748] | **separated** |
| TACTICAL | plan == logged manoeuvre | 0.4353 [0.1610, 0.7133] | 0.4000 [0.1597, 0.6402] | **+0.0353** [-0.4354, +0.5118] | not separated |
| TACTICAL | manoeuvre HEAD == logged | 0.0353 [0.0053, 0.0738] | 0.2294 [0.0579, 0.4444] | - |  |
| TACTICAL | manoeuvre HEAD == own plan | 0.1824 [0.0157, 0.4261] | 0.6059 [0.3432, 0.8429] | - |  |
| STRATEGIC | route HEAD == logged route | 1.0000 [1.0000, 1.0000] ⚠️echo-unidentifiable ⛔degenerate | 0.2712 [0.0156, 0.6818] ⚠️echo-unidentifiable ⛔degenerate | **+0.7288** [+0.3182, +0.9844] | **separated** |
| STRATEGIC | route side vs graded proxy | 0.1412 [0.0000, 0.3905] ⚠️echo-unidentifiable | 0.3706 [0.1447, 0.6191] ⚠️echo-unidentifiable | - |  |

#### empty road (matched control, identical logged poses)

`flagship-v1` (A) vs `refc-base` (B) · **170 paired windows · 9 clusters** · paired episode-cluster bootstrap.

| family | metric | A | B | paired Δ (A−B) | |
|---|---|---|---|---|---|
| ADE | ADE 0–2 s (m) | 7.1479 [6.0645, 8.3711] | 0.6444 [0.4392, 0.8497] | **+6.5035** [+5.4961, +7.7434] | **separated** |
| ADE | displacement @2 s (m) | 11.2721 [9.4975, 13.4087] | 1.3930 [0.9684, 1.8284] | - |  |
| LONGITUDINAL | target-speed err (abs, m/s) | 5.9517 [5.1379, 6.7660] | 0.0526 [0.0430, 0.0613] | **+5.8991** [+5.0862, +6.7143] | **separated** |
| LONGITUDINAL | target-speed err signed (m/s) | -5.9517 [-6.7660, -5.1379] | 0.0118 [-0.0086, 0.0291] | - |  |
| LONGITUDINAL | along-track ADE (m) | 7.1374 [6.0540, 8.3617] | 0.6022 [0.3899, 0.8092] | **+6.5353** [+5.5348, +7.7702] | **separated** |
| LATERAL | heading err (rad) | 0.0469 [0.0366, 0.0554] | 0.0277 [0.0179, 0.0413] | **+0.0192** [+0.0033, +0.0328] | **separated** |
| LATERAL | curvature err (1/m) | 0.0028 [0.0018, 0.0037] | 0.0011 [0.0006, 0.0017] | **+0.0017** [+0.0008, +0.0026] | **separated** |
| LATERAL | yaw-rate err (rad/s) | 0.0425 [0.0312, 0.0547] | 0.0085 [0.0057, 0.0122] | **+0.0340** [+0.0225, +0.0466] | **separated** |
| LATERAL | lateral ADE (m) | 0.3077 [0.2478, 0.3732] | 0.1644 [0.1138, 0.2246] | **+0.1433** [+0.0688, +0.2196] | **separated** |
| TACTICAL | plan == logged manoeuvre | 0.4412 [0.1302, 0.7707] | 0.4059 [0.1657, 0.6413] | **+0.0353** [-0.4765, +0.5296] | not separated |
| TACTICAL | manoeuvre HEAD == logged | 0.0765 [0.0134, 0.1728] | 0.1706 [0.0053, 0.3822] | - |  |
| TACTICAL | manoeuvre HEAD == own plan | 0.1529 [0.0419, 0.2789] | 0.5118 [0.2189, 0.7854] | - |  |
| STRATEGIC | route HEAD == logged route | 1.0000 [1.0000, 1.0000] ⚠️echo-unidentifiable ⛔degenerate | 0.2373 [0.0000, 0.6364] ⚠️echo-unidentifiable ⛔degenerate | **+0.7627** [+0.3636, +1.0000] | **separated** |
| STRATEGIC | route side vs graded proxy | 0.1412 [0.0000, 0.3905] ⚠️echo-unidentifiable | 0.4176 [0.1562, 0.6883] ⚠️echo-unidentifiable | - |  |

## ⭐ LONGITUDINAL distance-keeping — the family, measured

Since the ego-to-lead gap is a log property here, what IS policy-dependent is **the gap the plan would produce**: project the ego to +2 s along the emitted plan, put the lead where the annotation says it will be at +2 s, and measure. Instrument: `ol_distance_keeping.py`.

#### scene 00040136, with objects

Lead present on **82/170** windows (rate 0.4824); 82 paired.

| metric | A | B | paired Δ (A−B) | |
|---|---|---|---|---|
| headway the PLAN implies @2 s (m) | 21.7758 [4.8819, 38.7559] *(n=82, 7cl)* | 13.3278 [-4.5816, 31.5377] *(n=82, 7cl)* | **+8.4480** [+5.6715, +10.4899] *(n=82, 7cl)* | **separated** |
| time gap the PLAN implies @2 s (s) | 1.9119 [0.5444, 3.8578] *(n=82, 7cl)* | 0.9479 [-0.1227, 2.3624] *(n=82, 7cl)* | **+0.9640** [+0.5941, +1.4956] *(n=82, 7cl)* | **separated** |
| TTC the PLAN implies (s, while closing) | 3.6905 [2.5902, 5.2011] *(n=64, 7cl)* | 2.0153 [1.2359, 3.3554] *(n=59, 6cl)* | **+1.9331** [+1.5301, +2.5331] *(n=59, 6cl)* | **separated** |
| frac of windows: plan time gap < 1 s | 0.2683 [0.0000, 0.6557] *(n=82, 7cl)* | 0.4634 [0.0882, 0.7654] *(n=82, 7cl)* | **-0.1951** [-0.4468, -0.0110] *(n=82, 7cl)* | **separated** |
| frac of windows: plan time gap < 0.5 s | 0.2439 [0.0000, 0.5902] *(n=82, 7cl)* | 0.3537 [0.0287, 0.6990] *(n=82, 7cl)* | **-0.1098** [-0.2143, +0.0000] *(n=82, 7cl)* | not separated |
| frac: plan would drive INTO the lead (in-lane) | 0.2195 [0.0000, 0.5246] *(n=82, 7cl)* | 0.2805 [0.0000, 0.6885] *(n=82, 7cl)* | **-0.0610** [-0.1613, +0.0000] *(n=82, 7cl)* | not separated |
* **A in-lane precision of the ungated test** — 18/18 fires were genuinely in-lane (precision 1.0), median |y| of a fire 0.475 m.
* **B in-lane precision of the ungated test** — 23/23 fires were genuinely in-lane (precision 1.0), median |y| of a fire 0.42 m.

⚠️ `headway_err_2s` is **algebraically −(along-track error @2 s)** — the lead term cancels — so it is NOT an independent separation and is excluded above.
⚠️ `n` is the **jointly-finite** window count for the paired column and the arm's own finite count for the marginals — TTC is defined only while CLOSING, so those three denominators differ and the marginals cannot be subtracted to reproduce the delta. Rows on fewer than 3 clusters, or with a zero-width interval, are marked NOT QUOTABLE.

## The four families + ADE — JUNCTION scene 7c72937c (day)

⭐ This scene was added **because the strategic family cannot be evaluated on 00040136**: the nav command there is constant (`follow` on 170/170 windows), so the circularity guard is structurally unable to run. On 7c72937c the nav command **varies** and the guard becomes identifiable.

#### junction 7c72937c, with objects

`flagship-v1` (A) vs `refc-base` (B) · **170 paired windows · 9 clusters** · paired episode-cluster bootstrap.

| family | metric | A | B | paired Δ (A−B) | |
|---|---|---|---|---|---|
| ADE | ADE 0–2 s (m) | 2.2431 [1.6524, 3.0405] | 0.7600 [0.5366, 0.9828] | **+1.4831** [+0.8427, +2.2610] | **separated** |
| ADE | displacement @2 s (m) | 3.7362 [2.7240, 5.1399] | 1.6358 [1.1889, 2.0802] | - |  |
| LONGITUDINAL | target-speed err (abs, m/s) | 1.5897 [1.1881, 2.0622] | 0.1885 [0.1189, 0.2820] | **+1.4013** [+0.9754, +1.9154] | **separated** |
| LONGITUDINAL | target-speed err signed (m/s) | -0.3737 [-1.2913, 0.3988] | 0.0639 [-0.0665, 0.2088] | - |  |
| LONGITUDINAL | along-track ADE (m) | 2.1574 [1.5528, 2.9802] | 0.6785 [0.4316, 0.9369] | **+1.4790** [+0.8340, +2.2657] | **separated** |
| LATERAL | heading err (rad) | 0.1484 [0.0720, 0.2446] | 0.1436 [0.0525, 0.2433] | **+0.0048** [-0.0536, +0.0647] | not separated |
| LATERAL | curvature err (1/m) | 0.0192 [0.0092, 0.0289] | 0.0402 [0.0087, 0.0887] | **-0.0210** [-0.0676, +0.0088] | not separated |
| LATERAL | yaw-rate err (rad/s) | 0.1132 [0.0313, 0.2248] | 0.1006 [0.0196, 0.2158] | **+0.0126** [+0.0043, +0.0212] | **separated** |
| LATERAL | lateral ADE (m) | 0.3353 [0.1703, 0.5185] | 0.1721 [0.0592, 0.3023] | **+0.1631** [+0.1082, +0.2249] | **separated** |
| TACTICAL | plan == logged manoeuvre | 0.4882 [0.2924, 0.7054] | 0.4412 [0.2282, 0.6721] | **+0.0471** [-0.0588, +0.1385] | not separated |
| TACTICAL | manoeuvre HEAD == logged | 0.5765 [0.3764, 0.7831] | 0.3941 [0.1600, 0.6335] | - |  |
| TACTICAL | manoeuvre HEAD == own plan | 0.5176 [0.3117, 0.7427] | 0.5882 [0.3757, 0.7948] | - |  |
| STRATEGIC | route HEAD == logged route | 1.0000 [1.0000, 1.0000] ⛔CIRCULAR | 0.7613 [0.5437, 0.9736] | **+0.2387** [+0.0264, +0.4563] | **separated** |
| STRATEGIC | route side vs graded proxy | 0.9235 [0.8254, 1.0000] ⛔CIRCULAR | 0.7824 [0.5755, 0.9764] | - |  |

#### junction 7c72937c, with objects — distance-keeping

Lead present on **129/170** windows (rate 0.7588); 129 paired.

| metric | A | B | paired Δ (A−B) | |
|---|---|---|---|---|
| headway the PLAN implies @2 s (m) | 10.9800 [-0.0241, 24.6495] *(n=129, 8cl)* | 10.5399 [0.8881, 21.9155] *(n=129, 8cl)* | **+0.4401** [-1.4963, +3.2076] *(n=129, 8cl)* | not separated |
| time gap the PLAN implies @2 s (s) | 3.9101 [0.8724, 6.8697] *(n=129, 8cl)* | 10.9357 [0.6796, 24.9601] *(n=129, 8cl)* | **-7.0256** [-18.6494, +0.6259] *(n=129, 8cl)* | not separated |
| TTC the PLAN implies (s, while closing) | 4.3813 [1.1971, 4.9120] *(n=7, 2cl)* ⛔too-few-clusters | 7.7219 [7.7219, 7.7219] *(n=8, 1cl)* ⛔too-few-clusters | **-8.0286** [-8.0286, -8.0286] *(n=5, 1cl)* ⛔too-few-clusters | ⛔ **NOT QUOTABLE** (too few clusters) |
| frac of windows: plan time gap < 1 s | 0.4264 [0.1189, 0.7534] *(n=129, 8cl)* | 0.3256 [0.0467, 0.6563] *(n=129, 8cl)* | **+0.1008** [+0.0141, +0.1870] *(n=129, 8cl)* | **separated** |
| frac of windows: plan time gap < 0.5 s | 0.3566 [0.0769, 0.6695] *(n=129, 8cl)* | 0.3178 [0.0467, 0.6429] *(n=129, 8cl)* | **+0.0388** [+0.0000, +0.0960] *(n=129, 8cl)* | not separated |
| frac: plan would drive INTO the lead (in-lane) | 0.1473 [0.0244, 0.2963] *(n=129, 8cl)* | 0.1085 [0.0067, 0.2234] *(n=129, 8cl)* | **+0.0388** [+0.0000, +0.0867] *(n=129, 8cl)* | not separated |
* **A in-lane precision of the ungated test** — 19/40 fires were genuinely in-lane (precision 0.475), median |y| of a fire 2.044 m.
* **B in-lane precision of the ungated test** — 14/35 fires were genuinely in-lane (precision 0.4), median |y| of a fire 2.247 m.

⚠️ `headway_err_2s` is **algebraically −(along-track error @2 s)** — the lead term cancels — so it is NOT an independent separation and is excluded above.
⚠️ `n` is the **jointly-finite** window count for the paired column and the arm's own finite count for the marginals — TTC is defined only while CLOSING, so those three denominators differ and the marginals cannot be subtracted to reproduce the delta. Rows on fewer than 3 clusters, or with a zero-width interval, are marked NOT QUOTABLE.

## ⛔⭐ THE STRATEGIC RESULT — flagship's route head is a CONFIRMED nav echo

The harness FEEDS a nav command to the policy. If the route head is a deterministic function of that input, `route_head_eq_logged` measures the echo of the model's own conditioning — and it scores perfectly, because the nav command is derived from the same log the route label is. The guard is **computed**, and it needs ≥2 distinct nav values to separate an echo from a constant head.

* **`flagship-v1` on junction 7c72937c** — `identifiable=True`, nav takes 2 value(s), head takes 2; map `{'1': [0], '0': [1]}`. **CIRCULAR — route_head_eq_logged above reproduces the nav command the policy was GIVEN and is NOT evidence of strategic skill; do not quote it**
* **`refc-base` on junction 7c72937c** — `identifiable=True`, nav takes 2 value(s), head takes 3; map `{'1': [0, 1, 2], '0': [0, 1]}`. **not an echo — the head is not a function of nav on these windows**

* **`flagship-v1` on scene 00040136** — `identifiable=False`, nav takes 1 value(s), head takes 1; map `{'0': [1]}`. **UNIDENTIFIABLE — nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 1 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'.**
* **`refc-base` on scene 00040136** — `identifiable=False`, nav takes 1 value(s), head takes 3; map `{'0': [0, 1, 2]}`. **UNIDENTIFIABLE — nav takes only 1 distinct value(s) on these 170 windows, so an echo cannot be distinguished from a constant head (the head takes 3 distinct value(s) here). Needs a scene where the nav command actually varies. NOT a clearance: do not read this as 'not an echo'.**

⇒ **flagship v1's `route_head_eq_logged` = 1.0000 [1.0000, 1.0000] ⛔CIRCULAR is NOT strategic skill** — the head is an exact bijection of the nav command on 170/170 windows. Its graded proxy 0.9235 [0.8254, 1.0000] ⛔CIRCULAR carries the same stamp. **Neither may be quoted.**

⇒ **REF-C base's 0.7613 [0.5437, 0.9736] is NOT an echo** (the head is not a function of nav) and is measured on a scene whose logged route is genuinely non-degenerate: `{'left': 0.7118, 'straight': 0.2, 'right': 0.0, 'unknown': 0.0882}`, valid on 0.9118 of windows. **This is the only admissible strategic-accuracy number the programme currently has.**

⚠️ This independently reproduces, in OPEN loop, what the closed-loop panel found — so the echo is a property of the arm, not of the closed-loop harness.

## The objects-vs-empty contrast, with control drift REMOVED

The closed-loop panel found that rendering the agents makes flagship better and flagged it as **not yet interpretable**: flagship is the arm whose plan moves 9 m under *any* appearance change, so 'extra gaussians change the frame' and 'the model reasons about the agents' predict the same sign. **In open loop the ego cannot drift**, so that confound is gone. Same logged poses, only the pixels differ.

| arm | metric | Δ (objects − empty) | |
|---|---|---|---|
| flagship v1 | `ade_0_2s` | **-2.4175** [-3.4848, -1.3549] | **separated** |
| flagship v1 | `abs_target_speed_err_ms` | **-1.9201** [-2.7510, -1.0874] | **separated** |
| flagship v1 | `along_track_ade_m` | **-2.4255** [-3.5012, -1.3617] | **separated** |
| flagship v1 | `lateral_ade_m` | **-0.0683** [-0.1338, -0.0029] | **separated** |
| flagship v1 | `manoeuvre_plan_eq_logged` | **-0.0059** [-0.0710, +0.0785] | not separated |
| REF-C base | `ade_0_2s` | **-0.0206** [-0.0371, -0.0061] | **separated** |
| REF-C base | `abs_target_speed_err_ms` | **+0.0075** [-0.0020, +0.0207] | not separated |
| REF-C base | `along_track_ade_m` | **-0.0152** [-0.0256, -0.0034] | **separated** |
| REF-C base | `lateral_ade_m` | **-0.0186** [-0.0470, +0.0026] | not separated |
| REF-C base | `manoeuvre_plan_eq_logged` | **-0.0059** [-0.0237, +0.0134] | not separated |

Negative Δ = the arm is BETTER with the agents rendered.

## ⚠️ Caveats that travel with every number here

* ⛔ **WITHIN-SIM RELATIVE ONLY.** REF-C's open-loop ADE is **1.5157 on these reconstructions vs 0.4728 on real footage — 3.21× OOD**. Orderings survive; absolute rates do not. Never quote a sim rate as a real-world rate.
* **Scope:** this exercises AlpaSim's renderer **wire contract** with a gsplat backend, driven by a TanitAD harness. It is **not** `alpasim_runtime.simulate`, so there is **no AlpaSim collision / offroad / scene score** here.
* ✅ **grad-NCC is the ONLY admissible render metric on these clips.** PSNR, NCC **and MAE** are RETRACTED — over 5 frames × 6 wrong references grad-NCC identifies the correct frame **5/5 on every arm** while MAE and PSNR manage **1–4/5 with arm-dependent reliability**.
* ⚠️ **flagship v1's route head is a CIRCULAR nav echo** (proved above on the junction scene, and previously 369/369 + 81/81 in closed loop). Its `route_head_eq_logged = 1.0000` is not strategic skill and must never appear on a video or in a README without this guard.
* ⚠️ **Precision travels with every rate**, and the denominator is stated: the per-class PR blocks in `OL_PANEL_*.md` carry precision, recall, support and n_fires, plus the majority-class baseline any constant predictor achieves.
* **Clusters are DISJOINT SEGMENTS OF ONE CLIP**, not 40 independent val episodes — 9 of them. The interval is the right estimator for the resampling unit available; the unit is named so it is never mistaken for the 40-episode val bootstrap.
* ⚠️ The renderer is a **step function of pose**; all production numbers here come from one numerical path (in-process, one render pass, both arms).

## Provenance

Code — `stack/experiments/alpasim-gsplat/`: `openloop_drive.py` (the sweep), `cl_metrics.py` (the four families, **unchanged** — the record schema is the same as a closed-loop rollout), `ol_distance_keeping.py` (the open-loop distance-keeping instrument), `ol_report.py` (degeneracy audit + panel), `overlay_video.py --mode open_loop` (the video), `run_openloop_videos.sh` (the runner). This page is generated by `ol_readme.py` **from the JSONs**, so no number on it was hand-copied.

Results — `stack/experiments/alpasim-gsplat/results/openloop-thor-2026-08-03/`: `OL_flagship_vs_refc_{objects,empty}.json` (scene 00040136), `junction/OL_flagship_vs_refc_{objects,empty}.json` (7c72937c), `OL_DISTKEEP_*.json`, `OL_{arm}_objects_vs_empty.json`, `OL_AUDIT_*.json` (the degeneracy audit), `OL_INVARIANTS.json` (the four invariants above), `OL_PANEL_*.md` (the full per-family tables including every confusion matrix and per-class PR block), `openloop_summary_*.json` (render config, timings, per-tick frame md5 pointer).

⭐ **The RAW per-window surface is banked, not stranded** — `results/openloop-thor-2026-08-03/rollouts/{scene00040136,junction7c72937c}/<condition>_<arm>.json` (8 files, 1.8 MB) hold every step's ego pose, emitted plan, head logits, nav command and frame md5. Every number on this page can be re-derived from them with **zero GPU**, and a re-render can be checked bit-exactly against the digests.

On the device — `thor:~/ol_out` (scene 00040136), `thor:~/ol_out_junction`, `thor:~/ol_videos`, `thor:~/ol_videos_junction`. Rollout JSONs carry the per-tick `frame_md5` so a re-render can be checked bit-exactly.

⚠️ `*.mp4` is gitignored — these are committed with `git add -f`. Any new video needs the same or it silently never lands.

*Closed-loop counterparts: `../alpasim-closedloop-thor-2026-08-03/`.*
