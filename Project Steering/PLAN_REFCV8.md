# PLAN_REFCV8 — understand every effect, measure it, validate the fix cheaply, then retrain once

*Master Mind, 2026-10-04. Status: **DRAFT for the PI** — Phase A is running; Phases B–E need the decisions in §8.*
*PI, 2026-10-04: "I need a big plan for refcv8, I need to understand all effects define measures and validate them to
retrain without loosing a lot of time. ... is our training data ok? are any thing missing or uplausible, are we using
them wrongly? Why the video is saying, no gt label in this frame?"*

Every number below carries its evidence class. Planner numbers are open-loop on held-out eval139 from the launch tree
`fec3a0d` (one training seed), where the speed ceiling does not reach the emitted plan (SPEC_REFCV7 §26.1).

---

## 1. The answer to "why does the replay say *GT goals: not labelled at this instant*?"

MEASURED (label file `s2_labels_v8_train.jsonl.gz`, md5 `b45377a1…`, 4,572 records; trainer `refc_v3_train.py:3884-3916`,
`v7_labels.py:678-710`):

* The tactical labels (8 lateral + 8 longitudinal actions, 22 goal tokens) come from **ONE record per clip**. Every one of
  the 4,572 records is anchored at the **same instant, raw t0 = 8.0 s**, with a tactical band of 2–6 s ahead.
* The trainer accepts a window only when its NOW is within **±2.0 s of t0** (`window_in_band`). The clip cache covers
  raw ≈ 0.1–20.1 s (`grid_start_s` median 0.113 s, dt 0.1007 s), so windows whose NOW is outside **6.0–10.0 s** get
  IGNORE: no tactical loss in training, and "not labelled at this instant" in the replay.
* ESTIMATED ≈ 23 % of windows labelled (≈ 40 of ≈ 171 per clip); stream D1 measures it exactly. Route package, MEASURED:
  the lateral label is IGNORE on **72 of 107** eval turn windows.

The replay is reporting the truth: the tactical head of refcv7 was trained on about a quarter of the windows, and on
the minority of turns that fall inside that 4-s band.

## 2. What we already know about the data (before the audit finishes)

| # | finding | evidence | consequence |
|---|---|---|---|
| K1 | Tactical + goal GT only within ±2 s of one anchor per clip (§1) | MEASURED | tactical head barely sees turns; selection cannot lean on it (lateral head right side 0.39 on turns) |
| K2 | The **nav input is one token per clip, fed on every window**. For L/R clips the turn starts a median **7.3 s** after the anchor; 52.7 % > 6 s, 42.0 % > 10 s (p90 26.2 s) | MEASURED, label file | nav says "turn" while the car drives straight: 193/291 L/R-nav eval windows are straight; the nav-compliance gate learned 0.163 (near inert) |
| K3 | The time-localised route **is already in the file** (`nav_30s.entries`: every turn's start/end time and arc-length distance) — refcv7 did not use it | MEASURED | the cheapest nav fix is a loader change, not new data |
| K4 | The **max-speed input is one value per clip** = the ego's realised max speed over [t0+2, t0+6] s, fed on every window. The label file itself records that the bin recovers **75.4 %** of the future-speed information v0 lacks, and that **75 % of intersection clips** get ≤ 30 km/h because the ego was stopped | MEASURED (quoted from the file's own audit fields) | an oracle hint that is stale away from the anchor and wrong at stops; speed profile is the largest measured planner lever (−0.606 m ADE if right) |
| K5 | The VLM (Alpamayo) disagrees with our geometric tactical label on **36 %** of lateral (1,537/4,275) and **36 %** of longitudinal (1,655/4,567) records | MEASURED | label noise of unknown direction — D2 decides which side is right |
| K6 | `a_tac.lat_args.lat_peak_m`: median 19.7 m, max 305 m | MEASURED; **SETTLED by D2**: mis-NAMED, not broken — the peak lateral displacement in the first-sample frame over the clip's first 20 s (reproduced at corr 0.9995, n = 4,369); refcv7 never read it | docstring fix only |
| K10 | **NUDGE labels curves, not nudges**: on train only 10.3 % (NUDGE_L) / 12.1 % (NUDGE_R) agree with an independent ego geometry; the heading never returns on 89.7 % of 1,028 NUDGE records; NUDGE is 23.5 % of the lateral mass. TURN agrees 85–87 %, LANE_KEEP 92 %; lateral overall 72.1 % (3,149/4,369) | MEASURED (D2, analytic controls 17/17) | the tactical head is trained to call a road curve a "nudge" |
| K11 | Longitudinal labels are computed over [0, 6] s, not the declared [2, 6] band (HOLD/CREEP excepted); the effective ACCELERATE bar is +1.0 m/s, not the documented +1.5. As the trainer applies them, agreement decays 80 % → 64 % across the ±2 s band | MEASURED (D2) | a band-vs-computation mismatch; fix the window and the bar |
| K12 | In-band, the fed speed ceiling equals the window's own realised future max at **R² 0.988** (mean abs error 0.44 m/s): a near-perfect future-speed oracle. Outside the band it is stale (over all windows v_now beats it, R² 0.899 vs 0.889); 11.9 % of windows exceed their bin; slow ego ⇒ low bin (intersections ≤ 30 km/h 74 %, 69.8 % even when moving) | MEASURED (D2, 100 Hz egomotion) | the largest measured planner lever (speed profile) rides on an oracle input; L3 must be redesigned, see §8-2 |
| K13 | Nav files: `build_v8_nav30s.py:83` reads `suppressed` while the emitter writes `applied`, so 66 suppressed turns stay in `nav_30s`; 68 `nav_command` turns (> 30 s ahead) have no `nav_30s` entry. Only 24.7 % of TURN-token windows see a same-side turn begin in their own next 6 s | MEASURED (D2) | L2 must use announced entries only and fix the cap |
| K7 | 5 of 22 goal tokens are untrainable (no supervised negatives); several have ≤ 23 positive clips in 4,572 (LANE_CHANGE_L 23, LANE_CHANGE_R 15, OVERTAKE 20, TAKE_EXIT_L 20, YIELD_FOR_TURN_L/R 21/20) | MEASURED (config.json `tac_goal_stats`; label file) | those tokens cannot be learned from this corpus as labelled |
| K8 | Lateral classes: LANE_KEEP 65 %, NUDGE 24 %, TURN 12 % of records | MEASURED | turns are rare AND mostly outside the band |
| K9 | No map topology, lane graph, posted speed limit or traffic-light state exists in the published PhysicalAI-AV corpus; our only route and speed-limit suppliers are the ego's own future | PUBLISHED (dataset card) + programme record | nav and max-speed inputs are optimistic by construction; say so on every result |

| K14 | **Raw data is sound (D3)**: 897 k pose rows with 0 non-finite and no physically implausible yaw-rate / lateral-acceleration rows; 0 black / frozen / over-exposed frames in 88 k decoded; 27.6 M boxes all carry z and h; the ego's next-6-s path is road-like on 99.92 % of the SAM3 map GT; train/eval balance within ~1 pp on turns, stops, speed, night. **The GT boxes do NOT carry duplicates** (0.34 % of boxes) — the reel's ~2.1 boxes per object is prediction-side | MEASURED (D3, controls read known values) | no raw-data blocker for refcv8 |
| K15 | **Split leakage across recordings**: no shared clip id, but 3 (image-confirmed) to 5 eval clips share a ~140 s source recording with a train clip; 135 train recording groups cover 296 clips (6.8 %); 2 train pairs are the same video under two ids. **The ego appears as an agent box in 18 train clips (693 frames)** | MEASURED (D3) | eval mildly optimistic on ~2–4 % of clips; fix both in the refcv8 corpus manifest |
| K16 | **Eval caveats**: left-turn route following rests on **13** eval clips (eval nav mix skewed, p = 0.0062); day/night is a clock label (only ~8 % of clips are actually dark vs 46 % "night"); box z is reliable near-field only; 13.4 % of train clips have no lane line (real, by country) | MEASURED (D3) | report left turns with that n; cut night by brightness; no z claims beyond ~30 m |

| K17 | **Usage audit (D4)** — three channels WRONG, each with a measured cost: nav (one token per clip; the nav-compliance graft's own training signal is right on only **35.7 %** of informative windows, which is why its gate stuck at 0.163), tactical labels (23.4 % of windows supervised; left/right-nav records carry LANE_KEEP on 49–52 %; all v7-label tactical terms together are **0.29 % of the loss**, no class weights), max speed (future oracle; the "unknown" row NavSim feeds on 45 % of navtest tokens was never trained). SUBOPTIMAL: the residual prior (14 of 17 wrong-direction turn picks follow its side), goal-token negatives (the PI's 2026-09-16 caption-absence ruling never reached the run — the sidecar is bound to another label blob), ego dropout 0.5 withholds the speed input on half of training, the box store's x ≤ 61 m scope excludes 84 % of joined agents (distance keeping) | MEASURED, label-level (D4; controls: eval grid rebuilt bit-exactly, A5 reproduced, label file's leak 75.3 % vs 75.4 %, analytic tracks, two mutations red) | the design in `REFCV8_LABEL_DESIGN.md`: L1 labelled windows **23.4 % → 88.9 %**, turn windows 36 → 107 / 107; L2 wrongly commanded straight windows **193 → 14 / 588**, graft signal 35.7 % → 80.0 % right |

**Verdict so far:** the images, poses, boxes and maps are sound (K14) apart from two small defects (K15). The **labels are used wrongly in time**: per-clip
records are applied **too narrowly** for tactical supervision (±2 s) and **too broadly** for the nav and speed inputs
(the whole 20 s clip). This is consistent with, and explains, the route-following diagnosis: the 117-candidate fan holds
a correct turn on 100 % of turn windows, but the pick turns the right way on only 84 %, and its heading is within 15° on
only 51 % (best available 76 %).

## 3. Phase A — data audit (RUNNING since 2026-10-04 ~12:10 Berlin; ETA ~15:00)

Package: `TanitAD Research Lab/Data Engineering/Research/2026-10-04-refcv8-data-audit/` (shared `CONTEXT.md`).

| stream | question | measures it produces |
|---|---|---|
| **D1** coverage census | which channel carries a value on which window? | the per-window label truth table (746,946 train + 23,772 eval windows, built through the trainer's own dataset code) with ego-future geometry; % windows with tactical GT, % turn windows labelled, % windows where nav says turn but no turn comes within 6 s / 10 s, % windows whose own speed exceeds the fed ceiling. Controls: reproduce 72/107 and 719,739/746,946 |
| **D2** label validity | do labels mean what we assume, and agree with the ego's own motion? | field-by-field semantics with file:line; confusion matrices of a_tac lat/lon vs an independent geometric derivation (analytic control: a circle must read "turn"); VLM-vs-geometry verdict; lat_peak_m verdict; speed-ceiling validity |
| **D3** raw plausibility | are poses, boxes, maps, frames and the split sane? | train/eval recording overlap; pose jumps/accel/yaw-rate outliers; GT box duplicates and implausible sizes; SAM3 GT drivable-on-own-path; frozen/black frames; rig mix; corpus balance |
| **D4** usage + design | are we using the data wrongly; what replaces it? | per-input row: per-clip vs per-window, oracle at inference, train/deploy mismatch (incl. what NavSim feeds), loss share, verdict; `REFCV8_LABEL_DESIGN.md` with each replacement's admissibility check, validation test and targeted number |
| **D5** effects inventory | every measured refcv7 effect in one table | effect · size + CI · tier · root-cause class · lever · measure · cheapest validation rung; gaps with no measure; missing metric families |

## 4. Phase B — the v9 label set (ZERO GPU; ~1 day)

Built from data we already hold. Every change ships with an independent cross-check, an analytic or known-value
control, and a mutation that must go red.

| change | definition (final form from D4's design) | validation (must pass before any GPU) |
|---|---|---|
| **L1 dense tactical labels** | lat/lon action for **every** window from the ego's own next 6 s (labels may use ego — PI 2026-08-03); NUDGE redefined (it labels curves today, K10) or replaced by a 3-way lateral head LANE_KEEP / TURN_L / TURN_R as the first arm; longitudinal on one declared window with the documented +1.5 m/s bar (K11); VLM-only semantics stay anchor-bound unless D4 gives a propagation rule | coverage ≥ 95 % of windows; on eval turn windows agreement with the turn direction ≥ 0.95; a synthetic circle reads TURN, a straight reads LANE_KEEP; time-shuffled labels must FAIL the agreement bar |
| **L2 time-localised nav** | per window: the next upcoming `nav_30s` entry → token + distance/time to the turn; FOLLOW once no turn is ahead | 0 windows with a turn token and no turn inside the stated horizon; the 193/291 straight-under-L/R count must fall to the expected residual (turns beyond 6 s, reported separately) |
| **L3 per-window speed input** | DECISION §8-2: (a) window-local realised max snapped UP to the posted ladder, or (b) a coarser posted-limit proxy | the file's own OOF test (R² of the future max from v0 vs from bin+v0) reported per option — the leak is measured, not assumed; a shuffled-ceiling arm must lose the speed gain |
| **L4 goal negatives** | supervised negatives for the 5 masked tokens where geometry can supply them; drop tokens with < 50 positives from the loss | per-token positive/negative counts; AP floor vs prevalence |
| **L5 fixes from D2/D3** | whatever the audits mark DEFECT (e.g. lat_peak_m, duplicate GT boxes, misregistered maps) | each with its own known-value control |

## 5. Measures — the pre-registered metric set for refcv8 (all four families + the route set)

Bars are committed in SPEC_REFCV8 before any refcv8 number exists; the baseline row is refcv7 at 50,400 on the same
windows. Estimator: paired episode-cluster bootstrap over eval139; a **replicate** (second sampler seed AND, for the
final run, the question "would another training run say this?" stated) for every lever claim.

| family | measure | refcv7 baseline (MEASURED) |
|---|---|---|
| route / LATERAL | turn-direction-correct pick on GT-turn windows; heading within 15° at 6 s; curvature and yaw-rate error; cross-track | 0.84; 0.51 (fan best 0.76); turn ADE 3.19 m (fan best 1.19) |
| selection | fan→pick drop (+ random and oracle on the same windows); pick regret | +0.159 [+0.080, +0.258]; random 0.39 |
| LONGITUDINAL | speed MAE 0–2 s / 2–6 s; speed-profile ADE attribution; distance keeping (headway, TTC to lead); ceiling compliance | speed lever −0.606 m; plan > fed ceiling on 110/2,059 reel windows |
| TACTICAL | lat/lon accuracy + macro-F1 on DENSE v9 labels and on the anchor band (continuity); goal-token AP vs prevalence | lateral head right side 0.39 on turns |
| STRATEGIC | N/A — strategic layer OFF by PI ruling R5 (2026-09-27); reported as absent with the reason, never silently dropped | — |
| nav compliance | on windows with a turn starting within 6 s: pick turns the commanded way | to be measured on v9 nav (D1 gives the denominator) |
| perception | map thin-class IoU (F1 thresholds), box AP@2 m, boxes/object, conf_ratio | lane 0.164, edge 0.042; AP 0.248 / 0.131 → 0.349 / 0.300 with NMS; 2.12 boxes/object |
| closed loop (NavSim) | navtest PDMS vs STOP; navhard two-stage EPDMS incl. DAC-zero and NC-zero rates; warmup S2-EPDMS-u | 5k: PDMS 65.60 vs STOP 61.82 PASS; 30k navhard 0.227 vs STOP 0.299 FAIL (DAC-zero 26.0 %, NC-zero 15.6 %); 50.4k RUNNING |

Controls on every panel: STOP / hold-action, refcv7-50.4k itself (paired), a deliberate-regression arm per lever (e.g.
time-shuffled dense labels, permuted nav), and the identity controls the replay already carries.

## 6. Phase C — the validation ladder (this is where the time is saved)

Each lever must clear its rung before it may enter the run. Rungs run in parallel where they do not share a GPU.

| rung | what | cost | gate to the next rung |
|---|---|---|---|
| **R0** zero-GPU | Phase B label checks; leak tests | hours, CPU | every L-change passes its control and mutation |
| **R0b** time-localised nav at inference — **A5 DONE** | route package addendum **A5** (sha256 `5f102584…`, registered 10:01:55Z before any number; A4's bound for a perfectly timed nav: −0.566 m on turns): **reported arm T2 FAILED** on criterion 4 only (seed 0: turn ΔADE −0.264 [−0.568, −0.0005], turn direction +0.093, straight +0.009 [+0.001, +0.020], all −0.043; seed 1 turn −0.204 [−0.447, +0.028] — not separated). Straight damage vs the clip token: **+0.326 → +0.009**. Control T2c failed as required. Secondary T3 (soft ×10) cleared all four criteria (turn −0.159 [−0.312, −0.040], seed 1 −0.143 [−0.286, −0.032]) — secondary, not promotable. Post hoc: 54 % of T2's turn gain came from turns `nav_command` had SUPPRESSED (curves, obstacle passes) — information a real nav would not give; nav is silent on 55 of 107 GT-turn windows (that half needs a vision turn cue, L1) | done | ⇒ **A6** (sha256 `b4099578…`, 10:32:25Z): T3 with ANNOUNCED turns only (the deployable nav) as the reported arm, on a denser capture of the same 139 held-out episodes (Thor GPU, ~30–40 min). RUNNING |
| **R1** head-only, frozen trunk | cache refcv7-50.4k's decoder inputs (trunk + BEV features) for a seeded train sample (~1,000 clips × 20 windows) and all eval139 windows on Thor; retrain ONLY the tactical decoder and the selector on v9 labels | ~1–2 h caching + minutes per arm | ⭐ **decides branch vs full retrain**: if dense labels + time-localised nav lift turn-correct pick ≥ 0.95 and heading-15 ≥ 0.70 with the trunk frozen, the information is in the trunk ⇒ branch fine-tune; if not ⇒ the trunk must learn it ⇒ full run |
| **R2** v7-tiny ladder | the real trainer at ~19 M params (~17 min/arm on Thor): each L-change ON + its deliberate regression; declared-vs-built and gradient-reach checks | ~3–4 h | each lever moves its metric and its regression arm does not |
| **R3** launch gate | `run_gate.py` PASS token (binding since 2026-09-26) | ~3 h | PASS |
| **R4** branch run `refcv8-b` | fine-tune from refcv7 50,400 with v9 labels + the selector fix + ride-alongs (R4 speed bundle, F1 map thresholds, F4 gates, box NMS at inference); 12k steps | ~1.4 days at 9.9 s/step | — |
| **R5** eval | the §5 metric set, G0, NavSim navtest/navhard/warmup, route set, replay + tactical video | ~1 day | SPEC_REFCV8 bars |
| (R4′) full run `refcv8` | only if R1 says the trunk lacks the information, or §8-4 approves trunk changes (F3 stride-4 map tap) | ~6 days (refcv7: 2026-09-28 00:03 → 10-03 20:38) | — |

Why branch-first: refcv7's in-run eval plateaued from ~35,000 (MEASURED, registry block REFCV7-2026-10-04-FINAL), the
measured defects sit in labels, inputs and selection, not in the trunk, and R1 tests that assumption directly in hours
instead of assuming it for six days.

## 7. Timeline (Berlin)

| when | what |
|---|---|
| Sun 2026-10-04 (today) | Phase A audit lands (ETA ~15:00); D1–D5 results → this plan's §2 updated; register rows |
| Mon 10-05 | Phase B v9 labels + R0; R1 feature caching on Thor in parallel; SPEC_REFCV8 pre-registration drafted |
| Tue 10-06 | R1 head-only arms (hours); R2 v7-tiny ladder; R3 launch gate; **PI go** → R4 launches Tue evening |
| Wed–Thu 10-07/08 | R4 branch run (~1.4 days) |
| Fri 10-09 | R5 evals; report with the four families and NavSim |

≈ **5 days to a refcv8 result**. The full-run path (R4′) adds ~4.5 days. NavSim leaderboard submission stays OFF until
the leaderboard is examined with the PI (PI 2026-10-04).

## 8. Decisions for the PI

1. **Branch-first** (fine-tune from 50,400, full run only on R1 evidence) — recommended.
2. **Speed input L3** (REVISED after D2, K12): the in-band ceiling is the window's own future max at R² 0.988, so a
   per-window 4-s realised max would put a speed-profile ORACLE on every window — and speed profile is the largest
   planner lever we measured, so a gain could be the leak. The input stays (PI requirement R1, 2026-09-27; the PI's
   2026-09-16 authorisation of a future-ego value inside its posted-limit window still governs), but closer
   to what a posted limit is: per window, the max realised speed over a long stretch of road ([NOW−10 s, NOW+10 s])
   snapped UP to the 8-step road-law ladder {20, 30, 50, 70, 80, 100, 120, 130} km/h (already in the v8 file), with
   the leak (OOF R² of the 6-s future max from v0 vs from bin + v0) and a shuffled-input arm reported beside every
   speed result. Default: this; alternative: the 4-s per-window max (stronger signal, larger leak).
   **⇒ SUPERSEDED by D4's measurement (2026-10-04 ~13:00):** ANY value derived from the ego's future is an oracle — a
   per-window realised max leaks **57.0 %** of the future-speed information v0 lacks (71.7 % on [NOW+2, NOW+6]). D4
   measured NON-oracle options that read only PAST ego speed (admissible at inference: the car knows its own history):
   **N2** past-20-s max snapped up to the road-law ladder with a 50 km/h urban floor — leak **3.5 %**, the human
   exceeds it on 6.5 % of windows; **N3** coarse urban / rural / motorway — leak 1.9 %, exceeded 4.4 %; no input — 0 %.
   **New default: N2**, trained with an "unknown" row on a share of windows (NavSim feeds the never-trained all-zero
   "unknown" row on **45 %** of navtest tokens — a train/deploy mismatch refcv7 carries), and every speed result
   reported beside the input-REMOVED and input-SHUFFLED arms. The PI's 2026-09-16 future-ego authorisation is then not
   needed; R1 (input + cap) is satisfied by a past-only limit.
3. **Nav L2** is a supplied route ("turn left in X m"), optimistic on PhysicalAI because it comes from the ego's own
   future — the same caveat as today's token, now time-correct. Default: adopt.
4. **Trunk changes** (map F3 stride-4 tap; anything that needs R4′): defer to after refcv8-b unless R1 forces a full run.
