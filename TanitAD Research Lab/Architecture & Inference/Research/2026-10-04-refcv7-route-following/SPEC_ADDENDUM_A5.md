# SPEC addendum A5 — a TIME-LOCALISED nav input for the pick (pre-registered BEFORE it is computed)

**Written 2026-10-04 ~12:05 Berlin (10:05 UTC) by the Master Mind**, after A1–A4 were scored and before any number
with a time-localised nav exists. POST-HOC in the sense that A4's bound B1t (a perfectly timed nav filter: −0.566 m
[−0.986, −0.239] on GT-turn windows, turn direction 1.000, straight 0.000) motivated it; the arms, the reported arm and
the bar are fixed here and cannot move after the numbers.

## Why
refcv7 fed ONE nav token per clip on every window (`--nav-from-v7`). The v8 record also carries `nav_30s.entries`:
every turn with anchor-relative `t_start_s` / `t_end_s` (0 = the anchor, RAW 8.0 s) and arc-length distance. SPEC §5's
nav arms V2 (hard nav-side filter) and V3 (nav-compliance ×10) gained turn direction but damaged straight windows
(V2: straight ΔADE +0.326 [+0.197, +0.459]), which is what a clip-constant "turn" token does on the straight part of a
clip. A5 asks: **with a time-correct nav, do the same rules gain the turns without the straight damage?**

⚠️ The nav stays an ORACLE supplied route (provenance ego-future), admissible at inference as "the navigation system
says turn left in X m", optimistic on PhysicalAI by construction. A pass is an inference-time route-following gain
under that caveat, not a training lever.

## Definition (literal; no parameter is fitted on EVAL)
* `t_now_raw` = the trainer's own clock for the window: `grid_start_s + (t + W − 1 + n_stack − 1) · dt_s` from
  `refcv6_clip_clock_sidecar.jsonl` (the rule `refc_v3_train.py::_now_s` implements; W and n_stack from the run's
  config). Clips without a sidecar row use the trainer's own fallback. **Known-value control K1:** on the reel windows,
  the computed `t_now_raw` must equal the replay bank's `t_label_s` (`rows.jsonl`) to ≤ 1e-3 s on every window.
* `t_rel = t_now_raw − 8.0` (the anchor). Join clip ↔ record by `sha12(clip_id)` == the bank's `win_sha12`.
* **nav_tl (horizon H = 6.0 s):** the first entry with `t_end_s > t_rel` (not yet finished). If its
  `t_start_s − t_rel ≤ 6.0` (includes a turn already under way), nav_tl = its side: `NAV_TURN_L` → left,
  `NAV_TURN_R` → right; every other token, no such entry, or a start > 6.0 s ahead → follow (uninformative).
  The token tally is reported. **Known-value control K2:** on windows whose t_rel ∈ [−0.05, +0.05] and whose
  `nav_command.args.time_s ≤ 6.0`, nav_tl must equal the clip token's side on 100 %.
* Sensitivity (reported, NEVER chosen from): H = 8.0 s.

## Arms (all on the SAME captured fans; scored on eval_s0g, replicated on eval_s1; nothing fitted on EVAL)
| id | rule | status |
|---|---|---|
| **T2** | V2's rule (nav-side hard filter, τ 0.05, empty → V0) with nav_tl in place of the clip token | **REPORTED ARM** |
| T3 | V3's rule (nav-compliance term × 10) with the compliance predicate recomputed against nav_tl | secondary |
| T4 | A2's X1 linear re-scorer with `navc` and `agree_nav_side` recomputed from nav_tl, refit on train_s0 with the same 5-fold episode-grouped CV and α grid | secondary |
| T2c | CONTROL: T2 with each clip's nav_tl series taken from another clip (a fixed seeded derangement of clips, seed 0) | must NOT pass the bar |
| T2h8 | T2 at H = 8.0 s | sensitivity only |

## Bar (unchanged from SPEC §5)
1. turn ΔADE < 0 and turn dir-correct up, both CIs excluding 0;
2. straight ΔADE ≤ +0.05 m with CI upper bound ≤ +0.10 m;
3. all-window ΔADE ≤ 0;
4. the effect replicates on seed 1.
Estimator: paired episode-cluster bootstrap vs V0, the same draws as SPEC §5. Reported with the four metric families
(§2.1 format) and, for scale, the fraction of A4's B1t bound captured: ΔADE_turn(T2) / (−0.566).

## Reading rule (fixed now)
* T2 passes, T2c fails ⇒ the route-following defect is largely nav TIMING: L2 (time-localised nav) enters refcv8 as a
  training input AND ships as an inference rule meanwhile.
* T2 fails on straight damage despite time-localisation ⇒ the fan's "right direction" candidates are poor on
  straight-adjacent windows; the lever is selector training (R1), not nav.
* T2 fails on turns (no turn gain) ⇒ nav timing is not the bottleneck; selector training ranks first.
* T2c passes ⇒ the instrument is broken; nothing from A5 is quotable.
