# STRATEGIC family on v8f-mini reads: audit and minimal instrument plan (Benchmarks agent, 2026-11-16)

Tip read: `861a557f` (`agent/arch-inf-20260803`, via `git ls-remote`, then `git show`). Evidence classes: M = MEASURED (artifact named), P = PUBLISHED, I = INHERITED (not re-verified), E = ESTIMATED, H = HYPOTHESIS, U = UNVERIFIED.
Tier of everything below: T0 read of own held outputs unless stated; no number here is a driving number.

## 0. Headline

1. **The four-family read the ACTSENS and HWM-PLAN reads use has no STRATEGIC cell at all.** `hwm_eval.py:13` lists three families; `:29` hardcodes the string "PAUSED (PI item 25 ...)"; `:231-233` prints it; `assemble_arm.py:37` and `verdict_actsens.py:126,167` copy it. With the pause ended (PI 2026-11-16) these strings now state something false and each is a work item (M, grep).
2. **v8f-mini does emit a strategic output: S's route checkpoints** (x, y, heading at 30 / 50 / 80 m of arc, NOW frame), `V8FMini.run_s` `v8f_mini.py:800-808`, returned as `out["ckpt"]` (`:878`). It has no output beyond 6 s in time, and no 6-30 s intent head. The 80 m checkpoint passes 6 s only when v0 < 13.3 m/s (E, arithmetic). The HWM planner's goal is a latent (`hwm_plan.py:108-150`), not decodable to a route without a decoder (U on decoder existence).
3. **Existing, runnable instruments cover goal/route-setting error only** (checkpoint pos/heading error, plan-point-vs-checkpoint). They score against v9 RC-A, which is derived from the ego's own future, and the arm's `nav` input is derived from the same ego future (NAVTOK_REPORT section 1, manifest policy `allow_oracle_nav (ego-future)`). They are echo-prone; the repo already holds the split arms (`v8f_mini_eval.py:552-581`).
4. **Decision-grade STRATEGIC cells from the NVIDIA nav text are computable now, but only at 3 of 4 places with enough n.** On the eval141 grid (2,399 windows, 141 episodes): 46 / 120 / 214 windows (18 / 22 / 24 episodes) at 30 / 50 / 80 m (M, `proto_eval141_counts.json`). The in-plan cell (does the 6 s plan take the announced turn) has only 13 scorable windows in 3 episodes on the ACTSENS score half (M, `proto_actsens_read.json`): too thin to decide anything.
5. **PI/MM decisions to escalate:** (a) a 6-30 s strategic intent cannot be scored because no arm outputs one; adding a head (for example checkpoints at 120 / 200 / 300 m, or a route-intent token) is an architecture change and training, so it needs a PI go. (b) Whether the next v8f-mini eval gets an N1 arm fed by the nav text (PREREG_NAVTEXT_DRAFT); the cells below are designed to be echo-split for it. (c) Pre-register `COMPLETE_M` and the thresholds (section 3) before any decisive read.

## 1. Inventory of STRATEGIC implementations (tip `861a557f`)

| # | file:line | computes | inputs | horizon | runs on v8f-mini outputs? |
|---|---|---|---|---|---|
| 1 | `stack/scripts/v8f_mini_eval.py:111-120` `checkpoint_metrics` (called `:160`); `taniteval/v8f_consistency.py:294` `checkpoint_errors` | S checkpoint position error (m) and heading error (deg) at 30/50/80 m, episode-cluster CI | `out["ckpt"]` vs v9 RC-A `rc`, `rc_ok` | 30-80 m of arc | **Yes**, it is v8f-mini's own read. n_valid 96.8 / 96.7 / 95.6 % of windows (M, `join_eval141.npz`) |
| 2 | `taniteval/v8f_beta_eval.py:89` `route_adherence`; used `v8f_mini_eval.py:427-430` | distance from the plan's point at 30/50/80 m of arc to the GT checkpoint, NaN if plan shorter than arc | plan xy (30 knots, 6 s) vs `rc` | arcs 30-80 m inside the 6 s plan | **Yes** (any 30-knot plan, including HWM plans). Only inside the `--plan beta` path today |
| 3 | `v8f_mini_eval.py:552-581` | nav-only checkpoint baseline floor; `M1-navshuf` and `M1-navonly` paired contrasts | `nav_baseline_ckpt`, seeds | 50 m, 80 m | **Yes**, this is the existing echo split (see section 3) |
| 4 | `taniteval/four_families.py:1627` `strategic`, `:1487` `strategic_unavailable`, `:1462` `STRATEGIC_NA_BY_CORPUS` | route-class decision (legacy) or the honest n/a with reason and n | refc route head `route_acc`, option sets | route class | **No**: v8f-mini has no route-class head. The n/a path is the binding form for what cannot be computed |
| 5 | `taniteval/strategic_optionset.py:558` `strategic_family` | decision vs map-derived option sets, best-constant control | `map.xodr` option sets, predictions | junction decisions | **No**: needs maps (PhysicalAI-AV ships none; only nurec/alpasim clips) |
| 6 | `stack/scripts/eval_v58f.py:454`, `e_wc2_sigma_star.py:962` | STRATEGIC written as n/a with reason | none | none | n/a statements; the reason text ("no route/goal label exists on PhysicalAI-AV") is false for the nav text and for the v7.x labels, and is a stale-claim work item |
| 7 | `stack/scripts/lan_probe.py:425-640` `STRATEGIC_route_accuracy` | probe accuracy of a route label | refc latent probes | route | **No** (refc trunk) |
| 8 | `stack/scripts/eval_behavior.py:579` `strategic_probe_eval` | decodability probe of coarse route intent | shared state | coarse | **No** (flagship ckpt); its own header says the layer emits no decodable code |
| 9 | `stack/scripts/p_floor.py:54`; `gate_emitters.py:188-233`; `verdict_refcv6.py:111-136,301-338` | S1 clause `families.STRATEGIC.route_acc`; pause flag with `--unpause-strategic` | refc `route_acc` | route | **No**: v8f-mini has no `route_acc`, so S1 reads MISSING_DATA there. The pause logic itself has the PI's 2026-11-16 ruling to act on (lifting it is `--unpause-strategic` or setting the pause record to None, `:111-119`) |
| 10 | `hwm_eval.py:13,29,204-233` | families LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC a string | HWM plans | 0-2, 2-6, 0-6 s | the gap this plan fills |

Greps run: `STRATEGIC`, `strategic`, `route`, `goal_point`, `S1`, `unpause` over `taniteval`, `stack/scripts`, `stack/tanitad` at the tip (M). Many hits under `Research/**/incoming` are banked per-run copies; not enumerated individually (U on completeness there).

## 2. What v8f-mini emits beyond 6 s, and per-member computability

* **Emitted:** O's 6 s trajectory (60 x 0.1 s); T's goal logits, lat/lon logits; **S's three route checkpoints** `[B, 3, 3]` = (x, y, psi) at arcs 30 / 50 / 80 m (`v8f_mini.py:43,800-808`; label RC-A built by `v8f_mini_data.py:134-147`). S sees `nav` (5 channels: FOLLOW, TURN_L, TURN_R, d_next/100, valid; `v8f_mini_data.py:198-209`, `e_nav` Linear(5, d) at `:698`), scene summary, ego dynamics, max speed.
* **Not emitted:** any intent, goal or route token at 6-30 s; any detected-context token (the HWM `ctx` token is a separate HWM input, `hwm_plan.py:113`); no checkpoint beyond 80 m.
* **HWM planner (the ACTSENS reads):** returns a 30-knot (6 s) plan; the goal is a level-1 latent at the goal frame (default the end of the level-2 horizon, 6 s; 12 s only in the G2 variant, `hwm_plan.py:63-73`; U on the exact c3 stride). The ACTSENS reads use **HWM-ORACLE: the goal is the encoded true future**, so any strategic cell on those plans measures the world model's use of an oracle, not a strategic decision (T0 diagnostic).

| member | computable now? | with what n (eval141 grid unless stated) | reason when not |
|---|---|---|---|
| decision: route adherence inside the plan (S_ADH) | yes, thin | M: 13 windows, 3 episodes on ACTSENS (994 windows). E: about 31 windows on the 2,399 grid (13/994 scaled; scale not verified) | the plan is 6 s (about 60 m at 10 m/s); the text announces at median 127 m (NAVPROBE) |
| decision: false turn (S_FALSE) | yes | M: 919 of 994 ACTSENS windows | the text does not announce bends or ramps the logged path takes; the logged path reads 13.3 % "false turns" (M), so S_FALSE is read only against the logged path, never as an absolute |
| decision: S checkpoint heading vs announced side (S_CKPT) | yes (needs an `eval_<tag>.npz` carrying `ckpt`) | M: n 46 / 120 / 214 at 30 / 50 / 80 m; episodes 18 / 22 / 24; side split L/R 17/29, 31/89, 44/170 | none. Right-turn heavy: report a best-constant control |
| goal/route-setting: checkpoint position and heading error | yes | M: n_valid about 2,300 per arc, 141 episodes | none; echo-prone, see section 3 |
| strategic intent 6-30 s | **no** | would-have n: E from ACTSENS, text turn windows by time-to-manoeuvre at d/v0: 6-12 s 32, 12-30 s 69, over 30 s 20 (M, `count_n.py`) | no arm outputs anything at that horizon |
| derived goals, detected info | **no** | n/a | no such output on v8f-mini; HWM `ctx` token is an input, not scored |

Report the bottom two rows per family as `strategic_unavailable`-style entries with reason and would-have n (clause 5), not as a pass.

## 3. Minimal cells from the nav text

Ground truth: `navigation.parquet` text, parsed by `navtok.py` into 44 classes and attached per cache step by `navtok_table.npz` (sha256 `2836369e...c8992`, rule: latest row with ts <= now_s + T0, T0 = -195,000 us). Coarse side from the class name: `TURN_L/R`, `TURN_SLIGHT_*`, `TURN_SHARP_*`; UTURN, CONTINUE, KEEP, MERGE, RAMP, roundabouts give side 0 (not turns). Rows with flag != 0 (no text row) are NaN in every cell, never 0.

| cell | definition | inputs | n on eval141 |
|---|---|---|---|
| **S_ADH** | announced L/R turn at d_text with d_text + COMPLETE_M inside the plan's own arc; 1 if side-signed heading change of the plan over [0, d_text + COMPLETE_M] >= theta (25 deg, SLIGHT 10 deg); continuous twin = signed degrees | plan xy and psi (HWM `integrate`, `hwm_plan.py:523`), text class and distance, v0 | E about 31 windows; **not decision-grade** |
| **S_FALSE** | no turn announced within plan arc + 30 m; 1 if plan net heading change >= 25 deg | same | M 919 / 994 on ACTSENS; read only as plan minus logged |
| **S_CKPT** | S heading at arc a (30/50/80) is on the announced side by >= theta, scored where d_text + COMPLETE_M <= a | `out["ckpt"]` psi, `rc_ok`, text | M 46 / 120 / 214 |
| **S_INT by band** | S_CKPT split by distance band; metres to seconds at the window's v0 (6-30 s is 6 v0 to 30 v0 m) | v0 | reported only for bands with n >= 30 and >= 10 episodes, else n/a with n |

`COMPLETE_M = 20`, the thetas and the 30 m margin are H: pre-register them, and run one sensitivity row (COMPLETE_M 10 and 30) before any decisive read. Prototype: `strategic_cells.py`; 9 tests green (M).

**Controls that must read known values:** HOLD plan: S_ADH exactly 0.0 (test, passes). Logged path: S_ADH ceiling; on 10 scorable windows it read 1.0 (M), so the text label and the human path agree where scorable. Mirrored side: must flip to 0 (test). Best-constant predictor on S_CKPT (always "right" scores 170/214 at 80 m among scorable windows, E from side counts): report the arm against it.

**Leak and echo check (CLAUDE.md goal-input rule):**
* The text is GPS + OSM routing text; it carries no situation-classifier output (M, parquet schema metadata quoted in NAVTOK_REPORT). As a goal/route input it is admissible; as a scoring label it is independent of the model.
* **Arms fed the v9 oracle nav** (every trained v8f-mini arm today): the text is a different derivation (router vs sustained-yaw) and agrees with the trainer token on the same side in 1,200 of 2,334 trainer-turn eval windows = 51.4 % (M, NAVTOK section 4). Scoring against the text is therefore not an echo of the input, but both come from the drive's own path, so the text cells are optimistic as route-following evidence; say so beside the number.
* **Arms fed the text (N1):** cells against the same text are an echo. Split by three inference passes on the same windows: real nav, donor-episode nav (the existing `M1-navshuf`), and nav withheld (zeros, the trained-in "no row" state; the existing `M1-navonly` handles the scene-only variant). The reported skill is cell(real) minus cell(donor), paired. Cell(real) alone is never quoted.
* **S_CKPT vs RC-A** (existing rows): the label and the input share the ego future. Quote only with the nav-only baseline floor (`:552`) and the navshuf contrast (`:576`).
* The goal path and the situation path stay information-disjoint; nothing here feeds a classifier output into a goal.

**Estimator:** paired episode-cluster bootstrap, B = 2000, `taniteval/ci.py:279` (arm vs arm, arm vs logged, arm vs HOLD), with `episode_cluster_bootstrap` (`:229`) for single arms. It answers "another draw of episodes". A claim that a lever moved a cell needs a replicate arm (H-ESTIM-SEED-1), and a sampling planner needs an inference-seed replicate. Prototype check: `paired_hwm_oracle_minus_logged_adh` on 8 windows / 3 episodes returned delta -0.75, CI [-1.0, 0.0], not separated (M); it shows the plumbing and the thinness, nothing else.

## 4. Build plan

1. **Table join (new, about 40 lines).** `strategic_text.py`: load `navtok_table.npz`, map (clip, step) to a row. Window step for ACTSENS windows is `now + 3` (994 of 994 land on the eval grid, a necessary check only; U on the offset until compared on `v0` against the join). Plug in at `hwm_eval.py:204` `build_report` and at `v8f_mini_eval.py:160` beside `checkpoint_metrics`.
2. **Cells module** = `strategic_cells.py` (done, prototype). Production imports `hwm_plan.integrate`.
3. **hwm_eval.py:** add `STRATEGIC: (s_adh, s_false)` to `FAMILIES` (`:13`) with band `0-2s` and `2-6s` rows empty by construction and `full6s` the only one filled; delete the `STRATEGIC` string at `:29`; replace `"strategic"` at `assemble_arm.py:37` and `verdict_actsens.py:126,167` with the computed block and the section 2 n/a entries. Add `NOT_COMPUTED["STRATEGIC.intent_6_30s"]` at `:21` in the same style as the existing entries.
4. **v8f_mini_eval.py:** `strategic_text_metrics(extras["ckpt"], X, table)` returning S_CKPT per arc with CI and the best-constant row; call it next to `checkpoint_metrics` (`:160`). Needs `eval_<tag>.npz` with `ckpt` (written at `:339`); U whether the banked eval npz files carry it.
5. **Pause clean-up:** `verdict_refcv6.py` pause record per PI ruling (lifting is the PI's recorded call, 2026-11-16: set the record None or pass `--unpause-strategic`); `four_families.py:1462` and `eval_v58f.py:454` reason texts to be rewritten to name the layer (published corpus vs episode build vs augmented v7.x labels vs nav text).
6. **Decision-grade plan:** read S_CKPT on the full eval141 grid for every arm; an arm needing S_ADH to decide gets a larger window set first (a stride-1 grid on the 40 episodes with a text turn, E about 10 x more windows but the same 40 episodes, so the CI width is episode-limited; say so).

**Tests (extend `test_strategic_cells.py`):** HOLD reads 0.0; mirrored side flips; unreachable is NaN; no text row is NaN; UTURN/CONTINUE are not turns; a threshold-0 mutation makes HOLD "adhere" (this proves the check is not vacuous, and it is the deliberate-regression arm). Add for production: (a) deleting the single line that attaches the text row must turn the integration test RED; (b) donor-shuffled nav must move the N1 cell to the shuffled value on a synthetic arm that echoes its input; (c) a `grep` of `hwm_eval.py` for the literal "PAUSED" must return 0 and a same-breath control on a string that exists must return non-zero (guard-in-implementation rule).

**Cost (E):** about 150 lines of new code plus 100 lines of wiring; one Sonnet agent, roughly half a day; CPU only, under 1 minute and under 1 GB for the cells. The S_CKPT read needs one v8f-mini `run` per arm if no banked npz carries `ckpt` (a job on the 8 GB cache track; not requested here).

## 5. Blocked levers (named)

* 6-30 s strategic intent: blocked on an output that does not exist. Unblocks with a PI go on adding a strategic head (checkpoints at larger arcs or a route-intent token) and training.
* S_ADH decision-grade n: blocked on window count in the plan horizon; unblocked by a denser eval window set or by S_CKPT as the primary cell.
* N1 echo split: blocked on an N1 arm and its launch gate.
