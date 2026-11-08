# AMENDMENT 10 (DRAFT, not hashed, not frozen) to BEHAVIOUR_SCORECARD.md (sha256 fac6c7f1...): TACTICAL executed-decision agreement (work item A-14)

**Status:** DRAFT for Master Mind review. **Date:** 2026-11-08.
**Written without reading any T2 outcome or any P6 output.** Only schemas were read (`code/ROLLOUT_SCHEMA.md`). No threshold below was chosen from a score. If frozen, it must be hashed before the first tier-2 ADOPT read and applies only to checkpoints read after it.
**Scope:** defines what AMENDMENT_1 A-14 left open. No threshold of any existing rule, no window and no adoption bar changes. Implementation: `tactical_detector.py`; tests: `test_tactical_detector.py` (37 checks).

## 1. Principle
REFe emits no tactical output, so the decision is read from the **executed trajectory**. The same pure function `decide()` reads the human log and the driven trajectory. **Events are defined from the scene and the human log only, never from the driven run**, so the event set is identical for every checkpoint (a precondition for pairing). Tier-1 reads keep TACTICAL executed-decision as N/A (A-14 text); this amendment is for tier 2.

## 2. Inputs (all on the 10 Hz metric grid, `dt = 0.1 s`; 20 Hz schema arrays are taken every second sample)
Ego `t, x, y, yaw, v`; agents (`agent_*`, scalar speeds, A-3); `signal_state`, `signal_governs_ego`, `stop_line_s`, `conflict_zone_xy`, `conflict_agent_ids`, `human_ref_valid_until_s`; ego length and width. **ASSUMED additional field (open item 1): `lane_offset` (T,), signed distance in m of the ego box centre from the route-lane centre line, + left, in both `rollout.npz` and `human.npz`.**

## 3. Reference path (identical to the scorer's conventions, re-implemented)
The human log is the reference path: 0.2 m distance filter (A-7); straight extension along the chord of the last 5.0 m of the moving part, ticks with `v >= 0.5` (A-6). Station `s` and signed offset come from projection onto it. Front-bumper station = `s + L/2`. Ticks after `human_ref_valid_until_s` (A-5) are cut for every station-based class (LINE, MERGE, LEAD, FREE); the count is reported as `ref_invalid_ticks`.

## 4. Primitives
- **STOPPED** (a run): `v <= 1.0 m/s` for >= 10 consecutive ticks (1.0 s). A creep counts as stopped (A-8).
- **Lane departure** (a run): `|lane_offset| > 1.75 m` for >= 3 ticks (A-5 half-lane; persistence from section 1). Its sign is the sign of the mean offset.
- **Lead** at a tick: nearest vehicle (class 0) whose box lies in the corridor (`|offset| - w_agent/2 <= 1.5 m`) and whose rear is at or ahead of the ego front. `gap` = rear station minus ego front station.
- **Free road, held ticks**: as section 1 and A-1 (no lead within `max(4 s * v, 15 m)`, no lead with TTC <= 8 s), minus ticks on which the governing signal is red and the human front bumper is 0 to 40 m before the line.
- **Governing state**: the most restrictive of the governing heads (red > yellow > green > none).
- Events start no earlier than the 2.0 s warm-up.

## 5. Events (what counts as one; anchor, window, closed form)
| class | anchor (scene or human only) | window |
|---|---|---|
| **LINE_R** | first tick in a contiguous red phase with the human front bumper 0 to 40 m before `stop_line_s` | trigger to end of the red phase |
| **LINE_G** | same on a contiguous green-or-none phase (so it is the launch after a red, or an approach on green). Yellow belongs to no event. Needs signal heads and a finite `stop_line_s` | trigger to end of the phase |
| **CONFLICT** | run (>= 3 ticks) in which some conflicting agent (all agents if the id list is empty, A-13) is inside the zone or has `dist(agent centre, zone) / max(v_agent, 0.1) <= 2.0 s` (c3) | the run |
| **MERGE** | a vehicle with `|offset| >= 2.0 m` at its first tick after warm-up that reaches `|offset| <= 0.5 m` with gap in [0, 30] m and heading within 30 degrees of the path (section 3.1, AMENDMENT_4 AMB-10) | the entry tick only |
| **LEAD** | same lead id continuously >= 3.0 s with gap <= 60 m (steady-lead predicate, section 3.1), from the lead series of the **human** | first 8.0 s of that run |
| **FREE** | a free-road interval >= 2.0 s of the **human** | the interval |
| **LANE** | a lane departure of the **human** | from 4.0 s before to 4.0 s after the human run (clipped to the scenario) |

Events of different classes may overlap in time (a merging vehicle becomes the lead). Each is its own row. The pooled table is therefore a sum over classes, **descriptive only**; the per-class tables are the reading. A phase or run that never meets its trigger, or is shorter than the persistence, is counted (`phases_without_trigger`, `sub_persistence_runs`), never silently dropped.

## 6. The five decisions (closed form; first match wins in the order given)
`front` = front-bumper station; `line` = `stop_line_s`; "window" = the event window.
- **LINE_R / LINE_G.** NOT_ENGAGED if the ego is never within 40 m before the line in the window. **go** = `front > line` at some tick of the window. **stop** = STOPPED with `front <= line`. Otherwise AMBIG_UNRESOLVED. (Crossing dominates: stop-then-cross is go; on LINE_R that is the c2 violation.)
- **CONFLICT.** **go** = the ego footprint (9 box points, plus zone vertices inside the box) intersects the zone at any tick of the window. **yield** = it never does while the ego centre comes within 40 m of the zone (wait, slow, or enter only after the conflict run ended). Otherwise NOT_ENGAGED.
- **MERGE.** At the entry tick, `d = s_agent - s_ego`; `h = (L_agent + L_ego)/2`. NOT_ENGAGED if `|d| > 40 m`. **yield** (ego lets it in, agent ahead) if `d >= h`; **go** (ego first) if `d <= -h`; otherwise AMBIG_CONTESTED.
- **LEAD.** **stop** = STOPPED in the window. Else **change-lane** = a lane departure in the window. Else **follow** = the event's lead is the ego's lead on >= 90 % of window ticks. Otherwise AMBIG_LEAD_LOST.
- **FREE.** **stop** if STOPPED; else **change-lane** if a lane departure; else **go** if `v > 1.0 m/s` on >= 90 % of ticks; else AMBIG_SLOW.
- **LANE.** NOT_COMPUTABLE if `lane_offset` is absent. **change-lane** = a same-sign lane departure in the window. AMBIG_WRONG_SIDE = only opposite-sign departures. No departure: the generic reading: **stop** > **follow** (any lead on >= 90 % of ticks) > **go** (`v > 1.0` on >= 90 %) > AMBIG_SLOW.

The five labels are `go, yield, stop, follow, change-lane`. A driven lane change in a LEAD or FREE window where the human did not change lane is captured there (human follow/go, driven change-lane); a human lane change is captured by LANE.

## 7. Alignment, ambiguity, absence
- **Alignment.** Events live on the scene clock (same origin and 10 Hz grid for both runs, ROLLOUT_SCHEMA). Both runs are read on the same index window. Station-based quantities use the one human reference path for both. No time warping and no event matching by nearest decision: a driven run that does the right thing in a different tick is still read correctly if it falls inside the window; LANE alone carries a 4.0 s tolerance each side because a manoeuvre has no scene anchor.
- **Named classes** (rows and columns of the table, each with its count, never dropped): `AMBIG_UNRESOLVED, AMBIG_LEAD_LOST, AMBIG_SLOW, AMBIG_CONTESTED, AMBIG_WRONG_SIDE, NOT_ENGAGED, NOT_COMPUTABLE`.
- **No event** in a scenario: no row; the audit records the phases and runs that did not trigger. A suite with no event at all reports `NO_EVENTS (family N/A, n = 0)`. This is not a pass.
- A synthetic or real scene whose class cannot occur (no heads, no zone, no lane offset) simply contributes no such events; the report lists the classes present with n.

## 8. Confusion table, estimator, small n
- **Table:** rows = human label, columns = driven label, over the 5 decisions plus the 7 named classes (12 x 12), one table per event class (primary) and one pooled (descriptive). Every cell is a count; n per class and per cell is printed.
- **Agreement** = diagonal share over events where **both** labels are among the five decisions (`n_resolved`). Events with a named class on either side are counted and shown, and excluded from that denominator. **Human-named rows are an instrument statement** (the human log itself is ambiguous there), as in Gate H.
- **Estimator.** Single arm: episode-cluster bootstrap (cluster = log, falling back to scenario), 10,000 draws, seed 20260927, 95 % percentile (section 4 settings). Two arms: **paired** bootstrap of `agreement(cand) - agreement(base)` over the same event keys, same settings; unpaired keys are counted. It answers "would another draw of scenarios say this?" only; it is blind to training and sim variance (`H-ESTIM-SEED-1`), so the baseline-pair floor of section 4 (i) applies to any difference claim.
- **Small n.** An interval is computed only with >= 10 clusters having resolved events and, per class, >= 20 resolved events (ESTIMATED: a percentile bootstrap over fewer clusters is not stable; with tier-1's 10 scenarios it almost never holds). Below that the report gives counts and the table only, status `INSUFFICIENT_N (counts only; no rate verdict)`. It is never a pass and never an omission.
- **Role in the scorecard (for the Master Mind to rule, open item 5).** Proposed: reported as TACTICAL row **c5**, descriptive in tier 1 (N/A), and in tier 2 a rule in the section 4 test (bad direction = lower agreement, paired CI above the sim floor) only for classes that clear the small-n rule; classes that do not clear it never enter a pass count. Until ruled, c5 is DESCRIPTIVE and cannot ADOPT or REJECT. STRATEGIC stays PAUSED (PI 2026-09-19, item 25) and is not touched.

## 9. Threshold ledger (every value; none chosen from a T2 outcome)
| value | source |
|---|---|
| 10 Hz, 2.0 s warm-up, persistence 3 ticks, free road (4 s, 15 m, TTC 8 s), corridor 1.5 m, 0.90 share | scorecard section 1; 0.90 from a1 |
| STOPPED `v <= 1.0` for 1.0 s | A-8 and `STOP_MIN_S` / `C4_STOP_V` (`behaviour_score.py:50-51`) |
| 40 m approach / engagement | c1 (ESTIMATED there) |
| 2.0 s zone horizon | c3 |
| lane half-width 1.75 m | AMENDMENT_5 |
| steady lead 3 s, gap 60 m; window 8 s | section 3.1 (steady lead: 8 s, gap <= 60 m) |
| merge 2.0 / 0.5 m, gap 30 m, 30 degrees | section 3.1 and AMENDMENT_4 AMB-10 |
| `go` needs `v > 1.0` on 90 % of ticks | 1.0 m/s from c1/c4, 0.90 from a1: ESTIMATED combination |
| LANE tolerance 4.0 s | ESTIMATED: the a2 horizon; wide on purpose because a manoeuvre has no scene anchor |
| free-interval window = whole interval; FREE `go` share | ESTIMATED, as above |
| min 10 clusters, 20 events | ESTIMATED (section 8) |
| bootstrap 10,000 / 20260927 / 95 % | section 4 |

## 10. What this cannot say
- `follow` and `go` are kinematic readings: a driver that is slow for no reason looks like `follow` only if a lead is present; otherwise it is AMBIG_SLOW, not an error.
- Traffic is replayed (section 0), so `yield` at CONFLICT/MERGE is scored against a fixed agent track.
- Junction and light events are **synthetic until map lifting exists** and carry that label; they are not pooled with nuPlan-seeded events in any adoption statement.
- The kinematic lane readings depend on `lane_offset`; without it LANE is NOT_COMPUTABLE and change-lane cannot appear in LEAD or FREE, which is reported per family with the reason and n.
- The detector reads decisions, not their quality: a late or harsh go is still `go` (rules c1 to c4, d, f carry quality).

## 11. Open items for the T2 session (assumed names; not contacted)
1. **`lane_offset`** (T,) float, + left, from the sim's route-lane chain, in `rollout.npz` and `human.npz`. Assumed from the existence of `human_ref_valid_until_s` (A-5), which already needs the same quantity. If the sim cannot supply it for the driven run, LANE and change-lane are NOT_COMPUTABLE.
2. `agent_cls` (A-2) so that only vehicles can be leads; the detector defaults to vehicle and the output must carry the defaulted flag as the scorer does.
3. **Equivalence with the scorer.** `RefPath` and `lead_series` re-implement `HumanPath` (`behaviour_score.py:292`, A-6/A-7) and the lead logic; the corridor test uses the agent half-width, not full box overlap. Before freezing, compare station, gap and free-road flags against the scorer's `--dump-ticks` on `example/` and on Gate H output (any difference is a defect in one of them).
4. The scorer's "held" latch (CH-1, latched to the first `v > 1.0` after green) is not reproduced; FREE excludes only red-and-near-line ticks.
5. MM ruling on the role of c5 in section 4 (section 8 proposal) and on a count/pooling convention for overlapping events.
6. Whether yellow phases deserve their own event class (none here: c2 is red-only).
7. The Gate H run of the suite should include this detector: the human-vs-human table must be purely diagonal (it is, on all synthetic cases: test C1).

## 12. Master Mind rulings on the open items (2026-11-08 ~19:00Z; still a DRAFT, not frozen)
1. **c5 is DESCRIPTIVE in the first tier-2 read.** It is reported per class with n and cannot ADOPT or REJECT. c1 to c4 keep their role as the deciding TACTICAL rows. The detector has never run on a real driven scene, and adding a new deciding row before its first read would change the adoption bar. Promotion needs its own amendment after that read.
2. **Overlapping events are counted per event class.** A pooled table is descriptive only and never enters a decision.
3. **No yellow event class:** c2 is red-only, and yellow phases are counted inside the phase they are reported under.
4. **Freeze waits on items 1 to 3 of section 11,** which only the T2 session can answer: the field names, and equivalence with the scorer's `--dump-ticks`. If `lane_offset` is unavailable for the driven run, LANE and change-lane stay NOT_COMPUTABLE with that reason, and the family is reported as such, never as a pass.
5. **`lane_offset` (ruling 18:56Z, on the T2 session's report that it is not saved for either run):** a post-hoc per-scene sidecar `lane_offset.npz` is accepted, no rollout schema change. Conditions: one deterministic script with its sha256 recorded; the same route-lane chain for both paths, its id or hash in the file; agreement with `d` in refe_calls.json checked first; unprojectable ticks are NaN, counted and NOT_COMPUTABLE, never filled; no outcome read. To be built after tier 1. `agent_cls` is saved in the driven files only: the source of the human side's agents is stated here before freezing.
