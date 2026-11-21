# PREREG_NAVTEXT (DRAFT, NOT RUN, NOT FROZEN): nav-text input arm against the current nav, v8f-mini on box 2

Status: draft by the Data-Eng agent, 2026-11-16. No arm has been trained and no eval has been run. The MM freezes it (sha256) before any launch; the launch gate (`launch_gate.py` PASS token, G-DVB entry per new flag) applies. Evidence for the inputs: `NAVTOK_REPORT.md`, `agreement.json`, `navtok_table.npz` (sha256 2836369e...c8992).

## 1. Question

Can the nav the trainer reads (ego-future turn token plus distance, 3 classes) be replaced by NVIDIA's nav text (GPS trace matched to OpenStreetMap routing, parsed by `navtok.py`) without losing driving quality, and does the text carry usable information the ego-future nav lacks (roundabouts, ramps, merges, distant turns)?

## 2. Admissibility (CLAUDE.md goal-input rule)

* **Not built from the situation classifier.** The text comes from GPS plus OSM routing, the current nav from ego-future yaw segments. Neither path touches the classifier's output. The goal path and the situation path stay information-disjoint at inference.
* **Optimistic by construction.** The text is computed along the drive's own GPS trace, so as a route input it is optimistic on PhysicalAI, in the same sense as the existing ego-future nav. The deployable form (next manoeuvre and its distance from a router) is what the text mimics. Every result is worded "with the route supplied along the driven path".
* **Nav is an INPUT at training and inference** (PI, repeatedly). The vision-only inference rule applies to the situation head and is unchanged.
* Labels may use ego; inference inputs stay as listed in section 3.

## 3. Arms (box 2, one job at a time)

Baseline recipe: the DAgger fine-tune of the banked M2b (3,000 steps, lr 1e-4), argv verbatim from `PREREG_V8F_DAG_OACC.md` (INHERITED, the exact argv file is to be pinned by the MM).

| arm | nav input | seed |
|---|---|---|
| **A0_s0, A0_s1** | current v9 nav (existing banked DAG arms if their argv and data sha match, else retrained) | 0, 1 |
| **A0r** (replicate) | A0's flags verbatim | **2** (a same-seed replicate is not admissible: the full trainer is not bit-identical, and the floor must come from a different seed) |
| **N1_s0, N1_s1** | text, drop-in: channels 0-2 = TURN_L/TURN_R from the text coarse class when a text turn (any degree, U-turn excluded) is in progress or its text distance <= d_ann = max(30 m, 6 s x v) (the same rule as the current token), else FOLLOW; channel 3 = text distance/100 clipped to [0, 3]; channel 4 = distance valid. No architecture change. | 0, 1 |

Eval-time controls on every trained arm: **nav-shuffle** (nav taken from another episode, as `M1-navshuf`) and **nav-withheld** (all five channels zero).

Stage 2 (only if N1 is NON-INFERIOR, under its own PREREG): N2, the 44-class embedding plus distance, which changes `e_nav` (`v8f_mini.py:698`) and needs a G-DVB entry.

Windows without a text row (8 train clips, 1 eval clip, 36 + 2 steps before the first row) get zeros and valid 0, as the current nav does for no-row steps. The paired comparison is made on the same windows for both arms, and the 1 eval clip without rows is reported with and without.

## 4. Reads

Tier: T1-kin with DAG's decisive settings, eval-141, per seed (INHERITED from `PREREG_V8F_DAG_OACC.md`; MM confirms). Estimator: **paired episode-cluster bootstrap** over the eval-141 episodes (clusters) (`taniteval/ci.py`), B = 2,000, delta = N1 - A0 per seed, CI as [lo, hi]; "separated" = CI excludes 0. A separated CI from a one-seed arm is necessary and not sufficient: it is read against the replicate floor F_rep below.

**Four families, each per-family, never pooled** (EVAL_DOCTRINE tier stamp on every number):

| family | rows | status |
|---|---|---|
| LONGITUDINAL | target-speed error, headway / time-gap / TTC to the lead | reported |
| LATERAL | heading error, curvature error, yaw-rate error, cross-track | reported |
| TACTICAL | manoeuvre-decision quality (selected vs executed), confusion over classes, goal/anchor selection | reported, with the echo split of section 5 |
| STRATEGIC | strategic decision and goal/route quality | **PAUSED** (PI decision queue item 25, 2026-09-19: strategic layer switched off until tactical and operative drive with high quality). Not a pass, not omitted. Note: `PREREG_V8F_DAG_OACC.md` line 74 says STRATEGIC is reported for v8F; the MM rules which applies. |

ADE6 is reported beside the families, never as the result. A family that cannot be computed is stated per family with the reason and n.

**Noise floor.** F_rep = |A0r - A0_s0| and |A0_s1 - A0_s0| paired deltas per row (training-run variance), plus the existing F = 0.1207 m ADE floor (INHERITED, DAG-OACC). An effect smaller than F_rep on a row is not an effect for that row. The interval answers "another draw of episodes", F_rep answers "another training run".

## 5. Tactical nav-echo split

Split every eval window by the text at NOW: **announced** (text turn within d_ann, any side) versus **not announced** (text FOLLOW-type or turn beyond d_ann). Train distribution: of 96,346 trainer-turn windows, 56.1 % have a same-side text turn, 38.1 % a FOLLOW-type text, 4.7 % the opposite side (MEASURED, `agreement.json`). Report per arm and split: TACTICAL decision accuracy and confusion, with the paired CI, and the shuffle and withheld controls.

Echo flags (the C6 / nav-echo family): N1 is flagged as echoing the text if (a) its tactical turn accuracy on announced windows exceeds A0's by a separated margin AND (b) it collapses under nav-shuffle by more than A0 does AND (c) its accuracy on not-announced windows is not above the withheld control. An echo-flagged gain is reported as an echo, not as skill.

## 6. Outcomes, committed before any number is seen

| outcome | condition (at both seeds) | consequence |
|---|---|---|
| **NON-INFERIOR** | ADE6 N1 - A0 upper bound <= +F/2 = +0.0604 m; no LONGITUDINAL, LATERAL or TACTICAL row separated-worse by more than F_rep | the text replaces the geometry nav as a route input in the same channels; N2 (44 classes, roundabouts, ramps) is carded next |
| **BETTER** | NON-INFERIOR plus a TACTICAL or LONGITUDINAL row separated-better by more than F_rep and no echo flag | text adopted; the gain is attributed to the text's extra horizon or classes; N2 carded |
| **ECHO** | a gain with an echo flag | not adopted as skill; next: train with text dropout so the tactical head cannot copy it, same reads |
| **WORSE** | any family row separated-worse by more than F_rep at >= 1 seed, or ADE6 upper bound above +0.0604 m | text is not a drop-in. Next lever (run, not just reported): diagnose by window class using `disagreements.csv` (roundabout and bend windows where the two navs disagree), then N1b with the text restricted to the classes where the two navs agree (TURN_L/R within d_ann), same arms and reads |
| **NOISE** | the N1 deltas are inside F_rep on every row | no lever effect is claimed; spend goes to more seeds, not windows |

Bars, controls and arm list do not change after any N1 number is seen. A bar that fails is reported as written.

## 7. Cost (ESTIMATED, INHERITED from DAG-OACC "~2 h per seed", not re-measured)

* 5 trained arms (A0 reused if banked: 3 new arms minimum = N1_s0, N1_s1, A0r; 5 if A0 must be retrained) x ~2 h = **~6 h** (minimum) to **~10 h** box-2 GPU, sequential, plus eval per arm and two eval-time controls (ESTIMATED, not measured here).
* Tokens: table and parser already built; the arm needs one new loader path (~50 lines) feeding `J["nav"]` from `navtok_table.npz`, a G-DVB entry and a launch gate PASS. A launch needs the PI's go for box-2 time (ruling 11-15 covers ACTSENS only).
* Pre-launch checks the MM must run (reachability, class "built, tested, unreachable"): trace the flag from argparse to the `J["nav"]` tensor with `file:line`, a non-zero gradient on `e_nav` and a control that reads exactly 0.0, and a mutation that swaps the table for zeros and must turn the check red.

## 8. Known risks

* T0 alignment is a constant (-195 ms), ESTIMATED; the T0 scan shows a flat curve, so the risk is small.
* The text's 127 m median look-ahead exceeds the 6 s token horizon; the drop-in keeps the current horizon on purpose, so N1 isolates the source change only. A longer-horizon input is N2.
* The text and our geometry define "turn" differently (router node vs sustained yaw >= 30 degrees); 38 % of trainer-turn windows have a FOLLOW-type text. This is expected to show up in the TACTICAL family and is why the WORSE branch exists.
