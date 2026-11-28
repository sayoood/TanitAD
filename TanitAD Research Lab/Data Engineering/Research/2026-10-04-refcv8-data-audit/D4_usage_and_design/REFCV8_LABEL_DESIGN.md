# D4 — refcv8 LABEL DESIGN: replacements for every channel the usage audit judged WRONG or SUBOPTIMAL

Stream D4 of the refcv8 data audit · 2026-10-04 · revision 2. This revision folds in D2 (label validity) and the
route package's A5/A6 at the Master Mind's instruction; none of their numbers were re-measured here.

## Stamps

**Evidence classes.**
* Every number is `MEASURED (D4)` unless it says otherwise. Sources: `raw/d4_measure.json`,
  `raw/d4_nav_announced.json`, `raw/d4_vmax_nonoracle.json`, `raw/d4_lanechange.json`, `raw/controls.json`; code in
  `code/` (CPU only, < 1 min per script).
* `MEASURED (D2)` = `../D2_label_validity/RESULT.md` and `SEMANTICS.md`.
* `MEASURED (A&I)` = the route-following `RESULT.md` and `RESULT_A5.md`.
* `INHERITED` = another package, not re-run here.

**Data.** TRAIN = the 4,369 clips / 746,946 windows refcv7 read (stride 0.1 s, NOW at provider row t+7).
EVAL-DIAG = the route RESULT's 1,112-window grid, reproduced bit-exactly (digest `92e36a1a…`).

**What these numbers are.** Every number is a label-level measurement; no model was trained. The effect of each
proposed label on a model is a HYPOTHESIS until its validation rung runs.

**Binding constraints.**
* Labels may use anything; inference is vision-only plus the supplied nav and max-speed inputs.
* A goal input must not carry the situation classifier's output (TL / YIELD stay targets only).
* A supplied route on PhysicalAI is an ego-future oracle and optimistic by construction (stamped).
* The PI authorised the max-speed future-ego derivation (2026-09-16), provided the value lands in the containing
  posted-limit window.
* The v7 vocabulary is FROZEN (PI 2026-08-23), so no new token is proposed.

## 0. The top 5 changes, ranked by the measured size of what they fix

### 1. L1 — dense ego-derived tactical labels on every window with ≥ 4 s of future

**Definition.**
* Lateral is 3-way: LANE_KEEP / TURN_L / TURN_R. NUDGE is dropped, per D2.
* Longitudinal is 6 kinematic classes plus FOLLOW from the lead agent, over [NOW, NOW+6 s] with the documented
  ±1.5 m/s bar.

**What it fixes (measured).**
* Labels exist on only 23.4 % of windows (D2).
* NUDGE is 23.5 % of the lateral mass, and only 10–12 % of it is corroborated (D2).
* As applied, longitudinal accuracy decays 80.5 → 63.8 % across the band (D2).
* Nav is silent on 55/107 GT-turn windows (A5).

**Expected effect / bar.**
* Labelled windows: 23.4 % → **88.9 %**.
* GT-turn windows labelled: 36/107 → **107/107**.
* In-band side-correct: 65.4 % → **85.0 %**.
* TURN on GT-straight: **0.28 %**.
* Model bar: head side-correct on GT turns **0.393 → ≥ 0.80**. D2's cheapest arm is the 3-way vs 5-way lateral
  head.

### 2. L2 — time-localised ANNOUNCED nav, H = 6 s, plus args

**Definition.**
* Source: a rebuilt whole-recording table of announced turns.
* Builder fixes:
  * the `applied` key;
  * no 30 s cap;
  * turns hidden behind a curve are announced.

**What it fixes (measured).**
* The per-clip token wrongly commands a turn on 193/193 GT-straight windows of L/R clips.
* The graft's predicate is right on only 35.7 % of informative windows (gate 0.163, inert).
* A5: time-localising removes the straight damage of the nav filter (+0.326 → +0.009 m).

**Expected effect / bar.**
* Wrong-straight commands: 193 → **14** of 588.
* Predicate correctness: 35.7 % → **80.0 %**.
* On GT-turn windows: side-correct 45/107, silent 60/107. The silent half is L1's job.
* Model bar: A6 / T3 form.

### 3. L3 — max speed: a NON-ORACLE default plus mandatory reliance controls

**Definition.**
* A past-only posted-limit proxy (N2 or N3) with UNKNOWN on 45 % of windows.
* The per-window realised max is used only as a privileged diagnostic arm.

**What it fixes (measured).**
* The fed value is a future oracle in-band (R² 0.988, D2) and stale elsewhere.
* The human exceeds it on 11.9–13.4 % of windows.
* Enforcing it costs +0.086 m ADE (A&I).
* The all-zero row (45 % of navtest) was never trained.

**Leak and violations of the replacement.**
* Leak 44.5 % at the anchor → **0.9–3.5 %**.
* The human exceeds the proxy on 3.6–6.5 % of windows (vs 13.4 %).
* ⚠️ Needs a PI decision: the authorised future-ego derivation and R1 are both affected.

### 4. Goal negatives with evidence

**Definition.**
* Rebuild the cot-absence sidecar over the refcv8 blob md5 (the PI's 2026-09-16 ruling).
* Add TL probe negatives.
* Drop SPEED_BAND from the BCE.
* Detach EVADE / LANE_CHANGE from NUDGE.

**What it fixes (measured).** 5/22 tokens are masked, and RED has 403 negatives against 3,793 ignored records. The
ruling never reached the run (sidecar md5 mismatch).

**Expected effect.**
* Trainable tokens: 17 → 21.
* RED negatives: 403 → ~4,196 records.
* The cost travels with it: 79.8 % of visible-light clips become TL negatives (INHERITED).

### 5. VLM tokens re-timed and selectively propagated, plus class-weighted tactical CE

**Definition.**
* Apply each VLM token on the VLM's OWN window (anchor offset −2.9 s).
* Propagate RED over the stop episode.
* Take non-turn lateral and highway lateral from the VLM rather than from our geometry.

**What it fixes (measured).**
* The VLM's side peaks at a window starting 5.1–6.1 s, but it is applied at the 8.0 s anchor (D2).
* On NUDGE rows the VLM is right 57 % vs ours 34 %; on highway lateral 63 % vs 20 % (D2).
* The tactical terms are 0.29 % of the loss value.

**Expected effect.** RED windows: 14,021 → 20,387 on 223/353 clips. The rest stays anchor-only.

**Non-label levers this audit points at (§6).**
* Residual-prior dropout or a route-aware prior: 14/17 wrong-direction picks follow the prior (A&I).
* Ego dropout 0.5 → ≤ 0.1 (UNVERIFIED).
* Map/box items stay with A&I (F1 / F3 / NMS).

## 1. L1 — dense ego-derived tactical labels (replaces the ±2 s-of-anchor `lat_v7` / `lon_v7` rule)

### Definitions (literal thresholds; `code/d4_lib.py::dense_tactical`)

**Horizon:** [NOW, NOW+6 s], truncated to the available rows. The label is IGNORE (−100) when fewer than 40 future
rows (4 s) exist.

**Lateral, 3-way (D2's proposal, with the frozen 8-wide head and a loss mask).** Let Δψ be the heading change since
NOW and κ = |ω̄| / max(v̄, 1 m/s) on a 0.5 s moving average.

| label | rule |
|---|---|
| `TURN_L` / `TURN_R` (variant **a**, junction) | max \|Δψ\| ≥ **30°** AND min radius 1/κ_max ≤ **40 m**; side = sign of Δψ at the first ≥ 30° crossing |
| `TURN_L` / `TURN_R` (variant **b**, heading-only) | max \|Δψ\| ≥ 30°, as in D2's independent rule and the route GT. Covers the bends variant **a** misses. |
| `LANE_KEEP` | otherwise |

* The **NUDGE, LANE_CHANGE and ABORT_LC logits are masked** from the lateral CE (the existing `effective_mask`
  mechanism).
* The **in-band record is overridden** wherever it disagrees on TURN vs not-TURN. Builder NUDGE becomes LANE_KEEP;
  D2 found 89.7 % of NUDGE records are curves or shifts and only 21 of 1,028 are nudges.
* Variant **a vs b** and **3-way vs 5-way** are the two pre-registered arms of the v7-tiny rung. Both variants are
  measured below.

**Why no dense NUDGE / LANE_CHANGE: ego geometry cannot identify them.**
* D2 calls this an identifiability limit of any pose-only derivation.
* My two LC rules fail an independent referee, the Alpamayo lane-change TEXT (`EXECUTED`, 72 L / 29 R clips;
  `raw/d4_lanechange.json`):
  * **v1:** same side 48.6 % vs **opposite 50.0 %**, and it fires on 52.7 % of no-text clips;
  * **v2:** 5.9 % of text clips vs 4.9 % of no-text clips — no discrimination.
* LC therefore needs SAM3 lane lines. The proposal:
  * **Rule:** the path crosses a `lane` marking within 6 s, ends ≥ 2.5 m displaced, and its heading is restored
    within 5°.
  * **Bar:** dominant side correct ≥ 0.80 on the 101 text clips, firing on ≤ 0.10 of no-text clips, and a mirrored-y
    mutation must flip the side.
  * Until that bar passes, lateral offsets come from the VLM, anchor-only and re-timed (§4). The VLM beats our
    geometry on NUDGE rows (57 % vs 34 %) and on highway lateral (63 % vs 20 %) (D2).

**Longitudinal.** v0 = v at NOW. "First" means the event that happens first within the horizon.

| class | rule |
|---|---|
| `HOLD` | v0 ≤ 0.5 and vmax ≤ 0.5 m/s |
| `CREEP` | vmax ≤ 2.0 m/s |
| `BRAKE_TO` | vmin ≤ v0 − **1.5 m/s** (the documented v7 bar; D2 found the builder effectively used +1.0 for ACCELERATE), first |
| `ADAPT_SPEED_FOR_CURVE` | `BRAKE_TO` with κ_max ≥ 1/60 m. Only BRAKING curves, so ADAPT no longer hides cruise or accelerate on a curve (D2: 272 cruise / 486 accelerate / 192 brake underneath the builder's ADAPT). |
| `ACCELERATE` | vmax ≥ v0 + 1.5 m/s, first |
| `CRUISE` | otherwise |
| `FOLLOW` | needs a lead agent: a join3d centre with \|y\| ≤ 1.75 m, 0 < x ≤ 60 m, x/v0 ≤ 3.0 s, and no `ACCELERATE`. Where the join is absent, the target is the partial label {CRUISE, FOLLOW}. |

* Per-window labels remove D2's measured 80.5 → 63.8 % decay by construction: each window is labelled on its own
  future.
* The window is [NOW, NOW+6], matching the builder's actual [0, 6] span where D2 found 89.5 % agreement, rather
  than the declared [2, 6].

**Admissibility.** These are labels only (ego future, plus other agents for FOLLOW), which the PI allowed on
2026-08-03. They are never an input.

### Cross-checks

* **Analytic** (`raw/controls.json` C1, 7/7):
  * straight → LANE_KEEP / CRUISE;
  * R 15 m arcs → TURN_L / TURN_R;
  * 1.5 m/s² stop → BRAKE_TO;
  * +1 m/s² → ACCELERATE;
  * 3.5 m lane change and 1.2 m nudge → the v1 classes. Those classes are not shipped; see above.
* **Mutation (M1).** Moving the horizon 6 s earlier loses the TURN: RED.
* **Against the independent 6 s GT class** (route SPEC §3; TRAIN):
  * variant **a** labels TURN on **81.4 %** of GT-turn windows (66,499/81,703) and on **0.28 %** of GT-straight;
  * **98.2 %** of its misses are bends (≥ 30° at R > 40 m), which variant **b** recovers;
  * variant **b** misses only the 1.8 % of misses whose heading stays under 30° within the horizon, i.e. about 0.3 %
    of GT-turn windows.
* **Agreement with D2.** D2's LANE_KEEP misses are 109 wide-radius bends, and D2's independent turn count (609
  clips) exceeds the builder's 517.

### Measured shares (TRAIN, 663,935 labelled windows = 88.9 %)

* **lat (variant a):** TURN_L 6.37 % · TURN_R 6.19 % · LANE_KEEP 87.4 % (with the masked NUDGE/LC mass folded in).
* **lon:** CRUISE 39.1 % · ACCELERATE 29.5 % · BRAKE_TO 18.8 % · ADAPT 8.6 % · HOLD 2.1 % · CREEP 2.0 %.

### Expected effect, bar, rungs

**Label level.**
* Labelled rows: 23.4 % → 88.9 %.
* EVAL-DIAG GT-turn windows labelled: 36/107 → 107/107, of which 70 are TURN with the right side (variant **a**).
* In-band GT-turn windows side-correct: 65.4 % → **85.0 %**.

**Add class-weighted CE** (inverse-√frequency). LANE_KEEP is 64.7 % of today's lat labels and the tactical terms are
0.29 % of the loss value. ⚠️ Raising the 0.1 budget is a PI / MM decision.

**Model bar.**
* Head side-correct on GT turns: 0.393 → ≥ 0.80 on EVAL-DIAG, replicated on seed 1.
* GT-straight: no worse than 0.828.
* On the 55 nav-silent GT-turn windows (A5): turn dir-correct of the pick must improve with a CI excluding 0. That is
  the half no nav can reach.

**Rungs.**
1. Zero-GPU (done here).
2. v7-tiny: 3-way vs 5-way × variant a vs b, plus a deliberate regression (labels time-shifted +6 s, which must lose)
   and a same-flag replicate (`H-ESTIM-SEED-1`).
3. A short branch from `ckpt_50400`.

A frozen-trunk probe is cheaper, but it is a SLOT: no banked per-window feature bank of the 50,400 checkpoint is
known.

## 2. L2 — time-localised nav from ANNOUNCED entries (replaces `--nav-from-v7`'s per-clip token)

### Why "announced", and why not the driven-path rule

* A token built from the ego's own driven path (the 20 m / 2 m rule, NavSim's geometry) fires on **6.7 %** of all
  windows where no turn is announced (50,349): curves, lane changes, obstacle passes. On PhysicalAI those are the
  ego's own future decisions.
* A5 measured the cost of feeding such information: **54 %** of the hard rule's turn gain sat on turns that
  `nav_command` had suppressed (curves, obstacle passes), which a real nav would not announce.
* The path rule stays a NavSim-parity **diagnostic** only. It is never the training token.

### Definitions

* **Announced turn table, rebuilt by the builder over the WHOLE recording** (~140 s; MM). It replaces `nav_30s`.
  For every sustained segment, the entry is announced iff:
  * the builder's own `is_turn` is True (|Δyaw| ≥ 15°, R_arc ≤ 140 m, v_min ≤ 8 m/s, `s2_geom_emit_v7.py:156-176`,
    per D2);
  * AND it is not a PI 2026-08-29 suppressed turn.
* **Builder fixes**, each from D2 SEMANTICS:
  * read `applied`, not `suppressed` (row 6; 66 records);
  * **no 30 s cap** (row 7: 68 TURN tokens 30–35 s ahead have no entry);
  * a junction turn behind a leading curve IS announced (row 8: `seq[0]`-only hid 76).
  * Fields per entry: side, `s_start` / `s_end` (driven arc from the recording start), `t_start` / `t_end` (raw s).
* **Token per window**, A5/A6's rule restricted to announced entries:
  * take the first unfinished announced entry (`t_end > NOW`);
  * if it starts within **H = 6.0 s** or is under way → `NAV_TURN_x`;
  * otherwise `NAV_FOLLOW_ROAD`.
* **Args per window:** `next_side` ∈ {L, R, none}, `d_next = max(s_start − s_now, 0)`,
  `t_next = max(t_start − t_now, 0)`, `d_end`, `args_valid`.
  * With no announced turn before the end of the recording, `args_valid = 0`.
  * ⛔ That case is never encoded as d = 0: the existing reader validates by key presence (`refc_v3_train.py:2771`).
* **Normalisation and dropout.** Normalise with TRAIN statistics only. Train with **args dropout p 0.5**, because
  NavSim supplies no args.

### My label-level announced() in this package (an approximation from the record's flags)

`code/d4_nav_announced.py` reads `nav_30s` entries, the builder's `manoeuvre_sequence.is_turn`, and
`turn_suppression`. The suppression time is clipped to the band start when the turn began earlier, so it is matched
by side and interval (MEASURED on the records).

**Control K-A5.** A5's own all-entry rule reproduces A5's EVAL-DIAG table exactly: GT-turn 12/55/40, straight
4/572/12.

**A6's gate G1** (announced(entries[0]) == `nav_command` side): **97.1 %** train (4,438/4,572), 95.2 % eval
(140/147), and 100 % on the 66 suppressed records.
* All 134 train residuals are the two documented `nav_command` artefacts, not announcement-rule differences:
  * 68 tokens name a turn > 30 s ahead that has no entry;
  * 66 are "curve, not a turn" clips whose later junction turn `seq[0]` hid.
* ⚠️ This approximation therefore misses A6's ≥ 99 % bar by exactly those artefacts. A6 calls the builder's function
  directly. The design keeps the rebuilt table, so those clips announce the junction turn, which a real nav would
  do.

### Admissibility

* Same oracle class as refcv7's token (ego-future, `allow_oracle_nav=True` stamped; optimistic on PhysicalAI).
* It carries no situation-classifier output.
* It is restricted to what a turn-by-turn nav announces.
* A predicted goal (E15) stays the preferred route signal; this is the supplied-route arm.

### Measured (label level)

| token | EVAL-DIAG GT-turn: side correct / silent / opposite | GT-straight windows commanded to turn | TRAIN: predicate right on informative windows |
|---|---|---|---|
| refcv7 per-clip | 47 / 57 / 3 of 107 | **193** of 588 | 35.7 % (on 37.1 % of classified windows) |
| A5 nav_tl, all entries, H 6 s | 50 / 55 / 2 | 16 | 80.3 % (15.8 %) |
| **announced, H 6 s (design)** | **45 / 60 / 2** | **14** | **80.0 %** (14.7 %) |
| driven path, 20 m / 2 m (DIAGNOSTIC: leaks the ego's own manoeuvres) | 68 / 33 / 1 | 11 | — |

**Reading the table.** The announced token removes the straight damage and makes the compliance signal learnable. It
does NOT reach the 60 nav-silent GT-turn windows; those are L1's job.

**Cross-checks.**
* **Analytic.** The entry rule fires first at 6.6 s against 6.5 s analytic (R 15 m turn, `raw/controls.json`).
* **Two independent derivations.** Path rule vs entry rule agree on 92.6 % of the 373,313 windows where both are
  defined.
* **Mutation (must go RED).** Measuring the window's arc from the clip start instead of the anchor drops agreement to
  26.2 % on turn-ish windows.

### NavSim parity — a deploy item for EvalFlyWheel

* NavSim's `driving_command` uses route-centreline geometry (20 m / 2 m; INHERITED), so it fires on curves the
  announced token never trains on.
* The NavSim bridge should therefore derive the announced token from the nuPlan route's **junction lane connectors**
  plus distance and time, rather than pass `driving_command` raw. Both the route and the map are available in
  NavSim.
* Otherwise the model meets L/R on curves out of distribution: the proxy gap is 6.7 % of windows.

### Bar and rungs

* **Model bar: A6's.** T3a (soft ×10 term with the announced token) clears SPEC §5, and its derangement control
  fails (A&I runs it).
* **Training arm.** The v7-tiny rung needs, besides the announced arm:
  * a deliberate regression: the per-window nav deranged across clips (A5's T2c, which hurt by +0.089 m);
  * a same-flag replicate.
* **Then** a branch from 50,400. The `navc_gate` must leave 0.163.

## 3. L3 — max speed (replaces `--max-speed-input-v6`'s per-clip value)

### What each candidate leaks

**Test.** The same OOF test as the label file: 5-fold clip-grouped OLS; target = the window's own max v over
[NOW+2, NOW+6] s; n = 576,555 windows; v0 alone gives R² 0.8926. "Recovered" = the share of the future information
v0 lacks that the candidate supplies.

**Controls.**
* The label file reproduced: 75.3 % (known 75.4 %).
* Shuffled: −0.0005.
* The target itself: 1.0000.

**Source.** `raw/d4_measure.json` D/F and `raw/d4_vmax_nonoracle.json`.

| candidate (training value) | reads | recovered | human 0–6 s max exceeds it | class |
|---|---|---|---|---|
| refcv7: per-clip `v_hi`, 4-way | ego future at the anchor | 23.8 % (44.5 % at the anchor; in-band R² 0.988, D2) | 13.4 % (D2: 11.9 %) | **oracle**, stale |
| O1: per-window containing step of the [NOW, NOW+6] max, 8-step | ego future, every window | **57.0 %** | 0.7 % (only above the 130 km/h top step) | **oracle on every window** (Master Mind, D2) |
| O1′: the [NOW+2, NOW+6] max, 8-step | ego future | **71.7 %** | — | oracle |
| O2: O1 + one step w.p. 0.52 + UNKNOWN w.p. 0.45 (NavSim statistics, INHERITED) | ego future + noise | 5.9 % | 0 % | oracle, diluted. ⚠️ No longer the "containing" step. |
| **N0: no input** | — | 0 % | — | non-oracle. ⚠️ Conflicts with the PI's R1 (max-speed input + cap). |
| N1: past-20 s realised max, snapped up, 8-step | ego past | 6.2 % | 17.4 % | non-oracle; "slow ego ⇒ low limit" |
| **N2: N1 with an URBAN FLOOR of 50 km/h** | ego past + road-law default | **3.5 %** | **6.5 %** (> 2 m/s: 1.8 %) | **non-oracle (recommended)** |
| **N3: coarse 3-level from N2: urban ≤ 50 / rural 70–100 / motorway ≥ 120** | same | **1.9 %** | **4.4 %** (> 2 m/s: 1.1 %) | **non-oracle (recommended)**. Shares urban 70.7 / rural 24.5 / motorway 4.9 %. |
| N2u: N2 with UNKNOWN on 45 % of windows | same | 0.9 % | 3.6 % | non-oracle, deploy-matched |
| country one-hot | external | 0.2 % | — | non-oracle, uninformative |
| ⛔ `strata.road_class` | **defined by ego speed** (highway = ≥ 20 m/s sustained; `constraints.py:18-20`) | — | — | inadmissible as a proxy |

### Recommendation

**Default for refcv8: N2 or N3, plus UNKNOWN dropout p 0.45.** Neither is an oracle: each reads only the ego's past
(an ego-history statistic) plus a road-law floor. They are close to the deployed semantics of a loose posted limit
that is often unknown, and they train the all-zero row that is 45 % of navtest.
* **N3 also undoes the 4-way ladder's lumping**: one input currently spans 50–100 km/h, 21.7 % of clips (D2).
* ⚠️ **Admissibility of the stand-in.** At deployment the channel carries the MAP limit, not the past max. The
  past-20 s statistic is a training stand-in. Should it ever be fed at inference instead of a map value, it is an
  ego-history input beyond the 0.8 s `--ego-history`, which needs the PI to extend the 2026-09-02 reading.

**Oracle arms stay as privileged DIAGNOSTICS** (as NavSim's `R6_VMAXORACLE`), never as the training default:
* O1 is inside the 2026-09-16 authorisation (containing window), but it recovers 57 % of the future, so a planner
  trained on it learns to read its speed off the input.

**This changes a PI-authorised design, so it is a PI decision.**

### Mandatory reliance controls (every arm with a max-speed input, on EVAL-DIAG and NavSim)

* **(i) REMOVED:** `v_max_valid = 0`, the all-zero row, at eval.
* **(ii) SHUFFLED:** each window gets a donor clip's value (seeded derangement, A5's T2c construction).
* **Report** the four families: LONGITUDINAL speed MAE, signed along-track and **speed-to-limit gap on free-road
  windows**, plus ADE.
* **Reliance** = metric(fed) − metric(removed). An arm whose reliance on PhysicalAI exceeds the 2 × inference seed
  floor while its input is an oracle is **non-deployable by construction**.
* **For N2 / N3**, reliance should sit near the floor on PhysicalAI. Its value is obedience under a real map
  (NavSim `VMAXOFF` vs map).

### Cross-checks, expected effect, rungs

* **Mutation (must go RED).** Feed the per-clip value: "human exceeds" must return to 13.4 % (it does).
* **Expected.**
  * The ceiling the expert breaks falls from 13.4 % to 4.4–6.5 % of windows (N3 / N2).
  * The +0.086 m §26.1 cost (A&I) should shrink accordingly. That needs re-emulation on the banked fan: a SLOT for
    A&I.
* **Rungs.**
  1. Zero-GPU (done here).
  2. v7-tiny: refcv7 per-clip vs N3 + unknown vs O1 vs no input, each with the REMOVED and SHUFFLED evals.
  3. A branch.

## 4. VLM-only semantics: re-timing and what may be propagated beyond the anchor band

### Timing fix (MEASURED, D2 §3)

* The VLM's lateral side matches geometry best on the window starting at 6.1 s (70.1 %), and at 5.1 s (69.6 %), its
  stated `alpamayo.anchor_offset_s` −2.9 s relative to the 8.0 s anchor.
* Our label peaks at 8.1 s. 15 % of the lateral "conflicts" are pure timing.
* **Rule:** a VLM-provenance token is supervised on windows with |NOW − (t0 + anchor_offset_s)| ≤ 2 s, i.e.
  NOW ∈ [3.1, 7.1] raw for −2.9 s. It is never supervised on the geometry band.

### Which source wins (D2)

* Our geometry wins on TURN (90–94 % vs 40–49 % in conflicts) and on longitudinal (89.5 % vs 63.5 %).
* The VLM wins on NUDGE rows (57 % vs 34 %) and on highway lateral (63 % vs 20 %).
* ⇒ Lateral SIDE for turns and all longitudinal classes come from geometry (L1). Non-turn lateral offsets
  (`EVADE_IN_CORRIDOR`, `CORRIDOR_OFFSET`, `LANE_CHANGE_*`) come from the VLM, re-timed and anchor-only.
* Those tokens must be **detached from our NUDGE**: the builder requires `lat ∈ {NUDGE_L, NUDGE_R}` for them
  (`s2_geom_emit_v7.py:329-335`, D2 SEMANTICS §8), so today they inherit NUDGE's defect.

### Propagation

**General rule.** A text token is extended beyond its band only when a measurable kinematic or perception signature
ties it to an interval, and that signature holds on ≥ **0.90** of its anchor positives. The INHERITED precedent is
YIELD: 548/609 = 0.900.

| token | rule | status |
|---|---|---|
| `TRAFFIC_LIGHT_REACT_RED` | Over the stop episode (v ≤ 0.5 m/s) overlapping [t0−2, t0+6]: from the last row with v ≥ 5 m/s before it (≤ 8 s back) to the launch | **Applies on 223/353 RED clips; covers 20,387 windows** vs 14,021 in-band (union not computed). The other 130 clips stay anchor-only. SLOT (D2/D3): the stop must be at a junction (a SAM3 crosswalk or stop line ≤ 30 m ahead) and not in a queue (no stopped lead within 10 m). |
| `TRAFFIC_LIGHT_REACT_GREEN` | forward from the band until the junction is crossed, and only if the ego does not stop; never backward | proposed; validate as RED |
| `TRAFFIC_LIGHT_REACT_YELLOW`, colourless `TRAFFIC_LIGHT_REACT` | — | anchor-only (yellow phases last 3–6 s) |
| `YIELD`, `YIELD_FOR_TURN_x` | the contiguous deceleration / hold segment overlapping the band | candidate (signature 0.900, at the bar) |
| `EVADE_IN_CORRIDOR`, `CORRIDOR_OFFSET` | while the cited static agent (join3d, \|v\| < 0.5 m/s, lateral 0.5–4 m, x ≤ 40 m) is present | SLOT (needs a text→box association); anchor-only until then |
| `MERGE`, `TAKE_EXIT_x`, `OVERTAKE_VEHICLE`, `LANE_CHANGE_x`, `REACT_ON_ONCOMING`, `GAP_TARGET` | — | **anchor-only (VLM window)**: the text is untimed, and the geometric LC rules failed their referee |

**Admissibility (unchanged).** TL / YIELD are situation outputs: auxiliary TARGETS only, never goal or selection
inputs.

## 5. Goal-token negatives (replaces `negatives = "measured"` without a sidecar)

1. **The PI ruling (2026-09-16), applied as written.**
   * Rebuild `cot_absence_negative_*.json.gz` over the refcv8 blob's md5. The one on disk is bound to `fa89ea55…`;
     refcv7 read `b45377a1…`.
   * Pass `--tac-goal-negatives cot-absence-negative --cot-negative-sidecar`. The code path exists:
     `v7_labels.py:970-1047`, `refc_v3_train.py:8483-8507`.
   * The cost travels with it: 692/867 = 79.8 % of visible-light clips become TL negatives (INHERITED; an exposure
     ceiling, not an error count).
2. **Evidence-first negatives**, stated per token in `config.json`:
   * **Probe negatives.** 998 clips were asked the traffic-light question, and ~131 answered "no light visible". Those
     are TRUE negatives for every TL token (INHERITED 998 / 867).
   * **Kinematically entailed negatives**, only where the entailment holds on ≥ 0.95 of positives. YIELD's 0.900
     fails that bar, so YIELD stays on the PI ruling.
3. **Drop `SPEED_BAND` from the BCE.** It is positive on 4,572/4,572 records: a band argument, not a decision. E8 and
   the trajectory already supervise speed.
4. **Make the geometry goal tokens dense from L1** (D2: TURN 87/85 %, STOP_POINT 291/291 corroborated):
   * `TURN_x` ⇔ the dense TURN_x;
   * `FOLLOW_LANE` ⇔ not TURN and no stop;
   * `STOP_POINT` ⇔ vmin ≤ 0.5 m/s within 6 s.
   `EVADE_IN_CORRIDOR` and `LANE_CHANGE_x` leave the NUDGE entailment (§4).

**Expected.**
* Trainable tokens: 17 → 21.
* RED negatives: 403 → ~4,196 records (cot-absence), or ~534 with probes only.
* Per-class reports against the majority control, never pooled.

**Rung.** A zero-GPU `goal_supervision_census` on the rebuilt sidecar, then v7-tiny.

## 6. Non-label levers (owners: Arch / MM)

* **Residual prior.** 14/17 wrong-direction turn picks share its side (A&I). Two options: prior dropout p ≈ 0.3, or
  a route-aware prior that clamps κ0 toward the announced nav side.
  * Zero-GPU check on the banked fans: remove the prior's contribution where it disagrees with the announced token,
    and re-score.
* **Ego dropout 0.5.** Half the rows train without the v0 that deployment always has.
  * Proposal: ≤ 0.1; keep the withheld bank as a diagnostic.
  * UNVERIFIED as a lever; it needs a v7-tiny pair.
* **Box / agent store scope.** x ≤ 61 m leaves 84 % of joined agents unsupervised, so there is no box target at
  highway headways. The distance-keeping instrument is UNAVAILABLE and must exist before this can be judged.

## 7. Implementation sketch (owner: DataFlyWheel builder + trainer reader)

**Builder.**
* Rebuild the announced turn table per recording, applying the three builder fixes of §2.
* Write per-window sidecars keyed by (sid, t):

| sidecar | carries |
|---|---|
| `refcv8_nav_window_{split}.npz` | token, `next_side`, `d_next`, `t_next`, `d_end`, `args_valid` |
| `refcv8_tac_dense_{split}.npz` | lat 3-way (variants a and b), lon, FOLLOW-available |
| `refcv8_vmax_window_{split}.npz` | N2 raw m/s, N3 class, the O1 diagnostic, valid |

* Each sidecar is ~12 MB (746,946 × < 16 B). It carries inside the file:
  * the label, manifest and clock md5s;
  * the rule literals.

**Trainer.**
* `V3Dataset.__getitem__` reads each sidecar by (sid, t) and refuses on a missing key.
* The lateral CE masks the NUDGE/LC/ABORT logits.
* The VLM-provenance goal cells use the re-timed band.

**Launch gate.**
* Every sidecar's window count must equal the dataset's (746,946 / 23,772).
* A mutation test per sidecar: shifting t by +60 must drop its agreement with the GT class.
* The G3 clock check is unchanged.

## 8. Remaining slots

**D1.**
* Per-token window coverage under the re-timed VLM band.
* The share of L/R-clip windows inside their turn interval.

**D2.**
* The per-window conflict count between the record's lat class and the dense TURN / not-TURN decision (the
  override rule in §1).
* The RED stop-episode validity, shared with D3.

**A&I.**
* A6 (announced nav at inference).
* Re-emulating §26.1 with N2 / N3 on the banked fan.
* Re-scoring the residual prior on the banked fans.

**EvalFlyWheel.**
* An announced-junction nav token from the nuPlan route for the NavSim bridge.
* The removed / shuffled max-speed arms on NavSim.
