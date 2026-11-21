# RESULT_V8F_LAUNCH_INTENT: O's launch-sample along-track weight on DAG-OACC's recipe, aimed at the crawl rows. Outcome **HARMS** (arm A, W [[W_A]], not adopted)

Claim row `H-V8F-LAUNCH-INTENT-1`. Pre-registration `Project Steering/PREREG_V8F_LAUNCH_INTENT.md`, promoted at `77234276` ([[gtime('77234276')]]; sha256 `469bac1f…`), with **Amendment 1** (`A1-2026-10-28`, the K-BUILD data-module clause), landed at `e28516e8` ([[gtime('e28516e8')]]; sha256 of the amended text `6ae79f2f…`) before any row was read. Read 2026-10-28 by the pre-registered reader of record `launch_intent_read.py` (sha256 `fb3d2d7d…`, the Amendment 1 reader): all **[[ncontrols]] reader controls PASS**, wall [[wall_s]] s. Output `raw/2026-10-28-launch-intent-read/read_A1/LAUNCH_INTENT_READ.json` (sha256 `e451e1a6…`), banked at `d7dc9f11` ([[gtime('d7dc9f11')]]).

* **The word, as registered.** **HARMS.** Arm A is not adopted. It failed its pre-registered bar: the decisive R-MOVE is false at both seeds, and the guard R-ADE fails at both seeds. Nothing below softens that.
* **Tier.** [[tier1]] The R-T0 points and the slow-draw cells are at T0: [[tier0]]
* **Estimator.** `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed and per pair, paired, on the same windows and matched cells (JSON `estimator`). Intervals are 95 % (INHERITED: the convention of `RESULT_V8F_DAG_ORC.md`; the JSON stores the bounds only). It answers *"would another draw of EPISODES say this?"*. The question *"would another training run say this?"* is answered by the replicate pair at each seed: a lever effect is credited at a seed only when both pairs show it (`H-ESTIM-SEED-1`). The planner is deterministic, so no inference-seed question arises. A T0 point carries no interval: the JSON stores the point only.
* **Arms** (box 2, launch commit `d8c9f7bf`, [[gtime('d8c9f7bf')]]; four gates PASS, four run records PASS, four K-SHARE PASS). A_s = DAG-OACC_s's training argv verbatim plus `--launch-intent-w` [[W_A]] (one third of the along-track weight on the launch samples). A_rep,s = REP_s's argv (trainer seed 10 / 11) plus the same flag. CTRL_s = DAG-OACC_s and REP_s are banked, so no control was retrained. Checkpoint sha256 prefixes: A `deb8f28b…` (s0) / `abe0fd24…` (s1); A_rep `b86f7d43…` / `2900ff73…`. **Arm B (W [[W_B]]) was not trained and is not read.**
* **Coverage.** The four families at T1-kin are in section 3. **The T0 four families are NOT RUN** (section 3.6). **M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows are NOT BUILT** by the reader (section 3.7). They are work items, and the outcome above does not use them.

**Summary** (each value is in the section named; evidence class and tier as stated there).

* **HARMS (section 1).** Controls pass. R-ADE fails at s0 in the replicate pair (ADE6, in m: [[cellt('xy','s0','rep','ADE6')]]) and at s1 in the main pair (ADE6, in m: [[cellt('xy','s1','main','ADE6')]]). R-TLAT, R-TLON, R-T0 and R-NEW hold at both seeds. R-KEEP and R-NOBASE hold at both seeds.
* **R-MOVE is false at both seeds (section 2.3).** [[n_crawl_better]] of the [[n_crawl_cells]] crawl cells are separated-better and [[n_crawl_clear]] clears its bar: L2 curvature in the replicate pair at s0, [[cellt('xy','s0','rep','L2 curvature MAE (GT>=5.2)')]] against a bar of [[bar('L2 curvature MAE (GT>=5.2)')]]. Its main-pair partner reads [[cellt('xy','s0','main','L2 curvature MAE (GT>=5.2)')]]. No crawl row moves by its bar in both pairs at either seed.
* **R-CRAWL is false at both seeds (section 2.4).** All [[n_xp_worse]] X − P10 crawl cells (4 rows × 2 pairs × 2 seeds) are separated-worse.
* **Reading, decides nothing (section 5).** At each seed the separated ADE6 cells flip sign between the main pair and the replicate pair.
* **Next (section 6).** Committed: B (W [[W_B]], 4 box-2 runs) with its own PI go. NEXT-LAUNCH (PREREG section 11) routes by M-LAUNCH, which is built separately; its slot is open in section 6.2.

## 1. The outcome, as the table decides it

Evidence class MEASURED (ours). Tier and estimator as in the header: T1-kin unless a cell says T0; paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Source: `rows.A.<seed>.pairs.<pair>` and `rows.A.<seed>` in the JSON. X is A (main pair) or A_rep (replicate pair); Y is the pair's control, CTRL = DAG-OACC or REP.

[[t_outcome()]]

**HARMS.** The table's condition, verbatim from the JSON: *"[[D['OUTCOME']['condition']]]"*. A guard fails at a seed when it fails in either pair (PREREG section 10), so R-ADE fails at s0 and at s1. REFUSED is decided first and HARMS second; the controls pass, so the R-MOVE and R-CRAWL flags do not change the word. Committed next, verbatim from the JSON: *"[[D['OUTCOME']['committed_next']]]"* Cascade: `cascade.final` reads [[D['cascade']['final']]], and `cascade.b_status` reads *"[[D['cascade']['b_status']]]"*.

⚠ **Reported as written, beside the word.** R-MOVE is false at both seeds. R-CRAWL is false at both seeds. The registered guard failure is R-ADE; R-TLAT, R-TLON, R-T0 and R-NEW hold, and R-KEEP and R-NOBASE hold.

The word is the reader's output (`matching_rows_in_table_order`: one row, HARMS). `check_result.py` recomputes R-ADE, R-MOVE, R-CRAWL, R-NEW, R-NOBASE, R-T0 and R-KEEP per pair and per seed from the JSON's cells, floors and points, derives the word from them, and finds every flag equal to the reader's.

## 2. The decisive rows

Tier T1-kin unless a heading says T0. Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, per pair. Evidence class MEASURED (ours): every cell is a copy of `read_a.<contrast>.<seed>.<row>` in the JSON. Cells read point [95 % CI] and a flag: B = separated-better, W = separated-worse, – = not separated. P10 is the kinematic floor. The `n` of a cell is windows / episodes.

### 2.1 The floors and the bar (`floors`)

`bar_r = 2 × max(F_r, RF0_r)`, one value per row, applied at both seeds and in both pairs, inclusive. F is [[un(FL['F']['ADE6'])]] m on ADE6 and 0 on every other row. RF0 is the frozen same-seed cross-build floor (`P0_RF0.json`, sha256 `1608d70b…`). RF is the larger of two REP − DAG-OACC draws, reported beside every row. It is not a bar term (PREREG section 6).

[[t_floors()]]

### 2.2 ADE6: the guard R-ADE and the adoption row R-KEEP (T1-kin)

[[t_ade()]]

R-ADE needs X − Y not separated-worse. It fails in the replicate pair at s0 and in the main pair at s1. R-KEEP reads the point against +[[un(F_HALF)]] m and holds in all four pairs. R-ADE and R-KEEP read one contrast, as the PREREG discloses.

### 2.3 R-MOVE: the crawl rows, X − Y against the bar (T1-kin)

R-MOVE needs, in each pair, every crawl row separated-better with an improvement of at least the bar. Improvement is −delta.

[[t_move()]]

[[n_crawl_better]] of the [[n_crawl_cells]] cells are separated-better and [[n_crawl_clear]] clears its bar. No pair clears all four rows, and no main-pair cell clears.

### 2.4 R-CRAWL: the crawl rows, X − P10 (T1-kin)

The share closed is (Y − P10 − (X − P10)) / (Y − P10) from the point deltas; a negative share means the gap to P10 widens. MEASURED (arithmetic on the banked JSON).

[[t_crawl()]]

All [[n_xp_worse]] X − P10 crawl cells are separated-worse. So are the Y − P10 cells in every pair: that is how the crawl set C was fixed.

### 2.5 R-NEW and R-NOBASE (T1-kin; domain D is every row not separated-worse for Y − P10, no exempt rows)

R-NEW fires on a row of D when three things hold: (1) X − P10 is separated-worse, (2) Y − P10 is not (the row is in D), and (3) X − Y is separated-worse with a worsening of at least the bar. R-NOBASE reads (1) and (3).

[[t_new()]]

**R-NEW and R-NOBASE hold because condition (1) fails, not because condition (3) does.** In the main pair at s1, four rows meet (3): speed MAE [[cellt('xy','s1','main','speed MAE')]], accel MAE [[cellt('xy','s1','main','accel MAE')]], cross-track MAE [[cellt('xy','s1','main','cross-track MAE')]] and L1 lateral-accel MAE [[cellt('xy','s1','main','L1 lateral-accel MAE')]]. None of the four has X − P10 separated-worse. In the replicate pair at s0, along-track MAE meets (3): [[cellt('xy','s0','rep','along-track MAE')]]. The registered predicate is applied as written.

### 2.6 R-T0, R-KEEP, R-TLAT and R-TLON

R-T0 is at tier T0 (a point, no interval in the JSON). R-KEEP is T1-kin.

[[t_t0keep()]]

R-TLAT and R-TLON are booleans in the JSON and read *held* in all four pairs. The macro-F1 values behind them are not carried (UNVERIFIED as numbers).

## 3. The four families (T1-kin; per family, per seed, per pair, never pooled)

Tier: [[tier1]] Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, per pair. Evidence class MEASURED (ours); source `four_families.A.<seed>.<pair>.<family>.rows`. Each family block carries the tier and estimator strings, and its leaves equal the `read_a` cells cell for cell ([[n_fam_cells]] cells, asserted by `check_result.py`). Each table shows X − Y, the one-lever contrast of the pair. The X − P10 and Y − P10 cells are in section 3.8. STRATEGIC is **not paused** for v8F (PI ruling 2026-10-17).

### 3.1 Separated cells per family, per seed, per pair (B / W / not separated; X − Y)

[[t_famcount()]]

### 3.2 LONGITUDINAL (read first: the lever acts on along-track pressure)

[[t_family('LONGITUDINAL')]]

Not built for this family: M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows (section 3.7).

### 3.3 LATERAL

[[t_family('LATERAL')]]

### 3.4 TACTICAL

[[t_family('TACTICAL')]]

TAC-LAT and TAC-LON are the plan-derived agreements (higher is better). The guards R-TLAT and R-TLON are in section 2.6. T1-kin goal-head F1 and head accuracy are **not computable** here: the paired tool reads eval dumps, not tick dumps (JSON note).

### 3.5 STRATEGIC

[[t_family('STRATEGIC')]]

S-POS@30 / @50 / @80 (lower is better). The X − Y cell is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows. Goal and route-setting quality is not a Read-A row; whether another tool computes it for these arms is UNVERIFIED here.

### 3.6 T0 (box 2's standard eval): the four families are NOT RUN

[[t_notrun()]]

`not_built.t0_four_families` reads: *"[[D['not_built']['t0_four_families']]]"*. The only T0 numbers in this read are the R-T0 points (section 2.6) and the slow-draw cells (section 4.2). The inputs exist: each run's box-2 T0 eval JSON is in the run record (section 10), and its `.npz` is in the pulled run directory on E: (listed in the run's `SHA256SUMS.txt`). **Nothing is claimed for T0 LONGITUDINAL, LATERAL, TACTICAL or STRATEGIC.**

### 3.7 Registered rows that are NOT BUILT or not computable

Verbatim from the JSON `not_built`:

* `M-LAUNCH`: *"[[D['not_built']['M-LAUNCH']]]"*
* `anti_echo_block`: *"[[D['not_built']['anti_echo_block']]]"*
* `M2_and_L2_launch_subset`: *"[[D['not_built']['M2_and_L2_launch_subset']]]"*

By family: LONGITUDINAL carries all three (the mechanism row, the anti-echo block, the launch-subset rows). TACTICAL: goal-head F1 and head accuracy are not computable at T1-kin, and the R-TLAT / R-TLON macro-F1 values are not in the JSON. LATERAL and STRATEGIC: every registered Read-A row is computed.

### 3.8 Against P10: X − P10 and Y − P10, every row (T1-kin)

For the three S-POS rows the P10 contrast is the run's own S − CTRV and the JSON labels its `n` *"ticks>=1, unpaired vs CTRV (DAG_READ.json)"*: those cells are not paired across runs.

**s0**

[[t_p10('s0')]]

**s1**

[[t_p10('s1')]]

## 4. Beside, deciding nothing

### 4.1 RF beside every row

The RF column of section 2.1 is this block (`beside.A.RF_beside_every_row`): RF, RF0 and the bar for each of the [[len(ALL_ROWS)]] rows. RF is not a bar term.

### 4.2 The slow-draw flag (T0; the five speed cells, per-window paired difference, episode-cluster bootstrap, B = 2,000, seed 0)

The rule (PREREG section 7): an arm is a **slow-T draw** when all five cells are separated from the pair's control and each shift exceeds [[D['beside']['A']['slow_draw']['main:A-CTRL:s0']['threshold_m_s']]] m/s in magnitude. The flag decides nothing.

[[t_slow()]]

No arm is labelled. The largest shift in magnitude over the 20 cells is [[slow_stats()[0]]] m/s, and [[slow_stats()[1]]] of the [[slow_stats()[2]]] cell intervals exclude 0.

### 4.3 The lever reached the data: K-SHARE and K-MASK (MEASURED, box-2 chain log)

The realised weight share is recomputed by the run record from `launch_n`, `launch_eligible` and the flag. The target is 1/3 ± 0.02. Rows equal eligible rows in all four runs.

[[t_share()]]

The four shares equal the PREREG's ESTIMATED shares for trainer seeds 0 / 1 / 10 / 11 ([[est_A]], simulated from the sampler stream) at four decimals.

### 4.4 The launch gates (MEASURED, box-2 chain log)

Each run went through its own launch gate before training: G-DVB, the swap probe, the acc probe, the launch probe, and the one-lever check against its pair control's banked argv. `head_o grad diff` is the gradient the launch term puts on O's head; it is non-zero in all four.

[[t_gate()]]

## 5. Reading (decides nothing)

**Label: MEASURED facts from the tables above, and one mechanism statement. The outcome is HARMS, as registered. Nothing in this section changes a flag, a bar or the word.**

**Fact 1. At each seed, the separated ADE6 cells flip sign between the main pair and the replicate pair.**

[[t_flip()]]

All four cells are separated. The lever is the same in both pairs (W [[W_A]]). The two pairs of a seed differ in the trainer seed and the control. The reader's K-PAIR control passes for both. Beyond ADE6, the rows whose separated X − Y cells have opposite signs between the two pairs of the same seed are: at s0, [[flips_text('s0')]]; at s1, [[flips_text('s1')]]. At s1 that includes accel MAE: main pair [[cellt('xy','s1','main','accel MAE')]], replicate pair [[cellt('xy','s1','rep','accel MAE')]].

**Fact 2. No crawl row moved by its bar in both pairs at any seed.** The crawl cells that are separated-better:

[[t_crawlcells()]]

The only cell that clears its bar is in the replicate pair at s0. Its main-pair partner is [[cellt('xy','s0','main','L2 curvature MAE (GT>=5.2)')]], not separated. Every main-pair crawl cell is below its bar.

**Why this is the pattern the pair design exists to expose.** The episode-cluster bootstrap resamples EPISODES with the models held fixed (`H-ESTIM-SEED-1`). It answers *"would another draw of EPISODES say this?"*. It cannot answer *"would another training run say this?"*. The replicate pair asks that second question: the same lever on the same recipe, another trainer seed. Here the answer on ADE6 is the opposite sign at each seed. A read on either pair alone would have called the lever better on ADE6 at one seed and worse at the other, depending on which pair was run. Each separated cell is right about its episodes and silent about its training run. That is why a separated cell from one pair is necessary and not sufficient, and why the registered predicates hold only when they hold in both pairs.

The sizes are small. The four ADE6 points lie between [[un(min(abs(C('xy',s,p,'ADE6')['delta']) for s in SEEDS for p in PAIRS))]] and [[un(max(abs(C('xy',s,p,'ADE6')['delta']) for s in SEEDS for p in PAIRS))]] m, all inside ±[[un(F_HALF)]] m, which is why R-KEEP holds in all four. The registered table does not use this reading: R-ADE counts as failed at a seed when it fails in either pair, so the word stays HARMS.

## 6. Next (Rule Zero)

### 6.1 Committed next: B, with its own PI go

PREREG section 10, verbatim from the JSON: *"[[D['OUTCOME']['committed_next']]]"* B is the same lever at a lower dose: W [[W_B]], a launch-sample weight share of one quarter (K-SHARE band 1/4 ± 0.02; ESTIMATED realised shares [[est_B]] at trainer seeds 0 / 1 / 10 / 11, PREREG section 3). It is four box-2 runs (B_0, B_1, B_rep,0, B_rep,1) in the same pair design, read under the same table with its own seeds, and never pooled with A. B's HARMS routes to NEXT-LAUNCH (section 6.2).

**Cost (ESTIMATED from a MEASURED span).** The A chain ran [[FIRST]] to [[LAST]] on box 2 ([[SPAN_MIN]] min, [[round(SPAN_MIN/60.0,1)]] h; MEASURED, `chain_v8f_launchintent.out`), against the PREREG's ESTIMATED [[prereg_est_h]] h (section 13). B is about the same.

**Blocked on:** the PI's go for box-2 time. The A chain's closing line says *"B is not queued"*. What unblocks it: the go, and a B chain on the pattern of the A chain (0 GPU to build; UNVERIFIED that nothing else is needed).

### 6.2 NEXT-LAUNCH (PREREG section 11) and the M-LAUNCH slot

LAUNCH-INTENT is the last lever `PREREG_V8F_DAG_ORC` section 8 ranks. If the crawl does not move, the routing variable is M-LAUNCH: X − Y of the mean (v_own − v_GT) at ticks 1–2 in the GT-launch windows, from the tick dumps, in each pair, at T1-kin. It is read on A, or on B when B is the arm that sends the programme here. *Moved* means separated toward 0 in both pairs at both seeds.

| M-LAUNCH | what it says | committed next |
|---|---|---|
| moved | the plan's launch speed was repaired and the crawl rows did not follow: the crawl is not in O's plan profile | **LOOP-STATE** (zero training, dev-box CPU): decompose the crawl by plan index in X's tick dumps; its read names the loop term |
| not moved | the lever did not reach O's launch behaviour at this dose | **LAUNCH-INTENT-HI**: the same lever at a launch-sample share of one half (W_HI [[W_HI]]), under the section 10 table, with its own PI go |

This reader does not build M-LAUNCH (`not_built.M-LAUNCH`: NOT BUILT). It is built separately, and its result goes in the slot below.

> **OPEN SLOT @@MLAUNCH@@** (until M-LAUNCH lands this reads NOT BUILT). The slot takes four things: the M-LAUNCH value X − Y per pair at both seeds with its interval (T1-kin, paired episode-cluster bootstrap, B = 2,000, per pair); the PREREG section 11 label, *moved* or *not moved*; the route it gives, LOOP-STATE or LAUNCH-INTENT-HI; and the artifact path with its sha256.

### 6.3 What is blocked, and on what (Rule Zero: every remaining lever)

| lever | status | blocker, and what unblocks it |
|---|---|---|
| B (W [[W_B]], 4 runs) | committed next | **the PI's go** for box-2 time; then a B chain |
| M-LAUNCH (the section 11 routing variable) | NOT BUILT in the reader; being built separately | its build; it reads the tick dumps, which are single-copy on F: (section 8, item 6) |
| LOOP-STATE (zero training) | routed from M-LAUNCH *moved* | M-LAUNCH |
| LAUNCH-INTENT-HI (W [[W_HI]]) | routed from M-LAUNCH *not moved* | M-LAUNCH, and its own PI go for box 2 |
| T0 four families | NOT RUN | a build item: `orc_families.py` pointed at the four arms and their controls; inputs: each run's eval JSON (repo) and `.npz` (E:) |
| anti-echo block; M2 / L2 launch subset | NOT BUILT | build items for the reader |

If both NEXT-LAUNCH branches are exhausted, the programme has no registered lever for the crawl rows on DAG-OACC (PREREG section 11). The blockers then are the ones that section names: route extent and nav timing (eliminated as the lever by N0, INHERITED), the state-channel levers (DAgger iteration 2, INHERITED), and the zero-training post-map (it moves the crawl but failed KEEP at s0, INHERITED). What would unblock: a new diagnosis of the loop term, or a PI decision on an architecture change that gives O a launch-intent input.

## 7. What this read cannot claim

* **It is not closed loop.** T1-kin is a self-action open loop: the lead follows its logged future and the cameras are logged. T0 is teacher-forced and is not driving performance.
* **One training run per arm, pair and seed.** The two pairs of a seed and the two seeds are four training draws of the lever. They disagree on ADE6 (section 5). No row here is an effect of the lever on its own (`H-ESTIM-SEED-1`).
* **It attributes to a weighting, not to a mechanism.** The lever is one label-defined weight on one term. The read does not say which of the 12 knots or which launch windows carried any cell.
* **The mask uses future GT speed.** That is admissible for labels. A result here says nothing about a deployable launch detector. Nothing new enters at inference.
* **The share is a training-weight fact.** K-SHARE and K-MASK are measured on the TRAIN windows of each run. They are not an eval fact.
* **B and HI are not read.** Nothing here says what a lower or a higher dose does.
* **The T0 four families, M-LAUNCH, the anti-echo block and the M2 / L2 launch-subset rows are not computed** (sections 3.6 and 3.7). The T-head macro-F1 behind R-TLAT / R-TLON is not in the JSON.

## 8. Disclosures

1. **Amendment 1 (`A1-2026-10-28`), before the read.**
   * **What was wrong.** The PREREG's K-BUILD row states a rule. Against `aeb54468`, the commit DAG-OACC trained at, each of the three files is byte-identical, or every changed hunk sits behind a default-off flag whose bit-identity test passes. The row then adds a fact: `v8f_mini_data.py` is byte-identical. That fact had been measured against `d7208328`, not against `aeb54468`. The reader coded it as a rule, so it refused the launch commit on the data module.
   * **What changed.** K-BUILD is read by its registered rule, with the data module pinned to DAG-ORC r2's build (sha256 `c117e1cf…`) and its twelve flag-gated hunks listed with a PASSing bit-identity test. The reader changed in that clause alone. No bar, row, outcome, predicate or other control changed.
   * **What had been seen: only controls.** These were the T1-kin chain's dry run (where K-BUILD REFUSES), A_0's run record and K-SHARE (both PASS), and the four launch-gate lines. The statement, and its time "written [[written_time]]", are the Amendment's own (INHERITED).
   * **Timing.** It landed at `e28516e8` ([[gtime('e28516e8')]]). The read of record started after that (first console stamp [[read_first_stamp]]). A_0's three T1-kin steps ran from [[a0_first_start]] to [[a0_last_end]], between the writing and the landing. Each fresh-T0 log prints one ADE6 summary line. Whether anyone read those lines in that window is not recorded (UNVERIFIED).
   * **Both readers are on record.** The launch-time reader (sha256 `5e950dcf…`) and the PREREG as promoted (sha256 `469bac1f…`) are what the launch-gate tokens bind. The reader of record (sha256 `fb3d2d7d…`) ran with the amended PREREG (sha256 `6ae79f2f…`).
   * **The T1-kin chain's own read with the launch-time reader REFUSED** at [[refused_time]] (chain log), with one failed control, K-BUILD, and wrote nothing: *"[[refused_msg]]"*. That read is banked as `lit1_logs/launch_intent_read_REFUSED.out`. The read of record ran the amended reader on the same manifest (sha256 `98d2cd3d…`).
   * **Effect on the pairs.** The replicate pair is unaffected: A_rep and REP have the same data module. The main pair carries DAG-ORC's flag-gated data hunks on the A side, as DAG-ORC's own main pair did.
2. **The T1-kin chain was stopped once, on its own RAM guard.**
   * **What happened.** It was stopped at 15:11, still waiting on the guard, and before any rollout. The stopped launch's last line, [[prev_last_stamp]], reads *"[[ram_msg]]"*, and its log holds no `start` line for a rollout. It was relaunched with a 2,048 MB guard.
   * **Where the numbers come from.** The 15:11 stop time and the 2,048 MB value are from the Master Mind's brief (INHERITED). The relaunch's log does not echo the guard value.
   * **What it did not change.** The relaunch started at [[relaunch_start]] and its first rollout at [[a0_first_start]]. All eight rollouts in the relaunch's log show attempt 1. The template proofs (rollout argv equal to the banked argv, token for token) pass in both launches. The guard gates dev-box memory. It is not a measurement gate.
   * **Logs.** The stopped launch: `lit1_logs/chain.out.prev_1028151251`. The relaunch: `lit1_logs/chain.out`.
3. **Pin slip (dated note).**
   * **The slip.** `PINS_FILLED.txt` (in `raw/2026-10-28-launch-intent-chain/`) says the chain pins were filled [[pins_filled_time]]. They were filled ~13:50, before commit `d8c9f7bf` ([[gtime('d8c9f7bf')]]) and after the PREREG was promoted at `77234276` ([[gtime('77234276')]]).
   * **Where the numbers come from.** The ~13:50 is from the Master Mind's brief (INHERITED). The bracket between the two commit times is MEASURED from the commits. The box-2 pins line is stamped [[FIRST]].
   * **The pins are the promoted ones.** The PREREG blob and the reader blob at `77234276` hash to the `469bac1f…` and `5e950dcf…` that the pins carry (re-measured here).
4. **A label inside the reader.** The JSON's `provenance.prereg` reads *"[[D['provenance']['prereg']]]"*. That is a string inside the reader. The reader of record was run with `--prereg-sha` equal to the amended text's sha256 `6ae79f2f…` (`read_A1/run_read_A1.sh`, `read_A1.out`).
5. **The bar and arms were redesigned before promotion.** The PREREG's disclosure, verbatim: *"The bar and arms were changed by the Master Mind before promotion, after DAG-ORC's floors and the REP s0 diagnosis were known, and before any LAUNCH-INTENT number existed."* The bar is 2 × max(F, RF0), with RF beside.
6. **Single copies.** The per-episode dumps and tick arrays are under `F:/TanitAD-scratch/launch_intent_t1_out/` (the chain's `--out-dir` and `--dump-ticks`). The repo listing at `d7dc9f11` holds the rollout JSONs under `t1/`, the fresh-T0 evals under `t0fresh/` and each run's record, and no tick arrays. M-LAUNCH reads tick dumps, so it needs F:. The checkpoints are on E: only.
7. **Disclosed coupling.** DAG-ORC and LAUNCH-INTENT share the REP_s runs. In DAG-ORC they set the floor RF; here they are the control of the replicate pair, and RF is not a bar term. Neither read's result changes the other's bar.
8. **What `check_result.py` is and is not.** It re-finds every quoted number in the banked JSON, logs and PREREG, recomputes the decision from the cells, and checks the document for the banned words. It does **not** recompute the cells from the per-window dumps or re-run the bootstrap (the dumps are a single copy on F:). A defect inside the estimator or the reader's heavy layer would not show there.

## 9. Run record

Evidence class MEASURED (ours): the box-2 chain log and the T1-kin chain logs, as banked.

**Box-2 chain** (`chain_v8f_launchintent.out`; tip `d8c9f7bf`). The chain waited for the DAG-ORC r2 chain's DONE marker, then ran: refresh, four launch gates ([[gate_span]]), then train and T0 eval in the order A_0, A_1, A_rep,0, A_rep,1, with a pull and a run record after each. First stamped line [[FIRST]], last [[LAST]]: [[SPAN_MIN]] min, [[round(SPAN_MIN/60.0,1)]] h. REP-LICENCE conditions 1 and 2 passed at both seeds in the chain; the reader checks conditions 3 and 4 (controls `K-REP-LICENCE:s0` and `K-REP-LICENCE:s1`, both PASS in the JSON). The closing line: *"[[zzdone]]"*

[[t_jobs()]]

**T1-kin chain** (`lit1_chain`, dev box CPU, one process at a time). Twelve steps (four runs × causal rollout, logged rollout, fresh-T0 eval); the logged rollouts and fresh-T0 evals are controls (K2-logged, K4), not rows. Step wall times:

[[t_walls()]]

The twelve steps sum to [[sum_walls()]] s. The chain's own read ran [[reader_start]] to [[refused_time]] and REFUSED (disclosure 1). The read of record's console ends with *"[[read_done]]"* and *"[[read_rc]]"*.

## 10. Provenance (artifact path, sha256)

Evidence class MEASURED (ours). Every sha256 below is of the bytes at `d7dc9f11`, re-measured for this document. The run directories are under `raw/2026-10-28-launch-intent-read/box2/`.

[[t_prov()]]

The reader's pins (`provenance.rf0_sha256`, `provenance.reader_sha256`, `provenance.manifest_sha256`) are in the JSON. The PREREG read for this document is the file at `d7dc9f11`, which is the file at `e28516e8` (sha256 `6ae79f2f…`).
