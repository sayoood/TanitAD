# PREREG_V8F_DAGGER: do the model's OWN T1-kin states, supervised with the human's labels at the matched route point, make the T head's decisions survive its own motion?

**DRAFT, written 2026-10-26, before any O-RECOVER number exists and before any DAgger code, bank or run exists.** Claim row `H-V8F-DAGGER-1`.
No ORC or ORC-0 output was read to write it (none existed to read; the drafting agent was barred from `v8f-orecover*` and every ORC log).

**What commits this arm.** `PREREG_V8F_ORECOVER.md` §4, two outcome rows, verbatim:
* **NO EFFECT:** *"perturbing the inputs does not teach own-state decisions. Next: on-policy states, i.e. the model's own T1-kin states supervised
  with the human's labels at the matched route point (DAgger form), under its own PREREG"*.
* **HARMS:** *"next is NO EFFECT's arm at p = 0.1"*.

⇒ **This PREREG runs if and only if O-RECOVER reads NO EFFECT (arm at p = 0.3) or HARMS (arm at p = 0.1).** Under ROBUST or PARTIAL it is parked,
not run, and the register row says so. Under REFUSED it waits for O-RECOVER's re-run. The mapping from ORC's outcome to p is fixed here, before
ORC's data, so no ORC number can choose it.

## Why (MEASURED, quoted from the primary files)

* `RESULT_V8F_T1_HEADS.md` (T1-kin, m = 10, causal bucket): the T head's decisions **DEGRADE** on its own state.
  * Lateral macro-F1 0.848 → 0.614 (s0) and 0.858 → 0.632 (s1); longitudinal 0.461 → 0.336 / 0.487 → 0.357. Both separated at both seeds.
  * The loss is in the turns: TURN_L / TURN_R F1 0.81 / 0.75 → 0.38 / 0.50 at s0, LANE_KEEP 0.98 → 0.96.
  * At zero row offset (label row = the row whose cameras T saw) lateral F1 is still 0.606 / 0.647: most of the drop is T reacting to its OWN ego
    state, not to the replayed cameras (descriptive probe, decides nothing).
* `RESULT_V8F_HACC_MC.md`: MECHANISM-NOT-ENOUGH. The zero-training post-map fix left L2 separated-worse at s0 (+0.0028 [+0.0004, +0.0055]); the
  committed next was the box-2 training lever, i.e. O-RECOVER.
* `raw/2026-10-26-lcdiag/DIAGNOSIS.md` (`R76`): 88–96 % of the L2 excess sits on launch-crawl pairs (the human moving ≥ 5.2 m/s, the model's own
  car below it). Those states exist only in the model's own rollouts; a logged training set never contains them.
* **What T reads of its own state** (code, `stack/tanitad/models/v8f_mini.py:686`): ONE ego token, `_ego(p10f[last_t])` (the 8 P10 features at T's
  tick frame), plus `nav` at that frame and S's checkpoints moved by the held-frame odometry (`:745`). These are the tensors this arm replaces.
* ⇒ ORC trains on states drawn from a fixed parametric law. If that law does not reach the states the loop actually visits (correlated
  turn-timing lag, launch deceleration over the history), the cheapest next lever is to train on the visited states themselves: DAgger, one pass.

## 1. The data: the own-state bank

**Generator (frozen, one pass, not iterated).** M2b_s{seed} (`D:/tanitad/runs/v8f_mini_full/M2b_s{seed}/ckpt_last.pt`, sha256 `601cb435…` /
`6104fb15…`), PLAIN (no HACC post-map), rolled out by `v8f_t1_eval.py`'s own machinery: `--replan-steps 10`, `--limit causal`, `--nav deadreckon`,
`--scene logged`, `--schedule periodic`, executor `plan`. This is the decisive read's configuration, so the states are the ones that read scores.
* Each seed's bank comes from that seed's M2b. DAG s0 trains on M2b_s0's states and DAG s1 on M2b_s1's: on-policy for the init it fine-tunes.
* **One pass.** The states are M2b's, the policy whose failure was measured. A 3,000-step fine-tune at peak lr 1e-4 moves the policy little, so a
  second pass is a separate lever (iteration 2), committed below only as a next step.

**Clips and windows: TRAIN only.**
* Every window of M2b's own TRAIN list, rebuilt by the trainer's `build_data` from M2b's `argv.json`: 4,512 clips, **306,566** windows
  (`MODEL_REGISTRY.md` §17; `--exclude 540:600`, the holdout withheld). One rollout per window start t0.
* ⛔ No eval-141 clip, window or frame. The bank's clip set must be disjoint from eval-141's (`v8f_mini_data.assert_disjoint`), and a bank holding
  one eval clip must be refused (deliberate-regression arm of control K-B3).
* ⚠️ `MODEL_REGISTRY.md` §17: box 2's `cache_full` is **not** the parity corpus `physicalai-train-e438721ae894`. This arm uses M2b's TRAIN set
  exactly as M2b did and re-selects nothing. It stays comparable with the v8F arms, never with parity arms.

**Which states.** Ticks 1 and 2 of each rollout: j = t0 + 10 and t0 + 20 (2 s and 4 s of own motion). Tick 0 is the logged state and is never an
own state. Two forwards per window (ticks 0 and 1); tick 2's state is reached by committing tick 1's plan. Before cuts: 2 × 306,566 = 613,132 own
states per seed (arithmetic).

**What is stored per own state:** clip index, t0, tick, j; the own history rows j − 15 … j (hmax = 16 in M2b's config: p10f [16, 8], world pose
[16, 3], travelled arc [16]), the own kin3 at j; the matched row r′, its lateral distance and heading error, and the cut reason. Rows before the
clip start are NaN, as in the store. No reader indexes them: the ages are ≤ 6, and j ≥ 13. The bank header records the generating
checkpoint's sha256, the rollout configuration, the window-list sha256, and the sha256 of the matching constants (§2).

## 2. The matched route point r′ and the sample it makes

Constants fixed before any probe ran (`probe/THRESHOLDS_FIXED.txt`, sha256 `438366d5…`).

**Definition.** For an own state at clock row j with world pose (x, y, ψ), of a rollout started at t0:
1. Candidates: the logged 0.2 s pose grid of the same clip, rows [t0, T − 1] (the T1-HEADS candidate set, `t1hd_read.py:161`).
2. Project (x, y) onto the logged polyline through those rows, segment by segment. **e_lat** = the smallest perpendicular distance; the foot's
   cumulative arc is s*.
3. **r′** = the row whose cumulative logged arc is nearest s*.
4. **Ties:** rows whose arc lies within 0.05 m of that nearest arc are tied (a stationary stretch: the human waiting). Among them, the row nearest in
   time to j; then the earlier row.
5. **Off-route cut:** e_lat > 3.0 m (the decisive reader's own `OFF_ROUTE_M`, `t1hd_read.py:34`), or |wrap(ψ − ψ_logged[r′])| > 30°. No sample.
6. **End edge:** r′ > T − 31 (no complete 6 s logged future at r′). No sample.

Every cut is counted by reason and reported.

**Why not the reader's first-minimum point rule.**
* At a stop, the human's rows share one position, and first-minimum picks the first row of the wait. A crawling own car at the stop line would then be
  paired with "the human has just stopped", never with "the human launches here".
* The launch is the failure (`R76`). The time-nearest tie pairs the own state with the human's row at the same moment of the wait.
* Point distance also charges up to half a step of along-track spacing (3 m at 30 m/s) to the 3 m cut. Projection measures the lateral offset only.
* Disclosed: the decisive rows are still scored by the T1-HEADS reader's own rule, unchanged. A beside row re-scores them at this rule.

**The sample (route-paired).** A DAgger sample is **the logged training sample at (clip, r′)** with its ego tensors replaced by the own state's.
This is exactly ORC's injection path (`EgoPerturb`, `stack/tanitad/data/v8f_mini_data.py:593`), with the replacement drawn from the bank instead of
from a law.

| tensors | treatment | rule |
|---|---|---|
| `p10f`, `kin3`, `p10_t`, `base`, `pose`, `pose_t`, `pose_s`, `nav` (= `EGO_PERTURB_REBUILT`, `:584`) | **replaced** by the own state's | own history rows j − 15 … j; `base` = `kdx_base(own kin3)`; poses re-anchored by the rigid motion mapping the own pose at j onto the logged pose at r′; `nav` channels 0–2 and 4 kept, channel 3 = nav[r′, 3] + own arc behind NOW / 100 on frames sharing NOW's token and validity (ORC's rule) |
| `feats`, `frame_ok`, `limit`, `limit_ok`, `a_t`, `a_s`, `pose_now` (= `EGO_PERTURB_KEPT`, `:587`) | **kept** | the logged images at r′'s history; the clip's logged speed bucket (M2b's trained input; ORC kept it too); the sampler's ages; `pose_now` is the anchor |
| `act` (`EGO_PERTURB_RECORD`, `:590`) | **kept** | the world model's transition record. Own actions paired with logged frames would corrupt the IDM term |
| `gt`, `gt_ok`, `s_h`, `lead_gap_h`, `lab`, `lab_t` | **kept: the human's, at r′** | the trajectory target is the human's 6 s future from r′, in r′'s frame (the re-anchored own NOW frame). Speed is carried by `gt`'s spacing, `s_h` and the lon class. Lat / lon classes, goal tokens and `cons` are read at r′ − a_t |
| `lab_s.rc` | **re-expressed, not changed** | the same world points (the human's route checkpoints at r′ − a_s), moved from the logged frame at r′ − a_s into the sample's S frame `pose_s` (`transform_ckpt`). The identity at a_s = 0. It is the only label tensor touched, and only by a frame change |

**Ages.** The sampler's (a_t, a_s), as clamped by the logged item at r′ (`V8FMiniWindows.item`, `:494`), index both the logged rows and the own history.

## 3. The images stay logged (no re-perception tier): the limitation, stated

**The images are the logged frames of the human's drive at r′.** No pixel is re-rendered: T2 is not provisioned. Three consequences:
1. **The scene is at the human's time.** Agents and lights are where they were when the human passed r′. The sample asks: what would the human have
   decided here, in the scene the human saw, had the car arrived with this kinematic state?
2. **The lateral offset (≤ 3 m) and heading error (≤ 30°) are invisible** in the image and are absorbed by the re-anchoring. ⇒ **This arm cannot
   teach lateral recovery.** That needs re-perception, or a synthesised rejoin target, which is a different lever.
3. **The T1-kin read shows the clock-row cameras** (`v8f_t1_eval.py:239`), so at evaluation an own state meets a camera from another place than in
   training. That mismatch is a property of the rig, identical for DAG and DAG-0.

**Why the arm still tests the hypothesis.**
* The hypothesis is about T's response to its own EGO-STATE inputs: the p10f token, `base`, the held-frame odometry, and nav progress. Those are
  exactly the tensors replaced with on-policy values, while everything else is a real logged sample.
* T1-HEADS localised most of the drop to that channel: lateral F1 still falls by about 0.22 at zero row offset.
* ⇒ If on-policy ego states with the human's labels do not move T's F1 at T1-kin, this channel cannot be repaired by data of this kind at this dose.

**Rejected: the time-paired form** (the clock-row images at j, labels at r′). It would train on the rig's own camera/position mismatch, which is absent
at deployment. It would also join two levers (own ego state, and camera mismatch), so a result could not be attributed.

## 4. The arms (box 2; through the trainer's own launch gate; both seeds)

| arm | what | role |
|---|---|---|
| **DAG** | M2b_s{seed} fine-tuned **3,000 steps**; a fraction **p** of eligible samples per batch replaced by a route-paired own state (§2) | **decisive** |
| **DAG-0** | the same fine-tune with `--dagger-p 0`: same init, steps, data, LR path, seed | **the control**: separates the own states from "3,000 more steps" |

* **p = 0.3** if O-RECOVER read NO EFFECT; **p = 0.1** if it read HARMS. That second case is "NO EFFECT's arm at p = 0.1", verbatim.
* **Eligible** = the drawn window (ci, t) has at least one bank own state with r′ = t that passed every cut. Among several, one is drawn uniformly.
  * The draws come from a private generator seeded by (`--dagger-seed`, step). It never touches the global RNG, so a resumed run draws what an
    uninterrupted run drew (`EgoPerturb`'s rule).
* **Fine-tune mechanics = `PREREG_V8F_ORECOVER` Amendment 1, unchanged:** `--init-ckpt <M2b_s{seed}/ckpt_last.pt> --steps 3000 --lr 1e-4
  --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2 --seed <seed>`, plus M2b's argv. `--ego-perturb-p 0` is named on both arms.
* **New flags** (default off; a default run is bit-identical to the pre-change trainer): `--dagger-p`, `--dagger-bank`, `--dagger-bank-sha256`,
  `--dagger-seed`.
  * DAG names all four.
  * DAG-0 names only `--dagger-p 0`. That is ORC Amendment 3's rule: the gate refuses declared-and-inert flags at p = 0.
  * The pair rule: the argv of DAG and DAG-0 are identical except `--out`, `--dagger-p` and the three DAG-only flags.
* ⇒ **DAG-0's argv equals ORC-0's plus `--dagger-p 0`.** ORC-0 is therefore a same-argv REPLICATE of DAG-0, trained by another build. It is used in
  §5 as the run-to-run floor, at no extra compute.
* **One lever.** A separated effect is attributed to "own-state replacement as a whole" (kinematics, odometry and nav progress together). It is never
  split into its parts.
* `--ego-perturb-p > 0` together with `--dagger-p > 0` is refused by the gate. The two levers are never combined in one arm.

## 5. Evaluation, rows, bars (ORC's, reused)

| where | what |
|---|---|
| **dev-box CPU (decisive)** | `v8f_t1_eval run --replan-steps 10 --limit causal --dump-ticks --dump-heads` on eval-141, **plain**, for DAG and DAG-0 × 2 seeds; plus one `--limit logged` run each for K4 (ORC Amendment 2) and one fresh-age T0 eval each as K4's comparator (ORC Amendment 4) |
| box 2 | the chain's standard T0 eval (eval-141) of every checkpoint: R-T0, T0-CKPT, the four families at T0 |

**Tier.** T1-kin is a **self-action OPEN loop**, never "closed loop". T0 is the guard tier.
**Estimator.** Paired episode-cluster bootstrap (`taniteval.ci`), B = 2,000, seed 0, 141 episodes, per seed, DAG − DAG-0, same windows and matched
cells.
**What each interval answers.** Another draw of EPISODES. Another TRAINING RUN is answered by the two seeds plus the replicate floor below. Another
INFERENCE RUN does not arise: the planner is deterministic (`PREREG_V8F_PGT`).

| row | definition | bar / direction |
|---|---|---|
| **R-TLAT / R-TLON** | T's lat / lon macro-F1 at ticks ≥ 1 (the T1-HEADS definition and reducer, via the ORC reader unchanged), DAG − DAG-0 | a head **clears** at seed s iff separated-better AND point ≥ **bar(h, s) = max(2F_h, 2·RF_h,s)** |
| **RF (replicate floor)** | \|F1_h(DAG-0_s) − F1_h(ORC-0_s)\|, ticks ≥ 1, cells scored in both, point | enters the bar. It reads 0 if the two runs are bit-identical |
| **R-ADE** | T1-kin plain ADE6, DAG − DAG-0 | guard: not separated-worse |
| **R-T0** | T0 own-code ADE6, DAG − DAG-0 | guard: point ≤ +F/2 = **0.0604 m** (F = F_M2b 0.1207 m, `MODEL_REGISTRY.md` §17) |

* 2F = **0.0248** (lat) / **0.0146** (lon), the T0 fresh seed floors (ORC §2). Bars are inclusive (ORC A2).
* Without the replicate floor the bar is ORC's 2F. It can only rise, and only by the measured run-to-run gap.

**Beside (never decisive):**
* DAG − ORC on R-TLAT / R-TLON: on-policy states against the law. This is the scientific comparison, but its arms differ in more than the lever.
* Per-class turn F1 (TURN_L / TURN_R), where the loss was.
* Tick-0 F1: did DAG lose logged-state skill?
* The `R76` crawl rows M1 and L2.
* The T1 off-route fraction at tick 2: does DAG stay nearer the route?
* S-POS@30 / 50 / 80 against CTRV. STRATEGIC is not paused for v8F (PI 2026-10-17).
* The four families at T0 and at T1, each with its tier stamp, estimator and n; a family that cannot be computed is reported with its reason and n.
* DAG − M2b at T0.
* R-TLAT / R-TLON re-scored at §2's tie rule.

## 6. Controls (each must pass, or the read is REFUSED)

**Bank (before any training):**
* **K-B0** tick-0 identity: an own state at tick 0 reproduces the logged item's ego tensors bit for bit.
* **K-B1** replay: a seeded 0.5 % of bank rollouts are re-run and must reproduce their stored rows bit for bit.
* **K-B2** on-policy: the bank's generating checkpoint sha256 equals the arm's `--init-ckpt` sha256.
* **K-B3** TRAIN-only: the bank's clips ⊆ M2b's TRAIN clips, its window-list sha256 equals the trainer's, and ∩ eval-141 = ∅. Deliberate
  regression: a bank holding one eval clip is refused.
* **K-B4** the matcher's synthetic cases (straight, stationary tie, lateral cut, heading cut, end edge, loop) all pass.
* **K-B5** re-anchoring: `pose_now` equals the logged pose at r′, and the held-frame relative transforms are preserved to ≤ 1e-4 m.
* **COVERAGE:** the eligible fraction of TRAIN windows is ≥ 0.5. Below it the arm does not launch: outcome BLOCKED (§7).

**Launch gate** (`v8f_mini_train.py --selfcheck-only`, as for every v8F arm; ORC A2): a PASS token per run, four in all, each binding the commit, that
run's argv sha256 and the data manifest (including the bank's sha256).
* **G-DVB:** every new flag has a `FLAG_CHECKS` entry (the gate refuses an unlisted flag, `v8f_mini_train.py:407`). Plus `bad_dagger`, which checks:
  * the built p;
  * the bank's path and sha256 against the declaration;
  * K-B2;
  * that the bank's configuration equals §1;
  * the exclusivity with `--ego-perturb-p`;
  * that DAG-only flags are refused at p = 0.
* **G-LIVE `dagger_probe`** (the structure of `ego_perturb_probe`, `:765`), on the real gate batch at p = 1:
  * every key `forward_infer` reads is replaced or kept, so none is unaccounted for;
  * every selected row changes in `p10f`, `base`, `kin3` and `p10_t`. A selected row whose own ego is bit-identical to the logged one is exempted
    and counted (ORC A4's at-rest rule, generalised), and at least one non-identical row must exist;
  * no unselected row changes;
  * every kept key is the same object, and `lab_s.rc` equals its frame change to 1e-5;
  * `live_check` on the swapped batch;
  * the loss AND its gradient move;
  * replacing `p10f` alone moves the loss.
* **G-LIVE zero controls**, each reading **exactly 0.0**:
  * the same loss twice;
  * p = 0 returns the same object, with a loss and gradient difference of 0;
  * the frozen P10 prior's gradient difference;
  * the world model's gradient difference (`act` is kept).
* **Default-off identity:** DAG-0's 3-step loss trace equals the pre-change trainer's bit for bit.
* **Deliberate regression:** proven at build time. Deleting the swap line, or the `dag.apply` line in the loop, turns the lever's tests and the
  G-LIVE probe RED, by mutation. No GPU arm is spent on it.

**Reached training:**
* DAG: the run's totals `dagger_n / dagger_eligible` lie in p ± 0.02. ESTIMATED binomial 3σ: about 0.0035 at about 150,000 eligible samples.
* DAG-0: `dagger_n` is exactly 0.

**Read:**
* The T1-HEADS reader's K1–K7 on every run, as amended by ORC A2 and A4. K4's comparator is the arm's own fresh-age T0 eval. K7 requires the rebuilt
  track to equal the dumped one.
* The pair rule (§4).
* A reader K1 on its own rows: (a, a) reads +0 [0, 0].

## 7. Outcomes, committed now (DAG vs DAG-0)

| outcome | condition | committed next |
|---|---|---|
| **ROBUST** | R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2 at both | adopt DAG as the base. Re-read T1 dominance over P10 and FLAT's committed repeat on it. If the crawl beside rows (M1, L2) still fail, next is **DAgger iteration 2** (DAG's own states), under its own read |
| **PARTIAL** | exactly one head clears at both seeds; guards held | adopt for the head that moved. Next: **iteration 2** for the other head, under its own read |
| **NO EFFECT** | no head clears at both seeds; guards held. A head separated-better but under its bar is NO EFFECT, reported and never called an effect | own-state supervision does not repair T at T1-kin at this dose. With ORC also null or harmful, training-distribution levers on T's ego channel are ELIMINATED. Next: **T-EGOCUT**: T's ego token (`v8f_mini.py:686`) replaced by a constant at training AND inference, M2b init, 3,000 steps, both seeds, under its own PREREG. It decides whether T should read its own ego state at all |
| **HARMS** | R-ADE separated-worse at ≥ 1 seed, or R-T0 above +F/2 at ≥ 1 seed | DAG is not adopted; reported with its rows. If it ran at p = 0.3: next is **this PREREG's arm at p = 0.1**. If it ran at p = 0.1: next is NO EFFECT's next (T-EGOCUT) |
| **BLOCKED** | COVERAGE < 0.5 (§6) | no launch. The stats are reported. Next: a **rejoin-target form** (a synthesised return to the route for off-route own states, ChauffeurNet), under its own PREREG |
| **REFUSED** | any control fails | fix and re-run |

The table is complete:
* a failed control gives REFUSED, and a coverage failure gives BLOCKED;
* with controls and coverage passing, a failed guard gives HARMS;
* with the guards held, the number of heads that clear at both seeds gives ROBUST (2), PARTIAL (1) or NO EFFECT (0).

## 8. Queue

* **Box 2**, after O-RECOVER's four runs:
  1. **bank s0, bank s1.** These depend only on M2b and the TRAIN cache, so they may run as soon as box 2's GPU is free, before ORC's read.
     ESTIMATED 1.1–1.5 h per seed with four clip-sharded processes (DESIGN_NOTES §4).
  2. Then, only on ORC's NO EFFECT or HARMS: gate and train DAG s0, DAG-0 s0, DAG s1, DAG-0 s1, each through its gate. The chain's predecessor
     predicate reads ORC's read outcome, not only its end marker.
  3. The T0 evals.
* **Dev-box CPU**, gated on C: ≥ 2 GB free: the T1 chain (orct1, re-parametrised for DAG / DAG-0) as each pair of checkpoints arrives. Then the
  reader.
* ⚠️ **Box-2 time for the bank and the four runs needs the PI's go**, as O-RECOVER's did.

Nothing in this file may be changed after ORC's read except by a dated amendment that says what it moves and why. The arms, rows, bars, controls
and the outcome table are frozen at landing.

## Amendment 1 (`A1-2026-10-26`): three readings fixed while building, before any DAgger code landed and before any bank or run

**When.** Written 2026-10-26, after O-RECOVER's read (HARMS, `raw/2026-10-26-orecover-run/read/ORC_READ.json`, sha256 a50dbba9…) and
before any DAgger bank, run or read.
* The draft above was banked unchanged at `ac139f71`, before that read (sha256 719e5f15…).
* This amendment changes no arm, row, bar, control or outcome. It fixes three readings that the build found ambiguous, and it applies §7's
  own mapping.

### 0. p, by §7's own mapping (not a change)
O-RECOVER read **HARMS**, so this arm runs at **p = 0.1**, as the draft's opening section and §7 commit.

### 1. The tie set (§2, item 4): the TEXT governs
Tied rows are those with |arc − arc[r*]| ≤ 0.05 m, where r* is the nearest row.
* The draft's cost probe (`probe/probe_dagger_cost.py`) coded a band around the nearest DISTANCE instead.
* MEASURED on 20,000 random polylines (`raw/2026-10-26-dagger-build/diff_vs_probe.log`): the two rules disagree on 2.1 % of cases. With only
  that line swapped to the probe's rule, the built matcher reproduces the probe on 20,000 / 20,000.
* The probe measured cost, not the rule. The text is the registered rule.

### 2. Nav progress on frames with no route (§2): it stays exactly 0
On frames whose route is invalid, nav channel 3 keeps the value the join stores, exactly 0.0. That is the rule ORC's rebuild and the T1 tool
apply.
* Read to the letter, "nav[r′, 3] + own arc behind NOW / 100 on frames sharing NOW's token and validity" would write arc / 100 onto frames
  that carry no route.
* The arm's hypothesis is about T's response to its own ego state (§3). Writing a route-progress value where the logged corpus has no route
  would add a NAV signal the logged data never shows. That is a second lever, so it is excluded.
* The build pins both readings by test (`NAV_REBUILD_INVALID = False` is the registered one).

### 3. K-B3 is three checks, each read on its own (§6)
1. **Clips:** the bank's clips ⊆ M2b's TRAIN clips, matched BY NAME. The bank may hold fewer clips than TRAIN, because a clip whose every
   own state is cut has no record. Equality is not required.
2. **Windows:** the TRAIN window-list sha256 that the bank records equals the trainer's.
3. **Eval:** the bank's clips ∩ eval-141 = ∅.

Each is its own control, so a failure names which one. The deliberate regression (one eval clip injected) must fail check 3.

Nothing else moves: the arms, the rows, F, 2F, the replicate floor, the bars, the controls, the outcome table, the committed next steps and
the queue.

## Amendment 2 (`A2-2026-10-27`): BANK-CONFIG reads `limit_clips` with the generator's own convention. POST-DATA, CONTROL-ONLY

**When, exactly.**
* The four runs finished on box 2 at 05:55 on 2026-10-27, and the dev-box T1 chain finished its 8 rollouts.
* The pre-registered reader (`dag_read.py`, sha256 `f5e9d618…`) ran from 06:02:06 to 06:02:10 and returned **REFUSED** (rc 2, nothing
  written). Its log is `raw/2026-10-27-dagger-run/dag_read_refused.out`, sha256 `14e4c387…`.
* It refused on exactly one control, at both seeds: **BANK-CONFIG**, *"the bank is PARTIAL (partial False, limit_clips 0)"*.
* The refusal is raised at the bank stage, before COVERAGE and before any arm is read. No outcome row was computed, and none was seen by
  anyone.
* This amendment was written after that refusal and before any re-read.

**What was wrong.** The control was wrong, not the bank.
* The generator (`stack/scripts/v8f_dagger_bank.py`) declares `--limit-clips` with `default=0`, meaning no limit (:399).
* It writes `"limit_clips": a.limit_clips, "partial": bool(a.limit_clips)` (:263, :289), and reads 0 as no limit itself (:254, :336).
* The reader's predicate `c.get("limit_clips") is not None` (`dag_read.py:775`) treats that 0 as a limit.
* Its tests built their synthetic full banks with `None`, and the only real bank they saw (the 12-clip smoke) is genuinely partial. So the
  mismatch was never exercised.
* MEASURED, the banks are complete:
  * `PROVENANCE.json` reads `partial: False, limit_clips: None` at both seeds;
  * the trainer's launch gate measured **4,512 clips / 306,566 windows** per seed, which is §1's registered TRAIN size;
  * COVERAGE reads 0.756 (s0) and 0.769 (s1).

**What changes.** BANK-CONFIG's completeness clause only. A bank is complete iff `partial is False` and `limit_clips` is 0 or None. The
inconsistent pairs are refused with their own message:
* `partial` False with `limit_clips` > 0;
* `partial` True with `limit_clips` 0.

The real 12-clip smoke is still refused. A test pins the generator's convention against its source text. New mutants of the predicate are
killed, and the reader's full suite and mutation proof are re-run on the final bytes before the re-read.

**What does not change.** Every other clause of BANK-CONFIG, every other control, the arms, the rows, F, 2F, the replicate floor, the bars,
the outcome table and the committed next steps. The re-read uses the same manifest, the same runs and the same estimator. It runs once,
after this amendment and the fixed reader have landed.
