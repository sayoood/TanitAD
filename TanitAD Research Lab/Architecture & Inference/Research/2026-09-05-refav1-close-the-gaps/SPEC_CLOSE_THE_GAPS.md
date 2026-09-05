# PREREG — CLOSING refav1's GAPS: the constraint base, the penalty knee, and a GOAL-CONDITIONED lateral cost

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-close-the-gaps`
**Written** 2026-09-06 ~00:40 Europe/Berlin — **BEFORE any arm below was launched.**
**Rig** refav1 `ckpt_ep3` step 21109, p4 panel: **40 windows / 8 episodes**, stride 16,
`--no-lead-block --no-navshuf`, planner arm `cl` at **T1 (self-action open loop)**.
**Estimator** paired episode-cluster bootstrap over the 8 episodes (`taniteval/ci.py`).
⛔ **`sep` is not read.** Deltas and intervals only, each against the **per-METRIC** seed floor.

---

## §0 — THE FRONTIER, RE-DERIVED FROM THE RECORDS (not inherited)

MEASURED by re-reading every `rec_*.json` in `C:/Users/Admin/refav1_margin/p4out` at 00:30
(`raw/frontier.txt`), arm `cl`, n = 40. **W_KAPPA is read from `cost.weights`, not from the arm name.**

| arm | metric | `W_KAPPA` | ADE m | curv MAE | turn_L | turn_R | lane_keep | speed MAE |
|---|---|---|---|---|---|---|---|---|
| `cos_argmax` | cos | 0 | 1.8944 | 0.08048 | 0.0000 | 0.2500 | 0.6190 | 0.7274 |
| `ccos_argmax` | ccos | 0 | 1.3272 | 0.05537 | **0.3636** | **0.7500** | 0.7143 | 0.7155 |
| `ccosh_w000` | ccosh | 0 | 1.3272 | 0.05537 | 0.3636 | 0.7500 | 0.7143 | 0.7155 |
| `l3ladder` | ccos | 0 | 1.3236 | 0.05373 | 0.3636 | 0.7500 | 0.5714 | 0.7303 |
| `combined` | ccos | 0 | 1.0504 | 0.04686 | **0.3636** | **0.7500** | 0.7619 | 0.7327 |
| **`kamm07`** | ccos | **0** | **0.9927** | 0.04846 | **0.3636** | **0.6250** | 0.7619 | 0.7138 |
| `cos_wk` | cos | 0.05 | 0.9251 | 0.04008 | 0.0000 | 0.0000 | 1.0000 | 0.7217 |
| `wk15` | ccos | 15.11245 | **0.8934** | 0.03098 | ⛔ 0.0000 | 0.5000 | 1.0000 | 0.7919 |
| `lonseam` | ccos | 15.11245 | 0.8934 | 0.03098 | 0.0000 | 0.5000 | 1.0000 | 0.7919 |
| `wk15_ladder` | ccos | 15.11245 | 0.9408 | 0.03979 | 0.0000 | 0.0000 | 1.0000 | 0.7609 |
| `best` | ccos | 15.11245 | **0.8838** | 0.03128 | ⛔ 0.0000 | 0.5000 | 1.0000 | 0.7751 |
| `wk151` | ccos | 151.1245 | 0.9084 | 0.03802 | ⛔ 0.0000 | ⛔ 0.0000 | 1.0000 | 0.7995 |
| `lonshift` | ccos | 15.11245 | **0.7868** | 0.04116 | ⛔ 0.0000 | ⛔ 0.0000 | 1.0000 | **0.5657** |

⭐ **The claim this package is built on, and it REPRODUCES:** the CONSTRAINT captures **77 %** of the
PENALTY's ADE gain at **zero turning cost**. `ccos_argmax -> wk15` = **-0.4338 m** and kills
`turn_left`; `ccos_argmax -> kamm07` = **-0.3345 m** with `turn_left` **unchanged at 0.3636**.
`0.3345 / 0.4338 = 0.7712`.

### §0.1 — THREE PREDECESSOR CAUTIONS, DISCHARGED

1. ⚠️ **`cos_wk` is NOT usable as evidence that a tiny penalty collapses turning.** Confirmed:
   `kappa_by_goal_all.txt` reads `cos_wk` **med|k|max 0.00000, frac k!=0 0.0000 on ALL THREE goal
   tokens** — an all-zero path. Its 0.9251 is a degenerate-baseline property. **Excluded.**
2. ⭐ **The `oracle_s0` / `ccosh_w000` / `ccos_argmax` bit-identity is EXPLAINED, and it is not an
   M40 shared denominator.** `raw/ccosh_null.txt` (banked, predecessor stream) MEASURED the `ccosh`
   hold branch **firing** — `basecost_cv` frac exactly 1.0 goes 0.2500 -> 0.0000, min
   0.884896 -> 3.9105e-08, cem frac 0.750 -> 0.925 — while the **plans are bit-identical**
   (max abs diff **0.000000e+00** over 40 windows) against a same-breath control that MUST differ
   (`wk15` vs `ccos`, max abs diff **1.508402e+01**). ⇒ the metric change is a **real, measured
   null on behaviour**, not an inert flag. `oracle_s0` is the same run under a name that describes
   its nav provenance, not a lever (`M51`: a name is not provenance — the GT arm is `ol`).
3. ⛔ **`lonseam` reads bit-identical to `wk15` on every column.** That is the tool's own documented
   guard being right: `--jerk-seam a0` with `W_JERK = 0.0` is **INERT BY CONSTRUCTION**
   (`refav1_arm.py:809`). It is a **null control that reads a known value**, and it is quoted here
   as one — never as a lever result.

### §0.2 — ⭐⭐ THE MECHANISM, MEASURED, AND IT REWRITES A3

From `raw/kappa_by_goal_all.txt` (predecessor stream, n = 40, vocabulary imported from
`tanitad.models.vocab_v7`), realised planner curvature **split by the DECODED lateral goal token**:

| arm | LANE_KEEP (n=18) mean k2 | LANE_KEEP frac k!=0 | **TURN_L (n=9) med\|k\|max** | TURN_R (n=13) med\|k\|max |
|---|---|---|---|---|
| `ccos_argmax` (W_KAPPA 0) | 0.002908 | 0.4444 | **0.08000** | **0.08000** |
| `wk15` (W_KAPPA 15.11) | **0.000000** | **0.0000** | ⛔ **0.02066** | 0.08000 |
| `wk151` (W_KAPPA 151.1) | 0.000000 | 0.0000 | ⛔ 0.00840 | ⛔ 0.01317 |

⛔ **THE PENALTY DOES NOT SUPPRESS TURNING TO ZERO — IT UNDER-TURNS BY 3.9x ON THE VERY WINDOWS
WHOSE GOAL SAYS `TURN_L`.** `frac k != 0` stays **1.0000** on TURN_L under `wk15`; the *magnitude*
falls from the commanded 0.08000 to 0.02066, i.e. **26 % of what the tactical brain asked for**,
which is below the trajectory labeller's turn gate — so the recall reads 0.0000.
⇒ **This is the hierarchy defeating itself, MEASURED: the tactical brain commands TURN_L and the
operative cost fines the planner for obeying it.** It is exactly `M48` — *a penalty RANKS, so a
quadratic curvature penalty always prefers the cheapest non-zero curvature and under-turns
everywhere, including where turning is CORRECT.*
⚠️ **And it is ASYMMETRIC**: TURN_R is **unchanged at 0.08000** under the same symmetric k2 penalty.
A symmetric penalty cannot be the asymmetry's source — the **goal term's magnitude must differ by
direction**, so the same k2 overwhelms the left-turn goal gradient and not the right-turn one.
⇒ This is where refav1's turn asymmetry lives, and it is a COST-BALANCE fact, not a head fact.

### §0.3 — PER-METRIC SEED FLOORS (⛔ quoted for the exact STATISTIC, never the family)

From `.../2026-09-05-refav1-cost-geometry/raw/seed_floor_ext_ccos.txt` (A) and
`seed_floor_ext_combined.txt` (B) — two inference-seed replicate pairs, controls PASSED
(`ha`/`ha0`/`ha0_ext`/`ol` bit-identical in both):

| statistic | floor A (ccos pair) | floor B (combined pair) | **BINDING (max)** |
|---|---|---|---|
| `ADE_m` | 0.06070 | 0.10350 | **0.10350** |
| `LAT curv_MAE` | 0.00066 | 0.00200 | **0.00200** |
| `LON speed_MAE` | 0.00380 | 0.00120 | **0.00380** |
| **`TACpc turn_left_rec`** | **0.00000** | **0.00000** | **0.00000** |
| **`TACpc turn_right_rec`** | **0.00000** | **0.00000** | **0.00000** |
| `TACpc lane_keep_rec` | 0.14290 | 0.09520 | **0.14290** |
| `TAC lat_acc` | 0.07500 | 0.05000 | **0.07500** |

⭐ **turn_left and turn_right recall have a measured inference-seed floor of EXACTLY 0.0000 on both
pairs** — both replicates reproduced 0.3636 / 0.7500 to four decimals. ⇒ a turn-recall change of
**any** size on this rig is above its own noise floor, which is what makes `wk15`'s
0.3636 -> 0.0000 admissible as a lever effect.
⛔ **lane_keep recall's floor is 0.1429 — LARGE.** No lane_keep-recall claim below that is made here.

---

## §1 — A1 `kammshift` = `kamm07` + `a0_shift` (RUNS FIRST)

**Flags** = `kamm07`'s command line **+ `--a-sustain-mode a0_shift`**. ONE variable.
**Why**: `lonshift` (ADE 0.7868, the programme's best refav1 ADE) is **`wk15` + `a0_shift`** — it
inherits `W_KAPPA = 15.11245`, **which is why it cannot turn**. Apply the same longitudinal lever to
the **CONSTRAINT** base instead of the **PENALTY** base.

### ⛔ COMMITTED PREDICTION — written before the arm exists

**Point prediction: ADE ~ 0.9927 - 0.1066 = 0.886**, from the measured `wk15 -> lonshift` transfer
(0.8934 -> 0.7868 = **-0.1066**), **with `turn_left` recall unchanged at 0.3636.**
Secondary: `LON speed_MAE` ~ 0.7138 - 0.2262 = **0.488** (the `wk15 -> lonshift` speed transfer).

⚠️ **THE RISK I AM NAMING IN ADVANCE, BECAUSE IT IS VISIBLE IN THE PREDECESSOR ARM.** `a0_shift`
moves the goal's **target speed** `v_t` (`refa_v1.py:499`), and the goal control profile is rolled
through the TACTICAL predictor to build the goal FIELD — so it changes the lateral ranking too.
MEASURED: `wk15 -> lonshift` took **turn_right 0.5000 -> 0.0000**. So the lever is NOT known to be
laterally neutral, and a turning collapse here is a live outcome, not a surprise.

| # | outcome | reading | consequence |
|---|---|---|---|
| **P** | **PASS** | ADE <= **0.8892** (= 0.9927 - 0.1035 binding floor) **AND** `turn_left` recall **> 0.0000** | ⭐ refav1 has, for the first time, an arm that is BOTH more accurate than the penalty base AND still turns. This becomes the shippable configuration. |
| **Q** | PARTIAL | ADE <= 0.8892 **AND** `turn_left` = 0.0000 | `a0_shift` destroys turning **independently of `W_KAPPA`** — the longitudinal lever is itself lateral, and the goal-FIELD path (not the k penalty) is the culprit. A1's ADE is then not quotable as a driving win. |
| **R** | FAIL | ADE > 0.8892 | the longitudinal lever **does not transfer** off the penalty base: -0.1066 was `wk15`-specific, so the two levers are not additive and the `lonshift` gain was partly the penalty. |

**Controls that must hold or the reading is void** (all three checked in `raw/`):
1. `ha`, `ha0`, `ha0_ext`, `ol` ADE **bit-identical to `kamm07`'s** (0.8888 / 0.9251 / 0.8772 /
   0.8052) — they do not depend on the lever, so a difference means a different window grid.
2. `a_shift_used` present and **non-null** in the record, and `a_sustain_mode == "a0_shift"` — the
   tool's own `flag did not reach it` guard (`refav1_arm.py:1114`) must not have been silently skipped.
3. `kamm_mu == 0.7` still in the record: A1 must not have lost its constraint base.

---

## §2 — A2 the `W_KAPPA` MICRO-SWEEP: {1, 3, 7}, turn recall BESIDE ADE at every point

**Flags** = `ccos_argmax`'s command line with `--cost-weights 0.0,<W>,64.29715042415070`.
Endpoints are **already banked and cost nothing**: `W = 0` is `ccos_argmax`
(ADE 1.3272 / turn_L 0.3636) and `W = 15.11245` is `wk15` (ADE 0.8934 / turn_L 0.0000).
**Launch order `1, 3, 7`** — `W = 1` is the highest-information single arm, so a killed run still
answers the question.

### ⛔ COMMITTED READING — both outcomes written in advance

| # | outcome | reading | consequence |
|---|---|---|---|
| **K** | **A KNEE EXISTS** | some `W` in {1,3,7} has `turn_left` **> 0.0000** AND ADE <= 1.3272 - 0.1035 = **1.2237** | the penalty is a **tunable** lateral lever; the knee's `W` is the shippable weight and the sweep is extended around it. |
| **N** | ⛔ **NO KNEE — THE PENALTY IS RETIRED** | `turn_left` = 0.0000 at **every** `W` that buys ADE <= 1.2237 (equivalently: turning dies before the accuracy arrives) | ⭐ **the penalty is retired as a lateral lever.** The constraint (`kamm_mu`) and the goal-conditioned cost (§3) become the only admissible lateral levers, and every `W_KAPPA > 0` arm in the register is re-labelled *"accurate because it does not turn"*. |
| **M** | MIXED | `turn_left` > 0 at some `W` but no `W` reaches ADE 1.2237 | the penalty neither helps enough nor is cleanly refuted; report the curve and prefer §3. |

⭐ **Both outcomes are a real finding**, which is why the sweep runs even though N is expected:
§0.2 shows the TURN_L magnitude already at 26 % of command at `W = 15.11`, and the attenuation is
monotone in `W` over the three banked points, so a collapse well below 15 is the prior.

**Reported at every point, no exceptions:** ADE · curv MAE · **turn_left recall** · turn_right
recall · lane_keep recall · speed MAE · **and the `TURN_L med|k|max` from `kappa_by_goal.py`**, which
is the quantity that actually moves (§0.2) and which recall only thresholds.

---

## §3 — A3 ⭐ THE GOAL-CONDITIONED LATERAL COST (0-GPU to implement; the actual fix)

⛔ **THE DEFECT, RESTATED FROM THE MEASUREMENT IN §0.2, NOT FROM A HYPOTHESIS.** The k term is
`c = c + w_kappa * controls[..., 1].pow(2).mean(-1)` (`refa_v1.py`, the one site) — **one scalar
weight applied to every window regardless of what the tactical brain decoded**. It charges curvature
identically when the goal is `LANE_KEEP` and when the goal is `TURN_L`. MEASURED consequence: TURN_L
windows execute **0.02066** against a commanded **0.08000**.

**The change:** make `w_kappa` a function of the **decoded lateral goal token**.
`--w-kappa-by-goal <w_lane_keep>,<w_turn>[,<w_shift>]`; absent is the shipped path and is
**bit-identical** to every arm banked before today.

### ⛔ ADMISSIBILITY — the binding check, answered from SOURCE before a line of code

**The conditioning signal is `lat_i = argmax(self.lat_head(intent))`** (`refa_v1.py:2203-2212`) —
the model's own decoded tactical lateral **action** token.

1. ⭐ **"Could this have been computed from the SITUATION CLASSIFIER's output?" — NO, and it is not
   merely disjoint, it is a DIFFERENT LABEL FAMILY.** The situation classifier is
   `stack/tanitad/data/situations.py` and emits `lane_change` / `intersection` / `roundabout`.
   `lat_head` is supervised by `lat_label` from the **v7.2 label release**
   (`refa_v1.py:1988-2012`) over `TACTICAL_LAT_ACTIONS_V7` =
   `LANE_KEEP, LANE_CHANGE_L, LANE_CHANGE_R, ABORT_LC, NUDGE_L, NUDGE_R, TURN_L, TURN_R`
   (`vocab_v7.py:290`). **Literal probe with a same-breath control:**
   `grep -cF situations stack/scripts/s2_labels.py` = **0** against a control
   `grep -cF "def " ...` = **23** on the same file, so the channel read. No situation-classifier
   output, posterior, argmax, embedding or derivative enters this path.
2. ⭐⭐ **AND THE STRONGEST FORM: A3 ADDS NO INFORMATION AT ALL.** The decoded token is **already**
   consumed by the cost function — it builds `goal_t` (the imagined tactical field that IS the goal
   term, `refa_v1.py:2546`) and it is the iCEM **seed** (`goal_action["controls"]`,
   `refa_v1.py:2572`). A3 re-uses a signal the planner already reads; it opens **no new channel**,
   so there is nothing it could smuggle.
3. ⭐ **Inference stays VISION-ONLY.** `intent` comes from `_run_brains(pooled_win, nav_cmd)`
   (`refa_v1.py:1564-1589`): `pooled_win` is the vision feature window; in these arms `nav_cmd = 0`
   with `nav_valid = False` (the join line in every arm log). The only ego quantity anywhere in the
   path is **`v0` measured at t0**, legal under the PI ruling of 2026-09-02. **No future, no ego
   state beyond cycle time, no privileged channel.**
4. ⭐ **Attribution survives.** The goal path and the k weight are the same path by construction, so
   there is no second path to confound. The lever is one scalar per goal class and is recorded
   per-window in the arm record, so the claim is auditable from the artifact.

### The arms, and what each must read

| arm | `--w-kappa-by-goal` | role | committed reading |
|---|---|---|---|
| **`gk_off`** | (absent) | ⛔ **CONTROL THAT MUST READ A KNOWN VALUE** | must be **bit-identical to `wk15`** (ADE 0.8934, curv 0.03098, turn_L 0.0000). Any difference ⇒ the flag is not inert when off and NOTHING else in §3 is readable. |
| **`gkappa`** | `15.11245,0.0` | ⭐ **THE ARM** | ADE <= **1.2237** (better than `ccos_argmax` by more than the floor) **AND** `turn_left` recall **> 0.0000** **AND** `TURN_L med\|k\|max` **>= 0.06** (back toward the commanded 0.08000). |
| **`gkappa_inv`** | `0.0,15.11245` | ⛔ **DELIBERATE-REGRESSION ARM** | the INVERTED cost — free curvature on LANE_KEEP, fined on TURN. It **must be WORSE than `gkappa` on `turn_left` recall AND worse on ADE**. If the inverted arm is not worse, the goal conditioning is doing nothing and `gkappa`'s result is noise. |

**Prediction, committed:** `gkappa` recovers most of `wk15`'s ADE gain — which §0.2 localises to the
LANE_KEEP / GT-straight windows (`ccos_argmax` GT-straight `frac k!=0` **0.6667 -> 0.3333** under
`wk15`) — while restoring TURN_L to its commanded magnitude. **Target: ADE in [0.90, 1.10] with
`turn_left` >= 0.3636.** ⚠️ It is NOT predicted to beat `wk15`'s 0.8934: part of that number is bought
by not turning, and A3 gives that part back on purpose.

---

## §4 — WHAT WOULD MAKE ME WRONG

* If A2 finds a knee (outcome **K**), §3's premise weakens — the penalty would be tunable and the
  goal conditioning merely a better parameterisation of the same thing. **A2 therefore runs before
  §3's arms are believed**, not after.
* If `gk_off` is not bit-identical to `wk15`, every §3 number is void.
* If `gkappa_inv` is not worse than `gkappa`, the conditioning is inert and `gkappa` is noise.
* ⛔ **All of §1–§3 is ONE inference seed.** refav1's planner samples, and the floors in §0.3 are the
  only defence. Any A1/A3 arm that clears its bar by **less than its per-metric floor** is reported
  as NOT SEPARATED FROM SEED NOISE, and a seed replicate is the required next arm — not a claim.

---

## §5 — GPU DISCIPLINE

⛔ Thor and the dev box are both loaded. The dev-box ceiling is **MAX_ARMS = 2 arms**, counted by
`--dump-dir` (an arm is 2-4 `python.exe` entries; a gate on processes can never open).
⭐ A sibling queue `ta_queue3.py` (PIDs 2160 / 43140) holds a slot for the turn-asymmetry panel.
**This package's queue DEFERS to it**: it launches only when a slot is free **and** the sibling
queue's own launcher is no longer alive, so the two can never both take the same freed slot.
`OMP_NUM_THREADS=6` on every launch (7 concurrent arms once sat at 0-6 % GPU for 50 minutes).

---

## §6 — AMENDMENT, 01:35, WRITTEN BEFORE ANY A3 RESULT EXISTS

Two things changed after `raw/frontier.txt` was generated and before the A3 arms
could produce a number. Both are recorded here rather than in the RESULT, because
both change what the arms are read AGAINST.

### §6.1 — THE A3 ARMS RUN ON THOR, AND SO DOES THEIR BASELINE

The dev box is at its measured 2-arm ceiling (6737/8188 MiB) and a sibling queue
owns the next free slot. `queueTHOR.sh` has **exhausted its 7-arm plan and sits in
`wait`**, so two Thor slots are free and nothing contends for them.
⛔ **Cross-rig pairing stays inadmissible** — `queueTHOR.sh`'s own note: CEM on a
different GPU is not bit-reproducible. The A3 arms are therefore paired against
the **THOR-LOCAL `T_wk15`** (same ckpt, labels, episodes, window grid and
`--cost-weights` triple; `rec_T_wk15.json`), whose **seed replicate `T_wk15_s1`
also exists**. Pair and floor are both in-rig.
**Cross-rig sanity, quoted as a RIG control and never as a lever result:**
dev-box `wk15` 0.8934 vs Thor `T_wk15` 0.8910 — a 0.0024 difference, inside every
ADE seed floor in the table.

### §6.2 — ⭐⭐ THE SEED FLOOR IS A PROPERTY OF THE ARM, NOT OF THE RIG

MEASURED in `raw/frontier.txt`, three replicate pairs on the identical grid:

| pair | configuration | **ADE seed floor** |
|---|---|---|
| `ccos_argmax` / `ccos_seed1` | W_KAPPA **0** — search unconstrained | **0.06070** |
| `combined` / `combined_seed1` | W_KAPPA 0 + Kamm cap + ladder | **0.10350** |
| `T_wk15` / `T_wk15_s1` | W_KAPPA **15.11245** — curvature crushed | **0.00470** |

⛔ **A 22x spread.** The heavily penalised arm barely moves under a seed change
because the penalty has removed the search's freedom; the unconstrained arm moves
13x more. ⇒ **Quoting one arm's floor for a different arm's configuration is the
same scope error as reading `df` on a pod or `free` on Thor** — a true measurement
applied outside the thing it measured.

**Consequence, committed now:** `G_gkappa` sets W_KAPPA to **0 on TURN-goal
windows**, so it is *less* constrained than `T_wk15` and its own floor is **not**
0.00470. Until `G_gkappa` has its own seed replicate, its ADE bar is the
**unconstrained** neighbour's floor, **0.06070** — the conservative choice — and
any ADE claim smaller than that is reported as **NOT SEPARATED FROM SEED NOISE**,
with the replicate named as the required next arm.
⭐ `turn_left` / `turn_right` recall floors are **0.00000 on all THREE pairs**, so
the turning verdict does not depend on this at all.

### §6.3 — `gk_off` IS REPLACED BY A CHEAPER AND STRONGER **WITHIN-ARM** CONTROL

The SPEC's §3 control arm `gk_off` (the new code with the flag absent, expected
bit-identical to `wk15`) would cost a full GPU hour to prove something two cheaper
instruments already prove better:

1. **At the library level** — `stack/tests/test_refa_v1_goal_kappa_cost.py::
   test_a_off_is_bit_identical` plus **394 passing / 1 skipped** `refa_v1` tests
   on the edited tree. The flag-off path is pinned bit-identical, with a
   same-breath control that must differ.
2. ⭐ **On the REAL rig, for FREE — the within-arm control.** `G_gkappa` charges
   the **same 15.11245** on LANE_KEEP-goal windows that `T_wk15` charged there.
   ⇒ **`G_gkappa`'s LANE_KEEP-goal windows must reproduce `T_wk15`'s, and its
   TURN-goal windows must differ.** That is a stronger statement than `gk_off`:
   it proves the flag is inert exactly where it should be **and** active exactly
   where it should be, on the trained checkpoint, in one arm.
   ⛔ **If both columns differ, or neither does, §3 is void.**

### §6.4 — WHAT THE LIBRARY TEST ALREADY MEASURED, AND IT CONSTRAINS THE PREDICTION

⚠️ MEASURED while writing the test, and it is a caution the RESULT must carry: on
a TURN-goal window whose **canonical goal seed is already the argmin**, dropping
`w_kappa` to 0 changes the **objective** by exactly `dw * mean(kappa^2)` and
changes the **plan not at all**. The first draft of the control asserted on the
argmin and FAILED against a correct implementation.
⇒ **A null ARGMIN is not a null LEVER** — the mirror image of `M23`(4) (*a NULL
arm with a live optimiser is not a null*). The real-rig effect can only come from
windows where the penalty was making a **different** candidate win, and
`kappa_by_goal_all.txt` says those exist: `wk15` drove TURN_L to med|k|max 0.02066
against the seed's own 0.08000, so on those windows the seed was NOT winning.
