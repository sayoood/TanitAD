# H-REFAV1-SURFACE-1 — the paired delta and the cost-surface weight sweep

**status: PHASE 1 DELIVERED · PHASE 2 STAGE A DELIVERED · STAGE B ARMED AND SELF-COMPLETING — done: (§1) the decision-grade four-family paired bootstrap — the ccos refutation is UPHELD and the shipped cos planner's lateral row is a WINDOW-LEVEL identity with constant velocity; (SPEC) phase 2 pre-registered with the SELECT/SCORE split banked before any weight was computed; (§2) the 59-setting exhaustive screen at ZERO GPU — W_JERK is inert on 24 of 26 candidates and functions only as a switch that turns the SEARCH off, the decoded goal is directionally right (0.778 [0.625, 0.923], excludes chance) but commits to a turn on only 0.205 [0.128, 0.293] of the windows where GT turns, and cv wins the argmin on 0.507 of windows at ALL 59 settings so half the grid is unreachable by any re-weighting; (§3) Stage B armed on Thor behind the running chord regression arm, with the finishing sequence written out; (§4) the SCORE-half reference panel precomputed so the arm slots straight in; (§5) a second independent probe of the goal-recall finding — the goal head emits only 3 of its 8 lateral tokens and TURN_R 4.4x more often than TURN_L. / next: pull `dump_B1_puregoal` when `B1.DONE` appears (~3 h after the chord arm ends), run §3c, and read the answer against the outcomes committed in SPEC.md §5.**

Arch+Inference FlyWheel · 2026-09-05 · checkpoint `refav1-b1-v72-ep3-speed/ckpt.pt` step **21,109** ·
**T1** (self-action open loop) for every planner arm; **T0** for `ol` (world-model diagnostic only).
⛔ Nothing in this document is driving performance (PI ruling 2026-09-02: a planner feeding its own
predictor is still OPEN LOOP).

## The question the PI asked

> *"our goal is to let refav1 drive and prove the performance of WM based architectures and also our
> hierarchy architecture."*

Two open items from `D-REFAV1-CCOS-EVAL`:

1. **Phase 1 — the paired delta.** The `ccos` refutation rested on PER-ARM intervals plus a
   `paired_decision_grade` block carrying only **two** metrics (`ade_m`, `speed_mae_mps`). The
   decision-grade form is the PAIRED episode-cluster bootstrap **on every family separately**.
2. **Phase 2 — the weight sweep** (`H-REFAV1-SURFACE-1`). With the goal term audible (`ccos`
   compensated), is there a weight setting at which refav1's planner beats the trivial controls on
   the four families — or is the iCEM line dead?

⚠️ **Scope statement that travels with every number below:** the world model is SOUND and separately
banked (TURN_L +0.502 [+0.383, +0.579], TURN_R −0.387 [−0.498, −0.267], separated). Everything here
is a statement about the **PLANNER over that model**, never about the world model itself.

---

## 1. PHASE 1 — the decision-grade paired delta (zero GPU)

**Instrument:** `tools/paired_delta_refav1.py` (this stream). It takes N dumps at once, so every
pair is drawn from ONE set of per-window components; components come from
`refav1_arm._components` (four_families' own geometry + the canonical trajectory labeller), never
re-derived. Estimator `taniteval.ci.paired_episode_cluster_bootstrap`, **n_boot 2000**, over the
**141 episode clusters**. ⛔ No composite: every family is reported separately.

**Inputs — the three arms as banked, pulled from Thor by content (not re-run):**

| name | dump | cost metric | weights |
|---|---|---|---|
| `cos` | `/home/nvidia/refav1_ccos/dump_cos_ext` | `cos` (shipped) | shipped (0.02, 0.05, 0.1) |
| `ccos_naive` | `/home/nvidia/refav1_ccos/dump_ccos_naive` | `ccos` | shipped (0.02, 0.05, 0.1) |
| `ccos_comp` | `/home/nvidia/refav1_ccos/dump_ccos_comp` | `ccos` | **(12.859430, 32.148575, 64.297150)** |

⚠️ **The dev box also holds an independent `ccos_comp` replicate, and it is NOT a substitute.**
`C:\Users\Admin\ccos_eval\devbox\rec_ccos_comp.json` differs from the Thor record by up to **0.044**
on 570 shared scalars (worst: `mean_min_ttc_s` 23.4644 vs 23.4251, `along_final_bias_m` −0.6940 vs
−0.6606). Cross-box agreement is ulp-level for GT and the *floors* (`D-REFAV1-CCOS-EVAL` §5) but
**the searched `cl` arm is not ulp-reproducible across boxes**, so the paired delta uses the Thor
dumps that produced the banked headline numbers. *(That is itself a MEASURED fact worth carrying: a
planner arm is a cross-box REPLICATE, not a cross-box identity.)*

### 1a. Controls — all read their known values

| control | expected | read |
|---|---|---|
| **known-value** (`cos.cl − cos.cl`) | exactly 0.0000 [0, 0] on every metric | **PASS**, 0 of 14 metrics failing |
| **same windows** | `ws`, `v0`, `g`, `clip_index` bit-exact across all three dumps | asserted, held |
| **shared floors identical across dumps** | the floors do not depend on the cost metric | `ha` **0.0**, `ha0` **0.0**, `ol` **0.0** (bit-identical); `ha0_ext` **7.63e-06** — float32 ulp, because on the `cos` dump it was added post hoc by `refav1_add_floor.py` while the `ccos` arms computed it inline |
| **lead block** | present, with its n | `b1_eval_lead_block.npz`: **PRESENT**, LEAD **90**, NO_LEAD 63, NOT_STRAIGHT 108, NO_LABEL 21 |

### 1b. ⛔ THE HEADLINE — the refutation survives, and it hardens

**`ccos` compensated − `cos`** (T1 − T1, n = 282 / 141). **Bold = the interval excludes zero.**

| family | metric | Δ (ccos_comp − cos) | verdict |
|---|---|---|---|
| **ADE** | `ade_m` | **+0.1642 [+0.1075, +0.2282]** | separated WORSE |
| | `fde_m` | **+0.4467 [+0.2984, +0.6131]** | separated WORSE |
| **LONGITUDINAL** | speed MAE (m/s) | **+0.1763 [+0.1155, +0.2425]** | separated WORSE |
| | along MAE (m) | **+0.1264 [+0.0800, +0.1763]** | separated WORSE |
| | accel MAE (m/s²) | **+0.1848 [+0.1223, +0.2531]** | separated WORSE |
| | dist-keep headway min (m) | **+0.2786 [+0.0372, +0.6463]** (n = 87) | separated — the plan keeps MORE headway |
| | dist-keep time-gap min (s) | **+0.0483 [+0.0026, +0.1119]** (n = 81) | separated — larger gap |
| | dist-keep min TTC (s) | +0.5616 [−0.0365, +1.3938] (n = 87) | straddles |
| **LATERAL** | cross-track MAE (m) | **+0.0645 [+0.0211, +0.1193]** | separated WORSE |
| | heading MAE (°) | **+0.7940 [+0.2059, +1.4960]** (n = 272) | separated WORSE |
| | yaw-rate MAE (rad/s) | **+0.0159 [+0.0056, +0.0289]** | separated WORSE |
| **TACTICAL** | traj lat correct | −0.0213 [−0.0567, +0.0071] | straddles |
| | traj lon correct | **−0.1028 [−0.1596, −0.0496]** | separated WORSE |
| **STRATEGIC** | — | **UNAVAILABLE, n = 0** | see §1e |

⇒ **Nine of the thirteen available metrics are separated worse; two straddle; the only two that
move in `ccos`'s favour are distance-keeping surrogates** — and those are not error metrics (§1d).
`D-REFAV1-CCOS-ARMS` said *"the direction is not in doubt"*. The paired test agrees and is stronger:
**the refutation stands on the decision-grade estimator, on every family separately, not only on
ADE.**

### 1c. Against the trivial controls — and the fingerprint, now proven at window level

| pair | ADE | LON speed | LAT cross | TAC lon |
|---|---|---|---|---|
| `cos − ha` | +0.0083 [−0.0574, +0.0703] | **+0.2706 [+0.2140, +0.3312]** | **−0.1569 [−0.2131, −0.1072]** | **−0.1277 [−0.1879, −0.0674]** |
| `cos − ha0` | **+0.0158 [+0.0007, +0.0315]** | **+0.0308 [+0.0049, +0.0585]** | **0.0000 [0, 0]** | −0.0035 [−0.0248, +0.0213] |
| `cos − ha0_ext` | +0.0267 [−0.0285, +0.0813] | **+0.2706 [+0.2140, +0.3312]** | **−0.1409 [−0.1844, −0.1015]** | **−0.1277 [−0.1879, −0.0674]** |
| `ccos_comp − ha` | **+0.1725 [+0.0902, +0.2585]** | **+0.4469 [+0.3707, +0.5236]** | **−0.0924 [−0.1649, −0.0170]** | **−0.2305 [−0.2980, −0.1667]** |
| `ccos_comp − ha0` | **+0.1800 [+0.1188, +0.2456]** | **+0.2071 [+0.1401, +0.2781]** | **+0.0645 [+0.0211, +0.1193]** | **−0.1064 [−0.1667, −0.0496]** |
| `ccos_comp − ha0_ext` | **+0.1909 [+0.1119, +0.2724]** | **+0.4469 [+0.3707, +0.5236]** | **−0.0764 [−0.1398, −0.0078]** | **−0.2305 [−0.2980, −0.1667]** |
| `ccos_naive − ccos_comp` | **+0.2152 [+0.1022, +0.3530]** | **−0.0361 [−0.0667, −0.0045]** | **+0.2375 [+0.1245, +0.3751]** | +0.0177 [+0.0000, +0.0390] |

⭐ **`cos − ha0` on the LATERAL family reads exactly 0.0000 with a ZERO-WIDTH interval on all three
metrics — and on `TAC_traj_lat_correct` too.** `D-REFAV1-CCOS-EVAL` called the bit-identical lateral
row "the fingerprint" from four-decimal means. The paired estimator upgrades that from a coincidence
of means to a **window-level identity**: on all 282 windows the shipped planner's lateral output IS
the constant-velocity rollout. There is nothing left to attribute to lateral planning under `cos`.

⚠️ **The compensated arm does not simply lose everywhere.** `ccos_comp − ha0_ext` is separated
**worse** on ADE, on all three core LON metrics and on TAC-lon, but separated **better** on LAT
cross-track (−0.0764) and yaw-rate (−0.0250), with heading straddling. `ha0_ext` holds the recorded
κ₀, which drifts; the compensated planner beats that particular floor laterally while losing to it
longitudinally. **No arm here beats any floor on all four families.**

⭐ **Weight compensation is a real effect, paired-confirmed.** `ccos_naive − ccos_comp` is separated
on ADE (+0.2152), FDE, all three LAT metrics and TAC-lat: compensating the implicit re-weight
recovers a measurable part of the naive flip's damage. It does not recover enough to reach any floor.

### 1d. ⚠️ How to read the distance-keeping rows (and why they are not a win)

`distance_keeping` is computed **on each arm's own predicted path against the lead track** — it is a
safety surrogate, **not an error against ground truth**. A paired delta on it says *"arm B's plan
runs X m further from the lead than arm A's"*, which is only good if the plan is otherwise right.
The straight-line `cos` plan reads **−0.7878 m headway** and **−1.8913 s min-TTC** against `ha`
(both separated): a plan that ignores the lead runs closer to it. `ccos_comp` reads *more* headway
than `cos` largely because it is slower and wanders. **The GT's own headway/TTC is the reference
that makes these rows interpretable, and it is the immediate next work item.**

### 1e. STRATEGIC — UNAVAILABLE, with its reason and n

**n = 0.** No route/goal channel exists in this eval surface. The grid is evaluated `--no-navshuf`
on the v7.2 EVAL labels, and the planner's own strategic input is the **tactical head's imagined
goal token** — model output, not a route label — so scoring the plan against it would score the
model against itself. PhysicalAI-AV ships no map, lane graph, junction annotation or route signal
(CLAUDE.md's read-set table), so no external reference exists either. **This is a WORK ITEM** (the
strategic reference must come from AlpaSim's `map.xodr` or an external corpus), **not a pass and not
an omission.**

### 1f. Verdict on `D-REFAV1-CCOS-ARMS`

**The refutation SURVIVES the paired test and is strengthened.** No amendment to the row's status is
warranted; it is annotated with the paired numbers instead. `ccos` must not become the default; the
shipped `cos` stays.

⚠️ **But `cos` is not thereby vindicated.** Against `ha` and `ha0_ext` its ADE straddles zero, its
longitudinal family is separated WORSE, its tactical-longitudinal is separated WORSE, and its
lateral family is *bit-identical to constant velocity*. The shipped planner ties the trivial
controls by **being** one of them.

**Artifacts:** `raw/paired_delta_phase1.json`, `raw/paired_delta_phase1.md`,
`tools/paired_delta_refav1.py`.

---

*(§2 lands below; Stage B arms follow when GPU frees)*

---

## 2. PHASE 2, Stage A — the exhaustive weight screen (zero GPU) and the goal-quality read

**Pre-registered in `SPEC.md` before this ran; the SELECT/SCORE split was banked in the same
commit.** Instrument `tools/surface_screen2.py`, raw `raw/screen_stageA2.json` /
`raw/screen_stageA2.md`, `raw/goal_quality_ci.json`.

The planner's own closure `total(n) = goal(n) + W_JERK·jerk_raw(n) + W_KAPPA·kap_raw(n)`
reproduces on the banked box panel to **1.6e-08**, so `argmin_n total(n)` at any weight triple is
exact arithmetic on already-measured fields. **59 weight settings** were evaluated (two
one-variable log sweeps ×{0, 1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 1e3, 1e6}, a 5×5 joint grid, shipped,
compensated, and the regression arm) at **zero GPU cost**.

⛔ **What Stage A may not conclude, stated in the SPEC and repeated here:** the box is **26**
injected/seeded candidates while the real planner searches **300 samples × 30 iterations**. This is
a SCREEN. The committed outcomes are decided by Stage B.

### 2a. ⛔ THE DELIBERATE REGRESSION FAILED FIRST — and the reason is a finding

The pre-registered regression arm was `W_JERK × 1e4`, expected to force `cv` on ~100 % of windows.
**It did not move a single decision.** The reason, MEASURED:

> **24 of the 26 candidates have `jerk_raw` identically 0.** Only `seed0` (median 0, max 0.280) and
> `proposal` (median **2.471**, max 6.197) carry any jerk at all.

⇒ **`W_JERK` cannot re-rank 24 of the 26 candidates, at any magnitude.** Its ONLY function on this
surface is to penalise the **iCEM proposal** — the *searched* plan — and, weakly, the seed. At the
compensated `W_JERK` = 12.86 the proposal's median penalty is **31.8**, against a goal term whose
whole range is [0, 2]. At the *shipped* 0.02 it is 0.049, against a `cos` goal term of ~1e-7.

⭐ **This re-reads the ranked suspect in `H-REFAV1-SURFACE-1`.** `W_JERK` is not "dominating a goal
term that can now compete" across the candidate set — it is a **switch that turns the SEARCH off**.
The `proposal` wins **0.0 %** of windows at every setting tested except the fully unpenalised corner
(2.1 %). ⚠️ Correspondingly, *no* value of `W_JERK` was found that lets the search win.

The regression arm was re-designed for this instrument (`W_KAPPA × 1e6`, which must drive κ ≡ 0) and
**PASSES**: `frac_plan_turns` **0.0000**. *(Both the failure and the re-design are recorded rather
than quietly swapped: the original arm was mis-specified for the box, and finding that out is what
the control is for.)*

### 2b. ⭐⭐ THE GOAL IS PRECISE AND HAS ALMOST NO RECALL — and no weight can fix that

This block involves **no weights at all**. It compares the decoded goal's own canonical curvature
(`seed0`'s κ, i.e. what the tactical head asks for) against the GT trajectory's curvature.
Episode-cluster bootstrap, n_boot 2000.

| quantity | value [95 % CI] | n windows / clusters |
|---|---|---|
| GT is actually turning | 0.4681 [0.4043, 0.5355] | 282 / 141 |
| **the decoded goal proposes a turn WHEN GT TURNS** | **0.2045 [0.1280, 0.2932]** | 132 / 92 |
| the decoded goal proposes a turn when GT is straight | 0.0733 [0.0347, 0.1169] | 150 / 101 |
| **the goal's DIRECTION is correct, given both turn** | **0.7778 [0.6249, 0.9231]** | 27 / 21 |

⇒ **When the goal commits to a turn it is right — 77.8 %, and the interval excludes chance (0.5).
But it commits on only 20.5 % of the windows where the car actually turns.** The failure is
**recall, not direction**. On the other ~79 % the goal *is* the straight/zero-action rollout, and
there the cost comparison has nothing to prefer: `cv`'s share of the argmin is **0.507 at every one
of the 59 weight settings**, unchanged from zero penalty to 1e6× — because on the HOLD stratum every
candidate scores exactly 1.0 (banked: ptp 0.0 on 133/133).

⛔ **Half of the grid is unreachable by ANY re-weighting, by construction.** That is not a
hypothesis; it is arithmetic on the banked panel.

### 2c. The weight screen — what the surface can and cannot buy

Selected rows (SELECT half, n = 140; the full 59-row table is in `raw/screen_stageA2.md`):

| setting | W_JERK | W_KAPPA | plan turns | turns when GT turns | sign correct given both turn (n) | follows goal sign | `proposal` share |
|---|---|---|---|---|---|---|---|
| shipped | 0.02 | 0.05 | 0.186 | 0.271 | 0.579 (19) | 0.864 | 0.000 |
| compensated | 12.86 | 32.15 | 0.129 | 0.186 | 0.846 (13) | 0.818 | 0.000 |
| `W_KAPPA` → 0 | 12.86 | 0 | 0.293 | 0.400 | 0.571 (28) | 0.864 | 0.000 |
| **both → 0 (pure goal)** | **0** | **0** | **0.300** | **0.414** | **0.586 (29)** | **0.909** | **0.021** |
| regression `W_KAPPA` × 1e6 ⛔ | 12.86 | 3.2e7 | 0.000 | 0.000 | — (0) | 0.000 | 0.000 |

Three things this says, none of which needed a GPU:

1. **The weights trade turn RATE against turn CORRECTNESS, and the product barely moves.** The rate
   of *correct* turns on GT-turning windows (`turns when GT turns` × `sign correct`) is **0.157** at
   shipped, **0.157** at compensated, and **0.243** at zero penalty. Removing the entire penalty
   budget buys **0.157 → 0.243**; it never approaches 1.
2. **The `W_JERK` axis is flat.** Nine settings spanning **1e13** in magnitude give an identical
   turn rate (0.129) and identical sign correctness (0.846). Only its effect on `seed0`'s share
   (0.171 → 0.000) moves at all.
3. **`W_KAPPA` is the only live axis, and only below ~0.3.** Between 0.32 and 3.2e4 nothing changes;
   above that everything is straight; below it the extra turns bought are mostly the *wrong*
   direction (sign correctness falls 0.846 → 0.571).

### 2d. What Stage A hands to Stage B

The SPEC's selection rule (maximise turn-goal κ-sign agreement on SELECT, subject to
`frac_nontrivial ≥ 0.10`) picks **`W_JERK = 0, W_KAPPA = 0`** — the **pure-goal surface**, agreement
**0.909** on SELECT. That is also the setting maximising the correct-turn rate, so the rule and the
alternative reading agree.

⚠️ **A zero-penalty arm is not obviously safe in the REAL planner**, and that is precisely why it
must be run rather than assumed: the box holds only tame designed candidates, while the live search
draws 300 coloured-noise samples per iteration with nothing penalising them. Stage B measures what
that does to the four families. ⛔ The screen cannot answer it.

**Stage B is prepared and blocked on GPU only** — Thor is running the `chord` deliberate-regression
arm (B3, 42 min in of ~3 h at the time of writing) and the dev-box RTX 4060 is running a sibling's
withheld-bank panel. Neither is displaced. The SCORE-only episode and cache views
(`/home/nvidia/refav1_ccos/score_{eps,cache}`, 71 + 71 symlinks, verified) are already built so B1
and B2 start the moment Thor frees.

---

## 3. HANDOFF — Stage B is armed on Thor and finishes without this session

⭐ **Nothing below needs re-deriving. The launcher is running; a successor pulls and analyses.**

### 3a. What is running right now

| what | where | state at hand-off |
|---|---|---|
| **B3 `chord_shipped`** ⛔ deliberate regression | Thor, PID 3052445 | 47/141 episodes at 59 min ⇒ ~3 h total. Writes `/home/nvidia/refav1_ccos/{dump_chord_shipped, rec_chord_shipped.json}` |
| **B1 + B2 launcher** | Thor, PID 3058070, `thor_stageB.sh` (md5 `a3d47399e6f07c11b6477a1eecc0d70c`, verified both ends, `bash -n` clean) | waiting on PID 3052445 **by explicit PID** — never `pgrep -f`, which self-matches. Log `/home/nvidia/refav1_ccos/stageB.log` |
| sibling panel | dev-box RTX 4060 | not displaced |

**The launcher's own guards:** it sleeps 60 s after the chord PID disappears, then counts live arms
with a bracketed pattern and emits an **opaque marker** (`ZZOTHERARMS-<n>ZZ`) so a client-side
filter can never match its own command text; it **refuses (exit 3)** if any arm is running.

### 3b. What B1/B2 is

One pass, `--with-oracle-goal-arm`, so both arms come out of it:

* **B1 `cl`** — **T1** — `--cost-metric ccos --cost-weights 0.0,0.0,64.29715042415070`, the
  pure-goal surface chosen by the SPEC's pre-registered rule on the SELECT half.
* **B2 `cl_oraclegoal`** — ⛔ **T0 ORACLE**, the true future field as the planning goal, stamped
  T0 by the tool itself (`refav1_arm.py:188`). **An upper bound. Never a deployable arm.**
* Run on the **SCORE half only** — `/home/nvidia/refav1_ccos/score_{cache,eps}`, **71 + 71**
  symlinks built and verified (`ZZ71-71ZZ`), plus `index.json`. The weights were chosen on
  SELECT, so this is the held-out read the SPEC committed to.
* Outputs `/home/nvidia/refav1_ccos/{dump_B1_puregoal, rec_B1_puregoal.json, B1.EXIT, B1.DONE}`.

⚠️ **Expect ~3 h** (71 episodes × 2 plan arms ≈ one full-grid single-arm run).

### 3c. The exact finishing sequence (zero GPU, ~10 min)

```bash
# 1. is it done?  (opaque markers; never grep the raw stream for your own pattern)
ssh -n tanitad-thor 'echo "ZZ$(cat /home/nvidia/refav1_ccos/B1.EXIT 2>/dev/null)ZZ"; \
  echo "ZZEPS-$(ls /home/nvidia/refav1_ccos/dump_B1_puregoal/ep*.npz 2>/dev/null | wc -l)ZZ"'

# 2. pull (tiny — the ccos dumps were 2.3 MB each)
cd C:/Users/Admin/refav1_sweep/dumps
ssh -n tanitad-thor 'cd /home/nvidia/refav1_ccos && tar cf - dump_B1_puregoal | gzip -1' > b1.tgz
tar xzf b1.tgz
ssh -n tanitad-thor 'cat /home/nvidia/refav1_ccos/rec_B1_puregoal.json' > ../rec/rec_B1_puregoal.json

# 3. the four families, paired, against the banked floors AND against cos, on the SCORE windows
#    (the tool refuses unless ws/v0/g/clip_index are bit-exact, so it will REFUSE a cos dump on
#    the full grid — slice cos to the SCORE episodes first, or pair only inside the B1 dump,
#    which already carries ha / ha0 / ha0_ext / ol on those same windows)
python "<repo>/TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refav1-surface-sweep/tools/paired_delta_refav1.py" \
  --dump B1=C:/Users/Admin/refav1_sweep/dumps/dump_B1_puregoal \
  --arm cl --lead-block C:/Users/Admin/refav1_sweep/b1_eval_lead_block.npz \
  --stack C:/Users/Admin/tanitad-wt/stack --taniteval C:/Users/Admin/tanitad-wt/taniteval \
  --n-boot 2000 --out raw/paired_B1.json --md raw/paired_B1.md
#    …then again with --arm cl_oraclegoal for B2 (⛔ stamp every B2 number T0).

# 4. echo gate 1 + criteria checker on the record
python "<ccos-eval>/tools/run_echo_gate.py"   # references ha, ha0_ext (required), ha0
python "<repo>/tools/criteria_check.py" raw/rec_B1_puregoal.json
```

### 3d. How to read the answer — the outcomes were committed in `SPEC.md` §5, before the data

| what B1/B2 shows on the SCORE half | the committed verdict |
|---|---|
| B1's four families beat `ha0_ext` paired-separated, echo gate holds | **the surface IS the lever** — report the setting |
| B1 does not, **but the T0 oracle-goal B2 does** | **the GOAL, not the weights, is binding** — refutes `H-REFAV1-SURFACE-1`, and the next work is the goal source |
| **neither** | ⛔ **the iCEM planner over this world model is not repairable by re-weighting or re-goaling — the line closes.** State it plainly; it is a real result about the PLANNER, and the world model underneath it is sound |
| B1 wins some families and loses others | ⛔ **NOT a success.** Pre-declared in `SPEC.md` §5 so it cannot be reported as one |

⭐ **Stage A already tilts this**, without spending the GPU: the goal commits to a turn on only
**20.5 %** of the windows where the car turns, and `cv` wins the argmin on **50.7 %** of windows at
**every one of the 59** weight settings. If that survives into B1's four families while B2's oracle
goal moves them, the answer is outcome 2 — and the programme's next question is the tactical head's
goal recall, not the cost weights.

⚠️ **And the standing scope statement applies to whatever comes out:** this is the PLANNER over the
world model. The world model responds correctly to turn actions (TURN_L +0.502 [+0.383, +0.579],
TURN_R −0.387 [−0.498, −0.267], separated, banked). **No verdict here may be restated as a verdict
about the world model.**

### 3e. Two defects found in passing, escalated rather than worked around

1. ⛔ **`Research/2026-09-05-refav1-ccos-eval/tools/insert_rows.py` can MANGLE
   `GOALS_AND_CLAIMS.md`.** It picks its line ending with `"\r\n" if "\r\n" in s else "\n"`. The
   register is **MIXED** — MEASURED 2026-09-05: **2,776 LF and 24 CR** — so the test is True and the
   split yields **25 "lines"** for a 2,776-line file; a write would re-join the whole register on
   CRLF. `tools/register_insert.py` here splits on LF only, which round-trips the bytes whatever the
   mix. **The predecessor's tool should be fixed or retired.**
2. ⚠️ **A searched planner arm is a cross-box REPLICATE, not an identity** (§1). Any future arm
   comparison must use dumps from the SAME box, or state the cross-box delta.

---

## 4. The SCORE-half reference panel — computed NOW so Stage B slots straight into it

Stage B runs on the **SCORE** half (71 episodes / 142 windows), while every banked arm runs the
full 141/282 grid. `paired_delta_refav1.py` correctly **refuses** to pair dumps whose
`ws`/`v0`/`g`/`clip_index` are not bit-exact, so the banked arms have been sliced to the same half
in advance (`tools/slice_dump_to_split.py`; the only field rewritten is `clip_index`, renumbered
0..70 to match a 71-episode loader, and the slice is stamped `sliced_from` in its manifest —
trajectories, floors and decisions are the originals).

**Result: the phase-1 conclusions reproduce on the held-out half.** Known-value control PASS;
`raw/paired_SCORE_reference.json` / `.md`, n = 142 / 71, n_boot 2000.

| pair (SCORE half) | ADE | LON speed | LAT cross | TAC lon |
|---|---|---|---|---|
| `ccos_comp − cos` | **+0.1860 [+0.1066, +0.2909]** | separated worse | **+0.0832 [+0.0150, +0.1732]** | separated worse |
| `ccos_naive − ccos_comp` | **+0.2902 [+0.1061, +0.5111]** | — | **+0.3181 [+0.1342, +0.5430]** | — |
| `cos − ha` | −0.0409 [−0.1259, +0.0377] | separated worse | **−0.1736 [−0.2536, −0.1088]** | separated worse |
| `cos − ha0` | **+0.0169 [+0.0024, +0.0389]** | separated worse | **0.0000 [0, 0]** | straddles |
| `cos − ha0_ext` | −0.0145 [−0.0866, +0.0593] | separated worse | **−0.1499 [−0.2123, −0.0979]** | separated worse |
| `ccos_comp − ha0_ext` | (see raw) | separated worse | −0.0668 [−0.1665, +0.0423] | separated worse |

⭐ **The lateral identity survives the split**: `cos − ha0` reads exactly **0.0000 with a zero-width
interval** on all three lateral metrics on the held-out half as well. It is not a property of the
particular 282 windows; it is what the shipped planner does.

⇒ **B1 and B2 are compared against THIS table**, on the same windows, with the same estimator. The
comparison is already set up; only the arm is missing.

---

## 5. ⭐ The goal head's lateral vocabulary is three tokens wide, and lopsided

A **second, independent probe** of §2b's goal-recall finding — a different mechanism, as the
absence rule requires: §2b read the goal's own *canonical control* (`seed0`'s κ from the box
panel); this reads the **decoded token** in the arm's `decisions/*.npz`. They agree exactly
(**38** turn goals by both routes), and the token histogram adds something the κ probe could not
see.

Over all 282 windows, `goal_lat_cl` (the goal the planner was actually given; `goal_source_cl` is
**`tactical_imagined` on 282/282**, i.e. every goal is the tactical head's imagination):

| token | v7.2 lateral vocabulary (`TACTICAL_LAT_ACTIONS_V7`) | count |
|---|---|---|
| 0 | `LANE_KEEP` | **244** |
| 6 | `TURN_L` | **7** |
| 7 | `TURN_R` | **31** |
| 1–5 | `LANE_CHANGE_L`, `LANE_CHANGE_R`, `ABORT_LC`, `NUDGE_L`, `NUDGE_R` | **0** |

Two things follow, neither of which any cost weight can touch:

1. **Five of the eight lateral tokens are never emitted at all** on this grid — including both
   lane-change tokens and both nudges. The goal head's effective lateral vocabulary is
   `{LANE_KEEP, TURN_L, TURN_R}`.
2. ⚠️ **`TURN_R` is emitted 4.4× more often than `TURN_L`** (31 vs 7). GT turns on 132 of 282
   windows, so this is not a corpus in which right turns outnumber left ones 4:1. Combined with
   §2b — the goal is *directionally right* 0.7778 [0.6249, 0.9231] when both turn, on n = 27 — the
   honest reading is that the direction signal is real but the emission is **both sparse and
   skewed**, and the n available to test the skew is small.

⇒ **This is where the next work is**, and it is a statement about the **goal head**, not the cost
surface and not the world model. It is exactly what Stage B's T0 oracle-goal arm is pre-registered
to bound: if replacing the goal moves the four families while the whole 59-setting weight grid does
not, the binding constraint is the goal's recall and its lateral vocabulary collapse.

⚠️ **Evidence class:** MEASURED (ours), T1 grid, artifact
`dumps/dump_cos_ext/decisions/ep*.npz` and `raw/screen_stageA2.json`. Token names resolved from
`stack/tanitad/models/vocab_v7.py:290` — index 6 `TURN_L`, index 7 `TURN_R`, matching the
`refav1_stratified_read.py` convention this stream inherited.
