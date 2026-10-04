# SPEC_REFCV8 — DRAFT pre-registration of the refcv8 run (binds NOTHING until the Master Mind registers it)

**Status: DRAFT.** Written 2026-10-04 ~21:20 Berlin (19:20Z) by a design agent for the Master Mind. It binds nothing until
the Master Mind registers it: the sha256 of the registered file and the UTC time, written **before any refcv8 number
exists**. At the time of writing no refcv8 checkpoint, no refcv8 training step and no refcv8 metric exists. The only
refcv8-labelled measurements are instrument and warm-start checks on refcv7 weights (cited where used).

**Authority and inputs.** `Project Steering/PLAN_REFCV8.md` (requirements R8-1…R8-7, X1–X10, the metric set §5, the
ladder §6, decisions 1–11) and `Project Steering/SPEC_REFCV7.md` (structural precedent). Every number in this file
carries an evidence class and a tier, and its artifact is listed in `SPEC_REFCV8_DRAFT_SOURCES.md`.

**What refcv7 data was visible when the bars were written.** Every refcv7 baseline quoted below was visible. The bars
are therefore NOT blind to refcv7. §14 lists, bar by bar, which refcv7 value was visible when the bar was chosen
(the calibration disclosure). No bar can have been chosen after seeing refcv8 data, because none exists.

---

## 0. Reading order

1. §1 (what refcv8 is) and §2 (rules common to every row). They fix the windows, the estimator and the stamps.
2. §3 (warm-start identity, launch gate, required flags). Nothing trains until §3 holds.
3. §4 (R8-1…R8-4): the PI's tactical and route requirements. These are the deciding rows.
4. §5 (four metric families) and §6 (NavSim).
5. §7 (R8-5 / R8-6, perception) and §8 (R8-7 and X1–X10).
6. §9 (training-variance rule), §10 (decided by pending evidence), §11 (open PI decisions).
7. §12 (amendment protocol), §13 (open questions for the Master Mind), §14 (calibration disclosure).

Every requirement row has the same seven fields:
* **(a)** the measure, defined operationally: function, windows, tier;
* **(b)** the refcv7-50,400 baseline, with its evidence class and artifact, or `PENDING: <artifact that will produce it>`;
* **(c)** the BAR, as a literal;
* **(d)** the estimator and the replicate rule;
* **(e)** the control that must read a known value;
* **(f)** the deliberate-regression arm that must FAIL;
* **(g)** the outcome table: what PASS means, what FAIL means, and the next arm on FAIL (Rule Zero).

---

## 1. What refcv8 is

**Object.** One training run, `refcv8-r101-s0`, **warm-started** from `refcv7-r101-s0` at step 50,400. PLAN §0.1 makes
this the default. §10.2 fixes the rule that could still turn it into a branch.
* Every module whose shape is unchanged loads strictly from refcv7-50,400: trunk, lift, BEV, existing heads.
* Every NEW path starts zero-gated, so step 0 reproduces refcv7-50,400 (§3.1).
* Fresh optimiser state for every tensor (DESIGN §3.7; `--init-from`, `refcv8_train.warm_start_from`).

**What changes relative to refcv7** (each item is a row below):

| component | source | row |
|---|---|---|
| v9 per-frame labels: tactical actions + goals over [NOW+2 s, NOW+8 s] with constraints; per-frame nav; route checkpoint input | WP-A release (train md5 `f63ece410b725febb8a5242cf2b01d3c`, eval139 md5 `6b5c7f207cffc3b7eb3cd527fd433599`) | R8-1, R8-2, R8-3 |
| tactical conditioning of fan generation and selection; listwise selector; constraint heads | WP-B `code/fix/` overlay, registered `SPEC_WPB.md` (sha256 `c952d4b4…`) | R8-4, X1 |
| perception fixes F1 / F4 / F4b / ego-box mask / track-id-switch mask / z-h trust | WP-C, opt-in modules | R8-5 |
| perception architecture levers B1–B4, M1, M5, M6, W1–W7 | WP-D design; entry decided by probes (§10.1) | R8-6 |
| residual-prior dropout / prior-free group; past-only speed input; loss budget; label-state isolation; vocabulary; distance keeping; NavSim bridge; pose timing | Master Mind list | X1–X10 |

**What does NOT change:** the corpus (refcv7's 4,369 train clips and the 139 eval clips, unless PI decision 8 says
otherwise, §11), the 416×1024 cylindrical front camera, the ResNet-101 trunk, the 117 anchors, the 10 cm map head
extent (100 m × ±30 m), the box head (A9 + `learned_ref`), and the strategic layer (OFF, PI ruling R5).

**Tier of every refcv8 number in this SPEC** (EVAL_DOCTRINE):
* **T1 self-action OPEN LOOP** on logged eval139 frames for every planner, route and four-family number. The
  status of T1 for an action-free model stays **UNRULED**, as for refcv3–refcv7.
* **T0** (GT-nearest anchor / oracle-117) is a ceiling diagnostic only and is never compared with a T1 number.
* **OPEN-LOOP PERCEPTION DIAGNOSTIC** for every map and box number.
* **NavSim open-loop benchmark** (T1-family, zero-shot PhysicalAI → nuPlan cameras, non-parity) for §6.
* No number in this SPEC is a closed-loop or driving-performance claim.

---

## 2. Rules common to every row

### 2.1 Eval surfaces (windows), fixed now

| id | windows | used for | why |
|---|---|---|---|
| **S-ROUTE** (PRIMARY for route, selection, tactical, controllability) | the A6 / R1 dense capture: **4,634** eval139 windows = **every** GT-turn window (**2,317**: 849 left / 1,468 right, from 43 episodes) + 2,317 seeded others (seed 0). Window list: route package `raw/a6_window_list.json` | R8-2, R8-3, R8-4, X1, X2 | every turn window of the 139 held-out episodes, so turn rates use all turn evidence the eval set holds |
| **S-GRID** (continuity, no bar) | the route package's 1,112-window EVAL-DIAG grid (107 GT-turn windows) | every S-ROUTE row is also reported here | the PI's quoted refcv7 values (0.84 / 0.51) live on this grid |
| **S-V9** | every eval139 window with a v9 lateral AND longitudinal label (94.57 % of 23,772; the 136 tactically scored clips) restricted to S-ROUTE | R8-1, R8-4 (i), X5 | the dense label population |
| **S2** | the battery's 2-s grid: **4,754** windows / 139 episodes (`battery/SPEC.md` §3.1); **S6** = its 6-s subset | §5 four-family non-regression | continuity with every refcv7 BAR-R7 |
| **S-PERC** | EVAL-DIAG: 1,112 windows / 139 episodes (map, 137 clips with `/3` GT); 1,061 labelled windows / 137 episodes (boxes) | R8-5, R8-6 | the WP-C / WP-D diagnostics surface; every refcv7 perception baseline is on it |
| **NAVSIM** | navtest (12,146 tokens, FULL), navhard two-stage (5,912 scorer tokens), warmup (204 scenes) | §6, X6 | the refcv7 NavSim suite |

GT-turn / GT-straight classes on S-ROUTE and S-GRID follow the route package's SPEC §3 rule (terminal heading class,
τ 10.35°; GT-turn = |terminal heading| ≥ 30°). They are model-free and fixed.

### 2.2 Estimator, and the three variance questions

* **Point estimate:** the FULL-SET pooled mean over windows (rates: pooled ratio). Never a mean of split-means.
* **Interval:** `taniteval.ci.paired_episode_cluster_bootstrap`, cluster = eval episode (clip), B = 2,000, seed 0,
  95 % percentile, refcv8 vs refcv7-50,400 on the **same windows** and the **same sampler seed**. NavSim: the paired
  log-cluster bootstrap of the NavSim suite (`paired_navsim_log_cluster_bootstrap`, log_name AND nuplan_drive).
  ⛔ `overlapping_holdout_se` appears nowhere.
* **Which question each interval answers** (CLAUDE.md; name it beside every interval):
  1. *another draw of EPISODES?* — the bootstrap above.
  2. *another INFERENCE run?* — refcv8 samples (DDIM ε at eval, plus allocation noise). **Sampler seeds 0 and 1 for
     every planner row; every planner bar must hold at BOTH.** The inference floor is `|m(s0) − m(s1)|` per metric.
  3. *another TRAINING run?* — not answerable from one run. §9 fixes what may be claimed.
* **NOT PROVEN rule** (SPEC_REFCV7 §3, unchanged): a bar that is met and separated in the right direction, but
  whose margin is within **2×** the inference floor of that metric, reads **NOT PROVEN**, never PASS.
* ⛔ A missed bar is **FAILED**. No goalpost moves after data. A FAILED row names its next arm (Rule Zero).

### 2.3 Stamps every refcv8 number carries

1. Its tier (§1) and its surface (§2.1) with n (windows, episodes).
2. *"single training seed; training-seed floor at full scale unmeasured"* (§9).
3. **Privileged inputs, named** (GATE_PROTOCOL §0.8): on PhysicalAI the per-frame nav and the route checkpoint are
   derived from the ego's own future path. They are optimistic by construction. Every route number is quoted beside
   its **RC-OFF** and **LEGAL** rows (§4.3, §6).
4. The speed input: which channel was fed (N2 / N3 / the v8 future-max sidecar / unknown), §8 X3.
5. The decision rule for map and box numbers (F1 thresholds, F4 gates, F4b NMS radius), refitted on refcv8's own
   TRAIN-DIAG pass (WP-C I4), never on eval.

### 2.4 Admissibility (binding rulings applied to refcv8)

* **Vision-only inference** (PI 2026-08-03): labels may use ego, other agents, maps and future poses; inference
  receives vision + the admitted inputs only: measured v0 and past ego history at cycle time (PI 2026-09-02), the
  supplied nav, the route checkpoint (PI R8-3), and the speed limit (PI R1, 2026-09-27).
* **Goal input ≠ situation classifier** (PI 2026-08-03): no situation-classifier output (TL tokens, YIELD, tac_SIT)
  enters the route checkpoint, the per-candidate tags φ_k, the allocation or the selection terms. WP-B pins this by
  `assert_no_situation_feed` and an interventional test (the RC tensor is bit-identical under a forced tactical
  posterior). The tactical layer's own lat/lon posteriors conditioning the planner ARE the hierarchy the PI asked for
  (R8-4); they are a separate, declared path.
* ⛔ Never inputs: `nav_t_next_s`, `nav_token_ttime`, `rcH_*`, any v9 TARGET field, any VLM token (WP-A
  `INTEGRATION.md` §3.2).

---

## 3. Before any training step

### 3.1 Warm-start identity (step 0 reproduces refcv7-50,400)

**I-1 (BINDING, the seams):** with every refcv8 seam attached and allocation OFF, the refcv8 build loaded from
refcv7-50,400 by the trainer's own warm-start rule reproduces refcv7's eval forward **bit for bit** on `traj`,
`sel_idx`, `anchor_traj`, `sel_score_v3` (`torch.equal`), on the launch tree, on ≥ 4 eval139 windows spread over the
index (0, ¼, ½, end), with zero G-DVB mismatches and every missing key a declared refcv8 seam.
* **Status: MEASURED PASS** on the dev-box CPU at full size: variant `A_seams_no_alloc`, 4 / 4 windows bit-identical,
  `base_score_max_abs_diff` 0.0, 1,131 keys loaded strictly, 28 new keys all refcv8 seams, 122,513 new parameters,
  eval v9 join 23,772 / 23,772 (WP-B `raw/r8_fullsize_warmstart.json`; checkpoint `ckpt_50400.pt` md5
  `b418d0fc4a92a6848c246a6a7c50207b`). The registered SPEC_WPB I-W on Thor GPU also read the seams-only case
  bit-identical on 218 grid windows (INHERITED from the docstring of WP-B `code/iw_diag.py`; the artifact
  `arms/I_W.json` lives on Thor).
* **Re-run on the launch commit** (the overlay is not the launch tree). I-1 is a launch prerequisite in the gate.
* **Deliberate regression (must FAIL I-1):** one new gate initialised to 1e-3 (WP-B `test_refcv8_warm_start.py`).

**I-2 (allocation attached, emission OFF): OPEN — see §13 Q1.** MEASURED: with allocation M = 32 attached and
emission OFF, `sel_idx` stays identical but `traj` and the base fan differ, and base scores move by up to 5.7e-6 on
the CPU (variant `B_alloc32_emit_off`, `identity_PASS: false` against its own criterion) and by up to 2.1e-5 on
Thor (> the registered 1e-5; INHERITED, `iw_diag.py` docstring). The mechanism proposed by WP-B (GEMM shape over 149
or 266 rows instead of 117) is a HYPOTHESIS until `iw_diag.py` reports.
* **Proposed literal** (if allocation is in the launch argv): `sel_idx` identical on **100 %** of the I-1 windows AND
  **max |Δ traj| ≤ 1e-3 m** AND max |Δ base score| ≤ 1e-4, with emission OFF at step 0.
* If allocation is emitted from step 0 (`--r8-alloc-emit`, as in the WP-B smoke argv), step-0 identity of the
  EMITTED plan does not hold by design. §13 Q1 asks the Master Mind to choose: emission from step `S_emit` > 0, or
  identity stated for the 117-candidate base fan only.

### 3.2 The launch gate (BINDING, PI 2026-09-26)

No refcv8 training step runs without a PASS token from `stack/scripts/launch_gate.py`, profile **`refcv8`**, bound to
the exact commit, argv sha256 and data-manifest sha256s. The profile (WP-B overlay, base tip `5388a43`) inherits every
refcv7 rule and adds:
* **required flags:** `--refcv8 --init-from --r8-v9-labels --r8-v9-labels-eval --r8-nav-from-v9 --r8-rc-variant
  --join-defect-masks --grad-share-every`;
* **required values:** `--r8-v9-md5 f63ece410b725febb8a5242cf2b01d3c`, `--r8-v9-eval-md5
  6b5c7f207cffc3b7eb3cd527fd433599`, `--r8-v9-lat-variant a`, `--r8-rc-noise-along-m 2.0`, `--r8-rc-noise-lat-m
  0.75`, `--r8-rc-dropout 0.3`, `--r8-nav-args-dropout 0.5`, **`--w-r8-cons 0.05`**;
* **forbidden levers:** the regression flags `--r8-derange-feed`, `--r8-rc-roll`, `--r8-roll-targets`.

**This SPEC adds (binding on the launch argv):**
1. **`--w-r8-cons 0.05` always**, and **`--w-r8-alloc-l1 1.0` whenever `--r8-n-alloc > 0`**. `_pin_refcv8` refuses
   `--refcv8` with `--w-r8-cons 0` and allocation with `--w-r8-alloc-l1 0`. `--w-r8-sat`, `--w-r8-listwise` and
   `--w-r8-subscore` take the values §10.3 selects and are STATED in the argv even when 0.
2. **The four open items of the profile must be CLOSED, by name, before a PASS token:** R8-RECIPE, R8-BUDGET,
   R8-WARMUP, R8-INHERITED-RECORDS (§10). While one is open the token is not a PASS.
3. **I-1 on the launch commit** (§3.1), and I-2 if allocation is in the argv.
4. **G-MAP-OVERFIT / G-BOX-OVERFIT re-run on the refcv8 tree** (R8-INHERITED-RECORDS): the refcv7 records bind the
   refcv7 argv sha and cannot bind refcv8's.
5. **Reachability (CLAUDE.md "built, tested and unreachable"):** in the G-LIVE smoke, every refcv8 seam takes a
   non-zero gradient, with a control that must read exactly 0.0 (a seam whose weight is 0 in that smoke).
6. **The speed-input channel is named in the argv** (§8 X3). The WP-B smoke argv
   (`stack/ops/runs.d/refcv8-wpb-smoke.argv.json`, 197 tokens) still feeds refcv7's v8 future-max sidecar
   (`--speed-max-sidecar-v6 …/refcv6_speed_max_v8_train.jsonl`). A launch that keeps it is stamped "future-ego speed
   oracle input (K12)" on every number, and X3 then reads FAILED (§8).
7. **The refcv7-50,400 four-family baseline row exists** (X8) before launch, or every §5 bar reads PENDING.

### 3.3 After training (post-training gate)

* **Zero-gradient census:** run the optimizer-state probe (`…/2026-09-10-refcv5v2-zerograd-heads/probe_zerograd3.py`
  form) on the final refcv8 checkpoint. Every registered refcv8 parameter must have optimizer state. Control: a
  parameter frozen by design must read "no state". A refcv8 seam with no state is a FAILED reachability check, and
  every row that depends on that seam is reported VOID for the mechanism, not FAILED for the lever.
* **Restart / continue during training** follows `GATE_PROTOCOL.md` via `stack/scripts/run_gate.py`: the horizon and
  n are named; no learning-curve exponent decides anything.

---

## 4. The PI's tactical and route requirements (R8-1 … R8-4) — the deciding rows

### 4.1 R8-1 — per-frame tactical goals and actions over [NOW+2 s, NOW+8 s], with constraints

R8-1 has two layers. The label layer was pre-registered and judged by WP-A (`SPEC.md` sha256 `d26ed9e8…`); its
verdicts are reported as registered and are not re-barred here. The model layer asks whether the run actually
CONSUMES the labels.

* **(a) Measure.**
  * *Label layer (judged):* WP-A V1–V11 on the train and eval139 releases (label-level, no model).
  * *Model layer (this SPEC):* **R8-1-REACH** = the share of training windows whose lateral AND longitudinal tactical
    target is supervised (a single class or a non-empty partial mask), as the trainer's workers produced them,
    from the run's own logged rows over all logged steps. It is the D1 K2 form ("tactical rows / window logged").
  * Tier: label-level; in-run census. No planner number.
* **(b) Baseline (refcv7).** Label layer: tactical labels on **23.22 %** of 746,946 train windows (173,409), 22.73 %
  of eval139 (D1 `COVERAGE.md` row 1). Model layer: refcv7 logged **0.2357** [0.2295, 0.2426] tactical rows per window
  vs the census 0.2322 (D1 control K2). Both MEASURED.
  * WP-A's verdicts on the v9 release (MEASURED, `RESULT.md` §S3.1): V1 coverage train **95.09 %** PASS, eval139
    **94.57 %** FAIL as registered (95.58 % on the 136 tactically scored clips); V2 TURN side train 0.980 / 0.988 PASS,
    eval 0.943 / 0.966 FAIL; STOP 0.991 / 0.994 PASS; V3 timing / angle / radius / stop PASS; V5 lane change FAIL ⇒
    never-positive; V7 FAIL ⇒ `H_ABS_MIN = 8.0`.
* **(c) Bar.** **R8-1-REACH ≥ 0.93** of logged training windows, AND the run's `config.json` label census equals the
  v9 release manifest (train md5 `f63ece41…`, 746,946 / 746,946 windows joined).
* **(d) Estimator.** A census on the run's own logs; the interval is the bootstrap over logged steps used by D1 K2.
  No replicate is needed (a property of the data path, not a lever effect).
* **(e) Control.** The same instrument on refcv7's own log must read **0.2357** (D1 K2). The release census (95.09 %)
  is the analytic expectation; the logged share must lie within ±0.02 of 0.9509.
* **(f) Deliberate regression (must FAIL).** The refcv8 G-LIVE smoke argv with `--r8-v9-labels` removed (v8 labels
  only) must read ≤ 0.30, i.e. FAIL the 0.93 bar.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| PASS | the v9 per-frame labels reach the loss on ~95 % of windows (refcv7: 23 %) | — |
| FAIL with the control holding | a data-path defect between the release and the workers (join, mask or label scope) | read the per-window census against the release by clip; fix the join or scope; re-gate. No training continues on a run that fails R8-1-REACH at its first logged eval |
| control fails | the instrument is broken; nothing in this row is quotable | fix the logging key, re-read |

⚠️ **Not consumed in the WP-B overlay: the v9 CONSTRAINT vectors** (`lat_c`, `lon_c`, `speed_goal`). The constraint
heads regress plan-derived [0, 6] s quantities (terminal heading, turn onset, 6-s progress, v at 6 s), not the v9
fields (stop distance and time, lead gap and time gap, target speed and time to reach). WP-B DESIGN §3.9 items 2 and 10
name this. R8-1's "with constraints like distance and time" is therefore only partly delivered by the current build.
See §13 Q2.

### 4.2 R8-2 — per-frame nav command (announced junction turns, with distance)

* **(a) Measure.**
  * *Label layer (judged by WP-A V8):* turn-commanded windows with no announced turn starting within 6 s; realised
    announced turns fed the matching token.
  * *Model layer:*
    * **NAV-COMPLY** = on S-ROUTE GT-turn windows whose v9 nav token is TURN_x and whose turn starts within 6 s, the
      share of picks whose direction class equals x.
    * **NAV-PREMATURE** = on S-ROUTE windows whose v9 nav token is TURN_x, whose announced turn starts **> 6 s** ahead
      and whose GT 6-s path is GT-straight, the share of picks whose direction class equals x (a premature turn).
  * Tier: T1 open loop.
* **(b) Baseline (refcv7).**
  * Label layer, MEASURED: v8 clip token "turn but no turn within 6 s" on **74.3 %** of L/R train windows (205,292,
    D1); v9 spatial token **22.2 %** train / 30.4 % eval (WP-A V8, FAIL as written against ≤ 5 %); realised announced
    turns fed the matching token **99.55 %** train / 99.34 % eval (PASS).
  * Model layer: **PENDING** — NAV-COMPLY and NAV-PREMATURE on refcv7-50,400 V0 are computable with zero GPU from the
    A6 dense capture (`raw/a6_dense_score.json` inputs) joined to the v9 eval139 release. Owner: EvalFlyWheel / WP-B.
    The closest MEASURED reading: refcv7's pick complies with its clip-token nav on **0.685** of the 2,317 GT-turn
    windows (`a6_dense_score.json`, V0, `nav_complies`), a different token.
* **(c) Bars.** Both sampler seeds:
  * **NAV-COMPLY ≥ 0.95**;
  * **NAV-PREMATURE ≤ 0.05**.
* **(d) Estimator.** §2.2, paired vs refcv7-50,400 on the same windows; inference floor; single training seed (§9).
* **(e) Control.** The model-free ego-path class on the same windows reads NAV-COMPLY = 1.000 and NAV-PREMATURE =
  0.000 by construction (the GT path is the reference). The window counts must equal the v9 release's nav census on
  S-ROUTE.
* **(f) Deliberate regression (eval-time, no training).** `NAVSHUF`: the nav token and args taken from a deranged
  other window (the A6 T3a-c derangement form, 0 fixed points). NAVSHUF must read NAV-COMPLY < 0.95, and refcv8 −
  NAVSHUF on NAV-COMPLY must be separated (CI lower > 0). If NAVSHUF passes, the nav is not being read and the row is
  VOID.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| both PASS | the planner follows the announced nav and does not turn early | — |
| NAV-COMPLY FAIL, NAV-PREMATURE PASS | nav is read but not executed | read X1 and R8-4 (iv): if the fan holds the turn (fan-contains ≥ 0.99), the selector; else generation conditioning (R8-4 ii) |
| NAV-PREMATURE FAIL | the spatial early announcement (D-WPA-4) causes premature turns, the D6 navhard pattern (~59 % of 724 command-but-straight scenes, D6 §7) | the onset constraint: condition on `nav_d_next` with the turn-onset constraint head (t_onset) in selection (γ_head term) — WP-B arm T4 on the frozen trunk first |
| NAVSHUF passes | instrument or input path broken | VOID; trace the nav path (argparse → forward → loss) per the reachability rule |

### 4.3 R8-3 — the route checkpoint INPUT (simulated navigation system)

**Variant fixed by the registered WP-A follow-up:** RC-A50, fed **with** the certified training noise (σ_along 2.0 m,
σ_lat 0.75 m) and dropout 0.3. E2′ PASS (Δ −0.035 vs the road-level route, bar ≤ 0.01, certified instrument C1
−0.0045, C2 0.997, O1 0.379) and E3′ PASS on both populations (WP-A `RESULT.md` §S2.7, `raw/s2_e2prime.json`;
MEASURED, label-level). ⚠️ This rests on the Master Mind's PROVISIONAL ruling that road-geometry speed information is
admissible inside a route input. **PI decision 11 is open (§11).** If the PI overrules, the run carries `--r8-no-rc`
and every row below reads NOT APPLICABLE.

* **(a) Measure.** On S-ROUTE, sampler seeds 0 and 1, four eval rows of the SAME refcv8 checkpoint (no training):
  * **RC-ON** (the model as configured);
  * **RC-OFF** (`route_cp_valid = 0` on every window);
  * **RC-SHUF** (the checkpoint of a deranged other window);
  * **E-1** (WP-A's trivial planner: a constant-curvature arc tangent to the NOW heading through the RC-A50 point,
    driven at v0; model-free).
  * Metrics: turn direction-correct and heading-within-15° at 6 s on GT-turn windows; GT-straight ADE 0–6 s;
    all-window ADE 0–6 s. Tier T1.
* **(b) Baseline.**
  * refcv7 has no RC input. Its route rates: dense S-ROUTE turn direction-correct **0.8153** [0.730, 0.885]
    (`a6_dense_score.json` V0 seed 0); grid S-GRID 0.8411 [0.742, 0.920] / heading-15 **0.514** [0.423, 0.603]
    (`route_analysis.json`). Dense heading-15: 0.505 [0.432, 0.579] (R1 H0; INHERITED from `RESULT_R1.md`, artifact on
    Thor).
  * E-1 floor, MEASURED on the 1,112-window grid, 1,071 windows with every RC variant valid (WP-A `RESULT.md` §S2.1,
    raw `s2_echo_leak.json`): RC-A50 direction 0.916, heading-15 **0.654** [0.531, 0.763]; on all eval139 windows
    heading-15 0.640. On S-ROUTE: **PENDING** — `code/` of WP-A Stage 2 run on `a6_window_list.json` (zero GPU).
* **(c) Bars.** Both sampler seeds:
  1. **RC-ON beats E-1:** RC-ON − E-1 on turn direction-correct > 0 AND on heading-15 > 0, each separated;
  2. **RC-SHUF loses the gain:** RC-SHUF − RC-OFF on turn direction-correct ≤ **+0.02** (CI upper);
  3. **RC-OFF robustness:** RC-OFF − refcv7 on all-window ADE 0–6 s ≤ **+0.05 m** (CI upper ≤ +0.10 m).
* **(d) Estimator.** §2.2, paired on the same windows and seed. Bars 1–2 compare rows of ONE checkpoint, so they are
  not exposed to training variance; bar 3 is a refcv8-vs-refcv7 comparison (§9 wording).
* **(e) Controls.**
  * E-1 on S-GRID must reproduce WP-A's banked E1 row (RC-A50 direction 0.916, heading-15 0.654) exactly.
  * A deranged-RC E-1 must read direction ≤ 0.25 (WP-A banked 0.178).
  * The interventional pin: the RC tensor reaching `rc_to_cond` is bit-identical under a forced tactical posterior.
* **(f) Deliberate regression.** RC-SHUF (bar 2) is the regression row. At the lever level, the ladder's `V-R8d`
  (`--r8-rc-roll` + `--r8-derange-feed`) must fail B-RECIPE (`SPEC_WPB_LADDER_DRAFT.md` §5 B-REG).
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| 1–3 PASS | the RC adds route information the model uses beyond the trivial floor, does not echo it, and the model still drives without it | — |
| 1 FAIL (does not beat E-1) | the model is no better than aiming at the point: the echo risk the PI flagged (Hidden Biases' TP shortcut) | per-candidate RC bearing in selection only (ρ term), RC removed from the operative condition; then yaw augmentation (V-YAW, ladder) |
| 2 FAIL (shuffled RC keeps the gain) | the RC path is not what helps; the gain is elsewhere or the eval is broken | VOID for RC attribution; check `route_cp_valid` reaches the forward |
| 3 FAIL (collapses without RC) | the dropout regime did not take; the NavSim LEGAL row will collapse | raise RC dropout to 0.5 (ladder arm), or train with an RC-off fraction on the LEGAL schedule |

### 4.4 R8-4 — the tactical layer learns goals, actions and constraints; it CONDITIONS generation and selection; the planner is consistent with it

R8-4 has four parts, each with its own bars. **(iv) route following is the deciding bar** (Master Mind 2026-10-04:
"the real head's conditioned arm decides"); (i)–(iii) are necessary.

#### 4.4.1 (i) Tactical accuracy on per-frame labels

* **(a) Measure** (`taniteval.tactical_conditioning.class_report` / `constraint_mae` / `goal_ap`; the module imports
  numpy and `taniteval.ci` only, pinned by an AST test):
  * **TAC-LAT-TURN** = on S-ROUTE GT-turn windows, the share whose tactical lateral argmax (lat3, variant a, frozen v7
    ids) has the GT side (the D0 "side-correct on GT turns" reading).
  * **TAC-LAT-F1** = lat3 macro-F1 vs the v9 variant-a label on S-V9 (partial labels: a prediction inside the allowed
    set counts as correct), with the majority-class control beside it.
  * **TAC-LON-F1** = longitudinal macro-F1 over the v9 classes mapped to v7 ids, on S-V9, majority control beside it.
  * **CONS-PROG** = median relative error of the predicted 6-s progress (all classified windows with GT progress
    > 0.5 m); **CONS-HEAD** = RMS terminal-heading error of the constraint head on GT-turn windows. Each beside the
    TRAIN-median constant.
  * **GOAL-AP** = AP per goal token on S-V9 vs its prevalence (a constant scorer's AP = prevalence); tokens with eval
    n_pos < 200 are reported UNSCOREABLE, never pooled.
  * Per-class support (windows, clips) printed with every number (X5).
* **(b) Baseline (refcv7, MEASURED, open loop, grid 1,112 windows):** side-correct on GT turns **0.402** (D0,
  `d0_dose_response.json` `acc3_turn` 0.4019); lat3 accuracy 0.856 vs majority 0.8645 on 834 windows, macro-F1
  **0.492**; lon accuracy 0.418 vs majority 0.456, macro-F1 **0.351** (`r84_baseline_refcv7.json`, WP-B
  `RESULT_STAGE2.md`). Constraint MAE and goal AP: ABSENT (refcv7 has no constraint heads; the capture holds no
  per-window goal GT). On S-V9 (the v9 labels): **PENDING** — `r84_baseline_refcv7.py` re-run on R1's eval capture
  (it stores `p_lat` / `p_lon` / `p_goal` for the same 4,634 scored windows, `R1/RESULT_R1.md` §2) joined to the v9
  eval139 release, zero GPU; goal AP then becomes computable from the v9 goal bits.
* **(c) Bars** (sampler seed 0; the tactical head is seed-independent, WP-B RESULT_STAGE2 §2):
  * **TAC-LAT-TURN ≥ 0.95** (D0's q\*, the lateral-class accuracy at which conditioning selection clears on its own);
  * **TAC-LAT-F1 ≥ 0.80** AND head − majority on accuracy separated (CI lower > 0);
  * **TAC-LON-F1 ≥ 0.60** AND head − majority separated;
  * **CONS-PROG ≤ 0.0674** (D0's σ\* = 0.10) AND better than the TRAIN-median constant, separated;
  * **CONS-HEAD ≤ 15°** (D0c, post hoc there, committed here for this new run) AND better than the constant, separated;
  * **GOAL-AP:** for every scoreable token, AP − prevalence > 0, separated. A token that fails is listed by name.
* **(d) Estimator.** §2.2. Lever attribution of any tactical gain needs the ladder replicate (§9).
* **(e) Controls.** The majority-class scorer reads its prevalence exactly; a constant goal scorer reads AP =
  prevalence exactly; labels shifted +6 s must lower TAC-LAT-F1 (the module's mutation tests, 60 / 60).
* **(f) Deliberate regression.** At the lever level: R1's `H2s` (time-shuffled dense labels) and the ladder's
  `V-TACk-roll` (tactical targets rolled by one row) must not clear. At eval: a sign-flipped lateral readout must
  swap L/R recall.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| all PASS | the tactical layer learned actions and the constraints D0 sized | — |
| TAC-LAT-TURN FAIL, (iv) PASS | the planner already transmits what the head gets right; the head is not the bottleneck | report the gap; the trunk-trainable run is expected to raise it (not assumed) |
| CONS-* FAIL | the constraint heads do not reach D0's σ\* / 15°; D0 §2.3 says a weak constraint HURTS when conditioned on (+0.213 m) | zero the γ_prog / γ_head selection terms at eval (`sat` off) and re-read (iv); next: supervise the v9 constraint fields (§13 Q2) |
| TAC-*-F1 at majority | the head still has no skill (the refcv7 state) | the budget lever: ladder L2 (`--w-tac-v6` × k, §10.3) |

#### 4.4.2 (ii) Controllability — forcing the tactical condition moves the fan and the pick

* **(a) Measure** (`tactical_conditioning.controllability`; SPEC_WPB §4 definitions): on every classified S-ROUTE
  window (GT path ≥ 5 m), force the tactical posterior into every planner feed (`set_r8_force`, one-hot):
  * TURN_L / TURN_R / LANE_KEEP: read the pick's direction class (LANE_KEEP = class 0), and the share of ALLOCATED
    candidates whose own path class equals the forced class;
  * STOP-at-d, d ∈ {10, 20} m (lon BRAKE_TO, progress d, v_end 0), on windows with v0 ≥ 3 m/s and v0² / 2d ≤ 4 m/s²:
    the pick's stop distance within max(2 m, 0.1·d) of d.
  * Only hypotheses observable inside the 6-s plan count (turn onset ≤ 5 s); the rest are reported with n.
* **(b) Baseline.** ABSENT: refcv7 has no conditioning input that can be forced (WP-B RESULT_STAGE2 §1). R1's H5 is the
  closest frozen-trunk reading (PENDING, R1 arms running).
* **(c) Bars** (both sampler seeds): pick follows the forced class **≥ 0.95 for EACH** of TURN_L, TURN_R, LANE_KEEP;
  allocated share **≥ 0.95** for each (if allocation is in the argv); STOP-at-d within tolerance on **≥ 0.95** of
  eligible windows for each d.
* **(d) Estimator.** §2.2 bootstrap over episodes on the forced rates; both seeds.
* **(e) Controls.** Analytic tracks: a 15 m-radius arc reads TURN, a straight reads LANE_KEEP, a 1.5 m/s² stop at d
  reads STOP-at-d = 1.0, and the mirrored arc reads 0.0 for the other side. A model whose forced fan equals its
  unforced fan reads the base rate. The METRIC's path-class rule is written independently of the model's tagger and
  held against it on analytic tracks.
* **(f) Deliberate regression.** **CTRL-SHUF** (eval-time): the forced condition taken from a deranged other window
  (SPEC_WPB (ii′)). It must fall below 0.95 on at least one forced class AND below refcv8 with a separated paired CI on
  the pooled forced-follow rate. If CTRL-SHUF passes, the row is VOID. Lever level: WP-B `T1d` / `T2d`.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| PASS | the tactical decision steers generation and selection (the PI's "constraining the trajectory planner") | — |
| FAIL on TURN, PASS on LANE_KEEP | generation does not obey a turn condition | L_sat (`--w-r8-sat 0.1`, WP-B T2s) if not already on; then CFG-style guidance at inference (optional arm) |
| FAIL on STOP-at-d | the longitudinal constraint does not reach the plan | supervise v9 stop constraints (§13 Q2); the STOP-at-d sat hinge |
| CTRL-SHUF passes | the condition is not what moves the plan | VOID; trace the φ_k path per the reachability rule |

#### 4.4.3 (iii) Consistency — the fan and the pick agree with the tactical decision

* **(a) Measure** (`tactical_conditioning.consistency`): (1) **CONS-ALLOC** = share of allocated candidates whose path
  class equals their tag, all classified S-ROUTE windows; (2) **CONS-PICK-TAG** = the pick's path class vs its own tag;
  (3) pick vs tactical argmax, on all classified windows and on GT-turn windows (reported).
* **(b) Baseline (refcv7, MEASURED, grid):** pick vs tactical argmax **0.886** all classified, **0.439** [0.318, 0.564]
  on GT-turn windows (lat3 reading); candidate share at the tag 0.416 / 0.352, at the permuted-tag null 0.361
  (`r84_baseline_refcv7.json`). CONS-ALLOC / CONS-PICK-TAG: ABSENT (no tags in refcv7).
* **(c) Bars.** **CONS-ALLOC ≥ 0.95** (if allocation is on); **CONS-PICK-TAG ≥ 0.95**. Reading (3) has no separate
  bar: TAC-LAT-TURN ≥ 0.95 and (iv) direction ≥ 0.95 together imply ≥ 0.90 on GT-turn windows.
* **(d) Estimator.** §2.2.
* **(e) Control.** A fan built from its own labels reads 1.0; permuted tags read the class base rate.
* **(f) Deliberate regression.** Tags permuted across candidates at eval must fall to the base rate.
* **(g) Outcome.** PASS ⇒ the planner is consistent with its tactical layer. FAIL on CONS-ALLOC while (ii) passes ⇒ the
  allocation's anchor source (tag-match level 0 fallbacks) is the defect: report the tag-match histogram; next arm =
  allocation restricted to tag-match level ≥ 1. FAIL on CONS-PICK-TAG ⇒ the tagger and the metric disagree, or
  selection overrides the tag: read the β / γ telemetry.

⚠️ By construction an allocated TURN candidate's own direction share is 1.0 (sources are chosen by tag, WP-B DESIGN
§3.9.5). CONS-ALLOC is therefore an instrument check, and (ii)'s PICK reading carries the controllability claim.

#### 4.4.4 (iv) Route following — THE DECIDING BAR

* **(a) Measure** (`tactical_conditioning.route_following` ≡ the route package's `route_metrics`): on S-ROUTE,
  RC-ON row (the model as configured), sampler seeds 0 and 1: turn direction-correct pick (terminal-heading class,
  τ 10.35°) on GT-turn windows; heading within 15° of GT at 6 s on GT-turn windows; GT-straight ADE 0–6 s. Tier T1.
* **(b) Baseline (refcv7-50,400, MEASURED):** S-ROUTE direction **0.8153** [0.730, 0.885] (seed 0), straight ADE 1.678
  m, straight direction 0.963 (`a6_dense_score.json`, V0); S-ROUTE heading-15 **0.505** [0.432, 0.579] seed 0 / 0.510
  seed 1 (INHERITED, `RESULT_R1.md` H0; artifact on Thor); S-GRID 0.8411 / **0.514** (`route_analysis.json`).
* **(c) Bars** (BOTH sampler seeds, all four):
  1. turn direction-correct **≥ 0.95**, AND refcv8 − refcv7 on it separated (CI lower > 0);
  2. heading within 15° **≥ 0.70**;
  3. GT-straight ΔADE vs refcv7 **≤ +0.05 m** with CI upper **≤ +0.10 m**;
  4. the **RC-OFF** row also clears 1 and 2 at **≥ 0.90 / ≥ 0.60** (route following must not live only in the
     privileged input; §4.3).
* **(d) Estimator.** §2.2. The comparison is two single training runs (§9): a PASS is worded "the refcv8 run vs the
  refcv7 run", never "lever X lifts route following".
* **(e) Controls.** The fan contains a direction-correct candidate (refcv7: 1.000) and the random-in-reach pick reads
  chance (refcv7 grid 0.385, `route_analysis.json`); oracle-117 reported. B1 / B3 bounds reproduce (D0 C1 form).
* **(f) Deliberate regression.** Eval-time: RC-SHUF + NAVSHUF + CTRL-SHUF together (the "all route evidence
  deranged" row) must FAIL bar 1. Lever level: the ladder's `V-R8d` must fail B-RECIPE.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| 1–4 PASS | R8-4 met: the conditioned planner follows the route | — |
| 1–3 PASS, 4 FAIL | route following lives in the privileged RC | RC dropout 0.5 and the LEGAL schedule (§4.3 g); NavSim LEGAL row decides severity |
| 1 FAIL, fan-contains ≥ 0.99 | the pick loses a turn the fan holds (refcv7's D_core +0.159) | X1: per-candidate sub-score critics (`--w-r8-subscore`), then the terminal-heading constraint in selection (D0c: ≤ 10–15° reaches the bars by re-selection alone) |
| 2 FAIL with 1 PASS | direction right, geometry wrong (over/under-steer, D6 classes) | the heading constraint (CONS-HEAD) and the curvature family; then yaw augmentation (V-YAW) |
| 3 FAIL | straight driving pays for turning | condition dropout up; read NAV-PREMATURE (§4.2) |
| regression row passes | the instrument cannot see the route inputs | VOID |

---

## 5. The four metric families (BINDING, CLAUDE.md 2026-08-02) — each reported separately, never pooled

Every refcv8 eval reports all four families, on S2 / S6 (the battery surface) AND on S-ROUTE, per family, each with
its estimator, its CI, its n and its tier. ADE stays and is one row among them. ⛔ A horizon sweep of ADE is never "the
result". A family that cannot be computed is reported absent **with its reason and n**, per family.

**Instrument.** The refcv7 four-family battery (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/
battery/`, SPEC + amendments A1–A7), re-pointed at the refcv8 run. Its G0 gate (the loader reproduces the run's own
in-run eval row) runs under **A7**. ⚠️ A7.2–A7.4 must be implemented before any refcv8 G0: until then a refcv8 G0
under A7 REFUSES (battery SPEC A7, "Implementation owed"). No refcv8 battery number counts unless its G0 passes.

| family | metrics (definitions as the battery SPEC §3.3 and the route package) | refcv7-50,400 baseline |
|---|---|---|
| **LONGITUDINAL** | speed MAE 0–2 s and 2–6 s; along-track error at 2 s and 6 s (signed and absolute); target-speed accuracy @ 0.5 m/s; 6-s progress relative error; **distance keeping** (min headway, time gap, min TTC to the in-path lead; X7); ceiling compliance of the EMITTED plan (X3) | S2: **PENDING** — the step-50,400 battery (`battery/raw/step50400/`) produced no family number: G0-A6 FAIL as coded, PASS under the A7 text ruling (`g0_A6_text.json`); the battery must be re-run under the ruling (X8). S-ROUTE, MEASURED (`a6_dense_score.json`, V0 seed 0): turn speed MAE 0–2 s **0.347** m/s, straight 0.252; turn \|along\| 6 s 7.37 m, straight 5.00 m |
| **LATERAL** | heading error (MAE 0–2 s; terminal at 6 s); **curvature** MAE 0–2 s (masked, beside the `ha0` straight-line floor); **yaw-rate** MAE with the standstill mask (RETR-2026-09-26-YAWMASK); cross-track at 2 s and 6 s; route direction and heading-15 (R8-4 iv) | S2: PENDING (as above). S-ROUTE, MEASURED (V0 seed 0): turn terminal-heading error **23.8°**, curvature MAE 0–2 s 0.0098, \|cross\| 6 s 6.33 m; straight 2.29°, 0.0016, 1.05 m |
| **TACTICAL** | R8-4 (i)–(iii): lat / lon accuracy and macro-F1 on dense v9 labels with per-class support; confusion matrices; constraint MAE; goal-token AP vs prevalence; controllability; consistency; anchor selection vs chance 1/117 | refcv7 grid, MEASURED (`r84_baseline_refcv7.json`): lat3 0.856 vs majority 0.8645, macro-F1 0.492; lon 0.418 vs 0.456, macro-F1 0.351; side-correct on GT turns 0.402 (D0) |
| **STRATEGIC** | **ABSENT by PI ruling R5 (2026-09-27): the strategic layer is OFF** (`--no-strategic`, SPEC_REFCV7 §1). Its route loss is gated off in training, so the route head is untrained and is not scored. Reported as **NOT APPLICABLE, n = 0, reason: layer off**, in every table, never silently dropped. The route-level decision quality the PI asked for (R8-3, R8-4 iv) is measured inside LATERAL / TACTICAL, and is not relabelled "strategic". | NOT APPLICABLE |

**Bars on the four-family block** (S2 / S6, both sampler seeds; refcv8 `os` vs refcv7-50,400 `os`, paired):
* **BAR-R8-E1 (inherits BAR-R7-1):** `os − ha0_ext` (the echo) ADE 0–2 s < 0, separated, at both seeds.
* **BAR-R8-E3 (inherits BAR-R7-3):** on S6, `os − ha0_ext` ADE 0–6 s < 0, separated, at both seeds.
* **BAR-R8-NR (non-regression, per family):** for each headline below, refcv8 − refcv7 is **not separated worse**
  (for an error metric: CI lower ≤ 0) at both seeds:
  * LONGITUDINAL: speed MAE 0–2 s; \|along-track\| at 2 s;
  * LATERAL: heading MAE 0–2 s; curvature MAE 0–2 s; yaw-rate MAE (masked);
  * ADE 0–2 s (S2) and ADE 0–6 s (S6).
* **Controls (VOID gates of the battery §3.6):** `ha0` curvature and tactical κ exactly 0; `stop` displacement exactly
  0; the model-free arms (`ha`, `ha0`, `ha0_ext`, `stop`) and GT bit-identical between the refcv8 and refcv7 dumps
  (paired Δ 0.0, CI [0, 0]).
* **Deliberate regression:** the battery's G0 mutation M1 (bottom-row equalisation removed) must be detected, else G0
  is VOID; M5 (A7.2, rare-class index swap) is reported.
* **Outcome.** E1 / E3 PASS ⇒ refcv8 beats its own echo, as refcv7 was required to. NR FAIL on a family ⇒ that family
  regressed while R8-x moved: the trade-off is reported, never averaged away; the next arm is the family's own lever
  (LONG: X1 / X3; LAT: the heading constraint, §4.4.4 g).

---

## 6. The NavSim block — every NavSim number has TWO rows

**PI, 2026-10-04: NavSim leaderboard submission stays OFF until the PI has examined the leaderboard.** Nothing in this
SPEC submits. A refcv8 NavSim number is never compared with a published leaderboard number as a bar (refcv7 NavSim
SPEC §10.2: camera-only, zero-shot, single front stitch).

**Tier:** NavSim open-loop benchmark (T1-family), zero-shot PhysicalAI-AV → nuPlan cameras, non-parity; never closed
loop. **Estimator:** the paired log-cluster bootstrap (navhard 76 logs, navtest 136 logs, B = 2,000, seed 0; log_name
AND nuplan_drive), inference seeds 0 and 1; training variance untested (§9).

### 6.1 The two rows

| input | **LEGAL** (the leaderboard form; carries every bar) | **PRIVILEGED** (diagnostic only; ⛔ never submitted, never a bar) |
|---|---|---|
| nav token | NavSim's bare `driving_command` mapped: left → TURN_L, forward → FOLLOW, right → TURN_R, unknown → FOLLOW (WP-A `INTEGRATION.md` §5) | OUR announced token re-derived from the scene route's lane connectors (a junction connector whose heading change meets `is_turn`'s 15°) |
| nav args (turn distance, angle) | none: `nav_args_known = 0` | from the route geometry (distance along the route centreline) |
| route checkpoint | none: `route_cp_valid = 0` | the centreline point at arc L = 50 m from the metric-cache route |
| speed input | **`unknown` row** (default; §13 Q3): NavSim's `AgentInput` carries ego statuses, cameras and lidars only (`navsim/common/dataclasses.py:150-155`, INHERITED via WP-A INTEGRATION §5); a past-only N2 needs 20 s of ego history the agent does not receive | the refcv7 suite's map speed limit (`max_speed_input`, "map") |
| frames, v0, ego history | as the refcv7 NavSim bridge (`refcv6_bridge` functions, imported unchanged) | same |

⚠️ **A deliberate mismatch, stated on every NavSim number:** NavSim's command also fires on curves (any \|y\| ≥ 2 m at
20 m ahead) and carries no distance; the v9 token is announced junction turns only, with distance. The nav-args
dropout (0.5, token kept) trains the token-only regime; the curve-firing difference remains (D6: premature turns on
~59 % of 724 command-but-straight navhard scenes, refcv7 30k).

The gap PRIVILEGED − LEGAL is reported with its paired CI. It is the price of the privileged route, never a result.

### 6.2 Bars (LEGAL row, both inference seeds)

| bar | statement | refcv7 baseline (MEASURED, `navsim/raw/milestones/step*/BARS.json`; the refcv7 A1 arm fed the MAP speed limit, so it is NOT a legal-row baseline) |
|---|---|---|
| **BAR-R8-N1** (navtest, FULL 12,146 tokens) | PDMS(refcv8 LEGAL) > STOP, paired interval excluding 0, margin > 2× the full-split inference floor | step 30,000: **71.8849** vs STOP 61.8202, Δ +0.1006 [+0.0829, +0.1178], PASS; step 5,000: 65.5976, PASS. Step 50,400: PENDING (`step50400/runner.log`: navtest scoring aborted by the RAM guard, last line 19:03Z) |
| **BAR-R8-N2** (navhard, official two-stage EPDMS) — **the decisive NavSim bar** | EPDMS(refcv8 LEGAL) > max(STOP, CV, ECHO); paired interval vs STOP excluding 0; margin > 2× the seed floor | step 30,000: **0.2269** vs STOP 0.2985, Δ −0.0716 [−0.1112, −0.0346], **FAILED**; step 5,000: 0.1624, FAILED. 50,400: PENDING |
| **BAR-R8-N3** (warmup, S2-EPDMS-u) | > max(CV, STOP, ECHO), margin > 2× the seed floor | step 30,000: 0.5224 vs STOP 0.5212, NOT PROVEN; step 5,000: FAILED |
| **BAR-R8-N4** (vs refcv7, same inputs) | navtest PDMS(refcv8 LEGAL) − PDMS(refcv7-50,400 LEGAL) > 0, separated | **PENDING: refcv7-50,400 rolled through the LEGAL inputs** (bare command, unknown speed row) on navtest FULL and navhard, by the EvalFlyWheel bridge with `R7_VMAXOFF`-style inputs (that arm exists for warmup and a navtest subset only) |

**Reported beside every navhard number (no bar; the X6 anatomy):** DAC-zero and NC-zero rates per stage, with paired
CIs vs refcv7 30k (stage 2: DAC-zero **26.0 %**, NC-zero **15.6 %**; D6 `RESULT.md` §0); the D6 geometry classes of the
DAC-zero scenes (LATERAL-DRIFT, ROUTE-FOLLOWING, ON-ROUTE, OVER-STEER, SPEED, NO-RECOVERY, WRONG-SIDE) by the D6
literals; the NC-zero front-collision share (refcv7 30k: 94.8 %) and the plan − reference speed at 4 s in NC-zero
scenes (refcv7 30k: median +3.87 m/s).

**Controls (must read known values, from the refcv7 suite):** STOP reproduces the banked official frame (D6 P3: max
abs 8.3e-17); a re-run of 300 banked scenes reproduces their 8 sub-scores exactly (D6 K7 form); KPR (the emitted prior
equals the bridge's model-free prior, ≤ 1e-4 m); KL-lift bit-exact.

**Deliberate regression:** `R8_BLIND` (constant grey frames, LEGAL inputs) must FAIL BAR-R8-N3 on warmup. If it passes,
the panel is VOID.

**Outcome (Rule Zero).**

| reading | means | next arm |
|---|---|---|
| N1–N3 PASS on LEGAL | refcv8 beats STOP on the official navhard metric with leaderboard-legal inputs; the leaderboard examination with the PI may start | — |
| N2 FAIL, D6 classes dominated by LATERAL-DRIFT / ROUTE-FOLLOWING / OVER-STEER | the steering-geometry failure persists. The D6 P3 ranking says the route / lateral lever (+0.113 EPDMS ceiling) exceeds the whole deficit to STOP (−0.072); speed alone (+0.061) does not | P1′ (`SPEC_P1P2_A2_P1PRIME.md`, registered) re-run on the refcv8 checkpoint decides selection vs generation: the fan holds a DAC-clean candidate ⇒ a drivable-area sub-score critic from the 10 cm map in selection (X1 family); not in the fan ⇒ generation (dense turn labels + the heading constraint) |
| N2 FAIL, NC-zero dominated by front collisions with the plan faster than the reference | the longitudinal commitment failure persists (D6 P3: a perfect speed fix clears 46.0 % of navhard NC-zero) | X7: the FOLLOW lead gap / time gap from v9 as the longitudinal condition, plus a speed-profile critic in selection (X1) |
| PRIVILEGED PASS, LEGAL FAIL | the model needs the privileged route | the RC-dropout / LEGAL schedule lever (§4.3 g) |
| R8_BLIND passes | the panel cannot see the frames | VOID |

---

## 7. Perception (R8-5, R8-6)

**Common to both rows.** Surface S-PERC (§2.1). Tier: OPEN-LOOP PERCEPTION DIAGNOSTIC; the four driving families are
N/A for these rows (perception feeds LONGITUDINAL distance keeping and the planner's BEV; those links are read in §5 and
X7). Decision rules are refitted on refcv8's OWN TRAIN-DIAG pass (WP-C I4: `refit_perception_thresholds.py`, each file
re-read by the stack loader), never on eval. Every AP is reported raw AND after centre-distance NMS; boxes per detected
object at the TRAIN P = R gate; the map per class × band. Beyond 40 m the map is reported with WP-D's range-adaptive
metric M5 and carries **no IoU bar** (ANALYTIC: 40–100 m is decoded from ~1.3 stride-8 feature rows; one row in 80–100 m
explains 903 label rows; WP-D `RESULT.md` §3, `raw/geom_bound.json`).
**Estimator:** paired episode-cluster bootstrap over the 137 labelled eval episodes (§2.2). Perception is checked
seed-invariant across sampler seeds 0 / 1 (battery VOID gate 6); if not, it is rolled at both. Single training seed
(§9).

### 7.1 R8-5 — all identified map and box fixes are applied

* **(a) Measure.**
  * **R8-5-APPLIED** (data hygiene, from `config.json` and the readers' censuses): ego-footprint boxes removed from the
    TRAIN agent / box3d targets (`--join-defect-masks`); track-id-switch rate rows masked; the 2 eval clips without map
    GT masked at eval; z / h quoted from the near-field keys (< 30 m) only (`--det-zh-trust`).
  * **R8-5-MAP:** per-class 10 cm IoU under the F1 rule (TRAIN-fitted per-class thresholds), bands 0–20 and 20–40 m and
    "all".
  * **R8-5-BOX:** box3d and agent AP@2 m after F4b NMS at refcv8's refitted radius and gate; boxes per detected object
    after NMS; `conf_ratio` at the declared gate.
* **(b) Baseline (refcv7-50,400 with the WP-C fixes, MEASURED):**
  * F1 map IoU (all bands): lane **0.1643**, crosswalk 0.1052, arrow 0.0588, edge **0.0421**, hatched 0.0616, drivable
    0.5764, sidewalk 0.5001, nocls 0.6082 (`perception-fixes/raw/f1_f4_f4b_reproduction.json`, 0.0 difference from the
    diagnostics). As trained (`prior_corrected`): lane 0.068 [0.056, 0.078], edge 0.000 (diagnostics `RESULT.md`).
  * Box: box3d AP@2 m 0.2483 → **0.3494** after NMS (r 2.5 m, gate 0.2145); agent 0.1314 → **0.2996** (r 3.0 m, gate
    0.1809); boxes per detected object (gate 0.2589) box3d 2.12 → **1.07**, agent 2.60 → **1.01**; `conf_ratio` 0.995 /
    1.068 at the refitted gates (same file).
  * Hygiene: 693 ego-footprint boxes in 18 train clips; on the 40 most-jumping clips the largest \|v_rel\| 764.1 → 77.9
    m/s, rate rows above 100 m/s 5 → 0, 3,470 rate rows masked (`perception-fixes/raw/join_masks_acceptance.json`).
* **(c) Bars.**
  1. **R8-5-APPLIED:** the train reader's `defect_masks` census in `config.json` equals the shipped list exactly (693
     boxes / 18 clips; the masked event frames equal the list's count), and the WP-A / WP-C mask cross-check reads
     693 / 693 frames (WP-B `raw/mask_crosscheck.json` form).
  2. **R8-5-MAP:** for every class and each band 0–20 / 20–40 m, refcv8 (its own F1 file) − refcv7 (F1) is not
     separated worse.
  3. **R8-5-BOX:** boxes per detected object after NMS **≤ 1.20** for both heads; `conf_ratio` in **[0.5, 1.5]** for
     both heads.
* **(d) Estimator.** §7 common.
* **(e) Control.** refcv7-50,400 with the shipped F1 / F4 / F4b files, through the same instrument, reproduces lane
  0.1643, box3d NMS AP@2 m 0.3494 and agent 0.2996 exactly (\|Δ\| 0.0, the WP-C reproduction).
* **(f) Deliberate regression.** (1) The G-LIVE smoke without `--join-defect-masks` must report 693 ego boxes in the
  targets (FAIL bar 1). (2) NMS radius 0 must reproduce the raw AP exactly and FAIL bar 3 (refcv7 raw 2.12 / 2.60).
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| 1–3 PASS | every R8-5 fix is applied and none of refcv7's fixed baseline is lost | — |
| 1 FAIL | a wiring defect (WP-C I3) | launch blocker if caught at the gate; after training: the box rows are VOID for the masked clips |
| 3 FAIL | the NMS operating point does not fit refcv8's score distribution | refit the radius on TRAIN-DIAG over the grid {2.0, 2.5, 3.0} m (fixed now); if still FAIL: B2's de-duplicating decoder (CDN / Group-DETR), §10.1 |

### 7.2 R8-6 — architecture and training workflow that DRAMATICALLY improve map and box

The bars are WP-D's registered "dramatic" bars (`PREREG_WPD_PROBES.md` §P-BOX PB6, §P-MAP PM4), transposed from the
frozen-memory probes to the full run. They apply whatever subset of WP-D levers enters (§10.1): the PI's requirement is
the outcome.

* **(a) Measure** (S-PERC, class-agnostic AP unless stated): box3d AP@1 m and AP@2 m **raw** (no NMS); boxes per
  detected object **raw** at the TRAIN P = R gate; AP@2 m of the detector the PLANNER reads (the agent head, or box3d
  if B3 entered); map IoU_2 (0.2 m Chebyshev tolerance) for edge and lane per band, under the F1 rule. Reported beside
  them: AP@{0.5, 1, 2, 4} m raw and NMS per band 0–20 / 20–40 / 40–60 m; longitudinal vs lateral centre error per band;
  presence ECE; every class × band of the map; 40–100 m with M5.
* **(b) Baseline (refcv7-50,400, MEASURED):** box3d AP@2 m raw **0.248**, AP@1 m raw **0.130**, boxes per detected
  object **1.97** (box3d) / 3.12 (agent) at the TRAIN P = R gate, agent AP@2 m raw **0.131** (WP-D `raw/box_probe0.json`;
  diagnostics `raw/B_box.json`); map IoU_2 under `thr_phat`: edge 0–20 m **0.2351**, edge 20–40 m **0.1176**, lane
  20–40 m **0.3077** (diagnostics `raw/M_b.json`, `eval/thr_phat`). The box error is longitudinal: median \|Δx\| 0.96 /
  1.56 m vs \|Δy\| 0.38 / 0.46 m at 0–20 / 20–40 m (WP-D `RESULT.md` §2.5).
* **(c) Bars** (all literals; each also requires refcv8 − refcv7 separated in the right direction):
  * **R8-6-BOX-1:** box3d AP@2 m **raw ≥ 0.339** (refcv7's admissible NMS level: the decoder must de-duplicate as well
    as NMS does);
  * **R8-6-BOX-2:** box3d AP@1 m **raw ≥ 0.260** (2 × refcv7);
  * **R8-6-BOX-3:** boxes per detected object **raw ≤ 1.30** (box3d);
  * **R8-6-BOX-4:** the planner's detector AP@2 m **raw ≥ 0.248** (refcv7's box3d level; refcv7's agent head 0.131);
  * **R8-6-MAP-1:** edge IoU_2, 0–20 m **≥ 0.33**;
  * **R8-6-MAP-2:** edge IoU_2, 20–40 m **≥ 0.18**;
  * **R8-6-MAP-3:** lane IoU_2, 20–40 m **≥ 0.38**;
  * **R8-6-NR:** drivable IoU not separated WORSE than refcv7 in any band 0–20 … 80–100 m (BAR-M7-4 continuity), and no
    band's box3d AP@2 m after NMS drops by more than 0.02.
* **(d) Estimator.** §7 common.
* **(e) Controls.** refcv7-50,400 through the same instrument reads box3d AP@2 m 0.24828 and agent 0.13140 (WP-D C2)
  and edge IoU_2 0–20 m 0.2351; a paired bootstrap of an arm against itself reads exactly 0 (WP-D C3); NMS radius 0
  reproduces raw exactly (C4).
* **(f) Deliberate regression (eval-time).** (1) The box memory permuted across the windows of each batch (the
  `R-memshuf` form) must read AP@2 m raw ≤ 0.05. (2) Lane / edge GT shifted +0.5 m laterally (the `R-gtshift` form,
  metric side) must lower lane IoU_0 at 0–20 m by ≥ 0.05. If either fails, the panel is VOID.
* **(g) Outcome.**

| reading | means | next arm |
|---|---|---|
| all PASS | the "dramatic" perception gain is met on the full run | — |
| BOX-3 FAIL (duplicates) | the decoder still does not de-duplicate | B2's CDN / Group-DETR if not in the run; then HQS (`--slot-query-select heatmap`) |
| BOX-2 FAIL, BOX-1 PASS | found but misplaced; the error is longitudinal | B4 (a) ray embedding and (b) BEV point sampling; then LiDAR depth (PI decision 9) |
| BOX-4 FAIL | the planner still reads the weaker detector | B3 (box3d slots → `AgentTokenEmbed`, zero-gated) |
| MAP-1/2/3 FAIL | the map has no say over its features (P-GRAD: map 0.01 % of the trunk update, −0.4 % of its own encoder's) | M6 (a map-owned copy of the 0.25 m encoder, initialised from the shared one) or ladder L3 (×4 map weight), then M1 (a) stride-4 near lift |
| NR FAIL | the planner's BEV or a band paid for the gain | the zero-gated path that shifted the shared encoder (M1 b) is reverted first |

---

## 8. R8-7 and the Master Mind's fixes X1–X10

### 8.0 R8-7 — every other identified refcv7 fix (the roll-up)

* **(a)** R8-7 is the conjunction of X1–X10 below, each read on its own surface.
* **(b)** Per X row.
* **(c) Bar:** R8-7 PASSES iff every X row that carries a bar reads PASS and every report-only X row is present with
  its n. An X row reported NOT BUILT counts as FAILED for R8-7 (never as deferred silently).
* **(d)–(f)** Per X row.
* **(g)** R8-7 FAIL names the failing X rows and each row's next arm.

### 8.1 X1 — the selector is trained on the emitted fan (listwise + sub-score critics)

* **(a) Measure.** On S-ROUTE GT-turn windows, both seeds: **D-CORE** = (share of windows whose fan contains a
  direction-correct candidate) − (share whose pick is direction-correct); **REGRET** = ADE 0–6 s of the pick − ADE of
  oracle-117 (the ADE-nearest candidate; T0, a denominator only). Random-in-reach pick beside both. Tier T1.
* **(b) Baseline (MEASURED).** S-GRID D-CORE **+0.159** [+0.080, +0.258] (seed 1 +0.150) (`route_analysis.json`).
  S-ROUTE: pick direction 0.8153; oracle-117 turn ADE 1.225 vs pick 3.297, i.e. **REGRET 2.072 m** [1.743, 2.448]
  (`a6_dense_score.json`, ORACLE row, seed 0); fan-contains on S-ROUTE 1.000 (INHERITED, `RESULT_R1.md` H0).
* **(c) Bars.** **D-CORE ≤ +0.05** at both seeds; **REGRET ≤ 1.55 m** (0.75 × refcv7's 2.072 m, fixed now as a literal)
  with refcv8 − refcv7 on REGRET separated.
* **(d)** §2.2.
* **(e) Control.** Random-in-reach pick reads chance (S-GRID refcv7: 0.385); fan-contains reads ≥ 0.99, else D-CORE is
  not a selection measure (generation lost the turn).
* **(f) Deliberate regression.** Eval-time: the selection score permuted within the reach-kept set must read the
  random rate. Lever level: WP-B `T1d` / `T2d` (registered).
* **(g) Outcome.** PASS ⇒ the pick no longer loses the turn its fan holds. FAIL with fan-contains ≥ 0.99 ⇒ Hydra
  sub-score critics (`--w-r8-subscore`, X1h) if off, else the terminal-heading constraint in selection (D0c). FAIL
  with fan-contains < 0.99 ⇒ conditioning collapsed the fan's diversity: condition dropout up, allocation off, re-read.

### 8.2 X2 — the residual-prior copycat

* **(a) Measure.** **PRIOR-WRONG-PICK** = on S-ROUTE GT-turn windows where the residual prior's side (`ha0_ext_pose`)
  differs from GT, the pick's direction-correct rate; beside it the prior-right rate and the share of wrong turn picks
  that take the prior's side.
* **(b) Baseline (MEASURED, S-GRID, `d0b_prior_follow.json`):** **0.6739** (n 46; seed 1 0.6957) vs **0.9672** when
  the prior is right (n 61); 14 of 17 wrong turn picks take the prior's side (seed 1: 13 of 16). S-ROUTE: **PENDING** —
  `d0b_prior_follow.py` on the A6 dense capture (zero GPU).
* **(c) Bar.** **PRIOR-WRONG-PICK ≥ 0.90** at both seeds.
* **(d)** §2.2; n printed.
* **(e) Control.** The model's emitted `residual_prior_path` equals the model-free prior (KPR, ≤ 1e-4 m).
* **(f) Deliberate regression.** Lever level only: X2a at p = 0 vs p = 0.3 (WP-B arm X2a vs T1, with the T0r floor).
* **(g) Outcome.** PASS ⇒ the copycat bias is gone. FAIL ⇒ X2b (the prior-free sampler group) if not in the run; if in,
  read where the wrong picks come from (prior group vs prior-free group).

### 8.3 X3 — a past-only speed input with a trained "unknown" row, and the ceiling on the emitted plan

⚠️ **NOT BUILT in the WP-B overlay** (base tip `5388a43`): no `--r8-*` flag carries N2 / N3 or the unknown-row dropout,
and the smoke argv feeds refcv7's v8 future-max sidecar. PI decision 2 (N2 vs N3) is open (§11).

* **(a) Measure.** (1) **LEAK** = the D4 instrument's recovered share of future-speed information (OOF R², 6-s future
  max) for the channel actually fed; (2) **CEIL-OBEY** = share of S-ROUTE windows whose EMITTED plan's max speed
  exceeds the fed finite ceiling by > 0.5 m/s; (3) **VMAX-OFF** and **VMAX-SHUF** eval rows (input removed / taken
  from a deranged window): speed MAE 2–6 s and \|along-track\| at 6 s; (4) **UNKNOWN** row: the input set to unknown on
  every window.
* **(b) Baseline.** LEAK (MEASURED, D4 `raw/d4_vmax_nonoracle.json`, 576,555 train windows): refcv7's per-clip value
  recovers **23.8 %** (`USAGE_AUDIT.md`), N2 **3.46 %**, N3 1.93 %, N2 with the unknown row at p 0.45 0.93 %, the
  window oracle 56.98 %. CEIL-OBEY: refcv7 as launched did not apply the ceiling to the emitted plan (SPEC_REFCV7 §26.1);
  110 / 2,059 reel windows exceeded it (INHERITED, PLAN X3). The unknown row: never trained in refcv7, fed on 45 % of
  navtest tokens (INHERITED, D4 / NavSim).
* **(c) Bars.** (1) **LEAK ≤ 0.05** for the fed channel; (2) **CEIL-OBEY ≤ 0.01**; (3) VMAX-SHUF − VMAX-OFF on speed
  MAE 2–6 s **not separated better** (a shuffled limit must not help); (4) UNKNOWN − known on speed MAE 2–6 s **≤ +0.10
  m/s** (CI upper ≤ +0.20).
* **(d)** §2.2 for (2)–(4); (1) is the D4 census instrument.
* **(e) Control.** The D4 instrument reads the window oracle at 0.5698 and "no input" at 0.0 (MEASURED).
* **(f) Deliberate regression.** VMAX-SHUF (bar 3).
* **(g) Outcome.** PASS ⇒ the speed channel is admissible and used as a limit. **If the launch keeps the v8 sidecar,
  X3 reads FAILED on bar 1 (0.238 > 0.05) at launch**, and every speed-family number carries the K12 oracle stamp; the
  next arm is the N2 / N3 wiring on the ladder (one variable vs V-R8) with the unknown row at p 0.45, then a re-launch
  decision for the PI.

### 8.4 X4 — loss budget from measured gradient shares; label-state isolation

* **(a) Measure.** (1) **LABEL-SCOPE** = windows whose (y, w) goal targets, produced by a pickled TRAIN dataset (the
  worker path) after the TRAIN then EVAL label loads, differ from the TRAIN policy; (2) **TAC-SHARE** = the median of
  the in-run `gs_trunk_proj_tac_v6` over the last 50 % of logged steps (`--grad-share-every`, P-GRAD's statistic).
* **(b) Baseline (MEASURED).** refcv7 trained `LANE_CHANGE_L` as a negative on 100 % of in-band windows (173,409
  scored vs the census' 12,145; D1 `COVERAGE.md`); tactical projection share of the whole-trunk update **−0.0005**,
  trajectory 0.0002, map 0.0001, agent 0.501, box3d 0.357 (WP-D `raw/pgrad.json`).
* **(c) Bars.** (1) **LABEL-SCOPE = 0** windows. (2) Only if the ladder adopted a tactical multiplier (§10.3):
  **TAC-SHARE ≥ 0.01** (the ladder's B-PREM literal); otherwise TAC-SHARE is reported with no bar.
* **(d)** (1) a census; (2) the instrument's median and IQR.
* **(e) Control.** grad_share linearity ≤ 1e-4 on every reading; a weight-0 term reads exactly 0.
* **(f) Deliberate regression.** The unfixed `v7_labels` module (a module-global policy read) must go RED
  (`test_v9_labels_v7_policy_isolation.py`, `test_refcv8_label_scope.py`).
* **(g) Outcome.** (1) FAIL ⇒ launch blocker. (2) FAIL ⇒ the budget cannot be bought by weight: a stop-gradient on the
  agent term into the trunk's last stages (agent·tac cosine −0.297), or per-term gradient normalisation (ladder §8).

### 8.5 X5 — the vocabulary: dead classes dropped, NUDGE redefined

* **(a) Measure.** The trainer's contract census at load (WP-B DESIGN §3.9.10: no exact LANE_CHANGE action; reversing
  rows carry no action or geometry goal; no absence class on a band < 8 s; exact masks equal their one bit; SPEED_BAND
  never supervised) and the per-class support table (windows, clips) printed with every tactical number.
* **(b) Baseline (MEASURED).** refcv7: NUDGE 23.5 % of the lateral mass, 10.3 % / 12.1 % corroborated by ego geometry
  (D2 / D4); 3 of 8 lateral and 1 of 8 longitudinal classes with zero windows (D1). v9: every class populated (WP-A
  §S3.3); both releases read 0 on all census counts (WP-B DESIGN §3.9.10).
* **(c) Bar.** Every census count **= 0** on the launch release; every tactical class with eval support **≥ 30**
  windows has its own F1.
* **(d)** A census.
* **(e) Control.** The real releases read 0 (MEASURED).
* **(f) Deliberate regression.** A release with one NUDGE target injected must be REFUSED at load.
* **(g) Outcome.** FAIL ⇒ launch blocker.

### 8.6 X6 — NavSim navhard

The §6 block. **Bar: BAR-R8-N2** on the LEGAL row. Baseline: refcv7 30k 0.2269 vs STOP 0.2985, FAILED. Control,
regression and outcome table: §6.2.

### 8.7 X7 — distance keeping

* **(a) Measure.** On S2 and S-ROUTE windows with a v9 in-path lead (`lead_valid`): the plan's minimum time gap and
  minimum TTC to the lead's logged future (join3d); **LEAD-VIOL** = share of lead windows whose plan's minimum time gap
  is < 1.0 s while the human's is ≥ 1.0 s.
* **(b) Baseline.** **PENDING:** refcv7-50,400 through the same instrument (the battery's distance-keeping block, X8).
  Context (MEASURED): refcv7's agent store scope x ≤ 61 m excluded 84 % of joined agents (D4); NavSim navhard NC-zero is
  94.8 % front collisions (D6).
* **(c) Bar.** refcv8 − refcv7 on LEAD-VIOL **< 0, separated**, at both seeds; and the LONGITUDINAL family carries the
  distance-keeping metrics with n (completeness).
* **(d)** §2.2.
* **(e) Control.** The human path reads LEAD-VIOL = 0 by construction; STOP's TTC-censoring count is reported.
* **(f) Deliberate regression.** The lead taken from a deranged window must change LEAD-VIOL (the instrument reads the
  lead).
* **(g) Outcome.** FAIL ⇒ the FOLLOW lead-gap / time-gap constraints from v9 as a longitudinal condition (§13 Q2); the
  agent-store range extension (§13 Q4).

### 8.8 X8 — the refcv7-50,400 four-family baseline row

* **(a)** The battery on refcv7-50,400 under the A7 ruling, all four families on S2 / S6, both seeds.
* **(b) MEASURED status:** G0-A6 FAIL as coded, PASS under the A7 text ruling (`battery/raw/step50400/g0_A6_text.json`);
  `battery_summary.json` carries no family number.
* **(c) Bar (precondition, not a model bar):** the baseline row exists before the refcv8 launch.
* **(e) Control / (f) regression:** the battery's VOID gates and M1.
* **(g)** FAIL ⇒ every §5 bar reads PENDING; the launch proceeds only on the Master Mind's written waiver, named in the
  launch record.

### 8.9 X9 — training seeds

§9. A second training seed of the full run is a PI compute decision. Without it, every claim is restricted as §9 says.

### 8.10 X10 — pose-to-image timing, night by brightness, left-turn n

* **(a) Measure.** **POSE-SYNC** = mean \|t_pose − t_image\| after interpolating poses to the camera timestamp;
  **NIGHT-LUMA** = the night stratum defined by mean luma < 30 (not the clock label); **LEFT-N** = every left-turn
  result with its clip count.
* **(b) Baseline (MEASURED, D3):** poses lead the image by 0–34 ms (mean 16.5 ms ⇒ 0.19 m mean longitudinal offset); ~8 %
  of clips are actually dark vs 46 % labelled night; left-turn route following rests on 13 eval clips.
* **(c) Bar.** POSE-SYNC **≤ 1 ms** if the interpolation is in the build; otherwise the row reads NOT BUILT. NIGHT-LUMA
  and LEFT-N are report-only.
* **(e) Control.** A synthetic constant-velocity track shifted by 16.5 ms moves by v × 0.0165 m exactly.
* **(f) Deliberate regression.** Interpolation off must read ≈ 16.5 ms.
* **(g) Outcome.** NOT BUILT ⇒ R8-7 reads FAILED on X10; the next arm is the interpolation in the dataset reader
  (Data FlyWheel), a label-time fix with its own G-CLOCK-style guard.

---

## 9. The training-variance rule (H-ESTIM-SEED-1) — what a single-seed refcv8 run may claim

**The fact.** MEASURED 2026-09-05 on the v7-tiny rig: a replicate with A0's flags and A0's seed, zero levers moved,
produced "separated" differences on **6 of 42** family cells (14.3 %; CLAUDE.md, `H-ESTIM-SEED-1`). The episode-cluster
bootstrap resamples EPISODES with the models held fixed; it never answers "would another training run say this?".
The ladder draft quotes a corrected 9.5 % (4 / 42, `RETR-2026-09-26-YAWMASK`) measured on a different tiny rig
(INHERITED); this SPEC uses neither number as a full-scale floor.

**What the v7-tiny replicate floor supplies.** For each LEVER the ladder tests (`SPEC_WPB_LADDER_DRAFT.md` L1 recipe,
L2 tactical budget, L3 map weight), a per-metric training-run floor on THAT rig, F = max(\|V0r − V0\|, \|V-R8r − V-R8\|),
and the ratio R = E / F. A lever earns its place in the refcv8 argv only with a separated CI **and** R > 1 (its
regression arm failing). The floor is a property of the tiny rig: it is **not transferred as a number** to the
full-scale run, and it is never used to declare a full-run difference "real".

**What a single-seed refcv8 run may claim.**
1. Every refcv8-vs-refcv7 bar compares **two single training runs** (refcv8 seed 0 vs refcv7 seed 0). A PASS is worded:
   *"the refcv8 run beats the refcv7 run on <metric>, separated over episodes, at both sampler seeds; training-seed
   floor at full scale unmeasured."*
2. ⛔ A full-run result is **never worded as a lever effect** ("tactical conditioning lifts route following by X").
   Lever attribution comes only from:
   * the frozen-trunk arms (R1, with its seed-1 training replicate; SPEC_WPB, with T0r);
   * the ladder rungs, with their replicates and R > 1;
   * eval-time intervention rows of ONE refcv8 checkpoint (RC-OFF, RC-SHUF, NAVSHUF, CTRL-SHUF, forced conditions,
     VMAX-OFF / VMAX-SHUF, UNKNOWN). These compare rows of the same weights, so they carry no training variance; they
     still carry inference variance (both sampler seeds).
3. **Inference variance:** refcv8 samples (DDIM ε at eval, allocation noise). Every planner bar holds at sampler seeds 0
   AND 1; a margin within 2× \|m(s0) − m(s1)\| is NOT PROVEN. Per battery A7.4, the seed-0 / seed-1 floor is one draw
   and is quoted as such.
4. **If the PI authorises a second training seed (X9):** `refcv8-r101-s1`, the identical argv except `--seed`. Then
   every refcv8-vs-refcv7 bar must PASS on BOTH training seeds, \|s1 − s0\| is reported per metric as the full-scale
   training floor, and a bar whose margin is within 2× that floor reads NOT PROVEN. Cost: ESTIMATED as one more run
   (PLAN §0.1: ~3.5–4 days of Thor). This SPEC does not assume it.

---

## 10. DECIDED BY PENDING EVIDENCE — the rules are fixed now; the outcomes are not known

Each item closes by an amendment that cites the deciding artifact (path + hash). The RULE below never changes. A
lever whose deciding evidence has not reported when the launch is decided is **OUT** (the default), never in by
expectation.

### 10.1 Which WP-D perception levers enter (decided by P-BOX and P-MAP, `PREREG_WPD_PROBES.md` sha256 `c054190b…` + `PREREG_WPD_A1.md` sha256 `eb6de12b…`)

* **A P-BOX lever** (PB1 CDN, PB2 hybrid groups, PB3 quality-aware presence, PB4 per-layer refinement, PB4h HQS, PB7
  ray embedding, PB8 BEV point sampling) **ENTERS iff ALL hold:**
  1. its registered bars PASS as written: ΔAP@2 m after NMS r 2 m vs PB0 ≥ +0.02, paired CI > 0 and > \|PB0r − PB0\|
     (or, for PB1 / PB2 / PB3, boxes per detected object −0.30 with ΔAP@2 m ≥ −0.005); no band's AP@2 m after NMS
     drops by more than 0.02; for PB3 / PB4 / PB7 / PB8 also ΔAP@1 m raw ≥ +0.02 under the same rule;
  2. P-BOX is not VOID: R-pres0z, R-memshuf and R-qshuf hold, and C-cache, C-fwd, C-loss and C-step PASS;
  3. its new path is zero-initialised, so I-1 (§3.1) holds with it attached.
  If PB6 (the bundle) PASSES its "dramatic" bar, B2 enters as one bundle.
* **WP-D's reading caveat, carried as a disclosure:** PB0 (the decoder continued alone on its frozen memory) got WORSE
  than refcv7's step-0 decoder (AP@2 m NMS +0.0301 [+0.0190, +0.0412] for step 0 − PB0; MEASURED, descriptive,
  `raw/pbox_step0_vs_PB0_descriptive.json`). Every lever's arm − step 0 is reported beside arm − PB0; the entry rule
  reads the registered comparison (vs PB0) as written.
* **A P-MAP lever** (PM1 stride-4 near lift, PM2 deeper-stage fusion, PM3 line-aware target, PM4 bundle) **ENTERS iff**
  its registered bar (P-MAP bars 1–4) PASSES with R-s4zero, R-s4shuf and R-gtshift holding. PM2 enters only
  zero-gated (it shifts the planner's BEV input).
* **B1** (keep training the heads: warm start, re-warmed LR, EMA eval; W1 / W2) is in by construction. **Guard
  (literal, WP-D §2.7):** if the EMA box3d AP@2 m on the in-run eval set at refcv8 step 1,000 is below refcv7-50,400's
  value on the same set, the head re-warm is cut to the trunk's 5e-5.
* **B3** (one detector for the planner) has no registered probe. **Default OUT.** It enters only after a registered
  R1-harness arm that reads box3d slots shows route following (R8-4 iv) non-inferior at both seeds.
* **M6** (a map-owned 0.25 m encoder) has no registered test. **Default OUT**; the ladder's L3 (×4 map weight) is the
  registered alternative (§10.3).
* **B4 (c) LiDAR depth:** OUT (PI decision 9, default defer). **P-TEMP:** OUT (PI decision 10).

### 10.2 Branch or full warm-started run (decided by the R1 bars, `SPEC_R1.md` sha256 `e9873e12…` + `SPEC_R1_A1.md` sha256 `cf3caf0b…`)

* **Default (PLAN §0.1): a warm-started FULL run, trunk trainable.**
* **The run becomes a head-only BRANCH** (trunk frozen; the planner-side heads trained as in R1) **iff ALL hold:**
  1. the first R1 arm in the order H2 → H3 → H4 → H5 clears the R1 gate (SPEC_R1 §6: at both sampler seeds, turn
     direction ≥ 0.95 with its paired gain vs H0 CI lower > 0; heading-15 ≥ 0.70; straight ΔADE ≤ +0.05 m with CI
     upper ≤ +0.10 m), H2s fails, and the training-seed-1 replicate of that arm clears again;
  2. no lever that needs the trunk to train is in the argv. That excludes B1 (P-BOX PB0 measured that refcv7's late
     box gain is a trunk / memory gain), PM2, M6, and the ladder's L2 / L3 multipliers;
  3. the PI waives R8-6 for the branch (R8-6's B1 lever is a full-run lever).
* ⇒ Stated now, before R1 reads: **unless the PI waives R8-6, the branch is ruled out by (2) and (3)**, and R1 informs
  the conditioning recipe (§10.3) rather than the run type.
* If no R1 arm clears: R1's TRAIN-FIT diagnostic decides the reading (fails on its own training windows ⇒ the trunk
  must learn it ⇒ the full run stands; fits TRAIN but not EVAL ⇒ generalisation, reported).
* A from-scratch run (~6 days) only if a required lever cannot be attached with I-1 holding.

### 10.3 The loss-budget weights and the conditioning recipe values

* **R8-RECIPE** (`--r8-n-alloc`, `--r8-alloc-emit`, `--w-r8-alloc-l1`, `--w-r8-listwise`, `--w-r8-sat`,
  `--w-r8-subscore`, `--r8-lat-prior-dropout`, `--r8-prior-free-group`): **SPEC_WPB §6's reading rule** (registered,
  sha256 `c952d4b4…`) on the frozen trunk. The first arm in the order T1 → T2 that clears B-ROUTE, with T1d / T2d
  failing, names the base. T2 is adopted over T1 only if T2 − T1 is CI-separated on turn direction or heading-15 at
  both seeds. X1h, T2s, X2a and X2b are adopted only on a CI-separated gain over their base arm, with straight ΔADE ≤
  +0.05 m. Then ladder **L1 (B-RECIPE)** must clear on the trunk-trainable tiny rig. If no WP-B arm clears B-ROUTE, the
  run carries T1's seams with the listwise selector, and refcv8's own R8-4 (iv) is the only deciding bar.
* **R8-BUDGET** (`--w-tac-v6`, `--w-map-hires`): `SPEC_WPB_LADDER_DRAFT.md` L2 / L3, **once registered** (it is a DRAFT
  at this writing, §13 Q9). `--w-tac-v6` = 1.0 × k, with k from the ladder's §4.2 rule, iff B-PREM and B-BUDGET pass;
  `--w-map-hires` = 4.0 iff B-PREM and B-MAPW pass. Otherwise refcv7's 1.0 stands for each. A PREMISE-UNMET arm is
  never read as a lever negative.
* ⚠️ The values in the WP-B smoke argv (`--w-r8-sat 0.1 --w-r8-subscore 0.5 --r8-lat-prior-dropout 0.3 --r8-n-alloc 32
  --r8-alloc-emit`) are smoke placeholders. They are not this SPEC's values until R8-RECIPE closes.
* **R8-WARMUP** (Master Mind ruling; proposed default from WP-D W1 / W2): re-warm both LR groups over 1,000 steps to
  refcv7's peaks (head 1e-4, trunk 5e-5 under `--opt dd`), cosine over the run; EMA 0.9998 evaluated at every in-run
  eval and saved; the B1 guard above.
* **Steps and cost:** ~30,000 steps (PLAN §0.1, ESTIMATED 3.5–4 days at ~10–11 s/step). The projected s/step must be
  **≤ 10.5** (the PI's approved refcv7 ceiling, `pi_cost_approval.json` `max_s_per_step` 10.5); above it, the PI decides
  before launch.

---

## 11. PI decisions still open, each with its default

| # | decision | default (what the run does if the PI does not overrule) | consequence of the alternative |
|---|---|---|---|
| **11** | road-geometry speed information admissible inside the route input? | **admissible** (Master Mind provisional ruling): RC-A50 with the certified noise, dropout 0.3 | `--r8-no-rc`; §4.3 rows NOT APPLICABLE; R8-4 (iv) bar 4 (the RC-OFF row) becomes the main route row |
| **8** | drop the 8 train clips that share a recording with eval clips or duplicate a video? | **keep** the refcv7 train corpus; mask the 693 ego boxes only; every eval139 number is also reported on the eval clips that share no recording with train (the leak-free sensitivity row, PLAN decision 8) | the train corpus is re-selected (≈ 0.2 % of train); refcv8-vs-refcv7 comparability carries that stamp |
| **2** | N2 or N3 as the past-only speed proxy | **N2** (past-20-s max snapped up the road-law ladder, urban floor 50 km/h) with an unknown row at p 0.45 | N3 (coarse urban / rural / motorway); X3's bars are unchanged. Either needs X3 built (§8.3) |
| **9** | LiDAR depth as an auxiliary target | **defer**: not in refcv8 | a Data FlyWheel build (~342 MB of streaming per clip, ESTIMATED ≈ 1.5 TB) and a P-DEPTH ladder arm before any entry |
| **10** | past ego-motion for temporal BEV warping | **not in refcv8** (P-TEMP stays deferred until the PI rules; PLAN's provisional reading is "admissible under the 2026-09-02 ruling") | P-TEMP is registered in PREREG_WPD_PROBES and would run on the ladder before entry |

Also owed to the PI (no default assumed): a **second training seed** (X9, §9); the **NavSim leaderboard examination**
before any submission (§6); a launch whose projected s/step exceeds 10.5.

PLAN decisions 1, 3, 4, 5, 6 and 7 are resolved or superseded: 1 and 6 by §10.2; 3 (nav L2) by the v9 release; 4 by
R8-6; 5 by E2′ (RC-A50, conditional on 11); 7 by §10.3.

---

## 12. Amendment protocol

1. **Registration.** The Master Mind renames this file `SPEC_REFCV8.md`, records its sha256 and the UTC time (in
   `Project Steering/` beside it, e.g. `SPEC_REFCV8_SHA256.txt`) **before any refcv8 number exists**: no refcv8
   training step's in-run eval row, no refcv8 eval, no refcv8 NavSim score.
2. **Amendments** are appended as dated sections A1, A2, …, each stating: the time (UTC and Berlin); exactly which
   refcv8 data had been read when it was written ("none", or a list); what changes; why; and what does NOT change. The
   file's sha256 after each amendment is recorded the same way.
3. **Goalposts.** Changing a bar literal, a window set, an estimator, a control or a regression arm AFTER any refcv8
   number exists is a goalpost move. It needs the PI's dated word, and every result under the old text stays on the
   record as written: FAILED stays FAILED.
4. **Instrument corrections** (a control found ill-posed, as WP-D A1's `R-pres0` under a warm start) are admissible
   only before the affected row's first number. A control that fails after data makes the row VOID or INCONCLUSIVE,
   never PASS.
5. **§10 items** close by an amendment that cites the deciding artifact (path + hash); the rule in §10 is not edited.
6. **Non-binding numbers.** Smoke, early and pipeline-validation numbers are stamped `binding: false` and never enter a
   bar.

---

## 13. OPEN QUESTIONS FOR THE MASTER MIND

* **Q1 — I-2, allocation identity.** Allocation attached with emission OFF moves `traj` and base scores (CPU 5.7e-6,
  Thor 2.1e-5) while `sel_idx` holds. Accept the proposed I-2 literals (`sel_idx` 100 %, max \|Δ traj\| ≤ 1e-3 m,
  max \|Δ base score\| ≤ 1e-4)? And choose: allocation emitted from step `S_emit` > 0, or identity stated for the
  117-candidate base fan only? `iw_diag.py`'s result (Thor) should be read first.
* **Q2 — the v9 constraint vectors are not consumed.** `lat_c`, `lon_c` and `speed_goal` (stop distance and time, lead
  gap and time gap, target speed and time to reach) are not supervised; the constraint heads regress plan-derived
  [0, 6] s quantities. R8-1's "constraints like distance and time" and R8-4 (ii) STOP-at-d / X7 FOLLOW therefore rest
  on partial machinery. Build before launch, or launch and report R8-1 constraints as partly delivered? Also owed:
  the label ↔ tag agreement on the [2, 6] s overlap (v9 band [2, 8] s vs plan [0, 6] s; WP-B DESIGN §3.9.4).
* **Q3 — NavSim LEGAL speed input.** Default here: the unknown row (NavSim's `AgentInput` has no map and only a short
  ego history). Is a map speed limit leaderboard-legal? The refcv7 A1 bar arm fed it, so refcv7's NavSim numbers are
  not LEGAL-row baselines; BAR-R8-N4 needs a refcv7-50,400 LEGAL roll (dev-box GPU).
* **Q4 — agent-store range (X7).** D4 measured that x ≤ 61 m excludes 84 % of joined agents. I found no range change in
  the WP-B overlay. In scope for refcv8?
* **Q5 — X3 before launch?** The smoke argv keeps refcv7's v8 future-max sidecar. If refcv8 launches with it, X3 reads
  FAILED on LEAK at launch, and every speed number carries the K12 oracle stamp.
* **Q6 — MODEL_REGISTRY on D:.** The D: working-tree `MODEL_REGISTRY.md` (mtime 2026-10-01; HEAD `37645fc`) has no
  refcv7 row. PLAN cites block `REFCV7-2026-10-04-FINAL`, which is only at the lander tip. Every refcv7 number here is
  from raw JSON on D:; please cross-check against that registry block before registration.
* **Q7 — primary route surface.** I chose S-ROUTE (the dense 4,634 windows: every one of the 2,317 GT-turn windows, 43
  episodes) over S-GRID (107 GT-turn windows), where the PI's quoted 0.84 / 0.51 live. Confirm.
* **Q8 — R8-1-REACH key.** Which in-run key carries the supervised-window share under v9 partial labels (refcv7:
  `tac_label_rows`)?
* **Q9 — SPEC_WPB_LADDER is a DRAFT.** §10.3's budget rule depends on its registration.
* **Q10 — X10 pose interpolation** is not built. In scope, or accept R8-7 FAILED on X10?
* **Q11 — R8-4 (iv) bar 4** (RC-OFF ≥ 0.90 / ≥ 0.60) is my proposal, not from any registered document. Confirm or
  strike.
* **Q12 — B3 default OUT.** WP-D ranks it third and proposes a zero-gated entry; with no registered probe I set it OUT.
  Register the R1-harness box3d-slots arm, or accept OUT?

### 13a. Master Mind rulings on Q1–Q12 (2026-10-04 ~22:00 Europe/Berlin; draft-stage, before any refcv8 number exists)

* **Q2 — BUILD before launch.** R8-1 is a binding PI requirement ("with corresponding constraints like distance and
  time"). The trainer must supervise the v9 constraint vectors (`lat_c`, `lon_c`, `speed_goal`: turn start/end time and
  distance, stop distance and time, lead gap and time gap, target speed and time/distance to reach), masked where the v9
  row is PARTIAL or absent. "Partly delivered" is not an admissible launch state. Assigned to WP-B, together with the
  [2, 6] s label ↔ tag agreement owed by DESIGN §3.9.4.
* **Q5 — BUILD before launch.** X3 enters as the past-only speed input **N2** (PI decision 2 is still open; N2 is the
  plan's default, leak 3.46 %), with a trained "unknown" row (input dropout), and the ceiling applied to the EMITTED plan.
  A refcv8 launch carrying refcv7's v8 future-max sidecar is refused. Assigned to WP-B; N3 behind a flag so decision 2
  needs no new build.
* **Q1 — accept the I-2 literals** (`sel_idx` 100 %, max |Δ traj| ≤ 1e-3 m, max |Δ base score| ≤ 1e-4). Identity is
  checked with allocation EMISSION OFF; emission switches on at a declared step `S_emit` chosen by WP-B from `iw_diag.py`'s
  Thor result and written into the launch argv before registration. An argv emitting allocation at step 0 fails the
  identity row.
* **Q3 — LEGAL speed input = the unknown row.** NavSim's `AgentInput` carries no map; a map speed limit is privileged.
  BAR-R8-N4 needs a refcv7-50,400 LEGAL roll (bare command, no route checkpoint, no turn distance, unknown speed row);
  assigned to the EvalFlyWheel, queued behind the running 50,400 NavSim scoring and P1'.
* **Q4 — X7's METRIC is built from the v9 lead tables** (`lead_{train,eval139}.npz`: lead ≤ 1.75 m from the ego's path,
  ≤ 100 m), not from the box store. Extending the box store's x ≤ 61 m range changes the box head's training target and
  is a WP-D decision: OUT unless WP-D registers it.
* **Q6 — the Master Mind cross-checks every refcv7 number against the tip's MODEL_REGISTRY block
  `REFCV7-2026-10-04-FINAL` before registration.**
* **Q7 — confirmed:** S-ROUTE primary; S-GRID (107 GT-turn windows, the PI's 0.84 / 0.51) reported beside it on every
  route row for continuity, never as the bar.
* **Q8 — WP-B names the key** (the v9 supervised-window share logged in-run) before registration.
* **Q9 — SPEC_WPB_LADDER** is registered by the Master Mind when WP-B hands it over; §10.3 binds on that registration.
* **Q10 — in scope, as a loader-side correction:** poses interpolated to the camera timestamp, validated by a
  known-value test (a synthetic constant-velocity track must shift by exactly v·Δt). Assigned to the Data FlyWheel. If
  it is not built and validated before registration, X10 reads DEFERRED with the build named as the next arm, and the
  0–34 ms lead stays a stated bias on every ADE number.
  **RESOLVED 2026-10-04 ~22:20: built and validated** (`FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing/`):
  one shift per window (the NOW row's own delta) applied to the cached 10 Hz track. Per-row interpolation was rejected:
  MEASURED, it adds 0.109 m mean jitter at 0–2 s, ~11× the defect itself. Measured offset 16.5 ms mean, max 35.4 ms;
  targets move 0.0093 m at 0–2 s and 0.060 m at 6 s (mean, eval139); 27 tests, 7 / 7 mutations caught. **ENABLED for
  refcv8** (`--pose-sync-sidecar`, sidecar md5 `ce00a130…` on Thor). The trainer hook lands through WP-B's tree. The
  standalone eval kit stays on the uncorrected clock for refcv7 comparability.
* **Q11 — confirmed.** RC-OFF ≥ 0.90 / ≥ 0.60 stays: it is the bar that protects the leaderboard-legal case (no route
  checkpoint at inference). Calibration disclosure in §14 stands.
* **Q12 — B3 OUT** unless WP-D registers a probe for it before SPEC_REFCV8 registration.

---

## 14. Calibration disclosure — which refcv7 value was visible when each bar was chosen

No bar below was chosen after a refcv8 number (none exists). Every bar was chosen with the listed refcv7 value
visible.

| bar | literal | refcv7 value visible | where the literal comes from |
|---|---|---|---|
| R8-1-REACH | ≥ 0.93 | 0.2357 logged; v9 release 0.9509 | the release census minus 0.02 (this SPEC) |
| NAV-COMPLY / NAV-PREMATURE | ≥ 0.95 / ≤ 0.05 | clip-token compliance 0.685 (dense) | PLAN R8-2 measure column (≤ 5 % / ≥ 90 %), tightened to 0.95 to match R8-4 (iv) |
| R8-3 bars 1–3 | > E-1 / ≤ +0.02 / ≤ +0.05 m | E-1 RC-A50 0.916 / 0.654 (grid); refcv7 0.841 / 0.514 | PLAN R8-3 ("a trivial planner … sets a floor"; "a shuffled-checkpoint arm must lose the gain"); the ladder's B-RECIPE LEGAL rule (+0.05 m) |
| TAC-LAT-TURN | ≥ 0.95 | 0.402 | D0's q\* (registered `SPEC_D0_DOSE_RESPONSE.md`) |
| TAC-LAT-F1 / TAC-LON-F1 | ≥ 0.80 / ≥ 0.60 | 0.492 / 0.351 | this SPEC (ambitious; not from any registered source) |
| CONS-PROG / CONS-HEAD | ≤ 0.0674 / ≤ 15° | refcv7's progress head median 0.176 (D0 §2.3) | D0 σ\* 0.10; D0c (post hoc there, committed in SPEC_WPB B-TAC) |
| (ii) controllability, (iii) consistency | ≥ 0.95 | ABSENT in refcv7 | SPEC_WPB B-CTRL / B-CONS (registered) |
| R8-4 (iv) bars 1–3 | ≥ 0.95 / ≥ 0.70 / ≤ +0.05 m | 0.841 / 0.514 (grid), 0.815 / 0.505 (dense) | the PI's measure column (R8-4 iv); SPEC_R1 / SPEC_WPB B-ROUTE |
| R8-4 (iv) bar 4 | ≥ 0.90 / ≥ 0.60 (RC-OFF) | as above | this SPEC (§13 Q11) |
| BAR-R8-E1 / E3 / NR | < 0 separated / not separated worse | refcv7-50,400 battery values do NOT exist yet | SPEC_REFCV7 BAR-R7-1 / R7-3 |
| BAR-R8-N1 / N2 / N3 | > STOP / > max(STOP, CV, ECHO) | refcv7 30k: navtest PASS 71.88; navhard FAILED 0.2269 vs 0.2985; warmup NOT PROVEN | the refcv7 NavSim suite's registered bars, unchanged |
| R8-5-BOX | ≤ 1.20 boxes / object; conf_ratio [0.5, 1.5] | 1.07 / 1.01 with NMS; 0.995 / 1.068 | this SPEC (1.20); SPEC_REFCV7 A10 §15.3 band |
| R8-6 bars | AP@2 m raw ≥ 0.339; AP@1 m raw ≥ 0.260; ≤ 1.30 boxes / object; planner detector ≥ 0.248; edge IoU_2 ≥ 0.33 / ≥ 0.18; lane ≥ 0.38 | 0.248 / 0.130 / 1.97 / 0.131; 0.235 / 0.118 / 0.308 | PREREG_WPD_PROBES "dramatic" bars (registered), transposed to the full run; BOX-4 by this SPEC |
| X1 | D-CORE ≤ +0.05; REGRET ≤ 1.55 m | +0.159 (grid); 2.072 m (dense) | D-CORE: implied by R8-4 (iv) with fan-contains 1.0; REGRET: 0.75 × refcv7, this SPEC |
| X2 | ≥ 0.90 | 0.674 (prior wrong) vs 0.967 (prior right) | this SPEC (the prior-right neighbourhood) |
| X3 | LEAK ≤ 0.05; CEIL-OBEY ≤ 0.01; UNKNOWN ≤ +0.10 m/s | 0.238 leak; 110 / 2,059 over the ceiling | LEAK: above D4's N2 0.0346; others this SPEC |
| X4 | LABEL-SCOPE = 0; TAC-SHARE ≥ 0.01 | 100 % flipped; −0.0005 | the WP-A / WP-B tests; ladder B-PREM |
| X7 | LEAD-VIOL refcv8 − refcv7 < 0, separated | refcv7 value not yet measured | this SPEC |
| X10 | POSE-SYNC ≤ 1 ms | 16.5 ms mean | this SPEC |

---

## Proposed register rows (the Master Mind owns `GOALS_AND_CLAIMS.md`; nothing is asserted here)

* **H-R8-RUN-1** (proposed): the refcv8 run meets R8-4 (iv) on S-ROUTE at both sampler seeds (single training seed).
* **H-R8-NAVSIM-1** (proposed): the refcv8 run beats STOP on navhard official EPDMS with leaderboard-legal inputs
  (BAR-R8-N2).
* **H-R8-PERC-1** (proposed): the refcv8 run meets the R8-6 "dramatic" perception bars on S-PERC.
* Lever hypotheses stay with their own registrations: H-R8-RECIPE-1 / H-R8-BUDGET-1 / H-R8-MAPW-1 (ladder draft), the
  SPEC_WPB arms, R1, P-BOX / P-MAP.
