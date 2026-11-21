# v7f — THE SIX PROBLEMS: STATUS BOARD

**ArchInf FlyWheel · 2026-09-05 · 0 GPU** (all reads are file reads; Thor's GPU was
**busy** with three sibling `refav1_lon` eval arms at 97–98 % throughout — checked with
`nvidia-smi --query-compute-apps`, not assumed. ⚠️ **The brief's premise "Thor is FREE" is
wrong as of 2026-09-05 20:43 UTC.**)

Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-six-problems/`

⭐⭐⭐ **UPDATED 2026-09-06 — BOTH `UNVERIFIED-CLAIM` ROWS ARE NOW RULED, 0 GPU.**
Follow-up package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/`
**Row 2 → PARTIAL** (the three never-measured T1 arms read 24.66/24.98/22.97 against
`rdw8p30k` 25.58 on the identical clips; and there is **no admissible absolute bar** —
the gate that rejected `splitp30k` never used one). **Row 5 → the effect is NULL** (the
freeze arm's own control falls with its signal). ⭐ **Row 6's missing L3 arm was also run:
`splitp30k` max |t| 1.08 vs a null bar of 2.9 ⇒ L3 is rulable and NO ARM PASSES IT.**
⛔ A live defect was found: the O6 gate in `train_v6_staged.py` rules on `effective_rank`
and never consults `participation_ratio`.

---

## 0. THE ONE THING TO READ FIRST

⛔⛔ **FIVE OF THE SIX PROBLEMS ARE MEASURED ON `v7-tiny`, NOT ON `v7f`. `v7f` HAS NEVER BEEN
TRAINED** (`M32`, re-verified today: no run directory under any name on Thor, no registry
row). `v7-tiny` is a **~19 M-param** proxy — **6.5 % of the 300 M budget** — trained on stage
`S-W` with **every planner objective at zero**. Its `summary.json` `param_report` reads
`total 19,300,297 / budget 300,000,000`.

⇒ **Every row below whose "on what" column says `v7-tiny` is evidence about a proxy.** None
of it is refuted by that, and none of it transfers automatically either.

⭐⭐ **AND THE ONE THE PI SINGLED OUT IS THE WEAKEST OF ALL — TWICE OVER.** *"Non-collapse
solved"* rests on **ONE arm** (`rdw8p30k`); the other arms measured on the same criterion
**FAIL** it; and ⛔⛔ **the bar itself cannot be reproduced — the programme's own reconcile
artifacts read 5.76, 8.56, 20.23 and 40.77 for the same frozen encoder on the same instrument,
and say verbatim *"do not fail any arm on it."*** The stamp that carried the claim was
`Reports/2026-08-28-0757-program-report.md:94` — *"**L1 collapse** ✅ solved at parity
(participation 25.58), **do-not-relitigate**"*. The arm behind `25.58` is not the arm the
programme carried forward, and *"do-not-relitigate"* is exactly the stamp that stops the next
reader checking — the `M33` lesson, one document later.

⭐⭐⭐ **THE FINDING THAT REFRAMES ALL SIX: THE T1 READOUT EMITS ≈ ZERO EGO MOTION.**
MEASURED, `thor:/home/nvidia/t1dumps/emao14_30k/t1.json`, T1, 40 episodes / 6,924 windows:
`speed_bias_mps` **−10.381**, `under_progress` **0.9925**, and the arm is beaten by
**hold-v0 — a zero-parameter copy of its own `v0` input — by 10.2192 m/s
[7.4371, 13.1021], separated against it.** Arm `speed_mae` 10.7028 ⇒ hold-v0 ≈ **0.4836**.
⇒ over the 2.0 s horizon (`dt_s` 0.1 × `horizon_steps` 20) the model predicts a **nearly
stationary vehicle**. **Every distance metric in that read is measuring a dead readout, not a
world model** — and that is why closed-loop and hold-action differ by only ~1 %.

---

## 1. THE BOARD

### ROW 1 — STABILITY

| | |
|---|---|
| **status** | ⚠️ **PARTIAL** — bounded and recipe-dependent, not solved |
| **the claim** | `V7_LAUNCH_GATE.md:~370` — *"this arm trained through an OPEN-ENDED intermittent-spiking regime — worst window **16,000–18,000** at 4.0 spikes/1000, max gnorm **1.24e10**, i.e. the instability got WORSE late, not better"* (also `GOALS_AND_CLAIMS.md:2413`) |
| **evidence class** | **MEASURED (ours)** — `thor:/home/nvidia/v7tiny/*/train_log.jsonl`, full series, re-derived today across **43 run directories** |
| **the number** | ⛔ **THE CLAIM UNDERSTATES IT.** True max gnorm is **1.931e10 at step 21,800**, not 1.24e10 — the quoted figure is the step-**18,000** spike (1.2379e10). The worst window is **not** 16,000–18,000. **31 spikes >10× median, 21 of them in the late half.** `k60p30k` at clip 1.0 **DIVERGED** (2.106e9 at step 9,000, killed, dir renamed `.DIVERGED-clip1.0.DO-NOT-CENSUS`). ⭐ **Countervailing and equally MEASURED: ZERO genuine NaN/Inf rows in ANY arm** (43 dirs; my first pass reported 150 "NaN" rows per arm — that was **my own probe artifact**, counting *field-absent* log rows, and is retracted here). Stability is **strongly recipe-dependent**: `postrain30k_freeze` gnorm median **0.512**, max **1.08**, **0 spikes**; `splitp30k` median 0.488, max 1.045, 0 spikes; while `k8clip05p30k` runs at median **10.096** with 4 late spikes. |
| **tier** | **T0-DIAGNOSTIC** (training-time telemetry; not a capability tier) |
| **on what** | ⛔ **v7-tiny only.** 19.3 M params, batch 8, stage S-W. **v7f has never trained, so v7f's stability is UNMEASURED** — and it is the arm whose depth and rollout make BPTT instability *more* likely, not less. |
| **what would close it** | **Zero GPU, minutes:** re-issue the corrected spike census as the standing instrument (script banked: `raw/stab2.py`) and put a **gnorm-spike abort** in `train_v6_staged.py` keyed on `>10× running median`, so a v7f run cannot silently train through a 1.9e10 excursion. **For v7f proper:** a **500-step gnorm probe at the real width** before committing the full run (~20 min GPU). ⚠️ Do **not** infer v7f stability from `postrain30k_freeze`'s clean trace — the freeze is exactly what removes the unstable path. |

⇒ **Learned-the-world vs echoed-the-action?** ⛔ **Cannot distinguish.** Gradient norms are
blind to both. This row constrains only whether a run *completes*.

---

### ROW 2 — NON-COLLAPSE ⭐⭐ (the PI's flagged row)

| | |
|---|---|
| **status** | ⚠️ **PARTIAL (was UNVERIFIED-CLAIM; re-ruled 2026-09-06)** — the ARMS are fine and the CRITERION is retired. ⭐ The three T1-read arms, never measured, now read `emao14_30k_tauramp` **24.975** / `emao14_30k` **24.659** / `o14fut30k` **22.969** against `rdw8p30k` **25.582** on the identical 12 clips (`rdw8p30k` reproduces its banked 25.583 at **rel 0.0000**; `splitp30k` reproduces 6.383 as **6.384**) ⇒ **the arms we evaluate are NOT collapsed.** ⛔ But the bar question resolves the other way from the way this row framed it: **NO absolute participation bar is admissible at all** — `v6.py:1521-1567` and `test_participation_floor_provenance.py` (6 passed) already pin that a scalar floor is invalid across corpora AND across d (`z_op` 2048 vs DINOv3 1024) — **and the kill-gate that actually rejected `splitp30k` never referenced 8.56**, so those verdicts were never floor-dependent. See `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/` §1. |
| **the claim** | `Reports/2026-08-28-0757-program-report.md:94` — *"**L1 collapse** ✅ solved at parity (participation 25.58), **do-not-relitigate**"*. Restated in `V7_LAUNCH_GATE.md` P5 as *"**L1** no collapse … 3.80/3.62 → **25.58/26.96** ✅"* |
| **evidence class** | ⛔ **INHERITED as stated.** The underlying numbers are MEASURED, but the *"solved"* generalisation is a document claim, and the gate's L1 row **attributes them to the wrong arm** (see below). |
| **the criterion** | `V7_RECIPE_AND_SCALEUP.md:266` — **participation (σ²), VAL-SIDE, ≥ 8.56** (frozen DINOv3 on our frames). *"⛔ Never `effective_rank` (C132: they disagree up to 141×)"* |
| **the number** | ⛔ **(a) THE CRITERION IS ARM-DEPENDENT AND MOSTLY FAILS:**<br>• `rdw8p30k` **25.583 / 26.965** ✅ — `…/2026-08-19-simwam-analysis/raw/gateb_panel.json` (= `MODEL_REGISTRY.md:4282`)<br>• `splitp30k` **6.38 / 7.63** ❌ — `MODEL_REGISTRY.md:4332`, whose own text reads *"Participation drops 4× (25.58 → 6.38) and **the pre-registered kill-gate REJECTS the arm on rank**"*<br>• `champ30k` **6.499** ❌ — `…/2026-08-19-simwam-analysis/raw/v7tiny_val_rank_5way.json` (in-repo; identical values to `thor:/home/nvidia/v7tiny/val_rank_3way.json`, which is **not** in the repo), `_floor 8.56`, stamped *"VAL (unbiased) — NOT the O4-biased train-pooled gate stream"*<br>• 2k arms `k1` 4.052, `sub64c` 3.564, `cap` 3.548, `lewm` 3.5, `sub32c` 3.329 ❌<br>⇒ **1 PASS, 8 FAIL.** And **`postrain30k`, `postrain30k_freeze`, `emao14_30k`, `emao14_30k_tauramp`, `o14fut30k` have NO banked val-side participation at all — none of the three T1-read arms has ever been measured on L1.**<br><br>⛔⛔ **(b) AND THE BAR ITSELF CANNOT BE REPRODUCED — WHICH MAKES EVERY VERDICT IN (a) UNDETERMINED.** Same frozen encoder (`dinov3-vitl16`), same instrument (`spectrum_report → participation_ratio`), and the programme's own reconcile artifacts read **four different values**: **5.756** on *the same 12 val clips as the 8.56 provenance* (`raw/h_rank16_floor_valclips.json`) · **8.56** (`raw/v6F_v7tiny_rank_probe.txt:41`) · **20.23** at n-matched 1440, 20.516 full-sample (`raw/h_rank16_floor_reconcile.json`) · **40.77** (`reference_E_TRUNK_3`). Their verdicts are verbatim: *"NOT explained by n alone … The floor's 8.56 came from a different corpus or a different pooling; **it must be re-derived before any arm is judged against it**"* and *"a FOURTH value (5.76) … **do not fail any arm on it**."*<br>⇒ At a bar of 5.76 most arms pass; at 20.23 only `rdw8p30k` does; at 40.77 **nothing does, including `rdw8p30k`.** |
| **tier** | **T0-DIAGNOSTIC.** ⚠️ **Stream discipline (H-RANK-9): the in-trainer stage gate pools the O4-BIASED TRAIN stream and OVERSTATES participation (~5.5 gate vs ~3.4 val on the same model).** Never mix the two in one comparison — the "solved" number and the failing numbers must both be val-side. |
| **on what** | ⛔ **v7-tiny.** `25.58` is `rdw8p30k`, a *two-term scratch* arm (O5+O6, no teacher, `--o5-k 1`). The gate's P5 table pairs that L1 number with an L2 result from `splitp30k` — **two different arms in one row**, and `splitp30k` is the one that fails L1. |
| **what would close it** | **~10 min GPU per arm, no training.** Run the existing participation probe on the **three T1-read checkpoints** (`emao14_30k`, `emao14_30k_tauramp`, `o14fut30k`) plus `postrain30k_freeze`, val-side, and publish the table with `_floor 8.56` beside every cell. Until then *"non-collapse is solved"* must be written *"`rdw8p30k` clears L1; the arms we evaluate do not, or were never measured."* |

⇒ **Learned-the-world vs echoed-the-action?** ⛔⛔ **STRUCTURALLY CANNOT DISTINGUISH, AND THE
CRITERION SAYS SO ITSELF.** `V7_RECIPE_AND_SCALEUP.md:266` carries: *"⛔ **rank is NECESSARY,
NOT SUFFICIENT (C131): v1-era had the HIGHEST RANK EVER MEASURED HERE and no environment
interpretation**"*. ⭐ **v1 — the family the PI says failed by mimicking ego dynamics — is the
programme's own participation record-holder.** A high participation number is therefore
*positive evidence of nothing* about the failure v7f exists to avoid.

---

### ROW 3 — PREDICTION QUALITY

| | |
|---|---|
| **status** | ⛔ **OPEN** — and the T1 half is worse than "open": the readout is dead |
| **the claim** | `V7_LAUNCH_GATE.md` P1 — *"**In every distance metric, for every arm, the closed-loop arm is worse than its own hold-action control.** All three `holdv0=LOSES_TO_HOLDV0`"*; and `GOALS_AND_CLAIMS.md:1705` (`D-T1-V7-READ`) — *"THE PROGRAMME'S FIRST T1 CAPABILITY READ, AND IT IS A FLOOR, NOT A CAPABILITY"* |
| **evidence class** | **MEASURED (ours)** — `thor:/home/nvidia/t1dumps/{emao14_30k,emao14_30k_tauramp,o14fut30k}/t1.json` |
| **the number** | **T1, 40 episodes / 6,924 windows, episode-cluster bootstrap (`taniteval.ci`, n_boot 2000), horizon 2.0 s (`dt_s` 0.1 × `horizon_steps` 20):**<br>`emao14_30k` cl/ha — ade **14.0688 [11.08, 17.11]** / 13.8793 · fde 26.2969 / 26.1307 · cross 1.3105 / 1.1240 · heading **94.63° [89.38, 100.16]** / 95.13° · speed_mae 10.7028 / 10.9002<br>⭐⭐ **THE DECISIVE NUMBER, and it is not ADE:** `speed_bias_mps` **−10.381**, `under_progress` **0.9925**, and `holdv0_baseline` — *"the arm is WORSE than hold-v0 on speed_mae_mps by **10.2192 m/s [7.4371, 13.1021], separated against it. A 0-parameter copy of the arm's own input outperforms the arm.**"* ⇒ hold-v0 ≈ **0.4836 m/s** vs the arm's **10.7028**. **The model predicts a nearly stationary vehicle over 2 s.**<br>**Four families (binding, per-family):** LON ⛔ `_longitudinal_claim_admissible = false` on all three arms · LAT heading **~94–99° = chance** · TAC **all three arms at chance**: lateral κ 0.0035 / 0.0032 / −0.0065, longitudinal κ −0.0099 / −0.0128 / +0.0054, 5-way κ 0.0029 / 0.0017 / −0.0051 · STR **UNAVAILABLE** (`missing ['route_pred','route_gt']`; per `D-VAL40-NOLABELS` a **corpus** fact — val40 carries v7 labels on 6 of 40 clips — not an instrument gap)<br>**T0 side:** predictor `nrmse` vs constant floor — `rdw8p30k` **0.7903**, `splitp30k` **0.8416** (both beat a constant); predictor cos h=1 0.6224 / 0.5481 (`MODEL_REGISTRY.md` §13.0/§13.0d) |
| **corrections found** | ⛔ **(a) THE "EVERY DISTANCE METRIC" CLAIM IS OVERSTATED.** `emao14_30k_tauramp` cl/ha: cross **1.0722 / 1.1704 = −0.0982 — closed-loop BEATS hold-action on lateral cross-track.** For `emao14_30k`, cl also beats ha on heading (−0.4957) and speed (−0.1974).<br>⛔ **(b) NO PAIRED INTERVAL EXISTS FOR ANY OF IT.** `paired_decision_grade` and `paired_legacy` are **`{}` — EMPTY — on all three arms.** ⇒ per the brief's own rule (*read every claim by its DELTA and INTERVAL, never by the flag*) the only admissible cl-vs-control statement is the **`holdv0_baseline` longitudinal one**, which *is* separated. Every ade/fde/cross/heading delta is a point estimate with no interval.<br>⭐ **(c) A THIRD T1 ARM EXISTS AND IS UNBANKED.** `D-T1-V7-READ` records **two** arms; `thor:/home/nvidia/t1dumps/` holds **three** — `emao14_30k_tauramp` (2026-08-30 08:48), the τ-ramp arm, with the best ade of the three (13.8645). It appears in no register row. |
| **on what** | ⛔ **v7-tiny**, stage S-W, **every planner objective at zero** (`--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0`). ⚠️ Per `D-T1-V7-READ`'s own binding scope clause, *"v7 cannot drive"* is **NOT supported** by this — DINO-WM and Navigation World Models plan closed-loop with no planner objective at all. |
| **what would close it** | ⭐ **FIRST, AND IT IS NOT A TRAINING ARM: find out why the readout emits ≈0 motion.** `speed_bias −10.381` with `under_progress 0.9925` is a **decoder/scale defect signature**, not a slow-learning signature — and it is the same family as the two banked scale traps (`actdiv_thor.py:47` feeding `v/30` where arms trained on `v/10`, per `H-LEAK-1`; and the `SPEED_SCALE = 30.0` finding). **Probe: decode one banked checkpoint on 20 windows and compare the readout's speed scale against the trainer's** (`~15 min GPU`). If it is a scale error, **every T1 number in the programme is recoverable by re-analysis with zero retraining** — the same shape as the `--analyze-only` recovery in CLAUDE.md. **Second:** compute `paired_decision_grade` on the three banked dumps (CPU-only re-analysis, no rollout). |

⇒ **Learned-the-world vs echoed-the-action?** ⛔ **NOT YET TESTABLE AT T1, and the gate's
"one genuinely positive result" is vacuous.** The gate presents `copy_detector = CLEAN,
echo_index 0.0000` as *"v7 traded fake skill by echo for honest absence of skill … a real
structural change."* ⚠️ **A model emitting ≈0 motion cannot echo anything** — `echo_index
0.0000` is what a dead readout produces, not evidence of non-echo. The instrument itself says
so verbatim in the artifact: *"CLEAN here is NOT an admissibility verdict … **This detector
can only refute a copy, never establish skill.**"* ⇒ the anti-echo property of the v7 line is
**UNTESTED**, not established.

---

### ROW 4 — ACTION SENSITIVENESS

| | |
|---|---|
| **status** | ⛔ **OPEN** — the sign is robust; two of the three headline numbers are withdrawn |
| **the claim** | `V7_LAUNCH_GATE.md` P2 — *"THE MODEL DOES NOT USE ITS ACTIONS, AND WE DO NOT KNOW WHY … action moves the prediction **0.4–0.6 %** as much as the scene"*; `MM-E10` — *"H-ARCH-ACTINS RESOLVED: THE PREDICTOR IS ACTION-DEAF"* |
| **evidence class** | **MEASURED (ours)** — `thor:/home/nvidia/staging/actdiv.json`, `actdiv_2k.json`, `actdiv_o1ctrl.json` (all `_evidence_class: MEASURED (ours; Thor)`, `eval_tier: T0-DIAGNOSTIC`, 24 val clips, 8 action variants, **variant_draw: "roll (never permutation)"**) |
| **the number** | **h=1 `ratio_action_over_scene`:** `emao14_30k` **0.004162** · `o14fut30k` **0.004080** · `postrain30k` **0.005947** · `o1ctrl30k` **0.002359** · 2k arms `o14base2k` 0.000212, `enc3f_2k` 0.000119, `enc3f_2k_s1` 0.000035.<br>⭐ **BOTH CONTROLS READ THEIR KNOWN VALUES, which is what makes this admissible:** `C0_identity_max_abs_diff` **exactly 0.0** on every row (forward is deterministic ⇒ small numbers are real absence, not noise) and `scene_spread` **0.18–0.43** (large) ⇒ ⛔ the "degenerate rollout" account is refuted *in latent space*.<br>**Interventions, all measured, all the wrong way:** `MM-E11` O1 restored → **0.40×** against a pre-registered **≥10×** bar; `MM-E19` k=60 → **0.50×** (0.002983 vs 0.005950). |
| **corrections found** | ⛔ **(a) EVERY h≥2 NUMBER IS WITHDRAWN AND THE GATE STILL QUOTES THEM.** `MM-E14` (`GOALS_AND_CLAIMS.md:1719`) MEASURED from `postrain30k`'s weights: ‖W1‖ **8.8386** vs ‖W2‖ **0.0262** / ‖W4‖ **0.0261** — the h≥2 heads are **337× smaller and still at initialisation**, because O5 trains through the 1-step head applied iteratively (`metric_dynamics.py:273`). Its verdict: *"the h=1 columns … STAND; **every h≥2 number in both is WITHDRAWN**."* ⛔ `V7_LAUNCH_GATE.md` nonetheless asserts *"`h2`/`h4` sit at **1.8e-05** on BOTH arms: beyond one step the action does essentially nothing, k=60 included."* **That sentence quotes withdrawn numbers measuring untrained heads.**<br>⛔ **(b) THE INSTRUMENT FED THE WRONG SPEED SCALE.** `H-LEAK-1` (`GOALS_AND_CLAIMS.md:2324`, 2026-09-03): `actdiv_thor.py:47` fed the predictor **`v/30`** where every arm trained on **`v/10`**. Inert for k=8 arms (−5.7 %, −9 %) but **+54 % for `k60clip05p30k`** ⇒ at the trained scale the k=60 arm's spread vs its one-variable control is **1.41× [1.03, 1.78]**, not the banked 0.83×. **MM-E19's "bar not cleared" verdict STANDS; the decomposition ("action response fell 17 %") is WITHDRAWN.**<br>⛔⛔ **(c) THE 0.4–0.6 % IS STEER/ACCEL AT FIXED SPEED, AND AT THE CORRECTED SCALE ESSENTIALLY ALL OF THE MODEL'S CONDITIONING RESPONSE IS THE MEASURED SPEED CHANNEL.** MEASURED, `…/2026-09-03-p2-probe-leak-audit/raw/actdiv.json`, 24 clips / 144 windows, C0 = 0.0 everywhere, 2,000-draw paired clip bootstrap: on `postrain30k` the full `[steer, accel, v]` tuple moves ẑ **0.027902** and **`v` ALONE moves it 0.027047** — while `[steer, accel]` at fixed `v` moves it **0.005612**. ⇒ the full-tuple response is **4.97× [3.64, 7.13]** the command-only response, **and the v-only column reproduces 97 % of it.** ⭐ **So the ~0.5 % "action response" is not merely small — the part that is not small is the model reading back its own MEASURED STATE, which is exactly the v1 failure mode in a new place.** **P2's sign stands; its magnitude is scoped.**<br>⭐ **(c2) AND THE CORRECTED SCALE GIVES A NUMBER THE DRIFT ROW NEEDS:** `postrain30k_freeze` reads **0.000781** against `postrain30k`'s **0.005612** — **the freeze that lowers drift makes the action channel ~7× deader** (its scene spread rises to 0.7503). ⇒ **P3's "lever" and P2 TRADE AGAINST EACH OTHER**, measured on one instrument.<br>⭐ **(d) THE O11 "BREAKOUT" IS ALREADY ADJUDICATED — the gate treats it as an open question worth an 8.6 h arm.** `V7_LAUNCH_GATE.md` says `o11p30k` *"stopped at 7,600 of 30,000 — **killed, not crashed**"* and queues a 4-min probe (#1) to gate a re-run (#4). ⛔ **`thor:/home/nvidia/v7tiny/o11p30k/summary.json` answers it and predates the gate by a week (2026-08-24):** `stopped_by: "PREREG DEGENERATE branch, 2026-08-24"`, reason — *"o11_loss 1.3956 → 0.00046 against the ln4 floor with pick_acc 0.25 → 1.000, **but sep_rel 0.01 → 12–17** and o5 **+18.7 %** … **This is `zhat = f(z) + λ·a`, the trivial minimiser named in the term's own docstring.**"* The last log row confirms it: `o11_sep_rel 12.29`, `o11_pick_acc 1.0`. ⇒ **it was a deliberate pre-registered kill of a degenerate solution, not an abandoned success.** `H-LEAK-1` §5 concurs independently (*"class-(i) positive by construction"*). |
| **on what** | ⛔ **v7-tiny**, all of it. |
| **what would close it** | ⛔ **NOT another horizon or objective arm — two have already moved it the wrong way (0.40×, 0.50×).** The two untested causes, in cost order: **(i) P2(d) teacher-forced targets** — `train_v6_staged.py:6920` encodes `b["future_frames"]`, the TRUE future, so an action-invariant solution is admissible (`UWM-JEPA` arXiv 2605.25313 states the finding *"applies beyond the unitary parameterisation"*); **(ii) P2(b) a command channel with UNITS** — `M32` §4 is the sharpest statement of it: refcv4b's `(a_lon, a_lat)` is a command with units, v7f's `omega_accel_v` *"has no units — it is realised motion played back"*, and `M31` measured that *"a constraint with units binds exactly where its physics says it should."* ⭐ **Cheapest first move is FREE and needs no arm:** re-run `actdiv` at the **corrected `v/10` scale** on the banked `postrain30k_freeze` and `o11p30k` checkpoints (~4 min each) — it retires the last open O11 question and gives the k-ladder a clean denominator. |

⇒ **Learned-the-world vs echoed-the-action?** ⭐ **YES — THIS ROW DISTINGUISHES THEM, AND IT IS
THE ONLY ONE THAT CLEANLY DOES.** `C0_identity` reads exactly 0.0 (a *known value*, not a
comparison) and `scene_spread` is large while `action_spread` is ~0.4 % of it. ⇒ the model
**responds to the world and is specifically deaf to the action** — the *opposite* pole from
v1, which echoed the action and had no world. ⚠️ But note (c): ~5× of even that small response
is the **measured speed channel**, i.e. state playback, not command.

---

### ROW 5 — DRIFT

| | |
|---|---|
| **status** | ⛔ **SOLVED AS A MEASUREMENT · THE EFFECT IS NULL (re-ruled 2026-09-06)** — the control was subtracted **on the freeze arm itself**, and it **falls with the signal**: true **0.3902** (reproducing the banked 0.3905) vs endpoint-shuffled control **0.4487**, share of drift that is arithmetic **1.1488**, control higher on **8 of 8** directions (mean Δ −0.0580). ⭐⭐ **The ordering INVERTS**: the arm that looked 1.71× better on the raw metric is the WORST of the three after subtraction (−0.0585 vs `postrain30k` −0.0077 and `rdw8p30k` +0.0010). ⇒ the freeze lowered the **latent geometry the metric reads**, not any drift property. `H-LEAK-3` confirmed on the arm it matters for; readiness decision **G.1 must not be adopted**; ⛔ **queue item #3 (8.6 h) must not be run — the metric it would be read on is null.** See `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/` §4. |
| **the claim** | `V7_LAUNCH_GATE.md` P3 — *"A REAL, SEED-STABLE 3.3× EFFECT WHOSE CAUSE IS UNKNOWN … **CLOSES WHEN:** one arm resolves it — `postrain30k` + `--freeze-encoder`, **one variable**, matched otherwise"*, listed in the GPU queue as item **#3, ~8.6 h, unrun** |
| **evidence class** | **MEASURED (ours)** for the numbers; ⛔ **the gate's "unrun" status is a stale document claim** |
| **the number** | ⭐⭐ **THE ABLATION RAN ON 2026-08-26 — FIVE DAYS BEFORE THE GATE CALLED IT UNRUN — AND WAS READ.**<br>`thor:/home/nvidia/v7tiny/postrain30k_freeze/`: `summary.done true`, `steps 30000`, `elapsed_s 28,753` (7.99 h, matching the 8.6 h estimate). **A full key-by-key `config.json` diff against `postrain30k` gives exactly TWO differences: `freeze_encoder False→True` and `out` (the directory name). 194 args each, zero keys present in one and absent in the other.** It is genuinely one-variable.<br>**Its drift is banked IN-REPO:** `…/2026-08-24-action-conditioning-and-heldout/raw/freezedrift.json` — **r 0.3905, t 32.21**, HELD-OUT, n_clips 80, n_rows 7680, k 4, pca_band [0,8].<br>**Same-instrument ladder** (identical split/n/k/band): `rdw8p30k` **0.6718** (`latentmotion.json`) · `postrain30k` **0.6674** · `postrain30k_seed1` **0.6741** (both `seedrep.json`) · **`postrain30k_freeze` 0.3905**.<br>⭐ **The gate offered a binary — *"≈0.199 ⇒ freezing explains it; ≈0.669 ⇒ `o5_k` is the cause"*. IT READS NEITHER.** Freeze accounts for **0.2769** of the 0.4684 total gap (**59 %**); `o5_k` 8→4 for the remaining **0.1915** (41 %). **Both variables move it.**<br>⭐⭐ **AND IT SURVIVES THE `H-ESTIM-SEED-1` OBJECTION, which almost nothing here does:** the replicate pair `postrain30k` / `postrain30k_seed1` differ by **0.0067**, so the freeze effect is **≈41× the rig's own run-to-run spread.** |
| **corrections found** | ⛔⛔ **THE DRIFT METRIC DOES NOT MEASURE DRIFT — AND THE CONTROL READS *HIGHER* THAN THE SIGNAL.** MEASURED, `…/2026-09-03-p2-probe-leak-audit/raw/drift.json`, endpoint-shuffled control (`Δz′ = z_{π(t)+K} − z_t`, start kept) through the identical code path, n 80 clips / 7,680 rows, K 4, band [0,8): `postrain30k` true **0.6697** vs **control 0.6774 (t −3.8 — the CONTROL IS HIGHER)**, share **1.0154**; `rdw8p30k` true 0.6737 vs control 0.6727, share 1.0019. **Control ≥ true on 13 of 16 directions.** Verdict (`H-LEAK-3`): *"the whole of 'drift' is class-(ii) arithmetic"* — it measures the decay of frame-specific latent content toward the clip mean, an **encoder time-series property, not a predictor/transition property**; *"E-DEC-40/45/59/60/61/64/66 and MM-E4/E6 **keep their numbers and lose their interpretation**."* It names *"the 0.199-vs-0.669 freeze story"* as affected and flags readiness decision **G.1 (accept P3 as met by `postrain30k_freeze`, 0.3905) NOT to be adopted.** ⚠️ **Status PROPOSED until the PI reads them.**<br>⚠️ **`splitp30k`'s 0.199 is a DIFFERENT INSTRUMENT.** `deltaz_splitp30k.json` is n_rows **1920**, LEAD-MATCHED, RFF+ridge nonlinear (`mean_r 0.1993`, t 8.38, n_scores 160). The 0.6674/0.3905 family is n_rows **7680**, 80 clips. ⇒ **the gate's headline *"0.199 against 0.657–0.679"* is a cross-instrument ratio and is not quotable as a like-for-like 3.3×.**<br>⚠️ **`MM-E4-L1` binds any use of this row:** *"**drift alone is never a valid objective; a drift number quoted without its paired prediction reads is INADMISSIBLE**"* — because innovation-SIGReg cut drift 25 % **and destroyed prediction** (cos 0.2462 → 0.0108). |
| **on what** | ⛔ **v7-tiny.** |
| **what would close it** | ⭐ **ZERO GPU, MINUTES PER ARM, AND IT IS ALREADY SPECIFIED:** the leak audit names the fix — **re-measure drift with the shuffled-endpoint control SUBTRACTED** (`latentmotion.py`). Run it on the four-arm ladder above; the freeze contrast is already one-variable and replicate-anchored, so **the moment the metric is valid the answer lands with no new training.** ⛔ **Do not run the 8.6 h queue item #3 — the arm exists.** |

⇒ **Learned-the-world vs echoed-the-action?** ⛔ **Cannot distinguish, twice over.** Drift asks
how much of Δz is predictable from `z_t` — a *self-consistency* question that is silent about
both the world and the action. And with the shuffled-endpoint control at 100–102 %, it is
currently silent about drift too.

---

### ROW 6 — DECODABILITY

| | |
|---|---|
| **status** | ⚠️ **PARTIAL → and L3 IS NOW RULED (2026-09-06): NO ARM PASSES IT.** The missing deliberate-regression arm was re-scored on the corrected instrument (`splitp30k`, true LOCO, 124 labelled → 70 lead-matched → **24 used**): **max |t| = 1.08 across all 9 (K × target) cells** against the measured null bar **|t| ≥ 2.9**, every CI straddling zero. With `rdw8p30k`, `postrain30k` and `postrain30k_freeze` already re-scored and none separating, **all four arms are in and the predictor adds nothing over `z_t`** at any K on any environment target. ⭐ **A second fact fell out: `ẑ_GT − ẑ_HOLD` is ±0.0017 at worst and exactly 0.0000 in 6 of 9 cells** — feeding the TRUE FUTURE actions and HOLDING the last one give the same rollout ⇒ Row 4's action-deafness and this L3 null are **the same fact seen twice**. L2 is unchanged. See `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/` §5. |
| **the claim** | `V7_LAUNCH_GATE.md` P5 — *"**DECODABILITY ITSELF IS NOT OPEN. L1 and L2 are MEASURED and PASSED**"*, with L3 *"the dissociation gate, and the actual v7 bar"* open, and `splitp30k` as the deliberate-regression arm *"(t −3.69 / −5.62 / −6.26)"* |
| **evidence class** | **L2 = MEASURED (ours)**; ⛔ **L1 as stated = INHERITED and mis-attributed (Row 2)**; ⛔ **L3's t-statistics = WITHDRAWN** |
| **the number** | **L1** — see Row 2: 1 arm passes, 8 fail, 5 never measured, **and the bar itself is unreproducible.**<br>**L2 ✅ genuinely MEASURED, held-out and lead-matched** — `…/2026-08-24-action-conditioning-and-heldout/raw/spatialenv_heldout_leadmatched.json`, corpus `physicalai-val130-heldout`, **n_rows 2,399 / n_clips 24**, LOEO paired, PCA d_eff 128 fit on training clips only: `splitp30k` `n_agents` **+0.1220**, pixel floor +0.0281 (**Δ +0.0939, t 3.75**), **frozen DINOv3 +0.0998** ⇒ **beats the pixel floor, the constant control AND the frozen teacher.** `occ_center` **+0.3351** (Δ +0.5071, t 6.81); `n_free_cols` **+0.2080** (Δ +0.1791, t 10.47).<br>⛔ **BUT L2 IS A 1-OF-5 WIN, NOT A SWEEP, AND THE GATE'S FRAMING OMITS IT:** `splitp30k` **fails the raw-input floor on `occ_left` (−0.0159, t −0.66) and `occ_right` (+0.1728, t −0.62), `beats_raw_input: false` on both**, and frozen DINOv3 beats it on **4 of 5** aggregate spatial targets (`occ_left` +0.3315, `occ_center` +0.3998, `occ_right` +0.3477, `n_free_cols` +0.4857). ⇒ **the one target we win is `n_agents`.**<br>⚠️ **And the same arm's lead targets go the WRONG way:** `lead_gap_m` +0.0063 → **−0.0940** (t −3.91, below the raw-pixel floor); repaired target `lead_range_m` **−0.1611**, worst of any arm (`MODEL_REGISTRY.md:4332`) ⇒ **the environment targets DISSOCIATE; `n_agents` alone must never be quoted as "environment."**<br>**L3 ⛔ UNDECIDED — and its two banked reads CONTRADICT EACH OTHER.** The t-values the gate quotes come from `raw/envpred.json`, which **carries no `split` key — it is the IN-SAMPLE read** (`splitp30k` `n_agents` `zhat` vs `z_t`: K1 −0.0008 t −3.69, K3 −0.0320 t −5.62, K6 −0.0158 t −6.26). Its held-out companion `raw/envpred_heldout.json` (124 val clips, zero overlap) **disagrees in sign at K=1: +0.0122, t 0.64.** `D-P2-LEAK-AUDIT`/`H-LEAK-2` then withdrew the t's outright — `envpred.loeo` hands `probe()` a pooled list in which each "per-clip" score shares ~11 of 12 clips with its neighbours, then divides by √24 as if independent: *"**L3 has never been passed or failed on an admissible read**."* **Corrected re-score** (`…/2026-09-03-p2-probe-leak-audit/raw/l3.json`, true leave-one-clip-out, 24 lead-matched held-out clips): **no ẑ column separates from `z_t` on `rdw8p30k`, `postrain30k` or `postrain30k_freeze` at any K (|t| ≤ 1.5)**. ⛔ **`splitp30k` — the deliberate-regression arm the whole gate depends on — was NOT re-scored; it is QUEUED** (checkpoint Thor-only, ~133 MB pull + ~11 min CPU). ⇒ **the regression arm's "predictor-dead" premise is itself currently unestablished.** |
| **tier** | **T0-DIAGNOSTIC** throughout. ⛔ No decodability number is a driving claim. |
| **on what** | ⛔ **v7-tiny.** L2's winner `splitp30k` is a **frozen distilled encoder** — `MODEL_REGISTRY.md:4332` records it is **TEACHER-DEPENDENT AT INIT**, so it does **not** satisfy the PI's *"clear preference without pretrained labels."* |
| **what would close it** | ⭐ **ONE CPU JOB, ~11 min + a 133 MB pull, and it is the cheapest decisive item on this board: re-score `splitp30k` on the corrected L3 instrument.** Three arms are already re-scored and none separates; the *only* missing arm is the deliberate-regression arm. ⛔ Per the validation standard, **until it is re-scored a PASS by any other arm would mean nothing** — so this single read is what makes L3 rulable at all. Then run the panel against the **measured null bar \|t\| ≥ 2.9** (`null_calibration.py`, 104 draws). |

⇒ **Learned-the-world vs echoed-the-action?** ⭐⭐ **L2 DOES distinguish, and it is the
programme's strongest evidence against a v1 repeat — held-out, lead-matched, with both required
controls.** `n_agents` decoded from the latent at **+0.1220 vs a raw-pixel floor of +0.0281
(t 3.75)** and above **frozen DINOv3 (+0.0998)** is a statement about **other road users** — a
quantity the ego's own action cannot supply, and one **v1 could never have produced.** ⚠️ Scope
it honestly: it is **one target of five**, and the same arm fails the pixel floor on `occ_left`
and `occ_right`. ⛔ **And L2 is about the ENCODER. L3 — does the PREDICTOR add anything over
simply looking at now? — is the question that separates a world model from a look-up, and it is
unanswered on an admissible read.**

---

## 2. SUMMARY TABLE

| # | problem | status | on what | the one artifact it needs |
|---|---|---|---|---|
| 1 | **stability** | ⚠️ PARTIAL | v7-tiny (v7f **unmeasured**) | a gnorm-spike abort + a 500-step probe at v7f width |
| 2 | **non-collapse** | ⚠️ **PARTIAL** *(was UNVERIFIED-CLAIM)* | v7-tiny, **7 arms, corpus-matched, 2 controls reproduced** | ✅ done — the 4 unmeasured ckpts are measured and pass relatively; **retire `O6_PARTICIPATION_FLOOR` from the decision path** (PI) |
| 3 | **prediction quality** | ⛔ OPEN | v7-tiny | **why the readout emits ≈0 motion** (speed-scale probe, ~15 min) |
| 4 | **action sensitiveness** | ⛔ OPEN | v7-tiny | already re-scored at `v/10`; **it is the one row whose evidence is now clean** |
| 5 | **drift** | ⛔ **RULED — EFFECT IS NULL** *(was UNVERIFIED-CLAIM)* | v7-tiny | ✅ done — control subtracted on the freeze arm; **nothing further to run, and ⛔ not queue item #3** |
| 6 | **decodability** | ⚠️ PARTIAL (**L3 RULED: no arm passes**) | v7-tiny | ✅ done — `splitp30k` re-scored, max |t| 1.08 vs bar 2.9; L2 unchanged |

⭐⭐⭐ **UPDATE 2026-09-06 — `UNVERIFIED-CLAIM` rows remaining: ZERO.** Both were ruled with 0 GPU, and they went opposite ways. **Row 2:** the criterion's control DID behave once it was corpus-matched — the never-measured arms pass relatively, and the absolute bar turns out to be inadmissible *by construction* rather than merely unreproduced (it is corpus-specific AND dimension-specific, already pinned in a passing test), while the gate that actually rejected an arm never used it. **Row 5:** the control did NOT behave, the freeze arm's own control falls with its signal, and the 3.3x effect is entirely arithmetic — **the row is ruled and its answer is null.** ⭐ Row 6's L3 was ruled in the same turn. ⛔ **New live defect found:** the O6 gate in `train_v6_staged.py` rules on `effective_rank`, the statistic C132 forbids, against a floor calibrated on synthetic data. Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-v7f-instrument-repair/`.

*(original text, kept for the record)* **`UNVERIFIED-CLAIM` rows: 2 (non-collapse) and 5 (drift).** Both were *marked closed or
closeable in a document* while the artifacts say otherwise, and **both fail in the same way —
the criterion's own control does not behave**: non-collapse is scored against a floor that reads
5.76 / 8.56 / 20.23 / 40.77 depending on the pooling, and drift against a metric whose
endpoint-shuffled control reads **100–102 %** of the signal. ⇒ **Neither is a "we measured it and
it is bad" row. Both are "the instrument is not established" rows** — which is cheaper to fix
than either sounds, and is why neither needs a training arm.

**Rows measured on v7-tiny rather than v7f: ALL SIX.** v7f has never been trained.

---

## 3. ⭐ WHICH SINGLE PROBLEM, IF SOLVED, UNBLOCKS THE MOST OF THE OTHERS

⭐⭐⭐ **PREDICTION QUALITY — specifically, the ≈0-motion readout (Row 3).** It is upstream of
four of the other five, and it is the cheapest to test:

* **It gates every T1 number.** ADE 14.07 m, heading 94.6°, tactical κ ≈ 0, and the whole
  `LOSES_TO_HOLDV0` verdict are all downstream of a decoder that predicts a stationary
  vehicle. Fix the readout and **P1 gets its first informative measurement** — with no
  retraining, because the rollouts are banked (`t1dumps/*/dumps/`).
* **It gates the echo question.** `echo_index 0.0000` is currently vacuous (Row 3). Only a
  readout that actually moves can be tested for echo — i.e. **the v1-repeat question cannot be
  answered at all until this is fixed.**
* **It gates action sensitiveness at the metric level.** If the readout emits ≈0 displacement,
  no action can change the output, so the T1 cl≈ha gap of ~1 % is *guaranteed* regardless of
  how action-sensitive the latent is. Row 4's T0 probes say the latent is 0.4–0.6 % action-
  responsive; the T1 read cannot corroborate or refute that while the decoder is dead.
* **It is a scale-defect signature with two banked precedents in this exact codebase** —
  `H-LEAK-1`'s `v/30`-vs-`v/10` and `SPEED_SCALE = 30.0`. `speed_bias −10.381` against a
  hold-v0 of ≈0.48 m/s is not a slow-learning curve; it is a constant offset the size of the
  corpus's mean speed.
* **Cost: ~15 minutes of GPU, no training, on a checkpoint we already hold.**

⚠️ **It does NOT unblock Row 2 (non-collapse) or Row 5 (drift)** — those are independent
measurement-validity failures, and both are also zero- or near-zero-GPU. They should run in
parallel, not behind it.

---

## 4. WHAT I DID NOT DO, AND WHY

* ⛔ **No v7f training was launched.** The PI's 2026-08-31 directive stands.
* ⛔ **No GPU job of any kind was run.** Thor was saturated by three sibling `refav1_lon` eval
  arms for the whole session; adding load would have violated the never-load-a-busy-box rule.
  Every number above is a file read.
* ⚠️ **The G: mount went fully down mid-session** (errno 22 on reads; a same-breath control
  file read 0 lines **together with** the target, proving a mount outage rather than an
  unreadable file). Affected reads were retried to a clean read; this document was authored on
  local disk and copied in. The off-Drive mirror `C:\Users\Admin\tanitad-wt` is **partial**
  (its `Project Steering/` holds only `BACKLOG.md`) and is not a substitute.
* ⚠️ **Every Thor script was shipped base64 + md5-verified on both sides** before running, and
  every pull was md5-verified end-to-end (`census.json` `fcbb4df0fe0e673347167a3d313058e7`).

---

## 5. PROVENANCE — every artifact this board rests on

**On Thor (never banked until now):**
`/home/nvidia/v7tiny/{postrain30k,postrain30k_freeze,postrain30k_seed1,splitp30k,o11p30k,emao14_30k,emao14_30k_tauramp,o14fut30k,k60clip05p30k,k8clip05p30k,champ30k,…}/{config,metrics,summary,stage_gate}.json` + `train_log.jsonl` (43 run dirs) ·
`/home/nvidia/v7tiny/val_rank_3way.json` ·
`/home/nvidia/staging/{actdiv,actdiv_2k,actdiv_o1ctrl,actinfo,condpath}.json` ·
`/home/nvidia/t1dumps/{emao14_30k,emao14_30k_tauramp,o14fut30k}/t1.json`

**In repo:**
`…/2026-08-24-action-conditioning-and-heldout/raw/{seedrep,freezedrift,latentmotion,deltaz_splitp30k,deltaz_top8,spatialenv_heldout_leadmatched,envpred,envpred_heldout}.json` + `probes/latentmotion.py`, `code/{spatialenv,envpred}.py` ·
`…/2026-08-19-simwam-analysis/raw/{gateb_panel,gatec_panel,e_dec14_split_panel,v7tiny_val_rank_5way,h_rank16_floor_reconcile,h_rank16_floor_valclips}.json` + `raw/v6F_v7tiny_rank_probe.txt:41` (the 8.56/17.25 provenance) ·
`…/2026-09-03-p2-probe-leak-audit/{RESULT.md,raw/{drift,l3,l3_pooled,actdiv,actdiv_pairs}.json,tools/p2_leak_rescore.py}` ·
`…/2026-08-31-mm-e19-k60-horizon/raw/mm_e19_read_step30000.json` · `…/2026-09-01-mm-e19-k8-attribution/raw/MISLABELLED_INSIDE_AS_k60__actdiv_local.json` (⚠️ its internal arm key reads `k60clip05p30k` while the arm measured is `k8clip05p30k` — never quote it by the internal key) ·
`stack/tanitad/eval/spectral.py` (`participation_ratio`:229, `effective_rank`:54 — different statistics) · `stack/scripts/val_rank_probe.py` ·
`MODEL_REGISTRY.md` §13.0 / §13.0d (:4282, :4332) ·
`GOALS_AND_CLAIMS.md` :1705 (`D-T1-V7-READ`), :1719 (`MM-E14`), :2320/:2324 (`D-P2-LEAK-AUDIT`, `H-LEAK-1`), :2413 (MM-E19) ·
`V7_RECIPE_AND_SCALEUP.md`:266 (the L-ladder) ·
`V7_LAUNCH_GATE.md` (P1–P5) ·
`Decisions/2026-09-05-mm-decisions.md` §M32, §M33 ·
`Reports/2026-08-28-0757-program-report.md`:94

**Raw banked with this package:** `raw/census.json` (43 run dirs, md5
`fcbb4df0fe0e673347167a3d313058e7`), `raw/census.py`, `raw/stab2.py`, `raw/diff2.py`,
`raw/t1b.py`, `raw/STABILITY_CENSUS.txt`, `raw/CONFIG_DIFF.txt`, `raw/T1_CL_VS_HA.txt`.

**Evidence-class discipline:** every *status* in this board that cites a document rather than
an artifact is stamped `INHERITED` in its own row, including where I believe it.

---

## UPDATE 2026-09-06 — the two gate defects this board's audit found are FIXED

Package: `…/2026-09-06-v7f-gate-repair/RESULT.md`. **16 of 18 new tests FAIL on the unfixed tree.**
Affected suite `84 passed, 0 failed`. ⚠️ No arm's number moves and **no launch behaviour changes**:
`O6_spectrum` is `reported`, not `required`, at every stage.

| defect | before | after |
|---|---|---|
| **D1** `o6_rank_verdict` (`train_v6_staged.py:5148`, `:7347`) | ruled on `effective_rank` (p ∝ σ) and never read `participation_ratio` | rules on **`participation_ratio`** (p ∝ σ², energy); `effective_rank` still reported, marked `effective_rank_is_ruling: False` |
| **D2** `full_panel.py`, `rolled_predict.py` | baseline = `present[0]`; first arm got no verdict; `None` rendered as "passes gate" | **named** `BASELINE` via `stack/tanitad/eval/panel_gate.py`; one three-valued entry per arm; **no substitute baseline promoted** |

⛔ **The D1 inversion, executed against HEAD:** a representation with **55 % of its energy in one
direction** reads `effective_rank` **769.09** vs a healthier arm's **659.69** — the collapsed arm
reads HIGHER — and the unfixed gate returns **`status=PASS pass=True`**. Participation orders them
correctly (3.31 vs 31.27). Same two readings, opposite verdicts:
`OLD retention 1.0815 ci=[1.056,1.107] -> PASS` · `NEW retention 0.1577 ci=[0.126,0.194] -> FAIL`.

⛔ **The D2 order flip, executed:** with `[rdw8, o7w1p0, o8w1p0]`, `o8w1p0` is **REJECTED** and
`rdw8` has no row; reorder to `[o8w1p0, o7w1p0, rdw8]` and **`o8w1p0` becomes the BASELINE**.

🔴 **PI DECISION SURFACED (`D-O6-FLOOR-PI`):** whether `O6_PARTICIPATION_FLOOR` should gate at all.
The absolute clause is now **REPORTED_NOT_RULING by default** and fires only with an explicit
`participation_floor` **and** a named `participation_reference`. **Recommended default: keep it
disarmed** — 8.56 is reproduced by no live instrument and moves 3.51× on episode diversity alone,
and the `effective_rank` floor of 64 sits on the inverting statistic.

### Row 3 (≈0-motion readout) — still OPEN, narrowed without a GPU, and its blocker named

* ⛔ **GPU BLOCKED, not skipped:** dev-box 4060 at **100 %**, four sibling `refav1_arm.py` arms.
  Polled with a check-command monitor (`arms=0` release condition), sibling queues untouched; no
  slot freed this turn.
* ⭐ **`D-T1-NO-SCALE-ON-EMISSION-PATH` (MEASURED, source):** the T1 emission path carries **no
  multiplicative scale constant anywhere** — `StepDisplacementReadout.forward`
  (`metric_dynamics.py:212`) is a bare MLP with no scale and no output activation; `rollout_decode`
  (`:233-244`) only stacks + `accumulate_se2`; `dense_speed_profile` (`rollout.py:280-284`) is
  `norm(Δp)/dt`. With the input side's `SPEED_SCALE = 10.0` already matching the trained 10.0,
  ⇒ **this row's "if it is a scale error, every T1 number is recoverable by re-analysis with zero
  retraining" is NOT supported by the eval code path.** ⚠️ **Scoped: rules out the HARNESS, not the
  TRAINER** — a readout supervised against a wrongly-scaled target gives the same ≈0 signature and
  that path is *not* closed.
* ⚠️ **A number this row should NOT be given:** quantifying the corpus speed against
  `speed_bias −10.381` was attempted and **withheld**. `windows_*.pt`'s `gt` is at **`wp_steps`
  spacing, not 10 Hz ticks** (`norm(diff(gt))/0.1` gives a nonsense 63.85 m/s), and more importantly
  `windows_flagship-30k` is a **different arm** from v7-tiny. Dead-readout vs mis-scaled therefore
  stays **OPEN** and needs v7-tiny's *own* val speed.
* ⇒ **Revised cheapest probe — ZERO GPU:** load the banked **v7-tiny** checkpoint on CPU and read
  `step_readout_op`'s final `Linear` weight/bias norms. A weight term ≈0 against the bias means a
  **constant emitter**, which would make `echo_index 0.0000` vacuous and mean **no re-analysis can
  recover the T1 numbers**. 🔴 **BLOCKED ON THE ARTIFACT** (verified, not assumed):
  `taniteval/results` holds **29** dumps and **none** is a v7 arm; locally reachable checkpoints are
  refav1 and phase-0 flagship only. **What unblocks it: the path to a banked v7-tiny checkpoint.**

### One more instrument defect found in passing

⚠️ **`D-X4PIN-DEAD`:** `test_x4_layer_spectrum.py`'s pin on `o6_rank_verdict` has been **silently
skipping** — its loader read a historical `v6.py` under a bare module name, hit `v6.py`'s **relative
imports**, and a bare `except Exception` turned that into the skip message *"git could not supply a
pre-X4 revision"*. Git was fine; the loader was not. Fixed, skip reason made honest, contract
updated so the *ruling* keys may move. ⚠️ Its git-history walk exceeds **15 min** on the G: mount —
bounding that runtime is a named work item before it joins a routine `pytest -q`.
