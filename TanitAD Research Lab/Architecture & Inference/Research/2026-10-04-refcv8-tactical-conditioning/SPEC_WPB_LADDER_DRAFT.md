# SPEC_WPB_LADDER (DRAFT for registration) — refcv8 §7: the tiny-rung ladder against the v9 release

*WP-B (Architecture & Inference), 2026-10-04. This replaces the outline in `SPEC_WPB.md` §7. It is a DRAFT for the Master
Mind to register. No arm of it has run, and no number below was produced on the rig. Every number quoted is stamped with
its evidence class.*

## 0. What this ladder decides

The ladder decides three lever questions that the frozen-trunk harnesses (R1, SPEC_WPB arms, WP-D's P-BOX / P-MAP)
cannot answer, because each one needs the trunk to train:

| rung | hypothesis (proposed register ID) | the one variable |
|---|---|---|
| L1 | **H-R8-RECIPE-1**: the refcv8 recipe improves route following over the refcv7 recipe when the trunk trains, and the improvement comes through the tactical condition. The recipe is v9 labels + tactical-conditioned generation and selection + route checkpoint + nav args. | `--refcv8` + the v9 release, versus the refcv7 recipe on v7.2 labels |
| L2 | **H-R8-BUDGET-1**: raising the tactical decoder's share of the trunk update to ≥ 1 % (MEASURED in-run) improves route following, and perception does not pay for it. | `--w-tac-v6` × k (k fixed by the §4.2 rule before the arm runs) |
| L3 | **H-R8-MAPW-1**: raising the 10 cm map's loss weight ×4 gives the map a positive share of its own shared 0.25 m encoder, improves the map, and costs neither boxes nor route. | `--w-map-hires` × 4 |
| L4 | **H-R8-X3-1** (SPEC_REFCV8 §8.3, MM ruling Q5): the past-only speed input N2 carries usable speed information to the plan. A training arm whose N2 is taken from ANOTHER window must lose the speed gain. | the speed input's information (N2 vs N2 rolled by one row) |

**Evidence for the levers.** All of it is MEASURED by WP-D's P-GRAD on refcv7-50,400 (`…/2026-10-04-refcv8-perception-architecture/raw/pgrad.json`,
RESULT §7.2; 64 TRAIN-DIAG windows; all weights 1.0; linearity control 6.6e-6, zero control exactly 0).
- **Whole-trunk projection shares:** agent 0.501, box3d 0.357, planner auxiliaries 0.142, traj 0.00016,
  map_hires 0.00014, tac_v6 −0.00051.
- **The tactical gradient opposes the dominant term:** cos(agent, tac_v6) −0.297. In the shared 0.25 m encoder, box3d's
  share is 0.973 and the map's is −0.0041, with cos(box3d, map) −0.137.

**The tiny rung makes lever claims only.** It never makes a model claim (the argparse help of `--size tiny`: "NOTHING
measured at it is a model claim"). A lever that clears here earns a place in the refcv8 launch argv. It does not earn a
refcv8 number.

## 1. The rig — stated so that a launch on anything else is refused

* **Trainer and size.** `stack/scripts/refc_v3_train.py --size tiny`, tip `c39798d` or later with this SPEC's
  instrument items (§6) landed.
  - ⛔ This is NOT the `train_v6_staged.py` "v7-tiny" (19.3 M, MODEL_REGISTRY:4837). That rig's "~17 min/arm on Thor"
    is INHERITED and is not quoted for this one.
  - This rig has no measured Thor wall time; §4.1 measures it.
* **Corpus.** refcv7's B1 416×1024 cylindrical caches:
  - Thor `/home/nvidia/data/physicalai-b1-w120-416x1024cyl`, train-a6 (4,369 clips) and eval139 (139 clips), exactly
    the sets the v9 release covers. MEASURED on eval139 through the trainer's path: 23,772 / 23,772 windows joined,
    clock residual 0.0 s (`raw/v9_join_eval139.json`). The train join is INHERITED from WP-A (746,946 / 746,946) and is
    re-measured by the arm's own `enable_r8_v9` census at launch (a short join refuses).
  - The 256×640 cache is NOT used: refcv7's perception targets (SAM3 10 cm map, join3d, VIS-1) are banked for the
    416×1024 corpus. ⚠ It is listed as a Thor deletion candidate (INVENTORY_THOR C01); that is irrelevant here.
* **Trunk.** `--trunk timm --trunk-name resnet34.a1_in1k --trunk-in-channels 9`, ImageNet init. This is the
  precedent of the 2026-09-19 A7 rung (`…/2026-09-19-a7-imagenet-knockout/code/a7_run.sh:90-107`). refcv7's resnet101
  is not used at tiny size.
* **Recipe flags.** Every refcv7 recipe flag of `stack/ops/runs.d/refcv7-r101-s0.argv.json` except the size / trunk /
  batch / steps / lr schedule tokens, which this rig sets:
  - `--batch 8 --lr 1e-4 --warmup 200 --log-every 50 --eval-every <steps> --eval-batches 64`;
  - steps N from §4.1, the same N for every arm.

  The token diff of every arm against its base is audited before launch (an `onevar_check` of the actual argv, not
  the intent). A second difference refuses the arm.
* **Labels.**
  - Base (V0): v7.2 (`--v7-labels`).
  - refcv8 arms: `--r8-v9-labels / --r8-v9-labels-eval` (train md5 `f63ece41…`, eval139 md5 `6b5c7f20…`), RC-A50 with
    the certified noise (σ 2.0 / 0.75 m), RC dropout 0.3, nav-args dropout 0.5, `--r8-nav-from-v9`.
  - The 693-box ego mask via `--join-defect-masks`. The 8-clip drop list is NOT applied (PI decision 8 open).
  - **Speed input (MM ruling Q5).** EVERY arm, V0 included, feeds the past-only **N2** with the trained unknown row
    (SPEC_REFCV8 §8.3). The v8 future-max sidecar is refused. Because both sides carry the same speed channel, L1–L3 do
    not move it; only L4 does.
  - **v9 constraint supervision (MM ruling Q2)** is ON in every refcv8 arm once built (§6 P8).
  - **Allocation emission.** The tiny arms train from ImageNet, so there is no warm-start identity to protect, and
    allocation is emitted from step 0. The `--r8-alloc-emit-start` schedule (MM ruling Q1) binds the WARM-STARTED
    launch only (§7).
* **Seeds.** Training seed 0 for every primary arm; seed 1 for the replicates. Sampler seeds 0 and 1 at eval.
* **Compute.** Thor, under `flock /home/nvidia/refcv7_post/thor_gpu.lock`, one job at a time.
  - Jobs are ≤ 40 min; an arm longer than that runs as resumable chunks (the trainer's implicit resume from `<out>/ckpt.pt`).
  - ⚠ A chunked arm does not restore RNG state, so it is not bit-reproducible. The replicate arms quantify exactly
    that variance.
  - Queue position: after R1 → WP-B arms → WP-D P-BOX / P-MAP → WP-RL.

## 2. Arms (run order = priority order)

| arm | = | role |
|---|---|---|
| **V0** | refcv7 recipe at the rig, v7.2 labels | L1 base |
| **V-R8** | V0 + `--refcv8` + v9 + RC + nav args; weights `--w-r8-cons 0.05 --r8-n-alloc 32 --r8-alloc-emit --w-r8-alloc-l1 1.0 --w-r8-listwise 1.0`; `--grad-share-every 250` | L1 treatment; the L2 / L3 base; the L2 sizing substrate (§4.2) |
| **V-R8d** | V-R8 + `--r8-derange-feed` (the planner reads ANOTHER window's tactical output in training) + `--r8-rc-roll` (each row gets another window's route checkpoint) | L1 deliberate regression |
| **V0r**, **V-R8r** | V0 / V-R8 at training seed 1 | L1 training-replicate floors |
| **V-TACk** | V-R8 with `--w-tac-v6` × k (§4.2) | L2 treatment |
| **V-TACk-roll** | V-TACk + `--r8-roll-targets tac` (tactical targets rolled by one row: same gradient magnitude, zero information) | L2 deliberate regression |
| **V-TACkr** | V-TACk at seed 1 | L2 replicate |
| **V-MAP4** | V-R8 with `--w-map-hires` × 4 (WP-D's ESTIMATE; fixed now, not tuned) | L3 treatment |
| **V-MAP4-roll** | V-MAP4 + `--r8-roll-targets map` | L3 deliberate regression |
| **V-MAP4r** | V-MAP4 at seed 1 | L3 replicate |
| **V-VSHUF** | V-R8 with the N2 speed input rolled by one row in TRAINING (`--r8-roll-speed-input`, §6 P7) | L4 deliberate regression: the speed information is removed and the input's magnitude distribution kept |

L4 reads V-R8 against V-VSHUF and needs no extra treatment arm: V-R8 already feeds N2. Its eval rows on V-R8 are
VMAX-OFF (unknown row everywhere) and VMAX-SHUF (N2 from a deranged window), as in SPEC_REFCV8 §8.3.

That is 12 training arms in total. **Eval-only rows on every refcv8 arm's checkpoint (no training):**
- **RC-OFF:** valid 0 on every window.
- **LEGAL:** the NavSim-legal row — known 0, RC invalid, token kept.
- **RC-shuffled:** the checkpoint taken from another window.
- **E-1 floor:** the trivial drive-to-checkpoint plan at constant v0.

## 3. Measures (every arm, sampler seeds 0 and 1; per family, never pooled)

* **Eval set.** eval139 through the launch tree's loader, on the route package's 1,112-window EVAL-DIAG grid
  (`run_route.py --split eval_diag`, generalised per §6 P4), plus the in-run eval rows.
* **Route / LATERAL (deciding for L1 and L2):**
  - turn direction-correct (terminal-heading class, τ 10.35°) on GT-turn windows;
  - heading within 15° at 6 s;
  - cross-track at 6 s;
  - curvature / yaw-rate MAE 0–2 s with the yaw-rate standstill mask (RETR-2026-09-26-YAWMASK).
* **LONGITUDINAL:** along-track at 6 s; speed MAE 0–2 s; 6-s progress relative error. Distance keeping: v9's lead gap
  is a TARGET only; reported as UNAVAILABLE in this harness.
* **TACTICAL (R8-4 i):** lat3 accuracy and macro-F1 on v9 labels beside the majority control; constraint MAE beside
  the TRAIN-median constant (`taniteval.tactical_conditioning`).
* **STRATEGIC:** N/A (PI R5); reported as absent.
* **Controllability / consistency (R8-4 ii / iii):** on refcv8 arms only, the SPEC_WPB §4 definitions.
* **Perception (the L2 / L3 guard and the L3 decider):**
  - `eval_box3d_ap2m`, `eval_agent_ap2m`;
  - the 10 cm map IoU per class × band (`eval_map_hires_iou_<class>_<band>`). Headline classes: drivable and lane in
    the 0_20 and 20_40 bands. Every class is reported.
* **The premise instrument (in-run, every arm with `--grad-share-every`):** `gs_trunk_proj_<term>` and
  `gs_bev025_proj_<term>`, which is P-GRAD's statistic on the run's own batches (§6 P2). The median over logged
  readings in the last 50 % of steps is reported with its IQR.
* **Estimators.**
  - Paired episode-cluster bootstrap over the eval139 episodes (B 2000, fixed seed-0 draws), every treatment against
    its base on the same windows and sampler seed.
  - The TRAINING-variance floor of a metric is `F = max(|V0r − V0|, |V-R8r − V-R8|)` on the same metric; for L2 / L3,
    the treatment's own replicate is included.
  - The headline ratio is `R = E / F`, where E is the treatment − base effect (the A7.6 rule).
  - ⚠ The floor is NOT the 14.3 % separated rate quoted in SPEC_WPB §7. That figure was corrected to 9.5 % (4 / 42,
    `RETR-2026-09-26-YAWMASK`, `…/2026-09-26-yaw-rate-mask/raw/claims/replicate_fp_rate_fixed.json`) and was measured
    on a different tiny rig. Here the floor is measured on this rig, per metric.
* **Three questions, named.** The bootstrap answers "another draw of episodes". The replicate arms answer "another
  training run". The sampler seed 1 answers "another inference".

## 4. Sizing rules (committed now; the numbers they produce are set before the arm they size runs)

### 4.1 Steps
- **S0 timing:** 30 training steps each of V0, V-R8 and V-MAP4, logged, with nothing saved.
- **Rule:** N = the largest multiple of 250 with N × s_step(V-R8) ≤ 3 h, capped at 6,000 and floored at 2,000.
- **Fallback:** if N < 2,000 the rig is too slow; the ladder STOPS and the rig change goes to the MM.
- N is the same for every arm.

### 4.2 The L2 multiplier k
Sizing reads V-R8's own in-run readings over its first 20 % of logged steps (an instrument output, not a scored outcome).
Take the medians p = `gs_trunk_proj_tac_v6`, n = `gs_trunk_norm_tac_v6` and G = `gs_trunk_norm_total`. Then:

- c = p·G² − n² and R² = G² − 2p·G² + n²;
- k = the smallest root above 1 of (0.99·n²)·k² + (0.98·c)·k − 0.01·R² = 0, rounded UP to 2 significant figures and
  capped at 100.

This is a frozen-direction linear extrapolation, so k is ESTIMATED. Whether the share actually reaches 1 % is the
in-run premise check (§5 B-PREM).
- **Worked example on P-GRAD's refcv7-50,400 numbers** (p −0.000506, n 3.282, G 872.4): **k ≈ 51**.
  - Because the tactical gradient opposes the agent term, the share is NEGATIVE for every k < ~37: at k = 10 it is
    ESTIMATED −0.38 %.
  - So no gentle dose exists for this lever, and a ×2 to ×10 ladder would test the wrong side of the curve. That is why
    the rule sizes to the target instead of fixing a dose.
- The same algebra for traj gives k ≈ 33, so traj is reported in-run but not armed. TRAJ_WEIGHT is a constant, not a
  flag.

### 4.3 The L3 multiplier
×4, fixed (WP-D's ESTIMATE). The same frozen algebra on P-GRAD's shared-encoder numbers (p −0.0041, n 4.638, G 69.73)
ESTIMATES the map's share of its own encoder's update at **+3.6 %** at ×4.

## 5. Bars (committed now, both directions)

* **I-0 (instrument identities; must PASS before any arm is scored).**
  - Arm-level identity: V-R8's step-0 loss terms on the first batch equal V0's on every refcv7 key.
  - Model-level identity: the warm-start identity test (refcv8 seams zero-init).
  - grad_share's linearity control ≤ 1e-4 on every reading.
  - The conflict detector's `cd_*` rows exist on every arm and carry the agent term (§6 P1).
  - The v9 join census = 100 % on train and eval.
  - A failure STOPS the ladder.
* **B-REG (the regression arms must fail; otherwise the instrument is broken and nothing from the rung is quotable).**
  - V-R8d's turn direction-correct gain over V0 is ≤ F, and V-R8 − V-R8d is separated with R > 1.
  - V-TACk-roll's route gain over V-R8 is ≤ F.
  - V-MAP4-roll's map gain over V-R8 is ≤ F on both headline classes.
* **B-PREM (a premise, not a result).**
  - V-TACk's median `gs_trunk_proj_tac_v6` over the last 50 % of steps is ≥ 0.01.
  - V-MAP4's median `gs_bev025_proj_map_hires` is > 0.
  - A miss makes the arm PREMISE-UNMET, which is reported as such and is never read as a lever negative.
* **B-RECIPE (L1).** On BOTH sampler seeds:
  - turn direction-correct and heading-15 gains of V-R8 over V0, each separated (CI lower > 0) with R > 1;
  - GT-straight ΔADE ≤ +0.05 m with CI upper ≤ +0.10 m;
  - no family worse beyond F;
  - the LEGAL row of V-R8 is within +0.05 m ADE of V0, which is the NavSim-legal robustness check.
* **B-BUDGET (L2).** All of the following:
  - B-PREM holds;
  - V-TACk's route gains over V-R8 are separated with R > 1 on both seeds;
  - `eval_box3d_ap2m`, `eval_agent_ap2m` and the headline map IoUs each drop by ≤ F, and by ≤ 0.02 absolute, against
    V-R8.
* **B-X3 (L4).** On BOTH sampler seeds:
  - speed MAE 2–6 s of V-R8 is better than V-VSHUF's, separated, with R > 1;
  - SPEC_REFCV8 §8.3 bars (3) and (4) hold as eval rows on V-R8: VMAX-SHUF − VMAX-OFF is not separated better, and
    UNKNOWN − known ≤ +0.10 m/s (CI upper ≤ +0.20).
  - The LEAK bar (≤ 0.05) is a property of the fed channel, read by the D4 census instrument (§6 P7), not by an arm.
* **B-MAPW (L3).** All of the following:
  - B-PREM holds;
  - the headline map IoUs gain over V-R8, separated, with R > 1 on both classes;
  - box / agent AP and the route criteria are no worse than V-R8 by more than F.

## 6. Instrument prerequisites (WP-B owns all; status at this draft)

| id | item | status |
|---|---|---|
| P1 | The conflict detector sees the agent head. `("agent", _w_agent)` is on the AUX side (the plan side stays the pre-registered traj term), and `losses["agent"]` is emitted attached. | **BUILT + TESTED** (`tests/test_refcv8_grad_share.py`: row, producer, aux value AND gradient, deliberate regression RED). ⚠ `cd_*` rows on agent arms are not comparable with pre-2026-10-04 runs. |
| P2 | The in-run gradient-share instrument `--grad-share-every N` (`tanitad/train/grad_share.py`). It uses autograd.grad on logged steps only, leaves `.grad` and the RNG untouched, and logs projection share, norms and a linearity control per group (trunk / bev025 / bev_pool). | **BUILT + TESTED.** The closed-form literals hold; the norm-share wrong statistic goes RED; on the real agent rig the shares sum to 1 and linearity ≤ 1e-4. |
| P3 | Regression-arm flags `--r8-derange-feed`, `--r8-rc-roll`, `--r8-roll-targets {tac,map}`. They are training-only, consume no RNG (roll-by-one, the `--bev-aux-shuffle` precedent), and are refused without their prerequisite (batch ≥ 2, an RC to roll, a live tactical / map loss). | **BUILT + TESTED.** `refcv8_train.roll_targets` rolls exactly its family, never an input; the RC roll equals `roll(plain)` under identical draws and is off at eval; the derange is set per call and only in training; each pin refusal has a GREEN control. |
| P4 | The eval path for refcv8 checkpoints. `refcv7_loader.build_eval_dataset` attaches the v9 EVAL join when the run carries `--r8-v9-labels-eval`; `run_route.py` takes the run dir and step instead of refcv7's constants. | Loader **BUILT**, exercised by the full-size check (§7). The `run_route.py` generalisation is a harness item for the ladder chain. |
| P5 | `--init-from <ckpt>`: warm start, strict on every key of the source, new keys must be exactly the refcv8 seams, fresh optimiser, ignored when the run resumes its own `ckpt.pt`. This serves the full-scale smoke (§7), not the tiny arms, which train from ImageNet. | **BUILT + TESTED** (`tests/test_refcv8_init_from.py`): the rig forward is bit-identical after the load; a moved gate goes RED; an extra key, a non-seam missing key, a shape mismatch or a missing file is each refused; it runs before G-DVB and never over a resume. |
| P7 | **X3** (MM ruling Q5): the past-only N2 speed input with a trained unknown row (input dropout), the ceiling on the EMITTED plan, N3 behind a flag, and the gate refusing the v8 future-max sidecar. Also `--r8-roll-speed-input` (training-only roll, the V-VSHUF regression arm) and a leak check reproducing WP-A's 3.46 % / 23.8 %. | building (MM item 1) |
| P8 | **v9 constraint supervision** (MM ruling Q2): `lat_c`, `lon_c` and `speed_goal` supervised, masked where the v9 row is PARTIAL or absent, plus the [2, 6] s label ↔ tag agreement. | building (MM item 2) |
| P9 | **`--r8-alloc-emit-start S_emit`** (MM ruling Q1). Identity is checked with emission OFF on the I-2 literals; S_emit is chosen from `iw_diag.py`. | building (MM item 3) |
| P6 | The `refcv8` launch-gate profile. It inherits every refcv7 rule, adds the refcv8 required flags and values, forbids the regression flags, and marks the RC as PI-DECISION. It also lists four OPEN ITEMS: R8-RECIPE, R8-BUDGET, R8-WARMUP and R8-INHERITED-RECORDS. The canonical smoke argv is `stack/ops/runs.d/refcv8-wpb-smoke.argv.json`. | **BUILT + TESTED** (`tests/test_refcv8_launch_profile.py`). The smoke argv passes the trainer's FULL pin chain and the effective-weight audit, with all five `w_r8_*` weights TRAINS (MEASURED, dev box CPU). |

## 7. The full-trainer smoke (item 5 of the MM's request)

This is not an arm and is not scored. It is the refcv8 launch argv at FULL size:
- refcv7-r101-s0's tokens + `--refcv8` with every seam on, v9, RC, nav args and the stated weights;
- `--init-from` refcv7-50,400;
- `--grad-share-every 50`.

It runs on Thor under the lock for 30 steps plus one in-run eval, through the `refcv8` launch-gate profile's G-SMOKE.

**Reads:**
- the warm-start identity at step 0 (planner outputs = refcv7's);
- every new parameter takes gradient (G-LIVE rows for the five `w_r8_*` weights);
- s/step against the 10.5 s ceiling (B-COST);
- the in-run grad_share at full scale, which is the first refcv8 number for L2's premise.

### 7.1 Status at hand-over (2026-10-04)
* **SPEC_WPB's registered I-W FAILED on Thor on the extended fan, and the defect is fixed** (`RESULT_IW.md`).
  - Seams only: bit-identical on 218 / 218 windows.
  - Allocation + prior-free, emission off: sel_idx identical on 218 / 218, but the plan differed on 205 windows (by
    ≤ 7.6e-5 m on 48 diagnosed windows) and the scores by up to 2.1e-5.
  - Cause (MEASURED by `iw_diag.py`): the GEMM shape over 117 + extras rows. The determinism control reads 0.0, and
    the effect is 5–15× smaller than the GPU's own TF32 switch.
  - Fix: the base and extra candidates are decoded in separate passes. After the fix, full size on CPU is
    bit-identical, scores included, and a Thor CPU I-W on 4 windows PASSES.
  - The registered I-W re-runs on the Thor GPU with the SAME criterion (`wpb_chain2.sh`).
* **The full-size smoke (§7) is queued on Thor** (`code/r8_smoke_thor.sh`, PID-gated on the re-run I-W, one ≤ 40 min
  job). Its argv is the CURRENT smoke argv; the launch smoke re-runs once P7–P9 land.

## 8. Reading rule (fixed now)

* I-0 fails ⇒ stop.
* A B-REG failure in a rung ⇒ that rung is not quotable.
* L1 clears ⇒ the recipe enters the refcv8 launch argv.
* L1 fails while SPEC_WPB's frozen-trunk arms clear B-ROUTE ⇒ "the condition transmits on frozen features but trunk
  training dilutes it". L2 then decides whether the budget is the lever.
* L2 is PREMISE-UNMET ⇒ the budget cannot be bought by weight alone at this rig. The next lever is architectural, and it
  is named in the RESULT rather than tested here: a stop-gradient on the agent term into the trunk's last stages
  (agent·tac cos −0.30), or per-term gradient normalisation.
* L2 clears B-PREM but not B-BUDGET ⇒ "share without effect": the budget is not the lever.
* L3 clears ⇒ ×4 enters the launch argv. L3 PREMISE-UNMET ⇒ a map-only encoder (WP-D's alternative) is the next lever.

## 9. Stamps every number from this ladder carries

- TINY RUNG (lever claims only; resnet34; N steps; one training seed per primary arm, seed 1 replicates).
- OPEN-LOOP on logged eval139 frames.
- Nav, RC and dense labels are ego-future derived, so they are optimistic on PhysicalAI.
- The RC is RC-A50 noised, under a PROVISIONAL MM ruling (PI confirmation pending). Every RC result is quoted beside its
  RC-OFF, LEGAL and shuffled rows.
