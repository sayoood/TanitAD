# PREREG_REFCV9_INMODEL_PA -- does the in-model R9-PF head move refcv8's MD/FD frontier (Phase A, refcv8 trunk, 0 Thor-h)? (SEALED by the Master Mind 2026-10-26 13:19:05 Berlin)

*Architecture FlyWheel, 2026-10-26. **DRAFT -- NOT SEALED.** The Master Mind seals it (sha256) before any R9-PF fit or eval-window
output exists. Design: `REFCV9_INMODEL_DESIGN.md`. Code: `stack/tanitad/refs/refcv9_progress.py` + the flag-gated edits listed in the
hand-back. No number below is a result of this registration; every quoted number is a prior MEASUREMENT with its source.*

## 0. Question and why
Four post-hoc levers on refcv8 (H2cn, H2cn-L, SELFSHARP, SELCOR) slide along ONE missed-decel (MD) / false-decel (FD) frontier; the
GT-oracle pick from the same fan reads MD 0.075 / FD 0.057 vs BASE 0.47 / 0.14 (PLAN_REFCV9 §12, MEASURED). R9-PF adds a clean-split
DECISION (decel / cruise / accel) whose profile alone sets the speed. **Prediction (HYPOTHESIS):** it dominates both measured ends --
H2CN_DEPLOY8's MD and FACTORED_DEPLOY8's FD -- because the decision is fitted on clips the trunk never saw (L1 O2 lesson).

## 1. Phases and arms
**Phase A -- refcv8's frozen trunk (EMA-48,852), 0 Thor-h.** Stage 2 fits the R9 head on the FRESH 300-clip split (K-DISJ, the
H2cn split; 297 clips / 9,135 S2 rows MEASURED in `H2CN_FREEZE_INFO.json`). The head is deterministic given its inputs, so the fit
and the eval plans may be computed OFFLINE from banked arrays (fresh `ztac_fresh.npz`; eval `traj_own` + z_tac banks) -- the
in-forward path must reproduce them (K-EQ).

| arm | what | role |
|---|---|---|
| **R9PF** | `--r9-decision regime`, argmax | the lever |
| R9PF-TL | `--r9-decision trunk_lon` (same profiles; decision = trunk lon head argmax grouped to regimes = in-model SELFSHARP) | attributes the decision SOURCE |
| R9PF-CONSTZ | z_tac replaced by its calibration-split mean, at fit AND eval | deliberate regression: vision removed |
| R9PF-SHUF | GT regime labels permuted across calibration EPISODES before the fit | information control |
| beside (banked, not re-run) | RETIME_DEPLOY8, H2CN_DEPLOY8, FACTORED_DEPLOY8, OSONLY, GT-oracle pick | references |

Stage-2 recipe (all arms): H2cn's selection rule -- weight decay and epochs by 5-fold EPISODE CV on the calibration rows, grid as
`l1_fit_score.MLP_WD` / `MLP_MAX_EP`; fit seeds **0, 1, 2**. `--r9-decel-tau` stays at argmax for every registered bar; a tau tuned on
the calibration split is EXPLORATORY and reported beside.

**Phase B -- the refcv9 trunk (PI compute).** Whatever trunk levers the PI authorises, trained as refcv8 (R9 flags OFF in stage 1,
bit-identical seams), then stage 2 on the same fresh split from the FINAL EMA checkpoint. Arms: refcv9-s0 + R9PF, refcv9-s1 + R9PF
(the replicate), refcv8-final + the PI's shipped stage (the bar).

## 2. Population, estimator, tier (as SHIP-R8 / L1C)
eval139, compliant windows: S2 4,159 w / 138 ep; S6 3,251 w (GT-decel 639, GT-cruise 1,856, 136 ep -- PREREG_SELFSHARP), at
inference seeds 0 and 1, per-frame nav feed. Paired episode-cluster bootstrap, B 2000, seed 0 (`taniteval.ci`). Tier **T1** self-action
open loop on logged eval139 frames. **Questions answered, named:** another draw of EPISODES (bootstrap); another TRAINING run (Phase A:
fit seeds 0/1/2; Phase B: trunk seed s1); another INFERENCE run (seeds 0/1; the inference-seed floor is printed beside every effect).
Every eval reports the FOUR families (ADE, LONGITUDINAL, LATERAL, TACTICAL; STRATEGIC if on) per family, never pooled.

## 3. Controls (a failed control VOIDS the bars)
* **K-OFF** -- flags-off: every forward output and state tensor bit-equal to the pristine tip (MEASURED for the CPU rig: 3 arms, 234-280
  outputs, 391-408 tensors, `bitid_REPORT_pristine_vs_modified_flags_off.json`); on the eval roll, `traj_preretime` == the banked
  os-only plan on every window.
* **K-EQ** -- the in-forward R9 plan equals the offline composition (same head file, same inputs) on a >= 500-window subset, max |d|
  <= 1e-5 m; **K-GEO** -- torch re-timing == `retime_stage` on every eval window (1e-6 m); **K-LAT** -- every R9 point on the selected
  polyline / end ray (<= 1e-5 m).
* **K-DISJ** -- fresh-split disjointness vs train4572 and eval139, with its mutation arm RED (`ship_r8_sub_common.fresh_disjointness`).
* **K-CONSTZ** -- R9PF-CONSTZ must NOT pass PA-1; **K-SHUF** -- R9PF-SHUF must NOT pass PA-1a. Either passing means the bar is
  passable without vision / without the decision, and the run is VOID.

## 4. Bars
**Phase A** (S6 compliant unless stated; both inference seeds; paired on the SAME draw):
* **PA-1a** vs H2CN_DEPLOY8: MD separated BELOW and FD NOT separated above.
* **PA-1b** vs FACTORED_DEPLOY8: FD separated BELOW and MD NOT separated above.
* **PA-2** vs RETIME_DEPLOY8: S2 E1 and S6 E3 separated below (H2cn's own pass condition).
* **PA-G** no S2 or S6 four-family row separated WORSE than H2CN_DEPLOY8; the ceiling-exceedance rate not separated above it.
* **PA-R** PA-1a, PA-1b and PA-2 hold at fit seeds 0, 1 AND 2.
* **PASS = PA-1a + PA-1b + PA-2 + PA-G + PA-R.**

**Phase B** (PLAN_REFCV9 §0, verbatim intent):
* **B9-0** no family separated worse than refcv8-final + the shipped stage, at both inference seeds, on the same windows.
* **B9-1** refcv9 + R9PF with NO post-hoc stage passes E1 (S2) and E3 (S6) vs `ha0_ext_pose` at its final EMA checkpoint, AND beats
  the ego-scalar floor PROG_SCALAR_OS (E1 -0.0328, annex §1.5) separated -- the condition for "vision carries it".
* **B9-2** every lever claim holds at trunk seeds s0 AND s1 and fit seeds 0/1/2, and exceeds the s0-vs-s1 replicate difference
  (`H-ESTIM-SEED-1`); the inference-seed floor printed beside.

## 5. Committed outcomes
* **PASS (A)** -> R9PF is refcv9's speed decode; the Phase-B compute request goes to the PI with these numbers. R9PF on refcv8's trunk is
  also a refcv8 deliverable candidate, which needs its OWN SHIP-R8 amendment (never assumed from this read).
* **PARTIAL (PA-1a xor PA-1b)** -> a new operating point, not a frontier move. Next lever, named: a larger clean split (V-C, more fresh
  clips; availability UNVERIFIED) and the decision's AUC vs the lon head's on the same windows read beside.
* **INERT (R9PF within the fit-seed floor of R9PF-TL, both on the frontier)** -> the decision SOURCE adds nothing at 297 clips; the lever
  moves to training the lon decision in the trunk with richer supervision (PLAN §12), a Phase-B design item.
* **FAIL-GUARD (PA-G)** -> name the rows; next a constrained stage-2 objective, pre-registered first.
* **Phase B FAIL on B9-1 only** -> the in-model head is not enough without a trunk change; report which of E1 / E3 fails and by how much.

## 6. Compute (ESTIMATED -- the Phase-B part is a PI decision)
* **Phase A: 0 Thor-h.** CPU stage-2 fits from the banked fresh z_tac (H2cn's two fits took 392 s + 241 s, MEASURED) x 4 arms x 3 fit
  seeds ~ 2-4 CPU-h; offline eval composition on CPU (minutes); the K-EQ subset roll on the dev-box GPU ~0.5 h (ESTIMATED; only if
  `nvidia-smi` shows room). ⚠️ Prerequisite UNVERIFIED: the eval-side z_tac bank at EMA-48,852 for both inference seeds (z_tac is
  seed-independent per L1 K6, INHERITED).
* **Phase B: ~240 Thor-h** = two refcv9 trunk seeds at ~120 Thor-h each (ESTIMATED from refcv8-r101-s0: launched 2026-10-15 04:54,
  final EMA-48,852 read by 2026-10-20 -- dates, not a run record), + per seed a stage-2 roll of the fresh split (~1.5 dev-box GPU-h at the
  annex's 0.57 s/window) + the eval battery. **PI decision.**

## 7. Master Mind sealing notes (binding; added at seal)
* **Only PHASE A is registered here.** Phase B (two refcv9 trunk seeds, ~240 Thor-h ESTIMATED) is a PI decision (D-R9-INMODEL Q3) and needs
  its own seal; nothing in Phase B may be launched on this registration.
* **Labels.** The stage-2 regime target (decel / cruise / accel) is the KINEMATIC class of the human future, the same definition as the
  MD / FD cells (`l2_common.plan_class` on the GT future, `lon_indicators`). It does NOT read the fresh v9 `lon_v7` labels, so the fresh
  release's missing lead pass (H-R9-FRESHLABEL-1) does not touch this arm -- asserted by a control: **K-LABSRC-R9** = the stage-2 target is
  recomputed from the GT future independently of the fit code and matches on every calibration row; a mutation (labels taken from
  `lon_v7`) must go RED.
* **Registration order.** The executor freezes every stage-2 head (4 arms x fit seeds 0/1/2) and the eval-side z_tac bank's sha256 and STOPS;
  the Master Mind registers them (REGISTERED_R9PF_PA.sha256) before ANY eval139 window output exists (K-EQ's subset roll included).
* **Prerequisite.** If the eval-side z_tac bank at EMA-48,852 cannot be produced or verified, the run stops BLOCKED with the reason; no
  substitute trunk or checkpoint.
* **Code.** The R9-PF code is the banked patch `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv9-inmodel/REFCV9_INMODEL.patch`
  (4b5f7fb), applied to a COPY of the tip tree off-repo; `stack/` on the branch is not modified by this run.
* **The SELFSHARP 'self-label' note.** R9PF-TL uses the trunk's own lon argmax grouped to regimes; that is an inference-available decision,
  not a GT label.
