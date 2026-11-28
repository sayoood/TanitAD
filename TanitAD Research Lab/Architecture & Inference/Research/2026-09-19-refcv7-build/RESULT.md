<title>refcv7 build and implementation review</title>

# refcv7 — built, wired, tested, and reviewed (PI request 2026-09-19)

`Architecture & Inference · 2026-09-19/20 · "Build the DriveR-T refc variant, combine all our
advantages with their proven tricks, keep the hierarchy etc., prepare it to be trained with our
last data corpus, review the implementation and check its quality."`
`Spec: Project Steering/SPEC_REFCV7.md · Pre-registration: Project Steering/PREREG_REFCV7.md`
`Evidence for every ported trick: …/Research/2026-09-19-drivor-deep-analysis/RESULT.md`

---

## 0 · What was built

refcv7 = **refcv6 exactly as PI-ruled** (416 × 1024, pretrained timm trunk, K = 3 frame history,
ego history, nav mandatory, 4-value max-speed input, SAM3 map head, 3-D box head, DETR tactical
behaviour decoder, DD-faithful diffusion over the 117 anchors, strategic layer off) **plus three
DrivoR/TOAD components**, each behind its own flag and absent from the graph when off:

| # | component | file | params (MEASURED) |
|---|---|---|---|
| D1 | WTA proposal decoder — 64 learned queries, ONE token per trajectory, winner-takes-all L1 | `stack/tanitad/refs/refcv7_heads.py` | **4,493,840** |
| D2 | Disentangled scorer — own decoder, candidates re-embedded behind a stop-gradient, 7 oracle sub-score heads, condition-aware | same | **3,429,127** |
| D3 | PhysicalAI oracle — NC, TTC (3-D agent join at every waypoint slot), DAC (SAM3 drivable), EP, COMF (NAVSIM limits), SPD (set speed), NAV (timing-aware) | `refcv7_oracle.py` | 0 (label-side) |
| D4 | Off-proposal coverage — smooth control-space-like perturbations shown to the scorer | `refcv7_oracle.smooth_perturbations` | 0 |
| D5 | Selection by the aggregated scorer (PDMS-shaped gates × quality, behaviour-profile weights) + reachability + set-speed mask | `refc_v3.py::_refcv7_forward` | 0 |
| D6 | TOAD test-time CEM over the slot controls, through the vocabulary's OWN integrator | `refcv7_toad.py` | 0 |

Total added: **7.92 M** parameters, both on their own ledger lines (so a checkpoint stays
rollable through `refcv3_arm.cross_check_config`).

## 1 · The five defects this build found and fixed — the quality review

⭐ Each was MEASURED while wiring, not reasoned about.

| # | defect | how it showed | fix |
|---|---|---|---|
| **1** | ⛔ **A side-only nav label would teach the planner to turn EARLY.** The nav token is per CLIP: on eval-139 split A the median \|GT terminal heading\| at 6 s on LEFT/RIGHT windows is **0.048 rad**, Youden J **0.22** (`raw/nav_tau_eval139_splitA_PIPELINE_ONLY.json`) — the commanded turn is usually outside the plan | the derivation script's own separation numbers | the nav sub-score became **timing-aware**: the target is *"turns on the commanded side exactly when the human did within the horizon"*; FOLLOW/STRAIGHT abstain |
| **2** | ⛔ **A truncated GT was being fed to the scorer as a POSITIVE example.** `waypoint_targets` clamps past the clip end, so a window whose 6 s runs off the clip has a GT that stalls — it scores comfort 0 and its length corrupts EP | comfort rate read **0.0 over every candidate** in the first real-data smoke, while the human's own complete path fails comfort **0/161** times | rows without a complete GT: the GT candidate is masked out of every sub-score, and the GT-derived sub-scores (EP, NAV) abstain; `r7_gt_full_frac` is logged |
| **3** | ⛔ **TOAD's no-worse guarantee was void for free-form proposals.** It compares against the ROLLED base, which equals the pick only if the control inversion is exact; on an untrained model the fit residual is **1.97 m** (a free-form proposal need not be rollable at all) | the wiring test asserted a small residual and went red | the last word is now a reward comparison against the **actual** pick; the residual is emitted every step (`r7_toad_fit_rms_m`) |
| **4** | ⚠️ **A scorer with candidate self-attention is not a stationary reward.** DrivoR's scorer is a vanilla decoder, so a candidate's score depends on which others were scored with it | by construction; proven both ways in `test_refcv7_heads.py` | self-attention **OFF** by default, `--r7-scorer-self-attn` keeps DrivoR's form as an arm |
| **5** | ⚠️ **Padded agent slots would have been attended to** | — | `SceneMemory` carries key-padding masks; a fully-padded row is unmasked rather than producing NaN, and the test's control proves the mask is load-bearing |

## 2 · Tests — 60 refcv7 tests (COUNTED by `pytest --collect-only`), every guard with a control that must go red

| file | n | what it pins |
|---|---:|---|
| `test_refcv7_oracle.py` | 15 | analytic LITERALS: constant-speed kinematics; a car on the path is a collision; 10 m to the side is not; an agent just ahead fails TTC but not NC; a lane corridor; **abstention** on unseen ground; EP = the length ratio; a hard brake fails comfort; set-speed; the PDMS aggregate; the timing-aware nav label (4 cases) |
| `test_refcv7_heads.py` | 9 | WTA supervises ONLY the winner (gradient 0 on the others); the GT mask; the scorer REFUSES a non-detached candidate; candidate independence **and its self-attention control**; condition dependence; masked BCE; padding |
| `test_refcv7_toad.py` | 4 | the inversion reproduces the programme's integrator; never worse than base; moves toward the reward; same seed → same plan, different seed → different |
| `test_refcv7_model.py` | 11 | OFF build is bit-identical (no key, no ledger line, **no moved parameter**); the refusal without the scene hook; candidate/pick shapes; scorer ⇏ proposals but scorer → trunk; ledger sums; TOAD at eval only, deterministic, never worse |
| `test_refcv7_training.py` | 21 | 9 pin refusals **each for its own reason** + green controls; the two weight-gate rows discriminate; the conflict detector sees both terms; the future-agent transform at three ego headings; unlabelled and past-the-end slots |
| regression | — | `test_refcv6_tactical.py` 93 pass; `test_refc_v3*.py`, `test_refcv6_geometry_agnostic.py` (refcv7 modules added to its OWNED list), `test_refcv6_bev_tactical_wiring.py` pass |

⚠️ `test_refc_v3_u8_batches.py::test_one_step_train_on_cuda…` was **not run**: it needs the GPU,
which is running the live a8 training. Not a refcv7 result either way.

## 3 · End-to-end on REAL data (dev box, CPU, eval-139 at 416 × 1024)

`stack/scripts/refc_v3_train.py` with the live refcv6 flag set plus refcv7 — trunk `resnet34`
ImageNet, agents head, SAM3 map head, 3-D box head, tactical decoder, `--refcv7 --w-r7-wta 1
--w-r7-scorer 1 --r7-toad`:

* exit **0**; both refcv7 terms read **`TRAINS` / `graph=yes`** in the trainer's own
  effective-weights table (the instrument that exists because a built head with no gradient once
  trained for 40,284 steps);
* `config.json` carries the two ledger lines and the stamped `w_r7_*` knobs;
* every `r7_*` telemetry key is in `metrics.jsonl`: the 7 BCEs, the 7 oracle **rates AND
  coverages**, the selection triple (**pick / random / best** on the same windows — an oracle gap
  without its random control is not a claim), the WTA L1, and the TOAD block.

### 3.1 The oracle's own coverage, MEASURED (6 steps x batch 4 = 24 windows, `raw/smoke2_metrics.jsonl`)

| sub-score | coverage | rate | reading |
|---|---:|---:|---|
| NC / TTC | **0.708** | 0.774 / 0.459 | the 3-D join labels EVERY frame (spacing 1, MEASURED), so the future slots resolve; the shortfall is windows whose 6 s runs off the clip |
| DAC | 0.680 | 0.087 | 135/139 eval clips carry a SAM3 map; a low rate is expected from an UNTRAINED fan |
| EP | **0.750** | 0.748 | ⭐ exactly `r7_gt_full_frac` = 0.75 — the truncated-GT fix (defect 2) doing its job |
| COMF | 0.999 | **0.0037** | non-zero now; before the fix it was EXACTLY 0 on every candidate |
| SPD | **0.000** | — | correct: this smoke carries no max-speed channel, so the oracle ABSTAINS rather than passing |
| NAV | 0.208 | 0.613 | LEFT/RIGHT windows with a complete GT only (40 of 70 clips are FOLLOW) |
| selection | — | pick **0.2257** · random **0.2165** · best **0.8333** | an untrained scorer picks at chance, which is the honest reading at step 6 — and the random control is quoted beside it |

⚠️ **A risk this surfaces, named rather than absorbed:** at initialisation the comfort target is
**~0.4 % positive**, and DAC **~9 %**. Those two heads start on a heavily imbalanced BCE, so a
refcv7 arm must report their per-head base rates (it does, every step) and treat an early
"comfort head learned nothing" as a class-balance question first. Candidate remedy if it bites:
a `pos_weight` fitted on the TRAIN split, exactly as `--w-tac-goal` already does for its 22
tokens — ⛔ never a literal.

⛔ **This is a WIRING fact and nothing else.** No refcv7 arm has trained; no quality claim is
made or implied. The first real numbers are the ones `PREREG_REFCV7.md` commits to.

## 4 · What is NOT done — named, not buried

1. ⛔ **The 416 × 1024 TRAIN cache does not exist** (10.6 h, 386.5 GB, HF quota check first), and
   **SAM3 production is still running** (ETA ~22 Sep). refcv7 cannot train at corpus scale today.
2. ⛔ **The nav tolerance must be re-derived on the TRAIN split.** The 0.0487 used in the smoke is
   from eval split A and is a PIPELINE number, never a training input.
3. ⚠️ **No GPU measurement**: per-step cost, TOAD's eval-time ms/window, and whether the
   `resnet101` primary fits are all unmeasured (the dev-box card is busy; `resnet101` OOMs at
   8 GB anyway — a pod-sizing fact from refcv6).
4. ⚠️ **The oracle is partial by construction**: no lane/direction terms exist on PhysicalAI, so
   DAC is drivable-area only. Coverage is logged per sub-score every step rather than assumed.
5. ⚠️ **One assumption is stated but not re-derived**: that `ep.poses` are the rig poses the
   agent join's boxes are expressed in (the future-agent transform rests on it; it has an
   analytic test at three headings, and `waypoint_targets` makes the same assumption).
6. ⛔ **Registers are NOT in refcv7** (DrivoR's compressor lives inside a ViT; our trunk is the
   PI-ruled ResNet) — that is backlog row **DR-4**.

## 5 · Deliverables

| path | what |
|---|---|
| `stack/tanitad/refs/refcv7_heads.py` | D1 + D2 |
| `stack/tanitad/refs/refcv7_oracle.py` | D3 + D4 |
| `stack/tanitad/refs/refcv7_toad.py` | D6 |
| `stack/tanitad/refs/refc_v3.py` | cfg fields, build, scene-hook capture, `_refcv7_forward`, TOAD, ledger |
| `stack/scripts/refc_v3_train.py` | dataset future-agent block, CLI group, `_pin_refcv7`, weight gates, conflict terms, loss block |
| `stack/scripts/refcv7_derive_nav_tau.py` | the tolerance derivation (TRAIN split) |
| `stack/tests/test_refcv7_{oracle,heads,toad,model,training}.py` | 60 refcv7 tests |
| `stack/tests/test_refcv6_geometry_agnostic.py` | refcv7 modules added to OWNED |
| `Project Steering/SPEC_REFCV7.md`, `PREREG_REFCV7.md` | the spec and the pre-registration |
| `raw/nav_tau_eval139_splitA_PIPELINE_ONLY.json` | the tolerance derivation's own output |
| `raw/smoke2_metrics.jsonl`, `raw/smoke2_config.json` | the real-data smoke's own metrics and run record |
