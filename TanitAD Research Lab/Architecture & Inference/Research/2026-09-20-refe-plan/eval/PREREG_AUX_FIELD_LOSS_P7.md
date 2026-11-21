# Proposal-level field loss (L_field) on host H-P7: pre-registration (FINAL, 2026-11-17)

> ⭐ **FINAL. IN FORCE from the commit that lands it**, which also lands the gate profiles `refe_field_p7` / `refe_field_p7_ctrl` (Amendment A, written before any F or C data exists). It supersedes `PREREG_AUX_FIELD_LOSS_DRAFT.md` (sha256 `2749db168a0b046d…`, left unedited) **for host H-P7 only**; everything the old draft fixes is carried below unchanged unless a line says "H-P7". After landing nothing changes without a dated amendment written before the read.
> In force only when all hold: (1) PI D-AUX-4 **MET 2026-10-12** ("yes"); (2) label store S0 gate **MET** (below); (3) the wiring exists (**MET in the trainer**, `REACHABILITY_FIELDP7.md`), the launch-gate entries exist (**MET in the landing commit**: G1, G2, G5, G6 of `LAUNCH_GATE_GAPS_FIELDP7.md`; G3 MET, the full store is on the pod; G4 the replicate profiles: the replicates are APPROVED (Amendment A) and their profiles land before they launch; G7, the real-bank reach re-run, is the pod gate's own G-LIVE) and a PASS token binds the arm's commit, argv sha256 and data-manifest sha256s (**per arm, on the pod, before its first step**); (4) **MET**: `w_field` 2.0 MM-ratified pre-data and PI-approved ("Approve F+C", 2026-11-17), the comparator ruled (Amendment A), this file landed before the first training step.
> Evidence classes: MEASURED (ours + artifact) · PUBLISHED · INHERITED (not re-verified) · ESTIMATED · HYPOTHESIS. `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`; `HOLD` = `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/field_p7`.

## Why H-P7, and what changed since the old draft
* The old draft names host P2 (never finished) and P6. The current best checkpoint is **P7** (`p7_reserve_head`): `model_final.pt` sha256 `2613fc17b9e5d63342c9f42120c5612a6ec7061fda72f83dac46ddd2e70fd3fe`, 1,298,292,047 B, 9,075 steps, full-navtest PDMS x100 **91.0724** (`Project Steering/MODEL_REGISTRY.md` §14.4, MEASURED), vs P6 91.1965, paired −0.12 [−0.51, +0.28] (NOT PROVEN). Copy on this box: `F:/TanitAD_p7read/run/model_final.pt`.
* **The head-veto read is VH-NULL** (`PK/raw/2026-11-16-p7-read/veto/P7_HEADVETO_READ.md`, MEASURED, 1,058 C3 tokens, stratified): veto −0.117 [−0.602, +0.253], score term −0.103 [−0.557, +0.383]. **The headroom is real:** an oracle DAC veto gains **+1.640 [+0.902, +2.531]**. The head's quality, not the headroom, is the limit; the pre-registered next lever is this arm (D-AUX-4).
* **Superseded in the old draft:** its three hosts (H-P2, H-P6, H-0) and its ~20 A40-h-per-arm sizing. H-P7 replaces them; the trainer and the gate machinery are P5-C's (same base, same bank, same graft recipe).

## The question (one)
Does a hinge on the **footprint corners of all 64 proposals** against the stored ground-truth SDF, added to P7's trajectory loss in a short graft, cut the drivable-area failures of the *picked* plan on navtest, at no cost in progress or fan diversity? Separate arm from the head; shares the labels only.

## The treatment (fixed BEFORE any read)
| item | value | status |
|---|---|---|
| loss | `refe/field_loss.field_loss`: mean over labelled samples, 64 proposals, 40 steps (0.1 … 4.0 s), 4 corners of `relu(m − SDF_gt(corner))` | module sha256 `e29d91c349539ba366c73739fd2fe7fc6f1e7a6a015c4c0749df5f15df36c994` (tip `6b789b7e`; last change `7da149be`; `test_field_loss.py` sha256 prefix `a8d97adb…`); `ZZTEST_PASS 42` re-run 2026-11-17 (`HOLD/raw/test_field_loss.out`, MEASURED). **Re-hash at launch.** |
| margin m | **0.25 m** | FIXED (old draft) |
| up-sampling | `5hz` (20 poses at 5 Hz → 10 Hz, linear from the origin, wrapped heading) | FIXED |
| footprint, lookup, reduction | unchanged: nuPlan Pacifica corners (max dev 1.4e-14 m f64), bilinear `grid_sample` on the int8 SDF x 0.1 m, out-of-extent corners carry exactly 0 loss and 0 gradient, sum / (n_labelled x 64 x 40 x 4), unlabelled sample = exactly 0 | FIXED, MEASURED by the module's tests |
| **w_field** | **2.0**, by the committed gradient-norm balance on P7's init: r(w) = 0.0493 / 0.0985 / 0.197 / **0.394** / 0.788 / 1.576 / 3.152 / 6.305 / 12.61 for w = 1/4 … 64; smallest w with r >= 0.25; K2, K2b not triggered. Never tuned on PDMS. | MEASURED on a **dev-box proxy pool** (deviations 1-6 in `WEIGHT_FIELDP7.md`), `HOLD/raw/weight_rule_p7_cpu.json` sha256 `90ae1b30…`, 8 micro-batch ids inside. **PROPOSED-FROZEN: MM ratifies.** The launch gate re-measures on the arm's own bank and may only veto (K2/K2b), never re-pick. |
| total loss | `(l_traj + l_score + w_field · L_field) / accum`; `--w-field` defaults to 0.0 and **every arm's argv names it** (C names `0.0`) | MEASURED reach: `REACHABILITY_FIELDP7.md` |

## Host H-P7: arms, recipe, cost
* **Init (all arms):** P7 `model_final.pt` (`2613fc17…`), strict warm start (`--init-from`); optimiser and schedule start fresh.
* **Recipe = the P5-C template** (`PK/refe/launch_gate.py` `p5c_argv`, tip `6b789b7e`) = P7's `ARGS` (`refe/p7_pod.sh:346` at P7's commit `181c7a3b`, INHERITED: P7's own `config.json` lives on the pod/HF and was **not** read) with: aux head OFF (the head is a sibling saved beside `model`, not in `model_final.pt`'s `model`: `train.py:1342-1355`; re-initialising it would add a confound), `--init-from`, `--epochs 0 --steps 2000`, `--lr 5e-5`, `--seed` named, plus for arm F: `--w-field 2.0 --field-labels <store> --field-margin 0.25 --field-interp 5hz`. Same P7 bank `train_grow_cmd_3c_p7` (150,603 tuples, `targets_sha256 92af8d66…`, registered in the gate), same `--ego-yaw-table`, `--yaw-wrap off`, `--labeller online`, batch 4 x accum 64, bf16, `--tf32 --compile`.
* **Steps: 2,000** (not P7's 9,075). Reason: a graft on a converged trunk at a quarter of peak LR is the P5-C recipe, its gate machinery exists, and 9,075 steps is ~127 h per arm (P6 rate). A NOT PROVEN here is **not** a refutation at scale (a graft is a weaker version of the effect than one from step 0).
* **Arms:** **F** `--w-field 2.0` · **C** `--w-field 0.0` (same init, bank, steps, seed 0) · replicates **F-seed** and **C-seed** (seed 1): **APPROVED** (PI 2026-11-17, "I approve your recommendation"), run after C (after C0r if the P5-C read triggers it), C-seed first; Amendment A.
* **Comparator: RULED (PI "Approve F+C", 2026-11-17). C is F's OWN control, a new run.** The proposal below is kept for the record and is NOT taken: P5-C's arm **C0** (P7 fine-tune, 2,000 steps, lr 5e-5, seed 0, raw labels, `--w-field 0.0`; occupies the pod until ~2026-11-19) can serve as **C**, and **C0r** (seed 1) as **C-seed**, **iff** `argv(F) − argv(C0) = {--w-field, --field-labels, --field-margin, --field-interp, --out}` exactly and the trainer's zero-weight identity holds. MEASURED here: with `--w-field 0.0 --field-labels`, the first-step gradient equals the run with no field flag **bit for bit on all 277 gradient-carrying parameters** (`HOLD/raw/reach/COMPARE.out`); the multi-step identity is the tiny-scale proof of 2026-10-13 (INHERITED). If not ruled, C and C-seed are new runs.
* **Cost (ESTIMATED, A40-h x 0.863 = L40-h; a decision size, not a commitment):** per arm 2,000 steps x ~60 s (the old draft's A40 rate) = 33.3 A40-h = **~28.7 L40-h**; cross-check 2,000 x 50.5 s (P6's MEASURED L40 rate) = 28.1 L40-h. Pair (F + C) **~57 L40-h**; F alone with C0 shared **~29**; four arms **~115**; F + F-seed with C0/C0r shared **~57**. Loss compute: 2.8-4.6 ms/step on CPU (old draft, MEASURED); K3 reads it on the pod.
* **Box:** the L40 pod, after P5-C (C0/C1) ends (~2026-11-19), in the PI's order **C1 → F → C → (C0r if the P5-C read triggers it) → C-seed → F-seed** (PI 2026-11-17). Eval is on the dev box, never on a training pod.

## Label store and S0 evidence
| item | value |
|---|---|
| train store | dev box `E:/Projects/TanitAD-artifacts/aux_drivable/label_store` (the old draft's `D:` path; drive letter moved); 103,139 frames (100 % of the 103,037-row rank-0 bank + 102 DB frames), `store_sha256` `4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a`, `manifest.json` sha256 `5b629af3ca5354ca260bafdcb179ad182a9acb68f8357e119076ee197c3671a9` (re-hashed 2026-11-17, MEASURED). HF copy `Sayood/tanitad-refe-navtrain-banks` `refe_aux_drivable_label_store_2026-10-23.tar` (353,802,240 B, equal 1,196/1,196 files, INHERITED `PK/raw/2026-11-06-p7-launch-ready/HANDOFF.md`) |
| **S0 gate (train store): PASSED** | `PK/raw/2026-10-12-aux-label-store/s0_gate_post.json` (`ZZGATE_PASS post`, `gate_pass: true`; both strata `db` and `inferred` pass AB / ORIGIN / CORNER / VERDICT / TARGET / LEAK; token and log overlap with navtest 0; train-bank coverage 1.0). MEASURED, read from the tip blob. |
| navtest store (evaluation only) | `…/label_store_navtest`, 12,146 frames, `store_sha256 985a3bc7…`, S0 PASS `PK/raw/2026-10-13-aux-navtest-store/s0_gate_navtest.json`, 0 overlap with the train store |
| ⚠️ **pod store mismatch (blocker G3)** | the field loader hard-pins the FULL store above (`train.py:664`); the pod's `aux_label_store_p7` was re-finalised in place (92,993 keys, sha `df2ddd1b…`) and would be REFUSED. Needs the full store beside it on the pod, or a loader change (`LAUNCH_GATE_GAPS_FIELDP7.md`) |
So K1 is not open for lack of S0; it is open for the pod-side pin only.

## Estimator, tier, endpoints, adverse checks, rule (carried from the old draft; PDMS levels updated)
* **Tier:** NAVSIM v1 PDMS, open-loop pseudo-simulation: **T1-class, not closed loop**. **Population:** full navtest, 12,146 tokens over 136 logs; harness and seam as for P7's read (`stage2_sft.py --ego-yaw-table <navtest table sha f5ebe24f…>`: P7 is a yaw-input model and the read route refuses without it; route fix ON, A7 repair ON, `navsim_v1` selection), dev box.
* **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927, for every difference. It answers *another draw of logs* only; inference is deterministic (ASSERT: two evaluations of one checkpoint bit-identical per token; a mismatch is a defect); training variance is the replicate (H-ESTIM-SEED-1).
* **MDE (ESTIMATED):** ~0.75 PDMS (SFT-4's paired half-width), plus the seed floor. **Ceiling (ESTIMATED):** the oracle DAC veto read **+1.64 [+0.90, +2.53]** on the 1,058-token C3 stratified population (MEASURED, veto read); the loss acts on raw plans and cannot touch the tracker-made share (29.2 %, INHERITED), so the plausible range is **+0.3 to ~+1.6** and **NOT PROVEN is a likely outcome, written down as such.**
* **Endpoints (paired, same tokens):** **E1** PDMS (F − C), the decision endpoint · **E2** DAC sub-score and DAC-zero share (the mechanism: a PDMS gain without fewer DAC zeros is not credited) · **E3** zeroed-token share · **M1** fan cleanliness: share of the 64 proposals NAVSIM-DAC-compliant, also on the raw plan (`navsim_dac_violation(mode="raw")`), on the C3 tokens (failure-enriched: not a rate) · **M2** raw → tracked transfer (ΔDAC raw vs ΔDAC LQR-tracked; raw improves but tracked does not ⇒ routes to tracker-aware compensation). All six sub-scores with paired CI; nothing pooled across families.
* **Adverse (thresholds proposed; PI may amend before landing):** **A1** EP Δ separated below 0 and point ≤ −1.0 (P7's EP is already its weakest: 86.68, P7−P6 −0.35 [−0.75, +0.02]) · **A2** picked plan's 4 s along-track, F/C upper bound < 0.97 · **A3** fan spread (mean pairwise 4 s endpoint distance) F/C upper bound < 0.90.
* **ADOPT** (per host) iff: E1 lower bound > 0; E2 DAC-zero-share upper bound < 0; none of A1-A3 adverse and no other longitudinal/lateral family component separates adversely; the replicate pair reproduces both signs (E1 point > 0 and DAC-zero-share point < 0); E1's lower bound exceeds the host's measured seed floor |E1(C-seed − C)|. Floor unmeasured at read time ⇒ **"separated, single seed, not yet a lever claim"**, never ADOPT.
* **REFUTED** iff E1 upper bound < 0, or E2 DAC-zero-share lower bound > 0. **TRADE** iff 1 and 2 hold and at least one of A1-A3 adverse. **NOT PROVEN** otherwise.
* **Size label** (reported with every ADOPT): LARGE iff E1 Δ ≥ +1.0 and DAC-zero share cut ≥ 30 %; else SMALL. Labels, not bars.

## Kill conditions (carried K1-K7; H-P7 readings)
| id | condition | then |
|---|---|---|
| **K1** | label gate fails (alignment < 95 % on ≥ 64 real frames, or a mutation arm not RED) | no launch. **S0 PASSED (above)**; the open item is the pod store pin (G3) |
| **K2 / K2b** | r(64) < 0.25 / r(1/4) > 1.0 | not triggered on the proxy pool (r(64) = 12.6, r(1/4) = 0.049); the gate re-measures on the real bank and vetoes only |
| **K3** | step time rises > 3 % or peak memory > 0.5 GB at micro-batch 4 | profile the loss on the L40 (the label read first) |
| **K4** | at any full-navtest read E1's upper bound < 0 | REFUTED; stop |
| **K5** | a zero-weight / reach control is not exact | no launch. **Reach MEASURED: GREEN** (bank-labeller site); online site by the pod gate |
| **K6** | logged `field_viol` at 25 % of steps > 0.8 x its step-0 value. **Operationalised pre-data (Amendment A):** mean `field_viol` over the logged rows with step in [475, 525] divided by the mean over step in [0, 10] > 0.8 ⇒ **FIRE**; any row with a non-finite `field_L` or a `field_w` other than the arm's weight ⇒ **INVALID** (`field_k6.py`). Read on **F only** (C carries `--w-field 0.0`), at step 525, by the pod chain | FIRE or INVALID: stop F and the chain (C is not launched); check labels, extent, `field_w` |
| **K7** | DAC-zero share falls but A1 or A2 adverse and E1 Δ ≤ 0 | stop; selection/progress calibration (0 GPU) |

## Deliberate-regression controls
Clean-fan identity (loss and gradient exactly 0.0): in `test_field_loss.py`, GREEN. Module mutation arms: 14, each fails ≥ 1 check, `ZZTEST_PASS 42`, GREEN. **Zero weight:** first-step gradients with `--w-field 0.0` equal the no-flag trainer on all 277 gradient-carrying parameters, max abs diff 0.0, equal sha256 (`HOLD/raw/reach`, MEASURED). **Term-not-added mutation** (line `train.py:1926` deleted): the check goes **RED** on all six proposal-side groups (MEASURED). Label-shuffle control (derangement seed 20261011): optional, not budgeted; forbidden by the gate until a dated amendment adds the arm (`--field-shuffle-seed`). Label misalignment: the S0 gate reads RED on a y-flip or a one-frame offset (S0 mutation arms).

## Replicate plan and the seed floor
* ⭐ **APPROVED** (PI 2026-11-17, "I approve your recommendation"; ~57 L40-h ESTIMATED): C-seed then F-seed, after C. A read taken before both exist is capped by the ADOPT rule at **"separated, single seed, not yet a lever claim"**; the bar is NOT lowered, and once both are read the floor is MEASURED and the rule as written applies.
* **C-seed** (= P5-C **C0r** if the comparator ruling allows, else new) = the zero-lever replicate, seed 1. Floor = |E1(C-seed − C)| and per sub-score. One pair is a single draw of the floor.
* **F-seed** = arm F at seed 1; ADOPT needs it to reproduce both signs.
* Three variances ride one interval: another draw of **logs** (the bootstrap), another **training run** (these replicates), another **inference run** (none: deterministic, asserted). Name which one a CI answered. With only the seed-0 pair the best verdict is "separated, single seed, not yet a lever claim".

## Reports: four families, per family, never pooled (binding), plus STRATEGIC
Instruments: `eval/instruments/families_run.py` (+ `fam_ci.py`) for LONGITUDINAL / LATERAL / TACTICAL on the identical tokens, each tested component with a paired CI; **`eval/instruments/strategic_family.py`** for STRATEGIC (REPORTED, not gating; the 2026-09-19 pause ended with the PI's 2026-11-16 ruling, so STRATEGIC is reported in full: S-DEC command adherence with per-command confusion, S-GOAL endpoint error / heading error / route-progress ratio / lateral offset per command class, paired log-cluster bootstrap).
| family | prediction registered before the read |
|---|---|
| LONGITUDINAL | EP is the adverse risk (A1); TTC unchanged; P7's under-progress rate (+0.013 vs P6) must not rise |
| LATERAL | **DAC up**; cross-track MAE not worse; no yaw gradient of its own beyond corner positions, so yaw-rate MAE should not move |
| TACTICAL | decision accuracy / κ and selection quality (random / actual / best-of-64, **random control mandatory**) rise because the fan is cleaner; actual − random may shrink (not adverse by itself) |
| STRATEGIC | command adherence and endpoint/route metrics unchanged within the interval; a move is reported, not gated |

## Admissibility (SPEC §6, unchanged)
Training-only: the label is a map-derived training target (PI 2026-10-11), never an inference input; the planner imports no label path (`grep -c "field_loss\|sdf" refe/model.py refe/planner.py` must stay 0). Navtest labels are never used to train or select; the weight rule read **navtrain** rows only. Goal ⟂ situation (PI 2026-08-03): the loss takes no goal, ego or navigation input. Anything built on it is "recipe + a declared deviation", never "reproduction".

## Outcomes committed in advance, and the next arm (Rule Zero)
| outcome | next lever, executed in the same turn that reads the verdict |
|---|---|
| **ADOPT, LARGE** | F-seed/C-seed if not run → longer graft or from-scratch rung (declared deviation) → combined arm with the head (H1) on the same labels → scorer ensemble (P7.2) |
| **ADOPT, SMALL** | the replicate first; no 7-day run on a lever whose lower bound sits at the floor |
| **TRADE** | selection/progress calibration on the same proposals (0 GPU), then a narrower variant (loss on the top-k proposals by score only) |
| **REFUTED** | bisect: (i) did the gradient reach (K6, r)? (ii) **M2**: raw improved, tracked not ⇒ tracker-aware compensation, not a new loss; (iii) label-shuffle arm (regularisation?); (iv) second arm at m = 0.5 m (the tracker's tolerance) |
| **NOT PROVEN** | the replicate (C-seed first: it prices the floor); more logs are not the answer. If the DAC-zero share did not move while the raw-plan fan cleanliness (M1) did, the picked plan is selected by the scorer, not the fan: route to the scorer arm (P5-C C1 / SCORER) |
| K7 | selection/progress calibration, 0 GPU |

## Blanks at landing
| blank | value |
|---|---|
| host, init, sha256 | **H-P7**, P7 `model_final.pt`, `2613fc17b9e5d63342c9f42120c5612a6ec7061fda72f83dac46ddd2e70fd3fe` (re-verified 2026-11-17: sha256 of `F:/TanitAD_p7read/run/model_final.pt` equals `summary.json`'s `model_final_sha256`) |
| trainer commit | P7 trained from **`181c7a3b86d4`** (`train.py` sha256 `eafd03e5…`). Candidate for the arm package: tip `6b789b7e` (`train.py` sha256 `56ea5e34f5bb…`; field-related lines identical to `181c7a3b`'s, MEASURED by diff; adds only the P5-C traj-label hook, default OFF). **The arm package = this file's landing commit**, built from that tree with autocrlf off and shipped as `/workspace/data/refe_p5c/code_<sha8>`; its sha8 and tree sha are recorded in each arm's launch evidence |
| module / test | `field_loss.py` `e29d91c3…`, `ZZTEST_PASS 42` (above) |
| label store | pod `/workspace/data/refe_navtrain/aux_label_store_full`: manifest `store_sha256 4a17db67` = the loader's pin, 103,139 keys, 1,192/1,192 files sha-verified (HF tar `refe_aux_drivable_label_store_2026-10-23.tar`, 353,802,240 B), MEASURED 2026-11-17 |
| `w_field` | **2.0** (proposed-frozen) |
| steps / lr / bank | 2,000 / 5e-5 / `train_grow_cmd_3c_p7` 150,603 tuples |
| comparator checkpoint and sha256 (C) | the new C run (`refe_field_p7_ctrl`, out `p7field_c`); its `model_final` sha256 is recorded at its post-gate. C-seed: the new C-seed run (seed 1), likewise |
| seeds | F, C: 0 · F-seed, C-seed: 1 (approved; after C) · derangement seed 20261011 (shuffle arm, not run) |
| harness commit and seam flags | as P7's read (`PK/raw/2026-11-13-p7-read-route/`), `--ego-yaw-table` navtest; commit **_ fill at read_** |
| C3 token set for M1/A3 | `PK/raw/2026-10-11-c3-selection-headroom/tokens.json` (1,058 tokens), instrument `c3_run.py`, forward identity `aux_navtest_forward.py` (INHERITED, old draft) |
| gate PASS token (commit, argv sha256, data-manifest sha256s) | per arm on the pod (`launch_gate.py`, profiles `refe_field_p7` / `refe_field_p7_ctrl`), banked with the arm's launch evidence; never hand-written |
| cost | ~28.7 L40-h per arm, ESTIMATED (above) |

## Amendment A (2026-11-17, written BEFORE any F or C data exists; MM, under the PI's rulings of the same day)
1. **Comparator:** C is F's own control, a new run (PI "Approve F+C"). P5-C's C0 / C0r are not comparators for this card.
2. **Replicates F-seed / C-seed: APPROVED** (PI 2026-11-17, "I approve your recommendation"), ~57 L40-h ESTIMATED, after C (after C0r if triggered), C-seed first; their gate profiles land before they launch. The ADOPT rule is unchanged; a read before both replicates exist is capped at "separated, single seed, not yet a lever claim".
3. **K6 operationalised** (K-table above), read on F only, at step 525, by the pod chain (`post_c1_chain.sh` → `field_k6.py`).
4. **Pod order** C1 → F → C → (C0r if triggered) → C-seed → F-seed (PI 2026-11-17).
5. **Gate:** G5 closed in the landing commit: G-LIVE runs P5-C's (f) warm-start identity and (h) no-table check on both arms, run_checks builds the capture for G-LIVE, G-CLOCK requires the bank content pin (`test_field_p7_profile.py` `chk_live_wiring`, with mutation arms).
6. **Cost:** F + C ~57 L40-h plus C-seed + F-seed ~57 L40-h, ESTIMATED (both PI-approved).

Nothing in the endpoints, the estimator, the thresholds or the decision rule changed.

## Amendment B (2026-11-19, written BEFORE any F or C training step exists; MM)
Both field arms' launch gates refused them on the pod. No F or C step ran. The two causes and their fixes:
1. **F, G-CKPT FAIL** (2026-11-19 05:43Z, `lg_F/FAIL_e9ecd8859f1c.json`): G-CKPT replays training under `torch.use_deterministic_algorithms(True)` and both runs died with `grid_sampler_2d_backward_cuda does not have a deterministic implementation`. The dev-box tests ran on CPU, where that backward is deterministic. **Fix: `refe/field_loss.py` `sample_sdf` is an explicit bilinear gather** (clamp = border padding, `align_corners=False` fractional index, `torch.gather` of the four neighbours from the constant label, the bilinear blend). The loss DEFINITION in the table above is unchanged, and so is the in-extent rule; only the module pin moves, **`e29d91c3…` -> `da7b43030cbc89b43da74c3c9ce48b12450d3080c702374dc2e62dea929563b3`**. Equivalence against a frozen copy of the old `grid_sample` lookup, MEASURED on CPU (`refe/test_field_loss.py`, `ZZTEST_PASS 54`, 19 source mutations RED): forward float64 max |diff| 3.5e-13 (noise SDF) and 5.3e-15 (smooth SDF) on 100,000 points per case, in-extent flags identical; forward float32 max 2.9e-6 m, mean 1.8e-7 m (both implementations sit ~3.5e-6 m from the float64 value at the same points: rounding order, not a different function); gradient w.r.t. the points, float64, 3.0e-12 on 200,000 points away from cell boundaries (measure-zero kinks); `field_loss` value 5.6e-17 and gradient w.r.t. `traj` 2.5e-18 over both interp modes and 5 fixtures; two deterministic-mode runs bit-identical (CPU). The CUDA determinism proof is G-CKPT itself on the pod.
2. **C, G-LIVE FAIL** (2026-11-19 06:15Z, `lg_C/FAIL_e9ecd8859f1c.json`; its G-CKPT PASSED): "declared loss term `field_L` ABSENT from the trainer's metrics". `train.py` computes and logs `field_L` only when `--w-field > 0`, so a control at 0.0 must not declare it. **Fix: `field_L` is a declared live term exactly for the arms that train it** (F, F-seed); C and C-seed no longer declare it. No argv, weight, seed, bank, step count or other profile key changes: `test_field_p7_profile.py` `chk_unchanged` undoes exactly these two changes and reproduces the `e9ecd885` profile hashes bit for bit (F `68f73903`, C `258b53f9`).
3. **One package for all four arms**, built from the commit that lands this amendment (C no longer runs from `code_e9ecd885`). Training-path identity against `e9ecd885`, MEASURED: of the 296 files the F profile ships, 293 are byte-identical; the 3 that differ are `field_loss.py`, `launch_gate.py` and `test_field_loss.py` (no trainer, model or data module).
4. **Pod order** C -> F -> (C0r if triggered) -> C-seed -> F-seed. C was started first (05:51Z) while F's lookup was fixed, to keep the pod busy (PI 2026-11-17, "dont let the pod idle"); its gate then failed (item 2) before any step. The order of independent runs does not enter any endpoint. The one cost: K6 (read on F at step 525) can no longer spare C's ~29 L40-h. C is kept anyway, because it is F's own control and a same-seed repeat of P5-C's C0.

Nothing in the endpoints, the estimator, the thresholds, K6 or the decision rule changed.
