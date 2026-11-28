# Gate-1 closed-loop-aware planner fine-tune — PROTOTYPE / mechanism de-risk — 2026-07-23

**One line:** Fine-tuned REF-C-base's anchored-diffusion **planner** (decoder only) on the 675 on-policy
junction states with the collected GT expert **recovery** targets (DAgger/CAT-K-style, no reward), then
re-ran the closed-loop junction suite. **The lever WORKS: junction offroad 11/15 → 7/15, at-fault
collisions 5 → 1, pass 3 → 8 — mechanism + pipeline DE-RISKED** (within-sim / ~3.2× OOD / in-scene —
a prototype de-risk, not a clean Gate-1 number).

## ⚠️ FRAMING (mandatory — this is NOT a clean Gate-1 number)
- **WITHIN-SIM RELATIVE, NuRec ~3.2× OOD.** The on-policy states are reconstructions ~3.2× off REF-C's
  real-footage distribution (RUN_RECIPE §13; REF-C open-loop ADE 1.47 on recon vs 0.47 real). A fine-tune
  on them **partly teaches reconstruction-OOD robustness, not pure closed-loop recovery.** This de-risks
  the **pipeline + objective + mechanism**; a clean Gate-1 read needs a lower-OOD on-policy source (worked
  in parallel).
- **n=15 → overfit-dominated.** The decoder **memorizes** per-scene recovery (generalization diagnostic
  below). The primary re-eval is **in-scene** (same 15 scenes trained on) → any offroad change confounds
  recovery-learning with scene-memorization. Read the DIRECTION + mechanism, not the magnitude.
- Every number here is **MEASURED** (artifact path given) unless tagged ESTIMATED/HYPOTHESIS.

## What was done (MEASURED)
1. **Extracted** the on-policy fine-tune data from the 15 junction `rollout.asl` logs
   (`gate1_extract.py`): per planned drive step, reconstructed the **exact** REF-C input the driver saw
   (`refc_driver.RefCPolicy.plan` preprocessing: `ftheta_crop_resize(center='principal')` →
   `stack_frames(3)` → `[8,9,256,256]`), paired with `v0` (= `preds.jsonl` speed, exact), `nav` (driver's
   `_nav_from_route`), and the DAgger label `ref_lookahead_rig` [4,2] (GT expert path 0.5/1/1.5/2 s ahead,
   rig frame = the decoder's own output space). **675 steps / 15 scenes.** Built-in checks: **f_eff
   265.5–266.4** (canonicalization correct, matches the driver's 266.1) and **pose_err = 0.0 m** on all
   675 (frame↔target alignment exact). Artifact: `/workspace/gate1_ft_data/*.pt` (287 MB, indexed by
   `manifest.json`).
2. **Fine-tuned the decoder only** (`gate1_finetune.py`). The 90.5 M ResNet encoder + all aux heads are
   **FROZEN**; only the **8,634,505-param anchored-diffusion decoder** (the planner) moves. Objective =
   the REF-C trainer's trajectory primaries with the target swapped for the recovery path: **anchor-cls CE
   + traj-recon L1**, diffusion mode (steps=2, as trained AND deployed). Adam lr 3e-5. The frozen
   encoder/strategic/measurement/maneuver forward is **cached once** (refc_v12_cache pattern; eval-mode, no
   ego-dropout = the deployment condition) so the decoder trains at ~0 encoder cost.
3. **Re-ran the closed-loop junction suite** (`gate1_reeval.sh`, same validated M2 topology, same 480×854,
   same renderer/session) for **fresh baseline** and the **fine-tuned** planner and scored AlpaSim's own
   offroad/collision/pass from each `metrics.parquet` (`gate1_score.py`, same `score_rollout` as the
   collection).

## Open-loop recovery gap the fine-tune closes (MEASURED)
The frozen REF-C-base planner's **selected plan** is far from the expert recovery path on its own
on-policy states — it does not recover:

| | selected-plan → recovery-target L1 (675 on-policy steps) | anchor cls-acc |
|---|---|---|
| **REF-C-base (frozen)** | **5.06 m** | 0.16 |
| **+ Gate-1 FT (step 800)** | **0.55 m** | 0.985 |

(`gate1_ft_result.json`.) Example, roundabout `c3d4065e`: base plan→recovery **11.44 m → 0.95 m** (FT).
The planner **can** be taught to emit the recovery path from off-policy states — the objective + loop work.

## Generalization diagnostic — the n=15 overfit is REAL (MEASURED)
Leave-3-scenes-out (`780ece49`, `c3d4065e`, `fd3a49fa` held out; train on 12; `gate1_ft_holdout_diag.json`):

| step | train recovery-L1 | held-out-scene recovery-L1 |
|---|---|---|
| 0 | 5.16 | 4.65 |
| 300 | 1.21 | **4.15 (best)** |
| 800 | 0.43 (cls 0.99) | 4.87 (degraded) |

**Train recovery-L1 collapses to near-zero (memorization); held-out-scene recovery barely moves (11 % at
best, then degrades).** The decoder **memorizes per-scene recovery** — cross-scene generalization is weak.
⇒ **sufficient to prototype/de-risk, NOT to train a robust policy** — exactly the collection note's verdict,
now quantified. Best cross-scene generalization is early (~step 300), so both a moderate (step 300) and a
maximal (step 800) FT ckpt were re-evaluated closed-loop.

## Closed-loop re-eval — junction OFF-ROAD (the pre-registered read)
Fresh **paired** runs (same renderer/session, 480×854; `gate1_reeval_scores.json`). The **baseline rerun
reproduces the original collection EXACTLY** (offroad 11/15, at-fault 5/15, pass 3/15, identical per-scene
outcomes) → the pipeline is deterministic and the paired delta is clean, not run-to-run noise.

| arm | offroad | at-fault collision | pass | intersection offroad | roundabout offroad | mean plan-dev |
|---|---|---|---|---|---|---|
| baseline (rerun) | **11/15 (0.73)** | 5/15 | 3/15 | 5/7 | 6/8 | 0.41 |
| **Gate-1 FT (step 800)** | **7/15 (0.47)** | **1/15** | **8/15** | **2/7** | 5/8 | 2.69 |
| Gate-1 FT (step 300) | **8/15 (0.53)** | **1/15** | 7/15 | 3/7 | 5/8 | 2.49 |

**Paired (ft800 − baseline), same 15 scenes:** offroad **11 → 7 (−4)** · at-fault **5 → 1 (−4)** ·
pass **3 → 8 (+5)**. (step 300: offroad 11 → 8 (−3), at-fault 5 → 1, pass 3 → 7.)
- **7 scenes RECOVERED on-road** (were offroad, now pass): `00169207 0810968d 41c06176 59cb0598` (int) ·
  `6dcd2117 c3d4065e d3267951` (rnd) — incl. `c3d4065e`, the clean covariate-shift roundabout. (step 300
  recovers 6 of these — all but `0810968d`.)
- **All 5 intersection at-fault collisions eliminated** (5→0; the corridor-tracking recovery also keeps the
  ego out of crossing traffic — a bonus beyond the pre-registered offroad scope; 1 new at-fault appears in
  roundabout `fd3a49fa`).
- **3 scenes went NEWLY offroad** (were on-road, now depart): `8b04d54e` (int) · `adb72a39 fd3a49fa` (rnd).
  **The SAME 3 scenes fail in BOTH ckpts**, and mean plan-deviation jumps **~6×** at BOTH (0.41 → 2.49 /
  2.69) → **the high-deviation trade is INTRINSIC to the recovery objective, not overtraining.** The
  recovery-trained planner is a **higher-deviation / swervier** planner — the *same* "high-deviation planner
  → offroad" mechanism the 2026-07-23 retraction flagged for flagship v1 (plan_dev 1.12 vs 0.34). It
  recovers most scenes but over-corrects a few previously-fine ones.
- **step 800 (max in-scene fit) edges step 300 closed-loop (7 vs 8 offroad)** even though step 300 had the
  better *cross-scene* open-loop generalization — consistent with the in-scene/memorization confound (the
  closed-loop suite IS the trained scenes).

## Pre-registered outcome
- **reduces junction off-road vs baseline → the lever WORKS** (mechanism de-risked; promote to a clean run).
- **no reduction / worse → warning, the lever needs rework.**

### ✅ VERDICT: the lever WORKS — mechanism + pipeline DE-RISKED (promote to a clean run)
Closed-loop-aware fine-tuning of the planner on on-policy recovery targets **reduced junction offroad
11→7 (−36 %)**, **cut at-fault collisions 5→1**, and **raised the pass rate 3→8** on the pre-registered
15-scene suite. The full chain is proven end-to-end: on-policy `.asl` frames extracted with **byte-exact**
alignment (f_eff 266, pose_err 0), the planner **learns** the recovery (open-loop 5.06→0.55 m), and driving
with it **reduces closed-loop departure**. Both outcomes were pre-registered; this is the **positive** one.

**This is a MECHANISM/PIPELINE de-risk, NOT a clean Gate-1 number — three load-bearing confounds:**
1. **~3.2× OOD (NuRec).** The reduction partly reflects reconstruction-OOD robustness, not pure
   closed-loop recovery. A clean number needs a lower-OOD on-policy source.
2. **In-scene + n=15.** Re-eval is on the trained scenes and the decoder **memorizes** per-scene recovery
   (holdout diagnostic: cross-scene recovery-L1 barely improves). It is *not* pure replay — the FT model
   visits its **own** new on-policy states and still recovers 7 scenes — but scene-appearance memorization
   cannot be excluded. Magnitude is not trustworthy; **direction is.**
3. **The recovery is high-deviation.** plan-dev **0.41→2.69 (6.6×)**; the FT trades stability for recovery
   and pushes **3 previously-fine scenes offroad** — the flagship-v1 offroad mechanism. A clean run needs a
   **deviation/stability regularizer** + **CAT-K/RoAD target filtering** (drop the catastrophic-state labels
   that point backward), so the net offroad win is not partly given back.

## Honest gaps / refinements for a clean Gate-1 run
- **Late-step recovery targets are a DAgger artifact.** At catastrophically-departed states (exec_xte up
  to 24.6 m) `ref_lookahead_rig` points far **backward** (nearest corridor is 20 m behind); the FT
  reproduces these. A robust run should **filter/weight to the recoverable regime** (CAT-K top-K / RoAD)
  rather than clone every state.
- **Intersection collision mode** (at-fault 5/7 baseline) is **not** addressed by a route-following
  recovery target — it needs an added collision/agent-cost signal. The clean covariate-shift lever lives
  in the **roundabouts** (0 at-fault).
- **OOD** (load-bearing): a real Gate-1 number needs on-policy states at REF-C's real-footage distribution
  (or the recon gap closed).
- **n**: 15 scenes / 675 steps de-risks the loop; a generalizable policy needs many more (the roundabout
  classifier/collection pipeline is staged: `select_suite.py` / `kf_batch.py`).

## Deliverable manifest
| artifact | where | status |
|---|---|---|
| `GATE1_PROTO_NOTE.md` (this) | repo (staged) | write-up + verdict |
| `gate1_extract.py` | repo (staged) · pod `/workspace/` | on-policy (frames+recovery) extractor, self-checking |
| `gate1_finetune.py` | repo (staged) · pod `/workspace/` | decoder-only DAgger fine-tune (cached frozen fwd) |
| `gate1_reeval.sh` / `gate1_score.py` | repo (staged) · pod `/workspace/` | paired closed-loop re-eval + AlpaSim-metric scorer |
| `gate1_ft_result.json` | repo (staged) | FT curve + baseline/final recovery-L1 |
| `gate1_ft_holdout_diag.json` | repo (staged) | leave-3-scenes-out generalization curve |
| `gate1_reeval_scores.json` | repo (staged) | per-scene + aggregate offroad/collision/pass, paired |
| FT ckpts `ckpt.pt` (=step800) + `ckpt_step{300..800}.pt` + `config.json` | **pod only** `/root/models/refc-gate1-ft/` | fine-tuned planner (indexed here + in `gate1_ft_result.json`); step0/100/200 pruned |
| extracted data `*.pt` | **pod only** `/workspace/gate1_ft_data/` | 675 on-policy (frames,v0,nav,recovery) bundles |
| re-eval rollouts (`metrics.parquet`, `.asl`) | **pod only** `/workspace/gate1_reeval_{base,ft800,ft300}/` | AlpaSim closed-loop outputs |

**Pod left CLEAN:** services down, GPU 0 MiB, `gpu_lock` released. No deletions of the baseline collection.

## Reproduction
```
gpu_lock.sh acquire gate1-proto
# 1. extract on-policy fine-tune data from the 15 rollout.asl
bash /workspace/run_extract.sh --out /workspace/gate1_ft_data
# 2. decoder-only DAgger fine-tune
bash /workspace/run_ft.sh --steps 800 --save-every 100 --lr 3e-5 --out /root/models/refc-gate1-ft
# 3. paired closed-loop re-eval (renderer persists across tags)
bash /workspace/gate1_reeval.sh base /root/models/refc-base-30k/ckpt.pt base
bash /workspace/gate1_reeval.sh ft800 /root/models/refc-gate1-ft/ckpt.pt base
CUDA_VISIBLE_DEVICES="" .venv/bin/python /workspace/gate1_score.py \
    base:/workspace/gate1_reeval_base ft800:/workspace/gate1_reeval_ft800
```
