# Phase-0 GO criteria & the three-arm gate harness

This document defines the **decode + open-loop GO criteria** the automated gate
harness (`stack/scripts/compare_arms.py`, `stack/scripts/watch_gates.py`)
decides, ties them to **Phase 0 Plan §4** gates D1–D9, and states exactly which
val artifacts must be provisioned before the real three-arm comparison.

It is deliberately explicit that these are **necessary, not sufficient**
conditions: per `tanitad/eval/gates.py` doctrine (and arXiv 2512.24497, *"decode
quality does not reliably predict planning success"*), the closed-loop gates
**D4–D6 remain the arbiters** of the hierarchy edge and are computed in sim, not
by this harness.

---

## 1. The three arms (the controlled comparison)

All three train on the **byte-identical** PhysicalAI set; only the architecture
differs — that identity is the whole point.

| arm | architecture | eval input | native trajectory decode | imagination (D2/D3) |
|---|---|---|---|---|
| **flagship** | 261 M 4-brain: from-scratch ViT + operative/tactical/strategic predictors + H15 + metric-dynamics grounding + SIGReg (`flagship4b`) | raw frames | grounded operative rollout (`rollout_decode`, `grounding.step['op']`) | yes |
| **REF-A** | frozen-DINO → trainable adapter → shared predictor (`--adapter grid`) | DINO features `[T,256,768]` | grounded operative rollout (`ck['step_readout']`) | yes |
| **REF-B** | from-scratch ViT, behaviour cloning, **no world model** | raw frames | direct tactical waypoint head | **no** (pre-registered structural gap) |

---

## 2. The defined benchmarks (wired, not reinvented)

The harness reuses the existing eval code verbatim — it assembles, it does not
re-implement:

- **D1–D3 decode gates** — `tanitad.eval.gates.run_d1/run_d2/run_d3` (the same
  functions `evaluate_checkpoint.py` calls), instrument-doctrine PASS/FAIL/BLOCKED.
- **Grounded-decode ADE** — `tanitad.models.metric_dynamics.rollout_decode` (the
  `eval_grounded_rollout_4b.py` method): roll the operative predictor under the
  **true** actions, decode each transition's metric Δpose, SE(2)-accumulate to
  ego waypoints at 0.5/1/1.5/2 s.
- **Trivial baselines + oracle ceiling** — `driving_diagnostic.py` helpers:
  constant-velocity / go-straight / constant-yaw-rate + the in-distribution
  (fit==eval) oracle decode ceiling.
- **Behaviour (tactical/strategic)** — `eval_behavior.py` (optional add-on; see
  §7 caveat — its `main()` hard-codes `base250cam_config()` so behaviour is wired
  through its internal functions, not its CLI).

**Metric identity (rigorous).** Every arm's compact state
(`encode_window(...)[:,-1]`) goes through the *same* `decode_parity` — same
frozen `RidgeProbe` ladder, same `run_d1`, same route-resampled episode splits.
The GT ego-waypoints, the baselines and the strata are built **once** from the
val poses and shared by every arm. `ade_0_2s` (the 4-waypoint mean) means the
same thing in the D1 row, the grounded row and the baseline row. The only thing
that differs per arm is the state tensor (the architecture axis under test) and
the trajectory *mechanism* (labelled honestly: flagship/REF-A roll a grounded
predictor; REF-B reads its BC waypoint head).

---

## 3. Gates D1–D9 (Phase 0 Plan §4, verbatim thresholds)

| Gate | Claim | Threshold | Harness? |
|---|---|---|---|
| **D1** | encoder state decodable | frozen-probe **ADE@1s < 0.5 m (BEV) / < 1.0 m (camera)**; I2, I3 pass | **decides** (camera, `run_d1`) |
| **D2** | imagination usable for selection | calibrated **dir-acc > 0.7 OR P4 forward-dynamics acc > 0.7**; I1 ≈ 1.0 first (imag-rel is *diagnostic*, D-017/A13) | **decides** (`run_d2`) |
| **D3** | trajectory decode from imagination | **imagined-ADE@2s ≤ 1.5× oracle-decode ADE@2s** | **decides** (`run_d3`) |
| **D4** | tactical beats greedy | +15 % success on interactive scenarios | closed-loop (sim) — **not here** |
| **D5** | strategic routing beats greedy on topology | blocked-route success 4B ≫ operative (≥ +30 % abs) | closed-loop (sim) — **not here** |
| **D6** | hierarchy generalizes simple→complex | success-degradation slope 4B < flat at matched params | closed-loop (sim) — **not here** |
| **D7** | memory helps rare scenarios *(stretch)* | repeat-exposure improvement, no interference regression | out of scope |
| **D8** | monitor detects OOD | AUROC > 0.85 (unseen town/weather) | separate harness (WP7) |
| **D9** | H15 imagination in unobserved areas | hidden-sector cosine ≥ shuffled + 0.2; calibration gap > 0 | separate (train-time D9 rows) |

**Program rule (Plan §4):** *no architecture change may be motivated by a gate
that hasn't passed its instrument rows.* The harness enforces this mechanically —
a gate whose I1–I4 instruments fail is reported **BLOCKED**, never FAIL, and
contributes no claim.

---

## 4. Phase-0 GO verdict (what this harness decides)

Combining Plan §4 (D1–D3) with the **DRIVING_DIAGNOSTIC_FRAMEWORK §3** revised
open-loop exit criteria (Sayed: *solid single-cam driving before scaling*):

**Necessary conditions the harness checks (the `verdict` block):**
1. **D1 decode** — flagship frozen-probe `ade_0_2s` **beats REF-A and REF-B**
   (the parity edge) and the D1 gate is admissible (I2/I3 pass).
2. **Beat the floor** — the flagship's grounded trajectory beats the
   constant-velocity baseline (overall and, per §3, on the **straight AND curve**
   strata). CV/go-straight/constant-yaw are the interpretability floor everyone
   must clear.
3. **Decodability matures** — held-out `ade_0_2s` within a defined factor of the
   **measured** oracle ceiling (readout generalizes), and the grounded ADE
   trends toward that ceiling as steps increase.
4. **D2 usable / D3 faithful** — flagship D2 dir-acc > 0.7 (admissible); D3
   imagined/oracle ≤ 1.5× (admissible).
5. **Hierarchy edge (open-loop portion)** — flagship ≥ REF-A/REF-B on the
   grounded trajectory metric.

**A GO decision is NOT made on these alone.** The `DOCTRINE` field of every
report restates that D4–D6 closed-loop (interactive success, blocked-route,
simple→complex slope) arbitrate, per Plan §7 Definition of Done ("D1–D3 pass; D5
or D6 shows the hierarchy edge"). This harness produces the D1–D3 + open-loop
evidence; the sim harness produces D4–D6.

---

## 5. Eval on VAL (determinism + what to provision)

Everything runs on the **val** split, never train. The split is
**seed-deterministic** (not a hard-coded episode count): `val_frac=0.2`, fixed
`seed` (0), episode-level disjoint (I3), guarded by
`validate_data.check4_mix_split` (zero train/val id overlap). The route-resampled
protocol means ADE/FDE are **means over 8 episode-level splits** with a 95 % CI —
a single split's ADE swung 5.2→11.5 m on identical latents (step-21k incident),
so single-split numbers are split-luck, not measurements.

**Val artifacts to provision before the real three-arm run** (the harness
consumes these — it does not build them; provisioning is upstream):

| artifact | consumed by | contract | path passed to harness |
|---|---|---|---|
| **frame val cache** | flagship, REF-B | `<root>/*val*/ep_*.pt` (`mixing.save_episode`: `frames_u8`, `actions`, `poses`, `episode_id`) | `--frame-cache-dirs` |
| **DINO feature val** | REF-A | `<dir>/ep_*.pt` (`dino_precompute`: `feats_fp16 [T,256,768]`, `actions`, `poses`, `episode_id`) for the **same** val episode ids | `--refa-feat-dir` |

**Known dependency (must be resolved before the real comparison):** REF-A/REF-B
builds currently **SKIP the val split** to save disk, and REF-A needs **DINO
features of val**. Provision (a) the frame val cache for flagship/REF-B and (b)
the DINO-feature val cache for REF-A, for the **same** episode ids. A shared
**subset** (100–200 episodes, `--episodes`) is acceptable and preferred for
cross-pod fairness and disk — all three arms must be able to hold it.

**Same-episode fairness guarantee (in code).** `load_common_val` intersects the
frame and feature val caches by `episode_id`, keeps only episodes present in
**both**, and asserts their `poses` match to `--pose-tol` (the same clip in two
representations must have the same odometry). The reference grid is built from
those shared poses; every arm is scored on the same windows of the same clips.
The exact episode-id list used is emitted in the report (`val.common_episode_ids`).

---

## 6. ⚠ Two constants to confirm before the real run (repo evidence disagrees)

The harness makes both **configurable and measured** rather than hard-coding a
number the tree does not support:

1. **Oracle ceiling "0.68 m".** Cited by steering as the grounded-decode target,
   but **not present anywhere in the repo** (word-boundary search). The
   in-tree oracle-decode references are **1.52 m (ADE@2s)** and **1.65 m
   (ADE@1s, MLP)** — see `Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`
   §Results (27k) and `gate_results/flagship_step27000_2026-07-12.json`. The
   harness **measures** the in-distribution oracle ceiling from the val data
   every run (the honest, data-derived number) and reports the grounded ADE
   relative to it; `--oracle-target` (default **1.65**) is a *reference only*.
   **Action: confirm whether 0.68 m is a pod-side grounded-rollout artifact or an
   aspirational target, then set `--oracle-target` accordingly.**
2. **Val size "2376 / 600 episodes".** No `2376` in the tree; the documented
   PhysicalAI val is **100 episodes / ~2240 windows** (`PRE_FLIGHT_VALIDATION.md`
   Axis 1: comma 410/90, physicalai 400/100). Steering notes val is being rebuilt
   to **600**. The harness does **not** assume a count — it takes `--episodes`
   (a configurable subset of whatever val is provisioned) and reports the actual
   episode ids used.

---

## 7. Run commands

**Three-arm comparison** (dev-box 4060 or pod), any subset of arms:

```bash
python scripts/compare_arms.py \
    --flagship-ckpt <dir>/ckpt.pt --flagship-config flagship4b \
    --refa-ckpt     <dir>/ckpt.pt --refa-adapter grid \
    --refb-ckpt     <dir>/ckpt.pt \
    --frame-cache-dirs /workspace/data/physicalai/_epcache \
    --refa-feat-dir    /opt/dino_feats/physicalai-val-dinov2-b14 \
    --episodes 150 --out <dir>/arm_compare
```

Emits `arm_compare.json` + `arm_compare.md` (the comparison table + per-metric
winner + hierarchy-edge necessary conditions).

**Checkpoint-watch** (auto-run the gate suite on val as milestones land at
1k/5k/10k/20k/30k):

```bash
# poll loop (dev-box, RECOMMENDED — see below)
python scripts/watch_gates.py --arm flagship --flagship-config flagship4b \
    --exp-dir <local-pull>/flagship-30k \
    --frame-cache-dirs <local-pull>/physicalai_val \
    --episodes 150 --interval-s 300 --out <local-pull>/flagship-30k/gates

# single pass after copying a milestone checkpoint
python scripts/watch_gates.py --arm flagship --flagship-config flagship4b \
    --exp-dir <dir> --frame-cache-dirs <val> --episodes 150 --once
```

Appends one JSON line per checkpoint to `<out>/gate_log.jsonl`
(`{ts, step, summary{D1,D2,D3}, d1_ade_0_2s, grounded_ade_0_2s, cv_ade_0_2s,
grounded_beats_cv, ...}`) plus a full `gates_step<STEP>.json/.md` per milestone.

**⚠ eval_behavior caveat.** `eval_behavior.py::main()` hard-codes
`base250cam_config()` and strict-loads — it will **not** load a `flagship4b`
checkpoint via its CLI. Wire behaviour through its internal functions
(`eb.collect`, `eb.maneuver_probe_eval`, `eb.strategic_probe_eval`) with a
correctly-configured `WorldModel`, or generalize its `main()` to accept
`--config` (small follow-up, out of this harness's scope).

---

## 8. Deployment recommendation (dev-box vs pod — GPU contention)

**Recommended: dev-box 4060, pulling each new `ckpt.pt` + a fixed val subset off
the pod.** The gate suite competes for GPU; running it on the training pod steals
cycles from the 30k run it is measuring (already GPU-bound). The 4060 evaluates a
150-episode val subset in a few minutes and never touches the training GPU. Cost:
copy the checkpoint (~0.5–1 GB) + the val subset once per milestone.

**Alternative: `--once` on the pod** at a checkpoint boundary (trainer between
saves) — fine for a single milestone, not for a tight poll loop.

The dev-box path also keeps the comparison honest across pods: all three arms'
checkpoints are pulled to one machine and evaluated on one shared val subset with
one code path.
