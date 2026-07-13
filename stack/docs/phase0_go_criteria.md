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

---

## 9. Unified eval in TanitResim (visual overlays + formal gates, one tool)

The formal gate suite is wired into **TanitResim** so one tool gives visual
overlays + per-arm metrics + the formal D1–D3 gates + the Phase-0 GO verdict, in
BOTH the script and the web UI. The gate code has ONE home
(`compare_arms.py`); `replay_app.py` adapts each loaded replay arm
(`MainArm`/`RefAArm`/`RefBArm`) into the same `ArmSpec` and calls
`compare_arms.compute_arm_gates` — so a checkpoint gated inside TanitResim
reconciles **byte-for-byte** with a standalone `compare_arms.py` run on the same
episodes (pinned by `tests/test_compare_arms.test_resim_arm_reconciles_with_builder`).

- **`replay_app.py --mode test`** — `stats.json` gains, per arm,
  `arms[name].gates` (D1/D2/D3 PASS/FAIL/BLOCKED + `d1_ade_0_2s`,
  `oracle_ceiling_ade_0_2s`, `grounded_ade_0_2s`, `d2_dir_acc`, `d3_ratio`) plus
  a top-level `gates` block (shared baselines + Phase-0 GO verdict). The existing
  ADE/FDE/action/maneuver/latency stats are untouched.
- **`replay_app.py --mode export` / `viz`** — the session bundle
  (`session.json`) carries `meta.arms[*].gates` + `meta.gates` (verdict +
  baselines); `--mode viz` writes the same gated `stats.json` as `test`.
- **Web UI** (`resim/static/app.js` + `style.css`) — a per-arm **Formal gates**
  panel (D1/D2/D3 status badges + D1 ADE / oracle ceiling / grounded ADE / D2
  dir-acc) renders alongside the camera-fan and head-readout panels, and a
  **Phase-0 GO banner** (CV floor, D1 decode winner, flagship hierarchy-edge
  conditions, the necessary-not-sufficient caveat) sits under the session header.

Set `--main-config flagship4b` so TanitResim's `main` arm IS the 4-brain
flagship (visualized + gated). Gates run over ALL loaded val episodes (the gate
does its own internal route-resampled splits), NOT the resim fit/replay overlay
split — that is what makes the numbers reconcile with `compare_arms.py`. Opt out
with `--no-gates`; a gate failure never breaks the overlay/stats output.

Unified command (flagship + REF-A + REF-B, overlays + gates + UI bundle):
```bash
python scripts/replay_app.py --mode export --main-config flagship4b \
    --arms main:<flagship>/ckpt.pt refa:<refa>/ckpt.pt:grid refb:<refb>/ckpt.pt \
    --data-root /workspace/data/physicalai/_epcache --corpus-glob '*val*' \
    --episodes 150 --out /workspace/resim/flagship-30k
python scripts/resim_app.py --port 8888 --sessions-root /workspace/resim
```

---

## 10. Behavior gate (tactical maneuver + strategic route), unified

`eval_behavior.py` now takes `--config` (`flagship4b` / `flagship4b_reduced` /
`base250cam` / `smoke` / ...) — it previously hard-coded `base250cam_config()`
and strict-loaded, so it could not open a flagship4b checkpoint (the 4-brain
policy keys + rebalanced depths). Standalone:

```bash
python scripts/eval_behavior.py --ckpt <dir>/ckpt.pt --config flagship4b \
    --cache-dirs /workspace/data/physicalai_phase0/_epcache --out <dir>/behavior
```

**Behavior is also wired into the unified suite** (compare_arms / replay_app /
watch_gates) as a per-arm `behavior` block, arm-agnostic and on the SAME episode
grid as D1–D3:

- **maneuver decodability** (tactical): balanced-accuracy of a class-weighted
  linear probe `compact state -> GT maneuver class` (kinematic labels,
  `eval_behavior.gt_maneuver`), reported vs chance (1/5) with `beats_chance`;
- **route-intent decodability** (strategic): the same probe `state -> GT route
  {left,straight,right}` on route-valid windows.

These are `eval_behavior`'s PRIMARY instrument (the flagship has no trained
maneuver/route *head* in Phase 0 — decodability is a prerequisite for any future
selection head; the selector itself is a Phase-0 gap). The block reuses
`eval_behavior.fit_classifier` + `probe_metrics` + labels **verbatim** and
replicates the `maneuver_probe_eval` encoder_state/_all/linear cell exactly, so
the unified number **reconciles byte-for-byte** with a standalone
`eval_behavior.py --config flagship4b` run (pinned by
`tests/test_compare_arms.test_behavior_probe_reconciles_with_eval_behavior`).
The imagine-and-select secondary and the per-arch native maneuver/route heads
(flagship `tactical_policy`, REF-B `TacticalHead`) stay in `eval_behavior.py`
proper. Toggle with `--behavior-epochs` / `--no-behavior` (on by default in the
CLI, replay, and watch; off in the `compare()` library default to keep unit
tests fast).

---

## 11. Turnkey watch_gates deploy (dev-box 4060, auto-pull from pod2)

`watch_gates.py` can auto-pull the flagship's checkpoint + a val subset off the
training pod and gate them on the 4060 — non-disruptive to training (read-only
scp; the gate suite never runs on the training GPU). The pull is shell-free
(one ssh to list the val subset on the pod, one scp for it; a `stat`-guarded scp
of the checkpoint only when its mtime changes, so a ~1 GB ckpt is not re-pulled
each poll).

**Standing command** (fires 5k/10k/20k/30k gates automatically as pod2
overwrites `ckpt.pt` every 1000 steps):

```bash
python scripts/watch_gates.py --arm flagship --flagship-config flagship4b \
    --exp-dir  /local/pull/flagship-30k \
    --pull-host tanitad-pod2 \
    --pull-ckpt /workspace/experiments/flagship4b-phase0-30k/ckpt.pt \
    --pull-val  /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
    --pull-val-episodes 150 \
    --episodes 150 --interval-s 600
```

- val subset pulled ONCE to `--exp-dir/val_subset/physicalai-val/`; ckpt scp'd
  to `--exp-dir/ckpt.pt` whenever its pod mtime changes; each new step appends a
  row to `--exp-dir/gates/gate_log.jsonl` + writes `gates_step<STEP>.json/.md`.
- Preview the exact scp/ssh commands without running anything (or without a
  configured `tanitad-pod2` alias) via `--dry-run-pull`.
- Requires the `tanitad-pod2` ssh alias in `~/.ssh/config` (IP/port change on pod
  restart — update the alias). On Git Bash, prefix `MSYS_NO_PATHCONV=1` (or run
  from PowerShell) so the POSIX remote paths are not mangled.

**Reconcile check:** the on-pod first gate at step 1000 reported grounded ADE
7.18 m vs CV 0.825 m (immature, as expected pre-training). A dev-box watch run at
the same step on the same 150-ep val subset must land in that neighbourhood
(within route-split CI); a large divergence means a val-subset or config
mismatch, not a real gate change.

---

## 12. Val provisioning for the 3-arm comparison (trigger on the REF pods)

All three arms must be evaluated on the **same val episode ids** (§5). Here is
exactly what each pod must produce and how `compare_arms.py` consumes it. The
flagship's `physicalai-val` cache is the **canonical** episode set; REF-A's
features and REF-B's eval both key off its `episode_id`s.

| # | producer | artifact | contract | status |
|---|---|---|---|---|
| 1 | **pod2 (flagship)** | frame val cache | `physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11/ep_*.pt` (600 eps): `frames_u8`, `actions`, `poses`, `episode_id` | **exists** — the canonical set |
| 2 | **REF-A pod** | val **DINO features** | `<feat-dir>/ep_*.pt`: `feats_fp16 [T,256,768]`, `actions`, `poses`, `episode_id` — computed on the SAME `physicalai-val` clips (same ids/poses) | **MUST PROVISION** (REF builds skip val) |
| 3 | **REF-B pod** | — (reuses #1) | REF-B evaluates on the shared frame val cache #1; no REF-B-specific val artifact | reuses #1 |

**Provisioning actions when a REF pod finishes training:**

- **REF-A** — run `dino_precompute.py` over the physicalai epcache root (it
  globs `*train*`/`*val*`); `--train-n 0` skips train so only the val subset is
  encoded, `--val-n 150` bounds the shared subset:
  ```bash
  # on the REF-A pod — writes <out>/physicalai-val-0c5f7dac3b11-<tag>/ep_*.pt
  # with feats_fp16 [T,256,768] + the SOURCE actions/poses/episode_id (same clip)
  python scripts/dino_precompute.py \
      --cache-root /workspace/data/physicalai_phase0/_epcache \
      --out /opt/dino_feats --train-n 0 --val-n 150
  ```
  (The `_epcache` root must contain both a `*train*` and a `*val*` dir — the
  flagship's does. Output tag is `dinov3-b16` or `dinov2-b14`, recorded in
  `<out>/META.json`.) Ship `<out>/physicalai-val-...-<tag>/ep_*.pt` to wherever
  `compare_arms.py` runs; `episode_id`/`poses` are copied from the source, so the
  feature file is the same clip in DINO space.
- **REF-B / flagship** — only the checkpoints need to reach the compare host
  (`scp <pod>:<exp>/ckpt.pt`); both read frame val cache #1.
- **Shared subset** — use the first `N` (deterministic sorted `ep_*.pt`, e.g.
  150) so every arm can hold it in memory and cross-pod fairness holds. Compute
  REF-A features for exactly those ids.

**How `compare_arms.py` consumes them (fairness enforced in code):**
```bash
python scripts/compare_arms.py \
    --flagship-ckpt <flagship>/ckpt.pt --flagship-config flagship4b \
    --refa-ckpt <refa>/ckpt.pt --refa-adapter grid \
    --refb-ckpt <refb>/ckpt.pt \
    --frame-cache-dirs <dir with physicalai-val-0c5f7dac3b11> \
    --refa-feat-dir    <dir with physicalai-val-...-dinov2-b14> \
    --episodes 150 --out <out>/arm_compare
```
`load_common_val` intersects the frame cache (#1) and the DINO features (#2) by
`episode_id`, keeps only episodes present in **both**, and asserts their `poses`
match to `--pose-tol` — so even if the feature subset and the frame subset differ
at the edges, the arms are scored on exactly the shared intersection, and the
episode-id list used is emitted in the report. If REF-A features are missing,
omit `--refa-ckpt`/`--refa-feat-dir` and the flagship↔REF-B comparison still runs
on the full frame val.
