# V6 STAGED TRAINER — design, per-stage runbook, gates, costs
### the dev-side build that lets v6 training start tomorrow

**Status:** implemented and STAGED (not committed, not pushed — the agent contract).
**Implements:** `V6_TRAINING_MEASURES.md` (O1–O6 / T1–T5 / S1–S3 / C1–C2 / X1–X5),
`HIERARCHY_VOCABULARY.md` v0.2 (§3/§4 vocabulary, §4b the binding 6 s horizon, §5 wiring),
`JEPA_PHYSICS_SURVEY.md` (LF0–LF4, §4 staged-training evidence, §5 the S-W/S-P/S-J recipe).
**Tier stamp:** everything here is MECHANISM. No number produced by this trainer is quotable
as driving performance — capability claims are **T1** (`taniteval/tools/t1_eval.py`) with the
**four metric families** and the **paired episode-cluster bootstrap** (`taniteval/ci.py`).

---

## 1. Module map — what was built, and what it reuses

| file | what it is | LOC-scale role |
|---|---|---|
| `stack/tanitad/models/v6.py` | the v6 **composition module**: `V6Config`, `V6Stack`, the goal-token vocabulary, the X3 isolation check, and the pure measure primitives (O2/O3/O4/O6) | the model |
| `stack/scripts/train_v6_staged.py` | the **staged trainer CLI** (`--stage S-W|S-T|S-S|S-J`), the measure losses, the per-stage gate hook, `--dry-run` | the loop |
| `stack/tests/test_v6_staged.py` | 80 CPU-only pins — no GPU, no corpus, no checkpoint | the proof it assembles |

### 1.1 What is IMPORTED, never re-implemented

This is the point of the build: v6 is a **composition** of parts that already passed gates.

| imported from | what for | why it is the right source |
|---|---|---|
| `tanitad/models/encoder.py` `ViTEncoder` + `readout.py` `SpatialGridReadout` | the trunk | the readout is the **geometry firewall** — a wide 256×640 input still yields `state_dim` 2048 |
| `tanitad/models/predictor.py` `OperativePredictor` | layer O's predictor | its `intent` FiLM port **is** the `g_tac` conditioning seam: `P_O(z_op, (a,κ) | g_tac)` |
| `tanitad/models/tactical.py` `FTac` | layers T **and** S predictors | the residual-MLP latent-dynamics family; a tactical roll is cheap because the state is small and the clock is slow |
| `tanitad/models/metric_dynamics.py` `StepDisplacementReadout`, `rollout_transitions` | the metric decode + the shared roll | one rollout convention in the programme, not two |
| `scripts/train_stage_a.py` `stage_a_losses` + the counterfactual machinery | **O1** | the response-form `L_ctrl` that PASSED the W3 gain gate (0.27 → 0.97). O1 is that loss **from step 0** instead of as a post-hoc repair |
| `scripts/train_v58f_unicycle_head.py` `UnicycleEmission` | the 60-step emission | W4-gated: `a = a_max·tanh`, `κ = κ_max·tanh` — feasible **by construction**, census violations 0.0 |
| `tanitad/models/sigreg.py` `SigReg` + `position_relaxed` | **O6** | LeJEPA, λ=0.1, 512 slices; the "never divide by n" bug is already guarded inside |
| `tanitad/eval/spectral.py` `effective_rank`, `participation_ratio` | O6's spectrum monitor | the training monitor and the offline instrument **cannot drift apart** |
| `scripts/train_v58f_unicycle_head.py` `build_train_episodes`, `make_sampler` | data seam | parity guard + geometry binding + the MEASURED episode-grouped I/O shape (~8× fewer cold payload loads) |

### 1.2 `V6Stack` — the wiring (HIERARCHY_VOCABULARY §5, verbatim)

```
z_str_{t+K} = P_S(z_str_t, a_str_t)
z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)
z_op_{t+j}  = P_O(z_op_t, (a,κ)_t | g_tac)
```

```
frames ──► encoder ──► readout ──► z_op ─┬─────────────────────────────► P_O  (10 Hz)
                                          │                                ▲
                        ┌── stop-grad/EMA ─┘                    g_tac ──────┘
                        ▼
                  adapter_tac ──► z_tac ─┬───────────────────► P_T  (~2 Hz)
                        │                 │                       ▲
                        │   ┌─ stop-grad ─┘             g_str ────┘
                        │   ▼
                        │ adapter_str ──► z_str ──────────────► P_S  (~0.5 Hz)
                        │                   │
                        │                   └─► goal_head_str ─┐
                        └──────────────────────► goal_head_tac ◄┘   goals flow DOWN
                                                     │
                       z_op (planner view) ──► emission ◄── g_tac embedding
                                                     │
                                          ONE 60-step (a,κ) @10 Hz → 6 s
```

**Goals flow DOWN only; latents flow UP through stop-grad / EMA.**

### 1.3 The three things enforced mechanically

1. **X3 gradient isolation.** `V6Stack.assert_isolation()` is a real **autograd probe**, not a
   comment: it backprops the module's own DECLARED planner-side surface and reads which
   parameters actually received gradient (`torch.autograd.grad(..., allow_unused=True)` — it
   never touches `.grad`, so it is safe mid-training). Three edges are probed:
   `planner → encoder`, `tactical → below`, `strategic → below`.
   * ⚠️ It temporarily makes every parameter differentiable **for the probe only**, because
     X3 is an *architecture* property. A frozen parameter records no autograd edge, so a
     check run mid-S-T would find the encoder "isolated" simply because it is frozen — a
     **vacuous pass**, which is how an isolation guarantee rots.
   * ⚠️ The emission head is **zero-init** (the CV warm start), so `d(controls)/d(input) = 0`
     at step 0. A probe that only looked at the emitted controls would report a **live**
     mis-wire as isolated. The declared surface therefore includes the pre-emission feature.
     `test_isolation_is_not_fooled_by_the_zero_init_emission_head` measures exactly this.
   * The two isolation flags are **independent levers on purpose** — the planner cut and the
     uplink cut are separate rules. If the goal heads read the raw layer latents, disabling
     the uplink alone would silently also open a planner→encoder path, and the two control
     arms would stop being one lever each (the `--v2` conflation failure in miniature).
2. **One vocabulary, two views (§5).** The goal-token table is the **same `nn.Module` object**
   in the emitting head above and the consuming conditioner below. Pinned by `is` identity,
   not equality — two tables that merely start equal are two vocabularies. `named_parameters`
   de-duplicates, so `param_report()` doubles as the sharing check: an accidental copy makes
   the total jump.
3. **§4b seam-free by construction.** ONE 60-step `(a, κ)` @10 Hz sequence, ONE unicycle
   rollout 0→6 s. `0–2 s` and `2–6 s` are **slices** of that rollout (`cfg.split_bands`), and
   `V6Config` **refuses** a band gap or overlap at construction. X2's seam metrics therefore
   *verify*; they have nothing to repair.

### 1.4 The measure primitives (pure, importable, unit-pinned)

| id | function | the correction it encodes |
|---|---|---|
| O2 | `time_to_reach_weights(dist_m, v_ego, tau_s)` | **TIME-scaled, not metre-scaled** (HIERARCHY_VOCABULARY §2, PI correction): *"a fixed 40 m band cannot cover a 6 s horizon (180 m at 30 m/s)"*. Equal time-to-reach ⇒ equal weight; the half-weight **metre** distance is `v·τ·ln2`, linear in speed. Weights are mean-1 normalised so the term **re-allocates** the loss instead of rescaling it (a weighting that also changes gradient magnitude is a learning-rate change in disguise). |
| O2 | `readout_grid_ranges(gh, gw)` | ⚠️ **ESTIMATED — a declared monotone image-row prior, NOT calibrated depth.** PhysicalAI ships no map and no depth channel (the five-probe settled result). What is defensible is monotonicity; a calibrated table drops in later without touching O2, since everything downstream consumes metres. |
| O3 | `sample_cell_block_mask`, `near_field_band_mask` | **CONTIGUOUS** blocks. Scattered per-cell dropout is trivially inpainted from neighbours and teaches nothing about permanence — that is RC3 in the survey. Blocks may overlap (rejection sampling would make the mask rate data-dependent and the loss non-stationary), so the realised rate is reported, never assumed. |
| O4 | `kinematic_saliency`, `saliency_weights`, `InteractionSampler` | from **ACTIONS ONLY** (|jerk|, |decel|, steering reversals) — label-free, = LF1. `floor > 0` keeps free flow reachable: O4 **reweights the draw, never removes a window**, so parity holds and every arm still sees the same 2376 episodes. `alpha=0` reproduces uniform exactly (the attributability control). |
| O5 | `rollout_step_weights`, `o5_rollout_consistency_loss` | error at **every** step. `mode="endpoint"` exists only as the ablation that reproduces the defect O5 fixes: an endpoint-only loss is minimisable by a trajectory that is wrong throughout and right at the end. |
| O6 | `spectrum_report` | participation ratio + effective rank + top-k share on the **centred** covariance spectrum. O6's gate is *retention ≥ 0.8×*, which is a **ratio** — so it needs a SERIES, and the trainer logs it every `--spectrum-every` steps. |

---

## 2. Parameter budget and the E-ENC arm switch

**MEASURED at instantiation** (`V6Stack.param_report()`, defaults as shipped):

| arm | total | encoder | readout | predictor_op | layer_tac | layer_str | planner | aux |
|---|---|---|---|---|---|---|---|---|
| **(a) shared encoder + adapters** (`shared_encoder=True`, default) | **87.89 M** | 15.33 | 0.05 | 60.29 | 5.77 | 4.15 | 0.66 | 1.65 |
| **(b) per-layer encoders** (`--per-layer-encoders`) | **120.74 M** | 45.98 | 0.15 | 60.29 | 6.81 | 5.20 | 0.66 | 1.65 |

Both are inside the **sub-300M INVARIANT**; `build_stack_from_args` refuses to launch otherwise,
**before any GPU time is spent**.

⚠️ **E-ENC decides at MATCHED TOTAL PARAMS** (`V6_TRAINING_MEASURES` §0 Q1), not at matched
per-layer widths. Matching by eye is how an arm wins on **capacity** and gets read as winning on
**architecture** — the same confound class as the C6 "decoder compared on its marginal". So:

```python
from tanitad.models.v6 import V6Config, V6Stack, matched_param_config
from dataclasses import replace
base = V6Config()
target = V6Stack(replace(base, shared_encoder=False)).param_report()["total"]  # 120.74 M
cfg, rep = matched_param_config(base, target)
# MEASURED: chosen predictor d_model 960 -> 118.11 M, gap_frac 0.022
```

⇒ **the matched pair to run is `--per-layer-encoders` (120.74 M) vs shared with
`--pred-dim 960` (118.11 M), a 2.2 % residual gap** — quote the gap, because "matched" with a
30 % gap is not matched. (The probe grid is multiples of `lcm(n_heads, 64)`: a naive
multiples-of-64 grid leaves only 384/768/1152 at `n_heads=12` and reports a bad match as best.)

**Prior from the field:** every frontier system (V-JEPA2, DINO-WM, Drive-JEPA) uses ONE encoder
with downstream consumers. Separate encoders must **earn** their params; a tie goes to (a).

---

## 3. The per-stage runbook — copy-pasteable

⛔ `PYTHONPATH=/workspace/TanitAD/stack` is **REQUIRED** or the trainer dies with
`ModuleNotFound: tanitad`. `cd` alone is not enough.
⛔ `OMP_NUM_THREADS=6` — torch spawns ~113 threads per process; 7 concurrent arms sat at
GPU `sm` **0–6 % for 50 minutes** without it. The trainer sets it defensively, but set it in the
launch line too so it is visible in `ps`.

### 3.0 STEP ZERO on the pod — verify the shipped code, then dry-run

```bash
# 0a. the files arrived by md5-verified FILE-SHIP, never by git (pods have NO git
#     credentials: `git fetch` HANGS, and a `checkout -B` after a failed fetch RESETS
#     the tree to an ancient HEAD and destroys the shipped files).
md5sum /workspace/TanitAD/stack/tanitad/models/v6.py \
       /workspace/TanitAD/stack/scripts/train_v6_staged.py

# 0b. the fix you think is there IS there (grep-verify before every launch)
grep -c "assert_isolation" /workspace/TanitAD/stack/tanitad/models/v6.py

# 0c. it imports and steps — 0 GPU, 0 corpus, ~20 s
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
python3 scripts/train_v6_staged.py --stage S-W --dry-run \
    --out /workspace/experiments/v6-dryrun --dry-batch 1 --dry-steps 2 \
    --dry-k 12 --o5-k 12
# expect: "params 87.89 M / budget 300 M", "X3 isolation pass=True",
#         two [v6 dry N] rows, then "dry-run OK".
```

### 3.1 S-W — the WORLD stage (WM only, λ_plan ≡ 0, planner ABSENT)

```bash
cd /workspace/TanitAD/stack && \
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-W \
  --out /workspace/experiments/v6-SW-30k \
  --v2-cache /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
  --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
  --v2-subframe 176x624 --require-parity \
  --steps 30000 --batch 16 --lr 1e-4 \
  --o1-k 10 --o5-k 20 --o5-mode uniform \
  --o3-mode action --o3-blocks 2 --o3-block-h 2 --o3-block-w 2 \
  --o2-tau-s 2.0 --o4-alpha 1.0 --w-o6 0.1 --spectrum-every 200 \
  > /workspace/experiments/v6-SW-30k/train.out 2>&1 &
```

* **trains:** `encoder`, `readout`, `predictor_op` (+ `step_readout_op`), `aux` (the O3 masked-cell head)
* **frozen:** `layer_tac`, `layer_str`, `planner`
* **losses in force:** O1 (ctrl 1.0 / fact 1.0 / scene 0.3) · O2 1.0 · O3 1.0 · O5 1.0 · O6 0.1
* **λ_plan ≡ 0** — the trainer **refuses to start** S-W with a non-zero `--lambda-plan`.
* ⚠️ **`--o5-k 60` is the §4b horizon as a REPRESENTATION lever (= LF4).** It costs the extra
  future-frame encodes and needs `--max-horizon 60` (see §3.6). `--o5-k 20` is the catalog's
  ≤2 s row; whichever is run, **say which in the run row**.

### 3.2 S-T — tactical layer + the operative planner, on the FROZEN S-W trunk

```bash
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-T \
  --prev-gate  /workspace/experiments/v6-SW-30k/stage_gate.json \
  --init-from  /workspace/experiments/v6-SW-30k/ckpt.pt \
  --max-horizon 60 \
  --out /workspace/experiments/v6-ST-10k \
  --v2-cache ... --v2-val-cache ...  (same corpus args as S-W) \
  --steps 10000 --batch 16 --lr 1e-4 --w-t1 1.0 \
  > /workspace/experiments/v6-ST-10k/train.out 2>&1 &
```

* ⛔ **`--init-from` is REQUIRED** for S-T/S-S/S-J and preflight refuses without it. A gate
  saying "S-W passed" is worthless if this stage then trains on a randomly-initialised trunk —
  that is not the staged protocol, it is four unrelated models with a gate between them. The
  load is `strict=True` (a key mismatch = different geometry) and the run config records the
  **md5 of the loaded trunk**, so the row names exactly which S-W it stands on.
* **trains:** `layer_tac` (adapter, `P_T`, `goal_head_tac`, factored LAT/LON action heads,
  `vocab_tac`, `vocab_a_lat/lon`) and `planner` (`cond_op`, `plan_proj`, `cand_queries`, emission)
* **frozen:** everything below — this is Drive-JEPA's shape (the planner is a **post-trained consumer**)
* **λ_plan defaults to 1.0** at this stage (`STAGE_LAMBDA_PLAN`); the plan loss reports
  `plan_ade_0_2s` and `plan_ade_2_6s` **separately**, because a pooled 0–6 s number cannot see
  the seam.
* ⚠️ the tactical target sits one **tactical** tick ahead (`stride_tac = 5` at 10/2 Hz).
  Predicting one *operative* tick ahead and calling it a tactical prediction is an identity
  map wearing a hierarchy's name; the trainer builds the target from the frame at `t+5`.

### 3.3 S-S — strategic layer on the FROZEN S-T stack

```bash
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-S \
  --prev-gate  /workspace/experiments/v6-ST-10k/stage_gate.json \
  --init-from  /workspace/experiments/v6-ST-10k/ckpt.pt \
  --out /workspace/experiments/v6-SS-8k \
  --v2-cache ... --steps 8000 --batch 16 --lr 1e-4 --w-s1 1.0 \
  > /workspace/experiments/v6-SS-8k/train.out 2>&1 &
```

* **trains:** `layer_str` only (adapter, `P_S`, `goal_head_str`, `act_head_str`, `vocab_str`,
  `vocab_a_str`) · **frozen:** everything else · **λ_plan 0**
* target is one **strategic** tick ahead (`stride_str = 20`).
* ⚠️ **S2 (`g_str` supervision) is not wired here and must not be faked.** It arrives from the
  PH0→PH1→PH2 VLM/geometric pipeline. Until it lands, S-S trains the strategic **latent
  prediction** (S1) only, and the STRATEGIC metric family is reported as `n/a` **with its
  reason and its n** — never silently dropped.

### 3.4 S-J — optional brief joint polish, isolation ON

```bash
PYTHONPATH=/workspace/TanitAD/stack OMP_NUM_THREADS=6 \
nohup python3 scripts/train_v6_staged.py \
  --stage S-J \
  --prev-gate  /workspace/experiments/v6-SS-8k/stage_gate.json \
  --init-from  /workspace/experiments/v6-SS-8k/ckpt.pt \
  --max-horizon 60 \
  --out /workspace/experiments/v6-SJ-3k \
  --v2-cache ... --steps 3000 --batch 16 --lr 3e-5 \
  > /workspace/experiments/v6-SJ-3k/train.out 2>&1 &
```

* **trains:** everything, **isolation still ON** (planner never into any encoder; uplinks still
  stop-grad/EMA). Gate: the frozen battery **FLAT** across the joint phase (the H-COTRAIN rule).
* Run S-J **only if S-T/S-S plateau** (`JEPA_PHYSICS_SURVEY` §5).

### 3.4b ⛔ Resume + the done-marker — the supervisor discipline, in code

`supervise_run.sh` **sources its manifest ONCE, at supervisor startup**, and replays the
`TRAIN_CMD` it captured. Every relaunch therefore runs the SAME flags. Two guards, each from a
measured incident, fire **before** anything expensive:

| guard | the incident it prevents |
|---|---|
| `<out>/summary.json` says `"done": true` ⇒ **REFUSE** (only `--force-rerun` passes) | MEASURED 2026-08-09/11: the v5f run finished but never wrote its done-marker. Its supervisor relaunched for **two days**; when the crash-cause was fixed a relaunch SUCCEEDED, resumed from a stale `ckpt.pt`, and began overwriting `config.json`/`metrics.json`/`ckpt.pt` in the canonical run directory **while burning GPU next to a live eval**. |
| `--resume off` with an existing `ckpt.pt` ⇒ **REFUSE** | a replayed command would restart at **step 0 on top of a live checkpoint** |

`--resume auto` (default) restores stack + optimiser + step and **replays the LR schedule**, so a
relaunch continues rather than restarting. A checkpoint already at or past `--steps` is refused
with an explicit message rather than looping over an empty range.

⭐ **The done-marker is also the correct REMOTE OFF-SWITCH**: `train()` writes
`summary.json {"done": true}` in the same turn the run finishes, and writing it by hand makes a
live supervisor exit cleanly (~20 min) with no kill needed.

⚠️ `step_s` is divided by the steps **this process** ran, not by the resumed step number — a
resumed run would otherwise report an absurdly small per-step time and look 10x faster than it is.

### 3.5 The pre-registered CONTROL arms (never defaults)

| arm | flag | what it isolates |
|---|---|---|
| E-ENC (b) | `--per-layer-encoders` | one visual substrate vs three, at matched total params |
| uplink target | `--uplink ema --ema-decay 0.996` | V-JEPA teacher vs plain stop-grad target |
| O4 off | `--o4-alpha 0` | interaction weighting is a single lever |
| O5 endpoint | `--o5-mode endpoint` | reproduces the compounding defect O5 fixes |
| ⛔ isolation off | `--no-isolate-planner` / `--no-isolate-uplink` **+ `--i-know-this-is-the-control-arm`** | the co-trained path. Preflight refuses without the explicit acknowledgement. |

### 3.6 ⚠️ `--max-horizon` — the finding that would otherwise have cost a pod day

**MEASURED dev-side (2026-08-11):** `_plan(_eval_cfg())` — the horizon plan every existing
flagship trainer inherits — returns **`max_horizon = 20`**, i.e. each window carries **2 s** of
future. A v6 trainer that inherited it would make §4b's 6 s horizon **structurally
untrainable**, and it would fail *looking like a corpus limitation*:

* S-W `--o5-k 60` → refused (needs 60 future latents);
* **S-T at all** → refused, because `λ_plan` needs a 60-step ground-truth plan target. The
  6 s planner has no target inside a 2 s window.

**But `max_horizon` is not a property of the cache.** It is a **windowing parameter**
(`tanitad/data/_contract.py:118-121`: `t_max = frames − window − max_horizon`); the cache
stores whole episodes. So v6 **derives its own** from the stage's actual needs
(`o1_k`, `o5_k`, `stride_tac`, `stride_str`, `plan_steps`), overridable with `--max-horizon`,
and **prints what it chose next to what v4's plan would have said**.

| consequence | detail |
|---|---|
| ✅ parity is untouched | parity is **EPISODE selection** (`physicalai-train-e438721ae894`, 2376 episodes, skip-hash `f09e44db`). Windowing inside an episode is a training-config choice, and O4 reweights rather than removes. |
| ⚠️ the window COUNT drops | a 120-frame episode yields `120 − 6 − 20 = 94` windows at v4's horizon and `120 − 6 − 60 = 54` at 6 s — **≈43 % fewer**. That is a real distribution change vs v5f and belongs in the run row. |
| ⛔ refusals, not silence | `--max-horizon` below what the stage needs, below `maneuver_h`, or yielding **0 windows** all raise with an explicit message. |

⇒ **S-T/S-J launches must add `--max-horizon 60`** (or accept the derived value, which is 60
whenever `λ_plan > 0`). Confirm on the pod from the trainer's own `[v6] windowing:` line.

---

## 4. Gates — what each stage must pass, and what "gated" means

`--gate` behaviour is automatic at the end of every run: `run_stage_gate` writes
`<out>/stage_gate.json`. Launching stage N+1 runs `assert_stage_precondition`, which reads
stage N's gate.

| stage | REQUIRED probes | criteria (pre-registered) | reported-only |
|---|---|---|---|
| **S-W** | P1, P3, P6 | P1 retention ≥ 0.85× R²(z) at k=10 per target · P3 sign ≥ 0.95 **both** channels · P3 gain median ∈ [0.5, 2.0] **without post-training** · P6 action-subspace dims ≤ 32 | P2, P5, P8, O6 spectrum |
| **S-T** | TACTICAL family, `sel_gap` | `sel_gap` ≤ 0.5× the fan oracle at **T1** tier · TACTICAL confusion improves on the E4.1-derived strata | P7 (ρ ≥ 0.3, CI excluding 0, **per stratum**), LATERAL family, X2 seam |
| **S-S** | STRATEGIC family | computable **at all** (measured vs `n/a` today) · S1 ADE(8–30 s) beats CV/corridor baselines at T1 | X2 seam |
| **S-J** | X3 isolation, no-harm | zero live forbidden edges · battery **FLAT** across the joint phase | P1/P3/P6, four families |

### 4.1 Three verdicts, and why the third exists

| verdict | meaning | effect on the next stage |
|---|---|---|
| `pass: true` | every required probe ran and passed | may launch |
| `pass: false` | a required probe **FAILED** | ⛔ **REFUSED, and there is no override.** X5: *a failed stage never propagates upward.* A FAIL is a finding about the layer below; propagating it is how a defect gets attributed to the wrong layer three stages later. |
| `pass: null` | a required probe **did not run** or reported null | ⛔ **INCONCLUSIVE IS NOT A PASS.** Refused unless `--allow-inconclusive-gate` **AND** a non-empty `--gate-off-reason`, which is stamped into `config.json` and printed as a banner. |

An unavailable probe is recorded with **what** could not be reached and **where it lives**
(`STAGE_GATE_SPEC[...]["owners"]`) — rule 2 (*absence at one location is not absence*) applied
to the battery. It is never silently dropped and never counts as a pass.

**X3 is the one gate this module measures on its own, always** — `assert_isolation` runs at the
end of every stage regardless of what else was importable.

### 4.2 Folding in externally-run battery probes

The battery is a **separate, frozen instrument by design** — the trainer does not run it
in-loop (an instrument that moves with the trainer stops being a yardstick). Run it, then:

```bash
# after e.g. scripts/stage_a_probes.py (P3/P6) and scripts/probe_latent_state.py (P1/P2)
cat > /workspace/experiments/v6-SW-30k/probes.json <<'JSON'
{"P1": {"pass": true, "status": "run", "artifact": "p12_gate.json"},
 "P3": {"pass": true, "status": "run", "artifact": "w3_gate.json"},
 "P6": {"pass": true, "status": "run", "artifact": "w3_gate.json"}}
JSON
# re-run the trainer's gate hook with --gate-probes, or pass it on the next launch
```

---

## 5. Cost — ESTIMATED, with its basis, and the first thing to do on the pod

⚠️ **EVIDENCE CLASS: ESTIMATED.** These are extrapolations, not measurements, and the
programme's own history says estimates here run **~11 % low** (`MODEL_REGISTRY` §: v4's
*"~53 h ESTIMATED"* against **MEASURED 59.04 h**, understating the spend by ≈2.5 GPU-days).

**Basis (MEASURED, A40):**
* flagship v1 `flagship4b-speedjerk-30k`: `wallclock_s` **191 206.2** for 30 k steps → **6.4 s/step**
* v4.2: **59.04 h** for 30 k steps → **7.08 s/step**

**Why S-W is more expensive per step than either:**
* **encoder passes/sample:** window 6 + `max(o1_k, o5_k)` = 20 future frames = **26**, vs v1's ~8
  (the future frames are encoded in **one** batched pass, not a Python loop — that optimisation
  is already in the trainer);
* **predictor forwards/sample:** O1 rolls 6 arms × k=10 = **60**, O5 rolls **20** → ~80, vs v1's
  handful. The predictor is 60.3 M params and is the dominant term.

| stage | steps | ESTIMATED s/step (A40) | ESTIMATED A40-hours | note |
|---|---|---|---|---|
| **S-W** | 30 000 | 21–35 | **175–290** (7–12 A40-days) | the whole cost centre; every lever below acts here |
| **S-T** | 10 000 | 6–10 | **17–28** | trunk frozen, no O1/O5 rolls, and **no future-frame encodes** (the trainer skips them when no O-layer term is live) — forward-only through the encoder |
| **S-S** | 8 000 | 5–9 | **11–20** | strategic layer only; one future-frame encode at `stride_str` |
| **S-J** | 3 000 | 21–35 | **18–29** | all terms live again |
| **total** | | | **≈220–370 A40-hours** | |

**One-off startup cost — the O4 pre-pass.** `--o4-alpha > 0` scores every training window
once before step 1. It reads the ACTION arrays straight off the episode providers
(`ep.actions[t : t+W+H]`) and **decodes no frames**: the saliency needs no pixels by
construction, which is what makes O4 label-free in the first place. Going through
`ds_train[i]` instead would decode the entire corpus for a scalar over two channels —
hundreds of thousands of payload loads on MooseFS before the first step. Expect seconds,
not hours; if it is slow, that is a symptom, not the design.

⛔ **THE FIRST ACTION ON THE POD IS TO RE-COST FROM THE RUN'S OWN LOG.** The trainer logs
`step_s` **already divided** (with `step_s_note` naming the divisor) precisely so nobody
re-derives the false "430 s/step" alarm from an accumulated counter. Read it at step 500 and
re-cost before letting 30 k steps run.

**Cost levers, in the order they cost the least science:**

| lever | effect | what it costs |
|---|---|---|
| `--o5-k 10` (from 20) | halves the O5 roll **and** the future-frame encode | O5's compounding shaping is measured over 1 s instead of 2 s |
| `--batch 8 --steps 30000` | fits a smaller GPU | fewer windows/step; **not** a free swap |
| drop the O1 `random` arm | 5 rolls instead of 6 | −17 % of O1; the named channels are the gated ones |
| `--steps 20000` | linear | the ladder is shorter; state it in the run row |
| ⛔ **not** a lever | reducing `--plan-steps` below 60 | that is §4b, and it is **binding** |

---

## 6. What could go wrong — mapped to the CLAUDE.md traps that earned each rule

| # | failure | why it happens here | the defence already built in |
|---|---|---|---|
| 1 | `ModuleNotFound: tanitad` | `cd` alone is not enough on a pod | `--print-launch` emits the `PYTHONPATH=`-correct line; §3.0 |
| 2 | **a launch from a stale pod checkout resurrects a fixed bug** | pods have **NO git credentials**: `git fetch` **HANGS**, and a `checkout -B` after a failed fetch **RESETS the tree to an ancient HEAD and destroys shipped files** (MEASURED 2026-08-11: pod5 HEAD at `6d714ad`, weeks old, while its tree was current) | ⛔ **never put git sync in a pod chain.** Ship files (md5-verified), then §3.0's md5 + grep-verify + `--dry-run` **before** every launch |
| 3 | a supervised run is **RESURRECTED** days after it finished, or a relaunch **restarts at step 0 over a live checkpoint** | the run never wrote its done-marker, so the supervisor kept relaunching; when the crash-cause was fixed a relaunch SUCCEEDED, resumed from a stale `ckpt.pt`, and started overwriting `config.json`/`metrics.json` next to a live eval | the trainer writes `summary.json` with `"done": true` **in the same turn it finishes**, and `resume_guard` **REFUSES to launch** into a DONE directory (only `--force-rerun` passes) or to restart fresh over an existing `ckpt.pt`; §3.4b |
| 4 | editing `runs.d/<run>.env` under a live supervisor changes nothing | `supervise_run.sh` **sources its manifest ONCE, at supervisor startup**, and replays the captured `TRAIN_CMD` | to change a supervised v6 run: edit the manifest → kill the **SUPERVISOR** first → kill the trainer → start a fresh supervisor. **Verify by grepping the flags out of the RUNNING process**, never by reading the manifest |
| 5 | the restarted supervisor races the old one's `flock` and **nothing runs** | the new one prints *"another supervisor holds …lock — exiting"* and dies while the log looks like a normal startup | poll `ps` until BOTH the old supervisor and the trainer are gone, then start. A lock with no holder (scan `/proc/*/fd`) is debris — `rm` it |
| 6 | **two trainers on one pod** ⇒ both crawl, or an eval is contaminated | *"never add GPU/RAM load to a pod that is training, and never eval on a training pod"* | one stage per pod; S-T cannot start before S-W's gate exists anyway (§4) |
| 7 | `pkill -f train_v6_staged` **kills your own ssh session** | the pattern self-matches the ssh command line; it returns empty output and looks like nothing happened | kill by **explicit PID** |
| 8 | a full MooseFS quota kills the run mid-checkpoint | **`df` reports the 965 TB cluster and hides the per-pod quota** | judge disk with a real `dd` write test before launching |
| 9 | "training is 430 s/step!" | `step_s` in the older trainers is **ACCUMULATED over `--log-every`** | this trainer logs `step_s` **already divided** and ships `step_s_note` naming the divisor |
| 10 | 7 concurrent arms at GPU `sm` 0–6 % for 50 minutes, looking exactly like a hang | torch spawns ~113 threads **per process** | `OMP_NUM_THREADS=6` is set defensively in `main()` **and** belongs in the launch line so it is visible in `ps` |
| 11 | an "OOM" that is not one | `memory.usage_in_bytes` counts **reclaimable page cache** (MEASURED 37.2 GB of a 50 GB cap with *nothing running*) | read `memory.stat`'s `rss` and `memory.failcnt` — `failcnt 0` settles it. This cost ~40 min of training and an invented container-OOM diagnosis |
| 12 | **the 6 s horizon looks like a corpus limitation and gets quietly abandoned** | inheriting v4's `plan.max_horizon` (**MEASURED 20**) makes §4b untrainable — S-T could never start at all, because a 6 s planner has no target in a 2 s window | v6 **derives its own** `max_horizon` from the stage's needs and prints it beside v4's; §3.6. Refusals (below-need, below-`maneuver_h`, 0 windows) are explicit — *a silently shortened horizon is not the same experiment* |
| 13a | **stage N+1 silently trains on a random trunk** | `--init-from` omitted; the gate passed, the log looks healthy, and the ladder is four unrelated models | preflight **refuses** S-T/S-S/S-J without `--init-from`; the load is `strict=True` and the config records the loaded trunk's md5 |
| 13 | a stage runs with **no trainable parameters** | the freeze map and the stage disagree | `train()` refuses; `test_stage_freeze_trains_exactly_the_declared_groups` pins the map |
| 14 | a new head escapes the isolation probe | it was added without appending to the declared planner-side surface | `test_planner_surface_is_total` fails when a planner param becomes unreachable from the declaration |
| 15 | a gate is "passed" that never ran | a missing probe read as satisfied | `pass: null` ≠ `pass: true`; the override needs a **stated reason**, printed as a banner and stored in `config.json` |
| 16 | `git add` reports success and stages **nothing** | MEASURED: exit 0, the usual CRLF warning, file **not** staged, in a newly created directory | ⛔ `git add` exit codes are not evidence — verify with `git ls-files --cached <path>` (done for this deliverable, §7) |

---

## 7. Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `stack/tanitad/models/v6.py` | repo working tree, index | **staged**; verified with `git ls-files --cached` + an index-blob-vs-worktree md5 compare (`git add` exit codes are not evidence) |
| `stack/scripts/train_v6_staged.py` | repo working tree, index | **staged**, verified the same way |
| `stack/tests/test_v6_staged.py` | repo working tree, index | **staged**, verified · **80 tests green, CPU-only** (run under `-W error::UserWarning`) (no GPU, no corpus, no checkpoint) |
| `…/incoming/2026-08-07-hierarchical-wm-redesign/V6_TRAINER_DESIGN.md` | this file, repo working tree, index | **staged**, verified |

**This agent committed nothing and pushed nothing** — the `AGENT_OPERATING_STANDARD` contract.

⚠️ **The index was committed by ANOTHER agent mid-task.** Commit `0c30a0f` (a W7-FULL banking
commit) swept an in-progress snapshot of all four files in alongside its own work — the exact
hazard CLAUDE.md records twice (*"`git commit` commits the ENTIRE INDEX, not the files you just
`git add`ed"*; `60265d3` swallowed the eval tooling, `3d41bd0` swallowed REF-C v1.2's rescorer).
**Nothing is lost and nothing is stranded:** the post-commit improvements — the `--max-horizon`
derivation (§3.6), the O4 action-array pre-pass, the S-T/S-S future-encode skip, and the doc
updates — are **staged on top of that commit** (`git diff --cached`: v6.py +5/−3,
train_v6_staged.py +170/−48, V6_TRAINER_DESIGN.md +55/−9; `test_v6_staged.py` is already
current in HEAD). Whoever commits next gets the finished versions.

⚠️ **Foreign staged entries are present in the index** (a concurrent T1-adapter stream:
`stack/scripts/T1_ADAPTER_NOTES.md`, `stack/scripts/t1_v58f_chain.sh`). Anyone committing must
check `git status --short` FIRST and either name them in the message or use a pathspec — and
⛔ **`git commit -- <pathspec>` SEGFAULTS on this repo**, so the admissible route is a
pathspec-free `git commit -F <msgfile>` **after** listing `git diff --cached --name-only` and
confirming every entry is intended.

### 7.1 Escalations for the PI (not "please merge" buried in a doc)

1. **S-W's cost is the decision.** ESTIMATED **175–290 A40-hours** for 30 k steps. Levers are in
   §5; the honest move is to launch, read `step_s` at step 500, and re-cost before committing
   the full ladder.
2. **The 6 s horizon is trainable on today's caches — but it costs ~43 % of the windows.**
   MEASURED: `plan.max_horizon` is **20**, and inheriting it would have made S-T structurally
   impossible. `max_horizon` is a *windowing* parameter, so v6 sets its own (§3.6). Parity
   (episode selection) is untouched; the **window count** drops from `120−6−20 = 94` to
   `120−6−60 = 54` per 120-frame episode. That is a real distribution change vs v5f and needs
   a PI decision: accept it, or rebuild the cache with longer episodes.
3. **W5/E-H1 is a REQUIRED precursor** (§4b promotes it): v5.8f must be baselined at 6 s
   **before** v6 trains against it, or there is no yardstick for the thing v6 changes.
4. **S2 (`g_str` supervision) is not wired** and must not be improvised — it comes from
   PH0→PH1→PH2. Until then the STRATEGIC family is `n/a` **with its reason and n**.
5. **E-ENC pre-registration:** run the matched pair from §2 (`--per-layer-encoders` at 120.74 M
   vs shared `--pred-dim 960` at 118.11 M), decide on **per-layer P-battery pass rate**, tie to
   the common encoder.
