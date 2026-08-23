# v6 architecture — full review, component by component

**PI, 2026-08-12:** *"I want a full review of v6 architecture as we aligned in the
architecture diagram including a detailed report about the size and architecture of the
different model components."*

Evidence class: **MEASURED (ours)** — every number below is counted at instantiation from
`V6Stack(V6Config()).named_children()` in this repo, and the totals are reproduced on
**pod5** by the S-W dry-run gate (2026-08-12): `params 87.89 M / budget 300 M`,
`X3 isolation pass=True`, two real training steps. Nothing here is quoted from a doc.

---

## 0. The shape in one paragraph

Three latent levels over **one** shared visual encoder, each with its **own predictor**,
its **own goal/action vocabulary**, and its **own clock**. Goals flow **down** as
conditioning; latents flow **up** through a severed gradient path. The operative level runs
at **10 Hz** and carries the only attention-over-tokens in the model; tactical runs at
**2 Hz** and strategic at **0.5 Hz**, both as residual MLPs over small latents. A single
**60-step (6 s) unicycle rollout** is emitted from the operative level and *shared* by the
operative (0–2 s) and tactical (2–6 s) bands, which is what makes the 2 s seam
discontinuity-free by construction rather than by repair.

---

## 1. Total, and the E-ENC arms

| arm | total | encoder | readout | predictor_op | layer_tac | layer_str | planner | aux |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **(a) shared encoder + adapters** (default) | **87.89 M** | 15.33 | 0.05 | 60.29 | 5.77 | 4.15 | 0.66 | 1.65 |
| (b) per-layer encoders | 120.74 M | 45.98 | 0.15 | 60.29 | 6.81 | 5.20 | 0.66 | 1.65 |
| (c) v5-width encoder 768×12 | 159.93 M | 87.32 | 0.05 | 60.29 | 5.77 | 4.15 | 0.66 | 1.65 |
| (d) (c) + predictor depth 10 | 193.01 M | 87.32 | 0.05 | 93.37 | 5.77 | 4.15 | 0.66 | 1.65 |

Budget invariant **sub-300 M**; `build_stack_from_args` refuses to launch outside it
**before any GPU time is spent**. See `V6_SIZING.md` for the v5→v6 delta and the
recommendation to add arm (c) to the E-ENC pre-registration.

---

## 2. Component by component

### 2.1 Encoder — 15.327 M · `ViTEncoder`
| | |
|---|---|
| input | **9 channels**, 256 × 640 (the w120 cylindrical geometry) |
| patching | 16 × 16 → 16 × 40 = **640 tokens** |
| width / depth / heads | **384 / 8 / 6** |
| breakdown | `patch` 0.885 · `blocks` 14.196 · `norm` 0.001 |

⚠️ **The single most consequential default in the model.** v5's encoder was 768×12 =
87.16 M; this is **5.7× smaller** and accounts for 41 % of the whole v5→v6 reduction. It
was never *decided* — see `V6_SIZING.md` §4.

### 2.2 Readout — 0.049 M · `SpatialGridReadout`
4 × 4 grid × `d_readout` 128 → **state_dim 2048**. This is the **geometry firewall**: a wide
256 × 640 input still yields a fixed 2048-d state, so downstream widths do not track the
image geometry. Tiny in parameters, load-bearing in interface.

### 2.3 Operative layer — 60.292 M (**69 % of the model**)
| module | params | note |
|---|---:|---|
| `predictor_op` | **58.185** | d_model **768**, depth 6, heads 12, window 6, action_dim 3 |
| ↳ `blocks` | 49.614 | the only attention-over-token-sequence in v6 |
| ↳ `heads` | 4.725 | horizons (1, 2, 4) |
| ↳ `in_proj` / `out_proj` | 1.574 / 1.574 | 2048 ↔ 768 |
| ↳ `act_emb` | 0.594 | action conditioning |
| ↳ `intent_proj` | 0.099 | **the `g_tac` FiLM port** — `P_O(z_op, (a,κ) | g_tac)` |
| `step_readout_op` | 2.107 | per-step decode |
| `cond_op` | 0.003 | goal-vocabulary conditioner |

**This is where the parameters live, and deliberately so:** 10 Hz control over a token
window is the expensive problem; the levels above it are not.

### 2.4 Tactical layer — 5.769 M · 2 Hz, band 2–6 s
| module | params | note |
|---|---:|---|
| `adapter_tac` | 1.313 | 2048 → `d_tac` 512 |
| `predictor_tac` | 3.810 | `FTac` residual MLP, `f_hidden` 512, **3 blocks** — NOT a transformer |
| `goal_head_tac` | 0.237 | emits `g_tac` |
| `act_head_lat` | 0.203 | LAT axis |
| `act_head_lon` | 0.203 | LON axis |
| `cond_tac` | 0.003 | consumes `g_str` |

⭐ **The factored LAT/LON action heads are a fix, not a style choice.** The programme's
single largest known defect is a 5-way softmax that MIXES lateral and longitudinal; two
heads on two axes cannot express that confusion.

### 2.5 Strategic layer — 4.153 M · 0.5 Hz, horizon 8–30 s+
`adapter_str` 0.394 (2048 → `d_str` 256) · `predictor_str` 3.482 (same `FTac` family) ·
`goal_head_str` 0.139 · `act_head_str` 0.137.

### 2.6 Vocabularies — 0.012 M total, and they are SHARED
`vocab_str` · `vocab_tac` · `vocab_a_str` · `vocab_a_lat` · `vocab_a_lon`, ~0.002–0.003 M
each. Each is one table with an `arg_proj` for the continuous envelope args.

⚠️ **Sharing is asserted by IDENTITY, not equality** — the goal-EMITTING head above and the
goal-CONSUMING conditioner below must be the *same* tensor. Two tables that merely start
equal are two vocabularies that drift apart. `named_parameters` de-duplicates, so the param
count itself is the sharing check.

### 2.7 Planner — 0.656 M
`plan_proj` 0.525 · `cand_queries` 0.002 · `emission` 0.130.
Emits **`n_candidates` 8** control sequences of **`plan_steps` 60** at `dt` 0.1 →
**one 6 s unicycle rollout**, bounded by `a_max` 4.0 m/s² and `kappa_max` 0.2 m⁻¹.

⭐ **Why it is 34× smaller than v5's 22.74 M tactical policy:** selection moved out of the
parameters and into the **roll-cost** (the W7 pattern, per level). That is eval-time
compute, not weights.

### 2.8 Aux — 1.650 M
`masked_cells` (the H15-family imagination/masked-cell objective) 1.650 · `sigreg` 0.000
(512 slices, β 1.0, free_dims 0 — a loss, not parameters).

---

## 3. The wiring, and how it is enforced

```
z_str_{t+K} = P_S(z_str_t, a_str_t)                 0.5 Hz
z_tac_{t+k} = P_T(z_tac_t, a_tac_t | g_str)         2 Hz
z_op_{t+j}  = P_O(z_op_t, (a,κ)_t | g_tac)          10 Hz
```

Goals flow **DOWN** only; latents flow **UP** through `uplink=stopgrad` (EMA 0.996
available). Three switches make this checkable rather than aspirational:

| flag | default | what it forbids |
|---|---|---|
| `isolate_planner_from_encoder` | True | planner gradients reaching the encoder |
| `isolate_uplink` | True | a level's gradient reaching the level below |
| `uplink` | `stopgrad` | — |

**X3 isolation probe — MEASURED on pod5 2026-08-12:**
`pass=True  violations={'planner_to_encoder': 0, 'tactical_to_below': 0, 'strategic_to_below': 0}`

The probe works by temporarily making **every** parameter differentiable and asking
`torch.autograd.grad(..., allow_unused=True)` which ones actually received gradient — an
*architecture* property, not a *freeze* property. A frozen parameter records no autograd
edge, which is why the freeze map and the isolation matrix are separate rules.

---

## 4. The 6 s trajectory contract

**One 60-step control sequence (a, κ) @ 10 Hz through ONE unicycle rollout, 0 → 6 s** —
never two stitched trajectories. `op_band_s (0.0, 2.0)` is the dense-control segment;
`tac_band_s (2.0, 6.0)` is the same controls shaped by `g_tac` conditioning. The shared
integrator is what makes the 2 s seam discontinuity-free **by construction**; the X2 seam
metrics verify it, they do not repair it.

**Eval consequence (binding):** four families + oracle/selected at **both** 0–2 s and
0–6 s, T1/P5 compounding to 6 s, and **E-H1/W5** (v5.8f at 6 s) as the required precursor
baseline.

---

## 5. Staged training, and what each stage may touch

| stage | trains | frozen | gate |
|---|---|---|---|
| **S-W** | encoder, readout, `predictor_op`, aux | `layer_tac`, `layer_str`, `planner` | P1 retention ≥ 0.85× R²(z) @ k=10 · P3 sign ≥ 0.95 both channels · P3 gain median ∈ [0.5, 2.0] **without post-training** · P6 action-subspace dims ≤ 32 |
| **S-T** | `layer_tac` (adapter, `P_T`, `goal_head_tac`, LAT/LON heads) + operative planner | trunk | tactical gates |
| **S-S** | `layer_str` only | S-T stack | strategic gates |

⛔ **`pass: false` is REFUSED and there is no override.** X5: a failed stage never
propagates upward — a FAIL is a finding about the layer below, and propagating it is how a
defect gets attributed to the wrong layer three stages later.

---

## 6. Review findings — what I would change, and what I would not

1. ⚠️ **The encoder width is the open question.** 384×8 is a default, not a decision, and it
   is 41 % of the v5→v6 reduction. **Recommend arm (c) as a pre-registered S-W-to-step-500
   pair decided on the P-battery.** (`V6_SIZING.md` §4.)
2. ✅ **The parameter split is right for the diagnosed defects.** 69 % in the operative
   predictor matches where the measured failure is: T1 found ~99 % of the gap longitudinal
   and hold-action beating the closed loop 22×. Those are operative-band control failures.
3. ✅ **Factored LAT/LON heads** directly retire the mixed 5-way softmax.
4. ✅ **Isolation is measured, not asserted** — and it passes on the pod, not just locally.
5. ⚠️ **`plan_steps` 60 is set, but E-H1/W5 has NOT been run**, so the incumbent's 6 s
   number does not exist yet. v6 is currently specified to beat a baseline nobody has
   measured. This is the highest-value unblocked item on the v5.8f side.
6. ⚠️ **`predictor.horizons (1, 2, 4)`** are the *latent-transition* horizons and are
   unchanged from v5's family. Worth confirming they still suit a 6 s emission — flagged,
   not asserted.
