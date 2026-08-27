# v7r — design proposal (PI-directed, 2026-08-27)

**Author** Master Mind · **Status** PROPOSAL for PI review — nothing here launches
without its gate · **PI directive (verbatim intent):** design v7r using the survey's
main findings combined with our own; **keep the multi-hierarchy design and the
diffusion-planner design**; carry **the whole v1.7 decoding learnings**, the
**optimized action space from Alpamayo**, and the **module optimizations**.

Every design input below is MEASURED (ours) or PUBLISHED (banked primary); the
combination is the proposal.

---

## 0. Design inputs, one line each

| input | source |
|---|---|
| Encoder TRAINABLE; freezing trades drift for miscalibration | D1 / E-DEC-64 (MEASURED) |
| Predictor at the ceiling of the current latent; missing content is pixel-borne, non-photometric, displaced by token structure | E-DEC-63 + F1–F4 (MEASURED) |
| Scene→action, not action→scene, in observational data; the planner owns the reaction | E-DEC-48b (MEASURED) |
| Conditioning = `[ω, a_long, v]` measured state | PI directive; D2 arm (read pending) |
| JEPA WM + light planner reaches NAVSIM 93.3 PDMS at 307 M / 208 h | Drive-JEPA (PUBLISHED, banked) |
| Distilled-foundation latent + causal predictor, 89.3 EPDMS perception-free | Latent-WAM (PUBLISHED, banked) |
| Latent-subgoal hierarchy: 70 % vs 0 % flat, 3× less planning compute | arXiv 2604.03208 (PUBLISHED, banked) |
| Occupancy works as a JEPA WM target | arXiv 2602.12540 (PUBLISHED, banked) |
| v1.7 decoding: speed-L1 helps globally (ADE CI-better, jerk≈human, reliance 1.18) but NOT near-term response; next lever = event-weighted near-term accel matching | registry §1.11 (MEASURED) |
| Open-loop lateral skill can be an action echo; hold-action control is mandatory | registry §1.12 (MEASURED, T1) |
| REF-C's selection flaw: ranks with the UN-refined anchor's score | registry §4 (MEASURED) |
| Alpamayo-2 augmentation: 4,729 clips × {trajectory, meta_action (CoT 100 %), auto_labeling, vqa, grounding} | registry §Alpamayo (MEASURED asset) |

---

## 1. The architecture — four brains kept, interfaces now specified

```
                    ┌────────────────────────────────────────────┐
   STRATEGIC        │ route/goal over map-derived option sets    │  slow tick
   (kept)           │ emits: LATENT SUBGOAL g_str                │
                    └───────────────┬────────────────────────────┘
                                    │ latent-subgoal matching (2604.03208)
                    ┌───────────────▼────────────────────────────┐
   TACTICAL         │ meta-action head (ALPAMAYO vocabulary) +   │  mid tick
   (kept, re-armed) │ emits: LATENT SUBGOAL g_tac                │
                    └───────────────┬────────────────────────────┘
                                    │ latent-subgoal matching
                    ┌───────────────▼────────────────────────────┐
   OPERATIVE WM     │ trainable ViT encoder (distill-init)       │  fast tick
   (revised)        │ + latent predictor, cond = [ω, a_long, v]  │
                    │ + O14-fut (R2, approved)                   │
                    │ + occupancy/free-space target (new)        │
                    └───────────────┬────────────────────────────┘
                                    │ imagination surface
                    ┌───────────────▼────────────────────────────┐
   PLANNER+DECODER  │ anchored-DIFFUSION planner (kept) over a   │
   (kept, fixed)    │ TRAJECTORY VOCABULARY; v1.7 decoder recipe │
                    └────────────────────────────────────────────┘
```

### 1.1 Operative world model (the revision core)

- **Encoder:** trainable (D1), distill-init. **Pretraining bake-off** (P0, tiny
  ladder): our two-term core **vs** the Drive-JEPA-style **EMA-teacher masked-latent
  target** — judged by the E-DEC-63 absorption metric and the standard gates.
  ⚠️ The EMA teacher is still model-generated (E-DEC-7's class); it enters as a
  gated candidate, never as an assumption.
- **Objectives:** two-term core + **O14-fut (R2, approved — PREREG_O14)** +
  ⭐ **an occupancy/free-space prediction target** — the E-DEC-63 "coarser
  structured target", with published JEPA precedent (2602.12540). Source of
  occupancy labels: `obstacle.offline` (Stage B staging) — labels may use
  privileged data; inference stays vision-only.
- **Conditioning:** `--cond-param omega_accel_v` if D2 reads CHANNEL-MATTERS or
  neutral-with-no-side-effect; the prereg's four outcomes decide, not taste.
- ⛔ **What the WM does NOT do (E-DEC-48b):** encode the reaction. *"If the lead
  brakes, ego must react"* lives in the planner.

### 1.2 Tactical level — the Alpamayo action space

- **Action space:** the **meta-action vocabulary** of the Alpamayo-2 augmentation
  (4,729 clips, `meta_action` with CoT at 100 % coverage; `trajectory` task as the
  dense counterpart). This replaces the 5-way softmax whose LAT/LON mixing is our
  single largest known decoder defect (the longitudinal-blindness root cause).
  ⚠️ C136/C138 bind: every agreement number against these labels ships with a
  chance baseline; no threshold is calibrated on a witness axis measured at chance.
- **Output:** meta-action distribution **and a latent subgoal `g_tac`** for the
  planner — so the tactical decision is scoreable (four-families TACTICAL) *and*
  drives planning through one typed interface.

### 1.3 Strategic level

Map-derived option sets (AlpaSim `map.xodr` / external corpus — Stage B). Emits
`g_str` as a latent subgoal constraining tactical. Until B2 data exists this level
trains last; **the interface is specified now so nothing downstream has to change.**

### 1.4 The inter-level interface — latent-subgoal matching (new, published-backed)

Upper level emits a latent subgoal; the lower level plans/predicts **to match it in
the shared latent space** (2604.03208's mechanism: 70 % vs 0 % flat, 3× cheaper
planning). Properties we require and it satisfies: **reward-free**,
**information-disjoint** (the goal path carries no situation-classifier output —
2026-08-03 rule), and **evaluable per level** (subgoal-match error is a per-level
metric ⇒ feeds the hierarchy-traversing eval Stage D needs).

### 1.5 Planner — the diffusion design kept, with two measured fixes

- **Kept:** anchored-diffusion proposal generation (REF-C lineage; the v6 stack
  already carries it: 4 diffusion steps, 8 candidates, query proposals).
- **Fix 1 (measured defect):** rank candidates by the **REFINED** trajectory's
  score — REF-C ranks with the un-refined anchor's score (registry §4 flaw).
- **Fix 2 (adopted from Drive-JEPA):** ⭐ **trajectory-vocabulary distillation** —
  k-means vocabulary over our corpus trajectories; candidates scored by
  rule/simulator (AlpaSim) and EPDMS-style filtering; pseudo-trajectories join
  human ones as supervision. Multi-modal supervision without new labels.
- **Selector:** trained on the WM's imagination surface (sel_gap becomes
  computable — the T1 tool named the missing fan as a work item).

### 1.6 Decoder — the v1.7 learnings, carried whole

| carried | evidence |
|---|---|
| **speed-L1 aux** (w 0.5) | v1.7: ADE **CI-better** −0.0549, jerk 1.567 ≈ human 1.71 |
| **event-weighted near-term accel-matching term** | v1.7's pre-registered NEXT lever (P1/P2 failed on response/lag; weight tuning explicitly ruled out) |
| **reliance gate N4 ≥ 0.5** | v1.7 measured 1.18 — the decoder must *need* the WM |
| **anti-echo discipline** | §1.12: hold-action control mandatory at T1; no open-loop lateral claim without it |
| decode from the WM state, never from an echoed action channel | E-DEC-57 + §1.12 |

### 1.7 Module optimization (kept as constraints, not revisited)

Sub-300M total with per-group budgets and **X3 isolation** (planner cannot reach
encoder gradients, tactical/strategic cannot reach below — already enforced and
logged every run); the perflib fused-kernel/bf16 learnings and the Thor batch-8
saturation fact carry into deployment; `cuda_max_mem_gb` logging stays; eval and
training never share a box.

---

## 2. Training schema — four phases, each gated

| phase | trains | gate to pass (instrumented today) |
|---|---|---|
| **P0** encoder | two-term vs EMA-teacher bake-off (tiny ladder) | E-DEC-63 absorption + G-RANK + G-DECODE |
| **P1** operative WM | O14-fut + occupancy + conditioning per D2 | L1–L3 (esp. **L3: `zhat` beats `z_t`**, the dissociation gate — no arm has ever passed it) |
| **P2** tactical + strategic | Alpamayo meta-actions; map option sets (B2) | four-families TACTICAL/STRATEGIC vs chance baselines |
| **P3** planner + decoder | vocabulary distillation + v1.7 recipe | **L4: T1, S-rate > hold-action, ADE < CV floor; v1.7 as the harness's deliberate-regression arm** |
| **P4** scale + dominance | the full v7r at scale | **L5 vs REF-C, REF-D, frozen-DINOv3/V-JEPA-2.1 fallback WM** |

**Parallelism:** P0/P1 now; P2's data (Stage B FlyWheel) in parallel; P3 needs P1;
P4 needs all. This is the PI's staged plan with the gates attached.

---

## 3. Open decisions for the PI

1. **Fallback teacher:** DINOv3 (current) vs **V-JEPA 2.1** dense features for the
   distill/fallback arm — one extra probe column decides; default: measure both.
2. **NAVSIM as external yardstick** for Stage D (the surveyed field proves itself
   there) — provisioning call, not scheduled.
3. **P0 bake-off budget:** default 4 tiny arms (~2 h Thor total). Approve or trim.
4. Whether "optimized action space from Alpamayo" means MORE than the meta-action
   vocabulary + trajectory parameterisation adopted in §1.2 — **if there is a
   specific action-space document I have not found, point me to it and §1.2 will be
   revised against it.**
