# Pre-registration — does seam projection hold the WM where the scalar floor degraded it?

**Date:** 2026-07-23 (Berlin) · **Stream:** GradCouple (`a4fde3c6`) · **Design:** `DESIGN.md` (same dir).
**Binding:** bars are committed **before** the run (GATE_PROTOCOL discipline); **all three verdicts are
written down here in advance.** This is a *cheap discriminating* experiment, not a gate — it decides
which of three worlds we are in, so no 30k GPU-day is spent blind. Nothing is launched by this document.

---

## 1. The question, and why it is decidable cheaply

The scalar `lam_mult` floor degraded the WM canary at `lr_trunk 1e-4` (v4.2: **0.86@2k / 0.72@4k /
0.77@5k**, `MEASURED`, LOOP_STATE v4.2b stream). **Does one-sided PCGrad at the `states` seam
(`DESIGN.md` §3) hold the canary ≤ 0.55 through the same window, at a *higher* effective planner
coupling than the floor allowed?** The degradation shows up by **step ~4000** (v4.2 was already 0.72
there), so a **~4000-step run settles it** — ≈ 0.66 A40-day per arm, not the 5 A40-days of a full run.

The experiment must also separate the **confound** (RETRACTION_LOG C6; `DESIGN.md` §7.1): part of v4's
degradation was the warm trunk re-optimising under its **own** WM loss (v4 degraded at `λ_plan=0`). So a
`λ_plan=0` control is run **alongside**, and the surgery bar is defined **relative** to it.

---

## 2. Arms (one thing changed vs v4.2; everything else byte-identical)

Fixed for all arms (v4.2 config): warm-start `flagship4b-speedjerk-30k`; `lr_trunk 1e-4`, `lr_head 1e-4`;
phases A[0,2000)/B[2000,8000); micro-batch 16 × accum 4 = **eff-batch 64**; `rollout_k 4`;
`ego_null_row true`; parity `physicalai-train-e438721ae894` / val `physicalai-val-0c5f7dac3b11` (the
CLEAN split — never the 78%-leaked `f1b378f295ae`, LOOP_STATE data-integrity invariant); canary =
operative rollout ADE@2s, 40 val eps, baseline set at step 0.

| arm | coupling operator | planner→trunk gradient | GPU | purpose |
|---|---|---|---|---|
| **S — surgery** | `--coupling seam` (controller floor 1.0, inert) | **full magnitude in the non-conflicting subspace, 0 in the conflicting one** (one-sided PCGrad, per-sample) | ~0.66 A40-day (4k steps) | the arm under test |
| **C₀ — λ=0 control** | `--lambda-plan 0` | **zero** (planner heads train; trunk moves under the WM loss only) | ~0.66 A40-day (4k steps) | isolates trunk-LR re-optimisation (the confound floor) |
| **Cₛ — scalar (reuse)** | v4.2, floor 0.25 | scalar `∈[0.25,1]`, all directions | **0** (reuse MEASURED v4.2 trace) | the baseline that degraded |

Optional stress variant (only if S PASSES and Sayed wants the hardest case): **S′** with `λ_plan` pinned
at **1.0 from step 2000** (no ramp) — full projected coupling from the first joint step.

**Phase 0 — the near-free pre-probe (run this FIRST; eval pod, minutes, ~0 GPU-day).** On the warm v1
trunk, over ~500 val windows, compute `g_wm = ∂L_predict/∂states` and `g_plan = ∂L_plan/∂states` for a
few batches **without updating**, and log the per-sample cosine distribution and `seam_frac_removed`.
This is pod-free to *design* and eval-pod to *run* (no training pod touched). It can **kill the whole
experiment for free**: if the cosine is rarely negative on the warm trunk, the conflict is **not
directional**, surgery cannot help, and we skip straight to §4's trunk-LR / from-scratch branch without
spending the 1.3 A40-days. (`MEASURED`-first, before asserting the mechanism — RETRACTION_LOG C3.)

---

## 3. Primary metric, instrument, and the bars (committed now)

**Primary:** WM canary `canary_ade@2s` at each eval (every 500 steps) over steps [2000, 4000] — the
window where the scalar degraded. Let `c_λ0 = C₀'s canary at step 4000` (the trunk-LR-only floor,
`MEASURED` in-experiment).

**Instrument (logged every `log_every`, decision-grade regardless of the canary):**
`seam_cos_mean`, `seam_cos_min`, `seam_frac_conflict`, `seam_frac_removed_mean` (from
`_SeamProject.last_diag`), plus `gnorm_encoder`, `oracle_in_fan`, `plan_ade`.

### The three pre-committed verdicts

**✅ PASS — surgery is the lever (proceed to a 10k gate run of S).** ALL of:
1. S's canary **≤ 0.55** at every eval in [2000, 4000] (vs v4.2's 0.72–0.86), **and**
2. S's canary − `c_λ0` **≤ +0.05** (the projected planner coupling costs the WM ≤ 0.05 m over the
   no-planner floor — i.e. it is nearly free for the world model), **and**
3. the planner is **not** geometrically starved: `seam_frac_removed_mean` **< 0.70** (a real orthogonal
   subspace survives) **and** S's `oracle_in_fan` / `plan_ade` improve vs C₀ (the coupling is actually
   teaching the planner, unlike v4.1's starved 0.8522).
   *Reading:* a directional conflict with a real non-conflicting subspace; projection resolves what a
   scalar could not. **Decision:** launch S to the 10k gate (one thing changed vs v4.2 = the operator).

**❌ FAIL — surgery is not the lever (do NOT spend a 30k run on it).** ANY of:
1. S's canary **> 0.60** by step 4000 (no better than the scalar floor), **or**
2. `c_λ0` **> 0.55** — the trunk re-optimises past the bar with **zero** planner gradient, so the
   degradation is trunk-LR, not the planner (`DESIGN.md` §7.1), **or**
3. S holds the canary **only** by `seam_frac_removed_mean` **≈ 1.0** (removing essentially all of
   `g_plan`) while the planner does not learn — a fundamentally opposed objective (`DESIGN.md` §7.2), the
   honest v4.1 starvation.
   *Decision by sub-cause:* (2) → lower `lr_trunk` or **from-scratch** (`--from-scratch`, READY,
   `a05a5c9e`); (3) → **from-scratch** (co-evolve; no warm optimum to protect). Surgery is retired with a
   measured reason, not a hunch.

**⚠️ PARTIAL — surgery helps but does not fully hold (report to Sayed; do not auto-escalate).**
`0.55 < S canary ≤ 0.60` with `seam_frac_removed_mean < 0.7`. *Reading:* the direction is right, the
projection is under-protecting. **Options, pre-named:** GradVaccine `--seam-target-cos 0.1` (actively
align the planner toward the WM), or `--coupling seam+floor` with floor 0.5, or a modest `lr_trunk 7e-5`
— each changes one further thing; re-read at a 2000-step follow-up.

---

## 4. What each verdict decides (the fork, drawn in advance)

```
Phase-0 pre-probe cosine on warm trunk
   ├─ rarely negative (frac_conflict low)  → conflict NOT directional → skip to  ▶ trunk-LR / from-scratch
   └─ meaningfully negative                → run S + C₀ (1.3 A40-day)
         ├─ PASS      → seam projection is the coupling; 10k gate run of S; update O-20 lever note
         ├─ PARTIAL   → GradVaccine φ / seam+floor 0.5 / lr_trunk 7e-5 (one more thing) → re-read
         └─ FAIL
              ├─ c_λ0 > 0.55           → trunk-LR re-optimisation → ▶ from-scratch (v1's proven regime)
              └─ frac_removed ≈ 1.0    → fundamentally opposed     → ▶ from-scratch
```

Every branch ends in a **measured** decision. The two expensive fallbacks (from-scratch 30k ≈ 2.2
A40-day; or floor-roulette, already 3× burned) are only taken **after** the 1.3 A40-day experiment says
which one — never fired blind.

---

## 5. Cost, provenance, and safety

- **Cost:** Phase-0 pre-probe ≈ 0 GPU-day (eval pod, minutes). S + C₀ = **~1.3 A40-day** total. Cₛ = 0
  (reuse v4.2's MEASURED trace). **Cheaper than one wrong 30k run** (5 A40-day) or a blind from-scratch
  (2.2 A40-day).
- **Where it runs:** an A40 the fleet frees (NOT pod2 while v4.2b trains; NOT the leaked val split).
  Eval pre-probe on `tanitad-eval`. This design + the reference module are **pod-free**; the runs are
  Sayed's go, launched by the orchestrator (RETRACTION_LOG C1: a launch is not a completion).
- **Staged launch commands (NOT executed):**
  ```
  # Arm S (surgery)
  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/train_flagship_v4.py \
    --train-cache …/physicalai-train-e438721ae894 --val-cache …/physicalai-val-0c5f7dac3b11 \
    --trunk …/flagship4b-speedjerk-30k/ckpt.pt --anchors-dense …/flagship_v4_anchors_dense.pt \
    --out …/flagship-v4.3-surgery-4k --labels v3 --lambda-plan sched --phase-a-steps 2000 \
    --phase-b-steps 8000 --strategic full --steps 4000 --gate-step 8000 --batch 16 --accum 4 \
    --lr-head 1e-4 --lr-trunk 1e-4 --coupling seam --seam-per-sample --eval-every 500 --rollout-k 4
  # Arm C₀ (λ=0 control): identical but  --lambda-plan 0  --coupling scalar
  ```
  (`--coupling`, `--seam-per-sample`, `--seam-target-cos` are the CLI additions from `DESIGN.md` §5; the
  `--gate-step 8000 > steps` keeps the O-17 preflight happy for a sub-gate probe run.)
- **Safety:** `--coupling scalar` remains the default and is byte-identical to today, so merging the
  module cannot perturb any existing arm; the surgery path is opt-in and unit-tested (`test_grad_surgery.py`,
  9 green).

---

## 6. Why both outcomes are genuinely informative

- **PASS** gives the program its first coupling that is neither blunt (scalar) nor expensive
  (from-scratch): a principled, ~free operator that spends only the *conflicting fraction* of the planner
  gradient — and `seam_frac_removed` quantifies that fraction for the record.
- **FAIL** is worth as much: it converts "we mis-tuned the floor three times" into a **measured**
  statement about *why* — trunk-LR re-optimisation (§7.1) or fundamental opposition (§7.2) — which
  **justifies from-scratch on evidence** instead of on exhaustion, and stops any fourth floor-roulette
  attempt. Either way the program stops guessing at the scalar.

**Nothing in this pre-registration was launched, and no training pod was touched.**
