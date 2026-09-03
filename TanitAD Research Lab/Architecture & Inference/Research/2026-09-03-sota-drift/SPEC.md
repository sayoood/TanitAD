<title>SPEC - SOTA pass: latent drift over the rollout (theme 1 of 4)</title>

# SPEC - E-ARCH-SOTA-DRIFT-1: how does the field keep multi-step latent rollouts on-manifold, and does any of it have a MEASURED effect?

**Package** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-sota-drift/` · Research Lab overnight pass 2026-09-03 · literature only · 0 GPU
**Written BEFORE any primary of this pass was opened** (constitution §6.1: spec before evidence). Staged in the same turn it was written.
**Trigger.** Master Mind brief for the v7f design freeze; `V7_LAUNCH_GATE` P3; `LAB_BACKLOG` rows P-6, 4, 9, FS-1 (re-scoped), L-17, L-18.

## 1. Our MEASURED state (registry / register only; anything else is marked INHERITED)

| fact | value | source (class · tier) |
|---|---|---|
| "drift" = the `r` of a k-fold linear probe predicting `Δz = z_{t+k} − z_t` from `z_t` — a **per-window predictability statistic at one fixed k**, NOT a rollout-vs-truth divergence | definition | `mm-e19-probes/latentmotion.py`, E-DEC-59 (`GOALS_AND_CLAIMS.md:254`) · MEASURED · T0 |
| trainable line, 30k, parity | **0.6674** / **0.6741** (seed 1) | `MODEL_REGISTRY.md` §13.9 `postrain30k` / `_seed1` · MEASURED · T0 |
| same recipe at 10k | 0.359 (→ 0.669 at 30k): **training manufactures drift** | E-DEC-61 (`GOALS_AND_CLAIMS.md:142`) · MEASURED · T0 |
| `--init-from` is NOT the lever | two arms share `distill_init.pt`, drift differs by 0.47 | E-DEC-60 (`:194`) · MEASURED · T0 |
| one-variable freeze | **0.3905**, but held-out nrmse 0.9301 vs 0.8115 (**+14.6 %**) ⇒ DEGENERATE | §13.9 `postrain30k_freeze`; E-DEC-64 (`:102`) · MEASURED · T0 |
| `o5_k` 4 on the trainable line | 0.6701 — INERT; helps only under freeze (0.3905 → 0.199 = `splitp30k`) | §13.10 `k4_30k`; E-DEC-66 (`:104`) · MEASURED · T0 |
| O14 future-observation auxiliary | absorbs pixel content; drift 0.6709 UNCHANGED | §13.10 `o14fut30k`; E-DEC-67 · MEASURED · T0 |
| EMA teacher (`--o5-target ema`, τ 0.996 fixed) | drift 0.6952 vs 0.6709 (+3.6 %, EMA-OUT on drift) **but** cos_ctr 0.7524 vs 0.6043 (+24.5 %), nrmse 0.7466 vs 0.8288 (−9.9 %) — the largest prediction gain on the trainable line | §13.10 `emao14_30k` · MEASURED · T0 |
| τ-ramp (cosine, BYOL form) | NEUTRAL at 30k: drift 0.6936, nrmse 0.7408, cos 0.7513 | §13.10 `emao14_30k_tauramp` · MEASURED · T0 |
| deliberate-regression arm | a SHUFFLED innovation constraint reproduces −20.1 % drift (L1's "gain" was −25.0 %) ⇒ **drift is trivially reducible; a symptom, not a target** | §13.10 `e4_ctrl_shuf` · MEASURED · T0 |
| drift is flat across a 15× change in k (4→60): −2.39 % … +0.38 %, constant control 0.0 on 16/16 reads | INHERITED from `2026-09-02-drift-metric-axis-audit/RESULT.md` (not re-verified here) |
| the v7 trainer's rollout is NOT truncated BPTT; k=60 full-chain diverged (gnorm 5.71 → 2.1e9) | INHERITED from `PREREG_MM_E19_K60_HORIZON.md` §3c / readiness report §B.3 |

## 2. Questions (each answered in RESULT.md with class + library key)

- **Q-D1.** Which mechanisms does the field use to keep multi-step latent rollouts on-manifold — EMA/teacher targets, latent regularisers (VICReg / SIGReg / covariance / subspace), rollout-consistency losses, horizon curricula, noise injection, truncation (Looped-WM / InfinityDrive)?
- **Q-D2.** For which of these does a primary publish an **ABLATION with a measured effect on a drift/divergence quantity** (not only on planning success)?
- **Q-D3.** Is the field's "drift" the same quantity as ours? (Ours is `r(Δz | z_t)`; the field's is typically rollout-vs-encoded-future divergence or reconstruction error at step k.)
- **Q-D4.** What is the CHEAPEST discriminating arm on our rigs (v7-tiny ~19 M on the 4060 at bs 1–4, or a probe on banked checkpoints)?

## 3. Hypotheses — both outcomes committed in advance

| id | hypothesis | outcome A (supported) | outcome B (refuted) |
|---|---|---|---|
| **H-SOTA-D1** | At least one banked primary reports an ablation in which a latent regulariser or rollout-consistency term reduces a multi-step rollout divergence metric by a stated amount | the term becomes a v7-tiny arm candidate with the published effect size as its PRIOR; pre-register it against our drift AND a rollout-divergence read (L-17) | no primary measures drift directly — the field reads it through planning success only ⇒ our instrument is ahead of the field, there is **no published prior** for any anti-drift arm, and P3 stays a TanitAD-internal question |
| **H-SOTA-D2** | The field's "drift" is a DIFFERENT quantity from ours (rollout-vs-truth divergence vs. predictability of `Δz` from `z_t`) | rename ours in the register (L-18) and add the rollout-divergence instrument (L-17) BEFORE comparing any number to the field; no field number may be quoted against our 0.669 | the quantities coincide ⇒ published drift numbers are comparable to ours after unit conversion |
| **H-SOTA-D3** | EMA/teacher targets are the field's dominant anti-drift mechanism AND at least one primary shows a measured drift effect from them | our EMA finding (drift +3.6 %, prediction +24.5 %) is a DISAGREEMENT with the field on the drift axis — record it as such and design the experiment that separates the two | EMA is used for collapse-avoidance, not drift; our reading (EMA moves prediction, not drift) AGREES with the field and closes backlog row 4's remaining question |
| **H-SOTA-D4** | Some primary shows that a rollout that drifts can still plan (drift is NOT the capability-limiting quantity) | P3 is deprioritised relative to P1/P2 in the v7f prereg; drift stays a diagnostic | drift is shown to limit planning ⇒ P3 keeps its rank |

## 4. What would change our plan

1. A published, MEASURED anti-drift lever with cost ≤ one v7-tiny arm ⇒ registered as the next arm after the P2 actdiv probe (readiness report §G.2 ordering).
2. Evidence that our "drift" is not the field's drift ⇒ L-17/L-18 move up; no cross-paper comparison until then.
3. Evidence that drift does not limit planning ⇒ P3 is demoted in the v7f prereg; the freeze question stays closed (E-DEC-64).

## 5. Method and admissibility

- Literature only. Every quoted number is read from a PDF banked in `TanitAD Research Lab/Library/` (`tools/kb_add.py`, tag `sota-2026-09-03-drift`); nothing `PUBLISHED-SECONDARY` enters RESULT.md.
- ≥ 3 primaries read IN FULL for this theme; section/table numbers quoted.
- Every field number is stamped `PUBLISHED (PRIMARY, lib:<key>)`; every internal number `MEASURED` with its registry/register line, or `INHERITED` if taken from a summary.
- The field's numbers are used to state what they PREDICT for ours — agreement/disagreement stated explicitly.
- Named empty searches go to `raw/search_log.md`; absence claims need two independent probes.

## 6. Falsifiers

- H-SOTA-D1 outcome B is refuted by a single banked primary with a table row of the form "drift/divergence metric with vs. without term X".
- H-SOTA-D2 outcome A is refuted if a primary defines drift as predictability of the latent increment from the current latent.
