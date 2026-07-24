# MODEL_REGISTRY.md — v4 line addition note

**Date:** 2026-07-23 (Europe/Berlin, UTC+2) · **Type:** doc-synthesis (pod-free; no GPU, no training pod touched).
**What this records:** every number added to `Project Steering/MODEL_REGISTRY.md` **§1.5 (new) flagship-v4 line**,
its evidence class, and the raw source artifact it was read from. No number below was quoted from `LOOP_STATE.md`
prose; where LOOP_STATE and a raw JSON disagreed, the JSON is used and the discrepancy is flagged in the registry.

**Placement:** the v4 lineage was inserted as **§1.5**, mirroring §1's existing per-version schema (shared preamble
+ config/param block, then per-version subsections). The pre-existing "Flagship variants that are not versions" block
was renumbered **§1.5 → §1.6** (verified: no `§1.5` cross-reference exists anywhere in the registry — only §1.4b is
referenced, at the §8 decision log).

**Evidence-class key (CLAUDE.md / operating standard):** MEASURED (ours + artifact path) · PUBLISHED · INHERITED
(another doc/agent, not re-verified) · ESTIMATED · HYPOTHESIS · PENDING (in-flight, no number).

---

## A. Architecture / parameters (§1.5 preamble)

| number added | value | class | source artifact |
|---|---|---|---|
| v4 trainable total | **247,878,786** (≈247.9 M; 62 % of 400 M cap; ~30 M < v1) | **MEASURED** (local instantiation, not a run-side print) | `V4_FLAGSHIP_DESIGN.md` §3.1 (`scratchpad/v4_param_budget.py`); G0-preflight faithfulness check §17 |
| faithfulness check | `WorldModel(flagship4b_config())` = **263,440,533** = §1.1 | MEASURED | same, §3.1 |
| module breakdown | encoder+readout 87,121,280 · operative pred 96,609,283 · H15 22,055,683 · strategic planner 5,152,911 · direct-head baselines 3,149,824 · tactical `FlagshipV15Head` 9,767,320 · operative dense head 9,778,604 · factorised LAT/LON/DIST ≤811,543 · grounding 13,432,338 · removals −22,736,141/−26,534,912/−8,385,027/−528,897 | MEASURED (instantiation) | `V4_FLAGSHIP_DESIGN.md` §3.1 |
| v1 comparison | v1 trainable **277,404,073** | MEASURED | `MODEL_REGISTRY.md` §1.2 |
| **not-frozen proof #1** (run-side) | `not_frozen:true`, encoder `149/149` req-grad, predictor `159/159`, `trunk_tensors_frozen 0`, `trunk_group_lr 1e-4` | **MEASURED** (run config) | `taniteval/results/trainlogs/flagship-v4.2-step4000_config.json` → `not_frozen_proof` |
| **not-frozen proof #2** (run-side) | trainer banner `warm-started trunk+grounding … (TRAINABLE)` | **MEASURED** | `…/incoming/2026-07-23-v41-10k-gate/v4.1_train.log:1` |
| head config | `cond_imagination=false`, `cond_vtarget/route=true`, `factorised=true`, `n_anchors=256`, dense 1..20 horizons, decoder d384×4L | MEASURED (run config) | `flagship-v4.1-10k_config.json` → `head_cfg` |

> ⚠️ The 247,878,786 total is **not** printed in any run `config.json`; its authority is §3.1 instantiation + the
> G0-preflight check. Marked MEASURED-by-instantiation in the registry, with an explicit caveat not to quote it bare.
> This is the honest class — it is a measurement record, not inherited prose, but also not a run artifact.

## B. Root-cause through-line (§1.5 preamble) — HYPOTHESIS + MEASURED support

| claim | class | source |
|---|---|---|
| WM degradation is a **warm-start artifact** (not intrinsic) | **HYPOTHESIS** | `V4_FROMSCRATCH_LAUNCH.md` §0; tied to the pending v4.2b Phase-B test |
| v4 hot-trunk canary 0.452 → ~1.3 by ~step 3500 | MEASURED (in-loop trainer canary, NOT held-out) | `V4_FROMSCRATCH_LAUNCH.md` §0 (from MODEL_REGISTRY/[PM]) |
| v1 co-evolved WM+planner from scratch, canary held 0.42 | MEASURED | `MODEL_REGISTRY.md` §1.2 |
| from-scratch smoke: canary 1.52 → 1.165, WM co-evolves, no collapse | **MEASURED** (dev-box `--smoke-loop`) | `V4_FROMSCRATCH_LAUNCH.md` §3.1 |

## C. §1.5.1 flagship-v4 (original, hot trunk) — KILLED

| number | value | class | source |
|---|---|---|---|
| step-0 canary baseline | 0.42148 | MEASURED (in-loop) | LOOP_STATE S2 (in-loop trainer) — **no held-out eval exists for v4-original** |
| WM canary runaway / WM loss 2.3→4.24 | in-loop | MEASURED (in-loop trainer canary; kill trigger only, C1) | `V4_FROMSCRATCH_LAUNCH.md` §0; LOOP_STATE |
| distinguishing lever | `--lr-trunk 3e-4` (hot), eff batch 16 | MEASURED | LOOP_STATE launch command |

## D. §1.5.2 flagship-v4.1 @ 10 k — the FIRST decision-grade held-out v4 numbers

**Primary source: `taniteval/results/flagship-v4.1-10k.json`** (`eval_flagship_v4.py` MODE B, gate stream `a938e1c0`;
duplicate staged copy `…/incoming/2026-07-23-v4-eval-harness/flagship-v4.1-10k.json`). All **MEASURED**.

| number | value | source field |
|---|---|---|
| **`ade_0_2s`** (gate primary) | **0.8522 [0.7468, 0.9800]** (full-set 0.8521944) | `cluster_bootstrap.model.ade_0_2s` / `full_set.model` |
| ADE@0.5/1/1.5s | 0.2376 / 0.4075 / 0.6304 | `cluster_bootstrap.model` |
| FDE@2s | 1.5176 [1.2563, 1.8213] | `cluster_bootstrap.model.fde@2s` |
| **`miss_at_2m`** | **0.2486 [0.1714, 0.3379]** | `driving.headline.miss_2m` / `miss_rate@2m` |
| **`oracle_in_fan`** (4wp) | **0.4838** (dense-20 oracle 0.3603) | `v4_diagnostics.wp4_oracle_ade_0_2s` / `dense_headhorizons_oracle_ade` |
| **`wm_canary_ade_2s`** | **0.4599** (PASS; v1 base 0.452) | `wm_canary_ade_2s` |
| `seam_norm_ratio_max` | 0.1796 (PASS) | `v4_diagnostics.seam_norm_ratio_max` |
| ADE vs CV (paired) | Δ −0.0145 [−0.1508, +0.1448], not separated | `driving.verdict.ade_vs_cv` |
| speed-MAE vs CV / vs hold-v0 | Δ −0.3662 [−0.4908,−0.2446] / −0.3523 — separated, favours floor | `driving.verdict.speed_mae_vs_cv/holdv0` |
| steady-cruise speed vs hold-v0 (n=639) | Δ −0.5593 [−0.6482,−0.4689] — separated, favours floor | `driving.verdict.cruise_speed_vs_holdv0` |
| path-geometry vs CV | Δ +0.1145 [+0.0171,+0.24] — separated, favours **model** | `driving.verdict.path_geometry_vs_cv` |
| straight heading model/CV | 8.25° / 1.399° | `driving.verdict.straight_heading_model_vs_cv_deg` |
| CV / hold-v0 floors | CV ade_0_2s 0.8377, hold-v0 0.7876 | `full_set.cv` / `driving.floor_values.holdv0` |

**Gate verdict — `Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json`** (MEASURED, `run_gate.py check`):

| field | value | class |
|---|---|---|
| formal machine verdict | **`INCOMPLETE`** (3/8 KILL secondaries have no emitter) | MEASURED |
| primary `ade_0_2s` ≤ 0.60 | **pass:false** (0.8522) | MEASURED |
| KILL secondaries | PASS `wm_canary`/`seam`/`levers`; **FAIL** `oracle_in_fan`/`miss_at_2m` | MEASURED |
| restart budget | 0/2 for family `joint-planner-wm` | MEASURED |

- **Discrepancy flagged in registry:** LOOP_STATE calls this "FAIL"; raw gate JSON verdict is **`INCOMPLETE`** with
  a decisive primary `pass:false`. Registry presents both (formal INCOMPLETE, substantive FAIL). Class: MEASURED.
- Harness validation (MODE A on v1): **0.42148** vs registry full-set 0.4271, `HARNESS_VALIDATED:true`. MEASURED —
  `…/incoming/2026-07-23-v4-eval-harness/v1_validation_proof.json` / `taniteval/results/v1-validation.json`.
- ckpt: `pod2:/workspace/experiments/flagship-v4.1-30k/ckpt_step10000.pt`, 3,243,109,310 B, md5 `8ae1ca68…`; eval copy
  `tanitad-eval:/root/models/flagship-v4.1-10k/`. MEASURED — `…/2026-07-23-v4-eval-harness/STATUS.md` §5.
- ⚠️ in-loop `train.log` numbers (val.ade 0.7054 / oracle 0.3598) are **dense-20-mean, a different convention** —
  registry explicitly forbids quoting them as the gate primary (C1). Class of that caveat: MEASURED (the metric
  definition, `flagship-v4.1-10k_train_log.jsonl`).

## E. §1.5.3 flagship-v4.2 interim @ step 4000

**Primary source: `taniteval/results/flagship-v4.2-step4000.json`** (MODE B). All **MEASURED**.

| number | value | source field |
|---|---|---|
| **`ade_0_2s`** | **0.9869 [0.8795, 1.1088]** (full-set 0.9869228) | `cluster_bootstrap.model.ade_0_2s` / `full_set` |
| **`wm_canary_ade_2s`** | **0.7222** (breaches KILL ≤0.55) | `wm_canary_ade_2s` (0.7222376) |
| miss@2m | 0.2940 [0.2216, 0.3716] | `driving.headline.miss_2m` |
| oracle (4wp) | 0.5009 | `v4_diagnostics.wp4_oracle_ade_0_2s` |
| controller / lr / batch | cap-and-hold floor **0.25**, lr_trunk **1e-4**, eff batch **64** (16×4) | MEASURED — `flagship-v4.2-step4000_config.json` (`canary_controller`, `optimizer`, `args`) |
| in-loop canary trend 0.86@2k / 0.72@4k / 0.77@5k | — | **INHERITED** (in-loop, LOOP_STATE); the 0.72@4k is corroborated MEASURED by the JSON canary above |
| eval-copy relay md5 | `c42ae39c…` | MEASURED — `STATUS.md` §5b |

## F. §1.5.4 flagship-v4.2b (floor 0.15) — PENDING

| item | value | class |
|---|---|---|
| held-out result | **PENDING — no number; not fabricated** | PENDING |
| status | LIVE pod2 (streams-table PID 99197), ~step 900 Phase A, in-loop canary 0.495 | INHERITED (LOOP_STATE, in-loop) |
| distinguishing lever | `--lam-mult-floor 0.15`, else = v4.2 | MEASURED (design/launch flag) |
| pre-registered tell | Phase-B canary @ 2500–3000: ≤0.55&<v4.2 → PASS; ≥0.65 → FAIL→from-scratch; 0.55–0.65 → floor 0.10/pivot | PUBLISHED (pre-registration, LOOP_STATE) |

## G. §1.5.5 from-scratch fallback — READY, not launched

| item | value | class | source |
|---|---|---|---|
| validation | pytest **786/2**; 14 `test_train_flagship_v4.py`; smoke-loop canary 1.5189→1.165 | MEASURED | `V4_FROMSCRATCH_LAUNCH.md` §3.1 |
| cost | ~53 h / 30 k | **ESTIMATED** (MEASURED basis: v4.1 ~1.57 s/step ×4 accum) | `V4_FROMSCRATCH_LAUNCH.md` §6 |
| lever | `--from-scratch` (skip warm-start), else = v4.2b | MEASURED | `V4_FROMSCRATCH_LAUNCH.md` §1–2 |

## H. Reconstruction / gate-completeness risk (added to registry)

3 of 8 KILL secondaries have **no emitter** (`speed_benefit_recovered_frac`, `deploy_tick_p99_ms`,
`nonav_route_beats_majority`) → no v4 gate can render a *complete* formal verdict today. `g_op_fwd_ade_m` never
reaches `train_log.jsonl` (whitelist gap) → the comparative matched-step diagnostic is dead for v4. Checkpoints are
on single pod disks, not HF-backed. Class: MEASURED — `…/incoming/2026-07-23-v41-10k-gate/STATUS_BLOCKED.md`,
`…/2026-07-23-v4-eval-harness/STATUS.md` §3–4.

---

## Deliverable manifest

| artifact | where | status |
|---|---|---|
| edited registry (new §1.5 v4 line; old variants → §1.6) | `repo:Project Steering/MODEL_REGISTRY.md` | **STAGED** (git add; NOT committed, NOT pushed) |
| this addition note | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-07-23-registry-v4-section/REGISTRY_V4_ADDITION_NOTE.md` | **STAGED** |

**Primary sources consumed (all already in-repo, read-only):** `taniteval/results/flagship-v4.1-10k.json`,
`taniteval/results/flagship-v4.2-step4000.json`, `Project Steering/Gates/flagship-v4-gate-10k-2026-07-23.json`,
`Project Steering/Gates/flagship-v4.card.json`, `taniteval/results/trainlogs/flagship-v4.{1-10k,2-step4000}_config.json`,
`…/incoming/2026-07-23-v41-10k-gate/{v4.1_train.log,v41_step10000_inloop_health.json,STATUS_BLOCKED.md}`,
`…/incoming/2026-07-23-v4-eval-harness/STATUS.md`, `…/incoming/2026-07-23-v4-fromscratch/V4_FROMSCRATCH_LAUNCH.md`,
`TanitAD Research Hub/Architecture & Inference/V4_FLAGSHIP_DESIGN.md` §3.1/§17.

**Escalations (surfaced, not acted on):**
- The v4 line does **not** touch `Project Steering/Mission Plan.md` (agents never edit it).
- The v4.1 ckpt (3.24 GB) and both v4.2/v4.2b ckpts live on **single pod disks**, not HF-backed — a standing loss risk.
- 3 KILL secondaries have no emitter → whoever runs the next v4 gate must build them or accept a 5/8 adjudication
  (Sayed's call, per `STATUS.md` §6).
