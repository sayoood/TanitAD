# TANITAD VOCABULARY — the single glossary

`Created 2026-08-22 (TANITAD_PROGRAMME.md §5). Append-mostly; renames require a
deprecation line. Every abbreviation used twice gets an entry. Agents use these
terms VERBATIM — no synonyms, no drift.`

## Naming schemas

| kind | schema | example |
|---|---|---|
| experiment | `E-<AREA>-<N>` | E-DETECT-1, E-ENC-3WAY |
| hypothesis | `H-<TOPIC>-<N>` | H-RANK-5 |
| retraction | `C<N>` (append-only log) | C132 |
| decision | `D-<TOPIC or N>` | D-GATE-FLOOR |
| product | `P<1..8>` (TANITAD_PROGRAMME §1) | P5 = TanitScena |
| work package | `<area>/<YYYY-MM-DD>-<slug>/` | `…/2026-08-19-simwam-analysis/` |
| model version | registry key, never prose | `flagship4b-speedjerk-30k` |

## Core terms

| term | meaning | since |
|---|---|---|
| **4B** | the four-brain hierarchy: operative / tactical / strategic / (context) | phase 0 |
| **T0 / T1 / T2** | eval tiers: teacher-forced WM diagnostic / action-closed loop (PRIMARY) / re-perception sim | EVAL_DOCTRINE |
| **parity key** | `physicalai-train-e438721ae894`, 2376 eps, skip-hash `f09e44db` — the canonical corpus identity | phase 0 |
| **EM (explained movement)** | 1 − ‖ẑ−z⁺‖²/‖z−z⁺‖² vs the HOLD baseline | 2026-08-22 |
| **HOLD** | the predict-no-change baseline | 2026-08-22 |
| **collapse (dimensional)** | representation variance concentrated in few directions; measured by PARTICIPATION (σ²), never effective_rank(σ) alone | C132 |
| **participation (ratio)** | (Σλ)²/Σλ² on covariance eigenvalues — THE collapse statistic; floor 8.56 (frozen DINOv3, measured on our frames) | C132 |
| **decodability** | linear/probe recovery of world state (ego speed/yaw/d_ego; detection AP) above the pixel floor AND the constant control | 2026-08-22 |
| **rank AND decodability** | the pass condition for a representation; rank alone admits v1 (C131) | C131 |
| **the pixel floor** | raw pooled frames as a feature baseline; a learned representation below it has added nothing | 2026-08-22 |
| **constant control** | a no-information feature that must read exactly the no-information value; validity gate for every panel | 2026-08-22 |
| **evidence class** | MEASURED / PUBLISHED / INHERITED / ESTIMATED / HYPOTHESIS — on every claim | 2026-07-21 |
| **LeWM recipe** | two-term loss: next-latent prediction + SIGReg(λ=0.1); arXiv 2603.19312 | 2026-08-22 |
| **Sub-JEPA** | SIGReg in K frozen orthogonal subspaces (d_s=D/K); arXiv 2605.09241 | 2026-08-22 |
| **o-terms O1..O6** | v6 loss terms: O1 response-form ctrl · O2 near-field · O3 masked cells · O4 saliency sampling · O5 rollout consistency · O6 SIGReg | v6 |
| **FlyWheel** | a production subprogramme (Data/Training/Deploy/Eval) run by a teammate agent | 2026-08-22 |
| **Research Lab** | the single daily research agent (supersedes "Research Hub" rotation) | 2026-08-22 |
| **Master Mind** | the main orchestrating agent | 2026-08-22 |
| **LAN** | Lane-Anchored Navigation: leak-guarded geometric route corridor label (replaces 4-way nav_cmd) | lan.py |
| **aligned vocabulary** | the tactical/strategic token set of HIERARCHY_VOCABULARY.md — NOT the ego-geometric subset | 2026-08-22 |
| **v7-tiny** | the 29-min validation rig: v6's real trainer at 19M params on the parity corpus | 2026-08-22 |
| **TanitResim** | P9 — replay & visualization pipeline, UI + CLI (owned by EvalFlyWheel; early versions exist) | 2026-08-22 |
| **TanitSpear** | future product — own data generation/rendering/augmentation pipeline (GAIA-class, done cheaper; small-scale proofs first) | 2026-08-22 |
| **TanitSim** | future product — closed-loop environment on our own real data (AlpaSim-class), linked to TanitSpear | 2026-08-22 |
| **transfer handoff** | the mandatory Lab⇄FlyWheel carry of an extremely good result — accept or reject with a reason, never silent (§7 charters) | 2026-08-22 |
| **verify by content** | never trust exit codes, file counts, names, or "success" prints; assert on the bytes | C77/C79 |
| **gate participation (train-pooled)** | the O6 gate pools rows from the O4-weighted TRAIN stream — cross-arm comparable, but inflated vs a val-side read (~5.5 vs ~3.4 on the same model); quote val-side participation for representation claims | H-RANK-9 |

## Deprecated

| term | replaced by | when |
|---|---|---|
| "Research Hub" (agent rotation) | TanitAD Research Lab | 2026-08-22 |
| effective_rank as the collapse gate | participation ratio (σ²) | C132 |
| "the 0.452 m driving result" | `wm_fidelity_ade_2s` (T0 WM fidelity; closed-loop is 1.7318) | C131 |
