# Hub Deep-Screen Digest — 2026-07-08 (D-020 §1, orchestrator)

Deep screen of ALL hub agent outputs (6 disciplines, 6 intake packages, bench/regulation docs) by a
dedicated screening agent; this file is the derived-usable-findings record. Full agent output
archived in session transcript. Items marked ➜MVP are pulled into the main stream; ➜BACKLOG items
were seeded into the per-discipline `BACKLOG.md` files (D-020 §4).

## 1. Integration verification
All 6 triaged packages confirmed LIVE in `stack/` HEAD (gate runner w/ P4+I7+imag-rel-demotion,
spectral, cosmos loader, metadrive front-cam, work-zone scenario, metric suite). 139 tests passing.
No re-triage needed.

## 2. Actionable, not yet in stack/ — disposition
| Item | Evidence | Disposition |
|---|---|---|
| K-step rollout loss (K=2) bake-off | arXiv 2512.24497 multistep-as-augmentation | ➜BACKLOG Architecture P0.2 (behind 30k checkpoint; matched-compute arms; falsifier: no D2/D3 gain) |
| RoPE in FiLM/AdaLN conditioning | 2512.24497 "AdaLN+RoPE best" | ➜BACKLOG Architecture P1.3 (smoke first) |
| σ-gated tactical MoE | uncertainty routing vs DriveMoE black-box | ➜BACKLOG Architecture P2.6 (Phase 1, WP4) |
| ViT INT8 via OwLite/ModelOpt (not native TRT) | known ViT+MHA TRT trap | ➜BACKLOG Production P1.6 |
| Spectral-sizing on real trained latents | validates D-008 dim 2048; knee method | ➜MVP: run at every checkpoint eval (already in evaluate_checkpoint); full study ➜BACKLOG Architecture P0.1. Feeds proposed **D-021** |

## 3. Research findings → design state
- **L1 (2606.27014):** latent-dim trade-off formalized; `max_a` term vindicates D-010 off-expert sim
  data. → proposed **D-021**: latent dim k is a *measured* design variable (spectral knee), not a
  hyperparameter; 2048 readout is a candidate pending validation. Escalated to Sayed (PROJECT_STATE §4).
- **L2 (2605.00066):** open-loop ⊥ closed-loop — already embedded in D-017/leaderboard hygiene.
  Standing rule added: never rank a checkpoint on ADE alone.
- **L3 (HiT-JEPA):** external evidence hierarchy⇒generalization (D6 expectation); H3-hex place codes
  ➜BACKLOG (strategic keys, Phase 1).
- **L4 (2512.24497):** decode ≠ planning — motivates BLOCKED-on-instruments doctrine; DINO>V-JEPA
  caveat strengthens H4 arm-B (honest risk).
- **L5 (ecosystem):** all opponents scale-first; nobody at our Pareto point → every public comparison
  leads with CNCE. Production stream owns the denominator (latency baseline).

## 4. Data opportunities → DataEng BACKLOG
Zenseact ZOD pilot (P1.3, real-CAN #2, EU/night); PhysicalAI-WorldModel-Synthetic-Scenarios license
check (P1.4 — H6/H15/D9 long-tail); `data:physicalai` tag audit (P1.5); monthly HF sweep (P2.8);
nuScenes reserved as D8 OOD probe (never trained).

## 5. Blockers surfaced (➜MVP §4 escalations / monitors)
1. **Supervised MetaDrive install** (~10 min, Sayed) — unblocks D-010 sim-mix live rollout + D5/D6 +
   D9 occluder scenarios. Escalated.
2. CARLA-on-pod harness (W31–32, planned) — gate for D4–D6 + Bench2Drive entry.
3. Cosmos T=39 temporal semantics — before cosmos enters training mix (DataEng P0.1).
4. PhysicalAI-AV license audit before any Phase-1 publication (DataEng P1.5).

## 6. Quality observations → protocol fixes (all applied 2026-07-08)
- Scenario→telemetry→metric wiring only oracle-deep → full dry-run duty added (Benchmarks BACKLOG P1.4).
- Agent runs were research-heavy, experiment-light → G-H gate (≥1 measured experiment/run) now in
  `_common-protocol.md`; per-discipline BACKLOG.md seeded; burst-compute section added (Colab CLI,
  idle-pod etiquette, 4060 default).
- WP.29 extraction incomplete → recurring close-read duty (Benchmarks BACKLOG P1.3).
- W-01…W-07 weakness catalog → formalized into `Opponent Analyzer/SCENARIO_DATABASE.md` (SC-01…SC-12)
  with lifecycle + excellence scoreboard (D-020 §5).
