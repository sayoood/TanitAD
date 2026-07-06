# STATE — Tools&DevEnv

LAST_RUN: 2026-07-13 (W2, second weekly run)
QUALITY: full (G-A…G-F + G-T1 met; live MetaDrive rollout still gated on the supervised source install)

## HANDOFF
Backlog #1 is now b/c-complete **as an intake package** (D-011), pending orchestrator triage:
`Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`
(`tanitad_metadrive_frontcam.py` + tests + `INTAKE.md`). It adds the **front-camera RGB path**
(6ch/256, comma2k19-identical) that unblocks the D-010 sim arm — the merged 1ch BEV adapter is
structurally rejected by `MixedWindowDataset` (proven by test). 17 standalone tests pass; 0 new deps.

**To finish the LIVE path (supervised session, ~10 min):**
1. `pip install git+https://github.com/metadriverse/metadrive.git` then `pip install -e stack/.[sim]`.
2. `pytest stack/tests/test_metadrive_frontcam.py -m slow -q` (after integration) — confirm 6ch/256
   frames render and `frame_change_fraction > 0.01` (A8) on real front-cam.
3. Settle the `obs["image"]` BGR/row-flip caveat vs a saved PNG.
4. Wire `populate_scene()` object spawns for occluder/blocked-route (currently `NotImplementedError`;
   MetaDrive `engine.spawn_object` signature is version-sensitive).

Backlog #1(a) verdict (PyPI no-go py3.13, source=GO) unchanged.

**Next backlog item (#2):** `episode → Rerun .rrd` replay/viz overlay (predicted-vs-actual trajectory +
BEV). Doubles as the D3 imagined-vs-oracle visual. Rerun is the chosen tool (`pip install rerun-sdk`);
measure its setup cost for G-T1.

## Done this run
- Intake pkg `2026-07-13-metadrive-frontcam-perturbation/`: front-cam RGB (`frontcam_frame`,
  `assemble_frontcam_episode` → 6ch/256 comma2k19-identical), perturbation policy (`perturb_action`),
  scenario configs (cruise/occluder/blocked-route), `generate_and_save` via `mixing.save_episode`.
- 17 standalone tests pass (1.67 s, no sim). Full stack suite unaffected: 46 passed / 1 skipped.
- Research note `2026-07-13-metadrive-frontcam-rgb-and-perturbation.md`; KB delta (3 findings).
- G-T1 measured: pure path GO (import 1.38 s, 0 new deps); live rollout gated on supervised install.

## Open threads / proposals to raise
- AlpaGym closed-loop RL post-training with our own <100 M driver — A100-gated Phase-1 proposal (draft to
  `Project Steering/Proposals/` once D1–D3 pass). NVIDIA Alpamayo 2 Super (32 B) + OmniDreams confirm the
  closed-loop-RL-on-sim direction at scale; our edge stays efficiency/labels (P5/C2), not scale.
- Note to Wed (Architecture): sim frames are now the SAME tensor as real (`[6,256,256]`) — no ONNX-shape
  divergence between sim-eval and deployment. Keep the encoder input static at `[6,256,256]`; keep ViT
  shapes static + norms batch-free for the ONNX→TensorRT FP16 Orin path (INT8 deferred, must be measured).
