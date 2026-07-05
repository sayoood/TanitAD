# STATE — Tools&DevEnv

LAST_RUN: 2026-07-06 (W1→W2, first weekly run)
QUALITY: full (G-A…G-F + G-T1 met; live MetaDrive smoke deferred to a supervised install)

## HANDOFF
Backlog item #1 (MetaDrive wrapper, WP2) is **implemented and CI-green** — the pure contract helpers are
merged and tested; only the *live* rollout is unvalidated because installing MetaDrive from source needs
user-approved trust (blocked in the unattended run). To finish it in a supervised session:
1. `pip install git+https://github.com/metadriverse/metadrive.git` (native deps panda3d/gymnasium already in venv).
2. `pytest stack/tests/test_metadrive_env.py -m slow -q` to run the live smoke.
3. If the installed MetaDrive's `env.render(mode="topdown", ...)` returns None/wrong shape, switch
   `_topdown_frame` to `TopDownObservation` (see module docstring). Contract helpers need no change.

**Next backlog item (#2):** `episode → Rerun .rrd` replay/viz overlay (predicted-vs-actual trajectory +
BEV). Doubles as the D3 imagined-vs-oracle visual. Rerun is the chosen tool (`pip install rerun-sdk`);
measure its setup cost for G-T1.

## Done this run
- `stack/tanitad/data/metadrive_env.py` — MetaDrive→toy-contract adapter (lazy import, pure helpers).
- `stack/tests/test_metadrive_env.py` — 7 contract tests pass; 1 live test skips w/o MetaDrive. Suite 17✓/1s.
- `pyproject.toml` `[sim]` extra fixed to installable native deps + `slow` marker registered.
- Research note `2026-07-06-metadrive-adoption-and-alpasim-verdict.md`; KB delta (4 findings).

## Open threads / proposals to raise
- AlpaGym closed-loop RL post-training with our own <100 M driver — A100-gated Phase-1 proposal (draft to
  `Project Steering/Proposals/` once D1–D3 pass).
- Note to Wed (Architecture): keep ViT shapes static + norms batch-free for a clean ONNX→TensorRT FP16
  Orin path (INT8 deferred, must be measured).
