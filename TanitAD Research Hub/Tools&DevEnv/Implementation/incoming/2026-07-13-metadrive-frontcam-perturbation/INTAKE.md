# INTAKE — MetaDrive front-camera RGB + perturbation episode generator

- **Package:** `Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`
- **Author agent / date:** tools-devenv-agent, 2026-07-13
- **Proposed target:** `stack/tanitad/data/metadrive_frontcam.py` (new module, sibling of
  `metadrive_env.py`) + `stack/tests/test_metadrive_frontcam.py`
- **Hypothesis / WP served:** WP2 / D-010 mix / H15 + D9 (occluder) / D5-D6 (blocked route) / A8

## What & why (≤10 lines)

The merged MetaDrive adapter renders a single-channel top-down BEV `[T,1,64,64]`. The real corpus
(comma2k19 `base250cam`, D-009) is `[T,6,256,256]` — 2 stacked RGB front-camera frames.
`MixedWindowDataset._check_contract` (D-010) **rejects any sim source** whose frame shape differs, so
the D-010 mix currently has **no admissible sim arm** — the whole point of sim (off-expert consequences,
occluders, blocked routes) is unreachable. This package adds the **front-camera RGB path**: a MetaDrive
episode becomes byte-for-byte contract-identical to a comma2k19 episode (6ch, 256px, identical 2-frame
stacking and t+1 action/pose alignment), plus a **scripted perturbation policy** (off-expert steer/throttle
coverage) and three **scenario configs** (cruise / scripted-occluder / blocked-route). The single-channel
BEV path in `metadrive_env.py` is untouched and stays for the D3 imagined-vs-oracle probe. Frames persist
via the existing `mixing.save_episode` (channel-agnostic uint8) → backlog #1(c).
Research note: `Tools&DevEnv/Research/2026-07-13-metadrive-frontcam-rgb-and-perturbation.md`.

## Evidence & tests

- Tests included: `tests/test_metadrive_frontcam.py` — **17 passed in 1.67 s** on the author machine
  (RTX 4060 / py3.13, no simulator). Full stack suite unaffected: **46 passed, 1 skipped** (baseline).
- Load-bearing tests: `test_frontcam_mixes_with_real_contract` proves a front-cam episode mixes with a
  comma2k19-shaped 6ch episode via `MixedWindowDataset`; `test_single_channel_sim_is_rejected_by_mix`
  proves the old 1ch BEV path is (correctly) refused — i.e. this package is exactly the missing piece.
  `test_assemble_frontcam_episode_matches_real_contract` verifies 6ch/256, action=finite-diff-accel,
  t+1 alignment identical to `comma2k19.build_episode`.
- **G-T1 (measured setup cost + go/no-go):** module import **1.38 s** (torch-dominated; **0 new
  dependencies** beyond the existing stack), real-size conversion 1164×874→256 verified. `frontcam_frame`
  / `assemble_frontcam_episode` / `perturb_action` / scenario configs: **GO, in-envelope** (RTX 4060,
  free). **Live rollout: still gated** on the supervised MetaDrive *source* install (PyPI no-go on py3.13,
  verdict unchanged from 2026-07-06); once installed, `pytest -m slow` exercises it.

## Risk & rollback

- Blast radius if integrated: one new module + one new test file; **zero edits** to existing stack code
  (additive). No import-time dependency on MetaDrive (lazy). Contract primitives are imported/reused, not
  copied, so no drift risk.
- **Supervised-run TODOs before the live path is trusted** (flagged in code, not silently assumed):
  (1) confirm `obs["image"]` orientation/channel order (BGR/row-flip) against a saved PNG — world model is
  channel-agnostic but the replay overlay is not; (2) wire `populate_scene()` object-spawn signatures for
  the occluder/blocked-route scenarios (currently `NotImplementedError`, version-sensitive API);
  (3) run `-m slow` to confirm `frame_change_fraction > 0.01` (A8 consequence-dominance) on real renders.
- Rollback: delete the two files; nothing else references them.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate-with-changes
- **Date / by:** 2026-07-06, MVP orchestrator session
- **Reason & notes:** Exemplary package: additive-only, zero new deps, reuses contract primitives
  instead of copying them, honest NotImplementedError on version-sensitive live-sim APIs, and the two
  load-bearing contract-compatibility tests are exactly what D-010 needed. Only change: replaced the
  test's sys.path import hack with the proper `tanitad.data.metadrive_frontcam` import. Standalone
  17/17; full stack suite 65 passed / 1 sim-skip after integration. The three supervised-run TODOs
  (BGR/orientation check, populate_scene wiring, live A8 check) remain open and tracked in your STATE —
  correctly NOT blockers for the offline path.
- **Integrated as:** `stack/tanitad/data/metadrive_frontcam.py` + `stack/tests/test_metadrive_frontcam.py`
  (commit: see `intake(tools-devenv)` in git log, 2026-07-06)
