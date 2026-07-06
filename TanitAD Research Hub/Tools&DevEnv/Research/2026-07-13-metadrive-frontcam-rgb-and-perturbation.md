# Tools&DevEnv — 2026-07-13 — MetaDrive front-camera RGB + perturbation sim arm

> Note on dating: the prior Tools&DevEnv run and the DataEng comma2k19 validation are dated
> 2026-07-06/07; this weekly Monday run is dated 2026-07-13 to keep the session log chronological.

**Run:** weekly Tools&DevEnv agent. Focus: unblock the D-010 sim arm (backlog #1 b/c) + scan
AlpaSim/AlpaGym and MetaDrive updates. Budget used: 3 web searches, 1 loop iteration, well under caps.

---

## 1. The problem this run fixes (why the sim arm was dead)

D-010 mixes real (comma2k19, representation + public open-loop numbers) with sim (off-expert
consequences, occluders, blocked routes, closed-loop). But the two contracts did not match:

| Source | Frame contract |
|---|---|
| comma2k19 `base250cam` (D-009, real) | `[T, 6, 256, 256]` — 2 stacked RGB front-cam frames (t-1, t) |
| MetaDrive adapter (WP2, as merged 07-06) | `[T, 1, 64, 64]` — single-channel top-down BEV |

`MixedWindowDataset._check_contract` (`stack/tanitad/data/mixing.py:71`) asserts every sim source's
`frames`/`future_frames` shape equals the real source's, with the explicit message *"sim episodes must be
rendered at the SAME size/channels as the real data (front-camera RGB 2-frame stacks)"*. So the BEV
adapter was **structurally inadmissible** to the mix — the sim arm of the headline co-training design
had no code path. Evidence, reproduced this run: `test_single_channel_sim_is_rejected_by_mix` (the 1ch
episode raises `AssertionError` in the mix constructor). — impact: D-010 / WP2 — repo-path.

## 2. What was built (implementation increment — intake package)

Per D-011 (hub/MVP separation), delivered as an intake package, **not** written into `stack/`:
`Tools&DevEnv/Implementation/incoming/2026-07-13-metadrive-frontcam-perturbation/`
(`tanitad_metadrive_frontcam.py` + `tests/test_metadrive_frontcam.py` + `INTAKE.md`).

- **Front-camera RGB path.** `frontcam_frame(rgb, size=256)` → `[3,S,S]` in [0,1]; geometry
  (center-crop to largest square → bilinear resize) mirrors `comma2k19._decode_video` exactly.
  `assemble_frontcam_episode(...)` stacks consecutive frames into `[T,6,S,S]` reusing comma2k19's
  `stack_two_frames`, computes `accel` as the finite-diff of pose speed, and keeps the **t+1**
  action/pose — byte-for-byte the comma2k19 alignment. Asserts `channels=6`. The single-channel BEV
  path is untouched (stays for the D3 probe). — impact: D-010 / A8.
- **Perturbation policy.** `perturb_action(base, t, cfg, rng)`: deterministic sinusoidal steer bias +
  stochastic throttle pulses / brake stabs, clipped to `[-1,1]`; identity when disabled. This is the
  sim arm's *reason to exist* — action-consequence pairs expert logs never contain. Reproducible from
  the episode seed. — impact: D-010 (off-expert coverage), H1/H11.
- **Scenario configs → env kwargs.** `cruise` / `scripted_occluder` (H15/D9 object permanence) /
  `blocked_route` (D5/D6 fallback). `env_config()` is pure (grounded on the MetaDrive RGBCamera API,
  §4) and offline-testable; live object placement is isolated in `populate_scene()`
  (`NotImplementedError` until the supervised install, version-sensitive spawn API — flagged honestly,
  P8). — impact: D9 / D5 / D6.
- **Persistence.** `generate_and_save(...)` writes each episode via the existing
  `mixing.save_episode` (uint8, channel-agnostic → 6ch works unchanged) — backlog #1(c) entry point
  the pod calls. — impact: WP2.

**Tests:** 17 passed / 1.67 s, no simulator; full stack suite unaffected (46 passed, 1 skipped).

## 3. G-T1 — measured setup cost & go/no-go (no aspirational tooling)

- Pure path: module import **1.38 s** (torch-dominated), **0 new dependencies**, real-size conversion
  1164×874 → 256 verified. **GO, in-envelope** (RTX 4060, free — Master Plan §4 local-first). ✅
- Live rollout: **still NO-GO in an unattended run** — needs the supervised MetaDrive *source* install
  (PyPI `metadrive-simulator` unbuildable on py3.13, verdict unchanged 2026-07-06). Once installed:
  `pip install -e stack/.[sim]` native deps already present; `pytest -m slow` runs the live smoke.
  Estimated supervised cost: ~5–10 min (git source install + one 20-step render check). — impact: P5.

## 4. MetaDrive front-camera API (grounded this run)

RGB front camera (docs.metadrive-simulator, DeepWiki): `image_observation=True`;
`sensors={"rgb_camera": (RGBCamera, W, H)}`; `vehicle_config.image_source="rgb_camera"`. Observation is a
dict, `obs["image"]` shape `(H, W, 3, stack_size)` (default `stack_size=3`), most-recent frame at
`[..., -1]`, normalized [0,1] float32 by default. `image_on_cuda=True` ≈ 10× rollout throughput (pod
only, VRAM-gated). Caveat for the supervised run: image buffer has historically been BGR / row-flipped on
some backends — verify once against a saved PNG. — impact: WP2 —
[MetaDrive sensors](https://metadrive-simulator.readthedocs.io/en/latest/sensors.html),
[obs](https://metadrive-simulator.readthedocs.io/en/latest/obs.html).

## 5. Opponent / ecosystem scan (research focus)

- **NVIDIA Alpamayo 2 Super** (GTC Taipei, 2026-06-01): 32 B-param open VLA *reasoning* model for L4
  robotaxis; expected on GitHub/HF "this summer". Trained closed-loop via **AlpaGym** (high-throughput
  closed-loop RL) on **AlpaSim**. 32 B is ~100–300× our envelope → reinforces, not threatens, our thesis:
  we prove the *mechanism* (hierarchical latent WM at 10–100 M, labels=0, data≤35 h), we do not chase
  scale (Master Plan C2/P5). — impact: opponent context / P5 —
  [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis).
- **NVIDIA OmniDreams**: generative world model for *photorealistic closed-loop* AV scenario generation
  (rare/long-tail), built on AlpaSim + Omniverse NuRec. Same-family as our H15 imagination goal but
  A100/Omniverse-scale and asset-heavy → **watch, do not adopt** in Phase 0 (consistent with the
  2026-07-06 AlpaSim NO-GO on 4060). Possible Phase-1 comparison target for closed-loop scenario realism.
  — impact: H15 / Phase-1 sim —
  [Alpamayo closed-loop blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/).

## 6. Actionable recommendations (tied to hypotheses / WPs)

1. **(Ready for triage)** Integrate the front-cam intake package → unblocks the D-010 sim arm so Stage-0
   bake-off (Phase 0 Plan §4, W2) can actually run the real-vs-mixed comparison. Owner: MVP orchestrator.
2. **(Sayed / supervised)** One 10-min supervised MetaDrive source install → run `-m slow` to confirm
   `frame_change_fraction > 0.01` on real front-cam renders (A8) and settle the BGR/orientation caveat.
   This is the last blocker on live sim episodes for D5/D6.
3. **(Architecture, Wed)** The 6ch/256 sim frames are now the same tensor as real — no ONNX-shape divergence
   between the sim-eval and deployment paths. Keep encoder input static at `[6,256,256]`.
4. **(Do not adopt)** AlpaSim/OmniDreams remain Phase-1 cloud watch items — no resource-ledger entry.

## 7. Self-critique (quality gates)

- G-A sources: all §-claims carry a link or repo path. ✅  G-B: 4 actionable recs tied to
  D-010/D5/D6/A8/H15. ✅  G-E: 17 passing standalone tests + measurable next step. ✅  G-T1: measured
  setup cost + explicit go/no-go. ✅  Honesty (P8): live path openly gated; `populate_scene` raises
  rather than pretending. ✅
- Gap (recorded): live A8 consequence-dominance on real renders is asserted-by-test-only until the
  supervised install; no hypothesis status is upgraded on that basis (see HYPOTHESIS_LEDGER note).
