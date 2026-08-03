# AlpaSim closed-loop videos — REF-C and flagship v1

**These are REAL closed-loop rollouts** on NuRec neural reconstructions: a renderer generates each
camera frame, the policy returns a trajectory, a controller executes it, and the next frame is
rendered from where the car actually went. Not a replay, not an imagination proxy.

## A. LONG videos, rendered ON THE JETSON THOR with our own gsplat renderer (2026-08-03)

| file | arm | condition | duration | size |
|---|---|---|---|---|
| `flagship-v1_empty_road.mp4` | flagship v1 | background + road only | **18.0 s** (180 f @10 fps) | 1800×850 |
| `flagship-v1_with_objects.mp4` | flagship v1 | + 26 gaussian agents | 18.0 s | 1800×850 |
| `refc-base_empty_road.mp4` | REF-C base | background + road only | 18.0 s | 1800×850 |
| `refc-base_with_objects.mp4` | REF-C base | + 26 gaussian agents | 18.0 s | 1800×850 |

Layout = the programme's standing viz standard, **camera + metric BEV + decision HUD together**:
the f-theta frame the policy actually consumed, with its plan projected through the **real f-theta
forward polynomial** and the logged route in green; a metric BEV (ego bottom-centre, rotated to ego,
range rings, driven path, plan, agents with the lead called out); and a HUD naming the decoded
**tactical manoeuvre** (planned / head / executed / logged), the **strategic** route command and
corridor state, and live lateral + longitudinal errors.

Every file was verified by **decoding it back** (180/180 frames, first frame decodes, non-black).

Code + reproduction: `stack/experiments/alpasim-gsplat/`.
Raw metrics: `stack/experiments/alpasim-gsplat/results/metrics_*.json`.
Contract test: `…/results/contract_test.json`. Actor-placement falsifier: `…/results/actor_map.json`.

### ⛔ Two corrections this run forces on the text that used to be here

1. **"Regenerating longer ones needs the `tanitad-eval` pod restarted" — RETRACTED.** They were
   regenerated on Thor, longer and richer, with no A40 and no NVIDIA renderer.
2. **"Why not on the Jetson Thor / the NRE renderer is a closed x86_64 binary" — the premise is
   true, the conclusion is RETRACTED.** We do not need NRE: `volume.nurec` is gzip+MessagePack and
   gsplat 1.5.3 renders it natively on aarch64, **including the f-theta camera model**, at
   **16–28 ms per 1920×1080 frame** with the scene resident on the GPU. Thor IS a simulation node.
   *(Root-cause class: an absence claim about one implementation generalised to the capability.)*

### ⚠️ What these videos are NOT

They are **AlpaSim's renderer contract satisfied by our renderer, driven by a TanitAD closed-loop
harness** — NOT `alpasim_runtime.simulate`. MEASURED on Thor: `alpasim_grpc`, `alpasim_utils` and
`alpasim_wizard` import; **`alpasim_runtime`, `alpasim_controller`, `alpasim_physics` and `utils_rs`
do not**, and `uv` is absent. So there is **no AlpaSim collision / offroad / scene score** for these
four videos — the four TanitAD metric families are what is measured instead.

## B. Earlier short videos, rendered on the `tanitad-eval` A40 with NVIDIA's NRE (2026-07-22)

| file | model | duration | resolution |
|---|---|---|---|
| `flagship-v1_alpasim-closedloop_10s.mp4` | flagship v1 | 10.4 s (52 f @5 fps) | 900×1000 |
| `refc-base_alpasim-closedloop_10s.mp4` | REF-C base | 10.4 s | 900×1000 |
| `refc-xl_alpasim-closedloop_10s.mp4` | REF-C XL | 10.4 s | 900×1000 |
| `refc-small_alpasim-closedloop_10s.mp4` | REF-C small | 10.4 s | 900×1000 |

These DO carry AlpaSim's own eval overlay (collision / offroad / progress) because they came out of
the full runtime. Recipe: `…/Implementation/incoming/2026-07-22-alpasim-closedloop-evalpod/`.

## Reading any number from either set

⛔ **Every closed-loop number from this asset is WITHIN-SIM RELATIVE.** REF-C's own open-loop ADE is
**1.5157 on these reconstructions vs 0.4728 on real footage — 3.21× OOD**
(`REFC_openloop_diagnostic.json`). **Orderings survive; absolute rates do not.** Never quote a sim
rate as a real-world rate.

⛔ **ADE alone is an incomplete result** (binding, 2026-08-02). The Thor run reports LONGITUDINAL,
LATERAL, TACTICAL and STRATEGIC alongside it, each with a paired episode-cluster bootstrap CI.
Headline: **REF-C base beats flagship v1 closed-loop and the separation is entirely LATERAL** —
heading, curvature, yaw-rate and cross-track all have CIs excluding zero, while **ADE does not
separate** (+0.789 [−0.865, +2.728]).

⚠️ `.gitignore` excludes `*.mp4` — every file here is committed with `git add -f`. A new video needs
the same or it silently never lands.
