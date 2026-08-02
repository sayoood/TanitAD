# AlpaSim closed-loop videos — REF-C and flagship v1

**These are REAL AlpaSim closed-loop rollouts** on NuRec neural reconstructions: the renderer
generates each camera frame, our policy adapter returns a trajectory, the MPC controller executes
it, and the next frame is rendered from where the car actually went. Not a replay, not an
imagination proxy.

| file | model | duration | resolution |
|---|---|---|---|
| `flagship-v1_alpasim-closedloop_10s.mp4` | flagship v1 | **10.4 s** (52 frames @ 5 fps) | 900x1000 |
| `refc-base_alpasim-closedloop_10s.mp4` | REF-C base | 10.4 s | 900x1000 |
| `refc-xl_alpasim-closedloop_10s.mp4` | REF-C XL | 10.4 s | 900x1000 |
| `refc-small_alpasim-closedloop_10s.mp4` | REF-C small | 10.4 s | 900x1000 |

⚠️ **THESE ARE SHORT.** 10.4 s each. The PI asked for LONG videos with richer visualisation; these
are the existing artifacts collected into one place, not that deliverable.

⛔ **Regenerating longer ones needs the `tanitad-eval` pod restarted** — MEASURED 2026-08-02:
`Connection refused`. That pod holds the AlpaSim source (98 GB), the extracted NRE renderer
(38 GB at `/workspace/nre/rootfs`) and the built scenesets. None of it is in the repo and none of
it is on Thor.

## Why not on the Jetson Thor

The NRE renderer is a **closed x86_64 binary**; Thor is `aarch64`. Everything else we do on Thor
(TensorRT, latency, four-family scoring) is unaffected — Thor is an evaluation and deployment node,
not a simulation node.

⚠️ **Correction to an earlier claim of mine:** I probed `docker pull nvcr.io/nvidia/nre/nre-ga:26.04`,
got `no matching manifest for linux/arm64/v8`, and generalised that to "AlpaSim cannot run on our
hardware". **That was too broad and the probe tested a path we do not use.** We run AlpaSim **bare on
one A40** — upstream's Docker Compose deployment was deliberately bypassed, and the renderer was
acquired by a bearer-token layer-fetch into a rootfs, not by `docker pull`. See
`…/2026-07-26-alpasim-consolidation/ALPASIM_STATE.md` §1.

## Reading these numbers

⛔ **Every closed-loop number from this asset is a WITHIN-SIM RELATIVE number.** REF-C's own
open-loop ADE is **1.5157 on these reconstructions vs 0.4728 on real footage — 3.21x OOD**
(`REFC_openloop_diagnostic.json`, 4 scenes / 288 predictions). **Orderings survive; absolute rates
do not.** Never quote a sim rate as a real-world rate.

## For the long-video regeneration

- ⭐ The PI's steer: **render the front camera only** — the multi-camera rig is the dominant render
  cost, and a single 120-deg front view is what our models consume anyway.
- Source recipe: `…/2026-07-22-alpasim-closedloop-evalpod/` (`alpasim_setup.sh`, `refc_driver.py`,
  `flagship_v1_driver.py`, launchers) and `RUN_RECIPE.md` (588 lines).
- ⚠️ `.gitignore:24` excludes `*.mp4` — these are committed with `git add -f`. Any new video needs
  the same, or it silently never lands (that mismatch is exactly what stranded the originals).
