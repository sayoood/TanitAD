# TanitAD Replay — User Tutorial (Sayed edition, 2026-07-12)

One engine, two modes: `test` (headless stats + regression gate) and `viz`
(interactive rerun visualization). Everything below runs from `stack/`.

## 1. The arms — what you are comparing

`--arms` takes one or more `name:checkpoint[:option]` specs:

| Spec | Model | What it emits |
|---|---|---|
| `main:/path/ckpt.pt` | TanitAD-4B-M world model | probe-decoded waypoints, per-horizon imagination error, H15 belief sigma, inv-dyn action readout |
| `refa:/path/ckpt.pt:pool` or `:grid` | frozen-DINO reference | same decode path over DINO tokens (online DINOv2 tokenizer) |
| `refb:/path/ckpt.pt` | E2E reference | tactical waypoints, maneuver distribution, nav command, action sequence, confidence, OOD score |

One arm of each name per session (colors are fixed per arm everywhere).

## 2. Test mode — statistics & regression gate

```
python scripts/replay_app.py --mode test \
    --arms main:/workspace/ckpt.pt refa:/workspace/refa/ckpt.pt:pool \
    --data-root /opt/comma_epcache --episodes 24 --stride 8 \
    --out /workspace/replay
```
Produces `stats.json`: per-arm/per-horizon ADE/FDE, action MAE, maneuver
accuracy, imagination-error means, latency p50/p95, per-episode ADE, and the
**worst-K windows** (episode + timestep of the hardest moments — feed these
to the scenario DB). Add `--rrd out.rrd` to also record a replayable artifact.

**Regression gate** (CI / nightly): compare against a pinned baseline —
non-zero exit on regression, direction-aware tolerances:
```
python scripts/replay_app.py --mode test ... \
    --baseline /workspace/replay_baseline/stats.json --tol ade=0.05
```

## 3. Viz mode — the interactive viewer

Two ways to look at a session:

**(a) Live from a pod (what is running now on pod2):**
```
python scripts/replay_app.py --mode viz --arms ... \
    --data-root /opt/comma_epcache --episodes 8 --stride 4 \
    --out /workspace/replay --serve 8888 --grpc-only --rrd /workspace/replay/session1.rrd
```
`--grpc-only` serves ONLY the data stream on the proxied port (single-port
pods). Open the HOSTED viewer in your browser:

```
https://app.rerun.io/version/0.34.1/?url=rerun%2Bhttps%3A%2F%2F<POD-ID>-8888.proxy.runpod.net%2Fproxy
```
(The `?url=` value is `rerun+https://<POD-ID>-8888.proxy.runpod.net/proxy`,
URL-encoded. Nothing installs locally; the viewer runs in the browser and
pulls the stream from the pod.)

**(b) Artifact (`.rrd`) — most robust:** every session records one; download
it and open it locally with the NATIVE viewer (`rerun session1.rrd` — on the
dev box: `C:\Users\Admin\venvs\tanitad\Scripts\rerun.exe <file>`). Doubles as
the per-checkpoint review archive.

**VERSION GOTCHA (cost us a confused hour, 2026-07-12):** .rrd files are
viewer-version-locked. The unversioned https://app.rerun.io serves the LATEST
viewer, which silently shows the welcome page for older recordings. Use the
version-pinned web viewer https://app.rerun.io/version/0.34.1/ for drag &
drop, or the native viewer (always version-matched to the SDK that recorded).
Known logger bug (fix pending): `--serve` + `--rrd` together — the last sink
wins and the file stays empty; record and serve in separate runs until fixed.

## 4. Reading the viewer

- **Camera + fans** (top-left): input frames with each arm's trajectory fan
  overlaid in its color + ground truth in white. THE at-a-glance comparison.
- **BEV ego**: same trajectories top-down in metres.
- **Actions**: steer/accel per arm vs ground truth (dashed).
- **Error strip**: per-step ADE per arm — spot a spike, click it, the whole
  view scrubs to that moment. This is how you find failure moments fast.
- **Heads**: imagination error + sigma (main), confidence + OOD (refb),
  maneuver distribution bars (refb), ego speed/yaw.
- **Timelines**: `step` scrubs globally; `episode` jumps between episodes.

## 5. Recipes

- Worst-moment review: run test mode, open stats.json `worst_k`, then scrub
  the paired .rrd to those steps.
- Checkpoint A/B: two sessions with the same episodes, different ckpts —
  compare stats deltas, then eyeball both .rrds.
- Fresh pod: `pip install rerun-sdk`, checkpoints + an epcache dir are the
  only inputs.
