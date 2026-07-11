# tanitad.replay — open-loop replay, regression test & visualization app

Replays identical cached episodes through the three architecture arms —
**main** latent world model (blue), **REF-A** frozen-DINO (purple), **REF-B**
E2E (orange), against **GT** (green) — as (a) an automated regression harness
with a CI-hookable exit code and (b) an interactive rerun visualization.

Entry point: `stack/scripts/replay_app.py`. Commissioned by Sayed 2026-07-11.

```
# regression gate (test mode)
python scripts/replay_app.py --mode test \
    --arms main:/workspace/exp/ckpt.pt refa:/opt/refa/ckpt.pt:grid refb:/opt/refb/ckpt.pt \
    --data-root /opt/comma_epcache --corpus-glob '*val*' --episodes 24 \
    --out /workspace/replay --baseline /workspace/replay_base/stats.json

# visualization artifact (viz mode)
python scripts/replay_app.py --mode viz --arms ... --data-root ... \
    --out /workspace/replay --rrd /workspace/replay/replay.rrd
```

Exit codes: `0` green, `1` at least one metric regressed beyond tolerance.

---

## Deep-research verdict: rerun as the viz backbone — ADOPTED

Evaluated 2026-07-11 (`rerun-sdk` 0.34.1, pip) in the tanitad venv
(py3.13/Windows dev box; pure-pip wheel, no system deps, so pods are covered
too):

- **Headless logging works**: `rr.save(path.rrd)` writes complete artifacts
  with no display/viewer attached (pinned by `tests/test_replay.py::
  test_rr_log_writes_rrd`). This is the property the regression harness
  needs — viz artifacts fall out of CI runs for free.
- **Full archetype coverage** for our schema: `Image` (camera frames, JPEG-
  compressible), `LineStrips2D` (trajectory fans, BEV paths), `Scalars` +
  `SeriesLines` styling (actions/error/monitor time series), `TextLog`
  (episode boundaries), multiple timelines (`step`, `episode`) with free
  scrubbing, and a programmatic blueprint API for the default layout.
- **API note (0.34)**: `rr.serve_web()` is gone. Live serving =
  `rr.serve_grpc()` (data stream, default port 9876, URL form
  `rerun+http://host:9876/proxy` — i.e. HTTP-tunnelable) +
  `rr.serve_web_viewer(web_port=...)` (viewer UI). Verified working locally.
- **Artifacts open anywhere**: `rerun replay.rrd` locally, or drag the file
  into <https://app.rerun.io> (no install).

**Fallback (not needed)**: a FastAPI + static canvas viewer remains the
documented plan B only if rerun's web viewer ever proves un-proxyable on a
pod — the `.rrd`-artifact path cannot break that way, so the trigger is
unlikely to fire.

`rerun-sdk>=0.34` is deliberately NOT a hard dependency: everything in
`--mode test` runs without it; `rr_log` raises with an install hint only
when a rerun sink is actually requested. (Kept out of `pyproject.toml` to
honor the new-files-only commission; add `viz = ["rerun-sdk>=0.34"]` to the
optional deps when touching that file next.)

## Repo patterns reused (where the numbers' meaning comes from)

- `_ego` waypoint convention + collect loop: `scripts/evaluate_checkpoint.py`,
  `scripts/d1_probe_capacity.py` — waypoints at 5/10/15/20 steps from the
  anchor (last window) pose, ridge probes at `alpha=10`.
- Probe doctrine: encoder-state waypoint probe fitted on a HELD-OUT leading
  split (episode-level, I3); imagination probes fitted ON imagined latents
  (A3 calibrated decode) — both from `scripts/viz_trajectory_fan.py`.
- Camera ground-plane projection (approximate, intuition only):
  `viz_trajectory_fan.py` constants (f_eff 266 px @ 256 px, cam height
  1.22 m, D-016), rescaled by frame size.
- REF-A online tokenization: `scripts/dino_precompute.py` DINOv2-B/14
  fallback path exactly (latest 3 channels, 224 px, ImageNet norm, 16x16
  fp16 grid) — run online per frame with an LRU cache so overlapping
  windows tokenize each frame once.
- REF-B pseudo-labels and nav commands: `scripts/refb_labels.py` imported
  (not duplicated) — maneuver accuracy and the strategic input use the
  pinned thresholds.
- Fail-loud windowing: the REF-B `build_window_index` doctrine (2026-07-10
  review) — short episodes, empty corpora, empty fit splits and unfitted
  standardizers RAISE. No silent skips anywhere.
- Episode I/O: `tanitad.data.mixing.load_episode(mmap=True)` (F-7), corpus
  tag = cache-dir name, `--episodes/--stride` bounds.
- Tools&DevEnv 2026-07-09 test-suite-I/O note: the timing-guard idea pairs
  with test mode — latency p50/p95 are first-class stats metrics with a
  loose default tolerance (wall-clock is machine-dependent; tighten via
  `--tol latency=0.1` when baseline and candidate share a box). The future
  `ci.ps1` hooks the gate as
  `python scripts/replay_app.py --mode test ... --baseline <pinned>/stats.json`.

## Architecture

```
tanitad/replay/engine.py   ReplayEngine: corpora -> deterministic window
                           batches (engine window = max arm window, shared
                           anchor frame) -> TimestepRecord stream
tanitad/replay/arms.py     ArmAdapter protocol + MainArm / RefAArm / RefBArm
                           (+ DinoV2Tokenizer, ToyTokenizer, ARM_COLORS)
tanitad/replay/stats.py    aggregate() -> stats.json; compare() -> delta
                           table + regressions (exit-code source)
tanitad/replay/rr_log.py   RerunLogger: records -> rerun entities + default
                           blueprint (.rrd and/or live gRPC+web viewer)
scripts/replay_app.py      CLI wiring all of the above
tests/test_replay.py       CPU-only synthetic coverage (see file docstring)
```

Per-window emissions (all arms): GT waypoints/action, predicted waypoints at
(5, 10, 15, 20) steps, an action readout (REF-B: operative head row 0;
main/REF-A: inverse-dynamics head on the 1-step imagined latent), control-
path latency. Main/REF-A add per-horizon imagination decodes (the fan),
imag_rel (A9 self-monitor, per-sample persistence-normalized). Main adds the
H15 mean belief sigma. REF-B adds the 0.5 s action sequence, maneuver
distribution + pseudo-label GT, nav command, confidence and feature-OOD.

Efficiency levers: mmap episodes, batched windows (`--batch`), fp16 autocast
on CUDA (`--half`), REF-A per-frame token LRU cache, `--no-imag-rel` to skip
the future-frame encodes when only waypoint metrics matter.

## Regression mode

`stats.json` = `{"meta": {...}, "arms": {arm: metrics...}}` with per-arm:
`ade@k`, `ade`, `fde@20`, `steer_mae`, `accel_mae`, `maneuver_acc` (REF-B),
`imag_rel_k*`, `conf/ood/sigma_mean`, `latency_p50/p95_ms`,
`per_episode_ade`, `worst_windows` (top-K by ADE with corpus/episode/t/step —
the scrub-here list for the viz).

`compare()` direction rules: metrics containing `maneuver_acc`/`r2` regress
DOWNWARD, everything else UPWARD; `n_windows`/`fit_windows` are
informational. Default relative tolerances: 5 % (quality), 10 % (imag_rel),
25 % (monitor means), 50 % (latency); override with `--tol ade=0.10 ...`
(substring match). Full delta table goes to stdout and `regression.json`.

## Viz schema (one color per arm, everywhere)

| entity | content |
|---|---|
| `/camera` + `/camera/traj/*` | anchor frame + projected trajectory fans |
| `/bev/gt`, `/bev/<arm>` | ego-frame waypoint paths (m, forward = up) |
| `/bev/<arm>/imagination` | per-horizon A3 imagination fan rays |
| `/actions/steer\|accel/*` | per-arm action readout vs GT |
| `/error/<arm>` | per-step waypoint ADE — the error strip; scrub to spikes |
| `/heads/conf\|ood\|sigma/<arm>`, `/heads/imag_rel/<arm>/k<k>` | monitors |
| `/maneuver/*` | REF-B tactical distribution + GT class |
| `/ego/speed`, `/ego/yaw_rate`, `/meta` | replayed kinematics, episode log |

Default blueprint (sent on init — the screenshot view): left column camera +
BEV + episode log; middle steer/accel/error-strip time series; right monitor
heads + maneuver distribution + ego kinematics. Timelines: `step` (global
scrub axis), `episode` (route jumping). Colors: GT `#2ca02c`, main
`#1f77b4`, refa `#9467bd`, refb `#ff7f0e`.

## Pod deployment (RunPod)

1. `pip install rerun-sdk` in the pod venv (pure wheel, no system deps).
2. **Recommended**: produce an artifact — `--mode viz --rrd
   /workspace/replay/replay.rrd`, download it, open locally with
   `rerun replay.rrd` or at app.rerun.io. This path has no proxy moving
   parts and doubles as a per-checkpoint archive.
3. **Live viewer**: `--serve 9090` starts the web viewer on 9090 and the
   data stream on gRPC 9876 (`rerun+http://...:9876/proxy` — plain HTTP/2,
   proxyable). Expose BOTH ports via the RunPod HTTP proxy and point the
   browser-side data URL at the proxied stream:
   `--connect-url rerun+https://<pod-id>-9876.proxy.runpod.net/proxy`,
   then open `https://<pod-id>-9090.proxy.runpod.net`. The `--connect-url`
   override exists precisely because the default localhost URL is resolved
   by the BROWSER, not the pod. (Verified locally; first live-pod run should
   sanity-check the proxy hop and fall back to the artifact path if RunPod's
   proxy chokes on gRPC-web framing.)
4. Regression gate in CI/agents: keep a pinned `stats.json` per corpus next
   to the baseline checkpoint; the app's exit code 1 fails the pipeline.

## Extensions

Implemented now: error-strip timeline per arm; worst-K windows in stats
(scrub the `step` timeline to the listed steps); imagination fan for
main/REF-A (`/bev/<arm>/imagination`); H15 mean belief sigma head; REF-B
maneuver-distribution view; dual-corpus replay with per-episode ADE table.

Roadmap (deliberately deferred):
- **Sigma-vs-realized-error calibration scatter** — needs per-SECTOR sigma
  paired with realized per-sector imagination error; lands with the D9/H15
  eval loop (log as `rr.Points2D` under `/calibration`).
- **Latent-PCA inspector** — fit a 3-component PCA on fit-split states per
  arm, log `/latent/<arm>` as `Points3D` per step; cheap, high insight for
  collapse forensics.
- **Regression-diff view, two checkpoints same arm** — the engine already
  takes arbitrary arm lists; needs only name suffixing (`main@14k`,
  `main@30k`) in the CLI spec parser and a paired-delta stats section.
- **Worst-K reel blueprint** — auto-generate a blueprint whose time panels
  are pre-zoomed to the worst-K steps (rerun blueprint API supports
  per-view time ranges).
- **Closed-loop replay** — swap the episode iterator for a MetaDrive env
  stepper once the closed-loop harness lands (D-014 scope).
