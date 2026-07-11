# TanitResim — TanitAD's branded replay-visualization web app

A purpose-built front-end for reviewing open-loop replays: **scenario cards**,
**curated side-by-side arm panels**, the TanitAD design language, served as a
proper single-port web app. Commissioned by Sayed 2026-07-11 to fix the raw
rerun viewer's real pain points — *signals hard to tell apart, poor legends*
(overlay + BEV). TanitResim's answer: one column per arm, a labelled metric
BEV, and **a legend on every canvas**.

TanitResim *consumes* the existing replay engine (`tanitad.replay`); it does
not re-run models. `replay_app.py --mode export` streams the engine's
`TimestepRecord`s into a portable **session bundle**, which
`scripts/resim_app.py` serves to a vanilla-JS SPA. No build step, no CDN, no
external runtime deps in the browser.

```
tanitad/resim/
  export.py            records -> session bundle (session.json + frames/*.jpg)
  static/index.html    SPA shell (wordmark, #app root)
  static/style.css     TanitAD design language (dark slate, gold/cyan/magenta)
  static/app.js        vanilla JS + canvas: home cards + session panels
scripts/resim_app.py   FastAPI single-port server (build_app factory + CLI)
scripts/replay_app.py  --mode export writes a bundle (added; test/viz intact)
tests/test_resim.py    exporter schema + portability + FastAPI TestClient
```

## The two views

**Home = scenario cards.** A top summary strip (session name, arm legend, global
per-arm ADE + latency), then a responsive grid with **one card per episode**:
camera thumbnail, corpus/scenario tag chip, per-arm ADE mini-bars in the arm
colors, and a worst-moment badge. A session picker (top-right dropdown) switches
between bundles. Click a card → session view.

**Session view = side-by-side arm panels.** One **column per arm** (header: name
+ color dot + episode ADE + p50 latency), each carrying: a **camera canvas** with
that arm's trajectory fan projected onto the road plus a toggleable GT overlay
(white dashed); **steer/accel small-multiple charts** vs GT over the episode with
a current-step marker; and **head readouts** (main: imag_rel per horizon + belief
σ; refa: imag_rel; refb: confidence + OOD + a maneuver-probability bar chart with
the GT class marked + nav command). Below the columns, a shared **master panel**:
a **BEV canvas** with all arms + GT on a metric grid — **metre axis labels, a
scale bar, and a legend box** — an **all-arm error strip** (per-step ADE lines in
arm colors, **click-to-seek** to a spike), and a **scrubber** with play/pause
(~10 fps), step/episode indicators and arrow-key nav (←/→ step, ↑/↓ episode,
space = play).

**Legends everywhere** (the top-priority ask): every canvas has a legend chip
row (arm color + name, GT = white dashed); the BEV is labelled in metres with a
scale bar; every line chart names its series. Nothing is unlabelled.

**Design language "TanitAD":** dark slate background (`#0e1420`), arm colors
**MAIN = gold/amber `#f5b301`**, **REF-A = cyan `#22d3ee`**, **REF-B = magenta
`#e35ce0`**, **GT = near-white dashed `#eef2f7`**; system font stack, generous
whitespace, card shadows, a subtle grid, and a "TanitResim" wordmark top-left.
URL hash carries `session/episode/step` (`#/s/<id>/e/<ep>/t/<step>`) so any view
is shareable.

> The branded palette lives in `tanitad.resim.export.RESIM_COLORS` and is
> emitted into each bundle's `session.json` (the SPA is data-driven off it). It
> is intentionally **separate** from `tanitad.replay.arms.ARM_COLORS`, which
> stays the canonical rerun palette (arms.py is not modified).

## Export a session bundle

```
python scripts/replay_app.py --mode export \
    --arms main:/workspace/exp/ckpt.pt refa:/opt/refa/ckpt.pt:grid refb:/opt/refb/ckpt.pt \
    --data-root /opt/comma_epcache --corpus-glob '*val*' --episodes 8 --stride 6 \
    --out /workspace/resim/run-30k --session-name "main@30k vs refs"
```

Writes a self-contained, **relative-path-only** bundle:

```
/workspace/resim/run-30k/
    session.json                 # meta (arms, colors, per-episode ADE, worst step) + per-step data
    frames/ep<i>_step<j>.jpg     # one shared camera frame per step (~q80, downscaled if >640px)
```

Serve the **parent** directory so multiple bundles show up in the picker.

## Serve on pod2 (RunPod)

Single plain-HTTP port — proxy-friendly (the earlier gRPC stream failed the
RunPod proxy; this does not):

```
python scripts/resim_app.py --port 8888 --sessions-root /workspace/resim
```

Then open the proxied URL:

```
https://<pod-id>-8888.proxy.runpod.net
```

Expose port **8888** on the pod's HTTP proxy. Everything (SPA, API, frames) is
same-origin under that one port, so no second port or `--connect-url` dance is
needed. FastAPI + uvicorn only; no browser-side CDN or external assets.

Endpoints: `GET /` (SPA) · `GET /static/*` (assets) · `GET /api/sessions`
(bundle summaries) · `GET /api/session/{id}` (full `session.json`) ·
`GET /frames/{id}/{name}` (a camera frame, path-traversal guarded).

## Tests

`tests/test_resim.py` (CPU-only): the exporter builds a valid bundle from
synthetic records (schema + one JPEG per step), checkpoints are stored
basename-only and **no absolute path leaks into session.json** (portability),
empty/frameless streams fail loud, wide frames are downscaled, and a
`fastapi.testclient.TestClient` drives index / sessions-list / session-fetch
(+404) / frame-fetch (+404 guards) / static-asset serving.
