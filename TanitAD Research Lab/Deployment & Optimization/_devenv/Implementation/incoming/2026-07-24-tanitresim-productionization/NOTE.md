# TanitResim productionization — 2026-07-24

Turned TanitResim (the trajectory/replay visualization tool) into a documented,
tested, production-ready tool; de-stranded it from the worktrees; wired it to
render the **full viz standard** end-to-end. Evidence class of every claim here:
**MEASURED** (dev-box run + `pytest`) unless marked otherwise.

## What was done

**P1 — consolidate + verify.** TanitResim exists in **20 worktree copies + the
main tree**. Hashed every key file across all copies: the **main tree is the
canonical superset** — its `app.js` / `export.py` / `style.css` / `test_resim.py`
are the largest and byte-identical to the 3 most-recent worktrees
(`data-engineering-20260721`, `opponent-20260721`, `dazzling-villani-bb4728`).
The one "in-between" worktree (`agent-a5ab55…`) had **0 unique app.js lines** and
1 stale export.py signature line. **Nothing to fold in — no stranding.** Verified
the server runs end-to-end (FastAPI + real browser, below); baseline `pytest`
green.

**P2 — viz-standard compliance.** Closed the gaps against THE STANDARD
(`taniteval/taniteval/corpus_overlay.py`):
- **Decoded-intent HUD** (new): each arm's camera pane now carries a text overlay
  — `tactical: <maneuver>` (argmax of that arm's `maneuver_probs`) +
  `strategic: route <goal>` (`nav_cmd`) + `ADE … · v … m/s`, mirroring the
  standard's 3-line HUD, per arm and co-located with the camera.
- **`nav_cmd` label bug fixed**: the SPA hard-coded `["straight","left","right"]`
  which mislabels indices 0 and 3 vs the canonical
  `refb.NAV_COMMANDS=("follow","left","right","straight")`. Now data-driven from
  `meta.nav_commands`. (The old test only exercised `nav_cmd=1`, correct by
  coincidence, so it missed the bug.)
- **BEV-only fallback** (new): `export_bundle(..., uncalibrated_corpora=…)` nulls
  the image-plane paths for corpora whose camera geometry is unrecoverable (e.g.
  cosmos f-theta); the SPA then shows the raw frame + *"camera overlay disabled —
  see BEV"* and the metric BEV carries the comparison — exactly the standard's
  per-clip degrade path.
- **Synthetic sample bundle** (`tanitad/resim/sample.py`): a deterministic,
  pod-free/checkpoint-free demo exercising every element (3 arms, decoded
  maneuver + route, varied maneuvers, formal gates, one fallback episode).

**P3 — hardening + docs.** One-command demo (`resim_app.py --demo`); rewrote
`tanitad/resim/README.md` with a viz-standard compliance table, the demo, the
export/serve/API reference, and the fallback contract. Confirmed the
path-traversal guard (encoded `..%2f` on both `sid` and frame `name` → 404) and
the fail-loud CLI (no `--sessions-root`/`--demo` → SystemExit).

## Run it (dev box or pod)

```
# one command, zero external state (synthetic bundle):
python scripts/resim_app.py --demo

# real bundles from an eval run:
python scripts/replay_app.py --mode export --arms main:… refa:…:grid refb:… \
    --data-root … --corpus-glob '*val*' --episodes 8 --out /workspace/resim/run --session-name "…"
python scripts/resim_app.py --port 8888 --sessions-root /workspace/resim   # -> https://<pod>-8888.proxy.runpod.net

# serve the bundle shipped in this folder:
python scripts/resim_app.py --sessions-root "…/2026-07-24-tanitresim-productionization/sample-bundle"
```

## End-to-end verification (MEASURED, dev box, 2026-07-24)

Server on `127.0.0.1:8899` (`--demo`). Real-browser DOM (no console errors):
- **Home**: 4 scenario cards, per-arm ADE bars, maneuver-mix ribbons.
- **Session**: per-arm HUD e.g. MAIN `tactical: TURN L · strategic: route left ·
  ADE 0.09 m · v 9.0 m/s`; REFA `tactical: BRAKE/STOP · gt turn l` (decoded≠GT
  annotation working); gate panels + GO banner; BEV; error strip; maneuver band;
  action panel; scrubber.
- **Fallback (ep3, cosmos-ood-demo)**: each camera shows *"camera overlay
  disabled — calibration unverified · see BEV"*; BEV still renders all arms + GT.

API probes: `/`, `/static/app.js`, `/api/sessions`, `/api/session/{id}`,
`/frames/{id}/{name}` all 200; three path-traversal probes → 404.

## Tests

`stack/tests/test_resim.py`: **39 passed** (was 33; +6 for nav_commands, the
BEV-only fallback, and the sample bundle). Full non-slow suite: **829 passed, 1
skipped**; the only failures (4, `tests/test_tlc_metric.py`) are a **sibling
agent's in-progress** `tanitad.eval.metrics` traffic-light work — unrelated to
TanitResim and outside this agent's territory (not touched).

## Deliverable manifest

| Artifact | Location | Staged |
|---|---|---|
| Exporter (+ `uncalibrated_corpora`, `NAV_COMMANDS`, `meta.nav_commands`) | `repo:stack/tanitad/resim/export.py` | yes |
| Synthetic demo generator | `repo:stack/tanitad/resim/sample.py` (new) | yes |
| SPA (decoded-intent HUD, nav fix, fallback note) | `repo:stack/tanitad/resim/static/app.js` | yes |
| SPA styles (HUD) | `repo:stack/tanitad/resim/static/style.css` | yes |
| Server (`--demo`, arg-guard) | `repo:stack/scripts/resim_app.py` | yes |
| Tests (39) | `repo:stack/tests/test_resim.py` | yes |
| README (viz-standard table, demo, API) | `repo:stack/tanitad/resim/README.md` | yes |
| This note | `repo:…/incoming/2026-07-24-tanitresim-productionization/NOTE.md` | yes |
| Concrete demo bundle (serve-ready) | `repo:…/2026-07-24-tanitresim-productionization/sample-bundle/` | yes |

Every artifact lives in the repo working tree and is `git add`-ed (staged, **not
committed**, per the operating standard). Nothing lives only on a pod or only in
a worktree.

## Honest status

**Production-ready now**: consolidated to one canonical tree, one-command demo,
full viz standard incl. fallback, 39 green tests, path-traversal + fail-loud
hardening, documented API/deploy.

**Still rough / remains**:
- The **flagship (`main`) arm does not yet emit `maneuver_probs`/`nav_cmd`** in
  `tanitad/replay/arms.py` (only REF-B does today), so on a *real* bundle the
  main arm's decoded-intent HUD shows only `ADE · v` until arms.py is wired to
  the trained `tactical_policy`/`strategic_policy`. The SPA + exporter already
  render it the moment arms.py populates those heads. **This is `arms.py`
  territory (shared) — escalated, not edited here.** (INHERITED from reading
  `arms.py`, not re-verified against a live checkpoint.)
- `uncalibrated_corpora` is per-**corpus**; the standard's finest grain is
  per-**clip**. Per-corpus matches how calibration actually fails (cosmos) and is
  sufficient today.
- Frontend has no automated headless (JS) test — verification is the real-browser
  DOM check above + the Python server/exporter tests. A Playwright smoke would
  close this.
