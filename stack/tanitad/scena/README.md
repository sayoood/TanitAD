# TanitScena — TanitAD's scenario-database web app

Turns the Opponent-Analyzer **Opponent-Weakness Scenario Database**
(`TanitAD Research Hub/Opponent Analyzer/SCENARIO_DATABASE.md` — the SC-01..SC-14
catalogue of documented opponent "dumb situations") into a searchable,
browsable, **visual** single-port app with **local semantic (vector) search**.
Commissioned by Sayed (D-029). Self-contained, pod-servable, no build step, no
CDN, no network at query time — it reuses TanitResim's proven patterns and
design language (dark slate, gold/cyan/magenta accents, wordmark, card grid,
"a legend on every canvas").

```
tanitad/scena/
  parse.py             SCENARIO_DATABASE.md -> Scenario dicts (+ scenarios.json)
  vector.py            local semantic index (MiniLM if present, else hashing TF-IDF)
  static/index.html    SPA shell (TanitScena wordmark, #app root)
  static/style.css     TanitAD design language + search / detail / stepper styles
  static/app.js        vanilla JS + canvas: search home + scenario detail + BEV glyphs
scripts/scena_app.py   FastAPI single-port server (build_app factory + CLI)
tests/test_scena.py    parser + both vector paths + FastAPI TestClient + portability
```

## Pipeline

1. **`parse.py`** — section-heading / regex driven, **fail-soft per field**
   (a per-scenario `parse_warnings` list; one malformed entry never crashes the
   parse). Each `## SC-xx — <title> [W-xx] ★★★` block yields: `id`, `title`,
   `w_code`/`family`/`stars`/`headline`/`is_new` (+ a display `tags` list),
   `evidence_label` (FACT/CLAIM/INFER, compound labels collapse to primary),
   `opponent_evidence`, `description`, `correct_behavior`, `mechanism`,
   `data_sources` (`{kind, ref, status, link, replay}` — kind ∈
   CARLA/Cosmos/HF/nuScenes/comma2k19/NuRec/dashcam/other; `link` resolves an
   explicit URL or a real dataset landing page / HF search, never a fabricated
   repo id), `metric_hooks` (+ full text), `lifecycle_stage` (first canonical
   ladder stage named in Status), `status_text`, `evidence_links`. Emits a
   portable `scenarios.json` (source stored **basename-only** — no absolute path
   leaks).

2. **`vector.py`** — self-contained cosine search over an in-memory `float32`
   matrix (unit rows, so dot = cosine). Two embedder paths:
   - **MiniLM** (`sentence-transformers` `all-MiniLM-L6-v2`) **iff it imports**
     in the venv (~80 MB, downloads once at build time). Best quality.
   - **Hashing TF-IDF fallback** — pure-numpy hashing vectoriser (word 1/2-grams
     + char 3/4/5-grams, **md5-bucketed so it is deterministic across
     processes/machines**, sublinear TF × corpus IDF, L2-normalised). Needs only
     numpy, so the feature and its tests stay green fully **offline**.

   The chosen embedder is recorded in the index metadata and persisted with the
   vectors to `vectors.npz`. `search(query, k)` → ranked `(scenario_id, score)`.

3. **`scripts/scena_app.py`** — FastAPI, one plain-HTTP port (RunPod-proxy
   friendly, same rationale as TanitResim). `build_app(db_md, static, cache_dir,
   prefer)` is the importable factory. The index builds **lazily on first
   search** (and persists to `vectors.npz`); `/api/reindex` forces a full
   re-parse + re-embed for live DB edits.

## Embedder note (MiniLM vs fallback)

`prefer` ∈ `auto` (default) | `minilm` | `hashing`. `auto` uses MiniLM when
`sentence-transformers` is importable, else the deterministic hashing fallback.
On a box without `sentence-transformers` (the current dev/pod default) the app
runs fully on the hashing fallback — a cone query still ranks **SC-01** top,
stop-arm → **SC-04**, fog/glare → **SC-05**, red-light → **SC-14**. Install
`sentence-transformers` and reindex (`prefer=minilm` or just `auto`) to upgrade
ranking quality; nothing else changes. `/api/meta` reports the live embedder and
whether MiniLM is available.

## The two views

**Home = search-first browser.** A prominent semantic search box (live queries
to `/api/search`, ranked result cards with score bars), filter chips (lifecycle
stage · evidence label · ★ rating · data-source kind), and a responsive card
grid: id, title, ★, a color-coded evidence chip (FACT = green / CLAIM = amber /
INFER = grey), a lifecycle badge, W-code / headline / new tags, and data-source
kind chips.

**Detail = one scenario in full.** Opponent evidence (with label + source
links), description + a highlighted **correct-behavior** callout + the TanitAD
mechanism, a **schematic top-down BEV canvas** hand-authored per scenario family
(cone-taper for SC-01/09, stop-arm for SC-04, occlusion for SC-02/03, degraded
visibility for SC-05, stationary lead for SC-13, red-light for SC-14, … + a
generic fallback glyph) **with a legend on the canvas**, the **metric hooks**,
the **lifecycle stepper** (catalogued → … → excellence-proven, current stage
highlighted), and the **Dataset links** section — one clickable card per data
source (kind badge + ref + status + a "get dataset ▸" link; an "open in replay
▸" affordance when a source references a `.rrd` / TanitResim bundle).

URL-hash routing (`#/s/<id>` detail, `#/q/<query>` search) makes any view
shareable. `/` focuses search, `Esc` clears/backs out.

## Serve on a pod (RunPod)

Single plain-HTTP port — proxy-friendly:

```
python scripts/scena_app.py --port 8890 \
    --db-md "TanitAD Research Hub/Opponent Analyzer/SCENARIO_DATABASE.md"
```

Then open the proxied URL:

```
https://<pod-id>-8890.proxy.runpod.net
```

Expose port **8890** on the pod's HTTP proxy. Everything (SPA, API) is
same-origin under that one port. FastAPI + uvicorn only; no browser-side CDN or
external assets. Flags: `--static` (override SPA dir), `--cache-dir` (index
location, default `<db dir>/.scena_cache`), `--prefer auto|minilm|hashing`.

Endpoints: `GET /` (SPA) · `GET /static/*` · `GET /api/scenarios` (summaries) ·
`GET /api/scenario/{id}` (full; id guarded `^SC-\d+$`) · `GET /api/search?q=&k=`
(ranked) · `POST|GET /api/reindex` · `GET /api/meta`.

## Staying in sync with SCENARIO_DATABASE.md

The database is authored by hand and edited continuously (Opponent Analyzer adds
entries Friday, Data Engineering fills data-source rows Tuesday, …). TanitScena
re-parses the live file on startup and on demand:

- **`POST /api/reindex`** (or `GET`) re-reads the markdown, re-parses, rewrites
  `scenarios.json`, and re-embeds — no restart needed after a DB edit.
- Or just restart the server; it parses on boot.

The parser is tolerant of markdown variation and reports coverage + any
`parse_warnings` (also surfaced in `/api/meta`), so a new or reworded entry is
picked up without code changes and any field it can't read is flagged rather
than silently dropped.

## Tests

`tests/test_scena.py` (CPU-only, offline-safe): the parser runs against the
**real** `SCENARIO_DATABASE.md` (all SC-xx parse; SC-01 keeps its Waymo FACT
evidence, ★★★, W-01, a resolvable data source, its `closure_incursion_m` hook);
the hashing fallback ranks SC-01 top for a cone query, is byte-for-byte
deterministic, and round-trips through `vectors.npz`; the MiniLM path is
asserted when importable (skipped otherwise); a `fastapi.testclient.TestClient`
drives list / detail (+404 + id guard) / search / reindex / meta / static
serving; and `scenarios.json` is checked portable (no absolute filesystem path,
source basename-only).
