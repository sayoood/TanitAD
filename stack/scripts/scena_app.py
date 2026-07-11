"""TanitScena server — single-port FastAPI host for the scenario database.

Serves the vanilla-JS SPA and a local semantic-search API over the
Opponent-Analyzer ``SCENARIO_DATABASE.md`` on ONE plain-HTTP port
(RunPod-proxy friendly, same rationale as TanitResim)::

    GET  /                     the SPA (tanitad/scena/static/index.html)
    GET  /static/*             SPA assets (app.js, style.css)
    GET  /api/scenarios        list scenarios (summary fields)
    GET  /api/scenario/{id}    one scenario (full; id guarded ^SC-\\d+$)
    GET  /api/search?q=&k=     local vector search -> ranked scenarios + scores
    POST /api/reindex          re-parse the DB + re-embed (for live DB edits)
    GET  /api/meta             embedder + counts + parse-warning summary

Run on a pod::

    python scripts/scena_app.py --port 8890 \\
        --db-md "TanitAD Research Hub/Opponent Analyzer/SCENARIO_DATABASE.md"

then open ``https://<pod-id>-8890.proxy.runpod.net``. ``build_app`` is the
importable factory the tests drive with ``fastapi.testclient.TestClient``.

The index (scenarios.json + vectors.npz) is written under ``--cache-dir``
(default ``<db-md dir>/.scena_cache``) and rebuilt lazily on first search if
absent; ``/api/reindex`` forces a full re-parse + re-embed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Bind to THIS checkout's tanitad (worktree/pod editable-install skew guard —
# same rationale as resim_app.py / replay_app.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tanitad.scena import (VectorIndex, parse_file, static_dir as _static_dir,
                           to_index, write_scenarios_json)

_ID_RE = re.compile(r"SC-\d+")


def _summary(s: dict) -> dict:
    """Light card payload for list + search results (no full-text bloat)."""
    return {
        "id": s.get("id"),
        "title": s.get("title"),
        "stars": s.get("stars", 0),
        "w_code": s.get("w_code"),
        "family": s.get("family", False),
        "headline": s.get("headline", False),
        "is_new": s.get("is_new", False),
        "tags": s.get("tags", []),
        "evidence_label": s.get("evidence_label"),
        "lifecycle_stage": s.get("lifecycle_stage"),
        "description": s.get("description", ""),
        "data_source_kinds": sorted({d.get("kind", "other")
                                     for d in s.get("data_sources", [])}),
        "n_data_sources": len(s.get("data_sources", [])),
        "metric_hooks": s.get("metric_hooks", []),
    }


def _reload(app: FastAPI) -> None:
    """(Re)parse the DB markdown, refresh state, emit scenarios.json, drop the
    stale in-memory index so the next search rebuilds it."""
    st = app.state
    scenarios = parse_file(st.db_md)
    st.cache_dir.mkdir(parents=True, exist_ok=True)
    write_scenarios_json(scenarios, st.cache_dir / "scenarios.json",
                         Path(st.db_md).name)
    st.scenarios = scenarios
    st.by_id = {s["id"]: s for s in scenarios}
    st.index = None


def _ensure_index(app: FastAPI, force: bool = False) -> VectorIndex:
    """Return a vector index over the current scenarios, building lazily.

    On build it is cached in memory and persisted to ``vectors.npz``; a cached
    npz is reused only if its ids still match the current DB (else rebuilt).
    """
    st = app.state
    if force:
        st.index = None
    if st.index is not None:
        return st.index

    npz = st.cache_dir / "vectors.npz"
    cur_ids = {s["id"] for s in st.scenarios}
    if not force and npz.is_file():
        try:
            idx = VectorIndex.load(npz)
            if set(idx.ids) == cur_ids:
                st.index = idx
                return idx
        except Exception:
            pass                                  # corrupt/stale -> rebuild

    idx = VectorIndex.build(st.scenarios, prefer=st.prefer)
    st.cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        idx.save(npz)
    except Exception:
        pass                                      # read-only cache is non-fatal
    st.index = idx
    return idx


def build_app(db_md: str | Path, static: str | Path | None = None,
              cache_dir: str | Path | None = None,
              prefer: str = "auto") -> FastAPI:
    """FastAPI app serving the SPA + scenario API over ``db_md``.

    ``db_md``     path to SCENARIO_DATABASE.md (parsed on startup + reindex).
    ``static``    override the SPA static dir (default: bundled scena/static).
    ``cache_dir`` where scenarios.json + vectors.npz live (default:
                  ``<db_md dir>/.scena_cache``).
    ``prefer``    embedder preference: auto | minilm | hashing.
    """
    db_path = Path(db_md).resolve()
    if not db_path.is_file():
        raise FileNotFoundError(f"SCENARIO_DATABASE.md not found at {db_path}")
    static_path = Path(static).resolve() if static else _static_dir()
    if not (static_path / "index.html").is_file():
        raise FileNotFoundError(
            f"TanitScena static SPA not found at {static_path} "
            f"(expected index.html/app.js/style.css)")
    cache = Path(cache_dir).resolve() if cache_dir \
        else db_path.parent / ".scena_cache"

    app = FastAPI(title="TanitScena", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    app.state.db_md = db_path
    app.state.cache_dir = cache
    app.state.static_path = static_path
    app.state.prefer = prefer
    app.state.index = None
    _reload(app)

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((static_path / "index.html").read_text(
            encoding="utf-8"))

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        idx_name = app.state.index.embedder if app.state.index else None
        payload = to_index(app.state.scenarios, Path(app.state.db_md).name)
        return JSONResponse({
            "source": payload["source"],
            "generated": payload["generated"],
            "stages": payload["stages"],
            "n": payload["n"],
            "n_warnings": payload["n_warnings"],
            "warnings": payload["warnings"],
            "embedder": idx_name,
            "minilm_available": _minilm_flag(),
        })

    @app.get("/api/scenarios")
    def scenarios() -> JSONResponse:
        return JSONResponse([_summary(s) for s in app.state.scenarios])

    @app.get("/api/scenario/{sid}")
    def scenario(sid: str) -> JSONResponse:
        if not _ID_RE.fullmatch(sid) or sid not in app.state.by_id:
            raise HTTPException(status_code=404, detail=f"no scenario {sid!r}")
        return JSONResponse(app.state.by_id[sid])

    @app.get("/api/search")
    def search(q: str = "", k: int = 8) -> JSONResponse:
        idx = _ensure_index(app)
        k = max(1, min(int(k), 50))
        ranked = idx.search(q, k=k)
        results = []
        for sid, score in ranked:
            s = app.state.by_id.get(sid)
            if s is None:
                continue
            item = _summary(s)
            item["score"] = score
            results.append(item)
        return JSONResponse({"query": q, "k": k, "embedder": idx.embedder,
                             "n_results": len(results), "results": results})

    @app.api_route("/api/reindex", methods=["POST", "GET"])
    def reindex() -> JSONResponse:
        _reload(app)
        idx = _ensure_index(app, force=True)
        payload = to_index(app.state.scenarios, Path(app.state.db_md).name)
        return JSONResponse({"reindexed": True, "n": payload["n"],
                             "embedder": idx.embedder,
                             "n_warnings": payload["n_warnings"],
                             "warnings": payload["warnings"]})

    return app


def _minilm_flag() -> bool:
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitScena single-port server")
    ap.add_argument("--port", type=int, default=8890)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--db-md", required=True,
                    help="path to SCENARIO_DATABASE.md")
    ap.add_argument("--static", default=None,
                    help="override the SPA static dir (default: bundled)")
    ap.add_argument("--cache-dir", default=None,
                    help="index cache dir (default: <db dir>/.scena_cache)")
    ap.add_argument("--prefer", default="auto",
                    choices=["auto", "minilm", "hashing"],
                    help="embedder preference (default: auto)")
    args = ap.parse_args(argv)

    app = build_app(args.db_md, static=args.static, cache_dir=args.cache_dir,
                    prefer=args.prefer)
    n = len(app.state.scenarios)
    idx = _ensure_index(app)                      # warm the index at startup
    print(f"[scena] {n} scenarios from {Path(args.db_md).name} · "
          f"embedder={idx.embedder} · serving on "
          f"http://{args.host}:{args.port}  (open / for the app)", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
