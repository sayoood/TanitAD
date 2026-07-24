"""TanitResim server — single-port FastAPI host for session bundles.

Serves the vanilla-JS SPA and the session bundles produced by
``replay_app.py --mode export`` over ONE plain-HTTP port (RunPod-proxy
friendly — unlike the rerun gRPC-web stream, which the proxy choked on):

    GET /                       the SPA (tanitad/resim/static/index.html)
    GET /static/*               SPA assets (app.js, style.css, vendored)
    GET /api/sessions           list bundles under --sessions-root (summaries)
    GET /api/session/{id}       one bundle's session.json
    GET /frames/{id}/{name}     a bundle camera frame (path-traversal guarded)

Run on a pod::

    python scripts/resim_app.py --port 8888 --sessions-root /workspace/resim

then open ``https://<pod-id>-8888.proxy.runpod.net``. ``build_app`` is the
importable factory the tests drive with ``fastapi.testclient.TestClient``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Bind to THIS checkout's tanitad (worktree/pod editable-install skew guard —
# same rationale as replay_app.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from tanitad.resim.export import static_dir as _default_static_dir


def _bundle_dirs(root: Path) -> list[Path]:
    """Sorted bundle directories under ``root`` (each has a session.json)."""
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and (d / "session.json").is_file())


def _session_summary(bundle: Path) -> dict:
    """Light card for the session picker (no per-step payload)."""
    meta = json.loads((bundle / "session.json").read_text(
        encoding="utf-8")).get("meta", {})
    return {
        "id": bundle.name,
        "session_name": meta.get("session_name", bundle.name),
        "created": meta.get("created"),
        "n_episodes": len(meta.get("episodes", [])),
        "corpora": meta.get("corpora", []),
        "arms": [{"name": a.get("name"), "color": a.get("color"),
                  "ade": a.get("ade"), "latency_p50": a.get("latency_p50")}
                 for a in meta.get("arms", [])],
    }


def build_app(sessions_root: str | Path,
              static: str | Path | None = None) -> FastAPI:
    """FastAPI app serving the SPA + bundles under ``sessions_root``."""
    root = Path(sessions_root).resolve()
    static_path = Path(static).resolve() if static \
        else _default_static_dir()
    if not (static_path / "index.html").is_file():
        raise FileNotFoundError(
            f"TanitResim static SPA not found at {static_path} "
            f"(expected index.html/app.js/style.css)")

    app = FastAPI(title="TanitResim", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(static_path)),
              name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((static_path / "index.html").read_text(
            encoding="utf-8"))

    @app.get("/api/sessions")
    def sessions() -> JSONResponse:
        return JSONResponse([_session_summary(d) for d in _bundle_dirs(root)])

    @app.get("/api/session/{sid}")
    def session(sid: str) -> JSONResponse:
        bundle = (root / sid)
        if bundle.parent != root or not (bundle / "session.json").is_file():
            raise HTTPException(status_code=404, detail=f"no session {sid!r}")
        return JSONResponse(json.loads(
            (bundle / "session.json").read_text(encoding="utf-8")))

    @app.get("/frames/{sid}/{name}")
    def frame(sid: str, name: str) -> FileResponse:
        base = (root / sid / "frames").resolve()
        target = (base / name).resolve()
        # Path-traversal guard: the resolved file must live under the bundle's
        # frames dir and the session id must be a direct child of the root.
        if (root / sid).resolve().parent != root \
                or base not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(str(target), media_type="image/jpeg")

    app.state.sessions_root = root
    app.state.static_path = static_path
    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TanitResim single-port server")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--sessions-root", default=None,
                    help="directory of exported session bundles "
                         "(default: a temp dir when --demo is set)")
    ap.add_argument("--static", default=None,
                    help="override the SPA static dir (default: bundled)")
    ap.add_argument("--demo", action="store_true",
                    help="generate a synthetic full-viz-standard bundle and "
                         "serve it — a zero-dependency, pod-free demo")
    args = ap.parse_args(argv)

    if not args.sessions_root and not args.demo:
        ap.error("--sessions-root is required (or pass --demo)")

    if args.sessions_root:
        root = Path(args.sessions_root)
    else:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix="tanitresim-demo-"))
    root.mkdir(parents=True, exist_ok=True)

    if args.demo:
        from tanitad.resim.sample import make_sample_bundle
        make_sample_bundle(root / "demo-synthetic")
        print(f"[resim] --demo: wrote a synthetic bundle to {root}", flush=True)

    app = build_app(root, static=args.static)
    n = len(_bundle_dirs(root))
    print(f"[resim] serving {n} bundle(s) from {root} on "
          f"http://{args.host}:{args.port}  (open / for the app)", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
