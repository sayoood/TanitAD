"""TanitScena — TanitAD's scenario-database web app.

Turns the Opponent-Analyzer :file:`SCENARIO_DATABASE.md` (the SC-01..SC-14
"dumb situation" catalogue) into a searchable, browsable, visual single-port
app with **local** semantic (vector) search — no build step, no CDN, no
network at query time.

Pipeline: :mod:`tanitad.scena.parse` turns the Markdown into portable
``Scenario`` dicts (+ ``scenarios.json``); :mod:`tanitad.scena.vector` embeds
them (MiniLM if importable, else a deterministic pure-numpy hashing TF-IDF
fallback) into a cosine index; ``stack/scripts/scena_app.py`` serves the
vanilla-JS SPA in :mod:`tanitad.scena.static` over one FastAPI port.

Design language + view descriptions + pod deploy live in
``tanitad/scena/README.md``. Reuses TanitResim's proven patterns.
"""

from pathlib import Path

from tanitad.scena.parse import (STAGES, parse_file, parse_scenarios, to_index,
                                 write_scenarios_json)
from tanitad.scena.vector import HashingTfidf, VectorIndex, doc_text


def static_dir() -> Path:
    """Absolute path to the bundled SPA static assets (index/app/style)."""
    return Path(__file__).resolve().parent / "static"


__all__ = ["STAGES", "parse_file", "parse_scenarios", "to_index",
           "write_scenarios_json", "HashingTfidf", "VectorIndex", "doc_text",
           "static_dir"]
