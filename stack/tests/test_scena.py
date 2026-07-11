"""TanitScena tests — parser + local vector search + FastAPI single-port server.

CPU-only and **offline-safe**: the deterministic hashing TF-IDF fallback needs
only numpy, so every core assertion runs with no network. The MiniLM path is
exercised only when ``sentence-transformers`` is importable (skipped, not
failed, otherwise). Pins:

(a) parser — all SC-xx entries parse off the REAL database; SC-01 carries its
    Waymo FACT evidence, ★★★, W-01, a resolvable data source and its
    ``closure_incursion_m`` hook; fail-soft records warnings, never raises;
(b) vector — a cone query ranks SC-01 top on BOTH embedder paths; the hashing
    fallback is byte-for-byte deterministic and round-trips through vectors.npz;
(c) server — index served, scenarios list, detail (+404 + id guard), search
    (ranked, SC-01 for a lane-closure query), reindex, meta;
(d) portability — scenarios.json embeds no absolute filesystem path (URLs ok),
    source stored basename-only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.scena import (VectorIndex, parse, vector, write_scenarios_json)
from tanitad.scena.parse import STAGES

DB = (Path(__file__).resolve().parents[2] / "TanitAD Research Hub" /
      "Opponent Analyzer" / "SCENARIO_DATABASE.md")

pytestmark = pytest.mark.skipif(
    not DB.is_file(), reason=f"SCENARIO_DATABASE.md not found at {DB}")

CONE_Q = "construction cones lane closure"


@pytest.fixture(scope="module")
def scenarios():
    return parse.parse_file(DB)


# ---------- (a) parser --------------------------------------------------------

def test_parses_all_scenarios(scenarios):
    ids = [s["id"] for s in scenarios]
    assert len(scenarios) >= 14
    assert len(ids) == len(set(ids))                 # ids unique
    for want in ("SC-01", "SC-02", "SC-04", "SC-05", "SC-14"):
        assert want in ids


def test_sc01_fields(scenarios):
    s = next(x for x in scenarios if x["id"] == "SC-01")
    tl = s["title"].lower()
    assert "cone" in tl or "work-zone" in tl
    assert s["evidence_label"] == "FACT"
    assert "waymo" in s["opponent_evidence"].lower()
    assert s["stars"] == 3
    assert s["headline"] is True
    assert s["w_code"] == "W-01"
    assert s["lifecycle_stage"] in STAGES
    assert s["correct_behavior"]                      # non-empty
    assert "closure_incursion_m" in s["metric_hooks"]
    assert len(s["data_sources"]) >= 1
    d0 = s["data_sources"][0]
    assert set(d0) >= {"kind", "ref", "status", "link"}
    assert any(d.get("link") for d in s["data_sources"])   # resolvable link


def test_evidence_labels_and_lifecycle(scenarios):
    by = {s["id"]: s for s in scenarios}
    assert by["SC-04"]["evidence_label"] == "CLAIM"
    assert by["SC-12"]["evidence_label"] == "INFER"
    assert by["SC-01"]["lifecycle_stage"] == "live-measured"
    assert by["SC-04"]["lifecycle_stage"] == "spec-drafted"
    assert by["SC-05"]["lifecycle_stage"] == "data-sourced"
    for s in scenarios:
        assert s["lifecycle_stage"] is None or s["lifecycle_stage"] in STAGES


def test_data_source_kinds_detected(scenarios):
    by = {s["id"]: s for s in scenarios}
    kinds01 = {d["kind"] for d in by["SC-01"]["data_sources"]}
    assert "CARLA" in kinds01 and "Cosmos" in kinds01
    # SC-05 verified-available Cosmos weather + nuScenes held-out probe
    kinds05 = {d["kind"] for d in by["SC-05"]["data_sources"]}
    assert "Cosmos" in kinds05 and "nuScenes" in kinds05


def test_failsoft_never_raises_on_malformed():
    md = "## SC-99 — Broken entry\n\njust prose, no bullet fields at all\n"
    scs = parse.parse_scenarios(md)
    assert len(scs) == 1 and scs[0]["id"] == "SC-99"
    assert scs[0]["parse_warnings"]                   # recorded, not raised


# ---------- (b) local vector search ------------------------------------------

def test_hashing_search_ranks_sc01_top(scenarios):
    idx = VectorIndex.build(scenarios, prefer="hashing")
    assert idx.embedder == "hashing-tfidf"
    ids = [i for i, _ in idx.search(CONE_Q, k=5)]
    assert ids[0] == "SC-01"
    assert "SC-01" in ids[:3]


def test_hashing_is_deterministic(scenarios):
    a = VectorIndex.build(scenarios, prefer="hashing")
    b = VectorIndex.build(scenarios, prefer="hashing")
    assert np.array_equal(a.matrix, b.matrix)
    assert a.search(CONE_Q, 5) == b.search(CONE_Q, 5)


def test_index_save_load_roundtrip(scenarios, tmp_path):
    idx = VectorIndex.build(scenarios, prefer="hashing")
    p = tmp_path / "vectors.npz"
    idx.save(p)
    ld = VectorIndex.load(p)
    assert ld.embedder == idx.embedder
    assert set(ld.ids) == set(idx.ids)
    assert ld.search(CONE_Q, 5) == idx.search(CONE_Q, 5)


def test_other_queries_rank_expected(scenarios):
    idx = VectorIndex.build(scenarios, prefer="hashing")
    assert idx.search("school bus stop arm child", 3)[0][0] == "SC-04"
    assert idx.search("fog glare rain reduced visibility", 3)[0][0] == "SC-05"
    assert idx.search("red light running signal", 3)[0][0] == "SC-14"


def test_empty_query_returns_nothing(scenarios):
    idx = VectorIndex.build(scenarios, prefer="hashing")
    assert idx.search("", 5) == []
    assert idx.search("   ", 5) == []


def test_auto_prefer_matches_availability(scenarios):
    idx = VectorIndex.build(scenarios, prefer="auto")
    want = "minilm" if vector._minilm_available() else "hashing-tfidf"
    assert idx.embedder == want


@pytest.mark.skipif(not vector._minilm_available(),
                    reason="sentence-transformers not installed")
def test_minilm_search_ranks_sc01(scenarios):
    idx = VectorIndex.build(scenarios, prefer="minilm")
    assert idx.embedder == "minilm"
    ids = [i for i, _ in idx.search(CONE_Q, k=5)]
    assert "SC-01" in ids[:3]


# ---------- (c) FastAPI server -----------------------------------------------

@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    import scena_app
    app = scena_app.build_app(DB, cache_dir=tmp_path, prefer="hashing")
    return TestClient(app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "TanitScena" in r.text


def test_static_asset_served(client):
    r = client.get("/static/app.js")
    assert r.status_code == 200 and "TanitScena" in r.text


def test_scenarios_list(client):
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 14
    ids = {s["id"] for s in body}
    assert "SC-01" in ids
    s = next(x for x in body if x["id"] == "SC-01")
    assert set(s) >= {"id", "title", "stars", "evidence_label",
                      "lifecycle_stage", "data_source_kinds"}


def test_scenario_detail_and_guards(client):
    r = client.get("/api/scenario/SC-01")
    assert r.status_code == 200 and r.json()["evidence_label"] == "FACT"
    assert client.get("/api/scenario/SC-999").status_code == 404
    assert client.get("/api/scenario/nope").status_code == 404
    # id that is not ^SC-\d+$ is rejected by the guard (path-traversal-safe)
    assert client.get("/api/scenario/..%2f..%2fsecret").status_code in (404, 400)


def test_search_endpoint_ranks_sc01(client):
    r = client.get("/api/search", params={"q": "construction lane closure", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["embedder"] == "hashing-tfidf"
    ids = [x["id"] for x in body["results"]]
    assert ids and ids[0] == "SC-01" and "SC-01" in ids[:3]
    scores = [x["score"] for x in body["results"]]
    assert scores == sorted(scores, reverse=True)     # ranked descending


def test_search_empty_query(client):
    body = client.get("/api/search", params={"q": ""}).json()
    assert body["results"] == []


def test_reindex(client):
    r = client.get("/api/reindex")
    assert r.status_code == 200 and r.json()["n"] >= 14
    # still searchable after a live reindex
    after = client.get("/api/search", params={"q": "red light"}).json()
    assert after["results"] and after["results"][0]["id"] == "SC-14"


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["n"] >= 14
    assert m["stages"] == list(STAGES)
    assert "n_warnings" in m


def test_build_app_fails_loud_on_missing_db(tmp_path):
    import scena_app
    with pytest.raises(FileNotFoundError):
        scena_app.build_app(tmp_path / "nope.md")


# ---------- (d) portability ---------------------------------------------------

_URL = re.compile(r"https?://\S+")


def test_scenarios_json_is_portable(tmp_path, scenarios):
    out = tmp_path / "scenarios.json"
    payload = write_scenarios_json(scenarios, out, "SCENARIO_DATABASE.md")
    assert payload["source"] == "SCENARIO_DATABASE.md"        # basename only
    text = out.read_text(encoding="utf-8")
    # the machine-specific absolute DB path (and its parent) must not leak
    assert str(DB) not in text
    assert str(DB).replace("\\", "/") not in text
    assert str(DB.parent) not in text
    # no absolute filesystem paths — strip legitimate URLs first, then look for
    # a Windows drive-letter path (the real leak risk on this box).
    stripped = _URL.sub("", text)
    assert not re.search(r"[A-Za-z]:[\\/]", stripped), "drive-letter path leaked"
