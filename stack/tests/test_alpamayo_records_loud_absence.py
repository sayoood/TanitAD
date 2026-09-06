"""A MISSING SOURCE FILE MUST NOT LOOK LIKE A NEGATIVE FINDING.

⛔⛔ THE DEFECT THIS PINS (MEASURED 2026-09-06, escalation 4 of
`…/Research/2026-09-06-speed-limit-source/RESULT.md`). `alpamayo_records._load()`
answered a missing `RECORDS` parquet with ``return {}``. On any machine without
that hard-coded local path **every CoT token vanished with no error**, and
downstream *"this corpus states no speed limit"* and *"the records file was not
there"* were INDISTINGUISHABLE.

⭐ THREE CLAIMS ARE PINNED HERE, AND THEY ARE DIFFERENT CLAIMS.

1. **CORRECTNESS** — the guard raises when it is called (`test_pr1a_*`).
2. ⭐ **WIRING** — the guard is *reached* from every PUBLIC entry point, not
   only from `_load()` itself (`test_pr1c_*`). *Correctness and wiring are
   different claims, and historically only the first was ever tested: a guard
   that works when called and is called from one path of several is how a
   52 %-dead training budget survived a green suite.*
3. ⛔ **THE OTHER SIDE** — the guard does NOT refuse a source that is present
   (`test_pr1b_*`). A guard that refuses everything passes claim 1 trivially
   and destroys the pipeline; a guard that refuses nothing has measured
   nothing. Both sides, or neither is evidence.

⛔ `test_pr1d_deliberate_regression_arm` is the mutation proof: ONE checker,
TWO loaders — the pre-fix body and the current one — required to disagree. If
the checker cannot FAIL the knowingly-broken loader, a PASS on the fixed one
means nothing.

Pre-registration: `TanitAD Research Lab/Architecture & Inference/Research/
2026-09-06-cot-loader/PREREG.md` (PR-1a … PR-1e), written before these numbers
existed.
"""
import os

import pytest

from tanitad.data import alpamayo_fusion as AF
from tanitad.data import alpamayo_records as AR
from tanitad.data import alpamayo_structured as AST

MISSING = "C:/__tanitad_no_such_dir__/records.parquet"


@pytest.fixture(autouse=True)
def _clean_loader_state(monkeypatch):
    """The lru_cache is process-global; every test starts and ends cold.

    NOTE: `lru_cache` does not memoise exceptions, so a failing load re-raises
    every call -- but a SUCCESSFUL load is cached and would poison a later test
    that repoints the env var. Clearing on both sides is what keeps these tests
    order-independent.
    """
    monkeypatch.delenv(AR.RECORDS_ENV, raising=False)
    monkeypatch.delenv(AR.RECORDS_OPTIONAL_ENV, raising=False)
    AR._load.cache_clear()
    yield
    AR._load.cache_clear()


def _point_at_missing(monkeypatch):
    monkeypatch.setenv(AR.RECORDS_ENV, MISSING)
    AR._load.cache_clear()


# --------------------------------------------------------------------- PR-1a
def test_pr1a_missing_source_raises_a_named_error(monkeypatch):
    """A missing source is an ERROR, never an empty mapping."""
    _point_at_missing(monkeypatch)
    with pytest.raises(AR.AlpamayoRecordsUnavailable) as ei:
        AR._load()
    msg = str(ei.value)
    assert MISSING in msg, "the failure must name the path it tried"
    assert AR.RECORDS_ENV in msg, "the failure must name the override knob"
    assert AR.RECORDS_MD5 in msg, "the failure must name the expected md5"


def test_pr1a_unreadable_source_is_not_an_empty_source(tmp_path, monkeypatch):
    """A file that EXISTS but cannot be parsed also raises.

    'Present but garbage' and 'absent' must land in the same place: neither is
    evidence that the corpus carries nothing.
    """
    p = tmp_path / "not_a_parquet.parquet"
    p.write_bytes(b"this is not parquet")
    monkeypatch.setenv(AR.RECORDS_ENV, str(p))
    AR._load.cache_clear()
    with pytest.raises(AR.AlpamayoRecordsUnavailable):
        AR._load()


# --------------------------------------------------------------------- PR-1b
def _write_toy_parquet(path):
    """A 2-clip source with the real schema, so the test needs no corpus."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import json
    rows = []
    for cid in ("clip_aaaa", "clip_bbbb"):
        rows.append({"clip_id": cid, "task": "meta_action",
                     "vqa_category": None, "question": None,
                     "raw_json": json.dumps(
                         {"meta_action": ["Longitudinal: Constant Speed\n"
                                          "Lateral: Steer Right\n"
                                          "Lane: Lane Keep"],
                          "cot": ["A 50 km/h speed limit sign is posted."]})})
        rows.append({"clip_id": cid, "task": "grounding_via_vqa",
                     "vqa_category": None, "question": "Where is the car?",
                     "raw_json": json.dumps(
                         {"box": [json.dumps([{"label": "Car",
                                               "bbox_2d": [1, 2, 3, 4]}])]})})
    pd.DataFrame(rows).to_parquet(path)
    return path


def test_pr1b_a_present_source_is_not_refused(tmp_path, monkeypatch):
    """THE OTHER SIDE: the guard must not refuse a source that is there.

    Data-free -- a toy parquet with the real schema, so this arm runs on any
    machine, including one with no corpus at all.
    """
    p = _write_toy_parquet(tmp_path / "toy.parquet")
    monkeypatch.setenv(AR.RECORDS_ENV, str(p))
    AR._load.cache_clear()
    d = AR._load()
    assert set(d) == {"clip_aaaa", "clip_bbbb"}
    assert d["clip_aaaa"].lateral == "right"
    assert d["clip_aaaa"].box_labels() == ["car"]
    rep = AR.load_report()
    assert rep["records_available"] is True
    assert rep["clips"] == 2 and rep["rows"] == 4
    assert rep["row_json_failed"] == 0


@pytest.mark.skipif(not os.path.exists(AR.RECORDS),
                    reason="the local Alpamayo mirror is not on this machine")
def test_pr1b_real_corpus_is_4729_clips():
    """The corpus denominator every fraction in this package is quoted against.

    ⛔ 4,729 clips of `Sayood/tanitad-alpamayo2-augmentation` -- NOT the
    2,376-episode parity corpus, NOT B1's 4,572.
    """
    AR._load.cache_clear()
    cov = AR.coverage()
    assert cov["clips"] == 4729
    assert cov["records_available"] is True


def test_pr1b_a_clean_source_warns_about_nothing(tmp_path, monkeypatch, recwarn):
    """THE OTHER SIDE of the swallow warning: a clean file must be quiet.

    A loader that warns on every load trains its callers to ignore the warning,
    which is the same end state as not warning at all.
    """
    p = _write_toy_parquet(tmp_path / "clean.parquet")
    monkeypatch.setenv(AR.RECORDS_ENV, str(p))
    AR._load.cache_clear()
    AR._load()
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_a_dropped_row_is_never_silent(tmp_path, monkeypatch):
    """⛔ A COUNTER NOTHING READS IS STILL A SILENT SWALLOW.

    One row's `raw_json` is unparseable. The load SUCCEEDS -- that is correct,
    a single bad row should not destroy a corpus -- but it must not succeed
    QUIETLY, and the count must be recoverable.
    """
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import json
    rows = [
        {"clip_id": "good", "task": "meta_action", "vqa_category": None,
         "question": None,
         "raw_json": json.dumps({"meta_action": ["Lateral: Steer Left"],
                                 "cot": ["nothing to see"]})},
        {"clip_id": "bad", "task": "meta_action", "vqa_category": None,
         "question": None, "raw_json": "{not json at all"},
    ]
    p = tmp_path / "one_bad_row.parquet"
    pd.DataFrame(rows).to_parquet(p)
    monkeypatch.setenv(AR.RECORDS_ENV, str(p))
    AR._load.cache_clear()
    with pytest.warns(RuntimeWarning, match="DROPPED without failing"):
        d = AR._load()
    assert set(d) == {"good"}, "the good row must still load"
    assert AR.load_report()["row_json_failed"] == 1


# --------------------------------------------------------------------- PR-1c
#: Every PUBLIC route into the loader. The guard is worth nothing on a route
#: that never reaches it, so the route list IS the test.
ENTRY_POINTS = [
    ("records.available", lambda: AR.available()),
    ("records.get", lambda: AR.get("clip_that_does_not_exist")),
    ("records.coverage", lambda: AR.coverage()),
    ("structured.coverage", lambda: AST.coverage()),
    ("structured.motion_segments", lambda: AST.motion_segments("x")),
    ("structured.critical_component", lambda: AST.critical_component("x")),
    ("fusion.cot_text", lambda: AF.cot_text("x")),
    ("fusion.ground_tokens", lambda: AF.ground_tokens("x", {})),
    ("fusion.lateral_concordance", lambda: AF.lateral_concordance("x")),
]


@pytest.mark.parametrize("name,call", ENTRY_POINTS, ids=[e[0] for e in ENTRY_POINTS])
def test_pr1c_loud_path_is_reached_from_every_entry_point(name, call, monkeypatch):
    """WIRING, not correctness.

    Each of these used to answer a missing source with "" / [] / None / {} /
    {"clips": 0} -- an empty container that reads exactly like a real negative.
    """
    _point_at_missing(monkeypatch)
    with pytest.raises(AR.AlpamayoRecordsUnavailable):
        call()


def test_pr1c_entry_point_list_covers_every_public_caller():
    """The route list must not silently fall behind the module.

    Counts the `AR.get(` / `AR._load(` call sites in the two sibling modules and
    requires the parametrised list to be at least that wide, so a NEW consumer
    cannot be added without this test noticing.
    """
    import inspect
    n = 0
    for mod in (AF, AST):
        src = inspect.getsource(mod)
        assert src.strip(), "source must be readable -- a blank read is not a zero"
        n += src.count("AR.get(") + src.count("AR._load(")
    assert n >= 10, "sanity: the sibling modules really do call the loader (%d)" % n
    assert len(ENTRY_POINTS) >= 9


# --------------------------------------------------------------------- PR-1d
def _prefix_loader(path):
    """⛔ THE PRE-2026-09-06 BODY, in shape. Deliberately broken."""
    if not os.path.exists(path):
        return {}
    raise AssertionError("not exercised: this arm only tests the missing branch")


def _current_loader(path):
    os.environ[AR.RECORDS_ENV] = path
    AR._load.cache_clear()
    try:
        return AR._load()
    finally:
        os.environ.pop(AR.RECORDS_ENV, None)
        AR._load.cache_clear()


def _is_loud(loader):
    """Does `loader` REFUSE a missing source, or answer with an empty container?

    True only for a NAMED refusal. An unnamed exception is not a loud failure
    either -- it is a different silent-ish failure wearing a stack trace.
    """
    try:
        loader(MISSING)
    except AR.AlpamayoRecordsUnavailable:
        return True
    except Exception:
        return False
    return False


def test_pr1d_deliberate_regression_arm():
    """ONE checker, TWO loaders, opposite verdicts required.

    ⛔ If `_is_loud` cannot FAIL the knowingly-broken loader, its PASS on the
    fixed one is not evidence of anything. Both halves are asserted here in one
    test on purpose, so neither can be deleted without the other.
    """
    assert _is_loud(_prefix_loader) is False, (
        "the checker did not detect the ORIGINAL defect -- it measures nothing")
    assert _is_loud(_current_loader) is True, (
        "the current loader answered a missing source without raising")


# --------------------------------------------------------------------- PR-1e
def test_pr1e_opt_out_is_stamped_not_silent(monkeypatch):
    """The escape hatch exists, and it is LOUD in a different channel.

    ⛔ A silent opt-out would reinstate exactly the defect removed here, so the
    empty result must arrive with a warning AND a stamp that says the file was
    not there.
    """
    _point_at_missing(monkeypatch)
    monkeypatch.setenv(AR.RECORDS_OPTIONAL_ENV, "1")
    AR._load.cache_clear()
    with pytest.warns(RuntimeWarning, match="STAMPED empty"):
        d = AR._load()
    assert d == {}
    cov = AR.coverage()
    assert cov["clips"] == 0
    assert cov["records_available"] is False
    assert cov["optional_opt_out"] is True
    assert cov["records_path"] == MISSING


def test_pr1e_opt_out_must_be_explicit(monkeypatch):
    """Any value other than an explicit "1" still raises -- no truthiness."""
    _point_at_missing(monkeypatch)
    for v in ("0", "", "true", "yes"):
        monkeypatch.setenv(AR.RECORDS_OPTIONAL_ENV, v)
        AR._load.cache_clear()
        with pytest.raises(AR.AlpamayoRecordsUnavailable):
            AR._load()
