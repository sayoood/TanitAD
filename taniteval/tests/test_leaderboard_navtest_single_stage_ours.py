"""The leaderboard SHOWS every run it reads — in the right place — and never shows a withdrawn one as a result.

⛔ WHY (2026-09-26; W8's escalation E1, then what a general invariant test found):

1. ``EPDMS_v2_navtest_single_stage`` mixes pre-, post- and unverified-#151 published rows. Our harness is post-#151
   and every W8 artifact says so in ``provenance.harness.fix151``. But the table printed ONLY published rows plus a
   "TanitAD — pending" placeholder, and because the protocol HAS a table, ``orphan_runs_block`` did not list our runs:
   a real run would have been read, validated, counted and rendered NOWHERE. The placeholder sat after whichever
   group came last — the one "out of comparison".
2. The invariant "a run the page reads is a run the page shows" then failed on real inputs: SIX nuScenes runs of
   2026-09-26 were read and rendered nowhere (their table, too, prints published rows only).
3. And the loader never looked for ``TOMBSTONE.json``: the next build would have printed FOUR withdrawn refcv4b
   navhard runs as live result blocks beside the run that superseded them.

Driven by a REAL W8 summary (``…/2026-09-26-navtest-single-stage/raw/smoke218_run/summary.json``, tracked) through the
REAL loader (``sources.summary_rows``), and by the real results tree for the page-level invariants.
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_TE = _HERE.parent
REPO = _TE.parent
if str(_TE) not in sys.path:
    sys.path.insert(0, str(_TE))

from taniteval.leaderboard import render, sources      # noqa: E402

PROTO = "EPDMS_v2_navtest_single_stage"
SUMMARY = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/raw/smoke218_run/summary.json"
POST, PRE, UNV = ("*corrected EPDMS (post-#151):*", "*legacy EPDMS* (pre-#151):*",
                  "*implementation NOT STATED by the source — out of comparison:*")
NOSIDE = "*Our runs whose artifact states NO usable #151 side"
NL = "\n"


@pytest.fixture(scope="module")
def pub():
    return sources.load_published(REPO, sources.load_config(REPO))


@pytest.fixture(scope="module")
def summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def _rows(s):
    return sources.summary_rows(s, "test:" + SUMMARY.name, REPO)


def _groups(lines):
    """{group header -> the lines under it, up to the next group header}."""
    heads = [i for i, ln in enumerate(lines) if ln in (POST, PRE, UNV) or ln.startswith(NOSIDE)]
    return {lines[i]: lines[i:(heads[k + 1] if k + 1 < len(heads) else len(lines))] for k, i in enumerate(heads)}


def _who(r):
    return f"**TanitAD — {r['label']}**" if r["kind"] == "model" else f"**our {r['kind']} — {r['label']}**"


# ---------------------------------------------------------------- 1. single-stage: our rows, placed by the artifact
def test_the_real_artifact_states_its_side_and_the_loader_carries_it(summary):
    assert summary["provenance"]["harness"]["fix151"] == "post"          # the producer's own field
    rows = _rows(summary)
    assert {r["key"] for r in rows} == {"CV", "STOP", "HUMAN"}
    assert {r["fix151"] for r in rows} == {"post"}
    assert all("FIX151_POST_ONLY_COMPARABLE" in r["caveats"] for r in rows)


def test_our_post151_run_is_rendered_in_the_post_group_and_nowhere_else(pub, summary):
    rows = _rows(summary)
    lines = render.navtest2_tables({"published": pub, "navsim": rows}, PROTO)
    g = _groups(lines)
    run_id = summary["run_id"]
    post_text = NL.join(g[POST])
    for r in rows:
        assert _who(r) in post_text, r["key"]
    for other in (PRE, UNV):
        assert run_id not in NL.join(g.get(other, [])), other
    assert not any(k.startswith(NOSIDE) for k in g)
    assert any("FIX151_POST_ONLY_COMPARABLE" in ln and "non-reactive" in ln for ln in lines)
    # the producer's caveat IDS are printed — never a Python repr of the {"id", "one_line"} objects
    assert "`SUBSET_RUN`" in post_text and "'one_line'" not in post_text and "{'id'" not in post_text


def test_harness_floors_are_never_labelled_as_tanitad_and_do_not_retire_the_placeholder(pub, summary):
    """The W8 run is CV / STOP / HUMAN with checkpoint none: floors, not a TanitAD model. 'TanitAD — any arm:
    pending' stays true beside them, and no floor row may read as a TanitAD system."""
    rows = _rows(summary)
    assert {r["kind"] for r in rows} <= {"floor", "control", "reference"}
    g = _groups(render.navtest2_tables({"published": pub, "navsim": rows}, PROTO))
    post_text = NL.join(g[POST])
    assert "TanitAD — any arm" in post_text
    assert not any(f"**TanitAD — {r['label']}**" in post_text for r in rows)


def test_a_model_arm_is_labelled_tanitad_and_retires_the_placeholder(pub, summary):
    rows = _rows(copy.deepcopy(summary))
    rows[0] = {**rows[0], "kind": "model", "label": "refcvX-test"}
    lines = render.navtest2_tables({"published": pub, "navsim": rows}, PROTO)
    assert any("**TanitAD — refcvX-test**" in ln for ln in lines)
    assert not any("TanitAD — any arm" in ln for ln in lines)


def test_a_run_whose_artifact_states_no_side_is_never_placed_in_a_comparison_group(pub, summary):
    s = copy.deepcopy(summary)
    del s["provenance"]["harness"]["fix151"]
    rows = _rows(s)
    assert {r["fix151"] for r in rows} == {None}
    g = _groups(render.navtest2_tables({"published": pub, "navsim": rows}, PROTO))
    for side in (POST, PRE, UNV):
        assert s["run_id"] not in NL.join(g.get(side, [])), side
    noside = next(v for k, v in g.items() if k.startswith(NOSIDE))
    assert s["run_id"] in NL.join(noside) and "`None`" in NL.join(noside)


def test_a_run_whose_artifact_says_pre_sits_with_the_pre_rows(pub, summary):
    s = copy.deepcopy(summary)
    s["provenance"]["harness"]["fix151"] = "pre"
    g = _groups(render.navtest2_tables({"published": pub, "navsim": _rows(s)}, PROTO))
    assert s["run_id"] in NL.join(g[PRE]) and s["run_id"] not in NL.join(g[POST])


def test_without_runs_the_placeholder_sits_in_the_post_group(pub):
    g = _groups(render.navtest2_tables({"published": pub, "navsim": []}, PROTO))
    assert any("TanitAD — any arm" in ln for ln in g[POST])
    assert not any("TanitAD — any arm" in ln for ln in g.get(UNV, []) + g.get(PRE, []))


# ---------------------------------------------------------------- 2 + 3. the page-level invariants, on real inputs
def _page(summary):
    build = importlib.import_module("taniteval.leaderboard.build")   # the package re-exports a FUNCTION `build`
    cfg = sources.load_config(REPO)
    model = build.load_model(REPO, cfg)
    model["navsim"] = model["navsim"] + [{**r, "tombstone": None} for r in _rows(summary)]
    return model, render.render_md(model, cfg)


def test_every_live_run_on_the_page_is_rendered_somewhere(summary):
    """A run the page READ is a run the page SHOWS. Real inputs, plus the W8 run injected as if it had landed in
    ``results/bench``."""
    model, md = _page(summary)
    missing = sorted({r["run_id"] for r in model["navsim"] if r.get("run_id") and r["run_id"] not in md})
    assert not missing, f"runs read but rendered nowhere: {missing}"


def test_a_withdrawn_run_appears_only_in_the_withdrawn_block(summary):
    """Each withdrawn run appears ONLY in the 0E.y record, never in a results table."""
    model, md = _page(summary)
    withdrawn = sorted({r["run_id"] for r in model["withdrawn"]})
    assert len(withdrawn) >= 6                                   # 4 navhard + 2 nuScenes on this tree today
    lines = md.split(NL)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("### 0E.y Withdrawn runs"))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(("### ", "## "))), len(lines))
    for rid in withdrawn:
        where = [i for i, ln in enumerate(lines) if rid in ln]
        assert where and all(start <= i < end for i in where), (rid, where[:3])
    assert not any(r.get("tombstone") for r in model["navsim"])


def test_a_tombstone_is_read_and_a_malformed_one_is_refused(tmp_path):
    s = tmp_path / "summary.json"
    s.write_text("{}", encoding="utf-8")
    assert sources.tombstone_of(s) is None
    (tmp_path / "TOMBSTONE.json").write_text(json.dumps({"TOMBSTONE": "True", "superseded_by": "run-b",
                                                         "why": "wrong denominator", "removed_utc": "x"}),
                                             encoding="utf-8")
    assert sources.tombstone_of(s) == {"superseded_by": "run-b", "why": "wrong denominator", "removed_utc": "x"}
    (tmp_path / "TOMBSTONE.json").write_text(json.dumps({"TOMBSTONE": "maybe"}), encoding="utf-8")
    with pytest.raises(sources.SourceError):
        sources.tombstone_of(s)
