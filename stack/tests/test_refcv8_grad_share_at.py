"""refcv8 X4 -- the gradient-share reading's SCHEDULE lever: ``--grad-share-at`` / ``grad_share.parse_at`` / ``due``.

MEASURED 2026-10-11 (the R2 ladder's V0 on Thor): one reading inside the fp32 replay is ~16 min even with
``--grad-share-terms tac_v6,map_hires`` (forward ~228 s + ~181 s per backward pass, 4 passes), against a 3.15 s
training step -- at the tip's every-250 schedule that is 12 readings = ~3.2 h per 3000-step arm, more than the training.
The ladder's consumers read only two windows (ladder_score: size_k over (0, 0.2 smax], L2/L3 over (0.5, 1.0]), so an
arm reads at rows 500, 2000, 2500 and nowhere else.

Pinned here:
  * ``parse_at`` literals and every refusal (instrument off, non-integer, non-positive, not a multiple, past --steps);
  * ``due`` over a whole 3000-step run gives EXACTLY the literal rows -- and the tip's 12 rows when ``at`` is None;
    a deliberate regression (``at`` ignored) goes RED;
  * the schedule covers both ladder_score windows (a reading in each) -- derived from the windows' literal bounds;
  * the trainer: the pin refuses each bad spec before config.json; the due site calls ``_gshare.due`` with ``_gs_at``.
"""
from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig  # noqa: E402,F401  (puts scripts/ on sys.path)
from tanitad.train import grad_share as GS  # noqa: E402

T = importlib.import_module("refc_v3_train")
LADDER_AT = "500,2000,2500"


def test_parse_at_literals():
    assert GS.parse_at(None, 250, 3000) is None
    assert GS.parse_at("", 250, 3000) is None
    assert GS.parse_at(" , ", 250, 3000) is None
    assert GS.parse_at(LADDER_AT, 250, 3000) == frozenset({500, 2000, 2500})
    assert GS.parse_at("2500, 500,500", 250, 3000) == frozenset({500, 2500})
    assert GS.parse_at([750, 250], 250, 0) == frozenset({250, 750})          # steps 0 = no upper bound
    assert GS.parse_at("3000", 250, 3000) == frozenset({3000})                # the final row is admissible


@pytest.mark.parametrize("spec,every,steps,match", [
    ("500", 0, 3000, "instrument is off"),
    ("5x0", 250, 3000, "not an integer"),
    ("0", 250, 3000, "not positive"),
    ("-250", 250, 3000, "not positive"),
    ("600", 250, 3000, "not a multiple"),
    ("3250", 250, 3000, "past --steps"),
])
def test_parse_at_refusals(spec, every, steps, match):
    with pytest.raises(ValueError, match=match):
        GS.parse_at(spec, every, steps)


def _rows(every, at, steps=3000):
    return [s + 1 for s in range(steps) if GS.due(s + 1, every, at)]


def test_due_over_a_whole_run_is_exactly_the_literal_rows():
    assert _rows(250, GS.parse_at(LADDER_AT, 250, 3000)) == [500, 2000, 2500]
    assert _rows(250, None) == [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]   # the tip
    assert _rows(0, None) == [] and _rows(0, frozenset({500})) == []                                 # off is off
    assert _rows(-250, None) == []


def test_DELIBERATE_REGRESSION_at_ignored_goes_RED(monkeypatch):
    real = GS.due
    monkeypatch.setattr(GS, "due", lambda r, e, at=None: real(r, e, None))
    assert _rows(250, GS.parse_at(LADDER_AT, 250, 3000)) != [500, 2000, 2500]


def test_the_ladder_schedule_has_a_reading_in_each_consumer_window():
    """ladder_score.gs_median selects steps in (lo * smax, hi * smax]; size_k reads (0, 0.2], L2/L3 read (0.5, 1.0].
    smax is the run's last row = 3000. Bounds typed as literals, not imported, so a scorer edit cannot move both."""
    rows = _rows(250, GS.parse_at(LADDER_AT, 250, 3000))
    assert [r for r in rows if 0 < r <= 600] == [500]
    assert [r for r in rows if 1500 < r <= 3000] == [2000, 2500]


def _pin(extra):
    from tanitad.refs import refcv8_conditioning as r8c
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--log-every", "50", "--steps", "3000"] + extra)
    T._pin_refcv8(types.SimpleNamespace(refcv8=r8c.R8Config()), args)
    return args


def test_the_pin_refuses_each_bad_schedule_before_config_json():
    for bad in (["--grad-share-at", "500"],                                        # every defaults to 0
                ["--grad-share-every", "250", "--grad-share-at", "600"],
                ["--grad-share-every", "250", "--grad-share-at", "3250"],
                ["--grad-share-every", "250", "--grad-share-at", "five"]):
        with pytest.raises(SystemExit, match="grad-share-at"):
            _pin(bad)
    a = _pin(["--grad-share-every", "250", "--grad-share-at", LADDER_AT, "--grad-share-terms", "tac_v6,map_hires"])
    assert a.grad_share_at == LADDER_AT
    assert _pin(["--grad-share-every", "250"]).grad_share_at is None                # the tip's schedule


def test_the_due_site_uses_the_schedule():
    src = inspect.getsource(T).replace("\r\n", "\n")
    tr = src[src.index("\ndef train("):]
    assert "_gs_at = _gshare.parse_at(getattr(args, \"grad_share_at\", None), _gs_every," in tr
    assert "and _gshare.due(step + 1, _gs_every, _gs_at))" in tr
    # DELIBERATE REGRESSION: the tip's due line (no schedule) must not satisfy the pin above
    mut = tr.replace("and _gshare.due(step + 1, _gs_every, _gs_at))", "and (step + 1) % _gs_every == 0)")
    assert mut != tr and "_gshare.due(step + 1, _gs_every, _gs_at)" not in mut
