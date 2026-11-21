"""⛔⛔ D3 -- per-head GRADIENT REACH (`ga_*`) must reach metrics.jsonl at the LOG CADENCE.

MEASURED 2026-09-26 (the map-signal audit): refcv6-r101-s0 reports per-head gradient reach, and
**0 of its 4,621 metrics rows carry a `ga_*` key**. The reach row was filled on
`step % log_every == 0` BEFORE `step += 1` and written on `step % log_every == 0` AFTER it, so fill
and write never met; only a run's FINAL row (the `step + 1 >= steps` clause) could ever carry one,
and refcv6 has not reached its final row. A declared instrument that never runs: the
F3-WHITELIST class.

The pins run the REAL `train()` on the synthetic rig. The smoke arm builds neither seam that
switches the report on (no perception branch, no tac_decoder_v6), so `_grad_reach_declared` is
forced ON -- the move `test_conflict_readings_are_logged.py` makes with `_conflict_aux_weights`.
Everything downstream is the trainer's own code: the backward, `grad_reach_report` on the real
trunk and planner, the fill, the log write, the config.json declaration and the first-row check.
Expectations are LITERALS, never expressions over the code under test.

RED arms reintroduce the historical condition verbatim (`step % max(1, log_every) == 0 or
step + 1 >= steps`): with the first-row guard switched off it logs ZERO `ga_*` keys on every
cadence row but the final one, and with the guard on the run REFUSES at its first log row.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402
from tanitad.train import declared_vs_built as dvb  # noqa: E402

#: the parts `grad_reach_report` names on the smoke rig -- trunk (core.encoder) and planner
#: (core.decoder); no perception branch, no tactical decoder. A LITERAL.
SMOKE_KEYS = ["ga_planner", "ga_planner_n", "ga_trunk", "ga_trunk_n"]
ARGV = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "6", "--batch", "2",
        "--device", "cpu", "--save-every", "100", "--log-every", "2"]
#: the loop's own post-increment rule at --steps 6 --log-every 2 -- a LITERAL
LOG_STEPS = [2, 4, 6]


def _OLD_RULE(step, log_every, steps):
    """The historical fill condition, verbatim from ab436ee:stack/scripts/refc_v3_train.py."""
    return step % max(1, log_every) == 0 or step + 1 >= steps


def _run(tmp_path, name):
    out = tmp_path / name
    T.train(T.build_parser().parse_args(["--out", str(out), *ARGV]))
    rows = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    return rows, cfg


def _ga_steps(rows):
    return sorted(r["step"] for r in rows if any(str(k).startswith("ga_") for k in r))


def _assert_every_log_row_has_the_keys(rows, keys):
    log_rows = {r["step"]: r for r in rows if "loss" in r}
    assert sorted(log_rows) == LOG_STEPS, sorted(log_rows)
    for s in sorted(log_rows):
        miss = [k for k in keys if k not in log_rows[s]]
        assert not miss, f"log row {s} is missing {miss}"


# ---- the rule ----------------------------------------------------------------------------- #
def test_the_fill_rule_is_the_POST_increment_log_rule():
    assert [s for s in range(120) if T._logged_after(s, 50, 120)] == [49, 99, 119]
    assert [s for s in range(6) if T._logged_after(s, 2, 6)] == [1, 3, 5]
    assert [s for s in range(7) if T._logged_after(s, 3, 7)] == [2, 5, 6]
    # the historical rule filled pre-steps that are written as 1, 51, 101 -- never a log step --
    # except the final one, which is why a still-running arm carries ZERO
    assert [s for s in range(120) if _OLD_RULE(s, 50, 120)] == [0, 50, 100, 119]


# ---- GREEN, through the real loop ---------------------------------------------------------- #
def test_ga_keys_reach_EVERY_log_row_through_the_REAL_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_grad_reach_declared", lambda model: True)
    rows, cfg = _run(tmp_path, "green")
    dec = cfg["grad_reach_logging"]
    assert dec["declared"] is True, dec
    assert dec["keys"] == SMOKE_KEYS, dec
    _assert_every_log_row_has_the_keys(rows, SMOKE_KEYS)
    assert _ga_steps(rows) == LOG_STEPS
    # the reading is a real one: the planner received gradient on every logged step
    assert all(r["ga_planner_n"] > 0 for r in rows if "loss" in r), \
        [(r["step"], r["ga_planner_n"]) for r in rows if "loss" in r]
    assert dvb.check_logged_rows(cfg, rows) == []


# ---- RED arms: the historical condition ---------------------------------------------------- #
def test_RED_the_old_cadence_logs_ZERO_ga_keys_on_the_cadence_rows(tmp_path, monkeypatch):
    real_check = dvb.check_logged_rows                      # saved: the next patch hits the module
    monkeypatch.setattr(T, "_grad_reach_declared", lambda model: True)
    monkeypatch.setattr(T, "_logged_after", _OLD_RULE)      # the defect, reintroduced verbatim
    monkeypatch.setattr(T._dvb, "check_logged_rows", lambda cfg, rows: [])  # first-row guard OFF
    rows, cfg = _run(tmp_path, "red")
    # ZERO on the cadence rows 2 and 4; only the FINAL row carries them (the `steps` clause)
    assert _ga_steps(rows) == [6]
    with pytest.raises(AssertionError, match="log row 2 is missing"):
        _assert_every_log_row_has_the_keys(rows, SMOKE_KEYS)
    bad = real_check(cfg, rows)
    assert sorted(m.read_from for m in bad) == ["metrics.jsonl row step=2",
                                                "metrics.jsonl row step=4"]


def test_RED_the_first_row_guard_REFUSES_the_old_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_grad_reach_declared", lambda model: True)
    monkeypatch.setattr(T, "_logged_after", _OLD_RULE)
    with pytest.raises(SystemExit, match=r"D3: a declared instrument did not reach metrics\.jsonl "
                                         r"at the first log row \(step 2\)"):
        _run(tmp_path, "red_guard")


# ---- the OFF arm keeps the pre-instrument schema ------------------------------------------- #
def test_an_arm_with_NO_reach_seam_declares_OFF_and_logs_no_ga_key(tmp_path):
    rows, cfg = _run(tmp_path, "off")
    assert cfg["grad_reach_logging"]["declared"] is False
    assert cfg["grad_reach_logging"]["keys"] == []
    assert _ga_steps(rows) == []
    assert dvb.check_logged_rows(cfg, rows) == []


# ---- the checker on literal rows ----------------------------------------------------------- #
def test_check_logged_rows_on_LITERAL_rows():
    on = {"grad_reach_logging": {"declared": True, "keys": ["ga_a", "ga_a_n"]}}
    good = [{"step": 2, "loss": 1.0, "ga_a": 0.5, "ga_a_n": 3}, {"step": 3, "cd_x": 0.1}]
    assert dvb.check_logged_rows(on, good) == []
    bad = dvb.check_logged_rows(on, [{"step": 2, "loss": 1.0, "ga_a": 0.5}])
    assert [(m.read_from, m.why) for m in bad] == [
        ("metrics.jsonl row step=2", "declared ga_* keys MISSING: ['ga_a_n']")]
    assert len(dvb.check_logged_rows(on, [])) == 1                   # nothing to hold it against
    assert len(dvb.check_logged_rows(
        {"grad_reach_logging": {"declared": True, "keys": []}}, good)) == 1   # ON, no keys
    off = {"grad_reach_logging": {"declared": False, "keys": []}}
    assert dvb.check_logged_rows(off, [{"step": 2, "loss": 1.0}]) == []
    assert [m.built for m in dvb.check_logged_rows(
        off, [{"step": 2, "loss": 1.0, "ga_x": 0.0}])] == [["ga_x"]]
    assert [m.read_from for m in dvb.check_logged_rows({}, good)] == ["config.json"]  # pre-D3
