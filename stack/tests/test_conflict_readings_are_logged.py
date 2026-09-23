"""⛔ EVERY conflict reading reaches metrics.jsonl, whatever --log-every is.

MEASURED 2026-09-23 on the launched refcv6 run (Thor, `refcv6-r101-s0`): the gradient-conflict
detector measures on pre-increment steps divisible by `--conflict-every`, i.e. on LOGGED steps
10k+1, and its reading was merged ONLY into a log row. With `--log-every 50` those never coincide,
so every reading was computed -- ~10 % of the run's time -- and thrown away: rows 50, 100 and 150
carried no `cd_*` key at all. The smokes had logged every step and never showed it.

The pin runs the REAL `train()` on the synthetic rig with the detector forced on and its aux side
pointed at a loss that reaches the trunk there (`law`), `--conflict-every 2 --log-every 5`, six
steps. Readings are taken on pre-steps 0, 2, 4 = logged steps 1, 3, 5, and step 5 is also a log
step. ⇒ steps 1 and 3 must arrive as rows of their own (a step, the readings, no loss), and step 5
inside its log row. The expected set is written as a LITERAL, not derived from the code under test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T  # noqa: E402


def test_every_reading_is_logged_even_off_the_log_cadence(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_conflict_aux_weights", lambda model: {"law": 1.0})
    out = tmp_path / "run"
    T.train(T.build_parser().parse_args(
        ["--arm", "hier", "--out", str(out), "--smoke", "--synth-episodes", "2",
         "--steps", "6", "--batch", "2", "--device", "cpu", "--save-every", "100",
         "--conflict-detector", "on", "--conflict-every", "2", "--log-every", "5"]))
    rows = [json.loads(ln) for ln in (out / "metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    read = {r["step"]: r for r in rows
            if any(k.startswith("cd_") and k != "cd_deferred" for k in r)}
    assert sorted(read) == [1, 3, 5], sorted(read)
    for s in (1, 3):
        assert "loss" not in read[s], "an off-cadence reading must be its own row"
    assert "loss" in read[5], "an on-cadence reading rides in its log row"
    losses = sorted(r["step"] for r in rows if "loss" in r)
    assert losses == [5, 6], losses          # the log cadence itself is unchanged
