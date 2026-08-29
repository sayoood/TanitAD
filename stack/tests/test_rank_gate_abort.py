"""P4-1 — an unrulable O6 must ABORT when required, and only WARN when reported.

⛔ WHY BOTH BRANCHES ARE LOAD-BEARING. `INCONCLUSIVE` counts as NOT-PASS, so a
REQUIRED criterion that cannot rule at the configured settings blocks the ladder
for an INSTRUMENT reason after the compute is already spent. Warning and
proceeding is the failure the banner was written to prevent, one level up.
Conversely a REPORTED criterion cannot block anything, so refusing on it would be
the opposite error — a launch killed by a probe that was never going to gate.

⚠️ `O6_spectrum` is REPORTED at every stage today, so the abort branch is not
reachable from any current launch line. It is armed for the PROMOTION, which is
precisely when a stale hardcoded policy fails silently — hence
`test_policy_is_read_from_the_spec_not_hardcoded`.
"""
import contextlib
import io
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import train_v6_staged as T  # noqa: E402


def _stack():
    return SimpleNamespace(cfg=SimpleNamespace(
        predictor=SimpleNamespace(window=6), d_op=2048))


def _run(stage, accum=1):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        T._warn_rank_gate_unrulable(
            SimpleNamespace(batch=8, spectrum_accum=accum, stage=stage), _stack())
    return buf.getvalue()


def test_o6_is_reported_at_every_stage_today():
    """The premise the abort branch is dormant under. If this fails, O6 was
    promoted — which is intended, but the ladder's launch lines must then carry
    --spectrum-accum, so the failure should be read as a TODO, not a bug."""
    required = {st: tuple(sp.get("required", ()))
                for st, sp in T.STAGE_GATE_SPEC.items()}
    assert not any("O6_spectrum" in r for r in required.values()), required


def test_reported_stage_warns_and_proceeds():
    out = _run("S-W")
    assert "CANNOT RULE" in out
    assert "--spectrum-accum 22" in out, "the warning must name the fix"
    # ⛔ and it must NOT have raised — a reported criterion cannot block a launch


def test_sufficient_accum_never_warns_or_aborts():
    assert "CAN rule" in _run("S-W", accum=22)


def test_required_stage_aborts_before_any_compute(monkeypatch):
    """The P4-1 behaviour: promote O6 to required, and the launch refuses."""
    spec = {k: dict(v) for k, v in T.STAGE_GATE_SPEC.items()}
    spec["S-W"]["required"] = tuple(spec["S-W"]["required"]) + ("O6_spectrum",)
    monkeypatch.setattr(T, "STAGE_GATE_SPEC", spec)
    with pytest.raises(SystemExit) as e:
        _run("S-W")
    msg = str(e.value)
    assert "REFUSING TO LAUNCH" in msg
    assert "--spectrum-accum 22" in msg, "the refusal must name the fix, not just refuse"
    assert "O6_spectrum" in msg


def test_required_stage_with_enough_accum_does_not_abort(monkeypatch):
    """⭐ The abort must key on CANNOT-RULE, not on being required."""
    spec = {k: dict(v) for k, v in T.STAGE_GATE_SPEC.items()}
    spec["S-W"]["required"] = tuple(spec["S-W"]["required"]) + ("O6_spectrum",)
    monkeypatch.setattr(T, "STAGE_GATE_SPEC", spec)
    assert "CAN rule" in _run("S-W", accum=22)


def test_policy_is_read_from_the_spec_not_hardcoded(monkeypatch):
    """⛔ THE ONE THAT PROTECTS THE FUTURE.

    `o6_is_required` must consult STAGE_GATE_SPEC live. A copied constant would
    keep warning after the promotion — the exact silent-staleness this guard
    exists to prevent, reproduced inside the guard itself.
    """
    assert T.o6_is_required("S-W") is False
    spec = {"S-W": {"required": ("O6_spectrum",), "reported": ()}}
    monkeypatch.setattr(T, "STAGE_GATE_SPEC", spec)
    assert T.o6_is_required("S-W") is True


def test_unknown_or_missing_stage_does_not_abort():
    """A stage with no spec entry has no required criteria — warn, never refuse."""
    assert "CANNOT RULE" in _run("S-NOPE")
    assert "CANNOT RULE" in _run(None)
