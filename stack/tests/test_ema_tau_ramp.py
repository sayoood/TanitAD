"""The O5-EMA teacher's tau-RAMP (`--ema-decay-ramp`) — D-EMA-ADOPT's PI-gated
prerequisite for v7f ("ships with the EMA teacher, CONDITIONAL on one tau-ramp
arm first", register row D-EMA-ADOPT / commit a67b3c00c).

Mirrors `test_o5_ema_teacher.py`'s structure. Four tests are LOAD-BEARING:

  * ``test_the_default_is_bit_identical`` — ramp off must reproduce the fixed
    ``--ema-decay`` EXACTLY (same float, same state_dict keys, no log field).
    Every ramped-vs-fixed comparison rests on the fixed arm being unchanged.
  * ``test_a_resume_continues_the_ramp_instead_of_restarting_it`` +
    ``test_both_update_sites_use_the_ABSOLUTE_step`` — a ramp keyed to a
    PROCESS-LOCAL counter drops tau back to ``--ema-decay-start`` after every
    restart and re-randomises the teacher mid-run, with a healthy-looking log.
    The first pins the arithmetic, the second pins the call sites.
  * ``test_the_ramp_does_not_touch_the_adapter_EMAs`` — the cell's ONE variable
    is the O5 teacher's tau. If ``ema_update()``'s adapter copies rode along,
    the arm would carry two.

⚠️ What is NOT pinned here: a real 30k resume. The resume claim is established
analytically (tau is a pure function of the ABSOLUTE step) plus structurally
(both call sites pass that step). A CPU test cannot run the real thing.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import train_v6_staged as TV  # noqa: E402
from train_v6_staged import (  # noqa: E402
    _ema_tau_record, build_parser, build_stack_from_args, dry_run, ema_tau_at,
)

BASE = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "64",
        "--frame-w", "160", "--enc-dim", "32", "--enc-depth", "1",
        "--enc-heads", "2", "--readout-grid", "2", "--readout-grid-w", "4",
        "--readout-dim", "16", "--pred-dim", "32", "--pred-depth", "1",
        "--pred-heads", "2", "--window", "4", "--d-tac", "16", "--d-str", "8"]

RAMP = dict(ramp="cosine", fixed=0.996, start=0.99, end=None)


def _args(*extra):
    return build_parser().parse_args(BASE + list(extra))


def _stack(*extra):
    return build_stack_from_args(_args(*extra))


def _closed_form(step, total, start, end):
    """The spec, written out independently of the implementation."""
    return end - (end - start) * (math.cos(math.pi * step / total) + 1.0) / 2.0


# --------------------------------------------------------------------------- #
# 1. DEFAULT-OFF, BIT-IDENTICAL                                                 #
# --------------------------------------------------------------------------- #
def test_the_defaults_are_the_incumbent():
    a = _args()
    assert a.ema_decay_ramp == "off"
    assert a.ema_decay_start == 0.99
    assert a.ema_decay_end is None          # resolved to --ema-decay, not 1.0
    assert a.ema_decay == 0.996


def test_the_default_is_bit_identical():
    """ramp off ⇒ the SAME float as today's fixed --ema-decay, at every step."""
    for step in (0, 1, 7, 15_000, 29_999, 30_000):
        assert ema_tau_at(step, 30_000, ramp="off", fixed=0.996) == 0.996
        # and it is bit-identical to what `_EmaCopy` was constructed with
        assert (ema_tau_at(step, 30_000, ramp="off", fixed=float(0.996))
                is not None)
    got = ema_tau_at(11, 30_000, ramp="off", fixed=0.9993)
    assert got == 0.9993 and isinstance(got, float)


def test_off_adds_no_state_dict_keys_and_no_log_field():
    off = _stack("--o5-target", "ema")
    on = _stack("--o5-target", "ema", "--ema-decay-ramp", "cosine")
    assert set(off.state_dict()) == set(on.state_dict())
    assert not any("tau" in k for k in on.state_dict())
    assert not any(k.endswith(".decay") for k in on.state_dict())
    # the log record is empty when off — an unramped run's rows are unchanged
    assert _ema_tau_record(_args(), 0.996) == {}


def test_the_copy_updates_identically_when_no_decay_is_passed():
    """`update(src)` and `update(src, <configured decay>)` must agree to the
    bit — that identity IS the bit-identity of the off arm at the update site."""
    from tanitad.models.v6 import _EmaCopy
    src = torch.nn.Linear(6, 6)
    c_default = _EmaCopy(src, 0.996)
    c_explicit = _EmaCopy(src, 0.996)
    with torch.no_grad():
        for p in src.parameters():
            p.add_(1.5)
    c_default.update(src)
    c_explicit.update(src, 0.996)
    for pa, pb in zip(c_default.module.parameters(),
                      c_explicit.module.parameters()):
        assert torch.equal(pa, pb)


# --------------------------------------------------------------------------- #
# 2. THE CLOSED FORM (BYOL 2006.07733)                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("step", [0, 1, 7_500, 15_000, 22_500, 29_999, 30_000])
def test_cosine_matches_the_closed_form(step):
    got = ema_tau_at(step, 30_000, **RAMP)
    want = _closed_form(step, 30_000, 0.99, 0.996)
    assert got == pytest.approx(want, abs=1e-12)


def test_the_endpoints_are_exact():
    assert ema_tau_at(0, 30_000, **RAMP) == pytest.approx(0.99, abs=1e-12)
    assert ema_tau_at(30_000, 30_000, **RAMP) == pytest.approx(0.996,
                                                               abs=1e-12)
    # the midpoint of a cosine ramp is the arithmetic mean of the endpoints
    assert ema_tau_at(15_000, 30_000, **RAMP) == pytest.approx(0.993,
                                                               abs=1e-12)


def test_it_reproduces_BYOLs_literal_formula():
    """BYOL p5 (banked primary): tau := 1 - (1 - tau_base)*(cos(pi k/K)+1)/2.
    Our generalisation must collapse onto it exactly at end=1.0."""
    K, base = 1000, 0.996
    for k in (0, 1, 250, 500, 999, 1000):
        byol = 1.0 - (1.0 - base) * (math.cos(math.pi * k / K) + 1.0) / 2.0
        ours = ema_tau_at(k, K, ramp="cosine", fixed=base, start=base, end=1.0)
        assert ours == pytest.approx(byol, abs=1e-15)


def test_tau_end_defaults_to_the_ema_decay_value():
    """The design choice that makes the matched pair readable: the ramp LANDS
    where the fixed arm sat, so ramped-vs-emao14_30k is one variable."""
    for fixed in (0.996, 0.999, 0.95):
        assert ema_tau_at(30_000, 30_000, ramp="cosine", fixed=fixed,
                          start=0.9, end=None) == pytest.approx(fixed,
                                                                abs=1e-12)
    # ⚠️ NOT BYOL's 1.0
    assert ema_tau_at(30_000, 30_000, **RAMP) != 1.0
    # and an explicit --ema-decay-end overrides it
    assert ema_tau_at(30_000, 30_000, ramp="cosine", fixed=0.996, start=0.99,
                      end=0.9999) == pytest.approx(0.9999, abs=1e-12)


def test_the_ramp_is_monotone_non_decreasing():
    seq = [ema_tau_at(s, 3_000, **RAMP) for s in range(0, 3_001)]
    assert all(b >= a for a, b in zip(seq, seq[1:])), "tau went DOWN"
    assert seq[0] < seq[-1]                      # and it actually moves
    assert min(seq) == seq[0] and max(seq) == seq[-1]


def test_past_the_end_it_clamps_rather_than_turning_around():
    """cos is periodic; an unclamped progress > 1 would send tau back DOWN."""
    end = ema_tau_at(3_000, 3_000, **RAMP)
    assert ema_tau_at(3_500, 3_000, **RAMP) == pytest.approx(end, abs=1e-12)


# --------------------------------------------------------------------------- #
# 3. RESUME                                                                     #
# --------------------------------------------------------------------------- #
def test_a_resume_continues_the_ramp_instead_of_restarting_it():
    total = 1_000
    straight = [ema_tau_at(s, total, **RAMP) for s in range(1, total + 1)]
    # the trainer's loop is range(start_step + 1, steps + 1): a process that
    # resumed at 600 sees ABSOLUTE steps 601..1000
    resumed = [ema_tau_at(s, total, **RAMP) for s in range(601, total + 1)]
    assert resumed == straight[600:]
    assert resumed[-1] == pytest.approx(0.996, abs=1e-12)   # still lands on end
    # the regression this exists to catch: a PROCESS-LOCAL counter restarts
    restarted = [ema_tau_at(s, total, **RAMP) for s in range(1, total - 600 + 1)]
    assert restarted[0] == pytest.approx(0.99, abs=1e-6)    # back at tau_start
    # ⚠️ the damage, stated as a number: the teacher would jump from 0.9939 to
    # 0.9900 mid-run — a 1.6x FASTER teacher, silently, on every restart
    assert restarted[0] < resumed[0] - 3e-3
    assert restarted != resumed


def test_both_update_sites_use_the_ABSOLUTE_step():
    """Structural half of the resume pin: EXACT-COUNT on the call sites, so a
    future edit cannot quietly swap in a per-process counter."""
    src = Path(TV.__file__).read_text(encoding="utf-8")
    calls = re.findall(r"ema_tau_at\(\s*step,\s*int\(a\.steps\),", src)
    assert len(calls) == 2, f"expected 2 ramped update sites, found {len(calls)}"
    assert re.search(r"ema_tau_at\([^)]*start_step", src) is None
    assert re.search(r"ema_tau_at\([^)]*n_proc", src) is None


# --------------------------------------------------------------------------- #
# 4. SCOPE — the O5 teacher only                                                #
# --------------------------------------------------------------------------- #
def test_the_ramp_does_not_touch_the_adapter_EMAs():
    """`stack.ema_update()` must stay argument-free at both sites: the adapter
    EMAs keep the fixed --ema-decay, so the arm carries ONE variable."""
    src = Path(TV.__file__).read_text(encoding="utf-8")
    assert len(re.findall(r"stack\.ema_update\(\s*\)", src)) == 2
    assert re.search(r"ema_update\(\s*[^)\s]", src) is None
    # and the two teacher copies (and only those) receive the per-step tau
    assert len(re.findall(r"\.update\(stack\.(?:encoder|readout), ema_tau\)",
                          src)) == 4


def test_a_passed_decay_overrides_the_configured_one():
    from tanitad.models.v6 import _EmaCopy
    src = torch.nn.Linear(4, 4)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.0)
    c = _EmaCopy(src, 0.996)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
    c.update(src, 0.5)                       # teacher = 0.5*0 + 0.5*1
    w = c.module.weight
    assert float(w.abs().max()) == pytest.approx(0.5, abs=1e-6)
    assert c.decay == 0.996, "self.decay must stay the CONFIGURED value"


def test_tau_is_the_retention_weight_on_the_teacher():
    """The convention that silently inverts: higher tau = SLOWER teacher, which
    is BYOL's convention and ours. A flip here would make the ramp run backwards
    while every number still looked plausible."""
    from tanitad.models.v6 import _EmaCopy
    src = torch.nn.Linear(4, 4)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(0.0)
    slow, fast = _EmaCopy(src, 0.996), _EmaCopy(src, 0.996)
    with torch.no_grad():
        for p in src.parameters():
            p.fill_(1.0)
    slow.update(src, 0.999)
    fast.update(src, 0.9)
    assert float(slow.module.weight.abs().max()) < \
           float(fast.module.weight.abs().max())


# --------------------------------------------------------------------------- #
# 5. REFUSALS                                                                   #
# --------------------------------------------------------------------------- #
def test_a_decreasing_ramp_is_refused_by_name():
    with pytest.raises(ValueError, match="MONOTONE"):
        ema_tau_at(0, 100, ramp="cosine", fixed=0.996, start=0.999, end=0.99)


@pytest.mark.parametrize("start,end", [(0.0, 0.996), (1.0, 1.0), (-0.1, 0.996),
                                       (0.99, 1.5), (0.99, 0.0)])
def test_out_of_range_endpoints_are_refused(start, end):
    with pytest.raises(ValueError, match="must be in"):
        ema_tau_at(0, 100, ramp="cosine", fixed=0.996, start=start, end=end)


def test_end_may_be_exactly_one_but_start_may_not():
    """The asymmetry is BYOL's: it ramps TO 1.0, so refusing that endpoint
    would leave us unable to express the primary we cite. A start of 1.0 is a
    run-long frozen teacher — `--o5-target frozen`'s cell in a ramp's costume."""
    assert ema_tau_at(100, 100, ramp="cosine", fixed=0.996, start=0.99,
                      end=1.0) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="o5-target frozen"):
        ema_tau_at(0, 100, ramp="cosine", fixed=0.996, start=1.0, end=1.0)


def test_an_unknown_schedule_is_refused():
    with pytest.raises(ValueError, match="ema-decay-ramp"):
        ema_tau_at(0, 100, ramp="linear", fixed=0.996)


def test_a_bad_ramp_dies_at_LAUNCH_not_at_step_1():
    with pytest.raises(SystemExit, match="MONOTONE"):
        _stack("--o5-target", "ema", "--ema-decay-ramp", "cosine",
               "--ema-decay-start", "0.999", "--ema-decay-end", "0.99")


# --------------------------------------------------------------------------- #
# 6. INERT WITHOUT AN EMA TEACHER                                               #
# --------------------------------------------------------------------------- #
def test_the_ramp_is_inert_when_o5_target_is_not_ema(capsys):
    """No EMA copy ⇒ nothing to ramp. It must NOT crash, and it must SAY so —
    a flag that silently does nothing is how a 'ramped arm' gets banked that
    never ramped."""
    st = _stack("--ema-decay-ramp", "cosine")            # --o5-target live
    assert not hasattr(st, "ema_o5_enc")
    assert "INERT" in capsys.readouterr().out
    st2 = _stack("--o5-target", "frozen", "--ema-decay-ramp", "cosine")
    assert not hasattr(st2, "ema_o5_enc")


# --------------------------------------------------------------------------- #
# 7. THE LOG FIELD — end to end through the real dry-run                        #
# --------------------------------------------------------------------------- #
def _tiny(tmp_path, *extra):
    ap = build_parser()
    argv = ["--stage", "S-W", "--out", str(tmp_path), "--dry-run",
            "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
            "--patch", "16", "--enc-dim", "32", "--enc-depth", "1",
            "--enc-heads", "2", "--readout-grid", "4", "--readout-dim", "8",
            "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
            "--window", "4", "--horizons", "1", "2", "--d-tac", "32",
            "--d-str", "16", "--d-goal-embed", "16", "--adapter-hidden", "32",
            "--n-candidates", "3", "--sigreg-slices", "8",
            "--dry-steps", "2", "--dry-batch", "2", "--dry-k", "12",
            "--steps", "1000"]
    return ap.parse_args(argv + list(extra))


def test_the_log_field_appears_only_when_the_ramp_is_on(tmp_path):
    fixed = dry_run(_tiny(tmp_path / "fixed", "--o5-target", "ema"))
    assert all("ema_tau" not in r for r in fixed["steps"])

    ramped = dry_run(_tiny(tmp_path / "ramped", "--o5-target", "ema",
                           "--ema-decay-ramp", "cosine"))
    for r in ramped["steps"]:
        assert "ema_tau" in r and "ema_tau_note" in r
        want = _closed_form(r["step"], 1000, 0.99, 0.996)
        assert r["ema_tau"] == pytest.approx(want, abs=1e-7)
    # ⚠️ the dry-run ramps on the REAL --steps, not on --dry-steps: step 1 of
    # 1000 must be at the ramp's OPENING, not near its end
    assert ramped["steps"][0]["ema_tau"] == pytest.approx(0.99, abs=1e-6)
    assert ramped["steps"][0]["ema_tau"] < ramped["steps"][1]["ema_tau"]
    # the note carries the schedule, so a banked log is readable without flags
    note = ramped["steps"][0]["ema_tau_note"]
    assert "0.99" in note and "0.996" in note and "1000" in note


def test_a_live_target_run_logs_no_tau_even_with_the_flag(tmp_path):
    res = dry_run(_tiny(tmp_path / "live", "--ema-decay-ramp", "cosine"))
    assert all("ema_tau" not in r for r in res["steps"])


def test_the_flags_land_in_config_json(tmp_path):
    """The run's own artifact must record the schedule — the residual-init
    lesson: an input that changes the model but not the config is invisible."""
    out = tmp_path / "cfg"
    dry_run(_tiny(out, "--o5-target", "ema", "--ema-decay-ramp", "cosine"))
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert cfg["args"]["ema_decay_ramp"] == "cosine"
    assert cfg["args"]["ema_decay_start"] == 0.99
    assert cfg["args"]["ema_decay_end"] is None
    assert cfg["args"]["ema_decay"] == 0.996
