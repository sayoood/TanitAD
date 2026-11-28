"""X2 producer side — `taniteval.seam_dump`: the thing that banks the plan.

⛔ WHAT THIS PROTECTS. The seam probe was built, validated and left with ZERO
real-arm numbers because nothing banked `V6Stack.emit`'s 60-step plan. This
module is that missing half, so the tests that matter are (a) a real `emit()`
output round-trips through it into something `seam_probe` accepts, and (b)
every way of banking a MEANINGLESS dump is refused — a dump that loads and
scores but answers the wrong question is worse than no dump, because it looks
like a number.

⚠️ No GPU, no checkpoint, no corpus: the plan is produced by a default CPU
`V6Stack` and by hand-built tensors.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

_STACK = Path(__file__).resolve().parents[1]
_ROOT = _STACK.parent
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_ROOT / "taniteval"))

from taniteval.seam_dump import (  # noqa: E402
    SEAM_DUMP_KEYS, SeamDumpError, plan_is_degenerate, save_seam_dump,
    seam_dump_from_plan)


def _plan(n=4, c=3, t=60, fan=True, zero=False, sel=True):
    shape = (n, c, t, 2) if fan else (n, t, 2)
    mk = torch.zeros if zero else torch.randn
    p = {"controls": mk(*shape), "waypoints": mk(*shape)}
    if fan and sel:
        p["sel_score"] = torch.randn(n, c)
    return p


def test_a_fan_plan_banks_the_WINNER_and_every_required_key():
    d = seam_dump_from_plan(_plan(), eids=[7, 7, 8, 8], tier="T1", arm="a@1")
    for k in SEAM_DUMP_KEYS:
        assert k in d, k
    assert d["controls"].shape == (4, 3, 60, 2)
    assert d["sel"].shape == (4,) and d["sel"].dtype == torch.long
    assert d["eid"].tolist() == [7, 7, 8, 8]
    assert d["plan_steps"] == 60 and d["dt"] == 0.1
    assert d["op_band_s"] == [0.0, 2.0] and d["tac_band_s"] == [2.0, 6.0]
    assert d["tier"] == "T1"


def test_a_fan_with_NO_selector_REFUSES_rather_than_guessing_candidate_0():
    """F-16 asks about the EMITTED WINNER. Defaulting to candidate 0 would
    silently probe a trajectory the planner never proposed."""
    with pytest.raises(SeamDumpError, match="no emitted winner|sel_score"):
        seam_dump_from_plan(_plan(sel=False), eids=[1, 1, 2, 2], tier="T1",
                            arm="a")


def test_a_single_candidate_plan_needs_no_selector():
    d = seam_dump_from_plan(_plan(fan=False), eids=range(4), tier="T1",
                            arm="a")
    assert d["controls"].ndim == 3 and d["sel"].tolist() == [0, 0, 0, 0]


def test_the_ZERO_INIT_S_W_plan_is_REFUSED_unless_asked_for():
    """⚠️ THE S-W CASE, and the one most likely to be banked by accident: the
    emission head is zero-init, every control is exactly (0, 0), and the probe
    correctly returns DEGENERATE. Banking it silently is how a DEGENERATE
    verdict later gets quoted as a seam result."""
    p = _plan(zero=True)
    p["sel_score"] = torch.randn(4, 3)
    assert plan_is_degenerate(p["controls"]) is True
    with pytest.raises(SeamDumpError, match="EXACTLY zero|DEGENERATE"):
        seam_dump_from_plan(p, eids=range(4), tier="T1", arm="sw@8900")
    d = seam_dump_from_plan(p, eids=range(4), tier="T1", arm="sw@8900",
                            allow_degenerate=True)
    assert float(d["controls"].abs().max()) == 0.0


def test_a_WINDOW_length_eid_is_REFUSED():
    """The eid is the CI's resampling unit; a wrong-length one does not fail
    loudly downstream, it narrows every interval."""
    with pytest.raises(SeamDumpError, match="eids for"):
        seam_dump_from_plan(_plan(), eids=[1, 2], tier="T1", arm="a")


def test_a_MISALIGNED_gt_is_REFUSED_and_an_absent_one_is_simply_absent():
    p = _plan()
    d = seam_dump_from_plan(p, eids=range(4), tier="T1", arm="a")
    assert "gt" not in d, "an absent GT must be absent, never fabricated"
    d2 = seam_dump_from_plan(p, eids=range(4), tier="T1", arm="a",
                             gt=torch.randn(4, 60, 2))
    assert d2["gt"].shape == (4, 60, 2)
    with pytest.raises(SeamDumpError, match="does not match the plan"):
        seam_dump_from_plan(p, eids=range(4), tier="T1", arm="a",
                            gt=torch.randn(4, 40, 2))


def test_mismatched_controls_and_waypoints_REFUSE():
    p = _plan()
    p["waypoints"] = torch.randn(4, 5, 60, 2)          # different fan width
    with pytest.raises(SeamDumpError, match="disagree on the batch"):
        seam_dump_from_plan(p, eids=range(4), tier="T1", arm="a")
    with pytest.raises(SeamDumpError, match="needs BOTH|no 'waypoints'"):
        seam_dump_from_plan({"controls": torch.randn(4, 60, 2)},
                            eids=range(4), tier="T1", arm="a")


def test_save_REFUSES_a_dump_missing_a_required_key(tmp_path):
    """The `t1_eval` lesson: an artifact seam_probe would reject must fail at
    WRITE time, not after the expensive part."""
    d = seam_dump_from_plan(_plan(), eids=range(4), tier="T1", arm="a")
    d.pop("tier")
    with pytest.raises(SeamDumpError, match="missing"):
        save_seam_dump(d, tmp_path / "x.pt")


def test_the_dump_is_DETACHED_and_on_CPU_so_it_cannot_hold_the_graph():
    p = _plan()
    for k in ("controls", "waypoints"):
        p[k].requires_grad_(True)
    d = seam_dump_from_plan(p, eids=range(4), tier="T1", arm="a")
    for k in ("controls", "waypoints"):
        assert not d[k].requires_grad and d[k].device.type == "cpu"


# =========================================================================== #
# the end-to-end half: a REAL emit() -> a dump -> the REAL probe
# =========================================================================== #

def test_a_REAL_v6_emit_round_trips_into_a_dump_the_PROBE_accepts(tmp_path):
    """⛔ THE TEST THAT ACTUALLY CLOSES F-16's GAP. Everything above checks
    this module in isolation; only this one shows the instrument can now be
    RUN — a default `V6Stack.emit` output, banked here, scored by the real
    `seam_probe.py` CLI as a subprocess.

    ⚠️ The stack is randomly initialised, so the VERDICT is meaningless and is
    deliberately not asserted. What is asserted is that the probe consumed the
    artifact and produced a scored record — i.e. the producer/consumer
    contract holds end to end."""
    from tanitad.models.v6 import V6Config, V6Stack

    stack = V6Stack(V6Config()).eval()
    b = 6
    with torch.no_grad():
        plan = stack.emit(torch.randn(b, stack.cfg.d_op),
                          torch.randn(b, stack.cfg.d_goal_embed),
                          torch.full((b,), 8.0))
    plan = dict(plan)
    # a zero-init emission head is the S-W case; nudge it so the round trip is
    # not testing the degenerate refusal a second time.
    if plan_is_degenerate(plan["controls"]):
        plan = {k: (v + torch.randn_like(v) * 0.01 if torch.is_tensor(v)
                    and v.is_floating_point() else v)
                for k, v in plan.items()}
    # ⚠️ MEASURED: the DEFAULT `V6Config` emits a fan of 8 with NO
    # `sel_score` — the selector is an opt-in head (`--selector`). A real S-T
    # arm carries it (the S-S preflight refuses dropping it), and without it
    # this module REFUSES rather than guessing a winner — proved separately by
    # `test_a_fan_with_NO_selector_REFUSES...`. Supplying it here is what
    # makes this the S-T-shaped case rather than a second copy of that test.
    assert "sel_score" not in plan, \
        "the default config grew a selector — re-read this comment"
    plan["sel_score"] = torch.randn(b, plan["controls"].shape[1])
    d = seam_dump_from_plan(plan, eids=[1, 1, 2, 2, 3, 3], tier="T1",
                            arm="unit@0")
    p = save_seam_dump(d, tmp_path / "seam.pt")
    out = tmp_path / "seam.json"
    r = subprocess.run(
        [sys.executable, str(_ROOT / "taniteval" / "tools" / "seam_probe.py"),
         "--dump", p, "--out", str(out), "--n-boot", "20", "--no-scan",
         "--quiet"],
        capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert out.exists(), (
        f"the probe did not produce a record.\nrc={r.returncode}\n"
        f"stdout={r.stdout[-3000:]}\nstderr={r.stderr[-3000:]}")
    import json
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec.get("tier") == "T1"
    assert rec.get("arm") == "unit@0"
