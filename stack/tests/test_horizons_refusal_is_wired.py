"""THE --horizons REFUSAL EXISTS, FIRES, AND EXEMPTS RESUMES -- pinned.

WHY THIS FILE EXISTS. `predictor.py` promised a trainer preflight and named it
`_refuse_untrained_horizons`. No function by that name was ever written -- the
check lives inside `preflight(a)` instead. On 2026-09-06 a reader grepped for
the promised NAME, got 0 with a valid same-breath control, and concluded the
GUARD did not exist. The grep was right; the conclusion was wrong, and it was
about to be actioned as "implement the refusal or change the default".

MEASURED the same day by EXECUTION rather than by grep: a fresh
`--horizons 1 2 4` launch exits 2. The refusal is also present in the pre-edit
baseline tree (`train_v6_staged.py` md5 `d2ade650...`), so it predates the
claim.

=> the durable fix is not another comment. It is a test that answers
"does this guard exist" by RUNNING it, so the question cannot be re-answered by
grepping an identifier that was never the guard's name.

NO GPU, NO CORPUS, NO CHECKPOINT.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_v6_staged import build_parser, preflight  # noqa: E402

MARKER = "which NO loss consumes"


def args_for(tmp_path, horizons):
    return build_parser().parse_args(
        ["--stage", "S-W", "--out", str(tmp_path),
         "--horizons", *[str(h) for h in horizons]])


def horizon_problems(problems):
    return [p for p in problems if MARKER in p]


# ---------------------------------------------------------------------------
# direction A -- it REFUSES a fresh arm that declares untrained horizons
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizons", [(1, 2, 4), (1, 2), (2,), (1, 4)])
def test_a_FRESH_run_declaring_untrained_horizons_is_REFUSED(tmp_path, horizons):
    probs = horizon_problems(preflight(args_for(tmp_path, horizons)))
    assert len(probs) == 1, (
        "the horizons refusal did not fire for %r -- this is the guard "
        "`predictor.py` promises and a 2026-09-06 reading declared absent"
        % (horizons,))
    # the message must carry the FIX, not just the complaint
    assert "--horizons 1" in probs[0]
    assert "--o5-k" in probs[0]


# ---------------------------------------------------------------------------
# the control -- it does NOT fire on the correct launch line
# ---------------------------------------------------------------------------
def test_it_does_NOT_fire_on_horizons_1(tmp_path):
    """Without this, a guard that refused unconditionally would pass the test
    above. A refusal that always refuses gets deleted."""
    assert horizon_problems(preflight(args_for(tmp_path, (1,)))) == []


# ---------------------------------------------------------------------------
# the RESUME exemption -- load-bearing, and its absence would destroy work
# ---------------------------------------------------------------------------
def test_a_RESUME_is_EXEMPT_because_a_run_under_way_is_not_a_new_decision(tmp_path):
    """`k60p30k` (MM-E19) runs `--horizons 1 2 4` DELIBERATELY as a matched
    control against a banked incumbent. A preflight that fired on RESUME would
    refuse that arm's own restart after any crash and make 22 h unrecoverable
    -- the guard destroying the work it was written to protect."""
    (tmp_path / "ckpt.pt").write_bytes(b"not a real checkpoint")
    assert horizon_problems(preflight(args_for(tmp_path, (1, 2, 4)))) == []


def test_the_resume_exemption_is_keyed_on_the_checkpoint_not_the_flag(tmp_path):
    """Control for the test above: same args, no ckpt.pt -> still refused."""
    assert len(horizon_problems(preflight(args_for(tmp_path, (1, 2, 4))))) == 1


# ---------------------------------------------------------------------------
# the WIRING -- preflight() gates BOTH launch paths, not only the dry run
# ---------------------------------------------------------------------------
def test_main_runs_preflight_before_BOTH_dry_run_and_train():
    import inspect

    import train_v6_staged as T

    src = inspect.getsource(T.main)
    assert "preflight(a)" in src
    i = src.index("preflight(a)")
    for call in ("dry_run(a)", "train(a)"):
        assert call in src, "main() no longer calls %s" % call
        assert i < src.index(call), (
            "preflight runs AFTER %s -- the refusal would not gate the launch"
            % call)


def test_the_default_is_still_1_2_4_and_that_is_a_RECORDED_decision():
    """The default is deliberately NOT changed: changing it would silently
    change the geometry of every run that omits the flag, and a model built at
    (1,) cannot strict-load a banked checkpoint. The refusal above is what makes
    the wrong default un-launchable for a fresh arm. If someone changes the
    default, this test fails and they must re-read that reasoning."""
    a = build_parser().parse_args(["--stage", "S-W", "--out", "."])
    assert list(a.horizons) == [1, 2, 4]
