"""V6Stack nav-injection gates (PI directive 2026-08-30).

⛔ THE CHEAPEST TEST HERE IS THE MOST VALUABLE ONE.
`test_nav_OFF_constructs_and_runs` would have caught, in seconds, a scope bug
that blocked ALL stack construction: the nav locals were assigned in `uplink_tac`
and consumed in `forward`, because the insertion anchor `        cfg = self.cfg`
is NOT UNIQUE in v6.py and a first-match replace edited the wrong method. The
edit was syntactically perfect and the anchor LOOKED specific.
⇒ Two lessons, both encoded below: build the nav-OFF path in a test, and never
anchor a patch on a string you have not proved unique.
"""
import dataclasses

import pytest
import torch

from tanitad.models.nav_conditioning import NavTokenMissing
from tanitad.models.v6 import V6Config, V6Stack


def _cfg(nav: bool):
    return dataclasses.replace(V6Config(), nav_cond=nav)


# ------------------------------------------------------------- strict subset
def test_nav_cond_defaults_FALSE():
    """⛔ Mandatory is enforced by a PREFLIGHT REFUSAL, not by flipping this
    default. Flipping it silently changes the construction of every existing
    arm — the comparability break the PI ruled against. An explicit flag is
    recorded in config.json; a default is not."""
    assert V6Config().nav_cond is False


def test_nav_OFF_constructs_and_runs():
    """⭐ THE REGRESSION TEST FOR THE SCOPE BUG. Constructing with nav off and
    entering forward must not raise NameError — that is what a nav local defined
    in the wrong method produces, and it blocks every stack construction."""
    s = V6Stack(_cfg(False))
    assert s.nav is None
    try:
        s.forward(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    except NameError as e:                      # the failure mode under test
        pytest.fail(f"nav local out of scope on the nav-off path: {e}")
    except Exception:
        pass                                    # real tensor work, dummy shapes


def test_nav_ON_constructs_with_the_widths_the_SITES_require():
    """operative = cfg.predictor.d_model (the act_emb/cond width);
    tactical = 2*d_goal_embed (== e_a_tac); strategic = d_goal_embed (== e_a_str).
    FTac CONCATENATES (tactical.py), so nav is projected into the existing g_flat
    width and ADDED — the cond_tac_dyn idiom. A uniform width dies at the first
    forward; widening the cat would orphan every existing checkpoint."""
    c = _cfg(True)
    s = V6Stack(c)
    assert s.nav.widths == {"operative": int(c.predictor.d_model),
                            "tactical": 2 * int(c.d_goal_embed),
                            "strategic": int(c.d_goal_embed)}


# ------------------------------------------------------------- the guard
def test_missing_nav_RAISES_and_raises_FIRST():
    """⛔ Never defaults, and the refusal must precede any tensor work so the
    message names the real cause rather than surfacing as a shape error three
    modules away."""
    s = V6Stack(_cfg(True))
    with pytest.raises(NavTokenMissing) as e:
        s.forward(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert "MANDATORY" in str(e.value)
    assert "collate" in str(e.value), "the refusal must name where to fix it"


# --------------------------------------------------- ALL FIVE call sites
@pytest.mark.parametrize("site,call", [
    ("rollout helper", "predictor_op(win_s, win_a"),
    ("strategic", "predictor_str(z_str"),
    ("tactical", "predictor_tac(z_tac"),
    ("operative main", "predictor_op(z_op_win, actions"),
    ("detached seam", "predictor_op(z_op_win.detach()"),
])
def test_every_predictor_call_site_receives_nav(site, call):
    """⛔ FIVE sites, not three, and the two extra ones fail SILENTLY:

    * the ROLLOUT helper — conditioned training with an unconditioned rollout is
      the nav-echo shape v6.py names, "a conditioning path present in the diagram,
      absent from the gradient". It would LOOK like a working nav arm.
    * the DETACHED SEAM — it exists BECAUSE a downlink was once trained on the
      main path only. Missing it reproduces the defect the seam was added to fix.

    Both would pass a naive test and produce plausible numbers, which is exactly
    why they are pinned here.
    """
    import inspect

    from tanitad.models import v6
    src = inspect.getsource(v6)
    i = src.find(call)
    assert i > 0, f"call site not found: {call}"
    # the injection may precede the call (strategic/tactical) or be an argument
    assert "nav" in src[max(0, i - 420):i + 340], f"{site} does not receive nav"


def test_nav_reaches_the_seam_DETACHED():
    """The seam's whole discipline is that gradients reach the port's own
    parameters without the WM loss training the conditioning path backwards."""
    import inspect

    from tanitad.models import v6
    src = inspect.getsource(v6)
    i = src.find("predictor_op(z_op_win.detach()")
    assert "nav_op.detach()" in src[i:i + 340]


# ------------------------------------------------- the trainer-side enforcement
def _parser():
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import train_v6_staged as T
    return T, T.build_parser()


BASE = ["--stage", "S-W", "--out", "x", "--v2-cache", "y"]


def _nav_problems(extra):
    T, ap = _parser()
    return [p for p in T.preflight(ap.parse_args(BASE + extra)) if "nav-cond" in p]


def test_preflight_REFUSES_without_nav_cond():
    """⛔ Mandatory is enforced HERE, by a refusal, not by a default flip. A
    refusal is auditable and the flag lands in config.json; a default is
    invisible and would silently change every existing arm's construction."""
    probs = _nav_problems([])
    assert probs, "a v6 arm without --nav-cond must be refused"
    assert "ARCHITECTURALLY DIFFERENT" in probs[0], "the refusal must say why"


def test_preflight_accepts_with_nav_cond():
    assert not _nav_problems(["--nav-cond"])


def test_the_advertised_ESCAPE_HATCH_ACTUALLY_OPENS():
    """⚠️ REGRESSION. The first version of this refusal NAMED
    --i-know-this-arm-predates-nav in its message and refused anyway — an escape
    hatch that does not open. A flag advertised but not honoured is the same
    defect family as a flag that validates itself without being wired
    (cf. --newest-frame-only, inert while its consistency guard implied
    otherwise)."""
    assert not _nav_problems(["--i-know-this-arm-predates-nav"])


def test_the_optout_is_reachable_from_build_parser():
    """The refusal's advice must be actionable from the same parser that
    produced the refusal — otherwise it names a flag the user cannot pass."""
    _, ap = _parser()
    a = ap.parse_args(BASE + ["--i-know-this-arm-predates-nav"])
    assert a.predates_nav is True


def test_nav_is_forwarded_through_the_batch_WHITELIST():
    """⛔ The batch dict is a WHITELIST, not a view of the collated item — a key
    added to the dataset reaches `b` and stops there (measured: PSG's first
    smoke died KeyError('ep_idx') while everything upstream was correct)."""
    import inspect
    T, _ = _parser()
    src = inspect.getsource(T)
    assert '"nav_token": b.get("nav_token")' in src
    assert 'nav_token=batch.get("nav_token")' in src, "and forwarded to the stack"
