"""⛔ THE E1 GUARD — and the proof that it CAN FAIL.

A guard that cannot fail is worse than none (class C13, and this program has
shipped several). Every rule below is exercised in BOTH directions: the value
that makes it pass and the value that makes it FAIL. The failing direction is
the point of the file; the passing direction only proves it is not vacuous.

THE BUG (MEASURED 2026-07-27/28)
--------------------------------
``ego=`` is passed at exactly three call sites in the repo, all three in the
TRAINER (``tanitad/train/flagship_losses.py:245,246,351``). Not one evaluation
path passed it, so a checkpoint TRAINED with ``v2_ego_to_planners`` was
EVALUATED ego-blind, silently. ``flagship-v2corpus-30k`` is training with that
lever on right now.
"""
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from tanitad.config import flagship4b_config
from tanitad.ego_plan import EgoInputDropped
from tanitad.models.fourbrain import WorldModel
from taniteval import ego_guard as EG
from taniteval import pseudosim as PS

_REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# tiny 4-brain fixtures: one WITHOUT the ego lever, one WITH it                 #
# --------------------------------------------------------------------------- #
def _cfg(ego_lever: bool):
    c = flagship4b_config()
    c.encoder.d_model = 32
    c.encoder.depth = 1
    c.encoder.n_heads = 2
    c.encoder.patch = 32
    c.encoder.img_size = 64
    c.readout.d_readout = 32
    c.predictor.d_model = 32
    c.predictor.depth = 1
    c.predictor.n_heads = 2
    c.tactical_policy.d_model = 32
    c.tactical_policy.depth = 1
    c.tactical_policy.n_heads = 2
    c.strategic_policy.d_model = 32
    c.strategic_policy.depth = 1
    c.strategic_policy.n_heads = 2
    c.v2_ego_to_planners = bool(ego_lever)
    return c


@pytest.fixture(scope="module")
def m_plain():
    return WorldModel(_cfg(False)).eval()


@pytest.fixture(scope="module")
def m_ego():
    return WorldModel(_cfg(True)).eval()


# --------------------------------------------------------------------------- #
# 1. THE CAPABILITY PROBE — it must read the two arms differently               #
# --------------------------------------------------------------------------- #
def test_capability_probe_separates_an_ego_ckpt_from_a_plain_one(m_plain, m_ego):
    """If this returned the same for both, every rule below would be vacuous."""
    plain = EG.planner_ego_capability(m_plain)
    ego = EG.planner_ego_capability(m_ego)
    assert plain["ego_input_on_planners"] is False
    assert ego["ego_input_on_planners"] is True
    assert plain["tactical_policy"]["has_trained_ego_emb"] is False
    assert ego["tactical_policy"]["has_trained_ego_emb"] is True
    assert ego["strategic_policy"]["has_trained_ego_emb"] is True
    # the shape is [2 -> d_cond], the same nn.Linear the --v2 lever builds
    assert ego["tactical_policy"]["shape"][0] == 2


# --------------------------------------------------------------------------- #
# 2. ⛔ THE DEMONSTRATED FAILURE — the exact live-bug shape                     #
# --------------------------------------------------------------------------- #
def test_GUARD_FAILS_on_the_live_bug_shape(m_ego):
    """⛔ THE FAILING VALUE, named: an ego-TRAINED policy + ``ego=None``.

    This is the state ``flagship-v2corpus-30k`` would have been scored in by
    every harness in the repo before 2026-07-28."""
    with pytest.raises(EgoInputDropped) as e:
        EG.assert_planner_ego(m_ego, None, where="test.livebug")
    msg = str(e.value)
    assert "test.livebug" in msg, "the refusal must name the call site"
    assert "silently unused" in msg


def test_guard_PASSES_when_the_ego_is_actually_fed(m_ego):
    ego = torch.zeros(3, 2)
    prov = EG.assert_planner_ego(m_ego, ego, where="test.fed")
    assert prov["ego_fed"] is True
    assert prov["ego_input_DROPPED"] is False


def test_guard_is_a_provable_NO_OP_for_every_published_arm(m_plain):
    """Every arm in the 2026-07-27 panel and every registry checkpoint has
    ``ego_emb is None``. Adding the call therefore changes NO published number,
    which is what makes refusing (rather than warning) free."""
    prov = EG.assert_planner_ego(m_plain, None, where="test.published")
    assert prov["ego_input_DROPPED"] is False
    assert prov["capability"]["ego_input_on_planners"] is False


def test_an_EXPLICIT_zero_ego_is_accepted_because_it_is_a_REAL_ablation(m_ego):
    """``ego = 0`` is IN-distribution when the run used ``v2_ego_dropout`` and is
    a different object from ``ego=None`` (which skips the ``ego_emb`` bias too).
    The ablation has to say so in code — that is the escape hatch."""
    prov = EG.assert_planner_ego(m_ego, torch.zeros(2, 2), where="test.ablate",
                                 ego_source="deliberate v0=0 ablation")
    assert prov["ego_input_DROPPED"] is False
    assert prov["ego_source"] == "deliberate v0=0 ablation"


# --------------------------------------------------------------------------- #
# 3. THE MODES — and warn mode is NOT free                                      #
# --------------------------------------------------------------------------- #
def test_default_mode_is_REFUSE(monkeypatch):
    monkeypatch.delenv(EG.ENV_VAR, raising=False)
    assert EG.guard_mode() == EG.MODE_REFUSE


def test_warn_mode_scores_but_STAMPS_the_defect_into_the_node(m_ego):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        prov = EG.assert_planner_ego(m_ego, None, where="test.warn",
                                     mode=EG.MODE_WARN)
    assert prov["ego_input_DROPPED"] is True, (
        "warn mode must still mark the number as produced ego-blind; a warning "
        "that leaves no trace in the artifact is not a record")
    assert any(issubclass(x.category, EG.EgoInputDroppedWarning) for x in w)


def test_a_TYPO_in_the_env_var_is_an_ERROR_not_a_silent_disable(monkeypatch):
    """⛔ THE FAILING VALUE for the mode parser. Falling back to the permissive
    mode on an unrecognised value is how a guard silently stops guarding."""
    monkeypatch.setenv(EG.ENV_VAR, "wrn")
    with pytest.raises(ValueError, match="not a mode"):
        EG.guard_mode()


def test_env_var_can_select_warn_mode(monkeypatch, m_ego):
    monkeypatch.setenv(EG.ENV_VAR, EG.MODE_WARN)
    assert EG.guard_mode() == EG.MODE_WARN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prov = EG.assert_planner_ego(m_ego, None, where="test.envwarn")
    assert prov["guard_mode"] == EG.MODE_WARN


# --------------------------------------------------------------------------- #
# 4. THE EGO VECTOR — the trainer's contract, not a re-derivation               #
# --------------------------------------------------------------------------- #
def test_ego_from_poses_matches_the_trainer_construction():
    """``ego = [v0/pose_scale, yr0]``, ``yr0`` = wrapped dyaw / 0.1, from
    OBSERVED poses at t and t-1 (``flagship_losses.py:202-210``)."""
    T = 6
    poses = torch.zeros(T, 4)
    poses[:, 2] = torch.tensor([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    poses[:, 3] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ego = EG.ego_from_poses(poses, torch.tensor([3, 4]), pose_scale=10.0)
    assert ego.shape == (2, 2)
    assert ego[0, 0].item() == pytest.approx(4.0 / 10.0)
    assert ego[0, 1].item() == pytest.approx(0.05 / 0.1, abs=1e-5)
    assert ego[1, 0].item() == pytest.approx(5.0 / 10.0)


def test_ego_from_poses_wraps_the_yaw_difference():
    """A pi -> -pi crossing must not read as a 62 rad/s yaw rate."""
    poses = torch.zeros(3, 4)
    poses[:, 2] = torch.tensor([0.0, np.pi - 0.05, -np.pi + 0.05])
    ego = EG.ego_from_poses(poses, torch.tensor([2]), pose_scale=10.0)
    assert abs(ego[0, 1].item()) == pytest.approx(1.0, abs=1e-4)


def test_ego_from_poses_uses_the_trainers_own_fallback_at_t0():
    """``last == 0`` has no t-1; the trainer's fallback is ``yr0 = 0``
    (``flagship_losses.py:209``), reproduced rather than invented."""
    poses = torch.zeros(3, 4)
    poses[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    poses[:, 3] = 7.0
    ego = EG.ego_from_poses(poses, torch.tensor([0]), pose_scale=10.0)
    assert ego[0, 1].item() == 0.0
    assert ego[0, 0].item() == pytest.approx(0.7)


def test_the_pose_scale_is_NOT_the_operative_speed_scale():
    """Different port, different scale — swapping them decodes garbage."""
    poses = torch.zeros(2, 4)
    poses[:, 3] = 20.0
    a = EG.ego_from_poses(poses, torch.tensor([1]), pose_scale=10.0)
    b = EG.ego_from_poses(poses, torch.tensor([1]), pose_scale=5.0)
    assert a[0, 0].item() != b[0, 0].item()


# --------------------------------------------------------------------------- #
# 5. THE PSEUDO-SIMULATION HOOK — the surface v5 is gated on                    #
# --------------------------------------------------------------------------- #
class _EgoBlindAdapter:
    """The ``panel_run.py`` shape: wraps a model, calls the policy positionally,
    declares nothing. This is what would have scored `flagship-v2corpus-30k`."""

    def __init__(self, model):
        self.model = model

    def traj(self, frames, v0, goal=None):        # pragma: no cover - refused
        raise AssertionError("must never be reached: the guard refuses first")


class _DeclaringAdapter(_EgoBlindAdapter):
    def __init__(self, model):
        super().__init__(model)
        self.ego_provenance = EG.assert_planner_ego(
            model, torch.zeros(1, 2), where="test.adapter")


def test_GUARD_FAILS_for_an_undeclared_adapter_over_an_ego_ckpt(m_ego):
    """⛔ THE FAILING VALUE at the pseudo-sim surface."""
    with pytest.raises(EgoInputDropped, match="declares no"):
        EG.assert_adapter_declares_ego(_EgoBlindAdapter(m_ego),
                                       where="test.adapter")


def test_a_DECLARING_adapter_passes(m_ego):
    prov = EG.assert_adapter_declares_ego(_DeclaringAdapter(m_ego),
                                          where="test.adapter")
    assert prov["ego_input_DROPPED"] is False
    assert prov["declared"]["ego_fed"] is True


def test_an_adapter_over_a_PLAIN_ckpt_needs_no_declaration(m_plain):
    """v4/v5 heads and every published panel arm: unaffected, no ceremony."""
    prov = EG.assert_adapter_declares_ego(_EgoBlindAdapter(m_plain),
                                          where="test.adapter")
    assert prov["ego_input_DROPPED"] is False
    assert prov["wrapped_models_with_trained_ego_emb"] == []


def test_pseudo_evaluate_REFUSES_an_undeclared_adapter_BEFORE_touching_a_model(
        m_ego):
    """⛔ The guard runs beside ``assert_grid_in_envelope`` — before any GPU
    second is spent. ``.traj`` raises if reached, so reaching the refusal proves
    no model was touched."""
    grid = PS.GridSpec(dyaw_deg=(0.0,), dlon_steps=(0,))
    with pytest.raises(EgoInputDropped):
        PS.pseudo_evaluate(_EgoBlindAdapter(m_ego), [], grid, stride=8, frame=PS.LEGACY_WARP)


# --------------------------------------------------------------------------- #
# 6. ⭐ THE COVERAGE SCAN — the guard cannot be forgotten by the next author    #
# --------------------------------------------------------------------------- #
#: Shipped evaluation modules that call the policy brains directly. Each pairs a
#: file with the guard token that must appear in it. `blindimag.py` is NOT here:
#: the 2026-07-28 census listed `blindimag.py:101` as a call site, but that line
#: is inside the MODULE DOCSTRING — the code path takes an injected `plan_fn`.
#: MEASURED correction to the census: 7 real call-site files, not 8.
_GUARDED_EVAL_FILES = (
    "taniteval/taniteval/closedloop.py",
    "taniteval/taniteval/planning.py",
    "taniteval/taniteval/planner_p2.py",
    "taniteval/taniteval/corpus_overlay.py",
    "taniteval/probe_overlay.py",
)
_CALL = re.compile(r"model\.(tactical|strategic)_policy\(|"
                   r"self\.world\.(tactical|strategic)_policy\(")


def test_every_shipped_eval_call_site_passes_ego_explicitly():
    """⭐ A NEW eval call site that forgets ``ego=`` FAILS THIS TEST.

    The 2026-07-28 finding was not that a guard was missing — it was that a
    two-condition gate had been audited at one condition, so every call site
    silently satisfied the false half. A scan is the only control that survives
    the next author: `assert_ego_is_fed` existed, was tested, and had never been
    called."""
    missing = []
    for rel in _GUARDED_EVAL_FILES:
        src = (_REPO / rel).read_text(encoding="utf-8")
        for i, line in enumerate(src.splitlines(), 1):
            if _CALL.search(line) and "ego=" not in line:
                missing.append(f"{rel}:{i}: {line.strip()}")
    assert not missing, (
        "these planner call sites do not pass ego= and would score an "
        "ego-trained checkpoint EGO-BLIND:\n  " + "\n  ".join(missing))


def test_every_guarded_eval_file_actually_calls_the_guard():
    """Passing ``ego=`` is not enough: the file must also REFUSE when the ego it
    can build is None but the checkpoint owns trained weights."""
    missing = [rel for rel in _GUARDED_EVAL_FILES
               if "assert_planner_ego" not in
               (_REPO / rel).read_text(encoding="utf-8")]
    assert not missing, f"no guard call in: {missing}"


def test_the_guard_scan_itself_can_fail():
    """⛔ The scan's own failing value, demonstrated on a synthetic line — a
    coverage test that cannot fail is the same C13 defect one level up."""
    bad = "            wp = model.tactical_policy(states, ctx)['waypoints']"
    assert _CALL.search(bad) and "ego=" not in bad
    good = "            wp = model.tactical_policy(states, ctx, ego=ego)['w']"
    assert _CALL.search(good) and "ego=" in good


#: ⚠️ KNOWN, DELIBERATE EXEMPTIONS — listed so they cannot grow silently.
#: Each pairs a call site with the reason it is NOT guarded. Adding a row here
#: is a decision someone has to write down; forgetting a call site is not.
_EXEMPTIONS = {
    "stack/tanitad/refs/refa.py:260":
        "run_hierarchy(self, states, actions, nav_cmd) — forwards ego=None. It "
        "is a MODEL file in refa_train4b's import graph, not an eval path, and "
        "no REF-A checkpoint has ever set v2_ego_to_planners (registry §3). "
        "Guarding it means editing stack/ model code for zero current benefit. "
        "ESCALATED instead: a REF-A trained with --v2 would be ego-blind.",
    "stack/tanitad/models/fourbrain.py:614":
        "WorldModel.propose_and_score's tactical_policy call (the P2 candidate "
        "scorer). Same reason: stack/ model code, and pod1 is mid-run on a "
        "trainer that imports this exact file. ESCALATED.",
    "taniteval/taniteval/blindimag.py:101":
        "NOT A CALL SITE. The 2026-07-28 census listed it as one; the line is "
        "inside the MODULE DOCSTRING. blindimag takes an injected plan_fn and "
        "never touches the policy brains. MEASURED correction to the census.",
}


def test_the_exemption_list_is_a_decision_not_an_oversight():
    """A guard with undocumented holes is the C13 defect one level down. Every
    unguarded call site must be HERE, with a reason, or the scan must cover it."""
    for site, reason in _EXEMPTIONS.items():
        path, _, line = site.rpartition(":")
        assert (_REPO / path).is_file(), f"stale exemption: {path}"
        assert len(reason) > 60, f"exemption {site} has no real reason"
    assert len(_EXEMPTIONS) == 3, (
        "the exemption list changed. That is allowed, but it is a DECISION: "
        "update this count and say why in the report.")


def test_the_trainer_is_untouched_so_pod1_stays_resumable():
    """⛔ pod1 is mid-run on ``train_flagship4b.py``. Nothing in this change may
    enter its import graph: it imports neither ``taniteval`` nor
    ``tanitad.train.heldout_gate``, and this test pins that."""
    src = (_REPO / "stack/scripts/train_flagship4b.py").read_text(
        encoding="utf-8")
    assert "taniteval" not in src, (
        "train_flagship4b.py must not import taniteval — every edit in this "
        "change lives there")
    assert "heldout_gate" not in src
    assert "ego_guard" not in src
    assert "pseudosim" not in src


def test_repo_layout_assumption_holds():
    """If the repo-relative paths above stop resolving, the two scans above go
    green by scanning nothing. Pin them."""
    for rel in _GUARDED_EVAL_FILES + ("stack/scripts/train_flagship4b.py",):
        assert (_REPO / rel).is_file(), f"missing {rel} (scan would be vacuous)"
    assert os.path.isdir(_REPO / "taniteval")
