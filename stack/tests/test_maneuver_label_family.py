"""The manoeuvre-label family must follow the ARM, not the eval module.

⛔ THE DEFECT THIS PINS (MEASURED 2026-08-17). ``taniteval/hierarchy.py:592``
computed the ground-truth manoeuvre label as ``refb_labels.classify_maneuver``
— the **v1 net-yaw** classifier — **unconditionally**, with no branch on the
arm's ``cfg.v2_labels``. It flows to ``rec["man_corr_real"]`` and thence to
``seam_ctx_to_tactical.maneuver_acc``, the ``ctx -> tactical`` seam ``CLAUDE.md``
records as the programme's one "load-bearing" hierarchy seam. Every arm trained
with ``--v2`` (which implies ``--labels-v2``) therefore had its manoeuvre head
scored against labels it was never trained to produce.

⚠️ **THIS IS NOT A THRESHOLD DISAGREEMENT.** v1 gates the **net yaw** in rad
(``YAW_TURN_RAD = 0.15``); v2 gates the **path curvature** in 1/m
(``CURV_TURN_MAN_PER_M = 1/60``). Different physical quantities — no choice of
threshold makes the two commensurable, which is why the fix is a branch and not
a constant.

NON-VACUITY. Every behavioural assertion below runs on ONE fixture,
:func:`_gentle_curve`, on which the two labelers **provably disagree**
(``test_fixture_actually_separates_the_two_label_families`` proves it, and every
other test would pass vacuously if it stopped holding). Against the pre-fix code
the three eval-path tests fail: ``hierarchy.run`` / ``rollout._man_gt`` /
``planning.run`` had no ``labels_v2`` parameter at all (``TypeError``), and the
source guard finds the unconditional call.

⛔ NO GPU, NO CORPUS, NO CHECKPOINT — synthetic geometry and stub modules only.
"""
from __future__ import annotations

import math
import os
import re
import sys

import pytest
import torch
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/stack/tests
_REPO = os.path.dirname(os.path.dirname(_HERE))              # <repo>
_STACK = os.path.join(_REPO, "stack")
_TE = os.path.join(_REPO, "taniteval")
for _p in (_STACK, os.path.join(_STACK, "scripts"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

R = pytest.importorskip("refb_labels")


# --------------------------------------------------------------------------- #
# The fixture the whole file rests on                                          #
# --------------------------------------------------------------------------- #
def _gentle_curve(n=40, radius_m=200.0, v=20.0, dt=0.1):
    """A constant-radius LEFT curve at motorway speed: poses [n, 4].

    Chosen so the two label families give DIFFERENT answers over the 2 s
    horizon. At R = 200 m the curvature is 1/200 = 0.005 /m, comfortably under
    v2's junction gate (1/60 = 0.01667 /m) -> v2 says ``lane_keep``. Over 2 s at
    20 m/s the car travels 40 m of arc, so the net heading change is
    40/200 = 0.2 rad > v1's 0.15 rad gate -> v1 says ``turn_left``.

    This is exactly the case ``classify_maneuver_v2``'s own docstring names: a
    gentle highway curve that stays ``lane_keep`` *"even when its net dyaw
    exceeds v1's 0.15 rad"*.
    """
    ds = v * dt                                   # arc per step (2.0 m)
    dpsi = ds / radius_m                          # yaw per step (0.01 rad)
    poses = torch.zeros(n, 4, dtype=torch.float32)
    x = y = yaw = 0.0
    for t in range(n):
        poses[t] = torch.tensor([x, y, yaw, v])
        x += ds * math.cos(yaw)
        y += ds * math.sin(yaw)
        yaw += dpsi
    return poses


H = 20                                            # LABEL_HORIZON (2 s @ 10 Hz)


def test_fixture_actually_separates_the_two_label_families():
    """⛔ THE NON-VACUITY PROOF. If this ever stops holding, every behavioural
    assertion in this file becomes an assertion about nothing."""
    p = _gentle_curve()
    pose_last, fut = p[:1], p[1:1 + H][None]
    v1 = R.window_maneuver_labels(pose_last, fut, H)
    v2 = R.window_maneuver_labels_v2(pose_last, fut, H)
    assert int(v1[0]) == R.TURN_LEFT, (
        "fixture no longer trips v1's net-yaw gate", int(v1[0]))
    assert int(v2[0]) == R.LANE_KEEP, (
        "fixture no longer stays under v2's curvature gate", int(v2[0]))
    assert int(v1[0]) != int(v2[0])
    # and the two gates really are different QUANTITIES, not one rescaled
    net_yaw = float(R.wrap_to_pi(fut[0, H - 1, 2] - pose_last[0, 2]))
    arc = float((fut[0, :H, :2] - torch.cat([pose_last[:, :2], fut[0, :H - 1, :2]])
                 ).norm(dim=-1).sum())
    assert net_yaw > R.YAW_TURN_RAD                      # rad, v1's gate
    assert abs(net_yaw / arc) < R.CURV_TURN_MAN_PER_M    # 1/m, v2's gate


# --------------------------------------------------------------------------- #
# 1. ONE definition of the flag -> labeler mapping                             #
# --------------------------------------------------------------------------- #
def test_dispatcher_selects_the_family_and_is_byte_identical_to_each():
    p = _gentle_curve()
    pose_last, fut = p[:1], p[1:1 + H][None]
    got_v1 = R.window_maneuver_labels_for(pose_last, fut, H, v2=False)
    got_v2 = R.window_maneuver_labels_for(pose_last, fut, H, v2=True)
    assert torch.equal(got_v1, R.window_maneuver_labels(pose_last, fut, H))
    assert torch.equal(got_v2, R.window_maneuver_labels_v2(pose_last, fut, H))
    assert not torch.equal(got_v1, got_v2)


def test_dispatcher_refuses_to_guess_the_family():
    """``v2`` is keyword-only and REQUIRED: a caller that has not resolved the
    arm's label family must fail at the call site, not inherit a default. Two
    definitions of this mapping is how the eval and trainer drifted apart."""
    p = _gentle_curve()
    with pytest.raises(TypeError):
        R.window_maneuver_labels_for(p[:1], p[1:1 + H][None], H)
    with pytest.raises(TypeError):                  # positional is not allowed
        R.window_maneuver_labels_for(p[:1], p[1:1 + H][None], H, True)


def test_trainer_and_eval_share_the_one_definition():
    """The trainer must not carry a SECOND copy of the branch. Source-level,
    because the point is the absence of a duplicate, not a value."""
    src = open(os.path.join(_STACK, "scripts", "train_flagship4b.py"),
               encoding="utf-8").read()
    ds = src.split("class FlagshipWindowDataset", 1)[1].split("\ndef _wrap", 1)[0]
    assert "window_maneuver_labels_for(" in ds, (
        "the trainer must select its manoeuvre labeler through the shared "
        "dispatcher, not by re-stating the branch")
    assert "refb_labels.classify_maneuver(" not in ds
    assert "refb_labels.window_maneuver_labels_v2(" not in ds


# --------------------------------------------------------------------------- #
# 2. The eval path follows the arm — behavioural                               #
# --------------------------------------------------------------------------- #
def test_rollout_man_gt_follows_the_arm():
    """``rollout._man_gt`` feeds ``win['maneuver_gt']`` -> the four-families
    TACTICAL block. Pre-fix it took no ``labels_v2`` at all."""
    rollout = pytest.importorskip("taniteval.rollout")
    p = _gentle_curve(n=60)
    last = torch.tensor([0, 5, 10])
    g1 = rollout._man_gt(p, last, horizon=H, labels_v2=False)
    g2 = rollout._man_gt(p, last, horizon=H, labels_v2=True)
    assert (g1 == R.TURN_LEFT).all(), g1
    assert (g2 == R.LANE_KEEP).all(), g2
    # and the v1 branch is byte-identical to what the pre-fix code produced
    t1 = torch.clamp(last + H, max=p.shape[0] - 1)
    assert torch.equal(g1, R.classify_maneuver(p[last][:, 2], p[t1][:, 2],
                                               p[last][:, 3], p[t1][:, 3]))


# --------------------------------------------------------------------------- #
# 3. hierarchy.run — the seam the defect actually lands on                     #
# --------------------------------------------------------------------------- #
S_DIM, A_DIM, C_DIM, I_DIM, F_DIM = 8, 2, 4, 4, 6


class _Strategic(nn.Module):
    def __init__(self):
        super().__init__()
        self.nav_emb = nn.Embedding(4, C_DIM)
        self.route = nn.Linear(S_DIM, 3)

    def forward(self, states, nav, ego=None):
        z = states[:, -1]
        return {"ctx": z[:, :C_DIM] + self.nav_emb(nav),
                "route_logits": self.route(z)}


class _Tactical(nn.Module):
    """The manoeuvre head is deliberately restricted to {lane_keep, turn_left}
    — the exact two classes the fixture's v2 and v1 targets take — so
    ``maneuver_acc`` under the two families is forced to sum to 1.0 and the
    'did the number actually move' assertion is arithmetic, not a golden file."""

    def __init__(self):
        super().__init__()
        self.man2 = nn.Linear(S_DIM + C_DIM, 2)     # -> {lane_keep, turn_left}
        self.wp = nn.Linear(S_DIM + C_DIM, 2)
        self.tl = nn.Linear(S_DIM + C_DIM, S_DIM)
        self.it = nn.Linear(S_DIM + C_DIM, I_DIM)

    def forward(self, states, ctx, ego=None):
        h = torch.cat([states[:, -1], ctx], -1)
        m2 = self.man2(h)
        man = torch.cat([m2, torch.full((h.shape[0], 3), -1e4)], -1)
        return {"intent": self.it(h), "maneuver_logits": man,
                "target_latent": self.tl(h),
                "waypoints": {k: self.wp(h) for k in (5, 10, 15, 20)}}


class _Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_emb = nn.Linear(A_DIM, S_DIM)
        self.intent_proj = nn.Linear(I_DIM, S_DIM)
        self.head = nn.Linear(S_DIM, S_DIM)

    def forward(self, states, actions, intent=None):
        z = self.head(states[:, -1]) + self.act_emb(actions)[:, -1]
        if intent is not None:
            z = z + self.intent_proj(intent)
        return {h: z for h in (1, 2, 4)}


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dim = S_DIM
        self.enc = nn.Linear(F_DIM, S_DIM)
        self.strategic_policy = _Strategic()
        self.tactical_policy = _Tactical()
        self.predictor = _Predictor()

    def encode_window(self, fw):
        return self.enc(fw)


class _Readout(nn.Module):
    def forward(self, z_last, z_hat):
        d = (z_hat - z_last)[:, :3] * 0.01
        return torch.stack([d[:, 0].abs() + 2.0, d[:, 1], d[:, 2]], -1)


class _Ep:
    def __init__(self, eid, T=260):
        self.episode_id = eid
        self.poses = _gentle_curve(n=T)
        self.feats = torch.randn(T, F_DIM)
        self.actions = torch.zeros(T, A_DIM)


def _stamp_key():
    return pytest.importorskip("taniteval.hierarchy").MANEUVER_LABEL_KEY


def _run_panel(labels_v2):
    hierarchy = pytest.importorskip("taniteval.hierarchy")
    torch.manual_seed(0)
    eps = [_Ep(str(i)) for i in range(3)]
    return hierarchy.run(_Model().eval(), _Readout(), eps, "cpu",
                         max_eps=3, stride=64, batch=8, n_boot=32,
                         labels_v2=labels_v2)


@pytest.mark.parametrize("labels_v2, want", [(False, "v1"), (True, "v2")])
def test_hierarchy_panel_scores_against_the_arms_own_family(labels_v2, want):
    """⛔ THE CORE PIN. On the gentle-curve corpus the v1 target is
    ``turn_left`` for every window and the v2 target is ``lane_keep`` for every
    window, so ``maneuver_acc`` is a DIFFERENT measurement under the two — which
    is the whole point. Pre-fix, ``labels_v2`` was not a parameter."""
    res = _run_panel(labels_v2)
    assert not res.get("skipped"), res.get("skipped")
    assert res["seam_ctx_to_tactical"][_stamp_key()] == want
    assert res[_stamp_key()] == want


def test_hierarchy_maneuver_acc_actually_moves_with_the_family():
    """The stamp alone could be cosmetic. This asserts the NUMBER changes: the
    stub's manoeuvre head is fixed, the two label families disagree on every
    window of the fixture, so the two accuracies must sum to 1.0."""
    lo = _run_panel(False)["seam_ctx_to_tactical"]["maneuver_acc"]["real"]
    hi = _run_panel(True)["seam_ctx_to_tactical"]["maneuver_acc"]["real"]
    assert lo is not None and hi is not None
    assert lo != hi, ("maneuver_acc did not move when the label family changed "
                      "— the branch is not reaching the scored label")
    assert abs((lo + hi) - 1.0) < 1e-6, (lo, hi)


def test_panel_banks_the_per_window_arrays_it_used_to_drop():
    """⛔ THE INSTRUMENT GAP. When this defect was found, the ONE affected banked
    panel could not be corrected offline — `man_pred` was on disk nowhere. The
    panel builds every array it needs and used to persist only their means."""
    hierarchy = pytest.importorskip("taniteval.hierarchy")
    res = _run_panel(True)
    pw = res["per_window"]
    n = res["n_windows"]
    for k in hierarchy.PER_WINDOW_KEYS:
        assert k in pw, f"{k} is not banked — a re-read still needs a GPU"
        assert len(pw[k]) == n, (k, len(pw[k]), n)
    assert pw[hierarchy.MANEUVER_LABEL_KEY] == "v2"
    # the two families really are both there, and they differ on this fixture
    assert pw["man_tgt"] != pw["man_tgt_alt"]
    assert set(pw["man_tgt"]) == {R.LANE_KEEP}          # v2 on a gentle curve
    assert set(pw["man_tgt_alt"]) == {R.TURN_LEFT}      # v1 on the same windows


def test_the_other_familys_number_is_banked_so_a_rescore_costs_no_gpu():
    """⭐ THE PROPERTY THAT MATTERS: the diagnostic emitted beside the arm's own
    number is EXACTLY what a re-run under the other family would produce. Assert
    it against an actual re-run, so the claim is arithmetic, not a promise."""
    v1, v2 = _run_panel(False), _run_panel(True)
    a1 = v1["seam_ctx_to_tactical"]["maneuver_acc_under_other_label_family"]
    a2 = v2["seam_ctx_to_tactical"]["maneuver_acc_under_other_label_family"]
    assert a1["label_version"] == "v2" and a2["label_version"] == "v1"
    assert a1["real"] == v2["seam_ctx_to_tactical"]["maneuver_acc"]["real"]
    assert a2["real"] == v1["seam_ctx_to_tactical"]["maneuver_acc"]["real"]
    assert a1["mean_ctx"] == v2["seam_ctx_to_tactical"]["maneuver_acc"]["mean_ctx"]
    assert a1["zero_ctx"] == v2["seam_ctx_to_tactical"]["maneuver_acc"]["zero_ctx"]
    # every window of the fixture disagrees between the families, by construction
    assert a1["label_disagreement_rate"] == 1.0 == a2["label_disagreement_rate"]


# --------------------------------------------------------------------------- #
# 4. Source guard — no unconditional v1 labeler may return to the eval path    #
# --------------------------------------------------------------------------- #
_EVAL_LABEL_SITES = ("taniteval/hierarchy.py", "taniteval/planning.py",
                     "taniteval/rollout.py")


@pytest.mark.parametrize("rel", _EVAL_LABEL_SITES)
def test_no_unconditional_v1_labeler_in_the_eval_path(rel):
    """A regression that re-hardcodes the family would otherwise be invisible:
    the panel keeps producing a plausible number. `classify_maneuver` may still
    be REFERENCED in prose; what is banned is CALLING it."""
    path = os.path.join(_TE, *rel.split("/"))
    src = open(path, encoding="utf-8").read()
    body = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    hits = re.findall(r"\brl\.classify_maneuver\s*\(|"
                      r"\brefb_labels\.classify_maneuver\s*\(", body)
    assert not hits, (
        f"{rel} calls the v1 manoeuvre labeler directly ({hits}); it must go "
        "through refb_labels.window_maneuver_labels_for(..., v2=<arm's "
        "cfg.v2_labels>) or every --v2 arm is mis-scored again")
    assert "window_maneuver_labels_for(" in body, (
        f"{rel} derives no manoeuvre label through the shared dispatcher")


def test_loader_exposes_the_arms_label_family():
    """The resolution rule is the TRAINER's ``cfg.v2_labels``, read off the
    arm's own rebuilt config — not a second rule invented eval-side."""
    loaders = pytest.importorskip("taniteval.loaders")

    class _C:
        v2_labels = True

    assert loaders.resolve_labels_v2({"key": "a"}, _C()) is True
    assert loaders.resolve_labels_v2({"key": "a"}, type("X", (), {})()) is False
    # an entry may declare it for arms whose cfg cannot be rebuilt ...
    assert loaders.resolve_labels_v2({"key": "a", "labels_v2": True},
                                     type("X", (), {})()) is True
    # ... but a declaration that CONTRADICTS the run config is not tolerated
    with pytest.raises(ValueError, match="labels_v2 disagreement"):
        loaders.resolve_labels_v2({"key": "a", "labels_v2": False}, _C())


def test_run_signature_carries_the_flag():
    """Cheap canary: the three eval entry points must all take it, so a caller
    physically cannot omit the decision by accident."""
    import inspect
    for mod, fn in (("taniteval.hierarchy", "run"),
                    ("taniteval.planning", "run"),
                    ("taniteval.rollout", "collect")):
        m = pytest.importorskip(mod)
        assert "labels_v2" in inspect.signature(getattr(m, fn)).parameters, \
            f"{mod}.{fn} cannot be told which label family the arm was trained on"
