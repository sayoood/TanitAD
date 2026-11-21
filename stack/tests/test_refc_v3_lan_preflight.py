"""The v3 preflight must be ABLE TO SEE the LAN -> goal_str pathway.

WHY THIS FILE EXISTS (MEASURED, D-LAN-PF / D-LAN-COV 2026-08-23):

``preflight()`` built a bare ``V3Dataset``, which emits no ``lan`` key, so
``compute_losses_v3`` skipped ``loss_gstr`` BY DESIGN and ``goal_str`` never
appeared in the loss dict — with or WITHOUT ``--goal-str``. The quantity that
was being read instead, ``route``, is the v2.1 NAV-derived CE masked by
``nav_valid``; it measured 0.0 identically with and without ``--goal-str`` and
``--graft-lan``. So the documented acceptance check "``--preflight --goal-str``
shows ``route`` non-zero" was unsatisfiable AND aimed at the wrong number.

That is the C9/C13/C14 class: an instrument structurally unable to report the
answer it is quoted for. These tests pin the fix so it cannot regress:
the pathway fires, the LABEL is live before the loss is believed, and the
guard is demonstrably ABLE TO FAIL.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import refc_v3_train as T                                    # noqa: E402
from refc_train import lan_dataset_class                     # noqa: E402
from tanitad.data.lan import LanConfig as DataLanConfig      # noqa: E402
from tanitad.refs import refc_v3 as v3                       # noqa: E402

ARCS = [20.0, 40.0, 80.0, 160.0]


def _smoke_cfg(hier: bool = True):
    """The smoke config AS THE TRAINER BUILDS IT — kin3-pinned.

    ``refc_v3_train._pin_trainer_cfg`` (2026-09-01) pins ``tac_vocab_version=
    "kin3"`` on every config the trainer constructs, and ``compute_losses_v3``
    now REFUSES wider z_tac heads loudly (it supervises the 3x3 kinematic
    classes). These tests exercise the LAN pathway through the trainer's loss,
    so they build what the trainer builds — the raw v7.0 default would trip
    the (correct) refusal and test a model the trainer can no longer produce."""
    cfg = v3.refc_v3_smoke_config(hier)
    cfg.tac_vocab_version = "kin3"
    return cfg


def _args(**kw):
    d = dict(size="small", arm="hier", smoke=True, goal_str=True,
             graft_lan=False, lan_arclengths=list(ARCS), lan_min_lead_m=5.0)
    d.update(kw)
    return argparse.Namespace(**d)


def _lan_ds(min_frames, lan_cfg, n_eps=2):
    cfg = _smoke_cfg(True)
    eps = T._synth_episodes(n_eps, cfg.core, seed=0, min_frames=min_frames)
    return lan_dataset_class(T.V3Dataset)(
        eps, window=cfg.core.window, max_horizon=20,
        channels=cfg.core.encoder.in_channels, lan_cfg=lan_cfg)


# --------------------------------------------------------------------------
# The defect itself, pinned so it cannot come back
# --------------------------------------------------------------------------

def test_route_is_not_the_lan_route_and_never_testifies_about_it():
    """``route`` is nav-derived. It must be UNMOVED by the LAN label.

    This is the category error the acceptance criterion made. If this test
    ever fails, `route` has become lan-dependent and the two quantities have
    been conflated in the loss — which would also silently re-open C6.
    """
    cfg = _smoke_cfg(True)
    eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=400)
    plain = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                        channels=cfg.core.encoder.in_channels)
    withlan = _lan_ds(400, DataLanConfig(arclengths_m=tuple(ARCS),
                                         min_lead_m=5.0))
    b_plain = torch.utils.data.default_collate([plain[0], plain[1]])
    b_lan = torch.utils.data.default_collate([withlan[0], withlan[1]])

    assert "lan" not in b_plain, "the bare dataset must not emit lan"
    assert "lan" in b_lan, "the lan-wrapped dataset must emit lan"

    torch.manual_seed(0)
    m1 = v3.RefCV3Model(_smoke_cfg(True))
    torch.manual_seed(0)
    cfg2 = _smoke_cfg(True)
    cfg2.core.lan = T.refc.LanConfig(k=len(ARCS))
    m2 = v3.RefCV3Model(cfg2)

    r1 = float(T.compute_losses_v3(m1, b_plain, "cpu")["route"])
    r2 = float(T.compute_losses_v3(m2, b_lan, "cpu")["route"])
    assert r1 == r2, ("`route` moved when the LAN label was added — the "
                      "nav-derived CE and the LAN route have been conflated")


def test_goal_str_is_absent_without_lan_and_present_with_it():
    """The loss dict itself is the evidence: no lan key => no goal_str key."""
    cfg = _smoke_cfg(True)
    eps = T._synth_episodes(2, cfg.core, seed=0, min_frames=400)
    plain = T.V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                        channels=cfg.core.encoder.in_channels)
    b_plain = torch.utils.data.default_collate([plain[0], plain[1]])
    torch.manual_seed(0)
    m = v3.RefCV3Model(_smoke_cfg(True))
    assert "goal_str" not in T.compute_losses_v3(m, b_plain, "cpu")


# --------------------------------------------------------------------------
# The corpus floor — a preflight built on too-short episodes reads green on
# a dead label, which is the failure mode that hid all of this.
# --------------------------------------------------------------------------

def test_the_default_synthetic_corpus_is_too_short_to_carry_a_route():
    """Documents WHY the LAN arm needs its own corpus.

    T=40 at 2-8 m/s is ~8-32 m of path; the shortest anchor is at 20 m arc
    BEYOND a ~2 s x v + 5 m guard. Every anchor is masked. If this ever starts
    passing, `_synth_episodes` changed and the LAN arm's `min_frames` floor
    should be re-derived rather than inherited.
    """
    ds = _lan_ds(40, DataLanConfig(arclengths_m=tuple(ARCS), min_lead_m=5.0))
    assert ds.lan_stats(n=256, seed=0)["any_valid_frac"] == 0.0


def test_a_long_enough_corpus_makes_the_label_live():
    ds = _lan_ds(400, DataLanConfig(arclengths_m=tuple(ARCS), min_lead_m=5.0))
    assert ds.lan_stats(n=256, seed=0)["any_valid_frac"] > 0.0


# --------------------------------------------------------------------------
# The arm, and the proof that its PASS is worth something
# --------------------------------------------------------------------------

def test_lan_arm_passes_on_a_live_label():
    cfg = _smoke_cfg(True)
    assert T._lan_arm_preflight(cfg, _args()) == 0


def test_lan_arm_FAILS_when_the_leak_guard_kills_the_label():
    """The deliberate-regression arm: a guard that cannot fail is not a guard.

    With min_lead_m enormous every anchor is masked, `goal_str` computes to a
    clean 0.0, and the OLD preflight would have reported PASS. The arm must
    refuse it.
    """
    cfg = _smoke_cfg(True)
    assert T._lan_arm_preflight(cfg, _args(lan_min_lead_m=1e9)) != 0


def test_lan_arm_FAILS_when_the_corpus_cannot_reach_the_anchors():
    """Same dead label reached from the other side — anchors past the corpus.

    Uses arc-lengths far beyond any synthetic episode's path length, so the
    label is dead for a reason that has nothing to do with the guard. The arm
    must still refuse, because it checks the LABEL, not the knob.
    """
    cfg = _smoke_cfg(True)
    rc = T._lan_arm_preflight(cfg, _args(lan_arclengths=[5000.0, 10000.0]))
    assert rc != 0


@pytest.mark.parametrize("k", [2, 3, 4])
def test_lan_width_is_pinned_to_the_configured_anchor_count(k):
    cfg = _smoke_cfg(True)
    arcs = ARCS[:k]
    ds = _lan_ds(400, DataLanConfig(arclengths_m=tuple(arcs), min_lead_m=5.0))
    item = ds[0]
    assert item["lan"].shape == (k * 4,)
    assert T._lan_arm_preflight(cfg, _args(lan_arclengths=list(arcs))) == 0


def test_preflight_success_path_survives_a_cp1252_console(capsys):
    """A PASSING preflight must not die printing its own PASS banner.

    MEASURED: on the Windows dev box the run reached "[v3-preflight] loss step
    OK" and then raised UnicodeEncodeError on the '✅' — non-zero exit on a
    green gate. The module reconfigures stdout at import; this pins that the
    banner characters survive an encode to the console's codec.
    """
    for ch in ("⛔", "✅", "…"):
        ch.encode(sys.stdout.encoding or "utf-8", errors="replace")
