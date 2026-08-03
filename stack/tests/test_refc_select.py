"""D-SEL — REF-C's SELECTION surface. Rationale: tanitad/refs/refc_select.py.

What this file pins, in the order the pre-registration argues it:

(a) CAPACITY CONTROL. The whole selection surface is +385 parameters on
    REF-C-base and the ``param_breakdown`` still sums to ``total`` exactly. This
    is the control that caught a +272,001-parameter tactical head where +897
    sufficed; a lever whose cost is not pinned is a lever nobody can audit.
(b) BYTE-IDENTICAL WHEN OFF, and BIT-IDENTICAL where the maths says so:
    an all-off build has no new state_dict keys, ``steps=0`` makes
    ``refined_logits is anchor_logits`` by construction, and a seam clamp that
    is not binding rescales by exactly 1.0.
(c) SELF-CONSISTENCY, with a fail-loud runtime guard. The consequence score must
    have a real candidate axis and must not be constant along it — the exact
    silent failure ``imagine_probes`` shipped with for months (MEASURED: 32
    tokens serving all 256 candidates, so E-V5-1's imagination negative was
    over-determined). The guard raises instead of returning a clean nothing.
(d) THE SEAM ACTUATOR. The clamp bounds the graft/base norm ratio, and its
    fail-loud is a POPULATION-OVER-TIME condition that a transient spike can
    never trigger — the flagship's batch-max version killed two healthy arms.
(e) THE C6 GUARD. ``--graft-route --labels v1`` is refused at parse time,
    because that route target is a deterministic function of a model INPUT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import refc_train                                                   # noqa: E402
from tanitad.refs import refc_select as sl                          # noqa: E402
from tanitad.refs.refc import (RefCModel, SelectionConfig,          # noqa: E402
                               param_breakdown, refc_config,
                               refc_select_config, refc_smoke_config)
from tests.test_refc import _batch, _make_cached_root               # noqa: E402


def _cfg(**flags):
    cfg = refc_smoke_config()
    for k, v in flags.items():
        setattr(cfg, k, v)
    return cfg


def _build(seed: int = 0, **flags) -> RefCModel:
    torch.manual_seed(seed)
    return RefCModel(_cfg(**flags))


ALL_ON = dict(sel_refined=True, sel_reach_clamp=True, graft_cons=True,
              graft_route=True, seam_clamp=1.0)


# ---------- (a) capacity control ----------------------------------------------

def test_dsel_is_not_a_capacity_change():
    """The ENTIRE selection surface costs +385 parameters on REF-C-base.

    S1 (rank on the refined confidence), S2 (reachability band) and S4 (seam
    clamp) are structurally FREE. S3 is +1: the consequence score reuses the
    decoder's own ``feat_proj`` + ``conf_head`` plus a parameter-free
    ``layer_norm``, so only the gate is new — a dedicated projection would have
    cost ~270 k. S5 is +384 = N_ROUTE x n_anchors, the same shape and the same
    zero-init as ``lon_to_anchor``.

    +385 / 104,191,577 = +0.00037 %, the same order as the D-TAC1 F1 arm's +384
    and ~1/700 of the +272,001 an earlier two-MLP tactical head cost before its
    own capacity check caught it.
    """
    with torch.device("meta"):
        base = param_breakdown(RefCModel(refc_config()))
        sel = param_breakdown(RefCModel(refc_select_config()))
    print(f"[d-sel] base {base['total']:,} -> {sel['total']:,} "
          f"(+{sel['total'] - base['total']})")
    assert base["total"] == 104_191_577                  # the registry's number
    assert sel["total"] - base["total"] == 385
    assert sel["selection"] == 385 and base["selection"] == 0
    # The breakdown must keep summing exactly — `selection` is CARVED OUT of
    # `decoder`, never added on top of it.
    for bd in (base, sel):
        assert sum(v for k, v in bd.items() if k != "total") == bd["total"]
    assert sel["decoder"] == base["decoder"]             # carve-out, not growth
    assert sel["encoder"] == base["encoder"]             # nothing touches vision


def test_each_lever_costs_what_it_says():
    """Per-lever capacity, so a future edit cannot smuggle parameters into one."""
    with torch.device("meta"):
        n0 = param_breakdown(RefCModel(refc_config()))["total"]
        for flags, cost in (({"sel_refined": True}, 0),
                            ({"sel_reach_clamp": True}, 0),
                            ({"seam_clamp": 1.0}, 0),
                            ({"graft_cons": True}, 1),
                            ({"graft_route": True}, 384)):
            cfg = refc_config()
            for k, v in flags.items():
                setattr(cfg, k, v)
            got = param_breakdown(RefCModel(cfg))["total"] - n0
            assert got == cost, f"{flags}: +{got}, expected +{cost}"


def test_ego_valid_channel_is_exactly_one_input_column_per_reader():
    """X15's validity flag is one column into the measurement encoder, and one
    more into the tactical head ONLY when that head reads the speed at all —
    a flag is meaningless to a reader that never sees the channel it qualifies."""
    with torch.device("meta"):
        n0 = param_breakdown(RefCModel(refc_config()))["total"]
        cfg = refc_config()
        cfg.ego_valid_channel = True
        n1 = param_breakdown(RefCModel(cfg))["total"]
        cfg.tactical_speed_input = True
        n2 = param_breakdown(RefCModel(cfg))["total"]
        cfg2 = refc_config()
        cfg2.tactical_speed_input = True
        n_f1 = param_breakdown(RefCModel(cfg2))["total"]
    hid, aux = refc_config().measurement.hidden, refc_config().decoder.aux_hidden
    assert n1 - n0 == hid                              # +128 measurement only
    assert n_f1 - n0 == aux                            # +384: the shipped F1 arm
    #                                                    (unchanged by D-SEL)
    # Turning F1 on while the flag is already there adds BOTH tactical columns.
    assert n2 - n1 == 2 * aux                          # +768
    # The flag's own marginal cost on top of F1 is one column per reader.
    assert n2 - n_f1 == hid + aux                      # +512


# ---------- (b) byte-identical when off ---------------------------------------

def test_all_off_adds_no_keys_and_is_bit_identical():
    """A D-SEL-off build must be indistinguishable from one that never had it."""
    off1, off2 = _build(), _build()
    s1, s2 = off1.state_dict(), off2.state_dict()
    assert set(s1) == set(s2)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    assert not any("route_to_anchor" in k or "cons_gate" in k for k in s1)
    on = _build(**ALL_ON)
    assert set(on.state_dict()) - set(s1) == {"decoder.route_to_anchor.weight",
                                              "decoder.cons_gate"}
    assert not set(s1) - set(on.state_dict())


def test_zero_init_grafts_do_not_move_the_pick_at_step_zero():
    """S5's route graft and S3's gate are ZERO at construction, so the ranked
    score is bit-identical to the graft-free one on the first forward — the
    ``ctx_to_cond`` / ``lon_to_anchor`` discipline, so any later change is
    attributable to the seam rather than to a random init."""
    root = _make_cached_root(Path(__file__).parent / "_dsel_tmp") \
        if False else None                      # (kept explicit; see fixtures)
    del root
    frames = torch.randn(4, 4, 1, 64, 64)
    v0 = torch.tensor([3.0, 10.0, 20.0, 0.5])
    off = _build().eval()
    on = _build(graft_route=True, graft_cons=True).eval()
    on.load_state_dict(off.state_dict(), strict=False)
    with torch.no_grad():
        a = off(frames, v0=v0, steps=2)
        b = on(frames, v0=v0, steps=2)
    assert float(on.decoder.route_to_anchor.weight.abs().max()) == 0.0
    assert float(on.decoder.cons_gate.abs().max()) == 0.0
    assert torch.equal(a["sel_score"], b["sel_score"])
    assert torch.equal(a["sel_idx"], b["sel_idx"])


def test_s1_is_inert_at_zero_denoise_steps_by_construction():
    """``--mode classifier`` runs 0 denoise passes, so there IS no refined
    readout and S1 cannot change anything. Pinned as an IDENTITY (`is`), not an
    approximate equality: a future refactor that made these merely close would
    silently turn a control arm into a treatment arm."""
    m = _build(sel_refined=True).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 4, 1, 64, 64), v0=torch.tensor([5.0, 9.0]),
                steps=0)
    assert out["refined_logits"] is out["anchor_logits"]
    assert torch.equal(out["sel_score"], out["anchor_logits"])


def test_s1_actually_changes_the_ranking_when_steps_are_taken():
    """...and with denoise steps it is a REAL lever, not a no-op flag."""
    frames = torch.randn(6, 4, 1, 64, 64)
    v0 = torch.tensor([3.0, 10.0, 20.0, 0.5, 7.0, 14.0])
    off = _build().eval()
    on = _build(sel_refined=True).eval()
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        a, b = off(frames, v0=v0, steps=2), on(frames, v0=v0, steps=2)
    assert not torch.equal(a["sel_score"], b["sel_score"])
    assert torch.equal(a["anchor_logits"], b["anchor_logits"])   # t=0 untouched
    assert torch.equal(a["anchor_traj"], b["anchor_traj"])       # fan untouched


# ---------- (c) self-consistency: the candidate-axis guard --------------------

def test_consequence_score_has_a_real_candidate_axis():
    """S3's whole justification is that it CAN rank, unlike ``imagine_probes``.

    MEASURED 2026-07-27 (``t4_imagination_conditioning.json``): the flagship's
    probe rollout returns 32 tokens invariant to ``n_anchors`` and IDENTICAL for
    all 256 candidates, so ``IMAGINATION_HAS_CANDIDATE_AXIS is False`` and no
    ranking built on it could ever have worked. REF-C's ``law_head`` consumes
    the trajectory, so its output varies per candidate — and that is asserted
    here rather than assumed."""
    from tanitad.models.flagship_v15 import IMAGINATION_HAS_CANDIDATE_AXIS
    assert IMAGINATION_HAS_CANDIDATE_AXIS is False       # the thing NOT ported
    m = _build(graft_cons=True).eval()
    with torch.no_grad():
        out = m(torch.randn(3, 4, 1, 64, 64), v0=torch.tensor([4.0, 8.0, 12.0]),
                steps=2)
    cons = out["cons_score"]
    assert cons.shape == (3, m.cfg.anchors.n_anchors)
    # a REAL candidate axis: not constant along it, for every row
    assert float(cons.std(dim=1).min()) > 0.0
    sl.assert_candidate_axis(cons, m.cfg.anchors.n_anchors, name="test")


def test_the_guard_fires_on_a_candidate_blind_score():
    """The fail-loud runtime guard: a score that is CONSTANT along the candidate
    axis ranks nothing, and must raise rather than return a clean negative."""
    n = 8
    with pytest.raises(sl.NoCandidateAxis, match="CONSTANT along it"):
        sl.assert_candidate_axis(torch.ones(3, n) * 2.5, n, name="flat")
    with pytest.raises(sl.NoCandidateAxis, match="no candidate axis"):
        sl.assert_candidate_axis(torch.randn(3, n + 1), n, name="wrong width")
    # ...and it is THE flagship's guard, not a second implementation of it.
    from tanitad.models import flagship_v15
    assert sl.NoCandidateAxis is flagship_v15.NoCandidateAxis
    assert sl.reachability_mask.__module__ == "tanitad.refs.refc_select"
    m = _build(graft_cons=True).train()
    with pytest.raises(sl.NoCandidateAxis):
        m.decoder.conf_head.weight.data.zero_()      # collapse the readout
        m.decoder.conf_head.bias.data.zero_()
        m(torch.randn(2, 4, 1, 64, 64), v0=torch.tensor([5.0, 9.0]), steps=2)


def test_the_world_model_is_not_corrupted_by_the_ranking_objective():
    """``cons_detach`` (default) runs ``law_head`` under ``no_grad``, so LAW
    stays trained by its own MSE alone — the flagship's FROZEN-predictor
    discipline. The gate still receives gradient, so the lever is gated, not
    dead."""
    root = _make_cached_root(_tmp())
    m = _build(graft_cons=True).train()
    batch = _batch(root, m.cfg)
    out = refc_train.compute_losses(m, batch)
    out["loss"].backward()
    assert m.decoder.cons_gate.grad is not None
    assert float(m.decoder.cons_gate.grad.abs().sum()) > 0.0, "gated but DEAD"
    # law_head still trains (its own MSE), and every parameter has a gradient.
    assert float(m.law_head[0].weight.grad.abs().sum()) > 0.0
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


# ---------- (d) S2 reachability, S4 seam clamp --------------------------------

def test_reach_clamp_masks_only_the_argmax_and_never_empties_a_window():
    """The band filters the ARGMAX; the returned score stays unmasked so no
    ``-inf`` can reach a cross-entropy, and a row whose survivor set is empty
    keeps its whole fan — an unreachable-everywhere window is a measurement
    failure, not a licence to emit nothing."""
    frames = torch.randn(5, 4, 1, 64, 64)
    v0 = torch.tensor([0.0, 3.0, 10.0, 25.0, 40.0])
    m = _build(sel_reach_clamp=True).eval()
    with torch.no_grad():
        out = m(frames, v0=v0, steps=2)
    assert torch.isfinite(out["sel_score"]).all()          # no -inf escapes
    assert out["sel_tele"]["reach_frac_windows_empty"] == 0.0
    assert 0.0 < out["sel_tele"]["reach_frac_candidates_clipped"] < 1.0
    # the pick is always a real anchor index
    assert 0 <= int(out["sel_idx"].min()) and \
        int(out["sel_idx"].max()) < m.cfg.anchors.n_anchors


def test_reach_clamp_never_leaks_past_ego_dropout():
    """``v_ms`` is the speed BEFORE ego-dropout. Filtering candidates with it on
    a sample whose speed was WITHHELD would smuggle the channel back in through
    the ranking — the failure ``flagship_v15``'s ``vt_keep`` masking exists to
    prevent. With ``ego_dropout = 1.0`` every training sample is dropped, so the
    band must be inert; at eval nothing is dropped, so it must bite."""
    frames = torch.randn(4, 4, 1, 64, 64)
    v0 = torch.tensor([1.0, 8.0, 16.0, 24.0])
    m = _build(sel_reach_clamp=True, ego_dropout=1.0)
    torch.manual_seed(0)
    m.train()
    out_tr = m(frames, v0=v0, steps=2)
    assert out_tr["sel_tele"]["reach_frac_candidates_clipped"] == 0.0
    m.eval()
    with torch.no_grad():
        out_ev = m(frames, v0=v0, steps=2)
    assert out_ev["sel_tele"]["reach_frac_candidates_clipped"] > 0.0


def test_seam_clamp_is_bit_identical_below_the_cap_and_bounds_above_it():
    base = torch.randn(4, 16)
    small = torch.randn(4, 16) * 1e-3
    st = sl.SeamState()
    out, tele = sl.apply_seam_clamp(base, small, clamp=1.0, fail=1.5,
                                    fail_frac=0.75, patience=50, state=st,
                                    surface="rank")
    assert torch.equal(out, base + small)          # scale is EXACTLY 1.0
    assert tele["seam_rank_bound_frac"] == 0.0
    big = base * 12.0
    out2, tele2 = sl.apply_seam_clamp(base, big, clamp=1.0, fail=99.0,
                                      fail_frac=0.75, patience=50, state=st,
                                      surface="rank")
    ratio = (out2 - base).norm(dim=-1) / base.norm(dim=-1)
    assert float(ratio.max()) <= 1.0 + 1e-5
    assert tele2["seam_rank_bound_frac"] == 1.0
    assert tele2["seam_rank_ratio_max"] == 1.0
    # off => untouched, and the counter cannot accumulate while disabled
    out3, tele3 = sl.apply_seam_clamp(base, big, clamp=0.0, fail=1.5,
                                      fail_frac=0.75, patience=1, state=st,
                                      surface="rank")
    assert torch.equal(out3, base + big) and tele3 == {} and st.sat_steps == 0


def test_seam_fail_is_sustained_saturation_not_a_batch_spike():
    """The flagship's first fail-loud fired on ``ratio.max()`` and one sample of
    64 could kill a run — MEASURED, it lost BOTH wide arms of a geometry
    validation at ~step 350 on arms training at or below their control (C51).
    So: a transient must never fire, and the counter must RESET on any break."""
    base, big = torch.randn(4, 16), None
    big = torch.randn(4, 16) * 50.0
    st = sl.SeamState()
    kw = dict(clamp=1.0, fail=1.5, fail_frac=0.75, patience=3, state=st,
              surface="conf")
    for i in range(2):                                   # 2 < patience
        sl.apply_seam_clamp(base, big, **kw)
    assert st.sat_steps == 2
    sl.apply_seam_clamp(base, torch.zeros(4, 16), **kw)  # one healthy step
    assert st.sat_steps == 0, "a transient must not accumulate into a kill"
    for i in range(2):
        sl.apply_seam_clamp(base, big, **kw)
    with pytest.raises(RuntimeError, match="SATURATED"):
        sl.apply_seam_clamp(base, big, **kw)
    assert st.sat_steps == 0, "a caught error must stay recoverable"


def test_seam_clamp_bounds_the_real_tactical_grafts():
    """End-to-end: the actuator acts on the grafts REF-C already logs the norms
    of (``graft_lat_norm`` / ``graft_lon_norm`` / ``conf_norm``)."""
    m = _build(factored_maneuver=True, seam_clamp=1.0).eval()
    with torch.no_grad():
        out = m(torch.randn(4, 4, 1, 64, 64), v0=torch.tensor([3., 9., 15., 21.]),
                steps=2)
    t = out["sel_tele"]
    assert t["seam_conf_ratio_max"] <= 1.0
    assert "seam_conf_ratio_preclamp_mean" in t and "seam_conf_sat_steps" in t


# ---------- (e) the trainer ----------------------------------------------------

def _tmp():
    import tempfile
    return Path(tempfile.mkdtemp())


def test_trainer_runs_every_lever_and_reports_the_selection_family(tmp_path):
    root = _make_cached_root(tmp_path)
    out = tmp_path / "dsel"
    refc_train.main([
        "--data-root", str(root), "--out", str(out), "--steps", "2",
        "--batch", "4", "--smoke", "--log-every", "1", "--workers", "0",
        "--labels", "v21", "--mode", "diffusion",
        "--sel-refined", "--sel-reach-clamp", "--graft-cons", "--graft-route",
        "--seam-clamp", "1.0", "--ego-valid-channel"])
    conf = json.loads((out / "config.json").read_text())
    for k in ("sel_refined", "sel_reach_clamp", "graft_cons", "graft_route",
              "ego_valid_channel"):
        assert conf["cfg"][k] is True, k
    assert conf["cfg"]["seam_clamp"] == 1.0
    assert conf["param_breakdown"]["selection"] > 0
    met = json.loads((out / "metrics.json").read_text())
    # The ranking diagnostic must be in the log line, not reconstructed later.
    for k in ("cls_refined", "oracle_ade", "sel_ade", "sel_gap", "rank_acc",
              "frac_sel_2x_worse", "seam_conf_ratio_max",
              "reach_frac_candidates_clipped"):
        assert k in met["final"], f"{k} missing from the step log"
    assert met["final"]["sel_gap"] >= -1e-6      # selected is never < oracle
    assert met["param_breakdown"]["total"] == met["n_params_trainable"]


def test_trainer_refuses_the_circular_route_graft(tmp_path):
    """⛔ C6. Under ``--labels v1`` the route CE target is
    ``route_target(nav_cmd)`` — a deterministic function of a model INPUT — so
    grafting that readout onto SELECTION would train the ranking on a nav echo
    (the same class as R-2026-08-03-l, where flagship's route accuracy of
    1.0000 turned out to be an oracle-conditioning echo). Refused at parse
    time, before a GPU-second is spent."""
    root = _make_cached_root(tmp_path)
    with pytest.raises(SystemExit, match="requires --labels v21 or v3"):
        refc_train.main([
            "--data-root", str(root), "--out", str(tmp_path / "x"),
            "--steps", "1", "--batch", "4", "--smoke", "--labels", "v1",
            "--graft-route"])


def test_selection_config_defaults_are_todays_behaviour():
    """A decoder built with a default ``SelectionConfig`` must be the pre-D-SEL
    decoder — the projection cannot smuggle a lever on by omission."""
    s = SelectionConfig()
    assert not s.any_on
    assert (s.refined, s.reach_clamp, s.graft_cons, s.graft_route) == \
        (False, False, False, False)
    assert s.seam_clamp == 0.0
    # ...and the projection off RefCConfig derives the horizon rather than
    # hard-coding 2.0 s: the band and the anchors must agree on plan length.
    cfg = refc_config()
    assert cfg.selection().horizon_s == max(cfg.trajectory.horizons) * 0.1
