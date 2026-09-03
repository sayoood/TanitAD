"""GS-8 — the ANCHORED action-divergence diagnostic must detect what it hunts, and every
control must read its KNOWN value (CLAUDE.md probe-panel rule; C133 family).

What is pinned, and why each group exists:

  (1) THE STATISTIC on a synthetic predictor with a PLANTED linear action response:
      F_sep far above its label-permutation null, Spearman monotonicity +1 on both
      axes, cos(m(+L), m(−L)) = −1 exactly, and the verdict SENSITIVE.
  (2) THE CONTROLS read their known values through the REAL code path: C0 identity
      == 0.0 exactly, the zero model == 0.0 exactly (an action-blind predictor is VOID,
      never "insensitive"), the shuffled-label null sits at F ≈ 1, and the output-scale
      control is invariant.
  (3) THE SHUFFLED-ACTION CONTROL COLLAPSES SEPARATION: with the labels decoupled from
      the responses the between/within ratio falls into the null band.
  (4) THE WINDOW RULE and the LIFT reproduce the banked ``actdiv`` construction —
      ``range(0, n, n // 5)``; and the default lift IS the trainer's ``_lift3``
      (v_last / SPEED_SCALE = 10), not the banked scripts' v_first / 30.
  (5) THE VERDICT TABLE (SPEC.md §4) on crafted summaries, including the units of the
      magnitude bar (per-dim RMS over per-dim scene std).
  (6) A FRESH tiny ``V6Stack`` (zero-init FiLM ⇒ action-blind by construction) reads
      EXACTLY zero displacement through the real predictor path.
  (7) The refav1 absolute-level grid and its per-axis verdict table.

CPU only; no checkpoint, no corpus, no GPU.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[2]
_STACK = _REPO / "stack"
for _p in (str(_STACK), str(_STACK / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
TOOL = _REPO / "taniteval" / "tools" / "actdiv_anchored.py"
_spec = importlib.util.spec_from_file_location("actdiv_anchored_under_test", TOOL)
aa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aa)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _batch(n=64, W=6, S=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    zs = torch.randn(n, W, S, generator=g)
    a2 = torch.randn(n, W, 2, generator=g) * torch.tensor([0.02, 0.6])
    a3 = torch.cat([a2, torch.full((n, W, 1), 0.5)], -1)
    return zs, a3


@pytest.fixture(scope="module")
def sensitive_read():
    zs, a3 = _batch()
    return aa.read_arm(aa.LinearResponsePredictor(24, seed=0), zs, a3, [1], (1,),
                       n_perm=60, seed=0)["horizons"]["h1"]


# ---------------------------------------------------------------------------
# (1) planted structure is recovered
# ---------------------------------------------------------------------------
def test_planted_linear_response_reads_SENSITIVE(sensitive_read):
    g = sensitive_read["grid"]
    assert g["F_over_null_p95"] > 50, g["F_over_null_p95"]
    for ax in aa.AXIS_NAMES:
        a = g["axes"][ax]
        # 10 candidates per axis carry TIED |level| ranks (± signs), so the attainable
        # Spearman maximum is ~0.985, not 1.0; the tool's own bar is 0.8
        assert a["spearman_rho_abslevel_vs_norm"] > 0.95
        assert all(v == pytest.approx(-1.0, abs=1e-4) for v in a["sign_cos_by_level"].values())
        assert a["monotone"] and a["sign_consistent"]
    assert sensitive_read["verdict"]["verdict"] == "SENSITIVE"


def test_norms_grow_linearly_with_level_for_a_linear_response(sensitive_read):
    nb = sensitive_read["grid"]["axes"]["kappa"]["norm_by_level"]
    lv = np.array([float(k) for k in nb])
    nm = np.array(list(nb.values()))
    ratio = nm / lv
    assert np.allclose(ratio, ratio[0], rtol=0.15)


# ---------------------------------------------------------------------------
# (2) controls read their known values through the real code path
# ---------------------------------------------------------------------------
def test_C0_identity_and_zero_model_read_exactly_zero(sensitive_read):
    c = sensitive_read["controls"]
    assert c["C0_identity_max_abs_diff"] == 0.0 and c["C0_passes"]
    assert c["zero_model_max_abs_displacement"] == 0.0 and c["zero_model_passes"]


def test_shuffled_label_null_sits_at_one(sensitive_read):
    null = sensitive_read["grid"]["null"]
    assert 0.7 < null["median"] < 1.3, null
    assert null["p95"] < 2.0


def test_output_scale_control_is_invariant(sensitive_read):
    osc = sensitive_read["controls"]["output_scale"]
    assert osc["invariant"], osc


def test_action_blind_predictor_is_VOID_not_insensitive():
    zs, a3 = _batch(seed=3)
    r = aa.read_arm(aa.ActionBlindPredictor(), zs, a3, [1], (1,), n_perm=20)["horizons"]["h1"]
    assert r["grid"]["max_abs_displacement"] == 0.0
    assert r["verdict"]["verdict"] == "VOID"
    assert any("identically zero" in s for s in r["verdict"]["reasons"])


# ---------------------------------------------------------------------------
# (3) the shuffled-action control collapses separation
# ---------------------------------------------------------------------------
def test_shuffling_labels_collapses_F_sep_to_the_floor():
    rng = np.random.default_rng(0)
    N, C, S = 80, 8, 16
    M = rng.standard_normal((C, S)) * 3.0
    D = M[None] + 0.3 * rng.standard_normal((N, C, S))
    f_true = aa.between_within(D)["F_sep"]
    null = aa.shuffled_label_null(D, n_perm=40, seed=1)
    assert f_true > 20 * null["p95"]
    assert 0.6 < null["median"] < 1.5


def test_realised_mode_shuffled_actions_fall_to_the_floor():
    """Each window's own response is a function of its own action; feeding the ROLLED
    actions (window i gets window i+j's) while labelling by window i's ORIGINAL bin must
    drop the between/within ratio to the floor. Quartile bins of a continuous 2-D
    response leave the other channel's variation inside every bin, so the TRUE ratio is
    moderate by construction; what is pinned is the DROP under the roll."""
    rng = np.random.default_rng(2)
    N, S = 400, 12
    a2 = rng.standard_normal((N, 2)) * np.array([0.02, 0.6])
    M = rng.standard_normal((S, 2))
    d_true = (a2 / np.array([0.02, 0.6])) @ M.T + 0.05 * rng.standard_normal((N, S))
    rolled = [np.roll(d_true, j, axis=0) for j in (1, 2, 3)]
    rd = aa.realised_reads(d_true, rolled, a2, a2.std(0))
    for key in ("kappa", "accel"):                 # signed quartiles separate a linear response
        assert rd[key]["F_sep"] > 8.0, (key, rd[key]["F_sep"])
        assert rd[key]["F_sep_shuffled_actions_max"] < 3.0, (key, rd[key])
        assert rd[key]["F_sep"] > 4 * rd[key]["F_sep_shuffled_actions_max"], key
    # |a| bins pool +a with −a: an ODD (linear) response cancels inside every bin, so
    # this binning cannot separate it — pinned so the reading is never over-read
    assert rd["abs_a"]["F_sep"] < 3.0 and rd["abs_a"]["F_sep_shuffled_actions_max"] < 3.0


# ---------------------------------------------------------------------------
# (4) the banked window rule and the trainer's lift
# ---------------------------------------------------------------------------
def test_window_starts_reproduce_the_banked_actdiv_rule():
    assert aa.window_starts(54) == [0, 10, 20, 30, 40, 50]      # 60 frames, W=6 -> 6 windows
    assert aa.window_starts(3) == []


def test_default_lift_is_the_trainers_lift3_not_the_banked_v_over_30():
    from train_v6_staged import _lift3, COND_INCUMBENT
    from tanitad.models.flagship_v15 import SPEED_SCALE
    a2 = torch.randn(5, 6, 2)
    v = torch.rand(5) * 20
    got = aa.lift_trainer(a2, v, COND_INCUMBENT)
    exp = _lift3(a2, v, COND_INCUMBENT)
    assert torch.equal(got, exp)
    assert torch.allclose(got[:, :, 2], (v / SPEED_SCALE)[:, None].expand(-1, 6))
    legacy = aa.lift_legacy(a2, v)
    assert torch.allclose(legacy[:, :, 2], (v / 30.0)[:, None].expand(-1, 6))
    assert SPEED_SCALE == 10.0 and aa.LEGACY_SPEED_SCALE == 30.0


def test_with_last_action_replaces_only_the_requested_positions():
    a3 = torch.randn(3, 6, 3)
    out = aa.with_last_action(a3, [0.1, -0.2], "last")
    assert torch.allclose(out[:, -1, :2], torch.tensor([0.1, -0.2]).expand(3, -1))
    assert torch.equal(out[:, :-1], a3[:, :-1]) and torch.equal(out[:, -1, 2], a3[:, -1, 2])
    out_all = aa.with_last_action(a3, [0.1, -0.2], "all")
    assert torch.allclose(out_all[:, :, :2], torch.tensor([0.1, -0.2]).expand(3, 6, -1))
    assert torch.equal(out_all[:, :, 2], a3[:, :, 2])


# ---------------------------------------------------------------------------
# (5) the verdict table, including the magnitude units
# ---------------------------------------------------------------------------
def _summary(f_over=20.0, rho=0.95, cos=-0.9, rel=0.1, scene=0.2, degenerate=False):
    axes = {n: {"spearman_rho_abslevel_vs_norm": rho,
                "sign_cos_by_level": {"1": cos, "2": cos},
                "monotone": rho >= aa.THRESHOLDS["monotone_rho_min"],
                "sign_consistent": cos < aa.THRESHOLDS["sign_cos_max"]}
            for n in aa.AXIS_NAMES}
    return {"F_over_null_p95": f_over, "axes": axes, "rel_mag_at_material_level": rel,
            "scene_spread": scene, "degenerate": degenerate}


@pytest.mark.parametrize("kw,expect", [
    (dict(), "SENSITIVE"),
    (dict(rel=0.01), "STRUCTURED-WEAK"),
    (dict(rho=0.3), "SEPARATED-NONMONOTONE"),
    (dict(cos=+0.2), "SEPARATED-NONMONOTONE"),
    (dict(f_over=2.0), "INSENSITIVE"),
    (dict(scene=0.0), "VOID"),
])
def test_verdict_table(kw, expect):
    assert aa.verdict(_summary(**kw))["verdict"] == expect


def test_verdict_is_VOID_when_a_control_is_off_its_known_value():
    assert aa.verdict(_summary(), c0_max_abs=1e-6)["verdict"] == "VOID"
    assert aa.verdict(_summary(), zero_model_max_abs=1e-9)["verdict"] == "VOID"


def test_material_magnitude_uses_per_dim_rms_over_per_dim_scene_std():
    """A displacement of 0.01 per dim against a scene std of 0.2 per dim is 0.05 — NOT
    0.05·√S. Built directly: every candidate mean = 0.01 in every dim."""
    N, S = 40, 400
    cands = aa.candidate_grid([1.0, 1.0], levels=(2.0,))
    D = np.zeros((N, len(cands), S))
    for j, c in enumerate(cands):
        if c["axis"] >= 0:
            D[:, j, :] = 0.01 * np.sign(c["level"])
    D += 1e-4 * np.random.default_rng(0).standard_normal(D.shape)
    summ = aa.summarize_grid(D, cands, scene_spread=0.2, n_perm=10)
    assert summ["rel_mag_at_material_level"] == pytest.approx(0.05, rel=0.05)
    assert "per-dim RMS" in summ["rel_units"]


# ---------------------------------------------------------------------------
# (6) a fresh tiny V6Stack is action-blind by construction (zero-init FiLM) and
#     reads EXACTLY zero through the real predictor path
# ---------------------------------------------------------------------------
def test_fresh_tiny_v6stack_reads_exactly_zero_displacement():
    from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig
    from tanitad.models.v6 import V6Config, V6Stack
    cfg = V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=32, image_width=32, patch_size=16,
                              d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=4),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1, 2), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, d_plan_feat=16, emission_hidden=16, n_candidates=3,
        aux_hidden=16, sigreg_slices=8)
    torch.manual_seed(0)
    stack = V6Stack(cfg).eval()
    zs = torch.randn(12, 4, cfg.d_op)
    a3 = torch.randn(12, 4, 3)
    r = aa.read_arm(stack.predictor_op, zs, a3, [1, 2], (1,), n_perm=10,
                    scale_control=False)["horizons"]["h1"]
    assert r["controls"]["C0_passes"]
    assert r["grid"]["max_abs_displacement"] == 0.0
    assert r["verdict"]["verdict"] == "VOID"


# ---------------------------------------------------------------------------
# (7) refav1: absolute-level grid, channel mapping, per-axis verdict table
# ---------------------------------------------------------------------------
def test_refav1_grid_puts_kappa_on_channel_1_and_accel_on_channel_0():
    cands = aa.candidate_grid_abs((0.02, 0.1), (0.5,), aa.REFAV1_CHANNELS)
    ids = [c["id"] for c in cands]
    assert ids[0] == "zero" and len(cands) == 1 + 4 + 2
    k = next(c for c in cands if c["id"] == "kappa+0.1")
    a = next(c for c in cands if c["id"] == "accel-0.5")
    assert k["a2"] == [0.0, 0.1] and k["channel"] == 1
    assert a["a2"] == [-0.5, 0.0] and a["channel"] == 0


@pytest.mark.parametrize("fk,fa,expect", [
    (1.0, 20.0, "LAT-INSENSITIVE-CONFIRMED"),
    (20.0, 20.0, "LAT-INSENSITIVE-REFUTED"),
    (20.0, 1.0, "LAT-INSENSITIVE-REFUTED"),
    (1.0, 1.0, "BOTH-INSENSITIVE"),
])
def test_refav1_verdict_table(fk, fa, expect):
    pa = {"kappa": {"F_over_null_p95": fk}, "accel": {"F_over_null_p95": fa}}
    assert aa.verdict_refav1(pa, 0.0, 0.0, 0.1)["verdict"] == expect
    assert aa.verdict_refav1(pa, 1e-7, 0.0, 0.1)["verdict"] == "VOID"


def test_per_axis_separation_isolates_a_one_axis_response():
    """A response on the accel axis only: accel separates, kappa sits at its floor."""
    rng = np.random.default_rng(5)
    cands = aa.candidate_grid_abs((0.02, 0.05, 0.1), (0.5, 1.5), aa.REFAV1_CHANNELS)
    ck = [c for c in cands if c["axis"] >= 0]
    N, S = 60, 32
    M = rng.standard_normal(S)
    D = np.stack([np.stack([c["a2"][0] * M + 0.05 * rng.standard_normal(S) for c in ck])
                  for _ in range(N)])
    pa = aa.axis_separation(D, ck, n_perm=30, seed=0)
    assert pa["accel"]["F_over_null_p95"] > 20
    assert pa["kappa"]["F_over_null_p95"] < 3
    assert aa.verdict_refav1(pa, 0.0, 0.0, 1.0)["verdict"] == "LAT-INSENSITIVE-CONFIRMED"


def test_full_field_exact_stats_match_the_direct_computation():
    rng = np.random.default_rng(7)
    cands = aa.candidate_grid_abs((0.02, 0.1), (0.5, 1.5), aa.REFAV1_CHANNELS)
    ck = [c for c in cands if c["axis"] >= 0]
    N, S = 30, 50
    D = rng.standard_normal((N, len(ck), S)) + np.array([c["a2"][1] * 10 for c in ck])[None, :, None]
    sum_d = D.sum(0)
    sum_sq = (D ** 2).sum((0, 2))
    ex = aa._full_field_stats(sum_d, sum_sq, N, ck)
    direct = aa.between_within(D)
    assert ex["F_sep"] == pytest.approx(direct["F_sep"], rel=1e-9)
    for j, c in enumerate(ck):
        assert ex["mean_norm"][c["id"]] == pytest.approx(direct["norms"][j], rel=1e-9)


# ---------------------------------------------------------------------------
# (8) refav1 END-TO-END on a RANDOM-INIT TINY RefAV1 — the real read path, and
#     the speed-scale contract (D-P2-LEAK-AUDIT / H-LEAK-1).
#
#     A banked instrument family was invalidated this week for feeding v/30 to
#     models trained on v/10. The rule this group pins is not "use 30": it is
#     THIS TOOL NEVER BUILDS THE SPEED CHANNEL AT ALL — it calls the model's own
#     ``augment_actions``, so the scale is whatever the loaded checkpoint's own
#     module says it is, and cannot drift from it.
# ---------------------------------------------------------------------------
def _tiny_refav1(speed_channel=False, seed=0):
    """A random-init RefAV1 small enough for a unit test (~85k params), with a
    FITTED standardizer (the loader refuses an unfitted one at eval)."""
    from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
    torch.manual_seed(seed)
    cfg = RefAV1Config(d_enc=32, n_tokens=8, d_state=32, op_layers=1, op_heads=2,
                       op_window=2, tac_queries=4, tac_layers=1, str_layers=1,
                       str_dim=32, speed_channel=speed_channel)
    m = RefAV1(cfg).eval()
    m.std.fit(torch.randn(4, 8, 32, generator=torch.Generator().manual_seed(seed + 1)))
    for p in m.parameters():
        p.requires_grad_(False)
    return m, cfg


class _FakeLoader:
    """The two attributes and the one method ``refav1_read`` actually touches
    (``_order``/``_cursor`` then ``batch(1)``), with the real key names and
    shapes of ``refav1_arm.build_loader``'s batches."""

    def __init__(self, n=16, T=3, N=8, d=32, K=4, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.feats = torch.randn(n, T, N, d, generator=g)
        self.actions = torch.randn(n, K, 2, generator=g) * torch.tensor([0.8, 0.04])
        self.v0 = 5.0 + 10.0 * torch.rand(n, generator=g)
        self.nav = torch.randint(0, 4, (n,), generator=g)
        self.windows = [(0, i) for i in range(n)]
        self.names, self.W, self._order, self._cursor = ["ep0"], T, [0], 0

    def __len__(self):
        return len(self.windows)

    def batch(self, k):
        i = self._order[0]
        return {"feats": self.feats[i:i + 1], "actions": self.actions[i:i + 1],
                "v0": self.v0[i:i + 1], "nav_cmd": self.nav[i:i + 1]}


def _tiny_read(model, cfg, n=16, n_perm=20, seed=0):
    ld = _FakeLoader(n=n, T=3, N=cfg.n_tokens, d=cfg.d_enc, seed=seed)
    sel = [(i, 0, i) for i in range(n)]
    cands = aa.candidate_grid_abs(aa.REFAV1_KAPPA_LEVELS, aa.REFAV1_ACCEL_LEVELS,
                                  aa.REFAV1_CHANNELS)
    return aa.refav1_read(model, cfg, ld, sel, cands, n_perm=n_perm, seed=0, batch=4)


@pytest.fixture(scope="module")
def tiny_refav1_read():
    m, cfg = _tiny_refav1(speed_channel=False, seed=0)
    return _tiny_read(m, cfg)


def test_refav1_speed_channel_is_derived_by_the_model_never_by_this_tool():
    """The scale is the MODEL's constant, read through the MODEL's method."""
    from tanitad.refs.refa_v1 import SPEED_SCALE_MPS
    assert SPEED_SCALE_MPS == 30.0                      # refav1's trained scale
    from tanitad.models.flagship_v15 import SPEED_SCALE as V7_SPEED_SCALE
    assert V7_SPEED_SCALE == 10.0                       # the v7 arms' — NOT interchangeable
    assert SPEED_SCALE_MPS != V7_SPEED_SCALE

    m, cfg = _tiny_refav1(speed_channel=True)
    a = torch.zeros(2, 4, 2)
    a[:, :, 0] = 1.0                                    # 1 m/s^2 for every step
    v = torch.tensor([10.0, 20.0])
    out = m.augment_actions(a, v)
    assert out.shape[-1] == 3
    # v_k = v0 + sum_{j<k} a_j*op_dt, normalised by the MODEL's own constant
    exp = (v[:, None] + torch.arange(4).float()[None] * cfg.op_dt) / SPEED_SCALE_MPS
    assert torch.allclose(out[:, :, 2], exp, atol=1e-6)

    # a pre-widened action is REFUSED: a caller that built the channel itself
    # has read a speed from somewhere the model cannot audit (the leak shape).
    with pytest.raises(ValueError, match="never supplied"):
        m.augment_actions(torch.zeros(2, 4, 3), v)

    # and with the channel OFF (both banked step-1,000 refav1 configs) the
    # predictor input is (a, kappa) only — the constant is inert, not wrong.
    m_off, _ = _tiny_refav1(speed_channel=False)
    assert m_off.augment_actions(a, v).shape[-1] == 2


def test_refav1_tool_never_writes_a_speed_scale_of_its_own():
    """A grep-level guard: the refav1 read must not contain a hand-rolled
    normalisation. If someone adds one, this test names the leak."""
    src = TOOL.read_text(encoding="utf-8")
    body = src[src.index("def refav1_read("):src.index("def refav1_main(")]
    assert "augment_actions" in body
    for bad in ("/ 30", "/30.", "SPEED_SCALE_MPS =", "LEGACY_SPEED_SCALE"):
        assert bad not in body, "refav1_read builds its own speed channel: " + repr(bad)


def test_refav1_end_to_end_controls_read_their_known_values(tiny_refav1_read):
    r = tiny_refav1_read
    c = r["controls"]
    assert c["C0_identity_max_abs_diff"] == 0.0 and c["C0_passes"]
    assert c["zero_model_max_abs_displacement"] == 0.0 and c["zero_model_passes"]
    assert r["n_windows"] == 16
    assert r["n_candidates"] == 2 * (len(aa.REFAV1_KAPPA_LEVELS) + len(aa.REFAV1_ACCEL_LEVELS))
    for sp in ("tac", "pooled", "full_subsample"):
        assert r["spaces"][sp]["scene_spread"] > 1e-6          # C1: not 0/0
        assert r["spaces"][sp]["verdict"]["verdict"] != "VOID"
    assert r["verdict_space"] == "tac"
    assert set(r["space_agreement"]) == {"tac", "pooled", "full_subsample"}


def test_refav1_full_field_exact_stats_are_well_formed(tiny_refav1_read):
    fx = tiny_refav1_read["full_field_exact"]
    assert fx["state_dim"] == 8 * 32
    assert np.isfinite(fx["F_sep"]) and fx["F_sep"] > 0
    for ax, levels in (("kappa", aa.REFAV1_KAPPA_LEVELS), ("accel", aa.REFAV1_ACCEL_LEVELS)):
        nbl = fx["axes"][ax]["norm_by_level"]
        assert len(nbl) == len(levels)
        assert all(v >= 0 for v in nbl.values())


def test_refav1_action_blind_model_reads_exactly_zero():
    """Zeroing the operative's action encoder makes the predictor ignore its
    action EXACTLY. Every displacement must be 0.0 — the zero-model control
    reading its known value through the REAL predictor path."""
    m, cfg = _tiny_refav1(speed_channel=False, seed=3)
    m.operative.act[0].weight.data.zero_()
    m.operative.act[0].bias.data.zero_()
    r = _tiny_read(m, cfg, n=8, n_perm=10)
    assert r["controls"]["zero_model_max_abs_displacement"] == 0.0
    fx = r["full_field_exact"]
    assert max(fx["mean_norm"].values()) == 0.0, "an action-blind model moved"
    assert fx["ss_between"] == 0.0 and fx["ss_within"] == 0.0
    assert r["verdict"]["verdict"] == "BOTH-INSENSITIVE"


def test_refav1_planted_kappa_only_response_reads_REFUTED_end_to_end():
    """A model whose action encoder sees ONLY the kappa channel: the kappa axis
    must separate and the read must be LAT-INSENSITIVE-REFUTED — the instrument
    detects a lateral response when one is really there."""
    m, cfg = _tiny_refav1(speed_channel=False, seed=4)
    m.operative.act[0].weight.data[:, 0].zero_()          # blind to accel (ch 0)
    m.operative.act[0].weight.data[:, 1].mul_(50.0)       # loud on kappa (ch 1)
    r = _tiny_read(m, cfg, n=12, n_perm=30)
    pa = r["spaces"]["tac"]["grid"]["per_axis"]
    # ⚠️ AN EXACTLY DEAD AXIS READS ``nan``, NOT 0 — its F is 0/0 (no between
    # variance because every candidate mean is the zero vector, and no within
    # variance because every window's displacement is that same zero vector).
    # ``verdict_refav1`` guards with ``np.isfinite(...) and f >= bar``, so nan
    # is correctly NOT separated; a test that expected 0 would have failed on a
    # working instrument. Pinned so the nan cannot later be "fixed" into a 0
    # that would compare >= a bar of 0.
    assert not np.isfinite(pa["accel"]["F_over_null_p95"])   # dead axis: 0/0
    assert pa["kappa"]["F_over_null_p95"] > 5.0
    assert r["verdict"]["verdict"] == "LAT-INSENSITIVE-REFUTED"
    fx = r["full_field_exact"]
    assert max(fx["axes"]["accel"]["norm_by_level"].values()) == 0.0
    assert min(fx["axes"]["kappa"]["norm_by_level"].values()) > 0.0


def test_refav1_planted_accel_only_response_reads_CONFIRMED_end_to_end():
    """The mirror image, and the hypothesis' OWN signature: a model blind to
    kappa and loud on accel must read LAT-INSENSITIVE-CONFIRMED. Without this
    the instrument could only ever refute."""
    m, cfg = _tiny_refav1(speed_channel=False, seed=5)
    m.operative.act[0].weight.data[:, 1].zero_()          # blind to kappa (ch 1)
    m.operative.act[0].weight.data[:, 0].mul_(50.0)       # loud on accel (ch 0)
    r = _tiny_read(m, cfg, n=12, n_perm=30)
    pa = r["spaces"]["tac"]["grid"]["per_axis"]
    assert not np.isfinite(pa["kappa"]["F_over_null_p95"])   # dead axis: 0/0 (see above)
    assert pa["accel"]["F_over_null_p95"] > 5.0
    assert r["verdict"]["verdict"] == "LAT-INSENSITIVE-CONFIRMED"
    fx = r["full_field_exact"]
    assert max(fx["axes"]["kappa"]["norm_by_level"].values()) == 0.0
    assert min(fx["axes"]["accel"]["norm_by_level"].values()) > 0.0


# ---------------------------------------------------------------------------
# (8) --action-units: the units PASS-THROUGH (D-ACTDIV-UNITS-INVARIANCE, 2026-09-03)
#
# The banked refav1 read swept levels {0.02, 0.05, 0.1} on channel 1 and called them
# CURVATURES. `physicalai.signals_at` writes `steer = arctan(2.9 * curvature)` into that
# channel, so the swept numbers were STEER ANGLES. The pass-through lets the same grid be
# fed in either convention, ONE VARIABLE apart, so the "separation is invariant under a
# monotone reparametrisation" argument can be MEASURED rather than inherited.
# ---------------------------------------------------------------------------
def test_action_units_kappa_is_an_identity_pass_through():
    """The legacy path must be byte-identical: same objects, not a re-scaled copy."""
    cands = aa.candidate_grid_abs((0.02, 0.05, 0.1), (0.5, 1.5), aa.REFAV1_CHANNELS)
    out, prov = aa.apply_action_units(cands, "kappa")
    assert out is cands                                   # object-identical, not a copy
    assert prov["action_units"] == "kappa" and prov["wheelbase_m"] is None
    assert all("fed_value" not in c for c in out)         # nothing added to the legacy record


def test_action_units_steer_feeds_arctan_of_L_times_level():
    """steer: fed = arctan(2.9 * level); the NOMINAL level is kept so linearity is still
    measured against the swept action levels."""
    cands = aa.candidate_grid_abs((0.02, 0.05, 0.1), (0.5, 1.5), aa.REFAV1_CHANNELS)
    out, prov = aa.apply_action_units(cands, "steer")
    assert prov["action_units"] == "steer" and prov["wheelbase_m"] == pytest.approx(2.9)
    k = next(c for c in out if c["id"] == "kappa+0.1")
    assert k["level"] == pytest.approx(0.1)               # nominal curvature preserved
    assert k["abs_level"] == pytest.approx(0.1)
    assert k["fed_value"] == pytest.approx(0.28225742, abs=1e-7)
    assert k["a2"][1] == pytest.approx(0.28225742, abs=1e-7)
    assert k["a2"][0] == 0.0
    kn = next(c for c in out if c["id"] == "kappa-0.1")
    assert kn["fed_value"] == pytest.approx(-0.28225742, abs=1e-7)   # arctan is ODD


def test_action_units_steer_leaves_the_accel_axis_and_the_zero_anchor_untouched():
    cands = aa.candidate_grid_abs((0.02, 0.1), (0.5, 1.5), aa.REFAV1_CHANNELS)
    out, _ = aa.apply_action_units(cands, "steer")
    for cid in ("accel+1.5", "accel-0.5"):
        src = next(c for c in cands if c["id"] == cid)
        dst = next(c for c in out if c["id"] == cid)
        assert dst["a2"] == src["a2"]                     # longitudinal channel is NOT converted
    z = next(c for c in out if c["id"] == "zero")
    assert z["a2"] == [0.0, 0.0]                          # the anchor stays exactly zero


def test_action_units_steer_uses_the_repos_own_kinematic_inverse():
    """⛔ Not a hand-rolled atan: the conversion must go through kinematic.as_command, and
    it must invert exactly with kappa_of_steer at the ENCODING wheelbase."""
    from tanitad.models import kinematic as kin
    assert kin.STEER_WHEELBASE_M == 2.9
    cands = aa.candidate_grid_abs((0.05,), (0.5,), aa.REFAV1_CHANNELS)
    out, _ = aa.apply_action_units(cands, "steer")
    fed = next(c for c in out if c["id"] == "kappa+0.05")["fed_value"]
    ref = float(kin.as_command(torch.tensor([[0.0, 0.05]], dtype=torch.float64), "steer")[0, 1])
    assert fed == pytest.approx(ref, rel=0, abs=0.0)      # the same code path, to the bit
    back = float(kin.kappa_of_steer(torch.tensor(fed, dtype=torch.float64)))
    assert back == pytest.approx(0.05, abs=1e-12)


def test_action_units_steer_honours_an_explicit_wheelbase():
    cands = aa.candidate_grid_abs((0.1,), (0.5,), aa.REFAV1_CHANNELS)
    out, prov = aa.apply_action_units(cands, "steer", 3.085)
    assert prov["wheelbase_m"] == pytest.approx(3.085)
    assert "explicit" in prov["wheelbase_source"]
    assert next(c for c in out if c["id"] == "kappa+0.1")["fed_value"] == pytest.approx(
        float(np.arctan(3.085 * 0.1)), abs=1e-12)


def test_action_units_rejects_an_unknown_convention():
    cands = aa.candidate_grid_abs((0.1,), (0.5,), aa.REFAV1_CHANNELS)
    with pytest.raises(ValueError, match="action-units"):
        aa.apply_action_units(cands, "curvature_per_metre")


def test_action_units_steer_is_refused_on_the_v7_family():
    """⛔ STEER_WHEELBASE_M is OUR cache's ENCODING constant. ZOD / l2d / alpasim encode
    with different wheelbases, so the flag must not travel — refuse loudly, never ignore."""
    with pytest.raises(SystemExit, match="refav1-only"):
        aa.main(["--family", "v7", "--action-units", "steer", "--out", "x.json"])


def test_action_units_steer_changes_the_stimulus_by_the_arctan_gain_not_a_constant():
    """The discriminating arithmetic the SPEC commits: the per-level gain is NOT constant
    (2.897 / 2.880 / 2.823), so a response that is linear in the FED value cannot also be
    linear in the NOMINAL level. This is what makes the invariance argument testable."""
    cands = aa.candidate_grid_abs((0.02, 0.05, 0.1), (1.5,), aa.REFAV1_CHANNELS)
    out, _ = aa.apply_action_units(cands, "steer")
    fed = {c["abs_level"]: c["fed_value"] for c in out
           if c.get("axis_name") == "kappa" and c["level"] > 0}
    gains = {lv: fed[lv] / lv for lv in fed}
    assert gains[0.02] == pytest.approx(2.8968, abs=1e-3)
    assert gains[0.1] == pytest.approx(2.8226, abs=1e-3)
    assert gains[0.02] > gains[0.05] > gains[0.1]          # arctan compresses as |level| grows
    assert fed[0.1] / fed[0.02] == pytest.approx(4.8720, abs=1e-3)   # NOT 5.000
