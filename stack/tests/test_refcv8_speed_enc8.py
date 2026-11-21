"""refcv8 (B) (MM 2026-10-04) -- `--r8-speed-enc8`: the FULL N2 ladder (8 steps + unknown) through a zero-init FiLM seam
beside the inherited 4-way channel, and the ceiling = the fed N2 value.

The 4-way one-hot {30, 50, 100, 120} merges 70 / 80 km/h into 100 on 20.8 % of train rows and so discards 44 % of N2's
information (MEASURED, raw/x3_leak.json: 0.0346 -> 0.0193). The seam is zero-init, so the model at step 0 is unchanged.

Pinned:
1. the encoding literals (every ladder value, above the top, unknown = zeros next to a 0, float32 and bf16 input);
2. STEP-0 IDENTITY: with the seam attached (same weights otherwise) the TRAINING forward and the eval forward with an
   unknown speed are bit-identical on traj / sel_idx / anchor_traj / sel_score_v3; with a KNOWN speed at eval the fan
   and the scores are bit-identical (the emitted plan moves only where the tighter N2 ceiling binds -- the ruling);
3. the ceiling is the fed N2 value (70 km/h stays 70, not 100), an unknown row none;
4. the EMITTED plan obeys the N2-value ceiling on the EXTENDED fan, and the same windows under the inherited 4-way
   ceiling emit a plan above it on at least one row (the test is not vacuous);
5. the seam reads the TREATED value: under the V-VSHUF roll it encodes another window's speed;
6. the seam takes gradient;
7. the argv (needs N2; dead flag), the stamp, G-DVB.
The leak reading (the fed channel stays <= 0.05) is in test_refcv8_speed_input.py's real-artifact test.
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import _refcv8_rig as R  # noqa: E402
import test_refcv8_speed_input as SI  # noqa: E402
import test_refcv8_v9_wiring as W9  # noqa: E402
from tanitad.refs import refcv8_conditioning as C  # noqa: E402
from tanitad.train import refcv8_train as RT  # noqa: E402

KEYS = ("traj", "sel_idx", "anchor_traj", "sel_score_v3")
#: the behaviour decoder's OWN outputs -- the seam acts here first; at step 0 nothing downstream reads it (every
#: planner seam is zero-init too), so the identity must be asserted HERE or a live seam would pass unseen
TAC = ("tacv6_lat_logits", "tacv6_lon_logits", "tacv6_goal_logits")


# =========================================================================== #
# 1. the encoding                                                              #
# =========================================================================== #
def test_the_encoding_literals():
    """float32 only, by design: the scene hook hands the encoder the FED float32 value (never the model dtype)."""
    kmh = [20, 30, 50, 70, 80, 100, 120, 130, 150]
    v = torch.tensor([k / 3.6 for k in kmh], dtype=torch.float32)
    e = C.speed_enc8(v, torch.ones(len(kmh)))
    assert e[:, :8].argmax(-1).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 7]
    assert e[:, :8].sum(-1).tolist() == [1.0] * 9 and e[:, 8].tolist() == [1.0] * 9
    u = C.speed_enc8(torch.tensor([70 / 3.6, 0.0]), torch.tensor([0.0, 0.0]))
    assert u.abs().sum() == 0                                                   # unknown: zeros next to a 0
    assert C.SPEED_ENC8_LADDER_KMH == (20, 30, 50, 70, 80, 100, 120, 130) and C.SPEED_ENC8_DIMS == 9


# =========================================================================== #
# 2. step-0 identity                                                           #
# =========================================================================== #
@pytest.fixture(scope="module")
def pair():
    pytest.importorskip("timm")
    T = R.trainer()
    kw = dict(speed_input="n2", n_alloc=8, alloc_emit=True)
    cfgA, mA = R.build(T, True, vmax=True, **kw)
    cfgB, mB = R.build(T, True, vmax=True, speed_enc8=True, **kw)
    new = R.copy_into(mA, mB)
    assert new and all(k.startswith("tac_decoder_v6.r8_speed_film.") for k in new), new[:4]
    return T, (cfgA, mA), (cfgB, mB)


def _run(cfg, m, bt, kmh, valid, train):
    b = bt["frames"].shape[0]
    return R.forward(cfg, m, bt, seed=5, train=train,
                     v_max_ms=torch.full((b,), kmh / 3.6), v_max_valid=torch.full((b,), float(valid)))


@pytest.mark.parametrize("train,valid", [(True, 1.0), (False, 0.0)])
def test_STEP0_the_seam_is_bit_identical_where_no_ceiling_acts(pair, train, valid):
    T, (cA, mA), (cB, mB) = pair
    bt = R.batch(cA, b=3, seed=2)
    a = _run(cA, mA, bt, 70, valid, train)
    b = _run(cB, mB, bt, 70, valid, train)
    for k in KEYS + TAC:
        assert torch.equal(a[k], b[k]), k


def test_DELIBERATE_REGRESSION_a_non_zero_seam_init_breaks_the_identity(pair):
    """The identity above is the ZERO INIT's doing: one seam bias moved by 1e-3 must make the same check go RED."""
    T, (cA, mA), (cB, mB) = pair
    bias = mB.tac_decoder_v6.r8_speed_film[0].proj.bias
    keep = bias.detach().clone()
    try:
        with torch.no_grad():
            bias.fill_(1e-3)
        bt = R.batch(cA, b=3, seed=2)
        a = _run(cA, mA, bt, 70, 1.0, True)
        b = _run(cB, mB, bt, 70, 1.0, True)
        assert not all(torch.equal(a[k], b[k]) for k in TAC)               # the seam's own reader goes RED
    finally:
        with torch.no_grad():
            bias.copy_(keep)


def test_STEP0_with_a_known_speed_the_fan_and_scores_are_bit_identical(pair):
    T, (cA, mA), (cB, mB) = pair
    bt = R.batch(cA, b=3, seed=3)
    a = _run(cA, mA, bt, 70, 1.0, False)
    b = _run(cB, mB, bt, 70, 1.0, False)
    for k in ("anchor_traj", "sel_score_v3") + TAC:
        assert torch.equal(a[k], b[k]), k


# =========================================================================== #
# 3 + 4. the ceiling is the fed N2 value, and the emitted plan obeys it          #
# =========================================================================== #
def test_the_ceiling_is_the_fed_N2_value(monkeypatch, pair):
    T, (cA, mA), (cB, mB) = pair
    lim = SI._limits(monkeypatch, cB, mB, [50, 70, 80, 130], [1.0, 1.0, 1.0, 0.0])
    assert lim[:3].tolist() == pytest.approx([50 / 3.6, 70 / 3.6, 80 / 3.6], abs=1e-5) and math.isinf(float(lim[3]))
    limA = SI._limits(monkeypatch, cA, mA, [50, 70, 80, 130], [1.0, 1.0, 1.0, 0.0])
    assert limA[:3].tolist() == pytest.approx([50 / 3.6, 100 / 3.6, 100 / 3.6], abs=1e-5)   # without enc8: the bins


V0 = [14.0, 15.0, 16.0, 17.0, 17.5, 18.0, 18.5, 19.0]


def test_the_EMITTED_plan_obeys_the_N2_value_on_the_extended_fan_and_the_check_is_not_vacuous(pair):
    T, (cA, mA), (cB, mB) = pair
    from tanitad.refs.refcv6_selection import planned_max_speed
    lim = 70 / 3.6
    bt = SI._speed_batch(cB, V0, seed=6)
    b = len(V0)
    kw = dict(v_max_ms=torch.full((b,), 70 / 3.6), v_max_valid=torch.ones(b))
    oB = R.forward(cB, mB, bt, seed=7, **kw)
    oA = R.forward(cA, mA, bt, seed=7, **kw)                                     # the inherited 4-way ceiling (100)
    dec = mB.core.decoder
    assert oB["anchor_traj"].shape[1] > int(oB["r8_n_base"])                    # the extended fan is in play
    vmax = planned_max_speed(oB["anchor_traj"], horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)
    ar = torch.arange(b)
    keep = vmax <= lim
    if "reach_keep" in oB:
        keep = keep & oB["reach_keep"]
    exempt = ~keep.any(dim=1)
    assert bool((vmax[ar, oB["sel_idx"]][~exempt] <= lim + 1e-5).all())
    vA = planned_max_speed(oA["anchor_traj"], horizons=dec.anchor_horizons, tick_s=dec.anchor_dt)[ar, oA["sel_idx"]]
    assert bool(((vA > lim + 1e-5) & ~exempt).any()), "no window's 4-way pick exceeds 70 km/h -- nothing was tested"


# =========================================================================== #
# 5. the seam reads the TREATED value                                          #
# =========================================================================== #
def test_under_the_VVSHUF_roll_the_seam_encodes_another_windows_speed(pair):
    T, (cA, mA), (cB, mB) = pair
    seen = {}
    real = type(mB.tac_decoder_v6).forward

    def spy(self, *a, **k):
        seen["cs8"] = k.get("cond_speed8")
        return real(self, *a, **k)
    bt = R.batch(cB, b=3, seed=4)
    b = 3
    traj = torch.cumsum(torch.ones(b, 8, 2) * torch.tensor([3.0, 0.2]), 1)
    v = torch.tensor([50 / 3.6, 70 / 3.6, 120 / 3.6])
    mB.train()
    mB._r8_rc_dropout, mB._r8_rc_noise, mB._r8_nav_args_dropout = 0.3, (2.0, 0.75), 0.5
    mB._r8_speed_unknown_p, mB._r8_roll_speed_train = 0.45, True
    mB.core.decoder.r8_gen = C.R8Generator(3)
    prep = W9._prep(mB, bt, b, traj, {"v_max_ms": v, "v_max_valid": torch.ones(b)})
    type(mB.tac_decoder_v6).forward = spy
    try:
        R.forward(cB, mB, bt, seed=5, train=True, **prep["fwd"])
    finally:
        type(mB.tac_decoder_v6).forward = real
        mB._r8_roll_speed_train = False
    want = C.speed_enc8(prep["fwd"]["v_max_ms"], prep["fwd"]["v_max_valid"])
    assert torch.equal(seen["cs8"], want)
    # the treated value IS another window's: every kept row carries the PREVIOUS row's speed, never its own
    fv, fk = prep["fwd"]["v_max_ms"], prep["fwd"]["v_max_valid"]
    for i in range(b):
        if float(fk[i]) == 1.0:
            assert float(fv[i]) == pytest.approx(float(v[(i - 1) % b])) and float(fv[i]) != pytest.approx(float(v[i]))


# =========================================================================== #
# 6. gradient                                                                  #
# =========================================================================== #
def test_the_seam_takes_gradient(pair):
    T, (cA, mA), (cB, mB) = pair
    bt = R.batch(cB, b=3, seed=8)
    mB.zero_grad()
    out = _run(cB, mB, bt, 80, 1.0, True)
    (out["tacv6_lat_logits"].float().pow(2).sum() + out["tacv6_lon_logits"].float().pow(2).sum()).backward()
    g = sum(float(p.grad.abs().sum()) for p in mB.tac_decoder_v6.r8_speed_film.parameters() if p.grad is not None)
    assert g > 0.0


# =========================================================================== #
# 7. argv, stamp, G-DVB                                                        #
# =========================================================================== #
ON = ["--arm", "hier", "--tac-decoder-v6", "--sampler", "ddim", "--out", "X", "--refcv8", "--w-r8-cons", "0.05",
      "--max-speed-input-v6", "--r8-v9-labels", "v9.npz"]


def _pin(argv):
    tr = R.trainer()
    args = tr.build_parser().parse_args(argv)
    cfg = types.SimpleNamespace(refcv8=C.R8Config())
    tr._pin_refcv8(cfg, args)
    return cfg.refcv8


def test_the_argv_rules_and_the_stamp():
    assert _pin(ON + ["--r8-speed-input", "n2", "--r8-speed-enc8"]).speed_enc8 is True
    for argv, needle in ((ON + ["--r8-speed-input", "n3", "--r8-speed-enc8"], "needs --r8-speed-input n2"),
                         (["--arm", "hier", "--out", "X", "--r8-speed-enc8"], "without --refcv8")):
        with pytest.raises(SystemExit) as e:
            _pin(argv)
        assert needle in str(e.value), str(e.value)
    txt = RT.speed_derivation("n2", True)
    RT.assert_r8_speed_stamp({"r8_speed_derivation": txt}, "n2", enc8=True)
    with pytest.raises(SystemExit, match="does not declare"):
        RT.assert_r8_speed_stamp({"r8_speed_derivation": RT.speed_derivation("n2", False)}, "n2", enc8=True)
    from tanitad.train import declared_vs_built as dvb
    assert dvb.REGISTRY["r8_speed_enc8"].kind == "built"
