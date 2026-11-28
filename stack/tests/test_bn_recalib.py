"""A7 (2026-09-19): BatchNorm recalibration for the ImageNet knockout.

⛔⛔ THE DEFECT THIS EXISTS FOR. `--trunk-frozen-bn` (required by the memory levers
that make A7 fit on an 8 GiB card) pins every backbone BatchNorm to EVAL, where it
normalises with its STORED running statistics. An ImageNet trunk stores ImageNet's.
A RANDOM-init trunk stores what a fresh BN stores -- MEASURED on resnet34: every
running_mean exactly 0.0 and every running_var exactly 1.0 -- so its eval-mode BN is
the IDENTITY. Unfixed, A7 compares "ImageNet weights + a normalisation" against
"random weights + NO normalisation": two variables, and the second is invisible.

The fix (`--trunk-bn-recalib N`) re-estimates both arms' statistics on the SAME
fixed windows, then keeps them frozen. Pre-registered in
`PREREG_REFCV6_DEVBOX_PREPARATION.md` (A7 AMENDMENT, 2026-09-19) before any A7 data.

⛔ Per the 2026-09-10/11 binding rule ("built, tested, and unreachable from its
caller"), the LAST section drives the REAL trainer (`T.train`) end-to-end, so the
flag is proven to RUN from the training loop, not merely to work in isolation. Every
check was also shown to go RED under mutation (`mutate_bn_recalib.py`, banked with
the A7 package).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.models.timm_trunk import bn_staleness, build_timm_trunk  # noqa: E402


def _mk(**kw):
    """A tiny CPU trunk -- resnet18, no download: this file is about the
    recalibration's WIRING and ARITHMETIC, which are backbone-agnostic."""
    kw.setdefault("pretrained", False)
    return build_timm_trunk(in_channels=9, image_hw=(64, 128),
                            model_name="resnet18.a1_in1k", **kw)


def _batches(n=3, b=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(b, 9, 64, 128, generator=g) for _ in range(n)]


# ============================================================ the confound
def test_THE_CONFOUND_random_init_frozen_bn_is_the_identity():
    """The reason the flag exists, pinned as a measurement. If timm ever changed a
    fresh BN's stored statistics this test says so, and the A7 amendment's premise
    must be re-read."""
    t = _mk(frozen_bn=True)
    bns = t.backbone_bns()
    assert len(bns) > 0
    for m in bns:
        assert torch.equal(m.running_mean, torch.zeros_like(m.running_mean))
        assert torch.equal(m.running_var, torch.ones_like(m.running_var))
    t.train()                                     # what the trainer does every step
    assert t.bn_training_count() == 0             # ...and it stays frozen at identity


# ============================================================ the mechanism
def test_recalibration_moves_EVERY_layer_off_the_identity_and_keeps_it_frozen():
    t = _mk(frozen_bn=True)
    moms = [m.momentum for m in t.backbone_bns()]
    rep = t.recalibrate_bn_(_batches())
    assert rep["n_batches"] == 3 and rep["n_images"] == 12
    assert rep["n_bn"] == len(t.backbone_bns())
    for m in t.backbone_bns():
        assert not torch.equal(m.running_var, torch.ones_like(m.running_var)), (
            "a BN layer kept the identity statistics -- the pass did not reach it")
    # restored, exactly: momentum, the freeze, and the frozen train() pin
    assert [m.momentum for m in t.backbone_bns()] == moms
    t.train()
    assert t.bn_training_count() == 0, "recalibration un-froze the trunk"


def test_ANALYTIC_the_first_bn_equals_the_exact_moments_of_its_input():
    """⭐ An INDEPENDENTLY derived target, not the producer re-run: on ONE batch the
    cumulative mean is that batch's statistics, so the first BN's running stats must
    equal the per-channel mean and UNBIASED variance of the tensor that BN received,
    captured by a hook."""
    t = _mk(frozen_bn=True)
    bn0 = t.backbone_bns()[0]
    seen = []
    h = bn0.register_forward_pre_hook(lambda mod, a: seen.append(a[0].detach().clone()))
    x = _batches(n=1, b=6)[0]
    t.recalibrate_bn_([x])
    h.remove()
    inp = torch.cat(seen, dim=0)                   # every image that BN saw
    flat = inp.transpose(0, 1).reshape(inp.shape[1], -1).double()
    torch.testing.assert_close(bn0.running_mean.double(), flat.mean(1),
                               rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(bn0.running_var.double(), flat.var(1, unbiased=True),
                               rtol=1e-4, atol=1e-6)


def test_chunking_is_BYPASSED_so_a_bn_batch_is_never_one_image():
    """⛔ With chunk_ckpt=1 a BN batch would be ONE image, and averaging per-image
    variances omits the between-image variance. A chunked and an unchunked trunk with
    identical weights must therefore recalibrate IDENTICALLY."""
    torch.manual_seed(3)
    a = _mk(frozen_bn=True, chunk_ckpt=1)
    torch.manual_seed(3)
    b = _mk(frozen_bn=True)
    xs = _batches()
    rep = a.recalibrate_bn_(xs)
    b.recalibrate_bn_(xs)
    assert rep["chunk_bypassed"] is True
    assert a.memory_levers["chunk_ckpt"] == 1, "the lever was not restored"
    torch.testing.assert_close(a.bn_stats_snapshot()["var"],
                               b.bn_stats_snapshot()["var"], rtol=1e-5, atol=1e-7)


def test_SAME_BREATH_CONTROL_per_image_batches_really_do_underestimate_var():
    """⚠️ Without this the bypass test above is vacuous: it would pass even if the
    per-image path gave the same answer.

    ⛔ CORRECTED WHILE WRITING IT. The first version fed iid uniform noise and failed:
    every noise image has the SAME statistics, so there is NO between-image variance
    for the per-image path to omit, and it "under-estimated" on ~half the channels by
    chance. Real frames differ in brightness and content. So the inputs here carry a
    per-image brightness offset, and the check is read at the FIRST BatchNorm, where
    it is analytic: the stem conv is linear, so a per-image offset becomes a
    per-image, per-channel shift -- a between-image variance that averaging per-image
    variances provably drops."""
    torch.manual_seed(3)
    full = _mk(frozen_bn=True)
    torch.manual_seed(3)
    single = _mk(frozen_bn=True)
    g = torch.Generator().manual_seed(11)
    xs = [torch.rand(4, 9, 64, 128, generator=g) * 0.3
          + torch.rand(4, 1, 1, 1, generator=g) * 0.7 for _ in range(3)]
    full.recalibrate_bn_(xs)
    single.recalibrate_bn_([x[i:i + 1] for x in xs for i in range(x.shape[0])])
    v_full = full.backbone_bns()[0].running_var
    v_one = single.backbone_bns()[0].running_var
    frac = float((v_one < v_full).double().mean())
    assert frac > 0.9, (
        "per-image recalibration under-estimated the first BN's variance on only "
        "%.2f of channels -- the bypass would not be protecting anything" % frac)


def test_identical_weights_and_windows_recalibrate_BYTE_identically():
    """The CPU form of A7.3(a): both ImageNet arms share their weights and windows,
    so their statistics must agree. On CPU this is exact; on CUDA it is to cuDNN
    non-determinism, which is why the amendment reads it as a ratio, not an epsilon."""
    torch.manual_seed(5)
    a = _mk(frozen_bn=True)
    torch.manual_seed(5)
    b = _mk(frozen_bn=True)
    xs = _batches()
    ra, rb = a.recalibrate_bn_(xs), b.recalibrate_bn_(xs)
    assert ra["stats_sha12"] == rb["stats_sha12"]
    assert torch.equal(a.bn_stats_snapshot()["var"], b.bn_stats_snapshot()["var"])


def test_measure_true_leaves_EVERY_buffer_untouched():
    """The staleness probe must not alter the trained trunk it measures."""
    t = _mk(frozen_bn=True)
    t.recalibrate_bn_(_batches(seed=1))
    before = [(m.running_mean.clone(), m.running_var.clone(),
               m.num_batches_tracked.clone()) for m in t.backbone_bns()]
    true = t.measure_true_bn_stats_(_batches(seed=2))
    for m, (rm, rv, nb) in zip(t.backbone_bns(), before):
        assert torch.equal(m.running_mean, rm)
        assert torch.equal(m.running_var, rv)
        assert torch.equal(m.num_batches_tracked, nb)
    assert not torch.equal(true["var"], t.bn_stats_snapshot()["var"])


def test_zero_batches_is_REFUSED_not_trained_on():
    t = _mk(frozen_bn=True)
    with pytest.raises(ValueError, match="ZERO batches"):
        t.recalibrate_bn_([])


def test_bn_staleness_reads_KNOWN_values():
    snap = {"mean": torch.tensor([0.0, 1.0]), "var": torch.tensor([1.0, 4.0])}
    s = bn_staleness(snap, snap)
    assert s["bn_staleness_var"] == 0.0 and s["bn_staleness_mean"] == 0.0
    fz = {"mean": torch.tensor([0.0, 1.0]), "var": torch.tensor([4.0, 16.0])}
    s4 = bn_staleness(fz, snap)
    assert s4["bn_staleness_var"] == pytest.approx(float(torch.tensor(4.0).log()))
    with pytest.raises(ValueError, match="different channels"):
        bn_staleness({"mean": torch.zeros(3), "var": torch.ones(3)}, snap)


# ============================================================ the trainer
@pytest.fixture(scope="module")
def T():
    import refc_v3_train as trainer
    return trainer


def test_trunk_input_is_EXACTLY_what_the_forward_hands_the_encoder(T):
    """⛔ `refc.py`'s hierarchy path calls `encoder.forward_features(...)` DIRECTLY,
    so a module forward pre-hook NEVER FIRES there (measured while writing this).
    The capture therefore wraps `forward_features` itself -- the one call the
    forward really makes -- and both branches are checked."""
    import dataclasses as dc
    from tanitad.refs import refc
    for hierarchy in (True, False):
        c = refc.refc_smoke_config()
        enc = dc.replace(c.encoder, trunk="timm", trunk_name="resnet18.a1_in1k",
                         trunk_pretrained=False, trunk_frozen_bn=True,
                         image_size=64, image_width=128, in_channels=9)
        m = refc.RefCModel(dc.replace(c, encoder=enc, hierarchy=hierarchy)).eval()
        seen, orig = [], m.encoder.forward_features

        def spy(x, *a, _o=orig, **k):
            seen.append(x.detach().clone())
            return _o(x, *a, **k)

        m.encoder.forward_features = spy
        frames = torch.rand(2, c.window, 9, 64, 128)
        with torch.no_grad():
            m(frames, None, torch.tensor([3.0, 4.0]), steps=1)
        assert len(seen) == 1, "the forward did not call forward_features once"
        assert torch.equal(seen[0], T.bn_recalib_trunk_input(m, frames)), (
            "recalibration would calibrate on a different input from the forward's "
            "(hierarchy=%s)" % hierarchy)


def test_rng_isolation_SAME_BREATH_CONTROL_and_exact_restore(T):
    """⛔ Creating a DataLoader iterator DRAWS from the global torch RNG even with
    shuffle=False. The control proves that happens; the isolation must undo it."""
    import random
    import numpy as np
    dl = torch.utils.data.DataLoader(list(range(4)), batch_size=2, shuffle=False)
    torch.manual_seed(0)
    s0 = torch.get_rng_state()
    iter(dl)
    assert not torch.equal(torch.get_rng_state(), s0), (
        "control: iterating a DataLoader no longer consumes RNG -- the isolation "
        "would then be protecting nothing, and this test must be re-read")
    torch.manual_seed(0)
    np.random.seed(1)
    random.seed(2)
    t0, n0, p0 = torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()
    with T._RngIsolated("cpu", 7):
        list(iter(dl))
        torch.rand(10)
        np.random.rand(10)
        random.random()
    assert torch.equal(torch.get_rng_state(), t0)
    assert (np.random.get_state()[1] == n0).all()
    assert random.getstate() == p0


def test_the_default_is_OFF(T):
    a = T.build_parser().parse_args(["--arm", "hier", "--out", "x"])
    assert a.trunk_bn_recalib == 0
    assert a.trunk_bn_recalib_seed == 0


_BASE = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "2",
         "--batch", "2", "--device", "cpu", "--log-every", "1", "--save-every", "1",
         "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k",
         "--trunk-in-channels", "3", "--no-trunk-pretrained", "--trunk-frozen-bn",
         "--trunk-chunk-ckpt", "1"]


@pytest.fixture(scope="module")
def e2e(T, tmp_path_factory):
    """ONE ON run and ONE OFF run through the REAL trainer, same seed. The global
    RNG state after each is recorded -- see the stream test below."""
    out = {}
    for tag, extra in (("off", []), ("on", ["--trunk-bn-recalib", "2"])):
        d = tmp_path_factory.mktemp("bnr_" + tag) / "run"
        T.train(T.build_parser().parse_args(_BASE + ["--out", str(d)] + extra))
        out[tag] = (d, torch.get_rng_state().clone())
    return out


def test_END_TO_END_the_real_trainer_recalibrates_stamps_and_holds_the_freeze(e2e):
    """⭐ REACHABILITY FROM THE ACTUAL CALLER: `T.train`, the training loop itself."""
    d, _ = e2e["on"]
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    st = cfg["trunk_bn_recalib"]
    assert isinstance(st, dict), "the ON run did not stamp a recalibration"
    assert st["changed"] is True
    assert st["var_median_before"] == 1.0, (
        "the random-init trunk did not START at the identity -- the confound "
        "this flag fixes was not present, so this run proves nothing")
    assert st["var_median_after"] != 1.0
    assert st["chunk_bypassed"] is True
    assert st["n_windows"] == 2 and st["seed"] == 0 and len(st["windows_sha12"]) == 12
    assert (d / "bn_recalib_stats.pt").exists()
    fin = json.loads((d / "bn_recalib.json").read_text(encoding="utf-8"))
    assert fin["freeze_held"] is True, "the frozen statistics moved during training"
    assert fin["staleness_error"] is None
    assert set(fin["staleness"]) == {"bn_staleness_var", "bn_staleness_mean",
                                     "n_channels"}
    assert st["final"]["freeze_held"] is True


def test_OFF_writes_null_and_touches_nothing(e2e):
    d, _ = e2e["off"]
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    assert "trunk_bn_recalib" in cfg and cfg["trunk_bn_recalib"] is None
    assert not (d / "bn_recalib_stats.pt").exists()
    assert not (d / "bn_recalib.json").exists()


def test_ON_does_not_shift_the_TRAINING_rng_stream(e2e):
    """⭐ The two runs share a seed and differ ONLY in the flag. Because every
    recalibration draw is isolated, the global RNG must end in the SAME state -- i.e.
    the ON arm saw the same data order and the same noise as the OFF arm, and differs
    from it in BN statistics alone."""
    assert torch.equal(e2e["on"][1], e2e["off"][1]), (
        "the recalibration consumed training RNG: the ON arm is not the OFF arm "
        "plus recalibration, it is a different draw")


def test_REFUSED_without_frozen_bn_BEFORE_anything_is_written(T, tmp_path):
    d = tmp_path / "run"
    # ⚠️ built explicitly: filtering the token "1" out of _BASE also strips the VALUES
    # of --log-every / --save-every (measured: argparse "expected one argument").
    i = _BASE.index("--trunk-frozen-bn")
    argv = _BASE[:i]                  # everything before the two levers
    with pytest.raises(SystemExit, match="without --trunk-frozen-bn"):
        T.train(T.build_parser().parse_args(
            argv + ["--out", str(d), "--trunk-bn-recalib", "2"]))
    assert not (d / "config.json").exists()


def test_REFUSED_on_the_in_repo_trunk(T, tmp_path):
    d = tmp_path / "run"
    argv = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "1",
            "--batch", "2", "--device", "cpu", "--out", str(d),
            "--trunk-frozen-bn", "--trunk-bn-recalib", "2"]
    with pytest.raises(SystemExit, match="needs --trunk timm"):
        T.train(T.build_parser().parse_args(argv))
