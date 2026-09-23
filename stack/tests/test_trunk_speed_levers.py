"""⭐ The backbone speed levers: bf16 autocast and channels_last -- the default path untouched.

MEASURED 2026-09-23 (torch.profiler, Thor, resnet101 at 416x1024, batch 8, chunk 8): the refcv6
step is GPU-bound, ~two thirds of GPU time is memory-bound backbone elementwise work (BN 23 %,
ReLU 17 %, residual adds 14 %) and 12 % is cuDNN's NCHW<->NHWC conversions. `bf16` halves the bytes
those ops move; `channels_last` removes the conversions. Both touch the BACKBONE ONLY: its outputs
return to float32 and the default layout, so fusion, decoders, trajectory integration and every
loss see exactly what they saw before.

What is pinned, each observed rather than inferred:
* with both levers OFF, `_backbone` returns exactly `net(x)` -- bit for bit;
* with `bf16`, a forward hook on the stem conv SEES a bfloat16 activation (the lever is live in
  the forward, not only in the config), the outputs are float32, and they are close to fp32;
* with `channels_last`, the conv weights ARE NHWC, outputs equal NCHW's within fp32 tolerance and
  come back in the default layout;
* `bf16` composes with chunked checkpointing + frozen BN (the refcv6 launch configuration);
* the trainer flags reach the BUILT trunk, are stamped, and refuse the `refc` trunk.
CPU only (CPU autocast supports bfloat16); random-init weights, no download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from tanitad.models import timm_trunk as TT  # noqa: E402


def _trunk(**kw):
    torch.manual_seed(0)
    return TT.build_timm_trunk(in_channels=3, image_hw=(64, 128),
                               model_name="resnet18.a1_in1k", pretrained=False, **kw)


def _x(b=2):
    g = torch.Generator().manual_seed(1)
    return torch.rand(b, 3, 64, 128, generator=g)


def _stem_dtype(trunk, x):
    seen = {}
    conv = next(m for m in trunk.net.modules() if isinstance(m, torch.nn.Conv2d))

    def _hook(m, i, o):            # ⛔ returns None: a returned value REPLACES the output
        seen.setdefault("dtype", o.dtype)

    h = conv.register_forward_hook(_hook)
    try:
        out = trunk._backbone(x)
    finally:
        h.remove()
    return seen["dtype"], out


def test_levers_OFF_is_exactly_the_pre_lever_path():
    t = _trunk()
    x = t.normalise(_x())
    with torch.no_grad():
        got, want = t._backbone(x), t.net(x)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype == torch.float32 and torch.equal(a, b)


def test_bf16_RUNS_the_backbone_in_bf16_and_hands_back_float32():
    t32, t16 = _trunk(), _trunk(bf16=True)
    assert t16.memory_levers["bf16"] is True
    x = t32.normalise(_x())
    with torch.no_grad():
        d32, o32 = _stem_dtype(t32, x)
        d16, o16 = _stem_dtype(t16, x)
    assert d32 == torch.float32 and d16 == torch.bfloat16
    for a, b in zip(o16, o32):
        assert a.dtype == torch.float32
        rel = float((a - b).norm() / b.norm())
        assert rel < 5e-2, rel


def test_channels_last_weights_ARE_nhwc_and_outputs_match():
    t, tcl = _trunk(), _trunk(channels_last=True)
    conv = next(m for m in tcl.net.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv.weight.is_contiguous(memory_format=torch.channels_last)
    x = t.normalise(_x())
    with torch.no_grad():
        a, b = t._backbone(x), tcl._backbone(x)
    for u, v in zip(a, b):
        assert v.is_contiguous()                       # default layout downstream
        assert torch.allclose(u, v, rtol=1e-4, atol=1e-5)


def test_bf16_composes_with_chunked_checkpointing_and_frozen_bn():
    t = _trunk(bf16=True, chunk_ckpt=1, frozen_bn=True)
    x = t.normalise(_x()).requires_grad_(False)
    out = t._backbone(x)
    loss = sum(o.pow(2).mean() for o in out)
    loss.backward()
    grads = [p.grad for p in t.net.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_the_trainer_flags_reach_the_BUILT_trunk_and_are_stamped():
    import dataclasses as dc

    import refc_v3_train as T
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(
        ["--arm", "hier", "--out", "x", "--trunk", "timm", "--trunk-name",
         "resnet18.a1_in1k", "--trunk-bf16", "--trunk-channels-last"])
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    enc = cfg.core.encoder
    assert (enc.trunk_bf16, enc.trunk_channels_last) == (True, True)
    built = refc.build_encoder(dc.replace(enc, trunk_pretrained=False))
    assert built.memory_levers["bf16"] is True
    assert built.memory_levers["channels_last"] is True
    st = T._seam_stamp(cfg, args)
    assert (st["trunk_bf16"], st["trunk_channels_last"]) == (True, True)


def test_the_levers_REFUSE_the_refc_trunk():
    import refc_v3_train as T
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "x", "--trunk", "refc",
                                        "--trunk-bf16"])
    with pytest.raises(SystemExit, match="need --trunk timm"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), args)


def test_cudnn_benchmark_is_OPT_IN_applied_and_READ_BACK_into_the_record():
    """Off by default (the default path picks the same algorithms as before); when asked for,
    train() sets the backend, and config.json records the setting AS READ BACK from torch,
    not the flag that asked for it. MEASURED 2026-09-23 on Thor it bought nothing
    (11.24 vs 10.12-11.56 s/step), which is why it stays opt-in."""
    import refc_v3_train as T
    P = T.build_parser()
    assert P.parse_args(["--arm", "hier", "--out", "x"]).cudnn_benchmark is False
    assert P.parse_args(["--arm", "hier", "--out", "x",
                         "--cudnn-benchmark"]).cudnn_benchmark is True
    src = (ROOT / "scripts" / "refc_v3_train.py").read_text(encoding="utf-8")
    assert "torch.backends.cudnn.benchmark = True" in src
    assert '"cudnn_benchmark": bool(torch.backends.cudnn.benchmark),' in src


def _randomise_bn(t, seed=3):
    """Non-trivial running stats and affine, identical on every trunk built from `seed`:
    with the default init (mean 0, var 1, gamma 1, beta 0) a WRONG fold could still pass."""
    g = torch.Generator().manual_seed(seed)
    for m in t.net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.5)
            m.running_var.copy_(torch.rand(n, generator=g) * 2.0 + 0.1)
            with torch.no_grad():
                m.weight.copy_(torch.randn(n, generator=g) * 0.5 + 1.0)
                m.bias.copy_(torch.randn(n, generator=g) * 0.3)
    return t


def test_fold_bn_is_the_SAME_FUNCTION_and_the_same_GRADIENTS():
    ref = _randomise_bn(_trunk(frozen_bn=True))
    fold = _randomise_bn(_trunk(frozen_bn=True, fold_bn=True))
    assert fold.memory_levers["bn_folded"] == 20            # resnet18: 1 + 8x2 + 3 shortcuts
    x = ref.normalise(_x())
    oa, ob = ref._backbone(x), fold._backbone(x)
    # ⛔ RELATIVE NORM, not elementwise: the fold REORDERS fp32 arithmetic (scale the weight,
    # then convolve), and through 20 layers with randomised BN statistics the reassociation
    # rounding reaches ~1e-3 absolute on activations of ~100 -- largest near ReLU's zero, where
    # an elementwise relative tolerance is meaningless. A wrong fold is off by O(1), not 1e-6.
    for a, b in zip(oa, ob):
        rel = float(((a - b).norm() / b.norm()).detach())
        assert rel < 1e-5, rel
    sum(o.pow(2).mean() for o in oa).backward()
    sum(o.pow(2).mean() for o in ob).backward()
    ga = dict(ref.net.named_parameters()); gb = dict(fold.net.named_parameters())
    for k in ga:
        if ga[k].grad is None:
            continue
        rel = float((ga[k].grad - gb[k].grad).norm() / ga[k].grad.norm().clamp_min(1e-30))
        assert rel < 1e-4, (k, rel)


def test_fold_bn_makes_the_BN_pass_the_identity():
    fold = _trunk(frozen_bn=True, fold_bn=True)
    bn = next(m for m in fold.net.modules() if isinstance(m, torch.nn.BatchNorm2d))
    seen = {}

    def _hook(m, i, o):
        seen["same"] = o is i[0]

    h = bn.register_forward_hook(_hook)
    try:
        fold._backbone(fold.normalise(_x()))
    finally:
        h.remove()
    assert seen["same"] is True


def test_fold_bn_keeps_every_state_dict_key_and_LOADS_into_an_unfolded_trunk():
    fold = _randomise_bn(_trunk(frozen_bn=True, fold_bn=True))
    plain = _trunk(frozen_bn=True)
    assert list(fold.state_dict()) == list(plain.state_dict())
    plain.load_state_dict(fold.state_dict())
    x = plain.normalise(_x())
    with torch.no_grad():
        for a, b in zip(plain._backbone(x), fold._backbone(x)):
            assert float((a - b).norm() / b.norm()) < 1e-5


def test_fold_bn_REFUSES_a_bn_that_trains():
    with pytest.raises(ValueError, match="without frozen_bn"):
        _trunk(fold_bn=True)
    import refc_v3_train as T
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "x", "--trunk", "timm",
                                        "--trunk-fold-bn"])
    with pytest.raises(SystemExit, match="without --trunk-frozen-bn"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), args)


def test_fold_bn_composes_with_bf16_and_chunked_checkpointing():
    t = _trunk(frozen_bn=True, fold_bn=True, bf16=True, chunk_ckpt=1)
    out = t._backbone(t.normalise(_x()))
    sum(o.pow(2).mean() for o in out).backward()
    grads = [p.grad for p in t.net.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_fold_bn_flag_reaches_the_BUILT_trunk_FOLDED_and_is_stamped():
    """Stamped is not applied: the built trunk must report folded pairs AND its first BN must
    pass its input through untouched -- a flag that reached config.json but not the modules
    would read `True` in the record and run the unfolded net."""
    import dataclasses as dc

    import refc_v3_train as T
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(
        ["--arm", "hier", "--out", "x", "--trunk", "timm", "--trunk-name",
         "resnet18.a1_in1k", "--trunk-frozen-bn", "--trunk-fold-bn"])
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    assert cfg.core.encoder.trunk_fold_bn is True
    built = refc.build_encoder(dc.replace(cfg.core.encoder, trunk_pretrained=False))
    assert built.memory_levers["bn_folded"] == 20
    bn = next(m for m in built.net.modules() if isinstance(m, torch.nn.BatchNorm2d))
    probe = torch.randn(1, bn.num_features, 4, 4)
    assert bn(probe) is probe
    assert T._seam_stamp(cfg, args)["trunk_fold_bn"] is True


def test_fold_bn_REFUSES_the_refc_trunk_and_NAMES_the_flag():
    import refc_v3_train as T
    from tanitad.refs import refc_v3 as v3
    args = T.build_parser().parse_args(["--arm", "hier", "--out", "x", "--trunk", "refc",
                                        "--trunk-frozen-bn", "--trunk-fold-bn"])
    with pytest.raises(SystemExit, match=r"--trunk-fold-bn .*need --trunk timm"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), args)


def test_train_RECORDS_the_levers_the_BUILT_trunk_applied(tmp_path):
    """⭐ The run record carries the FACT, read off the constructed trunk: a real `train()` on
    the synthetic rig with frozen + folded BN must write `trunk_memory_levers.bn_folded == 20`
    (resnet18) into config.json -- beside the seam stamp, which only says it was ASKED for.
    The unfolded control records no `bn_folded` at all."""
    import json

    import refc_v3_train as T
    base = ["--arm", "hier", "--smoke", "--synth-episodes", "2", "--steps", "1",
            "--batch", "2", "--device", "cpu", "--log-every", "1", "--save-every", "100",
            "--trunk", "timm", "--trunk-name", "resnet18.a1_in1k", "--no-trunk-pretrained",
            "--trunk-in-channels", "3", "--trunk-frozen-bn"]
    T.train(T.build_parser().parse_args(base + ["--out", str(tmp_path / "f"),
                                                "--trunk-fold-bn"]))
    T.train(T.build_parser().parse_args(base + ["--out", str(tmp_path / "u")]))
    f = json.loads((tmp_path / "f" / "config.json").read_text(encoding="utf-8"))
    u = json.loads((tmp_path / "u" / "config.json").read_text(encoding="utf-8"))
    assert f["seams"]["trunk_fold_bn"] is True and u["seams"]["trunk_fold_bn"] is False
    assert f["trunk_memory_levers"]["bn_folded"] == 20
    assert f["trunk_memory_levers"]["frozen_bn"] is True
    assert f["trunk_memory_levers"]["bn_pinned"] == 20
    assert "bn_folded" not in u["trunk_memory_levers"]
    assert u["trunk_memory_levers"]["bn_pinned"] == 20
