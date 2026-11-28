"""⭐ `--trunk-compile`: the backbone through torch.compile -- same function, same keys.

MEASURED 2026-09-23 on Thor (`…/2026-09-23-refcv6-fixes/raw/compile_*_thor_2026-09-23.json`,
resnet101 at 416x1024, bf16 + NHWC, frozen + folded BN, chunk 8): backbone fwd+bwd 1.53x faster,
and CLOSER to strict fp32 than the eager bf16 path (1.75 % / 4.67 % vs 2.08 % / 5.74 %).

What is pinned here, on CPU with the `aot_eager` backend (the dev box has no Triton, so Inductor
itself is measured on Thor; `aot_eager` goes through the same Dynamo + AOTAutograd capture and
runs the captured graph eagerly):
* OFF is the eager network itself -- no compiled callable exists;
* ON gives the eager trunk's outputs AND parameter gradients;
* the compiled callable is what RUNS (a counting wrapper sees every backbone call) -- a flag
  that was stamped but never reached `_backbone` reads zero calls;
* the module tree and every state_dict key are unchanged, and weights load both ways;
* it composes with chunked checkpointing, bf16, NHWC, the fold and the dedup;
* the trainer flag reaches the BUILT trunk (backend recorded as "inductor"), is stamped, and
  is refused for the `refc` trunk by name.
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

K = 3


def _trunk(**kw):
    torch.manual_seed(0)
    return TT.build_timm_trunk(in_channels=3 * K, image_hw=(64, 128),
                               model_name="resnet18.a1_in1k", pretrained=False, **kw)


def _compiled(**kw):
    return _trunk(compile_backbone=True, compile_backend="aot_eager", **kw)


def _x(seed=1, n=4):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3 * K, 64, 128, generator=g)


def _rel(a, b):
    return float(((a - b).norm() / b.norm().clamp_min(1e-30)).detach())


def _count_calls(t):
    """Replace the compiled callable by a counting wrapper around it (outside the tree)."""
    seen = {"n": 0}
    inner = t._net_fn

    def _counting(x):
        seen["n"] += 1
        return inner(x)

    object.__setattr__(t, "_net_fn", _counting)
    return seen


def test_compile_OFF_is_the_eager_network_itself():
    t = _trunk()
    assert t._net_fn is None and "compile" not in t.memory_levers


def test_compiled_trunk_is_the_same_function_and_the_same_GRADIENTS():
    eager, comp = _trunk(frozen_bn=True), _compiled(frozen_bn=True)
    assert comp.memory_levers["compile"] == "aot_eager"
    x = _x()
    oa, ob = eager.forward_features(x), comp.forward_features(x)
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 1e-5
    sum(o.pow(2).mean() for o in oa).backward()
    sum(o.pow(2).mean() for o in ob).backward()
    ga, gb = dict(eager.named_parameters()), dict(comp.named_parameters())
    n = 0
    for k, p in ga.items():
        if p.grad is None:
            continue
        n += 1
        assert _rel(gb[k].grad, p.grad) < 1e-4, k
    assert n > 20


def test_the_COMPILED_callable_is_what_runs():
    comp = _compiled(frozen_bn=True, chunk_ckpt=2)
    seen = _count_calls(comp)
    with torch.no_grad():
        comp.forward_features(_x(n=2))          # 2 stacks x K=3 frames = 6 images, chunk 2
    assert seen["n"] == 3


def test_state_dict_keys_are_UNCHANGED_and_load_both_ways():
    eager, comp = _trunk(frozen_bn=True), _compiled(frozen_bn=True)
    assert list(comp.state_dict()) == list(eager.state_dict())
    assert not any("_orig_mod" in k for k in comp.state_dict())
    eager.load_state_dict(comp.state_dict())
    comp.load_state_dict(eager.state_dict())


def test_composes_with_chunk_bf16_nhwc_fold_and_dedup():
    kw = dict(frozen_bn=True, fold_bn=True, bf16=True, channels_last=True, chunk_ckpt=4,
              dedup_frames=True)
    eager, comp = _trunk(**kw), _compiled(**kw)
    g = torch.Generator().manual_seed(2)
    raw = torch.rand(2, 4 + K - 1, 3, 64, 128, generator=g)
    x = torch.stack([torch.cat([raw[:, i + j] for j in range(K)], dim=1)
                     for i in range(4)], dim=1).reshape(8, 3 * K, 64, 128)
    oa, ob = eager.forward_features(x), comp.forward_features(x)
    for a, b in zip(ob, oa):
        assert _rel(a, b) < 2e-2                  # both bf16
    assert comp.last_dedup == (8 * K, 2 * (4 + K - 1))
    sum(o.pow(2).mean() for o in ob).backward()
    grads = [p.grad for p in comp.net.parameters() if p.requires_grad]
    assert all(gr is not None and bool(torch.isfinite(gr).all()) for gr in grads)


def test_trainer_flag_reaches_the_BUILT_trunk_is_stamped_and_refuses_refc():
    import dataclasses as dc

    import refc_v3_train as T
    from tanitad.refs import refc
    from tanitad.refs import refc_v3 as v3
    P = T.build_parser()
    args = P.parse_args(["--arm", "hier", "--out", "x", "--trunk", "timm", "--trunk-name",
                         "resnet18.a1_in1k", "--trunk-compile"])
    cfg = v3.RefCV3Config(hier=True)
    T._pin_trainer_cfg(cfg, args)
    assert cfg.core.encoder.trunk_compile is True
    built = refc.build_encoder(dc.replace(cfg.core.encoder, trunk_pretrained=False))
    assert built.memory_levers["compile"] == "inductor"   # the run's backend; compiles lazily
    assert built._net_fn is not None
    assert T._seam_stamp(cfg, args)["trunk_compile"] is True
    bad = P.parse_args(["--arm", "hier", "--out", "x", "--trunk", "refc", "--trunk-compile"])
    with pytest.raises(SystemExit, match=r"--trunk-compile need --trunk timm"):
        T._pin_trainer_cfg(v3.RefCV3Config(hier=True), bad)


def test_compile_turns_off_DONATED_BUFFERS_because_the_trainer_backprops_twice():
    """⛔ MEASURED 2026-09-23 on Thor: the first compiled smoke (S11) died at step 1 --
    "compiled with non-empty donated buffers which requires create_graph=False and
    retain_graph=False" -- because the conflict detector takes per-term gradients with
    retain_graph=True before the step's own backward. The CPU backends here do not donate,
    so this pins the SWITCH (and runs the trainer's pattern); the Thor smoke proves the fix."""
    import torch._functorch.config as fc
    old = fc.donated_buffer
    try:
        fc.donated_buffer = True
        t = _compiled(frozen_bn=True)
        assert fc.donated_buffer is False
        assert t.memory_levers["compile_donated_buffer"] is False
        out = t.forward_features(_x(n=2))
        loss = sum(o.pow(2).mean() for o in out)
        torch.autograd.grad(loss, [p for p in t.net.parameters() if p.requires_grad],
                            retain_graph=True)
        loss.backward()
    finally:
        fc.donated_buffer = old
