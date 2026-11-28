"""PREREG_V8F_LORA: LoRA on the frozen DINOv3 ViT-L/16 inside v8F-mini -- the instruments' tests. CPU only.

Run: CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<repo>/stack;<repo>/taniteval pytest -q stack/tests/test_v8f_lora.py

THE CLASS THIS FILE EXISTS FOR (CLAUDE.md, "built, tested, and unreachable from its caller"): a LoRA that is importable, rollable and unit-tested
can still be a flag that reaches no loss. Pinned here, hop by hop (`file:line` below are as of the 2026-10-25 build; they rot, the function names do not):

  argv --encoder lora --lora-r/-alpha/-targets/-lr/--accum/--enc-ckpt  (v8f_mini_train.py build_parser)
    -> make_encoder (v8f_lora_encoder.DinoV3LoRAEncoder.from_pretrained) -> inject_lora: LoRALinear on q_proj / v_proj of every block
    -> declared_vs_built / bad_encoder (the exact trainable count; refuses every lie below)
    -> V8FMiniFrameWindows -> collate_frames: the frames behind the cache
    -> lora_step -> cached_grad_step: encode (LoRA in the forward) -> leaf -> model.loss -> total.backward() -> leaf.grad -> backward through the encoder
    -> a non-zero gradient on the LoRA B matrices at step 1 and on A after the first optimiser step, EXACTLY 0.0 on the frozen DINOv3 weights.

  * WITH B = 0 THE GRADIENT ON EVERY A MATRIX IS EXACTLY ZERO AT STEP 1, BY CONSTRUCTION (dL/dA = B^T ...). A test that asked for non-zero A grads at
    step 1 would be red on a correct implementation; the contract is: step 1 -> B != 0 and A == 0.0; after one optimiser step -> A != 0.
  * the IDENTITY property: B = 0 -> the LoRA encoder is the frozen encoder BIT FOR BIT, and the frozen encoder is the CACHE BUILDER's `encode_episode` bit for
    bit (tests below run the builder's own function against ours on the same v2ep).
  * `--encoder cache` is today's trainer, bit for bit.
  * the gradient-cache accumulation equals the full-batch gradient; plain micro-batch accumulation does not (the control that proves the test can fail).

FILE:LINE OF EVERY HOP (stack/scripts/v8f_mini_train.py unless named; lines as of 2026-10-25):
  1 argv -> namespace           :157 --encoder  :159 --lora-r  :160 --lora-alpha  :161 --lora-targets  :162 --lora-lr  :163 --frames-root  :164 --accum  :165 --enc-ckpt
  2 namespace -> built encoder  :654 make_encoder(a, dev) (:395) -> build_encoder_raw (:387) -> DinoV3LoRAEncoder.from_pretrained(r=a.lora_r, alpha=a.lora_alpha, ...);
                                v8f_lora_encoder.py:366 inject_lora (:263) wraps q_proj / v_proj of every block
  3 encoder -> optimiser        :660 the "lora" AdamW group = enc.lora_parameters(), lr = lora_lr_of(a)
  4 launch gate (G-DVB, G-LIVE) FLAG_CHECKS :81; bad_encoder :194 (called from declared_vs_built :252): every flag == the BUILT value incl. the exact trainable count;
                                --encoder cache + any LoRA flag off its default is refused (LORA_PARSER_DEFAULTS :84); live_check_lora :449 (called :680); identity_probe :436 (:681)
  5 step                        :762 lora_step (:413) -> v8f_lora_encoder.cached_grad_step (:558; called from v8f_mini_train.py:428)
  6 LoRA in the forward         v8f_lora_encoder.py:203 y + (F.linear(F.linear(h, A), B) * scaling); field :455, pooled :460, fp8_ste :464
  7 loss -> backward            run_loss (inside lora_step) = model.loss on the FULL effective batch with feats = leaf[frame_map]; cached_grad_step :582 total.backward()
                                fills the heads' grads and leaf.grad; :592 torch.autograd.backward(p, g[chunk]) pulls it into A and B
  8 clip + step                 :431-432 two clips (heads on B's rule, LoRA on its own norm); opt.step() in the caller

DESIGN CHOICES THE PREREG DID NOT SPECIFY (defaults chosen here; the Master Mind confirms or overrules before the arm is registered):
  clipping: heads on B's rule alone, LoRA on its own norm (a joint norm changes the heads' update whenever the LoRA gradient dominates) | weight decay: LoRA 2-D matrices
  follow --wd | A: kaiming-uniform(a=sqrt 5) from a private generator seeded by --seed (the global RNG, hence the heads' init, is never moved) | fp8 straight-through KEPT on
  the on-the-fly path (the cache IS quantised) | LoRA branch fp32 on the bf16 base, dropout 0 | encoder always eval (HF DINOv3 RoPE rescales coordinates at random in train
  mode: its own test) | --accum k = encoder micro-batches of batch/k windows, the heads' loss still once on the full batch (plain accumulation is inexact: control test) |
  G-LIVE probes 16 windows through the real encoder | the identity probe vs the cache is RECORDED, not gated (a GPU-built cache differs from CPU by bf16 noise) | LoRA lr
  defaults to --lr, the warm-up/decay multiplier is shared | decode+normalise are MIRRORED from the cache builder (pool_field is IMPORTED), pinned by a runtime bit-for-bit test
  against the builder's own encode_episode + the sha256 of its source + the constants, instead of editing the builder (which would change every future cache's provenance hash).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "stack", REPO / "taniteval", REPO / "stack" / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_lora_encoder as E  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

torch.set_num_threads(2)
PKG = REPO / "TanitAD Research Lab" / "Architecture & Inference" / "Research" / "2026-10-17-v8f-mini"
BUILDER = E.builder_paths()[0]
HAVE_BUILDER = BUILDER.is_file() and E.builder_paths()[1].is_file()
needs_builder = pytest.mark.skipif(not HAVE_BUILDER, reason="the cache builder (pull_encode_pooled.py + dinov3_fp8_encode_ship.py) is not in this checkout")
try:
    REAL_SAFETENSORS = E.model_safetensors_path()
except Exception:                                         # noqa: BLE001
    REAL_SAFETENSORS = None
needs_real = pytest.mark.skipif(REAL_SAFETENSORS is None or not HAVE_BUILDER, reason="DINOv3-L weights are not in the local HF cache")
try:
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
except Exception:                                         # noqa: BLE001
    pytest.skip("transformers has no DINOv3", allow_module_level=True)

TINY_HW = (64, 160)                                       # 4 x 10 patches: the test-scale geometry (40 tokens per frame)


# ------------------------------------------------------------------------------------------------ helpers
def tiny_backbone(hidden=1024, layers=2, heads=16, inter=64, dtype=torch.float32, seed=0):
    cfg = DINOv3ViTConfig(hidden_size=hidden, num_attention_heads=heads, intermediate_size=inter, num_hidden_layers=layers, patch_size=16,
                          num_register_tokens=4, image_size=224)
    g = torch.get_rng_state()
    torch.manual_seed(seed)
    m = DINOv3ViTModel(cfg).eval().to(dtype)
    torch.set_rng_state(g)
    return m


def smooth_frames(n, hw, seed=0, step0=0):
    """Low-entropy uint8 frames [n, 3, H, W] that differ per (seed, step): PNG-small, content-distinct."""
    H, W = hw
    y = torch.arange(H).view(H, 1).float()
    x = torch.arange(W).view(1, W).float()
    out = []
    for i in range(n):
        k = step0 + i
        ch = [((x * (1 + c) + y * (2 + c) + 11 * k + 37 * seed + 90 * c) % 256) for c in range(3)]
        out.append(torch.stack(ch).to(torch.uint8))
    return torch.stack(out)


def v2ep_from_frames(frames_u8, clip_id="c", only_even=True):
    """A v2ep-shaped dict whose raw frame 2g is frames_u8[g] (odd raw frames have length 0, as in `resize416_to_256`)."""
    import torchvision
    n = frames_u8.shape[0]
    raw = 2 * n - 1
    pngs, lens = [], []
    for j in range(raw):
        if j % 2 == 0:
            p = torchvision.io.encode_png(frames_u8[j // 2], compression_level=1)
            pngs.append(p)
            lens.append(int(p.numel()))
        else:
            lens.append(0)
    g = torch.Generator().manual_seed(len(lens))
    poses = torch.cat([torch.cumsum(torch.rand(raw, 2, generator=g), 0), torch.rand(raw, 1, generator=g) * 0.1, 5 + torch.rand(raw, 1, generator=g)], 1)
    return {"clip_id": clip_id, "codec": "png", "jpeg_buf": torch.cat(pngs), "jpeg_len": torch.tensor(lens, dtype=torch.int64),
            "image_h": int(frames_u8.shape[2]), "image_w": int(frames_u8.shape[3]), "poses": poses, "actions": torch.rand(raw, 2, generator=g) * 0.05}


def make_encoder(hidden=1024, layers=2, dtype=torch.float32, r=4, alpha=4.0, targets=("q", "v"), seed=0, quantize=True, grid="4x10", bb_seed=0):
    bb = tiny_backbone(hidden, layers, 16 if hidden % 16 == 0 and hidden >= 64 else 2, 64, dtype, bb_seed)
    m, s = E.mean_std(dtype)
    return E.DinoV3LoRAEncoder(bb, m, s, r=r, alpha=alpha, targets=targets, seed=seed, grid=grid, quantize=quantize)


@pytest.fixture(autouse=True)
def tiny_geometry(monkeypatch, request):
    """Test-scale geometry: a 64x160 frame is a 4x10 patch grid, so `pool4x10` has 1x1 cells and 40 tokens -- 16x cheaper than 256x640. The builder's module
    constants are patched, never its code. The tests named in REAL_GEOMETRY keep 256x640."""
    if request.node.originalname in REAL_GEOMETRY or not HAVE_BUILDER:
        yield
        return
    pep, _ = E.load_builder()
    monkeypatch.setattr(pep, "GH", 4)
    monkeypatch.setattr(pep, "GW", 10)
    monkeypatch.setattr(D, "FRAME_HW", TINY_HW)
    yield


REAL_GEOMETRY = {"test_the_lora_encoder_at_zero_is_the_cache_builders_encode_episode_bit_for_bit",
                 "test_real_dinov3_l_trainable_count_sha_and_one_real_frame_identity"}


# ------------------------------------------------------------------------------------------------ 1. the module: structure, count, identity
def test_trainable_count_is_the_preregistered_number_on_the_real_architecture():
    """32 x (1024 + 1024) x 2 x 24 = 3,145,728 -- built on the REAL ViT-L/16 architecture (meta device: no weights, no memory)."""
    cfg = DINOv3ViTConfig(hidden_size=1024, num_attention_heads=16, intermediate_size=4096, num_hidden_layers=24, patch_size=16, num_register_tokens=4,
                          query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True, layerscale_value=1.0, use_gated_mlp=False)
    with torch.device("meta"):
        bb = DINOv3ViTModel(cfg)
    n = E.inject_lora(bb, 32, 32.0, ("q", "v"))
    assert n == 24
    train = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    assert train == 32 * (1024 + 1024) * 2 * 24 == 3_145_728
    assert [t for t, p in bb.named_parameters() if p.requires_grad and "lora_" not in t] == []
    mods = E.lora_modules(bb)
    assert len(mods) == 24 * 2 and all(m.r == 32 and m.scaling == 1.0 for _, m in mods)
    assert {n.split(".")[-1] for n, _ in mods} == {"q_proj", "v_proj"}                    # q and v, not k, not o
    # r / targets change the count exactly as the formula says
    with torch.device("meta"):
        bb2 = DINOv3ViTModel(cfg)
    E.inject_lora(bb2, 8, 8.0, ("q", "k", "v", "o"))
    assert sum(p.numel() for p in bb2.parameters() if p.requires_grad) == 8 * 2048 * 4 * 24


def test_fused_qkv_adapts_only_the_q_and_v_slices():
    """A backbone with ONE fused qkv linear: LoRA touches rows [0:E] (q) and [2E:3E] (v); the k rows are untouched by construction."""
    class Attn(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.qkv = nn.Linear(e, 3 * e)
            self.proj = nn.Linear(e, e)

        def forward(self, x):
            q, k, v = self.qkv(x).chunk(3, -1)
            return self.proj(torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, -1) @ v)

    class Net(nn.Module):
        def __init__(self, e=16, L=3):
            super().__init__()
            self.blocks = nn.ModuleList([Attn(e) for _ in range(L)])

        def forward(self, x):
            for b in self.blocks:
                x = x + b(x)
            return x
    torch.manual_seed(0)
    net = Net()
    x = torch.randn(2, 5, 16)
    ref = net(x).detach()
    wrapped = copy.deepcopy(net)
    n = E.inject_lora(wrapped, 4, 4.0, ("q", "v"))
    assert n == 3
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) == 3 * 4 * (16 + 16) * 2
    assert torch.equal(wrapped(x), ref)                                                      # B = 0 -> bitwise the frozen net
    for b in wrapped.blocks:                                                                 # a non-zero delta moves q and v rows only
        for sl in b.qkv.slices.values():
            nn.init.normal_(sl.lora_B, std=0.1)
        d = b.qkv.delta_weight()
        assert float(d[:16].abs().sum()) > 0 and float(d[32:].abs().sum()) > 0 and float(d[16:32].abs().sum()) == 0.0
    merged = copy.deepcopy(wrapped)
    E._fold(merged)
    E._unwrap(merged)
    assert torch.allclose(merged(x), wrapped(x), atol=1e-5) and not torch.equal(wrapped(x), ref)
    assert sum(p.numel() for p in merged.parameters()) == sum(p.numel() for p in net.parameters())
    with pytest.raises(ValueError, match="targets"):
        E.inject_lora(copy.deepcopy(net), 4, 4.0, ("z",))
    with pytest.raises(ValueError, match="no attention block"):
        E.inject_lora(nn.Sequential(nn.Linear(4, 4)), 4, 4.0, ("q", "v"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_b_zero_forward_is_the_frozen_forward_bit_for_bit(dtype):
    enc = make_encoder(64, 2, dtype=dtype)
    plain = E._PlainEncoder(copy.deepcopy(enc.backbone), enc.mean.clone(), enc.std.clone())
    plain.backbone = copy.deepcopy(enc.merge())                                              # B = 0: merging adds exactly 0
    fr = smooth_frames(3, TINY_HW, seed=1)
    assert torch.equal(enc.field(fr), plain.field(fr))
    assert torch.equal(enc.pooled(fr), plain.pooled(fr))
    assert enc.built_levers()["lora_B_zero_at_build"] is True


@needs_builder
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_lora_encoder_at_zero_is_the_cache_builders_encode_episode_bit_for_bit(dtype):
    """The mirror of the builder's decode + normalise, proved against the builder's OWN `encode_episode` on the same v2ep (256x640, a tiny ViT with the real
    d = 1024 so the builder's own asserts hold): the fp32 patch field is bit-identical, and so is the pooled + fp8 payload via the builder's `payload_from_field`."""
    pep, ship = E.load_builder()
    enc = make_encoder(1024, 2, dtype=dtype)
    frames = smooth_frames(3, (256, 640), seed=2)
    o = v2ep_from_frames(frames)
    ref = ship.encode_episode(o, copy.deepcopy(enc.merge()), enc.mean, enc.std, 5, "cpu")           # the builder's own function, on the B = 0 weights
    mine = enc.field(E.decode_cache_frames(o, [0, 1, 2]))
    assert ref.shape == (3, 640, 1024) and torch.equal(mine, ref)
    o["poses"] = torch.zeros(5, 4)
    o["actions"] = torch.zeros(5, 2)
    pay = pep.payload_from_field(ref, o, ["4x10"], torch.float8_e4m3fn)
    assert torch.equal(enc.pooled(E.decode_cache_frames(o, [0, 1, 2]), "4x10").to(torch.float8_e4m3fn).view(torch.uint8), pay["pool4x10"].view(torch.uint8))


@needs_builder
def test_the_pool_is_the_builders_own_function_and_the_source_is_pinned():
    pep, ship = E.load_builder()
    assert E.builder_pool_field() is pep.pool_field                                          # imported, not re-implemented
    assert E.assert_builder_unchanged() == E.BUILDER_ENCODE_EPISODE_SHA256                   # the mirror's source pin
    assert E.GRIDS == pep.GRIDS and (E.GH, E.GW) == (16, 40) and E.N_SPECIAL == 5
    assert (E.IMAGE_MEAN, E.IMAGE_STD, E.PIXEL_SCALE) == ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 255.0)


def test_fp8_ste_is_the_caches_quantiser_and_passes_the_gradient_through():
    x = (torch.randn(4, 40, 64) * 3).requires_grad_(True)
    y = E.fp8_ste(x)
    assert torch.equal(y, x.detach().to(torch.float8_e4m3fn).float())                        # EXACTLY pool.to(fp8).float()
    assert torch.equal(y.to(torch.float16).float(), y)                                        # fp8 -> fp16 -> fp32 (the cache path) is lossless
    g = torch.randn_like(y)
    y.backward(g)
    assert torch.equal(x.grad, g)                                                             # straight-through
    assert not torch.equal(y.detach(), x.detach())                                            # and it really quantises


def test_building_the_encoder_never_touches_the_global_rng_and_a_is_seeded_privately():
    torch.manual_seed(123)
    st = torch.get_rng_state()
    e1 = make_encoder(64, 2, seed=5)
    assert torch.equal(torch.get_rng_state(), st)
    e2 = make_encoder(64, 2, seed=5)
    e3 = make_encoder(64, 2, seed=6)
    a1 = [p for n, p in e1.named_parameters() if "lora_A" in n]
    a2 = [p for n, p in e2.named_parameters() if "lora_A" in n]
    a3 = [p for n, p in e3.named_parameters() if "lora_A" in n]
    assert all(torch.equal(x, y) for x, y in zip(a1, a2)) and not all(torch.equal(x, y) for x, y in zip(a1, a3))
    assert all(float(p.abs().sum()) == 0.0 for n, p in e1.named_parameters() if "lora_B" in n)


# ------------------------------------------------------------------------------------------------ 2. reachability of the gradient
def _loss_through(enc, frames, w=None):
    f = enc.pooled(frames)
    w = torch.linspace(0.5, 1.5, f.shape[-1]) if w is None else w
    return (f * w).pow(2).mean()


def _grad_sums(enc):
    A = sum(float(p.grad.abs().sum()) for n, p in enc.named_parameters() if "lora_A" in n and p.grad is not None)
    B = sum(float(p.grad.abs().sum()) for n, p in enc.named_parameters() if "lora_B" in n and p.grad is not None)
    Z = sum(float(p.grad.abs().sum()) for n, p in enc.named_parameters() if "lora_" not in n and p.grad is not None)
    return A, B, Z


def test_step_one_has_nonzero_B_grads_and_exactly_zero_A_grads_then_A_wakes_up_and_the_frozen_control_reads_zero():
    enc = make_encoder(64, 2, r=4)
    fr = smooth_frames(4, TINY_HW, seed=3)
    opt = torch.optim.AdamW(enc.lora_parameters(), lr=1e-2)
    _loss_through(enc, fr).backward()
    A, B, Z = _grad_sums(enc)
    assert B > 0.0 and A == 0.0 and Z == 0.0                                                  # B != 0, A EXACTLY 0.0, frozen DINOv3 EXACTLY 0.0
    for n, p in enc.named_parameters():
        if "lora_B" in n:
            assert float(p.grad.abs().sum()) > 0.0, n                                         # EVERY B group (all blocks, q and v)
    opt.step()
    opt.zero_grad(set_to_none=True)
    _loss_through(enc, fr).backward()
    A2, B2, Z2 = _grad_sums(enc)
    assert A2 > 0.0 and B2 > 0.0 and Z2 == 0.0
    for n, p in enc.named_parameters():
        if "lora_A" in n:
            assert float(p.grad.abs().sum()) > 0.0, n                                         # EVERY A group, once B has moved
    assert all(p.grad is None for n, p in enc.named_parameters() if "lora_" not in n)         # frozen weights: no gradient AT ALL
    assert sum(float(p.abs().sum()) for n, p in enc.named_parameters() if "lora_B" in n) > 0.0  # and B moved


def test_a_detached_encoder_output_turns_the_gradient_check_red():
    """The instrument self-test: the same check, on an encoder whose output is detached, must FAIL (this is the in-process twin of the mutation harness)."""
    class Detached(E.DinoV3LoRAEncoder):
        def pooled(self, frames_u8, grid=None, quantize=None):
            return super().pooled(frames_u8, grid, quantize).detach()
    good = make_encoder(64, 2)
    bad = Detached(tiny_backbone(64, 2, 2, 64), good.mean.clone(), good.std.clone(), r=4, alpha=4.0)
    fr = smooth_frames(2, TINY_HW)
    _loss_through(good, fr).backward()
    assert _grad_sums(good)[1] > 0.0
    with pytest.raises(RuntimeError, match="does not require grad"):
        _loss_through(bad, fr).backward()
    # through the gradient cache: the detached encoder cannot pull the heads' gradient back
    leaf_fail = False
    try:
        E.cached_grad_step(bad, fr, torch.arange(2).view(1, 2), lambda f: {"terms": {"total": f.pow(2).mean()}}, chunk=1, device="cpu")
    except RuntimeError:
        leaf_fail = True
    assert leaf_fail


def test_dinov3_train_mode_randomises_rope_and_our_encoder_is_immune_to_it():
    """The trap: HF DINOv3 rescales its RoPE coordinates at random whenever the module is in train mode (config.pos_embed_rescale = 2.0). The cache was built
    under eval(); a trainer that calls train() on the encoder would silently feed the heads different features at every call."""
    fr = smooth_frames(2, TINY_HW, seed=8)
    bare = tiny_backbone(64, 2, 2, 64)
    x = E.normalise(fr, *E.mean_std(torch.float32))
    with torch.no_grad():
        ev = bare(pixel_values=x).last_hidden_state
        bare.train()
        t1, t2 = bare(pixel_values=x).last_hidden_state, bare(pixel_values=x).last_hidden_state
    assert not torch.equal(ev, t1) and not torch.equal(t1, t2)                              # the raw HF module is NOT deterministic in train mode
    enc = make_encoder(64, 2)
    ref = enc.field(fr)
    enc.train()                                                                              # a careless caller
    enc.backbone.train()                                                                     # ... and nn.Module.train() semantics reaching in
    enc.train()
    assert not enc.backbone.rope_embeddings.training and torch.equal(enc.field(fr), ref)
    enc.set_checkpointing(True)
    assert not enc.backbone.rope_embeddings.training and not enc.backbone.embeddings.training
    assert all(m.training for m in enc.backbone.modules() if type(m).__name__ == "DINOv3ViTLayer")
    assert torch.equal(enc.field(fr), ref) and torch.equal(enc.field(fr), enc.field(fr))     # checkpointing on, still the builder's features, deterministic
    enc.train()
    assert not enc.backbone.rope_embeddings.training and torch.equal(enc.field(fr), ref)
    enc.set_checkpointing(False)
    assert not any(m.training for m in enc.backbone.modules())


def test_checkpointing_changes_nothing_but_memory():
    fr = smooth_frames(3, TINY_HW, seed=4)
    a = make_encoder(64, 2, r=4)
    b = make_encoder(64, 2, r=4)
    for p in a.lora_parameters():
        nn.init.normal_(p, std=0.05)                                                          # B != 0 so the LoRA branch is live
    b.load_state_dict(a.state_dict())
    b.set_checkpointing(True)
    assert b.built_levers()["checkpointing"] is True and a.built_levers()["checkpointing"] is False
    _loss_through(a, fr).backward()
    _loss_through(b, fr).backward()
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if "lora_" in n:
            assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-6), n


def test_merge_equals_unmerged_and_has_exactly_the_frozen_encoders_parameter_count():
    enc = make_encoder(64, 3, r=4)
    for p in enc.lora_parameters():
        nn.init.normal_(p, std=0.05)
    plain = tiny_backbone(64, 3, 2, 64)
    fr = smooth_frames(3, TINY_HW, seed=5)
    merged = enc.merged_encoder()
    assert torch.allclose(enc.field(fr), merged.field(fr), rtol=1e-4, atol=1e-5)               # fp32 float order only
    base = E._PlainEncoder(copy.deepcopy(enc.backbone), enc.mean.clone(), enc.std.clone())
    base.backbone = tiny_backbone(64, 3, 2, 64)
    assert not torch.allclose(base.field(fr), enc.field(fr), rtol=1e-4, atol=1e-5)                # the update is not vacuous: B != 0 moved the output
    assert sum(p.numel() for p in merged.backbone.parameters()) == sum(p.numel() for p in plain.parameters())
    assert not any("lora_" in n for n, _ in merged.named_parameters()) and not E.lora_modules(merged)
    assert sum(1 for _ in merged.backbone.parameters()) == sum(1 for _ in plain.parameters())
    assert enc.trainable_count() > 0 and merged.trainable_count() == 0                          # the original keeps its adapters; the merged has none
    enc2 = copy.deepcopy(enc)
    enc2.merge()
    assert all(not m.merged for _, m in E.lora_modules(enc2))                                    # merge() is non-destructive
    # a rank-r update really is folded: W' - W == (alpha/r) B A on every adapted projection
    bb = enc.merge()
    for mod_name, m in [(n, m) for n, m in enc.backbone.named_modules() if isinstance(m, E.LoRALinear)]:
        w2 = dict(bb.named_modules())[mod_name]
        assert torch.allclose(w2.weight - m.base.weight, m.delta_weight(), atol=1e-6)


def test_lora_state_roundtrip_and_refuses_a_different_shape():
    a = make_encoder(64, 2, r=4)
    for p in a.lora_parameters():
        nn.init.normal_(p, std=0.05)
    sd = a.lora_state_dict()
    b = make_encoder(64, 2, r=4, seed=9)
    b.load_lora_state_dict(sd)
    assert all(torch.equal(p, q) for p, q in zip(a.lora_parameters(), b.lora_parameters()))
    with pytest.raises(ValueError, match="LoRA checkpoint r"):
        make_encoder(64, 2, r=8).load_lora_state_dict(sd)
    with pytest.raises(ValueError, match="targets"):
        make_encoder(64, 2, r=4, targets=("q", "k", "v")).load_lora_state_dict(sd)


# ------------------------------------------------------------------------------------------------ 3. exact accumulation
def _batch_stat_loss(feats, w):
    """A loss with the two things that break naive accumulation: a masked mean normalised by the MASK SUM, and a batch statistic (variance over the batch)."""
    z = feats.mean((1, 2))                                                  # [B, D]
    m = (z[:, 0] > z[:, 0].median()).float()                                # a data-dependent mask: its sum differs per micro-batch
    a = ((z * w).pow(2).mean(-1) * m).sum() / m.sum().clamp_min(1.0)
    return a + z.var(0).mean()                                              # SIGReg-like: a statistic of the whole batch


def test_the_gradient_cache_equals_the_full_batch_gradient_and_plain_accumulation_does_not():
    B, F = 6, 3
    fr_all = smooth_frames(B * F, TINY_HW, seed=7)
    fmap = torch.arange(B * F).view(B, F)
    w = torch.linspace(0.5, 1.5, 1024)

    def fresh():
        e = make_encoder(64, 2, r=4)
        g = torch.Generator().manual_seed(11)
        for p in e.lora_parameters():
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.05)                          # the same A and B in every copy
        return e
    ref = fresh()
    feats = ref.pooled(fr_all)[fmap]                                         # the full batch, plain autograd, no chunking
    _batch_stat_loss(feats, w[:feats.shape[-1]]).backward()
    g_ref = [p.grad.clone() for p in ref.lora_parameters()]
    for chunk in (1, 4, 7, B * F):
        e = fresh()
        out = E.cached_grad_step(e, fr_all, fmap, lambda f: {"terms": {"total": _batch_stat_loss(f, w[:f.shape[-1]])}}, chunk=chunk, device="cpu")
        assert out["leaf_grad_abs_sum"] > 0
        for p, g in zip(e.lora_parameters(), g_ref):
            assert torch.allclose(p.grad, g, rtol=2e-4, atol=1e-6), chunk
    # control: PLAIN accumulation (mean of micro-batch losses, 3 windows each) is NOT the full-batch gradient
    e = fresh()
    k = 3
    for s in range(0, B, k):
        f = e.pooled(fr_all[s * F:(s + k) * F])[torch.arange(k * F).view(k, F)]
        (_batch_stat_loss(f, w[:f.shape[-1]]) * (k / B)).backward()
    worst = max(float((p.grad - g).abs().max() / g.abs().max().clamp_min(1e-12)) for p, g in zip(e.lora_parameters(), g_ref))
    assert worst > 1e-2, worst


# ------------------------------------------------------------------------------------------------ 4. the real DINOv3-L, when its weights are local
@needs_real
def test_real_dinov3_l_trainable_count_sha_and_one_real_frame_identity():
    enc = E.DinoV3LoRAEncoder.from_pretrained(r=32, alpha=32.0, targets=("q", "v"), device="cpu", dtype=torch.bfloat16)
    lv = enc.built_levers()
    assert lv["lora_trainable"] == lv["lora_expected_trainable"] == 3_145_728 and lv["lora_blocks"] == 24 and lv["lora_modules"] == 48
    assert lv["weights_sha256"] == E.BUILDER_MODEL_SHA256 and lv["lora_B_zero_at_build"] and lv["backbone_dtype"] == "torch.bfloat16"
    enc.assert_frozen()
    a32 = TR.build_parser().parse_args(["--out", "x", "--encoder", "lora", "--frames-root", "x"])
    assert TR.bad_encoder(a32, enc, {"frames_missing": 0}) == []                              # the gate's arithmetic incl. the prereg's literal 3,145,728
    frozen = E.frozen_plain_encoder(device="cpu", dtype=torch.bfloat16)
    n_frozen = sum(p.numel() for p in frozen.backbone.parameters())
    assert lv["frozen_params"] == n_frozen                                                   # LoRA wraps the SAME frozen parameters
    assert enc.merged_encoder().frozen_count() == n_frozen and enc.merged_encoder().trainable_count() == 0
    fr = smooth_frames(1, (256, 640), seed=1)
    assert torch.equal(enc.pooled(fr), frozen.pooled(fr))                                    # B = 0 on the real weights: bitwise the frozen encoder
    assert enc.pooled(fr).shape == (1, 40, 1024)


# ------------------------------------------------------------------------------------------------ 5. data: the frames behind the cache
def make_frames_root(root, names, T=60, shift=0, hw=TINY_HW):
    """``<root>/<clip>.v2ep.pt`` for every clip: cache step g <-> raw frame 2 g, ``T`` cache steps. ``shift`` offsets the frame content in time (the off-by-one mutant)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(names):
        torch.save(v2ep_from_frames(smooth_frames(T, hw, seed=i, step0=shift), c), root / f"{c}.v2ep.pt")
    return root


@pytest.fixture(scope="module")
def lworld(tmp_path_factory):
    root = tmp_path_factory.mktemp("lw")
    w = D.make_synthetic_world(root / "w", n_train=6, n_eval=2, T=60, seed=0)
    names = w["names"]["train"]
    return {"root": root / "w", "names": names, "frames": make_frames_root(root / "frames", names), "joins": w["joins"]}


def _stores(lworld, hmax=10, keep=True):
    J = D.load_join(Path(lworld["root"]) / "join_train.npz")
    store = D.ClipStore(lworld["root"], "train", lru=8)
    wins = D.train_windows(store, J, lworld["names"])
    return store, J, wins, D.V8FMiniWindows(store, J, wins, hmax=hmax), D.V8FMiniFrameWindows(store, J, wins, hmax=hmax, keep_cached=keep)


def test_frame_windows_carry_the_same_indices_the_cached_feats_were_gathered_with(lworld):
    store, J, wins, ds_c, ds_f = _stores(lworld)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(wins), 25, replace=False):
        ci, t, _, _ = wins[i]
        a_t, a_s = int(rng.integers(0, 3)), int(rng.integers(3, 7))
        ic, if_ = ds_c.item(ci, t, a_t, a_s), ds_f.item(ci, t, a_t, a_s)
        clip = str(J["clip"][ci])
        assert if_["fclip"] == clip
        want = np.clip(np.arange(t - 10 + 1, t + 1), 0, ic["T"] - 1)
        assert np.array_equal(if_["fidx"], want)
        assert np.array_equal(store.get(clip)["feats"][if_["fidx"]], ic["feats"])             # fidx IS how feats was gathered
        assert np.array_equal(if_["feats"], ic["feats"])                                       # keep_cached keeps the cache's tensor
        for k in ic:
            if k not in ("feats", "lab", "lab_t", "lab_s"):
                assert np.array_equal(np.asarray(if_[k]), np.asarray(ic[k]), equal_nan=True), k
    # without keep_cached the cached tokens are NOT carried (they would be 1.3 MB per window through the prefetch queue)
    _, _, _, _, ds_n = _stores(lworld, keep=False)
    assert "feats" not in ds_n.item(wins[40][0], wins[40][1])


def test_collate_frames_hands_every_window_its_own_frames_and_leaves_collate_alone(lworld):
    store, J, wins, ds_c, ds_f = _stores(lworld)
    fs = D.FrameStore(lworld["frames"], lru=4)
    sel = [w for w in wins if w[0] == 0][20:23] + [w for w in wins if w[0] == 3][:2]       # three ADJACENT windows of one clip share frames
    items = [ds_f.item(ci, t, 1, 3) for ci, t, _, _ in sel]
    b = D.collate_frames(items, fs)
    fm, fr = b["frame_map"], b["frames_u8"]
    assert fm.shape == (5, 10) and fr.shape[1:] == (3, *TINY_HW) and fr.dtype == torch.uint8
    assert fr.shape[0] == len(set(fm.flatten().tolist())) < 5 * 10                          # shared frames are encoded ONCE
    for bi, it in enumerate(items):
        for f in range(10):
            assert torch.equal(fr[fm[bi, f]], fs.frames(it["fclip"], [it["fidx"][f]])[0])
    # collate() itself: the cache path's tensors are what they always were, and `feats` is simply optional
    items_c = [ds_c.item(ci, t, 1, 3) for ci, t, _, _ in sel]
    bc = D.collate(items_c)
    for k, v in bc.items():
        if k in ("lab", "lab_t", "lab_s"):
            continue
        assert torch.equal(torch.nan_to_num(b[k].float(), nan=7e9), torch.nan_to_num(v.float(), nan=7e9)), k        # (lead_gap_h is NaN where there is no lead)
    assert bc["feats"].dtype == torch.float16 and list(bc)[0] == "feats"
    assert "feats" not in D.collate([{k: v for k, v in it.items() if k != "feats"} for it in items_c])


def test_unique_frames_and_the_frame_store_refusals(lworld, tmp_path):
    keys, fm = D.unique_frames(["a", "a", "b"], np.array([[0, 0, 1], [0, 1, 2], [0, 0, 5]]))
    assert keys == [("a", 0), ("a", 1), ("a", 2), ("b", 0), ("b", 5)] and fm.tolist() == [[0, 0, 1], [0, 1, 2], [3, 3, 4]]
    fs = D.FrameStore(lworld["frames"])
    assert fs.missing(["syn-train-000", "nope"]) == ["nope"] and fs.n_cache_frames("syn-train-000") == 60
    torch.save(v2ep_from_frames(smooth_frames(3, TINY_HW), "some-other-clip"), tmp_path / "x.v2ep.pt")
    with pytest.raises(ValueError, match="clip_id"):
        D.FrameStore(tmp_path).get("x")
    torch.save(v2ep_from_frames(smooth_frames(3, (32, 64)), "y"), tmp_path / "y.v2ep.pt")
    with pytest.raises(ValueError, match="png"):
        D.FrameStore(tmp_path).get("y")
    store, J, wins, ds_c, ds_f = _stores(lworld)
    short = make_frames_root(tmp_path / "short", lworld["names"], T=40)
    with pytest.raises(ValueError, match="cache steps"):
        D.collate_frames([ds_f.item(wins[0][0], wins[0][1])], D.FrameStore(short))


def test_end_to_end_identity_the_on_the_fly_feats_are_the_caches_feats_exactly_and_a_one_step_shift_is_caught(lworld, tmp_path):
    """The cache is made BY THE ENCODER (tiny, fixed weights) from the same frames; the windows -> frames -> encoder -> gather path must then return the cache's
    `feats` bit for bit, in order, with the same frame_ok. Shift the frames by one cache step and the identity probe must go non-zero."""
    enc = make_encoder(1024, 2, r=4, seed=0, bb_seed=3)
    names = lworld["names"]
    root = tmp_path / "w"
    D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    for i, c in enumerate(names):                                   # overwrite the random cache tokens with the encoder's OWN pooled fp8 tokens
        p = root / "train" / "clips" / f"{c}.pt"
        o = torch.load(p, map_location="cpu", weights_only=True)
        with torch.no_grad():
            o["pool4x10"] = enc.pooled(smooth_frames(60, TINY_HW, seed=i), quantize=True).to(torch.float8_e4m3fn)
        torch.save(o, p)
    J = D.load_join(root / "join_train.npz")
    store = D.ClipStore(root, "train", lru=8)
    wins = D.train_windows(store, J, names)
    ds = D.V8FMiniFrameWindows(store, J, wins, hmax=16, keep_cached=True)
    sel = [wins[i] for i in (0, 7, 40, 41, 42, 100, 150)]
    items = [ds.item(ci, t, 1, 3) for ci, t, _, _ in sel]
    probe = D.collate_frames(items, D.FrameStore(make_frames_root(tmp_path / "f_ok", names)))
    ident = TR.identity_probe(enc, probe, "cpu", 16)
    assert ident["max_abs"] == 0.0 and ident["frac_bitwise_equal"] == 1.0 and ident["rel_l2"] == 0.0
    got = enc.pooled(probe["frames_u8"], quantize=True)[probe["frame_map"]]                 # [B, hmax, 40, 1024] vs the cache path's `feats`
    assert got.shape == probe["feats"].shape and torch.equal(got, probe["feats"].float())
    bad = D.collate_frames(items, D.FrameStore(make_frames_root(tmp_path / "f_shift", names, shift=1)))   # frames one cache step LATE
    assert TR.identity_probe(enc, bad, "cpu", 16)["max_abs"] > 0.0


# ------------------------------------------------------------------------------------------------ 6. the trainer: flags, launch gate, end to end
def _tiny_raw(a, dev, draw_global_rng=True):
    """The tiny stand-in for ``build_encoder_raw`` (the real one loads DINOv3-L). It DRAWS from the global RNG on purpose (HF's loader may): ``make_encoder``'s
    save / restore is what keeps the heads' initialisation B's."""
    if draw_global_rng:
        torch.randn(7)
    bb = tiny_backbone(1024, 2, 16, 64, torch.float32, 3)
    m, s = E.mean_std(torch.float32)
    return E.DinoV3LoRAEncoder(bb, m, s, r=a.lora_r, alpha=a.lora_alpha, targets=TR.parse_targets(a.lora_targets), seed=a.seed, grid=a.grid[len("pool"):]).to(dev)


@pytest.fixture
def tiny_trainer(monkeypatch):
    monkeypatch.setattr(TR, "build_encoder_raw", _tiny_raw)
    return TR


def lora_argv(tmp_path, out, extra=(), arm="M1", steps="3", batch="4"):
    frames = make_frames_root(tmp_path / "frames", [f"syn-train-{i:03d}" for i in range(6)])
    return ["--synthetic", "--batch", batch, "--d", "32", "--n-q", "2", "--steps", steps, "--log-every", "1", "--warmup", "2", "--ckpt-every", "2",
            "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", arm, "--encoder", "lora", "--frames-root", str(frames),
            "--lora-r", "4", "--lora-alpha", "4", "--out", str(out), *extra]


def test_every_new_flag_has_a_declared_vs_built_entry_and_a_default_that_builds_nothing():
    new = {"encoder", "lora_r", "lora_alpha", "lora_targets", "lora_lr", "frames_root", "accum", "enc_ckpt"}
    dests = {act.dest for act in TR.build_parser()._actions}
    assert new <= dests and new <= set(TR.FLAG_CHECKS)
    a = TR.build_parser().parse_args(["--out", "x"])
    assert a.encoder == "cache" and (a.lora_r, a.lora_alpha, a.lora_targets, a.lora_lr, a.frames_root, a.accum, a.enc_ckpt) == (32, 32.0, "q,v", None, None, 1, False)
    assert TR.LORA_PARSER_DEFAULTS == {"lora_r": 32, "lora_alpha": 32.0, "lora_targets": "q,v", "lora_lr": None, "frames_root": None, "accum": 1, "enc_ckpt": False}


@pytest.mark.parametrize("extra", [["--lora-r", "8"], ["--lora-alpha", "16"], ["--lora-targets", "q,k,v"], ["--lora-lr", "1e-4"], ["--accum", "2"], ["--enc-ckpt"],
                                   ["--frames-root", "x"]])
def test_cache_encoder_refuses_every_lora_flag_that_builds_nothing(extra):
    a = TR.build_parser().parse_args(["--out", "x", *extra])
    bad = TR.bad_encoder(a, None, {})
    assert bad and "builds nothing" in bad[0], bad
    assert TR.bad_encoder(TR.build_parser().parse_args(["--out", "x"]), None, {}) == []


def test_launch_gate_refuses_every_lie_about_the_built_encoder(tiny_trainer, tmp_path, monkeypatch):
    import copy as _c

    def run(extra=(), raw=None):
        out = tmp_path / f"o{len(list(tmp_path.glob('o*')))}"
        if raw is not None:
            monkeypatch.setattr(TR, "build_encoder_raw", raw)
        return TR.main(lora_argv(tmp_path, out, ["--selfcheck-only", *extra]))

    def lie(mutate):
        def raw(a, dev):
            enc = _tiny_raw(a, dev)
            mutate(enc)
            return enc
        return raw

    def with_args(**kw):
        def raw(a, dev):
            b = _c.copy(a)
            for k, v in kw.items():
                setattr(b, k, v)
            return _tiny_raw(b, dev)
        return raw
    assert run() == 0                                                                           # the honest build passes
    with pytest.raises(SystemExit, match="ZZDVB-REFUSEDZZ.*lora_r"):
        run(raw=with_args(lora_r=2))                                                             # built rank 2, declared 4
    with pytest.raises(SystemExit, match="lora_targets"):
        run(raw=with_args(lora_targets="q,k,v"))
    with pytest.raises(SystemExit, match="not zero at build"):
        run(raw=lie(lambda enc: [nn.init.normal_(p, std=0.1) for n, p in enc.named_parameters() if "lora_B" in n]))
    with pytest.raises(SystemExit, match="non-LoRA parameters are trainable"):
        run(raw=lie(lambda enc: enc.backbone.norm.weight.requires_grad_(True)))
    with pytest.raises(SystemExit, match="is not the cache builder's"):
        run(raw=lie(lambda enc: setattr(enc, "weights_sha256", "0" * 64)))
    with pytest.raises(SystemExit, match="accum 3 must divide"):
        run(extra=["--accum", "3"], raw=_tiny_raw)
    monkeypatch.setattr(TR, "build_encoder_raw", _tiny_raw)
    out = tmp_path / "miss"                                                                      # a missing v2ep for one train clip
    argv = lora_argv(tmp_path, out, ["--selfcheck-only"])
    (Path(argv[argv.index("--frames-root") + 1]) / "syn-train-003.v2ep.pt").unlink()
    with pytest.raises(SystemExit, match="1 train clips have no"):
        TR.main(argv)


def test_the_encoder_build_never_moves_the_heads_initialisation(tiny_trainer, tmp_path, monkeypatch):
    """B's heads and L's heads must start from the same bits and the same RNG stream: the trainer seeds, builds the heads, THEN the encoder; the encoder build (which here draws
    from the global RNG, as HF's loader may) is wrapped in a save / restore. The control: without the restore the stream that follows is NOT B's."""
    seen = {}
    orig = TR.declared_vs_built

    def spy(a, cfg, model, opt, info, *rest, **kw):                                             # forwards whatever the trainer passes (ego, sched, dag were added after this test)
        seen[a.encoder] = ({k: v.clone() for k, v in model.state_dict().items()}, torch.get_rng_state().clone())
        return orig(a, cfg, model, opt, info, *rest, **kw)
    monkeypatch.setattr(TR, "declared_vs_built", spy)
    cache_argv = ["--synthetic", "--batch", "4", "--d", "32", "--n-q", "2", "--steps", "2", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M1",
                  "--selfcheck-only", "--out"]
    assert TR.main([*cache_argv, str(tmp_path / "c")]) == 0
    assert TR.main(lora_argv(tmp_path, tmp_path / "l", ["--selfcheck-only"])) == 0
    (sd_c, rng_c), (sd_l, rng_l) = seen["cache"], seen["lora"]
    assert set(sd_c) == set(sd_l) and all(torch.equal(sd_c[k], sd_l[k]) for k in sd_c)         # the heads: identical to the last bit
    assert torch.equal(rng_c, rng_l)                                                             # ... and so is the stream that follows
    monkeypatch.setattr(TR, "make_encoder", lambda a, dev: TR.build_encoder_raw(a, dev))        # the mutant: no save / restore
    seen.clear()
    assert TR.main(lora_argv(tmp_path, tmp_path / "l2", ["--selfcheck-only"])) == 0
    assert not torch.equal(seen["lora"][1], rng_c)


def test_reachability_flag_to_loss_to_backward_end_to_end(tiny_trainer, tmp_path):
    out = tmp_path / "run"
    assert TR.main(lora_argv(tmp_path, out, ["--accum", "2", "--enc-ckpt"], steps="4")) == 0
    sc = json.load(open(out / "selfcheck.json", encoding="utf-8"))
    lv = sc["levers"]["encoder"]
    assert sc["declared_vs_built"] == "PASS" and lv["encoder"] == "lora" and lv["lora_trainable"] == lv["lora_expected_trainable"] == 4 * 2048 * 2 * 2
    assert lv["lora_r"] == 4 and lv["lora_targets"] == ["q", "v"] and lv["checkpointing"] is True and sc["levers"]["accum"]["micro_windows"] == 2
    live = sc["live"]
    assert live["pass"] and live["lora_A_grad_step0"] == 0.0 and live["lora_B_grad_step0"] > 0.0                      # B = 0 -> A's grad is EXACTLY 0 at step 1
    assert live["control_frozen_dinov3_grad_sum"] == 0.0 and live["control_prior_grad_sum"] == 0.0
    assert live["lora_B_modules_dead_at_step0"] == [] and live["lora_A_modules_dead_after_step1"] == [] and live["unexpected_dead"] == []
    assert "identity_probe" in sc and sc["identity_probe"]["n_frames_unique"] > 0                 # recorded (huge here: the synthetic cache is random), not gated
    s = json.load(open(out / "summary.json", encoding="utf-8"))
    assert s["done"] is True and s["encoder"] == "lora"
    ck = torch.load(out / "ckpt_last.pt", weights_only=False)
    assert set(ck["model"]) and "lora" in ck and ck["lora"]["meta"]["r"] == 4
    assert any(float(t.abs().sum()) > 0 for n, t in ck["lora"]["tensors"].items() if "lora_B" in n)    # training MOVED the B matrices
    assert any(float(t.abs().sum()) > 0 for n, t in ck["lora"]["tensors"].items() if "lora_A" in n)
    log = [json.loads(l) for l in open(out / "train_log.jsonl", encoding="utf-8")]
    assert all(np.isfinite(r["total"]) and r["lora_grad_norm"] > 0 for r in log)
    # `model` has EXACTLY the cache arm's layout: the unchanged v8f_mini_eval.py loads it
    cache_out = tmp_path / "cache"
    assert TR.main(["--synthetic", "--batch", "4", "--d", "32", "--n-q", "2", "--steps", "2", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M1",
                    "--out", str(cache_out)]) == 0
    ckc = torch.load(cache_out / "ckpt_last.pt", weights_only=False)
    assert set(ck["model"]) == set(ckc["model"]) and "lora" not in ckc


def test_gradient_accumulation_is_exact_end_to_end_including_the_wm_sigreg_term(tiny_trainer, tmp_path):
    """M2b carries the WM auxiliary (SIGReg = a statistic of the whole batch) and masked-mean losses: accum 1 and accum 4 must give the same heads AND the same LoRA."""
    res = {}
    for k, extra in (("a1", []), ("a4", ["--accum", "4", "--enc-ckpt"])):
        out = tmp_path / k
        assert TR.main(lora_argv(tmp_path, out, extra, arm="M2b", steps="2", batch="8")) == 0
        res[k] = torch.load(out / "ckpt_last.pt", weights_only=False)
    for n, t in res["a1"]["model"].items():
        assert torch.allclose(t.float(), res["a4"]["model"][n].float(), rtol=1e-3, atol=2e-5), n
    for n, t in res["a1"]["lora"]["tensors"].items():
        assert torch.allclose(t, res["a4"]["lora"]["tensors"][n], rtol=1e-3, atol=2e-5), n
    assert any(float(t.abs().sum()) > 0 for n, t in res["a1"]["lora"]["tensors"].items() if "lora_B" in n)


def test_lora_run_resumes_as_the_same_run(tiny_trainer, tmp_path, monkeypatch):
    common = ["--ckpt-every", "2"]
    full, res = tmp_path / "full", tmp_path / "res"
    assert TR.main(lora_argv(tmp_path, full, common, steps="5")) == 0
    monkeypatch.setenv("V8F_TRAIN_CRASH_AT", "3")
    with pytest.raises(RuntimeError, match="V8F_TRAIN_CRASH_AT"):
        TR.main(lora_argv(tmp_path, res, common, steps="5"))
    monkeypatch.delenv("V8F_TRAIN_CRASH_AT")
    assert TR.main(lora_argv(tmp_path, res, common, steps="5")) == 0
    A, B = (torch.load(d / "ckpt_last.pt", weights_only=False) for d in (full, res))
    for n in A["model"]:
        assert torch.allclose(A["model"][n].float(), B["model"][n].float(), atol=1e-5), n
    for n in A["lora"]["tensors"]:
        assert torch.allclose(A["lora"]["tensors"][n], B["lora"]["tensors"][n], atol=1e-6), n


# ------------------------------------------------------------------------------------------------ 7. --encoder cache IS today's trainer
GOLDEN_CACHE_PATH = {  # sha256 of the final `model` state (sorted keys, raw bytes) of the PRE-LoRA trainer on `--synthetic --batch 8 --d 32 --n-q 2 --steps 6 ... --seed 5`
    "torch": "2.11.0+cu128",
    "M1": "6d36d9d2cfca19a5dedeb9ab1fbfc425e27ea02bfbb7b3ba68d4685ef31b8563",
    "M2b": "9be55255fced3fb66b3c6950b3dcadfaf3aebdbba82c7cff3d64ccef2bda3b24"}


def _state_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if torch.is_tensor(v):
            h.update(k.encode())
            h.update(v.detach().contiguous().view(torch.uint8).numpy().tobytes() if v.numel() else b"")
    return h.hexdigest()


@pytest.mark.parametrize("arm", ["M1", "M2b"])
def test_encoder_cache_is_the_pre_lora_trainer_bit_for_bit(tmp_path, arm):
    """`--encoder cache` (explicit) and no flag must give identical final checkpoints, and both must equal the PRE-LoRA trainer's: the golden hash was taken from the
    tip's trainer (190be80a) -- see raw/2026-10-25-lora/bitwise_cache_path.py/.out for the side-by-side run (model, optimiser state and train_log, M1 and M2b). The golden is
    torch-version specific; set V8F_PRE_LORA_TREE to a pristine tree to compare against it live instead."""
    common = ["--synthetic", "--batch", "8", "--d", "32", "--n-q", "2", "--steps", "6", "--log-every", "1", "--warmup", "2", "--ckpt-every", "3", "--threads", "1",
              "--kin-mode", "yaw", "--seed", "5", "--arm", arm]
    hs = {}
    for k, extra in (("noflag", []), ("explicit", ["--encoder", "cache"])):
        out = tmp_path / k
        assert TR.main([*common, *extra, "--out", str(out)]) == 0
        ck = torch.load(out / "ckpt_last.pt", weights_only=False)
        assert "lora" not in ck and "encoder_levers" not in ck
        hs[k] = _state_hash(ck["model"])
        assert "encoder" not in json.load(open(out / "summary.json", encoding="utf-8"))
        assert "encoder" not in json.load(open(out / "selfcheck.json", encoding="utf-8"))["levers"]
    assert hs["noflag"] == hs["explicit"]
    pre = os.environ.get("V8F_PRE_LORA_TREE")
    if pre:
        import subprocess
        out = tmp_path / "pre"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=os.pathsep.join([str(Path(pre) / "stack"), str(Path(pre) / "taniteval")]))
        r = subprocess.run([sys.executable, str(Path(pre) / "stack" / "scripts" / "v8f_mini_train.py"), *common, "--out", str(out)], env=env, capture_output=True, text=True,
                           encoding="utf-8")
        assert r.returncode == 0, r.stderr[-800:]
        assert _state_hash(torch.load(out / "ckpt_last.pt", weights_only=False)["model"]) == hs["noflag"]
    elif torch.__version__ == GOLDEN_CACHE_PATH["torch"]:
        assert hs["noflag"] == GOLDEN_CACHE_PATH[arm]
    else:
        pytest.skip(f"golden hash is for torch {GOLDEN_CACHE_PATH['torch']}; set V8F_PRE_LORA_TREE to compare against a pristine tree")


# ------------------------------------------------------------------------------------------------ 8. real weights: the trainer's loader, the RNG, the launch gate
@needs_real
def test_real_loader_never_moves_the_global_rng_and_builds_what_argv_declares():
    """The TRAINER's real loader (`build_encoder_raw` via `make_encoder`) with NON-default flags: whatever is declared must be what is built (a loader that dropped `--lora-r`
    would build the default 32 and be refused here), and the heads' RNG stream must come out untouched."""
    a = TR.build_parser().parse_args(["--out", "x", "--encoder", "lora", "--frames-root", "x", "--device", "cpu", "--lora-r", "8", "--lora-alpha", "8", "--lora-targets", "q,v", "--seed", "5"])
    torch.manual_seed(5)
    st = torch.get_rng_state().clone()
    enc = TR.make_encoder(a, "cpu")
    assert torch.equal(torch.get_rng_state(), st)                                            # the heads' stream is untouched by the real loader
    assert enc.r == 8 and enc.alpha == 8.0 and enc.targets == ("q", "v") and enc.seed == 5
    assert enc.built_levers()["lora_trainable"] == 8 * 2048 * 2 * 24
    assert TR.bad_encoder(a, enc, {"frames_missing": 0}) == []                               # weights sha256 == the builder's, 24 blocks, frozen, B = 0
    a2 = TR.build_parser().parse_args(["--out", "x", "--encoder", "lora", "--frames-root", "x", "--lora-r", "16"])
    assert any("lora_r" in m for m in TR.bad_encoder(a2, enc, {"frames_missing": 0}))        # a rank the encoder was not built with


# ------------------------------------------------------------------------------------------------ 9. the eval route: LoRA ckpt -> merge -> the BUILDER encodes the eval clips -> unchanged eval
def _load_route():
    path = PKG / "code" / "v8f_lora_eval_cache.py"
    if not path.is_file():
        pytest.skip("the eval-route script is not in this checkout")
    spec = importlib.util.spec_from_file_location("v8f_lora_eval_cache_t", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert Path(m.REPO).resolve() == REPO.resolve(), (m.REPO, REPO)               # the script finds the repo root from ITS OWN location (a parents[] off-by-one once hid in both code/ scripts)
    return m


@needs_builder
@pytest.mark.parametrize("zero", [True, False])
def test_eval_route_reproduces_the_builders_cache_at_zero_lora_and_the_merged_encode_otherwise(tmp_path, monkeypatch, zero):
    R = _load_route()
    pep, ship = E.load_builder()
    monkeypatch.setattr(pep, "GH", 16)
    monkeypatch.setattr(pep, "GW", 40)                                                       # the route runs at the REAL geometry
    eps = tmp_path / "eps"
    eps.mkdir()
    clip = "route-clip-000"
    frames = smooth_frames(3, (256, 640), seed=4)
    torch.save(v2ep_from_frames(frames, clip), eps / f"{clip}.v2ep.pt")
    o = torch.load(eps / f"{clip}.v2ep.pt", weights_only=False)
    base_enc = make_encoder(1024, 2, r=4, bb_seed=3)
    if not zero:
        g = torch.Generator().manual_seed(2)
        for p in base_enc.lora_parameters():
            p.data.copy_(torch.randn(p.shape, generator=g) * 0.02)

    def factory(meta, device, dtype):
        e = make_encoder(1024, 2, r=(meta or {"r": 4})["r"], alpha=float((meta or {"alpha": 4.0})["alpha"]), bb_seed=3)
        return e
    monkeypatch.setattr(R, "BACKBONE_FACTORY", factory)
    # the base cache = what the BUILDER's own encode_clip writes for the frozen tiny encoder
    plain = E._PlainEncoder(copy.deepcopy(make_encoder(1024, 2, r=4, bb_seed=3).merge()), base_enc.mean.clone(), base_enc.std.clone())
    base_payload = pep.encode_clip(o, ship, plain.backbone, plain.mean, plain.std, 5, "cpu", ["4x10"], torch.float8_e4m3fn)
    base_root = tmp_path / "base"
    (base_root / "eval" / "clips").mkdir(parents=True)
    torch.save(dict(base_payload, meta=pep.make_meta(clip, o, base_payload, ["4x10"], torch.float8_e4m3fn)), base_root / "eval" / "clips" / f"{clip}.pt")
    json.dump({"clip": clip}, open(base_root / "eval" / "manifest.jsonl", "w", encoding="utf-8"))
    argv = ["--eval-eps", str(eps), "--out", str(tmp_path / "new"), "--base-cache", str(base_root), "--device", "cpu", "--dtype", "fp32", "--clips", clip]
    if zero:
        assert R.main([*argv, "--zero-lora"]) == 0
    else:
        torch.save({"lora": base_enc.lora_state_dict(), "step": 7}, tmp_path / "ck.pt")
        assert R.main([*argv, "--ckpt", str(tmp_path / "ck.pt")]) == 0
    new = torch.load(tmp_path / "new" / "eval" / "clips" / f"{clip}.pt", weights_only=True)
    ent = json.loads(open(tmp_path / "new" / "eval" / "manifest.jsonl", encoding="utf-8").read().strip().splitlines()[-1])
    mj = json.load(open(tmp_path / "new" / "eval" / "meta.json", encoding="utf-8"))
    assert mj["merged_trainable"] == 0 and mj["merged_params"] == sum(p.numel() for p in plain.backbone.parameters()) and mj["unmerged_lora_trainable"] > 0
    for k in ("v", "a", "steer", "yaw_rate", "pose_grid", "kin3", "p10"):
        assert R.same_bits(new[k], base_payload[k]), k
    if zero:
        assert torch.equal(new["pool4x10"].view(torch.uint8), base_payload["pool4x10"].view(torch.uint8))        # B = 0: BIT-identical to the builder's own cache clip
        assert ent["content_md5"] == pep.content_md5(base_payload)
    else:
        assert not torch.equal(new["pool4x10"].view(torch.uint8), base_payload["pool4x10"].view(torch.uint8))     # the LoRA moved the tokens
        want = base_enc.pooled(E.decode_cache_frames(o, [0, 1, 2]), "4x10", quantize=True)
        assert float((new["pool4x10"].float() == want).float().mean()) > 0.99                                    # merged == on-the-fly, up to rare fp8 ties
    # a base cache whose NON-feature arrays differ is a STOP
    bad = torch.load(base_root / "eval" / "clips" / f"{clip}.pt", weights_only=True)
    bad["pose_grid"] = bad["pose_grid"] + 1.0
    torch.save(bad, base_root / "eval" / "clips" / f"{clip}.pt")
    with pytest.raises(SystemExit, match="non-feature arrays differ"):
        R.main([*argv[:3], str(tmp_path / "new2"), *argv[4:], *(["--zero-lora"] if zero else ["--ckpt", str(tmp_path / "ck.pt")])])
    with pytest.raises(SystemExit, match="exactly one of"):
        R.main(argv)


# ------------------------------------------------------------------------------------------------ 10. the AR-2 smoke script
def test_smoke_script_runs_the_trainer_and_reports_the_ar2_checks_and_a_projection(tiny_trainer, tmp_path):
    path = PKG / "code" / "v8f_lora_smoke.py"
    if not path.is_file():
        pytest.skip("the smoke script is not in this checkout")
    spec = importlib.util.spec_from_file_location("v8f_lora_smoke_t", path)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    assert Path(S.REPO).resolve() == REPO.resolve(), (S.REPO, REPO)               # same pin as the eval route's
    frames = make_frames_root(tmp_path / "frames", [f"syn-train-{i:03d}" for i in range(6)])
    targv = ["--synthetic", "--d", "32", "--n-q", "2", "--warmup", "2", "--threads", "1", "--kin-mode", "yaw", "--seed", "5", "--arm", "M2b", "--lora-r", "4",
             "--lora-alpha", "4", "--frames-root", str(frames), "--batch", "64"]
    out = tmp_path / "smoke"
    rc = S.main(["--steps", "4", "--out", str(out), "--warmup", "1", "--census-batches", "2", "--heads-b64-steps", "1", "--", *targv])
    rep = json.load(open(out / "smoke_report.json", encoding="utf-8"))
    v = rep["verdict"]
    assert rc == 0 and v["AR2_gradient_and_finite_checks_pass"] and v["n_steps_run"] == 4
    assert v["lora_A_grad_exactly_zero_at_step_1"] and v["lora_A_grad_nonzero_from_step_2"] and v["lora_B_grad_nonzero_every_step"]
    assert v["frozen_dinov3_control_exactly_zero"] and v["frozen_prior_control_exactly_zero"] and v["every_head_group_nonzero_every_step"]
    assert {"ckpt", "goal", "lat", "lon", "cons", "ade", "fde", "total", "wm"} <= set(v["declared_terms"])                      # M2b: the WM term too
    steps = [json.loads(l) for l in open(out / "smoke_steps.jsonl", encoding="utf-8")]
    assert len(steps) == 4 and steps[0]["n_windows"] == 1                                                                       # batch 1
    assert steps[0]["groups"]["lora_A"]["abs_sum"] == 0.0 and steps[1]["groups"]["lora_A"]["abs_sum"] > 0.0
    assert {"phase1_encode_nograd_s", "heads_fwd_bwd_s", "phase2_encode_grad_s", "n_unique_frames"} <= set(steps[0]["timing"])
    pr = rep["projection_ESTIMATED"]
    assert pr["hours_one_seed_ESTIMATED"] > 0 and pr["steps"] == 19164 and 1.0 <= pr["U_unique_frames_per_window_b64"] <= 16.0 and pr["census"]["hmax"] == 16
    assert "ESTIMATED" in json.dumps(list(pr)) and pr["venue_rule_proof5"]["box2_eligible_by_this_estimate"] == "n/a (not measured on a GPU)"
    assert rep["selfcheck"]["levers"]["encoder"]["lora_trainable"] == 4 * 2048 * 2 * 2
