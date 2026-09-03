"""D-V7-DINO-SEED — the DINOv3 encoder seed, its refusals, and the trunk schedule.

⛔ WHAT THIS FILE IS DEFENDING AGAINST, in one sentence: **a silent partial
seed**. A load that puts most of the trunk in place and leaves the rest at random
init looks EXACTLY like a successful init in every artifact the run writes — the
same family as the `df`/cgroup/`step_s` traps, where a probe reporting the wrong
scope reads as an answer. Every refusal below is therefore SHOWN TO FIRE, and
the mapping is checked by VALUE, never by key count.

⭐ THE CENTREPIECE IS NOT A KEY-COUNT TEST. `test_the_seeded_encoder_reproduces_
a_dinov3_reference_forward` runs a from-scratch reference implementation of a
DINOv3 block (LayerNorm -> biased-q/v, UNbiased-k attention -> o_proj -> xLS1 ->
residual; LayerNorm -> GELU MLP -> xLS2 -> residual) beside the seeded
`ViTEncoder` and requires the two to agree numerically. That is what actually
proves the qkv row-concat, the zero-fill for DINOv3's absent key bias
(`key_bias: false`) and the LayerScale fold are TRANSFERS rather than
approximations — a key-count test passes for all three of those being wrong.

⛔ NO REAL DINOv3 WEIGHTS ARE REQUIRED, and none are downloaded. HF quota is a
hard ceiling. The synthetic state dict below reproduces the HuggingFace
`DINOv3ViTModel` LAYOUT verbatim — the layout was read off the real ViT-L/16
snapshot on this box (415 tensors,
`~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/
snapshots/ea8dc2863c51be0a264bab82070e3e8836b02d51`), and the converter was run
against those real weights once (292/293 target tensors mapped). ⚠️ UNVERIFIED
BY THIS FILE: a real ViT-B/16 load — that snapshot is NOT on this box.

The load-bearing test is the last one: at the flag defaults the optimizer and
the LR schedule are the incumbent objects, not merely equivalent ones.
"""
from __future__ import annotations

import json
import math
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.config import EncoderConfig                             # noqa: E402
from tanitad.models.encoder import ViTEncoder                        # noqa: E402

import dinov3_seed_checkpoint as S                                   # noqa: E402
from train_v6_staged import (build_lr_scheduler,                     # noqa: E402
                             build_parser, build_stack_from_args,
                             build_trunk_optimizer, load_encoder_seed,
                             apply_encoder_seed, assert_trunk_anchor_unwired,
                             trunk_lr_factor, trunk_lr_split_active,
                             O7_DEFAULT_MODEL, O7Distill)

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "train_v6_staged.py"


@contextmanager
def _quiet_sched():
    """torch warns when `sched.step()` runs without a preceding `opt.step()`.
    These tests read the SCHEDULE, not the updates, so the warning is noise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        yield

# ---------------------------------------------------------------------------
# the synthetic "DINOv3" — the HuggingFace DINOv3ViTModel LAYOUT, tiny
# ---------------------------------------------------------------------------
D, L, H, P, MLP = 32, 2, 4, 16, 128        # d_model, depth, n_heads, patch, ffn


def _syn_state(d=D, depth=L, mlp=MLP, patch=P, ch=3, *, key_bias=False,
               n_reg=4, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.05

    st = {
        "embeddings.patch_embeddings.weight": r(d, ch, patch, patch),
        "embeddings.patch_embeddings.bias": r(d),
        "embeddings.cls_token": r(1, 1, d),
        "embeddings.mask_token": r(1, 1, d),
        "embeddings.register_tokens": r(1, n_reg, d),
        "norm.weight": 1.0 + r(d),
        "norm.bias": r(d),
    }
    for i in range(depth):
        s = f"layer.{i}."
        st |= {
            s + "norm1.weight": 1.0 + r(d), s + "norm1.bias": r(d),
            s + "norm2.weight": 1.0 + r(d), s + "norm2.bias": r(d),
            s + "attention.q_proj.weight": r(d, d),
            s + "attention.k_proj.weight": r(d, d),
            s + "attention.v_proj.weight": r(d, d),
            s + "attention.q_proj.bias": r(d),
            s + "attention.v_proj.bias": r(d),
            s + "attention.o_proj.weight": r(d, d),
            s + "attention.o_proj.bias": r(d),
            s + "layer_scale1.lambda1": 0.5 + r(d),
            s + "layer_scale2.lambda1": 0.5 + r(d),
            s + "mlp.up_proj.weight": r(mlp, d), s + "mlp.up_proj.bias": r(mlp),
            s + "mlp.down_proj.weight": r(d, mlp), s + "mlp.down_proj.bias": r(d),
        }
        if key_bias:
            st[s + "attention.k_proj.bias"] = r(d)
    return st


def _enc_cfg(h=32, w=32, ch=3, d=D, depth=L, heads=H, patch=P) -> EncoderConfig:
    return EncoderConfig(in_channels=ch, image_size=h,
                         image_width=None if w == h else w,
                         patch_size=patch, d_model=d, depth=depth, n_heads=heads)


def _geo(state):
    return S.infer_source_geometry(state, {})


def _seed(state=None, cfg=None, **kw):
    state = _syn_state() if state is None else state
    cfg = _enc_cfg() if cfg is None else cfg
    return S.build_seed(state, cfg, src_geo=_geo(state), **kw)


# ===========================================================================
# 1. the mapping is exact and complete
# ===========================================================================

def test_the_mapping_is_exact_and_complete_on_a_synthetic_seed():
    st = _syn_state()
    sd, rep = _seed(st)
    ref = ViTEncoder(_enc_cfg()).state_dict()

    # complete: every target tensor is written except the DECLARED left-at-init
    assert set(sd) | set(S.TARGET_LEFT_AT_INIT) == set(ref), (
        f"seed does not cover the encoder: "
        f"missing={sorted(set(ref) - set(sd) - set(S.TARGET_LEFT_AT_INIT))}")
    assert rep["left_at_init_keys"] == ["pos"]
    assert rep["n_mapped"] == len(ref) - 1
    for k, v in sd.items():
        assert v.shape == ref[k].shape, k

    # exact, BY VALUE, transform by transform
    torch.testing.assert_close(sd["patch.weight"],
                               st["embeddings.patch_embeddings.weight"])
    torch.testing.assert_close(sd["norm.bias"], st["norm.bias"])
    for i in range(L):
        s, t = f"layer.{i}.", f"blocks.{i}."
        torch.testing.assert_close(sd[t + "norm1.weight"], st[s + "norm1.weight"])
        # qkv: row concat in q, k, v order
        torch.testing.assert_close(
            sd[t + "attn.in_proj_weight"],
            torch.cat([st[s + "attention.q_proj.weight"],
                       st[s + "attention.k_proj.weight"],
                       st[s + "attention.v_proj.weight"]], 0))
        # DINOv3 has key_bias=false -> the k slice must be EXACTLY zero
        b = sd[t + "attn.in_proj_bias"]
        torch.testing.assert_close(b[:D], st[s + "attention.q_proj.bias"])
        assert torch.count_nonzero(b[D:2 * D]) == 0, "k-bias slice must be zero"
        torch.testing.assert_close(b[2 * D:], st[s + "attention.v_proj.bias"])
        # LayerScale folded into the two branch-terminal linear maps
        ls1, ls2 = st[s + "layer_scale1.lambda1"], st[s + "layer_scale2.lambda1"]
        torch.testing.assert_close(sd[t + "attn.out_proj.weight"],
                                   st[s + "attention.o_proj.weight"] * ls1[:, None])
        torch.testing.assert_close(sd[t + "attn.out_proj.bias"],
                                   st[s + "attention.o_proj.bias"] * ls1)
        torch.testing.assert_close(sd[t + "mlp.2.weight"],
                                   st[s + "mlp.down_proj.weight"] * ls2[:, None])
        torch.testing.assert_close(sd[t + "mlp.2.bias"],
                                   st[s + "mlp.down_proj.bias"] * ls2)
        # the MLP's first layer is untouched
        torch.testing.assert_close(sd[t + "mlp.0.weight"], st[s + "mlp.up_proj.weight"])


def test_the_key_bias_slot_is_used_when_the_checkpoint_has_one():
    """`key_bias: false` is a DINOv3 config choice, not a law. If a checkpoint
    ships a k bias it must be transferred, not zeroed."""
    st = _syn_state(key_bias=True)
    sd, _ = _seed(st)
    b = sd["blocks.0.attn.in_proj_bias"]
    torch.testing.assert_close(b[D:2 * D], st["layer.0.attention.k_proj.bias"])


def test_the_left_at_init_list_is_exactly_pos_and_nothing_may_join_it():
    """⚠️ THE DECLARED LOSS, PINNED. `pos` is left at init because DINOv3 is
    RoPE-only and carries no absolute-position table at all. If this list ever
    grows, a seed that silently stopped transferring something would pass."""
    assert S.TARGET_LEFT_AT_INIT == ("pos",)
    assert set(S.DINOV3_ALLOW_UNMAPPED) == {
        "embeddings.cls_token", "embeddings.register_tokens",
        "embeddings.mask_token"}


# ===========================================================================
# 2. ⭐ the numerical proof: a DINOv3 reference block vs the seeded encoder
# ===========================================================================

class _RefDinoBlock(nn.Module):
    """DINOv3's block, written from its own state-dict layout, RoPE removed.

    ⛔ THIS IS THE ONLY TEST THAT CAN CATCH A WRONG FOLD OR A WRONG qkv ORDER.
    Key-count and shape tests pass for a transposed concat, for a k/q swap, and
    for dropping LayerScale entirely."""

    def __init__(self, st: dict, pre: str, d: int, heads: int):
        super().__init__()
        self.d, self.h, self.hd = d, heads, d // heads
        for k in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                  "attention.q_proj.weight", "attention.k_proj.weight",
                  "attention.v_proj.weight", "attention.q_proj.bias",
                  "attention.v_proj.bias", "attention.o_proj.weight",
                  "attention.o_proj.bias", "layer_scale1.lambda1",
                  "layer_scale2.lambda1", "mlp.up_proj.weight",
                  "mlp.up_proj.bias", "mlp.down_proj.weight",
                  "mlp.down_proj.bias"):
            self.register_buffer(k.replace(".", "_"), st[pre + k].clone())

    def _heads(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.h, self.hd).transpose(1, 2)

    def forward(self, x):
        hgt = torch.nn.functional.layer_norm(
            x, (self.d,), self.norm1_weight, self.norm1_bias, 1e-5)
        q = self._heads(hgt @ self.attention_q_proj_weight.T + self.attention_q_proj_bias)
        k = self._heads(hgt @ self.attention_k_proj_weight.T)      # NO key bias
        v = self._heads(hgt @ self.attention_v_proj_weight.T + self.attention_v_proj_bias)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(x.shape)
        o = o @ self.attention_o_proj_weight.T + self.attention_o_proj_bias
        x = x + self.layer_scale1_lambda1 * o
        h2 = torch.nn.functional.layer_norm(
            x, (self.d,), self.norm2_weight, self.norm2_bias, 1e-5)
        m = torch.nn.functional.gelu(
            h2 @ self.mlp_up_proj_weight.T + self.mlp_up_proj_bias)
        m = m @ self.mlp_down_proj_weight.T + self.mlp_down_proj_bias
        return x + self.layer_scale2_lambda1 * m


def test_the_seeded_encoder_reproduces_a_dinov3_reference_forward():
    st = _syn_state()
    cfg = _enc_cfg()
    sd, _ = _seed(st, cfg)
    enc = ViTEncoder(cfg).eval()
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    assert unexpected == [] and sorted(missing) == ["pos"]
    with torch.no_grad():
        enc.pos.zero_()                    # the one tensor DINOv3 cannot supply
        x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(7))
        got = enc(x)
        # the reference path: same patch embed, no APE, no cls/registers
        t = torch.nn.functional.conv2d(
            x, st["embeddings.patch_embeddings.weight"],
            st["embeddings.patch_embeddings.bias"], stride=P)
        t = t.flatten(2).transpose(1, 2)
        for i in range(L):
            t = _RefDinoBlock(st, f"layer.{i}.", D, H)(t)
        want = torch.nn.functional.layer_norm(
            t, (D,), st["norm.weight"], st["norm.bias"], 1e-5)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)


def test_the_reference_forward_disagrees_when_the_layerscale_is_dropped():
    """⚠️ THE NEGATIVE CONTROL. A guard that cannot fail has measured nothing —
    so the same comparison is run against a seed with LayerScale removed, and it
    MUST disagree. (Same discipline as the discarded-`nn.Linear` control in
    test_v6_anchor_loss.py.)"""
    st = _syn_state()
    cfg = _enc_cfg()
    sd, _ = _seed(st, cfg)
    for i in range(L):
        ls1 = st[f"layer.{i}.layer_scale1.lambda1"]
        ls2 = st[f"layer.{i}.layer_scale2.lambda1"]
        sd[f"blocks.{i}.attn.out_proj.weight"] /= ls1[:, None]
        sd[f"blocks.{i}.attn.out_proj.bias"] /= ls1
        sd[f"blocks.{i}.mlp.2.weight"] /= ls2[:, None]
        sd[f"blocks.{i}.mlp.2.bias"] /= ls2
    enc = ViTEncoder(cfg).eval()
    enc.load_state_dict(sd, strict=False)
    with torch.no_grad():
        enc.pos.zero_()
        x = torch.randn(2, 3, 32, 32, generator=torch.Generator().manual_seed(7))
        got = enc(x)
        t = torch.nn.functional.conv2d(
            x, st["embeddings.patch_embeddings.weight"],
            st["embeddings.patch_embeddings.bias"], stride=P)
        t = t.flatten(2).transpose(1, 2)
        for i in range(L):
            t = _RefDinoBlock(st, f"layer.{i}.", D, H)(t)
        want = torch.nn.functional.layer_norm(
            t, (D,), st["norm.weight"], st["norm.bias"], 1e-5)
    assert not torch.allclose(got, want, rtol=1e-4, atol=1e-5), (
        "the unfolded seed matched the reference — the comparison is not "
        "sensitive to the LayerScale fold and proves nothing")


# ===========================================================================
# 3. the refusals — each one SHOWN TO FIRE, each one naming the offender
# ===========================================================================

def test_a_shape_mismatch_refuses_and_names_the_tensor():
    st = _syn_state()
    st["layer.0.attention.q_proj.weight"] = torch.randn(D, D + 8)
    with pytest.raises(S.SeedError) as e:
        _seed(st)
    m = str(e.value)
    assert "SHAPE MISMATCH" in m
    assert "blocks.0.attn.in_proj_weight" in m, m
    assert "q_proj.weight" in m, m


def test_an_unmapped_tensor_outside_the_allowlist_refuses_and_names_it():
    st = _syn_state()
    st["distillation_head.weight"] = torch.randn(4, D)
    with pytest.raises(S.SeedError) as e:
        _seed(st)
    m = str(e.value)
    assert "NEITHER MAPPED NOR" in m and "distillation_head.weight" in m, m


def test_the_allowlist_can_be_extended_explicitly():
    """The refusal must be a gate, not a wall — otherwise the next DINOv3
    revision makes the tool unusable rather than making a human look."""
    st = _syn_state()
    st["distillation_head.weight"] = torch.randn(4, D)
    sd, rep = _seed(st, allow_unmapped=("distillation_head.weight",))
    assert "distillation_head.weight" in rep["skipped_source_keys"]
    assert rep["n_skipped_allowlist"] == 4          # 3 default + this one


def test_a_hole_in_the_source_refuses_rather_than_seeding_around_it():
    st = _syn_state()
    del st["layer.1.mlp.down_proj.weight"]
    with pytest.raises(S.SeedError) as e:
        _seed(st)
    assert "MISSING TENSORS" in str(e.value)
    assert "layer.1.mlp.down_proj.weight" in str(e.value)


def test_no_fold_layer_scale_refuses_rather_than_dropping_the_gain():
    with pytest.raises(S.SeedError) as e:
        _seed(fold_layer_scale=False)
    assert "layer_scale1.lambda1" in str(e.value)
    assert "DROPPING" in str(e.value)


@pytest.mark.parametrize("kw,flag", [
    (dict(d=64), "--enc-dim"), (dict(depth=4), "--enc-depth"),
    (dict(patch=8), "--patch"),
])
def test_a_geometry_mismatch_refuses_and_names_the_field(kw, flag):
    st = _syn_state(**({"d": kw.get("d", D), "depth": kw.get("depth", L),
                        "mlp": 4 * kw.get("d", D), "patch": kw.get("patch", P)}))
    with pytest.raises(S.SeedError) as e:
        S.assert_target_geometry(_geo(st), _enc_cfg())
    assert "GEOMETRY MISMATCH" in str(e.value) and flag in str(e.value)


def test_a_nine_channel_target_refuses_because_the_patch_embed_is_three():
    st = _syn_state(ch=9)
    with pytest.raises(S.SeedError) as e:
        S.assert_target_geometry(_geo(st), _enc_cfg(ch=9))
    assert "--in-channels 9" in str(e.value)


def test_a_config_json_that_disagrees_with_the_tensors_refuses():
    """⛔ The tensors win. A converter that quietly prefers the metadata is how
    a run gets seeded from a different model than its record names."""
    st = _syn_state()
    with pytest.raises(S.SeedError) as e:
        S.infer_source_geometry(st, {"hidden_size": 999})
    assert "hidden_size" in str(e.value) and "999" in str(e.value)


def test_a_non_dinov3_state_dict_refuses_with_the_layout_it_wanted():
    with pytest.raises(S.SeedError) as e:
        S.infer_source_geometry({"foo.weight": torch.zeros(2, 2)}, {})
    assert "patch_embeddings.weight" in str(e.value)


def test_the_vit5_target_is_refused_by_the_cli():
    with pytest.raises(SystemExit) as e:
        S.main(["--source", "x", "--out", "y", "--target", "vit5"])
    m = str(e.value)
    assert "D1 option B" in m and "lossy" in m, m


def test_a_missing_source_path_refuses_and_never_downloads():
    with pytest.raises(S.SeedError) as e:
        S.load_source("Z:/no/such/dinov3")
    assert "never downloads" in str(e.value)


# ===========================================================================
# 4. the provenance stamp round-trips
# ===========================================================================

def _write_seed(tmp_path, state=None, cfg=None, **kw) -> tuple[Path, dict]:
    state = _syn_state() if state is None else state
    cfg = _enc_cfg() if cfg is None else cfg
    sd, rep = S.build_seed(state, cfg, src_geo=_geo(state), **kw)
    prov = S.make_provenance(
        {"source_path": "SYNTHETIC", "source_sha256": "0" * 64,
         "source_bytes": 0},
        rep, model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        argv=["--synthetic"], stripped_prefix="", encoder_sd=sd)
    p = tmp_path / "seed.pt"
    torch.save({S.SEED_MARKER_KEY: S.SEED_FORMAT_VERSION,
                "encoder": sd, "_provenance": prov}, p)
    return p, prov


def test_the_provenance_stamp_round_trips(tmp_path):
    p, prov = _write_seed(tmp_path)
    back = torch.load(p, map_location="cpu", weights_only=False)
    got = back["_provenance"]
    assert got == prov
    assert json.loads(json.dumps(got)) == got, "the stamp must be JSON-safe"
    # the stamp carries every field the brief requires, and they are not empty
    assert got["format"] == S.SEED_FORMAT
    assert got["source_path"] and got["source_sha256"]
    assert got["dinov3_model_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert got["dinov3_variant"] == "custom"     # the tiny rig is not published
    assert len(got["mapping"]) == 2 + 12 * L + 2
    assert got["n_mapped"] + got["n_left_at_init"] == len(
        ViTEncoder(_enc_cfg()).state_dict())
    assert got["n_skipped_allowlist"] == 3
    assert got["left_at_init_keys"] == ["pos"]
    assert got["target_geometry"]["class"] == "ViTEncoder"
    assert got["target_geometry"]["n_tokens"] == 4
    # ⚠️ RESOLVED, so a square cfg's `image_width=None` cannot masquerade as a
    # geometry disagreement against a run that spells the width out.
    assert got["target_geometry"]["image_width"] == 32
    assert got["layer_scale_folded"] is True
    assert got["declared_losses"] and any(
        "RoPE-only" in x for x in got["declared_losses"])
    # the content hash is over the TENSORS, so it survives the container
    assert got["seed_sha256"] == S._sha256_state(back["encoder"])


def test_the_seed_hash_is_content_addressed_not_container_addressed(tmp_path):
    """⛔ C72: `torch.save`'s bytes are not canonical. The stamp's hash must
    depend on the tensors and NOT on dict order."""
    st = _syn_state()
    sd, _ = _seed(st)
    shuffled = {k: sd[k] for k in reversed(list(sd))}
    assert S._sha256_state(sd) == S._sha256_state(shuffled)
    sd2 = dict(sd)
    sd2["norm.bias"] = sd2["norm.bias"] + 1e-6
    assert S._sha256_state(sd) != S._sha256_state(sd2)


def test_the_printed_table_reports_the_same_counts_as_the_stamp():
    _, rep = _seed()
    txt = S.format_table(rep)
    assert f"MAPPED            : {rep['n_mapped']:4d}" in txt
    assert "LEFT AT INIT" in txt and "['pos']" in txt
    assert "DECLARED LOSSES" in txt


# ===========================================================================
# 5. --init-encoder-from seeds the encoder and NOTHING else
# ===========================================================================

_SMALL_ARGV = [
    "--stage", "S-T", "--out", "unused",
    "--in-channels", "3", "--frame-h", "32", "--frame-w", "32",
    "--patch", "16", "--enc-dim", "32", "--enc-depth", "2", "--enc-heads", "4",
    "--readout-grid", "2", "--readout-dim", "8",
    "--pred-dim", "32", "--pred-depth", "1", "--pred-heads", "2",
    "--window", "2", "--horizons", "1",
    "--d-tac", "32", "--d-str", "16", "--d-goal-embed", "16",
    "--adapter-hidden", "32", "--f-hidden-tac", "32", "--f-hidden-str", "32",
    "--f-blocks", "1", "--sigreg-slices", "8", "--n-candidates", "4",
]


def _args(*extra):
    return build_parser().parse_args(_SMALL_ARGV + list(extra))


def _stack(a, seed: int = 0):
    torch.manual_seed(seed)
    return build_stack_from_args(a)


def test_init_encoder_from_seeds_the_encoder_and_leaves_every_other_module(tmp_path):
    p, _ = _write_seed(tmp_path)
    a = _args("--init-encoder-from", str(p))
    st = _stack(a)
    before = {k: v.detach().clone() for k, v in st.state_dict().items()}
    rep = apply_encoder_seed(a, st)
    after = st.state_dict()

    assert rep["seeded_modules"] == ["encoder"]
    assert rep["left_at_init"] == ["pos"]
    assert rep["dinov3_model_id"] == "facebook/dinov3-vitb16-pretrain-lvd1689m"
    assert rep["encoder_md5_after_seed"]

    # ⛔ TENSOR BY TENSOR over the WHOLE stack — a param-count or a
    # "did the encoder change" check passes for a seed that also stomped the
    # readout, and passes for one that seeded only half the blocks.
    changed, unchanged = [], []
    for k, v in after.items():
        (changed if not torch.equal(before[k], v) else unchanged).append(k)
    seeded = set(torch.load(p, map_location="cpu",
                            weights_only=False)["encoder"])
    assert set(changed) == {"encoder." + k for k in seeded}, (
        f"unexpected changes: {sorted(set(changed) - {'encoder.' + k for k in seeded})}; "
        f"missed: {sorted({'encoder.' + k for k in seeded} - set(changed))}")
    assert "encoder.pos" in unchanged, "pos must stay at ITS OWN init"
    assert any(k.startswith("readout.") for k in unchanged)
    assert any(k.startswith("predictor_op.") for k in unchanged)
    # and by VALUE, not merely "changed"
    for k in seeded:
        torch.testing.assert_close(
            after["encoder." + k],
            torch.load(p, map_location="cpu", weights_only=False)["encoder"][k])


def test_no_flag_means_no_change_at_all(tmp_path):
    a = _args()
    st = _stack(a)
    before = {k: v.detach().clone() for k, v in st.state_dict().items()}
    rep = apply_encoder_seed(a, st)
    assert rep["init_encoder_from"] is None
    for k, v in st.state_dict().items():
        assert torch.equal(before[k], v), k


def test_a_seed_without_a_provenance_stamp_is_refused(tmp_path):
    sd, _ = _seed()
    p = tmp_path / "unstamped.pt"
    torch.save({S.SEED_MARKER_KEY: 1, "encoder": sd}, p)
    with pytest.raises(SystemExit) as e:
        load_encoder_seed(_stack(_args()), p)
    assert "NO PROVENANCE STAMP" in str(e.value)


def test_a_whole_stack_checkpoint_through_the_encoder_flag_is_refused(tmp_path):
    a = _args()
    st = _stack(a)
    p = tmp_path / "whole.pt"
    torch.save({"stack": st.state_dict(), "step": 5}, p)
    with pytest.raises(SystemExit) as e:
        load_encoder_seed(st, p)
    assert "not an encoder seed" in str(e.value)
    assert "--init-from" in str(e.value)


def test_a_geometry_disagreement_at_load_time_is_refused(tmp_path):
    p, _ = _write_seed(tmp_path, cfg=_enc_cfg(d=32, depth=2, heads=4),
                       state=_syn_state(d=32, depth=2, mlp=128))
    a = _args("--init-encoder-from", str(p), "--enc-depth", "1")
    with pytest.raises(SystemExit) as e:
        apply_encoder_seed(a, _stack(a))
    m = str(e.value)
    assert "geometry disagrees" in m and "depth" in m, m


def test_a_seed_that_dropped_a_tensor_is_refused_at_load(tmp_path):
    """⛔⛔ THE SILENT PARTIAL SEED ITSELF. `missing` must EQUAL the stamp's
    declared list, never merely be a superset of it."""
    p, prov = _write_seed(tmp_path)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    del ck["encoder"]["blocks.1.mlp.2.weight"]
    torch.save(ck, p)
    with pytest.raises(SystemExit) as e:
        load_encoder_seed(_stack(_args()), p)
    m = str(e.value)
    assert "RANDOM INIT" in m and "blocks.1.mlp.2.weight" in m, m


def test_a_class_mismatch_refuses_and_points_at_vit5(tmp_path):
    p, _ = _write_seed(tmp_path)
    a = _args("--init-encoder-from", str(p), "--vit5-encoder",
              "--n-registers", "4")
    with pytest.raises(SystemExit) as e:
        apply_encoder_seed(a, _stack(a))
    m = str(e.value)
    assert "ViT5Encoder" in m and "D1 option B" in m, m


def test_init_from_and_init_encoder_from_together_are_refused(tmp_path):
    p, _ = _write_seed(tmp_path)
    a = _args("--init-encoder-from", str(p), "--init-from", str(p))
    with pytest.raises(SystemExit) as e:
        apply_encoder_seed(a, _stack(a))
    assert "AMBIGUOUS" in str(e.value)


# ===========================================================================
# 6. the optimizer split, the warmup, and the anchor refusal
# ===========================================================================

def test_the_optimizer_has_exactly_two_groups_with_every_param_in_exactly_one():
    a = _args("--trunk-lr-scale", "0.1", "--trunk-lr-warmup-steps", "2000")
    st = _stack(a)
    trainable = [p for p in st.parameters() if p.requires_grad]
    opt, rep = build_trunk_optimizer(a, st, trainable)

    assert rep["trunk_lr_split"] is True and rep["n_groups"] == 2
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["tanitad_group"] == "encoder"
    assert opt.param_groups[1]["tanitad_group"] == "rest"

    ids = [[id(p) for p in g["params"]] for g in opt.param_groups]
    flat = ids[0] + ids[1]
    assert len(flat) == len(set(flat)), "a parameter is in BOTH groups"
    assert set(flat) == {id(p) for p in trainable}, "a parameter is in NEITHER"
    # the trunk group is EXACTLY V6Stack.group_of == "encoder"
    want = {id(p) for n, p in st.named_parameters()
            if st.group_of(n) == "encoder" and p.requires_grad}
    assert set(ids[0]) == want
    assert rep["n_trunk_numel"] == sum(p.numel() for p in opt.param_groups[0]["params"])
    # ⚠️ the readout is deliberately NOT in the trunk group
    ro = {id(p) for n, p in st.named_parameters() if n.startswith("readout.")}
    assert ro and ro.issubset(set(ids[1]))


def test_the_multiplier_is_applied_once_not_squared():
    """⚠️ Both groups are built at `a.lr`; the scale lives in the scheduler's
    lambda. If it were ALSO in `initial_lr` the trunk would run at scale**2."""
    a = _args("--trunk-lr-scale", "0.1", "--lr", "1e-4", "--steps", "1000")
    st = _stack(a)
    opt, _ = build_trunk_optimizer(a, st, [p for p in st.parameters()
                                           if p.requires_grad])
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    sched = build_lr_scheduler(a, opt)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4 * 0.1, rel=1e-9)
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-4, rel=1e-9)


def test_the_warmup_makes_the_trunk_lr_zero_before_its_step_and_the_value_after():
    warm, lr, scale, steps = 50, 1e-4, 0.1, 1000
    a = _args("--trunk-lr-scale", str(scale), "--trunk-lr-warmup-steps",
              str(warm), "--lr", str(lr), "--steps", str(steps))
    st = _stack(a)
    opt, _ = build_trunk_optimizer(a, st, [p for p in st.parameters()
                                           if p.requires_grad])
    sched = build_lr_scheduler(a, opt)
    seen = []
    with _quiet_sched():
        for step in range(warm + 5):
            seen.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
            sched.step()
    for s in range(warm):
        assert seen[s][0] == 0.0, f"trunk LR must be EXACTLY 0 at step {s}"
        assert seen[s][1] > 0.0, f"the rest must keep training at step {s}"
    cos = (1 + math.cos(math.pi * warm / steps)) / 2
    assert seen[warm][0] == pytest.approx(lr * scale * cos, rel=1e-9)
    assert seen[warm][1] == pytest.approx(lr * cos, rel=1e-9)
    assert trunk_lr_factor(warm - 1, a) == 0.0
    assert trunk_lr_factor(warm, a) == scale


def test_a_frozen_trunk_plus_a_trunk_schedule_refuses():
    a = _args("--trunk-lr-scale", "0.1", "--freeze-encoder")
    st = _stack(a)
    for p in st.encoder.parameters():
        p.requires_grad_(False)
    with pytest.raises(SystemExit) as e:
        build_trunk_optimizer(a, st, [p for p in st.parameters()
                                      if p.requires_grad])
    assert "NO trainable parameter" in str(e.value)


def test_the_trunk_anchor_flag_refuses_when_set_rather_than_running_inert():
    assert assert_trunk_anchor_unwired(_args()) == 0.0
    with pytest.raises(SystemExit) as e:
        assert_trunk_anchor_unwired(_args("--w-trunk-anchor", "1.0"))
    m = str(e.value)
    assert "DECLARED BUT NOT WIRED" in m
    assert "vitl16" in m and "readout" in m.lower()


def test_o7s_teacher_cannot_be_reused_for_the_trunk_anchor():
    """The brief asked which; this pins the three reasons AGAINST SOURCE so the
    answer cannot rot into prose.

    ⚠️ Evidence class MEASURED (ours; source read of train_v6_staged.py)."""
    # (a) wrong network: O7's teacher is ViT-L/16, the seed is ViT-B/16
    assert O7_DEFAULT_MODEL == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert "vitb16" not in O7_DEFAULT_MODEL
    # (b) wrong granularity: O7.target returns READOUT CELLS, not patch tokens
    assert "n_cells" in O7Distill.__init__.__code__.co_varnames
    assert "cells" in (O7Distill.target.__doc__ or "")
    # (c) it does not exist at all on the v7f launch line (--w-o7-distill 0)
    src = _SRC.read_text(encoding="utf-8")
    assert 'if float(getattr(a, "w_o7_distill", 0.0)) > 0:' in src


# ===========================================================================
# 7. ⭐ THE LOAD-BEARING TEST — at the defaults, nothing moved
# ===========================================================================

def test_at_the_defaults_the_optimizer_is_the_incumbent_single_group():
    """⛔ NOT "two equal groups". `load_resume` calls `opt.load_state_dict`,
    which REFUSES a different param-group count — an unconditional split would
    have made every existing checkpoint unresumable while every arithmetic test
    still passed."""
    a = _args()
    assert trunk_lr_split_active(a) is False
    st = _stack(a)
    trainable = [p for p in st.parameters() if p.requires_grad]
    opt, rep = build_trunk_optimizer(a, st, trainable)
    assert rep["trunk_lr_split"] is False and rep["n_groups"] == 1
    assert len(opt.param_groups) == 1
    ref = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
    assert len(opt.param_groups[0]["params"]) == len(ref.param_groups[0]["params"])
    for k in ("lr", "weight_decay", "betas", "eps", "amsgrad"):
        assert opt.param_groups[0][k] == ref.param_groups[0][k], k
    # the state dict shape is what a resume compares
    assert (len(opt.state_dict()["param_groups"])
            == len(ref.state_dict()["param_groups"]) == 1)
    assert (opt.state_dict()["param_groups"][0]["params"]
            == ref.state_dict()["param_groups"][0]["params"])


def test_at_the_defaults_the_scheduler_is_the_incumbent_cosine_object():
    a = _args("--steps", "1000")
    st = _stack(a)
    opt, _ = build_trunk_optimizer(a, st, [p for p in st.parameters()
                                           if p.requires_grad])
    sched = build_lr_scheduler(a, opt)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == a.steps and sched.eta_min == 0


def test_the_lambda_schedule_reproduces_the_incumbent_cosine_at_scale_one():
    """The split path leaves `CosineAnnealingLR` for a `LambdaLR` closed form
    (the cosine's recursive `get_lr` cannot be safely overwritten per group).
    The two must agree, or the split silently changes the non-trunk schedule."""
    steps = 500
    a_ref = _args("--steps", str(steps), "--lr", "1e-4")
    a_new = _args("--steps", str(steps), "--lr", "1e-4",
                  "--trunk-lr-warmup-steps", "1")     # scale stays 1.0
    st = _stack(a_ref)
    tr = [p for p in st.parameters() if p.requires_grad]
    o_ref, _ = build_trunk_optimizer(a_ref, st, tr)
    s_ref = build_lr_scheduler(a_ref, o_ref)
    o_new, _ = build_trunk_optimizer(a_new, st, tr)
    s_new = build_lr_scheduler(a_new, o_new)
    assert isinstance(s_new, torch.optim.lr_scheduler.LambdaLR)
    with _quiet_sched():
        for step in range(steps + 1):
            if step >= 1:                              # past the 1-step warmup
                assert o_new.param_groups[0]["lr"] == pytest.approx(
                    o_ref.param_groups[0]["lr"], rel=1e-9, abs=1e-15), step
            assert o_new.param_groups[1]["lr"] == pytest.approx(
                o_ref.param_groups[0]["lr"], rel=1e-9, abs=1e-15), step
            s_ref.step()
            s_new.step()


def test_the_new_flags_all_default_to_todays_behaviour():
    a = _args()
    assert a.init_encoder_from is None
    assert a.trunk_lr_scale == 1.0
    assert a.trunk_lr_warmup_steps == 0
    assert a.w_trunk_anchor == 0.0


def test_the_prereg_spelling_of_the_flag_still_parses():
    """`PREREG_V7F.md` §9's launch line writes `--enc-init-from`. The canonical
    name is `--init-encoder-from` (a sibling of `--init-from`; every other
    `--enc-*` flag is a GEOMETRY knob), and the pre-registered spelling is an
    alias so that launch line runs verbatim."""
    a = _args("--enc-init-from", "SEED.pt")
    assert a.init_encoder_from == "SEED.pt"
