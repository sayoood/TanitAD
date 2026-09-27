"""+R6 (SPEC_REFCV7 §15.5 A10.1) -- denoising queries for the box head: default OFF and bit-identical, the per-group
attention mask (with its red arm), finite losses with gradient at every layer, negatives taught "no object", and the
trainer / G-DVB wiring. BANKED UNLANDED: runs only in a tree that carries apply_r6_edits.py.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import box3d_head as B3         # noqa: E402
from tanitad.models import slot_denoise as DN       # noqa: E402


def _dec():
    torch.manual_seed(0)
    d = B3.Box3DSlotDecoder(16, 8, n_queries=10, d_model=32, depth=3, n_heads=4, enforce_band=False,
                            presence_prior=0.01)
    d.deep_supervision = True
    return d


def _embed(d):
    return DN.DenoiseQueryEmbed(32, 10, x_range=60.0, y_range=16.0, z_range=4.0)


def _tgt(counts=(3, 1)):
    B, A = len(counts), max(counts) + 1
    box = torch.zeros(B, A, 4)
    valid = torch.zeros(B, A, dtype=torch.bool)
    g = torch.Generator().manual_seed(2)
    for b, c in enumerate(counts):
        box[b, :c, 0] = torch.rand(c, generator=g) * 50 + 5
        box[b, :c, 1] = torch.rand(c, generator=g) * 20 - 10
        box[b, :c, 2], box[b, :c, 3] = 4.5, 1.9
        valid[b, :c] = True
    return {"box": box, "yaw": torch.zeros(B, A), "cls": torch.zeros(B, A, dtype=torch.long), "valid": valid,
            "cz": torch.full((B, A), 0.8), "h": torch.full((B, A), 1.6), "zh_mask": valid.clone()}


def test_the_arm_literals():
    assert (DN.DN_GROUPS_ARM, DN.DN_BOX_NOISE, DN.DN_LABEL_NOISE, DN.DN_LOSS_W) == (5, 0.4, 0.2, 1.0)


def test_the_mask_is_block_diagonal_per_group():
    m = DN.dn_attention_mask(3, 2)
    assert m.shape == (12, 12)
    assert not m[0:4, 0:4].any() and m[0:4, 4:].all() and not m[4:8, 4:8].any() and m[8:, :8].all()


def test_dn_queries_layout_positives_negatives_and_padding():
    d = _dec()
    t = _tgt((3, 1))
    q = DN.make_dn_queries(_embed(d), t, groups=2, gen=torch.Generator().manual_seed(0))
    assert q["A_max"] == 3 and q["q"].shape == (2, 2 * 2 * 3, 32)
    assert q["is_pos"][0].tolist() == [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3
    assert q["pad"][1].tolist() == [False, True, True] * 4                # element 1 has ONE positive
    assert DN.make_dn_queries(_embed(d), _tgt((0, 0)), groups=2, gen=torch.Generator())["q"] is None


def _group_outputs(d, mem, q, mask_on=True):
    if mask_on:
        return DN.dn_forward(d, mem, q)[-1]["presence_logit"]
    saved = DN.dn_attention_mask
    DN.dn_attention_mask = lambda g, a, device=None: torch.zeros(2 * g * a, 2 * g * a, dtype=torch.bool, device=device)
    try:
        return DN.dn_forward(d, mem, q)[-1]["presence_logit"]
    finally:
        DN.dn_attention_mask = saved


def test_groups_are_isolated_and_RED_ARM_without_the_mask_they_leak():
    d = _dec()
    mem = torch.randn(2, 8, 16)
    t = _tgt((3, 2))
    q = DN.make_dn_queries(_embed(d), t, groups=2, gen=torch.Generator().manual_seed(0))
    with torch.no_grad():
        a = _group_outputs(d, mem, q)
        q2 = dict(q)
        q2["q"] = q["q"].clone()
        # perturb GROUP 1 only -- with NOISE: a constant shift is erased by the pre-norm LayerNorm
        q2["q"][:, 6:] += torch.randn(q2["q"][:, 6:].shape, generator=torch.Generator().manual_seed(3)) * 2.0
        b = _group_outputs(d, mem, q2)
        assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6), "group 0 saw group 1"
        a0 = _group_outputs(d, mem, q, mask_on=False)
        b0 = _group_outputs(d, mem, q2, mask_on=False)
        assert not torch.allclose(a0[:, :6], b0[:, :6], atol=1e-4), "the red arm no longer leaks -- vacuous test"


def test_the_matching_pass_is_untouched_by_construction():
    d = _dec()
    mem = torch.randn(2, 8, 16)
    with torch.no_grad():
        before = d(mem)["presence_logit"].clone()
        q = DN.make_dn_queries(_embed(d), _tgt(), groups=3, gen=torch.Generator().manual_seed(0))
        DN.dn_forward(d, mem, q)
        after = d(mem)["presence_logit"]
    assert torch.equal(before, after)


def test_losses_finite_at_every_layer_with_gradient_and_negatives_pushed_down():
    d = _dec()
    emb = _embed(d)
    mem = torch.randn(2, 8, 16)
    t = _tgt((3, 1))
    q = DN.make_dn_queries(emb, t, groups=2, gen=torch.Generator().manual_seed(0))
    outs = DN.dn_forward(d, mem, q)
    outs[-1]["presence_logit"].retain_grad()
    r = DN.dn_losses(outs, t, q)
    assert r["parts"]["dn_n_layers"] == 3.0 and r["parts"]["dn_n_pos"] == 2 * 4 and r["parts"]["dn_n_neg"] == 2 * 4
    for k in ("dn_layer0", "dn_layer1", "dn_layer2", "dn_presence", "dn_centre"):
        assert torch.isfinite(torch.as_tensor(r["parts"][k])), k
    r["total"].backward()
    for name, p in (("queries-free decoder layer0", d.blocks.layers[0].linear1.weight), ("head", d.head.weight),
                    ("embed", emb.mlp[0].weight)):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name
    g = outs[-1]["presence_logit"].grad
    neg = ~q["is_pos"] & ~q["pad"]
    assert (g[neg] > 0).all(), "a negative's presence must be pushed DOWN (positive gradient on its logit)"
    assert (g[q["pad"]] == 0).all(), "a padded DN query must carry no loss"


def test_an_element_with_no_positive_does_not_poison_the_batch_with_nan():
    d = _dec()
    t = _tgt((2, 0))
    q = DN.make_dn_queries(_embed(d), t, groups=2, gen=torch.Generator().manual_seed(0))
    r = DN.dn_losses(DN.dn_forward(d, torch.randn(2, 8, 16), q), t, q)
    assert torch.isfinite(r["total"])


def test_the_dn_task_is_learnable():
    d = _dec()
    emb = _embed(d)
    mem = torch.randn(4, 8, 16)
    t = _tgt((3, 2, 3, 1))
    opt = torch.optim.AdamW(list(d.parameters()) + list(emb.parameters()), lr=2e-3)
    gen = torch.Generator().manual_seed(0)
    first = None
    for _ in range(150):
        opt.zero_grad()
        q = DN.make_dn_queries(emb, t, groups=2, gen=gen)
        L = DN.dn_losses(DN.dn_forward(d, mem, q), t, q)["total"]
        L.backward()
        opt.step()
        first = float(L) if first is None else first
    assert float(L) < 0.5 * first


# --------------------------------------------------------------------------------------------------------- #
# the trainer / G-DVB wiring (needs the R6 patch in this tree)                                              #
# --------------------------------------------------------------------------------------------------------- #
def _trainer():
    spec = importlib.util.spec_from_file_location("refc_v3_train_for_r6", str(ROOT / "scripts" / "refc_v3_train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_flag_defaults_off_and_dead_uses_refuse():
    T = _trainer()
    a = T.build_parser().parse_args(["--arm", "hier", "--out", "X"])
    assert a.slot_dn_groups == 0
    with pytest.raises(SystemExit, match="BOX head only"):
        T._pin_slot_refine(None, T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--slot-dn-groups", "5"]))
    with pytest.raises(SystemExit, match=">= 0"):
        T._pin_slot_refine(None, T.build_parser().parse_args(["--arm", "hier", "--out", "X", "--slot-dn-groups", "-1"]))


def test_the_branch_builds_the_embedder_only_when_asked_and_gdvb_reads_it():
    from tanitad.models import refcv6_perception_branch as PB
    from tanitad.train import declared_vs_built as dvb
    import types
    off = PB.PerceptionBranch(PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=4, d_model=32,
                                                        enforce_param_band=False), d_image=8, image_hw=(2, 4))
    on = PB.PerceptionBranch(PB.PerceptionBranchConfig(w_map=0.0, w_box3d=1.0, n_queries=4, d_model=32,
                                                       enforce_param_band=False, dn_groups=5), d_image=8,
                             image_hw=(2, 4))
    assert off.box_dn is None and on.box_dn is not None
    # the dn-OFF breakdown keeps the tip's key set (refcv6/refcv7 stamps are compared key for key); the DN key
    # appears only when the embedder is built, "total" stays last and counts it
    b_off, b_on = off.param_breakdown(), on.param_breakdown()
    assert "box_dn" not in b_off and list(b_off)[-1] == "total"
    assert b_on["box_dn"] > 0 and list(b_on)[-1] == "total"
    assert set(b_on) - set(b_off) == {"box_dn"}
    assert b_on["total"] - b_off["total"] == b_on["box_dn"]
    # RED arm: the first R6 build added the key unconditionally, so a dn-off stamp gained a key it never had
    assert set({**b_off, "box_dn": 0}) != set(b_off)
    on.train()
    assert "box_memory" in on(torch.randn(1, 8, 2, 4))
    on.eval()
    assert "box_memory" not in on(torch.randn(1, 8, 2, 4))              # inference unchanged
    m = types.SimpleNamespace(_perception=on)
    assert dvb.REGISTRY["slot_dn_groups"].check(m, types.SimpleNamespace(slot_dn_groups=5)) == []
    bad = dvb.REGISTRY["slot_dn_groups"].check(m, types.SimpleNamespace(slot_dn_groups=0))
    assert bad and bad[0].lever == "--slot-dn-groups"


def test_presence_w0_zeroes_the_DN_presence_term_too(monkeypatch):
    """The harness's must-fail presence_w0 arm patches slot_presence.FOCAL_PRESENCE_W; the DN presence term reads the
    SAME constant, so a DN arm cannot be rescued by its own presence term (RED arm: a literal 2.0 would keep it)."""
    from tanitad.models import slot_presence as SP
    d = _dec()
    t = _tgt((3, 1))
    q = DN.make_dn_queries(_embed(d), t, groups=2, gen=torch.Generator().manual_seed(0))
    outs = DN.dn_forward(d, torch.randn(2, 8, 16), q)
    for o in outs:
        o["presence_logit"].retain_grad()
    monkeypatch.setattr(SP, "FOCAL_PRESENCE_W", 0.0)
    DN.dn_losses(outs, t, q)["total"].backward()
    assert all(float(o["presence_logit"].grad.abs().sum()) == 0.0 for o in outs)
