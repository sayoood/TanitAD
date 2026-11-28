"""Is our rotary position encoding DINOv3's, or merely A rotary encoding?

⛔ BANKED FROM THE ADVERSARIAL FIX-VERIFICATION REVIEW, 2026-09-20, because it is the only
instrument in the programme that can tell a correct rotary encoding from a wrong one. Our first
implementation was self-consistent, passed every check we had, and was MEASURABLY WORSE THAN NO
ENCODING AT ALL: against the released reference on the real ViT-S checkpoint, no rope gave relative
L2 0.585 / cosine 0.834 while ours gave 0.760 / 0.688.

⭐ WHY OUR OWN ARCHITECTURE TEST COULD NOT SEE IT. `diag_architecture.py`'s D7 arm asserts that
the random frozen table is gone, that the rotation changes the vectors, and that it preserves the
norm. ALL THREE PASS FOR ANY ROTARY ENCODING, including a random one. A check that cannot
distinguish the right answer from a wrong one of the same SHAPE is not a check.

The controls that make this instrument trustworthy: a SELF arm that must read 0.00000, and a JITTER
arm that must read a small non-zero (0.00005) to prove the comparison is sensitive at all.
"""

import math
import sys
import torch

sys.path.insert(0, r"D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe")
import model as M                                             # noqa: E402
from load_dinov3 import load_state, map_into_backbone         # noqa: E402

torch.manual_seed(0)


def dinov3_rope(gh, gw, dh, base=100.0, shift=0.0):
    """The released formulation, verbatim in structure."""
    ch = (torch.arange(0.5, gh, dtype=torch.float32) + shift) / gh
    cw = (torch.arange(0.5, gw, dtype=torch.float32) + shift) / gw
    coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1)     # [gh,gw,2]
    coords = coords.flatten(0, 1)                                          # [N,2]
    coords = 2.0 * coords - 1.0
    periods = base ** (2 * torch.arange(dh // 4, dtype=torch.float32) / (dh // 2))
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]     # [N,2,dh//4]
    angles = angles.flatten(1, 2)                                          # [N,dh//2]
    angles = angles.tile(2)                                                # [N,dh]
    return angles.cos(), angles.sin()


def dinov3_apply(t, cos, sin, offset):
    n = cos.shape[0]
    seg = t[..., offset:offset + n, :]
    x1, x2 = seg.chunk(2, dim=-1)
    rot = torch.cat((-x2, x1), dim=-1)
    seg = seg * cos + rot * sin
    return torch.cat([t[..., :offset, :], seg, t[..., offset + n:, :]], dim=-2)


def forward_with(bb, img, mode, shift=0.0):
    """Run the trunk with a chosen rope. `mode` in {ref, ours, none}."""
    B = img.shape[0]
    pe = bb.patch_embed(img)
    gh, gw = pe.shape[-2], pe.shape[-1]
    x = pe.flatten(2).transpose(1, 2)
    x = torch.cat([bb.cls_token.expand(B, -1, -1), bb.reg_token.expand(B, -1, -1), x], dim=1)
    off = 1 + bb.N_DINOV3_REG
    dh = bb.cfg.width // bb.cfg.heads
    if mode == "ref":
        cos, sin = dinov3_rope(gh, gw, dh, shift=shift)
        app = dinov3_apply
    elif mode == "ours":
        cos, sin = M.build_axial_rope(gh, gw, dh)
        app = M.apply_rope
    else:
        cos = sin = None
        app = None

    for b in bb.blocks:
        h = b.n1(x)
        B_, N_, C_ = h.shape
        a = b.attn
        q = a.q(h).view(B_, N_, a.h, a.dh).transpose(1, 2)
        k = a.k(h).view(B_, N_, a.h, a.dh).transpose(1, 2)
        v = a.v(h).view(B_, N_, a.h, a.dh).transpose(1, 2)
        if app is not None:
            q = app(q, cos, sin, off)
            k = app(k, cos, sin, off)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + b.gamma_1 * a.proj(o.transpose(1, 2).reshape(B_, N_, C_))
        x = x + b.gamma_2 * b.mlp(b.n2(x))
    x = bb.norm(x)
    return x[:, off:]


def rel(a, b):
    return float((a - b).norm() / b.norm())


def cos_sim(a, b):
    return float(torch.nn.functional.cosine_similarity(
        a.reshape(-1), b.reshape(-1), dim=0))


ROPE_TOL = 1e-5


def main():
    cfg = M.REFeConfig.for_backbone("vits16")
    cfg.img_h, cfg.img_w = 128, 256
    net = M.REFe(cfg).eval()
    bb = net.backbone
    sd = load_state("D:/Projects/TanitAD/data/backbones/dinov3-vits16")
    loaded, missing, unused = map_into_backbone(bb, sd)
    print(f"checkpoint loaded: {len(loaded)} target tensors, {len(unused)} unconsumed, "
          f"{len(missing)} target tensors missing")
    assert not unused and not missing, "refusing to measure on a partial load"

    gh, gw = cfg.img_h // cfg.patch, cfg.img_w // cfg.patch
    dh = cfg.width // cfg.heads
    print(f"grid {gh}x{gw} = {gh*gw} patches, head dim {dh}")

    # --- the angle ladders themselves ------------------------------------------------
    rc, rs = dinov3_rope(gh, gw, dh)
    oc, os_ = M.build_axial_rope(gh, gw, dh)
    ra = torch.atan2(rs, rc)
    oa = torch.atan2(os_, oc)
    print("\nANGLE SPAN (highest-frequency channel, before wrapping):")
    # rebuild raw (unwrapped) angles for both, to show the scale error
    ch = torch.arange(0.5, gh, dtype=torch.float32) / gh
    cw = torch.arange(0.5, gw, dtype=torch.float32) / gw
    ref_y = 2 * math.pi * (2 * ch - 1) / 1.0
    ref_x = 2 * math.pi * (2 * cw - 1) / 1.0
    # ⛔ THIS ROW USED TO BE A HARDCODED `arange`, i.e. it DISPLAYED THE REMOVED DEFECT as if
    # it were a measurement of the current code. A probe that does not call the thing it reports
    # on cannot detect a regression in it -- it is a screenshot, not an instrument. Derive the
    # spans from the LIVE build_axial_rope instead.
    _c, _s = M.build_axial_rope(gh, gw, dh)
    _ang = torch.atan2(_s, _c)
    ours_y = _ang[:, 0]
    ours_x = _ang[:, dh // 4]
    print(f"  reference  y in [{ref_y.min():+.3f}, {ref_y.max():+.3f}]  "
          f"x in [{ref_x.min():+.3f}, {ref_x.max():+.3f}]   (span {float(ref_y.max()-ref_y.min()):.3f} / "
          f"{float(ref_x.max()-ref_x.min()):.3f})")
    print(f"  REFe       y in [{ours_y.min():+.3f}, {ours_y.max():+.3f}]  "
          f"x in [{ours_x.min():+.3f}, {ours_x.max():+.3f}]   (span {float(ours_y.max()-ours_y.min()):.3f} / "
          f"{float(ours_x.max()-ours_x.min()):.3f})")
    print(f"  cos table  mean |ref - ours| = {float((rc-oc).abs().mean()):.4f}  "
          f"(a table of cosines lives in [-1,1])")
    print(f"  ANGLE LAYOUT: reference first half == second half? "
          f"{bool(torch.allclose(ra[:, :dh//2], ra[:, dh//2:], atol=1e-5))}    "
          f"REFe channel 2j == 2j+1? {bool(torch.allclose(oa[:, 0::2], oa[:, 1::2], atol=1e-5))}")

    # --- feature-level divergence on the real frozen trunk ---------------------------
    img = torch.randn(1, 3, cfg.img_h, cfg.img_w)
    with torch.no_grad():
        f_ref = forward_with(bb, img, "ref")
        f_self = forward_with(bb, img, "ref")
        f_jit = forward_with(bb, img, "ref", shift=1e-3)
        f_none = forward_with(bb, img, "none")
        f_ours = forward_with(bb, img, "ours")
    print("\nPATCH-TOKEN FEATURES of the FROZEN PRETRAINED TRUNK, vs the reference encoding:")
    for nm, f in (("SELF   (control, must be 0)", f_self),
                  ("JITTER (control, must be small)", f_jit),
                  ("NONE   (no rope at all)", f_none),
                  ("OURS   (REFe build_axial_rope)", f_ours)):
        print(f"  {nm:34s} rel L2 {rel(f, f_ref):8.5f}   cos {cos_sim(f, f_ref):8.5f}")
    print("\nAND THE SAME QUESTION ON THE ATTENTION LOGITS OF BLOCK 0 "
          "(where the encoding enters):")
    with torch.no_grad():
        pe = bb.patch_embed(img)
        x = pe.flatten(2).transpose(1, 2)
        x = torch.cat([bb.cls_token, bb.reg_token, x], dim=1)
        h = bb.blocks[0].n1(x)
        a = bb.blocks[0].attn
        B_, N_, C_ = h.shape
        q0 = a.q(h).view(B_, N_, a.h, a.dh).transpose(1, 2)
        k0 = a.k(h).view(B_, N_, a.h, a.dh).transpose(1, 2)
        off = 1 + bb.N_DINOV3_REG
        lg = {}
        lg["ref"] = (dinov3_apply(q0, rc, rs, off) @ dinov3_apply(k0, rc, rs, off).transpose(-1, -2))
        lg["ours"] = (M.apply_rope(q0, oc, os_, off) @ M.apply_rope(k0, oc, os_, off).transpose(-1, -2))
        lg["none"] = q0 @ k0.transpose(-1, -2)
    for nm in ("ours", "none"):
        print(f"  {nm:34s} rel L2 {rel(lg[nm], lg['ref']):8.5f}   cos {cos_sim(lg[nm], lg['ref']):8.5f}")

    # ⛔ THIS PROBE USED TO HAVE NO ASSERTIONS, NO VERDICT TOKEN, AND `main()` RETURNED None -- so a
    # REGRESSION WOULD PRINT 0.760 AND EXIT 0. It established the rope's correctness twice and
    # could not have detected it breaking. An instrument without a failing branch is a report.
    checks = {
        "OURS matches the reference on patch features":
            rel(f_ours, f_ref) < ROPE_TOL,
        "OURS matches the reference on block-0 attention logits":
            rel(lg["ours"], lg["ref"]) < ROPE_TOL,
        "CONTROL the SELF comparison is exactly 0": rel(f_self, f_ref) < 1e-9,
        "CONTROL removing rope DOES move the features (the probe is sensitive)":
            rel(f_none, f_ref) > 0.05,
        # ⚠️ COMPARE LIKE WITH LIKE. My first version of this arm compared OUR angles -- recovered
        # via atan2 and therefore WRAPPED to [-pi, pi] -- against the reference's UNWRAPPED ladder,
        # which spans +-5.5. It can never match, and it failed on code that is bit-identical to the
        # reference on both features and attention logits. Compare the cos/sin TABLES instead:
        # they are what the rotation actually consumes, and they are wrap-invariant by definition.
        "CONTROL our rotation table equals the reference's":
            float((oc - rc).abs().max()) < 1e-5 and float((os_ - rs).abs().max()) < 1e-5,
    }
    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    good = all(checks.values())
    print("")
    print("ROPE_MATCHES_REFERENCE" if good else "ROPE_DIVERGES_FROM_REFERENCE")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
