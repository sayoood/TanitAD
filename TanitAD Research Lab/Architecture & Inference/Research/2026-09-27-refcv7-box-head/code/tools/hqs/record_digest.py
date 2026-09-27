"""Record MAIN_LEARNED_DIGEST on the MAIN variant (no HQS code): the same computation as test_refcv7_hqs._learned_digest
(copied verbatim in substance; the test module imports the HQS module, which the MAIN tree does not have)."""
import hashlib
import sys

import torch

from tanitad.data.semantic_map_gt import CART_SHAPE
from tanitad.models import refcv6_perception_branch as PB


def _geom(b, z):
    return torch.zeros(b, z, CART_SHAPE[0], CART_SHAPE[1], 2), torch.ones(b, z, CART_SHAPE[0], CART_SHAPE[1],
                                                                         dtype=torch.bool)


def _targets(B=2, A=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    box = torch.zeros(B, A, 4)
    box[..., 0] = torch.rand(B, A, generator=g) * 40 + 8
    box[..., 1] = torch.rand(B, A, generator=g) * 16 - 8
    box[..., 2], box[..., 3] = 4.5, 1.9
    v = torch.ones(B, A, dtype=torch.bool)
    tgt = {"box": box, "yaw": torch.zeros(B, A), "cls": torch.zeros(B, A, dtype=torch.long), "valid": v,
           "occ": torch.full((B, A), -1.0), "rates": torch.zeros(B, A, 3),
           "rates_mask": torch.zeros(B, A, dtype=torch.bool), "cz": torch.full((B, A), 0.8),
           "h": torch.full((B, A), 1.6), "zh_mask": v.clone()}
    nv = torch.full((B, A), 900, dtype=torch.int32)
    nv[:, 0] = 20
    vis = {"n_full": torch.full((B, A), 1000, dtype=torch.int32), "n_vis": nv, "known": v.clone()}
    return tgt, vis


def learned_digest(**cfg_kw) -> str:
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(0)
        cfg = PB.PerceptionBranchConfig(
            w_map=1.0, w_box3d=1.0, d_bev=8,
            bev_cfg=PB.BEVEncoderConfig(d_in=8, d_model=8, d_out=8, dilations=(1,), norm_groups=2),
            n_queries=6, d_model=16, bev_tokens_hw=(2, 2), enforce_param_band=False,
            presence_loss="focal", presence_prior=0.01, deep_supervision=True, vis1=True, **cfg_kw)
        br = PB.PerceptionBranch(cfg, d_image=8, image_hw=(4, 8))
        g = torch.Generator().manual_seed(1)
        f = torch.randn(2, 8, 4, 8, generator=g)
        gg, vv = _geom(2, len(cfg.heights_m))
        gg = torch.randn(gg.shape, generator=g) * 0.5
        out = br(f, gg, vv)
        tgt, vis = _targets()
        row = PB.box3d_loss_row(out["box_slots"], tgt, presence_loss="focal", vis1=True, vis=vis)
        row["loss"].backward()
        h = hashlib.sha256()
        for k in sorted(out["box_slots"]):
            v = out["box_slots"][k]
            if torch.is_tensor(v):
                h.update(k.encode() + v.detach().contiguous().numpy().tobytes())
        h.update(row["loss"].detach().numpy().tobytes())
        for n, p in sorted(br.named_parameters()):
            if p.grad is not None:
                h.update(n.encode() + p.grad.contiguous().numpy().tobytes())
        return h.hexdigest()
    finally:
        torch.set_num_threads(nt)


if __name__ == "__main__":
    import tanitad
    print("tanitad from", tanitad.__file__, "torch", torch.__version__, sys.platform)
    print(learned_digest())
    print(learned_digest())
