"""Local check of gbo_diagnose.Probe / summarise_step on a toy decoder (no Thor data): the hooks fire, every layer's
presence targets are traced to that layer's own matching, gradients land, and a deliberately BROKEN presence_term
(ignore weight NOT restored on matched slots) is caught by the effective check (matched slots with zero gradient)."""
import sys
import types
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gbo_diagnose as D  # noqa: E402
from tanitad.models import agent_slots as AS  # noqa: E402
from tanitad.models import box3d_head as B3  # noqa: E402
from tanitad.models import slot_presence as SP  # noqa: E402


def toy(seed=0, A=3, Q=12, B=2):
    g = torch.Generator().manual_seed(seed)
    box = torch.zeros(B, A, 4)
    box[..., 0] = torch.rand(B, A, generator=g) * 30 + 5
    box[..., 1] = torch.rand(B, A, generator=g) * 10 - 5
    box[..., 2], box[..., 3] = 4.5, 1.9
    v = torch.ones(B, A, dtype=torch.bool)
    tgt = {"box": box, "yaw": torch.zeros(B, A), "cls": torch.zeros(B, A, dtype=torch.long), "valid": v,
           "occ": torch.full((B, A), -1.0), "rates": torch.zeros(B, A, 3),
           "rates_mask": torch.zeros(B, A, dtype=torch.bool), "cz": torch.full((B, A), 0.8),
           "h": torch.full((B, A), 1.6), "zh_mask": v.clone()}
    # row 0 of each frame is HIDDEN (IGNORE): vis 0.02
    nv = torch.full((B, A), 900, dtype=torch.int32)
    nv[:, 0] = 20
    vis = {"n_full": torch.full((B, A), 1000, dtype=torch.int32), "n_vis": nv, "known": v.clone()}
    torch.manual_seed(seed)
    dec = B3.Box3DSlotDecoder(16, 8, n_queries=Q, d_model=32, depth=3, n_heads=4, enforce_band=False,
                              presence_prior=0.01)
    dec.deep_supervision = True
    model = types.SimpleNamespace(core=types.SimpleNamespace(encoder=nn.Linear(2, 2)),
                                  _perception=types.SimpleNamespace(lift=None, map_branch=None,
                                                                    box_mem=nn.Linear(1, 1), box_dec=dec))
    return dec, model, tgt, vis


def run_once(broken=False):
    dec, model, tgt, vis = toy()
    pr = D.Probe(SP, AS)
    pr.install(dec)
    orig_pt = SP.presence_term if not broken else None
    if broken:
        # the DEFECT the check exists for: IGNORE weight applied to matched slots too (no restore)
        def bad(logit, match, *, mode, ignore_w=None):
            # focal presence with the ignore weight applied to EVERY slot and NO restore on matched ones
            tgt_ = torch.zeros_like(logit)
            for b, rr in enumerate(match["rows"]):
                tgt_[b, rr] = 1.0
            w = torch.zeros_like(logit)                      # the defect: every slot zeroed, matched included
            el = SP.sigmoid_focal_elementwise(logit, tgt_)
            n_m = int(tgt_.sum())
            return {"loss": (el * w).sum() / max(n_m, 1), "n_matched": n_m, "n_exempt": 0}
        pr._orig["presence_term"] = bad
    try:
        pr.active = True
        base = torch.randn(2, 8, 16, requires_grad=True)
        mem = base * 1.0
        pred = dec(mem)
        r = SP.refined_box3d_losses(pred, tgt, presence_loss="focal", vis1=True, vis=vis)
        r["total"].backward()
        row = {"box3d": r["total"], "box3d_presence": r["loss_presence"]}
        s = D.summarise_step(pr, model, 1, row)
    finally:
        pr.remove()
    assert SP.presence_term is not None and AS.match_slots.__name__ == "match_slots", "wrappers not removed"
    return s


s = run_once()
assert "ERROR" not in s, s
assert len(s["layers"]) == 3
for d in s["layers"]:
    assert d["targets_from_own_layer_match"] and d["logit_is_decoder_output"], d
    assert d["n_pos_targets"] == 4, d            # 2 frames x (3 rows - 1 IGNORE)
    assert d["n_matched_zero_grad"] == 0, d
    assert d["frac_matched_grad_negative"] == 1.0, d
assert s["mem_grad_norm"] and s["mem_grad_norm"] > 0
print("clean toy:", {k: s["layers"][-1][k] for k in ("n_pos_targets", "n_exempt", "grad_abs_mean_matched",
                                                      "grad_abs_mean_unmatched", "logit_mean_matched",
                                                      "frac_same_slot_as_last_layer")}, "mem", s["mem_grad_norm"])
s2 = run_once(broken=True)
zero = [d["n_matched_zero_grad"] for d in s2["layers"]]
assert all(z == 4 for z in zero), zero
print("RED arm (ignore weight zeroes matched slots): matched slots with zero presence gradient per layer =", zero)
print("TOY CHECK PASS")
