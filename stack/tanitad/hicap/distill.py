"""Optional student (HiCAP-S): distil the cached teacher readouts into a smaller state side; the ACTION side is shared verbatim
(action embeddings never depend on images), so only the state side is distilled (paper sec:student, Proposition distill)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .model import HiCAP, HiCAPConfig


def student_config(teacher: HiCAPConfig, d: int = 256, layers: int = 4) -> HiCAPConfig:
    c = HiCAPConfig(**{**teacher.__dict__}); c.d = d; c.layers = layers; c.heads = max(4, d // 64); return c


def share_action_tables(student: HiCAP, teacher: HiCAP):
    """Student reuses the teacher's towers, theta and pair bias verbatim (frozen). Requires equal tower width d."""
    assert student.c.d == teacher.c.d, "shared tables need equal width; project the student readout to the teacher width otherwise"
    for n in ("tower1", "tower_lat", "tower_lon", "tower_path", "tower_prof", "theta"):
        getattr(student, n).load_state_dict(getattr(teacher, n).state_dict())
        for p in getattr(student, n).parameters(): p.requires_grad_(False)
    student.pair_bias.data.copy_(teacher.pair_bias.data); student.pair_bias.requires_grad_(False)


def distill_loss(s_stu: dict, s_tea: dict, z_stu: dict | None = None, z_tea: dict | None = None, proj: dict | None = None):
    """cos + MSE on every level readout; KL on per-level distributions if logits are given. proj maps student readout -> teacher width."""
    L = 0.0
    for k, t in s_tea.items():
        s = proj[k](s_stu[k]) if proj else s_stu[k]
        s = F.normalize(s, dim=-1)
        L = L + (1 - F.cosine_similarity(s, t.detach(), dim=-1)).mean() + F.mse_loss(s, t.detach())
    if z_stu is not None:
        for k in z_tea: L = L + F.kl_div(F.log_softmax(z_stu[k], -1), F.softmax(z_tea[k].detach(), -1), reduction="batchmean")
    return L


def margin_condition(scores: torch.Tensor, delta_s: float, tau: float, delta_a: float = 0.0) -> torch.Tensor:
    """Proposition (decision preservation): the student's argmax equals the teacher's if the teacher's top-1/top-2 margin exceeds
    2 tau (delta_s + delta_a). scores [B,K] teacher logits (already scaled by tau). Returns a bool per window."""
    t2 = scores.topk(2, dim=-1).values
    return (t2[:, 0] - t2[:, 1]) > 2 * tau * (delta_s + delta_a)
