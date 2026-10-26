"""CLM-pure arm (paper sec:protocol NC10): CLM's recipe transferred to driving with nothing added on the state side.

CLM-8B = frozen Qwen3-8B, last-token pooled embedding, a ~20 M state head and a ~20 M action head (MLP hidden -> width -> 512),
bidirectional InfoNCE (github.com/Contrastive-LM/CLM @ d5f9ef0, `src/clm/heads.py`, README "How it works"; PUBLISHED).
This arm keeps exactly that state side: one small MLP state head per hierarchy level on the frozen backbone's LAST-TOKEN state, and
nothing else — no fusion transformer, no motion adapter, no tap tokens, no ego / nav / route tokens, no audio, no rule memory.
The action side, vocabularies, losses, data and training loop are HiCAP's own (node towers on cached frozen-text embeddings of the
action definitions = CLM's action head; numeric towers for the geometric path / speed-profile actions that have no text), so the
paired difference HiCAP - CLM-pure measures what the added state machinery earns, per metric family.

Single clock: the state head reads the most recent refresh's last-token state, so at inference the arm costs a few MLPs per tick plus
the backbone refresh. With images every state is new (no exact-text cache hits as in CLM's text-state benchmarks), which is why the
refresh stays asynchronous at 2 Hz and the arm's tokens age exactly like HiCAP's.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from .model import HiCAP, HiCAPConfig, LEVELS, MLP


class CLMPure(HiCAP):
    def __init__(self, c: HiCAPConfig | None = None, width: int = 1024):
        super().__init__(c)
        c = self.c
        del self.fusion                                                    # the whole state-side machinery goes
        self.state_ln = nn.LayerNorm(c.d_v)
        self.state_heads = nn.ModuleDict({k: MLP(c.d_v, width, c.d) for k in LEVELS})   # CLM state head, one per level
        self.scene_proj = nn.Linear(c.d_v, c.d)                            # one scene token for the stage-3 re-ranker

    def encode(self, batch, width=None):
        z = self.state_ln(batch["last"])                                   # [B, d_v] frozen last-token state (CLM readout)
        q = {k: h(z) for k, h in self.state_heads.items()}
        s = {k: self.heads[k](q[k], width) for k in LEVELS if k != "goal"}
        g = self.goal_head(q["goal"])
        return s, g[:, : self.c.n_goals], g[:, self.c.n_goals:].view(-1, 3, 3), self.scene_proj(z).unsqueeze(1)

    def state_params(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if n.startswith(("state_ln", "state_heads", "heads", "goal_head", "scene_proj")))
