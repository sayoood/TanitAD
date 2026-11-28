"""Applies the MOTION edits to tree_patch/stack/tanitad/models/hjepa.py (exact-match replacements, each must match once). LF, UTF-8."""
import sys

P = "C:/Users/Admin/v8proof_c/motion/tree_patch/stack/tanitad/models/hjepa.py"
src = open(P, "rb").read().decode("utf-8")
assert "\r" not in src


def rep(old, new):
    global src
    assert src.count(old) == 1, (src.count(old), old[:80])
    src = src.replace(old, new)


# ---- exports + docstring
rep('"SIGReg", "IDMHead", "LevelEncoder", "MacroActionEncoder", "CausalPredictor", "CellCausalPredictor",\n',
    '"SIGReg", "IDMHead", "LevelEncoder", "MacroActionEncoder", "CausalPredictor", "CellCausalPredictor", "ResidualPredictor",\n')
rep('* :class:`HJEPAStack`       -- the levels,',
    '* :class:`ResidualPredictor` -- MOTION lever A: a thin wrapper that makes a level-1 predictor return ``z_ctx + f(z_ctx, a_ctx)`` for every caller.\n'
    '* :class:`HJEPAStack`       -- the levels,')

# ---- the wrapper (before the stack banner)
BANNER = '''# =============================================================================================
# The stack
# =============================================================================================
'''
WRAP = '''class ResidualPredictor(nn.Module):
    """MOTION lever A (``--wm-l1-residual on``): ``forward(z_ctx, a_ctx) = z_ctx + f(z_ctx, a_ctx)`` at EVERY position, for either predictor kind.

    ``f`` is the wrapped predictor (``self.inner``); its LAST output layer is zero-initialised here, so at initialisation the predictor equals
    the copy of the input state exactly (``z + 0.0 == z``). Every caller that reads ``stack.predictors[0]`` (``_pred_loss``, ``rollout``, the planner,
    readers outside this file) gets the skip without knowing: the wrapper exposes ``context``, ``latent_dim``, ``action_dim`` and, by fall-through,
    every other attribute of the wrapped module (``in_proj``, ``n_cells``, ``cell_dim``, ...). It is built ONLY when the lever is on: with the lever off
    there is no wrapper and the state-dict keys are those of the tree before this lever.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        out = getattr(inner, "out", None)
        if not (isinstance(out, nn.Sequential) and isinstance(out[-1], nn.Linear)):
            raise ValueError("ResidualPredictor needs a predictor whose last output layer is out[-1] (nn.Linear)")
        if getattr(inner, "residual", False):
            raise ValueError("the wrapped predictor already has its own skip (residual=True)")
        self.inner = inner
        self.latent_dim, self.action_dim, self.context = inner.latent_dim, inner.action_dim, inner.context
        nn.init.zeros_(out[-1].weight)
        nn.init.zeros_(out[-1].bias)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.__dict__.get("_modules", {}).get("inner")
            if inner is None:
                raise
            return getattr(inner, name)

    def forward(self, z_ctx: Tensor, a_ctx: Tensor) -> Tensor:
        return z_ctx + self.inner(z_ctx, a_ctx)


'''
rep(BANNER, WRAP + BANNER)

# ---- LevelConfig
rep('''    pred_cell_dim: int = 0           # cell kind only: numbers per cell
''', '''    pred_cell_dim: int = 0           # cell kind only: numbers per cell
    pred_residual: bool = False      # MOTION lever A, LEVEL 1 ONLY: the predictor returns z_ctx + f(z_ctx, a_ctx) (ResidualPredictor, last layer zero-initialised); False = no wrapper = every earlier arm
    roll_k: int = 0                  # MOTION lever B, LEVEL 1 ONLY: K, the rollout length of the multi-step term roll/L1; 0 = off (the term is not built, not in the loss, no key). Otherwise K >= 2 and roll_weight > 0
    roll_weight: float = 0.0         # MOTION lever B: W, the weight of roll/L1 in the level-1 loss (0.0 exactly when roll_k is 0)
''')

# ---- stack construction
rep('''        self.cfg: list[LevelConfig] = list(levels)
        self.pred_reduction = pred_reduction
        L = len(self.cfg)
''', '''        self.cfg: list[LevelConfig] = list(levels)
        self.pred_reduction = pred_reduction
        L = len(self.cfg)
        for lv, c in enumerate(self.cfg, start=1):          # MOTION: the two levers live at level 1 and only there
            if lv >= 2 and (c.pred_residual or c.roll_k or c.roll_weight):
                raise ValueError(f"level {lv}: pred_residual / roll_k / roll_weight are level-1 levers")
            if not ((c.roll_k == 0 and c.roll_weight == 0.0) or (c.roll_k >= 2 and c.roll_weight > 0.0 and math.isfinite(c.roll_weight))):
                raise ValueError(f"level {lv}: roll_k {c.roll_k} / roll_weight {c.roll_weight}: need (0, 0.0) or (K >= 2, W > 0)")
''')
rep('''        self.predictors = nn.ModuleList(preds)
        self.idm_heads''', '''        if c1.pred_residual:                                                  # MOTION lever A: wrap AFTER the predictor's own draws (the RNG stream is the unwrapped model's), zero the last layer
            if predictor_overrides and 1 in predictor_overrides:
                raise ValueError("pred_residual with a predictor override at level 1 is not supported")
            preds[0] = ResidualPredictor(preds[0])
        self.predictors = nn.ModuleList(preds)
        self.idm_heads''')

# ---- rollout: one core, used by rollout() and by the roll loss
rep('''        e = self.act_embed(a) if level == 1 else a
        zs = [z]
        for h in range(h_):
            lo = max(0, h + 1 - context)
            zc = torch.stack(zs[lo:h + 1], dim=1)
            nxt = pred(zc, e[:, lo:h + 1])[:, -1]
            zs.append(nxt)
        out = torch.stack(zs[1:], dim=1)
        return out.reshape(*lead, h_, out.shape[-1])
''', '''        e = self.act_embed(a) if level == 1 else a
        out = self._roll_embedded(pred, z, e, context)
        return out.reshape(*lead, h_, out.shape[-1])

    @staticmethod
    def _roll_embedded(pred: nn.Module, z: Tensor, e: Tensor, context: int) -> Tensor:
        """The autoregressive core of :meth:`rollout`: ``z`` ``(n, D)``, ``e`` ``(n, H, E)`` EMBEDDED actions -> ``(n, H, D)``. ONE function for the read-time rollout and
        the MOTION roll loss (the context grows from one state up to ``context``, then slides)."""
        zs = [z]
        for h in range(e.shape[1]):
            lo = max(0, h + 1 - context)
            zc = torch.stack(zs[lo:h + 1], dim=1)
            nxt = pred(zc, e[:, lo:h + 1])[:, -1]
            zs.append(nxt)
        return torch.stack(zs[1:], dim=1)

    def roll_pairs(self, z: Tensor, a: Tensor, k: int):
        """MOTION lever B: roll the level-1 predictor ``k`` steps from the single state ``z[:, s]`` for every start ``s in range(0, T - k, 2)``.
        ``z`` ``(B, T, D)`` states, ``a`` ``(B, T-1, E)`` EMBEDDED level-1 actions (what :meth:`_pred_loss` receives). The starts are stacked on the batch axis
        (rows are independent), the context is the predictor's own (what ``rollout(1, ..., context=<predictor context>)`` uses at read time).
        Returns ``(pred, tgt, starts)``: ``pred`` ``(S, B, k, D)`` keeps the graph (the start state too), ``tgt = z[:, s+1 : s+k+1]`` is DETACHED."""
        t_ = z.shape[1]
        if k < 2 or k > t_ - 1:
            raise ValueError(f"roll_k {k} does not fit a sequence of {t_} states (need 2 <= K <= T - 1)")
        pred = self.predictors[0]
        ctx = getattr(pred, "context", None) or k
        starts = list(range(0, t_ - k, 2))
        z0 = torch.cat([z[:, s] for s in starts], dim=0)
        ea = torch.cat([a[:, s:s + k] for s in starts], dim=0)
        out = self._roll_embedded(pred, z0, ea, ctx)
        tgt = torch.stack([z[:, s + 1:s + k + 1] for s in starts], dim=0).detach()
        return out.reshape(len(starts), z.shape[0], k, -1), tgt, starts

    def _roll_loss(self, z: Tensor, a: Tensor, k: int) -> Tensor:
        """``roll/L1``: mean over starts, steps and batch of the squared error of the K-step rollout, the width reduced as in :meth:`_pred_loss`."""
        out, tgt, _ = self.roll_pairs(z, a, k)
        sq = (out - tgt).pow(2)
        per = sq.mean(dim=-1) if self.pred_reduction == "mean" else sq.sum(dim=-1)
        return per.mean()
''')

# ---- loss
rep('''    def loss(self, z1_seq: Tensor, a_seq: Tensor, *, generator: Optional[torch.Generator] = None,
             detach_target: bool = False) -> dict:''', '''    def loss(self, z1_seq: Tensor, a_seq: Tensor, *, generator: Optional[torch.Generator] = None,
             detach_target: bool = False, diag: bool = False) -> dict:''')
rep('''            level_loss = pred + c.beta * sg + c.gamma * idm
            out[f"pred/L{lv}"], out[f"sigreg/L{lv}"], out[f"idm/L{lv}"] = pred, sg, idm
''', '''            level_loss = pred + c.beta * sg + c.gamma * idm
            out[f"pred/L{lv}"], out[f"sigreg/L{lv}"], out[f"idm/L{lv}"] = pred, sg, idm
            if lv == 1 and c.roll_k > 0:                                    # MOTION lever B: the K-step rollout term, targets detached, the start state keeps its graph
                roll = self._roll_loss(z, a, c.roll_k)
                out["roll/L1"] = roll
                level_loss = level_loss + c.roll_weight * roll              # <- the line the m5 mutation test guards: deleting it makes total(W) - total(0) read 0.0
            if lv == 1 and diag:                                            # MOTION diagnostics: no graph, no effect on the loss
                with torch.no_grad():
                    cp = (z[:, :-1] - z[:, 1:]).pow(2)
                    out["diag/copy_L1"] = (cp.mean(dim=-1) if self.pred_reduction == "mean" else cp.sum(dim=-1)).mean()
                    if c.roll_k > 0:
                        rp, rt, rs = self.roll_pairs(z.detach(), a.detach(), c.roll_k)
                        z0 = torch.stack([z.detach()[:, s] for s in rs], dim=0)[:, :, None]
                        rc = (z0 - rt).pow(2)
                        out["diag/roll_copy_L1"] = (rc.mean(dim=-1) if self.pred_reduction == "mean" else rc.sum(dim=-1)).mean()
''')
open(P, "wb").write(src.encode("utf-8"))
print("hjepa edits applied")
