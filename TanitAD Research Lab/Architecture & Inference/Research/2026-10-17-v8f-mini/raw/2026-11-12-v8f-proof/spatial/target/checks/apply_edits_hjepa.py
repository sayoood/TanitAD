"""Edits of stack/tanitad/models/hjepa.py for the FUTTOK lever (run once on tree_patch). Each edit must match exactly once."""
import sys

p = "C:/Users/Admin/v8proof_c/target/tree_patch/stack/tanitad/models/hjepa.py"
s = open(p, encoding="utf-8", newline="").read()
assert "\r\n" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


# 1. config fields
rep("""    context: int = 3
    adaln_init: str = "identity_gate"
""", """    futtok_k: int = 0                # TARGET lever FUTTOK, LEVEL 1 ONLY: K, the rollout length of the token-read term futtok/L1; 0 = off (no term, no key). Otherwise K >= 2 and futtok_weight > 0
    futtok_weight: float = 0.0       # TARGET lever FUTTOK: W, the weight of futtok/L1 in the level-1 loss (0.0 exactly when futtok_k is 0)
    context: int = 3
    adaln_init: str = "identity_gate"
""")

# 2. validation
rep("""            if lv >= 2 and (c.pred_residual or c.roll_k or c.roll_weight):
                raise ValueError(f"level {lv}: pred_residual / roll_k / roll_weight are level-1 levers")
""", """            if lv >= 2 and (c.pred_residual or c.roll_k or c.roll_weight or c.futtok_k or c.futtok_weight):
                raise ValueError(f"level {lv}: pred_residual / roll_k / roll_weight / futtok_k / futtok_weight are level-1 levers")
            if not ((c.futtok_k == 0 and c.futtok_weight == 0.0) or (c.futtok_k >= 2 and c.futtok_weight > 0.0 and math.isfinite(c.futtok_weight))):
                raise ValueError(f"level {lv}: futtok_k {c.futtok_k} / futtok_weight {c.futtok_weight}: need (0, 0.0) or (K >= 2, W > 0)")
""")

# 3. loss signature + docstring tail
rep("""    def loss(self, z1_seq: Tensor, a_seq: Tensor, *, generator: Optional[torch.Generator] = None,
             detach_target: bool = False, diag: bool = False) -> dict:""",
    """    def loss(self, z1_seq: Tensor, a_seq: Tensor, *, generator: Optional[torch.Generator] = None,
             detach_target: bool = False, diag: bool = False,
             tok_head: Optional[nn.Module] = None, tok_tgt: Optional[Tensor] = None) -> dict:""")

rep("""        nothing, and it saves the backward through the target frames.
        \"\"\"
        enc = self.encode(z1_seq, a_seq)
        out: dict = {}
""", """        nothing, and it saves the backward through the target frames.

        TARGET lever FUTTOK (level 1, ``futtok_k > 0``): ``tok_head`` maps one cell of a rolled-out state to a token and ``tok_tgt`` ``(B, T, n_tok, d_tok)`` holds the
        (normalised, detached) frozen token of every frame; both are required when the lever is on and refused when it is off.
        \"\"\"
        if (tok_head is None) != (tok_tgt is None):
            raise ValueError("tok_head and tok_tgt come together: the FUTTOK term needs both")
        if tok_head is not None and not self.cfg[0].futtok_k:
            raise ValueError("tok_head / tok_tgt given but the level-1 config has futtok_k 0: a declared input the loss would ignore")
        if tok_head is None and self.cfg[0].futtok_k:
            raise ValueError(f"futtok_k {self.cfg[0].futtok_k} on but the loss got no tok_head / tok_tgt (WMAux.forward needs the cached token grid `feats`)")
        enc = self.encode(z1_seq, a_seq)
        out: dict = {}
        rcache: dict = {}                                                   # K -> roll_pairs(...) of this call: ROLL and FUTTOK with the same K share ONE rollout
""")

# 4. the term
rep("""            if lv == 1 and c.roll_k > 0:                                    # MOTION lever B: the K-step rollout term, targets detached, the start state keeps its graph
                roll = self._roll_loss(z, a, c.roll_k)
""", """            if lv == 1:
                def _rolled(k, z=z, a=a):
                    if k not in rcache:
                        rcache[k] = self.roll_pairs(z, a, k)
                    return rcache[k]
            if lv == 1 and c.roll_k > 0:                                    # MOTION lever B: the K-step rollout term, targets detached, the start state keeps its graph
                roll = self._roll_loss(z, a, c.roll_k, rolled=_rolled(c.roll_k))
""")

rep("""            if lv == 1 and diag:                                            # MOTION diagnostics: no graph, no effect on the loss
""", """            if lv == 1 and c.futtok_k > 0:                                  # TARGET lever FUTTOK: a per-cell linear read of the rolled-out states against the frozen future tokens
                futt = self._futtok_loss(_rolled(c.futtok_k), tok_head, tok_tgt)
                out["futtok/L1"] = futt
                level_loss = level_loss + c.futtok_weight * futt            # <- the line the t5 mutation test guards: deleting it makes total(W) - total(0) read 0.0
                if diag:                                                    # copy through the same head: the START state against the same future tokens, no graph
                    with torch.no_grad():
                        out["diag/futtok_copy"] = self._futtok_copy(z.detach(), tok_head, tok_tgt, c.futtok_k)
            if lv == 1 and diag:                                            # MOTION diagnostics: no graph, no effect on the loss
""")

# 5. _roll_loss + the futtok helpers
rep("""    def _roll_loss(self, z: Tensor, a: Tensor, k: int) -> Tensor:
        \"\"\"``roll/L1``: mean over starts, steps and batch of the squared error of the K-step rollout, the width reduced as in :meth:`_pred_loss`.\"\"\"
        out, tgt, _ = self.roll_pairs(z, a, k)
""", """    def _roll_loss(self, z: Tensor, a: Tensor, k: int, rolled=None) -> Tensor:
        \"\"\"``roll/L1``: mean over starts, steps and batch of the squared error of the K-step rollout, the width reduced as in :meth:`_pred_loss`.
        ``rolled`` = an already computed ``roll_pairs(z, a, k)`` (shared with the FUTTOK term of the same K).\"\"\"
        out, tgt, _ = rolled if rolled is not None else self.roll_pairs(z, a, k)
""")

rep("""    def goal_cost(self, z_pred: Tensor, goal: Tensor, level: int, last_k: Optional[int] = None) -> Tensor:""",
    """    @staticmethod
    def _cells(x: Tensor, n_tok: int) -> Tensor:
        \"\"\"``(..., n_tok * cell)`` -> ``(..., n_tok, cell)``: the inverse of the cell-major flatten of ``WMAux.z1``.\"\"\"
        if x.shape[-1] % n_tok:
            raise ValueError(f"state width {x.shape[-1]} is not a multiple of the {n_tok} cells of the token target")
        return x.reshape(*x.shape[:-1], n_tok, x.shape[-1] // n_tok)

    def _futtok_loss(self, rolled, tok_head: nn.Module, tok_tgt: Tensor) -> Tensor:
        \"\"\"``futtok/L1``: ``rolled`` = ``roll_pairs`` output (the rollout states keep their graph, the start state too). Per start ``s`` the states of steps 1..K are read cell by cell by
        ``tok_head`` and compared with ``tok_tgt[:, s + k]`` (squared error, mean over the token width, cells, steps, starts and batch). The target never carries a gradient.\"\"\"
        out, _, starts = rolled
        n_tok, k = tok_tgt.shape[2], out.shape[2]
        if tok_tgt.shape[1] < starts[-1] + k + 1:
            raise ValueError(f"token target has {tok_tgt.shape[1]} frames, the rollouts need {starts[-1] + k + 1}")
        per = []
        for i, s in enumerate(starts):
            tgt = tok_tgt[:, s + 1:s + k + 1].detach()                       # (B, K, n_tok, d_tok)
            per.append((tok_head(self._cells(out[i], n_tok)) - tgt).pow(2).mean())
        return torch.stack(per).mean()                                      # every start has the same number of elements: the mean of means is the global mean

    def _futtok_copy(self, z: Tensor, tok_head: nn.Module, tok_tgt: Tensor, k: int) -> Tensor:
        \"\"\"Diagnostic: the same head on the START state ``z[:, s]`` against the same future tokens (what a model that never moves the scene would score).\"\"\"
        n_tok = tok_tgt.shape[2]
        per = []
        for s in range(0, z.shape[1] - k, 2):
            h0 = tok_head(self._cells(z[:, s], n_tok))[:, None]              # (B, 1, n_tok, d_tok)
            per.append((h0 - tok_tgt[:, s + 1:s + k + 1]).pow(2).mean())
        return torch.stack(per).mean()

    def goal_cost(self, z_pred: Tensor, goal: Tensor, level: int, last_k: Optional[int] = None) -> Tensor:""")

open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
