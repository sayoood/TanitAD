import sys
p = "F:/TanitAD-scratch/strat1/whead/tree/stack/tanitad/models/v8f_mini.py"
s = open(p, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (old[:60], s.count(old))
    s = s.replace(old, new)


# 1 constants
rep('''CTX_HEAD_MODES = ("off", "on")
''', '''CTX_HEAD_MODES = ("off", "on")
#: PREREG_V8F_WEATHERHEAD form (a) (`--ctx-pool`): the attention-pooled detected-information head, five sub-heads (one per CTX_VARS entry) over the planner's own scene tokens (frames t-3..t, 160 tokens). ``off`` (default) builds nothing
#: = a4357e79 bit for bit. The logits go to the fit loss and the evaluators ONLY: they never reach S, T, O, ``e_code``, ``e_ck`` or any goal path (T-ISO, pinned by ``ctx_pool_isolation_problems``).
CTX_POOL_MODES = ("off", "on")
CTX_POOL_INIT_SEED = 20261117
CTX_POOL_KEY_PREFIXES = ("ctx_pool_head.",)
CTX_POOL_HEADS = 4
''')
rep('+ ("e_stok.",) + CTX_KEY_PREFIXES ', '+ ("e_stok.",) + CTX_KEY_PREFIXES + CTX_POOL_KEY_PREFIXES ')

# 2 cfg field
rep('''    nav_source: str = "v9"             # PREREG_V8F_STRAT1 (`--nav-source`)''', '''    ctx_pool: str = "off"              # PREREG_V8F_WEATHERHEAD (`--ctx-pool`): off = nothing built (a4357e79, bit for bit); on = the attention-pooled detected-information head (CTX_VARS, CTX_DIM logits) on the planner's scene tokens
    nav_source: str = "v9"             # PREREG_V8F_STRAT1 (`--nav-source`)''')
rep('''        if self.nav_source not in NAV_SOURCES:''', '''        if self.ctx_pool not in CTX_POOL_MODES:
            raise ValueError(f"ctx_pool must be one of {CTX_POOL_MODES}, got {self.ctx_pool!r}")
        if self.nav_source not in NAV_SOURCES:''')

# 3 PoolHead
rep('''class WMAux(nn.Module):''', '''class PoolHead(nn.Module):
    """PREREG_V8F_WEATHERHEAD 4.1: LayerNorm(d) -> MultiheadAttention(d, 4 heads, batch_first) with ONE learned query ~ N(0, 0.02^2) and a key padding mask -> LayerNorm(d) -> Linear(d, n). 264,448 + 257 n parameters."""

    def __init__(self, d: int, n: int, heads: int = CTX_POOL_HEADS):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.out = nn.Linear(d, n)

    def forward(self, tok: Tensor, pad: Tensor | None = None) -> Tensor:
        x = self.n1(tok)
        o, _ = self.att(self.q.expand(x.shape[0], -1, -1), x, x, key_padding_mask=pad, need_weights=False)
        return self.out(self.n2(o[:, 0]))


def build_ctx_pool_heads(d: int, seed: int) -> "nn.ModuleDict":
    """The five sub-heads, built in CTX_VARS order under a forked RNG seeded ``seed`` (the global stream is untouched)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.ModuleDict({v: PoolHead(d, n) for v, n in zip(CTX_VARS, CTX_N)})


class WMAux(nn.Module):''')

# 4 build
rep('''        self.p_gt = 1.0

    # ---- helpers''', '''        self.ctx_pool_head = None
        if cfg.ctx_pool == "on":                                           # PREREG_V8F_WEATHERHEAD: built LAST under a forked RNG: every module above and the global stream after it are the off model's
            self.ctx_pool_head = build_ctx_pool_heads(d, CTX_POOL_INIT_SEED)
        self.p_gt = 1.0

    # ---- helpers''')

# 5 methods
rep('''    def _aux_in(self, b, adapt: Tensor, last: Tensor) -> Tensor:''', '''    def _ctx_pool_scene(self, b, adapt: Tensor, last: Tensor):
        """PREREG_V8F_WEATHERHEAD: the planner's own scene assembly (frames t-3..t, adapter + pos + time) and its pad mask. VISION ONLY: the only batch key read is ``frame_ok``. A row whose frames are ALL padded gets an
        unmasked NOW slot so attention never returns NaN (the row is masked in the loss and the read)."""
        tok, pad = self._scene(adapt, b["frame_ok"], last)
        n = adapt.shape[2]
        allpad = pad.all(1)
        pad = torch.cat([pad[:, :-n], pad[:, -n:] & ~allpad[:, None]], 1)
        return tok, pad

    def _ctx_pool_apply(self, tok: Tensor, pad: Tensor) -> Tensor:
        """PREREG_V8F_WEATHERHEAD: the five sub-heads on the scene tokens -> logits [B, CTX_DIM] float32 in CTX_VARS order (the layout of ``ctx_logits``)."""
        return torch.cat([self.ctx_pool_head[v](tok, pad) for v in CTX_VARS], -1).float()

    def _ctx_pool(self, b, adapt: Tensor, last: Tensor) -> Tensor:
        """PREREG_V8F_WEATHERHEAD: ``ctx_pool_logits`` [B, CTX_DIM] from the scene tokens. VISION ONLY (pinned by :func:`ctx_pool_source_problems` and :func:`assert_ctx_pool_vision_only`)."""
        tok, pad = self._ctx_pool_scene(b, adapt, last)
        return self._ctx_pool_apply(tok, pad)

    def ctx_pool_fit_logits(self, b, train_adapter: bool = False) -> Tensor:
        """PREREG_V8F_WEATHERHEAD 4.1 (the FIT path): the adapter, ``pos``, ``time`` and the token assembly run under no_grad (the frozen model gets no gradient path); only the sub-heads carry a graph.
        ``train_adapter`` (the FEAT control only) lets the adapter, ``pos`` and ``time`` carry gradient too."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and train_adapter):
            adapt = self._encode(b)
            last = torch.full((adapt.shape[0],), adapt.shape[1] - 1, dtype=torch.long, device=adapt.device)
            tok, pad = self._ctx_pool_scene(b, adapt, last)
        return self._ctx_pool_apply(tok, pad)

    def _aux_in(self, b, adapt: Tensor, last: Tensor) -> Tensor:''')

# 6 forward
rep('''            out["ctx_logits"] = self._ctx(b, adapt, last_now)
''', '''            out["ctx_logits"] = self._ctx(b, adapt, last_now)
        if self.ctx_pool_head is not None:                              # PREREG_V8F_WEATHERHEAD: logits for the fit loss and the evaluators ONLY (never an input of S, T, O)
            out["ctx_pool_logits"] = self._ctx_pool(b, adapt, last_now)
''')

# 7 groups
rep('''            g["CTX"] = [self.ctx_head]                                              # PREREG_V8F_STRAT1
''', '''            g["CTX"] = [self.ctx_head]                                              # PREREG_V8F_STRAT1
        if self.ctx_pool_head is not None:
            g["CTX_POOL"] = list(self.ctx_pool_head.values())                       # PREREG_V8F_WEATHERHEAD
''')

# 8 built levers
rep('''        if self.lead_head is not None or self.e_aux is not None:           # the BUILT wiring''', '''        if self.ctx_pool_head is not None:                                 # PREREG_V8F_WEATHERHEAD: present ONLY with the head built (an off model keeps the tip's key set)
            lv["ctx_pool"] = "on"
            lv["ctx_pool_dim"] = int(sum(self.ctx_pool_head[v].out.out_features for v in CTX_VARS))
            lv["ctx_pool_vision_only_source"] = not ctx_pool_source_problems()
        if self.lead_head is not None or self.e_aux is not None:           # the BUILT wiring''')

# 9 module functions
s += '''

# ---------------------------------------------------------------------------------------------------- PREREG_V8F_WEATHERHEAD form (a)
def ctx_pool_terms(logits: Tensor, y: Tensor, frame_ok_now: Tensor) -> dict:
    """The FIT loss (card 4.2): unweighted cross entropy per variable over the rows with a label (``y >= 0``) whose NOW frame is real; ``T["ctx_pool"]`` = the SUM over variables. ``y`` [B, 5] long, -1 = masked."""
    T, o = {}, 0
    for k, (v, n) in enumerate(zip(CTX_VARS, CTX_N)):
        t = y[:, k].long()
        m = (t >= 0) & frame_ok_now
        ce = F.cross_entropy(logits[:, o:o + n], t.clamp_min(0), reduction="none")
        T[f"ctx_pool/{v}"] = (ce * m).sum() / m.sum().clamp_min(1)
        o += n
    T["ctx_pool"] = sum(T[f"ctx_pool/{v}"] for v in CTX_VARS)
    return T


#: the methods whose CODE must read vision only (the scene assembly, the head application, the fit path)
CTX_POOL_FNS = ("_ctx_pool_scene", "_ctx_pool_apply", "_ctx_pool", "ctx_pool_fit_logits")


def ctx_pool_source_problems(cls=None) -> list:
    """PREREG_V8F_WEATHERHEAD T-VIS, STATIC half: the identifiers of the pooled head's CODE (:data:`CTX_POOL_FNS`) that name an ego, nav, limit, code, label or metadata input (:data:`CTX_FORBIDDEN_NAMES`). Empty = vision only."""
    cls = V8FMini if cls is None else cls
    return sorted({n for f in CTX_POOL_FNS for n in _forbidden_in(getattr(cls, f), CTX_FORBIDDEN_NAMES)})


def assert_ctx_pool_vision_only(model: V8FMini, batch: dict) -> dict:
    """PREREG_V8F_WEATHERHEAD T-VIS: ``ctx_pool_logits`` is a function of the VISION tokens alone. Raises AssertionError unless (1) :func:`ctx_pool_source_problems` is empty; (2) ``model._ctx_pool`` on a key-guarded batch READS only
    :data:`CTX_ALLOWED_KEYS`; (3) the logits are IDENTICAL (bit for bit) when every non-vision batch key is replaced, and DIFFERENT when ``feats`` is."""
    if model.ctx_pool_head is None:
        raise AssertionError("assert_ctx_pool_vision_only: the model has no pooled head")
    src = ctx_pool_source_problems(type(model))
    if src:
        raise AssertionError(f"the pooled head's code names non-vision inputs: {src}")
    rec = _vision_only_run(model, batch, lambda b, adapt, last: model._ctx_pool(b, adapt, last), CTX_ALLOWED_KEYS, "ctx pool head")
    return {**rec, "source_problems": src}


def ctx_pool_isolation_problems(model: V8FMini, batch: dict) -> list:
    """PREREG_V8F_WEATHERHEAD T-ISO (b), the extension of :func:`ctx_isolation_problems`: ``forward_infer`` twice, the second after adding N(0, 1) to every ``ctx_pool_head`` parameter (forked RNG). The planner and goal outputs
    (:data:`CTX_ISOLATED_OUTPUTS`) and ``ctx_logits`` (when the model has the shipped head) must be bit-identical and ``ctx_pool_logits`` must differ. A non-empty list = the head feeds the planner or the goal path (or the check is blind)."""
    if model.ctx_pool_head is None:
        return ["the model has no pooled head"]
    was = model.training
    model.eval()
    probs = []
    saved = [p.detach().clone() for p in model.ctx_pool_head.parameters()]
    try:
        with torch.no_grad():
            a = model.forward_infer(batch)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(7)
                for p in model.ctx_pool_head.parameters():
                    p.add_(torch.randn_like(p))
            b2 = model.forward_infer(batch)
        for k in CTX_ISOLATED_OUTPUTS + ("ctx_logits",):
            if k not in a:
                continue
            if not torch.equal(a[k], b2[k]):
                probs.append(f"{k} moved when only ctx_pool_head weights changed")
        if "ctx_pool_logits" not in a or torch.equal(a["ctx_pool_logits"], b2["ctx_pool_logits"]):
            probs.append("ctx_pool_logits did not move when ctx_pool_head weights changed")
    finally:
        with torch.no_grad():
            for p, q in zip(model.ctx_pool_head.parameters(), saved):
                p.copy_(q)
        model.train(was)
    return probs
'''
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
