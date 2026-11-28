"""Reproduces the PREREG_V8F_MRATE edit of stack/tanitad/models/v8f_mini.py on a fresh tip (idempotence: refuses a file it already patched)."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
assert "wm_k" not in s, "already patched"


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


# 1. config field
rep('''    wm_strides: tuple = (2, 2)
''', '''    wm_strides: tuple = (2, 2)
    wm_k: int = 1                      # PREREG_V8F_MRATE (`--wm-k`): how many past states each WM level's state reads, spaced at that level's own step. 1 = W0 = every earlier arm, bit for bit (no module, no key added);
                                       # K > 1 = W1: L1 reads the adapter tokens of frames t, t-1, .., t-K+1; L2 / L3 read K states of the level below at wm_strides[l-2] apart (hjepa LevelConfig.window / dilation / causal_pad)
''')
# 2. validation
rep('''        if self.wm != "off" and self.hmax < 2 * self.wm_strides[0] * self.wm_strides[1] + 4:
            raise ValueError("hmax too short for the WM levels")
''', '''        if self.wm != "off" and self.hmax < 2 * self.wm_strides[0] * self.wm_strides[1] + 4:
            raise ValueError("hmax too short for the WM levels")
        if not (isinstance(self.wm_k, int) and not isinstance(self.wm_k, bool) and self.wm_k >= 1):
            raise ValueError(f"wm_k must be an int >= 1, got {self.wm_k!r}")
        if self.wm_k > 1 and self.wm == "off":
            raise ValueError("wm_k > 1 needs a world model (wm a or b): a lever that is declared and builds nothing")
''')
# 3. WMAux
a = s.index("class WMAux(nn.Module):")
b = s.index("    def scale_actions(self, act: Tensor) -> Tensor:")
new_init = '''class WMAux(nn.Module):
    """M2: the H-JEPA world-model auxiliary (``tanitad.models.hjepa``), levels z1 / z2 / z3 on the O / T / S cycles
    (0.2 s / 0.4 s / 0.8 s on the cache grid -- the nearest multiples to 0.1 / 0.5 / 1.0 s). a = without IDM
    (gamma 0), b = with IDM (beta 0.08, gamma 100).

    PREREG_V8F_MRATE, ``cfg.wm_k`` = K. K = 1 (default) is every earlier build, bit for bit: no module and no state-dict key is added. K > 1 (W1): level 1's state reads the adapter tokens of
    frames t, t-1, .., t-K+1 (``enc1``, an MLP over the K concatenated projections, oldest first, the first frame's projection repeated before the window start); levels 2 and 3 read K states of
    the level below, ``wm_strides[l-2]`` of them apart (a ``LevelConfig`` with ``window=K, dilation=stride, causal_pad=True``), so every state reads only its own time and earlier ones.
    ``enc1`` is built LAST, on a private RNG stream derived from the run seed: W0 and W1 built under one seed share every other parameter bit for bit."""

    def __init__(self, cfg: V8FMiniConfig):
        super().__init__()
        from tanitad.models.hjepa import HJEPAStack, LevelConfig, LevelEncoder   # S3's module; absent => this arm cannot be built
        gamma = cfg.wm_gamma if cfg.wm == "b" else 0.0
        mk = lambda **kw: LevelConfig(latent_dim=cfg.wm_dim, beta=cfg.wm_beta, gamma=gamma, pred_width=128,
                                      pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32, **kw)
        K = int(cfg.wm_k)
        up = lambda i: dict(window=K, dilation=cfg.wm_strides[i], causal_pad=True) if K > 1 else {}      # K = 1: LevelConfig's defaults, i.e. the earlier build
        lv = [mk(action_dim=2, act_embed_dim=32),
              mk(action_dim=32, stride=cfg.wm_strides[0], **up(0)), mk(action_dim=32, stride=cfg.wm_strides[1], **up(1))]
        self.stack = HJEPAStack(lv, sigreg_projections=256)
        self.proj = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, cfg.wm_dim))
        self.gamma = gamma
        self.ch1_scale = CH1_SCALE[cfg.kin_mode]
        self.enc1 = None
        if K > 1:                                                              # PREREG_V8F_MRATE: built after everything else, on a private stream (the global RNG is the W0 build's)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed((torch.initial_seed() * 1_000_003 + 101) % (2 ** 63 - 1))
                self.enc1 = LevelEncoder(cfg.wm_dim, cfg.wm_dim, 128, K)

    def built_k(self) -> dict:
        """What the BUILT modules read (the G-DVB reader): per level the window, the dilation and whether the window is causally padded. Level 1's window is ``enc1``'s (1 when absent)."""
        st = self.stack
        ks = [self.enc1.window if self.enc1 is not None else 1] + [st.encoders[i].window for i in range(2)]
        return {"k": ks, "dilation": [1] + [st.cfg[i + 1].dilation for i in range(2)],
                "causal_pad": [self.enc1 is not None] + [bool(st.cfg[i + 1].causal_pad) for i in range(2)]}

    def z1(self, adapt: Tensor) -> Tensor:
        """Level-1 latent of every cached frame: the mean-pooled adapter token, projected. ``adapt`` [B, F, n_tok, d] -> [B, F, D].
        ONE spelling: ``forward`` (training) and rung beta's planner (``tanitad.models.v8f_beta``) both call it.
        With ``wm_k`` K > 1 (PREREG_V8F_MRATE) frame j's latent is ``enc1`` of the projections of frames j-K+1 .. j (oldest first); frames before the window's first are the first frame's projection."""
        p = self.proj(adapt.mean(2))
        if self.enc1 is None:
            return p
        from tanitad.models.hjepa import _windows
        K = self.enc1.window
        return self.enc1(_windows(torch.cat([p[:, :1].expand(-1, K - 1, -1), p], dim=1), K, 1, 1))

'''
s = s[:a] + new_init + s[b:]
# 4. wm_only_loss
rep('''    @staticmethod
    def _partial_ce(logits: Tensor, allowed: Tensor, has: Tensor) -> Tensor:''', '''    def wm_only_loss(self, batch: dict) -> dict:
        """PREREG_V8F_MRATE, the WM-only mode: ONLY the world model's loss terms, on a FROZEN adapter. ``adapt`` is detached, so the adapter (and ``pos``) receive exactly 0.0 even if they are trainable; the readers,
        the heads and the planner are never run. ``terms``: ``wm`` (the stack's total), ``wm/<stack term>`` per level, ``total`` = ``w_wm`` x ``wm``."""
        if self.wm is None:
            raise ValueError("wm_only_loss needs a world model (wm a or b)")
        wm = self.wm(self._encode(batch).detach(), batch["act"])
        T = {"wm": wm["total"]}
        T.update({f"wm/{k}": v for k, v in wm.items() if k != "total"})
        T["total"] = self.cfg.w_wm * wm["total"]
        return {"terms": T}

    @staticmethod
    def _partial_ce(logits: Tensor, allowed: Tensor, has: Tensor) -> Tensor:''')
# 5. built_levers
rep('''        if self.lvl_head is not None:                                      # PREREG_V8F_LEVEL: present ONLY with the head built (a flag-off model keeps its key set); the gate reads built.get("level_head", "off")''',
    '''        if self.wm is not None and any(k > 1 for k in self.wm.built_k()["k"]):      # PREREG_V8F_MRATE: present ONLY with a multi-frame WM built (a W0 model keeps its key set); the gate reads built.get("wm_k", 1)
            bk = self.wm.built_k()
            lv["wm_k"], lv["wm_k_levels"], lv["wm_dilation_levels"], lv["wm_causal_pad_levels"] = max(bk["k"]), bk["k"], bk["dilation"], bk["causal_pad"]
        if self.lvl_head is not None:                                      # PREREG_V8F_LEVEL: present ONLY with the head built (a flag-off model keeps its key set); the gate reads built.get("level_head", "off")''')
open(p, "w", encoding="utf-8", newline="").write(s)
print("patched", p)
