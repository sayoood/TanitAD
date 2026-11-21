"""MRATE step 1: dilated, causal-padded level windows in stack/tanitad/models/hjepa.py (defaults = every earlier build, bit for bit). Run once on the tree."""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8", newline="").read()
assert "\r" not in s and "causal_pad" not in s


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


rep('''def _windows(z: Tensor, w: int, s: int) -> Tensor:
    """``(N, T, D) -> (N, n, w*D)`` with ``n = (T - w)//s + 1``; window ``i`` = ``z[i*s : i*s+w]``."""
    n_, t_, d_ = z.shape
    if t_ < w:
        raise ValueError(f"need at least window={w} lower-level states, got {t_}")
    n = (t_ - w) // s + 1
    idx = torch.arange(n, device=z.device)[:, None] * s + torch.arange(w, device=z.device)[None, :]
    return z[:, idx].reshape(n_, n, w * d_)
''', '''def _windows(z: Tensor, w: int, s: int, d: int = 1) -> Tensor:
    """``(N, T, D) -> (N, n, w*D)`` with ``n = (T - ((w-1)*d + 1))//s + 1``; window ``i`` = ``z[i*s + m*d]``, ``m = 0..w-1`` (oldest first).
    ``d = 1`` is the consecutive window ``z[i*s : i*s+w]`` (every earlier call, bit for bit). PREREG_V8F_MRATE: ``d > 1`` = dilated at the level's own spacing."""
    n_, t_, d_ = z.shape
    span = (w - 1) * d + 1
    if t_ < span:
        raise ValueError(f"need at least window={w} (dilation {d}: span {span}) lower-level states, got {t_}")
    n = (t_ - span) // s + 1
    idx = torch.arange(n, device=z.device)[:, None] * s + torch.arange(w, device=z.device)[None, :] * d
    return z[:, idx].reshape(n_, n, w * d_)
''')
rep('''    act_sigreg: float = 0.005
''', '''    act_sigreg: float = 0.005
    dilation: int = 1          # PREREG_V8F_MRATE: spacing (lower-level states) between the ``window`` states a state reads; 1 = consecutive (every earlier build)
    causal_pad: bool = False   # PREREG_V8F_MRATE: front-pad ``(window-1)*dilation`` copies of the first lower state, so state i sits at lower index i*stride and reads only that state and earlier ones
''')
rep('''            if c.stride < 1 or c.window < 1:
                raise ValueError(f"level {lv}: stride/window must be >= 1")''', '''            if c.stride < 1 or c.window < 1 or c.dilation < 1:
                raise ValueError(f"level {lv}: stride/window/dilation must be >= 1")''')
rep('''        self.encoders = nn.ModuleList([
            LevelEncoder(self.cfg[i - 1].latent_dim, self.cfg[i].latent_dim,
                         self.cfg[i].enc_hidden, self.cfg[i].window) for i in range(1, L)])
''', '''        base_seed = torch.initial_seed()
        self.encoders = nn.ModuleList([self._build_encoder(i, base_seed) for i in range(1, L)])
''')
rep('''    # ---- bookkeeping ----------------------------------------------------------------------
    @property
    def n_levels(self) -> int:''', '''    def _build_encoder(self, i: int, base_seed: int) -> "LevelEncoder":
        """Level ``i + 1``'s state encoder. Window 1: built exactly as before. Window > 1 (PREREG_V8F_MRATE): weights drawn on a PRIVATE RNG stream
        derived from the run seed, so the global stream every later module reads is the window-1 stack's (W0 and W1 share every other parameter bit for bit)."""
        c = self.cfg[i]
        mk = lambda w: LevelEncoder(self.cfg[i - 1].latent_dim, c.latent_dim, c.enc_hidden, w)  # noqa: E731
        enc = mk(1)                              # the window-1 encoder ALWAYS draws from the global stream, so the stream every later module reads is the same for any window
        if c.window == 1:
            return enc
        with torch.random.fork_rng(devices=[]):  # the real window-K encoder draws on a private stream derived from the run seed
            torch.manual_seed((base_seed * 1_000_003 + 101 * (i + 1)) % (2 ** 63 - 1))
            return mk(c.window)

    # ---- bookkeeping ----------------------------------------------------------------------
    @property
    def n_levels(self) -> int:''')
rep('''                           act_sigreg_live=c.act_sigreg > 0.0, macro_mode=mac.mode)
''', '''                           act_sigreg_live=c.act_sigreg > 0.0, macro_mode=mac.mode)
                if enc.window > 1 or c.dilation > 1 or c.causal_pad:      # PREREG_V8F_MRATE: present only when moved (a window-1 stack keeps its key set)
                    row.update(dilation=c.dilation, causal_pad=c.causal_pad, enc_in_dim=enc.net[0].in_features)
''')
rep('''        win = _windows(z.reshape(-1, t_, d_), c.window, c.stride)
''', '''        zf = z.reshape(-1, t_, d_)
        if c.causal_pad and c.window > 1:       # PREREG_V8F_MRATE
            pad = (c.window - 1) * c.dilation
            zf = torch.cat([zf[:, :1].expand(-1, pad, -1), zf], dim=1)
        win = _windows(zf, c.window, c.stride, c.dilation)
''')
rep("        if strict and (t_ - c.window) % c.stride != 0:", "        if strict and not c.causal_pad and (t_ - ((c.window - 1) * c.dilation + 1)) % c.stride != 0:")
rep("            lo = c.window - 1\n", "            lo = 0 if c.causal_pad else (c.window - 1) * c.dilation\n")
open(p, "w", encoding="utf-8", newline="").write(s)
print("hjepa patched")
