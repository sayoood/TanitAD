"""PREREG_V8F_NAVRB2: patch v8f_mini.py / v8f_mini_data.py in the tree export (exact, unique replacements; refuses on any miss)."""
import sys

ROOT = sys.argv[1]


def sub(path, pairs):
    p = ROOT + "/" + path
    s = open(p, "rb").read().decode("utf-8")
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{path}: {n} matches for {old[:80]!r}")
        s = s.replace(old, new)
    open(p, "wb").write(s.encode("utf-8"))
    print("patched", path, len(pairs))


M = "stack/tanitad/models/v8f_mini.py"
sub(M, [
    ('NEW_LEVER_KEY_PREFIXES = LEVEL_KEY_PREFIXES + LEAD_KEY_PREFIXES + AUX_KEY_PREFIXES + ("e_stok.",) + CTX_KEY_PREFIXES',
     'NAV_RB_MODES = ("off", "on")\n'
     '#: PREREG_V8F_NAVRB2: the ROUNDABOUT state channel of the route-text nav (``--nav-rb``): its state-dict prefix and its batch key (read with the nav, for a model built with it; NOT in INFERENCE_INPUT_KEYS, which tests pin)\n'
     'NAV_RB_KEY_PREFIXES = ("e_nav_rb.",)\n'
     'NAV_RB_KEYS = frozenset({"nav_rb"})\n'
     'NEW_LEVER_KEY_PREFIXES = LEVEL_KEY_PREFIXES + LEAD_KEY_PREFIXES + AUX_KEY_PREFIXES + ("e_stok.",) + CTX_KEY_PREFIXES + NAV_RB_KEY_PREFIXES'),
    ('    nav_source: str = "v9"             # PREREG_V8F_STRAT1',
     '    nav_rb: str = "off"                # PREREG_V8F_NAVRB2 (`--nav-rb`): off = nothing built (every earlier arm, bit for bit); on = e_nav_rb, Linear(1, d, bias=False), zero-init, ADDED to e_nav(nav) in S, T and O\n'
     '    nav_source: str = "v9"             # PREREG_V8F_STRAT1'),
    ('        if self.nav_source not in NAV_SOURCES:\n',
     '        if self.nav_rb not in NAV_RB_MODES:\n'
     '            raise ValueError(f"nav_rb must be one of {NAV_RB_MODES}, got {self.nav_rb!r}")\n'
     '        if self.nav_source not in NAV_SOURCES:\n'),
    ('                self.ctx_head = _mlp(d, d, CTX_DIM)\n        self.p_gt = 1.0\n',
     '                self.ctx_head = _mlp(d, d, CTX_DIM)\n'
     '        self.e_nav_rb = None\n'
     '        if cfg.nav_rb == "on":                                             # PREREG_V8F_NAVRB2: built LAST under a forked RNG (no draw reaches the global stream) and zeroed: x + 0.0 == x, so the step-0 forward is the off model\'s\n'
     '            with torch.random.fork_rng(devices=[]):\n'
     '                self.e_nav_rb = nn.Linear(1, d, bias=False)\n'
     '            nn.init.zeros_(self.e_nav_rb.weight)\n'
     '        self.p_gt = 1.0\n'),
    ('    # ---- helpers -------------------------------------------------------------------------------------------\n    def _ego(self, f8):\n',
     '    # ---- helpers -------------------------------------------------------------------------------------------\n'
     '    def _nav_tok(self, b, i):\n'
     '        """The nav conditioning token of S, T and O: ``e_nav(nav)``, plus ``e_nav_rb(nav_rb)`` for a model built with the roundabout channel (PREREG_V8F_NAVRB2; a batch with no ``nav_rb`` key is then a KeyError, never a silent zero)."""\n'
     '        t = self.e_nav(self._at(b["nav"], i))\n'
     '        if self.e_nav_rb is not None:\n'
     '            t = t + self.e_nav_rb(self._at(b["nav_rb"], i).to(self.e_nav_rb.weight.dtype))\n'
     '        return t\n\n'
     '    def _ego(self, f8):\n'),
    ('        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),\n',
     '        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self._nav_tok(b, i),\n'),
    ('        toks = [self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),\n',
     '        toks = [self._ego(self._at(b["p10f"], i)), self._nav_tok(b, i),\n'),
    ('        toks = [ego_tok, self.e_nav(self._at(b["nav"], i)),\n',
     '        toks = [ego_tok, self._nav_tok(b, i),\n'),
    ('            g["CTX"] = [self.ctx_head]                                              # PREREG_V8F_STRAT1\n        return g\n',
     '            g["CTX"] = [self.ctx_head]                                              # PREREG_V8F_STRAT1\n'
     '        if self.e_nav_rb is not None:\n'
     '            g["NAVRB"] = [self.e_nav_rb]                                            # PREREG_V8F_NAVRB2: G-LIVE then requires its gradient\n'
     '        return g\n'),
    ('        allowed = INFERENCE_INPUT_KEYS | (STOK_KEYS if getattr(model, "e_stok", None) is not None else frozenset())      # PREREG_V8F_OTBAND: the strategic token\'s two keys, for a model built with it\n',
     '        allowed = INFERENCE_INPUT_KEYS | (STOK_KEYS if getattr(model, "e_stok", None) is not None else frozenset())      # PREREG_V8F_OTBAND: the strategic token\'s two keys, for a model built with it\n'
     '        allowed = allowed | (NAV_RB_KEYS if getattr(model, "e_nav_rb", None) is not None else frozenset())                 # PREREG_V8F_NAVRB2: the roundabout channel, for a model built with it\n'),
])

D = "stack/tanitad/data/v8f_mini_data.py"
sub(D, [
    ('#: the nav-text channel ranges (the v9 nav\'s layout:',
     'NAV_RB_VARIANTS = ("real", "zero")\n\n\n'
     'def nav_rb_arrays(J: dict, path, expect_sha256: str, split: str = "train", variant: str = "real"):\n'
     '    """PREREG_V8F_NAVRB2 (``--nav-rb``): the ROUNDABOUT state channel ``nav_rb`` [N, 1] float32 in {0, 1}, ROW-ALIGNED to the join (``build_navtext_nav_rb.py``: 1 iff the router row at the step is a flag-0\n'
     '    roundabout class). Re-hashes the file (refuses a mismatch or an undeclared sha), refuses a shifted clip / off / n, a wrong shape or a value outside {0, 1}. ``variant == "zero"`` (the zeroed-channel\n'
     '    control): every row 0.0. Returns ``(nav_rb, rec)``; ``rec`` carries the sha256, the variant, ``n_on`` (after the variant) and ``n_on_real``."""\n'
     '    if variant not in NAV_RB_VARIANTS:\n'
     '        raise ValueError(f"nav_rb_arrays: variant {variant!r}")\n'
     '    sha = _sha256_file(path)\n'
     '    if not expect_sha256 or sha != expect_sha256:\n'
     '        raise ValueError(f"nav_rb_arrays: sha256 {sha} != declared {expect_sha256!r}")\n'
     '    with np.load(path, allow_pickle=False) as z:\n'
     '        _assert_table_aligned(z, J, split, "nav_rb_arrays")\n'
     '        rb = np.asarray(z[f"{split}_nav_rb"], np.float32)\n'
     '    N = int(np.asarray(J["off"])[-1] + np.asarray(J["n"])[-1])\n'
     '    if rb.shape != (N, 1):\n'
     '        raise ValueError(f"nav_rb_arrays: shape {rb.shape} != ({N}, 1)")\n'
     '    if not np.isin(rb, (0.0, 1.0)).all():\n'
     '        raise ValueError("nav_rb_arrays: values outside {0, 1}")\n'
     '    real_on = int(rb.sum())\n'
     '    if variant == "zero":\n'
     '        rb = np.zeros_like(rb)\n'
     '    rec = {"sha256": sha, "variant": variant, "n_on": int(rb.sum()), "n_on_real": real_on, "digest": hashlib.sha256(np.ascontiguousarray(rb).tobytes()).hexdigest()}\n'
     '    return rb, rec\n\n\n'
     '#: the nav-text channel ranges (the v9 nav\'s layout:'),
    ('        it["pose_s"] = d["pose"][t - a_s][:3].astype(np.float32)\n',
     '        it["pose_s"] = d["pose"][t - a_s][:3].astype(np.float32)\n'
     '        if "nav_rb" in J:                                         # PREREG_V8F_NAVRB2: the roundabout channel of the hmax frames, read with the nav; absent from every earlier join = no key in the batch\n'
     '            it["nav_rb"] = J["nav_rb"][[self.row(ci, int(j)) for j in idc]]\n'),
    ('    for k in ("a_t", "a_s", "t", "ci"):\n        out[k] = torch.as_tensor([it[k] for it in items], dtype=torch.long)\n',
     '    for k in ("a_t", "a_s", "t", "ci"):\n        out[k] = torch.as_tensor([it[k] for it in items], dtype=torch.long)\n'
     '    if "nav_rb" in items[0]:                                     # PREREG_V8F_NAVRB2: present only when the join carries nav_rb\n'
     '        out["nav_rb"] = stack("nav_rb", torch.float32)\n'),
])
