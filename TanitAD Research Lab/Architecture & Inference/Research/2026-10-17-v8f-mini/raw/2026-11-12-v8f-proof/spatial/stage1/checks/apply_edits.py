"""Apply the SPATIAL lever edits to tree_patch (exact-string replacement, each must match once). LF preserved."""
import sys

ROOT = "F:/TanitAD-scratch/v8proof/spatial/stage1/tree_patch/stack/"


def sub(path, old, new, count=1):
    with open(ROOT + path, "r", encoding="utf-8", newline="") as fh:
        s = fh.read()
    n = s.count(old)
    if n != count:
        sys.exit(f"EDIT MISMATCH in {path}: expected {count}, found {n}: {old[:70]!r}")
    s = s.replace(old, new)
    with open(ROOT + path, "w", encoding="utf-8", newline="") as fh:
        fh.write(s)


M = "tanitad/models/v8f_mini.py"
T = "scripts/v8f_mini_train.py"

# ---------------------------------------------------------------- model: constants
sub(M, 'STOPAUX_CFG_FIELDS = ("stop_aux", "w_stop_aux")',
    'WM_STATE_MODES = ("mean", "grid")                        # PREREG_V8F_PROOF_ADD12 (SPATIAL): the level-1 state of the world-model auxiliary: mean = the mean of the adapter tokens (today), grid = one projection per cell, concatenated\n'
    'WM_UPPER_DIM = 128                                        # levels 2 and 3 of the WM stack keep this width whatever the level-1 state width is\n'
    'SPATIAL_CFG_FIELDS = ("wm_state", "wm_cell_dim")          # recorded in a checkpoint\'s cfg ONLY when wm_state is grid (a mean checkpoint keeps the tip\'s key set)\n'
    'STOPAUX_CFG_FIELDS = ("stop_aux", "w_stop_aux")')

# ---------------------------------------------------------------- model: config fields
sub(M, '    wm_dim: int = 128\n    wm_strides: tuple = (2, 2)\n',
    '    wm_dim: int = 128                  # mean state: the level-1 state width (levels 2 and 3 stay WM_UPPER_DIM); grid state: unused\n'
    '    wm_state: str = "mean"             # PREREG_V8F_PROOF_ADD12: mean | grid\n'
    '    wm_cell_dim: int = 16              # grid state: numbers per cell (level-1 state width = n_tok x wm_cell_dim)\n'
    '    wm_strides: tuple = (2, 2)\n')

sub(M, '''        if self.cons_absent not in ("mask", "zero"):''',
    '''        if self.wm_state not in WM_STATE_MODES:
            raise ValueError(f"wm_state must be one of {WM_STATE_MODES}, got {self.wm_state!r}")
        if self.wm_state == "grid" and self.wm == "off":
            raise ValueError("wm_state grid needs the world-model auxiliary (wm a or b): with wm off there is no level-1 state to keep the cells of")
        if not (isinstance(self.wm_cell_dim, int) and self.wm_cell_dim >= 1):
            raise ValueError(f"wm_cell_dim must be an integer >= 1, got {self.wm_cell_dim!r}")
        if not (isinstance(self.wm_dim, int) and self.wm_dim >= 1):
            raise ValueError(f"wm_dim must be an integer >= 1, got {self.wm_dim!r}")
        if self.cons_absent not in ("mask", "zero"):''')

sub(M, '''    @property
    def window_start_effective(self) -> int:''',
    '''    @property
    def wm_l1_dim(self) -> int:
        """Width of the level-1 state of the WM stack: the mean state's ``wm_dim``, or ``n_tok x wm_cell_dim`` for the grid state."""
        return self.n_tok * self.wm_cell_dim if self.wm_state == "grid" else self.wm_dim

    @property
    def wm_l1_pred_width(self) -> int:
        """Level-1 predictor width: 128 while the state is 128 wide (today), 256 when it is wider."""
        return 128 if self.wm_l1_dim == 128 else 256

    @property
    def window_start_effective(self) -> int:''')

# ---------------------------------------------------------------- model: WMAux
sub(M, '''        mk = lambda **kw: LevelConfig(latent_dim=cfg.wm_dim, beta=cfg.wm_beta, gamma=gamma, pred_width=128,
                                      pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32, **kw)
        lv = [mk(action_dim=2, act_embed_dim=32),
              mk(action_dim=32, stride=cfg.wm_strides[0]), mk(action_dim=32, stride=cfg.wm_strides[1])]
        self.stack = HJEPAStack(lv, sigreg_projections=256)
        self.proj = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, cfg.wm_dim))
''',
    '''        mk = lambda **kw: LevelConfig(beta=cfg.wm_beta, gamma=gamma, pred_depth=2, pred_heads=4, enc_hidden=128, macro_hidden=32, **kw)
        # PREREG_V8F_PROOF_ADD12: level 1 has the state width cfg.wm_l1_dim (128 today) and its own predictor width; levels 2 and 3 keep WM_UPPER_DIM / 128. At the defaults every number is the tip's.
        up = WM_UPPER_DIM
        lv = [mk(latent_dim=cfg.wm_l1_dim, pred_width=cfg.wm_l1_pred_width, action_dim=2, act_embed_dim=32),
              mk(latent_dim=up, pred_width=128, action_dim=32, stride=cfg.wm_strides[0]),
              mk(latent_dim=up, pred_width=128, action_dim=32, stride=cfg.wm_strides[1])]
        self.stack = HJEPAStack(lv, sigreg_projections=256)
        self.state_mode = cfg.wm_state
        self.n_tok = cfg.n_tok
        # mean: LayerNorm + Linear(d, wm_dim) on the mean token (today); grid: the SAME shape of module with the output width wm_cell_dim, applied to EVERY cell
        self.proj = nn.Sequential(nn.LayerNorm(cfg.d), nn.Linear(cfg.d, cfg.wm_cell_dim if cfg.wm_state == "grid" else cfg.wm_dim))
''')

sub(M, '''        return self.proj(adapt.mean(2))

    def scale_actions''',
    '''        if self.state_mode == "grid":
            if adapt.shape[2] != self.n_tok:
                raise ValueError(f"grid state: adapter tokens {adapt.shape[2]} != the {self.n_tok} the model was built for")
            p = self.proj(adapt)                                                # [B, F, n_tok, cell_dim]: the same projection on every cell
            return p.reshape(*p.shape[:2], p.shape[2] * p.shape[3])             # cell-major flatten: [B, F, n_tok * cell_dim]
        return self.proj(adapt.mean(2))

    @torch.no_grad()
    def probe_state(self, d: int) -> dict:
        """What ``z1`` BUILT does (the G-DVB 'built' side): the output width on a fixed random [1, 2, n_tok, d] input and whether swapping two cells of one frame moves it
        (a state that keeps location moves; a mean does not). Fixed generator: the global RNG stream is not touched."""
        g = torch.Generator().manual_seed(1234)
        x = torch.randn(1, 2, self.n_tok, d, generator=g)
        y = x.clone()
        y[:, 0, 0], y[:, 0, self.n_tok - 1] = x[:, 0, self.n_tok - 1].clone(), x[:, 0, 0].clone()
        was = self.training
        self.eval()
        try:
            a, b = self.z1(x), self.z1(y)
        finally:
            self.train(was)
        return {"z1_width": int(a.shape[-1]), "location_sensitive": bool((a - b).abs().max() > 1e-4)}

    def scale_actions''')

# ---------------------------------------------------------------- model: built_levers (present only with the WM built)
sub(M, '''        if self.lvl_head is not None:                                      # PREREG_V8F_LEVEL: present ONLY with the head built''',
    '''        if self.wm is not None:                                            # PREREG_V8F_PROOF_ADD12: what the BUILT WM does -- read from the stack's modules and from z1's behaviour, not from the config
            pr = self.wm.probe_state(cfg.d)
            lv["wm_state"] = "grid" if pr["location_sensitive"] else "mean"
            lv["wm_z1_width"] = pr["z1_width"]
            lv["wm_l1_dim"] = int(self.wm.stack.cfg[0].latent_dim)
            lv["wm_l1_pred_width"] = int(self.wm.stack.predictors[0].in_proj.weight.shape[0])
            lv["wm_upper_dims"] = [int(c.latent_dim) for c in self.wm.stack.cfg[1:]]
            lv["wm_cell_dim"] = int(self.wm.proj[1].out_features) if pr["location_sensitive"] else None
            lv["wm_dim"] = int(self.wm.proj[1].out_features) if not pr["location_sensitive"] else None
        if self.lvl_head is not None:                                      # PREREG_V8F_LEVEL: present ONLY with the head built''')

# ---------------------------------------------------------------- trainer
sub(T, '''    "arm": "lever:arm", "navonly": "lever:navonly", "no_consistency": "lever:consistency", "wm": "lever:wm",''',
    '''    "arm": "lever:arm", "navonly": "lever:navonly", "no_consistency": "lever:consistency", "wm": "lever:wm",
    # PREREG_V8F_PROOF_ADD12 (SPATIAL): argv -> make_config -> the BUILT config AND the BUILT model (built_levers: z1's measured width and location sensitivity, the stack's level-1 width and predictor width) -> declared_vs_built (bad_wm_state)
    "wm_state": "lever:wm_state", "wm_cell_dim": "cfg:wm_cell_dim", "wm_dim": "cfg:wm_dim",''')

sub(T, '''    ap.add_argument("--cons-absent", choices=("mask", "zero"), default=None,''',
    '''    ap.add_argument("--wm-state", dest="wm_state", choices=M.WM_STATE_MODES, default="mean",
                    help="PREREG_V8F_PROOF_ADD12: level-1 state of the WM auxiliary. mean (default, the tip bit for bit) = the mean of the adapter tokens; grid = one projection per cell, concatenated (width n_tok x --wm-cell-dim)")
    ap.add_argument("--wm-cell-dim", dest="wm_cell_dim", type=int, default=16, help="grid state: numbers per cell (default 16: 40 cells -> 640)")
    ap.add_argument("--wm-dim", dest="wm_dim", type=int, default=128, help="mean state: level-1 state width (default 128; 640 = the capacity control of PREREG_V8F_PROOF_ADD12)")
    ap.add_argument("--cons-absent", choices=("mask", "zero"), default=None,''')

sub(T, '''                           consistency=lv["consistency"], wm=lv["wm"], n_tok=40''',
    '''                           consistency=lv["consistency"], wm=lv["wm"], wm_state=a.wm_state, wm_cell_dim=a.wm_cell_dim, wm_dim=a.wm_dim, n_tok=40''')

sub(T, '''"wm", "wm_dim", "wm_strides", "wm_beta", "wm_gamma",
                   "kin_mode",''',
    '''"wm", "wm_dim", "wm_strides", "wm_beta", "wm_gamma",
                   "wm_state", "wm_cell_dim",        # PREREG_V8F_PROOF_ADD12: a checkpoint without the keys is a mean-state one (defaults supplied below), so a grid argv cannot start from it nor the reverse
                   "kin_mode",''')

sub(T, '''    diff = {k: (norm(ck["cfg"][k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k in ck["cfg"] and norm(ck["cfg"][k]) != norm(getattr(cfg, k))}''',
    '''    ck_cfg = dict(ck["cfg"])
    for k_, d_ in (("wm_state", "mean"), ("wm_cell_dim", 16)):                 # PREREG_V8F_PROOF_ADD12: an absent key is the default (a mean checkpoint is written without them)
        ck_cfg.setdefault(k_, d_)
    if ck_cfg["wm_state"] == "mean":
        ck_cfg["wm_cell_dim"] = getattr(cfg, "wm_cell_dim") if getattr(cfg, "wm_state") == "mean" else ck_cfg["wm_cell_dim"]     # the cell width of a mean state is unused: not a mismatch
    diff = {k: (norm(ck_cfg[k]), norm(getattr(cfg, k))) for k in INIT_CKPT_MATCH if k in ck_cfg and norm(ck_cfg[k]) != norm(getattr(cfg, k))}''')

sub(T, '''def declared_vs_built(a, cfg, model, opt, data_info, enc=None''',
    '''def bad_wm_state(a, cfg, model) -> list:
    """PREREG_V8F_PROOF_ADD12: --wm-state / --wm-cell-dim / --wm-dim read back from the BUILT model. A declared lever the model does not build, or a built one nobody declared, is a launch blocker."""
    bad = []
    built = model.built_levers()
    if model.wm is None:
        d = build_parser()
        for f in ("wm_state", "wm_cell_dim", "wm_dim"):
            if getattr(a, f) != d.get_default(f):
                bad.append(f"--{f.replace('_', '-')} {getattr(a, f)!r} declared but no WM auxiliary was built (wm off): a lever that is declared and builds nothing")
        return bad
    l1 = a.wm_cell_dim * cfg.n_tok if a.wm_state == "grid" else a.wm_dim
    want_pred = 128 if l1 == 128 else 256
    if built.get("wm_state") != a.wm_state:
        bad.append(f"lever wm_state: declared {a.wm_state} but built {built.get('wm_state')} (z1 location-sensitive: {built.get('wm_state') == 'grid'})")
    if built.get("wm_z1_width") != l1 or built.get("wm_l1_dim") != l1:
        bad.append(f"wm level-1 state width: declared {l1} but z1 emits {built.get('wm_z1_width')} and the stack's level 1 is {built.get('wm_l1_dim')}")
    if built.get("wm_l1_pred_width") != want_pred:
        bad.append(f"wm level-1 predictor width: declared {want_pred} (state {l1}) but built {built.get('wm_l1_pred_width')}")
    if built.get("wm_upper_dims") != [M.WM_UPPER_DIM] * 2:
        bad.append(f"wm levels 2 and 3 width: expected {M.WM_UPPER_DIM} but built {built.get('wm_upper_dims')}")
    if a.wm_state == "grid" and built.get("wm_cell_dim") != a.wm_cell_dim:
        bad.append(f"wm_cell_dim: declared {a.wm_cell_dim} but built {built.get('wm_cell_dim')}")
    if a.wm_state == "mean" and built.get("wm_dim") != a.wm_dim:
        bad.append(f"wm_dim: declared {a.wm_dim} but the mean projection emits {built.get('wm_dim')}")
    return bad


def declared_vs_built(a, cfg, model, opt, data_info, enc=None''')

sub(T, '''    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL''',
    '''    bad += bad_wm_state(a, cfg, model)                                                           # PREREG_V8F_PROOF_ADD12: --wm-state / --wm-cell-dim / --wm-dim read back from the BUILT model
    bad += bad_level(a, model, data_info)                                                        # PREREG_V8F_LEVEL''')

sub(T, '''"cfg": {k: v for k, v in cfg.__dict__.items() if cfg.stop_aux == "on" or k not in M.STOPAUX_CFG_FIELDS}''',
    '''"cfg": {k: v for k, v in cfg.__dict__.items() if (cfg.stop_aux == "on" or k not in M.STOPAUX_CFG_FIELDS) and (cfg.wm_state == "grid" or k not in M.SPATIAL_CFG_FIELDS)}''')
print("OK")
