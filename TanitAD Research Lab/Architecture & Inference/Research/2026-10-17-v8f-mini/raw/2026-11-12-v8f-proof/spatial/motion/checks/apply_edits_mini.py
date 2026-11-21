"""Applies the MOTION edits to tree_patch/stack/tanitad/models/v8f_mini.py and tree_patch/stack/scripts/v8f_mini_train.py (each replacement must match once)."""
R = "C:/Users/Admin/v8proof_c/motion/tree_patch/stack/"


def edit(path, pairs):
    src = open(R + path, "rb").read().decode("utf-8")
    assert "\r" not in src
    for old, new in pairs:
        assert src.count(old) == 1, (path, src.count(old), old[:90])
        src = src.replace(old, new)
    open(R + path, "wb").write(src.encode("utf-8"))


MINI = [
    # ---- constants
    ('WM_CELL_PRED_WIDTH = 128 ',
     'WM_RESIDUAL_MODES = ("off", "on")                         # MOTION lever A (--wm-l1-residual): the level-1 predictor returns z + f(z, a) (ResidualPredictor, last layer zero-initialised)\n'
     'WM_DIAG_MODES = ("off", "on")                             # MOTION diagnostics (--wm-l1-diag): extra train-log keys, no effect on the loss\n'
     '#: MOTION config fields and their defaults: a checkpoint records one ONLY when it differs from the default (every earlier checkpoint keeps its key set and loads bit for bit; an absent key = the default)\n'
     'MOTION_CFG_DEFAULTS = {"wm_l1_residual": "off", "wm_l1_roll_k": 0, "w_wm_roll": 0.0, "wm_l1_diag": "off"}\n'
     'WM_CELL_PRED_WIDTH = 128 '),
    # ---- config fields
    ('    wm_strides: tuple = (2, 2)\n',
     '    wm_l1_residual: str = "off"        # MOTION lever A: off | on (level-1 predictor = z + f(z, a), last layer zero-initialised: at init it equals the copy exactly)\n'
     '    wm_l1_roll_k: int = 0              # MOTION lever B: K, rollout length of the multi-step level-1 term (0 = off; else K >= 2 and K <= hmax - 1)\n'
     '    w_wm_roll: float = 0.0             # MOTION lever B: W, the weight of roll/L1 in the level-1 loss (0.0 exactly when K is 0; > 0 when K >= 2)\n'
     '    wm_l1_diag: str = "off"            # MOTION diagnostics: on = the train-log record gains wm_pred_l1, wm_copy_l1 (+ wm_roll_l1, wm_roll_copy_l1 when K > 0); the loss is unchanged\n'
     '    wm_strides: tuple = (2, 2)\n'),
    # ---- validation
    ('        if self.cons_absent not in ("mask", "zero"):\n',
     '        if self.wm_l1_residual not in WM_RESIDUAL_MODES:\n'
     '            raise ValueError(f"wm_l1_residual must be one of {WM_RESIDUAL_MODES}, got {self.wm_l1_residual!r}")\n'
     '        if self.wm_l1_diag not in WM_DIAG_MODES:\n'
     '            raise ValueError(f"wm_l1_diag must be one of {WM_DIAG_MODES}, got {self.wm_l1_diag!r}")\n'
     '        if not (isinstance(self.wm_l1_roll_k, int) and not isinstance(self.wm_l1_roll_k, bool) and self.wm_l1_roll_k >= 0):\n'
     '            raise ValueError(f"wm_l1_roll_k must be an integer >= 0, got {self.wm_l1_roll_k!r}")\n'
     '        if not (math.isfinite(self.w_wm_roll) and self.w_wm_roll >= 0.0):\n'
     '            raise ValueError(f"w_wm_roll must be a finite number >= 0, got {self.w_wm_roll!r}")\n'
     '        if not ((self.wm_l1_roll_k == 0 and self.w_wm_roll == 0.0) or (self.wm_l1_roll_k >= 2 and self.w_wm_roll > 0.0)):\n'
     '            raise ValueError(f"wm_l1_roll_k {self.wm_l1_roll_k} and w_wm_roll {self.w_wm_roll}: need K = 0 with W = 0.0 (off) or K >= 2 with W > 0 (a rollout term with no weight, or a weight with no rollout, is a lever that does nothing)")\n'
     '        if self.wm == "off" and (self.wm_l1_residual != "off" or self.wm_l1_roll_k != 0 or self.wm_l1_diag != "off"):\n'
     '            raise ValueError("wm_l1_residual / wm_l1_roll_k / wm_l1_diag need the world-model auxiliary (wm a or b): with wm off there is no level-1 predictor")\n'
     '        if self.wm_l1_roll_k > self.hmax - 1:\n'
     '            raise ValueError(f"wm_l1_roll_k {self.wm_l1_roll_k} does not fit the training sequence: the WM sees hmax = {self.hmax} frames, so K <= {self.hmax - 1}")\n'
     '        if self.cons_absent not in ("mask", "zero"):\n'),
    # ---- WMAux build
    ('        lv = [mk(latent_dim=cfg.wm_l1_dim, pred_width=cfg.wm_l1_pred_width, action_dim=2, act_embed_dim=32, **kind),\n',
     '        if cfg.wm_l1_residual == "on":                                  # MOTION lever A: absent argument = today\'s LevelConfig\n'
     '            kind["pred_residual"] = True\n'
     '        if cfg.wm_l1_roll_k > 0:                                        # MOTION lever B\n'
     '            kind["roll_k"], kind["roll_weight"] = cfg.wm_l1_roll_k, cfg.w_wm_roll\n'
     '        lv = [mk(latent_dim=cfg.wm_l1_dim, pred_width=cfg.wm_l1_pred_width, action_dim=2, act_embed_dim=32, **kind),\n'),
    ('        self.state_mode = cfg.wm_state\n',
     '        self.diag = cfg.wm_l1_diag == "on"                                # MOTION diagnostics: read by forward\n'
     '        self.state_mode = cfg.wm_state\n'),
    # ---- probe_pred sees through the wrapper
    ('        from tanitad.models.hjepa import CausalPredictor, CellCausalPredictor\n        pred = self.stack.predictors[0]\n',
     '        import copy\n'
     '        from tanitad.models.hjepa import CausalPredictor, CellCausalPredictor, ResidualPredictor\n'
     '        top = self.stack.predictors[0]\n'
     '        wrapped = isinstance(top, ResidualPredictor)                  # MOTION lever A: the kind is read from the wrapped module; the wrapper adds no parameters\n'
     '        pred = top.inner if wrapped else top\n'),
    ('        dev = next(pred.parameters()).device\n        g = torch.Generator().manual_seed(4321)\n',
     '        dev = next(pred.parameters()).device\n'
     '        probe = pred\n'
     '        if wrapped:                                                  # the last layer of a residual predictor is ZERO at init: probe a copy whose last layer is a fixed non-zero map, so cross_cell reads the trunk\n'
     '            probe = copy.deepcopy(pred)\n'
     '            gg = torch.Generator().manual_seed(99)\n'
     '            last = probe.out[-1]\n'
     '            with torch.no_grad():\n'
     '                last.weight.copy_(torch.randn(last.weight.shape, generator=gg).to(dev) * 0.5)\n'
     '        g = torch.Generator().manual_seed(4321)\n'),
    ('            d = (pred(z, a) - pred(z2, a))[..., cw:].abs().max()\n',
     '            d = (probe(z, a) - probe(z2, a))[..., cw:].abs().max()\n'),
    ('        return {"kind": kind, "cross_cell": bool(d > 1e-7), "params": n(pred), "params_flat_ref": n(ref)}\n',
     '        return {"kind": kind, "cross_cell": bool(d > 1e-7), "params": n(pred), "params_flat_ref": n(ref), "residual": "on" if wrapped else "off"}\n'),
    # ---- WMAux.forward
    ('        return self.stack.loss(z1, a)\n',
     '        if self.diag:                                                           # MOTION diagnostics: extra no-grad keys only, the terms and the total are the same\n'
     '            return self.stack.loss(z1, a, diag=True)\n'
     '        return self.stack.loss(z1, a)\n'),
    # ---- V8FMini.loss: diag keys
    ('            T["wm"] = wm["total"]\n',
     '            T["wm"] = wm["total"]\n'
     '            if cfg.wm_l1_diag == "on":                                      # MOTION diagnostics: logged only, detached, not in the total\n'
     '                T["wm_pred_l1"], T["wm_copy_l1"] = wm["pred/L1"].detach(), wm["diag/copy_L1"]\n'
     '                if cfg.wm_l1_roll_k > 0:\n'
     '                    T["wm_roll_l1"], T["wm_roll_copy_l1"] = wm["roll/L1"].detach(), wm["diag/roll_copy_L1"]\n'),
    # ---- built levers
    ('            lv["wm_l1_pred_cross_cell"] = pp["cross_cell"]\n',
     '            lv["wm_l1_pred_cross_cell"] = pp["cross_cell"]\n'
     '            lv["wm_l1_residual"] = pp["residual"]                      # MOTION lever A: read from the BUILT predictor module (is it the residual wrapper?)\n'
     '            lv["wm_l1_roll_k"] = int(self.wm.stack.cfg[0].roll_k)       # MOTION lever B: the K and the W the stack\'s loss reads (HJEPAStack.loss reads cfg[0].roll_k / .roll_weight)\n'
     '            lv["w_wm_roll"] = float(self.wm.stack.cfg[0].roll_weight)\n'
     '            lv["wm_l1_diag"] = "on" if self.wm.diag else "off"\n'),
]

TRAIN = [
    ('    "wm_l1_pred": "lever:wm_l1_pred",',
     '    "wm_l1_residual": "lever:wm_l1_residual", "wm_l1_roll_k": "lever:wm_l1_roll_k", "w_wm_roll": "lever:w_wm_roll", "wm_l1_diag": "lever:wm_l1_diag",   # MOTION: read back from the BUILT predictor module / the stack\'s loss config, see bad_wm_state\n'
     '    "wm_l1_pred": "lever:wm_l1_pred",'),
    ('    ap.add_argument("--wm-dim", dest="wm_dim",',
     '    ap.add_argument("--wm-l1-residual", dest="wm_l1_residual", choices=M.WM_RESIDUAL_MODES, default="off",\n'
     '                    help="MOTION lever A: level-1 predictor of the WM auxiliary returns z + f(z, a) (last output layer zero-initialised, so at init it equals the copy of the state). off (default) = every earlier arm bit for bit")\n'
     '    ap.add_argument("--wm-l1-roll-k", dest="wm_l1_roll_k", type=int, default=0,\n'
     '                    help="MOTION lever B: K, the rollout length of the multi-step level-1 term (rolled from one state under the level-1 actions, context growing to the predictor context, targets detached). 0 (default) = off; else K >= 2, K <= hmax - 1, with --w-wm-roll > 0")\n'
     '    ap.add_argument("--w-wm-roll", dest="w_wm_roll", type=float, default=0.0, help="MOTION lever B: W, the weight of roll/L1 in the level-1 loss (0.0 exactly when --wm-l1-roll-k is 0)")\n'
     '    ap.add_argument("--wm-l1-diag", dest="wm_l1_diag", choices=M.WM_DIAG_MODES, default="off",\n'
     '                    help="MOTION diagnostics: on = the train log gains wm_pred_l1, wm_copy_l1 (+ wm_roll_l1, wm_roll_copy_l1 when K > 0), computed without a graph; the loss is unchanged. off (default) adds no key")\n'
     '    ap.add_argument("--wm-dim", dest="wm_dim",'),
    ('wm_l1_pred=a.wm_l1_pred, wm_dim=a.wm_dim,',
     'wm_l1_pred=a.wm_l1_pred, wm_l1_residual=a.wm_l1_residual, wm_l1_roll_k=a.wm_l1_roll_k, w_wm_roll=a.w_wm_roll, wm_l1_diag=a.wm_l1_diag, wm_dim=a.wm_dim,'),
    ('"wm_state", "wm_cell_dim", "wm_l1_pred",        #',
     '"wm_state", "wm_cell_dim", "wm_l1_pred", "wm_l1_residual", "wm_l1_roll_k", "w_wm_roll",        # MOTION: an absent key = the default (off / 0 / 0.0); #'),
    ('(("wm_state", "mean"), ("wm_cell_dim", 16), ("wm_l1_pred", "flat")):',
     '(("wm_state", "mean"), ("wm_cell_dim", 16), ("wm_l1_pred", "flat"), *M.MOTION_CFG_DEFAULTS.items()):'),
    ('        for f in ("wm_state", "wm_cell_dim", "wm_dim", "wm_l1_pred"):\n',
     '        for f in ("wm_state", "wm_cell_dim", "wm_dim", "wm_l1_pred", "wm_l1_residual", "wm_l1_roll_k", "w_wm_roll", "wm_l1_diag"):\n'),
    ('    if built.get("wm_upper_dims") != [M.WM_UPPER_DIM] * 2:\n',
     '    # MOTION levers, each read back from the BUILT model: A from the class of the level-1 predictor, B from the K and W the stack\'s loss reads; a declared one that was not built (or a built one nobody declared) is refused\n'
     '    if built.get("wm_l1_residual") != a.wm_l1_residual:\n'
     '        bad.append(f"lever wm_l1_residual: declared {a.wm_l1_residual} but built {built.get(\'wm_l1_residual\')} (read from the level-1 predictor module: is it the residual wrapper)")\n'
     '    if built.get("wm_l1_roll_k") != a.wm_l1_roll_k:\n'
     '        bad.append(f"lever wm_l1_roll_k: declared {a.wm_l1_roll_k} but the built stack\'s loss reads {built.get(\'wm_l1_roll_k\')}")\n'
     '    if built.get("w_wm_roll") != a.w_wm_roll:\n'
     '        bad.append(f"lever w_wm_roll: declared {a.w_wm_roll} but the built stack\'s loss reads {built.get(\'w_wm_roll\')}")\n'
     '    if built.get("wm_l1_diag") != a.wm_l1_diag:\n'
     '        bad.append(f"lever wm_l1_diag: declared {a.wm_l1_diag} but built {built.get(\'wm_l1_diag\')}")\n'
     '    if not ((a.wm_l1_roll_k == 0 and a.w_wm_roll == 0.0) or (a.wm_l1_roll_k >= 2 and a.w_wm_roll > 0.0 and math.isfinite(a.w_wm_roll))):\n'
     '        bad.append(f"wm_l1_roll_k {a.wm_l1_roll_k} with w_wm_roll {a.w_wm_roll}: need (0, 0.0) or (K >= 2, W > 0): a lever that is declared and does nothing")\n'
     '    if built.get("wm_upper_dims") != [M.WM_UPPER_DIM] * 2:\n'),
    ('(cfg.wm_l1_pred == "cell" or k not in M.L1PRED_CFG_FIELDS)}',
     '(cfg.wm_l1_pred == "cell" or k not in M.L1PRED_CFG_FIELDS) and (k not in M.MOTION_CFG_DEFAULTS or v != M.MOTION_CFG_DEFAULTS[k])}'),
    ('{k: v for k, v in cfg.__dict__.items() if (cfg.stop_aux',
     '{k: v for k, v in cfg.__dict__.items() if (cfg.stop_aux'),
]
edit("tanitad/models/v8f_mini.py", MINI)
edit("scripts/v8f_mini_train.py", TRAIN)
print("applied")
