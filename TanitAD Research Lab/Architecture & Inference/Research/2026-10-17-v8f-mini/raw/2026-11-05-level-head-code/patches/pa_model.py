"""Lever (a) `--level-head {off,on}` + `--w-level`: model side. Reads tip -> writes new (LF, exact-anchor replace; each anchor must occur exactly once)."""
import sys
from pathlib import Path

SRC = Path(sys.argv[1]) / "stack/tanitad/models/v8f_mini.py"
DST = Path(sys.argv[2]) / "stack/tanitad/models/v8f_mini.py"
s = SRC.read_bytes().decode("utf-8")
assert "\r" not in s


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---- constants
rep('O_INTERFACE_MODES = ("full", "cut")\n', '''O_INTERFACE_MODES = ("full", "cut")

#: PREREG_V8F_LEVEL (`--level-head`, lever (a) of DECISION_K2): the speed LEVEL read from VISION. ``off`` (default) builds nothing = every earlier arm, bit for bit (state dict AND outputs). ``on`` builds a head
#: ``lvl_head`` (mean of the CURRENT tick frame's adapted tokens -> MLP(d, d, 1) -> v_hat, m/s; it reads vision tokens ONLY) and a conditioning embedding ``e_level`` (Linear(1, d), weight AND bias zero-initialised)
#: whose output O adds to its ego conditioning token. O reads ``v_hat.detach() / LEVEL_SCALE`` (train and inference alike: no exposure gap). The head's loss is ``w_level`` x Huber(v_hat, the LOGGED speed at NOW).
LEVEL_HEAD_MODES = ("off", "on")
LEVEL_SCALE = 10.0                         # m/s: e_level reads v_hat / LEVEL_SCALE
LEVEL_HUBER_DELTA = 1.0                    # m/s
#: the state-dict keys the lever adds (and ONLY these): an older checkpoint loads into a lever-ON model with exactly these keys missing (`--init-ckpt`)
LEVEL_KEY_PREFIXES = ("lvl_head.", "e_level.")
#: identifiers the level head's own code (``V8FMini._level``) must not mention: every ego / nav / limit / code / label input (the head is VISION ONLY; PREREG_V8F_LEVEL, CLAUDE.md "LABELS MAY USE EGO; INFERENCE IS VISION-ONLY")
LEVEL_FORBIDDEN_NAMES = ("p10f", "nav", "limit", "limit_ok", "act", "pose", "pose_now", "pose_t", "pose_s", "base", "kin3", "a_t", "a_s", "code", "ck", "_ego", "e_ego", "e_nav", "e_lim", "e_age", "e_ck", "e_code",
                         "level_y", "level_ok", "gt", "lab", "lab_s", "lab_t", "prior")
#: the only batch key the head's ``_level`` may read besides the adapted tokens it is handed (``feats`` is read by ``_encode``, upstream of it)
LEVEL_ALLOWED_KEYS = frozenset({"frame_ok"})
''')

# ---- config fields
rep('''    @property
    def window_start_effective(self) -> int:''', '''    level_head: str = "off"            # PREREG_V8F_LEVEL (`--level-head`): off = nothing built (every earlier arm, bit for bit); on = the vision-only speed-level head + O's level conditioning (LEVEL_HEAD_MODES)
    w_level: float = 0.0               # PREREG_V8F_LEVEL (`--w-level`): the weight of Huber(v_hat, logged speed) in the total. Must be > 0 with the head on and exactly 0.0 with it off (checked by the launch gate: a head with no loss, or a weight with no head, is refused)

    @property
    def window_start_effective(self) -> int:''')
rep('''        if self.hmax < self.hist + self.max_age_s:
            raise ValueError("hmax must cover hist + max_age_s")''', '''        if self.level_head not in LEVEL_HEAD_MODES:
            raise ValueError(f"level_head must be one of {LEVEL_HEAD_MODES}, got {self.level_head!r}")
        if not (math.isfinite(self.w_level) and self.w_level >= 0.0):
            raise ValueError(f"w_level must be a finite number >= 0, got {self.w_level!r}")
        if self.hmax < self.hist + self.max_age_s:
            raise ValueError("hmax must cover hist + max_age_s")''')

# ---- build (LAST in __init__, so every pre-existing module draws the same init as the off model)
rep('''        self.wm = WMAux(cfg) if cfg.wm != "off" else None
        self.p_gt = 1.0
''', '''        self.wm = WMAux(cfg) if cfg.wm != "off" else None
        self.lvl_head = None
        self.e_level = None
        if cfg.level_head == "on":                                         # PREREG_V8F_LEVEL: built LAST (the global RNG stream of every module above is the off model's)
            self.lvl_head = _mlp(d, d, 1)
            self.e_level = nn.Linear(1, d)
            nn.init.zeros_(self.e_level.weight)                            # zero-init: O's cond at step 0 is the off model's, bit for bit (x + 0.0 == x)
            nn.init.zeros_(self.e_level.bias)
        self.p_gt = 1.0
''')

# ---- _level (before the three layers)
rep('''    # ---- the three layers ---''', '''    def _level(self, b, adapt: Tensor, last: Tensor):
        """PREREG_V8F_LEVEL: ``(v_hat [B] float32 in m/s, v_in [B])``. The head reads the mean of the adapted tokens of the frame at ``last`` -- VISION ONLY (no ego, nav, limit, code or label: pinned by
        :func:`assert_level_head_vision_only`). ``v_in`` is what O is handed: ``v_hat.detach() / LEVEL_SCALE``, 0 where that frame is not a real frame."""
        tok = self._at(adapt, last)
        v_hat = self.lvl_head(tok.mean(1)).squeeze(-1).float()
        ok = self._at(b["frame_ok"], last)
        v_in = torch.where(ok, v_hat.detach(), torch.zeros_like(v_hat)) / LEVEL_SCALE
        return v_hat, v_in

    # ---- the three layers ---''')

# ---- run_o
rep('''        cond = torch.stack([self._ego(self._at(b["p10f"], i)), self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                            self._ck_cond(ck_o, ck_ok), self.e_code(code), self.e_age(ages)], 1)''', '''        ego_tok = self._ego(self._at(b["p10f"], i))
        if self.e_level is not None:                                    # PREREG_V8F_LEVEL: O reads e_level(v_hat.detach() / 10), ADDED to its ego conditioning token (zero-init: x + 0.0 == x, so the step-0 forward is the off model's;
            # an APPENDED token cannot be: it changes cond.mean(1), the query, and adds a softmax key). ``run_o`` keeps its six-argument signature (the trainer's probes, the T1 sub-eval and the tests wrap it) and computes
            # the level itself from ``adapt`` -- the same value ``_forward`` computes for the loss, so every caller of ``run_o`` gets the lever.
            ego_tok = ego_tok + self.e_level(self._level(b, adapt, i)[1][:, None])
        cond = torch.stack([ego_tok, self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                            self._ck_cond(ck_o, ck_ok), self.e_code(code), self.e_age(ages)], 1)''')

# ---- _forward
rep('''        out = {}
        # ---- S at its tick
''', '''        out = {}
        if self.lvl_head is not None:                                   # PREREG_V8F_LEVEL: v_hat from the CURRENT tick frame's vision tokens (for the loss); O recomputes the same value inside run_o and reads v_hat.detach() / 10
            out["level_v"] = self._level(b, adapt, last_now)[0]
        # ---- S at its tick
''')

# ---- loss
rep('''        # consistency penalties
        lim = limit_ms_from_onehot''', '''        if cfg.level_head == "on":                                   # PREREG_V8F_LEVEL: computed ONLY when the head is built (off = no new tensor op, no new T key = the tip, bit for bit)
            if "level_y" not in batch or "level_ok" not in batch:
                raise ValueError("level_head on but the batch carries no `level_y` / `level_ok`: the trainer computes them from the LOGGED speed before any ego stage (v8f_mini_train.py level_stage)")
            ly = torch.nan_to_num(batch["level_y"].float())
            lok = (batch["level_ok"] & batch["frame_ok"][:, -1]).float()
            T["level"] = (F.huber_loss(out["level_v"], ly, reduction="none", delta=LEVEL_HUBER_DELTA) * lok).sum() / lok.sum().clamp_min(1.0)
        # consistency penalties
        lim = limit_ms_from_onehot''')
rep('''        if cfg.consistency:
            total = total + cfg.w_pen * sum''', '''        total = total + cfg.w_level * T["level"] if cfg.level_head == "on" else total          # PREREG_V8F_LEVEL: ONE line (a mutation deletes it); head off adds nothing: the tip's total
        if cfg.consistency:
            total = total + cfg.w_pen * sum''')

# ---- groups and levers
rep('''        if self.wm is not None:
            g["WM"] = [self.wm]
        return g''', '''        if self.wm is not None:
            g["WM"] = [self.wm]
        if self.lvl_head is not None:
            g["LEVEL"] = [self.lvl_head, self.e_level]
        return g''')
rep('        return {"navonly": (not self.read_s.uses_scene)', '        lv = {"navonly": (not self.read_s.uses_scene)')
rep('''"w_tac": cfg.w_tac, "w_cons": cfg.w_cons}}                       # PREREG_V8F_WC: T's constraint term is w_tac x w_cons
''', '''"w_tac": cfg.w_tac, "w_cons": cfg.w_cons}}                       # PREREG_V8F_WC: T's constraint term is w_tac x w_cons
        if self.lvl_head is not None:                                      # PREREG_V8F_LEVEL: present ONLY with the head built (a flag-off model keeps its key set); the gate reads built.get("level_head", "off")
            lv["level_head"], lv["w_level"] = "on", cfg.w_level
            lv["level_token_zero_init"] = bool(float(self.e_level.weight.detach().abs().sum()) == 0.0 and float(self.e_level.bias.detach().abs().sum()) == 0.0)
            lv["level_vision_only_source"] = not level_head_source_problems()
        return lv
''')

# ---- module-level guard (appended)
s = s.rstrip("\n") + '''


def level_head_source_problems(fn=None) -> list:
    """PREREG_V8F_LEVEL admissibility, STATIC half: the identifiers / attributes / arguments / string subscripts of ``V8FMini._level``'s CODE (not its docstring) that name an ego, nav, limit, code or label input
    (:data:`LEVEL_FORBIDDEN_NAMES`) -- empty = the head's code reads vision only. The head is fed ``adapt`` (the adapted tokens) and ``b["frame_ok"]`` and nothing else."""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(V8FMini._level if fn is None else fn)))
    fn = tree.body[0]
    body = ast.Module(body=[st for st in fn.body if not (isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and isinstance(st.value.value, str))], type_ignores=[])
    names = set()
    for n in ast.walk(body):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Subscript) and isinstance(getattr(n, "slice", None), ast.Constant) and isinstance(n.slice.value, str):
            names.add(n.slice.value)
    return sorted(names & set(LEVEL_FORBIDDEN_NAMES))


def assert_level_head_vision_only(model: V8FMini, batch: dict) -> dict:
    """PREREG_V8F_LEVEL admissibility (binding: labels may use ego, inference is vision-only): the level head's v_hat is a function of the VISION tokens alone. Raises AssertionError unless
      (1) :func:`level_head_source_problems` is empty;
      (2) running ``model._level`` on a key-guarded batch READS only :data:`LEVEL_ALLOWED_KEYS`;
      (3) v_hat is IDENTICAL (bit for bit) when every non-vision batch key (ego history, nav, limit, actions, poses, base, kin3, ages, labels) is replaced by a different value, and DIFFERENT when ``feats`` is.
    Returns the record the launch gate carries."""
    if model.lvl_head is None:
        raise AssertionError("assert_level_head_vision_only: the model has no level head")
    src = level_head_source_problems(type(model)._level)               # the code of the model's OWN class (a subclass that rewires the head is read, not the base)
    if src:
        raise AssertionError(f"the level head's code names non-vision inputs: {src}")
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            Fn = batch["feats"].shape[1]
            last = torch.full((batch["feats"].shape[0],), Fn - 1, dtype=torch.long, device=batch["feats"].device)
            g = _KeyGuard(batch)
            adapt = model._encode(g)
            g.read.clear()
            v0, _ = model._level(g, adapt, last)
            extra = g.read - LEVEL_ALLOWED_KEYS
            if extra:
                raise AssertionError(f"the level head READS batch keys {sorted(extra)} (only {sorted(LEVEL_ALLOWED_KEYS)} are admissible)")
            other = {}
            for k, v in batch.items():
                if k in ("feats", "frame_ok") or not torch.is_tensor(v):
                    other[k] = v
                elif v.is_floating_point():
                    other[k] = torch.randn_like(v) * 3.0 + 1.0
                elif v.dtype == torch.bool:
                    other[k] = ~v
                else:
                    other[k] = v + 1
            v1, _ = model._level(other, model._encode(other), last)
            if not torch.equal(v0, v1):
                raise AssertionError(f"v_hat moved by {float((v0 - v1).abs().max())} when only non-vision inputs changed: the level head reads a non-vision input")
            moved = dict(batch)
            moved["feats"] = batch["feats"] + torch.randn(batch["feats"].shape, generator=torch.Generator().manual_seed(3)).to(batch["feats"].dtype) * 0.5
            v2, _ = model._level(moved, model._encode(moved), last)
            if torch.equal(v0, v2):
                raise AssertionError("v_hat did not move when the vision features changed: the head does not read the frame")
    finally:
        model.train(was)
    return {"keys_read": sorted(g.read), "source_problems": src, "vision_sensitivity": float((v0 - v2).abs().max())}
'''
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes(s.encode("utf-8"))
print("wrote", DST, len(s))
