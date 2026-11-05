"""Patch 1 of 3 (lead head): stack/tanitad/models/v8f_mini.py of the tip 219f6c7 -> the lead-head model.

usage: python -I pb_model.py <path of the tip file> <path of the file to write>
Every replacement must match EXACTLY ONCE (a drifted tip is refused, not guessed at)."""
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
t = src.read_bytes().decode("utf-8")
assert "\r\n" not in t


def rep(old, new):
    global t
    n = t.count(old)
    assert n == 1, f"{n} matches for: {old[:90]!r}"
    t = t.replace(old, new)


# ---- constants
rep('''LEVEL_ALLOWED_KEYS = frozenset({"frame_ok"})
''', '''LEVEL_ALLOWED_KEYS = frozenset({"frame_ok"})

#: PREREG_LEVELHEAD_A2 (`--lead-head`, `--w-lead`, `--aux-cond`; arm LLR): the LEAD head, a second vision-only head wired the same way as the level head. ``off`` (default) builds nothing = every earlier arm, bit for bit
#: (state dict AND outputs). ``on`` builds ``lead_head`` (mean of the CURRENT tick frame's adapted tokens -> MLP(d, d, 2) -> (presence logit for "a lead within LEAD_RANGE_M", gap_hat = sigmoid(.) = gap / LEAD_RANGE_M in
#: [0, 1])). O reads ``[sigmoid(logit), gap_hat]`` DETACHED. ``--aux-cond`` fixes HOW: ``add`` = ``e_lead`` (Linear(2, d), weight AND bias zero) ADDED to O's ego conditioning token like ``e_level``; ``token`` = ONE extra
#: conditioning token of O, ``e_aux`` (Linear(3, d), default non-zero init) of ``[v_hat / LEVEL_SCALE if the level head is built else 0, sigmoid(logit), gap_hat]`` (detached), which REPLACES the additive paths of BOTH heads
#: (``e_level`` and ``e_lead`` are not built). ``token`` is refused unless at least one head is on (the launch gate: bad_lead).
LEAD_HEAD_MODES = ("off", "on")
AUX_COND_MODES = ("add", "token")
LEAD_RANGE_M = 40.0                        # m: the presence label is "a lead within this range", the gap target is min(gap, this) / this
LEAD_HUBER_DELTA = 0.1                     # in gap / 40 units (= 4 m): the card names a Huber on the gap and no delta; this is the builder's choice, a module constant, pinned by a test
AUX_TOKEN_DIM = 3                          # [v_hat / LEVEL_SCALE, sigmoid(presence logit), gap_hat]
LEAD_KEY_PREFIXES = ("lead_head.", "e_lead.")
AUX_KEY_PREFIXES = ("e_aux.",)
#: every state-dict key a declared lever may ADD to an older checkpoint's key set (load_init_ckpt: EXACTLY these may be fresh)
NEW_LEVER_KEY_PREFIXES = LEVEL_KEY_PREFIXES + LEAD_KEY_PREFIXES + AUX_KEY_PREFIXES
#: identifiers the lead head's own code (``V8FMini._lead``) and the aux-token assembly (``V8FMini._aux_in``) must not mention (the level list + the lead labels): the head is VISION ONLY
LEAD_FORBIDDEN_NAMES = LEVEL_FORBIDDEN_NAMES + ("lead_y", "lead_has", "lead_near", "lead_gap_h", "s_h", "pose_now")
LEAD_ALLOWED_KEYS = frozenset({"frame_ok"})
''')

# ---- config fields
rep('''    w_level: float = 0.0               # PREREG_V8F_LEVEL (`--w-level`)''', '''    lead_head: str = "off"             # PREREG_LEVELHEAD_A2 (`--lead-head`): off = nothing built (every earlier arm, bit for bit); on = the vision-only lead head (presence + gap) + O's lead conditioning (LEAD_HEAD_MODES)
    w_lead: float = 0.0                # PREREG_LEVELHEAD_A2 (`--w-lead`): the weight of BCE(presence) + Huber(gap_hat) in the total. Must be > 0 with the head on and exactly 0.0 with it off (checked by the launch gate)
    aux_cond: str = "add"              # PREREG_LEVELHEAD_A2 (`--aux-cond`): how O reads the head outputs: add (e_level / e_lead ADDED to the ego token, zero-init) | token (ONE extra conditioning token, replaces both additive paths)
    w_level: float = 0.0               # PREREG_V8F_LEVEL (`--w-level`)''')

rep('''        if not (math.isfinite(self.w_level) and self.w_level >= 0.0):
            raise ValueError(f"w_level must be a finite number >= 0, got {self.w_level!r}")
''', '''        if not (math.isfinite(self.w_level) and self.w_level >= 0.0):
            raise ValueError(f"w_level must be a finite number >= 0, got {self.w_level!r}")
        if self.lead_head not in LEAD_HEAD_MODES:
            raise ValueError(f"lead_head must be one of {LEAD_HEAD_MODES}, got {self.lead_head!r}")
        if self.aux_cond not in AUX_COND_MODES:
            raise ValueError(f"aux_cond must be one of {AUX_COND_MODES}, got {self.aux_cond!r}")
        if not (math.isfinite(self.w_lead) and self.w_lead >= 0.0):
            raise ValueError(f"w_lead must be a finite number >= 0, got {self.w_lead!r}")
''')

# ---- build (LAST: no RNG shift of any older module; the level head's own draws come first, exactly as in the tip)
rep('''        self.lvl_head = None
        self.e_level = None
        if cfg.level_head == "on":                                         # PREREG_V8F_LEVEL: built LAST (the global RNG stream of every module above is the off model's)
            self.lvl_head = _mlp(d, d, 1)
            self.e_level = nn.Linear(1, d)
            nn.init.zeros_(self.e_level.weight)                            # zero-init: O's cond at step 0 is the off model's, bit for bit (x + 0.0 == x)
            nn.init.zeros_(self.e_level.bias)
''', '''        self.lvl_head = None
        self.e_level = None
        self.lead_head = None
        self.e_lead = None
        self.e_aux = None
        if cfg.level_head == "on":                                         # PREREG_V8F_LEVEL: built LAST (the global RNG stream of every module above is the off model's)
            self.lvl_head = _mlp(d, d, 1)
            if cfg.aux_cond == "add":                                      # PREREG_LEVELHEAD_A2: the ``token`` wiring replaces the additive path (e_level is not built: an unused module would be an untrained one)
                self.e_level = nn.Linear(1, d)
                nn.init.zeros_(self.e_level.weight)                        # zero-init: O's cond at step 0 is the off model's, bit for bit (x + 0.0 == x)
                nn.init.zeros_(self.e_level.bias)
        if cfg.lead_head == "on":                                          # PREREG_LEVELHEAD_A2: after the level head (its draws are the level-only model's)
            self.lead_head = _mlp(d, d, 2)
            if cfg.aux_cond == "add":
                self.e_lead = nn.Linear(2, d)
                nn.init.zeros_(self.e_lead.weight)
                nn.init.zeros_(self.e_lead.bias)
        if cfg.aux_cond == "token" and (self.lvl_head is not None or self.lead_head is not None):
            self.e_aux = nn.Linear(AUX_TOKEN_DIM, d)                       # default (non-zero) init: the extra token must carry signal from step 0
''')

# ---- _lead and _aux_in, after _level
rep('''        v_in = torch.where(ok, v_hat.detach(), torch.zeros_like(v_hat)) / LEVEL_SCALE
        return v_hat, v_in
''', '''        v_in = torch.where(ok, v_hat.detach(), torch.zeros_like(v_hat)) / LEVEL_SCALE
        return v_hat, v_in

    def _lead(self, b, adapt: Tensor, last: Tensor):
        """PREREG_LEVELHEAD_A2: ``(logit [B], gap_hat [B], lead_in [B, 2])``, float32. The head reads the mean of the adapted tokens of the frame at ``last`` -- VISION ONLY (no ego, nav, limit, code or label: pinned by
        :func:`assert_lead_head_vision_only`). ``gap_hat`` is sigmoid of the second output = gap / 40 m in [0, 1]. ``lead_in`` is what O is handed: ``[sigmoid(logit), gap_hat]`` DETACHED, zeros where that frame is not a real frame."""
        tok = self._at(adapt, last)
        o = self.lead_head(tok.mean(1)).float()
        logit, gap_hat = o[:, 0], torch.sigmoid(o[:, 1])
        ok = self._at(b["frame_ok"], last)
        lead_in = torch.where(ok[:, None], torch.stack([torch.sigmoid(logit.detach()), gap_hat.detach()], -1), torch.zeros_like(o))
        return logit, gap_hat, lead_in

    def _aux_in(self, b, adapt: Tensor, last: Tensor) -> Tensor:
        """PREREG_LEVELHEAD_A2 (``--aux-cond token``): the [B, 3] input of ``e_aux`` = ``[v_hat / LEVEL_SCALE (0 without the level head), sigmoid(presence logit), gap_hat (both 0 without the lead head)]``, detached,
        VISION ONLY (it reads ``_level`` / ``_lead`` and nothing else: pinned by :func:`aux_source_problems` and :func:`assert_aux_vision_only`)."""
        z = torch.zeros(adapt.shape[0], device=adapt.device, dtype=torch.float32)
        lv = self._level(b, adapt, last)[1] if self.lvl_head is not None else z
        ld = self._lead(b, adapt, last)[2] if self.lead_head is not None else torch.stack([z, z], -1)
        return torch.cat([lv[:, None], ld], -1)
''')

# ---- run_o
rep('''            ego_tok = ego_tok + self.e_level(self._level(b, adapt, i)[1][:, None])
        cond = torch.stack([ego_tok, self.e_nav(self._at(b["nav"], i)),
                            self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                            self._ck_cond(ck_o, ck_ok), self.e_code(code), self.e_age(ages)], 1)
''', '''            ego_tok = ego_tok + self.e_level(self._level(b, adapt, i)[1][:, None])
        if self.e_lead is not None:                                     # PREREG_LEVELHEAD_A2 (--aux-cond add): the same form for the lead head, e_lead([sigmoid(logit), gap_hat].detach()) ADDED to the ego token, zero-init
            ego_tok = ego_tok + self.e_lead(self._lead(b, adapt, i)[2])
        toks = [ego_tok, self.e_nav(self._at(b["nav"], i)),
                self.e_lim(torch.cat([b["limit"], b["limit_ok"][:, None].float()], -1)),
                self._ck_cond(ck_o, ck_ok), self.e_code(code), self.e_age(ages)]
        if self.e_aux is not None:                                      # PREREG_LEVELHEAD_A2 (--aux-cond token): ONE extra conditioning token built from both heads' detached outputs (default init: the step-0 forward differs from the init's by construction)
            toks.append(self.e_aux(self._aux_in(b, adapt, i)))
        cond = torch.stack(toks, 1)
''')

# ---- _forward
rep('''            out["level_v"] = self._level(b, adapt, last_now)[0]
''', '''            out["level_v"] = self._level(b, adapt, last_now)[0]
        if self.lead_head is not None:                                  # PREREG_LEVELHEAD_A2: (presence logit, gap_hat) from the CURRENT tick frame's vision tokens (for the loss); O recomputes the same values inside run_o
            out["lead_logit"], out["lead_gap"] = self._lead(b, adapt, last_now)[:2]
''')

# ---- loss
rep('''            T["level"] = (F.huber_loss(out["level_v"], ly, reduction="none", delta=LEVEL_HUBER_DELTA) * lok).sum() / lok.sum().clamp_min(1.0)
''', '''            T["level"] = (F.huber_loss(out["level_v"], ly, reduction="none", delta=LEVEL_HUBER_DELTA) * lok).sum() / lok.sum().clamp_min(1.0)
        if cfg.lead_head == "on":                                    # PREREG_LEVELHEAD_A2: computed ONLY when the head is built (off = no new tensor op, no new T key = the tip, bit for bit)
            if "lead_y" not in batch or "lead_has" not in batch or "lead_near" not in batch:
                raise ValueError("lead_head on but the batch carries no `lead_y` / `lead_has` / `lead_near`: the trainer computes them from the lead labels before any ego stage (v8f_mini_train.py lead_stage)")
            fok = batch["frame_ok"][:, -1]
            lhas = (batch["lead_has"] & fok).float()                                   # frames with a lead label: the presence BCE (positives AND negatives)
            lnear = (batch["lead_near"] & batch["lead_has"] & fok).float()            # frames with a lead label AND a lead within LEAD_RANGE_M: the gap Huber
            bce = F.binary_cross_entropy_with_logits(out["lead_logit"], batch["lead_near"].float(), reduction="none")
            hub = F.huber_loss(out["lead_gap"], torch.nan_to_num(batch["lead_y"].float()), reduction="none", delta=LEAD_HUBER_DELTA)
            T["lead_bce"] = (bce * lhas).sum() / lhas.sum().clamp_min(1.0)
            T["lead_huber"] = (hub * lnear).sum() / lnear.sum().clamp_min(1.0)
            T["lead"] = T["lead_bce"] + T["lead_huber"]
''')
rep('''        total = total + cfg.w_level * T["level"] if cfg.level_head == "on" else total          # PREREG_V8F_LEVEL: ONE line (a mutation deletes it); head off adds nothing: the tip's total
''', '''        total = total + cfg.w_level * T["level"] if cfg.level_head == "on" else total          # PREREG_V8F_LEVEL: ONE line (a mutation deletes it); head off adds nothing: the tip's total
        total = total + cfg.w_lead * T["lead"] if cfg.lead_head == "on" else total             # PREREG_LEVELHEAD_A2: ONE line (a mutation deletes it); head off adds nothing: the tip's total
''')

# ---- groups
rep('''        if self.lvl_head is not None:
            g["LEVEL"] = [self.lvl_head, self.e_level]
        return g
''', '''        if self.lvl_head is not None:
            g["LEVEL"] = [self.lvl_head] + ([self.e_level] if self.e_level is not None else [])      # token wiring: e_level is not built
        if self.lead_head is not None:                                              # PREREG_LEVELHEAD_A2
            g["LEAD"] = [self.lead_head] + ([self.e_lead] if self.e_lead is not None else [])
        if self.e_aux is not None:
            g["AUX"] = [self.e_aux]
        return g
''')

# ---- built levers
rep('''            lv["level_token_zero_init"] = bool(float(self.e_level.weight.detach().abs().sum()) == 0.0 and float(self.e_level.bias.detach().abs().sum()) == 0.0)
            lv["level_vision_only_source"] = not level_head_source_problems()
        return lv
''', '''            if self.e_level is not None:                                   # PREREG_LEVELHEAD_A2: absent under --aux-cond token (the additive path is not built)
                lv["level_token_zero_init"] = bool(float(self.e_level.weight.detach().abs().sum()) == 0.0 and float(self.e_level.bias.detach().abs().sum()) == 0.0)
            lv["level_vision_only_source"] = not level_head_source_problems()
        if self.lead_head is not None:                                     # PREREG_LEVELHEAD_A2: present ONLY with the head built (a flag-off model keeps its key set); the gate reads built.get("lead_head", "off")
            lv["lead_head"], lv["w_lead"] = "on", cfg.w_lead
            lv["lead_vision_only_source"] = not lead_head_source_problems()
            if self.e_lead is not None:
                lv["lead_token_zero_init"] = bool(float(self.e_lead.weight.detach().abs().sum()) == 0.0 and float(self.e_lead.bias.detach().abs().sum()) == 0.0)
        if self.lead_head is not None or self.e_aux is not None:           # the BUILT wiring, read back by the gate (absent = the additive default with no head: a token declared with no head builds nothing and is refused)
            lv["aux_cond"] = "token" if self.e_aux is not None else "add"
            lv["aux_vision_only_source"] = not aux_source_problems()
            if self.e_aux is not None:
                lv["aux_token_dim"] = int(self.e_aux.in_features)
        return lv
''')

# ---- the guards, appended after assert_level_head_vision_only (end of file)
t = t.rstrip("\n") + '''


def _forbidden_in(fn, forbidden) -> list:
    """The identifiers / attributes / string subscripts of ``fn``'s CODE (not its docstring) that are in ``forbidden`` (sorted)."""
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    fdef = tree.body[0]
    body = ast.Module(body=[st for st in fdef.body if not (isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) and isinstance(st.value.value, str))], type_ignores=[])
    names = set()
    for n in ast.walk(body):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
        elif isinstance(n, ast.Subscript) and isinstance(getattr(n, "slice", None), ast.Constant) and isinstance(n.slice.value, str):
            names.add(n.slice.value)
    return sorted(names & set(forbidden))


def lead_head_source_problems(fn=None) -> list:
    """PREREG_LEVELHEAD_A2 admissibility, STATIC half: the identifiers of ``V8FMini._lead``'s CODE that name an ego, nav, limit, code or label input (:data:`LEAD_FORBIDDEN_NAMES`) -- empty = the head's code reads vision only."""
    return _forbidden_in(V8FMini._lead if fn is None else fn, LEAD_FORBIDDEN_NAMES)


def aux_source_problems(fn=None) -> list:
    """The same static half for the aux-token assembly ``V8FMini._aux_in`` (it may read only what ``_level`` / ``_lead`` return)."""
    return _forbidden_in(V8FMini._aux_in if fn is None else fn, LEAD_FORBIDDEN_NAMES)


def _vision_only_run(model: V8FMini, batch: dict, call, allowed, what: str) -> dict:
    """Shared body of the two runtime guards: ``call(b, adapt, last)`` -> a tensor. It must read only ``allowed`` batch keys, be bit-identical when every non-vision key changes, and move when ``feats`` does."""
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            Fn = batch["feats"].shape[1]
            last = torch.full((batch["feats"].shape[0],), Fn - 1, dtype=torch.long, device=batch["feats"].device)
            g = _KeyGuard(batch)
            adapt = model._encode(g)
            g.read.clear()
            v0 = call(g, adapt, last)
            extra = g.read - allowed
            if extra:
                raise AssertionError(f"the {what} READS batch keys {sorted(extra)} (only {sorted(allowed)} are admissible)")
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
            v1 = call(other, model._encode(other), last)
            if not torch.equal(v0, v1):
                raise AssertionError(f"the {what} moved by {float((v0 - v1).abs().max())} when only non-vision inputs changed: it reads a non-vision input")
            moved = dict(batch)
            moved["feats"] = batch["feats"] + torch.randn(batch["feats"].shape, generator=torch.Generator().manual_seed(3)).to(batch["feats"].dtype) * 0.5
            v2 = call(moved, model._encode(moved), last)
            if torch.equal(v0, v2):
                raise AssertionError(f"the {what} did not move when the vision features changed: it does not read the frame")
    finally:
        model.train(was)
    return {"keys_read": sorted(g.read), "vision_sensitivity": float((v0 - v2).abs().max())}


def assert_lead_head_vision_only(model: V8FMini, batch: dict) -> dict:
    """PREREG_LEVELHEAD_A2 admissibility (binding: labels may use ego, inference is vision-only): the lead head's outputs (logit, gap_hat, and the [B, 2] input O is handed) are a function of the VISION tokens alone. Raises
    AssertionError unless (1) :func:`lead_head_source_problems` is empty; (2) running ``model._lead`` on a key-guarded batch READS only :data:`LEAD_ALLOWED_KEYS`; (3) its outputs are IDENTICAL (bit for bit) when every non-vision
    batch key (ego history, nav, limit, actions, poses, base, kin3, ages, labels) is replaced by a different value, and DIFFERENT when ``feats`` is. Returns the record the launch gate carries."""
    if model.lead_head is None:
        raise AssertionError("assert_lead_head_vision_only: the model has no lead head")
    src = lead_head_source_problems(type(model)._lead)                 # the code of the model's OWN class (a subclass that rewires the head is read, not the base)
    if src:
        raise AssertionError(f"the lead head's code names non-vision inputs: {src}")
    def call(b, adapt, last):
        lg, _, li = model._lead(b, adapt, last)
        return torch.cat([lg[:, None], li], -1)

    rec = _vision_only_run(model, batch, call, LEAD_ALLOWED_KEYS, "lead head")
    return {**rec, "source_problems": src}


def assert_aux_vision_only(model: V8FMini, batch: dict) -> dict:
    """PREREG_LEVELHEAD_A2 admissibility for ``--aux-cond token``: the [B, 3] vector ``e_aux`` reads is a function of the VISION tokens alone (same three conditions as :func:`assert_lead_head_vision_only`)."""
    if model.e_aux is None:
        raise AssertionError("assert_aux_vision_only: the model has no aux token")
    src = aux_source_problems(type(model)._aux_in)
    if src:
        raise AssertionError(f"the aux token's assembly names non-vision inputs: {src}")
    rec = _vision_only_run(model, batch, lambda b, adapt, last: model._aux_in(b, adapt, last), LEAD_ALLOWED_KEYS | LEVEL_ALLOWED_KEYS, "aux token input")
    return {**rec, "source_problems": src}
'''
dst.parent.mkdir(parents=True, exist_ok=True)
assert "\r\n" not in t
dst.write_bytes(t.encode("utf-8"))
print("wrote", dst, len(t))
