"""edit_n1dm_code.py -- anchored, counted edits for PREREG_V8F_N1DM on the 1f217bf0 export (model, trainer). Every edit asserts its anchor occurs EXACTLY the stated number of times."""
import sys
TREE = sys.argv[1]


def sub(rel, old, new, count=1):
    p = f"{TREE}/{rel}"
    s = open(p, encoding="utf-8", newline="").read()
    n = s.count(old)
    assert n == count, (rel, n, count, old[:90])
    open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))


MOD = "stack/tanitad/models/v8f_mini.py"
TRN = "stack/scripts/v8f_mini_train.py"

# ---------------------------------------------------------------- model
sub(MOD, 'AUX_ROUTE_MODES = ("off", "cond")\nROUTE_KEY_PREFIXES = ("route_head.", "route_emb.")\n',
    '#: PREREG_V8F_N1DM adds ``modes``: route_head + K = N_ROUTE zero-init residual plan branches (``route_br.k``), NO route_emb. Branch k is Linear(n_q * d, 2 * nb): the width of head_o\'s output (the coefficient vector before the basis).\n'
    'AUX_ROUTE_MODES = ("off", "cond", "modes")\nROUTE_KEY_PREFIXES = ("route_head.", "route_emb.")\nROUTE_BR_KEY_PREFIXES = ("route_br.",)\n')
sub(MOD, "+ O_CODE_KEY_PREFIXES + ROUTE_KEY_PREFIXES      #", "+ O_CODE_KEY_PREFIXES + ROUTE_KEY_PREFIXES + ROUTE_BR_KEY_PREFIXES      #")
sub(MOD, '        self.route_emb = None\n', '        self.route_emb = None\n        self.route_br = None                                               # PREREG_V8F_N1DM: the K residual plan branches of the ``modes`` mode (None in every other mode)\n')
sub(MOD, '                self.route_emb = nn.Linear(N_ROUTE, fl, bias=False)\n            nn.init.zeros_(self.route_emb.weight)\n',
    '                if cfg.aux_route == "cond":\n'
    '                    self.route_emb = nn.Linear(N_ROUTE, fl, bias=False)\n'
    '                else:                                                      # PREREG_V8F_N1DM ``modes``: K branches of head_o, Linear(fl -> head_o\'s output width), no bias, built under the same forked RNG AFTER route_head\n'
    '                    self.route_br = nn.ModuleList([nn.Linear(fl, int(self.head_o[2].out_features), bias=False) for _ in range(N_ROUTE)])\n'
    '            if self.route_emb is not None:\n'
    '                nn.init.zeros_(self.route_emb.weight)\n'
    '            else:\n'
    '                for br in self.route_br:\n'
    '                    nn.init.zeros_(br.weight)                              # step 0: every branch is 0.0, so coef + 0.0 == coef (the N1D plan, bit for bit)\n')
sub(MOD, '                hf = hf + self.route_emb(torch.softmax(self._route_logits, -1).detach())',
    '                if self.route_emb is not None:\n                    hf = hf + self.route_emb(torch.softmax(self._route_logits, -1).detach())', 1)
sub(MOD, '            coef = self.head_o(hf).reshape(-1, 2, cfg.nb) * cfg.res_scale_m      # [B, 2, nb]\n',
    '            coef = self.head_o(hf)\n'
    '            if self.route_br is not None:                               # PREREG_V8F_N1DM: ONE block (a mutation deletes the add): plan coefficients = head_o(h) + sum_k p_k * branch_k(h), p = softmax(route_head(h)) DETACHED (CE alone trains the route head),\n'
    '                p_mode = torch.softmax(self._route_logits.float(), -1).detach()       # the SAME soft form at training and at inference (no argmax); p reads h, an inference-input function, and no label source\n'
    '                for k, br in enumerate(self.route_br):\n'
    '                    coef = coef + p_mode[:, k:k + 1] * br(hf)\n'
    '            coef = coef.reshape(-1, 2, cfg.nb) * cfg.res_scale_m      # [B, 2, nb]\n')
sub(MOD, 'g["ROUTE"] = [self.route_head, self.route_emb]', 'g["ROUTE"] = [m_ for m_ in (self.route_head, self.route_emb, self.route_br) if m_ is not None]')

# ---------------------------------------------------------------- trainer
sub(TRN, '"""argparse type of ``--aux-route``: off | cond. ``ce``', '"""argparse type of ``--aux-route``: off | cond | modes (modes: PREREG_V8F_N1DM). ``ce``')
sub(TRN, 'help="PREREG_V8F_N1DH: off (default) = nothing built = every earlier arm, bit for bit; cond =',
    'help="PREREG_V8F_N1DM: modes = route_head (CE, own-code rows) + K = 3 zero-init residual plan branches of head_o, plan = head_o(h) + sum_k p_k branch_k(h), p = softmax(route_head(h)).detach(), NO route_emb. PREREG_V8F_N1DH: off (default) = nothing built = every earlier arm, bit for bit; cond =')
sub(TRN, "Needs --w-aux-route > 0, --code-mix whole", "Needs (cond / modes) --w-aux-route > 0, --code-mix whole")

old_present = '    present = getattr(model, "route_head", None) is not None and getattr(model, "route_emb", None) is not None\n    if lv != a.aux_route or present != on or '
new_present = ('    has_emb, has_br = getattr(model, "route_emb", None) is not None, getattr(model, "route_br", None) is not None\n'
               '    present = getattr(model, "route_head", None) is not None and (has_emb if a.aux_route == "cond" else has_br if a.aux_route == "modes" else False)\n'
               '    wrong_extra = has_emb != (a.aux_route == "cond") or has_br != (a.aux_route == "modes")          # PREREG_V8F_N1DM: route_emb iff cond, route_br iff modes, never both\n'
               '    if lv != a.aux_route or present != on or wrong_extra or ')
sub(TRN, old_present, new_present)
sub(TRN, "modules {'present' if present else 'absent'})\")", "modules {'present' if present else 'absent'}, route_emb {has_emb}, route_br {has_br})\")")
old_shapes = '''    if tuple(model.route_emb.weight.shape) != (want, M.N_ROUTE) or model.route_emb.bias is not None:
        bad.append(f"aux_route: route_emb is {tuple(model.route_emb.weight.shape)} (bias {model.route_emb.bias is not None}), declared ({want}, {M.N_ROUTE}) without a bias")
'''
new_shapes = '''    if a.aux_route == "cond" and (tuple(model.route_emb.weight.shape) != (want, M.N_ROUTE) or model.route_emb.bias is not None):
        bad.append(f"aux_route: route_emb is {tuple(model.route_emb.weight.shape)} (bias {model.route_emb.bias is not None}), declared ({want}, {M.N_ROUTE}) without a bias")
    if a.aux_route == "modes":                                                                      # PREREG_V8F_N1DM: K = N_ROUTE branches, each [head_o's output width, n_q * d], no bias, ZERO at the start
        do = int(model.head_o[2].out_features)
        if len(model.route_br) != M.N_ROUTE:
            bad.append(f"aux_route modes: {len(model.route_br)} branches built, declared {M.N_ROUTE}")
        for k, br in enumerate(model.route_br):
            if tuple(br.weight.shape) != (do, want) or br.bias is not None:
                bad.append(f"aux_route modes: route_br.{k} is {tuple(br.weight.shape)} (bias {br.bias is not None}), declared ({do}, {want}) without a bias (do = head_o's output width)")
            if float(br.weight.detach().abs().sum()) != 0.0:
                bad.append(f"aux_route modes: route_br.{k}.weight is not zero at the start (the step-0 plan would differ from N1D's)")
'''
sub(TRN, old_shapes, new_shapes)
sub(TRN, '''    if int(model.head_o[0].in_features) != want:
        bad.append(f"aux_route: head_o reads {int(model.head_o[0].in_features)} features but route_emb writes {want}")
    if float(model.route_emb.weight.detach().abs().sum()) != 0.0:
        bad.append("aux_route: route_emb.weight is not zero at the start (the step-0 forward would differ from the parent's)")
''', '''    if int(model.head_o[0].in_features) != want:
        bad.append(f"aux_route: head_o reads {int(model.head_o[0].in_features)} features but the route modules read/write {want}")
    if a.aux_route == "cond" and float(model.route_emb.weight.detach().abs().sum()) != 0.0:
        bad.append("aux_route: route_emb.weight is not zero at the start (the step-0 forward would differ from the parent's)")
''')
sub(TRN, '''        for nm, p in (("route_head.weight", model.route_head.weight), ("route_head.bias", model.route_head.bias), ("route_emb.weight", model.route_emb.weight)):''',
    '''        for nm, p in (("route_head.weight", model.route_head.weight), ("route_head.bias", model.route_head.bias)) + ((("route_emb.weight", model.route_emb.weight),) if a.aux_route == "cond" else tuple((f"route_br.{k}.weight", br.weight) for k, br in enumerate(model.route_br))):''')

# probe
sub(TRN, '''    def agrad(t_, w):
        g = torch.autograd.grad(t_, [w], retain_graph=True, allow_unused=True)[0]
        return 0.0 if g is None else float(g.abs().sum())
''', '''    modes = model.route_br is not None                                                                  # PREREG_V8F_N1DM: the conditioning path is the K branches (cond: route_emb)

    def agrad(t_, w):
        ws = list(w) if isinstance(w, (list, tuple)) else [w]
        gs = torch.autograd.grad(t_, ws, retain_graph=True, allow_unused=True)
        return float(sum(float(g.abs().sum()) for g in gs if g is not None))

    def cond_params(mm):
        return [br.weight for br in mm.route_br] if mm.route_br is not None else [mm.route_emb.weight]
''')
sub(TRN, '''        we = mm.route_emb.weight if mm.route_emb is not None else None
''', '''        we = cond_params(mm) if (mm.route_emb is not None or mm.route_br is not None) else None
''')
sub(TRN, '''            r["emb_total"], r["emb_ade"] = agrad(T["total"], we), agrad(T["ade"], we)
''', '''            r["emb_total"], r["emb_ade"] = agrad(T["total"], we), agrad(T["ade"], we)
            r["emb_each"] = [agrad(T["total"], w_) for w_ in we]                                         # one entry per conditioning module (cond: one; modes: each of the K branches)
''')
sub(TRN, '''    m_off.route_head, m_off.route_emb, m_off.cfg = None, None, dataclasses.replace(m.cfg, aux_route="off", w_aux_route=0.0)''',
    '''    m_off.route_head, m_off.route_emb, m_off.route_br, m_off.cfg = None, None, None, dataclasses.replace(m.cfg, aux_route="off", w_aux_route=0.0)''')
sub(TRN, '''    with torch.no_grad():
        m.route_emb.weight.copy_(torch.randn(m.route_emb.weight.shape, generator=torch.Generator().manual_seed(8)) * 0.05)
    o2 = run(m, batch, 0.0)''', '''    with torch.no_grad():
        for w_ in cond_params(m):
            w_.copy_(torch.randn(w_.shape, generator=torch.Generator().manual_seed(8)) * 0.05)
    o2 = run(m, batch, 0.0)''')
sub(TRN, '''    r3 = grads(m, o3)
    res = {"head_grad_total"''', '''    r3 = grads(m, o3)
    plan_perm = None
    if modes:                                                                                           # PREREG_V8F_N1DM: p permuted across ROWS must change the plan (the selection reaches the output per row)
        hk = m.route_head.register_forward_hook(lambda mod_, i_, out_: out_.roll(1, 0))
        try:
            o_perm = run(m, batch, 0.0)
        finally:
            hk.remove()
        plan_perm = float((o_perm["traj30"] - o2["traj30"]).detach().abs().max())
    res = {"head_grad_total"''')
sub(TRN, '''"control_real_init_emb_grad": r0["emb_total"],''', '''"control_real_init_emb_grad": r0["emb_total"], "real_init_emb_each": r0["emb_each"], "emb_each": r1["emb_each"], "plan_move_when_p_permuted": plan_perm,''')
sub(TRN, '''    if head_zero_at_init and r0["emb_total"] != 0.0:
        raise AssertionError''', '''    if modes:                                                                                           # PREREG_V8F_N1DM: the branch add is BEHIND head_o's zero last layer, so every branch reads a NON-ZERO gradient at the REAL init (no seeding needed), and p permuted across rows moves the plan
        if not all(g_ > 0.0 for g_ in res["real_init_emb_each"]) or not all(g_ > 0.0 for g_ in res["emb_each"]):
            raise AssertionError(f"aux_route probe: a plan branch has no gradient (real init {res['real_init_emb_each']!r}, seeded {res['emb_each']!r}): the branch add does not reach the loss")
        if not (plan_perm is not None and plan_perm > 0.0):
            raise AssertionError(f"aux_route probe: permuting p across rows moves the plan by {plan_perm!r} (must be > 0): the selection does not reach the output per row")
    elif head_zero_at_init and r0["emb_total"] != 0.0:
        raise AssertionError''')
print("code edits ok")
