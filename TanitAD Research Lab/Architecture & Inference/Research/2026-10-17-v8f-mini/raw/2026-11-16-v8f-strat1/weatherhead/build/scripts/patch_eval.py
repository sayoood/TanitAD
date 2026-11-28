"""patch the two evaluators with --ctx-pool (PREREG_V8F_WEATHERHEAD 4.4 item 1 and 4)"""
R = "F:/TanitAD-scratch/strat1/whead/tree/stack/scripts/"


def patch(fn, pairs):
    s = open(R + fn, encoding="utf-8", newline="").read()
    for old, new in pairs:
        assert s.count(old) == 1, (fn, old[:70], s.count(old))
        s = s.replace(old, new)
    open(R + fn, "w", encoding="utf-8", newline="").write(s)


CHECK = '''    if a.ctx_pool != cfg.ctx_pool:                    # PREREG_V8F_WEATHERHEAD: the head the checkpoint carries is the head the run declares (a cfg that drops it, or a flag that names a head the checkpoint lacks, is refused)
        raise SystemExit(f"REFUSED: --ctx-pool {a.ctx_pool} but the checkpoint's cfg has ctx_pool={cfg.ctx_pool!r}: the evaluator would silently skip (or invent) the pooled head")
'''
FLAG = '''    p.add_argument("--ctx-pool", choices=("off", "on"), default="off", help="PREREG_V8F_WEATHERHEAD: on = the checkpoint carries the attention-pooled detected-information head and the eval writes ctx_pool_logits; must equal the checkpoint's cfg (a mismatch is refused)")
'''

# ---- T0 evaluator
patch("v8f_mini_eval.py", [
    ('''    cfg = M.V8FMiniConfig(**ck["cfg"])                 # a checkpoint written before the ruling has no kin_mode => steer
''', '''    cfg = M.V8FMiniConfig(**ck["cfg"])                 # a checkpoint written before the ruling has no kin_mode => steer
''' + CHECK),
    ('''    p.add_argument("--nav-withheld", action="store_true", help="PREREG_V8F_STRAT1: all five nav channels 0''', FLAG + '''    p.add_argument("--nav-withheld", action="store_true", help="PREREG_V8F_STRAT1: all five nav channels 0'''),
    ('''    if model.ctx_head is not None:                     # PREREG_V8F_STRAT1: the context logits ride in the eval npz (the reader scores them)
        outs["ctx_logits"] = []
''', '''    if model.ctx_head is not None:                     # PREREG_V8F_STRAT1: the context logits ride in the eval npz (the reader scores them)
        outs["ctx_logits"] = []
    if model.ctx_pool_head is not None:                # PREREG_V8F_WEATHERHEAD: the pooled head's logits ride in the eval npz
        outs["ctx_pool_logits"] = []
'''),
    ('''            if model.ctx_head is not None:
                outs["ctx_logits"].append(o["ctx_logits"].float().cpu().numpy())
''', '''            if model.ctx_head is not None:
                outs["ctx_logits"].append(o["ctx_logits"].float().cpu().numpy())
            if model.ctx_pool_head is not None:
                outs["ctx_pool_logits"].append(o["ctx_pool_logits"].float().cpu().numpy())
'''),
])

# ---- T1 evaluator
patch("v8f_t1_eval.py", [
    ('''            plan.heads = {k: out[k].float().cpu().numpy() for k in HEAD_KEYS + (("ctx_logits",) if "ctx_logits" in out else ())}      # PREREG_V8F_STRAT1: only a ctx model carries the key''',
     '''            plan.heads = {k: out[k].float().cpu().numpy() for k in HEAD_KEYS + (("ctx_logits",) if "ctx_logits" in out else ()) + (("ctx_pool_logits",) if "ctx_pool_logits" in out else ())}      # PREREG_V8F_STRAT1 / PREREG_V8F_WEATHERHEAD: only a ctx / pooled-head model carries the key'''),
    ('''    def __init__(self, planner, store, J, heads: bool = False, nav: bool = False, ctx: bool = False):
        self.planner, self.store, self.J, self.heads = planner, store, J, bool(heads)
        self.nav, self.ctx = bool(nav), bool(ctx) and bool(heads)       # PREREG_V8F_STRAT1: --dump-nav adds nav_in; ctx_logits ride with --dump-heads on a ctx model
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("nav_in",) if self.nav else ()) + (("ctx_logits",) if self.ctx else ())}''',
     '''    def __init__(self, planner, store, J, heads: bool = False, nav: bool = False, ctx: bool = False, pool: bool = False):
        self.planner, self.store, self.J, self.heads = planner, store, J, bool(heads)
        self.nav, self.ctx = bool(nav), bool(ctx) and bool(heads)       # PREREG_V8F_STRAT1: --dump-nav adds nav_in; ctx_logits ride with --dump-heads on a ctx model
        self.pool = bool(pool) and bool(heads)                          # PREREG_V8F_WEATHERHEAD: ctx_pool_logits ride with --dump-heads on a pooled-head model
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("nav_in",) if self.nav else ()) + (("ctx_logits",) if self.ctx else ()) + (("ctx_pool_logits",) if self.pool else ())}'''),
    ('''            if self.ctx:
                r["ctx_logits"].append(hd["ctx_logits"][b])
''', '''            if self.ctx:
                r["ctx_logits"].append(hd["ctx_logits"][b])
            if self.pool:
                r["ctx_pool_logits"].append(hd["ctx_pool_logits"][b])
'''),
    ('''    p.add_argument("--dump-nav", action="store_true", help="PREREG_V8F_STRAT1 (needs --dump-ticks)''', FLAG + '''    p.add_argument("--dump-nav", action="store_true", help="PREREG_V8F_STRAT1 (needs --dump-ticks)'''),
    ('''    cfg = M.V8FMiniConfig(**ck["cfg"])
    if a.dump_nav and not a.dump_ticks:''', '''    cfg = M.V8FMiniConfig(**ck["cfg"])
''' + CHECK + '''    if a.dump_nav and not a.dump_ticks:'''),
    ('''ctx=model.ctx_head is not None)''', '''ctx=model.ctx_head is not None, pool=model.ctx_pool_head is not None)'''),
])
print("evaluators patched")
