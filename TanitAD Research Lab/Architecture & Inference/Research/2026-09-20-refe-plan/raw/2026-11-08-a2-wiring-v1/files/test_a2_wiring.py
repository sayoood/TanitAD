#!/usr/bin/env python3
"""A2 wiring tests (CPU).  Run: CUDA_VISIBLE_DEVICES=-1 python test_a2_wiring.py -> ZZTEST_PASS n.
Fixture: a tiny 12-row head over random features; the STOCK train_losses is a stand-in that calls the REAL expected_reward_loss with the stash main()'s losses6 passes (beh_cur.get('rfac'/'selfac')).
What is proven: G-BEH-LIVE (grad on weight[7:12] non-zero for B, exactly 0.0 for A and the w=0 control), G-BEH-ID (all bact False + listwise ON == pre-A2 loss bit for bit), mutation (delete the
w_beh line / the rfac line / the selfac line -> RED), leak grep (no bq/bact in model.py planner.py eval/), --beh-heads OFF leaves state_dict keys, OFF flags default, q_stall default exclusion."""
import ast, os, re, sys, types
import torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "scorer_finetune.py")
sys.path.insert(0, HERE)
import scorer_finetune as S

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)

def build(mod, w=1.0, listwise=True, stall=False, act_on=True, seed=0):
    torch.manual_seed(seed)
    B, M, D = 3, 8, 16
    feat = torch.randn(B, M, D)
    head = nn.Linear(D, 12)
    a = types.SimpleNamespace(lane_head=True)
    cur = dict(w=w, listwise=listwise, cols=(mod.BEH_REWARD_COLS_STALL if stall else mod.BEH_REWARD_COLS))
    tg = torch.rand(B, M, 7)
    bq = torch.rand(B, M, 5).clamp(0.2, 1.0)
    bact = (torch.rand(B, M, 5) > 0.3) if act_on else torch.zeros(B, M, 5, dtype=torch.bool)
    st = {"head": head}
    def fwd_stock(body):
        out = head(feat)
        return out[..., :7].detach() * 0, out, out, tg           # sxA = sxB = the shared 12-wide head output (arm A/B separation: see armA below)
    armA = nn.Linear(D, 12)
    def fwd_stock2(body):
        oB, oA = head(feat), armA(feat)
        return oB[..., :7].detach(), oA, oB, tg
    def stock(sx0, sxA, sxB, tg_, rows, upd, lam_pg=None, pair_w=None):
        lA = nn.functional.binary_cross_entropy_with_logits(sxA[..., :6], tg_[..., :6])
        bce = nn.functional.binary_cross_entropy_with_logits(sxB[..., :6], tg_[..., :6])
        er = mod.expected_reward_loss(sxB, tg_, 0.1, rfac=cur.get("rfac"), selfac=cur.get("selfac"))
        return lA, bce + 0.3 * er, er, torch.zeros(())
    fwd, tl = mod.make_beh_wrappers(a, cur, fwd_stock2, stock, "cpu")
    batch = (None,) * 9 + (bq, bact)
    return fwd, tl, stock, head, armA, cur, batch, tg

def grads(fwd, tl, head, armA, batch, **kw):
    head.zero_grad(); armA.zero_grad()
    s0, sA, sB, tg = fwd(batch)
    lA, lB, _l, _e = tl(s0, sA, sB, tg, None, 0, **kw)
    (lA + lB).backward()
    gB = float(head.weight.grad[7:12].abs().sum()); gA = float(armA.weight.grad[7:12].abs().sum())
    return gB, gA, float(lB)

def live(mod, **kw):
    fwd, tl, *_r = build(mod, **kw)
    head, armA, batch = _r[1], _r[2], _r[4]
    gB, gA, _ = grads(fwd, tl, head, armA, batch)
    fwd, tl, *_r = build(mod, **kw)
    g0, _, _ = grads(fwd, tl, _r[1], _r[2], _r[4], beh_w=0.0, beh_listwise=False)
    return gB, gA, g0

def load_src(src, name):
    m = types.ModuleType(name); m.__file__ = SRC; sys.modules[name] = m
    exec(compile(src, SRC, "exec"), m.__dict__)
    return m

# ---- G-BEH-LIVE
gB, gA, g0 = live(S, listwise=False)
check("G-BEH-LIVE: B grad on weight[7:12] non-zero", gB > 0.0)
check("G-BEH-LIVE: A grad on weight[7:12] EXACTLY 0.0", gA == 0.0)
check("G-BEH-LIVE: explicit w=0.0 control (graph live) EXACTLY 0.0", g0 == 0.0)
# listwise alone reaches rows 7:12 only through selfac
fwd, tl, stock, head, armA, cur, batch, tg = build(S, w=0.0, listwise=True)
g_lw, _, _ = grads(fwd, tl, head, armA, batch)
check("listwise product alone: grad on weight[7:12] non-zero (selfac reaches the head)", g_lw > 0.0)
fwd, tl, stock, head, armA, cur, batch, tg = build(S, w=0.0, listwise=True, stall=True)
check("q_stall default EXCLUDED from the listwise columns, flag adds col 3", S.BEH_REWARD_COLS == (0, 1, 2, 4) and cur["cols"] == (0, 1, 2, 3, 4))

# ---- G-BEH-ID: every bact False + listwise on == the pre-A2 loss bit for bit (value and gradient wrt sxB)
fwd, tl, stock, head, armA, cur, batch, tg = build(S, w=1.0, listwise=True, act_on=False)
head.zero_grad(); s0, sA, sB, t_ = fwd(batch); lA, lB, ln, _ = tl(s0, sA, sB, t_, None, 0)
g_new = torch.autograd.grad(lB, sB, retain_graph=True)[0][..., :6]
cur.pop("rfac", None); cur.pop("selfac", None)
lA0, lB0, ln0, _ = stock(s0, sA, sB, t_, None, 0)
g_old = torch.autograd.grad(lB0, sB)[0][..., :6]
check("G-BEH-ID: loss B bit-identical to pre-A2 (all bact False)", torch.equal(lB.detach(), lB0.detach()) and torch.equal(ln, ln0))
check("G-BEH-ID: d lossB / d sxB[:, :, :6] bit-identical", torch.equal(g_new, g_old))

# ---- mutation: delete the single line, require the probe to go RED
src = open(SRC, encoding="utf-8").read()
def mutated(line_re):
    n = len(re.findall(line_re, src, flags=re.M))
    assert n == 1, (line_re, n)
    return load_src(re.sub(line_re, "                pass", src, flags=re.M), "sf_mut")
m1 = mutated(r"^                lB = lB \+ w \* beh_bce\(cur\[\"sxB_beh\"\], cur\[\"bq\"\], cur\[\"bact\"\]\)$")
g1, _, _ = live(m1, listwise=False)
check("MUTATION delete `lB = lB + w * beh_bce(...)` -> G-BEH-LIVE RED (B grad == 0)", g1[0] == 0.0 if isinstance(g1, tuple) else g1 == 0.0)
m2 = mutated(r"^        sc = sc \* selfac$".replace("        sc", "        sc")) if False else None
s2 = src.replace("        sc = sc * selfac\n", "        pass\n", 1)
assert s2 != src
mm = load_src(s2, "sf_mut2")
gl, _, _ = grads(*(lambda t: (t[0], t[1], t[3], t[4], t[6]))(build(mm, w=0.0, listwise=True)))
check("MUTATION delete `sc = sc * selfac` -> listwise-only grad on weight[7:12] == 0 (RED)", gl == 0.0)
s3 = src.replace("        return -(pi * (agg_v1(tg[..., :6]) * rfac)).sum(1).mean()" + chr(10), "        pass" + chr(10), 1)
assert s3 != src
m3 = load_src(s3, "sf_mut3")
def lw_value(mod, bq_scale):
    fwd, tl, stock, head, armA, cur, batch, tg = build(mod, w=0.0, listwise=True)
    batch = batch[:9] + (batch[9] * bq_scale, batch[10])
    s0, sA, sB, t_ = fwd(batch); return float(tl(s0, sA, sB, t_, None, 0)[2])
check("listwise value depends on bq (rfac reaches the loss)", lw_value(S, 1.0) != lw_value(S, 0.5))
check("MUTATION delete `r = r * rfac` -> value no longer depends on bq (RED)", lw_value(m3, 1.0) == lw_value(m3, 0.5))

# ---- leak grep: the labels are in the loss only
base = os.path.dirname(HERE)
leaks = []
for rel in ("refe/model.py", "refe/planner.py"):
    p = os.path.join(base, rel)
    if os.path.isfile(p):
        t = open(p, encoding="utf-8").read()
        if re.search(r"\b(bq|bact|beh_)", t):
            leaks.append(rel)
for root, _d, fs in os.walk(os.path.join(base, "eval")):
    for f in fs:
        if f.endswith(".py") and re.search(r"\b(bq|bact)\b", open(os.path.join(root, f), encoding="utf-8", errors="replace").read()):
            leaks.append(os.path.join(root, f))
check(f"LEAK: no bq/bact/beh_ name in model.py, planner.py or eval/ ({leaks})", not leaks)
# --beh-heads OFF: the model's state_dict keys are the stock ones (the head is grown only inside `if a.beh_heads`)
tree = ast.parse(src)
main_src = ast.get_source_segment(src, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"))
pre = main_src.split("extend_beh_head(")[0].splitlines()[-6:]
check("OFF: extend_beh_head is called once, under `if a.beh_heads:`; defaults are OFF",
      main_src.count("extend_beh_head(") == 1 and any("if a.beh_heads:" in x for x in pre)
      and S.build_parser().get_default("beh_heads") is False and S.build_parser().get_default("w_beh") == 0.0 and S.build_parser().get_default("beh_listwise") is False)
m = nn.Module(); m.score_head = nn.Linear(4, 7)
k0 = sorted(m.state_dict()); S.extend_beh_head(m, "cpu", [0.0] * 5)
check("extend_beh_head: same state_dict KEYS, 12 rows, rows 0-6 verbatim, rows 7-11 zero weight", sorted(m.state_dict()) == k0 and m.score_head.weight.shape[0] == 12 and float(m.score_head.weight[7:].abs().sum()) == 0.0)
# ORDER pin
bt = os.path.join(HERE, "behaviour_terms.py")
if os.path.isfile(bt):
    check("BEH_ORDER == behaviour_terms.ORDER", __import__("behaviour_terms").ORDER == S.BEH_ORDER)

# ---- THE REAL CLOSURE: the nested defs of main() (sel_score, lane_bce, losses, losses6, pair_term, train_losses) are lifted by AST, as TEXT, out of scorer_finetune.main() and run on CPU with the same free
# variables main() gives them (a, compw, beh_cur, pairer, cap, pair_last).  What stays UNVERIFIED here: the code that builds the model / data around them (needs CUDA + a checkpoint).
import textwrap
CLOSURE_NAMES = ("sel_score", "lane_bce", "losses", "losses6", "pair_term", "train_losses")
def lift_closure(text):
    t = ast.parse(text)
    mn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    lines = text.splitlines()
    got = {}
    for n in mn.body:                                    # DIRECT children of main() only (the other `train_losses` at depth 2 is the speed-probe shim)
        if isinstance(n, ast.FunctionDef) and n.name in CLOSURE_NAMES:
            got[n.name] = "\n".join(lines[n.lineno - 1:n.end_lineno])
    assert set(got) == set(CLOSURE_NAMES), sorted(set(CLOSURE_NAMES) - set(got))
    fac = "def _factory(a, compw, beh_cur, pairer, cap, pair_last):\n" + "\n".join(got[k] for k in CLOSURE_NAMES) + "\n    return train_losses\n"
    return fac, got
def real_stock(mod_text, mod, a, beh_cur):
    fac, _ = lift_closure(mod_text)
    g = dict(mod.__dict__); exec(compile(fac, SRC + "::closure", "exec"), g)
    return g["_factory"](a, torch.ones(6), beh_cur, None, {}, {})
def build_real(mod_text, mod, w=1.0, listwise=True, act_on=True, seed=0):
    torch.manual_seed(seed)
    B, M, D = 3, 8, 16
    feat = torch.randn(B, M, D); head, armA = nn.Linear(D, 12), nn.Linear(D, 12)
    a = types.SimpleNamespace(lane_head=True, b_mode="expreward", lam=0.3, tau_s=0.1, lane_weight=1.0, pair_w=0.0, epo=False)
    cur = dict(w=w, listwise=listwise, cols=mod.BEH_REWARD_COLS)
    tg = torch.rand(B, M, 7); tg[..., 6] = -1.0
    bq = torch.rand(B, M, 5).clamp(0.2, 1.0); bact = (torch.rand(B, M, 5) > 0.3) if act_on else torch.zeros(B, M, 5, dtype=torch.bool)
    real = real_stock(mod_text, mod, a, cur)
    def fwd_stock(body):
        oB, oA = head(feat), armA(feat)
        return oB[..., :7].detach(), oA, oB, tg
    fwd, tl = mod.make_beh_wrappers(a, cur, fwd_stock, real, "cpu")
    return fwd, tl, real, head, armA, cur, (None,) * 9 + (bq, bact)
fwd, tl, real, head, armA, cur, batch = build_real(src, S, listwise=False)
gB, gA, _ = grads(fwd, tl, head, armA, batch)
fwd, tl, real, head, armA, cur, batch = build_real(src, S, listwise=False)
g0, _, _ = grads(fwd, tl, head, armA, batch, beh_w=0.0, beh_listwise=False)
check("REAL closure: G-BEH-LIVE B grad on weight[7:12] > 0, A exactly 0.0, explicit w=0 control exactly 0.0", gB > 0.0 and gA == 0.0 and g0 == 0.0)
fwd, tl, real, head, armA, cur, batch = build_real(src, S, w=0.0, listwise=True)
gl, _, _ = grads(fwd, tl, head, armA, batch)
check("REAL closure: listwise product alone (w=0) reaches weight[7:12] through losses6 -> expected_reward_loss", gl > 0.0)
fwd, tl, real, head, armA, cur, batch = build_real(src, S, w=1.0, listwise=True, act_on=False)
s0, sA, sB, t_ = fwd(batch); lA, lB, ln, _ = tl(s0, sA, sB, t_, None, 0)
gnew = torch.autograd.grad(lB, sB, retain_graph=True)[0][..., :6]
cur.pop("rfac", None); cur.pop("selfac", None)
lA0, lB0, ln0, _ = real(s0, sA, sB, t_, None, 0)
check("REAL closure: G-BEH-ID (all bact False + listwise) loss and d/dsxB[..., :6] bit-identical to the stock assembly",
      torch.equal(lB.detach(), lB0.detach()) and torch.equal(ln, ln0) and torch.equal(gnew, torch.autograd.grad(lB0, sB)[0][..., :6]))
# mutations of the REAL closure's text: the call in losses6 stops passing the stash -> the listwise lever is dead and the probes must go RED
call = 'expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score, rfac=beh_cur.get("rfac"), selfac=beh_cur.get("selfac"))'
assert src.count(call) == 1
mA = load_src(src.replace(call, 'expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score)'), "sf_mutA")
fwd, tl, real, head, armA, cur, batch = build_real(mA and open(SRC, encoding="utf-8").read().replace(call, 'expected_reward_loss(sxB_full, tg, a.tau_s, sel=sel_score)'), mA, w=0.0, listwise=True)
gm, _, _ = grads(fwd, tl, head, armA, batch)
check("REAL closure MUTATION losses6 stops passing rfac/selfac -> listwise-only grad == 0 (RED)", gm == 0.0)
mB_text = src.replace('selfac=beh_cur.get("selfac"))', 'selfac=None)')
assert mB_text != src
fwd, tl, real, head, armA, cur, batch = build_real(mB_text, load_src(mB_text, "sf_mutB"), w=0.0, listwise=True)
gm2, _, _ = grads(fwd, tl, head, armA, batch)
check("REAL closure MUTATION losses6 passes selfac=None -> listwise-only grad == 0 (RED)", gm2 == 0.0)
# main() rebinds fwd / train_losses with the wrapper BEFORE the first use of either by the training loop, the preflight and evaluate()
mi = main_src.index("make_beh_wrappers(")
check("main(): the rebinding `fwd, train_losses = make_beh_wrappers(...)` is under `if a.beh_heads:` and precedes every preflight / loop call of train_losses",
      main_src.count("make_beh_wrappers(") == 1 and "if a.beh_heads:" in main_src[:mi].splitlines()[-3] + main_src[:mi].splitlines()[-2]
      and main_src.index("def evaluate(") > mi and main_src.index("if a.preflight:") > mi and main_src.rindex("train_losses(") > mi)


# ---- G17 (the preflight probe, run on the REAL closure + the REAL wrappers): green on the good build, RED under each regression arm
def g17_run(text=None, mod=None, env=None, w=1.0, listwise=True):
    text = text or src; mod = mod or S
    fwd, tl, real, head, armA, cur, batch = build_real(text, mod, w=w, listwise=listwise)
    bias = [0.1 * k - 0.2 for k in range(5)]
    with torch.no_grad():
        head.weight[7:12].zero_(); head.bias[7:12].copy_(torch.tensor(bias))
    old = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    try:
        g = mod.g17_probe(cur, fwd, tl, real, lambda: (head.zero_grad(), armA.zero_grad()), armA, head, batch, bias, w, listwise)
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return g, mod.g17_ok(g, w, listwise)
g, ok17 = g17_run()
check("G17 GOOD: every raw field as required and g17_ok True", ok17 and g["grad_A_declared"] == 0.0 and g["grad_B_w0_control"] == 0.0 and g["grad_B_declared"] > 0.0 and g["id_loss_equal"] and g["id_grad_equal"])
for envk, why in (("REFE_BEH_MUTATE_DROP_LOSS", "term dropped"), ("REFE_BEH_MUTATE_LEAK_A", "arm A trains the heads"), ("REFE_BEH_MUTATE_ID", "act ignored in the listwise product")):
    g, ok17 = g17_run(env={envk: "1"}, listwise=True, w=1.0)
    check(f"G17 REGRESSION {envk} ({why}) -> g17_ok False", not ok17)
g, ok17 = g17_run(w=0.0, listwise=True)
check("G17 listwise-only declaration (w=0): live through the product, g17_ok True", ok17 and g["grad_B_declared"] > 0.0)
mC = src.replace('selfac=beh_cur.get("selfac"))', 'selfac=None)')
g, ok17 = g17_run(text=mC, mod=load_src(mC, "sf_mutC"), w=0.0, listwise=True)
check("G17 REGRESSION losses6 stops passing selfac (listwise-only declaration) -> g17_ok False", not ok17)


# ---- the gates (gate_checks / launch_gate): judge_g17 from raw fields, the G-DVB `beh` rows, the required-gate list, and that the callers exist
import gate_checks as C
g_good, ok_good = g17_run()
ns17 = types.SimpleNamespace(w_beh=1.0, beh_listwise=True, beh_listwise_stall=False, beh_heads=True, beh_labels_train="x", beh_labels_heldout="y")
check("judge_g17: the real probe output of the good build is accepted (no reason)", C.judge_g17(g_good, ns17) == [])
for name, mut in (("B not live (grad_B_declared 0)", lambda g: g.update(grad_B_declared=0.0)), ("bce term absent", lambda g: g.update(grad_B_bce_w1=0.0)),
                  ("listwise does not reach the head", lambda g: g.update(grad_B_listwise_only=0.0)), ("arm A leaks", lambda g: g.update(grad_A_declared=1e-9)),
                  ("w=0 control not exactly 0", lambda g: g.update(grad_B_w0_control=1e-12)), ("control absent", lambda g: g.pop("grad_A_w0_control")),
                  ("identity broken", lambda g: g.update(id_loss_equal=False)), ("identity grad broken", lambda g: g.update(id_grad_equal=False)),
                  ("a term has no active label in the probe", lambda g: g.update(probe_active=[3, 0, 2, 1, 4])), ("7-row head", lambda g: g.update(head_rows=7)),
                  ("bias not at init", lambda g: g.update(bias_rows_equal_init=False)), ("regression arm", lambda g: g.update(mutate_drop_loss=True)),
                  ("probe ran with another w", lambda g: g.update(w_declared=0.5)), ("NaN grad", lambda g: g.update(grad_B_declared=float("nan")))):
    gg = dict(g_good); mut(gg)
    check(f"judge_g17 REGRESSION '{name}' -> reasons non-empty", C.judge_g17(gg, ns17) != [])
req = lambda **kw: C.scorer_required_gates(types.SimpleNamespace(lane_labels_train="", lane_labels_heldout="", pdm_labels_train="", pdm_labels_heldout="", lane_head=False, epo=False, aux_drv=False, **kw))
check("scorer_required_gates: G17 iff --beh-heads", "G17" in req(beh_heads=True) and "G17" not in req(beh_heads=False) and "G17" not in req())
# G-DVB rows
beh_blk = {"order": list(C.SCORER_BEH_ORDER), "terms_version": "behaviour/1", "head_rows": 12, "row0": 7, "w": 1.0, "listwise": True, "listwise_cols": [0, 1, 2, 4], "stall_in_reward": False,
           "bias_init": [0.0] * 5, "train": {"matched": 10}, "heldout": {"matched": 4}, "train_active": [1] * 5}
nsd = types.SimpleNamespace(b_mode="expreward", b_compw="1,1,1,1,1,1", epo=False, aux_drv=False, pair_w=0.0, pair_calibrate=False, hn_bank="", ctx_cache="", **{k: getattr(ns17, k) for k in vars(ns17)})
data = {"b_mode": "expreward", "b_compw": "1,1,1,1,1,1", "beh": beh_blk}
prof = {"beh_profile": True}
rows_ok = lambda d, p=prof, n=nsd: all(r["ok"] for r in C.scorer_dvb_rows(n, d, p))
check("G-DVB: the `beh` rows are all ok on a faithful data event", rows_ok(data) and any(r["dest"] == "w_beh" for r in C.scorer_dvb_rows(nsd, data, prof)))
for name, d, p, n in (("lever ignored (no beh block)", {k: v for k, v in data.items() if k != "beh"}, prof, nsd),
                      ("built w differs", dict(data, beh=dict(beh_blk, w=0.5)), prof, nsd), ("built without listwise", dict(data, beh=dict(beh_blk, listwise=False)), prof, nsd),
                      ("stall in the product though not declared", dict(data, beh=dict(beh_blk, listwise_cols=[0, 1, 2, 3, 4], stall_in_reward=True)), prof, nsd),
                      ("7-row head built", dict(data, beh=dict(beh_blk, head_rows=7)), prof, nsd), ("train labels join no set", dict(data, beh=dict(beh_blk, train={"matched": 0})), prof, nsd),
                      ("heldout labels join no set", dict(data, beh=dict(beh_blk, heldout={})), prof, nsd), ("profile does not bind the label dirs", data, {}, nsd),
                      ("block present though --beh-heads is off", data, prof, types.SimpleNamespace(**{**vars(nsd), "beh_heads": False}))):
    check(f"G-DVB REGRESSION '{name}' -> a row is not ok", not rows_ok(d, p, n))
off_rows = C.scorer_dvb_rows(types.SimpleNamespace(**{**vars(nsd), "beh_heads": False}), {k: v for k, v in data.items() if k != "beh"}, {})
check("G-DVB: with --beh-heads off and no block, NO beh row exists (every stock profile's rows unchanged)", not any(r["dest"] in C.SCORER_BEH_DESTS for r in off_rows))
check("G-DVB registry: the six dests are all registered, and the trainer parser has exactly them (+ nothing unregistered)",
      set(C.SCORER_BEH_DESTS) <= set(C.DVB_KINDS_SCORER) and set(C.SCORER_BEH_DESTS) <= {a_.dest for a_ in S.build_parser()._actions})
# reachability from the callers
lg = open(os.path.join(HERE, "launch_gate.py"), encoding="utf-8").read()
check("REACH: launch_gate's preflight consumer calls C.judge_g17 on the G17 gate; the trainer's preflight calls g17_probe under `if a.beh_heads:` before G5",
      'prefix[k] == "G17"' in lg and "C.judge_g17(g17, ns)" in lg
      and re.search(r"if a\.beh_heads:[^\n]*\n(?:[^\n]*\n){0,6}[^\n]*g17_probe\(", main_src) is not None and main_src.index("g17_probe(") < main_src.index('gates["G5_ckpt_roundtrip_planner_loader"]'))

print(("ZZTEST_PASS " + str(len(fails))) if not fails else ("ZZTEST_FAIL " + str(fails)))
sys.exit(1 if fails else 0)
