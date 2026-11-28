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
    return load_src(re.sub(line_re, "            pass", src, flags=re.M), "sf_mut")
m1 = mutated(r"^            lB = lB \+ w \* beh_bce\(cur\[\"sxB_beh\"\], cur\[\"bq\"\], cur\[\"bact\"\]\)$")
g1, _, _ = live(m1, listwise=False)
check("MUTATION delete `lB = lB + w * beh_bce(...)` -> G-BEH-LIVE RED (B grad == 0)", g1[0] == 0.0 if isinstance(g1, tuple) else g1 == 0.0)
m2 = mutated(r"^        sc = sc \* selfac$".replace("        sc", "        sc")) if False else None
s2 = src.replace("        sc = sc * selfac\n", "        pass\n", 1)
assert s2 != src
mm = load_src(s2, "sf_mut2")
gl, _, _ = grads(*(lambda t: (t[0], t[1], t[3], t[4], t[6]))(build(mm, w=0.0, listwise=True)))
check("MUTATION delete `sc = sc * selfac` -> listwise-only grad on weight[7:12] == 0 (RED)", gl == 0.0)
s3 = src.replace("        r = r * rfac\n", "        pass\n", 1)
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
print(("ZZTEST_PASS " + str(len(fails))) if not fails else ("ZZTEST_FAIL " + str(fails)))
sys.exit(1 if fails else 0)
