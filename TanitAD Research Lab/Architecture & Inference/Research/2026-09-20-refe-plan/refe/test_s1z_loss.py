#!/usr/bin/env python3
"""S1Z L_twin tests (CPU, no data, tiny tensors; PREREG_S1Z_P6 sections 1 and 2).  Prints ZZTEST_PASS.

  T1 a HAND-COMPUTED set: 5 proposals, 3 heads, exact expected pairs and the exact loss (ln(1+e^-0.6) + ln(1+e^0.4) + ln 2) / 3 = 0.681217..., meaned per head then over the heads
  T2 the rule edges: ADE exactly at the radius is kept and just above it is not; an invalid (masked) proposal enters no pair; a set with no clean proposal contributes none; EP is not a head; the top-K keeps the
     K highest of the arm-B rule only (ties by proposal index); a set with no pair is a zero WITH graph
  T3 ANALYTIC IDENTITIES: (i) radius 0 -> no pairs and EXACTLY 0.0 (value and gradient); (ii) top-K 64 with an infinite radius == an independent ALL-CANDIDATES reference, to the last bit of the loss and the
     gradient; (iii) weight 0 -> the weighted term's gradient is EXACTLY 0.0 on every logit and the sum with a stock loss has bit-identical gradients to the stock loss alone
  T4 LIVE: a non-zero gradient reaches the scorer parameters through the real term; a frozen control module reads EXACTLY 0.0 (None); the same term at weight 0.0 reads EXACTLY 0.0 on the scorer
  T5 DELIBERATE REGRESSIONS (each must change what the test of T1/T2 asserts, i.e. turn it RED): the radius filter removed (REFE_TWIN_MUTATE_NORADIUS), the top-K mining removed (_NOTOPK); the built facts say so
  T6 twin_refusals: heads outside {NC,DAC,TTC}, a weight on no head, heads with weight 0 and no calibration, EP, a pair lever beside a twin lever, a non-positive radius, top-K < 1
Evidence class: MEASURED on CPU with hand-built tensors; nothing here touches the real model, the bank or the GPU.
"""
import math
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import scorer_finetune as S  # noqa: E402

torch.manual_seed(0)
K, T = 5, 2
HEADS = ["NC", "DAC", "TTC"]


def lab(nc, dac, ttc, ep=0.5):
    return [nc, dac, ep, ttc, 1.0, 1.0]


def hand_set():
    """p0 clean; p1 clean but 10 m away; p2 fails NC; p3 fails TTC; p4 fails DAC.  The three failing proposals sit 0.5 / 0.8 / 0.9 m from p0 (two poses each, a constant lateral offset)"""
    off = [0.0, 10.0, 0.5, 0.8, 0.9]
    xy = torch.tensor([[[0.0, y], [1.0, y]] for y in off])                                  # [K, T, 2]
    y = torch.tensor([lab(1, 1, 1), lab(1, 1, 1), lab(0, 1, 1), lab(1, 1, 0), lab(1, 0, 1)])  # [K, 6]
    return xy, y


def logits(nc, dac, ttc):
    sx = torch.zeros(1, K, 6)
    sx[0, :, 0], sx[0, :, 1], sx[0, :, 3] = torch.tensor(nc), torch.tensor(dac), torch.tensor(ttc)
    return sx


def clear_env():
    for e in S.TWIN_MUTATE_ENVS:
        os.environ.pop(e, None)


def ref_all_candidates(xy, y, valid, sx, heads, ade):
    """an independent loop-based reference: every clean i x every unclean-in-h j within the radius, no mining; -> the per-head-then-mean loss (a python float tensor with graph)"""
    per_head = []
    for h in heads:
        col = S.TWIN_HEADS[h]
        terms = []
        for i in range(y.shape[0]):
            if not (valid[i] and y[i, 0] == 1 and y[i, 1] == 1 and y[i, 3] == 1):
                continue
            for j in range(y.shape[0]):
                if not valid[j] or y[j, col] >= 1:
                    continue
                d = sum(math.sqrt(float(((xy[i, t] - xy[j, t]) ** 2).sum())) for t in range(xy.shape[1])) / xy.shape[1]
                if d <= ade:
                    terms.append(torch.nn.functional.softplus(-(sx[0, i, col] - sx[0, j, col])))
        if terms:
            per_head.append(torch.stack(terms).mean())
    return torch.stack(per_head).mean() if per_head else sx.sum() * 0.0


# ---------------------------------------------------------------- T1 the hand-computed set
clear_env()
xy, y = hand_set()
tl = S.TwinLoss(HEADS, 1.0, 8)
valid = torch.ones(K, dtype=torch.bool)
want = {"NC": ([0], [2]), "DAC": ([0], [4]), "TTC": ([0], [3])}
for h, (wi, wj) in want.items():
    ii, jj, n_clean, n_hard = tl.select(y, None, xy, valid, h)
    assert ii.tolist() == wi and jj.tolist() == wj and n_clean == 2 and n_hard == 1, (h, ii.tolist(), jj.tolist(), n_clean, n_hard)
sx = logits([1.0, 0.0, 0.4, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.9], [0.2, 0.0, 0.0, 0.2, 0.0])
tg = y[None]
cxy = torch.cat([xy, torch.zeros(K, T, 1)], -1)[None]                                       # [1, K, T, 3] as the trainer's proposals
cm = torch.ones(1, K)
loss = tl.term(sx, tg, cxy, cm)
exp = (math.log(1 + math.exp(-0.6)) + math.log(1 + math.exp(0.4)) + math.log(2.0)) / 3.0
assert abs(float(loss) - exp) < 1e-6 and abs(exp - 0.681217) < 1e-5, (float(loss), exp)
assert {h: v["pairs"] for h, v in tl.last.items()} == {"NC": 1, "DAC": 1, "TTC": 1} and all(v["sets_with_pair"] == 1 and v["sets"] == 1 and v["sets_with_clean"] == 1 for v in tl.last.values())
assert abs(tl.last_head_means["NC"] - math.log(1 + math.exp(-0.6))) < 1e-6 and abs(tl.last_head_means["TTC"] - math.log(2.0)) < 1e-6
sx.requires_grad_(True)
l2 = tl.term(sx, tg, cxy, cm)
l2.backward()
g = sx.grad[0]
# d/ds_i = -sigmoid(-(s_i - s_j)) / 3 for the winner, +sigmoid(-(s_i - s_j)) / 3 for the loser: NC pair (0, 2) with s_i - s_j = 0.6
sg = 1 / (1 + math.exp(0.6))
assert abs(float(g[0, 0]) + sg / 3) < 1e-6 and abs(float(g[2, 0]) - sg / 3) < 1e-6, g[:, 0]
assert float(g[1].abs().sum()) == 0.0 and float(g[:, 2].abs().sum()) == 0.0, "p1 is in no pair, and the EP column is not a head: exactly 0"
print("T1 ok (hand case: loss %.6f)" % exp)

# ---------------------------------------------------------------- T2 the rule edges
xy2 = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]], [[0.0, 1.0001], [1.0, 1.0001]]])        # p1 exactly 1.0 m from p0, p2 1.0001 m
y2 = torch.tensor([lab(1, 1, 1), lab(0, 1, 1), lab(0, 1, 1)])
ii, jj, _, _ = tl.select(y2, None, xy2, torch.ones(3, dtype=torch.bool), "NC")
assert (ii.tolist(), jj.tolist()) == ([0], [1]), "ADE == radius is kept, just above it is not"
ii, jj, _, _ = tl.select(y2, None, xy2, torch.tensor([True, False, True]), "NC")
assert ii.numel() == 0, "a masked (invalid) proposal enters no pair"
ii, jj, _, _ = tl.select(y2, None, xy2, torch.tensor([False, True, True]), "NC")
assert ii.numel() == 0, "an invalid clean proposal makes the set clean-less: no pair"
yb = y2.clone()
yb[0] = torch.tensor(lab(1, 1, 0.5))
ii, jj, nc_, _ = tl.select(yb, None, xy2, torch.ones(3, dtype=torch.bool), "NC")
assert ii.numel() == 0 and nc_ == 0, "a set with no clean proposal contributes none"
try:
    tl.select(y2, None, xy2, torch.ones(3, dtype=torch.bool), "EP")
    raise AssertionError("EP must not be a head")
except KeyError:
    pass
# the top-K: 6 proposals, p1..p5 are NC hard negatives at ADE 0.1 .. 0.5 from the clean p0; the rule ranks them by the (NC * DAC * ...) of the logits
xy3 = torch.tensor([[[0.0, 0.1 * k], [1.0, 0.1 * k]] for k in range(6)])
y3 = torch.tensor([lab(1, 1, 1)] + [lab(0, 1, 1)] * 5)
score = torch.tensor([0.9, 0.1, 0.5, 0.5, 0.7, 0.2])
tk = S.TwinLoss(["NC"], 1.0, 3)
ii, jj, _, nh = tk.select(y3, score, xy3, torch.ones(6, dtype=torch.bool), "NC")
assert sorted(jj.tolist()) == [2, 4] and nh == 2, (jj.tolist(), nh)         # top 3 by score: p0 (0.9, clean: not a hard negative), p4 (0.7), then the 0.5 tie broken by index: p2
tk8 = S.TwinLoss(["NC"], 1.0, 8)
ii, jj, _, nh = tk8.select(y3, score, xy3, torch.ones(6, dtype=torch.bool), "NC")
assert sorted(jj.tolist()) == [1, 2, 3, 4, 5] and nh == 5
sx_empty = torch.zeros(1, 3, 6, requires_grad=True)
z = S.TwinLoss(HEADS, 1.0, 8).term(sx_empty, torch.stack([torch.tensor(lab(0, 0, 0))] * 3)[None], torch.zeros(1, 3, 2, 3), torch.ones(1, 3))
assert float(z) == 0.0 and z.requires_grad, "a micro-batch with no pair is a zero WITH graph"
z.backward()
assert float(sx_empty.grad.abs().sum()) == 0.0
print("T2 ok")

# ---------------------------------------------------------------- T3 analytic identities
# (i) radius 0
t0 = S.TwinLoss(HEADS, 0.0, 8)
sx0 = logits([1.0, 0.0, 0.4, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.9], [0.2, 0.0, 0.0, 0.2, 0.0]).requires_grad_(True)
l0 = t0.term(sx0, tg, cxy, cm)
assert float(l0) == 0.0 and l0.requires_grad and all(v["pairs"] == 0 for v in t0.last.values())
l0.backward()
assert float(sx0.grad.abs().sum()) == 0.0, "radius 0: exactly 0.0 gradient"
# (ii) top-K 64 + infinite radius == the all-candidates form, on a random 64-proposal set
g = torch.Generator().manual_seed(7)
K64 = 64
off64 = torch.randn(K64, 1, 2, generator=g) * 3.0
xy64 = off64 + 0.05 * torch.randn(K64, 20, 2, generator=g)         # smooth proposals: ADE(i, j) ~ |off_i - off_j|, spread over 0 .. 9 m
y64 = torch.zeros(K64, 6)
y64[:, 0] = (torch.rand(K64, generator=g) > 0.35).float()
y64[:, 1] = (torch.rand(K64, generator=g) > 0.2).float()
y64[:, 3] = (torch.rand(K64, generator=g) > 0.3).float()
y64[:, 2] = torch.rand(K64, generator=g)
valid64 = torch.rand(K64, generator=g) > 0.05
sx64 = torch.randn(1, K64, 6, generator=g).requires_grad_(True)
cm64 = valid64.float()[None]
cxy64 = torch.cat([xy64, torch.zeros(K64, 20, 1)], -1)[None]
full = S.TwinLoss(HEADS, float("inf"), 64)
lf = full.term(sx64, y64[None], cxy64, cm64)
assert sum(v["pairs"] for v in full.last.values()) > 100, "the random set must hold many candidates"
lr_ = ref_all_candidates(xy64, y64, valid64, sx64, HEADS, float("inf"))
assert float(lf) == float(lr_) or abs(float(lf) - float(lr_)) < 1e-7, (float(lf), float(lr_))
gf, = torch.autograd.grad(lf, sx64, retain_graph=True)
gr, = torch.autograd.grad(lr_, sx64, retain_graph=True)
assert torch.allclose(gf, gr, atol=1e-7, rtol=0), float((gf - gr).abs().max())
# and with a finite radius the filtered form equals the reference at that radius (the radius is the only difference)
rad = S.TwinLoss(HEADS, 4.0, 64)
lrad = rad.term(sx64, y64[None], cxy64, cm64)
lrad_ref = ref_all_candidates(xy64, y64, valid64, sx64, HEADS, 4.0)
assert abs(float(lrad) - float(lrad_ref)) < 1e-7 and abs(float(lrad) - float(lf)) > 1e-4, (float(lrad), float(lrad_ref), float(lf))
# (iii) weight 0
sxw = sx64.detach().clone().requires_grad_(True)
stock = (sxw[..., :6] ** 2).mean()
(stock).backward(retain_graph=True)
g_stock = sxw.grad.clone()
sxw.grad = None
tw_w0 = S.TwinLoss(HEADS, 50.0, 8)
zero_twin = 0.0 * tw_w0.term(sxw, y64[None], cxy64, cm64)
assert sum(v["pairs"] for v in tw_w0.last.values()) > 10, "the weight-0 probe must have pairs to multiply by zero"
(stock + zero_twin).backward(retain_graph=True)
assert torch.equal(sxw.grad, g_stock), "weight 0: the gradients are bit-identical to the loss without the term"
sxw.grad = None
zero_twin.backward(retain_graph=True) if zero_twin.requires_grad else None
assert sxw.grad is None or float(sxw.grad.abs().sum()) == 0.0, "weight 0: the term alone has EXACTLY 0.0 gradient"
print("T3 ok (radius 0 -> 0.0; K=64 + infinite radius == all-candidates reference over %d pairs; weight 0 -> bit-identical gradients)" % sum(v["pairs"] for v in full.last.values()))

# ---------------------------------------------------------------- T4 live gradient, frozen control
scorer, frozen = nn.Linear(4, 6), nn.Linear(4, 6)
feat = torch.randn(1, K, 4, generator=torch.Generator().manual_seed(3))
sxs = scorer(feat)
with torch.no_grad():
    _ = frozen(feat)                                                                       # the control module is never in the graph of the loss
tw = S.TwinLoss(HEADS, 1.0, 8)
lv = 0.7 * tw.term(sxs, tg, cxy, cm)
assert sum(v["pairs"] for v in tw.last.values()) == 3
lv.backward()
assert float(scorer.weight.grad.abs().sum()) > 0.0, "L_twin must reach the scorer parameters"
assert float(scorer.bias.grad.abs().sum()) == 0.0, "a pairwise loss on the SAME head is shift-invariant: the head bias reads exactly 0.0 (the module still has a live weight)"
assert frozen.weight.grad is None and frozen.bias.grad is None, "the control reads exactly 0.0 (no gradient at all)"
scorer.zero_grad(set_to_none=True)
(0.0 * tw.term(scorer(feat), tg, cxy, cm)).backward()
assert float(scorer.weight.grad.abs().sum()) == 0.0 and float(scorer.bias.grad.abs().sum()) == 0.0, "the same term at weight 0.0 reads EXACTLY 0.0 on the scorer"
print("T4 ok")

# ---------------------------------------------------------------- T5 deliberate regressions
sx_h = logits([1.0, 0.0, 0.4, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.9], [0.2, 0.0, 0.0, 0.2, 0.0])
base = S.TwinLoss(HEADS, 1.0, 8)
good = float(base.term(sx_h, tg, cxy, cm))
# a set with a far hard negative (p5 at 7 m, clean p0): the radius keeps it out
xyf = torch.cat([xy, torch.tensor([[[0.0, 7.0], [1.0, 7.0]]])], 0)
yf = torch.cat([y, torch.tensor([lab(0, 1, 1)])], 0)
cxyf = torch.cat([xyf, torch.zeros(K + 1, T, 1)], -1)[None]
sxf = torch.zeros(1, K + 1, 6)
sxf[0, :K] = sx_h[0]
sxf[0, K, 0] = 3.0
def npairs(tl_):
    tl_.term(sxf, yf[None], cxyf, torch.ones(1, K + 1))
    return {h: v["pairs"] for h, v in tl_.last.items()}
assert npairs(S.TwinLoss(HEADS, 1.0, 8)) == {"NC": 1, "DAC": 1, "TTC": 1}
os.environ["REFE_TWIN_MUTATE_NORADIUS"] = "1"
try:
    mr = S.TwinLoss(HEADS, 1.0, 8)
    n_nr = npairs(mr)
    f_nr = mr.built_facts(0.3)
    loss_nr = float(mr.term(sx_h, tg, cxy, cm))
finally:
    clear_env()
assert n_nr["NC"] > 1 and f_nr["radius_filter"] is False and f_nr["ade"] == float(S.TWIN_UNBOUNDED), (n_nr, f_nr)
assert abs(loss_nr - good) > 1e-9 or n_nr != {"NC": 1, "DAC": 1, "TTC": 1}
assert S.TwinLoss(HEADS, 1.0, 8).built_facts(0.3)["radius_filter"] is True
os.environ["REFE_TWIN_MUTATE_NOTOPK"] = "1"
try:
    mk = S.TwinLoss(["NC"], 1.0, 1)
    ii, jj, _, nh = mk.select(y3, score, xy3, torch.ones(6, dtype=torch.bool), "NC")
    f_nk = mk.built_facts(0.3)
finally:
    clear_env()
ok_k = S.TwinLoss(["NC"], 1.0, 1)
ii2, jj2, _, nh2 = ok_k.select(y3, score, xy3, torch.ones(6, dtype=torch.bool), "NC")
assert nh2 == 0 or jj2.numel() <= 1, (nh2, jj2.tolist())             # top-1 of the rule is the clean p0: no hard negative survives
assert jj.numel() == 5 and f_nk["mining"] is False and f_nk["topk"] == S.TWIN_UNBOUNDED, (jj.tolist(), f_nk)
print("T5 ok (NORADIUS keeps the far pair and reads radius_filter False; NOTOPK keeps all 5 where top-1 keeps none)")

# ---------------------------------------------------------------- T6 refusals


def ns(**kw):
    d = dict(twin_w=0.3, twin_ade=1.0, twin_topk=8, twin_heads="NC,DAC,TTC", pair_calibrate=False, pair_w=0.0, pair_heads="", epo=False, aux_drv=False, hn_bank="")
    d.update(kw)
    return types.SimpleNamespace(**d)


assert S.twin_refusals(ns()) == []
assert S.twin_refusals(ns(twin_w=0.0, pair_calibrate=True)) == [], "calibration: heads named, weight 0"
for bad, frag in ((dict(twin_heads="EP"), "EP is not in L_twin"), (dict(twin_heads="NC,NC"), "twice"), (dict(twin_heads=""), "needs --twin-heads"),
                  (dict(twin_w=0.0), "never trained on"), (dict(twin_w=-1.0), "finite number >= 0"), (dict(twin_w=float("nan")), "finite number >= 0"),
                  (dict(twin_ade=0.0), "radius > 0"), (dict(twin_ade=float("inf")), "radius > 0"), (dict(twin_topk=0), ">= 1"),
                  (dict(twin_w=0.0, twin_heads="", twin_topk=4), "levers of a loss that is off"), (dict(pair_w=0.3, pair_heads="NC"), "L_twin alone"),
                  (dict(pair_calibrate=True), "must not also be given one"), (dict(epo=True), "--epo")):
    rs = S.twin_refusals(ns(**bad))
    assert any(frag in r for r in rs), (bad, rs)
print("T6 ok")
print("ZZTEST_PASS")
