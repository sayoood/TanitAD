#!/usr/bin/env python3
"""SCORER1P6 L_pair tests (CPU only, no data): PREREG_SCORER1P6 section 2, 'L_pair'.  Prints ZZTEST_PASS.

  P1 DEFAULTS ARE THE OLD LOSS: PairLoss with no per-head setting is bit-identical to the PREREG_SCORER1 / v3 PairLoss (pairs, generator stream, term value, gradient) on 200 random sets and 20 micro-batches
  P2 THE EP RULE: head EP keeps EVERY pair with y_i - y_j >= 0.05, with no speed-contrast filter and no F/S balancing, violating proposals (label 0) in; a gap of 0.04 is out; the generator is not consumed
  P3 INVALID PROPOSALS: a masked proposal (set mask 0), a NaN label and a NaN speed enter no pair of any head (incl. EP, whose speed filter is off, so the NaN speed is NOT caught by the filter any more)
  P4 THE REDUCTION: --pair-mean head = mean per head over its kept pairs, then over the heads with >= 1 kept pair; a head with no kept pair does not dilute; pooled is the old mean and differs
  P5 GRADIENT + CONTROL: a non-zero gradient on the scorer logits, and EXACTLY 0.0 at weight 0.0 (the graph is live)
  P6 THE MUTANTS AT THE LOSS: NOBALANCE unbalances a safety head and leaves EP alone; EPBALANCE balances EP and leaves the safety heads alone
  P7 PARSER: the four new flags default to the old rule; pair_refusals accepts the S1E argv and refuses a stray head, a malformed or negative gap, a twice-named head, --pair-mean head without heads
  P8 REACHABILITY: each hop from the flag to the PairLoss is a line of the trainer source
"""
import ast
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import scorer_finetune as S  # noqa: E402

K, T = 64, 20
SAFETY = ["NC", "DAC", "TTC"]
ALL = ["NC", "DAC", "EP", "TTC"]


class OldPairLoss:
    """the PREREG_SCORER1 / v3 PairLoss, verbatim in behaviour (the v3 trainer at git blob 73f3e2bd): the reference of P1"""

    def __init__(self, heads, dv, seed=S.PAIR_SEED):
        self.heads, self.dv, self.seed = tuple(heads), float(dv), int(seed)
        self.balance = os.environ.get("REFE_PAIR_MUTATE_NOBALANCE") != "1"
        self.gen = torch.Generator().manual_seed(self.seed)
        self.last = {}

    def select(self, y, v, v0):
        better = y[:, None] > y[None, :]
        far = (v[:, None] - v[None, :]).abs() / max(float(v0), 1.0) >= self.dv
        cand = better & far
        fast = v[:, None] > v[None, :]
        fi, fj = (cand & fast).nonzero(as_tuple=True)
        si, sj = (cand & ~fast).nonzero(as_tuple=True)
        nF, nS = int(fi.numel()), int(si.numel())
        if not self.balance:
            return torch.cat([fi, si]), torch.cat([fj, sj]), nF, nS, nF, nS
        k = min(nF, nS)
        if k == 0:
            e = torch.zeros(0, dtype=torch.long)
            return e, e, nF, nS, 0, 0
        pf = torch.randperm(nF, generator=self.gen)[:k]
        ps = torch.randperm(nS, generator=self.gen)[:k]
        return torch.cat([fi[pf], si[ps]]), torch.cat([fj[pf], sj[ps]]), nF, nS, k, k

    def term(self, sxB, tg, cxy, v0, cm):
        v4 = S.chord_speed(cxy).detach().cpu()
        ycpu = tg.detach().float().cpu()
        cmc = cm.detach().cpu()
        terms, st = [], {h: {"sets": 0, "sets_with_pair": 0, "cand_F": 0, "cand_S": 0, "kept_F": 0, "kept_S": 0} for h in self.heads}
        for b in range(sxB.shape[0]):
            if float(cmc[b].min()) <= 0.0:
                continue
            for h in self.heads:
                col = S.PAIR_HEADS[h]
                ii, jj, cF, cS, kF, kS = self.select(ycpu[b, :, col], v4[b], float(v0[b]))
                d = st[h]
                d["sets"] += 1; d["cand_F"] += cF; d["cand_S"] += cS; d["kept_F"] += kF; d["kept_S"] += kS; d["sets_with_pair"] += int(ii.numel() > 0)
                if ii.numel():
                    s = sxB[b, :, col]
                    terms.append(F.softplus(-(s[ii.to(s.device)] - s[jj.to(s.device)])))
        self.last = st
        if not terms:
            return sxB.sum() * 0.0
        return torch.cat(terms).mean()


def chord_set(speed):
    """[1, n, T, 3] proposals whose v4 chord speed equals `speed`"""
    speed = torch.as_tensor(speed, dtype=torch.float32)
    cxy = torch.zeros(1, speed.numel(), T, 3)
    cxy[0, :, :, 0] = (speed[:, None] * 0.2) * torch.arange(1, T + 1)[None, :]
    return cxy


def make_batch(B, seed, n=K):
    g = torch.Generator().manual_seed(seed)
    cxy = torch.zeros(B, n, T, 3)
    speed = torch.rand(B, n, generator=g) * 14.0
    cxy[:, :, :, 0] = (speed[:, :, None] * 0.2) * torch.arange(1, T + 1)[None, None, :]
    tg = torch.rand(B, n, 6, generator=g)
    tg[..., :2] = (tg[..., :2] > 0.4).float()                               # NC, DAC binary; EP, TTC, C, DDC graded
    tg[..., 3] = (tg[..., 3] > 0.5).float()
    return cxy, tg, torch.ones(B, n), torch.full((B,), 10.0), speed


def clear_env():
    for e in S.PAIR_MUTATE_ENVS:
        os.environ.pop(e, None)


clear_env()

# ---- P1 defaults are the old loss, bit for bit
new, old = S.PairLoss(SAFETY, 0.15), OldPairLoss(SAFETY, 0.15)
g = torch.Generator().manual_seed(11)
for n in range(200):
    cxy, tg, cm, v0, speed = make_batch(1, 100 + n)
    y = tg[0, :, n % 3 if n % 3 < 2 else 3]
    a_, b_ = new.select(y, speed[0], 10.0), old.select(y, speed[0], 10.0)
    assert all((torch.equal(x, z) if torch.is_tensor(x) else x == z) for x, z in zip(a_, b_)), n
assert torch.equal(new.gen.get_state(), old.gen.get_state()), "the generator stream moved"
new, old = S.PairLoss(SAFETY, 0.15), OldPairLoss(SAFETY, 0.15)
for n in range(20):
    cxy, tg, cm, v0, speed = make_batch(4, 500 + n)
    sx = torch.randn(4, K, 6, generator=g).requires_grad_(True)
    sx2 = sx.detach().clone().requires_grad_(True)
    ln, lo = new.term(sx, tg, cxy, v0, cm), old.term(sx2, tg, cxy, v0, cm)
    assert torch.equal(ln, lo) and new.last == old.last
    gn, = torch.autograd.grad(ln, sx)
    go, = torch.autograd.grad(lo, sx2)
    assert torch.equal(gn, go), "the gradient moved"
print("P1 ok (defaults bit-identical to the PREREG_SCORER1 loss: 200 sets, 20 micro-batches, generator stream, gradient)")

# ---- P2 the EP rule
#          idx:  0     1     2     3     4     5(NaN label)
y_ep = torch.tensor([0.00, 0.30, 0.34, 0.90, 0.90, float("nan")])
v_ep = torch.tensor([10.0, 10.0, 10.0, 8.0, 12.0, 5.0])
pl = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="head")
st0 = pl.gen.get_state().clone()
ii, jj, cF, cS, kF, kS = pl.select(y_ep, v_ep, 10.0, head="EP")
got = sorted(zip(ii.tolist(), jj.tolist()))
want = sorted((i, j) for i in range(6) for j in range(6) if i != j and not math.isnan(float(y_ep[i])) and not math.isnan(float(y_ep[j])) and float(y_ep[i]) - float(y_ep[j]) >= 0.05 - 1e-7)
assert got == want and (kF + kS) == len(want) and (cF, cS) == (kF, kS), (got, want)
assert (1, 0) in got and (2, 0) in got and (3, 0) in got, "a violating proposal (label 0) must be the worse side of its pairs"
assert (1, 2) not in got and (2, 1) not in got, "a label gap of 0.04 (< 0.05) must not make a pair"
assert (3, 4) not in got and (4, 3) not in got, "equal labels make no pair"
assert (2, 1) not in got and (2, 0) in got and (1, 0) in got, "equal speed (10 vs 10) must still pair: the speed filter is off"
assert torch.equal(pl.gen.get_state(), st0), "the EP head must not consume the sampling generator (no balancing)"
assert pl.rules["EP"] == {"balance": False, "speed_contrast": False, "min_gap": 0.05} and all(pl.rules[h] == {"balance": True, "speed_contrast": True, "min_gap": 0.0} for h in SAFETY)
assert pl.balance is True, "pair.balanced reads the safety heads: all built balanced"
# the safety heads keep the PREREG_SCORER1 rule on the same data: a strict gap, a speed filter, balancing
yb = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
vb = torch.tensor([2.0, 12.0, 4.0, 11.0, 8.0, 3.0, 6.0, 9.0])
ii, jj, cF, cS, kF, kS = pl.select(yb, vb, 10.0, head="NC")
assert kF == kS and ii.numel() == 2 * kF and ((vb[ii] - vb[jj]).abs() / 10.0 >= 0.15).all(), "NC must stay balanced and speed-filtered"
print("P2 ok (EP: natural pairs, no speed filter, no balancing, gap >= 0.05, label-0 proposals in; safety heads unchanged)")

# ---- P3 invalid proposals enter no pair of any head
y3 = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.6, 0.0])
v3 = torch.tensor([4.0, 9.0, 12.0, 3.0, 8.0, 14.0, 2.0, 6.0])
valid = torch.tensor([True, True, True, False, True, True, True, True])             # proposal 3 masked
y3n, v3n = y3.clone(), v3.clone()
y3n[5] = float("nan")                                                              # proposal 5: NaN label
v3n[6] = float("nan")                                                              # proposal 6: NaN speed (EP has no speed filter that would drop it)
for h in ALL:
    for ymat, vmat, bad in ((y3, v3, [3]), (y3n, v3, [5]), (y3, v3n, [6])):
        plx = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05})
        ii, jj, *_ = plx.select(ymat, vmat, 10.0, head=h, valid=valid)
        used = set(ii.tolist()) | set(jj.tolist())
        assert not (used & ({3} | set(bad))), (h, used, bad)
        if h == "EP":
            assert ii.numel() > 0
# and through term(): a masked proposal of a labelled set is excluded, a fully masked set is skipped
cxy = chord_set(v3)
tg = torch.zeros(1, 8, 6)
tg[0, :, 2] = y3
tg[0, 3, 2] = 1.0                                                                  # the masked proposal would be the best EP proposal
cm = torch.ones(1, 8)
cm[0, 3] = 0.0
plx = S.PairLoss(["EP"], 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="head")
sx = torch.zeros(1, 8, 6, requires_grad=True)
plx.term(sx, tg, cxy, torch.tensor([10.0]), cm)
l1 = plx.last["EP"]["kept_F"] + plx.last["EP"]["kept_S"]
ncand = sum(1 for i in range(8) for j in range(8) if i != j and 3 not in (i, j) and float(tg[0, i, 2] - tg[0, j, 2]) >= 0.05 - 1e-7)
assert l1 == ncand, (l1, ncand)
plx.term(sx, tg, cxy, torch.tensor([10.0]), torch.zeros(1, 8))
assert plx.last["EP"]["sets"] == 0, "a fully masked set must contribute nothing"
print("P3 ok (masked / NaN-label / NaN-speed proposals enter no pair, EP included)")

# ---- P4 the reduction: per head, then over the heads with >= 1 kept pair
cxy, tg, cm, v0, speed = make_batch(3, 7)
tg[..., 3] = 0.5                                                                   # TTC: all labels equal -> NO kept pair on that head
pl = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="head")
ref = pl.fresh()
sx = torch.randn(3, K, 6, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
loss = pl.term(sx, tg, cxy, v0, cm)
per_head = {h: [] for h in ALL}
v4 = S.chord_speed(cxy)
for b in range(3):
    for h in ALL:
        ii, jj, *_ = ref.select(tg[b, :, S.PAIR_HEADS[h]], v4[b], float(v0[b]), head=h, valid=cm[b] > 0)
        if ii.numel():
            per_head[h].append(F.softplus(-(sx[b, :, S.PAIR_HEADS[h]][ii] - sx[b, :, S.PAIR_HEADS[h]][jj])))
means = [torch.cat(v).mean() for v in per_head.values() if v]
assert not per_head["TTC"] and len(means) == 3, "TTC must have no kept pair here"
assert torch.allclose(loss, torch.stack(means).mean(), atol=0, rtol=1e-6), (float(loss), float(torch.stack(means).mean()))
assert set(pl.last_head_means) == {"NC", "DAC", "EP"}, pl.last_head_means
n_ep = sum(v.numel() for v in per_head["EP"])
n_nc = sum(v.numel() for v in per_head["NC"])
assert n_ep > 2 * n_nc, (n_ep, n_nc)                                               # unbalanced EP pairs outnumber the balanced ones: the reason for the per-head mean
pooled = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="pooled").term(sx, tg, cxy, v0, cm)
assert not torch.allclose(pooled, loss), "pooled and per-head must differ when the pair counts are unequal"
print(f"P4 ok (mean per head then over the 3 heads with a pair; EP {n_ep} pairs vs NC {n_nc}; pooled {float(pooled):.4f} != per-head {float(loss):.4f})")

# ---- P5 gradient and the exact-zero control
sx = torch.randn(3, K, 6, generator=torch.Generator().manual_seed(6)).requires_grad_(True)
pl = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05}, reduce="head")
gr, = torch.autograd.grad(pl.term(sx, tg, cxy, v0, cm), sx)
assert float(gr.abs().sum()) > 0.0 and float(gr[..., 2].abs().sum()) > 0.0 and float(gr[..., 4:].abs().sum()) == 0.0, "gradient on NC/DAC/EP/TTC columns only"
sx = sx.detach().clone().requires_grad_(True)
g0, = torch.autograd.grad(0.0 * pl.term(sx, tg, cxy, v0, cm), sx)
assert float(g0.abs().sum()) == 0.0, "weight 0.0 must read EXACTLY 0.0"
print("P5 ok")

# ---- P6 the mutants at the loss
cxy, tg, cm, v0, speed = make_batch(1, 9)
yk = tg[0, :, 2].clone()
try:
    os.environ["REFE_PAIR_MUTATE_NOBALANCE"] = "1"
    pm = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05})
    assert pm.rules["NC"]["balance"] is False and pm.rules["EP"]["balance"] is False and pm.balance is False
    kf = [pm.select(tg[0, :, 0], speed[0], 10.0, head="NC")[4:6]]
    assert any(a != b for a, b in kf), "NOBALANCE must unbalance a safety head"
finally:
    clear_env()
try:
    os.environ["REFE_PAIR_MUTATE_EPBALANCE"] = "1"
    pm = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05})
    assert pm.rules["EP"]["balance"] is True and all(pm.rules[h]["balance"] is True for h in SAFETY) and pm.balance is True
    r = pm.select(yk, speed[0], 10.0, head="EP")
    assert r[4] == r[5], "EPBALANCE must balance EP"
finally:
    clear_env()
pu = S.PairLoss(ALL, 0.15, nobalance=["EP"], nospeed=["EP"], gaps={"EP": 0.05})
assert pu.rules["EP"]["balance"] is False, "without the mutation EP is unbalanced"
print("P6 ok")

# ---- P7 parser and refusals
ap = S.build_parser()
base = ["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x"]
d = ap.parse_args(base)
assert (d.pair_nobalance_heads, d.pair_nospeed_heads, d.pair_gap_heads, d.pair_mean) == ("", "", "", "pooled") and S.pair_refusals(d) == []
s1e = ["--pair-heads", "NC,DAC,EP,TTC", "--pair-dv", "0.15", "--pair-nobalance-heads", "EP", "--pair-nospeed-heads", "EP", "--pair-gap-heads", "EP=0.05", "--pair-mean", "head", "--pair-w", "0.3"]


def refs(*extra):
    return S.pair_refusals(ap.parse_args(base + list(extra)))


assert refs(*s1e) == [], refs(*s1e)
assert refs("--pair-w", "0.3", "--pair-heads", "NC,TTC", "--pair-nobalance-heads", "EP"), "a rule for a head the loss does not act on"
assert refs(*s1e[:-2], "--pair-gap-heads", "EP=abc", "--pair-w", "0.3") and refs("--pair-w", "0.3", "--pair-heads", "EP", "--pair-gap-heads", "EP=-0.1")
assert refs("--pair-w", "0.3", "--pair-heads", "EP", "--pair-gap-heads", "EP")
assert refs("--pair-w", "0.3", "--pair-heads", "EP,EP") and refs("--pair-mean", "head")
assert refs("--pair-w", "0.3", "--pair-heads", "EP,NC", "--pair-nobalance-heads", "EP,EP")
assert refs("--pair-w", "0.3", "--pair-heads", "XX") and refs("--pair-w", "0.3", "--pair-heads", "NC", "--pair-nospeed-heads", "TTC")
print("P7 ok")

# ---- P8 reachability: the hops from the flags to the loss are lines of the trainer
src = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf8").read()
main = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main")
reads = {n.attr for n in ast.walk(main) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "a"}
assert {"pair_nobalance_heads", "pair_nospeed_heads", "pair_gap_heads", "pair_mean"} <= reads, reads
for needle in ('ap.add_argument("--pair-nobalance-heads"', 'ap.add_argument("--pair-nospeed-heads"', 'ap.add_argument("--pair-gap-heads"', 'ap.add_argument("--pair-mean"',
               "nobalance=[h for h in a.pair_nobalance_heads.split(',') if h]", "nospeed=[h for h in a.pair_nospeed_heads.split(',') if h]", "gaps=parse_pair_gaps(a.pair_gap_heads), reduce=a.pair_mean)",
               'ii, jj, cF, cS, kF, kS = self.select(ycpu[b, :, col], v4[b], float(v0[b]), head=h, valid=valid)', "probe_pl = pairer.fresh()",
               '"rules": {h_: dict(r_) for h_, r_ in pairer.rules.items()}, "reduce": pairer.reduce}', 'info["nav_mode"] = "command"'):
    assert src.count(needle) == 1, needle
print("P8 ok")
print("ZZTEST_PASS")
