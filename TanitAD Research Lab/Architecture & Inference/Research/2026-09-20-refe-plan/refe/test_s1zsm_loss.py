#!/usr/bin/env python3
"""S1ZSM L_twin SPEED-NEUTRAL tests (CPU, no data, tiny tensors; PREREG_S1ZSM_P6 sections 1 and 2).  Prints ZZTEST_PASS.

  T1 a HAND-COMPUTED set: 5 proposals, head NC, v0 = 10 (band < 1.5): the in-band pairs are exactly (0,2) S, (0,4) F, (1,2) F; the out-of-band (0,3), (1,3), (1,4) are dropped; balancing keeps 1 F + the 1 S;
     the F draw is the one of an independent torch.Generator seeded 20261107; the loss is the exact mean of the two softplus terms; a second loss object reproduces the pairs
  T2 the rule edges: |dv| / max(v0, 1) EXACTLY 0.15 is excluded and 0.149 kept; the v0 floor (v0 = 0.5 -> denominator 1); an empty F side (or S side) -> NO pair for that head and set, a zero WITH graph; all speeds
     equal -> no pair; the balancing draw does not touch the global torch RNG
  T3 BATTERY: 300 random sets x 3 heads against an INDEPENDENT numpy reference: every kept pair is in the band and is a candidate pair, kept F == kept S == min(|F|, |S|), no duplicate pair, the F/S split is the
     reference's; the counts the loss reports (band_pairs, kept_F, kept_S, out_of_band 0, unbalanced_sets 0) equal the reference's
  T4 IDENTITIES: (i) OFF (speed_neutral False) is the tip trainer's TwinLoss bit for bit (loss, gradient, `last` keys, built facts: no S1ZSM key anywhere); (ii) band and balancing both removed == S1Z's pair set exactly;
     (iii) weight 0 -> EXACTLY 0.0 gradient; (iv) a live gradient reaches the scorer weight, a frozen control reads exactly 0.0
  T5 DELIBERATE REGRESSIONS: REFE_TWIN_MUTATE_NOBAND (band removed: built facts sn_dv 1e6 + sn_band_filter False, out-of-band kept pairs appear) and REFE_TWIN_MUTATE_NOSNBALANCE (balancing removed: sn_balance False,
     unbalanced sets appear); neither acts without --twin-speed-neutral
  T6 twin_refusals: --twin-speed-neutral without heads is refused; the flag reaches the loss object (the parser carries it, default False)
Evidence class: MEASURED on CPU with hand-built tensors; nothing here touches the real model, the bank or the GPU.
"""
import importlib.util
import math
import os
import sys
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import s1zsm_layer as ZSM  # noqa: E402
import scorer_finetune as S  # noqa: E402

torch.manual_seed(0)
T = 20
HEADS = ["NC", "DAC", "TTC"]
BIG = 1000.0                                                  # a radius that keeps every pair: isolates the speed rule


def lab(nc, dac, ttc, ep=0.5):
    return [nc, dac, ep, ttc, 1.0, 1.0]


def poses(speeds, lat=None):
    """[K, T, 2] constant-speed proposals along x (v m/s, 0.2 s steps): the 0.2 s chord at step 19 is exactly v; `lat` the lateral offsets"""
    lat = lat if lat is not None else [0.0] * len(speeds)
    return torch.tensor([[[v * 0.2 * (t + 1), y] for t in range(T)] for v, y in zip(speeds, lat)], dtype=torch.float32)


def v4_ref(xy):
    return np.linalg.norm(xy[:, 19, :].numpy().astype(np.float64) - xy[:, 18, :].numpy().astype(np.float64), axis=-1) / 0.2


def clear_env():
    for e in S.TWIN_MUTATE_ENVS + S.TWIN_SN_MUTATE_ENVS:
        os.environ.pop(e, None)


clear_env()
# the constants are the card's
assert S.TWIN_SN_DV == 0.15 and S.TWIN_SN_SEED == 20261107 == S.PAIR_SEED and S.PAIR_V_IDX == 19 and S.PAIR_DT == 0.2

# ---------------------------------------------------------------- T1 the hand-computed set
speeds = [10.0, 11.0, 10.5, 14.0, 9.0]
K = len(speeds)
xy = poses(speeds, [0.0, 0.1, 0.2, 0.3, 0.4])
y = torch.tensor([lab(1, 1, 1), lab(1, 1, 1), lab(0, 1, 1), lab(0, 1, 1), lab(0, 1, 1)])
valid = torch.ones(K, dtype=torch.bool)
v4 = S.chord_speed(torch.cat([xy, torch.zeros(K, T, 1)], -1)[None])[0]
assert np.allclose(v4.numpy(), speeds, atol=1e-4), v4
tl = S.TwinLoss(["NC"], BIG, 8, speed_neutral=True)
ii, jj, nc, nh = tl.select(y, None, xy, valid, "NC", v4, 10.0)
got = sorted(zip(ii.tolist(), jj.tolist()))
assert nc == 2 and nh == 3 and len(got) == 2 and (0, 2) in got and (((0, 4) in got) != ((1, 2) in got)), got
assert tl.sn_sel == {"band_pairs": 3, "kept_F": 1, "kept_S": 1, "out_of_band": 0, "unbalanced_sets": 0}, tl.sn_sel
g = torch.Generator().manual_seed(20261107)
pf = torch.randperm(2, generator=g)[:1]
want_f = [(0, 4), (1, 2)][int(pf)]
assert want_f in got, ("the F draw must be the independent seeded generator's", want_f, got)
# the exact loss through term(): logits chosen so the four candidate terms differ
sx = torch.zeros(1, K, 6)
sx[0, :, 0] = torch.tensor([0.9, 0.3, 0.2, 1.5, -0.4])
cxy = torch.cat([xy, torch.zeros(K, T, 1)], -1)[None]
cm = torch.ones(1, K)
tl2 = S.TwinLoss(["NC"], BIG, 8, speed_neutral=True)
loss = tl2.term(sx, y[None], cxy, cm, torch.tensor([10.0]))
sp = lambda d: math.log(1 + math.exp(-d))                                              # noqa: E731
s_ = sx[0, :, 0].tolist()
exp = (sp(s_[0] - s_[2]) + sp(s_[want_f[0]] - s_[want_f[1]])) / 2.0
assert abs(float(loss) - exp) < 1e-6, (float(loss), exp)
assert tl2.last["NC"]["pairs"] == 2 and tl2.last["NC"]["kept_F"] == 1 and tl2.last["NC"]["kept_S"] == 1 and tl2.last["NC"]["band_pairs"] == 3 and tl2.last["NC"]["out_of_band"] == 0
tl3 = S.TwinLoss(["NC"], BIG, 8, speed_neutral=True)
ii3, jj3, _, _ = tl3.select(y, None, xy, valid, "NC", v4, 10.0)
assert sorted(zip(ii3.tolist(), jj3.tolist())) == got, "a second loss object (same seed) draws the same pairs"
facts = tl3.built_facts(0.3)
assert facts["speed_neutral"] is True and facts["sn_dv"] == 0.15 and facts["sn_balance"] is True and facts["sn_seed"] == 20261107 and facts["sn_band_filter"] is True, facts
print("T1 ok (hand case: pairs %s, loss %.6f)" % (got, exp))

# ---------------------------------------------------------------- T2 the rule edges
tle = S.TwinLoss(["NC"], BIG, 8, speed_neutral=True)
ii_ = torch.tensor([0, 0])
jj_ = torch.tensor([1, 2])
v_edge = torch.tensor([10.0, 11.5, 11.49])
tle._speed_neutral(ii_, jj_, v_edge, 10.0)
assert tle.sn_sel["band_pairs"] == 1, ("|dv| / max(v0,1) == 0.15 is EXCLUDED (strict <), 0.149 is kept", tle.sn_sel)
tle._speed_neutral(torch.tensor([0]), torch.tensor([1]), torch.tensor([10.0, 10.14]), 0.5)
assert tle.sn_sel["band_pairs"] == 1, "v0 = 0.5 -> denominator max(v0, 1) = 1: 0.14 / 1 is inside the band (0.14 / 0.5 would not be)"
tle._speed_neutral(torch.tensor([0]), torch.tensor([1]), torch.tensor([10.0, 10.16]), 0.5)
assert tle.sn_sel["band_pairs"] == 0
# an empty side: one in-band pair, it is S -> F empty -> NO pair
ii_e, jj_e = tle._speed_neutral(torch.tensor([0]), torch.tensor([1]), torch.tensor([10.0, 10.5]), 10.0)
assert ii_e.numel() == 0 and tle.sn_sel == {"band_pairs": 1, "kept_F": 0, "kept_S": 0, "out_of_band": 0, "unbalanced_sets": 0}, tle.sn_sel
ii_e, jj_e = tle._speed_neutral(torch.tensor([0]), torch.tensor([1]), torch.tensor([10.5, 10.0]), 10.0)
assert ii_e.numel() == 0, "F only: no pair either"
# all speeds equal: v_i > v_j is never true -> everything is S -> no pair
xy_eq = poses([10.0] * 4, [0.0, 0.1, 0.2, 0.3])
y_eq = torch.tensor([lab(1, 1, 1), lab(1, 1, 1), lab(0, 1, 1), lab(0, 1, 1)])
tq = S.TwinLoss(HEADS, BIG, 8, speed_neutral=True)
cxy_eq = torch.cat([xy_eq, torch.zeros(4, T, 1)], -1)[None]
sx_eq = torch.randn(1, 4, 6, requires_grad=True)
z = tq.term(sx_eq, y_eq[None], cxy_eq, torch.ones(1, 4), torch.tensor([10.0]))
assert float(z) == 0.0 and z.requires_grad and all(v["pairs"] == 0 and v["sets_with_pair"] == 0 and v["sets"] == 1 for v in tq.last.values()), tq.last
z.backward()
assert float(sx_eq.grad.abs().sum()) == 0.0
# the balancing draw has its OWN generator: the global torch RNG state is untouched
st0 = torch.get_rng_state()
S.TwinLoss(["NC"], BIG, 8, speed_neutral=True).select(y, None, xy, valid, "NC", v4, 10.0)
assert torch.equal(st0, torch.get_rng_state()), "the draw must come from the loss object's own generator"
# the term needs v0 when speed-neutral
try:
    S.TwinLoss(["NC"], BIG, 8, speed_neutral=True).term(sx, y[None], cxy, cm)
    raise AssertionError("a speed-neutral term without v0 must fail loudly")
except AssertionError as e:
    assert "needs v0" in str(e), e
print("T2 ok")

# ---------------------------------------------------------------- T3 the random battery vs an independent reference
rng = np.random.default_rng(11)
tb = S.TwinLoss(HEADS, BIG, 64, speed_neutral=True)
n_pairs_total, n_sets_with, n_sets = 0, 0, 0
for trial in range(300):
    Kb = 12
    spd = rng.uniform(0.0, 15.0, Kb)
    spd[rng.random(Kb) < 0.4] = spd[0] + rng.uniform(-1.0, 1.0, int((rng.random(Kb) < 0.4).sum() or 1))[:1][0]      # cluster some speeds so the band is populated
    xyb = poses(spd.tolist(), rng.uniform(-1, 1, Kb).tolist())
    yb = torch.zeros(Kb, 6)
    yb[:, 0] = torch.tensor((rng.random(Kb) > 0.4).astype(np.float32))
    yb[:, 1] = torch.tensor((rng.random(Kb) > 0.3).astype(np.float32))
    yb[:, 3] = torch.tensor((rng.random(Kb) > 0.3).astype(np.float32))
    v0 = float(rng.uniform(0.0, 15.0))
    vb = torch.tensor(v4_ref(xyb), dtype=torch.float32)
    vvalid = torch.tensor(rng.random(Kb) > 0.1)
    for h, col in S.TWIN_HEADS.items():
        ii, jj, ncl, nhd = tb.select(yb, None, xyb, vvalid, h, vb, v0)
        okp = [bool(vvalid[k]) for k in range(Kb)]
        clean = [okp[k] and yb[k, 0] == 1 and yb[k, 1] == 1 and yb[k, 3] == 1 for k in range(Kb)]
        hard = [okp[k] and yb[k, col] < 1 for k in range(Kb)]
        den = max(v0, 1.0)
        cand = [(i, j) for i in range(Kb) for j in range(Kb) if clean[i] and hard[j]]
        inband = [(i, j) for (i, j) in cand if abs(float(vb[i]) - float(vb[j])) / den < 0.15]
        F = [(i, j) for (i, j) in inband if float(vb[i]) > float(vb[j])]
        Sd = [(i, j) for (i, j) in inband if not float(vb[i]) > float(vb[j])]
        kept = list(zip(ii.tolist(), jj.tolist()))
        k = min(len(F), len(Sd))
        assert len(kept) == 2 * k and len(set(kept)) == len(kept), (trial, h, len(kept), k)
        assert all(p in inband for p in kept), (trial, h, "a kept pair outside the band / candidate set")
        kF = [p for p in kept if p in F]
        kS = [p for p in kept if p in Sd]
        assert len(kF) == k and len(kS) == k, (trial, h, len(kF), len(kS), k)
        assert tb.sn_sel == {"band_pairs": len(inband), "kept_F": k, "kept_S": k, "out_of_band": 0, "unbalanced_sets": 0}, (trial, h, tb.sn_sel, len(inband), k)
        n_sets += 1
        n_sets_with += int(k > 0)
        n_pairs_total += 2 * k
assert n_sets_with > 100 and n_pairs_total > 300, (n_sets, n_sets_with, n_pairs_total)
print("T3 ok (%d set x head cases, %d with a kept pair, %d kept pairs; every one in the band and F == S)" % (n_sets, n_sets_with, n_pairs_total))

# ---------------------------------------------------------------- T4 identities
# a 64-proposal random set for the OFF identity and the S1Z-pair-set identity
g64 = torch.Generator().manual_seed(7)
K64 = 64
spd64 = (10.0 + 3.0 * torch.randn(K64, generator=g64)).clamp(min=0.0)
lat64 = torch.randn(K64, generator=g64) * 2.0
xy64 = poses(spd64.tolist(), lat64.tolist())
y64 = torch.zeros(K64, 6)
y64[:, 0] = (torch.rand(K64, generator=g64) > 0.35).float()
y64[:, 1] = (torch.rand(K64, generator=g64) > 0.2).float()
y64[:, 3] = (torch.rand(K64, generator=g64) > 0.3).float()
y64[:, 2] = torch.rand(K64, generator=g64)
cm64 = (torch.rand(K64, generator=g64) > 0.05).float()[None]
sx64 = torch.randn(1, K64, 6, generator=g64).requires_grad_(True)
cxy64 = torch.cat([xy64, torch.zeros(K64, T, 1)], -1)[None]
# (i) OFF == the tip trainer's TwinLoss, bit for bit
tip_src = ZSM.peel(open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n"))
with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
    fp = os.path.join(td, "scorer_finetune_tip_s1z.py")
    open(fp, "w", encoding="utf-8", newline="").write(tip_src)
    spec = importlib.util.spec_from_file_location("scorer_finetune_tip_s1z", fp)
    TIP = importlib.util.module_from_spec(spec)
    sys.modules["scorer_finetune_tip_s1z"] = TIP
    spec.loader.exec_module(TIP)
assert not hasattr(TIP, "TWIN_SN_DV")
for ade, topk in ((1.0, 8), (4.0, 64), (BIG, 64)):
    t_new, t_tip = S.TwinLoss(HEADS, ade, topk), TIP.TwinLoss(HEADS, ade, topk)
    l_new, l_tip = t_new.term(sx64, y64[None], cxy64, cm64), t_tip.term(sx64, y64[None], cxy64, cm64)
    assert torch.equal(l_new, l_tip), (ade, topk, float(l_new), float(l_tip))
    g_new, = torch.autograd.grad(l_new, sx64, allow_unused=True)
    g_tip, = torch.autograd.grad(l_tip, sx64, allow_unused=True)
    assert (g_new is None and g_tip is None) or torch.equal(g_new, g_tip)
    assert t_new.last == t_tip.last and all(set(v) == {"sets", "sets_with_clean", "sets_with_pair", "pairs", "hard"} for v in t_new.last.values()), "OFF adds no key to `last`"
    assert t_new.built_facts(0.3) == t_tip.built_facts(0.3) and "speed_neutral" not in t_new.built_facts(0.3) and not any(k.startswith("sn_") for k in t_new.built_facts(0.3))
assert l_new.item() != 0.0
print("T4(i) ok (OFF: loss, gradient, `last` and built facts bit-identical to the tip trainer's TwinLoss)")
# (ii) band and balancing both REMOVED == S1Z's pair set (the speed-neutral step is the only difference)
os.environ["REFE_TWIN_MUTATE_NOBAND"] = "1"
os.environ["REFE_TWIN_MUTATE_NOSNBALANCE"] = "1"
try:
    t_nosn = S.TwinLoss(HEADS, 4.0, 64, speed_neutral=True)
finally:
    clear_env()
t_s1z = S.TwinLoss(HEADS, 4.0, 64)
v64 = S.chord_speed(cxy64).detach()[0]
n_cmp = 0
for h in HEADS:
    a_ = t_nosn.select(y64, None, xy64, cm64[0] > 0, h, v64, 10.0)
    b_ = t_s1z.select(y64, None, xy64, cm64[0] > 0, h)
    assert sorted(zip(a_[0].tolist(), a_[1].tolist())) == sorted(zip(b_[0].tolist(), b_[1].tolist())), h
    n_cmp += a_[0].numel()
assert n_cmp > 20
l_a = t_nosn.term(sx64, y64[None], cxy64, cm64, torch.tensor([10.0]))
l_b = t_s1z.term(sx64, y64[None], cxy64, cm64)
assert abs(float(l_a) - float(l_b)) < 1e-6, (float(l_a), float(l_b))
print("T4(ii) ok (band + balancing removed == S1Z's pair set over %d pairs)" % n_cmp)
# (iii) weight 0 and (iv) live gradient, frozen control
scorer, frozen = nn.Linear(4, 6), nn.Linear(4, 6)
feat = torch.randn(1, K, 4, generator=torch.Generator().manual_seed(3))
tg = y[None]
tw = S.TwinLoss(["NC"], BIG, 8, speed_neutral=True)
with torch.no_grad():
    _ = frozen(feat)
lv = 0.7 * tw.term(scorer(feat), tg, cxy, cm, torch.tensor([10.0]))
assert tw.last["NC"]["pairs"] == 2
lv.backward()
assert float(scorer.weight.grad.abs().sum()) > 0.0, "L_twin (speed-neutral) must reach the scorer parameters"
assert frozen.weight.grad is None and frozen.bias.grad is None
scorer.zero_grad(set_to_none=True)
(0.0 * S.TwinLoss(["NC"], BIG, 8, speed_neutral=True).term(scorer(feat), tg, cxy, cm, torch.tensor([10.0]))).backward()
assert float(scorer.weight.grad.abs().sum()) == 0.0 and float(scorer.bias.grad.abs().sum()) == 0.0, "weight 0: EXACTLY 0.0"
print("T4(iii,iv) ok")

# ---------------------------------------------------------------- T5 deliberate regressions
def battery(env):
    """the 64-proposal set through the speed-neutral loss under `env`: (sum out_of_band, sum unbalanced_sets, built facts)"""
    clear_env()
    os.environ.update(env)
    try:
        tl_ = S.TwinLoss(HEADS, BIG, 64, speed_neutral=True)
        facts_ = tl_.built_facts(0.3)
        oob_, unb_, pairs_ = 0, 0, 0
        for b_ in range(20):
            sxb = torch.randn(1, K64, 6, generator=torch.Generator().manual_seed(b_))
            tl_.term(sxb, y64[None], cxy64, cm64, torch.tensor([10.0]))
            oob_ += sum(v["out_of_band"] for v in tl_.last.values())
            unb_ += sum(v["unbalanced_sets"] for v in tl_.last.values())
            pairs_ += sum(v["pairs"] for v in tl_.last.values())
        return oob_, unb_, pairs_, facts_
    finally:
        clear_env()


oob0, unb0, pairs0, f0 = battery({})
assert oob0 == 0 and unb0 == 0 and pairs0 > 20 and f0["sn_band_filter"] is True and f0["sn_balance"] is True and f0["sn_dv"] == 0.15
oob1, unb1, pairs1, f1 = battery({"REFE_TWIN_MUTATE_NOBAND": "1"})
assert oob1 > 0 and f1["sn_band_filter"] is False and f1["sn_dv"] == float(S.TWIN_UNBOUNDED) and f1["sn_balance"] is True, (oob1, f1)
oob2, unb2, pairs2, f2 = battery({"REFE_TWIN_MUTATE_NOSNBALANCE": "1"})
assert unb2 > 0 and oob2 == 0 and f2["sn_balance"] is False and f2["sn_band_filter"] is True, (unb2, f2)
# the mutations act ONLY with --twin-speed-neutral (OFF stays S1Z whatever the env says)
os.environ["REFE_TWIN_MUTATE_NOBAND"] = "1"
try:
    off = S.TwinLoss(HEADS, BIG, 64)
finally:
    clear_env()
assert off.sn_band_filter is True and off.sn_balance is True and "speed_neutral" not in off.built_facts(0.3)
print("T5 ok (NOBAND: %d kept pairs outside the band, built sn_dv 1e6; NOSNBALANCE: %d unbalanced sets, built sn_balance False)" % (oob1, unb2))

# ---------------------------------------------------------------- T6 refusals and the parser


def ns(**kw):
    d = dict(twin_w=0.3, twin_ade=1.0, twin_topk=8, twin_heads="NC,DAC,TTC", pair_calibrate=False, pair_w=0.0, pair_heads="", epo=False, aux_drv=False, hn_bank="", twin_speed_neutral=True)
    d.update(kw)
    return types.SimpleNamespace(**d)


assert S.twin_refusals(ns()) == []
assert S.twin_refusals(ns(twin_speed_neutral=False)) == []
assert S.twin_refusals(ns(twin_w=0.0, pair_calibrate=True)) == [], "calibration: heads named, weight 0, speed-neutral"
rs = S.twin_refusals(ns(twin_w=0.0, twin_heads="", pair_calibrate=True))
assert any("--twin-speed-neutral needs --twin-heads" in r for r in rs), rs
p = S.build_parser()
dflt = p.parse_args(["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "o"])
assert dflt.twin_speed_neutral is False
on = p.parse_args(["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "o", "--twin-speed-neutral"])
assert on.twin_speed_neutral is True
print("T6 ok")
print("ZZTEST_PASS")
