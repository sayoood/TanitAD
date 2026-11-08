#!/usr/bin/env python3
"""SCORER1 L_pair tests (CPU only): PREREG_SCORER1 section 2.  Prints ZZTEST_PASS.

  T1 balance: kept F == kept S for every set and head; an empty side -> no pair; sampling is seeded (two PairLoss with seed 20261107 agree, another seed differs)
  T2 REFE_PAIR_MUTATE_NOBALANCE=1 keeps every candidate -> the F/S ratio check goes RED on a set with unequal sides
  T3 gradient: L_pair has a non-zero gradient on the scorer logits; weight 0.0 (explicit) gives gradients of EXACTLY 0.0 (graph live, factor 0)
  T4 informational: the Dv-weighted speed-direction gradient of a speed-only score, balanced vs unbalanced (the card's 'zero by construction' holds for pair COUNTS, not gradient mass)
  T5 OFF path: the module-level pieces are byte-identical to the base trainer (REFE_BASE_SCORER or ../base) -- expected_reward_loss + one seeded AdamW step bit-identical; and the diff base -> new
     removes only the registered lines (signature of train_losses, the PREFLIGHT_PASS mutation guard)
  T6 parser: defaults OFF; pair_refusals cases; the flags are read by main()/train_losses (reachability: AST names)
  T7 reachability: argparse -> namespace (a.pair_w, a.pair_heads, a.pair_dv, a.pair_calibrate) -> pairer -> pair_term -> train_losses -> lB -> ((lA + lB) / a.accum).backward(): each hop is a line in the source
"""
import ast
import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import scorer_finetune as S  # noqa: E402

torch.manual_seed(0)
K, T = 64, 20


def make_set(gen_seed=1, v0=10.0):
    g = torch.Generator().manual_seed(gen_seed)
    cxy = torch.zeros(1, K, T, 3)
    speed = torch.rand(K, generator=g) * 14.0                              # m/s
    cxy[0, :, :, 0] = (speed[:, None] * 0.2) * torch.arange(1, T + 1)[None, :]       # constant speed -> chord = speed * 0.2 / 0.2
    tg = (torch.rand(1, K, 7, generator=g) > 0.5).float()
    cm = torch.ones(1, K)
    return cxy, tg, cm, torch.tensor([v0]), speed


# ---- T1 balance, empty side, seeding
cxy, tg, cm, v0, speed = make_set()
v4 = S.chord_speed(cxy)
assert torch.allclose(v4[0], speed, atol=1e-4), "chord_speed must recover the constant speed"
pl = S.PairLoss(["NC", "DAC", "TTC"], 0.15)
for h, col in S.PAIR_HEADS.items():
    ii, jj, cF, cS, kF, kS = pl.select(tg[0, :, col], v4[0], 10.0)
    assert kF == kS and ii.numel() == kF + kS, (h, kF, kS)
    assert cF != cS or cF == 0, "the probe set must be unbalanced in candidates for T2 to bite"
    assert min(cF, cS) == kF
p1, p2 = S.PairLoss(["NC"], 0.15), S.PairLoss(["NC"], 0.15)
a1, a2 = p1.select(tg[0, :, 0], v4[0], 10.0), p2.select(tg[0, :, 0], v4[0], 10.0)
assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]), "same seed -> same pairs"
p3 = S.PairLoss(["NC"], 0.15, seed=1)
a3 = p3.select(tg[0, :, 0], v4[0], 10.0)
assert not (torch.equal(a1[0], a3[0]) and torch.equal(a1[1], a3[1])), "another seed -> other pairs"
# empty side: every proposal is slower than the better ones -> label increases as speed decreases => only S pairs
y = (-v4[0]).clone()
ii, jj, cF, cS, kF, kS = pl.select(y, v4[0], 10.0)
assert cF == 0 and cS > 0 and ii.numel() == 0 and kF == kS == 0, "an empty F side contributes no pair"
print("T1 ok")

# ---- T2 NOBALANCE mutation
os.environ["REFE_PAIR_MUTATE_NOBALANCE"] = "1"
plm = S.PairLoss(["NC", "DAC", "TTC"], 0.15)
assert plm.balance is False
bad = 0
for h, col in S.PAIR_HEADS.items():
    ii, jj, cF, cS, kF, kS = plm.select(tg[0, :, col], v4[0], 10.0)
    bad += int(kF != kS)
del os.environ["REFE_PAIR_MUTATE_NOBALANCE"]
assert bad >= 1, "the F/S ratio check must be able to read RED under the NOBALANCE mutation"
print("T2 ok (F/S unequal on %d of 3 heads under the mutation)" % bad)

# ---- T3 gradient and the exact-zero control
logits = torch.randn(1, K, 7, requires_grad=True)
raw = S.PairLoss(["NC", "DAC", "TTC"], 0.15).term(logits, tg, cxy, v0, cm)
raw.backward()
raw = raw.detach().requires_grad_(False) if False else raw
assert torch.isfinite(raw) and float(logits.grad.abs().sum()) > 0.0 and float(raw) > 0.0
logits.grad = None
(0.0 * S.PairLoss(["NC", "DAC", "TTC"], 0.15).term(logits, tg, cxy, v0, cm)).backward()
assert logits.grad is not None and float(logits.grad.abs().sum()) == 0.0, "the w = 0 control must read EXACTLY 0.0"
none_l = torch.randn(1, K, 7, requires_grad=True)       # no kept pair at all -> a zero WITH graph
z = S.PairLoss(["NC"], 0.15).term(none_l, torch.zeros(1, K, 7), cxy, v0, cm)
z.backward()
assert float(z) == 0.0 and none_l.grad is not None and float(none_l.grad.abs().sum()) == 0.0
print("T3 ok")

# ---- T4 zero speed direction: a score that is a linear function of speed gets ZERO net gradient along the speed direction from the BALANCED pairs
spd = v4[0].clone().requires_grad_(True)
lg = torch.zeros(1, K, 7)
lg[0, :, 0] = spd
lg = lg.detach().requires_grad_(True)
S.PairLoss(["NC"], 0.15).term(lg, tg, cxy, v0, cm).backward()
gdir = float((lg.grad[0, :, 0] * (v4[0] - v4[0].mean())).sum())
unb = torch.zeros(1, K, 7); unb[0, :, 0] = v4[0]; unb.requires_grad_(True)
os.environ["REFE_PAIR_MUTATE_NOBALANCE"] = "1"
S.PairLoss(["NC"], 0.15).term(unb, tg, cxy, v0, cm).backward()
del os.environ["REFE_PAIR_MUTATE_NOBALANCE"]
gdir_unb = float((unb.grad[0, :, 0] * (v4[0] - v4[0].mean())).sum())
print("T4 speed-direction gradient: balanced %.3e  unbalanced %.3e" % (gdir, gdir_unb))
# NOT asserted: balancing equalises the pair COUNTS (T1), not the Dv-weighted gradient mass, so the speed-direction gradient is not exactly zero (SPEC_GAPS.md item 1).  Informational only.

# ---- T5 OFF path identical to the base trainer
base_path = os.environ.get("REFE_BASE_SCORER", os.path.join(HERE, "..", "..", "..", "..", "..", "base_refe", "scorer_finetune.py"))
if os.path.exists(base_path):
    spec = importlib.util.spec_from_file_location("scorer_finetune_base", base_path)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    sx = torch.randn(4, K, 7); tgt = (torch.rand(4, K, 7) > 0.4).float()
    assert torch.equal(B.expected_reward_loss(sx[..., :6], tgt, 0.1), S.expected_reward_loss(sx[..., :6], tgt, 0.1))

    def one_step(mod):
        torch.manual_seed(7)
        net = torch.nn.Linear(16, 7)
        x = torch.randn(4 * K, 16, generator=torch.Generator().manual_seed(3))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
        sxl = net(x).view(4, K, 7)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(sxl, tgt) + 1.0 * mod.expected_reward_loss(sxl[..., :6], tgt, 0.1)
        loss.backward(); opt.step()
        return float(loss), [p.detach().clone() for p in net.parameters()]
    lb, pb = one_step(B); ln, pn = one_step(S)
    assert lb == ln and all(torch.equal(x, y) for x, y in zip(pb, pn)), "OFF path: loss and updated parameters must be bit-identical"
    import difflib
    a_ = open(base_path, encoding="utf8").read().splitlines(); b_ = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf8").read().splitlines()
    removed = [l[2:] for l in difflib.unified_diff(a_, b_, lineterm="", n=0) if l.startswith("- ") and not l.startswith("--- ")]
    removed = [l for l in removed if l.strip()]
    assert len(removed) == 2 and any("def train_losses(sx0, sxA, sxB, tg, rows, upd, lam_pg=None):" in l for l in removed) and any("EPO_CON_MUTATE_ENVS)" in l for l in removed), removed
    print("T5 ok (OFF path bit-identical; the diff removes exactly 2 lines)")
else:
    print("T5 SKIPPED: base trainer not found at", base_path)

# ---- T6 parser + refusals
ap = S.build_parser()
base_argv = ["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x"]
d = ap.parse_args(base_argv)
assert (d.pair_w, d.pair_dv, d.pair_heads, d.pair_calibrate) == (0.0, 0.15, "", False) and S.pair_refusals(d) == []


def refs(*extra):
    return S.pair_refusals(ap.parse_args(base_argv + list(extra)))
assert refs("--pair-w", "0.3", "--pair-heads", "NC,TTC") == []
assert refs("--pair-w", "0.3") and refs("--pair-heads", "NC") and refs("--pair-w", "0.3", "--pair-heads", "EP") and refs("--pair-w", "-1", "--pair-heads", "NC")
assert refs("--pair-calibrate") and refs("--pair-calibrate", "--pair-heads", "NC") == [] and refs("--pair-calibrate", "--pair-heads", "NC", "--pair-w", "0.1")
assert refs("--pair-w", "0.3", "--pair-heads", "NC", "--pair-dv", "0") and refs("--pair-w", "0.3", "--pair-heads", "NC,NC") and refs("--pair-w", "0.3", "--pair-heads", "NC", "--epo")
print("T6 ok")

# ---- T7 reachability: the hops exist in the source
src = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf8").read()
tree = ast.parse(src)
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
reads = {n.attr for n in ast.walk(main) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "a"}
assert {"pair_w", "pair_dv", "pair_heads", "pair_calibrate"} <= reads, reads
for needle in ('ap.add_argument("--pair-w"', "pairer = PairLoss(", "lB = lB + pair_term(sxB, tg, pair_w)", "train_losses(_s0, sA, sB, tg, rows, upd)", "((lA + lB) / a.accum).backward()",
               'cap["pair_in"] = (cxy, ego[:, 6].float(), cm)', 'info["pair"] = {"w": float(a.pair_w)', "g_on, g_off = _flat_loss_grad(1.0), _flat_loss_grad(0.0)"):
    assert src.count(needle) == 1, needle
print("T7 ok")
print("ZZTEST_PASS")
