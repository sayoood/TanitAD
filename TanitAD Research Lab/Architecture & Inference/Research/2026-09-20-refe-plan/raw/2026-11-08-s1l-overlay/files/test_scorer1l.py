#!/usr/bin/env python3
"""S1L tests (CPU, no data): AMENDMENT_2 "The lane head" + "Conditions on S1L".  Prints ZZTEST_PASS.
  T1 lane pair rule (margin 0.1, masked < 0 excluded, near-tie excluded, speed contrast + F/S balance)
  T2 base identity of NC/DAC/TTC select() (S1L_BASE_TRAINER = the d689f924 trainer) and the defaults of every pair flag
  T3 gradient on the lane row: S1L > 0, S1 EXACTLY 0.0, weight-0 control EXACTLY 0.0
  T4 reachability hops (trainer file:line)
  T5 profiles + argv   T6 deliberate-regression arms through the real consumer + G-DVB rows
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402
import scorer_finetune as S  # noqa: E402

K, T = 64, 20


def speed_cxy(speed):
    cxy = torch.zeros(1, speed.numel(), T, 3)
    cxy[0, :, :, 0] = (speed[:, None] * 0.2) * torch.arange(1, T + 1)[None, :]
    return cxy


# ---- T1
g = torch.Generator().manual_seed(3)
v4 = S.chord_speed(speed_cxy(torch.rand(K, generator=g) * 14.0))[0]
t = torch.rand(K, generator=g)
t[:8] = -1.0
pl = S.PairLoss(["lane"], 0.15)
ii, jj, cF, cS, kF, kS = pl.select(t, v4, 10.0, head="lane")
assert ii.numel() > 0 and kF == kS, (cF, cS, kF, kS)
assert bool(((t[ii] - t[jj]) >= S.PAIR_LANE_MARGIN - 1e-7).all()) and bool((t[ii] >= 0).all() and (t[jj] >= 0).all())
assert bool(((v4[ii] - v4[jj]).abs() / 10.0 >= 0.15 - 1e-6).all())
e = pl.select(torch.tensor([0.55, 0.50]), torch.tensor([0.0, 9.0]), 10.0, head="lane")
assert e[0].numel() == 0, "a 0.05 near-tie is no pair"
e3 = pl.select(torch.full((K,), -1.0), v4, 10.0, head="lane")
assert e3[0].numel() == 0 and e3[2] == 0 and e3[3] == 0
print("T1 ok (kept %d+%d of cand F %d S %d)" % (kF, kS, cF, cS))

# ---- T2
base_p = os.environ.get("S1L_BASE_TRAINER")
if base_p and os.path.exists(base_p):
    spec = importlib.util.spec_from_file_location("sf_base", base_p)
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    for h in ("NC", "DAC", "TTC"):
        for n in range(20):
            y = torch.rand(K, generator=g)
            sp = S.chord_speed(speed_cxy(torch.rand(K, generator=g) * 14.0))[0]
            x1, x2 = B.PairLoss([h], 0.15).select(y, sp, 10.0), S.PairLoss([h], 0.15).select(y, sp, 10.0, head=h)
            assert all(torch.equal(p, q) if torch.is_tensor(p) else p == q for p, q in zip(x1, x2)), (h, n)
    assert {k: v for k, v in S.PAIR_HEADS.items() if k != "lane"} == B.PAIR_HEADS
    print("T2 ok vs the base trainer: NC/DAC/TTC select() bit-identical (3 x 20 sets)")
else:
    print("T2 base comparison SKIPPED (set S1L_BASE_TRAINER)")
ap = S.build_parser()
dflt = {a.dest: a.default for a in ap._actions if a.dest in ("pair_w", "pair_heads", "pair_dv", "pair_calibrate")}
assert dflt == {"pair_w": 0.0, "pair_dv": 0.15, "pair_heads": "", "pair_calibrate": False}, dflt
assert S.PAIR_LANE_MARGIN == 0.1 and S.PAIR_HEADS["lane"] == 6
_A = L.scorer1_argv("S1L", 0.37)
_n_ok, _e1 = C.parse_scorer_argv(S.build_parser(), _A)
_n_no, _e2 = C.parse_scorer_argv(S.build_parser(), [t_ for t_ in _A if t_ != "--lane-head"])
assert not _e1 and _n_ok is not None and not any("needs --lane-head" in r for r in S.pair_refusals(_n_ok))
assert _n_no is not None and any("needs --lane-head" in r for r in S.pair_refusals(_n_no))
print("T2 defaults ok (no new CLI flag; lane is a value of --pair-heads, default ''):", dflt)


# ---- T3
def grads(heads, w, lane_valid=True):
    torch.manual_seed(0)
    feat, head, gen = torch.randn(1, K, 256), torch.nn.Linear(256, 7), torch.Generator().manual_seed(11)
    spd = torch.rand(K, generator=gen) * 14.0
    tg = torch.zeros(1, K, 7)
    tg[0, :, :6] = (torch.rand(K, 6, generator=gen) > 0.5).float()
    tg[0, :, 6] = torch.rand(K, generator=gen) if lane_valid else -1.0
    loss = w * S.PairLoss(heads, 0.15).term(head(feat), tg, speed_cxy(spd), torch.tensor([10.0]), torch.ones(1, K))
    loss.backward()
    row = torch.cat([head.weight.grad[6].flatten(), head.bias.grad[6:7]]).abs().sum().item()
    other = torch.cat([head.weight.grad[[0, 1, 3]].flatten(), head.bias.grad[[0, 1, 3]]]).abs().sum().item()
    return row, other


r_s1l, o_s1l = grads(["NC", "DAC", "TTC", "lane"], 1.0)
r_s1, o_s1 = grads(["NC", "DAC", "TTC"], 1.0)
r_ctl, o_ctl = grads(["NC", "DAC", "TTC", "lane"], 0.0)
r_nolab, _ = grads(["NC", "DAC", "TTC", "lane"], 1.0, lane_valid=False)
assert r_s1l > 0.0 and o_s1l > 0.0 and r_s1 == 0.0 and o_s1 > 0.0 and r_ctl == 0.0 and o_ctl == 0.0 and r_nolab == 0.0
print("T3 ok: lane row |grad| S1L %.3e | S1 %r | w=0 control %r | unlabelled set %r" % (r_s1l, r_s1, r_ctl, r_nolab))

# ---- T4
src = open(os.path.join(HERE, "scorer_finetune.py"), encoding="utf8").read().split("\n")


def where(sub):
    for n, ln in enumerate(src, 1):
        if sub in ln:
            return n
    raise AssertionError("missing hop: " + sub)


hops = [("argparse --pair-heads", where('ap.add_argument("--pair-heads"')),
        ("namespace -> PairLoss", where("pairer = PairLoss([h for h in a.pair_heads.split(',') if h], a.pair_dv)")),
        ("pair_term -> term", where("raw = pairer.term(sxB, tg, cxy_, v0_, cm_)")),
        ("term -> select(head=h)", where("self.select(ycpu[b, :, col], v4[b], float(v0[b]), head=h)")),
        ("lane pair definition", where("better = ((y[:, None] - y[None, :]) >= PAIR_LANE_MARGIN)")),
        ("train_losses lB", where("lB = lB + pair_term(sxB, tg, pair_w)")),
        ("PAIR_HEADS col 6", where('"lane": 6}'))]
print("T4 hops:", "; ".join("%s :%d" % h for h in hops))

# ---- T5
ARG1L, ARG1 = L.scorer1_argv("S1L", 0.37), L.scorer1_argv("S1", 0.37)
assert ARG1L[ARG1L.index("--pair-heads") + 1] == "NC,DAC,TTC,lane" and ARG1[ARG1.index("--pair-heads") + 1] == "NC,DAC,TTC"
assert ARG1L[-1].endswith("/refe_scorer1/s1l") and ARG1[-1].endswith("/refe_scorer1/s1")
assert [t_ for t_ in ARG1L[:-1] if t_ != "NC,DAC,TTC,lane"] == [t_ for t_ in ARG1[:-1] if t_ != "NC,DAC,TTC"]
bash = shutil.which("bash")
if bash:
    with tempfile.TemporaryDirectory() as td:
        ws = td.replace(chr(92), "/")
        os.makedirs(ws + "/data/refe_scorer1/calibrate_s1l")
        json.dump({"pair_w": 0.37}, open(ws + "/data/refe_scorer1/calibrate_s1l/pair_weight.json", "w"))
        out = subprocess.run([bash, os.path.join(HERE, "scorer1_pod.sh")], env=dict(os.environ, ARM="S1L", MODE="argv", SCORER1_WS=ws),
                             capture_output=True, encoding="utf-8").stdout.strip().split()
    assert [t_.replace(ws, "/workspace") for t_ in out] == ARG1L, out[-8:]
    print("T5 ok: pod script ARM=S1L MODE=argv == scorer1_argv('S1L', 0.37)")
ns1l, err = C.parse_scorer_argv(S.build_parser(), ARG1L)
assert not err, err
ns1, err = C.parse_scorer_argv(S.build_parser(), ARG1)
assert not err, err
P1L, P1 = L.PROFILES["refe_scorer1l"], L.PROFILES["refe_scorer1"]


def refusals(prof, argv):
    return L.scorer1_refusals(prof, argv, S, S.build_parser())[0]


assert any("is none of the arm dirs" in r for r in refusals(P1, ARG1L)) and any("is none of the arm dirs" in r for r in refusals(P1L, ARG1))
badh = [t_ if t_ != "NC,DAC,TTC,lane" else "NC,DAC,TTC" for t_ in ARG1L]
assert any("pinned" in r.lower() or "pin" in r.lower() or "NC,DAC,TTC,lane" in r for r in refusals(P1L, badh))
print("T5 ok: cross-profile refusals; S1L without the lane head refused")


# ---- T6
def pairs_for(heads, nobalance):
    for e_ in S.PAIR_MUTATE_ENVS:
        os.environ.pop(e_, None)
    if nobalance:
        os.environ["REFE_PAIR_MUTATE_NOBALANCE"] = "1"
    try:
        pl_, gg, mb, bank = S.PairLoss(heads, 0.15), torch.Generator().manual_seed(5), {}, {}
        for h in heads:
            kf = ks = sets = withp = lab = 0
            for _ in range(40):
                sp = S.chord_speed(speed_cxy(torch.rand(K, generator=gg) * 14.0))[0]
                if h == "lane":
                    y = torch.rand(K, generator=gg)
                    y[: int(torch.randint(0, 20, (1,), generator=gg))] = -1.0
                else:
                    y = (torch.rand(K, generator=gg) > 0.5).float()
                _i, _j, _a, _b, kF_, kS_ = pl_.select(y, sp, 10.0, head=h)
                sets += 1; withp += int(kF_ + kS_ > 0); lab += int(bool((y >= 0).any())); kf += kF_; ks += kS_
            mb[h] = {"kept_F": kf, "kept_S": ks}
            bank[h] = {"kept_F": kf, "kept_S": ks, "share_with_pair": float(withp / sets), "with_pair": withp, "sets": sets}
            if h == "lane":
                bank[h]["labelled"] = lab
        return mb, bank, bool(pl_.balance)
    finally:
        for e_ in S.PAIR_MUTATE_ENVS:
            os.environ.pop(e_, None)


def lane_row_delta(heads):
    return grads(heads, 1.0)[0] - grads(heads, 0.0)[0]


def g16(mode):
    heads = ["NC", "DAC", "TTC", "lane"]
    mb, bank, bal = pairs_for(heads, mode == "lane_nobalance")
    delta = lane_row_delta(heads if mode != "lane_drop" else ["NC", "DAC", "TTC"])
    if mode == "lane_share0":
        bank["lane"].update({"with_pair": 0, "share_with_pair": 0.0})
    return {"loss_grad_delta_w1_minus_w0": {m: 1.5e-3 for m in C.SCORER_GROUPS}, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS},
            "controls_exactly_zero": True, "micro_batch_pairs": mb, "bank_share": bank, "balanced_built": bal, "mutate_nobalance": False,
            "mutate_drop_loss": False, "lane_row": {"sets": 4, "delta_lane_row_w1_minus_w0": float(delta)}, "ok": True}


def consume(g16d, argv, prof, ns_):
    gates = {g_ + "_stub": {"ok": True} for g_ in C.scorer_required_gates(ns_) if g_ != "G16"}
    gates["G16_pair_live_balanced"] = g16d
    for k in list(gates):
        if k.startswith(("G11", "G12", "G13")):
            gates.pop(k)
    with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
        p = os.path.join(td, "preflight.jsonl")
        with open(p, "w") as fh:
            fh.write(json.dumps({"event": "data", "train_sets": 1}) + "\n")
            fh.write(json.dumps({"event": "preflight", "verdict": "PASS", "gates": {k: (dict(v, ok=True) if k.startswith("G16") else v) for k, v in gates.items()}}) + "\n")
        return L.consume_scorer_preflight(p, argv, prof)


ctl = consume(g16("control"), ARG1L, P1L, ns1l)
assert ctl["status"] == "PASS", ctl["reasons"]
ld = consume(g16("lane_drop"), ARG1L, P1L, ns1l)
assert ld["status"] == "FAIL" and any("lane row of the score head does not move" in r for r in ld["reasons"]), ld["reasons"]
lb = consume(g16("lane_nobalance"), ARG1L, P1L, ns1l)
assert lb["status"] == "FAIL" and any("F/S ratio != 1 on head lane" in r for r in lb["reasons"]), lb["reasons"]
l0 = consume(g16("lane_share0"), ARG1L, P1L, ns1l)
assert l0["status"] == "FAIL" and any("share 0" in r or "no lane-labelled set" in r for r in l0["reasons"]), l0["reasons"]
print("T6 control PASS; LANE-DROP RED; LANE-NOBALANCE RED; lane share 0 RED")
mb, bank, bal = pairs_for(["NC", "DAC", "TTC"], False)
gs1 = {"loss_grad_delta_w1_minus_w0": {m: 1.5e-3 for m in C.SCORER_GROUPS}, "control_w0_term_alone_grad_abs": {m: 0.0 for m in C.SCORER_GROUPS}, "controls_exactly_zero": True,
       "micro_batch_pairs": mb, "bank_share": bank, "balanced_built": bal, "mutate_nobalance": False, "mutate_drop_loss": False, "ok": True}
assert consume(gs1, ARG1, P1, ns1)["status"] == "PASS"
assert any("lane_row block without the lane head" in r for r in C.judge_g16(dict(gs1, lane_row={"sets": 1, "delta_lane_row_w1_minus_w0": 0.0}), ns1))
good = {"pair": {"w": 0.37, "dv": 0.15, "heads": ["NC", "DAC", "TTC", "lane"], "balanced": True, "seed": 20261107, "lane_col": 6, "lane_margin": 0.1}}
bad = {"pair": dict(good["pair"], lane_col=5, lane_margin=0.2)}
rg = [r for r in C.scorer_dvb_rows(ns1l, good, P1L) if "pair.lane" in r["how"]]
rb = [r for r in C.scorer_dvb_rows(ns1l, bad, P1L) if "pair.lane" in r["how"]]
assert len(rg) == 2 and all(r["ok"] for r in rg) and len(rb) == 2 and not any(r["ok"] for r in rb)
print("T6 S1 consumer unchanged; G-DVB lane rows ok (match ok, wrong column/margin RED)")
print("ZZTEST_PASS")
