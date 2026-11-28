#!/usr/bin/env python3
"""S1Z CPU preflight of the REAL scorer_finetune.main() on a tiny 6-OUTPUT stub model whose bank holds near-twin pairs (PREREG_S1Z_P6 section 2).  Prints ZZTEST_PASS.

  A OFF PATH: no --twin-* flag runs bit-identical (rows and every saved tensor) to the TIP trainer (s1z_layer.peel of this file = git blob f5c8b769..., commit 62df1e64), and writes no `twin` block
  B S1Z preflight: PREFLIGHT_PASS written, G18 ok (live on every scorer module, control EXACTLY 0.0, every head's share of sets with a kept pair > 0 on the live logits and on the labels), the data event's
    `twin` block equals the argv, every G-DVB row is green, judge_g18 passes; the weight-0 assembly has bit-identical gradients to the stock loss (injected into a COPY of the trainer); --twin-topk 1 changes
    the mining (the flag reaches the loss object)
  C DELIBERATE REGRESSIONS withhold PREFLIGHT_PASS, read G18 ok false and turn judge_g18 / the G-DVB rows RED: the radius filter removed, the top-K mining removed, L_twin dropped from the loss (by env AND by
    deleting the line from a copy of the trainer)
  D --pair-calibrate with --twin-heads writes a finite positive lambda_z with every head's share > 0 over 50 micro-batches
  E inconsistent --twin-* argvs never start (rc 4)
  F the training loop: 25 updates; the per-head share of sets with a kept pair is in the trainer's log and in the done event; arm A is bit-equal to the OFF run's arm A, arm B is not
  H G18 PROBE BATCH + NOISE FLOOR (2026-11-10 fix): the gradient probe runs on a micro-batch that holds a kept pair on EVERY head (a pairless micro-batch is never used; none within the search limit = G18 fails
    and names the limit); delta(w=1 vs w=0) must exceed 10x the w=0 vs w=0 delta on every scorer module (seeded logit noise injected: a correct trainer passes with tiny noise, fails when noise reaches delta; a
    dropped loss under noise has delta ~ noise and fails by the rule alone)
  G a navtest log or token among the training sets is COUNTED by the trainer's audit and the S1Z profile's G-DVB expect rows go RED
Evidence class: MEASURED on a stub trunk; the real ViT-L trunk, the real bank and bf16 autocast are NOT exercised here (the pod preflight does that).
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gate_checks as C  # noqa: E402
import launch_gate as L  # noqa: E402
import s1z_layer as ZL  # noqa: E402
import scorer_finetune as S  # noqa: E402
import test_aux_s1_wiring as AUXT  # noqa: E402
import test_rl_epo as TR  # noqa: E402

SRC = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n")
VOLATILE = AUXT.VOLATILE
M_PROP, T_H = 8, 20
TWIN_ARGS = ["--twin-heads", "NC,DAC,TTC", "--twin-ade", "1.0", "--twin-topk", "8"]
TW = TWIN_ARGS + ["--twin-w", "0.3"]
HEADS = ("NC", "DAC", "TTC")
SPARSE = {"keep": None}          # None = every set holds a near-twin hard negative per head; else only the sets whose index n has n % 16 in `keep` do (the others are all clean: no pair on any head)
NOISE = {"sigma": 0.0, "gen": None}      # seeded additive noise on the score logits of EVERY forward pass (a stand-in for GPU run-to-run noise between two forward passes)


class TwinBank(TR.StubBank):
    """8 proposals = 4 groups of 2 near-twins (0.5 m apart, groups 20 m apart).  Proposal 2g is clean; proposal 2g+1 is clean for one group of each set and fails NC, DAC or TTC in the others
    ((set index + g) % 4 = 0, 1, 2 -> NC, DAC, TTC; 3 -> clean), so EVERY set holds exactly one near-twin hard negative per head"""

    def __init__(self, path, n_prop, horizon, lane_labels=None, pdm_labels=None, teacher_lane_labels=None):
        super().__init__(path, n_prop, horizon, lane_labels=lane_labels, pdm_labels=pdm_labels, teacher_lane_labels=teacher_lane_labels)
        for n, k in enumerate(list(self.by)):
            e = self.by[k]
            tr = np.zeros((n_prop, T_H, 3), np.float32)
            tg = np.zeros((n_prop, 6), np.float32)
            for grp in range(4):
                for idx, off in ((2 * grp, 0.0), (2 * grp + 1, 0.5)):
                    tr[idx, :, 0] = 8.0 * 0.2 * np.arange(1, T_H + 1)
                    tr[idx, :, 1] = 20.0 * grp + off
                tg[2 * grp] = [1, 1, 0.8, 1, 1, 1]
                tg[2 * grp + 1] = [1, 1, 0.8, 1, 1, 1]
                fail = (n + grp) % 4
                if fail == 0:
                    tg[2 * grp + 1, [0, 2]] = 0.0
                elif fail == 1:
                    tg[2 * grp + 1, [1, 2]] = 0.0
                elif fail == 2:
                    tg[2 * grp + 1, 3] = 0.0
                    tg[2 * grp + 1, 2] = 0.5
            if SPARSE["keep"] is not None and (n % 16) not in SPARSE["keep"]:
                tg[:] = [1, 1, 0.8, 1, 1, 1]
            self.by[k] = (e[0], tr, tg, e[3], e[4])

    def get(self, log_name, token, step, rank=0):
        e = self.by.get((log_name, token, int(step), int(rank)))
        return None if e is None else (e[1], e[2], e[0])


class TwinTargets(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        tr, ho = TwinBank("train", M_PROP, T_H), TwinBank("held", M_PROP, T_H)
        self.rows = [{"log_name": k[0], "token": k[1], "step": k[2], "rank": k[3], "ego": [0.0] * 6 + [8.0, 0.0]} for k in list(tr.by) + list(ho.by)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        e = self.onpolicy.by[(r["log_name"], r["token"], int(r["step"]), int(r["rank"]))]
        g = torch.Generator().manual_seed(1000 + i)
        img = torch.randn(8, 3, generator=g) * (1.0 if i < 100 else 20.0)
        ego = torch.zeros(8)
        ego[6] = 8.0
        return (img, ego, torch.zeros(2), torch.zeros(1), torch.as_tensor(e[1]), torch.as_tensor(e[2]), torch.ones(M_PROP), torch.full((M_PROP,), 5), torch.zeros(0))


class TwinModel(AUXT.Stub256REFe):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def score_trajectories(self, trajs, sctx):
        s = AUXT.Stub256REFe.score_trajectories(self, trajs, sctx)
        if NOISE["sigma"] > 0.0:
            s = s + NOISE["sigma"] * torch.randn(s.shape, generator=NOISE["gen"])
        return s


INJECT_ANCHOR = "                ctl = twin_term(sB_, t_, 0.0)"
INJECT = '''                armB.zero_grad(set_to_none=True)
                _s0i, _sAi, _sBi, _ti = fwd(pb_batch)
                _lp = losses(_sAi, _sBi, _ti, a.lam)[1]
                _lo = train_losses(_s0i, _sAi, _sBi, _ti, pb_rows, 0, twin_w=0.0)[1]
                _prm = [p_ for m_ in SCORER for p_ in getattr(armB, m_).parameters()]
                _gp = torch.autograd.grad(_lp, _prm, retain_graph=True, allow_unused=True)
                _go = torch.autograd.grad(_lo, _prm, retain_graph=True, allow_unused=True)
                IDENT.append(bool(float(_lp) == float(_lo) and all(((x_ is None) == (y_ is None)) and (x_ is None or torch.equal(x_, y_)) for x_, y_ in zip(_gp, _go))))
                armB.zero_grad(set_to_none=True)
'''
DROP_LINE = "            lB = lB + twin_term(sxB, tg, twin_w)\n"


def run(td, tag, extra=(), env=None, meta=None, modes=("pre",), src=SRC, max_updates=6, inject=False, mutate_src=None, files=None):
    """the REAL main() of `src` on CPU against the stub trunk -> {rc_<mode>, rows, tensors, ident}"""
    import ckpt_io as real_ckpt_io
    csrc = (src.replace("pin_memory=True", "pin_memory=False").replace("num_workers=2", "num_workers=0").replace("n=10000, seed=20260927", "n=40, seed=20260927"))
    csrc = csrc.replace('dev = "cuda"', 'dev = "cpu"').replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", enabled=False)')
    if inject:
        assert csrc.count(INJECT_ANCHOR) == 1
        csrc = csrc.replace(INJECT_ANCHOR, INJECT + INJECT_ANCHOR)
    if mutate_src:
        assert csrc.count(mutate_src[0]) == 1, mutate_src[0]
        csrc = csrc.replace(mutate_src[0], mutate_src[1])
    sm = TR.load(csrc, f"s1zpre_{tag}")
    sm.IDENT = []

    class StubCkptIO:
        FORMAT_FULL = "full"
        config_for_checkpoint = staticmethod(real_ckpt_io.config_for_checkpoint)

        @staticmethod
        def load_for_inference(model, path, map_location="cpu", backbone=None, meta_out=None):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(sd["model"], strict=True)
            if meta_out is not None:
                meta_out.update(sd.get("meta", {}) or {})
            return "full"

        @staticmethod
        def trainable_names(model):
            return {n for n, p in model.named_parameters() if p.requires_grad}

        @staticmethod
        def atomic_save(obj, path):
            torch.save(obj, path)
    sm.REFe, sm.REFeConfig, sm.ckpt_io = TwinModel, TR.StubCfgFactory, StubCkptIO
    sm.T = types.SimpleNamespace(OnPolicyBank=TwinBank, TargetBank=TwinTargets)
    torch.manual_seed(0)
    ck = os.path.join(td, f"ck_{tag}.pt")
    torch.save({"format": "full", "model": TwinModel(types.SimpleNamespace(n_score_components=6)).state_dict(), "meta": dict(meta or {})}, ck)
    out = os.path.join(td, f"out_{tag}")
    with open(os.path.join(td, "ho_logs.txt"), "w") as fh:
        fh.write(" ".join(f"H{i}" for i in range(4)))
    args = ["--ckpt", ck, "--targets", "t", "--images", "i", "--calib", "c", "--onpolicy", "train", "--heldout", "held", "--heldout-logs", os.path.join(td, "ho_logs.txt"), "--out", out,
            "--batch", "2", "--accum", "2", "--workers", "0", "--max-updates", str(max_updates), "--eval-every", str(max_updates), "--warmup", "2", "--lr", "1e-3", "--b-mode", "expreward",
            "--lam", "1.0", "--pdm-labels-train", "d", "--pdm-labels-heldout", "d", "--require-pdm", *extra]
    saved_argv, saved_cuda = sys.argv, (torch.cuda.synchronize, torch.cuda.max_memory_allocated)
    torch.cuda.synchronize, torch.cuda.max_memory_allocated = (lambda *a, **k: None), (lambda *a, **k: 0)
    saved_env = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    res = {"out": out, "argv": args}
    try:
        for mode in modes:
            sys.argv = ["scorer_finetune.py"] + args + (["--preflight"] if mode == "pre" else [])
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    res["rc_" + mode] = sm.main()
            except Exception as e:                                          # noqa: BLE001 -- a crash IS a result
                res["rc_" + mode] = f"EXC {type(e).__name__}: {str(e)[:300]}"
            res["out_" + mode] = buf.getvalue()
    finally:
        sys.argv = saved_argv
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = saved_cuda
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    res["ident"] = list(sm.IDENT)
    return AUXT.harvest(res, out)


def pre_rows(res):
    rows = res["rows"]["preflight.jsonl"]
    return next(r for r in rows if r.get("event") == "data"), [r for r in rows if r.get("event") == "preflight"][-1]


def norm(rows):
    return [{k: v for k, v in r.items() if k not in VOLATILE and k != "out"} for r in rows]


def sd_of(x):
    return x["model"] if "model" in x else x


def g18_of(pre):
    return next(v for k, v in pre["gates"].items() if k.startswith("G18"))


with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
    # ---- A the OFF path against the TIP trainer
    tip = ZL.peel(SRC)
    assert tip != SRC and not ZL.has_layer(tip)
    base = run(td, "tip", src=tip, modes=("pre", "run"))
    new = run(td, "new", modes=("pre", "run"))
    for name, r in (("new", new),):
        assert base["rc_pre"] == r["rc_pre"] == 0 and base["rc_run"] == r["rc_run"] == 0, (base.get("rc_pre"), base.get("rc_run"), r.get("rc_pre"), r.get("rc_run"), r.get("out_pre", "")[-300:])
        for fn in ("preflight.jsonl", "sft.jsonl"):
            a_, b_ = norm(base["rows"][fn]), norm(r["rows"][fn])
            assert len(a_) == len(b_) and a_ == b_, (name, fn)
        for arm in ("A", "B"):
            sa, sb = sd_of(base[arm]), sd_of(r[arm])
            assert all(torch.equal(sa[k], sb[k]) for k in sa), (name, arm, "a saved tensor moved")
    assert "twin" not in pre_rows(new)[0] and all("G18" not in k for k in pre_rows(new)[1]["gates"]), "the OFF data event and gates carry nothing of S1Z"
    print("A ok (no --twin-* flag: rows and saved tensors bit-identical to the tip trainer)")

    # ---- B S1Z preflight
    b = run(td, "s1z", extra=TW, meta={"nav_mode": "command"}, inject=True)
    assert b["rc_pre"] == 0 and b["pass_json"] is not None, (b["rc_pre"], b.get("out_pre", "")[-600:])
    data, pre = pre_rows(b)
    g18 = g18_of(pre)
    assert g18["ok"] is True and g18["controls_exactly_zero"] is True and all(v == 0.0 for v in g18["control_w0_term_alone_grad_abs"].values()), g18
    assert all(v > 0.0 for v in g18["loss_grad_delta_w1_minus_w0"].values()) and set(g18["loss_grad_delta_w1_minus_w0"]) == set(C.SCORER_GROUPS), g18["loss_grad_delta_w1_minus_w0"]
    assert g18["micro_batches"] == S.TWIN_PREFLIGHT_MICRO == 16
    for h in HEADS:
        m_, bk = g18["share_micro"][h], g18["bank_share"][h]
        assert m_["share_with_pair"] == 1.0 and m_["pairs"] == m_["sets"] == 32 and bk["share_with_pair"] == 1.0, (h, m_, bk)      # every set holds exactly one near-twin hard negative per head
    assert not any(k.startswith("G16") for k in pre["gates"]) and "pair" not in data
    assert data["twin"] == {"w": 0.3, "heads": list(HEADS), "ade": 1.0, "topk": 8, "radius_filter": True, "mining": True, "calibrate": False, "ade_metric": "mean_xy_euclid_over_raw_poses",
                            "clean": "NC=1,DAC=1,TTC=1"}, data["twin"]
    ns, err = C.parse_scorer_argv(S.build_parser(), b["argv"])
    assert not err, err
    assert "G18" in C.scorer_required_gates(ns) and "G16" not in C.scorer_required_gates(ns)
    rows = C.scorer_dvb_rows(ns, data, {"name": "stub"})
    twin_rows = [r for r in rows if r["dest"].startswith("twin")]
    assert len(twin_rows) == 7 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert C.judge_g18(g18, ns) == [], C.judge_g18(g18, ns)
    assert b["ident"] == [True], ("weight 0 must give bit-identical gradients to the stock loss", b["ident"])
    # the flag reaches the loss object: --twin-topk 1 mines less (the hard negative must be the single top proposal of the rule)
    b1 = run(td, "s1z_k1", extra=[t if t != "8" else "1" for t in TW], meta={"nav_mode": "command"})
    d1, p1 = pre_rows(b1)
    assert d1["twin"]["topk"] == 1 and all(g18_of(p1)["share_micro"][h]["pairs"] < 32 for h in HEADS), [g18_of(p1)["share_micro"][h] for h in HEADS]
    print("B ok (G18 ok, shares 1.0 on the twin bank, `twin` block == argv, 7 G-DVB twin rows green, weight 0 -> bit-identical gradients, --twin-topk reaches the mining)")

    # ---- C deliberate regressions
    for tag, env, ms, frag, dvb in (("noradius", {"REFE_TWIN_MUTATE_NORADIUS": "1"}, None, "radius filter is missing", "twin_ade"),
                                    ("notopk", {"REFE_TWIN_MUTATE_NOTOPK": "1"}, None, "top-K mining is missing", "twin_topk"),
                                    ("droploss", {"REFE_TWIN_MUTATE_DROP_LOSS": "1"}, None, "not live IN the loss", None),
                                    ("srcdrop", None, (DROP_LINE, "            pass\n"), "not live IN the loss", None)):
        m = run(td, "mut_" + tag, extra=TW, env=env, meta={"nav_mode": "command"}, mutate_src=ms)
        assert m["pass_json"] is None and m["rc_pre"] != 0, (tag, m["rc_pre"])
        assert any(r.get("event") == "preflight" for r in m["rows"]["preflight.jsonl"]), (tag, m["rc_pre"], m["out_pre"][-500:])
        d_m, pre_m = pre_rows(m)
        gm = g18_of(pre_m)
        assert gm["ok"] is False, (tag, gm)
        reasons = C.judge_g18(gm, ns)
        assert reasons and any(frag in r for r in reasons), (tag, reasons)
        if dvb:
            assert any(not r["ok"] and r["dest"] == dvb for r in C.scorer_dvb_rows(ns, d_m, {"name": "stub"})), (tag, "the G-DVB row must go RED too")
    print("C ok (NORADIUS, NOTOPK, DROP_LOSS and a trainer with the line deleted: no PREFLIGHT_PASS, G18 ok false, judge_g18 RED, the built-fact G-DVB rows RED)")

    # ---- D calibrate
    d = run(td, "cal", extra=TWIN_ARGS + ["--pair-calibrate"], meta={"nav_mode": "command"}, modes=("run",))
    pw = os.path.join(d["out"], "pair_weight.json")
    assert d["rc_run"] == 0 and "ZZPAIR_CALIBRATED" in d["out_run"] and os.path.exists(pw), (d["rc_run"], d.get("out_run", "")[-400:])
    j = json.load(open(pw))
    assert j["micro_batches"] == 50 and j["twin_w"] and j["twin_w"] > 0 and "pair_w" not in j and j["lever"] == "twin" and j["ade"] == 1.0 and j["topk"] == 8, j
    assert set(j["sets_with_pair_share"]) == set(HEADS) and all(v == 1.0 for v in j["sets_with_pair_share"].values()) and sorted(j["heads"]) == sorted(HEADS), j
    cal_data = next(r for r in d["rows"]["sft.jsonl"] if r.get("event") == "data")
    assert cal_data["twin"]["calibrate"] is True and cal_data["twin"]["w"] == 0.0 and "pair" not in cal_data
    print(f"D ok (lambda_z {j['twin_w']:.4g} over 50 micro-batches)")

    # ---- E inconsistent argvs never start
    for tag, extra in (("e1", TWIN_ARGS), ("e2", ["--twin-w", "0.3"]), ("e3", ["--twin-w", "0.3", "--twin-heads", "EP"]), ("e4", TW + ["--pair-heads", "NC", "--pair-w", "0.2"]),
                       ("e5", TWIN_ARGS[:-2] + ["--twin-topk", "0", "--twin-w", "0.3"]), ("e6", ["--pair-calibrate"])):
        e = run(td, tag, extra=extra, modes=("pre",))
        assert e["rc_pre"] == 4 and "REFUSED" in e["out_pre"], (tag, e["rc_pre"], e["out_pre"][-300:])
    print("E ok")

    # ---- F the training loop
    base25 = run(td, "off25", meta={"nav_mode": "command"}, modes=("pre", "run"), max_updates=25)
    f = run(td, "s1z25", extra=TW, meta={"nav_mode": "command"}, modes=("pre", "run"), max_updates=25)
    assert f["rc_pre"] == 0 and f["rc_run"] == 0 and base25["rc_run"] == 0, (f["rc_pre"], f["rc_run"], f.get("out_run", "")[-400:])
    sa0, sa1 = sd_of(base25["A"]), sd_of(f["A"])
    sb0, sb1 = sd_of(base25["B"]), sd_of(f["B"])
    assert all(torch.equal(sa0[k], sa1[k]) for k in sa0), "arm A is the BCE-only control: bit-equal to the OFF run's arm A"
    assert any(not torch.equal(sb0[k], sb1[k]) for k in sb0), "arm B must move with L_twin"
    assert not np.allclose(sd_of(f["B"])["score_head.weight"].numpy(), sd_of(f["A"])["score_head.weight"].numpy())
    srows = f["rows"]["sft.jsonl"]
    trow = next(r for r in srows if r.get("event") == "train")
    assert set(trow["twin"]) == set(HEADS) and all(v["share_with_pair"] == 1.0 and v["sets"] == 100 and v["pairs"] == 100 for v in trow["twin"].values()), trow["twin"]
    done = next(r for r in srows if r.get("event") == "done")
    assert set(done["twin_share"]) == set(HEADS) and all(v["share_with_pair"] == 1.0 for v in done["twin_share"].values()), done["twin_share"]
    assert not any(("twin" in r or "twin_share" in r) for r in base25["rows"]["sft.jsonl"]), "the OFF run logs no twin share"
    print("F ok (25 updates: arm A bit-equal to the OFF run, arm B moved, the per-head share is in the train row and the done event)")

    # ---- H G18 probe batch and noise floor
    def with_world(keep, sigma, tag, env=None):
        SPARSE["keep"], NOISE["sigma"], NOISE["gen"] = keep, sigma, torch.Generator().manual_seed(4242)
        try:
            return run(td, tag, extra=TW, meta={"nav_mode": "command"}, env=env)
        finally:
            SPARSE["keep"], NOISE["sigma"], NOISE["gen"] = None, 0.0, None
    # H1 sparse bank (1 set in 8 holds pairs): the probe batch is searched for, holds a pair on every head, and the gradient is read there
    h1 = with_world({3, 11}, 0.0, "h1")
    assert h1["rc_pre"] == 0 and h1["pass_json"] is not None, (h1["rc_pre"], h1.get("out_pre", "")[-400:])
    g1 = g18_of(pre_rows(h1)[1])
    pbk = g1["probe_batch"]
    assert g1["ok"] is True and pbk["found"] is True and pbk["reason"] is None and pbk["search_limit"] == S.TWIN_PROBE_SEARCH_MAX == 400, g1
    assert pbk["searched"] == pbk["index"] + 1 and set(pbk["pairs_per_head"]) == set(HEADS), pbk
    assert all(pbk["pairs_per_head"][h_]["sets_with_pair"] > 0 and pbk["pairs_per_head"][h_]["pairs"] > 0 for h_ in HEADS), pbk      # a pairless micro-batch is never the probe batch
    assert all(g1["micro_batch_pairs"][h_]["sets_with_pair"] > 0 for h_ in HEADS), g1["micro_batch_pairs"]                              # the w=1 pass really saw pairs
    assert all(v == 0.0 for v in g1["loss_grad_noise_w0_vs_w0"].values()) and all(v > 0.0 for v in g1["loss_grad_delta_w1_minus_w0"].values()), g1
    assert pbk["index"] > 0, ("the sparse bank must have forced the search past at least one pairless micro-batch", pbk)
    # H2 a bank with no pair anywhere: no probe batch within the limit, G18 fails and says so; nothing is passed
    h2 = with_world(set(), 0.0, "h2")
    g2 = g18_of(pre_rows(h2)[1])
    assert h2["pass_json"] is None and g2["ok"] is False and g2["probe_batch"]["found"] is False and g2["probe_batch"]["searched"] == 400, g2["probe_batch"]
    assert "search limit of 400" in g2["probe_batch"]["reason"], g2["probe_batch"]
    # H3 tiny seeded noise: a correct trainer still passes (delta is far above 10x noise); the noise is really read (non-zero)
    h3 = with_world(None, 1e-6, "h3")
    g3 = g18_of(pre_rows(h3)[1])
    assert g3["ok"] is True and g3["noise_rule_ok"] is True and all(v > 0.0 for v in g3["loss_grad_noise_w0_vs_w0"].values()), g3
    assert all(g3["delta_over_noise"][m_] > 10.0 for m_ in C.SCORER_GROUPS), g3["delta_over_noise"]
    # H4 noise as large as the effect: the 10x rule fails on a CORRECT trainer (no mutation env), and judge-visible fields say why
    h4 = with_world(None, 1.0, "h4")
    g4 = g18_of(pre_rows(h4)[1])
    assert h4["pass_json"] is None and g4["ok"] is False and g4["noise_rule_ok"] is False and not g4["mutate_drop_loss"], g4
    assert any(g4["delta_over_noise"][m_] is not None and g4["delta_over_noise"][m_] <= 10.0 for m_ in C.SCORER_GROUPS), g4["delta_over_noise"]
    # H5 the dropped loss under noise: delta is noise (w=1 and w=0 are the same assembly); the rule alone fails every module, with the env read as a bonus
    h5 = with_world(None, 1e-3, "h5", env={"REFE_TWIN_MUTATE_DROP_LOSS": "1"})
    g5 = g18_of(pre_rows(h5)[1])
    assert h5["pass_json"] is None and g5["ok"] is False and g5["noise_rule_ok"] is False and all(g5["delta_over_noise"][m_] is not None and g5["delta_over_noise"][m_] < 10.0 for m_ in C.SCORER_GROUPS), g5
    print("H ok (probe batch holds a pair on every head and is found past a pairless one; no pair within 400 -> explicit failure; tiny noise passes, noise == effect fails, a dropped loss under noise fails by the 10x rule)")

    # ---- G a navtest log / token among the training sets
    nl = os.path.join(td, "nt_logs.json")
    json.dump({"logs": ["T0", "T1"]}, open(nl, "w"))
    nt = os.path.join(td, "nt_tokens.txt")
    open(nt, "w").write("tok3 tok4 tok5")
    g = run(td, "navtest", extra=TW + ["--navtest-logs", nl, "--navtest-tokens", nt], meta={"nav_mode": "command"})
    assert g["rc_pre"] == 0, g.get("out_pre", "")[-300:]
    dg, _pg = pre_rows(g)
    assert dg["navtest"]["train_rows_in_navtest_logs"] == 8 and dg["navtest"]["train_rows_in_navtest_tokens"] == 3 and dg["navtest"]["logs"] == 2 and dg["navtest"]["tokens"] == 3, dg["navtest"]
    nsg, _e = C.parse_scorer_argv(S.build_parser(), g["argv"])
    rows_g = {r["flag"]: r for r in C.scorer_expect_rows(L.PROFILES["refe_s1z_p6"], nsg, dg)}
    for key in ("train_rows_in_navtest_logs", "train_rows_in_navtest_tokens"):
        assert rows_g["data.navtest." + key]["ok"] is False, (key, rows_g["data.navtest." + key])
    print("G ok (the trainer counted 8 training sets in the navtest logs and 3 in its tokens; the S1Z profile's expect rows are RED)")
print("ZZTEST_PASS")
