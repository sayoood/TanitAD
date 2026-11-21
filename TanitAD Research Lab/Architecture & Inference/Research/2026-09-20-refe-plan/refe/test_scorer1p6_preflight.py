#!/usr/bin/env python3
"""SCORER1P6 CPU preflight of the REAL scorer_finetune.main() on a tiny 6-OUTPUT model (a 256-wide stub trunk, a 6-column bank with EP labels, no lane).  The 6-row path had never run.  Prints ZZTEST_PASS.

  A OFF PATH: no --pair-* flag, and `--pair-w 0`, run bit-identical (rows and every saved tensor) to the PRE-SCORER1 trainer (scorer1_layer.peel of this file = git blob 51a28163)
  B S1E preflight, 6 outputs: PREFLIGHT_PASS written, G16 ok with the per-head rules built (EP unbalanced / unfiltered / gap 0.05, safety heads balanced), every head's share of sets with a kept pair > 0,
    the data event carries pair.rules, pair.reduce and nav_mode=command; every G-DVB row of the pair levers and the nav_mode row is GREEN against the BUILT block, and a flipped rule turns its row RED
  C the three deliberate regressions each withhold PREFLIGHT_PASS and read G16 ok false: NOBALANCE (a safety head loses balancing), EPBALANCE (EP gets balancing), DROP_LOSS (L_pair dropped from the loss)
  D --pair-calibrate runs on 6 outputs and writes a finite positive lambda_p with every head's share of sets with a pair > 0
  E nav_mode: a checkpoint whose meta says command writes nav_mode=command; one that lost it writes nothing (and the profile's data_expect row would be RED)
  F the S1E training loop runs (6 updates) on the 6-row head and the loss reaches arm B only
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
import scorer1_layer as SL  # noqa: E402
import scorer_finetune as S  # noqa: E402
import test_aux_s1_wiring as AUXT  # noqa: E402
import test_rl_epo as TR  # noqa: E402

SRC = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n")
VOLATILE = AUXT.VOLATILE
M_PROP, T_H = 8, 20


class P6Bank(TR.StubBank):
    """the stub bank with trajectories and a get(): 8 proposals of known speed, 6 label columns (NC DAC binary; EP / TTC / C / DDC graded; EP 0 for violators), no lane column"""

    def __init__(self, path, n_prop, horizon, lane_labels=None, pdm_labels=None, teacher_lane_labels=None):
        super().__init__(path, n_prop, horizon, lane_labels=lane_labels, pdm_labels=pdm_labels, teacher_lane_labels=teacher_lane_labels)
        g = np.random.default_rng(3 if "train" in path else 4)
        for k, e in list(self.by.items()):
            speed = g.random(n_prop).astype(np.float32) * 14.0
            tr = np.zeros((n_prop, T_H, 3), np.float32)
            tr[:, :, 0] = speed[:, None] * 0.2 * np.arange(1, T_H + 1)[None, :]
            tg = e[2].copy()
            tg[:, 2] = np.where((tg[:, 0] < 0.5) | (tg[:, 1] < 0.5), 0.0, tg[:, 2])          # the EP label is 0 on a proposal with NC or DAC at 0 (card section 0)
            self.by[k] = (e[0], tr, tg, e[3], e[4])

    def get(self, log_name, token, step, rank=0):
        e = self.by.get((log_name, token, int(step), int(rank)))
        return None if e is None else (e[1], e[2], e[0])


class P6Targets(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        tr, ho = P6Bank("train", M_PROP, T_H), P6Bank("held", M_PROP, T_H)
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


class P6Model(AUXT.Stub256REFe):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg


S1E_ARGS = ["--pair-heads", "NC,DAC,EP,TTC", "--pair-dv", "0.15", "--pair-nobalance-heads", "EP", "--pair-nospeed-heads", "EP", "--pair-gap-heads", "EP=0.05", "--pair-mean", "head", "--pair-w", "0.3"]


def run(td, tag, extra=(), env=None, meta=None, modes=("pre",), src=SRC, max_updates=6):
    """the REAL main() of `src` on CPU against the stub trunk -> {rc_<mode>, rows, tensors}"""
    import ckpt_io as real_ckpt_io
    csrc = (src.replace("pin_memory=True", "pin_memory=False").replace("num_workers=2", "num_workers=0").replace("n=10000, seed=20260927", "n=40, seed=20260927"))
    csrc = csrc.replace('dev = "cuda"', 'dev = "cpu"').replace('torch.autocast("cuda", dtype=torch.bfloat16)', 'torch.autocast("cpu", enabled=False)')
    sm = TR.load(csrc, f"p6pre_{tag}")

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
    sm.REFe, sm.REFeConfig, sm.ckpt_io = P6Model, TR.StubCfgFactory, StubCkptIO
    sm.T = types.SimpleNamespace(OnPolicyBank=P6Bank, TargetBank=P6Targets)
    torch.manual_seed(0)
    ck = os.path.join(td, f"ck_{tag}.pt")
    torch.save({"format": "full", "model": P6Model(types.SimpleNamespace(n_score_components=6)).state_dict(), "meta": dict(meta or {})}, ck)
    assert torch.load(ck, weights_only=False)["model"]["score_head.weight"].shape[0] == 6, "the stub head must be 6 rows"
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
                res["rc_" + mode] = f"EXC {type(e).__name__}: {str(e)[:200]}"
            res["out_" + mode] = buf.getvalue()
    finally:
        sys.argv = saved_argv
        torch.cuda.synchronize, torch.cuda.max_memory_allocated = saved_cuda
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return AUXT.harvest(res, out)


def pre_rows(res):
    rows = res["rows"]["preflight.jsonl"]
    return next(r for r in rows if r.get("event") == "data"), [r for r in rows if r.get("event") == "preflight"][-1]


def norm(rows):
    return [{k: v for k, v in r.items() if k not in VOLATILE} for r in rows]


def same_tensors(a, b):
    if a is None or b is None:
        return a is b
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k])) if isinstance(a, dict) else True


with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
    # ---- A the OFF path
    peeled = SL.peel(SRC)
    assert peeled != SRC and not SL.has_layer(peeled)
    base = run(td, "peeled", src=peeled, modes=("pre", "run"))
    new = run(td, "new", modes=("pre", "run"))
    zero = run(td, "zero", extra=["--pair-w", "0"], modes=("pre", "run"))
    for name, r in (("new", new), ("--pair-w 0", zero)):
        assert base["rc_pre"] == r["rc_pre"] == 0 and base["rc_run"] == r["rc_run"] == 0, (base.get("rc_pre"), base.get("rc_run"), r.get("rc_pre"), r.get("rc_run"), r.get("out_pre", "")[-300:])
        for fn in ("preflight.jsonl", "sft.jsonl"):
            a_, b_ = norm(base["rows"][fn]), norm(r["rows"][fn])
            a_ = [{k: v for k, v in x.items() if k != "out"} for x in a_]
            b_ = [{k: v for k, v in x.items() if k != "out"} for x in b_]
            assert len(a_) == len(b_) and a_ == b_, (name, fn)
        for arm in ("A", "B"):
            assert base[arm] is not None and r[arm] is not None
            sa, sb = base[arm]["model"] if "model" in base[arm] else base[arm], r[arm]["model"] if "model" in r[arm] else r[arm]
            assert all(torch.equal(sa[k], sb[k]) for k in sa), (name, arm, "a saved tensor moved")
    assert new["rows"]["preflight.jsonl"][0].get("nav_mode") is None and "pair" not in pre_rows(new)[0], "the OFF data event carries neither pair nor nav_mode"
    print("A ok (OFF path and --pair-w 0 bit-identical to the pre-SCORER1 trainer on a 6-row head: rows and saved tensors)")

    # ---- B S1E preflight
    b = run(td, "s1e", extra=S1E_ARGS, meta={"nav_mode": "command"})
    assert b["rc_pre"] == 0 and b["pass_json"] is not None, (b["rc_pre"], b.get("out_pre", "")[-600:])
    data, pre = pre_rows(b)
    g16 = next(v for k, v in pre["gates"].items() if k.startswith("G16"))
    assert g16["ok"] is True and g16["fs_balanced"] is True and g16["controls_exactly_zero"] is True, g16
    assert g16["rules_built"]["EP"] == {"balance": False, "speed_contrast": False, "min_gap": 0.05} and all(g16["rules_built"][h] == {"balance": True, "speed_contrast": True, "min_gap": 0.0} for h in ("NC", "DAC", "TTC"))
    assert g16["reduce_built"] == "head" and all(g16["bank_share"][h]["share_with_pair"] > 0.0 for h in ("NC", "DAC", "EP", "TTC")), g16["bank_share"]
    assert all(g16["bank_share"][h]["kept_F"] == g16["bank_share"][h]["kept_S"] for h in ("NC", "DAC", "TTC"))
    assert data["pair"]["rules"]["EP"]["balance"] is False and data["pair"]["reduce"] == "head" and data["nav_mode"] == "command", data["pair"]
    ns, err = C.parse_scorer_argv(S.build_parser(), b["argv"])
    assert not err, err
    rows = C.scorer_dvb_rows(ns, data, {"name": "stub"})
    pair_rows = [r for r in rows if r["dest"] in ("pair_nobalance_heads", "pair_nospeed_heads", "pair_gap_heads", "pair_mean")]
    assert len(pair_rows) == 3 * 4 + 1 and all(r["ok"] for r in pair_rows), [r for r in pair_rows if not r["ok"]]
    assert all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    bad = json.loads(json.dumps(data))
    bad["pair"]["rules"]["EP"]["balance"] = True
    assert any(not r["ok"] and r["dest"] == "pair_nobalance_heads" for r in C.scorer_dvb_rows(ns, bad, {"name": "stub"})), "a balanced EP in the built block must turn its row RED"
    bad = json.loads(json.dumps(data))
    bad["pair"]["reduce"] = "pooled"
    assert any(not r["ok"] and r["dest"] == "pair_mean" for r in C.scorer_dvb_rows(ns, bad, {"name": "stub"}))
    jg = C.judge_g16(g16, ns)
    assert jg == [], jg
    print("B ok (6-output S1E preflight: G16 ok, per-head rules built, shares > 0, G-DVB pair + nav_mode rows green; a flipped rule is red)")

    # ---- C the regressions
    for tag, env, frag in (("nobalance", {"REFE_PAIR_MUTATE_NOBALANCE": "1"}, "F/S ratio"), ("epbalance", {"REFE_PAIR_MUTATE_EPBALANCE": "1"}, "head EP was built with rule"),
                           ("droploss", {"REFE_PAIR_MUTATE_DROP_LOSS": "1"}, "not live IN the loss")):
        m = run(td, "mut_" + tag, extra=S1E_ARGS, env=env, meta={"nav_mode": "command"})
        assert m["pass_json"] is None and m["rc_pre"] != 0, (tag, m["rc_pre"])
        assert any(r.get("event") == "preflight" for r in m["rows"]["preflight.jsonl"]), (tag, m["rc_pre"], m["out_pre"][-500:])
        _d, pre_m = pre_rows(m)
        gm = next(v for k, v in pre_m["gates"].items() if k.startswith("G16"))
        assert gm["ok"] is False, (tag, gm)
        reasons = C.judge_g16(gm, ns)
        assert reasons and any(frag in r for r in reasons), (tag, reasons)
    print("C ok (NOBALANCE, EPBALANCE and DROP_LOSS: no PREFLIGHT_PASS, G16 ok false, judge_g16 RED with the expected reason)")

    # ---- D calibrate
    d = run(td, "cal", extra=[t for t in S1E_ARGS[:-2]] + ["--pair-calibrate"], meta={"nav_mode": "command"}, modes=("run",))
    pw = os.path.join(d["out"], "pair_weight.json")
    assert d["rc_run"] == 0 and "ZZPAIR_CALIBRATED" in d["out_run"] and os.path.exists(pw), (d["rc_run"], d.get("out_run", "")[-400:])
    j = json.load(open(pw))
    assert j["micro_batches"] == 50 and j["pair_w"] and j["pair_w"] > 0 and all(v > 0 for v in j["sets_with_pair_share"].values()) and set(j["sets_with_pair_share"]) == {"NC", "DAC", "EP", "TTC"}, j
    print(f"D ok (calibration on 6 outputs: lambda_p {j['pair_w']:.4g} over 50 micro-batches)")

    # ---- E nav_mode
    e = run(td, "nonav", extra=S1E_ARGS, meta={})
    d_e, _ = pre_rows(e)
    assert "nav_mode" not in d_e, "a checkpoint that lost nav_mode builds goal mode and must write no command fact"
    rows_e = C.scorer_expect_rows({"data_expect": {"nav_mode": "command"}}, ns, d_e)
    assert [r["ok"] for r in rows_e] == [False]
    print("E ok")

    # ---- F the training loop on the 6-row head
    f = run(td, "train", extra=S1E_ARGS, meta={"nav_mode": "command"}, modes=("pre", "run"))
    assert f["rc_pre"] == 0 and f["rc_run"] == 0, (f["rc_pre"], f["rc_run"], f.get("out_run", "")[-400:])
    sd = f["B"]["model"] if "model" in f["B"] else f["B"]
    assert sd["score_head.weight"].shape[0] == 6 and not np.allclose(sd["score_head.weight"].numpy(), f["A"]["model"]["score_head.weight"].numpy() if "model" in f["A"] else f["A"]["score_head.weight"].numpy()), "arm B must differ from arm A"
    print("F ok")
print("ZZTEST_PASS")
