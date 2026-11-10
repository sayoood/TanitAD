#!/usr/bin/env python3
"""S1ZSM CPU preflight of the REAL scorer_finetune.main() on a tiny 6-OUTPUT stub model whose bank holds near-twin pairs WITH speed variety (PREREG_S1ZSM_P6 section 2).  Prints ZZTEST_PASS.

  The bank (every set): p0 clean 8.3 m/s, p1 clean 7.7, p5 clean 8.2 (all within 1 m of the three hard negatives), p2 / p3 / p4 the NC / DAC / TTC hard negatives at 8.0 m/s, p6 a clean proposal that sits on
  the hard negatives for 19 steps and jumps to 10 m/s at step 19 (INSIDE the 1 m radius, OUTSIDE the 0.15 speed band), p7 a far clean filler.  v0 = ego slot 6 = 8 m/s.  So per head: F candidates {p0, p5} (+ p6
  out of band), S candidate {p1}: the speed-neutral loss keeps exactly 1 F + 1 S = 2 pairs per head and set.

  A OFF PATH: (1) no --twin-* flag, and (2) the S1Z argv (--twin-* named, NO --twin-speed-neutral), run bit-identical (rows and every saved tensor) to the TIP trainer (s1zsm_layer.peel of this file = git blob
    6a16d35c..., commit 0c19b8fd); neither writes an S1ZSM key anywhere
  B S1ZSM preflight: PREFLIGHT_PASS written, G18 ok incl. the extended speed-neutral fields (probe batch AND live micro-batches: out_of_band 0, kept F == kept S, every head keeps pairs), the data event's `twin`
    block carries speed_neutral true / sn_dv 0.15 / sn_balance true / sn_seed 20261107, every G-DVB row is green (12 twin rows), judge_g18 passes; the weight-0 assembly has bit-identical gradients to the stock loss
  C DELIBERATE REGRESSIONS (the five of the chain) withhold PREFLIGHT_PASS, read G18 ok false, turn judge_g18 RED, and (the built-fact ones) the G-DVB rows RED: the speed band removed, the balancing removed, L_twin
    dropped from the loss (by env AND by deleting the line), S1Z's radius removed and top-K mining removed
  D --pair-calibrate --twin-heads --twin-speed-neutral writes a finite positive lambda_z that SAYS speed_neutral true (+ sn_dv, sn_seed, sn_balance); the same run without the flag does not (S1Z's file is recognisable)
  E inconsistent argvs never start (rc 4): --twin-speed-neutral without --twin-heads
  F the training loop: 25 updates; the per-head share and the speed-neutral counts are in the trainer's log and in the done event (kept F == kept S, out of band 0); arm A is bit-equal to the OFF run's arm A,
    arm B moved, and arm B differs from the S1Z (flag off) arm B: the flag reaches the loss
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
import s1zsm_layer as ZSM  # noqa: E402
import scorer_finetune as S  # noqa: E402
import test_aux_s1_wiring as AUXT  # noqa: E402
import test_rl_epo as TR  # noqa: E402

SRC = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n")
VOLATILE = AUXT.VOLATILE
M_PROP, T_H = 8, 20
TWIN_ARGS = ["--twin-heads", "NC,DAC,TTC", "--twin-ade", "1.0", "--twin-topk", "8"]
S1Z_ARGV = TWIN_ARGS + ["--twin-w", "0.3"]                                   # S1Z: the flag left OUT
SN = TWIN_ARGS + ["--twin-speed-neutral"]
TW = SN + ["--twin-w", "0.3"]
HEADS = ("NC", "DAC", "TTC")
NOISE = {"sigma": 0.0, "gen": None}      # seeded additive noise on the score logits of EVERY forward pass (unused here: sigma 0)
SPEEDS = [8.3, 7.7, 8.0, 8.0, 8.0, 8.2, 8.0, 8.0]
LAT = [0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.15, 30.0]


class SNBank(TR.StubBank):
    """8 proposals, identical in every set (see the module doc)"""

    def __init__(self, path, n_prop, horizon, lane_labels=None, pdm_labels=None, teacher_lane_labels=None):
        super().__init__(path, n_prop, horizon, lane_labels=lane_labels, pdm_labels=pdm_labels, teacher_lane_labels=teacher_lane_labels)
        for n, k in enumerate(list(self.by)):
            e = self.by[k]
            tr = np.zeros((n_prop, T_H, 3), np.float32)
            tg = np.zeros((n_prop, 6), np.float32)
            for idx in range(n_prop):
                tr[idx, :, 0] = SPEEDS[idx] * 0.2 * np.arange(1, T_H + 1)
                tr[idx, :, 1] = LAT[idx]
                tg[idx] = [1, 1, 0.8, 1, 1, 1]
            tr[6, 19, 0] = tr[6, 18, 0] + 10.0 * 0.2                      # the jump: chord speed 10 m/s at step 19 (0.4 m further), ADE to the hard negatives < 0.1 m
            tg[2, [0, 2]] = 0.0                                          # p2 fails NC
            tg[3, [1, 2]] = 0.0                                          # p3 fails DAC
            tg[4, 3] = 0.0                                               # p4 fails TTC
            tg[4, 2] = 0.5
            self.by[k] = (e[0], tr, tg, e[3], e[4])

    def get(self, log_name, token, step, rank=0):
        e = self.by.get((log_name, token, int(step), int(rank)))
        return None if e is None else (e[1], e[2], e[0])


class TwinTargets(torch.utils.data.Dataset):
    def __init__(self, targets, images, cfg, calib=None, onpolicy=None):
        self.onpolicy = onpolicy
        tr, ho = SNBank("train", M_PROP, T_H), SNBank("held", M_PROP, T_H)
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
    sm = TR.load(csrc, f"s1zsmpre_{tag}")
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
    sm.T = types.SimpleNamespace(OnPolicyBank=SNBank, TargetBank=TwinTargets)
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


SN_KEYS = {"speed_neutral", "sn_dv", "sn_balance", "sn_seed", "sn_band_filter", "sn_v_idx", "sn_v0"}


def has_s1zsm_key(obj):
    """True if any dict key at any depth is an S1ZSM key"""
    if isinstance(obj, dict):
        return any(k in SN_KEYS or k in S.TWIN_SN_KEYS or k.startswith("sn_") or k.startswith("mutate_no") and "band" in k for k in obj) or any(has_s1zsm_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_s1zsm_key(v) for v in obj)
    return False


with tempfile.TemporaryDirectory(dir=os.environ.get("SCORER1_TMP") or None) as td:
    # ---- A the OFF path against the TIP trainer
    tip = ZSM.peel(SRC)
    assert tip != SRC and not ZSM.has_layer(tip)
    for tag, extra in (("off", []), ("s1z", S1Z_ARGV)):
        base = run(td, "tip_" + tag, extra=extra, meta={"nav_mode": "command"}, src=tip, modes=("pre", "run"))
        new = run(td, "new_" + tag, extra=extra, meta={"nav_mode": "command"}, modes=("pre", "run"))
        assert base["rc_pre"] == new["rc_pre"] == 0 and base["rc_run"] == new["rc_run"] == 0, (tag, base.get("rc_pre"), base.get("rc_run"), new.get("rc_pre"), new.get("rc_run"), new.get("out_pre", "")[-300:])
        for fn in ("preflight.jsonl", "sft.jsonl"):
            a_, b_ = norm(base["rows"][fn]), norm(new["rows"][fn])
            assert len(a_) == len(b_) and a_ == b_, (tag, fn)
        for arm in ("A", "B"):
            sa, sb = sd_of(base[arm]), sd_of(new[arm])
            assert all(torch.equal(sa[k], sb[k]) for k in sa), (tag, arm, "a saved tensor moved")
        assert not has_s1zsm_key(new["rows"]["preflight.jsonl"]) and not has_s1zsm_key(new["rows"]["sft.jsonl"]), (tag, "an S1ZSM key leaked into an OFF run")
        if tag == "s1z":
            d_s, p_s = pre_rows(new)
            assert d_s["twin"]["w"] == 0.3 and g18_of(p_s)["ok"] is True, g18_of(p_s)
            assert all(g18_of(p_s)["share_micro"][h]["pairs"] > 0 for h in HEADS)
    print("A ok (no --twin-* flag, and the S1Z argv without --twin-speed-neutral: rows and saved tensors bit-identical to the tip trainer; no S1ZSM key in any row)")

    # ---- B S1ZSM preflight
    b = run(td, "sn", extra=TW, meta={"nav_mode": "command"}, inject=True)
    assert b["rc_pre"] == 0 and b["pass_json"] is not None, (b["rc_pre"], b.get("out_pre", "")[-600:])
    data, pre = pre_rows(b)
    g18 = g18_of(pre)
    assert g18["ok"] is True and g18["controls_exactly_zero"] is True and all(v == 0.0 for v in g18["control_w0_term_alone_grad_abs"].values()), g18
    assert all(v > 0.0 for v in g18["loss_grad_delta_w1_minus_w0"].values()) and set(g18["loss_grad_delta_w1_minus_w0"]) == set(C.SCORER_GROUPS)
    assert g18["speed_neutral"] is True and g18["sn_ok"] is True and g18["sn_probe_batch_ok"] is True and g18["sn_live_ok"] is True and g18["sn_dv_built"] == 0.15
    assert g18["sn_band_filter_built"] is True and g18["sn_balance_built"] is True and g18["sn_seed_built"] == 20261107 and g18["mutate_noband"] is False and g18["mutate_nosnbalance"] is False
    # MM fix 1: with --twin-speed-neutral the live window is TWIN_SN_PREFLIGHT_MICRO (128) micro-batches, not 16.  The fixture loader holds 50 micro-batches of 2 sets (100 sets),
    # more than 16 and fewer than 128, so the window reads the whole loader: 50 here proves the speed-neutral window is in force (S1Z's own preflight test still pins 16 = 32 sets).
    assert S.TWIN_SN_PREFLIGHT_MICRO == 128 and S.TWIN_PREFLIGHT_MICRO == 16 and g18["micro_batches"] == 50, (g18["micro_batches"], S.TWIN_SN_PREFLIGHT_MICRO)
    for h in HEADS:
        m_, pbh = g18["share_micro"][h], g18["micro_batch_pairs"][h]
        assert m_["share_with_pair"] == 1.0 and m_["sets"] == 100 and m_["pairs"] == 200 and m_["kept_F"] == m_["kept_S"] == 100 and m_["band_pairs"] == 300, (h, m_)
        assert m_["out_of_band"] == 0 and m_["unbalanced_sets"] == 0, (h, m_)
        assert pbh["out_of_band"] == 0 and pbh["unbalanced_sets"] == 0 and pbh["kept_F"] == pbh["kept_S"] > 0 and pbh["pairs"] == 2 * pbh["kept_F"], (h, pbh)
        assert g18["bank_share"][h]["share_with_pair"] == 1.0 and "band_pairs" not in g18["bank_share"][h], "the bank share is the labels-only upper bound (no speeds)"
    want_blk = {"w": 0.3, "heads": list(HEADS), "ade": 1.0, "topk": 8, "radius_filter": True, "mining": True, "calibrate": False, "ade_metric": "mean_xy_euclid_over_raw_poses", "clean": "NC=1,DAC=1,TTC=1",
                "speed_neutral": True, "sn_dv": 0.15, "sn_balance": True, "sn_seed": 20261107, "sn_band_filter": True, "sn_v_idx": 19, "sn_v0": "ego_slot_6"}
    assert data["twin"] == want_blk, data["twin"]
    ns, err = C.parse_scorer_argv(S.build_parser(), b["argv"])
    assert not err and ns.twin_speed_neutral is True, err
    assert "G18" in C.scorer_required_gates(ns)
    rows = C.scorer_dvb_rows(ns, data, {"name": "stub"})
    twin_rows = [r for r in rows if r["dest"].startswith("twin")]
    assert len(twin_rows) == 12 and all(r["ok"] for r in rows), [r for r in rows if not r["ok"]]
    assert sum(1 for r in twin_rows if r["dest"] == "twin_speed_neutral") == 5
    assert C.judge_g18(g18, ns) == [], C.judge_g18(g18, ns)
    assert b["ident"] == [True], ("weight 0 must give bit-identical gradients to the stock loss", b["ident"])
    # the flag reaches the loss object: the S1Z run (flag left out) on the same bank keeps the out-of-band-and-unbalanced pairs S1ZSM drops
    z = run(td, "s1z_pre", extra=S1Z_ARGV, meta={"nav_mode": "command"})
    gz = g18_of(pre_rows(z)[1])
    assert all(gz["share_micro"][h]["pairs"] > 64 for h in HEADS), [gz["share_micro"][h]["pairs"] for h in HEADS]
    dz, _pz = pre_rows(z)
    rows_z = C.scorer_dvb_rows(C.parse_scorer_argv(S.build_parser(), z["argv"])[0], dz, {"name": "stub"})
    assert len([r for r in rows_z if r["dest"].startswith("twin")]) == 7 and all(r["ok"] for r in rows_z), "an S1Z run keeps the S1Z G-DVB row count"
    # a built-ON block under an argv that leaves the flag out is RED (the default-off flag left out of the argv, seen from the data event)
    ns_off, _e = C.parse_scorer_argv(S.build_parser(), z["argv"])
    rows_x = C.scorer_dvb_rows(ns_off, data, {"name": "stub"})
    assert any(not r["ok"] and r["dest"] == "twin_speed_neutral" for r in rows_x), "speed-neutral built, flag not in the argv: G-DVB RED"
    assert C.judge_g18(g18, ns_off), "judge_g18: S1ZSM fields in the record but not in the argv: RED"
    print("B ok (G18 ok incl. the speed-neutral fields on the probe batch and over 16 live micro-batches; `twin` block == card; 12 G-DVB twin rows green; weight 0 -> bit-identical gradients; the S1Z argv on the same bank keeps more pairs)")

    # ---- C deliberate regressions (the five of the chain)
    for tag, env, ms, frag, dvb in (("noband", {"REFE_TWIN_MUTATE_NOBAND": "1"}, None, "speed band", "twin_speed_neutral"),
                                    ("nosnbalance", {"REFE_TWIN_MUTATE_NOSNBALANCE": "1"}, None, "F/S balanc", "twin_speed_neutral"),
                                    ("droploss", {"REFE_TWIN_MUTATE_DROP_LOSS": "1"}, None, "not live IN the loss", None),
                                    ("srcdrop", None, (DROP_LINE, "            pass\n"), "not live IN the loss", None),
                                    ("noradius", {"REFE_TWIN_MUTATE_NORADIUS": "1"}, None, "radius filter is missing", "twin_ade"),
                                    ("notopk", {"REFE_TWIN_MUTATE_NOTOPK": "1"}, None, "top-K mining is missing", "twin_topk")):
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
        if tag == "noband":
            assert gm["sn_ok"] is False and gm["mutate_noband"] is True and d_m["twin"]["sn_dv"] == float(S.TWIN_UNBOUNDED) and d_m["twin"]["sn_band_filter"] is False, (gm, d_m["twin"])
            assert gm["share_micro"]["NC"]["out_of_band"] > 0
        if tag == "nosnbalance":
            assert gm["sn_ok"] is False and gm["mutate_nosnbalance"] is True and d_m["twin"]["sn_balance"] is False, (gm, d_m["twin"])
            assert gm["micro_batch_pairs"]["NC"]["unbalanced_sets"] > 0 and gm["micro_batch_pairs"]["NC"]["kept_F"] != gm["micro_batch_pairs"]["NC"]["kept_S"]
    print("C ok (band removed, balancing removed, L_twin dropped (env + deleted line), radius removed, top-K removed: no PREFLIGHT_PASS, G18 ok false, judge_g18 RED, the built-fact G-DVB rows RED)")

    # ---- D calibrate
    d = run(td, "cal", extra=SN + ["--pair-calibrate"], meta={"nav_mode": "command"}, modes=("run",))
    pw = os.path.join(d["out"], "pair_weight.json")
    assert d["rc_run"] == 0 and "ZZPAIR_CALIBRATED" in d["out_run"] and os.path.exists(pw), (d["rc_run"], d.get("out_run", "")[-400:])
    j = json.load(open(pw))
    assert j["micro_batches"] == 50 and j["twin_w"] and j["twin_w"] > 0 and "pair_w" not in j and j["lever"] == "twin" and j["ade"] == 1.0 and j["topk"] == 8, j
    assert j["speed_neutral"] is True and j["sn_dv"] == 0.15 and j["sn_balance"] is True and j["sn_seed"] == 20261107, j
    assert set(j["sets_with_pair_share"]) == set(HEADS) and all(v == 1.0 for v in j["sets_with_pair_share"].values()) and sorted(j["heads"]) == sorted(HEADS), j
    cal_data = next(r for r in d["rows"]["sft.jsonl"] if r.get("event") == "data")
    assert cal_data["twin"]["calibrate"] is True and cal_data["twin"]["w"] == 0.0 and cal_data["twin"]["speed_neutral"] is True
    d0 = run(td, "cal0", extra=TWIN_ARGS + ["--pair-calibrate"], meta={"nav_mode": "command"}, modes=("run",))
    j0 = json.load(open(os.path.join(d0["out"], "pair_weight.json")))
    assert "speed_neutral" not in j0 and j0["twin_w"] != j["twin_w"], ("S1Z's calibration says nothing about speed_neutral and measures another weight", j0, j)
    print(f"D ok (lambda_z {j['twin_w']:.4g} speed-neutral vs {j0['twin_w']:.4g} S1Z-style on the same bank; only the first says speed_neutral true)")

    # ---- E inconsistent argvs never start
    for tag, extra in (("e1", ["--twin-speed-neutral"]), ("e2", ["--twin-speed-neutral", "--pair-calibrate"]), ("e3", SN + ["--twin-w", "0.3", "--pair-heads", "NC", "--pair-w", "0.2"])):
        e = run(td, tag, extra=extra, modes=("pre",))
        assert e["rc_pre"] == 4 and "REFUSED" in e["out_pre"], (tag, e["rc_pre"], e["out_pre"][-300:])
    print("E ok")

    # ---- F the training loop
    base25 = run(td, "off25", meta={"nav_mode": "command"}, modes=("pre", "run"), max_updates=25)
    z25 = run(td, "s1z25", extra=S1Z_ARGV, meta={"nav_mode": "command"}, modes=("pre", "run"), max_updates=25)
    f = run(td, "sn25", extra=TW, meta={"nav_mode": "command"}, modes=("pre", "run"), max_updates=25)
    assert f["rc_pre"] == 0 and f["rc_run"] == 0 and base25["rc_run"] == 0 and z25["rc_run"] == 0, (f["rc_pre"], f["rc_run"], f.get("out_run", "")[-400:])
    sa0, sa1 = sd_of(base25["A"]), sd_of(f["A"])
    sb0, sb1, sbz = sd_of(base25["B"]), sd_of(f["B"]), sd_of(z25["B"])
    assert all(torch.equal(sa0[k], sa1[k]) for k in sa0), "arm A is the BCE-only control: bit-equal to the OFF run's arm A"
    assert any(not torch.equal(sb0[k], sb1[k]) for k in sb0), "arm B must move with L_twin"
    assert any(not torch.equal(sbz[k], sb1[k]) for k in sbz), "the speed-neutral arm B differs from the S1Z arm B: the flag reaches the loss"
    srows = f["rows"]["sft.jsonl"]
    trow = next(r for r in srows if r.get("event") == "train")
    assert set(trow["twin"]) == set(HEADS), trow["twin"]
    for h, v in trow["twin"].items():
        assert v["share_with_pair"] == 1.0 and v["sets"] == 100 and v["pairs"] == 200 and v["kept_F"] == v["kept_S"] == 100 and v["out_of_band"] == 0 and v["unbalanced_sets"] == 0 and v["band_pairs"] == 300, (h, v)
    done = next(r for r in srows if r.get("event") == "done")
    assert set(done["twin_share"]) == set(HEADS), done["twin_share"]
    for h, v in done["twin_share"].items():
        assert v["share_with_pair"] == 1.0 and v["kept_F"] == v["kept_S"] > 0 and v["out_of_band"] == 0 and v["unbalanced_sets"] == 0 and v["pairs"] == 2 * v["kept_F"], (h, v)
    zrow = next(r for r in z25["rows"]["sft.jsonl"] if r.get("event") == "train")
    assert all("kept_F" not in v for v in zrow["twin"].values()), "the S1Z (flag off) run logs no speed-neutral counts"
    print("F ok (25 updates: arm A bit-equal to the OFF run, arm B moved and differs from the S1Z arm B; per-head shares + F/S counts in the train row and the done event)")
print("ZZTEST_PASS")
