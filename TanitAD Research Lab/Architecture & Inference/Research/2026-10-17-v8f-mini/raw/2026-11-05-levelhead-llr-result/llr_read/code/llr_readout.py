#!/usr/bin/env python3
"""llr_readout.py -- O's sensitivity to the TOKEN wiring of arm LLR (A2 card, Reads: A1's measurement adapted to ``--aux-cond token``), zero training, CPU only.

LLR's O reads ONE extra conditioning token e_aux([v_hat / 10, sigmoid(presence logit), gap_hat]) (detached, from the current tick frame).  Per LLR checkpoint (seed 0 / 1), over the 2,399 eval-141 windows at T0 = the tick-0
batch the T1-kin rollout feeds (logged history and frames), a forward PRE-hook on ``model.e_aux`` changes ONE input column, everything else fixed, and the change of the plan's mean speed over 0..3 s is measured
(``traj60`` rows 0..29, step length / 0.1 s, origin first; the statistic of clr_readout.py):
  LEVEL   column 0 (v_hat / 10) +0.3 / -0.3  = v_hat +- 3 m/s            (A1's committed labels: O-SENSITIVE >= 0.5 m/s |mean signed change| for +3, O-INSENSITIVE < 0.1, else WEAK)
  LEAD    column 1 (presence) set to 0.0 / 1.0, the gap column as the model's own  (the card names no threshold for it: A1's are applied BY ANALOGY to the 0 <-> 1 flip, stated as such)
  BESIDE  column 2 (gap_hat) 0.25 vs 0.75 at presence 1 (10 m vs 30 m); the e_aux OUTPUT zeroed (attribution); ||e_aux(x)|| / ||O's ego token||
Estimator: per-window values, episode-cluster bootstrap B = 2000, seed 0 (clr_readout.py's draws, pinned by sha256).  Controls run in-process and a failed control exits non-zero: a zero shift reads EXACTLY 0.0;
the e_aux input equals [v_hat / 10, sigmoid(logit), sigmoid(gap logit)] where the frame is real and 0 where it is not; +0.3 reaches column 0 and no other; the presence set reaches column 1 exactly; a norm-8
perturbation of e_aux's OUTPUT moves the plan (reach); the baseline tick-0 plan equals the banked LLR rollout's tick 0 (when present).
usage: python -I llr_readout.py --seed S [--max-windows N] [--threads 12] [--batch 32] [--ckpt PATH] [--out PATH]      |      python -I llr_readout.py --merge [--dir readout/]"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
CRO = f"{ART}/dagland/motion_c/clr_read/code/clr_readout.py"
CRO_SHA = "5526ad97a210481dbd3b83f5b44d6d90c2f1a8088ffa148b06ca533874c1bb1f"       # clr_readout.py (A1's instrument): its pure helpers are reused, an edit of it is a refusal
TREE = "F:/TanitAD-scratch/motion_c/llr_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TRACKS = f"{ART}/sfresh_inputs/inputs/native_tracks.pt"
EVAL_CACHE = f"{ART}/sfresh_inputs/cache"
CKPT = f"{ART}/v8f-recover/M2b_s{{s}}_recft_LLR/ckpt_last.pt"
BANKED_TICKS = "F:/TanitAD-scratch/motion_c/llr_read/out/LLR_s{s}/ticks/ticks_LLR_s{s}.npz"
OUT_DIR = f"{PK}/readout"
SHIFT_MS, LEVEL_SCALE, B_BOOT = 3.0, 10.0, 2000
GAP_LO, GAP_HI = 0.25, 0.75


def _load_cro():
    import hashlib
    got = hashlib.sha256(open(CRO, "rb").read()).hexdigest()
    if got != CRO_SHA:
        raise SystemExit(f"REFUSED: {CRO} is {got}, not the pinned {CRO_SHA}")
    spec = importlib.util.spec_from_file_location("clr_readout_base", CRO)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CR = _load_cro()
draws, ep_sums, point_and_ci, mean_stat, plan_speed_0_3, sens_label, sha256 = CR.draws, CR.ep_sums, CR.point_and_ci, CR.mean_stat, CR.plan_speed_0_3, CR.sens_label, CR.sha256
SENS_HI, SENS_LO = CR.SENS_HI, CR.SENS_LO


def load_tool():
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_ri", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v8f_t1_eval_ri"] = mod
    spec.loader.exec_module(mod)
    return mod


def forward_all(model, batch, dev):
    """One window batch -> dict of numpy arrays: the baseline plan, the head outputs, the e_aux input / output, and the plans under each input change.  Hooks only; the model is untouched."""
    import torch
    cap = {"v_hat": None, "lead_o": None, "a_in": None, "a_out": None}
    st = {"lvl": 0.0, "pres": None, "gap": None, "zero_e": False, "add": None}

    def h_lvl(mod, inp, out):
        cap["v_hat"] = out.detach().float().squeeze(-1).clone()

    def h_lead(mod, inp, out):
        cap["lead_o"] = out.detach().float().clone()

    def pre_a(mod, args):
        x = args[0].clone()
        if st["lvl"] != 0.0:
            x[:, 0] = x[:, 0] + st["lvl"]
        if st["pres"] is not None:
            x[:, 1] = st["pres"]
        if st["gap"] is not None:
            x[:, 2] = st["gap"]
        return (x,)

    def h_a(mod, args, out):
        cap["a_in"], cap["a_out"] = args[0].detach().float().clone(), out.detach().float().clone()          # AFTER the pre-hook: what the layer saw
        if st["zero_e"]:
            return torch.zeros_like(out)
        if st["add"] is not None:
            return out + st["add"]

    hs = [model.lvl_head.register_forward_hook(h_lvl), model.lead_head.register_forward_hook(h_lead), model.e_aux.register_forward_pre_hook(pre_a), model.e_aux.register_forward_hook(h_a)]
    res = {}

    def run(**kw):
        st.update({"lvl": 0.0, "pres": None, "gap": None, "zero_e": False, "add": None})
        st.update(kw)
        with torch.no_grad():
            out = model.forward_infer(batch)
        return out["traj60"].double().cpu().numpy(), out["level_v"].detach().float().cpu().numpy()
    try:
        t0, lv = run()
        res.update(traj60=t0, level_v=lv, v_hat=cap["v_hat"].cpu().numpy(), lead_o=cap["lead_o"].cpu().numpy(), a_in=cap["a_in"].cpu().numpy(), a_out=cap["a_out"].cpu().numpy())
        for name, kw in (("lp", dict(lvl=+SHIFT_MS / LEVEL_SCALE)), ("lm", dict(lvl=-SHIFT_MS / LEVEL_SCALE)), ("z", dict(lvl=0.0)), ("p0", dict(pres=0.0)), ("p1", dict(pres=1.0)),
                         ("glo", dict(pres=1.0, gap=GAP_LO)), ("ghi", dict(pres=1.0, gap=GAP_HI)), ("ez", dict(zero_e=True))):
            t, _ = run(**kw)
            res["traj60_" + name], res["a_in_" + name] = t, cap["a_in"].cpu().numpy()
        g = torch.Generator().manual_seed(7)                                  # REACH CONTROL (instrument only): a fixed random direction of norm 8 added to e_aux's OUTPUT must move the plan
        vec = torch.randn(1, model.e_aux.out_features, generator=g)
        res["traj60_big"] = run(add=(vec / vec.norm() * 8.0).to(dev))[0]
        Fn = batch["feats"].shape[1]
        last = torch.full((batch["feats"].shape[0],), Fn - 1, dtype=torch.long, device=dev)
        with torch.no_grad():
            res["ego_tok"] = model._ego(model._at(batch["p10f"], last)).float().cpu().numpy()
        res["frame_ok_now"] = batch["frame_ok"][:, -1].cpu().numpy().astype(bool)
    finally:
        for h in hs:
            h.remove()
    return res


VARIANTS = ("traj60", "traj60_lp", "traj60_lm", "traj60_z", "traj60_p0", "traj60_p1", "traj60_glo", "traj60_ghi", "traj60_ez", "traj60_big")
COLS = VARIANTS + ("v_hat", "level_v", "lead_o", "a_in", "a_in_lp", "a_in_lm", "a_in_z", "a_in_p0", "a_in_p1", "a_in_glo", "a_in_ghi", "a_out", "ego_tok", "frame_ok_now")


def run_seed(s, max_windows, threads, batch_size, ckpt=None):
    import torch
    torch.set_num_threads(threads)
    T = load_tool()
    ck_path = ckpt or CKPT.format(s=s)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = T.M.V8FMiniConfig(**ck["cfg"])
    assert cfg.level_head == "on" and cfg.lead_head == "on" and cfg.aux_cond == "token", "checkpoint cfg: level_head / lead_head / aux_cond is not on / on / token"
    model = T.M.V8FMini(cfg, T.M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * T.M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device("cpu")
    model.to(dev).eval()
    tracks = T.D.load_native_tracks(TRACKS)
    J = T.D.load_join(f"{DATA}/join_eval141.npz")
    store = T.D.ClipStore(EVAL_CACHE, "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins_all = T.D.eval_grid_windows(f"{DATA}/eval_grid_windows.json", J)
    sel = list(range(len(wins_all)))
    if max_windows and max_windows < len(wins_all):
        sel = sel[:max_windows]                                             # the FIRST N windows (the rollout's own batches)
    wins = [wins_all[i] for i in sel]
    proxy = T._OneClipStore()
    ds = T.D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = T.ages_for(0, "periodic", 3, 6)
    cols = {k: [] for k in COLS}
    ci_l, t0_l = [], []
    t_start = time.time()
    for b0 in range(0, len(wins), batch_size):
        chunk = wins[b0:b0 + batch_size]
        states = T.make_states(store, tracks, J, chunk, "deadreckon")
        items = [T.build_item(ds, proxy, st, 0, a_t, a_s, "logged", "causal", None) for st in states]     # EXACTLY the rollout's tick-0 batch
        full = T.D.collate(items)
        batch = {k: full[k].to(dev) for k in T.MODEL_KEYS}
        r = forward_all(model, batch, dev)
        for k in COLS:
            cols[k].append(r[k])
        for (ci_, t0_, _, _) in chunk:
            ci_l.append(int(ci_))
            t0_l.append(int(t0_))
        print(f"[s{s}] {b0 + len(chunk)}/{len(wins)} {time.time() - t_start:.0f}s", flush=True)
    A = {k: np.concatenate(v) for k, v in cols.items()}
    A.update(ci=np.array(ci_l), t0=np.array(t0_l), sel=np.array(sel), ckpt_sha256=sha256(ck_path))
    sd = ck["model"]
    A["e_aux_w"], A["e_aux_b"] = sd["e_aux.weight"].float().numpy(), sd["e_aux.bias"].float().numpy()
    A["cfg"] = {"level_head": cfg.level_head, "w_level": cfg.w_level, "lead_head": cfg.lead_head, "w_lead": cfg.w_lead, "aux_cond": cfg.aux_cond}
    return A


def analyze(s, A, full_set):
    n = len(A["v_hat"])
    eid = A["ci"]
    uniq = np.unique(eid)
    M = draws(uniq, B_BOOT, 0)
    sp = {k: plan_speed_0_3(A["traj60" + ("_" + k if k else "")]) for k in ("", "lp", "lm", "z", "p0", "p1", "glo", "ghi", "ez", "big")}

    def stat(d):
        S = ep_sums(eid, uniq, np.ones(n), d, np.abs(d))
        return {"mean_signed_change_ms": point_and_ci(S[:, [0, 1]], M, mean_stat), "mean_abs_change_ms": point_and_ci(S[:, [0, 2]], M, mean_stat),
                "median_signed_change_ms": float(np.median(d)), "p95_abs_change_ms": float(np.percentile(np.abs(d), 95)), "max_abs_change_ms": float(np.abs(d).max())}
    out = {"seed": s, "n_windows": int(n), "n_episodes": int(len(uniq)), "full_eval_set": bool(full_set), "ckpt_sha256": A["ckpt_sha256"], "ckpt_cfg": A["cfg"], "n_frame_ok_false_now": int((~A["frame_ok_now"]).sum()),
           "statistic": "plan mean speed over 0..3 s from traj60 rows 0..29 (origin first), step length / 0.1 s; change = changed - baseline, per window; episode-cluster bootstrap B=2000 seed 0",
           "mean_plan_speed_baseline": float(sp[""].mean())}
    lev = {"plus3": stat(sp["lp"] - sp[""]), "minus3": stat(sp["lm"] - sp[""])}
    lev["label"] = sens_label(abs(lev["plus3"]["mean_signed_change_ms"]["point"]))
    lev["label_alt_mean_abs_reading"] = sens_label(lev["plus3"]["mean_abs_change_ms"]["point"])
    lev["label_rule"] = f"A1's committed rule: O-SENSITIVE iff |mean signed plan-speed change| for v_hat +3 m/s >= {SENS_HI}; O-INSENSITIVE iff < {SENS_LO}; else WEAK (the mean-of-absolute reading is label_alt_mean_abs_reading)"
    lead = {"presence_0_vs_baseline": stat(sp["p0"] - sp[""]), "presence_1_vs_baseline": stat(sp["p1"] - sp[""]), "flip_1_minus_0": stat(sp["p1"] - sp["p0"])}
    lead["label_by_analogy"] = sens_label(abs(lead["flip_1_minus_0"]["mean_signed_change_ms"]["point"]))
    lead["label_alt_mean_abs_reading"] = sens_label(lead["flip_1_minus_0"]["mean_abs_change_ms"]["point"])
    lead["label_rule"] = "the A2 card commits no threshold for the lead input; A1's thresholds (0.5 / 0.1 m/s) are applied BY ANALOGY to the presence 0 -> 1 flip and labelled as such"
    beside = {"gap_0p75_minus_0p25_at_presence_1": stat(sp["ghi"] - sp["glo"]), "e_aux_output_zeroed": stat(sp["ez"] - sp[""]),
              "note": "beside, not committed measurements: the gap column at presence 1 (10 m vs 30 m), and e_aux's output replaced by zeros at inference (attribution)"}
    ai = A["a_in"]
    ok = A["frame_ok_now"]
    sg = lambda x: 1.0 / (1.0 + np.exp(-x.astype(np.float64)))
    exp_in = np.stack([A["v_hat"] / LEVEL_SCALE, sg(A["lead_o"][:, 0]), sg(A["lead_o"][:, 1])], -1)
    dev_ok = float(np.abs(ai[ok] - exp_in[ok]).max()) if ok.any() else None
    dev_bad = float(np.abs(ai[~ok]).max()) if (~ok).any() else None
    ctrl = {"zero_shift_change_max_abs": float(np.abs(sp["z"] - sp[""]).max()), "zero_shift_traj_max_abs_diff_m": float(np.abs(A["traj60_z"] - A["traj60"]).max()),
            "a_in_equals_vhat10_sigmoid_logit_sigmoid_gaplogit_where_frame_ok_max_abs_dev": dev_ok, "a_in_zero_where_frame_not_ok_max_abs": dev_bad,
            "level_shift_reaches_column0_max_abs_dev_from_0p3": float(max(np.abs(A["a_in_lp"][:, 0] - ai[:, 0] - 0.3).max(), np.abs(A["a_in_lm"][:, 0] - ai[:, 0] + 0.3).max())),
            "level_shift_leaves_columns_1_2_max_abs_dev": float(max(np.abs(A["a_in_lp"][:, 1:] - ai[:, 1:]).max(), np.abs(A["a_in_lm"][:, 1:] - ai[:, 1:]).max())),
            "presence_set_reaches_column1_exact": bool(np.all(A["a_in_p0"][:, 1] == 0.0) and np.all(A["a_in_p1"][:, 1] == 1.0)),
            "presence_set_leaves_columns_0_2_max_abs_dev": float(max(np.abs(A["a_in_p1"][:, [0, 2]] - ai[:, [0, 2]]).max(), np.abs(A["a_in_p0"][:, [0, 2]] - ai[:, [0, 2]]).max())),
            "gap_set_reaches_column2_exact": bool(np.all(A["a_in_glo"][:, 2] == GAP_LO) and np.all(A["a_in_ghi"][:, 2] == GAP_HI)),
            "level_v_equals_hook_vhat_max_abs_dev": float(np.abs(A["level_v"] - A["v_hat"]).max()),
            "reach_control_big_e_aux_output_perturbation": {"mean_abs_plan_speed_change_ms": float(np.abs(sp["big"] - sp[""]).mean()), "max_abs_ms": float(np.abs(sp["big"] - sp[""]).max()),
                                                           "note": "instrument control: a norm-8 perturbation of e_aux's output must move the plan (reads > 0 iff the aux token is wired into O's plan in this harness)"}}
    en, gn = np.linalg.norm(A["a_out"], axis=1), np.linalg.norm(A["ego_tok"], axis=1)
    ratio = en / np.where(gn > 0, gn, np.nan)
    gd = np.isfinite(ratio)
    ctx = {"e_aux_abs_weight_sum_per_input_column_level_presence_gap": [float(x) for x in np.abs(A["e_aux_w"]).sum(0)], "e_aux_abs_bias_sum": float(np.abs(A["e_aux_b"]).sum()),
           "e_aux_over_ego_norm_mean_of_per_window_ratio": point_and_ci(ep_sums(eid, uniq, gd, np.where(gd, ratio, 0)), M, mean_stat), "mean_e_aux_norm": float(en.mean()), "mean_ego_token_norm": float(gn.mean()),
           "head_outputs": {"v_hat_range": [float(A["v_hat"].min()), float(A["v_hat"].max())], "presence_prob_mean": float(sg(A["lead_o"][:, 0]).mean()), "gap_hat_mean": float(sg(A["lead_o"][:, 1]).mean())}}
    out.update(level=lev, lead=lead, beside=beside, controls=ctrl, context=ctx)
    out["labels"] = {"level": lev["label"], "lead_by_analogy": lead["label_by_analogy"], "level_same_under_both_readings": lev["label"] == lev["label_alt_mean_abs_reading"],
                     "lead_same_under_both_readings": lead["label_by_analogy"] == lead["label_alt_mean_abs_reading"]}
    return out


def banked_tick0_check(s, A):
    p = BANKED_TICKS.format(s=s)
    if not os.path.exists(p):
        return {"status": "NOT COMPUTED: banked LLR ticks file absent", "path": p}
    z = np.load(p, allow_pickle=True)
    t0rows = np.where(z["tick"] == 0)[0]
    pos = {(int(c), int(t)): r for c, t, r in zip(z["ci"][t0rows], z["t0"][t0rows], t0rows)}
    rows, mine = [], []
    for k, (c, t) in enumerate(zip(A["ci"], A["t0"])):
        if (int(c), int(t)) in pos:
            rows.append(pos[(int(c), int(t))])
            mine.append(k)
    rows, mine = np.array(rows), np.array(mine)
    dev = np.abs(A["traj60"][mine][:, 1::2] - z["plan_raw"][rows])
    return {"status": "computed", "path": p, "n_compared": int(len(rows)), "max_abs_dev_m": float(dev.max()), "mean_abs_dev_m": float(dev.mean()),
            "note": "banked rollout: 16 or 12 threads, batch 32; a different thread count / batch changes float rounding only (1e-5 m class); a gap >> 1e-3 m would mean a different tick-0 batch"}


def control_failures(res):
    c, bt, f = res["controls"], res["banked_tick0_check"], []
    if c["zero_shift_change_max_abs"] != 0.0:
        f.append("zero shift did not read exactly 0.0")
    if c["a_in_equals_vhat10_sigmoid_logit_sigmoid_gaplogit_where_frame_ok_max_abs_dev"] is not None and c["a_in_equals_vhat10_sigmoid_logit_sigmoid_gaplogit_where_frame_ok_max_abs_dev"] > 1e-5:
        f.append("the e_aux input is not [v_hat / 10, sigmoid(logit), gap_hat]")
    if c["a_in_zero_where_frame_not_ok_max_abs"] not in (None, 0.0):
        f.append("the e_aux input is not zero where the frame is not real")
    if c["level_shift_reaches_column0_max_abs_dev_from_0p3"] > 1e-5 or c["level_shift_leaves_columns_1_2_max_abs_dev"] != 0.0:
        f.append("the +-0.3 shift did not reach column 0 alone")
    if not c["presence_set_reaches_column1_exact"] or c["presence_set_leaves_columns_0_2_max_abs_dev"] != 0.0:
        f.append("the presence set did not reach column 1 alone")
    if not c["gap_set_reaches_column2_exact"]:
        f.append("the gap set did not reach column 2")
    if c["level_v_equals_hook_vhat_max_abs_dev"] != 0.0:
        f.append("forward hook v_hat != out['level_v']")
    if not c["reach_control_big_e_aux_output_perturbation"]["mean_abs_plan_speed_change_ms"] > 0.0:
        f.append("reach control: a norm-8 perturbation of e_aux's output did not move the plan")
    if bt.get("status") == "computed" and bt["max_abs_dev_m"] > 1e-2:
        f.append("tick-0 plan differs from the banked LLR rollout's tick 0")
    return f


def merge(d):
    rs = {s: json.load(open(f"{d}/LLR_READOUT_s{s}.json", encoding="utf-8")) for s in (0, 1)}
    fails = [f"s{s}: {x}" for s, r in rs.items() for x in r["control_failures"]] + [f"s{s}: not the full eval set" for s, r in rs.items() if not r["full_eval_set"]]
    row = lambda r, key: "%+.4f [%+.4f, %+.4f]" % (r["point"], r["ci95"][0], r["ci95"][1])
    L = ["# LLR read-out: O's sensitivity to the token wiring (A2 card Reads; A1's measurement adapted to --aux-cond token)", "",
         "Statistic: change of the plan's mean speed over 0..3 s (m/s), changed input minus baseline, per window, episode-cluster bootstrap B=2000 seed 0 over the full eval set. LEVEL labels = A1's committed rule on the +3 m/s shift; "
         "LEAD labels = A1's thresholds applied BY ANALOGY to the presence 0 -> 1 flip (the card commits none).", "",
         "| input change | s0 mean signed [95% CI] | s0 mean abs | s1 mean signed [95% CI] | s1 mean abs |", "|---|---|---|---|---|"]
    for title, path in (("level: v_hat +3 m/s (token column 0 +0.3)", ("level", "plus3")), ("level: v_hat -3 m/s", ("level", "minus3")), ("lead: presence set to 1 (vs baseline)", ("lead", "presence_1_vs_baseline")),
                        ("lead: presence set to 0 (vs baseline)", ("lead", "presence_0_vs_baseline")), ("lead: presence flip 1 minus 0", ("lead", "flip_1_minus_0")),
                        ("beside: gap 30 m minus 10 m at presence 1", ("beside", "gap_0p75_minus_0p25_at_presence_1")), ("beside: e_aux output zeroed", ("beside", "e_aux_output_zeroed"))):
        cells = []
        for s in (0, 1):
            x = rs[s]
            for k in path:
                x = x[k]
            cells += [row(x["mean_signed_change_ms"], ""), "%.4f" % x["mean_abs_change_ms"]["point"]]
        L.append(f"| {title} | " + " | ".join(cells) + " |")
    L += ["", "| label | s0 | s1 |", "|---|---|---|",
          f"| LEVEL (A1's rule, +3 m/s) | {rs[0]['level']['label']} (mean-abs reading {rs[0]['level']['label_alt_mean_abs_reading']}) | {rs[1]['level']['label']} (mean-abs reading {rs[1]['level']['label_alt_mean_abs_reading']}) |",
          f"| LEAD (by analogy, presence flip) | {rs[0]['lead']['label_by_analogy']} (mean-abs reading {rs[0]['lead']['label_alt_mean_abs_reading']}) | {rs[1]['lead']['label_by_analogy']} (mean-abs reading {rs[1]['lead']['label_alt_mean_abs_reading']}) |",
          "", "||e_aux|| / ||ego token|| (mean of per-window ratios): " + ", ".join("s%d %.4f" % (s, rs[s]["context"]["e_aux_over_ego_norm_mean_of_per_window_ratio"]["point"]) for s in (0, 1)),
          "e_aux |weight| column sums (level, presence, gap): " + "; ".join("s%d %s" % (s, [round(x, 3) for x in rs[s]["context"]["e_aux_abs_weight_sum_per_input_column_level_presence_gap"]]) for s in (0, 1)), "",
          "Controls: " + ("ALL PASS" if not fails else "FAILED: " + "; ".join(fails)),
          "No wiring decision follows from this table: LLR is the last arm of the A2 card; the table is evidence for the write-up of why the token was or was not used."]
    res = {"seeds": {f"s{s}": {"labels": rs[s]["labels"], "ckpt_sha256": rs[s]["ckpt_sha256"], "ckpt_cfg": rs[s]["ckpt_cfg"]} for s in (0, 1)}, "control_failures": fails, "PASS": not fails, "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(res, open(f"{d}/LLR_READOUT.json", "w", encoding="utf-8", newline="\n"), indent=1)
    open(f"{PK}/SENSITIVITY.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("MERGE", "PASS" if not fails else "FAIL: " + "; ".join(fails), "|", {f"s{s}": rs[s]["labels"] for s in (0, 1)})
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    if a.merge:
        return merge(a.dir)
    t0 = time.time()
    A = run_seed(a.seed, a.max_windows, min(a.threads, 12), a.batch, a.ckpt or None)
    res = analyze(a.seed, A, a.max_windows == 0)
    res["banked_tick0_check"] = banked_tick0_check(a.seed, A)
    res["control_failures"] = control_failures(res)
    res["wall_s"] = time.time() - t0
    res["prereg"] = {"card": "PREREG_LEVELHEAD_A2.md (Reads: O-sensitivity by A1's measurement; wiring token per FROZEN_LLR.txt)", "bootstrap": f"episode-cluster, B={B_BOOT}, seed 0", "helpers": f"clr_readout.py sha256 {CRO_SHA[:8]}"}
    out = a.out or os.path.join(OUT_DIR, f"LLR_READOUT_s{a.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", newline="\n"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)
    print(json.dumps({"labels": res["labels"], "control_failures": res["control_failures"]}, indent=1))
    return 1 if res["control_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
