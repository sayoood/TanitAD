#!/usr/bin/env python3
"""SFT-1: scorer-only fine-tune of REFe's final model on its on-policy sets -- two arms in ONE pass over the data.

WHY SCORER-ONLY: the scoring decoder reads the trajectories AND the visual context DETACHED (model.py:746, :751), so
the score loss reaches only `score_q_mlp`, `score_dec`, `score_head`. Freezing everything else changes nothing the
proposals depend on: the 64 hypotheses stay bit-identical, only the CHOICE among them moves. (The navtest evidence:
983 of 12,146 picks score 0, and on W3's 200 tokens 14 of 15 zero picks had a hypothesis scoring >= 80 -- selection,
not proposal, failures; raw/2026-10-04-navtest-final.)

ARMS (one trunk forward per batch; both scorers read the SAME captured context, so the comparison is paired by
construction):
  base   a frozen copy of the final scorer (the reference; must equal A and B exactly at step 0)
  A      the model's own scorer modules, trained with the TRAINING loss (per-component BCE on each labelled set) --
         the control: is it just more training at a low lr?
  B      a copy trained with that BCE + lam * a LISTWISE ranking loss over each 64-hypothesis set (ListNet: the
         predicted distribution p_i ~ the planner's own navsim_v1 aggregate of sigmoid scores, the target
         q_i ~ exp(true_v1_i / tau_t)) -- selection is a within-scene RANKING problem; BCE only calibrates.
DATA: train.py's own TargetBank + OnPolicyBank (the identical inputs and labels training used), restricted to samples
with a complete labelled set and to logs OUTSIDE the 24 held-out logs. HELD-OUT: the 3,137 v5-labelled sets of the
final model's own proposals (24 logs) -- selection metrics per arm: pick / random / best true navsim_v1, skill, picks
scoring 0, within-set AUC of NC and DAC, and the paired log-cluster bootstrap of pick(arm) - pick(base).
  python scorer_finetune.py --preflight ...      # the launch gate; prints ZZSFT_PREFLIGHT_PASS or _FAIL
  python scorer_finetune.py ...                  # the run (refuses without a preflight PASS file in --out)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io  # noqa: E402
import train as T  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402

SCORER = ("score_q_mlp", "score_dec", "score_head")
V1_W = (5.0, 5.0, 2.0)


def agg_v1(p):
    """planner.aggregate, rule navsim_v1, on [..., 6] probabilities OR labels: NC*DAC*(5EP+5TTC+2C)/12"""
    return p[..., 0] * p[..., 1] * (V1_W[0] * p[..., 2] + V1_W[1] * p[..., 3] + V1_W[2] * p[..., 4]) / sum(V1_W)


class ScorerCopy(nn.Module):
    """A detached copy of the three scorer modules; forward = REFe.score_trajectories itself (no re-implementation)."""

    def __init__(self, model):
        super().__init__()
        for m in SCORER:
            setattr(self, m, copy.deepcopy(getattr(model, m)))

    def forward(self, trajs, sctx):
        return REFe.score_trajectories(self, trajs, sctx)


def expected_reward_loss(sx, tg, tau_s):
    """-E_{i ~ pi}[R_i], pi = softmax(log agg_v1(sigmoid(sx)) / tau_s) over each set, R_i = agg_v1(labels) (the true score
    of hypothesis i). A one-step choice whose outcome is KNOWN for every action: this is the policy-gradient objective in
    closed form -- no sampled action, so no sampling variance. As tau_s -> 0, pi -> the planner's argmax."""
    s = torch.log(agg_v1(torch.sigmoid(sx)).clamp_min(1e-6))
    pi = torch.softmax(s / tau_s, dim=1)
    return -(pi * agg_v1(tg)).sum(1).mean()


def key(r):
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))


def set_auc(pred, lab):
    """within-set AUC of `pred` ranking PASS (lab >= 0.5) above FAIL; None if the set has one class only"""
    pos, neg = pred[lab >= 0.5], pred[lab < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()))


def boot_logs(diff: dict, logs: dict, n=10000, seed=20260927):
    toks = sorted(diff)
    by: dict = {}
    for i, t in enumerate(toks):
        by.setdefault(logs[t], []).append(i)
    groups = [np.asarray(v) for _k, v in sorted(by.items())]
    vals = np.asarray([diff[t] for t in toks], np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        means[b] = vals[idx].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--onpolicy", required=True, help="the training on-policy sets dir")
    ap.add_argument("--heldout", required=True, help="the held-out on-policy sets dir")
    ap.add_argument("--heldout-logs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.3, help="arm B: ListNet weight")
    ap.add_argument("--tau-t", type=float, default=0.1, help="arm B: target temperature on the true navsim_v1 score")
    ap.add_argument("--eval-every", type=int, default=400, help="optimizer updates between held-out evals")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--max-updates", type=int, default=0)
    # SFT-2 (LANE-1 follow-on); all default OFF = SFT-1 exactly
    ap.add_argument("--lane-labels-train", default="", help="side files of onpolicy_relabel_lane.py for --onpolicy")
    ap.add_argument("--lane-labels-heldout", default="", help="side files of onpolicy_relabel_lane.py for --heldout")
    ap.add_argument("--b-mode", default="listnet", choices=("listnet", "compw", "expreward"),
                    help="arm B: BCE + lam*ListNet (SFT-1) or per-component weighted BCE (--b-compw)")
    ap.add_argument("--b-compw", default="1,1,1,1,1,1", help="arm B component weights NC,DAC,EP,TTC,C,DDC (b-mode compw)")
    ap.add_argument("--tau-s", type=float, default=0.1,
                    help="b-mode expreward: temperature of the selection policy softmax(log agg_v1(p) / tau_s)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(a.seed)
    log_p = os.path.join(a.out, "preflight.jsonl" if a.preflight else "sft.jsonl")

    def log(row):
        row["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(log_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    if not a.preflight and not os.path.exists(os.path.join(a.out, "PREFLIGHT_PASS.json")):
        print("REFUSED: no PREFLIGHT_PASS.json in --out -- run --preflight first (the launch gate)"); return 4

    # ---------------- model: built and loaded EXACTLY as the planner does
    cfg = REFeConfig.for_backbone(a.backbone)
    model = REFe(cfg).to(dev)
    meta: dict = {}
    fmt = ckpt_io.load_for_inference(model, a.ckpt, map_location=dev, backbone=a.backbone, meta_out=meta)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for m in SCORER:
        for p in getattr(model, m).parameters():
            p.requires_grad_(True)
    if os.environ.get("REFE_SFT_MUTATE_UNFREEZE") == "1":            # deliberate regression for the gate
        for p in model.traj_head.parameters():
            p.requires_grad_(True)
    base = ScorerCopy(model).to(dev)
    for p in base.parameters():
        p.requires_grad_(False)
    armB = ScorerCopy(model).to(dev)
    for p in armB.parameters():
        p.requires_grad_(True)
    cap: dict = {}
    model.score_dec[0].register_forward_pre_hook(lambda mod, args: cap.__setitem__("sctx", args[1]))

    # ---------------- data: train.py's own banks
    op = T.OnPolicyBank(a.onpolicy, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_train or None)
    ho = T.OnPolicyBank(a.heldout, cfg.n_proposals, cfg.horizon_steps, lane_labels=a.lane_labels_heldout or None)
    compw = torch.tensor([float(x) for x in a.b_compw.split(",")], device=dev)
    assert compw.numel() == 6 and (compw > 0).all(), "--b-compw needs 6 positive weights"
    ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib, onpolicy=op)
    ho_logs = set(open(a.heldout_logs, encoding="utf-8").read().split())
    tr_idx = [i for i, r in enumerate(ds.rows) if key(r) in op.by and r.get("log_name") not in ho_logs]
    ds_ho = copy.copy(ds)
    ds_ho.onpolicy = ho
    ho_idx = [i for i, r in enumerate(ds.rows) if key(r) in ho.by]
    lv: dict = {}
    for k_ in (key(ds.rows[i]) for i in tr_idx):
        v = op.by[k_][3][1]
        lv[v] = lv.get(v, 0) + 1
    info = {"event": "data", "train_sets": len(tr_idx), "train_sets_in_bank": len(op.by),
            "train_label_versions": lv, "navsim_dac_sets": op.n_navsim_dac,
            "heldout_sets": len(ho_idx), "heldout_in_bank": len(ho.by), "heldout_logs": len(ho_logs),
            "train_rows_in_heldout_logs": sum(1 for i in tr_idx if ds.rows[i].get("log_name") in ho_logs),
            "ckpt_format": fmt, "per_sample_calib": meta.get("per_sample_calib"),
            "lane_labels": {"train_sets": op.n_lane, "train_stale": op.n_lane_stale, "train_superseded": op.n_lane_superseded,
                            "heldout_sets": ho.n_lane,
                            "heldout_stale": ho.n_lane_stale}, "b_mode": a.b_mode, "b_compw": a.b_compw}
    log(info)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(Subset(ds, tr_idx), batch_size=a.batch, shuffle=True, generator=g, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=a.workers > 0,
                    prefetch_factor=(4 if a.workers > 0 else None))

    def fwd(batch):
        img, ego, goal, _tgt, cxy, ctg, cm, _ci, cal = batch
        img, ego, goal, cxy, ctg = (x.to(dev, non_blocking=True) for x in (img, ego, goal, cxy, ctg))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _traj, _score, sxA = model(img, ego, goal, calib=cal if cal.numel() else None, score_extra=cxy)
            sctx = cap["sctx"]
            sxB = armB(cxy, sctx)
            with torch.no_grad():
                sx0 = base(cxy, sctx)
        return sx0.float(), sxA.float(), sxB.float(), ctg.float()

    def losses(sxA, sxB, tg, lam):
        bceA = F.binary_cross_entropy_with_logits(sxA, tg)
        if a.b_mode == "compw":                              # SFT-2: weighted per-component BCE, no ranking term
            w = compw.view(1, 1, 6)
            bceBw = (F.binary_cross_entropy_with_logits(sxB, tg, reduction="none") * w).sum(-1).mean() / w.sum()
            return bceA, bceBw, torch.zeros((), device=sxB.device)
        if a.b_mode == "expreward":                          # SFT-3: the selection's EXPECTED true score, all 64 outcomes known
            bceB = F.binary_cross_entropy_with_logits(sxB, tg)
            return bceA, bceB + lam * expected_reward_loss(sxB, tg, a.tau_s), expected_reward_loss(sxB, tg, a.tau_s)
        bceB = F.binary_cross_entropy_with_logits(sxB, tg)
        s = torch.log(agg_v1(torch.sigmoid(sxB)).clamp_min(1e-6))
        q = torch.softmax(agg_v1(tg) / a.tau_t, dim=1)
        listnet = -(q * torch.log_softmax(s, dim=1)).sum(1).mean()
        return bceA, bceB + lam * listnet, listnet

    @torch.no_grad()
    def evaluate(tag, limit=None):
        idx = ho_idx[:limit] if limit else ho_idx
        dlh = DataLoader(Subset(ds_ho, idx), batch_size=16, shuffle=False, num_workers=2)
        res = {k: {"pick": [], "rand": [], "best": [], "zero": [], "auc_nc": [], "auc_dac": [], "auc_ddc": [],
                   "onc": [], "pick_x": [], "onc_x": [], "zero_x": []} for k in ("base", "A", "B")}
        toks, logs_ = [], []
        j = 0
        for batch in dlh:
            sx0, sxA, sxB, tg = fwd(batch)
            ta = agg_v1(tg)                                                      # [b, M] true navsim_v1
            for k, sx in (("base", sx0), ("A", sxA), ("B", sxB)):
                p = torch.sigmoid(sx)
                ix = agg_v1(p).argmax(1, keepdim=True)
                pick = ta.gather(1, ix).squeeze(1)
                res[k]["pick"] += pick.tolist(); res[k]["rand"] += ta.mean(1).tolist()
                res[k]["best"] += ta.max(1).values.tolist(); res[k]["zero"] += (pick == 0).float().tolist()
                res[k]["onc"] += (tg[..., 5].gather(1, ix).squeeze(1) < 1).float().tolist()
                ixx = (agg_v1(p) * p[..., 5]).argmax(1, keepdim=True)          # LANE-1's L1: x the direction head
                pkx = ta.gather(1, ixx).squeeze(1)
                res[k]["pick_x"] += pkx.tolist(); res[k]["zero_x"] += (pkx == 0).float().tolist()
                res[k]["onc_x"] += (tg[..., 5].gather(1, ixx).squeeze(1) < 1).float().tolist()
                for bi in range(p.shape[0]):
                    for comp, nm in ((0, "auc_nc"), (1, "auc_dac"), (5, "auc_ddc")):
                        v = set_auc(p[bi, :, comp].cpu(), tg[bi, :, comp].cpu())
                        if v is not None:
                            res[k][nm].append(v)
            for _ in range(tg.shape[0]):
                r = ds.rows[idx[j]]; j += 1
                toks.append(f"{r.get('log_name')}|{r.get('token')}|{r.get('step')}|{r.get('rank', 0)}"); logs_.append(r.get("log_name"))
        out = {"event": "eval", "tag": tag, "n": len(toks)}
        for k, v in res.items():
            pk, rd, bs = np.mean(v["pick"]), np.mean(v["rand"]), np.mean(v["best"])
            out[k] = {"pick": 100 * pk, "random": 100 * rd, "best": 100 * bs, "skill": (pk - rd) / max(bs - rd, 1e-9),
                      "zero_picks": int(sum(v["zero"])), "auc_nc": float(np.mean(v["auc_nc"])) if v["auc_nc"] else None,
                      "auc_dac": float(np.mean(v["auc_dac"])) if v["auc_dac"] else None,
                      "auc_ddc": float(np.mean(v["auc_ddc"])) if v["auc_ddc"] else None,
                      "onc_picks": int(sum(v["onc"])),
                      "x_ddc": {"pick": 100 * float(np.mean(v["pick_x"])), "zero_picks": int(sum(v["zero_x"])),
                                "onc_picks": int(sum(v["onc_x"]))}}
        tl = dict(zip(toks, logs_))
        for k in ("A", "B"):
            d = {t: 100 * (res[k]["pick"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(d, tl)
            out[k]["pick_minus_base"] = {"mean": mu, "ci95": [lo, hi], "estimator": "log-cluster bootstrap over the held-out logs, 10,000, seed 20260927"}
            dx = {t: 100 * (res[k]["pick_x"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(dx, tl)
            out[k]["x_ddc"]["pick_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
            do = {t: 100 * (res[k]["onc_x"][i] - res["base"]["onc"][i]) for i, t in enumerate(toks)}
            mu, lo, hi = boot_logs(do, tl)
            out[k]["x_ddc"]["onc_pp_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        dx = {t: 100 * (res["base"]["pick_x"][i] - res["base"]["pick"][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(dx, tl)
        out["base"]["x_ddc"]["pick_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        do = {t: 100 * (res["base"]["onc_x"][i] - res["base"]["onc"][i]) for i, t in enumerate(toks)}
        mu, lo, hi = boot_logs(do, tl)
        out["base"]["x_ddc"]["onc_pp_minus_base_v1"] = {"mean": mu, "ci95": [lo, hi]}
        log(out)
        return out

    def save_full(arm_name, mods):
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        new = {}
        for m in SCORER:
            for k_, v in getattr(mods, m).state_dict().items():
                new[f"{m}.{k_}"] = v.detach().to("cpu", copy=True)
        assert set(new) <= set(state), "scorer keys missing from the checkpoint"
        for k_, v in new.items():
            assert state[k_].shape == v.shape
            state[k_] = v.to(state[k_].dtype)
        out = {"format": ckpt_io.FORMAT_FULL, "model": state,
               "meta": dict(sd.get("meta", {}) if isinstance(sd, dict) else {},
                            scorer_finetune={"arm": arm_name, "base_ckpt": a.ckpt, "args": vars(a)})}
        p = os.path.join(a.out, f"model_sft_{arm_name}.pt")
        ckpt_io.atomic_save(out, p)
        return p

    # ---------------- PREFLIGHT (the launch gate)
    if a.preflight:
        gates = {}
        tn = ckpt_io.trainable_names(model)
        want = {n for n, _ in model.named_parameters() if n.split(".")[0] in SCORER}
        gates["G1_trainable_is_exactly_the_scorer"] = {"ok": tn == want, "n_trainable": len(tn), "n_scorer": len(want),
                                                       "extra": sorted(tn - want)[:5], "missing": sorted(want - tn)[:5]}
        gates["G7_heldout_disjoint"] = {"ok": info["train_rows_in_heldout_logs"] == 0 and len(ho_idx) > 0,
                                        "heldout_sets": len(ho_idx), "train_rows_in_heldout_logs": info["train_rows_in_heldout_logs"]}
        batch = next(iter(DataLoader(Subset(ds, tr_idx[:a.batch]), batch_size=a.batch)))
        sx0, sxA, sxB, tg = fwd(batch)
        gates["G3_copies_equal_model_at_step0"] = {"ok": float((sx0 - sxA).abs().max()) == 0.0 and float((sxB - sxA).abs().max()) == 0.0,
                                                   "max_base_vs_A": float((sx0 - sxA).abs().max()), "max_B_vs_A": float((sxB - sxA).abs().max())}
        if a.b_mode == "compw":                              # SFT-2: unit weights must reproduce A; the declared ones must not
            keep_w = compw.clone()
            compw.fill_(1.0)
            bA, bB0, _ = losses(sxA, sxB, tg, 0.0)
            compw.copy_(keep_w)
            bA2, bB, ln = losses(sxA, sxB, tg, 0.0)
            differs = bool((keep_w != 1).any())
            gates["G4_compw_identity_and_live"] = {"ok": math.isclose(float(bA), float(bB0), rel_tol=1e-5) and
                                                   (float(bB) != float(bA) if differs else True) and math.isfinite(float(bB)),
                                                   "bceA": float(bA), "lossB_unit_w": float(bB0), "lossB": float(bB), "compw": keep_w.tolist()}
        else:
            bA, bB0, _ = losses(sxA, sxB, tg, 0.0)
            bA2, bB, ln = losses(sxA, sxB, tg, a.lam)
            gates["G4_listnet_identity_and_live"] = {"ok": float(bA) == float(bB0) and float(bB) != float(bA) and math.isfinite(float(ln)),
                                                     "bceA": float(bA), "lossB_lam0": float(bB0), "lossB": float(bB), "listnet": float(ln)}
        if a.lane_labels_train or a.lane_labels_heldout:      # G8: the NAVSIM direction labels really are in the banks
            def has_half(bank):
                return sum(1 for e in bank.by.values() if np.any(np.abs(e[2][:, 5] - 0.5) < 1e-6))
            g8 = {"train_lane_sets": op.n_lane, "train_stale": op.n_lane_stale, "train_superseded": op.n_lane_superseded,
                  "heldout_lane_sets": ho.n_lane, "heldout_stale": ho.n_lane_stale, "train_sets_with_ddc_0.5": has_half(op), "heldout_sets_with_ddc_0.5": has_half(ho)}
            g8["ok"] = all([(not a.lane_labels_train) or (op.n_lane > 0 and op.n_lane_stale <= 0.05 * max(op.n_lane, 1)
                                                          and g8["train_sets_with_ddc_0.5"] > 0),
                            (not a.lane_labels_heldout) or (ho.n_lane > 0 and ho.n_lane_stale <= 0.05 * max(ho.n_lane, 1)
                                                            and g8["heldout_sets_with_ddc_0.5"] > 0)])
            gates["G8_lane_labels_in_bank"] = g8
        (bA2 + bB).backward()
        live = {}
        for m in SCORER:
            ga = [p.grad for p in getattr(model, m).parameters()]
            gb = [p.grad for p in getattr(armB, m).parameters()]
            live[m] = {"A_nonzero": all(x is not None and torch.isfinite(x).all() and float(x.abs().sum()) > 0 for x in ga),
                       "B_nonzero": all(x is not None and torch.isfinite(x).all() and float(x.abs().sum()) > 0 for x in gb)}
        frozen_grads = sum(1 for n, p in model.named_parameters() if n.split(".")[0] not in SCORER and p.grad is not None)
        base_grads = sum(1 for p in base.parameters() if p.grad is not None)
        gates["G2_live"] = {"ok": all(v["A_nonzero"] and v["B_nonzero"] for v in live.values()) and frozen_grads == 0 and base_grads == 0,
                            "per_module": live, "frozen_params_with_grad": frozen_grads, "base_params_with_grad": base_grads}
        # G5: a CHANGED scorer, saved, reloads through the PLANNER's loader and reproduces its logits (and differs from
        # base) -- saving an unchanged scorer would pass even if the save wrote the original weights back
        with torch.no_grad():
            gen = torch.Generator(device=dev).manual_seed(1)
            for p in (p for m in SCORER for p in getattr(model, m).parameters()):
                p.add_(1e-2 * torch.randn(p.shape, device=dev, generator=gen))
        pth = save_full("A_preflight", model)
        m2 = REFe(cfg).to(dev)
        ckpt_io.load_for_inference(m2, pth, map_location=dev, backbone=a.backbone)
        m2.eval()
        cap2: dict = {}
        img, ego, goal, _t, cxy, _g, _m, _i, cal = batch
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, s_reload = m2(img.to(dev), ego.to(dev), goal.to(dev), calib=cal if cal.numel() else None, score_extra=cxy.to(dev))
            _, _, s_orig = model(img.to(dev), ego.to(dev), goal.to(dev), calib=cal if cal.numel() else None, score_extra=cxy.to(dev))
            s_base = base(cxy.to(dev), cap["sctx"])
        d_rt, d_base = float((s_reload.float() - s_orig.float()).abs().max()), float((s_orig.float() - s_base.float()).abs().max())
        gates["G5_ckpt_roundtrip_planner_loader"] = {"ok": d_rt == 0.0 and d_base > 0.0, "max_abs_reload_vs_saved": d_rt,
                                                     "max_abs_saved_vs_base": d_base}
        os.remove(pth)
        # restore the true starting weights (the perturbation was for the gate only)
        sd0 = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        st0 = sd0["model"] if isinstance(sd0, dict) and "model" in sd0 else sd0
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.split(".")[0] in SCORER:
                    p.copy_(st0[n].to(dev))
        del sd0, st0
        # speed: 3 timed micro-batches (fwd + both backwards)
        t0 = time.time()
        it = iter(dl)
        for _ in range(3):
            b_ = next(it)
            s0, sA, sB, t_ = fwd(b_)
            l1, l2, _ = losses(sA, sB, t_, a.lam)
            (l1 + l2).backward()
        torch.cuda.synchronize()
        sp = (time.time() - t0) / 3
        ok = all(v["ok"] for v in gates.values())
        res = {"event": "preflight", "verdict": "PASS" if ok else "FAIL", "gates": gates,
               "s_per_microbatch": round(sp, 2), "est_hours_per_epoch": round(len(tr_idx) / a.batch * sp / 3600, 2),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log(res)
        if ok and os.environ.get("REFE_SFT_MUTATE_UNFREEZE") != "1":
            json.dump(res, open(os.path.join(a.out, "PREFLIGHT_PASS.json"), "w"), indent=1, default=str)
        print("ZZSFT_PREFLIGHT_" + ("PASS" if ok else "FAIL"))
        return 0 if ok else 3

    # ---------------- the run
    paramsA = [p for m in SCORER for p in getattr(model, m).parameters()]
    paramsB = list(armB.parameters())
    optA = torch.optim.AdamW(paramsA, lr=a.lr, weight_decay=0.01)
    optB = torch.optim.AdamW(paramsB, lr=a.lr, weight_decay=0.01)
    total = int(math.ceil(len(tr_idx) / a.batch * a.epochs / a.accum))
    if a.max_updates:
        total = min(total, a.max_updates)
    lr_at = lambda u: a.lr * (u + 1) / a.warmup if u < a.warmup else a.lr * 0.5 * (1 + math.cos(math.pi * (u - a.warmup) / max(total - a.warmup, 1)))
    log({"event": "start", "updates": total, "micro_per_update": a.accum, "batch": a.batch, "train_sets": len(tr_idx)})
    evaluate("step0")
    upd, micro, t0 = 0, 0, time.time()
    accl = {"bceA": 0.0, "lossB": 0.0, "listnet": 0.0, "n": 0}
    done = False
    while not done:
        for batch in dl:
            _s0, sA, sB, tg = fwd(batch)
            lA, lB, ln = losses(sA, sB, tg, a.lam)
            ((lA + lB) / a.accum).backward()
            accl["bceA"] += float(lA); accl["lossB"] += float(lB); accl["listnet"] += float(ln); accl["n"] += 1
            micro += 1
            if micro % a.accum:
                continue
            for o, ps in ((optA, paramsA), (optB, paramsB)):
                torch.nn.utils.clip_grad_norm_(ps, 1.0)
                for gr in o.param_groups:
                    gr["lr"] = lr_at(upd)
                o.step()
                o.zero_grad(set_to_none=True)
            upd += 1
            if upd % 25 == 0:
                n = max(accl["n"], 1)
                log({"event": "train", "update": upd, "of": total, "lr": lr_at(upd), "bceA": accl["bceA"] / n,
                     "lossB": accl["lossB"] / n, "listnet": accl["listnet"] / n, "s_per_update": (time.time() - t0) / upd})
                accl = {"bceA": 0.0, "lossB": 0.0, "listnet": 0.0, "n": 0}
            if upd % a.eval_every == 0 or upd == total:
                evaluate(f"update{upd}")
                torch.save({"A": {m: getattr(model, m).state_dict() for m in SCORER},
                            "B": {m: getattr(armB, m).state_dict() for m in SCORER}, "update": upd},
                           os.path.join(a.out, "scorer_states_last.pt"))
            if upd >= total:
                done = True
                break
    pA, pB = save_full("A", model), save_full("B", armB)
    log({"event": "done", "updates": upd, "model_A": pA, "model_B": pB, "hours": (time.time() - t0) / 3600})
    print("ZZSFT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
