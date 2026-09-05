#!/usr/bin/env python3
"""THE 0-TRAINING PRODUCT: make refcv3's SELECTED path safer TODAY, and price it in ADE.

P1 (`rl_reward_envelope_rank.py`) measured something that is a deliverable rather than a
diagnosis: on 240 EVAL windows the RULE REWARD's own argmax picks an envelope-violating
candidate on **5.4 %** of windows, against the fan's base rate of **88.8 %** and refcv3's
own trained selector's **10.8 %**. The scorer we were about to spend GPU-days teaching a
network to imitate is already a BETTER feasibility ranker than the network -- and it needs
no training at all, because it is a rule over the emitted trajectory.

So this probe measures the SELECTION RULE as a product, on the same fan, with the cost
that decides whether it ships: **the ADE it gives up**.

RULES SCORED (every one of them re-ranks the SAME 128 emitted candidates; the model is
never retrained and never even reloaded differently):

  model          refcv3's own `sel_idx` -- the deployed baseline
  kin_only       argmax of `feasibility + comfort` -- NO SCENE INPUT AT ALL. These two
                 components read only the candidate's own waypoints, so this rule is
                 admissible under the vision-only rule with ZERO new perception: it needs
                 no lead, no obstacle track, no ego state.
  reward_full    argmax of the composed DEFAULT reward -- needs the lead track, i.e. a
                 perception dependency `kin_only` does not have. Reported for contrast.
  gateK_<k>      THE ONE THAT IS MEANT TO SHIP: keep the model's TOP-k by its own
                 `sel_score` (so the semantic/tactical ranking the network learned is
                 preserved) and re-rank ONLY those k by the kinematic score. k = 1 is the
                 model itself by construction -- a built-in identity control.
  oracle         the fan's best-ADE candidate -- the ceiling, T0, never deployable.

⛔ THE TRADE-OFF IS THE POINT, NOT A CAVEAT. `D-RL-FANSAFE-1` measured that the candidate
which best matches the human is the one the car can LEAST drive (`oracle_sel` infeasible
0.1105 > `os` 0.0865 > human 0.0068). A rule that only made the path more feasible while
destroying ADE would be worthless, so ADE is reported for every rule with a PAIRED
episode-cluster CI against the model, and the gate-k sweep IS the frontier.

Tier: T0 readout on the emitted fan over EVAL windows -- selection quality, never a
driving claim. Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.environ.get("TANITAD_REPO") or os.path.dirname(os.path.dirname(_HERE))


def _load_driver():
    path = os.path.join(_REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("_rl_refcv3_min_rerank", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


D = _load_driver()
RW = D.RW
FS = D.FS
import taniteval.ci as CI                                              # noqa: E402

TOOL = "stack/scripts/rl_fan_rerank_probe.py"
GATE_KS = (2, 4, 8, 16, 32, 128)
#: offset-shrink sweep. 1.0 is the shipped fan (identity control), 0.0 the raw bank.
LAMBDAS = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
REPORT = ("envelope", "kamm_over", "infeasible", "off_reach", "contact", "ttc_below",
          "flagged")


def boot(v, e, *, n_boot, seed):
    v = np.asarray(v, dtype=np.float64)
    e = np.asarray(e)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_windows": int(ok.sum()), "n_episodes": 0}
    r = CI.episode_cluster_bootstrap(v[ok], e[ok], n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes", "estimator")}


def pboot(a_, b_, e, *, n_boot, seed):
    a_ = np.asarray(a_, dtype=np.float64)
    b_ = np.asarray(b_, dtype=np.float64)
    e = np.asarray(e)
    ok = np.isfinite(a_) & np.isfinite(b_)
    if ok.sum() < 3:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "separated": False, "n_windows": int(ok.sum())}
    r = CI.paired_episode_cluster_bootstrap(a_[ok], b_[ok], e[ok], n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("delta", "lo", "hi", "separated", "n_windows", "n_episodes")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--expect-step", type=int, default=40284)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--windows", type=int, default=240)
    ap.add_argument("--window-seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    D.LEAD_MODE = "track"
    device = a.device if torch.cuda.is_available() else "cpu"

    class _A:
        ckpt, config, expect_step = a.ckpt, a.config, a.expect_step
    model, cfg, _t, prov = D.load(_A, device)
    model.eval()
    corp = D.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = D.load_lead_block(a.lead_block)
    pool = D.scoreable_windows(corp, lead)
    wis = sorted(random.Random(a.window_seed).sample(pool, min(len(pool), a.windows)))
    print(f"[rerank] {len(wis)} windows · device={device} · step={prov['step']}", flush=True)

    rules = ["model", "kin_only", "reward_full"] + [f"gate{k}" for k in GATE_KS] + ["oracle"]
    rows: list[dict] = []
    with torch.no_grad():
        for i in range(0, len(wis), a.batch):
            b = D.build_batch(corp, lead, wis[i:i + a.batch], device, with_gt=True)
            out = model(b["frames"], nav_cmd=b["nav_cmd"], v0=b["v0"],
                        steps=int(prov["decoder_steps"]))
            fan = out["anchor_traj"]                                     # [B, N, 8, 2]
            fan4 = fan[..., :D.N_REWARD_SLOTS, :]                        # [B, N, 4, 2]
            fan2 = D.with_origin(fan4)                                   # [B, N, 5, 2]
            ctx = D.reward_ctx(b, S5=fan2.shape[-2], cand_dims=1)
            comp = {n: RW.COMPONENTS[n](fan2, ctx) for n in
                    ("progress", "collision", "headway", "feasibility", "comfort")}
            r_full = sum(RW.DEFAULT_WEIGHTS[n] * comp[n] for n in RW.DEFAULT_WEIGHTS)
            r_kin = comp["feasibility"] + comp["comfort"]                # [B, N]
            lead5 = b["lead_track"].reshape(-1, 1, len(FS.GRID_S), 2)
            sc = FS.score_paths(fan2, b["v0"], lead5, lead_len_m=D.LEAD_LEN_DEFAULT_M)
            rank = out["sel_score"].detach().float()
            if out.get("reach_keep") is not None:
                rank = rank.masked_fill(~out["reach_keep"].bool(), float("-inf"))
            gt = b["gt_traj"]                                            # [B, 4, 2]
            ade = (fan4 - gt[:, None]).norm(dim=-1).mean(dim=-1)         # [B, N] per candidate
            order = rank.argsort(dim=1, descending=True)                 # [B, N]
            # ---- the OFFSET-SHRINK sweep: path(lambda) = bank + lambda * offset ---- #
            # `out["offset"]` is exactly what the decoder added to the bank, so the
            # interpolation is EXACT, not a reconstruction: at lambda = 1 the paths are
            # bitwise the emitted fan (asserted below), at lambda = 0 they are the bank.
            offs = out["offset"]                                         # [B, N, 8, 2]
            bank8 = fan - offs                                           # [B, N, 8, 2]
            lam_sc, lam_ade, lam_spread = {}, {}, {}
            for lam in LAMBDAS:
                p8 = bank8 + lam * offs
                p5 = D.with_origin(p8[..., :D.N_REWARD_SLOTS, :])
                lam_sc[lam] = FS.score_paths(p5, b["v0"], lead5,
                                             lead_len_m=D.LEAD_LEN_DEFAULT_M)
                lam_ade[lam] = (p8[..., :D.N_REWARD_SLOTS, :]
                                - gt[:, None]).norm(dim=-1).mean(dim=-1)
                lam_spread[lam] = p8[..., D.N_REWARD_SLOTS - 1, :].std(dim=1).norm(dim=-1)
            # identity control: lambda = 1 must reproduce the emitted fan EXACTLY.
            _d = float((lam_ade[1.0] - ade).abs().max())
            if _d > 1e-5:
                raise RuntimeError(f"lambda=1 identity control failed: max|diff| {_d:.3e} m "
                                   "- the sweep is not interpolating the shipped fan")

            for j, wi in enumerate(b["wis"]):
                e_i, _t = corp.ds.index[wi]
                pick = {"model": int(out["sel_idx"][j]),
                        "kin_only": int(r_kin[j].argmax()),
                        "reward_full": int(r_full[j].argmax()),
                        "oracle": int(ade[j].argmin())}
                for k in GATE_KS:
                    cand = order[j, :min(k, order.shape[1])]
                    pick[f"gate{k}"] = int(cand[r_kin[j][cand].argmax()])
                row = {"wi": int(wi), "eid": int(e_i), "has_lead": bool(b["has_lead"][j])}
                for rule, idx in pick.items():
                    row[f"{rule}__ade_m"] = float(ade[j, idx])
                    row[f"{rule}__peak_g"] = float(sc["peak_g"][j, idx])
                    for f in REPORT:
                        row[f"{rule}__{f}"] = float(sc[f][j, idx])
                    row[f"{rule}__idx"] = idx
                row["agrees_model__kin_only"] = float(pick["kin_only"] == pick["model"])
                # ---- the OFFSET-SHRINK sweep, same forward, no extra GPU ------- #
                for lam in LAMBDAS:
                    q = lam_sc[lam]
                    for f in ("envelope", "kamm_over", "off_reach", "infeasible"):
                        row[f"lam{lam}__fan_{f}"] = float(q[f][j].float().mean())
                    row[f"lam{lam}__fan_peak_g"] = float(q["peak_g"][j].mean())
                    row[f"lam{lam}__sel_envelope"] = float(q["envelope"][j, pick["model"]])
                    row[f"lam{lam}__sel_peak_g"] = float(q["peak_g"][j, pick["model"]])
                    row[f"lam{lam}__oracle_ade_m"] = float(lam_ade[lam][j].min())
                    row[f"lam{lam}__sel_ade_m"] = float(lam_ade[lam][j, pick["model"]])
                    row[f"lam{lam}__fan_spread_m"] = float(lam_spread[lam][j])
                for k in GATE_KS:
                    row[f"agrees_model__gate{k}"] = float(pick[f"gate{k}"] == pick["model"])
                rows.append(row)
            if (i // max(a.batch, 1)) % 12 == 0:
                print(f"  [{i + len(b['wis'])}/{len(wis)}]", flush=True)

    eids = np.array([r["eid"] for r in rows])
    lead_mask = np.array([r["has_lead"] for r in rows])
    LEAD_ONLY = set(FS.LEAD_ONLY)
    res = {"_tool": TOOL, "_tier": "T0 selection-quality readout on the emitted fan "
                                  "(never a driving claim)",
           "_evidence_class": "MEASURED (ours)", "ckpt": a.ckpt, "step": int(prov["step"]),
           "n_windows": len(rows), "n_episodes": int(len(set(eids.tolist()))),
           "n_lead_windows": int(lead_mask.sum()), "rules": rules,
           "kin_score": "feasibility + comfort (rewards.py) -- reads ONLY the candidate's "
                        "own waypoints: no lead, no obstacle track, no ego state",
           "estimator": "episode-cluster bootstrap; deltas vs `model` are PAIRED",
           "n_boot": a.n_boot, "seed": a.seed, "abs": {}, "paired_vs_model": {},
           "per_window": rows}
    metrics = ["ade_m", "peak_g"] + list(REPORT)
    for rule in rules:
        res["abs"][rule] = {}
        res["paired_vs_model"][rule] = {}
        for mname in metrics:
            k = f"{rule}__{mname}"
            sel = lead_mask if mname in LEAD_ONLY else np.ones(len(rows), bool)
            v = [r[k] for r, s in zip(rows, sel) if s]
            res["abs"][rule][mname] = boot(v, eids[sel], n_boot=a.n_boot, seed=a.seed)
            if rule != "model":
                mv = [r[f"model__{mname}"] for r, s in zip(rows, sel) if s]
                res["paired_vs_model"][rule][mname] = pboot(
                    mv, v, eids[sel], n_boot=a.n_boot, seed=a.seed)
        if rule != "model":
            res["abs"][rule]["agrees_model"] = boot(
                [r.get(f"agrees_model__{rule}", float("nan")) for r in rows], eids,
                n_boot=a.n_boot, seed=a.seed)

    res["lambda_sweep"] = {}
    for lam in LAMBDAS:
        res["lambda_sweep"][str(lam)] = {
            m: boot([r[f"lam{lam}__{m}"] for r in rows], eids, n_boot=a.n_boot, seed=a.seed)
            for m in ("fan_envelope", "fan_kamm_over", "fan_off_reach", "fan_infeasible",
                      "fan_peak_g", "sel_envelope", "sel_peak_g", "oracle_ade_m",
                      "sel_ade_m", "fan_spread_m")}

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    print(f"\n=== SELECTED-PATH SAFETY vs ADE, by selection rule "
          f"(n={len(rows)}w / {res['n_episodes']}ep, lead n={int(lead_mask.sum())}) ===")
    print(f"  {'rule':12s} {'ade_m':>8s} {'d_ade':>9s} {'sep':>4s} | {'envelope':>9s} "
          f"{'d_env':>9s} {'sep':>4s} | {'infeas':>8s} {'peak_g':>8s} {'agree':>6s}")
    for rule in rules:
        A = res["abs"][rule]
        P = res["paired_vs_model"].get(rule, {})
        da = P.get("ade_m", {})
        de = P.get("envelope", {})
        ag = A.get("agrees_model", {}).get("mean", float("nan")) if rule != "model" else 1.0
        print(f"  {rule:12s} {A['ade_m']['mean']:8.4f} "
              f"{da.get('delta', float('nan')):+9.4f} {str(da.get('separated', '-')):>4s} | "
              f"{A['envelope']['mean']:9.4f} {de.get('delta', float('nan')):+9.4f} "
              f"{str(de.get('separated', '-')):>4s} | {A['infeasible']['mean']:8.4f} "
              f"{A['peak_g']['mean']:8.4f} {ag:6.3f}")
    print("\n=== OFFSET-SHRINK SWEEP  path(lambda) = bank + lambda * offset ===")
    print(f"  {'lambda':>7s} {'fan_env':>8s} {'fan_peak_g':>11s} {'fan_offreach':>13s} "
          f"{'oracle_ade':>11s} {'sel_ade':>8s} {'sel_env':>8s} {'spread_m':>9s}")
    for lam in LAMBDAS:
        q = res["lambda_sweep"][str(lam)]
        print(f"  {lam:7.2f} {q['fan_envelope']['mean']:8.4f} {q['fan_peak_g']['mean']:11.4f} "
              f"{q['fan_off_reach']['mean']:13.4f} {q['oracle_ade_m']['mean']:11.4f} "
              f"{q['sel_ade_m']['mean']:8.4f} {q['sel_envelope']['mean']:8.4f} "
              f"{q['fan_spread_m']['mean']:9.4f}")
    print(f"[rerank] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
