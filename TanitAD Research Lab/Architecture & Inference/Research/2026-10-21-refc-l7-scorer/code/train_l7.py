"""train_l7.py -- train an L7 scorer (modes E / F; S once a scene-token bank exists) on a BANKED fan + its targets.

    python train_l7.py --fan-npz <bank>/fan_extras.npz --targets-npz <targets.npz> --mode F --seed 0 \
        --out-dir <run> [--epochs 30] [--val-frac 0.1]
    python train_l7.py --synthetic 256 --mode F --seed 0 --out-dir <run>       # machinery smoke, no real data

⛔ The bank must be a TRAIN-split capture (PREREG sec. 2). This script REFUSES a fan whose roll json names the eval139
windows file unless --i-know-this-is-eval is passed, and that flag exists only for a machinery smoke whose output is never
read as a metric.  Validation is an EPISODE-disjoint slice of the bank (seeded); early stopping reads only it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refc_scorer as RS  # noqa: E402

TARGET_KEYS = ("cost", "along", "cross", "comf", "ep", "nc", "ttc", "dac")


def synthetic_bank(n: int, seed: int, N: int = 149) -> tuple[dict, dict]:
    """A fan with a KNOWN best candidate per window: GT = a constant-speed arc; candidates = GT + structured offsets; the
    model's sel_score is a noisy function of the true cost, so a scorer that learns anything beats it."""
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(RS.SLOT_T_REFC)
    v = 5.0 + 10.0 * torch.rand(n, generator=g)
    k = (torch.rand(n, generator=g) - 0.5) * 0.04
    s = v[:, None] * t[None]
    gt = torch.stack([s, 0.5 * k[:, None] * s ** 2], -1)                 # constant speed, constant curvature (small angle)
    dv = (torch.rand(n, N, generator=g) - 0.5) * 0.6                     # speed scale error (longitudinal)
    dy = (torch.rand(n, N, generator=g) - 0.5) * 4.0                     # lateral offset growing with t
    fan = gt[:, None] * (1 + dv[..., None, None])
    fan[..., 1] = fan[..., 1] + dy[..., None] * (t / 6.0)[None, None]
    cost = (fan - gt[:, None]).norm(dim=-1).mean(-1)
    sel_score = -cost + torch.randn(n, N, generator=g) * cost.std() * 1.5
    refined = sel_score + torch.randn(n, N, generator=g) * 0.1
    eid = np.asarray([str(i // 8) for i in range(n)])
    fanz = {"anchor_traj": fan.numpy(), "sel_score": sel_score.numpy(), "refined_logits": refined.numpy(),
            "reach_keep": np.ones((n, N), np.float32), "v0": v.numpy(), "sel_idx": sel_score.argmax(1).numpy(),
            "clip_index": np.arange(n) // 8, "ws": np.arange(n) % 8}
    from tanitad.refs import refcv7_oracle as O
    o = O.oracle_subscores(fan, t, v, gt)
    al, cr = (fan - gt[:, None]).abs().unbind(-1)
    tg = {"cost": cost.numpy(), "along": al.mean(-1).numpy(), "cross": cr.mean(-1).numpy(), "eid": eid,
          "clip_index": fanz["clip_index"], "ws": fanz["ws"],
          **{k: o[k].numpy() for k in ("comf", "ep", "nc", "ttc", "dac")},
          **{k + "_mask": o[k + "_mask"].numpy() for k in ("comf", "ep", "nc", "ttc", "dac")}}
    return fanz, tg


def load_bank(fan_npz: str, targets_npz: str) -> tuple[dict, dict]:
    z = np.load(fan_npz, allow_pickle=True)
    t = np.load(targets_npz, allow_pickle=True)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(z["clip_index"], z["ws"]))}
    rows = np.asarray([key[(int(a), int(b))] for a, b in zip(t["clip_index"], t["ws"])])
    fanz = {k: np.asarray(z[k])[rows] for k in ("anchor_traj", "sel_score", "refined_logits", "reach_keep", "sel_idx",
                                                 "clip_index", "ws")}
    fanz["v0"] = np.asarray(t["v0"])
    tg = {k: np.asarray(t[k]) for k in t.files}
    return fanz, tg


def to_batch(fanz: dict, tg: dict, ii: np.ndarray, mode: str) -> tuple[dict, dict, torch.Tensor]:
    rec = {"anchor_traj": torch.from_numpy(np.asarray(fanz["anchor_traj"][ii], np.float32)),
           "v0": torch.from_numpy(np.asarray(fanz["v0"][ii], np.float32)),
           "sel_score": torch.from_numpy(np.asarray(fanz["sel_score"][ii], np.float32)),
           "refined_logits": torch.from_numpy(np.asarray(fanz["refined_logits"][ii], np.float32)),
           "reach_keep": torch.from_numpy(np.asarray(fanz["reach_keep"][ii]) > 0.5)}
    keep = rec["reach_keep"]
    inp = RS.assemble_inputs(rec, mode)
    t = {}
    for k in TARGET_KEYS:
        if k in tg:
            t[k] = torch.from_numpy(np.asarray(tg[k][ii], np.float32))
        if k + "_mask" in tg:
            t[k + "_mask"] = torch.from_numpy(np.asarray(tg[k + "_mask"][ii]).astype(bool))
    return inp, t, keep


def episode_split(eid: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(u, size=max(1, int(round(frac * len(u)))), replace=False).tolist())
    m = np.asarray([e in val for e in eid])
    return np.where(~m)[0], np.where(m)[0]


def evaluate(model, fanz, tg, ii, mode, bs=64) -> dict:
    model.eval()
    picks, costs, owns, orac = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(ii), bs):
            jj = ii[s:s + bs]
            inp, t, keep = to_batch(fanz, tg, jj, mode)
            out = model(**inp)
            p = RS.select(RS.rule_imit(out), keep)
            c = t["cost"]
            ar = torch.arange(len(jj))
            picks.append(p)
            costs.append(c[ar, p])
            owns.append(c[ar, torch.from_numpy(np.asarray(fanz["sel_idx"][jj]).astype(np.int64))])
            orac.append(c.masked_fill(~keep, float("inf")).min(1).values)
    model.train()
    cat = lambda x: torch.cat(x).double()
    return {"n": int(len(ii)), "pick_cost_m": round(float(cat(costs).mean()), 5),
            "own_pick_cost_m": round(float(cat(owns).mean()), 5), "oracle_cost_m": round(float(cat(orac).mean()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz")
    ap.add_argument("--targets-npz")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--mode", default="F", choices=("E", "F"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (smoke)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--i-know-this-is-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    if a.synthetic:
        fanz, tg = synthetic_bank(a.synthetic, a.seed)
        src = {"synthetic": a.synthetic}
    else:
        rj = os.path.join(os.path.dirname(a.fan_npz), "..", "fan_roll_s1.json")
        for cand in (rj, os.path.join(os.path.dirname(a.fan_npz), "..", "fan_roll_s0.json")):
            if os.path.exists(cand):
                w = json.load(open(cand, encoding="utf-8")).get("windows_json", "")
                if ("eval139" in w or "windows_s" in os.path.basename(w)) and not a.i_know_this_is_eval:
                    raise SystemExit(f"[train_l7] REFUSING: {a.fan_npz} is an EVAL capture ({w}); PREREG sec. 2/8")
        fanz, tg = load_bank(a.fan_npz, a.targets_npz)
        src = {"fan_npz": a.fan_npz, "targets_npz": a.targets_npz}
    eid = np.asarray(tg["eid"]).astype(str)
    tr, va = episode_split(eid, a.val_frac, 20261021)
    cfg = RS.L7Config(mode=a.mode)
    model = RS.L7Scorer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    log = {"args": vars(a), "source": src, "params": RS.n_params(model), "n_train": int(len(tr)), "n_val": int(len(va)),
           "epochs": []}
    best, bad, step = float("inf"), 0, 0
    rng = np.random.default_rng(a.seed)
    t_train = 0.0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        t0 = time.time()
        losses = []
        for s in range(0, len(perm), a.batch):
            jj = perm[s:s + a.batch]
            inp, t, keep = to_batch(fanz, tg, jj, a.mode)
            out = model(**inp)
            loss, lg = RS.pai_loss(out, t, keep, a.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(lg)
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        dt = time.time() - t0
        t_train += dt
        vr = evaluate(model, fanz, tg, va, a.mode)
        trr = evaluate(model, fanz, tg, tr[:512], a.mode)
        mean = {k: round(float(np.nanmean([d[k] for d in losses if k in d])), 5) for k in losses[0]}
        log["epochs"].append({"epoch": ep, "steps": step, "train_loss": mean, "val": vr, "train_subset": trr,
                              "epoch_s": round(dt, 2), "s_per_window": round(dt / max(len(perm), 1), 6)})
        print("ZZEP " + json.dumps(log["epochs"][-1]), flush=True)
        score = vr["pick_cost_m"]
        if score < best - 1e-6:
            best, bad = score, 0
            torch.save({"cfg": vars(cfg), "state_dict": model.state_dict(), "epoch": ep}, os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                break
        if a.max_steps and step >= a.max_steps:
            break
    log["train_wall_s"] = round(t_train, 2)
    with open(os.path.join(a.out_dir, "train_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print("ZZL7TRAIN " + json.dumps({"best_val_pick_cost_m": best, "epochs": len(log["epochs"]),
                                     "params": log["params"], "train_wall_s": log["train_wall_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
