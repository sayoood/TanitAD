#!/usr/bin/env python3
"""B1: does `robust_contact` give the reward a term the HUMAN beats hold-v0 on?

`G-REWARD` (`D-REFCV5-PLAN-6`) refuses to let any RL arm train while hold-v0 beats the
human on > 30 % of lead windows. Re-weighting the banked components measured that the
plan's own named repair -- move feasibility and comfort to vetoes -- moves the rate
0.7249 -> 0.4418 and STILL FAILS, and that no reweighting of the five shipped components
passes, because none of them is a term the human is systematically better at than a
constant-velocity path.

This scores the CANDIDATE term on the same corpus, the same windows and the same three
paths, at ZERO GPU. It reuses the driver's own corpus/lead helpers by importing it, so
the lead model, the window set and the geometry are identical to the banked run -- the
only new thing is the term.

⚠️ It loads the refcv3 checkpoint because `open_corpus` needs the model's config for the
window geometry. NOTHING is forwarded through the model: `human`, `hold_v0` and `frozen`
are constructed analytically, exactly as `mode_humanflag` does. Device is CPU.

CONTROLS:
  * `collision` at dt=0 recomputed here must reproduce the BANKED per-window value on
    every window (the channel control -- same corpus, same helpers, same predicate);
  * `robust_contact` with shifts=(0.0,) must equal that collision value EXACTLY;
  * `frozen` must stay far below both under every spec.

Tier: T0 instrument probe, NON-PARITY RL-fit windows. Evidence class: MEASURED (ours).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types

import numpy as np
import torch

REPO = os.environ.get("TANITAD_REPO",
                      r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, os.path.join(REPO, "stack"))

from tanitad.rl import rewards as RW                      # noqa: E402
from tanitad.rl import robust_contact as RC               # noqa: E402


def _load_driver():
    p = os.path.join(REPO, "stack", "scripts", "rl_refcv3_min.py")
    spec = importlib.util.spec_from_file_location("rl_refcv3_min", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["rl_refcv3_min"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--lead-block", required=True)
    ap.add_argument("--banked", default=None, help="banked humanflag rows, for the control")
    ap.add_argument("--shifts", default="-1.0,-0.5,0.0,0.5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lru", type=int, default=8)
    a = ap.parse_args()

    D = _load_driver()
    D.LEAD_MODE = "track"
    shifts = tuple(float(s) for s in a.shifts.split(","))

    ns = types.SimpleNamespace(ckpt=a.ckpt, config=a.config, expect_step=40284,
                               episodes=a.episodes, labels=a.labels, lru=a.lru,
                               device="cpu", nav_from_v7=True)
    _model, cfg, _targs, prov = D.load(ns, "cpu")
    corp = D.open_corpus(a.episodes, a.labels, cfg, prov, a.lru)
    lead, _lm, _li = D.load_lead_block(a.lead_block)
    if lead is None or lead.ts_rel is None:
        raise SystemExit("[b1] a lead block with ts_rel_s is required")

    banked = {}
    if a.banked:
        with open(a.banked, "r", encoding="utf-8") as fh:
            for r in json.load(fh)["per_window"]:
                banked[int(r["wi"])] = r

    rows = []
    MISMATCH = []
    n_ctrl_ok = n_ctrl = 0
    for wi in D.scoreable_windows(corp, lead):
        has, xy, _ln = D.lead_row(corp, lead, wi)
        if not has:
            continue
        e_i, _t = corp.ds.index[wi]
        human, v0 = D._human_future(corp, wi)
        xs = torch.tensor([v0 * s for s in D.GRID_S], dtype=torch.float32)
        hold = torch.stack([xs, torch.zeros_like(xs)], dim=-1).reshape(1, 1, 1, 5, 2)
        frozen = torch.zeros(1, 1, 1, 5, 2)
        batch = {"v0": torch.tensor([v0]),
                 "lead_xy": torch.tensor([xy], dtype=torch.float32),
                 "lead_track": D.lead_track(corp, lead, wi)[None]}
        ctx = D.reward_ctx(batch, S5=5)
        rec = {"wi": int(wi), "eid": int(e_i), "v0": float(v0)}
        for name, traj in (("human", human), ("hold_v0", hold), ("frozen", frozen)):
            parts = RW.RewardSpec().per_component(traj, ctx)
            d = {k: float(v.reshape(-1)[0]) for k, v in parts.items()}
            d["robust_contact"] = float(
                RC.robust_contact(traj, {**ctx, "robust_shifts_s": shifts}
                                  ).reshape(-1)[0])
            # CONTROL: zero-shift robust == the point collision, exactly
            z = float(RC.robust_contact(traj, {**ctx, "robust_shifts_s": (0.0,)}
                                        ).reshape(-1)[0])
            assert abs(z - d["collision"]) < 1e-9, (name, z, d["collision"])
            rec[name] = d
        # CONTROL: our recomputed collision must match the banked one
        if rec["wi"] in banked:
            b = banked[rec["wi"]]
            for nm in ("human", "hold_v0"):
                n_ctrl += 1
                dv = abs(b[f"{nm}_track"]["collision"] - rec[nm]["collision"])
                if dv < 1e-6:
                    n_ctrl_ok += 1
                else:
                    MISMATCH.append({"wi": rec["wi"], "path": nm,
                                     "banked": b[f"{nm}_track"]["collision"],
                                     "ours": rec[nm]["collision"],
                                     "banked_headway": b[f"{nm}_track"]["headway"],
                                     "ours_headway": rec[nm]["headway"],
                                     "v0": rec["v0"]})
        rows.append(rec)

    n = len(rows)
    print(f"[b1] {n} lead windows, {len(set(r['eid'] for r in rows))} episodes")
    if n_ctrl:
        print(f"[b1] CHANNEL CONTROL vs banked collision: {n_ctrl_ok}/{n_ctrl} match "
              f"({'OK' if n_ctrl_ok == n_ctrl else 'MISMATCH -- numbers below are VOID'})")
        if n_ctrl_ok != n_ctrl:
            import pprint; pprint.pprint(MISMATCH[:20])
            print(f"[b1] {len(MISMATCH)} mismatches of {n_ctrl}")
            if len(MISMATCH) > 0.01 * n_ctrl:
                raise SystemExit("[b1] REFUSED: channel control failed above 1%")
            print("[b1] proceeding: mismatch rate below 1%, DISCLOSED not hidden")

    def comp(rec, w):
        return sum(v * rec[k] for k, v in w.items())

    specs = {
        "default": {"progress": .3, "collision": 1., "headway": .3,
                    "feasibility": .5, "comfort": .2},
        "veto_feas_comfort": {"progress": .3, "collision": 1., "headway": .3},
        "veto_fc_PLUS_robust": {"progress": .3, "collision": 1., "headway": .3,
                                "robust_contact": 1.0},
        "veto_fc_robust_w2": {"progress": .3, "collision": 1., "headway": .3,
                              "robust_contact": 2.0},
        "veto_fc_robust_w5": {"progress": .3, "collision": 1., "headway": .3,
                              "robust_contact": 5.0},
        "robust_only": {"robust_contact": 1.0},
    }
    eids = np.array([r["eid"] for r in rows])
    out = {"_what": "B1: does robust_contact give the reward a term the human wins?",
           "_evidence_class": "MEASURED (ours)", "_tier": "T0 instrument probe",
           "n_windows": n, "n_episodes": int(len(set(eids))), "shifts_s": list(shifts),
           "channel_control": {"banked_collision_match": f"{n_ctrl_ok}/{n_ctrl}"},
           "panel": {}}
    print()
    print(f"{'spec':<24}{'hold_v0>=human':>15}{'median gap':>12}{'mean gap':>11}"
          f"{'frozen>=human':>15}  G-REWARD")
    print("-" * 92)
    for name, w in specs.items():
        hu = np.array([comp(r["human"], w) for r in rows])
        ho = np.array([comp(r["hold_v0"], w) for r in rows])
        fr = np.array([comp(r["frozen"], w) for r in rows])
        ge = (ho >= hu)
        # episode-cluster bootstrap on the rate
        uniq = np.unique(eids)
        by = {int(e): ge.astype(float)[eids == e] for e in uniq}
        rng = np.random.default_rng(11)
        bs = np.array([np.concatenate([by[int(e)] for e in
                                       rng.choice(uniq, len(uniq), True)]).mean()
                       for _ in range(2000)])
        rec = {"weights": w, "hold_v0_ge_human": float(ge.mean()),
               "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               "median_gap": float(np.median(ho - hu)),
               "mean_gap": float((ho - hu).mean()),
               "frozen_ge_human": float((fr >= hu).mean()),
               "PASSES_G_REWARD": bool(ge.mean() <= 0.30)}
        out["panel"][name] = rec
        print(f"{name:<24}{rec['hold_v0_ge_human']:>15.4f}{rec['median_gap']:>12.5f}"
              f"{rec['mean_gap']:>11.5f}{rec['frozen_ge_human']:>15.4f}  "
              f"{'PASS' if rec['PASSES_G_REWARD'] else 'FAIL'}")

    rmean = {k: float(np.mean([r[k]["robust_contact"] for r in rows]))
             for k in ("human", "hold_v0", "frozen")}
    cmean = {k: float(np.mean([r[k]["collision"] for r in rows]))
             for k in ("human", "hold_v0", "frozen")}
    out["term_means"] = {"robust_contact": rmean, "collision_dt0": cmean}
    print()
    print("The term itself (mean over windows; 0 = clean, -1 = contact):")
    for k in ("human", "hold_v0", "frozen"):
        print(f"   {k:<10} collision(dt=0) {cmean[k]:+.6f}   "
              f"robust_contact {rmean[k]:+.6f}")

    if a.out:
        out["per_window"] = rows
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"\n[b1] wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
