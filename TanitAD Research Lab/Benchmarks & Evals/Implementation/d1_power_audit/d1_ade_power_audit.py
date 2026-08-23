"""D1 ADE@1s statistical-power audit (Benchmarks & Eval, independent-test role).

The gate ladder currently reads a D1 "regression" of ADE@1s 5.18 m (step-14k,
9 val eps) -> 11.52 m (step-21k, 4 val eps). PROJECT_STATE flags it "small-sample
noise ... directional not decision-grade" but nobody has QUANTIFIED the estimator's
sampling variance. This script measures it on real hardware ($0, local RTX 4060).

`run_d1` (stack/tanitad/eval/gates.py) reports ADE as the mean over ALL val
windows from a SINGLE fixed seed=0 episode split. Windows within a route are
strongly correlated, so the effective sample size is ~the number of val EPISODES,
and the reported number can swing on *which* routes land in val. We characterise
two variance sources that differ between the 14k (n=9) and 21k (n=4) reads:

  A. Val-set sampling variance: fix a well-fit probe (trained on a disjoint
     20-episode pool), then bootstrap the val set of size n in {4, 9, 20} and
     report the 95%% band of the pooled window-ADE.
  B. Shipped-estimator swing: call the ACTUAL `run_d1` over the pool, varying the
     split seed 0..N, at val_frac tuned to ~4 and ~8 val eps -> dispersion of the
     number the program actually reads off a gate run.

Pre-registered falsifier: if the 95%% CI half-width (A) or the seed spread (B) at
n=4 is >= (11.52-5.18)/2 = 3.17 m, the step-21k "regression" is INSIDE the
estimator's own noise band -> NOT decision-grade; the D1 read needs >= K val eps.

Honesty (P8): the local checkpoint is step-6500, not 14k/21k. This audit measures
a PROPERTY OF THE ESTIMATOR (between-route ADE dispersion at small n), which is
driven by route-difficulty heterogeneity, not the small checkpoint delta. We report
the coefficient of variation so the result transfers across the mean-ADE level.

Usage (local 4060):
  python d1_ade_power_audit.py \
    --ckpt C:/Users/Admin/tanitad-data/eval/ckpt_full.pt \
    --cache C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
    --out d1_ade_power_audit.json --pool 40 --stride 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.eval.gates import run_d1
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.fourbrain import WorldModel
from tanitad.models.readout import RidgeProbe

STEPS_1S = 10          # ADE@1s = 10 steps @ 10 Hz (matches evaluate_checkpoint)
ALPHA = 1e-3           # run_d1 default ridge alpha
GAP_14K_21K = 11.52 - 5.18   # 6.34 m reported swing; falsifier band = half = 3.17


def _ego(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


@torch.no_grad()
def collect(world, episodes, device, window, stride, batch=8):
    """Per-window encoder state + ego waypoint@1s + episode id (I3-clean)."""
    S, Y, E = [], [], []
    for ei, ep in enumerate(episodes):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - STEPS_1S, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            st = world.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            wp = _ego(ep.poses[last + STEPS_1S, :2] - ep.poses[last, :2],
                      ep.poses[last, 2])
            S.append(st); Y.append(wp); E.extend([ei] * len(ch))
    return torch.cat(S).float(), torch.cat(Y).float(), torch.tensor(E)


def per_episode_ade(probe, S, Y, E, val_eps):
    """ADE@1s of the fixed probe, per val episode (window-mean within episode)."""
    out = {}
    pred = probe.predict(S)                      # [N, 2]
    d = (pred - Y).norm(dim=-1)                   # [N]
    for e in val_eps:
        m = (E == e)
        out[int(e)] = (float(d[m].mean()), int(m.sum()))
    return out, d


def bootstrap_val(d_by_ep, n, reps, gen):
    """Bootstrap pooled window-ADE over n episodes drawn with replacement."""
    eps = list(d_by_ep.keys())
    idx = torch.tensor(eps)
    vals = []
    for _ in range(reps):
        pick = idx[torch.randint(len(idx), (n,), generator=gen)]
        # pool ALL windows of the drawn episodes (run_d1 pools windows, not ep-means)
        allw = torch.cat([d_by_ep[int(e)] for e in pick])
        vals.append(float(allw.mean()))
    v = torch.tensor(vals)
    lo, hi = torch.quantile(v, torch.tensor([0.025, 0.975]))
    return {"n_eps": n, "reps": reps, "mean": round(float(v.mean()), 3),
            "sd": round(float(v.std()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "ci95_halfwidth": round(float((hi - lo) / 2), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool", type=int, default=40)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--reps", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    eps_files = sorted(Path(args.cache).glob("ep_*.pt"))[:args.pool]
    episodes = [load_episode(str(p), mmap=True) for p in eps_files]
    print(f"loaded {len(episodes)} episodes", flush=True)

    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()

    with strict_numerics():
        S, Y, E = collect(world, episodes, device, world.predictor.cfg.window,
                          args.stride)
    n_ep = int(E.max()) + 1
    print(f"collected {S.shape[0]} windows over {n_ep} eps @ step{step} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- A. fixed-probe val bootstrap -------------------------------------- #
    train_eps = list(range(0, n_ep, 2))          # disjoint 50/50 (I3-clean)
    val_eps = list(range(1, n_ep, 2))
    tr = torch.isin(E, torch.tensor(train_eps))
    probe = RidgeProbe(alpha=ALPHA).fit(S[tr], Y[tr])
    ep_ade, d_all = per_episode_ade(probe, S, Y, E, val_eps)
    d_by_ep = {e: (E == e).nonzero().squeeze(1) for e in val_eps}
    d_by_ep = {e: (probe.predict(S[i]) - Y[i]).norm(dim=-1) for e, i in d_by_ep.items()}

    ep_means = torch.tensor([v[0] for v in ep_ade.values()])
    gen = torch.Generator().manual_seed(0)
    boot = {f"n{n}": bootstrap_val(d_by_ep, n, args.reps, gen)
            for n in (4, 9, 20)}

    # ---- B. shipped run_d1 seed sensitivity -------------------------------- #
    # val_frac 0.2 -> ~n_ep*0.2 val eps (~8 @ pool40); 0.1 -> ~4 val eps.
    def seed_spread(val_frac, seeds=50):
        ades = []
        for sd in range(seeds):
            r = run_d1(S, Y, [int(x) for x in E.tolist()], unit="camera",
                       alpha=ALPHA, val_frac=val_frac, seed=sd)
            ades.append(r.metrics["ade@1s"])
        a = torch.tensor(ades)
        return {"val_frac": val_frac, "seeds": seeds,
                "approx_val_eps": round(n_ep * val_frac),
                "mean": round(float(a.mean()), 3), "sd": round(float(a.std()), 3),
                "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                "range": round(float(a.max() - a.min()), 3)}

    shipped = {"val_frac_0.2": seed_spread(0.2), "val_frac_0.1": seed_spread(0.1)}

    # ---- verdict ----------------------------------------------------------- #
    hw4 = boot["n4"]["ci95_halfwidth"]
    hw9 = boot["n9"]["ci95_halfwidth"]
    swing4 = shipped["val_frac_0.1"]["range"]
    falsifier_band = round(GAP_14K_21K / 2, 3)
    verdict = ("NOT decision-grade: estimator noise band at n=4 "
               f">= falsifier band {falsifier_band} m" if
               max(hw4, swing4) >= falsifier_band else
               "regression exceeds estimator noise band at n=4")

    report = {
        "exp": "d1-ade-power-audit", "ckpt_step": step, "device": device,
        "n_windows": int(S.shape[0]), "n_episodes": n_ep,
        "stride": args.stride, "alpha": ALPHA, "steps_1s": STEPS_1S,
        "ref_gate_swing": {"step14k_ade": 5.18, "step21k_ade": 11.52,
                           "delta": round(GAP_14K_21K, 3),
                           "falsifier_band_halfdelta": falsifier_band,
                           "n_val_eps_14k": 9, "n_val_eps_21k": 4},
        "per_episode_ade_dispersion": {
            "val_eps": len(val_eps),
            "ep_ade_mean": round(float(ep_means.mean()), 3),
            "ep_ade_sd": round(float(ep_means.std()), 3),
            "ep_ade_min": round(float(ep_means.min()), 3),
            "ep_ade_max": round(float(ep_means.max()), 3),
            "coef_of_variation": round(float(ep_means.std() / ep_means.mean()), 3)},
        "A_fixed_probe_val_bootstrap": boot,
        "B_shipped_run_d1_seed_spread": shipped,
        "verdict": verdict,
        "wallclock_s": round(time.time() - t0, 1), "cost_usd": 0.0,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("D1_POWER_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
