"""Driving-capability diagnostic for the TanitAD-4B-M world model.

Answers the pre-registered questions in
``Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md`` with REAL measurements on
the training val set (NO guessed numbers). Every section is a real computation
on encoder states / GT poses; anything that cannot be computed is emitted as
null with a note.

Sections (framework §A/§B/§C/instruments):
  1. Trivial baselines (NO model) — constant-velocity, go-straight,
     constant-yaw-rate kinematics from pose history. The interpretability floor:
     does the model even beat trivial predictors?
  2. Model decode ladder — ridge(alpha in {1,10,100}) + 2-layer MLP + the oracle
     in-distribution ceiling (fit==eval). Held-out vs oracle-ceiling routes the
     representation-vs-readout question (§B).
  3. Error localization — stratify val windows by future-2s path curvature
     (straight/gentle/sharp), ego-speed tertile, and corpus; report best-probe
     ADE vs constant-velocity ADE per stratum. "Cannot drive at all" vs
     "handles straights, fails curves" (§C).
  4. Instrument sanity — I1 oracle-probe fit R^2 (~1), I2 batch consistency
     (<1e-4), episode counts, per-stratum window counts (<30 flagged).

Conventions are reused verbatim from the proven code (no reinvention):
  - ``_ego`` ego-frame rotation from ``scripts/d1_probe_capacity.py``.
  - ``split_by_episode`` route/episode parity from ``tanitad.eval.gates``.
  - ``RidgeProbe`` (double, centered, closed form) from ``tanitad.models.readout``.
  - Frame preprocessing (uint8 -> float/255), checkpoint + WorldModel load, and
    the ``encode_window(...)[:, -1]`` last-frame state from
    ``scripts/evaluate_checkpoint.py`` / ``d1_probe_capacity.py``.
  - Everything runs under ``strict_numerics()`` in fp32.

METRIC DEFINITIONS (documented so nothing is ambiguous):
  waypoints are at steps (5,10,15,20) = (0.5,1,1.5,2)s @10Hz, ego frame.
  de@Ts  = mean over windows of ||pred(T) - gt(T)|| (point/final error at T;
           == FDE@Ts for the sub-trajectory ending at T).
  ade@Ts = mean over windows AND over the waypoints with step<=T of the point
           error (gates ``ade_fde`` sub-trajectory convention).
  ade_0_2s = mean over all 4 waypoints == the key reported by ``run_d1``
           (its metric named "ade@1s" is really this 4-waypoint mean).

Usage (pod1):
  python scripts/driving_diagnostic.py \
      --ckpt /workspace/ckpt27k_flagship.pt \
      --cache-dirs /workspace/data/comma2k19/_epcache \
                   /workspace/data/physicalai/_epcache \
      --out /workspace/experiments/driving_diagnostic.json --episodes 40
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from tanitad.data.mixing import load_episode
from tanitad.eval.gates import split_by_episode
from tanitad.instruments.checks import i2_batch_consistency
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.readout import RidgeProbe

WP_STEPS = (5, 10, 15, 20)               # 0.5/1/1.5/2 s @10Hz
K_MAX = max(WP_STEPS)
BASELINES = ("constant_velocity", "go_straight", "constant_yaw_rate")
CURV_STRAIGHT_DEG = 5.0                  # |net heading change@2s| < 5 deg
CURV_GENTLE_DEG = 20.0                   # 5-20 deg gentle; >20 sharp
I1_FLOOR = 0.9
I2_TOL = 1e-4
MIN_STRATUM_N = 30                       # flag strata below this as low-confidence


# --------------------------------------------------------------------------- #
# Frame convention — copied EXACTLY from scripts/d1_probe_capacity.py          #
# --------------------------------------------------------------------------- #
def _ego(dxy, yaw):
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    return torch.stack([dxy[..., 0] * c - dxy[..., 1] * s,
                        dxy[..., 0] * s + dxy[..., 1] * c], dim=-1)


def _wrap(a: Tensor) -> Tensor:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# GT + trivial baselines — pure kinematics from pose history (NO model)        #
# --------------------------------------------------------------------------- #
def gt_ego_waypoints(poses: Tensor, last: Tensor, wp_steps=WP_STEPS) -> Tensor:
    """GT ego-frame displacement to each horizon. poses [T,4], last [b] -> [b,H,2]."""
    p0 = poses[last, :2]
    yaw0 = poses[last, 2]
    return torch.stack([_ego(poses[last + k, :2] - p0, yaw0) for k in wp_steps],
                       dim=1)


def baseline_waypoints(poses: Tensor, last: Tensor, wp_steps=WP_STEPS
                       ) -> dict[str, Tensor]:
    """The three trivial predictors, ego frame, [b,H,2] each.

    All derive purely from the pose history at ``last`` (and ``last-1`` for the
    one-step velocity / yaw-rate). ``last`` must be >= 1.
      constant_velocity: extrapolate the last-step velocity vector linearly.
      go_straight:       zero lateral, keep current speed along ego heading.
      constant_yaw_rate: last yaw-rate + speed, forward-Euler circular arc.
    """
    p0, pm1 = poses[last, :2], poses[last - 1, :2]
    yaw0, yawm1 = poses[last, 2], poses[last - 1, 2]
    v_world = p0 - pm1                              # per-step world displacement
    speed = v_world.norm(dim=-1)                    # per-step speed magnitude
    omega = _wrap(yaw0 - yawm1)                      # per-step yaw rate
    ego_v = _ego(v_world, yaw0)                      # last velocity in ego frame
    zeros = torch.zeros_like(speed)
    cv, gs, cyr = [], [], []
    for k in wp_steps:
        cv.append(ego_v * k)
        gs.append(torch.stack([speed * k, zeros], dim=-1))
        js = torch.arange(k, dtype=poses.dtype, device=poses.device)
        ang = js[None, :] * omega[:, None]           # [b,k] heading per sub-step
        cyr.append(torch.stack([speed * ang.cos().sum(dim=1),
                                speed * ang.sin().sum(dim=1)], dim=-1))
    return {"constant_velocity": torch.stack(cv, dim=1),
            "go_straight": torch.stack(gs, dim=1),
            "constant_yaw_rate": torch.stack(cyr, dim=1)}


def net_heading_change_deg(poses: Tensor, last: Tensor, horizon: int = K_MAX
                           ) -> Tensor:
    """|net heading change| over the future ``horizon`` steps, degrees. [b]."""
    return _wrap(poses[last + horizon, 2] - poses[last, 2]).abs() * (180.0 / math.pi)


def curvature_bucket(deg: float) -> str:
    if deg < CURV_STRAIGHT_DEG:
        return "straight"
    if deg <= CURV_GENTLE_DEG:
        return "gentle"
    return "sharp"


# --------------------------------------------------------------------------- #
# Metric helpers                                                               #
# --------------------------------------------------------------------------- #
def scalar_metrics(de: Tensor) -> dict[str, float]:
    """de [n,H] point errors per waypoint -> ade@/de@ scalars (means over n)."""
    out: dict[str, float] = {}
    for i, k in enumerate(WP_STEPS):
        t = k / 10.0
        out[f"de@{t:g}s"] = float(de[:, i].mean())
        out[f"ade@{t:g}s"] = float(de[:, :i + 1].mean())
    out["ade_0_2s"] = float(de.mean())              # == gates run_d1 "ade@1s"
    return out


def mean_ci(vals: list[float]) -> dict:
    """mean, 95% CI (route-resampled protocol, matching gates.run_d1)."""
    n = len(vals)
    m = sum(vals) / n
    std = (sum((v - m) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
    return {"mean": round(m, 4), "ci95": round(1.96 * std / n ** 0.5, 4),
            "std": round(std, 4), "n_splits": n,
            "per_split": [round(v, 4) for v in vals]}


def agg_metric_dicts(dicts: list[dict]) -> dict:
    """List of per-split scalar dicts -> {metric: mean_ci over splits}."""
    keys = dicts[0].keys()
    return {k: mean_ci([d[k] for d in dicts]) for k in keys}


def _r2(pred: Tensor, tgt: Tensor) -> float:
    ss = (pred - tgt).pow(2).sum()
    tot = (tgt - tgt.mean(0)).pow(2).sum().clamp_min(1e-12)
    return float(1.0 - ss / tot)


def fit_predict(kind: str, alpha: float, Xtr: Tensor, Ytr: Tensor, Xev: Tensor,
                mlp_epochs: int) -> tuple[Tensor, float]:
    """Fit a probe on (Xtr,Ytr[n,8]) and predict Xev -> ([nev,4,2], fit R^2).

    ridge: the frozen RidgeProbe (double, centered) from tanitad.models.readout.
    mlp:   the 2-layer readout from d1_probe_capacity.fit_eval, widened to 8 out.
    """
    if kind == "ridge":
        pr = RidgeProbe(alpha=alpha).fit(Xtr, Ytr)
        return pr.predict(Xev).reshape(-1, len(WP_STEPS), 2), pr.r2(Xtr, Ytr)
    torch.manual_seed(0)
    net = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], 256),
                        nn.GELU(), nn.Linear(256, 2 * len(WP_STEPS)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(mlp_epochs):
        perm = torch.randperm(Xtr.shape[0])
        for j in range(0, len(perm), 512):
            b = perm[j:j + 512]
            loss = (net(Xtr[b]) - Ytr[b]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xev).reshape(-1, len(WP_STEPS), 2)
        fit_r2 = _r2(net(Xtr), Ytr)
    return pred, fit_r2


def de_of(pred_wp: Tensor, gt_wp: Tensor) -> Tensor:
    """[n,H,2] pred & gt -> [n,H] per-waypoint Euclidean point error."""
    return (pred_wp - gt_wp).norm(dim=-1)


# --------------------------------------------------------------------------- #
# Collection — encode every window ONCE; baselines + meta from poses           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect(world, episodes, corpora, device, window, stride, batch):
    S, GT = [], []
    BP = {n: [] for n in BASELINES}
    EID: list[int] = []
    COR: list[str] = []
    SPD, HDG = [], []
    for ep, corp in zip(episodes, corpora):
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames
        T = fr.shape[0]
        starts = list(range(0, T - window - K_MAX, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            fw = torch.stack([fr[t:t + window] for t in ch]).to(device)
            st = world.encode_window(fw)[:, -1].cpu()
            last = torch.tensor([t + window - 1 for t in ch])
            S.append(st)
            GT.append(gt_ego_waypoints(ep.poses, last))
            bp = baseline_waypoints(ep.poses, last)
            for n in BASELINES:
                BP[n].append(bp[n])
            EID.extend([ep.episode_id] * len(ch))
            COR.extend([corp] * len(ch))
            SPD.append(ep.poses[last, 3])
            HDG.append(net_heading_change_deg(ep.poses, last))
    return {"states": torch.cat(S).float(), "gt": torch.cat(GT).float(),
            "base": {n: torch.cat(BP[n]).float() for n in BASELINES},
            "eid": EID, "corpus": COR,
            "speed": torch.cat(SPD).float(), "head_deg": torch.cat(HDG).float()}


# --------------------------------------------------------------------------- #
# Sections                                                                     #
# --------------------------------------------------------------------------- #
def section1_baselines(data, splits) -> dict:
    """Trivial-baseline ADE/FDE per horizon, mean+-CI over the route splits."""
    gt = data["gt"]
    de_all = {n: de_of(data["base"][n], gt) for n in BASELINES}
    out = {}
    for n in BASELINES:
        per_split = [scalar_metrics(de_all[n][va]) for _tr, va in splits]
        out[n] = agg_metric_dicts(per_split)
    return out


def section2_decode_ladder(data, splits, mlp_epochs) -> dict:
    """Ridge(alpha)/MLP held-out vs oracle in-distribution ceiling."""
    states, gt = data["states"], data["gt"]
    flat = lambda idx: gt[idx].reshape(len(idx), 2 * len(WP_STEPS))
    probes = [("ridge", 1.0, "ridge_a1"), ("ridge", 10.0, "ridge_a10"),
              ("ridge", 100.0, "ridge_a100"), ("mlp", 1.0, "mlp")]
    held, oracle = {}, {}
    for kind, alpha, key in probes:
        ho, orc, ho_r2, orc_r2 = [], [], [], []
        for tr, va in splits:
            pred, fr = fit_predict(kind, alpha, states[tr], flat(tr),
                                   states[va], mlp_epochs)
            ho.append(scalar_metrics(de_of(pred, gt[va])))
            ho_r2.append(fr)
            predo, fro = fit_predict(kind, alpha, states[va], flat(va),
                                     states[va], mlp_epochs)
            orc.append(scalar_metrics(de_of(predo, gt[va])))
            orc_r2.append(fro)
        held[key] = agg_metric_dicts(ho)
        held[key]["fit_r2_train"] = mean_ci(ho_r2)
        oracle[key] = agg_metric_dicts(orc)
        oracle[key]["fit_r2"] = mean_ci(orc_r2)
        print(f"[ladder] {key}: held-out ade_0_2s={held[key]['ade_0_2s']['mean']} "
              f"oracle={oracle[key]['ade_0_2s']['mean']}", flush=True)
    best_key = min(held, key=lambda k: held[k]["ade_0_2s"]["mean"])
    return {"held_out": held, "oracle_ceiling": oracle,
            "best_probe_by_heldout_ade_0_2s": best_key,
            "model_trajectory_head": None,
            "model_trajectory_head_note":
                "WorldModel exposes no native waypoint/trajectory head; "
                "evaluate_checkpoint.py decodes via frozen RidgeProbe (D1). "
                "Arm (c) is therefore N/A for this checkpoint."}


def _strat(labels, model_de, cv_de) -> dict:
    out = {}
    for lab in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == lab]
        md, cd = model_de[idx], cv_de[idx]
        out[lab] = {
            "model_ade@1s": round(float(md[:, :2].mean()), 4),
            "cv_ade@1s": round(float(cd[:, :2].mean()), 4),
            "model_de@1s": round(float(md[:, 1].mean()), 4),
            "cv_de@1s": round(float(cd[:, 1].mean()), 4),
            "model_ade@2s": round(float(md.mean()), 4),
            "cv_ade@2s": round(float(cd.mean()), 4),
            "n": len(idx),
            "low_confidence": bool(len(idx) < MIN_STRATUM_N),
        }
    return out


def section3_localization(data, splits, best_key, mlp_epochs) -> dict:
    """Stratify held-out (seed-0 split) errors by curvature / speed / corpus."""
    states, gt = data["states"], data["gt"]
    tr0, va0 = splits[0]
    kind, alpha = ("mlp", 1.0) if best_key == "mlp" else \
        ("ridge", {"ridge_a1": 1.0, "ridge_a10": 10.0, "ridge_a100": 100.0}[best_key])
    pred, _ = fit_predict(kind, alpha, states[tr0],
                          gt[tr0].reshape(len(tr0), 2 * len(WP_STEPS)),
                          states[va0], mlp_epochs)
    model_de = de_of(pred, gt[va0])
    cv_de = de_of(data["base"]["constant_velocity"][va0], gt[va0])

    # speed tertiles over the WHOLE window population (stable thresholds)
    q = torch.quantile(data["speed"], torch.tensor([1 / 3, 2 / 3]))
    t1, t2 = float(q[0]), float(q[1])

    def sbucket(s: float) -> str:
        return "low" if s < t1 else ("high" if s >= t2 else "med")

    va_head = data["head_deg"][va0]
    va_speed = data["speed"][va0]
    curv_lab = [curvature_bucket(float(h)) for h in va_head]
    spd_lab = [sbucket(float(s)) for s in va_speed]
    cor_lab = [data["corpus"][i] for i in va0]
    return {
        "split_seed": 0,
        "n_val_windows": len(va0),
        "best_probe": best_key,
        "speed_tertile_thresholds_mps": [round(t1, 3), round(t2, 3)],
        "by_curvature": _strat(curv_lab, model_de, cv_de),
        "by_speed": _strat(spd_lab, model_de, cv_de),
        "by_corpus": _strat(cor_lab, model_de, cv_de),
        "curvature_definition": {"straight_deg": f"<{CURV_STRAIGHT_DEG}",
                                 "gentle_deg": f"{CURV_STRAIGHT_DEG}-{CURV_GENTLE_DEG}",
                                 "sharp_deg": f">{CURV_GENTLE_DEG}",
                                 "quantity": "|yaw[t+20]-yaw[t]| (net heading change @2s)"},
    }


def section4_instruments(world, episodes, data, splits, ladder, device) -> dict:
    # I1: the oracle-probe (fit==eval ridge a1) fit R^2 -> must be ~1.
    i1 = ladder["oracle_ceiling"]["ridge_a1"]["fit_r2"]
    # I2: batch-1 vs batched encode consistency (< 1e-4).
    fr = episodes[0].frames[:16]
    fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr
    i2_ok, i2_rel = i2_batch_consistency(lambda x: world.encode(x), fr.to(device),
                                         batch_size=8, tol=I2_TOL)
    eps_per_split = [{"train_episodes": len({data["eid"][i] for i in tr}),
                      "val_episodes": len({data["eid"][i] for i in va}),
                      "val_windows": len(va)} for tr, va in splits]
    loc = data.get("_localization", {})
    strata_n = {}
    low_conf = []
    for dim in ("by_curvature", "by_speed", "by_corpus"):
        for lab, v in loc.get(dim, {}).items():
            strata_n[f"{dim}:{lab}"] = v["n"]
            if v["n"] < MIN_STRATUM_N:
                low_conf.append(f"{dim}:{lab}({v['n']})")
    return {
        "I1_oracle_probe_fit_r2": i1,
        "I1_pass": bool(i1["mean"] >= I1_FLOOR),
        "I1_floor": I1_FLOOR,
        "I2_batch_consistency_max_rel": round(i2_rel, 8),
        "I2_pass": bool(i2_ok),
        "I2_tol": I2_TOL,
        "episodes_per_split": eps_per_split,
        "windows_per_stratum": strata_n,
        "low_confidence_strata": low_conf,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40,
                    help="val episodes per cache dir")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=60)
    ap.add_argument("--git-hash", default="unknown")
    ap.add_argument("--config",
                    choices=["base250cam", "flagship4b", "flagship4b_reduced"],
                    default="base250cam",
                    help="architecture to instantiate for the ckpt (must match "
                         "the training config so load_state_dict succeeds)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.config import (base250cam_config, flagship4b_config,
                                flagship4b_reduced_config)
    from tanitad.models.fourbrain import WorldModel
    _cfg_fn = {"base250cam": base250cam_config, "flagship4b": flagship4b_config,
               "flagship4b_reduced": flagship4b_reduced_config}[args.config]

    def corpus_of(cd: str) -> str:
        low = cd.lower()
        if "comma" in low:
            return "comma2k19"
        if "physicalai" in low or "physical" in low:
            return "physicalai"
        return Path(cd).name

    episodes, corpora = [], []
    for cd in args.cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if not val_dirs:
            print(f"[diag] WARNING no *val* dir under {cd}", flush=True)
            continue
        files = sorted(val_dirs[-1].glob("ep_*.pt"))[:args.episodes]
        for p in files:
            episodes.append(load_episode(str(p), mmap=True))
            corpora.append(corpus_of(cd))
    assert episodes, "no val episodes loaded"

    world = WorldModel(_cfg_fn())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    window = world.predictor.cfg.window
    print(f"[diag] {len(episodes)} val episodes, ckpt step {step}, window {window}, "
          f"device {device}", flush=True)

    with strict_numerics():
        data = collect(world, episodes, corpora, device, window,
                       args.stride, args.batch)
        n = data["states"].shape[0]
        print(f"[diag] collected {n} windows, state_dim {data['states'].shape[1]}",
              flush=True)
        splits = [split_by_episode(data["eid"], args.val_frac, s)
                  for s in range(args.seed, args.seed + args.n_splits)]

        sec1 = section1_baselines(data, splits)
        sec2 = section2_decode_ladder(data, splits, args.mlp_epochs)
        sec3 = section3_localization(data, splits,
                                     sec2["best_probe_by_heldout_ade_0_2s"],
                                     args.mlp_epochs)
        data["_localization"] = sec3
        sec4 = section4_instruments(world, episodes, data, splits, sec2, device)

    corpus_counts: dict[str, dict] = {}
    for c in data["corpus"]:
        corpus_counts.setdefault(c, {"windows": 0})
        corpus_counts[c]["windows"] += 1
    for c, eps in zip(corpora, episodes):
        corpus_counts.setdefault(c, {"windows": 0}).setdefault("episodes", 0)
        corpus_counts[c]["episodes"] = corpus_counts[c].get("episodes", 0) + 1

    report = {
        "exp": "driving-capability-diagnostic",
        "ckpt": args.ckpt, "step": step, "git_hash": args.git_hash,
        "config": {"episodes_per_dir": args.episodes, "stride": args.stride,
                   "window": window, "n_splits": args.n_splits,
                   "val_frac": args.val_frac, "seed": args.seed,
                   "mlp_epochs": args.mlp_epochs, "hz": 10,
                   "waypoint_steps": list(WP_STEPS), "fp32": True,
                   "strict_numerics": True},
        "corpora": corpus_counts, "n_windows_total": n,
        "definitions": {
            "de@Ts": "mean over windows of ||pred(T)-gt(T)|| (point/FDE error at T)",
            "ade@Ts": "mean over windows and waypoints<=T of point error (gates ade_fde convention)",
            "ade_0_2s": "mean over all 4 waypoints; == run_d1 metric keyed 'ade@1s'",
            "frame": "ego frame at departure yaw (scripts/d1_probe_capacity._ego)",
        },
        "section1_trivial_baselines": sec1,
        "section2_decode_ladder": sec2,
        "section3_error_localization": sec3,
        "section4_instrument_sanity": sec4,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    # compact console summary
    print("\n=== DRIVING DIAGNOSTIC SUMMARY ===", flush=True)
    for n_ in BASELINES:
        print(f"  baseline {n_:18s} ade@1s={sec1[n_]['ade@1s']['mean']:.3f} "
              f"ade@2s={sec1[n_]['ade@2s']['mean']:.3f}", flush=True)
    for key, v in sec2["held_out"].items():
        print(f"  probe {key:12s} held-out ade_0_2s={v['ade_0_2s']['mean']:.3f} "
              f"oracle={sec2['oracle_ceiling'][key]['ade_0_2s']['mean']:.3f}",
              flush=True)
    print(f"  I1 fit R^2={sec4['I1_oracle_probe_fit_r2']['mean']:.4f} "
          f"(pass={sec4['I1_pass']}) | I2 rel={sec4['I2_batch_consistency_max_rel']:.2e} "
          f"(pass={sec4['I2_pass']})", flush=True)
    print(f"[diag] report -> {args.out}", flush=True)
    print("DRIVING_DIAGNOSTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
