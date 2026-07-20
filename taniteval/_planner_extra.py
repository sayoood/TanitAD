"""Planner-native extras for REF-B v2 (no grounded rollout head):
  (A) gate-horizon long/lat decomposition (frenet + ego-axis) from saved
      windows — the pathspeed 'long vs lat' read at the 4 gate waypoints
      (dense speed/accel-profile metrics are N/A: the planner emits 4 gates,
      not a per-step path).
  (B) vision-use ablation: ADE with real frames vs zeroed frames — does the
      trained encoder actually drive the plan, or does it ride v0/yr0 ego?
Reuses taniteval.pathspeed.{frenet_residual,axis_residual} + refb_eval.collect.
"""
import os, sys, json, copy
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", "/root/models/assess-20260719/stack-v2b")
sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")
import torch
from taniteval import loaders, data
from taniteval import refb_eval  # noqa: F401
from taniteval.registry import MODELS
from taniteval.pathspeed import frenet_residual, axis_residual
from driving_diagnostic import WP_STEPS, gt_ego_waypoints, baseline_waypoints, net_heading_change_deg
import refb_labels as rl

RES = "/root/taniteval/results"
HZ = [k / 10.0 for k in WP_STEPS]                 # 0.5,1,1.5,2s
DT = 0.1


def _rmse(x):
    return round(float(x.pow(2).mean().sqrt()), 4)


def gate_longlat(win):
    p, g = win["pred"].float(), win["gt"].float()          # [N,4,2]
    along, cross = frenet_residual(p, g)                    # [N,4] frenet
    lon, lat = axis_residual(p, g)                          # [N,4] ego-axis
    spd = win.get("speed"); hd = win.get("head_deg")
    de = (p - g).norm(dim=-1)                               # [N,4]
    out = {"n": int(p.shape[0]), "per_horizon": {}}
    for i, t in enumerate(HZ):
        a, c = along[:, i], cross[:, i]
        sa, sc = float(a.pow(2).mean()), float(c.pow(2).mean())
        out["per_horizon"][f"{t:g}s"] = {
            "frenet_long_rmse_m": _rmse(a), "frenet_lat_rmse_m": _rmse(c),
            "long_frac_of_sqerr": round(sa / (sa + sc + 1e-9), 4),
            "ego_long_rmse_m": _rmse(lon[:, i]), "ego_lat_rmse_m": _rmse(lat[:, i]),
            "de_m": round(float(de[:, i].mean()), 4),
        }
    # 2s frenet split on strata
    def split(mask):
        a, c = along[mask, -1], cross[mask, -1]
        sa, sc = float(a.pow(2).mean()), float(c.pow(2).mean())
        return {"n": int(mask.sum()), "long_rmse_m": _rmse(a), "lat_rmse_m": _rmse(c),
                "long_frac_of_sqerr": round(sa / (sa + sc + 1e-9), 4),
                "ade_2s_m": round(float(de[mask].mean()), 4)}
    strata = {"all": split(torch.ones(p.shape[0], dtype=torch.bool))}
    if spd is not None:
        strata["fast_top10pct"] = split(spd >= torch.quantile(spd, 0.90))
        strata["slow_bottom50pct"] = split(spd <= torch.quantile(spd, 0.50))
    if hd is not None:
        strata["sharp_top10pct_curve"] = split(hd >= torch.quantile(hd, 0.90))
        strata["straight_lt5deg"] = split(hd < 5.0)
    out["strata_2s"] = strata
    return out


@torch.no_grad()
def collect_ablation(model, episodes, device, zero_vision, window=8, stride=8, batch=8):
    K = max(WP_STEPS)
    P, GT = [], []
    for ep in episodes:
        fr = ep.feats; T = fr.shape[0]
        starts = list(range(0, T - window - K, stride))
        for i in range(0, len(starts), batch):
            ch = starts[i:i + batch]
            last = torch.tensor([t + window - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + window]) for t in ch]).to(device).float().div_(255.0)
            if zero_vision:
                fw = torch.zeros_like(fw)
            v0 = ep.poses[last, 3].to(device)
            yr0 = (rl.wrap_to_pi(ep.poses[last, 2] - ep.poses[last - 1, 2]) / DT).to(device)
            out = model(fw, nav_cmd=None, v0=v0, yr0=yr0)
            wp = torch.stack([out["waypoints"][k] for k in WP_STEPS], dim=1).cpu().float()
            P.append(wp); GT.append(gt_ego_waypoints(ep.poses, last))
    P, GT = torch.cat(P), torch.cat(GT).float()
    return round(float((P - GT).norm(dim=-1).mean()), 4)


def main():
    report = {}
    for key in ["refb-v2-30k", "refb-v2-20k"]:
        wp = f"{RES}/windows_{key}.pt"
        if os.path.exists(wp):
            win = torch.load(wp, weights_only=False)
            report[f"gate_longlat_{key}"] = gate_longlat(win)
            print(f"[extra] gate_longlat {key}: done (n={win['pred'].shape[0]})", flush=True)
    # vision ablation on 30k
    e = [m for m in MODELS if m["key"] == "refb-v2-30k"][0]
    L = loaders.load(e, "cuda")
    files = data.list_val_episodes("/root/valdata/physicalai-val-0c5f7dac3b11", 40)
    eps = data.load_frames(files)
    ade_on = collect_ablation(L["model"], eps, "cuda", zero_vision=False)
    ade_off = collect_ablation(L["model"], eps, "cuda", zero_vision=True)
    report["vision_ablation_30k"] = {
        "ade_2s_frames_on_m": ade_on, "ade_2s_frames_zeroed_m": ade_off,
        "delta_m": round(ade_off - ade_on, 4),
        "vision_use_pct": round(100 * (ade_off - ade_on) / max(ade_off, 1e-6), 1),
        "note": "frames zeroed -> planner runs on v0/yr0 ego only; "
                "higher delta/pct = more vision-driven."}
    print(f"[extra] vision_ablation 30k: on={ade_on} off={ade_off} "
          f"vision_use={report['vision_ablation_30k']['vision_use_pct']}%", flush=True)
    open(f"{RES}/planner_extra_refb-v2-30k.json", "w").write(json.dumps(report, indent=2))
    print(f"[extra] wrote {RES}/planner_extra_refb-v2-30k.json", flush=True)


if __name__ == "__main__":
    main()
