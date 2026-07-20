"""flagship-v2 @6k mechanism diagnostic (2026-07-19, benchmarks-eval).

Why does the harness say 6.18 m ADE@2s while training telemetry says
g_op_fwd_ade_m=0.50 m?  Decompose the operative grounded-rollout path:

  EXP-1 readout swap : decode the SAME 20-step rollout with op/tac/str readouts
  EXP-2 per-step speed: pred |dxy| per rollout step vs GT per-step displacement
  EXP-3 speed probe   : ridge z_t -> v0 R^2 (encoder speed content), per model
  EXP-4 long/lat      : waypoint error in GT track frame at each WP step
  EXP-5 latent drift  : ||z_j||, cos(z_j, z_t) per rollout depth

Models: flagship-v2-6k, flagship-30k (v1 final), flagship-speed (19k relay).
Run ON the eval pod:
  TANITEVAL_STACK_OVERRIDE=/root/models/assess-20260719/stack-v2 \
  python3 diag_v2mech.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

ASSESS = "/root/models/assess-20260719"
os.environ.setdefault("TANITEVAL_STACK_OVERRIDE", f"{ASSESS}/stack-v2")

# Pin tanitad to the v2 stack copy BEFORE any taniteval import (sys.modules
# cache wins over the harness' own sys.path.insert of /root/TanitAD/stack).
sys.path.insert(0, os.environ["TANITEVAL_STACK_OVERRIDE"])
import tanitad  # noqa: F401,E402
print(f"[diag] tanitad -> {tanitad.__file__}", flush=True)

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import data  # noqa: E402
from driving_diagnostic import (WP_STEPS, gt_ego_waypoints,  # noqa: E402
                                baseline_waypoints)
from tanitad.models.metric_dynamics import (rollout_transitions,  # noqa: E402
                                            decode_transitions)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v2loaders = _load_mod("v2loaders", f"{ASSESS}/harness-v2-patches/loaders.py")
v2registry = _load_mod("v2registry", f"{ASSESS}/harness-v2-patches/registry.py")

K_MAX = max(WP_STEPS)   # 20
DT = 0.1
SPEED_SCALE = 10.0
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
DEVICE = "cuda"
WINDOW, STRIDE, BATCH = 8, 8, 8


def entry(key):
    e = [m for m in v2registry.MODELS if m["key"] == key]
    assert e, f"unknown {key}"
    return e[0]


def append_ego(aw, fa, poses, last, device):
    v0 = (poses[last, 3:4].float() / SPEED_SCALE).to(device)
    aw = torch.cat([aw, v0[:, None].expand(-1, aw.shape[1], -1)], dim=-1)
    fa = torch.cat([fa, v0[:, None].expand(-1, fa.shape[1], -1)], dim=-1)
    return aw, fa


@torch.no_grad()
def collect(model, grounding, eps):
    """Mirror taniteval.rollout.collect but keep the full per-step picture and
    decode the SAME rolled transitions with all three level readouts."""
    out = {k: [] for k in
           ("wp_op", "wp_tac", "wp_str", "gt", "cv", "speed",
            "d_op", "d_str", "gt_d", "znorm", "zcos", "z_t", "eid")}
    readouts = {lvl: grounding.step[lvl] for lvl in ("op", "tac", "str")}
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS], device=DEVICE)
    for ep in eps:
        feats = ep.feats
        T = min(feats.shape[0], ep.actions.shape[0], ep.poses.shape[0])
        starts = list(range(0, T - WINDOW - K_MAX, STRIDE))
        for i in range(0, len(starts), BATCH):
            ch = starts[i:i + BATCH]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(feats[t:t + WINDOW])
                              for t in ch]).to(DEVICE)
            if fw.dtype == torch.uint8:
                fw = fw.float().div_(255.0)
            elif fw.dtype == torch.float16:
                fw = fw.float()
            aw = torch.stack([ep.actions[t:t + WINDOW] for t in ch]).to(DEVICE)
            fa = torch.stack([ep.actions[t + WINDOW:t + WINDOW + K_MAX]
                              for t in ch]).to(DEVICE)
            aw, fa = append_ego(aw, fa, ep.poses, last, DEVICE)
            states = model.encode_window(fw)
            trans = rollout_transitions(model.predictor, states, aw, fa, K_MAX)
            z_t = states[:, -1]
            # latent drift over depth
            zs = torch.stack([t2 for _, t2 in trans], dim=1)      # [b, 20, S]
            out["znorm"].append(zs.norm(dim=-1).cpu())
            out["zcos"].append(torch.cosine_similarity(
                zs, z_t[:, None, :], dim=-1).cpu())
            out["z_t"].append(z_t.cpu())
            for lvl in ("op", "tac", "str"):
                wp_full, dpose = decode_transitions(readouts[lvl], trans, K_MAX)
                out[f"wp_{lvl}"].append(
                    wp_full.index_select(1, wp_idx).cpu().float())
                if lvl in ("op", "str"):
                    out[f"d_{lvl}"].append(
                        dpose[..., :2].norm(dim=-1).cpu().float())  # [b,20]
            # GT per-step displacement magnitudes
            p = ep.poses[:, :2]
            gd = torch.stack([
                (p[t + WINDOW:t + WINDOW + K_MAX] -
                 p[t + WINDOW - 1:t + WINDOW - 1 + K_MAX]).norm(dim=-1)
                for t in ch])                                       # [b,20]
            out["gt_d"].append(gd.float())
            out["gt"].append(gt_ego_waypoints(ep.poses, last).float())
            out["cv"].append(
                baseline_waypoints(ep.poses, last)["constant_velocity"].float())
            out["speed"].append(ep.poses[last, 3].float())
            out["eid"].extend([str(ep.episode_id)] * len(ch))
    return {k: (torch.cat(v) if k != "eid" else v) for k, v in out.items()}


def ade2s(pred, gt):
    return float((pred - gt).norm(dim=-1).mean())


def longlat(pred, gt):
    """Error in the GT track frame per WP step: unit vec = GT segment dir."""
    seg = torch.diff(
        torch.cat([torch.zeros_like(gt[:, :1]), gt], dim=1), dim=1)  # [N,4,2]
    u = seg / seg.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    err = pred - gt
    lon = (err * u).sum(-1)                       # signed along-track
    lat = err[..., 0] * (-u[..., 1]) + err[..., 1] * u[..., 0]
    return lon, lat


def ridge_r2(z, y, eids, lam_grid=(1e-2, 1e-1, 1.0, 10.0)):
    """Per-episode split ridge z->y, return heldout R^2."""
    uniq = sorted(set(eids))
    held = set(uniq[:: max(1, len(uniq) // 8)][:8])
    m = torch.tensor([e in held for e in eids])
    Xtr, Xte = z[~m].double(), z[m].double()
    ytr, yte = y[~m].double(), y[m].double()
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = ytr.mean()
    best = None
    for lam in lam_grid:
        A = Xtr.T @ Xtr + lam * len(Xtr) * torch.eye(Xtr.shape[1]).double()
        w = torch.linalg.solve(A, Xtr.T @ (ytr - ym))
        pr = Xte @ w + ym
        r2 = 1.0 - float(((yte - pr) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())
        if best is None or r2 > best[0]:
            best = (r2, lam)
    return best


def run_model(key):
    e = entry(key)
    L = v2loaders.load(e, DEVICE)
    print(f"[diag] {key}: step={L['step']} loaded", flush=True)
    files = data.list_val_episodes(VAL, 40)
    eps = data.load_frames(files)
    c = collect(L["model"], L["grounding"], eps)
    res = {"key": key, "step": L["step"], "n": int(c["gt"].shape[0])}
    for lvl in ("op", "tac", "str"):
        res[f"ade2s_{lvl}"] = round(ade2s(c[f"wp_{lvl}"], c["gt"]), 4)
    res["ade2s_cv"] = round(ade2s(c["cv"], c["gt"]), 4)
    # per-step speed tracking (m/s), rollout depth 1..20
    for lvl in ("op", "str"):
        pd = c[f"d_{lvl}"] / DT
        gd = c["gt_d"] / DT
        res[f"speed_pred_{lvl}"] = [round(float(x), 3) for x in pd.mean(0)]
        res[f"speed_err_{lvl}"] = [round(float(x), 3)
                                   for x in (pd - gd).mean(0)]
    res["speed_gt"] = [round(float(x), 3) for x in (c["gt_d"] / DT).mean(0)]
    # step-1 speed R^2 of the decode (operative path speed fidelity)
    for lvl in ("op", "str"):
        p1 = c[f"d_{lvl}"][:, 0] / DT
        g1 = c["gt_d"][:, 0] / DT
        ss = 1.0 - float(((g1 - p1) ** 2).sum() / ((g1 - g1.mean()) ** 2).sum())
        res[f"step1_speed_r2_{lvl}"] = round(ss, 4)
    # long/lat at each wp step (op readout)
    lon, lat = longlat(c["wp_op"], c["gt"])
    res["op_long_mean_by_wp"] = [round(float(x), 3) for x in lon.mean(0)]
    res["op_abs_long_by_wp"] = [round(float(x), 3) for x in lon.abs().mean(0)]
    res["op_abs_lat_by_wp"] = [round(float(x), 3) for x in lat.abs().mean(0)]
    # speed strata for the 2 s wp (op)
    spd = c["speed"]
    q = spd.quantile(torch.tensor([1 / 3, 2 / 3]))
    for name, m in (("low", spd < q[0]),
                    ("med", (spd >= q[0]) & (spd < q[1])),
                    ("high", spd >= q[1])):
        res[f"op_ade2s_{name}"] = round(ade2s(c["wp_op"][m], c["gt"][m]), 4)
        res[f"op_long2s_{name}"] = round(float(lon[m, -1].mean()), 3)
    # encoder speed probe
    r2, lam = ridge_r2(c["z_t"], c["speed"], c["eid"])
    res["probe_speed_r2"] = round(r2, 4)
    res["probe_lam"] = lam
    # latent drift
    res["znorm_by_step"] = [round(float(x), 2) for x in c["znorm"].mean(0)]
    res["zcos_by_step"] = [round(float(x), 4) for x in c["zcos"].mean(0)]
    torch.save(c, f"/root/taniteval/results/diagv2_{key}.pt")
    del L
    torch.cuda.empty_cache()
    return res


def main():
    keys = sys.argv[1:] or ["flagship-v2-6k", "flagship-30k", "flagship-speed"]
    all_res = {}
    for k in keys:
        try:
            all_res[k] = run_model(k)
            print(json.dumps(all_res[k], indent=1), flush=True)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            all_res[k] = {"error": str(ex)}
    with open("/root/taniteval/results/diagv2_summary.json", "w") as f:
        json.dump(all_res, f, indent=1)
    print("[diag] DONE", flush=True)


if __name__ == "__main__":
    main()
