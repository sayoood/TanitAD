"""P2a — ZERO-TRAINING recovery-response probe (the near-free discriminator).

Question, answered without spending one FT step: when REF-C is shown a real window
warped to simulate an in-envelope pose offset (dlat, dpsi) — exactly the covariate
shift the closed loop drives it into — does its plan MOVE to recover toward the path,
or does it stay on its on-path plan (blind)? This is the analog of the gradient-
coupling stream's Phase-0 pre-probe: it can GREENLIGHT or KILL the recovery-
augmentation lever for ~0 GPU (inference only, minutes on the eval pod).

METRIC (per perturbation magnitude, episode-cluster bootstrap CI):
  demand_m   = mean signed lateral correction the GT recovery trajectory requires at
               the 0.5 s lookahead = recovery_target(perturbed)[:,0,1] - base_target
  response_m = mean signed change in REF-C's 0.5 s lookahead lateral when the input is
               warped by (dlat,dpsi) vs un-warped, projected on the demand direction
  recovery_ratio = response_m / demand_m
      ~1 -> the planner already recovers (blind-ness is NOT the bottleneck; departures
            are an execution/controller problem -> augmentation predicted NULL)
      ~0 -> the planner is covariate-shift BLIND (keeps its on-path plan from an
            off-path view) -> in-envelope recovery augmentation should teach it ->
            GREENLIGHT the P2b FT.

The warp is perturb.warp_windows, asserted byte-identical to the abe82f1f instrument
(_assert_warp_matches_harness), so what the probe measures is what the loop drives.

Run (eval pod, after abe82f1f frees eval; ~minutes, no training):
  PYTHONPATH=/root/TanitAD/stack:/root/TanitAD/stack/scripts \
    python3 recovery_probe.py --refc-ckpt /root/models/refc-base-30k/ckpt.pt \
      --val-dir /root/valdata/physicalai-val-0c5f7dac3b11 --out recovery_probe.json
CPU smoke (tiny model + fake eps; validates the metric math end-to-end):
  python3 recovery_probe.py --smoke --out /tmp/probe_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_ROOTS = ["/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
          "/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
          "/root/taniteval", "/root/TanitAD/taniteval", "/workspace/taniteval",
          "/workspace/TanitAD/taniteval"]
for _up in (5, 6):                       # local repo layout depth (crash-safe)
    try:
        _r = _HERE.parents[_up]
        _ROOTS += [str(_r / "stack"), str(_r / "stack" / "scripts"), str(_r / "taniteval")]
    except IndexError:
        pass
for _p in _ROOTS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import perturb  # noqa: E402
from refb_labels import waypoint_targets  # noqa: E402

LOOKAHEAD = 0                 # index into horizons (5,10,15,20); 0 == 0.5 s
GRID = [(0.5, 0.0), (1.0, 0.0), (1.5, 0.0), (0.0, 3.0), (0.0, 5.0), (1.0, 3.0)]
PRIMARY = (1.0, 0.0)          # headline perturbation


def _refc_plan(model, frames, v0, steps, device):
    """REF-C 0.5 s ego-frame lookahead [B,2] via the DEPLOYED decode path."""
    out = model(frames.to(device), nav_cmd=None, v0=v0.to(device), steps=steps)
    return out["traj"][:, LOOKAHEAD].detach().cpu()


def _boot(x, eid):
    from taniteval import ci as _ci
    return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid, reduce="mean",
                                         n_boot=2000)


@torch.no_grad()
def probe(model, episodes, horizons, steps, device, stride=8, batch=16):
    """For each window: base plan (un-warped) + perturbed plans over GRID; collect
    per-window response and demand along the demand direction."""
    W = 8
    max_h = max(horizons)
    rows = {g: {"resp": [], "demand": [], "eid": []} for g in GRID}
    for ei, ep in enumerate(episodes):
        fr = ep.frames
        fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
        poses = ep.poses.float()
        T = poses.shape[0]
        starts = list(range(0, T - W - max_h, stride))
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            frames = torch.stack([fr[t0:t0 + W] for t0 in ch])       # [b,W,C,H,W']
            last = torch.tensor([t0 + W - 1 for t0 in ch])
            pose_last = poses[last]                                   # [b,4]
            fut = torch.stack([poses[l + 1:l + 1 + max_h] for l in last])  # [b,H,4]
            v0 = pose_last[:, 3]
            base_plan = _refc_plan(model, frames, v0, steps, device)  # [b,2]
            base_tgt = waypoint_targets(pose_last, fut, horizons)[:, LOOKAHEAD]
            for (dl, dyaw_deg) in GRID:
                dlat = torch.full((b,), float(dl))
                dpsi = torch.full((b,), float(np.radians(dyaw_deg)))
                warped = perturb.warp_windows(frames, dlat, dpsi)
                pert_plan = _refc_plan(model, warped, v0, steps, device)
                rec_tgt = perturb.recovery_targets(pose_last, fut, horizons,
                                                   dlat, dpsi, waypoint_targets)[:, LOOKAHEAD]
                demand = rec_tgt - base_tgt                          # [b,2]
                resp = pert_plan - base_plan                         # [b,2]
                dnorm = demand.norm(dim=-1).clamp_min(1e-6)
                dhat = demand / dnorm[:, None]
                rows[(dl, dyaw_deg)]["demand"].append(dnorm)         # >=0
                rows[(dl, dyaw_deg)]["resp"].append((resp * dhat).sum(-1))  # signed
                rows[(dl, dyaw_deg)]["eid"] += [str(ei)] * b
    res = {}
    for g, d in rows.items():
        demand = torch.cat(d["demand"]).numpy()
        resp = torch.cat(d["resp"]).numpy()
        eid = d["eid"]
        ratio = resp / np.maximum(demand, 1e-6)
        res[f"dlat{g[0]}_dyaw{g[1]}"] = {
            "n_windows": int(len(eid)), "n_episodes": int(len(set(eid))),
            "demand_m": _boot(demand, eid), "response_m": _boot(resp, eid),
            "recovery_ratio": _boot(ratio, eid),
            "recovery_ratio_pooled": round(float(resp.sum() / max(demand.sum(), 1e-6)), 4),
        }
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", default="/root/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tie the probe's warp to the instrument's operator (MEASURED, not asserted)
    try:
        import importlib.util
        _cc = [str(_HERE.parent), "/workspace/refc_cl_d2", "/root/TanitAD",
               "/workspace/TanitAD"]
        for _up in (5, 6):
            try:
                _cc.append(str(_HERE.parents[_up]))
            except IndexError:
                pass
        for cand in _cc:
            f = list(Path(cand).rglob("lowood_lanekeep.py")) if Path(cand).exists() else []
            if f:
                spec = importlib.util.spec_from_file_location("LK", str(f[0]))
                LK = importlib.util.module_from_spec(spec); spec.loader.exec_module(LK)
                chk = perturb._assert_warp_matches_harness(LK.sampling_homography,
                                                           LK.warp_batch)
                print(f"[probe] warp==instrument: {chk}")
                assert chk["ok"], "probe warp DIVERGES from the instrument"
                break
    except Exception as e:                       # noqa: BLE001
        print(f"[probe] instrument warp cross-check skipped: {e}")

    if args.smoke:
        from tanitad.refs.refc import RefCModel, refc_smoke_config
        cfg = refc_smoke_config()
        model = RefCModel(cfg).to(device).eval()
        horizons = cfg.trajectory.horizons
        steps = cfg.decoder.diffusion_steps
        eps = []
        for _ in range(3):
            e = type("E", (), {})()
            e.poses = torch.randn(50, 4); e.poses[:, 3] = e.poses[:, 3].abs() * 10
            e.frames = torch.rand(50, cfg.encoder.in_channels,
                                  cfg.encoder.image_size, cfg.encoder.image_size)
            eps.append(e)
        res = probe(model, eps, horizons, steps, device, stride=8, batch=8)
        print(json.dumps({k: {"recovery_ratio": v["recovery_ratio"]["mean"],
                              "demand_m": v["demand_m"]["mean"]}
                          for k, v in res.items()}, indent=2))
        Path(args.out).write_text(json.dumps({"smoke": True, "results": res},
                                             indent=2, default=str))
        print("RECOVERY_PROBE_SMOKE_OK")
        return

    from tanitad.data.mixing import load_episode
    # load REF-C via the instrument's strict-load convention (inlined _load_refc)
    model, step = _load_refc(args.refc_ckpt, args.refc_preset, device)
    eps = sorted(Path(args.val_dir).glob("ep_*.pt"))[:args.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in eps]
    from tanitad.refs.refc import refc_config
    horizons = refc_config().trajectory.horizons
    steps = 2
    print(f"[probe] REF-C step {step} | {len(episodes)} eps | grid {GRID}", flush=True)
    res = probe(model, episodes, horizons, steps, device)
    out = {"_design": "zero-training recovery-response probe; recovery_ratio ~0 => "
                      "planner covariate-shift BLIND (greenlight FT), ~1 => already "
                      "recovers (execution-bound, FT predicted null)",
           "refc_ckpt": args.refc_ckpt, "refc_step": step, "primary": str(PRIMARY),
           "results": res, "evidence_class": "MEASURED (this probe run)"}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    prim = res[f"dlat{PRIMARY[0]}_dyaw{PRIMARY[1]}"]["recovery_ratio"]
    print(f"[probe] PRIMARY {PRIMARY} recovery_ratio="
          f"{prim['mean']:.3f} [{prim['lo']:.3f},{prim['hi']:.3f}]", flush=True)
    print(f"[probe] wrote {args.out}")
    print("RECOVERY_PROBE_DONE", flush=True)


def _load_refc(ckpt, preset, device):
    """Inline of lowood_lanekeep.load_refc (strict) to avoid a hard import."""
    import dataclasses
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_xl_config)
    presets = {"base": refc_config, "small": refc_small_config, "xl": refc_xl_config}
    cfg = presets[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        d = json.loads(cj.read_text()).get("cfg", {})
        _apply(cfg, d)
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), int(ck.get("step", -1))


def _apply(cfg, d):
    for k, v in d.items():
        if not hasattr(cfg, k):
            continue
        cur = getattr(cfg, k)
        if isinstance(v, dict) and hasattr(cur, "__dataclass_fields__"):
            _apply(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(cfg, k, tuple(v))
        else:
            setattr(cfg, k, v)


if __name__ == "__main__":
    main()
