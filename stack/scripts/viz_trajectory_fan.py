"""Trajectory-fan overlay: imagine-and-select made visible.

For a handful of held-out real windows, renders side by side:

  LEFT  — the camera frame with the decoded trajectories projected onto the
          road (approximate pinhole ground-plane projection, f_eff=266 px,
          camera height ~1.22 m — labeled approximate, for intuition only).
  RIGHT — BEV (metres, ego frame): ground-truth future (green), the frozen
          D1 probe's decode of the CURRENT encoder state out to 2 s (blue),
          and the **imagination fan** (oranges): K steer-sweep candidate
          actions, each imagined by one batched predictor pass and decoded
          by the A3-calibrated probe at the predictor horizon (0.4 s).
          The candidate whose imagined consequence best matches the
          goal (here: the GT 0.4 s point) is highlighted — that argmin IS
          imagine-and-select.

Probes are fitted on a disjoint fit-half of val windows (route-level cache);
showcase windows come from the other half. No claim beyond D1/D2's existing
numbers — this is the communication artifact for those results.

Usage (local 4060):
  python stack/scripts/viz_trajectory_fan.py --ckpt .../ckpt_full.pt \
      --comma-cache .../comma2k19-val-<hash> --out-dir .../fan_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from tanitad.config import base250cam_config
from tanitad.data.mixing import load_episode
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.readout import RidgeProbe

WAYPOINT_STEPS = (5, 10, 15, 20)              # 0.5..2 s @ 10 Hz (D1 probe)
K_FAN = 9                                     # steer sweep candidates
STEER_SWEEP = np.linspace(-0.12, 0.12, K_FAN)  # rad, sustained maneuver
SUSTAIN = 4                                   # steer applied to last N actions
F_EFF, IMG, CAM_H = 266.0, 256, 1.22          # D-016 focal, size, cam height


def _ego_frame(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    x = dxy[..., 0] * c - dxy[..., 1] * s
    y = dxy[..., 0] * s + dxy[..., 1] * c
    return torch.stack([x, y], dim=-1)


def _tac_horizon(world) -> int:
    """Prefer the tactical predictor's longest horizon (1.6 s); fall back to
    the operative k_max if the tactical brain is absent."""
    if world.tactical_pred is not None:
        return max(world.tactical_pred.cfg.horizons)
    return max(world.predictor.cfg.horizons)


def _imagine_h(world, states, actions, h):
    pred = (world.tactical_pred if world.tactical_pred is not None
            else world.predictor)
    return pred(states, actions)[h]


@torch.no_grad()
def collect(world, episodes, device, window, stride=8, batch=4):
    """Per-window: state, GT waypoints, tactical imagined latent (true
    actions), GT displacement at the tactical horizon, plus window refs."""
    h = _tac_horizon(world)
    rows = []
    for ep in episodes:
        T = ep.frames.shape[0]
        need = max(max(WAYPOINT_STEPS), h)
        for i0 in range(0, T - window - need, stride):
            rows.append((ep, i0))
    out = {"state": [], "wp": [], "z_imag_h": [], "disp_h": [], "meta": []}
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        fw = torch.stack([(e.frames[t:t + window].float().div(255.0)
                           if e.frames.dtype == torch.uint8
                           else e.frames[t:t + window]) for e, t in chunk]).to(device)
        aw = torch.stack([e.actions[t:t + window] for e, t in chunk]).to(device)
        states = world.encode_window(fw)
        z_h = _imagine_h(world, states, aw, h)
        for j, (e, t) in enumerate(chunk):
            last = t + window - 1
            yaw0, p0 = e.poses[last, 2], e.poses[last, :2]
            wp = torch.stack([_ego_frame(e.poses[last + k, :2] - p0, yaw0)
                              for k in WAYPOINT_STEPS])
            out["state"].append(states[j, -1].cpu())
            out["wp"].append(wp)
            out["z_imag_h"].append(z_h[j].cpu())
            out["disp_h"].append(_ego_frame(e.poses[last + h, :2] - p0, yaw0))
            out["meta"].append((e, t))
    return {k: (torch.stack(v) if k != "meta" else v) for k, v in out.items()}


@torch.no_grad()
def fan_decode(world, ep, t0, window, probe_imag, device):
    """K sustained-steer maneuvers -> tactical imagined latents -> A3 decode
    at the tactical horizon (1.6 s)."""
    h = _tac_horizon(world)
    frames = ep.frames[t0:t0 + window].float().div(255.0) \
        if ep.frames.dtype == torch.uint8 else ep.frames[t0:t0 + window]
    fw = frames.unsqueeze(0).expand(K_FAN, -1, -1, -1, -1).to(device)
    aw = ep.actions[t0:t0 + window].unsqueeze(0).repeat(K_FAN, 1, 1).to(device)
    sweep = torch.tensor(STEER_SWEEP, dtype=aw.dtype, device=device)
    aw[:, -SUSTAIN:, 0] = aw[:, -SUSTAIN:, 0] + sweep[:, None]
    states = world.encode_window(fw)
    z_h = _imagine_h(world, states, aw, h)
    p = probe_imag.predict(z_h.cpu())
    return np.asarray(p).reshape(K_FAN, 2)


def to_image_plane(xy: np.ndarray) -> np.ndarray:
    """Ego ground points (x fwd, y left, m) -> approx pixel coords."""
    x = np.clip(xy[:, 0], 2.0, None)
    u = IMG / 2 - F_EFF * (xy[:, 1] / x)
    v = IMG / 2 + F_EFF * (CAM_H / x)
    return np.stack([u, v], axis=1)


def render(ax_img, ax_bev, frame_rgb, gt_wp, d1_wp, fan_xy, sel):
    import matplotlib.patheffects as pe
    ax_img.imshow(frame_rgb)
    ax_img.set_title("camera view (projection approx.)", fontsize=9)
    ax_img.axis("off")
    for xy, color, lw in ((gt_wp, "#2ca02c", 2.2), (d1_wp, "#1f77b4", 1.8)):
        px = to_image_plane(np.vstack([[2.0, 0.0], xy]))
        ax_img.plot(px[:, 0], px[:, 1], "-o", color=color, lw=lw, ms=3,
                    path_effects=[pe.withStroke(linewidth=3.2, foreground="k")])
    fx = to_image_plane(fan_xy)
    ax_img.scatter(fx[:, 0], fx[:, 1], c="#ff7f0e", s=14, zorder=5,
                   edgecolors="k", linewidths=0.4)
    ax_img.scatter(fx[sel, 0], fx[sel, 1], c="#d62728", s=48, zorder=6,
                   marker="*", edgecolors="k")
    ax_img.set_xlim(0, IMG); ax_img.set_ylim(IMG, 0)

    ax_bev.plot(-gt_wp[:, 1], gt_wp[:, 0], "-o", color="#2ca02c", lw=2.2,
                ms=4, label="ground truth (2 s)")
    ax_bev.plot(-d1_wp[:, 1], d1_wp[:, 0], "-o", color="#1f77b4", lw=1.8,
                ms=4, label="D1 probe decode (2 s)")
    for k in range(len(fan_xy)):
        ax_bev.plot([0, -fan_xy[k, 1]], [0, fan_xy[k, 0]], "-",
                    color="#ff7f0e", alpha=0.45, lw=1.1)
    ax_bev.scatter(-fan_xy[:, 1], fan_xy[:, 0], c="#ff7f0e", s=18,
                   label="imagined fan @1.6 s (K=9)")
    ax_bev.scatter(-fan_xy[sel, 1], fan_xy[sel, 0], c="#d62728", marker="*",
                   s=120, zorder=6, label="selected candidate")
    ax_bev.scatter([0], [0], c="k", marker="s", s=40)
    ax_bev.set_xlabel("left-right (m)"); ax_bev.set_ylabel("forward (m)")
    ax_bev.grid(alpha=0.3); ax_bev.legend(fontsize=7, loc="upper left")
    ax_bev.set_aspect("equal", adjustable="datalim")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--comma-cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-show", type=int, default=6)
    ap.add_argument("--max-eps", type=int, default=12)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from tanitad.models.fourbrain import WorldModel
    world = WorldModel(base250cam_config())
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.to(device).eval()
    window = world.predictor.cfg.window

    eps = [load_episode(str(p), mmap=True)
           for p in sorted(Path(args.comma_cache).glob("ep_*.pt"))[:args.max_eps]]
    with strict_numerics():
        col = collect(world, eps, device, window)
        n = col["state"].shape[0]
        half = n // 2
        probe_d1 = RidgeProbe(alpha=10.0).fit(
            col["state"][:half], col["wp"][:half].flatten(1))
        probe_imag = RidgeProbe(alpha=10.0).fit(          # A3: fit ON imagined
            col["z_imag_h"][:half], col["disp_h"][:half])
        print(f"[fan] ckpt step {step}; {n} windows; tac horizon "
              f"{_tac_horizon(world)} steps; D1 fit R2 "
              f"{probe_d1.r2(col['state'][:half], col['wp'][:half].flatten(1)):.3f}; "
              f"imag fit R2 {probe_imag.r2(col['z_imag_h'][:half], col['disp_h'][:half]):.3f}")

        out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
        show = np.linspace(half, n - 1, args.n_show).astype(int)
        for si, wi in enumerate(show):
            ep, t0 = col["meta"][wi]
            gt_wp = col["wp"][wi].numpy()
            d1_wp = np.asarray(
                probe_d1.predict(col["state"][wi:wi + 1])).reshape(-1, 2)
            fan = fan_decode(world, ep, t0, window, probe_imag, device)
            gt04 = col["disp_h"][wi].numpy()
            sel = int(np.linalg.norm(fan - gt04, axis=1).argmin())

            last = ep.frames[t0 + window - 1]
            rgb = (last[-3:].float() / 255.0 if last.dtype == torch.uint8
                   else last[-3:]).permute(1, 2, 0).numpy()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6),
                                           gridspec_kw={"width_ratios": [1, 1.15]})
            render(ax1, ax2, rgb, gt_wp, d1_wp, fan, sel)
            fig.suptitle(f"TanitAD imagine-and-select — step {step} ckpt, "
                         f"comma val ep {int(ep.episode_id)} t={t0} "
                         f"(steer sweep ±0.12 rad)", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"fan_{si:02d}.png", dpi=130)
            plt.close(fig)
            print(f"[fan] wrote fan_{si:02d}.png (sel candidate {sel}, "
                  f"steer {STEER_SWEEP[sel]:+.3f} rad)")


if __name__ == "__main__":
    main()
