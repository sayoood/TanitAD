"""Unified GT-vs-predicted BEV trajectory overlays for the 3 reset arms.

  --arm refa      frozen-DINO 4-brain (features)  -> grounded rollout (+v0)
  --arm flagship  4-brain WorldModel  (frames)    -> grounded rollout (+v0)
  --arm refb      from-scratch BC     (frames)    -> direct waypoint heads (+v0)

Scene selection (deterministic on the shared val poses) + the BEV render() are
byte-identical to render_overlays.py, so all three arms pick the SAME scenes and
the plots are directly comparable. GT (green solid) vs the arm's prediction
(vermillion dashed), ego frame, up = ahead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/workspace/tmp/refa_plus")            # refa only (harmless elsewhere)
sys.path.insert(0, "/workspace/TanitAD/stack/scripts")
sys.path.insert(0, "/workspace/TanitAD/stack")

from driving_diagnostic import (WP_STEPS, curvature_bucket, de_of,
                                gt_ego_waypoints, net_heading_change_deg)

SPEED_SCALE = 10.0
FWD_K = 20
SCAN_EPS = 300
GT_COL = (0, 158, 115)
PRED_COL = (213, 94, 0)
GRID_COL = (223, 223, 223)
AX_COL = (120, 120, 120)
TXT_COL = (20, 20, 20)
ARM_LABEL = {"refa": "REF-A 4-brain", "flagship": "Flagship 4-brain",
             "refb": "REF-B (BC)"}


def get_font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=sz)
    except TypeError:
        return ImageFont.load_default()


F_TITLE, F_AX, F_LEG = get_font(24), get_font(16), get_font(18)


def nice_step(rng):
    for s in (0.5, 1, 2, 5, 10, 20, 50, 100):
        if rng / s <= 8:
            return s
    return 200.0


def render(gt, pred, title, pred_label, out_path):
    """gt, pred: [K,2] arrays of (forward, lateral) metres, ego frame."""
    W, H = 1200, 1180
    L, R, TOP, BOT = 92, 40, 96, 84
    pw, ph = W - L - R, H - TOP - BOT
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    pts = np.concatenate([np.zeros((1, 2)), gt, pred], axis=0)
    f, lat = pts[:, 0], pts[:, 1]
    fmin, fmax, lmin, lmax = f.min(), f.max(), lat.min(), lat.max()
    fr, lr = max(fmax - fmin, 1.0), max(lmax - lmin, 1.0)
    fmin -= fr * 0.12; fmax += fr * 0.12
    lmin -= lr * 0.12; lmax += lr * 0.12
    fc, lc = (fmin + fmax) / 2, (lmin + lmax) / 2
    span_f, span_l = max(fmax - fmin, 4.0), max(lmax - lmin, 4.0)
    scale = min(pw / span_l, ph / span_f)
    cx, cy = L + pw / 2, TOP + ph / 2

    def to_px(fwd, latv):
        return (cx + (latv - lc) * scale, cy - (fwd - fc) * scale)

    vis_l0, vis_l1 = lc - (pw / 2) / scale, lc + (pw / 2) / scale
    vis_f0, vis_f1 = fc - (ph / 2) / scale, fc + (ph / 2) / scale
    d.rectangle([L, TOP, L + pw, TOP + ph], outline=AX_COL, width=1)
    sl = nice_step(vis_l1 - vis_l0)
    x = np.ceil(vis_l0 / sl) * sl
    while x <= vis_l1:
        px, _ = to_px(fc, x)
        if L <= px <= L + pw:
            d.line([px, TOP, px, TOP + ph], fill=GRID_COL, width=1)
            d.text((px - 10, TOP + ph + 6), f"{x:g}", font=F_AX, fill=AX_COL)
        x += sl
    sf = nice_step(vis_f1 - vis_f0)
    y = np.ceil(vis_f0 / sf) * sf
    while y <= vis_f1:
        _, py = to_px(y, lc)
        if TOP <= py <= TOP + ph:
            d.line([L, py, L + pw, py], fill=GRID_COL, width=1)
            d.text((L - 40, py - 8), f"{y:g}", font=F_AX, fill=AX_COL)
        y += sf

    wp_set = set(WP_STEPS)

    def draw_path(arr, col, dashed):
        chain = [to_px(0.0, 0.0)] + [to_px(a, b) for a, b in arr]
        if dashed:
            for i in range(len(chain) - 1):
                (x0, y0), (x1, y1) = chain[i], chain[i + 1]
                seglen = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                nd = max(int(seglen / 9), 1)
                for k in range(nd):
                    if k % 2 == 0:
                        t0, t1 = k / nd, (k + 1) / nd
                        d.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                                x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
                               fill=col, width=3)
        else:
            d.line(chain, fill=col, width=3)
        for j, (px, py) in enumerate(chain[1:]):
            r = 5 if (j + 1) in wp_set else 2.4
            d.ellipse([px - r, py - r, px + r, py + r], fill=col)

    draw_path(gt, GT_COL, dashed=False)
    draw_path(pred, PRED_COL, dashed=True)
    ox, oy = to_px(0.0, 0.0)
    d.rectangle([ox - 5, oy - 5, ox + 5, oy + 5], fill=(0, 0, 0))
    d.text((ox + 8, oy + 4), "ego", font=F_AX, fill=TXT_COL)

    d.text((L, 30), title, font=F_TITLE, fill=TXT_COL)
    d.text((L + pw / 2 - 40, H - 30), "lateral (m)", font=F_AX, fill=TXT_COL)
    d.text((10, TOP - 26), "forward (m), up = ahead", font=F_AX, fill=TXT_COL)
    lx, ly = L + pw - 260, TOP + 14
    d.line([lx, ly, lx + 34, ly], fill=GT_COL, width=3)
    d.ellipse([lx + 15, ly - 3, lx + 21, ly + 3], fill=GT_COL)
    d.text((lx + 42, ly - 9), "GT (odometry)", font=F_LEG, fill=TXT_COL)
    for k in range(4):
        d.line([lx + k * 9, ly + 24, lx + k * 9 + 5, ly + 24], fill=PRED_COL, width=3)
    d.ellipse([lx + 15, ly + 21, lx + 21, ly + 27], fill=PRED_COL)
    d.text((lx + 42, ly + 15), pred_label, font=F_LEG, fill=TXT_COL)
    img.save(out_path)


# --------------------------------------------------------------------------- #
# Per-arm model load + waypoint prediction                                     #
# --------------------------------------------------------------------------- #
def load_arm(arm, ckpt, device):
    if arm == "refa":
        from refa_plus import RefAModelPlus
        from tanitad.config import flagship4b_config
        from tanitad.models.metric_dynamics import StepDisplacementReadout
        cfg = flagship4b_config()
        object.__setattr__(cfg.predictor, "action_dim", 3)
        if cfg.tactical_pred is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        model = RefAModelPlus.from_stack_config(cfg, n_tokens=256,
                                                adapter_kind="temporal")
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        sr = StepDisplacementReadout(model.state_dim).to(device).eval()
        sr.load_state_dict(ck["step_readout"])
        return {"model": model, "sr": sr, "window": model.pred_cfg.window,
                "step": int(ck.get("step", -1)), "kind": "rollout"}
    if arm == "flagship":
        from tanitad.config import flagship4b_config
        from tanitad.models.fourbrain import WorldModel
        from tanitad.models.metric_dynamics import HierarchicalGrounding
        cfg = flagship4b_config()
        object.__setattr__(cfg.predictor, "action_dim", 3)
        if getattr(cfg, "tactical_pred", None) is not None:
            object.__setattr__(cfg.tactical_pred, "action_dim", 3)
        model = WorldModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        g = HierarchicalGrounding(model.state_dim).to(device).eval()
        g.load_state_dict(ck["grounding"])
        return {"model": model, "sr": g.step["op"],
                "window": model.predictor.cfg.window,
                "step": int(ck.get("step", -1)), "kind": "rollout"}
    if arm == "refb":
        from tanitad.refs.refb import RefBModel, refb_config
        cfg = refb_config()
        cfg.speed_input = True
        cfg.aux_accel = True
        model = RefBModel(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        model = model.to(device).eval()
        return {"model": model, "window": cfg.window,
                "step": int(ck.get("step", -1)), "kind": "bc"}
    raise ValueError(arm)


def load_ep(arm, path):
    if arm == "refa":
        d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return {"feats": d["feats_fp16"], "actions": d["actions"].float(),
                "poses": d["poses"].float(), "T": d["feats_fp16"].shape[0],
                "eid": int(d["episode_id"])}
    from tanitad.data.mixing import load_episode
    ep = load_episode(path, mmap=True)
    fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
    return {"frames": fr, "actions": ep.actions.float(), "poses": ep.poses.float(),
            "T": fr.shape[0], "eid": int(ep.episode_id)}


def read_poses(arm, path):
    """Cheap poses + T for scene selection — NO frame/feature materialization
    (raw mmap load; only touch poses + the big tensor's .shape)."""
    d = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    key = "feats_fp16" if arm == "refa" else "frames_u8"
    return d["poses"].float(), d[key].shape[0]


@torch.no_grad()
def predict_wp(arm, A, ep, last, device):
    W = A["window"]
    t = last - W + 1
    poses = ep["poses"]
    if A["kind"] == "rollout":
        from tanitad.models.metric_dynamics import rollout_decode
        fw = (ep["feats"] if arm == "refa" else ep["frames"])[t:t + W].unsqueeze(0).to(device)
        aw = ep["actions"][t:t + W].unsqueeze(0).to(device)
        fa = ep["actions"][t + W:t + W + FWD_K].unsqueeze(0).to(device)
        v0 = (poses[last, 3:4] / SPEED_SCALE).to(device)
        aw = torch.cat([aw, v0.view(1, 1, 1).expand(-1, W, -1)], dim=-1)
        fa = torch.cat([fa, v0.view(1, 1, 1).expand(-1, FWD_K, -1)], dim=-1)
        states = A["model"].encode_window(fw)
        wp_full, _ = rollout_decode(A["model"].predictor, states, aw, fa,
                                    A["sr"], FWD_K)
        return wp_full[0].cpu().numpy(), list(range(1, FWD_K + 1))
    # refb: direct waypoint heads at (5,10,15,20)
    fw = ep["frames"][t:t + W].unsqueeze(0).to(device)
    v0 = poses[last, 3].view(1).to(device)
    out = A["model"](fw, nav_cmd=None, v0=v0)
    hs = list(out["waypoints"].keys())
    hs = sorted(int(k) for k in hs)
    pred = torch.stack([out["waypoints"][k][0] for k in hs]).cpu().numpy()
    return pred, hs


def select_scenes(arm, files, window, device):
    cand = []
    for idx, fp in enumerate(files):
        try:
            poses, T = read_poses(arm, str(fp))
        except Exception:
            continue
        hi = T - 1 - FWD_K
        if hi <= window:
            continue
        lasts = torch.arange(window - 1, hi, 5)
        if len(lasts) == 0:
            continue
        heads = net_heading_change_deg(poses, lasts)
        j = int(torch.argmax(heads))
        last = int(lasts[j]); hd = float(heads[j])
        cand.append((curvature_bucket(hd), hd, idx, last, float(poses[last, 3])))
    sharp = sorted([c for c in cand if c[0] == "sharp"], key=lambda c: -c[1])
    gentle = sorted([c for c in cand if c[0] == "gentle"], key=lambda c: -c[1])
    straight = sorted([c for c in cand if c[0] == "straight" and c[4] > 3.0],
                      key=lambda c: -c[4])

    def spread(pool, k):
        if len(pool) <= k:
            return pool
        return [pool[round(i * (len(pool) - 1) / (k - 1))] for i in range(k)]

    return (sharp[:3] + gentle[:2] + spread(straight, 3))[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["refa", "flagship", "refb"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True, help="val dir of ep_*.pt")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    A = load_arm(args.arm, args.ckpt, device)
    label = ARM_LABEL[args.arm]
    print(f"[{args.arm}] step {A['step']} window {A['window']} dev {device}",
          flush=True)

    files = sorted(Path(args.data_dir).glob("ep_*.pt"))[:SCAN_EPS]
    pick = select_scenes(args.arm, files, A["window"], device)
    print(f"[{args.arm}] selected {len(pick)} scenes", flush=True)

    manifest = []
    coords = []
    for turn, hd, idx, last, spd in pick:
        ep = load_ep(args.arm, str(files[idx]))
        pred, hs = predict_wp(args.arm, A, ep, last, device)
        gt = gt_ego_waypoints(ep["poses"], torch.tensor([last]),
                              wp_steps=hs)[0].numpy()
        wp_i = [hs.index(k) for k in WP_STEPS if k in hs]
        ade = float(de_of(torch.tensor(pred)[wp_i], torch.tensor(gt)[wp_i]).mean())
        title = f"{label} | ep{idx:05d} {turn} (dpsi@2s={hd:.0f}deg) | ADE@2s={ade:.2f}m"
        out = outdir / f"{args.arm}_ep{idx:05d}_{turn}.png"
        render(gt, pred, title, label, str(out))
        manifest.append((out.name, idx, turn, round(hd, 1), round(spd, 1),
                         round(ade, 2), A["step"]))
        coords.append({"ep": idx, "turn": turn, "dpsi": round(hd, 1),
                       "spd": round(spd, 1), "ade": round(ade, 2),
                       "step": A["step"], "hs": hs,
                       "gt": [[round(float(a), 3), round(float(b), 3)] for a, b in gt],
                       "pred": [[round(float(a), 3), round(float(b), 3)] for a, b in pred]})
        print(f"[{args.arm}] {out.name}: {turn} dpsi={hd:.0f} spd={spd:.1f} "
              f"ADE@2s={ade:.2f}", flush=True)
    import json as _json
    (outdir / f"{args.arm}_coords.json").write_text(_json.dumps(coords))
    print(f"[{args.arm}] coords -> {outdir}/{args.arm}_coords.json", flush=True)

    print("\n=== MANIFEST ===", flush=True)
    for m in manifest:
        print("|".join(str(x) for x in m), flush=True)
    print(f"ARM_RENDER_DONE {args.arm} step={A['step']} n={len(manifest)}",
          flush=True)


if __name__ == "__main__":
    main()
