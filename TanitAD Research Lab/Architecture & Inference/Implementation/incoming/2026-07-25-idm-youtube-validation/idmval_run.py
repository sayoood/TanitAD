"""IDM reconstruction PROOF on a held-out PhysicalAI-val clip (+ 40-ep aggregate).

Purely-from-video reconstruction: frames_u8 -> frozen flagship-v1 encoder+readout
-> z -> persisted non-causal idm_head_v1 -> per-center {speed,yaw_rate,steer,
long_accel} + 2 s ego trajectory. GT is CAN-derived kinematics in the episode
contract (poses,actions). Reuses the shipped modules only.

Outputs (eval pod /root/idmval/results/):
  recon_ep{IDX}.npz  centers, scal_pred/gt, traj_pred/gt, per-frame ade, poses,
                     maneuvers, decoded man/route, dead-reckoned vs GT global path
  recon_metrics.json single-clip metrics + 40-ep aggregate (reproduce the card)
"""
from __future__ import annotations
import argparse, json, sys, time, math
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/root/v4eval/stack")
sys.path.insert(0, "/root/v4eval/stack/scripts")
import idm_head as ih
import run_idm_proof as R

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
ENC = "/root/models/flagship-30k/ckpt.pt"
POLICY_WINDOW = 8


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ep(idx):
    d = torch.load(f"{VAL}/ep_{idx:05d}.pt", weights_only=False)
    return d


def preds_for(head, z, poses, actions, k, stride, device):
    """encode-cached z -> windows -> head preds + GT. Returns centers, sp, sg, tp, tg."""
    zf = z.float()
    Zw, sg, tg = ih.build_windows(zf, poses.float(), actions.float(),
                                  k=k, horizons=ih.DEFAULT_HORIZONS, stride=stride)
    centers = ih.valid_centers(zf.shape[0], k, ih.DEFAULT_HORIZONS, stride).numpy()
    sp, tp = [], []
    with torch.no_grad():
        for i in range(0, Zw.shape[0], 512):
            o = head(Zw[i:i+512].to(device))
            sp.append(o["scalars"].cpu()); tp.append(o["traj"].cpu())
    sp = torch.cat(sp) if sp else torch.zeros(0, 4)
    tp = torch.cat(tp) if tp else torch.zeros(0, 4, 2)
    return centers, sp.numpy(), sg.numpy(), tp.numpy(), tg.numpy()


def metrics(sp, sg, tp, tg):
    r2 = {ih.SCALAR_NAMES[j]: ih.r2_score(torch.tensor(sp[:, j]), torch.tensor(sg[:, j]))
          for j in range(4)}
    mae = {ih.SCALAR_NAMES[j]: float(np.abs(sp[:, j] - sg[:, j]).mean()) for j in range(4)}
    de = np.linalg.norm(tp - tg, axis=-1)          # [N,H]
    return {"n": int(sp.shape[0]), "r2": r2, "mae": mae,
            "ade_2s": float(de.mean()),
            "de_per_horizon": [float(x) for x in de.mean(0)],
            "ade_per_window_mean": float(de.mean(1).mean())}


def build_flagship(ckpt_path, device):
    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if getattr(cfg, "tactical_pred", None) is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    model = WorldModel(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in list(missing) + list(unexpected)
           if k.split(".")[0] in ("encoder", "readout", "strategic_policy", "tactical_policy")]
    assert not bad, f"policy/encoder weights did not load cleanly: {bad[:8]}"
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


@torch.no_grad()
def policy_decode(model, frames9_u8, centers, device, batch=24):
    routes, mans = {}, {}
    valid = [int(t) for t in centers if t - POLICY_WINDOW + 1 >= 0]
    for i in range(0, len(valid), batch):
        ch = valid[i:i+batch]
        fw = torch.stack([frames9_u8[t-POLICY_WINDOW+1:t+1] for t in ch]).to(device).float().div_(255.)
        states = model.encode_window(fw)
        follow = torch.zeros(len(ch), dtype=torch.long, device=device)
        sf = model.strategic_policy(states, follow)
        r = sf["route_logits"].argmax(-1).cpu().tolist()
        tf = model.tactical_policy(states, sf["ctx"])
        m = tf["maneuver_logits"].argmax(-1).cpu().tolist()
        for j, t in enumerate(ch):
            routes[t], mans[t] = int(r[j]), int(m[j])
    return routes, mans


def deadreckon(centers, sp, poses):
    """Integrate per-center IDM (speed,yaw_rate) -> global path; GT from poses.
    Dense stride-1 centers assumed (consecutive). Anchored at GT start pose."""
    v = sp[:, 0]; yr = sp[:, 1]; dt = ih.DT
    t0 = int(centers[0])
    x = float(poses[t0, 0]); y = float(poses[t0, 1]); yaw = float(poses[t0, 2])
    xs, ys = [x], [y]
    for i in range(1, len(centers)):
        step = int(centers[i] - centers[i-1])
        yaw = yaw + yr[i-1] * dt * step
        x = x + v[i-1] * math.cos(yaw) * dt * step
        y = y + v[i-1] * math.sin(yaw) * dt * step
        xs.append(x); ys.append(y)
    pred = np.stack([xs, ys], 1)
    gt = poses[centers, :2].numpy()
    err = np.linalg.norm(pred - gt, axis=1)
    return pred, gt, {"endpoint_err_m": float(err[-1]),
                      "path_rmse_m": float(np.sqrt((err**2).mean())),
                      "path_len_m": float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, default=9)
    ap.add_argument("--head", default="/root/idmval/idm_head_v1.pt")
    ap.add_argument("--out", default="/root/idmval/results")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    enc, ro, emeta = R.load_encoder(ENC, dev)
    hd = torch.load(args.head, map_location="cpu", weights_only=False)
    head = ih.IDMHead(**hd["config"]["head_kwargs"]).to(dev); head.load_state_dict(hd["state_dict"]); head.eval()
    k = hd["config"]["window_k"]
    log(f"head {hd['config']['name']} k={k} params={hd.get('params')}")

    result = {"meta": {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "encoder_ckpt": ENC, "encoder_step": emeta["ckpt_step"],
                       "state_dim": emeta["state_dim"], "head": args.head,
                       "head_md5_expected": "fa4462f0b898b036be729c790278b823",
                       "val_cache": VAL, "k": k,
                       "horizons_s": [h*ih.DT for h in ih.DEFAULT_HORIZONS]}}

    # ---- single held-out clip (dense stride 1) ----
    d = load_ep(args.idx)
    fr = d["frames_u8"]; poses = d["poses"].float(); actions = d["actions"].float()
    man_gt = d.get("maneuvers"); epid = int(d.get("episode_id", -1))
    z = R.encode_frames(enc, ro, fr, dev, batch=32)
    log(f"ep_{args.idx:05d} epid={epid} T={fr.shape[0]} z{tuple(z.shape)}")
    centers, sp, sg, tp, tg = preds_for(head, z, poses, actions, k, 1, dev)
    m = metrics(sp, sg, tp, tg)
    ade_pw = np.linalg.norm(tp - tg, axis=-1).mean(1)   # per-window ADE
    log(f"clip ep_{args.idx:05d}: n={m['n']} speed_r2={m['r2']['speed']:.3f} "
        f"yaw_r2={m['r2']['yaw_rate']:.3f} ade_2s={m['ade_2s']:.3f} "
        f"de@h={[round(x,2) for x in m['de_per_horizon']]}")

    # policy HUD (best-effort)
    routes = mans = None
    try:
        fm = build_flagship(ENC, dev)
        routes, mans = policy_decode(fm, fr, centers, dev)
        del fm; torch.cuda.empty_cache()
        log(f"policy decode ok ({len(mans)} centers)")
    except Exception as e:
        log(f"policy decode SKIPPED: {type(e).__name__}: {e}")

    pred_g, gt_g, dr = deadreckon(centers, sp, poses)
    log(f"dead-reckon full-route: endpoint_err={dr['endpoint_err_m']:.2f}m "
        f"path_rmse={dr['path_rmse_m']:.2f}m over path_len={dr['path_len_m']:.1f}m")

    np.savez(f"{args.out}/recon_ep{args.idx:05d}.npz",
             centers=centers, scal_pred=sp, scal_gt=sg, traj_pred=tp, traj_gt=tg,
             ade_per_window=ade_pw, poses=poses.numpy(),
             maneuvers=(man_gt.numpy() if man_gt is not None else np.full(fr.shape[0], -1)),
             dec_route=np.array([routes.get(int(t), -1) if routes else -1 for t in centers]),
             dec_man=np.array([mans.get(int(t), -1) if mans else -1 for t in centers]),
             dr_pred=pred_g, dr_gt=gt_g, episode_id=epid)
    result["clip"] = {"idx": args.idx, "episode_id": epid, "T": int(fr.shape[0]),
                      "stride": 1, "metrics": m, "deadreckon_fullroute": dr}

    # ---- 40-ep aggregate (stride 2, reproduce card val_parityval) ----
    if args.aggregate:
        AS, AG, TP, TG = [], [], [], []
        for i in range(40):
            di = load_ep(i)
            zi = R.encode_frames(enc, ro, di["frames_u8"], dev, batch=32)
            _, spi, sgi, tpi, tgi = preds_for(head, zi, di["poses"].float(),
                                              di["actions"].float(), k, 2, dev)
            AS.append(spi); AG.append(sgi); TP.append(tpi); TG.append(tgi)
        sp2 = np.concatenate(AS); sg2 = np.concatenate(AG)
        tp2 = np.concatenate(TP); tg2 = np.concatenate(TG)
        agg = metrics(sp2, sg2, tp2, tg2); agg["n_clips"] = 40
        log(f"AGG 40ep stride2: n={agg['n']} speed_r2={agg['r2']['speed']:.4f} "
            f"yaw_r2={agg['r2']['yaw_rate']:.4f} ade_2s={agg['ade_2s']:.4f}")
        result["aggregate_40ep_stride2"] = agg
        result["card_val_parityval_expected"] = {
            "n": 3517, "speed_r2": 0.8853, "yaw_r2": 0.8075, "ade_2s": 2.7032}

    Path(f"{args.out}/recon_metrics.json").write_text(json.dumps(result, indent=2))
    log(f"WROTE {args.out}/recon_metrics.json")
    print("IDMVAL_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
