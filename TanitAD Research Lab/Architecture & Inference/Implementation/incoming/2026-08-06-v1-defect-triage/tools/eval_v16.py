"""FULL EVAL: flagship v1.6 (unicycle readout) vs v1arch baseline — four families,
paired episode-cluster bootstrap, temporal stability, and the video dump.

⛔ ONE LATENT ROLL, TWO DECODES. Both arms decode the SAME rolled transitions from the
SAME frozen trunk, so every contrast is attributable to the decoder alone — the same
discipline as rollout_decode_unicycle's byte-identical roll.

⛔ NAME: the registry key is `flagship-v16-unicycle`. There is an OLDER `flagship-v16-ab-ft`
("v1.6", §1.4b, the retracted best-in-program claim). These are DIFFERENT models.
"""
import glob, json, math, os, time
import numpy as np
import torch

DT = 0.1
K = 20

def controls(path):
    p = np.concatenate([np.zeros((path.shape[0],1,2)), path], 1)
    d = p[:,1:]-p[:,:-1]
    ds = np.sqrt((d**2).sum(-1)+1e-12)
    sp = ds/DT
    acc = (sp[:,1:]-sp[:,:-1])/DT
    h = np.arctan2(d[...,1], d[...,0])
    dh = (h[:,1:]-h[:,:-1]+math.pi)%(2*math.pi)-math.pi
    ok = (ds[:,1:]>0.05)&(ds[:,:-1]>0.05)
    return sp, acc, np.where(ok, dh, 0.0).sum(1), dh, ok

def fam_row(P, G):
    sp_p, ac_p, ny_p, dh_p, ok_p = controls(P)
    sp_g, ac_g, ny_g, dh_g, ok_g = controls(G)
    jerk = (ac_p[:,1:]-ac_p[:,:-1])/DT
    ct = P[:,:,1]-G[:,:,1]
    both = ok_p & ok_g
    hp = np.where(both, dh_p, 0.0); hg = np.where(both, dh_g, 0.0)
    n_ok = max(int(both.sum()),1)
    return {
      "ade_m": float(np.linalg.norm(P-G, axis=-1).mean()),
      "speed_bias_mps": float((sp_p-sp_g).mean()),
      "speed_mae_mps": float(np.abs(sp_p-sp_g).mean()),
      "along_final_bias_m": float((P[:,-1,0]-G[:,-1,0]).mean()),
      "accel_rms_mps2": float(np.sqrt((ac_p**2).mean())),
      "accel_mae_mps2": float(np.abs(ac_p-ac_g).mean()),
      "jerk_rms_mps3": float(np.sqrt((jerk**2).mean())),
      "net_yaw_err_rad": float(np.abs(ny_p-ny_g).mean()),
      "heading_mae_rad": float(np.abs(hp-hg).sum()/n_ok),
      "cross_mae_m": float(np.abs(ct).mean()),
    }

def to_frame(pa, A, B):
    ca, sa = math.cos(A[2]), math.sin(A[2])
    wx = A[0] + pa[:,0]*ca - pa[:,1]*sa
    wy = A[1] + pa[:,0]*sa + pa[:,1]*ca
    dx, dy = wx-B[0], wy-B[1]
    cb, sb = math.cos(B[2]), math.sin(B[2])
    return np.stack([dx*cb+dy*sb, -dx*sb+dy*cb], 1)

def temporal(preds_by_ep, poses_by_ep, ws_by_ep):
    sh, cj = [], []
    for X, poses, ws in zip(preds_by_ep, poses_by_ep, ws_by_ep):
        for i in range(1, len(ws)):
            s = ws[i]-ws[i-1]
            if s<=0 or s>=K: continue
            n = K-s
            a = to_frame(X[i-1][s:s+n], poses[ws[i-1]], poses[ws[i]])
            b = X[i][:n]
            sh.append(float(np.linalg.norm(a-b, axis=-1).mean()))
            _, aa, _, _, _ = controls(a[None]); _, ab, _, _, _ = controls(b[None])
            cj.append(float(np.abs(aa[0,:-1]-ab[0,:-1]).mean()))
    return {"replan_shift_m_mean": float(np.mean(sh)),
            "replan_accel_jump_mps2_mean": float(np.mean(cj)), "n_pairs": len(sh)}

def boot_delta(per_ep_a, per_ep_b, keys, n=2000, seed=0):
    """paired episode-cluster bootstrap over per-episode means: b - a."""
    rng = np.random.default_rng(seed)
    E = len(per_ep_a)
    out = {}
    for k in keys:
        da = np.array([e[k] for e in per_ep_a]); db = np.array([e[k] for e in per_ep_b])
        d = db - da
        draws = [float(d[rng.integers(0,E,E)].mean()) for _ in range(n)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out[k] = {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
                  "separated": bool(lo>0 or hi<0), "n_episodes": E,
                  "estimator": "paired episode-cluster bootstrap, 2000 draws"}
    return out

def main():
    dev = "cuda"
    from taniteval import loaders
    from taniteval.data import load_frames
    from taniteval.rollout import SPEED_SCALE
    from tanitad.models.metric_dynamics import (UnicycleStepReadout, accumulate_se2,
                                                decode_transitions, gt_ego_waypoints,
                                                rollout_transitions)
    L = loaders.load({"arch":"flagship-worldmodel-v2",
        "ckpt":"/workspace/experiments/flagship-v1arch-v2bal-30k/ckpt.pt",
        "run_config":"/workspace/experiments/flagship-v1arch-v2bal-30k/config.json",
        "speed_input":True}, device=dev)
    model, sr = L["model"].eval(), L["step_readout"]
    for p in model.parameters(): p.requires_grad_(False)
    ck = torch.load("/workspace/experiments/unicycle-readout-v2-latentsonly/unicycle_readout.pt",
                    map_location="cpu", weights_only=False)
    head = UnicycleStepReadout(2048, hidden=512, speed_input=False,
                               predict_delta=False).to(dev)
    head.load_state_dict(ck["head"]); head.eval()
    print("[eval] models loaded", flush=True)

    files = sorted(glob.glob("/workspace/pai_epcache/physicalai-oodval-6f4b94e4c7ce-q90/ep_*.pt"))[:40]
    W = 8
    os.makedirs("/workspace/v16_eval/dump", exist_ok=True)
    pe_a, pe_b = [], []           # per-episode family rows
    A_eps, B_eps, PO_eps, WS_eps = [], [], [], []
    t0 = time.time()
    for fi, f in enumerate(files):
        ep = load_frames([f])[0]
        poses = ep.poses.float()
        T = min(ep.feats.shape[0], poses.shape[0], ep.actions.shape[0])
        starts = list(range(0, T-W-K))
        wsA, wsB, gts, lastl = [], [], [], []
        with torch.no_grad():
            for i0 in range(0, len(starts), 16):
                ch = starts[i0:i0+16]
                last = torch.tensor([s+W-1 for s in ch])
                fw = torch.stack([torch.as_tensor(ep.feats[s:s+W]) for s in ch]).to(dev).float().div_(255.0)
                aw = torch.stack([ep.actions[s:s+W] for s in ch]).to(dev).float()
                fa = torch.stack([ep.actions[s+W:s+W+K] for s in ch]).to(dev).float()
                pl = poses[last].to(dev)
                ego = (pl[:,3:4]/SPEED_SCALE)
                aw = torch.cat([aw, ego[:,None].expand(-1,aw.shape[1],-1)],-1)
                fa = torch.cat([fa, ego[:,None].expand(-1,fa.shape[1],-1)],-1)
                states = model.encode_window(fw)
                trans = rollout_transitions(model.predictor, states, aw, fa, K)
                wp_a, _ = decode_transitions(sr, trans, K)         # v1arch
                v = pl[:,3].clone(); ap_=torch.zeros_like(v); yp=torch.zeros_like(v)
                rows=[]
                for zp, zh in trans:
                    aj, yj = head(zp, zh, v, ap_, yp)
                    rows.append(torch.stack([v*DT, torch.zeros_like(v), yj*DT],-1))
                    v=(v+aj*DT).clamp_min(0.0); ap_, yp = aj, yj
                wp_b = accumulate_se2(torch.stack(rows,1))          # v1.6
                fp = torch.stack([poses[s+W:s+W+K] for s in ch]).to(dev)
                gt = gt_ego_waypoints(pl, fp, list(range(1,K+1)))
                wsA.append(wp_a.float().cpu().numpy()); wsB.append(wp_b.float().cpu().numpy())
                gts.append(gt.cpu().numpy()); lastl += [int(x) for x in last]
        A = np.concatenate(wsA); B = np.concatenate(wsB); G = np.concatenate(gts)
        pe_a.append(fam_row(A, G)); pe_b.append(fam_row(B, G))
        A_eps.append(A); B_eps.append(B); PO_eps.append(poses.numpy()); WS_eps.append(lastl)
        np.savez_compressed(f"/workspace/v16_eval/dump/ep{fi:03d}.npz",
                            a=A.astype(np.float32), b=B.astype(np.float32),
                            g=G.astype(np.float32), ws=np.array(lastl))
        if (fi+1)%10==0: print(f"  [{fi+1}/40] {time.time()-t0:.0f}s", flush=True)

    keys = list(pe_a[0])
    pooled_a = {k: float(np.mean([e[k] for e in pe_a])) for k in keys}
    pooled_b = {k: float(np.mean([e[k] for e in pe_b])) for k in keys}
    res = {
      "_arms": {"a":"flagship-v1arch-v2bal-30k (displacement readout)",
                "b":"flagship-v16-unicycle == v1.6 (UnicycleStepReadout, latents-only, 2.11M, frozen trunk)"},
      "_corpus": "physicalai-oodval-6f4b94e4c7ce-q90, 40 episodes, stride-1 rollout grid",
      "_confound": ("action-conditioned WM rollout under TRUE future actions — NOT closed-loop "
                    "planning; applies to BOTH arms equally"),
      "_evidence_class": "MEASURED (ours)",
      "n_windows": int(sum(len(w) for w in WS_eps)),
      "pooled": {"v1arch": pooled_a, "v16": pooled_b},
      "delta_v16_minus_v1arch": boot_delta(pe_a, pe_b, keys),
      "temporal": {"v1arch": temporal(A_eps, PO_eps, WS_eps),
                   "v16": temporal(B_eps, PO_eps, WS_eps)},
      "TACTICAL_note": ("policies untouched by v1.6 — declared-head metrics identical to "
                        "v1arch's banked panel; executed-direction differences are the "
                        "net_yaw/heading rows above"),
      "STRATEGIC": {"status":"UNAVAILABLE","reason":"no map in PhysicalAI-AV (unchanged)"},
      "distance_keeping": {"status":"UNAVAILABLE",
        "reason":"lead block artifact not present on this pod; rebuild is a standing work item"},
      "per_episode": {"v1arch": pe_a, "v16": pe_b},
    }
    json.dump(res, open("/workspace/v16_eval/v16_full_eval.json","w"), indent=1)
    print(json.dumps({"pooled_v1arch":pooled_a,"pooled_v16":pooled_b},indent=1))
    print("DELTAS (v16-v1arch):")
    for k,v in res["delta_v16_minus_v1arch"].items():
        print(f"  {k:22s} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] sep={v['separated']}")
    print("TEMPORAL:", json.dumps(res["temporal"]))
    print("EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
