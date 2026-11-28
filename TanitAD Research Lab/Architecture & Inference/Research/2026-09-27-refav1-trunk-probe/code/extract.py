"""Extract, per eval window, the frozen refav1 trunk state + a matched raw-DINOv3 floor + kinematics + GT + floors.

No planner, no training. Reuses refav1_arm's OWN loader / window selection / GT / floor builders (imported),
so every quantity is the arm's definition. Output: one npz per episode under --out.
"""
import argparse, importlib.util, os, sys, time
import numpy as np


def load_arm(path):
    spec = importlib.util.spec_from_file_location("refav1_arm", path)
    m = importlib.util.module_from_spec(spec); sys.modules["refav1_arm"] = m; spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-tool", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--cache", required=True); ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", default=None); ap.add_argument("--nav", default=None)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--k6", type=int, default=30)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--lru", type=int, default=8)
    ap.add_argument("--spatial", action="store_true")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    ra = load_arm(a.arm_tool)
    # build_loader reads exactly these five attributes (refav1_arm.py:672-682)
    from types import SimpleNamespace
    ns = SimpleNamespace(cache=a.cache, episodes=a.episodes, lru=a.lru, labels=a.labels, nav=a.nav)
    model, cfg, prov = ra.load_model(a.ckpt, a.config, a.device, False)
    model.eval()
    k = 10; K6 = int(a.k6)
    names = ra.episode_names(a.cache)
    ld = ra.build_loader(ns, cfg, max(k, int(cfg.op_steps)), names)
    sel, _ = ra.select_windows(ld, a.stride, None)
    by_ep = {}
    for i, (wi, ei, t) in enumerate(sel):
        by_ep.setdefault(ei, []).append((i, wi, t))
    os.makedirs(a.out, exist_ok=True)
    DT = ra.DT
    print(f"[extract] model step={prov['step']} d_state={cfg.d_state} W={ld.W} windows={len(sel)} "
          f"episodes={len(by_ep)} stride={a.stride} DT={DT}", flush=True)
    t0 = time.time(); done = 0
    eps = sorted(by_ep)
    if a.max_episodes:
        eps = eps[: a.max_episodes]
    for fi, ei in enumerate(eps):
        nm = ld.names[ei]
        o = torch.load(ld.episode_dir / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        poses = o["poses"].float()
        F_ep, v_ep, kap_ep = ld._episode(nm)
        rec = {k_: [] for k_ in ("trunk", "raw1", "raw2", "raw3", "kin", "g", "g6", "g6_ok", "v0",
                                  "ha", "ha0", "ha0_ext", "ol", "ha0_6", "ha0_ext_6", "ws")}
        for (i, wi, t) in by_ep[ei]:
            ld._order, ld._cursor = [wi], 0
            b = ld.batch(1)
            feats = b["feats"].to(a.device)                       # [1,W,N,d]
            act = b["actions"].to(a.device)
            v0 = float(v_ep[2 * t])
            g = ra.gt_waypoints(poses, t, k)
            try:
                g6 = ra.gt_waypoints(poses, t, K6); g6_ok = True
            except Exception:
                g6 = torch.full((1, K6, 2), float("nan")); g6_ok = False
            if g6.shape[-2] != K6 or not torch.isfinite(g6).all():
                g6 = torch.full((1, K6, 2), float("nan")); g6_ok = False
            ol = ra.paths_from_controls(act[0], v0, DT, k)
            hold = ra.hold_action_controls(ld, v_ep, kap_ep, t)
            ha = ra.paths_from_controls(hold[None].expand(k, 2), v0, DT, k)
            ha0 = ra.paths_from_controls(ra.hold_v0_controls(k), v0, DT, k)
            ext = ra.hold_ext_controls(ld, v_ep, kap_ep, t)
            ha0_ext = ra.paths_from_controls(ext[None].expand(k, 2), v0, DT, k)
            ha0_6 = ra.paths_from_controls(ra.hold_v0_controls(K6), v0, DT, K6)
            ha0_ext_6 = ra.paths_from_controls(ext[None].expand(K6, 2), v0, DT, K6)
            field = model.encode(feats)                            # [1,W,N,d_state]
            pooled = field.mean(dim=-2)[:, -1]                     # EXACTLY plan()'s `pooled`
            if a.spatial:
                # A1: 4x10 regions of the row-major 16x40 token grid (4x4 tokens each)
                def sp(x):                                          # [N=640, d] -> [40, d]
                    return x.reshape(4, 4, 10, 4, -1).mean(dim=(1, 3)).reshape(40, -1)
                rec.setdefault("trunk_sp", []).append(sp(field[0, -1].float()).cpu().numpy().astype(np.float16))
                rec.setdefault("raw_sp1", []).append(sp(feats[0, -1].float()).cpu().numpy().astype(np.float16))
                rec.setdefault("raw_sp2", []).append(sp(feats[0, -2].float()).cpu().numpy().astype(np.float16))
            fr = feats.float().mean(dim=-2)                        # [1,W,d_enc] matched mean pooling
            rec["trunk"].append(pooled[0].float().cpu().numpy())
            rec["raw1"].append(fr[0, -1].cpu().numpy())
            rec["raw2"].append(fr[0, -2].cpu().numpy() if fr.shape[1] >= 2 else fr[0, -1].cpu().numpy())
            rec["raw3"].append(fr[0, -3].cpu().numpy() if fr.shape[1] >= 3 else fr[0, -1].cpu().numpy())
            rec["kin"].append(np.array([v0, float(ext[0]), float(ext[1])], dtype=np.float32))
            for key, val in (("g", g), ("g6", g6), ("ha", ha), ("ha0", ha0), ("ha0_ext", ha0_ext), ("ol", ol),
                             ("ha0_6", ha0_6), ("ha0_ext_6", ha0_ext_6)):
                rec[key].append(np.asarray(val.float().cpu().numpy()).reshape(-1, 2)[-(K6 if key.endswith("6") else k):])
            rec["g6_ok"].append(g6_ok); rec["v0"].append(v0); rec["ws"].append(int(t))
        out = {kk: np.asarray(v) for kk, v in rec.items()}
        out["eid"] = np.array([ei]); out["clip_index"] = np.array([ei])
        np.savez(os.path.join(a.out, f"ep{fi:03d}.npz"), **out)
        done += len(by_ep[ei])
        if fi < 3 or fi % 10 == 0:
            el = time.time() - t0
            print(f"[extract] ep {fi+1}/{len(eps)} windows={done} {el:.0f}s ({el/max(done,1):.2f} s/window)", flush=True)
    print(f"ZZEXTRACT-DONE windows={done} episodes={len(eps)} {time.time()-t0:.0f}sZZ", flush=True)


if __name__ == "__main__":
    main()
