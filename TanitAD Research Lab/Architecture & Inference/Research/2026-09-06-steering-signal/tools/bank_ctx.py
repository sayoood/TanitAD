"""D-GSTR-1 P1d/P2 -- bank the STRATEGIC CONTEXT `ctx` that `str_goal_head`
reads, on the same eval windows the banked dump scored.

Mechanism: a `register_forward_pre_hook` on `model.str_goal_head` captures its
INPUT and then raises a sentinel so the decoder never runs. No model file is
edited and no forward argument changes.

⛔ THE CONTROL THAT MAKES THE ABORT ADMISSIBLE: from the captured `ctx` this
script RECOMPUTES `g_str` exactly as `refc_v3.py:979-982` does and asserts it
matches the BANKED `gstr_nav_true` for the same (clip_index, window). If the
abort or the capture were wrong, that comparison cannot pass.

ASCII-only. Writes an .npz bank; no analysis here.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/nvidia/refcv4b/ckpt_40284_FINAL.pt")
    ap.add_argument("--config", default="/home/nvidia/refcv4b/config.json")
    ap.add_argument("--episodes", default="/home/nvidia/navpred/data_eval")
    ap.add_argument("--labels",
                    default="/home/nvidia/data/v72/labels/"
                            "s2_labels_v7.2_eval.jsonl.gz")
    ap.add_argument("--nav-source", default="v72")
    ap.add_argument("--episodes-n", type=int, default=0)
    ap.add_argument("--window-stride", type=int, default=5)
    ap.add_argument("--lru", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bank", default="/home/nvidia/navpred/navflip_dump")
    ap.add_argument("--out", default="/home/nvidia/navroute/CTX_BANK.npz")
    a = ap.parse_args()

    sys.path.insert(0, "/home/nvidia/navpred/taniteval/tools")
    import refcv3_arm as A                                       # noqa: E402
    A._bootstrap_paths()
    tr = A.trainer()
    from tanitad.data.lan import (LAN_FEATS_PER_ANCHOR,          # noqa: E402
                                  LanConfig, lan_window_features)

    model, cfg, targs, prov = A.load_model(a.ckpt, a.config, device=a.device)
    model.eval()
    eps, files, clip_ids, ds, lman, join, src, raw_off = A.build_corpus(
        a, cfg, prov)
    W = int(cfg.core.window)
    steps = int(prov["decoder_steps"])
    dev = a.device

    # ---- the capture ------------------------------------------------------
    class _Stop(Exception):
        pass

    grab = {}

    def pre(_mod, inp):
        grab["ctx"] = inp[0].detach().float().cpu().numpy()
        raise _Stop
    h = model.str_goal_head.register_forward_pre_hook(pre)

    Wt = model.str_goal_head.weight.detach().float().cpu().numpy()   # [3, d]
    bt = model.str_goal_head.bias.detach().float().cpu().numpy()     # [3]

    # ---- the banked g_str, for the control --------------------------------
    banked = {}
    for f in sorted(os.listdir(os.path.join(a.bank, "decisions"))):
        if not f.endswith(".npz"):
            continue
        top = os.path.join(a.bank, f)
        if not os.path.exists(top):
            continue
        t = np.load(top, allow_pickle=True)
        z = np.load(os.path.join(a.bank, "decisions", f), allow_pickle=True)
        if "clip_index" not in t.files or "gstr_nav_true" not in z.files:
            continue
        ci = int(np.asarray(t["clip_index"]).reshape(-1)[0])
        ws_b = np.asarray(z["ws"]).astype(np.int64).reshape(-1)
        g_b = np.asarray(z["gstr_nav_true"], dtype=np.float64)
        for i, w in enumerate(ws_b.tolist()):
            banked[(ci, int(w))] = g_b[i]

    # ---- windows ----------------------------------------------------------
    by_ep = {}
    for i, (e_i, t) in enumerate(ds.index):
        if (t % max(1, a.window_stride)) != 0:
            continue
        by_ep.setdefault(e_i, []).append((i, t))

    lcfg = LanConfig()
    k = lcfg.k
    CTX, GS, TGT, TVAL, EPI, WIN, NAV, GTY = [], [], [], [], [], [], [], []
    GTY6, GTV6 = [], []
    n_done = 0
    for e_i in sorted(by_ep):
        ep = eps[e_i]
        poses = np.asarray(ep.poses.float().cpu().numpy(), dtype=np.float64)
        ci = int(e_i)
        for (wi, t) in by_ep[e_i]:
            t0 = t + W - 1
            if t0 >= poses.shape[0]:
                continue
            item = ds[wi]
            fr = tr.frames_to_device(item["frames"][None], dev)
            v0_t = item["pose_last"].float()[3].reshape(1).to(dev)
            nav = item.get("nav_cmd")
            nav_t = (nav.reshape(1).long().to(dev)
                     if nav is not None else None)
            ego = None
            if bool(cfg.ego_state_inject):
                ego = tr.v3.ego_state_from_batch(
                    {"pose_last": item["pose_last"].float()[None],
                     "actions": item["actions"].float()[None]}, device=dev)
            grab.clear()
            try:
                with torch.no_grad():
                    model(fr, nav_cmd=nav_t, v0=v0_t, steps=steps,
                          ego_state=ego)
            except _Stop:
                pass
            if "ctx" not in grab:
                raise SystemExit("[bank_ctx] the pre-hook never fired -- "
                                 "str_goal_head was not reached; refusing")
            c = grab["ctx"][0]
            g = Wt @ c + bt
            n = float(np.linalg.norm(g[:2]))
            gs = np.array([g[0] / max(n, 1e-6), g[1] / max(n, 1e-6),
                           np.tanh(g[2])], dtype=np.float64)
            feats = lan_window_features(poses, t0, lcfg)
            f4 = np.asarray(feats, dtype=np.float64).reshape(
                k, LAN_FEATS_PER_ANCHOR)
            valid = f4[:, 3] > 0.5
            first = int(np.argmax(valid.astype(np.float64)))
            pk = f4[first]
            nn_ = np.linalg.norm(pk[:2])
            bear = pk[:2] / max(nn_, 1e-6)
            dpref = first / max(k - 1, 1) * 2.0 - 1.0
            # GT terminal lateral offset, at BOTH horizons. ⛔ 2 s and 6 s are
            # DIFFERENT questions and give different LEFT counts; the brief's
            # 1,597 GT-left windows are the 6 s number, so the gate uses 6 s
            # and the 2 s value ships beside it rather than being confused
            # for it.
            fpe = item.get("future_poses_ext")
            gy = gy6 = np.nan
            v6 = False
            if fpe is not None:
                fve = item.get("future_valid_ext")
                tgtxy = tr.refb_labels.waypoint_targets(
                    item["pose_last"].float()[None],
                    item["future_poses_ext"].float()[None], [20, 60])
                gy = float(tgtxy[0, 0, 1])
                gy6 = float(tgtxy[0, 1, 1])
                v6 = bool(fve[59]) if (fve is not None
                                       and len(fve) >= 60) else False
            CTX.append(c.astype(np.float32))
            GS.append(gs)
            TGT.append([bear[0], bear[1], dpref])
            TVAL.append(bool(valid.any()))
            EPI.append(ci)
            WIN.append(t0)
            NAV.append(int(nav) if nav is not None else -1)
            GTY.append(gy)
            GTY6.append(gy6)
            GTV6.append(v6)
            n_done += 1
        print(f"# ep {e_i} done, windows={n_done}", flush=True)

    h.remove()
    CTX = np.stack(CTX); GS = np.array(GS); TGT = np.array(TGT)
    TVAL = np.array(TVAL); EPI = np.array(EPI); WIN = np.array(WIN)
    NAV = np.array(NAV); GTY = np.array(GTY)
    GTY6 = np.array(GTY6); GTV6 = np.array(GTV6, dtype=bool)

    # ---- THE CONTROL: recomputed g_str must match the BANKED one ----------
    keys = [(int(EPI[i]), int(WIN[i])) for i in range(len(EPI))]
    have = np.array([k2 in banked for k2 in keys])
    ctrl = {"n_matched_keys": int(have.sum()), "n_windows": int(len(keys))}
    if have.any():
        gb = np.stack([banked[keys[i]] for i in range(len(keys)) if have[i]])
        gm = GS[have]
        d = np.abs(gb - gm).max(axis=1)
        ctrl.update(max_abs_diff=round(float(d.max()), 6),
                    median_abs_diff=round(float(np.median(d)), 8),
                    frac_within_1e3=round(float((d < 1e-3).mean()), 4),
                    PASS=bool(float(d.max()) < 1e-2))
    else:
        ctrl["PASS"] = False
        ctrl["ERROR"] = "no key overlap with the banked dump"

    np.savez_compressed(a.out, ctx=CTX, g_str=GS, target=TGT, tvalid=TVAL,
                        ep=EPI, win=WIN, nav=NAV, gt_y=GTY, gt_y6=GTY6, gt_valid6=GTV6,
                        W=Wt, b=bt)
    rep = {"tool": "D-GSTR-1 bank_ctx", "out": a.out,
           "n_windows": int(CTX.shape[0]), "d_ctx": int(CTX.shape[1]),
           "n_episodes": int(np.unique(EPI).size),
           "decoder_steps": steps, "window": W,
           "CONTROL_recomputed_gstr_vs_banked": ctrl}
    print(json.dumps(rep, indent=1), flush=True)
    with open(a.out + ".report.json", "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    main()
