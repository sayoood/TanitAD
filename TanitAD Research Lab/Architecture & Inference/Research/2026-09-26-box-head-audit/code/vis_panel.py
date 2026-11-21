#!/usr/bin/env python3
"""vis_panel.py -- box-head audit task 0b: the PI's frames, with every GT TARGET coloured by how much of it the
camera can see (vis_zbuf.zbuffer), drawn with the PATCHED border-clipped edges, plus the z-buffer ownership map.
Also the edge census on every GT-validation window: how many edges the OLD both-corners rule drew vs the edges
that actually have an in-frame part (the "one surface" defect, counted).

Runs on Thor (tanitad-train). sha12 only in every output.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vis_zbuf as VZ  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="/home/nvidia/bha_2325/tree")
    ap.add_argument("--code", default="/home/nvidia/bha_2325/code")
    ap.add_argument("--patched-rmv", default="/home/nvidia/bha_2325/patch/render_refcv6_map_video.py")
    ap.add_argument("--calib-dir", default="/home/nvidia/bha_2325/ov/calib_ship")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panels", default="9f8bedcfb9de:104,0191487845ef:64,34765c024267:108")
    ap.add_argument("--clips", default="0191487845ef,9f8bedcfb9de,34765c024267,22d47ae7e052,59d98b34ae8b")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    os.environ["REFCV6_REPO"] = a.tree
    sys.path.insert(0, a.code)
    import refcv6_loader as L
    L.bootstrap()
    import torch
    from PIL import Image, ImageDraw
    from tanitad.data import v2_dataset as v2d
    from tanitad.data.calib import CanonicalFrame
    from tanitad.data.physicalai import FrontWideExtrinsics
    from tanitad.data.rig_projection import RigCamera
    spec = importlib.util.spec_from_file_location("rmv_patched", a.patched_rmv)
    RMV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(RMV)
    tr = L.trainer()
    config = L.load_config("/home/nvidia/refcv6_run/runs/refcv6-r101-s0/config.json")
    model, cfg, targs, _ = L.build_model(config, "/home/nvidia/refcv6_run/runs/refcv6-r101-s0/ckpt.pt",
                                         device="cpu", remap={})
    del model
    table = json.load(open("/home/nvidia/data/refcv6_train_eval139_extrinsics.json", encoding="utf-8"))
    by12 = {VZ.sha12(c): c for c in table}
    keep = {by12[s] for s in a.clips.split(",") if s in by12}

    def clip_of(ep):
        from pathlib import Path
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]
    _orig = v2d.build_v2_providers
    v2d.build_v2_providers = lambda *aa, **kk: [e for e in _orig(*aa, **kk) if clip_of(e) in keep]

    class Win(tr.V3Dataset):
        u8_frames = True

        def _window_u8(self, i):
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w],
                    "future_frames": torch.zeros(0, dtype=torch.uint8),
                    "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                    "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                    "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}
    try:
        e_ds, e_eps, _ = L.build_eval_dataset(None, cfg, targs, config, with_perception_targets=True,
                                              dataset_cls=Win)
    finally:
        v2d.build_v2_providers = _orig
    Wn = int(cfg.core.window)
    frame = CanonicalFrame(height=VZ.H, width=VZ.W, f_ref=VZ.F, projection="cylindrical")
    wins = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        wins.setdefault(VZ.sha12(clip_of(e_eps[e_i])), []).append((int(t), wi, e_i))
    cams = {}
    census = {"n_targets": 0, "n_targets_any_edge_in_frame": 0, "old_lt12_new_gt_old": 0, "old_le4_new_gt4": 0,
              "old_zero_new_pos": 0, "by_clip": {}}
    panels = {tuple(p.split(":")) for p in a.panels.split(",")}
    out_rec = {"panels": []}
    for s12, lst in sorted(wins.items()):
        cid = by12[s12]
        e = table[cid]
        rcam = RigCamera.from_extrinsics(FrontWideExtrinsics(qx=float(e["qx"]), qy=float(e["qy"]), qz=float(e["qz"]),
                                                             qw=float(e["qw"]), x=float(e["x"]), y=float(e["y"]),
                                                             z=float(e["z"])), frame)
        R = VZ.R_from_quat(float(e["qx"]), float(e["qy"]), float(e["qz"]), float(e["qw"]))
        tt = np.array([float(e["x"]), float(e["y"]), float(e["z"])])
        import pandas as pd
        ch = int(e.get("chunk", -1))
        it = pd.read_parquet(f"{a.calib_dir}/camera_intrinsics/camera_intrinsics.chunk_{ch:04d}.parquet").reset_index()
        it = it[(it["clip_id"] == cid) & (it["camera_name"] == "camera_front_wide_120fov")].iloc[0]
        obs = VZ.observed_mask(tuple(float(it[f"fw_poly_{i}"]) for i in range(5)), float(it.cx), float(it.cy),
                               int(it.width), int(it.height))
        cc = {"n_targets": 0, "old_lt12_new_gt_old": 0, "old_le4_new_gt4": 0}
        for k, (t, wi, e_i) in enumerate(sorted(lst)):
            ep = e_eps[e_i]
            item = e_ds._agent_item(ep, t + Wn - 1)
            if not bool(item["agent_label"]):
                continue
            v = item["agent_valid"].numpy().astype(bool)
            idx = np.nonzero(v & item["agent_zh_mask"].numpy().astype(bool))[0]
            box = item["agent_box"].double().numpy()
            yaw = item["agent_yaw"].double().numpy()
            cz = item["agent_cz"].double().numpy()
            hh = item["agent_h"].double().numpy()
            cls = item["agent_cls"].numpy()
            bx = np.array([[box[j, 0], box[j, 1], cz[j], box[j, 2], box[j, 3], hh[j], yaw[j]] for j in idx]) \
                if len(idx) else np.zeros((0, 7))
            want_panel = (s12, str(k + 1)) in panels
            tg = [(0 <= box[j, 0] <= 60) and abs(box[j, 1]) <= 16 and
                  math.atan2(abs(box[j, 1]), box[j, 0]) <= math.radians(60) for j in idx]
            # edge census (old vs new drawing) on every target
            for ii, j in enumerate(idx):
                if not tg[ii]:
                    continue
                cor = RMV.cuboid_corners(box[j, 0], box[j, 1], cz[j], box[j, 2], box[j, 3], hh[j], yaw[j])
                c_, r_, ok = rcam.project(torch.as_tensor(cor, dtype=torch.float64))
                ok = ok.numpy().astype(bool)
                n_old = sum(1 for p, q in RMV.CUBOID_EDGES if ok[p] and ok[q])
                n_new = len({ee for ee, _ in RMV.cuboid_edge_runs(rcam, cor)})
                census["n_targets"] += 1
                cc["n_targets"] += 1
                census["n_targets_any_edge_in_frame"] += int(n_new > 0)
                census["old_lt12_new_gt_old"] += int(n_new > n_old)
                cc["old_lt12_new_gt_old"] += int(n_new > n_old)
                census["old_le4_new_gt4"] += int(n_old <= 4 and n_new > 4)
                cc["old_le4_new_gt4"] += int(n_old <= 4 and n_new > 4)
                census["old_zero_new_pos"] += int(n_old == 0 and n_new > 0)
            if not want_panel:
                continue
            nf, ni, nv, vr, idb = VZ.zbuffer(bx, R, tt, obs)
            it2 = e_ds[wi]
            rgb = it2["frames"][-1, -3:].permute(1, 2, 0).contiguous().numpy()
            img = Image.fromarray(rgb).convert("RGB")
            d = ImageDraw.Draw(img, "RGBA")
            rows = []
            for ii, j in enumerate(idx):
                if not tg[ii]:
                    continue
                vf = (nv[ii] / nf[ii]) if nf[ii] else float("nan")
                col = (60, 220, 60) if vf >= 0.5 else ((250, 210, 40) if vf >= 0.1 else (240, 50, 50))
                cor = RMV.cuboid_corners(box[j, 0], box[j, 1], cz[j], box[j, 2], box[j, 3], hh[j], yaw[j])
                for _ee, run in RMV.cuboid_edge_runs(rcam, cor):
                    d.line(run, fill=col + (255,), width=2, joint="curve")
                rows.append({"cls": VZ.CLASSES[int(cls[j])] if 0 <= int(cls[j]) < 10 else "unknown",
                             "range_m": round(math.hypot(box[j, 0], box[j, 1]), 1), "vis_frac": round(float(vf), 3),
                             "n_vis_px": int(nv[ii]), "in_img": round(float(ni[ii] / nf[ii]), 3) if nf[ii] else None})
            own = idb[VZ.PAD_V:VZ.PAD_V + VZ.H, VZ.PAD_U:VZ.PAD_U + VZ.W]
            rng_ = np.random.default_rng(1)
            pal = rng_.integers(40, 255, size=(max(1, len(idx)), 3))
            om = np.zeros((VZ.H, VZ.W, 3), np.uint8)
            m = own >= 0
            om[m] = pal[own[m] % len(pal)]
            om[~obs] = (30, 30, 30)
            both = Image.new("RGB", (VZ.W, 2 * VZ.H + 30), (10, 12, 16))
            both.paste(img, (0, 0))
            both.paste(Image.fromarray(om), (0, VZ.H + 30))
            dd = ImageDraw.Draw(both)
            n_t = len(rows)
            n10 = sum(1 for r in rows if r["vis_frac"] < 0.1)
            dd.text((6, VZ.H + 8), f"{s12} window {k + 1}: {n_t} targets; green >=50 % visible, yellow 10-50 %, "
                                   f"red <10 % ({n10}); below: z-buffer owner of every pixel (GT cuboids only)",
                    fill=(230, 230, 230))
            pth = os.path.join(a.out_dir, f"vis_{s12}_w{k + 1:03d}.png")
            both.save(pth)
            out_rec["panels"].append({"sha12": s12, "window_1based": k + 1, "t": t, "png": os.path.basename(pth),
                                      "targets": sorted(rows, key=lambda r: r["range_m"])})
        census["by_clip"][s12] = cc
    out_rec["edge_census"] = census
    out_rec["wall_s"] = round(time.time() - t0, 1)
    json.dump(out_rec, open(os.path.join(a.out_dir, "vis_panels.json"), "w"), indent=1, default=float)
    print(json.dumps(census, indent=1), flush=True)
    for p in out_rec["panels"]:
        print(p["png"], [(r["cls"], r["range_m"], r["vis_frac"]) for r in p["targets"] if r["range_m"] < 16])


if __name__ == "__main__":
    main()
