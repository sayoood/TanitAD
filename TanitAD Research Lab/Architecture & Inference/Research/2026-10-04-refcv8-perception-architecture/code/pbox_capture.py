"""WP-D P-BOX capture (PREREG_WPD_PROBES.md, registered sha256 c054190b…): the FROZEN box memory of refcv7-50,400.

ONE GPU pass of the run's own model (eval loader, STRICT, eval mode, no_grad) through the trainer's own
`compute_losses_v3` over ONE window set; read-only capture, nothing in the model changes:
  * a forward hook on `model._perception.box_mem` -> its OUTPUT (the decoder memory [B, 2144, 256]: 1,664 stride-16
    image tokens then 480 pooled BEV tokens) and its INPUT `bev` (the 0.5 m planner BEV [B, 96, 120, 64]);
  * a wrapper on the trainer's `_perc.box3d_loss_row` -> the labelled rows' targets (`tgt`), the VIS-1 block (`vis`)
    and the LIVE slots (presence logit + box of the last layer, for the cache-fidelity control C-cache);
  * `model._rig_camera.for_episodes(agent_ep)` -> each labelled row's own camera (R_cam_to_rig, t_cam_in_rig) for
    arm PB7.
Stored per LABELLED window (`batch["agent_label"]`, the trainer's own selection): memory as int8 affine per
(window, source block, channel) [image | BEV]; bev_feats int8 affine per (window, channel); targets truncated to the
last valid row; sha12 + t; camera; live slot outputs. Shards of <= 256 windows, written atomically (tmp -> rename),
resumable. Also `decoder.pt` (the box decoder, reference points, query-position embed and loss constants, from the
SAME built model) once per output dir.
Window sets: --split eval  -> EVAL-DIAG (the diagnostics' rule: episodes sorted by sha12, 8 evenly spaced windows)
             --split train -> the registered TRAIN set (REFCV6_REMAP_OVERRIDES = remap_train_pbox600.json: 600 clips,
                              5 evenly spaced windows each).
             --split train_diag -> TRAIN-DIAG, the registered P = R gate fit set (REFCV6_REMAP_OVERRIDES = the
                              diagnostics' remap_train_diag.json: 139 clips, 8 evenly spaced windows each).
⛔ Prints sha12 only. Env (caller): REFCV6_REPO, REFCV6_KIT, PYTHONPATH=<tree>/stack, [REFCV6_REMAP_OVERRIDES].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def q8(x, dim_reduce):
    """int8-affine over `dim_reduce` (min/max kept per remaining index) -> (uint8, lo fp32, scale fp32)."""
    import torch
    lo = x.amin(dim=dim_reduce, keepdim=True)
    hi = x.amax(dim=dim_reduce, keepdim=True)
    sc = ((hi - lo) / 255.0).clamp_min(1e-12)
    q = ((x - lo) / sc).round().clamp(0, 255).to(torch.uint8)
    return q, lo.float(), sc.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["eval", "train", "train_diag"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-dir", default="/home/nvidia/refcv7_run/runs/refcv7-r101-s0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-per-ep", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard", type=int, default=256)
    ap.add_argument("--part", type=int, default=0, help="contiguous part i of --nparts (a job must fit 40 min)")
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    K = a.k_per_ep or (5 if a.split == "train" else 8)

    import numpy as np
    import torch
    from tanitad.eval import refcv7_loader as L
    t0 = time.time()
    torch.manual_seed(0)
    tr = L.trainer()
    config = L.load_config(Path(a.run_dir) / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device="cuda", strict=True)
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config)
    sha_of_eid = dict(e_ds._vis1_sha12)
    out = Path(a.out) / a.split
    out.mkdir(parents=True, exist_ok=True)

    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    idx = []
    for e_i in sorted(by_ep, key=lambda e: sha_of_eid[int(e_ds.episodes[e].episode_id)]):
        ts = sorted(by_ep[e_i])
        for j in range(K):
            idx.append(ts[int((j + 0.5) * len(ts) / K)][1])
    win = [(sha_of_eid[int(e_ds.episodes[e_ds.index[i][0]].episode_id)], int(e_ds.index[i][1])) for i in idx]
    sel_sha = hashlib.sha256(json.dumps(win).encode()).hexdigest()
    (out / "selection.json").write_text(json.dumps({"split": a.split, "k_per_ep": K, "n_windows": len(idx),
                                                    "n_episodes": len(by_ep), "windows_sha12_t": win,
                                                    "sha256": sel_sha}), encoding="utf-8")
    print(f"[pbox-cap] {a.split}: {len(idx)} windows / {len(by_ep)} episodes, sel {sel_sha[:12]}, "
          f"build {time.time() - t0:.0f}s", flush=True)
    n_all = len(idx)
    lo, hi = (n_all * a.part) // a.nparts, (n_all * (a.part + 1)) // a.nparts
    idx, win = idx[lo:hi], win[lo:hi]
    tag = "" if a.nparts == 1 else f"p{a.part}_"
    print(f"[pbox-cap] part {a.part}/{a.nparts}: windows [{lo}, {hi})", flush=True)

    br = model._perception
    dec_path = Path(a.out) / "decoder.pt"
    if not dec_path.exists():
        torch.save({"box_dec": br.box_dec, "box_refpts": br.box_refpts, "box_qpos": br.box_qpos,
                    "cls_class_weight": getattr(model, "_cls_class_weight", None),
                    "visible_filter": bool(getattr(model, "_box3d_visible_filter", True)),
                    "presence_loss": str(br.cfg.presence_loss), "vis1": bool(br.cfg.vis1),
                    "planner_grid": br.cfg.planner_grid, "n_img_tokens": int(br.box_mem.image_hw[0] *
                                                                              br.box_mem.image_hw[1]),
                    "image_hw_tokens": list(br.box_mem.image_hw), "bev_tokens_hw": list(br.box_mem.bev_tokens_hw),
                    "frame": {"height": 416, "width": 1024, "f_ref": 488.92398517830253,
                              "projection": "cylindrical"},
                    "step": mrec["state_dict"]["step"], "ckpt": a.ckpt}, str(dec_path) + ".tmp")
        os.replace(str(dec_path) + ".tmp", dec_path)
    n_img = int(br.box_mem.image_hw[0] * br.box_mem.image_hw[1])

    CAP = {}

    def mem_hook(mod, inp, outp):
        CAP["mem"] = outp.detach()
        CAP["bev"] = inp[1].detach() if len(inp) > 1 and inp[1] is not None else None
    h = br.box_mem.register_forward_hook(mem_hook)
    orig_row = tr._perc.box3d_loss_row

    def row_wrapped(slots, tgt, **kw):
        CAP["slots_pl"] = slots["presence_logit"].detach()
        CAP["slots_box"] = slots["box"].detach()
        CAP["tgt"] = {k: v.detach() for k, v in tgt.items()}
        CAP["vis"] = {k: v.detach() for k, v in (kw.get("vis") or {}).items()}
        CAP["kw"] = {k: v for k, v in kw.items() if k != "vis"}
        return orig_row(slots, tgt, **kw)
    tr._perc.box3d_loss_row = row_wrapped

    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, idx), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, drop_last=False)
    buf, shard_i, n_lab, n_done = [], 0, 0, 0
    pos = 0
    model.eval()
    t_loop = time.time()

    def flush(buf, shard_i):
        p = out / f"shard_{tag}{shard_i:04d}.pt"
        torch.save(buf, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)

    with torch.no_grad():
        for bi, b in enumerate(dl):
            B = int(b["agent_valid"].shape[0])
            wins_b = win[pos:pos + B]
            pos += B
            CAP.clear()
            tr.compute_losses_v3(model, b, "cuda", mode=args.mode, ablate_frames=args.ablate_frames)
            keep = b.get("agent_label")
            sel = (keep.nonzero(as_tuple=False).flatten().tolist() if keep is not None else list(range(B)))
            if not sel:
                continue
            if "tgt" not in CAP or int(CAP["tgt"]["valid"].shape[0]) != len(sel):
                raise SystemExit(f"[pbox-cap] batch {bi}: captured targets do not align with the labelled rows")
            cams = model._rig_camera.for_episodes(b["agent_ep"].index_select(0, torch.as_tensor(sel)))
            for j, r in enumerate(sel):
                mem = CAP["mem"][r].float()                      # [2144, 256]
                qi, loi, sci = q8(mem[:n_img], 0)
                qb, lob, scb = q8(mem[n_img:], 0)
                bev = CAP["bev"][r].float()                      # [96, 120, 64]
                qv, lov, scv = q8(bev.reshape(bev.shape[0], -1), 1)
                v = CAP["tgt"]["valid"][j]
                A = int(v.nonzero().max()) + 1 if bool(v.any()) else 0
                rec = {"sha12": wins_b[r][0], "t": wins_b[r][1],
                       "mem_img_q": qi.cpu(), "mem_img_lo": loi.cpu(), "mem_img_sc": sci.cpu(),
                       "mem_bev_q": qb.cpu(), "mem_bev_lo": lob.cpu(), "mem_bev_sc": scb.cpu(),
                       "bev_q": qv.reshape(bev.shape).cpu(), "bev_lo": lov.cpu(), "bev_sc": scv.cpu(),
                       "tgt": {k: t_[j][:A].cpu() for k, t_ in CAP["tgt"].items()},
                       "vis": {k: t_[j][:A].cpu() for k, t_ in CAP["vis"].items()},
                       "R_cam_to_rig": torch.as_tensor(cams[j].R_cam_to_rig, dtype=torch.float32).cpu(),
                       "t_cam_in_rig": torch.as_tensor(cams[j].t_cam_in_rig, dtype=torch.float32).cpu(),
                       "live_presence_logit": CAP["slots_pl"][j].float().cpu(),
                       "live_box": CAP["slots_box"][j].half().cpu()}
                buf.append(rec)
                n_lab += 1
                if len(buf) >= a.shard:
                    flush(buf, shard_i)
                    buf, shard_i = [], shard_i + 1
            n_done = pos
            if bi % 20 == 0:
                el = time.time() - t_loop
                print(f"[pbox-cap] ZZ{pos}-{len(idx)}ZZ labelled {n_lab} {el:.0f}s "
                      f"cuda_max {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
    if buf:
        flush(buf, shard_i)
        shard_i += 1
    h.remove()
    tr._perc.box3d_loss_row = orig_row
    rec = {"split": a.split, "part": a.part, "nparts": a.nparts, "window_range": [lo, hi], "n_windows": len(idx), "n_windows_done": n_done, "n_labelled": n_lab,
           "n_shards": shard_i, "selection_sha256": sel_sha, "ckpt": a.ckpt, "step": mrec["state_dict"]["step"],
           "strict": mrec["state_dict"]["strict"], "loss_row_kwargs": {k: (v if isinstance(v, (bool, str, float, int))
                                                                          else type(v).__name__)
                                                                      for k, v in CAP.get("kw", {}).items()},
           "loader_departures": mrec.get("departures"), "quantisation": "int8 affine: memory per (window, "
           "image|BEV block, channel); bev_feats per (window, channel)", "wall_s": time.time() - t0,
           "cuda_max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
    (out / f"record{'' if a.nparts == 1 else '_p%d' % a.part}.json").write_text(json.dumps(rec, indent=1, default=str),
                                                                             encoding="utf-8")
    print(f"[pbox-cap] DONE {a.split}: {n_lab} labelled windows in {shard_i} shards, wall {rec['wall_s']:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
