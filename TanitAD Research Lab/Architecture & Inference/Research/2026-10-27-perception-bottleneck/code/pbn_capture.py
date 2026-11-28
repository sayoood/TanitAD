"""pbn_capture.py -- PERC-BOTTLENECK (PREREG_PERC_BOTTLENECK.md) Q1 feature capture on the FROZEN refcv8 EMA-48,852.

One forward per window through the trainer's own `compute_losses_v3` (the perc_pass_r7 path, eval mode, no_grad,
future frames zeroed), with forward hooks on `model._map_hires` (no model code changed). Per window it writes ONE
npz (uncompressed) to --out-dir with, on the corridor x in [0, 40) m:

  enc   [64, 160, 32]  fp16  HiresBEVEncoder output, 0.25 m rows 0..159, cols 104..135 (|y| < 4 m)
  lift  [64, 160, 32]  fp16  lift output (before the BEV encoder), same cells
  img   [2048, 80, 12] fp16  fmap_s8 sampled at the 4 lift heights (height-major, valid-masked) at the EVAL cells
                             (rows 0,2,..,158 ; cols 108,110,..,130 = |y| < 3 m)
  logit [8, 160, 32]   fp16  the head's 10 cm logits at the 10 cm cell containing each 0.25 m cell centre
  dec   [160, 32]      u8    the head's DECLARED decision (prior_corrected rule, its own class weights) there
  gt    [160, 32]      u8    the SAM3 10 cm GT code there (255 = not seen)
  lv    [160, 32]      bool  lift-valid (any height projects into the image)

Clip ids never leave this process: the file name and the record carry sha12 only.

    set R8_LAUNCH_TREE=C:/Users/Admin/refcv8_pf/gtmap_oracle/tree_fb47a58 & set R8_NO_POSE_SYNC=1 & set OMP_NUM_THREADS=4
    python -X utf8 pbn_capture.py --split eval --remap remap_eval_devbox.json --out-dir F:/perc_bottleneck/cap_eval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
R8_CODE = Path(os.environ.get("L2_R8_CODE", "C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                                            "2026-10-15-refcv8-milestone-battery/code"))
ROW_N, COL0, COL_N = 160, 104, 32            # patch: 0.25 m rows 0..159 (x 0-40 m), cols 104..135 (|y| < 4 m)
EV_ROWS = list(range(0, 160, 2))             # eval cells (rows of the patch)
EV_COLS = list(range(108, 132, 2))           # |y| < 3 m, absolute 0.25 m columns
QUANT = (0.125, 0.375, 0.625, 0.875)


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("eval", "train"), required=True)
    ap.add_argument("--remap", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default="E:/refcv8_eval_kit/ckpt/ckpt_ema_48852.pt")
    ap.add_argument("--config", default="E:/refcv8_eval_kit/ckpt/config.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-windows", type=int, default=0, help="SMOKE ONLY")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(R8_CODE))
    import r8_repoint  # noqa: E402
    r8_repoint.REMAP_OVERRIDES = Path(a.remap).resolve()
    L = r8_repoint.apply()
    sys.path.insert(0, str(r8_repoint.BATTERY_CODE))
    import perc_pass_r7 as PP  # noqa: E402  (READ-ONLY: make_perc_dataset_cls)
    import g0_refcv7 as G  # noqa: E402
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import torch.nn.functional as F  # noqa: E402
    from taniteval import map_hires_metrics as MH  # noqa: E402

    t0 = time.time()
    tr = L.trainer()
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, a.device)
    model.eval()
    W = int(cfg.core.window)
    LA = int(tr.LAW_AHEAD)
    law_idx = LA - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=PP.make_perc_dataset_cls(tr, LA))
    G.patch_frames_to_device(tr)
    from tanitad.data.v2_dataset import load_or_build_manifest
    clip_ids = [str(c) for c in load_or_build_manifest(args.eval_cache, verbose=False)["clip_id"]]
    # ---- window choice: 4 per clip at fixed quantiles of the clip's index positions (content-blind) ----------- #
    by_ep = {}
    for j, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), j))
    order = []
    for e_i in sorted(by_ep):
        lst = sorted(by_ep[e_i])
        for q in QUANT:
            order.append((e_i, lst[min(len(lst) - 1, int(q * len(lst)))]))
    if a.max_windows:
        order = order[:a.max_windows]
    br = model._map_hires
    cw = model._map_hires_class_weight
    cw_np = None if cw is None else cw.detach().float().cpu().numpy()
    rule = str(br.cfg.decision_rule)
    assert tuple(br.cfg.out_hw) == (1000, 600) and tuple(br.cfg.lift_grid.shape) == (400, 240), "extent drift"
    cap = {"n_branch": 0}

    def pre(mod, args_, kwargs_):
        cap["fmap"], cap["grid"], cap["valid"] = args_[0], args_[1], args_[2]
        cap["n_branch"] += 1

    h1 = br.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = br.lift.register_forward_hook(lambda m, i, o: cap.__setitem__("lift", o))
    h3 = br.encoder.register_forward_hook(lambda m, i, o: cap.__setitem__("enc", o))
    capo = {}
    orig_fwd = model.forward

    def fwd(*aa, **kk):
        o = orig_fwd(*aa, **kk)
        capo["out"] = o
        return o
    model.forward = fwd
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    fr = torch.tensor([int((i + 0.5) * 2.5) for i in range(ROW_N)])                 # 10 cm row containing the centre
    fc = torch.tensor([int((j + 0.5) * 2.5) for j in range(COL0, COL0 + COL_N)])
    ev_r = torch.tensor(EV_ROWS)
    ev_c = torch.tensor(EV_COLS)
    rec_w = []
    n_skip_nogt = 0
    torch.manual_seed(0)
    tl = time.time()
    with torch.no_grad():
        for n, (e_i, (t, j)) in enumerate(order):
            eb = G.collate(e_ds, [j], law_idx)
            lab = eb.get("map_fine_label")
            if lab is None or not bool(lab.reshape(-1)[0]):
                n_skip_nogt += 1
                continue
            cap["n_branch"] = 0
            tr.compute_losses_v3(model, eb, a.device, mode=mode, ablate_frames=abl)
            if cap["n_branch"] != 1:
                raise SystemExit(f"[pbn] the map branch ran {cap['n_branch']} times in one window (want 1)")
            enc = cap["enc"][0, :, :ROW_N, COL0:COL0 + COL_N].float()
            lift = cap["lift"][0, :, :ROW_N, COL0:COL0 + COL_N].float()
            fmap, grid, valid = cap["fmap"].float(), cap["grid"], cap["valid"]
            g = grid[:, :, ev_r][:, :, :, ev_c].to(fmap.dtype)                            # [1, Z, 80, 12, 2]
            v = valid[:, :, ev_r][:, :, :, ev_c]
            Z = g.shape[1]
            gz = torch.where(v.unsqueeze(-1), g, torch.zeros_like(g))
            s = F.grid_sample(fmap, gz.reshape(1, Z * len(EV_ROWS), len(EV_COLS), 2), mode="bilinear",
                              padding_mode="border", align_corners=False)              # BEVLift's own call
            s = s.reshape(1, fmap.shape[1], Z, len(EV_ROWS), len(EV_COLS)) * v.to(s.dtype).unsqueeze(1)
            img = s.permute(0, 2, 1, 3, 4).reshape(Z * fmap.shape[1], len(EV_ROWS), len(EV_COLS))
            lv = valid.any(dim=1)[0, :ROW_N, COL0:COL0 + COL_N]
            perc = capo["out"].get("perception") or {}
            lg_full = perc["map_hires_logits"][0].float()                                # [8, 1000, 600]
            lg = lg_full[:, fr][:, :, fc]
            dec_full = MH.logits_to_codes(lg_full.cpu().numpy(), rule=rule, class_weight=cw_np)
            dec = dec_full[fr.numpy()][:, fc.numpy()]
            gt_full = eb["map_fine"][0].cpu().numpy()
            gt = gt_full[fr.numpy()][:, fc.numpy()]
            arr = {"enc": enc.half().cpu().numpy(), "lift": lift.half().cpu().numpy(), "img": img.half().cpu().numpy(),
                   "logit": lg.half().cpu().numpy(), "dec": dec.astype(np.uint8), "gt": gt.astype(np.uint8),
                   "lv": lv.cpu().numpy().astype(bool)}
            for k_ in ("enc", "lift", "img", "logit"):
                if not np.isfinite(arr[k_].astype(np.float32)).all():
                    raise SystemExit(f"[pbn] non-finite {k_}")
            s12 = L.sha12(clip_ids[e_i])
            np.savez(out / f"w_{n:05d}_{s12}_{t:04d}.npz", **arr)
            rec_w.append({"n": n, "sha12": s12, "e_i": int(e_i), "t": int(t), "W": W,
                          "v0_ms": float(eb["pose_last"][0, 3]) if eb["pose_last"].shape[-1] > 3 else None,
                          "enc_absmean": float(enc.abs().mean()), "img_absmean": float(img.abs().mean()),
                          "gt_seen_frac": float((gt != 255).mean()), "lv_frac": float(lv.float().mean())})
            if n % 25 == 0:
                el = time.time() - tl
                print(f"[pbn] {a.split} {n + 1}/{len(order)} windows, {el:.0f} s, {el / max(1, len(rec_w)):.2f} s/win",
                      flush=True)
    for h in (h1, h2, h3):
        h.remove()
    model.forward = orig_fwd
    summ = {"tool": "pbn_capture.py", "split": a.split, "ckpt": a.ckpt, "ckpt_md5": md5(a.ckpt),
            "config_md5": md5(a.config), "decision_rule": rule,
            "class_weight": None if cw_np is None else [float(x) for x in cw_np],
            "n_order": len(order), "n_written": len(rec_w), "n_skip_nogt": n_skip_nogt,
            "n_clips_written": len({r["sha12"] for r in rec_w}), "W": W, "quantiles": QUANT,
            "patch": {"rows": [0, ROW_N], "cols": [COL0, COL0 + COL_N], "eval_rows": EV_ROWS, "eval_cols": EV_COLS},
            "repoint": r8_repoint.describe(L), "wall_s": time.time() - t0, "loop_s": time.time() - tl,
            "cuda_max_mem_gib": (torch.cuda.max_memory_allocated() / 2 ** 30) if torch.cuda.is_available() else None,
            "windows": rec_w}
    (out / "capture.json").write_text(json.dumps(summ, indent=1, default=str), encoding="utf-8")
    print(f"[pbn] DONE {a.split}: {len(rec_w)} windows, {summ['n_clips_written']} clips, skipped(no GT) {n_skip_nogt}, "
          f"{summ['wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
