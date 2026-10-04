"""SPEC_WPB_LADDER sec. 3 -- ONE eval pass of ONE ladder arm's checkpoint: one ROW (an intervention or none), one
SAMPLER seed, on the eval139 EVAL-DIAG grid (K windows per episode, the route package's rule), through the launch
tree's own loader (`refcv7_loader.build_model` STRICT + `build_eval_dataset`) and the trainer's own
`compute_losses_v3` in eval mode. Read-only: a forward hook and a ceiling-filter spy; nothing in the model changes.

Rows (model attributes, eval-only):
  base       -- as trained;
  rc_off     -- `_r8_no_rc` (refcv8 arms): the route checkpoint never fed;
  legal      -- `_r8_legal_row` (refcv8): the NavSim-LEGAL row (args unknown, RC invalid, speed unknown, token kept);
  rc_shuf    -- `_r8_rc_eval = "shuf"` (refcv8): another window's checkpoint (roll by one inside the batch);
  vmax_off   -- `_r8_speed_eval = "off"`: the unknown speed row everywhere (VMAX-OFF / UNKNOWN);
  vmax_shuf  -- `_r8_speed_eval = "shuf"`: another window's speed (VMAX-SHUF).
PAIRING: the window list (sha12 of episode + t, sha256 digest) and the batch composition are identical across arms;
the sampler seed of batch j is `seed * 100003 + j`. Batch 8 (the shuffled rows need >= 2 rows).

Writes <out>.npz (per window) + <out>.json (record: arm, step, row, seed, digest, compute). Clip ids: sha12 only.
Env: REFCV6_REPO=<tree>, REFCV6_KIT=/home/nvidia (Thor), PYTHONPATH=<tree>/stack:<tree>/taniteval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
ROWS = ("base", "rc_off", "legal", "rc_shuf", "vmax_off", "vmax_shuf")
#: L3's headline map cells (SPEC sec. 3): (code, name) x (fine rows, band) -- per-window inter / union on SEEN cells,
#: RAW argmax of the 10 cm logits (the declared prior_corrected rule is the in-run eval's; this is stated, not hidden)
MAP_CLASSES = ((1, "drivable"), (2, "lane"))
MAP_BANDS = ((0, 200, "0_20"), (200, 400, "20_40"))


def set_row(model, row: str) -> None:
    for k in ("_r8_no_rc", "_r8_legal_row", "_r8_rc_eval", "_r8_speed_eval"):
        setattr(model, k, None if k in ("_r8_rc_eval", "_r8_speed_eval") else False)
    if row == "rc_off":
        model._r8_no_rc = True
    elif row == "legal":
        model._r8_legal_row = True
    elif row == "rc_shuf":
        model._r8_rc_eval = "shuf"
    elif row == "vmax_off":
        model._r8_speed_eval = "off"
    elif row == "vmax_shuf":
        model._r8_speed_eval = "shuf"
    elif row != "base":
        raise SystemExit(f"[ladder-eval] unknown row {row!r} (one of {ROWS})")


def window_set(e_ds, k_per_ep: int, sha_of):
    by_ep = {}
    for i, (e_i, t) in enumerate(e_ds.index):
        by_ep.setdefault(int(e_i), []).append((int(t), i))
    idx = []
    for e_i in sorted(by_ep, key=lambda e: sha_of(e)):
        ts = sorted(by_ep[e_i])
        for j in range(k_per_ep):
            idx.append(ts[int((j + 0.5) * len(ts) / k_per_ep)][1])
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the arm's run dir (config.json, ckpt.pt)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--expect-step", type=int, required=True)
    ap.add_argument("--rows", default="base", help=f"comma list of {ROWS}")
    ap.add_argument("--seeds", default="0,1", help="sampler seeds (comma list)")
    ap.add_argument("--out-dir", required=True, help="writes <out-dir>/<row>_s<seed>.{npz,json}")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k-per-ep", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from tanitad.eval import refcv7_loader as L
    from tanitad.refs import refcv6_selection as v6sel
    from tanitad.refs import refcv8_conditioning as r8c
    tr = L.trainer()
    import refb_labels
    t0 = time.time()
    run = Path(a.run)
    config = L.load_config(run / "config.json")
    model, cfg, args, mrec = L.build_model(config, a.ckpt or str(run / "ckpt.pt"), device=a.device, strict=True)
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != a.expect_step or sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[ladder-eval] bad load: step {sd.get('step')} (want {a.expect_step}), missing "
                         f"{sd['missing'][:3]}, unexpected {sd['unexpected'][:3]}")
    if tuple(int(h) for h in cfg.core.trajectory.horizons) != HORIZONS:
        raise SystemExit("[ladder-eval] horizons drifted")
    model.eval()
    r8on = bool(getattr(model, "r8_enabled", False))
    rows = [r for r in a.rows.split(",") if r]
    seeds = [int(x) for x in a.seeds.split(",") if x != ""]
    for row in rows:
        if row not in ROWS:
            raise SystemExit(f"[ladder-eval] unknown row {row!r} (one of {ROWS})")
        if row in ("rc_off", "legal", "rc_shuf") and not r8on:
            raise SystemExit(f"[ladder-eval] row {row} needs the refcv8 seams (this arm has none)")
    # X10: every ladder arm TRAINS with --pose-sync-sidecar and its in-run eval rides the same clock
    # (refc_v3_train: enable_pose_sync right after the eval dataset is constructed, before any join). The standalone
    # kit (`refcv7_loader`) stays on the uncorrected clock BY MM RULING and is NOT changed: the harness applies the
    # shift itself through the loader's own `dataset_cls` seam, in the trainer's order.
    ps_path = getattr(args, "pose_sync_sidecar", None)
    trained = (config.get("pose_sync") or {}).get("train") if isinstance(config.get("pose_sync"), dict) else None
    if bool(ps_path) != bool(trained):
        raise SystemExit(f"[ladder-eval] clock MISMATCH: the arm's argv pose-sync sidecar {ps_path!r} vs its config.json "
                         f"pose_sync.train record {'present' if trained else 'absent'} -- refusing to pick a clock")
    if ps_path and str(trained.get("sidecar")) != str(ps_path):
        raise SystemExit(f"[ladder-eval] the arm trained on sidecar {trained.get('sidecar')!r}, argv says {ps_path!r}")
    PS = {}
    dcls = None
    if ps_path:
        base_cls = tr.lan_dataset_class(tr.V3Dataset) if bool(args.graft_lan or args.goal_str) else tr.V3Dataset

        class _PoseSynced(base_cls):
            def __init__(self, *a_, **k_):
                super().__init__(*a_, **k_)
                PS.update(self.enable_pose_sync(ps_path))
        dcls = _PoseSynced
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, dataset_cls=dcls)
    if ps_path and not PS:
        raise SystemExit("[ladder-eval] the arm trained with --pose-sync-sidecar but the eval split was not shifted")
    j8 = getattr(e_ds, "r8_join", None)
    if j8 is None:
        raise SystemExit("[ladder-eval] the eval split carries no v9 join (every ladder arm feeds N2)")
    s12 = getattr(e_ds, "_vis1_sha12", None) or {}

    def sha_of(e_i):
        eid = int(e_ds.episodes[e_i].episode_id)
        return s12.get(eid) or hashlib.sha256(str(eid).encode()).hexdigest()[:12]
    idx = window_set(e_ds, a.k_per_ep, sha_of)
    if a.max_windows:
        idx = idx[:a.max_windows]
    win_sha = [sha_of(e_ds.index[i][0]) for i in idx]
    win_t = [int(e_ds.index[i][1]) for i in idx]
    digest = hashlib.sha256(json.dumps(list(zip(win_sha, win_t))).encode()).hexdigest()
    rows_v9 = np.asarray(e_ds._r8_rows)[idx]
    R = j8.rel.rows
    lat_v9 = np.asarray(R["lat_v7id_a"]).astype(np.int64)[rows_v9]
    lon_v9 = np.asarray(R["lon_v7id"]).astype(np.int64)[rows_v9]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(e_ds, idx), batch_size=a.batch, shuffle=False,
                                     num_workers=a.workers, prefetch_factor=(2 if a.workers > 0 else None))

    def run_pass(row, seed):
        set_row(model, row)
        CAP = {}

        def hook(_m, _a, kw, out):
            if "out" not in CAP:
                CAP["out"], CAP["vmax"], CAP["vval"] = out, kw.get("v_max_ms"), kw.get("v_max_valid")
        h = model.register_forward_hook(hook, with_kwargs=True)
        orig_cf = v6sel.SpeedCeilingFilter.forward

        def _cf(self, cand, v_limit_ms):
            if "vlim" not in CAP:
                CAP["vlim"] = v_limit_ms.detach().float().reshape(-1).cpu().clone()
            return orig_cf(self, cand, v_limit_ms)
        v6sel.SpeedCeilingFilter.forward = _cf
        keep = {k: [] for k in ("traj", "gt", "gt_valid", "v0", "nav", "v_fed", "v_valid", "v_lim", "p_lat", "p_lon",
                                "off_drv", "drv_w", "rc_raw", "rc_valid", "cons_err_lat", "cons_err_lon", "cons_err_spd",
                                "cons_n_lat", "cons_n_lon", "cons_n_spd", "cons_pred", "cons_tgt", "cons_mask",
                                "map_inter", "map_union")}
        gpu = 0.0
        try:
            for j, b in enumerate(dl):
                CAP.clear()
                tg = time.time()
                torch.manual_seed(int(seed) * 100003 + j)
                with torch.no_grad():
                    tr.compute_losses_v3(model, b, a.device, mode=args.mode)
                gpu += time.time() - tg
                out = CAP["out"]
                n = out["traj"].shape[0]
                traj = out["traj"].detach().float().cpu().numpy()
                pl, fut = b["pose_last"].float(), b["future_poses_ext"].float()
                fv = b["future_valid_ext"].bool().numpy()
                gt = refb_labels.waypoint_targets(pl, fut, HORIZONS).numpy()
                keep["traj"].append(traj)
                keep["gt"].append(gt)
                keep["gt_valid"].append(np.stack([fv[:, h - 1] for h in HORIZONS], 1))
                keep["v0"].append(pl[:, 3].numpy())
                keep["nav"].append(b["nav_cmd"].numpy())
                vm = CAP.get("vmax")
                vv = CAP.get("vval")
                keep["v_fed"].append(np.full(n, np.nan) if vm is None else vm.detach().float().cpu().numpy())
                keep["v_valid"].append(np.zeros(n) if vv is None else vv.detach().float().cpu().numpy())
                keep["v_lim"].append(CAP["vlim"].numpy() if "vlim" in CAP else np.full(n, np.inf))
                keep["p_lat"].append(out["tacv6_lat_logits"].float().softmax(-1).cpu().numpy())
                keep["p_lon"].append(out["tacv6_lon_logits"].float().softmax(-1).cpu().numpy())
                if "map_fine" in b:          # A1 L5: the emitted plan's footprint at t 0.5..4.0 s (slots 0..5)
                    y, w = r8c.drivable_target(out["traj"].detach().float().cpu()[:, None, :6], b["map_fine"],
                                               b.get("map_fine_label"))
                    keep["off_drv"].append((1.0 - y[:, 0]).numpy())
                    keep["drv_w"].append(w[:, 0].numpy())
                else:
                    keep["off_drv"].append(np.full(n, np.nan))
                    keep["drv_w"].append(np.zeros(n))
                pl_ = (out.get("perception") or {}).get("map_hires_logits")
                if pl_ is not None and "map_fine" in b:
                    pred = pl_.detach().argmax(1).cpu()
                    gtm = b["map_fine"].long()
                    if tuple(pred.shape[-2:]) != tuple(gtm.shape[-2:]):
                        raise SystemExit(f"[ladder-eval] map grid {tuple(pred.shape)} vs target {tuple(gtm.shape)}")
                    seen = gtm != 255
                    ii, uu = [], []
                    for code, _nm in MAP_CLASSES:
                        for r0, r1, _bn in MAP_BANDS:
                            pc, gc, sn = pred[:, r0:r1] == code, gtm[:, r0:r1] == code, seen[:, r0:r1]
                            ii.append(((pc & gc) & sn).flatten(1).sum(1))
                            uu.append(((pc | gc) & sn).flatten(1).sum(1))
                    keep["map_inter"].append(torch.stack(ii, 1).numpy())
                    keep["map_union"].append(torch.stack(uu, 1).numpy())
                keep["rc_raw"].append(b["r8_rc_raw"].numpy() if "r8_rc_raw" in b else np.full((n, 3), np.nan))
                keep["rc_valid"].append(b["r8_rc_valid"].numpy().astype(np.float32) if "r8_rc_valid" in b
                                        else np.zeros(n))
                if "r8_v9_lat_c" in out and "r8_v9_lat_c" in b:
                    tg9 = r8c.v9_constraint_targets(b["r8_v9_lat_c"], b["r8_v9_lon_c"], b["r8_v9_speed"], b["lat_v7"],
                                                    b["lon_v7"])
                    ar = torch.arange(n)
                    pl9 = out["r8_v9_lat_c"].float().cpu()[ar, tg9["lat_cls"].clamp_min(0)]
                    po9 = out["r8_v9_lon_c"].float().cpu()[ar, tg9["lon_cls"].clamp_min(0)]
                    ps9 = out["r8_v9_speed"].float().cpu()[ar, tg9["lon_cls"].clamp_min(0)]
                    for nm, p_, t_, m_ in (("lat", pl9, tg9["lat"], tg9["lat_m"]), ("lon", po9, tg9["lon"], tg9["lon_m"]),
                                           ("spd", ps9, tg9["speed"], tg9["speed_m"])):
                        w_ = m_.float()
                        keep[f"cons_err_{nm}"].append(((p_ - t_).abs() * w_).sum(-1).numpy())
                        keep[f"cons_n_{nm}"].append(w_.sum(-1).numpy())
                    keep["cons_pred"].append(torch.cat([pl9, po9, ps9], -1).numpy())
                    keep["cons_tgt"].append(torch.cat([tg9["lat"], tg9["lon"], tg9["speed"]], -1).numpy())
                    keep["cons_mask"].append(torch.cat([tg9["lat_m"], tg9["lon_m"], tg9["speed_m"]], -1).numpy())
                if j % 20 == 0:
                    print(f"[ladder-eval] ZZ{min((j + 1) * a.batch, len(idx))}-{len(idx)}ZZ {time.time() - t0:.0f}s",
                          flush=True)
        finally:
            h.remove()
            v6sel.SpeedCeilingFilter.forward = orig_cf
        arrs = {k: np.concatenate(v) for k, v in keep.items() if v}
        arrs.update(win_sha12=np.asarray(win_sha), win_t=np.asarray(win_t), lat_v9=lat_v9, lon_v9=lon_v9,
                    map_cells=np.asarray([f"{nm}_{bn}" for _c, nm in MAP_CLASSES for _r0, _r1, bn in MAP_BANDS]))
        if len(arrs["traj"]) != len(idx):
            raise SystemExit(f"[ladder-eval] {len(arrs['traj'])} windows captured for {len(idx)} asked")
        outp = Path(a.out_dir) / f"{row}_s{seed}"
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(outp) + ".npz", **arrs)
        rec = {"run": str(run), "arm_out": config.get("argv", [])[config.get("argv", []).index("--out") + 1]
               if "--out" in config.get("argv", []) else None, "step": sd["step"], "row": row, "seed": seed,
               "batch": a.batch, "k_per_ep": a.k_per_ep, "n_windows": len(idx), "n_episodes": len(set(win_sha)),
               "window_sha12_t_sha256": digest, "refcv8": r8on,
               "speed_input": str(getattr(model.cfg.refcv8, "speed_input", "") or ""),
               "pose_sync": ({k: PS.get(k) for k in ("n_clips", "n_covered", "n_uncovered_read_unshifted")}
                             if PS else None),
               "compute": {"forward_s": round(gpu, 1), "wall_s": round(time.time() - t0, 1)}}
        Path(str(outp) + ".json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[ladder-eval] DONE {outp.name}: {len(idx)} windows digest {digest[:16]} wall {time.time() - t0:.0f}s",
              flush=True)

    for row in rows:
        for seed in seeds:
            run_pass(row, seed)
    set_row(model, "base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
