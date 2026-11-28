"""p2_tol_pass.py -- P2: the S2 map pass of ONE checkpoint with boundary-tolerant (Chebyshev k) statistics.

The forward is the perc8 / battery perception pass UNCHANGED (``perc_pass_r7.main`` semantics, batch 1, ``compute_losses_v3`` eval no_grad,
zero LAW frame, `MH.logits_to_codes(rule=<declared>, class_weight=<frozen>)`); what differs is only what is done with each window's
codes: besides ``MH.window_stats`` (so the output is a banked-format ``WindowTable``) the pass computes ``p2_tol_lib.tol_stats``
(per class x 20 m band: tpp_k / tpg_k for k in {1,2,5,10}) and the GT line-thickness census.  The box heads' packs are not kept.

    R8_NO_POSE_SYNC=1 python p2_tol_pass.py --ckpt <ckpt> --config <config.json> --windows-json <S2 {clip: [ws]}> \
        --arm <name> --out-dir <dir> [--device cpu --max-windows 20]   (cpu = SMOKE ONLY)

In-pass controls (aggregated into <out-dir>/p2_<arm>.json):
  C4  chessboard k=1 == the programme's Euclidean-EDT (<0.2 m) tp counts, classes 2,3,5, EVERY window (integer-exact);
  C1' GT-as-prediction on the first --controls-n map windows (P=R=F=IoU=1 at every k);
  C2' all-one-class prediction on the same windows (every other class: tpg_k = tpp_k = 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_battery_patch as BP  # noqa: E402

BP.apply_child()                 # r8_repoint: remap env + pose-sync policy (R8_NO_POSE_SYNC) BEFORE the loader is imported
import perc_pass_r7 as PP  # noqa: E402   (imports the loader + battery modules; L.bootstrap())
import numpy as np  # noqa: E402
import torch  # noqa: E402
import p2_tol_lib as T  # noqa: E402
from p2_tol_jobs import bank_rows, controls_job, window_job  # noqa: E402,F401

L, G = PP.L, PP.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--arm", required=True, help="arm label written into the tables")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--law-control", type=int, default=2)
    ap.add_argument("--controls-n", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--device", default="cuda", help="cpu = SMOKE ONLY (code-path check)")
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    t_all = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tr = L.trainer()
    device = a.device
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    ext = tr._mhr.declared_extent(args)
    W = int(cfg.core.window)
    LA = int(tr.LAW_AHEAD)
    law_idx = LA - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=PP.make_perc_dataset_cls(tr, LA))
    t_built = time.time()
    from tanitad.data.v2_dataset import load_or_build_manifest
    clip_ids = [str(c) for c in load_or_build_manifest(args.eval_cache, verbose=False)["clip_id"]]
    ci = {c: i for i, c in enumerate(clip_ids)}
    windows = json.load(open(a.windows_json, encoding="utf-8"))
    pos = {tuple(int(v) for v in x): j for j, x in enumerate(e_ds.index)}
    order = []
    for c in sorted(windows, key=lambda c: ci[c]):
        for w0 in windows[c]:
            key = (ci[c], int(w0) - (W - 1))
            if key not in pos:
                raise SystemExit(f"[p2] window ws={w0} of clip {L.sha12(c)} is not in the eval index")
            order.append((c, int(w0), pos[key]))
    if a.max_windows:
        order = order[:a.max_windows]
    G.patch_frames_to_device(tr)
    br = model._map_hires
    cw = model._map_hires_class_weight
    cw_np = None if cw is None else cw.detach().float().cpu().numpy()
    rule = str(br.cfg.decision_rule)
    cap = {}
    orig_fwd = model.forward

    def fwd(*aa, **kk):
        out = orig_fwd(*aa, **kk)
        cap["out"] = out
        return out
    model.forward = fwd
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))

    def fwd_losses(eb):
        cap.clear()
        return tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=abl)

    # ---- the LAW-frame control (the zero LAW frame must not change a perception tensor) -------------------- #
    law_ctl = []
    with torch.no_grad():
        for (c, w0, j) in order[:max(0, a.law_control)]:
            eb0 = G.collate(e_ds, [j], law_idx)
            e_i, t = e_ds.index[j]
            real = e_ds.episodes[e_i].frames[t + W:t + W + LA][law_idx]
            eb1 = dict(eb0)
            eb1["future_frames"] = G.LawOnly(real[None].clone(), law_idx)
            torch.manual_seed(a.seed)
            el0 = fwd_losses(eb0)
            p0 = {k: v.detach().clone() for k, v in PP._perc_tensors(cap["out"]).items()}
            torch.manual_seed(a.seed)
            el1 = fwd_losses(eb1)
            p1 = PP._perc_tensors(cap["out"])
            same = {k: bool(torch.equal(p0[k], p1[k])) for k in p0}
            law0, law1 = float(el0.get("law", float("nan"))), float(el1.get("law", float("nan")))
            law_ctl.append({"sha12": L.sha12(c), "ws": w0, "perception_equal": same,
                            "law_zero_frame": law0, "law_real_frame": law1, "law_differs": law0 != law1})
            del el0, el1, eb0, eb1
    ctl_ok = bool(law_ctl) and all(all(x["perception_equal"].values()) and x["perception_equal"] and x["law_differs"]
                                   for x in law_ctl)
    if a.law_control and not ctl_ok:
        model.forward = orig_fwd
        raise SystemExit(f"[p2] LAW-frame control FAILED: {law_ctl}")
    print(f"[p2] LAW-frame control: {'PASS' if ctl_ok else 'SKIPPED'} on {len(law_ctl)} windows", flush=True)

    # ---- per-window statistics (worker threads) ------------------------------------------------------------ #
    def stats_job(s12, w0, gt, logits_np, run_controls):
        return window_job(s12, w0, gt, logits_np, rule, cw_np, run_controls)

    rows = []                                                # drained results, in submission order
    n_map = 0
    tim = {"wait_decode": 0.0, "forward": 0.0, "stats_submit": 0.0}
    dec_pool = ThreadPoolExecutor(max_workers=1)
    st_pool = ThreadPoolExecutor(max_workers=3)
    pend_dec, pend_st = deque(), deque()
    it = iter(order)

    def submit_next():
        nxt = next(it, None)
        if nxt is not None:
            pend_dec.append((nxt, dec_pool.submit(G.collate, e_ds, [nxt[2]], law_idx)))

    def drain_stats(block: bool):
        while pend_st and (block or pend_st[0].done()):
            rows.append(pend_st.popleft().result())

    torch.manual_seed(a.seed)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.manual_seed_all(a.seed)
    t0 = time.time()
    submit_next()
    submit_next()
    try:
        with torch.no_grad():
            n = 0
            while pend_dec:
                (c, w0, j), fut = pend_dec.popleft()
                tw = time.time()
                eb = fut.result()
                tim["wait_decode"] += time.time() - tw
                submit_next()
                s12 = L.sha12(c)
                tf = time.time()
                el = fwd_losses(eb)
                pout = cap["out"].get("perception") or {}
                lab = eb.get("map_fine_label")
                has_map = (lab is not None and bool(lab.reshape(-1)[0]) and pout.get("map_hires_logits") is not None)
                logits_np = pout["map_hires_logits"][0].float().cpu().numpy() if has_map else None
                tim["forward"] += time.time() - tf
                ts = time.time()
                if has_map:
                    gt = eb["map_fine"][0].cpu().numpy()
                    pend_st.append(st_pool.submit(stats_job, s12, w0, gt, logits_np, n_map < a.controls_n))
                    n_map += 1
                tim["stats_submit"] += time.time() - ts
                del el, eb
                cap.clear()
                drain_stats(block=len(pend_st) > 6)
                if n % 200 == 0:
                    print(f"[p2] ZZ{n + 1}-{len(order)}ZZ windows ({time.time() - t0:.0f} s; map {n_map}; timing "
                          f"{ {k: round(v, 1) for k, v in tim.items()} }; peak "
                          f"{(torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0) / 2**30:.2f} GiB)", flush=True)
                n += 1
        drain_stats(block=True)
    finally:
        model.forward = orig_fwd
        dec_pool.shutdown(wait=True)
        st_pool.shutdown(wait=True)
    if len(rows) != n_map:
        raise SystemExit(f"[p2] {len(rows)} stat rows for {n_map} map windows")

    # ---- bank ------------------------------------------------------------------------------------------------ #
    tpath, xpath = bank_rows(rows, a.arm, out_dir, ext.fine_shape[0])
    ctl_rows = [r["ctl"] for r in rows if r["ctl"] is not None]
    meta = {"tool": "p2_tol_pass.py", "arm": a.arm, "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
            "step": mrec["state_dict"]["step"], "config": a.config, "config_md5": L.md5_file(a.config),
            "windows_json": a.windows_json, "windows_json_md5": hashlib.md5(open(a.windows_json, "rb").read()).hexdigest(),
            "n_windows": len(order), "n_map_windows": n_map, "decision_rule": rule,
            "class_weight": None if cw_np is None else [float(x) for x in cw_np],
            "extent_m": [float(ext.x_max_m), float(ext.y_half_m)], "ks": [0] + list(T.KS),
            "tol_definition": "Chebyshev (chessboard) distance in 0.1 m cells; k=0 exact",
            "controls": {
                "law_frame": {"pass": ctl_ok, "windows": law_ctl},
                "C4_cheb_k1_vs_EDT_0p2m": {"windows": len(rows), "mismatching_cells": int(sum(r["edt_bad"] for r in rows)),
                                           "rule": "classes 2,3,5; tp_pred and tp_gt; every window; integer-exact"},
                "C0_counts_vs_MH_window_stats": {"mismatching_arrays": int(sum(r["base_bad"] for r in rows)),
                                                 "rule": "inter/union/n_gt/n_pred of tol_stats == MH.window_stats, every window"},
                "C1p_gt_as_pred": {"windows": len(ctl_rows), "all_pass": bool(ctl_rows) and all(c_["gt_as_pred"] for c_ in ctl_rows)},
                "C2p_all_one_class": {"windows": len(ctl_rows), "all_pass": bool(ctl_rows) and all(c_["one_class"] for c_ in ctl_rows)},
            },
            "timing_s": {k: round(v, 1) for k, v in tim.items()}, "build_s": round(t_built - t_all, 1),
            "loop_s": round(time.time() - t0, 1), "wall_s": round(time.time() - t_all, 1), "seed": a.seed,
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                              if torch.cuda.is_initialized() else None),
            "device": device, "SMOKE_RESTRICTED": bool(a.max_windows),
            "map_table": str(tpath), "tol_npz": str(xpath),
            "map_table_sha256": hashlib.sha256(open(tpath, "rb").read()).hexdigest(),
            "tol_npz_sha256": hashlib.sha256(open(xpath, "rb").read()).hexdigest()}
    json.dump(meta, open(out_dir / f"p2_{a.arm}.json", "w", encoding="utf-8"), indent=1)
    print(f"[p2] done: {json.dumps({k: meta[k] for k in ('arm', 'n_windows', 'n_map_windows', 'controls', 'timing_s', 'build_s', 'loop_s')})}")


if __name__ == "__main__":
    main()
