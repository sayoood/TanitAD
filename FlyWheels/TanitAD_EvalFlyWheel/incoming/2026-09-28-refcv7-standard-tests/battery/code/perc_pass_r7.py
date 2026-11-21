"""refcv7 PERCEPTION pass on surface M (= S2 after SPEC AMENDMENT A1): the map at 10 cm and the box heads,
through the TRAINER'S OWN instrument, one window at a time (the W-BOOTSTRAP form, batch 1).

    python perc_pass_r7.py --ckpt <ckpt> --config <config.json> --windows-json <S2 {clip: [ws]}> \
        --out <prefix> [--refcv6-pkl <perc_dump_refcv6 .pkl>] [--prior-npz <prior_fit .npz>]

Per window: `compute_losses_v3` (eval mode, no_grad) on the loader's eval dataset WITH perception
targets. The forward's `perception.map_hires_logits` become 10 cm codes by
`taniteval.map_hires_metrics.logits_to_codes(rule=<the built branch's declared rule>, class_weight=
<the run's frozen weights>)` and are scored by `window_stats` against the batch's `map_fine` GT (seen
cells), IN THE SAME PASS as the refcv6@38k codes (`coarse_to_fine_codes` of its banked 0.5 m argmax,
NO_PREDICTION outside its 60 m x +-16 m window) and the positional prior (`predict_for`, which refuses
a fit clip): one `WindowTable`, every arm on every window (a paired test needs aligned arms).
Box: the trainer's own `_det_pack_box3d` / `_det_pack_agent`; refcv6@38k's banked box3d slots are packed
by the SAME `detection_metrics.window_packs` call on the IDENTICAL target + VIS-1 blocks refcv7's pack
was built from (intercepted, not rebuilt).

SPEED (same numbers, recorded): the LAW target frame is a ZERO frame -- compute_losses_v3's `law` term is
therefore meaningless here and is NOT read; the perception outputs come from the FORWARD, which never
sees future frames. ⛔ Verified per run, not assumed: `--law-control 2` rolls the first windows with the
REAL LAW frame too and requires every perception tensor to be `torch.equal` (and the `law` term to
DIFFER -- the positive control that the swap took effect). Frames are decoded one window ahead in a
background thread (only that thread touches the dataset) and the map statistics run in a small pool.

Outputs: `<prefix>_map_table.npz` (WindowTable), `<prefix>_det.pkl` (per-window packs, sha12 + ws
only -- no raw clip id), `<prefix>.json` (the record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tanitad.eval import refcv7_loader as L  # noqa: E402

L.bootstrap()
import numpy as np  # noqa: E402
import torch  # noqa: E402
import g0_refcv7 as G  # noqa: E402


def make_perc_dataset_cls(tr, law_ahead: int):
    """The trainer's V3Dataset; the future frames are ZEROS (never decoded). See the module docstring."""
    class PercWindows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:          # refb_train.py:220-231, future frames zeroed
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            fr = ep.frames[t:t + w]
            return {
                "frames": fr,
                "actions": ep.actions[t:t + w],
                "future_frames": torch.zeros((law_ahead,) + tuple(fr.shape[1:]), dtype=fr.dtype),
                "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                "pose_last": ep.poses[t + w - 1],
                "episode_id": ep.episode_id,
            }
    return PercWindows


def _perc_tensors(out) -> dict:
    p = out.get("perception") or {}
    d = {}
    if p.get("map_hires_logits") is not None:
        d["map_hires_logits"] = p["map_hires_logits"]
    bs = p.get("box_slots") or {}
    for k in ("presence_logit", "box", "cls_logits", "cz"):
        if torch.is_tensor(bs.get(k)):
            d[f"box_slots.{k}"] = bs[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out", required=True, help="output PREFIX")
    ap.add_argument("--refcv6-pkl", default=None)
    ap.add_argument("--prior-npz", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--law-control", type=int, default=2)
    ap.add_argument("--max-windows", type=int, default=0, help="SMOKE ONLY")
    ap.add_argument("--device", default="cuda", help="cpu = SMOKE ONLY (code-path check)")
    a = ap.parse_args()
    from taniteval import map_hires_metrics as MH
    t_all = time.time()
    tr = L.trainer()
    device = a.device
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, device)
    ext = tr._mhr.declared_extent(args)
    W = int(cfg.core.window)
    LA = int(tr.LAW_AHEAD)
    law_idx = LA - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=make_perc_dataset_cls(tr, LA))
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
                raise SystemExit(f"[perc] window ws={w0} of clip {L.sha12(c)} is not in the eval index")
            order.append((c, int(w0), pos[key]))
    if a.max_windows:
        order = order[:a.max_windows]
    # ---- the other arms' inputs --------------------------------------------------------- #
    r6 = None
    r6_meta = None
    if a.refcv6_pkl:
        r6 = {(r["sha12"], int(r["ws"])): r for r in pickle.load(open(a.refcv6_pkl, "rb"))}
        mp = Path(a.refcv6_pkl).with_suffix(".json")
        r6_meta = json.load(open(mp, encoding="utf-8")) if mp.exists() else None
        miss = [(L.sha12(c), w) for c, w, _ in order if (L.sha12(c), w) not in r6]
        if miss:
            raise SystemExit(f"[perc] the refcv6 dump lacks {len(miss)} of the windows -- refusing "
                             f"to pair (not the same windows)")
    prior = MH.PositionalPrior.load(a.prior_npz) if a.prior_npz else None
    if prior is not None and tuple(prior.extent.fine_shape) != tuple(ext.fine_shape):
        raise SystemExit(f"[perc] prior extent {prior.extent} != the model's {ext}")
    arms = ("refcv7",) + (("refcv6_38k",) if r6 is not None else ()) + \
        (("prior",) if prior is not None else ())
    table = MH.WindowTable(arms=arms)
    prior_map = None
    G.patch_frames_to_device(tr)
    br = model._map_hires
    cw = model._map_hires_class_weight
    cw_np = None if cw is None else cw.detach().float().cpu().numpy()
    rule = str(br.cfg.decision_rule)
    # ---- capture the forward's output and the box3d target block ------------------------ #
    cap = {}
    orig_fwd = model.forward

    def fwd(*aa, **kk):
        out = orig_fwd(*aa, **kk)
        cap["out"] = out
        return out
    model.forward = fwd
    dm = tr._det_metrics
    orig_wp = dm.window_packs
    calls = []

    def wp(pred, tgt_real, vis, **kw):
        res = orig_wp(pred, tgt_real, vis, **kw)
        calls.append({"ret": res, "tgt_real": tgt_real, "vis": vis})
        return res
    dm.window_packs = wp
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))

    def fwd_losses(eb):
        calls.clear()
        cap.clear()
        return tr.compute_losses_v3(model, eb, device, mode=mode, ablate_frames=abl)

    # ---- the LAW-frame control (SPEC: verified, not assumed) ---------------------------- #
    law_ctl = []
    with torch.no_grad():
        for (c, w0, j) in order[:max(0, a.law_control)]:
            eb0 = G.collate(e_ds, [j], law_idx)                       # zero LAW frame
            e_i, t = e_ds.index[j]
            real = e_ds.episodes[e_i].frames[t + W:t + W + LA][law_idx]
            eb1 = dict(eb0)
            eb1["future_frames"] = G.LawOnly(real[None].clone(), law_idx)
            torch.manual_seed(a.seed)
            el0 = fwd_losses(eb0)
            p0 = {k: v.detach().clone() for k, v in _perc_tensors(cap["out"]).items()}
            torch.manual_seed(a.seed)
            el1 = fwd_losses(eb1)
            p1 = _perc_tensors(cap["out"])
            same = {k: bool(torch.equal(p0[k], p1[k])) for k in p0}
            law0, law1 = float(el0.get("law", float("nan"))), float(el1.get("law", float("nan")))
            law_ctl.append({"sha12": L.sha12(c), "ws": w0, "perception_equal": same,
                            "law_zero_frame": law0, "law_real_frame": law1,
                            "law_differs": law0 != law1})
            del el0, el1, eb0, eb1
    ctl_ok = bool(law_ctl) and all(all(x["perception_equal"].values()) and x["perception_equal"]
                                   and x["law_differs"] for x in law_ctl)
    if a.law_control and not ctl_ok:
        dm.window_packs = orig_wp
        model.forward = orig_fwd
        json.dump({"tool": "perc_pass_r7.py", "law_control": law_ctl, "status": "REFUSED"},
                  open(str(a.out) + ".json.REFUSED", "w", encoding="utf-8"), indent=1)
        raise SystemExit(f"[perc] LAW-frame control FAILED -- the zero LAW frame changed a perception "
                         f"output (or the swap did not take): {law_ctl}")
    print(f"[perc] LAW-frame control: {'PASS' if ctl_ok else 'SKIPPED'} on {len(law_ctl)} windows",
          flush=True)

    def stats_job(s12, gt, logits_np, am6):
        preds = {"refcv7": MH.logits_to_codes(logits_np, rule=rule, class_weight=cw_np)}
        if am6 is not None:
            onehot = np.eye(8, dtype=np.float32)[am6.astype(np.int64)].transpose(2, 0, 1)
            preds["refcv6_38k"] = MH.coarse_to_fine_codes(onehot, extent=ext)
        if prior is not None:
            preds["prior"] = prior_map
        return {arm: MH.window_stats(preds[arm], gt) for arm in arms}

    det = []
    n_map = n_box = n_box6 = 0
    tim = {"wait_decode": 0.0, "forward": 0.0, "stats_submit": 0.0, "det": 0.0}
    dec_pool = ThreadPoolExecutor(max_workers=1)
    st_pool = ThreadPoolExecutor(max_workers=3)
    pend_dec = deque()
    pend_st = deque()
    it = iter(order)

    def submit_next():
        nxt = next(it, None)
        if nxt is not None:
            pend_dec.append((nxt, dec_pool.submit(G.collate, e_ds, [nxt[2]], law_idx)))

    def drain_stats(block: bool):
        while pend_st and (block or pend_st[0][1].done()):
            s12, fut = pend_st.popleft()
            st = fut.result()
            rows = int(next(iter(st.values()))["inter"].shape[1]) * 0 + int(ext.fine_shape[0])
            if table.n_rows and rows != table.n_rows:
                raise SystemExit("[perc] one table, one extent")
            table.n_rows = rows
            for arm in arms:
                dst = table.stats.setdefault(arm, {k: [] for k in MH.STATS})
                for k in MH.STATS:
                    dst[k].append(st[arm][k])
            table.eid.append(s12)

    torch.manual_seed(a.seed)
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
                has_map = (lab is not None and bool(lab.reshape(-1)[0])
                           and pout.get("map_hires_logits") is not None)
                logits_np = pout["map_hires_logits"][0].float().cpu().numpy() if has_map else None
                tim["forward"] += time.time() - tf
                ts = time.time()
                if has_map:
                    am6 = None
                    if r6 is not None:
                        am6 = r6[(s12, w0)].get("map05_argmax")
                        if am6 is None:
                            raise SystemExit("[perc] refcv6 has no map on a scored window")
                    if prior is not None:
                        if prior_map is None:
                            prior_map = prior.predict_for(s12)          # refuses a fit clip
                        else:
                            prior.predict_for(s12)                      # the refusal, every clip
                    gt = eb["map_fine"][0].cpu().numpy()
                    pend_st.append((s12, st_pool.submit(stats_job, s12, gt, logits_np, am6)))
                    n_map += 1
                tim["stats_submit"] += time.time() - ts
                td = time.time()
                rec = {"sha12": s12, "ws": w0}
                for hd in dm.HEADS:
                    pk = el.get(f"_det_pack_{hd}")
                    if not pk:
                        continue
                    rec[hd] = pk
                    if hd == "box3d":
                        n_box += 1
                        src = next((cc for cc in calls if cc["ret"] is pk), None)
                        if src is None:
                            raise SystemExit("[perc] the box3d pack's window_packs call was not seen")
                        if r6 is not None and r6[(s12, w0)].get("box3d") is not None:
                            p6 = {k: torch.as_tensor(v, device=device)[None]
                                  for k, v in r6[(s12, w0)]["box3d"].items()}
                            rec["box3d_refcv6_38k"] = orig_wp(
                                p6, src["tgt_real"], src["vis"], presence_cost="sigmoid",
                                cls_weight=None, with_match=True)
                            n_box6 += 1
                det.append(rec)
                tim["det"] += time.time() - td
                del el, eb
                cap.clear()
                drain_stats(block=len(pend_st) > 6)
                if n % 200 == 0:
                    print(f"[perc] {n + 1}/{len(order)} windows ({time.time() - t0:.0f} s; map "
                          f"{n_map}, box3d {n_box}, box3d refcv6 {n_box6}; timing "
                          f"{ {k: round(v, 1) for k, v in tim.items()} }; peak "
                          f"{(torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0) / 2**30:.2f} GiB)", flush=True)
                n += 1
        drain_stats(block=True)
    finally:
        dm.window_packs = orig_wp
        model.forward = orig_fwd
        dec_pool.shutdown(wait=True)
        st_pool.shutdown(wait=True)
    if table.n_windows != n_map:
        raise SystemExit(f"[perc] table holds {table.n_windows} windows, {n_map} were scored")
    pre = Path(a.out)
    pre.parent.mkdir(parents=True, exist_ok=True)
    tpath = str(pre) + "_map_table.npz"
    table.save(tpath)
    dpath = str(pre) + "_det.pkl"
    with open(dpath, "wb") as f:
        pickle.dump(det, f, protocol=4)
    meta = {"tool": "perc_pass_r7.py", "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
            "step": mrec["state_dict"]["step"], "config_md5": L.md5_file(a.config),
            "n_windows": len(order), "n_map_windows": n_map, "n_box3d_windows": n_box,
            "n_box3d_refcv6_windows": n_box6, "arms": list(arms), "decision_rule": rule,
            "class_weight": None if cw_np is None else [float(x) for x in cw_np],
            "extent_m": [float(ext.x_max_m), float(ext.y_half_m)],
            "prior": None if prior is None else {"npz": a.prior_npz, **prior.fingerprint()},
            "refcv6": None if r6 is None else {"pkl": a.refcv6_pkl, "meta": r6_meta},
            "law_control": {"pass": ctl_ok, "windows": law_ctl,
                            "rule": "zero LAW frame: every perception tensor torch.equal to the "
                                    "real-LAW-frame forward, and the law term differs"},
            "requires_grad_state": "loader (all False); G0 MEASURED the trained-flag difference",
            "timing_s": {k: round(v, 1) for k, v in tim.items()},
            "build_s": round(t_built - t_all, 1), "loop_s": round(time.time() - t0, 1),
            "wall_s": round(time.time() - t_all, 1), "seed": a.seed,
            "cuda_max_memory_allocated_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 3)
                                              if torch.cuda.is_initialized() else None),
            "device": device,
            "SMOKE_RESTRICTED": bool(a.max_windows), "map_table": tpath, "det_pkl": dpath,
            "map_table_sha256": hashlib.sha256(open(tpath, "rb").read()).hexdigest(),
            "det_pkl_sha256": hashlib.sha256(open(dpath, "rb").read()).hexdigest()}
    json.dump(meta, open(str(pre) + ".json", "w", encoding="utf-8"), indent=1)
    print(f"[perc] done: {json.dumps({k: meta[k] for k in ('n_windows', 'n_map_windows', 'arms', 'timing_s', 'build_s', 'loop_s')})}")


if __name__ == "__main__":
    main()
