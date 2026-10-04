"""refcv6@38k PERCEPTION dump on surface M (= S2, SPEC AMENDMENT A1), for BAR-M7 / BAR-B7's baseline.

Runs in the REFCV6 world only: the refcv6 battery package's own `refcv6_loader` on the 82c2331 tree
(the tree the post-switch 38k checkpoint trained on), exactly as that package's `chain_final_v2.sh` set
it up:
    REFCV6_REPO=C:/Users/Admin/ev6_82c2331  PYTHONPATH="C:/Users/Admin/ev6_82c2331/stack;C:/Users/Admin/ev6_82c2331/taniteval"
    python perc_dump_refcv6.py --ckpt D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt \
        --config D:/refcv6_eval_kit/ckpt_final/config.json --windows-json <S2> --out <npz>

Per window (batch 1, eval mode, no_grad): refcv6's own `compute_losses_v3`, the forward captured, and
banked: the 0.5 m map argmax over the 8 CLASS channels (the `not seen` channel dropped first, the
favourable reading `map_hires_metrics.coarse_to_fine_codes` also uses) as uint8 [120, 64], and the
box3d slots of the last decoder layer (`presence_logit`, `box`, `cls_logits`, `cz` when present).
Nothing is scored here: `perc_pass_r7.py` / `perc_score.py` score both models on the refcv7 targets.
SPEED: the LAW target frame is ZEROS (the forward never sees future frames; the `law` term is not read),
verified per run by `--law-control` (perception tensors torch.equal with the real frame; law differs);
frames are decoded one window ahead in a background thread.
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
R6PKG = Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
             "2026-09-23-refcv6-standard-tests/battery/code")
sys.path.insert(0, str(R6PKG))
import refcv6_loader as L  # noqa: E402  (the refcv6 package's loader; REFCV6_REPO selects the tree)

L.bootstrap()
import numpy as np  # noqa: E402,F401
import torch  # noqa: E402
import reproduce_inrun_eval as G6  # noqa: E402  (its LawOnly / frames_to_device patch)


def make_perc_dataset_cls(tr, law_ahead: int):
    class PercWindows(tr.V3Dataset):
        def _window_u8(self, i: int) -> dict:
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


def collate(e_ds, j, law_idx):
    import torch.utils.data as tud
    eb = tud.default_collate([e_ds[j]])
    ff = eb["future_frames"]
    eb["future_frames"] = G6.LawOnly(ff[:, law_idx].clone(), law_idx)
    return eb


def _perc(out):
    p = out.get("perception") or {}
    d = {}
    if torch.is_tensor(p.get("map_logits")):
        d["map_logits"] = p["map_logits"]
    for k, v in (p.get("box_slots") or {}).items():
        if torch.is_tensor(v):
            d[f"box_slots.{k}"] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--law-control", type=int, default=2)
    ap.add_argument("--max-windows", type=int, default=0, help="SMOKE ONLY")
    a = ap.parse_args()
    t_all = time.time()
    tr = L.trainer()
    import tanitad
    config = L.load_config(a.config)
    model, cfg, args, mrec = L.build_model(config, a.ckpt, "cuda")
    W = int(cfg.core.window)
    LA = int(tr.LAW_AHEAD)
    law_idx = LA - 1
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=make_perc_dataset_cls(tr, LA))
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
                raise SystemExit(f"[perc6] window ws={w0} of clip {L.sha12(c)} is not in the index")
            order.append((c, int(w0), pos[key]))
    if a.max_windows:
        order = order[:a.max_windows]
    G6.patch_frames_to_device(tr)
    cap = {}
    orig = model.forward

    def fwd(*aa, **kk):
        out = orig(*aa, **kk)
        cap["out"] = out
        return out
    model.forward = fwd
    mode = getattr(args, "mode", "diffusion")
    abl = bool(getattr(args, "ablate_frames", False))
    # ---- the LAW-frame control -------------------------------------------------------- #
    law_ctl = []
    with torch.no_grad():
        for (c, w0, j) in order[:max(0, a.law_control)]:
            eb0 = collate(e_ds, j, law_idx)
            e_i, t = e_ds.index[j]
            real = e_ds.episodes[e_i].frames[t + W:t + W + LA][law_idx]
            eb1 = dict(eb0)
            eb1["future_frames"] = G6.LawOnly(real[None].clone(), law_idx)
            torch.manual_seed(a.seed)
            el0 = tr.compute_losses_v3(model, eb0, "cuda", mode=mode, ablate_frames=abl)
            p0 = {k: v.detach().clone() for k, v in _perc(cap["out"]).items()}
            torch.manual_seed(a.seed)
            el1 = tr.compute_losses_v3(model, eb1, "cuda", mode=mode, ablate_frames=abl)
            p1 = _perc(cap["out"])
            same = {k: bool(torch.equal(p0[k], p1[k])) for k in p0}
            law0, law1 = float(el0.get("law", float("nan"))), float(el1.get("law", float("nan")))
            law_ctl.append({"sha12": L.sha12(c), "ws": w0, "perception_equal": same,
                            "law_zero_frame": law0, "law_real_frame": law1, "law_differs": law0 != law1})
    ctl_ok = bool(law_ctl) and all(x["perception_equal"] and all(x["perception_equal"].values())
                                   and x["law_differs"] for x in law_ctl)
    if a.law_control and not ctl_ok:
        model.forward = orig
        raise SystemExit(f"[perc6] LAW-frame control FAILED: {law_ctl}")
    print(f"[perc6] LAW-frame control {'PASS' if ctl_ok else 'SKIPPED'} ({len(law_ctl)} windows)", flush=True)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    recs = []
    n_map = n_box = 0
    t0 = time.time()
    pool = ThreadPoolExecutor(max_workers=1)
    pend = deque()
    it = iter(order)

    def submit_next():
        nxt = next(it, None)
        if nxt is not None:
            pend.append((nxt, pool.submit(collate, e_ds, nxt[2], law_idx)))
    submit_next()
    submit_next()
    try:
        with torch.no_grad():
            n = 0
            while pend:
                (c, w0, j), fut = pend.popleft()
                eb = fut.result()
                submit_next()
                cap.clear()
                tr.compute_losses_v3(model, eb, "cuda", mode=mode, ablate_frames=abl)
                pout = cap["out"].get("perception") or {}
                r = {"sha12": L.sha12(c), "ws": w0, "map05_argmax": None, "box3d": None}
                ml = pout.get("map_logits")
                if ml is not None:
                    p = ml[0].float()
                    if p.shape[0] not in (8, 9) or tuple(p.shape[1:]) != (120, 64):
                        raise SystemExit(f"[perc6] refcv6 map logits {tuple(p.shape)} are not [9|8,120,64]")
                    r["map05_argmax"] = p[:8].argmax(0).to(torch.uint8).cpu().numpy()
                    r["map05_channels"] = int(p.shape[0])
                    n_map += 1
                bs = pout.get("box_slots")
                if bs is not None:
                    r["box3d"] = {k: bs[k][0].detach().float().cpu().numpy()
                                  for k in ("presence_logit", "box", "cls_logits", "cz")
                                  if k in bs and torch.is_tensor(bs[k])}
                    n_box += 1
                recs.append(r)
                del eb
                if n % 200 == 0:
                    print(f"[perc6] {n + 1}/{len(order)} ({time.time() - t0:.0f} s; map {n_map}, box {n_box})",
                          flush=True)
                n += 1
    finally:
        model.forward = orig
        pool.shutdown(wait=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump(recs, f, protocol=4)
    meta = {"tool": "perc_dump_refcv6.py", "ckpt": a.ckpt, "ckpt_md5": L.md5_file(a.ckpt),
            "step": mrec["state_dict"]["step"], "config_md5": L.md5_file(a.config),
            "tree": str(L.REPO), "tanitad_file": tanitad.__file__, "n_windows": len(order),
            "n_map": n_map, "n_box3d": n_box, "law_control": {"pass": ctl_ok, "windows": law_ctl},
            "loop_s": round(time.time() - t0, 1), "wall_s": round(time.time() - t_all, 1),
            "SMOKE_RESTRICTED": bool(a.max_windows),
            "pkl_sha256": hashlib.sha256(open(out.with_suffix(".pkl"), "rb").read()).hexdigest()}
    json.dump(meta, open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=1)
    print(f"[perc6] wrote {out.with_suffix('.pkl')}: {json.dumps({k: meta[k] for k in ('n_windows', 'n_map', 'n_box3d', 'loop_s', 'wall_s')})}")


if __name__ == "__main__":
    main()
