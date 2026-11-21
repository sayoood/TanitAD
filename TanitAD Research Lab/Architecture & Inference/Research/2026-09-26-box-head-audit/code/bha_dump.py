#!/usr/bin/env python3
"""bha_dump.py -- box-head audit (2026-09-26): dump the per-slot outputs of BOTH refcv6 detector heads.

What it writes, per window (batch 1, the trainer's own forward, interrupted right after
``model(...)`` returns -- the render tool's ``forward_like_trainer`` pattern):

  * ``b3_raw``  [100, 23]  -- ``out['perception']['box_slots']['raw']`` (Box3DSlotDecoder, the 3-D box head)
  * ``ag_raw``  [100, 21]  -- ``out['agent_slots']['raw']``             (core AgentSlotDecoder, the planner's agent tokens)
  * ``b3_pres`` / ``ag_pres`` [100] -- the model's OWN decoded presence logits (a re-decode control offline)
  * the window's GT block exactly as the trainer's ``V3Dataset._agent_item`` emits it, valid rows only
    (box, yaw, cls, occ, cz, h, zh_mask, rates, rates_mask), plus ``agent_label`` / ``agent_n_raw``.

Nothing is scored here. Every loss, match and metric is recomputed OFFLINE from these tensors with the
tip tree's own functions (``bha_analyze.py``), so the Thor side carries no analysis code that could drift.

Sets (each a pre-declared, deterministic rule -- no window is chosen by looking at a prediction):
  * ``inrun``    -- ``refcv6_loader.inrun_eval_perm(e_ds, 8, 16)``: the trainer's OWN fixed in-run eval subset
                    (train():7561-7563, generator seed 12345). Its batch-level recomputation must reproduce the
                    run's own ``eval_box3d_*`` / ``eval_agent_*`` row at step 38,000 (the known-value control).
  * ``clipgrid`` -- every eval clip (sorted by sha12), ``--per-clip`` windows evenly spaced over the clip's
                    AGENT-LABELLED windows. Cluster-balanced; the representative eval sample.
  * ``train``    -- ``--train-clips`` TRAIN clips (every k-th clip of the train manifest sorted by sha12),
                    ``--per-clip`` labelled windows each. Tests train-vs-eval calibration.
  * ``smoke``    -- the first 2 inrun windows; timing only.

Binding rules honoured: CPU only (asserts no CUDA), sha12 only (no raw clip id is written), strict load, ckpt
md5 verified BY CONTENT, step asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


class _Captured(Exception):
    pass


def forward_like_trainer(tr, model, batch, device, *, mode, ablate_frames, seed):
    """``compute_losses_v3`` interrupted by a forward hook right after ``model(...)`` returns --
    the trainer's own input assembly, the trainer's own forward (render_refcv6_map_video.py:890)."""
    import torch
    box = {}

    def hook(_m, _inp, out):
        box["out"] = out
        raise _Captured()
    h = model.register_forward_hook(hook)
    torch.manual_seed(int(seed))
    try:
        with torch.no_grad():
            tr.compute_losses_v3(model, batch, device, mode=mode, ablate_frames=ablate_frames)
        raise SystemExit("[bha] compute_losses_v3 returned WITHOUT the forward hook firing")
    except _Captured:
        pass
    finally:
        h.remove()
    return box["out"]


def evenly(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({int(round((j + 0.5) * n / k - 0.5)) for j in range(k)})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True, help="clean git-archive tree (stack/ taniteval/ tools/)")
    ap.add_argument("--code", required=True, help="dir holding refcv6_loader.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expect-ckpt-md5", required=True)
    ap.add_argument("--expect-step", type=int, default=38000)
    ap.add_argument("--set", choices=("inrun", "clipgrid", "train", "smoke"), required=True)
    ap.add_argument("--per-clip", type=int, default=4)
    ap.add_argument("--train-clips", type=int, default=64)
    ap.add_argument("--train-cache", default="/home/nvidia/data/refcv6-b1-416x1024-train")
    ap.add_argument("--train-labels", default="/home/nvidia/data/v8labels/labels/s2_labels_v8_train.jsonl.gz")
    ap.add_argument("--train-sidecar", default="/home/nvidia/data/refcv6_speed_max_v8_train.jsonl")
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args(argv)

    t_all = time.time()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec: dict = {"tool": "bha_dump.py", "set": a.set, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "argv": sys.argv[1:], "departures": []}
    os.environ["REFCV6_REPO"] = str(a.tree)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(a.code))
    import refcv6_loader as L                                   # noqa: E402
    L.bootstrap()                                               # asserts tanitad resolves to THIS tree
    import tanitad                                              # noqa: E402
    want = os.path.normcase(os.path.abspath(os.path.join(a.tree, "stack")))
    got = os.path.normcase(os.path.abspath(tanitad.__file__))
    if not got.startswith(want):
        raise SystemExit(f"[bha] tanitad imported from {got}, not from {want}")
    rec["tanitad_file"] = tanitad.__file__
    import torch                                                # noqa: E402
    torch.set_num_threads(int(a.threads))
    if torch.cuda.is_available():
        raise SystemExit("[bha] CUDA is visible -- this dump is CPU-only by rule (CUDA_VISIBLE_DEVICES='')")
    rec["torch"] = torch.__version__
    rec["threads"] = torch.get_num_threads()
    rec["loader_md5"] = md5_file(Path(a.code) / "refcv6_loader.py")

    # ---- checkpoint + config BY CONTENT ------------------------------------------------------- #
    t0 = time.time()
    ck_md5 = md5_file(a.ckpt)
    if ck_md5 != a.expect_ckpt_md5:
        raise SystemExit(f"[bha] ckpt md5 {ck_md5} != expected {a.expect_ckpt_md5}")
    rec["ckpt"] = {"path": a.ckpt, "md5": ck_md5, "bytes": os.path.getsize(a.ckpt)}
    rec["config"] = {"path": a.config, "md5": md5_file(a.config)}
    config = L.load_config(a.config)
    print(f"[bha] ckpt md5 ok ({time.time() - t0:.1f} s); config md5 {rec['config']['md5']}", flush=True)

    tr = L.trainer()
    # ⛔ remap = {} : on Thor the run's own argv paths ARE the data (no kit remap). --trunk-compile is
    # still dropped by the loader (DROP_FLAGS): it wraps the backbone CALL only.
    model, cfg, targs, mrec = L.build_model(config, a.ckpt, device="cpu", remap={})
    sd = mrec["state_dict"]
    if int(sd.get("step") or -1) != int(a.expect_step):
        raise SystemExit(f"[bha] STEP MISMATCH: {sd.get('step')!r} != {a.expect_step}")
    if sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[bha] strict load not clean: {sd['missing'][:5]} {sd['unexpected'][:5]}")
    if not mrec["param_breakdown"]["equal"]:
        raise SystemExit("[bha] param_breakdown differs from config.json")
    bad_anc = [k for k, v in mrec["anchor_file_vs_ckpt_buffers"].items() if v["max_abs_diff"] != 0.0]
    if bad_anc:
        raise SystemExit(f"[bha] anchor file != checkpoint anchor buffers: {bad_anc}")
    if getattr(model, "_perception", None) is None or model._perception.box_dec is None:
        raise SystemExit("[bha] no perception branch / box decoder in this build")
    ag = getattr(cfg.core, "agents", None)
    if ag is None or bool(getattr(ag, "oracle", False)) or not bool(getattr(ag, "enable", False)):
        raise SystemExit("[bha] the agent seam is not the learned head (oracle or off)")
    rec["model"] = {"step": sd["step"], "strict": "0 missing / 0 unexpected", "build_s": mrec["build_s"],
                    "argv_remap": mrec["argv_remap"], "departures": mrec["departures"],
                    "trunk_equalize_as_trained": mrec.get("trunk_equalize_as_trained"),
                    "perception": mrec.get("perception"), "presence_gate": float(ag.presence_gate),
                    "presence_hard": bool(ag.presence_hard), "queries": int(ag.queries),
                    "box3d_visible_filter": bool(getattr(model, "_box3d_visible_filter", True)),
                    "cls_weight": mrec.get("cls_weight"),
                    "cls_class_weight": (None if model._cls_class_weight is None
                                         else [float(x) for x in model._cls_class_weight.cpu()])}
    mode = getattr(targs, "mode", "diffusion")
    ablate = bool(getattr(targs, "ablate_frames", False))
    W = int(cfg.core.window)
    print(f"[bha] model built {mrec['build_s']} s, step {sd['step']}, window {W}, gate {ag.presence_gate}",
          flush=True)

    # ---- dataset: the trainer's V3Dataset WITHOUT future-frame decode ------------------------- #
    class DumpWindows(tr.V3Dataset):
        u8_frames = True

        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w],
                    "future_frames": torch.zeros(0, dtype=torch.uint8),
                    "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                    "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                    "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def clip_of(ep):
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]

    from tanitad.data import v2_dataset as v2d
    t0 = time.time()
    keep_cids = None
    if a.set == "train":
        man = v2d.load_or_build_manifest(a.train_cache, verbose=False)
        cids = sorted((str(c) for c in man["clip_id"]), key=sha12)
        step = max(1, len(cids) // int(a.train_clips))
        keep_cids = set(cids[::step][: int(a.train_clips)])
        rec["train_selection"] = {"rule": f"train manifest clip ids sorted by sha12, every {step}-th, first "
                                          f"{a.train_clips}", "n_manifest": len(cids),
                                  "sha12": sorted(sha12(c) for c in keep_cids)}
        targs.eval_cache = a.train_cache
        targs.eval_labels = a.train_labels
        targs.speed_max_sidecar_v6_eval = a.train_sidecar
        rec["departures"].append("train set: --eval-cache/--eval-labels/--speed-max-sidecar-v6-eval pointed at the "
                                 "TRAIN cache/labels/sidecar (the A6 train-roll override), providers filtered to the "
                                 "selected clips")
    _orig_bvp = v2d.build_v2_providers
    if keep_cids is not None:
        def _only(*aa, **kk):
            eps = _orig_bvp(*aa, **kk)
            keep = [e for e in eps if clip_of(e) in keep_cids]
            if len(keep) != len(keep_cids):
                raise SystemExit(f"[bha] provider filter kept {len(keep)} of {len(keep_cids)}")
            return keep
        v2d.build_v2_providers = _only
    try:
        e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, targs, config, with_perception_targets=True,
                                                 dataset_cls=DumpWindows)
    finally:
        v2d.build_v2_providers = _orig_bvp
    if drec.get("agent_join") is None or drec.get("join3d") is None:
        raise SystemExit("[bha] the agent join / 3-D join was not attached")
    rec["dataset"] = {k: v for k, v in drec.items() if k not in ("label_clock_g3",)}
    rec["dataset"]["build_s"] = round(time.time() - t0, 1)
    print(f"[bha] dataset {drec['n_episodes']} episodes -> {drec['n_windows']} windows "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---- window selection (deterministic, declared) ------------------------------------------- #
    if a.set in ("inrun", "smoke"):
        idx = L.inrun_eval_perm(e_ds, 8, 16)
        rec["selection"] = {"rule": "refcv6_loader.inrun_eval_perm(e_ds, eval_batches=8, batch=16) -- "
                                    "train():7561-7563, torch.Generator().manual_seed(12345); batch b = "
                                    "positions [16b, 16b+16)"}
        if a.set == "smoke":
            idx = idx[:2]
    else:
        by_ep: dict = {}
        for wi, (e_i, t) in enumerate(e_ds.index):
            by_ep.setdefault(e_i, []).append((int(t), wi))
        idx = []
        n_unlab_clips = 0
        order = sorted(by_ep, key=lambda e_i: sha12(clip_of(e_eps[e_i])))
        for e_i in order:
            ep = e_eps[e_i]
            lab = [(t, wi) for (t, wi) in sorted(by_ep[e_i])
                   if e_ds.agent_join.lookup(int(ep.episode_id), t + W - 1) is not None]
            if not lab:
                n_unlab_clips += 1
                continue
            idx += [lab[j][1] for j in evenly(len(lab), int(a.per_clip))]
        rec["selection"] = {"rule": f"per clip (sorted by sha12): {a.per_clip} windows evenly spaced over the "
                                    f"clip's AGENT-LABELLED windows (join lookup at NOW = t + W - 1 is not None)",
                            "n_clips": len(order), "n_clips_without_labelled_window": n_unlab_clips}
    if a.max_windows:
        idx = idx[: int(a.max_windows)]
    rec["n_windows_selected"] = len(idx)
    print(f"[bha] {a.set}: {len(idx)} windows", flush=True)

    # ---- the dump --------------------------------------------------------------------------- #
    chunk, n_chunk, times = [], 0, []
    keys_t = ("agent_box", "agent_yaw", "agent_cls", "agent_occ", "agent_cz", "agent_h", "agent_zh_mask",
              "agent_rates", "agent_rates_mask")
    for k_i, wi in enumerate(idx):
        tw = time.time()
        item = e_ds[wi]
        batch = torch.utils.data.default_collate([item])
        out = forward_like_trainer(tr, model, batch, "cpu", mode=mode, ablate_frames=ablate, seed=0)
        pout = out.get("perception") or {}
        bs = pout.get("box_slots")
        ags = out.get("agent_slots")
        if bs is None or ags is None:
            raise SystemExit(f"[bha] forward emitted box_slots={bs is not None} agent_slots={ags is not None}")
        e_i, t = e_ds.index[wi]
        ep = e_eps[e_i]
        v = batch["agent_valid"][0].bool()
        r = {"set": a.set, "pos": k_i, "wi": int(wi), "sha12": sha12(clip_of(ep)), "t": int(t),
             "now_row": int(t + W - 1), "agent_label": bool(batch["agent_label"][0]),
             "agent_n_raw": int(batch["agent_n_raw"][0]), "v0": float(item["pose_last"][3]),
             "b3_raw": bs["raw"][0].detach().float().cpu().clone(),
             "ag_raw": ags["raw"][0].detach().float().cpu().clone(),
             "b3_pres": bs["presence_logit"][0].detach().float().cpu().clone(),
             "ag_pres": ags["presence_logit"][0].detach().float().cpu().clone(),
             "b3_box": bs["box"][0].detach().float().cpu().clone(),
             "ag_box": ags["box"][0].detach().float().cpu().clone()}
        for k in keys_t:
            if k in batch:
                r[k] = batch[k][0][v].detach().cpu().clone()
        del out
        chunk.append(r)
        times.append(time.time() - tw)
        if len(chunk) >= int(a.chunk) or k_i == len(idx) - 1:
            p = out_dir / f"{a.set}_chunk{n_chunk:03d}.pt"
            torch.save(chunk, p)
            n_chunk += 1
            chunk = []
            print(f"[bha] {k_i + 1}/{len(idx)} windows; last {times[-1]:.1f} s/w, median "
                  f"{sorted(times)[len(times) // 2]:.1f} s/w; wrote {p.name}", flush=True)
    rec["n_chunks"] = n_chunk
    rec["s_per_window"] = {"median": sorted(times)[len(times) // 2] if times else None,
                           "max": max(times) if times else None, "n": len(times)}
    rec["wall_s"] = round(time.time() - t_all, 1)
    (out_dir / f"{a.set}_record.json").write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"[bha] DONE {a.set}: {len(idx)} windows, {n_chunk} chunks, {rec['wall_s']} s", flush=True)
    (out_dir / f"{a.set}_DONE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
