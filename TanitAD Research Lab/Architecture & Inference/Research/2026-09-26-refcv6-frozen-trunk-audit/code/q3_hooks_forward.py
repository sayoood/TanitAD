"""Q3 (advisory class C) + Q1 at the AS-LAUNCHED configuration -- ONE real forward of the live
run's model, every attention hooked.

WHAT RUNS. The live run's own checkpoint (ckpt_30000.pt, md5 0c5c67b3...) is rebuilt through the
battery's `refcv6_loader.build_model`, which replays `refc_v3_train.train()`'s model construction
from the run's recorded argv (strict load), with ONE departure it records itself: `--trunk-compile`
is dropped (no Triton on Windows; compile wraps the backbone CALL, the module tree is unchanged).
Every other lever is ON exactly as launched: u8 batches, equalize_bottom_rows 43, chunk 8, frozen +
FOLDED BatchNorm, bf16 autocast, channels_last, per-frame dedup. The consumer is the trainer's own
`compute_losses_v3` on window perm[0] of the in-run eval's fixed subset, CPU, no_grad, seed 0.

Q3 -- every nn.MultiheadAttention / nn.TransformerDecoderLayer is hooked (forward pre-hook,
with_kwargs) and its query / key / value SHAPES and STORAGE POINTERS are recorded; F.grid_sample is
wrapped so the two geometric samplers (BEV lift, BEV waypoint coupling) print their operands too.
The key/value PROVENANCE is read by pointer against the outputs of named modules (held alive so a
pointer cannot be recycled), and the result is compared with the DESIGN, written below as
LITERALS from SPEC_REFCV6_V2 s1/s4/s6 + the run's config.json -- never from the code under test.

Q1 -- a pre-hook on the backbone's OWN stem conv records what the pretrained weights actually
receive, and it is compared with an INDEPENDENT reference: the same raw PNG bytes decoded by PIL
(not torchvision), /255, bottom 43 rows zeroed, normalised with the checkpoint's DECLARED
mean/std (HF config.json + timm registry, both read 2026-09-26). Plus: the folded stem conv is
re-computed as conv -> BatchNorm(eval, the checkpoint's running stats) in fp32.

RAM: a watchdog kills this process if the host's available RAM falls below 8 GB (brief rule 6).
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _common as C  # noqa: E402

os.environ["REFCV6_REPO"] = str(C.TIP)
os.environ["REFCV6_KIT"] = str(C.KIT)
C.bootstrap()
sys.path.insert(0, "C:/Users/Admin/ev6_battery/code")
import refcv6_loader as L  # noqa: E402  (read-only reuse)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CKPT = os.environ.get("Q3_CKPT", str(C.KIT / "ckpt/ckpt_30000.pt"))
DECLARED_MEAN = (0.485, 0.456, 0.406)     # HF timm/resnet101.a1_in1k config.json, pretrained_cfg
DECLARED_STD = (0.229, 0.224, 0.225)      # (and timm 1.0.29 data/constants.py:3-4)
EQ_ROWS = 43                              # config.json argv --equalize-bottom-rows 43

# ---- THE DESIGN, as literals (B = 1 window) ------------------------------------------------ #
# SPEC_REFCV6_V2 s12.3: 416x1024 -> stride-16 26x64 = 1664 tokens, stride-32 13x32 = 416 tokens.
# config.json refcv6_perception.bev_tokens_hw [30, 16] -> 480 BEV tokens; agents.queries 100;
# tac_decoder_v6.decoder_cfg: d_model 256, n_queries 38, sources [agent, bev];
# the diffusion decoder width 384 (core.decoder.* weights are [384, ...] in the checkpoint).
DESIGN = {
    "tac_decoder_v6.layers.*.cross_attn": {"q": [1, 38, 256], "kv": [1, 580, 256],
                                          "why": "SPEC s4: 38 behaviour queries over {100 agent slots, 30x16 BEV tokens}"},
    "tac_decoder_v6.layers.*.self_attn": {"q": [1, 38, 256], "kv": [1, 38, 256],
                                         "why": "SPEC s4: query-to-query attention"},
    "core.decoder.layers.*.cross": {"kv_tokens": 416, "d": 384,
                                    "why": "SPEC s1 (3): image tokens by content (stride-32, 13x32)"},
    "core.decoder.layers.*.cross_agent": {"kv_tokens": 100, "d": 384,
                                          "why": "SPEC s1 (2): agent slots addressed by waypoint"},
    "core.agent_head.*": {"memory_tokens": 416, "d": 256, "queries": 100,
                          "why": "agent head reads the stride-32 map (refc.py:4340-4341; mem_pos [1,416,256])"},
    "_perception.box_dec.*": {"memory_tokens": 2144, "d": 256, "queries": 100,
                              "why": "SPEC s6: stride-16 tokens (1664) + BEV (480)"},
}


def _watchdog(stop):
    while not stop.is_set():
        if C.ram_available_gb() < 8.0:
            print("[audit:RAM] available < 8 GB during the forward -- aborting (brief rule 6)",
                  flush=True)
            os._exit(4)
        time.sleep(0.5)


def shape(t):
    return list(t.shape) if torch.is_tensor(t) else None


def main():
    avail = C.ram_available_gb()
    if avail < 10.0:
        print(f"[audit:RAM] REFUSED: available {avail:.2f} GB < 10.0 GB (8 GB floor + ~2 GB job)")
        raise SystemExit(3)
    stop = threading.Event()
    threading.Thread(target=_watchdog, args=(stop,), daemon=True).start()
    t0 = time.time()
    rec = {"what": "Q3 hooks + Q1 at the as-launched configuration", "ckpt": CKPT,
           "evidence_class": "MEASURED (ours, dev box CPU, the live run's checkpoint)"}
    config = L.load_config(str(C.CONFIG_JSON))
    # RAM: the loader's torch.load(ckpt) is made mmap=True in THIS process (pages on touch, not a
    # resident 404 MB copy); the tensors loaded are the same bytes.
    _tl = torch.load

    def _tl_mmap(f, *a, **k):
        if str(f) == str(CKPT):
            k.setdefault("mmap", True)
        return _tl(f, *a, **k)
    torch.load = _tl_mmap
    try:
        model, cfg, args, mrec = L.build_model(config, CKPT, device="cpu")
    finally:
        torch.load = _tl
    rec["loader"] = {k: mrec[k] for k in ("argv_remap", "departures", "state_dict", "param_breakdown",
                                          "trunk_memory_levers_built", "trunk_memory_levers_run",
                                          "mode", "sampler", "decoder_steps", "build_s")
                     if k in mrec}
    print("[q3] model built", round(time.time() - t0, 1), "s; strict missing/unexpected:",
          len(mrec["state_dict"]["missing"]), len(mrec["state_dict"]["unexpected"]), flush=True)
    tr = L.trainer()
    import reproduce_inrun_eval as RIE          # battery helpers, read-only reuse
    law_ahead = int(getattr(tr, "LAW_AHEAD", 5))
    e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, args, config, with_perception_targets=True,
                                             dataset_cls=RIE.make_g0_dataset_cls(tr, law_ahead))
    RIE.patch_frames_to_device(tr)
    rec["departures"] = ["torch.load(ckpt) with mmap=True (same bytes)",
                         f"future_frames: only the LAW stack {law_ahead - 1} is decoded "
                         "(battery G0 class; the only index compute_losses_v3 reads)",
                         "--trunk-compile dropped by the loader (no Triton on Windows)"]
    rec["dataset"] = {k: v for k, v in drec.items() if k in ("n_episodes", "n_windows")}
    perm = L.inrun_eval_perm(e_ds, int(args.eval_batches), int(args.batch))
    w0 = int(os.environ.get("Q3_WINDOW", perm[0]))
    e_i, t_start = e_ds.index[w0]
    item = e_ds[w0]
    batch = torch.utils.data.default_collate([item])
    _ff = batch["future_frames"]
    batch["future_frames"] = RIE.LawOnly(_ff[:, law_ahead - 1].clone(), law_ahead - 1)
    del _ff
    _fp = e_eps[e_i].frames
    _payload = Path(_fp._cache.cache_dir) / _fp._cache.files[_fp._clip]
    rec["window"] = {"perm0": w0, "episode_sha12": C.sha12(_payload.name.split(".")[0]),
                     "t": int(t_start),
                     "frames": shape(batch["frames"]), "frames_dtype": str(batch["frames"].dtype),
                     "future_frames": "LAW stack only"}
    print("[q3] window", rec["window"], flush=True)

    # ------------------------------------------------------------------ hooks
    names = {m: n for n, m in model.named_modules()}
    calls = []            # attention calls
    keep_alive = []       # outputs of named modules, held so pointers are not recycled
    ptr_owner = {}        # data_ptr -> (module name, shape)
    trunk = next(m for m in model.modules() if type(m).__name__ == "TimmResNetTrunk")
    stem = trunk.net.conv1
    stem_calls = []

    def stem_pre(mod, args_):
        x = args_[0]
        entry = {"shape": shape(x), "dtype": str(x.dtype),
                 "channels_last": bool(x.is_contiguous(memory_format=torch.channels_last)),
                 "autocast_cpu_enabled": bool(torch.is_autocast_cpu_enabled()),
                 "autocast_dtype": str(torch.get_autocast_cpu_dtype())}
        if not stem_calls:
            entry["_x"] = x.detach().float().clone()
        stem_calls.append(entry)

    hs = [stem.register_forward_pre_hook(stem_pre)]

    def attn_pre(mod, args_, kwargs):
        q = args_[0] if len(args_) > 0 else kwargs.get("query")
        k = args_[1] if len(args_) > 1 else kwargs.get("key")
        v = args_[2] if len(args_) > 2 else kwargs.get("value")
        kpm = kwargs.get("key_padding_mask")
        am = kwargs.get("attn_mask")
        calls.append({"module": names.get(mod, "?"), "kind": "MHA", "q": shape(q), "k": shape(k),
                      "v": shape(v), "k_is_v": bool(k is v or (torch.is_tensor(k) and torch.is_tensor(v)
                                                               and k.data_ptr() == v.data_ptr())),
                      "k_ptr": int(k.data_ptr()) if torch.is_tensor(k) else None,
                      "q_ptr": int(q.data_ptr()) if torch.is_tensor(q) else None,
                      "kpm": shape(kpm), "kpm_true": (int(kpm.sum()) if torch.is_tensor(kpm)
                                                       and kpm.dtype == torch.bool else None),
                      "attn_mask": shape(am)})

    def dec_layer_pre(mod, args_, kwargs):
        tgt = args_[0] if args_ else kwargs.get("tgt")
        mem = args_[1] if len(args_) > 1 else kwargs.get("memory")
        calls.append({"module": names.get(mod, "?"), "kind": "TransformerDecoderLayer",
                      "tgt": shape(tgt), "memory": shape(mem),
                      "mem_ptr": int(mem.data_ptr()) if torch.is_tensor(mem) else None})

    def out_hook(mod, inp, out):
        nm = names.get(mod, "?")
        outs = out if isinstance(out, (tuple, list)) else (
            list(out.values()) if isinstance(out, dict) else [out])
        for i, o in enumerate(outs):
            if torch.is_tensor(o) and o.numel() > 0:
                ptr_owner[int(o.data_ptr())] = (nm + (f"[{i}]" if len(outs) > 1 else ""), shape(o))
                keep_alive.append(o)

    for n, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            hs.append(m.register_forward_pre_hook(attn_pre, with_kwargs=True))
        elif isinstance(m, nn.TransformerDecoderLayer):
            hs.append(m.register_forward_pre_hook(dec_layer_pre, with_kwargs=True))
        # provenance registry: every non-backbone module's output (backbone excluded: RAM)
        if not n.startswith("core.encoder.net") and n:
            hs.append(m.register_forward_hook(out_hook))
    hs.append(trunk.register_forward_hook(out_hook))
    tac_in = {}

    def tac_pre(mod, args_, kwargs):
        if not tac_in:
            tac_in["cond"] = args_[0].detach().clone()
            for k in ("agent_tokens", "agent_pad", "bev_tokens", "bev_pad"):
                v = kwargs.get(k)
                tac_in[k] = None if v is None else v.detach().clone()
    if getattr(model, "tac_decoder_v6", None) is not None:
        hs.append(model.tac_decoder_v6.register_forward_pre_hook(tac_pre, with_kwargs=True))

    gs_calls = []
    _orig_gs = F.grid_sample

    def gs_wrap(input, grid, *a, **k):
        import inspect
        fr = inspect.stack()[1]
        gs_calls.append({"caller": f"{Path(fr.filename).name}:{fr.lineno} {fr.function}",
                         "input": shape(input), "grid": shape(grid),
                         "input_ptr": int(input.data_ptr()),
                         "mode": k.get("mode"), "padding_mode": k.get("padding_mode"),
                         "align_corners": k.get("align_corners")})
        return _orig_gs(input, grid, *a, **k)

    F.grid_sample = gs_wrap
    import tanitad.models.bev_lift as _bl
    import tanitad.models.refc_bev_coupling as _bc
    _bl.F.grid_sample = gs_wrap
    _bc.F.grid_sample = gs_wrap

    # ------------------------------------------------------------------ the forward
    torch.manual_seed(0)
    t1 = time.time()
    with torch.no_grad():
        losses = tr.compute_losses_v3(model, batch, "cpu", mode=getattr(args, "mode", "diffusion"),
                                      ablate_frames=bool(getattr(args, "ablate_frames", False)))
    rec["forward_s"] = round(time.time() - t1, 1)
    for h in hs:
        h.remove()
    F.grid_sample = _orig_gs
    _bl.F.grid_sample = _orig_gs
    _bc.F.grid_sample = _orig_gs
    rec["losses_keys"] = sorted(losses.keys())
    rec["cascade_in_losses"] = "cascade" in losses
    rec["losses_scalar"] = {k: float(v) for k, v in losses.items()
                            if (torch.is_tensor(v) and v.ndim == 0) or isinstance(v, (int, float))}
    print("[q3] forward done", rec["forward_s"], "s", flush=True)

    # ------------------------------------------------------------------ Q2: non-persistent buffers
    sd_keys = set(model.state_dict().keys())
    nonpersist = [(n, list(b.shape)) for n, b in model.named_buffers() if n not in sd_keys]
    rec["q2_nonpersistent_buffers"] = nonpersist
    # ------------------------------------------------------------------ Q2: does the tactical
    # decoder read WHERE a BEV cell is? (permutation of its 480 map keys)
    perm_res = {}
    if tac_in:
        dec = model.tac_decoder_v6

        def run(**over):
            kw = {k: tac_in[k] for k in ("agent_tokens", "agent_pad", "bev_tokens", "bev_pad")}
            kw.update(over)
            with torch.no_grad():
                o = dec(tac_in["cond"], **kw)
            return torch.cat([o["goal_logits"], o["goal_conf"], o["lat_logits"], o["lon_logits"]], -1)
        base = run()
        g = torch.Generator().manual_seed(0)
        bt = tac_in["bev_tokens"]
        at = tac_in["agent_tokens"]
        pb = torch.randperm(bt.shape[1], generator=g)
        pa = torch.randperm(at.shape[1], generator=g) if at is not None else None
        # the 30x16 grid mirrored laterally (a PERMUTATION of the tokens, i.e. the map flipped L/R)
        X, Y = 30, 16
        mir = torch.arange(X * Y).reshape(X, Y).flip(1).reshape(-1)
        perm_res = {
            "control_same_inputs_maxabs": float((run() - base).abs().max()),
            "bev_tokens_permuted_maxabs": float((run(bev_tokens=bt[:, pb]) - base).abs().max()),
            "bev_grid_mirrored_LR_maxabs": float((run(bev_tokens=bt[:, mir]) - base).abs().max()),
            "agent_tokens_permuted_maxabs": (float((run(agent_tokens=at[:, pa],
                                                        agent_pad=tac_in["agent_pad"][:, pa]
                                                        if tac_in["agent_pad"] is not None else None)
                                                    - base).abs().max()) if at is not None else None),
            "POSITIVE_bev_tokens_zeroed_maxabs": float((run(bev_tokens=torch.zeros_like(bt)) - base).abs().max()),
            "POSITIVE_bev_features_channel_shuffled_maxabs": float(
                (run(bev_tokens=bt[:, :, torch.randperm(bt.shape[2], generator=g)]) - base).abs().max()),
            "base_output_absmax": float(base.abs().max()),
            "n_bev_tokens": int(bt.shape[1]), "d_bev": int(bt.shape[2]),
            "n_agent_tokens": None if at is None else int(at.shape[1])}
        # ---- does the BEV FEATURE itself carry its cell position (padding / FOV edge)? ----
        # ridge probe feature -> (row, col) of the cell, fit on a random half of the 480 cells,
        # scored on the other half; lambda chosen on the FIT half only (inner 4-fold);
        # controls: constant-only (must read R2 = 0 exactly... up to the fit-half mean) and
        # labels permuted.
        feats = bt[0].double()                                   # [480, d]
        rows = torch.arange(X * Y) // Y
        cols = torch.arange(X * Y) % Y
        tgt = torch.stack([rows, cols], 1).double()

        def ridge_r2(Xf, T, seed):
            gg = torch.Generator().manual_seed(seed)
            idx = torch.randperm(Xf.shape[0], generator=gg)
            fi, ti = idx[: Xf.shape[0] // 2], idx[Xf.shape[0] // 2:]
            mu, sd = Xf[fi].mean(0), Xf[fi].std(0).clamp_min(1e-9)
            A = (Xf - mu) / sd
            tm = T[fi].mean(0)
            best, best_l = None, None
            for lam in (1e-2, 1e-1, 1, 10, 100, 1000, 1e4):
                # inner 4-fold on the fit half
                errs = []
                for k in range(4):
                    va = fi[k::4]
                    trn = torch.cat([fi[j::4] for j in range(4) if j != k])
                    At, Tt = A[trn], T[trn] - T[trn].mean(0)
                    Wt = torch.linalg.solve(At.T @ At + lam * torch.eye(A.shape[1], dtype=A.dtype), At.T @ Tt)
                    pr = A[va] @ Wt + T[trn].mean(0)
                    errs.append(float(((pr - T[va]) ** 2).mean()))
                e = sum(errs) / 4
                if best is None or e < best:
                    best, best_l = e, lam
            At, Tt = A[fi], T[fi] - tm
            Wt = torch.linalg.solve(At.T @ At + best_l * torch.eye(A.shape[1], dtype=A.dtype), At.T @ Tt)
            pr = A[ti] @ Wt + tm
            ss_res = ((pr - T[ti]) ** 2).sum(0)
            ss_tot = ((T[ti] - T[ti].mean(0)) ** 2).sum(0)
            return [float(v) for v in (1 - ss_res / ss_tot)], best_l
        r2 = [ridge_r2(feats, tgt, s_)[0] for s_ in range(5)]
        r2_perm = [ridge_r2(feats, tgt[torch.randperm(tgt.shape[0], generator=torch.Generator().manual_seed(100 + s_))], s_)[0]
                   for s_ in range(5)]
        r2_const = [ridge_r2(torch.ones_like(feats[:, :1]), tgt, s_)[0] for s_ in range(5)]
        perm_res["position_probe"] = {
            "what": "ridge: BEV token feature -> its (row, col) in the 30x16 grid; held-out half of cells",
            "n_cells": int(X * Y), "d": int(feats.shape[1]), "n_fit": int(X * Y // 2),
            "R2_row_col_per_split": r2, "R2_labels_permuted": r2_perm, "R2_constant_only": r2_const}
    rec["q2_tactical_bev_position"] = perm_res
    print("[q3] tac permutation/probe", json.dumps(perm_res)[:1500], flush=True)

    # ------------------------------------------------------------------ Q3 summary
    for c in calls:
        p = c.get("k_ptr") or c.get("mem_ptr")
        own = ptr_owner.get(p)
        c["kv_source_module"] = own[0] if own else "functional (no module output owns this pointer)"
        c["kv_source_shape"] = own[1] if own else None
        c.pop("k_ptr", None)
        c.pop("mem_ptr", None)
        c.pop("q_ptr", None)
    by_mod = {}
    for c in calls:
        by_mod.setdefault(c["module"], []).append(c)
    all_attn = [n for n, m in model.named_modules()
                if isinstance(m, (nn.MultiheadAttention, nn.TransformerDecoderLayer))]
    rec["attention_modules_total"] = len(all_attn)
    rec["attention_modules_never_called"] = [n for n in all_attn if n not in by_mod]
    def _distinct(v):
        seen = []
        for x in v:
            sig = {kk: vv for kk, vv in x.items() if kk != "module"}
            if sig not in seen:
                seen.append(sig)
        return seen
    rec["attention_calls"] = {k: {"n_calls": len(v), "distinct": _distinct(v)}
                              for k, v in by_mod.items()}
    rec["grid_sample_calls"] = gs_calls
    rec["design_literals"] = DESIGN

    # ------------------------------------------------------------------ Q1 at launch
    q1 = {"stem_calls": [{k: v for k, v in s.items() if k != "_x"} for s in stem_calls]}
    x = stem_calls[0]["_x"] if stem_calls else None
    path = _payload
    q1["payload_found"] = bool(path is not None and Path(path).exists())
    if x is not None and q1["payload_found"]:
        from PIL import Image
        d = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        buf, lens = d["jpeg_buf"], d["jpeg_len"]
        offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens.to(torch.int64), 0)])
        q1["codec"] = d.get("codec")
        n_need = x.shape[0]
        ref = []
        for j in range(t_start, t_start + n_need):
            raw = bytes(buf[int(offs[j]):int(offs[j + 1])].numpy().tobytes())
            im = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.float32) / 255.0
            ref.append(torch.from_numpy(im).permute(2, 0, 1))
        ref = torch.stack(ref)                                   # [n, 3, H, W] in [0, 1]
        ref_raw = ref.clone()
        ref[..., -EQ_ROWS:, :] = 0.0
        mean = torch.tensor(DECLARED_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(DECLARED_STD).view(1, 3, 1, 1)
        refn = (ref - mean) / std
        q1["n_frames_compared"] = int(n_need)
        q1["stem_vs_independent_max_abs"] = float((x - refn).abs().max())
        q1["stem_vs_unnormalised_max_abs"] = float((x - ref).abs().max())
        q1["stem_vs_bgr_max_abs"] = float((x - ((ref.flip(1) - mean) / std)).abs().max())
        q1["stem_vs_no_equalize_max_abs"] = float((x - ((ref_raw - mean) / std)).abs().max())
        q1["stem_input_channel_mean"] = [float(v) for v in x.mean(dim=(0, 2, 3))]
        q1["stem_input_bottom43_values_frame0"] = [sorted({round(float(v), 5) for v in
                                                           x[0, c, -EQ_ROWS:, :].unique()})
                                                   for c in range(3)]
        q1["expected_bottom43_(0-mean)/std"] = [round(-m / s, 6) for m, s in zip(DECLARED_MEAN, DECLARED_STD)]
        q1["raw_range"] = [float(ref_raw.min()), float(ref_raw.max())]
        # fold check: the folded stem conv vs conv -> BN(eval, running stats), fp32, on x[:1]
        bn = trunk.net.bn1
        with torch.no_grad():
            folded = stem(x[:1].contiguous())
            plain = F.batch_norm(F.conv2d(x[:1], stem.weight, stem.bias, stem.stride, stem.padding,
                                          stem.dilation, stem.groups),
                                 bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps)
        q1["fold_stem_rel_err_fp32"] = float((folded - plain).norm() / plain.norm())
        q1["bn1_running_mean_head"] = [round(float(v), 6) for v in bn.running_mean[:4]]
        q1["levers"] = dict(trunk.memory_levers)
        q1["norm_calls"] = int(trunk.norm_calls)
        q1["equalize_calls"] = int(getattr(trunk, "equalize_calls", 0))
    rec["q1_as_launched"] = q1
    stop.set()
    print(json.dumps(q1, indent=1, default=str))
    for k, v in rec["attention_calls"].items():
        print(k, v["n_calls"], v["distinct"])
    for g in gs_calls:
        print("grid_sample", g)
    print("never called:", rec["attention_modules_never_called"])
    C.write_json(os.environ.get("Q3_OUT", "q3_hooks_forward.json"), rec)


if __name__ == "__main__":
    main()
