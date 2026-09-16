#!/usr/bin/env python3
"""H4 copy of P4a (REFCV6_CLARIFICATION §6.4): --ckpt makes the probed checkpoint the one variable. Frozen REF-C trunk tokens (+ a raw-pixel floor) for every stacked row of the
139 B1 EVAL clips. VISION-ONLY input, reproduced from the trainer's own code path.

WHAT THE TRUNK SAW IN TRAINING, and where each fact is read (never guessed):
  * frames: `*.v2ep.pt` PNG payloads at 256x640 CYLINDRICAL (`frame` dict in the payload),
    decoded + D-015 3-frame channel-stacked by `tanitad.data.v2_dataset._decode_stacked`
    -- the SAME function the trainer's `V2CompressedCache` calls. Oldest frame first,
    current frame in the LAST 3 channels (`comma2k19.stack_frames`).
  * dtype/scale: uint8 -> float32 [0,1] by `x.float().div_(<0-dim DEVICE tensor 255>)`,
    `refc_v3_train.py::frames_to_device` (the tensor divisor is bit-exact on CUDA; a Python
    scalar is 1 ulp off on 126/256 values). No normalisation anywhere in the model
    (`refc.py` ResNetEncoder: bare Conv2d stem).
  * numerics: fp32 -- `refc_v3_train.py` has no autocast (grep: 0 hits).
  * augmentation: none (grep `augment|jitter|flip|crop` over `refc_v3_train.py` and
    `train_flagship4b.py`: 0 hits).
  * encoder: `refc.ResNetEncoder(CNNEncoderConfig(in_channels=9, image_size=256,
    image_width=640, base_width=88, blocks=(3,6,16,6)))` -- `--size base` +
    `--image-hw 256 640` in `config.json['argv']`; weights = the 396 `core.encoder.*` keys
    of `ckpt_40284.pt`, loaded STRICT.
  * which map: `RefCModel.forward` uses the LAST window frame's `fmap` [B,704,8,20]
    (`refc.py:3161`), i.e. exactly encoder(stacked row j).

Outputs (local disk, NOT staged -- regenerable, ~9 GB):
  tokens_s32_fp16.npy  [N,704,8,20] float16   (fp32 inference, fp16 storage; the rounding
                                               error is MEASURED and written to the index)
  pix64_u8.npy         [N,9,64,160] uint8     (4x4 area mean of the same [0,255] stack)
  index.npz            clip ordinal, stacked row j, raw frame i = j+2, clip sha12 list

⛔ Clip ids never leave this process: the index carries sha12 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
JOIN = Path(r"C:/Users/Admin/tanitad-snap-20260915/TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz")
V2EP_DIR = Path(r"D:\Projects\TanitAD-artifacts\refcv5cmp\data\eval")
CKPT = Path(r"C:\Users\Admin\hf-refcv5v2\ckpt_40284.pt")
OUT = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913\tokens")


def sha12(c: str) -> str:
    return hashlib.sha256(c.encode()).hexdigest()[:12]


def join_clips() -> list[str]:
    """H4: the SAME 139 eval clips in the SAME order as the banked `tokens/index.npz`.

    The 2026-09-06 join file is not in this snapshot, so the clip set and its order are recovered from the
    banked index's `clip_sha12` (sha256(clip_id)[:12]) mapped onto the v2ep payloads present on this box.
    The resulting index.npz is asserted row-identical to the banked one in main(), which is what makes the
    two H4 extractions comparable with each other AND with the banked refcv5-v2 panel.
    """
    ref = np.load(Path(r"C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/index.npz"), allow_pickle=True)
    want = [str(s) for s in ref["clip_sha12"]]
    by_sha = {sha12(f.name.split(".")[0]): f.name.split(".")[0] for f in V2EP_DIR.glob("*.v2ep.pt")}
    missing = [s for s in want if s not in by_sha]
    if missing:
        raise SystemExit(f"[h4] {len(missing)} of {len(want)} banked clips have no v2ep payload on this box")
    return [by_sha[s] for s in want]


def build_encoder(device):
    from tanitad.refs import refc
    cfg = refc.CNNEncoderConfig(in_channels=9, image_size=256, image_width=640,
                                base_width=88, blocks=(3, 6, 16, 6))
    enc = refc.ResNetEncoder(cfg)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    sd = {k[len("core.encoder."):]: v for k, v in ck["model"].items()
          if k.startswith("core.encoder.")}
    missing, unexpected = enc.load_state_dict(sd, strict=True)
    n_params = sum(p.numel() for p in enc.parameters())
    info = {"n_encoder_keys_in_ckpt": len(sd), "missing": list(missing),
            "unexpected": list(unexpected), "encoder_params": n_params,
            "ckpt_step": int(ck.get("step", -1)), "grid_shape": list(cfg.grid_shape),
            "feat_dim": cfg.feat_dim}
    del ck
    return enc.to(device).eval(), info


def main() -> int:
    global CKPT, V2EP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--ckpt", default=str(CKPT), help="H4: the checkpoint whose core.encoder.* is probed")
    ap.add_argument("--v2ep", default=str(V2EP_DIR), help="H4: the v2ep frame payload dir on this box")
    ap.add_argument("--no-pix", action="store_true", help="H4: skip the pixel floor (checkpoint-independent, already banked)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rebuilt-eval", action="store_true",
                    help="AMENDMENT A1: the 139 eval clips from LOCALLY REBUILT frames -> tokens_rb/ "
                         "(row index asserted identical to tokens/index.npz)")
    ap.add_argument("--extra", action="store_true",
                    help="lever L4: the P6b extra TRAINING clips (frames-only payloads rebuilt from "
                         "mp4, OK rows of bev_gt_extra/manifest.jsonl) -> tokens_extra/")
    ap.add_argument("--what", default="s32pix", choices=["s32pix", "s16"],
                    help="s32pix: final stride-32 map + pixel floor; s16: the stage-3 "
                         "stride-16 map [352,16,40] (pre-registered lever L1)")
    args = ap.parse_args()
    CKPT = Path(args.ckpt)

    import tanitad
    from tanitad.data.v2_dataset import _decode_stacked, _jpeg_offsets
    print(f"[p4a] tanitad from {tanitad.__file__}", flush=True)
    dev = torch.device("cuda")
    enc, info = build_encoder(dev)
    print(f"[p4a] encoder {info}", flush=True)
    div = torch.full((), 255.0, device=dev, dtype=torch.float32)

    V2EP_DIR = Path(args.v2ep)
    if args.extra:
        import p6b_extra_corpus as X
        man = {}
        for line in (X.GT_EXTRA / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            man[r["clip_sha12"]] = r
        sel = json.loads((HERE.parent / "raw" / "p6b_extra_selection.json").read_text(encoding="utf-8"))
        by_sha = {}
        for f in X.FRAMES_DIR.glob("*.v2ep.pt"):
            cid = f.name.split(".")[0]
            by_sha[sha12(cid)] = cid
        clips = [by_sha[s] for s in sel["selected_sha12"] if man.get(s, {}).get("ok") and s in by_sha]
        V2EP_DIR = X.FRAMES_DIR
        if args.out == str(OUT):
            args.out = str(OUT.parent / "tokens_extra")
        print(f"[p4a] EXTRA clips with an OK GT artifact and a frames payload: {len(clips)}", flush=True)
    elif args.rebuilt_eval:
        clips = join_clips()
        V2EP_DIR = Path(r"C:\Users\Admin\tanitad-caches\bevhead-20260913\bevframes_eval_rb")
        if args.out == str(OUT):
            args.out = str(OUT.parent / "tokens_rb")
        print(f"[p4a] REBUILT-FRAME eval clips: {len(clips)} from {V2EP_DIR}", flush=True)
    else:
        clips = join_clips()
    if args.limit:
        clips = clips[: args.limit]
    # row counts first, so the memmaps are allocated once at their exact size
    rows = []
    for ci, c in enumerate(clips):
        d = torch.load(V2EP_DIR / f"{c}.v2ep.pt", map_location="cpu", weights_only=False, mmap=True)
        T = len(d["jpeg_len"])
        k = int(d["n_stack"]) - 1
        if (int(d["image_h"]), int(d["image_w"])) != (256, 640) or str(d.get("codec")) != "png":
            raise SystemExit(f"[p4a] unexpected payload geometry/codec for clip {sha12(c)}")
        rows += [(ci, j, j + k) for j in range(T - k)]
    N = len(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "s32pix":
        tok = np.lib.format.open_memmap(out / "tokens_s32_fp16.npy", mode="w+",
                                        dtype=np.float16, shape=(N, 704, 8, 20))
        pix = None if args.no_pix else np.lib.format.open_memmap(out / "pix64_u8.npy", mode="w+",
                                        dtype=np.uint8, shape=(N, 9, 64, 160))
    else:
        tok = np.lib.format.open_memmap(out / "tokens_s16_fp16.npy", mode="w+",
                                        dtype=np.float16, shape=(N, 352, 16, 40))
        pix = None
    print(f"[p4a] N={N} rows over {len(clips)} clips", flush=True)

    t0 = time.time()
    n = 0
    max_abs = 0.0
    fp16_err = 0.0
    fp32_vs_fp16_checked = False
    with torch.inference_mode():
        for ci, c in enumerate(clips):
            d = torch.load(V2EP_DIR / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
            offs = _jpeg_offsets(d["jpeg_len"])
            ns = int(d["n_stack"])
            T = len(d["jpeg_len"])
            n_rows = T - (ns - 1)
            for a in range(0, n_rows, args.batch):
                b = min(a + args.batch, n_rows)
                x_u8 = _decode_stacked(d["jpeg_buf"], offs, ns, a, b, str(d["codec"]))  # [B,9,256,640] u8
                x = x_u8.to(dev, non_blocking=True).float().div_(div)
                if args.what == "s32pix":
                    fmap, _ = enc(x)
                else:                         # stem + stages[0..2] = stride 16, 352 ch
                    h = enc.stem(x)
                    for st in enc.stages[:3]:
                        h = st(h)
                    fmap = h
                f = fmap.float()
                max_abs = max(max_abs, float(f.abs().max()))
                f16 = f.half()
                fp16_err = max(fp16_err, float((f16.float() - f).abs().max()))
                if not fp32_vs_fp16_checked:
                    fp32_vs_fp16_checked = True
                    info["fmap_shape"] = list(f.shape[1:])
                tok[n:n + (b - a)] = f16.cpu().numpy()
                if pix is not None:
                    p = torch.nn.functional.avg_pool2d(x_u8.to(dev).float(), 4)  # [B,9,64,160]
                    pix[n:n + (b - a)] = p.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                n += b - a
            if ci % 10 == 0 or ci == len(clips) - 1:
                el = time.time() - t0
                print(f"[p4a] clip {ci + 1}/{len(clips)} rows {n}/{N} "
                      f"{n / max(el, 1e-9):.1f} rows/s max|tok| {max_abs:.2f}", flush=True)
    tok.flush()
    if pix is not None:
        pix.flush()
    if n != N:
        raise SystemExit(f"[p4a] wrote {n} rows, expected {N}")

    # ---- content assertions on the FILES, not on memory ------------------------
    tk = np.load(out / ("tokens_s32_fp16.npy" if args.what == "s32pix" else "tokens_s16_fp16.npy"),
                 mmap_mode="r")
    px = np.load(out / "pix64_u8.npy", mmap_mode="r") if not args.no_pix else None
    probe = np.linspace(0, N - 1, 64).round().astype(int)
    tk_s = tk[probe].astype(np.float32)
    px_s = px[probe] if px is not None else np.zeros((1, 1), np.uint8)
    checks = {
        "tokens_finite": bool(np.isfinite(tk_s).all()),
        "tokens_nonzero_frac": float((np.abs(tk_s) > 0).mean()),
        "tokens_std_across_probe_rows": float(tk_s.std(axis=0).mean()),
        "pix_nonzero_frac": float((px_s > 0).mean()) if px is not None else 1.0,
        "pix_mean": float(px_s.mean()) if px is not None else 0.0,
        "max_abs_token_fp32": max_abs,
        "max_fp16_rounding_error": fp16_err,
    }
    ok = (checks["tokens_finite"] and checks["tokens_nonzero_frac"] > 0.05
          and checks["tokens_std_across_probe_rows"] > 1e-3 and checks["pix_nonzero_frac"] > 0.5
          and max_abs < 60000)
    idx = np.array(rows, dtype=np.int32)
    if not args.limit:
        ref = np.load(Path(r"C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/index.npz"), allow_pickle=True)
        same = (np.array_equal(ref["clip_ordinal"], idx[:, 0]) and np.array_equal(ref["stacked_row"], idx[:, 1])
                and np.array_equal(ref["raw_frame"], idx[:, 2])
                and [str(s) for s in ref["clip_sha12"]] == [sha12(c) for c in clips])
        if not same:
            raise SystemExit("[h4] rows/clips differ from the banked tokens/index.npz -- refusing")
        print("[h4] index row-identical to the banked panel", flush=True)
    if args.rebuilt_eval:
        ref = np.load(OUT / "index.npz")
        if not (np.array_equal(ref["clip_ordinal"], idx[:, 0]) and np.array_equal(ref["stacked_row"], idx[:, 1])
                and np.array_equal(ref["raw_frame"], idx[:, 2])):
            raise SystemExit("[p4a] rebuilt-eval rows differ from tokens/index.npz -- refusing")
    if args.what == "s16":
        old = np.load(out / "index.npz")
        if not (np.array_equal(old["clip_ordinal"], idx[:, 0]) and np.array_equal(old["stacked_row"], idx[:, 1])):
            raise SystemExit("[p4a] s16 rows do not match the s32 index -- refusing")
        (out / "index_s16_meta.json").write_text(json.dumps({**info, **checks, "N": N, "what": "s16",
                                                             "content_ok": ok}, indent=1))
        print(json.dumps({**info, **checks, "content_ok": ok, "N": N}, indent=1), flush=True)
        return 0 if ok else 2
    np.savez(out / "index.npz", clip_ordinal=idx[:, 0], stacked_row=idx[:, 1],
             raw_frame=idx[:, 2], clip_sha12=np.array([sha12(c) for c in clips]),
             meta_json=np.array(json.dumps({**info, **checks, "N": N,
                                            "rows_per_s": n / (time.time() - t0),
                                            "content_ok": ok})))
    print(json.dumps({**info, **checks, "content_ok": ok, "N": N,
                      "wall_s": round(time.time() - t0, 1)}, indent=1), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
