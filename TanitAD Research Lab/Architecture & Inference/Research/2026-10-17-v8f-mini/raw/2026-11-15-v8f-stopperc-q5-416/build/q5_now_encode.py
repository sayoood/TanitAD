"""q5_now_encode.py -- AMENDMENT 7 (Q5, 416x1024 source), stage Q5-0: the sibling encoder, the angle-matched area pool P-A, the fp8 path, the S0-P-format dir writer.

Pure functions, CPU-testable with synthetic inputs.  The banked files (dinov3_fp8_encode_ship.py, pull_encode_pooled.py, s0p_lib.py, s0p_fit.py) stay UNEDITED.

  encode_frames / encode_episode_sib : sibling of dinov3_fp8_encode_ship.encode_episode (lines 108-130): identical decode, /255, mean/std, batches of 8, CLS+registers dropped,
                                       finite + content asserts; ONLY the token-count assert differs (640 or 1,664).  `encode_frames` takes an explicit raw-frame index list
                                       (the sparse NOW-frame batching of section 2.1); `encode_episode_sib` is the full every-2nd-frame order (C8).
  pool_pa                            : P-A (SCOPE section 4): fractional-overlap, normalised, area weights.  4x10 = 6.4 x 6.4 patches at 416 (rows 0.2-25.8), integer 4x4 cells at 256.
  to_fp8 / pool_mean                 : fp8 e4m3fn after the fp32 pool, as the builder does; `mean` is the area-weighted mean over the same field, fp16.
  write_s0p_dir / write_s0p_npz      : S0-P-format outputs (s0p_lib.FIELDS): E, y, clip, ws, eid copied BITWISE from the source; only v40 / v160 / mean replaced.
"""
import hashlib
import json
import math
import os
import shutil

import numpy as np

FIELDS = ("E", "v40", "v160", "mean", "y", "clip", "ws", "eid")      # = s0p_lib.FIELDS (asserted in the tests)
COPIED = ("E", "y", "clip", "ws", "eid")
REPLACED = ("v40", "v160", "mean")
D_FEAT = 1024

# geometry of the two frames (SCOPE section 3/4): patch grid, the rows of the 416 field the 256 frame sees, degrees of azimuth per 16-px patch
RES = {
    "416": dict(h=416, w=1024, gh=26, gw=64, r0=0.2, rspan=25.6, deg_per_patch=120.0 / 64, tokens=1664, f_ref=488.92398517830253),
    "256": dict(h=256, w=640, gh=16, gw=40, r0=0.0, rspan=16.0, deg_per_patch=120.0 / 40, tokens=640, f_ref=305.5774907364391),
}


# ------------------------------------------------------------------------------------------------------------ the sibling encoder
def _decode_png(buf):
    import torchvision
    return torchvision.io.decode_png(buf)


def decode_frames(o, raw_idx, pool=None):
    """The v2ep dict's PNG frames at the given raw indices -> uint8 [n,3,H,W] (same decode as encode_episode)."""
    import torch
    assert str(o["codec"]) == "png"
    buf, lens = o["jpeg_buf"], o["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    raw_idx = [int(j) for j in raw_idx]
    assert all(0 <= j < len(lens) and int(lens[j]) > 0 for j in raw_idx), "a requested raw frame is absent"
    fn = lambda j: _decode_png(buf[offs[j]:offs[j + 1]].clone())
    items = list(pool.map(fn, raw_idx)) if pool is not None else [fn(j) for j in raw_idx]
    return torch.stack(items)


def encode_frames(frames_u8, model, mean, std, n_special, device, expect_tokens):
    """uint8 [n,3,H,W] -> float32 [n, expect_tokens, 1024]; batches of 8; byte-for-byte the loop of encode_episode except the token-count assert."""
    import torch
    feats = []
    with torch.inference_mode():
        for j in range(0, frames_u8.shape[0], 8):
            x = frames_u8[j:j + 8].to(device).to(mean.dtype) / 255.0
            h = model(pixel_values=(x - mean) / std).last_hidden_state
            feats.append(h[:, n_special:, :].cpu())
    f = torch.cat(feats).float()
    assert expect_tokens in (640, 1664), expect_tokens
    assert f.shape[1:] == (expect_tokens, D_FEAT) and torch.isfinite(f).all()
    assert float(f.abs().mean()) > 1e-3          # content, not zeros
    return f


def encode_episode_sib(o, model, mean, std, n_special, device, expect_tokens, pool=None):
    """Sibling of encode_episode: every 2nd RAW frame, the full order; token-count assert 640 or 1,664 (the one difference)."""
    raw = list(range(0, len(o["jpeg_len"]), 2))
    return encode_frames(decode_frames(o, raw, pool), model, mean, std, n_special, device, expect_tokens)


# ------------------------------------------------------------------------------------------------------------ P-A, the angle-matched area pool
def axis_edges(res, n_cells, axis):
    """Cell edges in PATCH units along an axis: rows over [r0, r0+rspan], columns over [0, gw]."""
    g = RES[res]
    start, span = (g["r0"], g["rspan"]) if axis == "row" else (0.0, float(g["gw"]))
    return start + span * np.arange(n_cells + 1, dtype=np.float64) / n_cells


def axis_weights(edges, n_patch):
    """[n_cells, n_patch] float64: fractional overlap of each cell with each patch, row-normalised (area-weighted mean)."""
    n = len(edges) - 1
    W = np.zeros((n, n_patch), np.float64)
    for k in range(n):
        lo, hi = float(edges[k]), float(edges[k + 1])
        for p in range(int(math.floor(lo)), min(n_patch, int(math.ceil(hi)))):
            W[k, p] = max(0.0, min(hi, p + 1) - max(lo, p))
    return W / W.sum(1, keepdims=True)


def pool_pa(f, res, gh, gw):
    """f float32 [T, tokens, D] (row-major patch grid) -> float32 [T, gh*gw, D] (cells row-major).  fp32 in, float64 accumulation, fp32 out.
    With integer cells (res '256') this equals reshape(...).mean exactly up to rounding (C7)."""
    import torch
    g = RES[res]
    T, N, D = f.shape
    assert N == g["gh"] * g["gw"], (N, res)
    Wr = torch.from_numpy(axis_weights(axis_edges(res, gh, "row"), g["gh"]))
    Wc = torch.from_numpy(axis_weights(axis_edges(res, gw, "col"), g["gw"]))
    x = f.reshape(T, g["gh"], g["gw"], D).double()
    x = torch.einsum("rh,thwd->trwd", Wr, x)
    x = torch.einsum("cw,trwd->trcd", Wc, x)
    return x.reshape(T, gh * gw, D).float().contiguous()


def pool_mean(f, res):
    """Area-weighted mean over the same field as P-A (rows 0.2-25.8 at 416, all rows at 256; all columns) -> float32 [T, D] (the caller stores fp16)."""
    import torch
    g = RES[res]
    T, N, D = f.shape
    Wr = torch.from_numpy(axis_weights(axis_edges(res, 1, "row"), g["gh"]))[0]
    Wc = torch.from_numpy(axis_weights(axis_edges(res, 1, "col"), g["gw"]))[0]
    x = f.reshape(T, g["gh"], g["gw"], D).double()
    return torch.einsum("h,w,thwd->td", Wr, Wc, x).float().contiguous()


def cell_azimuth_spans(res, n_cells, projection="cylindrical"):
    """Azimuth span in degrees of each column cell.  cylindrical: edges (patches) x deg_per_patch.  'pinhole' is the MUTATION of C7: equal-PIXEL cells read as pinhole angles,
    az = atan((u - W/2)/f), which is wrong on a cylindrical projection (TRAPS.md)."""
    g = RES[res]
    e = axis_edges(res, n_cells, "col")
    if projection == "cylindrical":
        return np.diff(e) * g["deg_per_patch"]
    u = e * 16.0
    az = np.degrees(np.arctan((u - g["w"] / 2.0) / g["f_ref"]))
    return np.diff(az)


def to_fp8(x):
    """float32 -> uint8 view of float8_e4m3fn, with the builder's finite check (fp8 e4m3fn has no inf: >448 -> NaN)."""
    import torch
    q = x.to(torch.float8_e4m3fn)
    assert bool(torch.isfinite(q.float()).all()), "non-finite after fp8"
    return q.contiguous().view(torch.uint8).numpy()


def rows_from_field(f, res):
    """float32 field [n, tokens, D] -> dict(v40 u8 [n,40,D], v160 u8 [n,160,D], mean f16 [n,D])."""
    return {"v40": to_fp8(pool_pa(f, res, 4, 10)), "v160": to_fp8(pool_pa(f, res, 8, 20)), "mean": pool_mean(f, res).numpy().astype(np.float16)}


# ------------------------------------------------------------------------------------------------------------ S0-P-format writers
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def write_s0p_dir(src_dir, out_dir, v40, v160, mean, extra_meta=None):
    """src_dir = an S0-P train-side directory (C0's / s0p_extract_train.py's).  E, y, clip, ws, eid are COPIED BYTE FOR BYTE (shutil.copyfile); v40 / v160 / mean are replaced.
    Shapes and dtypes are asserted against the source's.  meta.json carries the source's meta plus `q5`; MANIFEST.sha256 lists the dir's files."""
    os.makedirs(out_dir, exist_ok=True)
    for k in COPIED:
        shutil.copyfile(os.path.join(src_dir, k + ".npy"), os.path.join(out_dir, k + ".npy"))
    n = len(np.load(os.path.join(out_dir, "y.npy")))
    for k, a, dt, tail in (("v40", v40, np.uint8, (40, 1024)), ("v160", v160, np.uint8, (160, 1024)), ("mean", mean, np.float16, (1024,))):
        assert a.dtype == dt and a.shape == (n,) + tail, (k, a.dtype, a.shape, n)
        old = np.load(os.path.join(src_dir, k + ".npy"), mmap_mode="r") if os.path.exists(os.path.join(src_dir, k + ".npy")) else None
        if old is not None:
            assert old.shape == a.shape and old.dtype == a.dtype, (k, old.shape, a.shape)
        np.save(os.path.join(out_dir, k + ".npy"), a)
    meta = json.load(open(os.path.join(src_dir, "meta.json"), encoding="utf-8"))
    meta["q5"] = dict(extra_meta or {}, replaced=list(REPLACED), copied_bitwise=list(COPIED))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    if os.path.exists(os.path.join(src_dir, "sample.json")):
        shutil.copyfile(os.path.join(src_dir, "sample.json"), os.path.join(out_dir, "sample.json"))
    names = [k + ".npy" for k in FIELDS] + ["meta.json"] + (["sample.json"] if os.path.exists(os.path.join(out_dir, "sample.json")) else [])
    with open(os.path.join(out_dir, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
        for nme in names:
            f.write(f"{sha256_file(os.path.join(out_dir, nme))}  {nme}\n")
    return out_dir


def write_s0p_npz(src_npz, out_npz, v40, v160, mean, extra_meta=None):
    """Eval side: src_npz is s0p_extract_eval's s0p_eval.npz.  E, y, clip, ws, eid, ci copied bitwise (loaded arrays re-saved unchanged); v40 / v160 / mean replaced."""
    z = np.load(src_npz, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    n = len(d["y"])
    assert v40.shape == (n, 40, 1024) and v160.shape == (n, 160, 1024) and mean.shape == (n, 1024)
    assert v40.dtype == np.uint8 and v160.dtype == np.uint8 and mean.dtype == np.float16
    meta = json.loads(str(d["meta"]))
    meta["q5"] = dict(extra_meta or {}, replaced=list(REPLACED), copied_bitwise=list(COPIED) + ["ci"])
    d.update(v40=v40, v160=v160, mean=mean, meta=np.array(json.dumps(meta, sort_keys=True)))
    np.savez(out_npz, **d)
    return sha256_file(out_npz)
