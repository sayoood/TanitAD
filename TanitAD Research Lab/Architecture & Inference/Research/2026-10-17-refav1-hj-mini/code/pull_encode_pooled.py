"""RA-1a (refav1-HJ mini): the POOLED frozen-DINOv3 cache.

Derived from the banked trunk-probe encoder
`2026-09-27-refav1-trunk-probe/code/pull_encode.py` and keeps its pull logic, model, geometry and normalisation
IDENTICAL: the clip is encoded by `stack/scripts/dinov3_fp8_encode_ship.py::encode_episode` IMPORTED (never
re-implemented) -- DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m, local HF cache, no token read),
bf16, the 256x640 cylindrical frame fed directly (16 x 40 patch grid, 640 tokens, d = 1024), CLS + 4 registers
dropped, ImageNet mean/std from the model's own processor, every 2nd raw frame (the 0.2 s grid refav1 uses).

WHAT IS NEW (and only this):
  * the float32 patch field [T, 640, 1024] is average-pooled -- non-overlapping equal cells, in fp32, BEFORE
    quantisation -- to 4x10 (primary; 4x4-token cells) and 8x20 (secondary; 2x2-token cells). Cell order is
    row-major (r*Gw + c) over the row-major 16x40 token grid, exactly extract.py's `reshape(4,4,10,4,-1)` form.
  * the patch-mean vector [T, 1024] is stored too (fp16 -- NOT fp8: it is tiny and linear-probe-facing).
  * pooled grids are stored fp8 e4m3 (torch.float8_e4m3fn), like the refav1 cache. If fp8 is unavailable or fails
    its save/load preflight the script falls back to fp16 and says so LOUDLY (token ZZFP8-FALLBACK-FP16ZZ, and
    `dtype` in every manifest line / file meta).
  * per-frame ACTIONS + KINEMATICS on the same grid, and the 8 t0 P10 features, so RA-1 needs nothing else.

OUTPUT (under --out, default C:/Users/Admin/refav1_hj_cache, per split train|eval):
    <out>/<split>/clips/<clip>.pt     ONE file per clip, torch.load(weights_only=True)-safe dict:
        meta        dict   clip_id, t_ep, n_frames, grid_dt, pipeline, model, special_dropped, dtype, grids, units
        pool4x10    fp8 [T, 40, 1024]       (always, unless --grids says otherwise)
        pool8x20    fp8 [T, 160, 1024]      (only if requested)
        mean        fp16 [T, 1024]          patch-mean over all 640 tokens
        v           f32 [T]   m/s           poses[2j, 3]  (measured speed at grid frame j)
        a           f32 [T]   m/s^2         (v[2j+2]-v[2j])/0.2, the step j OPENS at j and CLOSES at j+1  (loader `_kin_actions`
                                            verbatim; last row clamped -> 0, no window ever reads it)
        steer       f32 [T]   rad           actions[2j, 0]: a ROAD-WHEEL STEER ANGLE = arctan(L*kappa), L_enc = 2.9 m
                                            (NOT a curvature: kappa = tan(steer)/2.9; ~2.9x kappa for small angles)
        yaw_rate    f32 [T]   rad/s         wrapped (yaw[2j+2]-yaw[2j])/0.2, same step convention as `a`
        pose_grid   f32 [T,4] (m,m,rad,m/s) poses[2j] = (x, y, yaw, v): the GT waypoint source on the 0.2 s grid
        kin3        f32 [T,3]               (v0, a0, k0) of a window whose LAST OBSERVED index is t=row:
                                            v0=v[2t], a0=(v[2t]-v[2t-2])/0.2 (backward, the action that CLOSED at t),
                                            k0=steer at 2t.  Row 0 is NaN (needs one closed step).  Bit-identical to the
                                            `kin` column probe_te/extract.py banked (hold_ext_controls form).
        p10         f32 [T,8]               kin_feats(kin3) = [v, a, k, v*v, v*a, v*k, v*v*k, a*k] (probe.py:41), float64
                                            arithmetic cast to f32; row 0 NaN.  These ARE the P10 features.
    <out>/<split>/manifest.jsonl      one line per finished clip: clip, file, bytes, md5 (file), content_md5 (tensor
                                      bytes, container-independent), n_frames, t_ep, shapes, dtype, grids, enc_s, pipeline
    <out>/<split>/meta.json           the last launch's full provenance (script/encoder sha256, model, torch, gates)
    <out>/<split>/runs.jsonl          append-only launch history

SPLITS (asserted, not assumed):
  eval  : the 141 EVAL clips (v2ep read from the local eps dir -- NOTHING is pulled from Thor).
  train : B1 listing (refav1's index.json, 4,713) minus the eval 141 == the v7.2 TRAIN label release's 4,572 clips.
          Ordered by sha256(clip)[:12]; --n takes the first N (so the 600-clip prefix is nested in the 4,572).
  Disjointness is checked against FOUR independent eval sources (eval v2ep dir, eval fp8 cache dir, the 147-record
  v7.2 eval label file, and the train label file) and the parity ingest gate (tanitad.data.parity, role 'train')
  runs on the clips about to be built.

RESUMABLE: a clip is done iff its file exists, its manifest line exists, size matches, the requested grids are a
subset of the stored ones, the dtype/pipeline match and (unless --fast-resume) the file md5 matches.
A partial file never carries a final name: files are written to <out>/<split>/.tmp/ and os.replace()d.

Thor is READ-ONLY and only ever touched by `scp` of a v2ep (train split); one clip prefetched ahead.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

PIPELINE = "ra1a-pooled-v1"
MID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
THOR = "tanitad-thor-wifi"
SRC = "/home/nvidia/data/physicalai-b1-w120-256x640cyl"
DT = 0.2                      # the 0.2 s operative grid; cache index j <-> v2ep frame 2j
L_ENC = 2.9                   # the ENCODER's wheelbase: steer = arctan(L_ENC * kappa)
GRIDS = {"4x10": (4, 10), "8x20": (8, 20)}
SRC416 = "/home/nvidia/data/physicalai-b1-w120-416x1024cyl"
SOURCES = ("auto", "thor256", "thor416", "local")
GH, GW = 16, 40               # the DINOv3 patch grid of the 256x640 cylindrical frame

_HERE = Path(__file__).resolve()
try:
    REPO = _HERE.parents[5]
    assert (REPO / "stack" / "scripts" / "dinov3_fp8_encode_ship.py").exists()
except Exception:             # the repo moved / package copied elsewhere
    REPO = Path("E:/Projects/TanitAD")

D_INDEX = "C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval/index.json"
D_EVAL_CACHE = "C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval"
D_EVAL_EPS = "C:/Users/Admin/tanitad-data/refav1-eval141/eps"
D_EVAL_LABELS = "C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz"
D_TRAIN_LABELS = str(REPO / "TanitAD Research Lab/Data Engineering/Implementation/incoming/"
                     "2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz")
D_ENCODER = str(REPO / "stack/scripts/dinov3_fp8_encode_ship.py")
D_OUT = "C:/Users/Admin/refav1_hj_cache"


# --------------------------------------------------------------------------------------------- small utils
def sha12(u: str) -> str:
    return hashlib.sha256(u.encode()).hexdigest()[:12]


def file_hash(path, algo="md5", chunk=1 << 22) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tensor_bytes(t: torch.Tensor) -> bytes:
    t = t.detach().contiguous().cpu()
    return t.view(torch.uint8).numpy().tobytes() if t.numel() else b""


def content_md5(d: dict) -> str:
    """md5 over the tensors in sorted key order: key, dtype, shape, raw bytes. Container-independent, so a
    bit-exact re-encode is decidable even if the zip wrapper differs (NaN rows compare by BIT PATTERN)."""
    h = hashlib.md5()
    for k in sorted(d):
        v = d[k]
        if isinstance(v, torch.Tensor):
            h.update(f"{k}|{v.dtype}|{tuple(v.shape)}|".encode())
            h.update(tensor_bytes(v))
    return h.hexdigest()


def read_jsonl_gz(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


# --------------------------------------------------------------------------------------------- clip sets
def clip_sets(a) -> dict:
    """Resolve and ASSERT the train/eval clip sets. Nothing here touches Thor unless --listing thor."""
    eps_dir, cache_dir = Path(a.eval_eps), Path(a.eval_cache)
    ev_eps = {p.name[:-len(".v2ep.pt")] for p in eps_dir.glob("*.v2ep.pt")}
    ev_cache = {p.stem for p in cache_dir.glob("*.pt")}
    ev_lab = {r["clip_id"] for r in read_jsonl_gz(a.eval_labels)}
    tr_lab = {r["clip_id"] for r in read_jsonl_gz(a.train_labels)}
    if a.listing == "thor":
        r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", a.thor, f"ls {SRC}"],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ssh ls failed rc={r.returncode}: {r.stderr[-300:]}")
        listing = {l.strip()[:-len(".v2ep.pt")] for l in r.stdout.splitlines() if l.endswith(".v2ep.pt")}
    else:
        listing = set(json.load(open(a.listing, encoding="utf-8"))["episodes"])
    ev = ev_eps
    train_all = listing - ev
    chk = {
        "eval_eps_eq_eval_cache": ev_eps == ev_cache,
        "n_eval": len(ev),
        "eval_subset_of_listing": ev <= listing,
        "eval_subset_of_eval_labels": ev <= ev_lab,
        "train_all_eq_train_labels": train_all == tr_lab,
        "train_and_eval_eps_empty": not (train_all & ev),
        "train_and_eval_cache_empty": not (train_all & ev_cache),
        "train_and_eval_labels_empty": not (train_all & ev_lab),
        "train_labels_and_eval_labels_empty": not (tr_lab & ev_lab),
        "n_listing": len(listing), "n_train_all": len(train_all), "n_train_labels": len(tr_lab),
        "n_eval_labels": len(ev_lab), "n_eval_labels_not_in_listing": len(ev_lab - listing),
    }
    bad = [k for k, v in chk.items() if isinstance(v, bool) and not v]
    if bad or chk["n_eval"] != 141:
        raise SystemExit(f"ZZSPLIT-REFUSEDZZ clip-set assertions failed: {bad} (n_eval={chk['n_eval']}) -- "
                         f"the split is not what this cache assumes")
    return {"eval": sorted(ev), "train_all": sorted(train_all), "checks": chk}


def pick_clips(sets: dict, split: str, n: int) -> list[str]:
    if split == "eval":
        return list(sets["eval"])
    pick = sorted(sets["train_all"], key=sha12)           # the trunk-probe's rule: B1 - eval141, by sha256[:12]
    return pick[:n] if n > 0 else pick


def parity_gate(clips: list[str], split: str, skip: bool) -> dict:
    """The §10c ingest gate (tanitad.data.parity). role 'train' = supervision: the deployed VAL may not be inside.
    On the eval split it is informational only (role 'eval', mode 'exclude' -- the 141 is fixed by the prereg)."""
    if skip:
        return {"skipped": True, "why": "--skip-parity-gate"}
    for p in (REPO / "stack", REPO / "taniteval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from tanitad.data import parity as P
    role, mode = ("train", "refuse") if split == "train" else ("eval", "exclude")
    kept, rec = P.guard_corpus_build(clips, label=f"pull_encode_pooled {split}", role=role, mode=mode)
    rec = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)[:200]) for k, v in rec.items()}
    rec["n_kept"] = len(kept)
    rec["informational_only"] = split == "eval"
    return rec


# --------------------------------------------------------------------------------------------- encoder
def load_encoder_module(path: str):
    try:
        import truststore  # noqa: F401  (the ship module imports it at top level)
    except Exception:
        import types
        m = types.ModuleType("truststore"); m.inject_into_ssl = lambda: None
        sys.modules["truststore"] = m
    spec = importlib.util.spec_from_file_location("dinov3_fp8_encode_ship", path)
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc


def build_model(device: str, dtype_name: str):
    """Same construction as the banked pull_encode.py, local HF cache only (no token file is read)."""
    from transformers import AutoImageProcessor, AutoModel
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name]
    proc = AutoImageProcessor.from_pretrained(MID, local_files_only=True)
    model = AutoModel.from_pretrained(MID, local_files_only=True, dtype=dt).eval().to(device)
    mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device, dt)
    std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device, dt)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    assert n_special == 5 and tuple(proc.image_mean) == (0.485, 0.456, 0.406), "encoder geometry changed"
    return model, mean, std, n_special


def fp8_ok(work: Path) -> bool:
    """The save/load preflight of the ship script: the LOADER (weights_only) must read what we write."""
    try:
        if not hasattr(torch, "float8_e4m3fn"):
            return False
        t = torch.randn(4, 8).to(torch.float8_e4m3fn)
        p = work / "_fp8_pre.pt"
        torch.save(t, p)
        back = torch.load(p, weights_only=True)
        p.unlink()
        return back.dtype == torch.float8_e4m3fn and bool(torch.isfinite(back.float()).all())
    except Exception:
        return False


# --------------------------------------------------------------------------------------------- the maths
def pool_field(f: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """[T, 640, D] fp32 (row-major 16x40 token grid) -> [T, gh*gw, D] by non-overlapping average pooling,
    cells row-major. Exactly `reshape(gh, 16/gh, gw, 40/gw, D).mean((1,3))` of extract.py's `sp()`."""
    T, N, D = f.shape
    assert N == GH * GW and GH % gh == 0 and GW % gw == 0
    x = f.reshape(T, gh, GH // gh, gw, GW // gw, D).mean(dim=(2, 4))
    return x.reshape(T, gh * gw, D)


def kin_feats(kin: np.ndarray) -> np.ndarray:
    """probe.py:41 verbatim."""
    v, a, k = kin[:, 0], kin[:, 1], kin[:, 2]
    return np.stack([v, a, k, v * v, v * a, v * k, v * v * k, a * k], 1)


def kinematics(o: dict) -> dict[str, torch.Tensor]:
    """Per-grid-frame actions / kinematics from the v2ep dict. Index conventions copy
    `RefAV1Windows._kin_actions` (a, steer) and `refav1_arm.hold_ext_controls` (kin3) with identical float32 ops."""
    poses = o["poses"].float()
    acts = o["actions"].float()
    t_ep = poses.shape[0]
    t_c = math.ceil(t_ep / 2)
    v_full, yaw_full, steer_full = poses[:, 3], poses[:, 2], acts[:, 0]
    idx = torch.arange(t_c)
    f = idx * 2
    f_next = torch.clamp((idx + 1) * 2, max=t_ep - 1)
    a = (v_full[f_next] - v_full[f]) / DT
    dyaw = yaw_full[f_next] - yaw_full[f]
    yaw_rate = torch.atan2(torch.sin(dyaw), torch.cos(dyaw)) / DT
    v, steer = v_full[f], steer_full[f]
    kin3 = torch.full((t_c, 3), float("nan"))
    t = idx[1:]
    kin3[1:, 0] = v_full[2 * t]
    kin3[1:, 1] = (v_full[2 * t] - v_full[2 * t - 2]) / DT       # the action that CLOSED at t (backward difference)
    kin3[1:, 2] = steer_full[2 * t]
    p10 = np.full((t_c, 8), np.nan, dtype=np.float32)
    p10[1:] = kin_feats(kin3[1:].numpy().astype(np.float64)).astype(np.float32)
    return {"v": v.contiguous(), "a": a.contiguous(), "steer": steer.contiguous(), "yaw_rate": yaw_rate.contiguous(),
            "pose_grid": poses[f].contiguous(), "kin3": kin3.contiguous(), "p10": torch.from_numpy(p10)}


UNITS = {"v": "m/s", "a": "m/s^2 (v[2j+2]-v[2j])/0.2, step j closes at j+1", "steer": "rad, road-wheel angle, "
         "arctan(2.9*kappa); kappa = tan(steer)/2.9", "yaw_rate": "rad/s wrapped, same step as a",
         "pose_grid": "x[m], y[m], yaw[rad], v[m/s] at frame 2j", "kin3": "(v0, a0, steer0) at t0=row; row0 NaN",
         "p10": "[v,a,k,v^2,v*a,v*k,v^2*k,a*k] of kin3; row0 NaN", "pool": "fp32 avg of equal token cells, row-major",
         "mean": "mean over all 640 patch tokens, fp16"}


def encode_clip(o: dict, enc, model, mean, std, n_special, device, grids, qdtype) -> dict:
    """v2ep dict -> the per-clip payload. `enc.encode_episode` is the banked path, byte for byte."""
    f = enc.encode_episode(o, model, mean, std, n_special, device)           # fp32 [T, 640, 1024]
    return payload_from_field(f, o, grids, qdtype)


def payload_from_field(f: torch.Tensor, o: dict, grids, qdtype) -> dict:
    """fp32 patch field [T, 640, 1024] + the v2ep dict -> pooled / mean / kinematics payload (split out so the
    validation harness can pool one encoded field two independent ways)."""
    out = {}
    for g in grids:
        gh, gw = GRIDS[g]
        q = pool_field(f, gh, gw).to(qdtype)
        qf = q.float()
        assert bool(torch.isfinite(qf).all()), f"{g}: non-finite after {qdtype} (fp8 e4m3fn has no inf: >448 -> NaN)"
        assert float(qf.abs().mean()) > 1e-3, f"{g}: content, not zeros"
        out["pool" + g] = q.contiguous()
    m = f.mean(dim=1)
    assert bool(torch.isfinite(m).all())
    out["mean"] = m.to(torch.float16).contiguous()
    out.update(kinematics(o))
    return out


def make_meta(clip, o, payload, grids, qdtype):
    return {"clip_id": clip, "t_ep": int(o["poses"].shape[0]), "n_frames": int(payload["v"].shape[0]),
            "grid_dt": DT, "pipeline": PIPELINE, "model": MID, "special_dropped": 5, "dtype": str(qdtype).replace("torch.", ""),
            "grids": list(grids), "token_grid": [GH, GW], "L_enc": L_ENC, "units": UNITS}


# --------------------------------------------------------------------------------------------- pull / source
def run_retry(cmd, tries=3, timeout=900):
    last = None
    for k in range(tries):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last = e
            time.sleep(5 * (k + 1))
    raise last


def remote_dir_exists(thor: str, path: str) -> bool:
    r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", thor,
                        f"test -d {path} && ls {path} | head -1 | grep -q v2ep && echo YES || echo NO"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ssh {thor} failed rc={r.returncode}: {r.stderr[-200:]}")
    return r.stdout.strip().endswith("YES")


def resolve_source(a) -> dict:
    """Where the v2ep frames come from. THE 256x640 B1 SOURCE ON THOR (`physicalai-b1-w120-256x640cyl`, 178 GB) WAS
    DELETED in the 2026-10-04..12 disk cleanup (INVENTORY_THOR.md C01), and the refav1 train fp8 cache with it. What
    remains on Thor is the 416x1024 cylindrical B1 cache; `thor416` resamples it back to the 256x640 cylinder
    (scale 0.625 on both axes, rows 2:258 of 260; MEASURED to match the native frames at MAE 0.59/255, see
    raw/validation.json). `auto` prefers the native 256x640 source and says LOUDLY when it falls back."""
    if a.source == "local" or (a.split == "eval" and a.source == "auto"):
        d = a.src_dir or a.eval_eps
        if not Path(d).is_dir():
            raise SystemExit(f"ZZSOURCE-MISSINGZZ local source dir {d} does not exist")
        resize = (a.src_geom == "416")
        return {"kind": "local416-resize" if resize else "local256", "dir": str(d), "remote": False, "resize": resize}
    cand = {"thor256": ("thor256", SRC, False), "thor416": ("thor416-resize", SRC416, True)}
    order = ["thor256", "thor416"] if a.source == "auto" else [a.source]
    for k in order:
        kind, path, resize = cand[k]
        if remote_dir_exists(a.thor, path):
            if k == "thor416" and a.source == "auto":
                print("ZZSOURCE-FALLBACK-416ZZ the native 256x640 v2ep source is gone from Thor; resampling the 416x1024 "
                      "cache to the 256x640 cylinder (see validation.json V3b for the measured fidelity)", flush=True)
            return {"kind": kind, "dir": path, "remote": True, "resize": resize}
    raise SystemExit(f"ZZSOURCE-MISSINGZZ none of {[cand[k][1] for k in order]} exists on {a.thor} (use --source local "
                     f"--src-dir <dir of 256x640 or 416x1024 v2ep> --src-geom 256|416, or rebuild from HF)")


def fetch(a, src: dict, clip: str, work: Path) -> tuple[Path, bool]:
    """-> (v2ep path, is_local_source). Remote clips are scp'd READ-ONLY from Thor to `<name>.part`, then renamed."""
    if not src["remote"]:
        p = Path(src["dir"]) / f"{clip}.v2ep.pt"
        if not p.exists():
            raise FileNotFoundError(f"{sha12(clip)}: no local v2ep at {p.parent}")
        return p, True
    dst = work / f"{clip}.v2ep.pt"
    part = work / f"{clip}.v2ep.pt.part"
    run_retry(["scp", "-q", "-o", "ConnectTimeout=25", "-o", "BatchMode=yes", f"{a.thor}:{src['dir']}/{clip}.v2ep.pt", str(part)])
    os.replace(part, dst)
    return dst, False


def resize416_to_256(o: dict, png_level: int = 1) -> dict:
    """416x1024 cylindrical v2ep -> a 256x640 cylindrical v2ep that `encode_episode` reads UNCHANGED.

    Both caches come from the same cylindrical_rectify at hfov 120 deg, so f_ref scales 488.924 -> 305.577 = x 0.625
    on BOTH axes: resize to 640 x 260 (bilinear, align_corners=False, NO antialias -- measured best of four kernels) and
    keep rows 2:258 (the vertical field of view is centred). Frames are re-stored as lossless PNG, so `encode_episode`'s
    decode_png path sees exactly the resampled pixels. Only even raw frames are converted (encode_episode decodes
    range(0, len, 2) only); odd entries get length 0. poses/actions/clip_id are untouched (measured bit-identical
    between the 416 and 256 v2ep of the same clip)."""
    import torch.nn.functional as F
    import torchvision
    assert (int(o["image_h"]), int(o["image_w"])) == (416, 1024) and str(o["codec"]) == "png", "not a 416x1024 png v2ep"
    lens, buf = o["jpeg_len"], o["jpeg_buf"]
    offs = torch.cat([torch.zeros(1, dtype=lens.dtype), lens.cumsum(0)])
    chunks, new_len = [], []
    for j in range(len(lens)):
        if j % 2:
            new_len.append(0)
            continue
        fr = torchvision.io.decode_png(buf[offs[j]:offs[j + 1]].clone()).float()[None]            # [1,3,416,1024]
        r = F.interpolate(fr, size=(260, 640), mode="bilinear", align_corners=False, antialias=False)[0, :, 2:258, :]
        png = torchvision.io.encode_png(r.round().clamp(0, 255).to(torch.uint8), compression_level=png_level)
        chunks.append(png)
        new_len.append(int(png.numel()))
    o2 = dict(o)
    o2.update(jpeg_buf=torch.cat(chunks), jpeg_len=torch.tensor(new_len, dtype=lens.dtype), image_h=256, image_w=640,
              frame={"height": 256, "width": 640, "f_ref": 305.5774907364391, "projection": "cylindrical",
                     "resampled_from": "416x1024"})
    return o2


def prepare(a, src: dict, clip: str, work: Path):
    """fetch + load + assert (+ resample). Runs in the prefetch workers, so the GPU never waits on PNG work."""
    path, local = fetch(a, src, clip, work)
    try:
        o = torch.load(path, map_location="cpu", weights_only=False)
        assert o.get("clip_id") == clip, "v2ep clip_id != file stem (the label join key would be wrong)"
        assert str(o["codec"]) == "png"
        want = (416, 1024) if src["resize"] else (256, 640)
        assert (int(o["image_h"]), int(o["image_w"])) == want, f"frame size {(o['image_h'], o['image_w'])} != {want}"
        if src["resize"]:
            o = resize416_to_256(o)
    except Exception:
        if not local:
            Path(path).unlink(missing_ok=True)                 # a torn transfer must not be reused
        raise
    return o, path, local


# --------------------------------------------------------------------------------------------- manifest / resume
def load_manifest(path: Path) -> dict:
    ents = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        e = json.loads(ln)
                        ents[e["clip"]] = e                      # last line wins
                    except Exception:
                        pass                                      # a torn trailing line is simply not an entry
    return ents


def is_done(clip, ents, root: Path, grids, qdtype_name, enc_dtype, source_kind, fast: bool) -> bool:
    e = ents.get(clip)
    if (e is None or e.get("pipeline") != PIPELINE or e.get("dtype") != qdtype_name
            or e.get("enc_dtype") != enc_dtype              # a bf16 run must not inherit an fp32 smoke test
            or e.get("source") != source_kind):             # ... nor a 416-resampled clip stand in for a native one
        return False
    if not set(grids) <= set(e.get("grids", [])):
        return False
    p = root / "clips" / f"{clip}.pt"
    if not p.exists() or p.stat().st_size != e["bytes"]:
        return False
    return True if fast else file_hash(p) == e["md5"]


def provenance(a, enc_path: str) -> dict:
    import transformers
    snap = Path.home() / ".cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m/snapshots"
    st = sorted(snap.glob("*/model.safetensors"))
    return {"pipeline": PIPELINE, "script_sha256": file_hash(_HERE, "sha256"), "encoder_module": enc_path,
            "encoder_module_sha256": file_hash(enc_path, "sha256"), "model": MID,
            "model_safetensors_sha256": file_hash(st[0], "sha256") if st else None,
            "torch": torch.__version__, "transformers": transformers.__version__, "device": a.device,
            "enc_dtype": a.dtype, "argv": sys.argv[1:], "started": time.strftime("%Y-%m-%d %H:%M:%S")}


# --------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--n", type=int, default=600, help="train: first N clips by sha12 order (0 = all 4,572)")
    ap.add_argument("--only", nargs="+", metavar="CLIP", help="build exactly these clips (still asserted in-split)")
    ap.add_argument("--only-file", default=None, help="like --only, one clip id per line (chunk lists exceed cmd's 8,191-char line)")
    ap.add_argument("--grids", default="4x10,8x20", help="comma list of 4x10,8x20")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="encoder dtype (the shipped cache is bf16)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default=D_OUT)
    ap.add_argument("--tag", default="", help="suffix of the split directory, e.g. '_m416' for a matched-preprocessing eval")
    ap.add_argument("--source", choices=SOURCES, default="auto",
                    help="auto: native thor 256x640 if it exists, else thor 416x1024 resampled (loud); eval+auto: the local "
                         "native eval v2ep. thor256 | thor416 | local (with --src-dir and --src-geom)")
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--src-geom", choices=["256", "416"], default="256")
    ap.add_argument("--listing", default=D_INDEX, help="refav1 index.json (B1, 4,713) or the literal 'thor'")
    ap.add_argument("--eval-eps", default=D_EVAL_EPS)
    ap.add_argument("--eval-cache", default=D_EVAL_CACHE)
    ap.add_argument("--eval-labels", default=D_EVAL_LABELS)
    ap.add_argument("--train-labels", default=D_TRAIN_LABELS)
    ap.add_argument("--encoder-module", default=D_ENCODER)
    ap.add_argument("--thor", default=THOR)
    ap.add_argument("--prefetch-workers", type=int, default=3, help="clips prepared ahead (scp + load + 416->256 resample); 0 = none")
    ap.add_argument("--keep-src", action="store_true", help="keep scp'd v2ep files (default: delete after encode)")
    ap.add_argument("--fast-resume", action="store_true", help="resume check = size only (skip md5)")
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--max-consecutive-skips", type=int, default=5)
    ap.add_argument("--skip-parity-gate", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="resolve + assert the sets and the source, print the plan, encode nothing")
    a = ap.parse_args(argv)
    grids = [g for g in a.grids.split(",") if g]
    if not grids or any(g not in GRIDS for g in grids):
        raise SystemExit(f"--grids must be a subset of {sorted(GRIDS)}")
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    root = Path(a.out) / (a.split + a.tag)
    if not a.dry_run:                                   # a preflight must leave no directories behind
        (root / "clips").mkdir(parents=True, exist_ok=True)
        (root / ".tmp").mkdir(exist_ok=True)
    work = Path(a.out) / "src_pull"
    work.mkdir(parents=True, exist_ok=True)

    sets = clip_sets(a)
    pick = pick_clips(sets, a.split, a.n)
    if a.only_file:
        if a.only:
            raise SystemExit("--only and --only-file are exclusive")
        a.only = [l.strip() for l in open(a.only_file, encoding="utf-8") if l.strip()]
        if not a.only:
            raise SystemExit(f"--only-file {a.only_file}: no clip ids")
    if a.only:
        missing = sorted(set(a.only) - set(sets[a.split if a.split == "eval" else "train_all"]))
        if missing:
            raise SystemExit(f"--only: {len(missing)} clip(s) not in the {a.split} set (refusing; train/eval must not mix)")
        pick = sorted(a.only, key=sha12)
    ck = sets["checks"]
    if a.split == "train":
        assert not (set(pick) & set(sets["eval"])), "a pick clip is inside the eval-141"
        assert set(pick) <= set(sets["train_all"]), "a pick clip is outside B1-minus-eval"
    else:
        assert set(pick) <= set(sets["eval"]) and not (set(pick) & set(sets["train_all"]))
    print(f"ZZSPLIT-OKZZ split={a.split} pick={len(pick)} listing={ck['n_listing']} eval={ck['n_eval']} "
          f"train_all={ck['n_train_all']} train==labels={ck['train_all_eq_train_labels']} "
          f"train&eval={not ck['train_and_eval_eps_empty']}", flush=True)
    gate = parity_gate(pick, a.split, a.skip_parity_gate)
    print(f"[parity] {json.dumps({k: gate[k] for k in gate if k in ('role', 'mode', 'n_in', 'n_disqualifying', 'n_kept', 'skipped')})}",
          flush=True)
    src = resolve_source(a)
    print(f"[source] kind={src['kind']} dir={src['dir']} remote={src['remote']} resample416={src['resize']}", flush=True)

    ents = load_manifest(root / "manifest.jsonl")
    qdtype = torch.float8_e4m3fn if fp8_ok(work) else torch.float16
    qname = str(qdtype).replace("torch.", "")
    if qdtype is torch.float16:
        print("ZZFP8-FALLBACK-FP16ZZ fp8 e4m3 unavailable or failed its save/load preflight; storing fp16 (2x disk)", flush=True)
    t0 = time.time()
    todo = [c for c in pick if not is_done(c, ents, root, grids, qname, a.dtype, src["kind"], a.fast_resume)]
    print(f"[plan] {len(pick)} in pick, {len(pick) - len(todo)} already done+verified ({time.time() - t0:.0f}s check), "
          f"{len(todo)} to build, grids={grids}, dtype={qname}, device={a.device}, out={root}", flush=True)
    if a.dry_run or not todo:
        print("ZZENC-DONE done=0 skipped=0 0s (dry-run or nothing to do)ZZ", flush=True)
        return 0
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("ZZNO-CUDAZZ --device cuda requested and torch.cuda is unavailable (no silent CPU fallback)")

    enc = load_encoder_module(a.encoder_module)
    model, mean, std, n_special = build_model(a.device, a.dtype)
    prov = provenance(a, a.encoder_module)
    prov.update({"parity_gate": gate, "clip_set_checks": ck, "n_pick": len(pick), "grids": grids, "stored_dtype": qname,
                 "source": src})
    (root / "meta.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    with open(root / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"started": prov["started"], "argv": prov["argv"], "n_todo": len(todo), "source": src["kind"]}) + "\n")

    pool = ThreadPoolExecutor(max_workers=a.prefetch_workers) if a.prefetch_workers > 0 else None
    futs: dict[int, object] = {}
    ahead = max(a.prefetch_workers, 1)

    def submit(i):
        if pool is not None and i < len(todo) and i not in futs:
            futs[i] = pool.submit(prepare, a, src, todo[i], work)

    mf = open(root / "manifest.jsonl", "a", encoding="utf-8")
    done = skipped = consec = 0
    t_start = time.time()
    for k in range(ahead):
        submit(k)
    for i, c in enumerate(todo):
        submit(i + ahead)
        free_gb = shutil.disk_usage(a.out).free / 1e9
        if free_gb < a.min_free_gb:
            print(f"ZZDISK-LOW free={free_gb:.1f}GB < {a.min_free_gb}GB -- stopping cleanly after {done} clips (resumable)ZZ",
                  flush=True)
            mf.close()
            return 7
        te = time.time()
        got = None
        for attempt in (0, 1):
            try:
                got = (futs.pop(i).result() if (i in futs and attempt == 0) else prepare(a, src, c, work))
                break
            except Exception as e:                           # torn transfer: re-pull once, then skip loudly
                if attempt == 1:
                    print(f"ZZSKIP {sha12(c)} {type(e).__name__}: {str(e)[:160]}ZZ", flush=True)
        if got is None:
            skipped += 1
            consec += 1
            if consec >= a.max_consecutive_skips:
                print(f"ZZSOURCE-BROKENZZ {consec} consecutive clips failed to load from {src['kind']} -- aborting", flush=True)
                mf.close()
                return 4
            continue
        consec = 0
        o, spath, local = got
        try:
            payload = encode_clip(o, enc, model, mean, std, n_special, a.device, grids, qdtype)
            meta = make_meta(c, o, payload, grids, qdtype)
            meta["source"] = src["kind"]
            blob = dict(payload, meta=meta)
            tmp = root / ".tmp" / f"{c}.pt"
            torch.save(blob, tmp)
            final = root / "clips" / f"{c}.pt"
            os.replace(tmp, final)
            ent = {"clip": c, "sha12": sha12(c), "file": f"clips/{c}.pt", "bytes": final.stat().st_size,
                   "md5": file_hash(final), "content_md5": content_md5(payload), "n_frames": meta["n_frames"],
                   "t_ep": meta["t_ep"], "dtype": qname, "enc_dtype": a.dtype, "device": a.device, "source": src["kind"],
                   "mean_dtype": "float16", "grids": grids,
                   "shapes": {k: list(v.shape) for k, v in payload.items()}, "enc_s": round(time.time() - te, 2),
                   "pipeline": PIPELINE}
        except Exception as e:            # a content assertion / CUDA error on ONE clip: loud, counted, never half-written
            print(f"ZZSKIP-ENCODE {sha12(c)} {type(e).__name__}: {str(e)[:160]}ZZ", flush=True)
            (root / ".tmp" / f"{c}.pt").unlink(missing_ok=True)
            skipped += 1
            consec += 1
            if consec >= a.max_consecutive_skips:
                print(f"ZZSOURCE-BROKENZZ {consec} consecutive clips failed -- aborting", flush=True)
                mf.close()
                return 4
            continue
        mf.write(json.dumps(ent) + "\n")
        mf.flush()
        os.fsync(mf.fileno())
        ents[c] = ent
        done += 1
        if not local and not a.keep_src:
            Path(spath).unlink(missing_ok=True)
        del o, payload, blob
        if i < 3 or (i + 1) % 25 == 0:
            el = time.time() - t_start
            print(f"[pull_encode_pooled] {i + 1}/{len(todo)} {sha12(c)} T={ent['n_frames']} {el:.0f}s "
                  f"({el / (i + 1):.2f} s/clip, free {free_gb:.0f} GB)", flush=True)
    mf.close()
    el = time.time() - t_start
    print(f"ZZENC-DONE done={done} skipped={skipped} {el:.0f}s {el / max(done, 1):.2f}s_per_clipZZ", flush=True)
    return 0 if skipped == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
