"""PREREG_V8F_LORA Amendment 2: the SAME-DEVICE identity control -- the on-the-fly B = 0 encoder against the PRODUCTION cache (``cache_full``), on the device that built it.

    python v8f_lora_identity_dev.py --cache-root D:/tanitad/cache_full --device cuda \\
        --train-frames <dir of <clip>.v2ep.pt> --train-clips A B ... --eval-frames <dir> --eval-clips C D ... --out identity_lora.json [--mutant NAME]

WHAT IS COMPARED. For every declared clip: the frames (``<clip>.v2ep.pt``, built by the PRODUCTION route -- ``pod_build_b1_epcache.py``, the route
``raw/2026-10-25-lora-route/ROUTE.md`` reproduced byte for byte) are encoded by ``v8f_lora_encoder.DinoV3LoRAEncoder`` at B = 0 in the CACHE BUILDER'S numerics
(weights loaded in pure bf16, NO autocast, batches of 8 from cache step 0, ``inference_mode``), the fp32 field is moved to the CPU, pooled with the builder's own
``pool_field`` (4 x 10) and only THEN quantised to fp8 e4m3 -- exactly ``encode_episode`` + ``payload_from_field`` -- and compared with the pooled tokens STORED
in the production cache for that clip. Reported per clip: rel-L2 (float64 norms of the fp8 values), max abs, and the fraction of fp8 BYTES that are equal.

THE BAR. PASS iff rel-L2 <= tau_device = 1e-3 on EVERY clip (Amendment 2). The reading also has to be QUALIFYING, else it is a FAIL whatever the numbers say: the
device is a CUDA device (Amendment 2 withdrew the CPU tau_route -- CPU fp32 vs the GPU-bf16 cache reads rel-L2 0.03 by construction), EVERY cache frame of every clip is
compared (a subset batches differently), no autocast, and at least 2 train and 2 eval clips are covered.

THE PRODUCTION ROWS ARE ASSERTED BEFORE ANYTHING IS COMPARED (and before the model is loaded). The manifest row of every declared clip must read
``source = local256, enc_dtype = bf16, dtype = float8_e4m3fn, device = cuda``; the stored file's md5 and the payload's ``content_md5`` must equal the row's; the row's
``n_frames`` / ``t_ep`` must equal the frames file's. A row that reads ``thor416`` (the dev-box test route), or a cache that is not the one the row describes, is REFUSED
(rc 2): comparing against a non-production reference is exactly what ``raw/2026-10-25-lora/identity_control.py`` did (``:38-41``, ``:212``), and why its numbers are no
route evidence. The json records the manifest row each clip matched.

``--mutant NAME`` changes ONE preprocessing constant of the mirror (``v8f_lora_encoder.PIXEL_SCALE`` / ``IMAGE_MEAN`` / ``IMAGE_STD``) and must FAIL: the instrument is
blind if it does not. In that mode rc 0 means the mutant was CAUGHT (a qualifying run whose numbers failed the bar), rc 1 that it was NOT.

``--manifest-only`` asserts the rows (and, where the files are present, their md5s) and stops: no frames, no model, no GPU. The chain runs it as a preflight on the
manifests it pulled from box 2.

rc: 0 PASS (or mutant caught, or manifest-only OK) | 1 FAIL (or mutant not caught) | 2 REFUSED (a row, a file, a device or an argument was wrong; nothing was compared).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
for _p in (REPO / "stack", REPO / "taniteval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from tanitad.models import v8f_lora_encoder as E  # noqa: E402

TAU_DEVICE = 1e-3                  # Amendment 1 item 2 (tau_device), unchanged by Amendment 2
BUILDER_BATCH = 8                  # dinov3_fp8_encode_ship.encode_episode: ``for j in range(0, frames.shape[0], 8)``
MIN_TRAIN, MIN_EVAL = 2, 2         # Amendment 2: ">= 2 train and >= 2 eval clips"
GRID = "4x10"
#: what a PRODUCTION cache row reads (MEASURED 2026-10-25: 4,572 / 4,572 train rows and 141 / 141 eval rows, ``probe_provenance.json``)
PRODUCTION_ROW = {"source": "local256", "enc_dtype": "bf16", "dtype": "float8_e4m3fn"}
PRODUCTION_DEVICE_PREFIX = "cuda"
REQUIRED_DEVICE_TYPE = "cuda"      # the device type of a QUALIFYING reading (Amendment 2); a module constant so the tests can exercise the PASS path on CPU
EXPECTED_TRAINABLE = 3_145_728     # PREREG_V8F_LORA: 32 x (1024 + 1024) x 2 x 24
#: one preprocessing constant each; every one reads rel-L2 >= 2e-2 against the builder at the same device (``identity_control.json`` SAME_DEVICE case): >= 20 x tau
MUTANTS = {"PIXEL_SCALE_255_to_256": ("PIXEL_SCALE", 256.0),
           "IMAGE_MEAN0_0.485_to_0.486": ("IMAGE_MEAN", (0.486, 0.456, 0.406)),
           "IMAGE_STD2_0.225_to_0.226": ("IMAGE_STD", (0.229, 0.224, 0.226))}
DEFAULT_MUTANT = "PIXEL_SCALE_255_to_256"


class Refusal(Exception):
    """Nothing was compared: a row, a file, a device or an argument was not what the control requires. rc 2."""


# --------------------------------------------------------------------------------------------- the production rows
def file_md5(path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def file_sha256(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def read_manifest(path) -> dict:
    """``manifest.jsonl`` -> ``{clip: row}``, the LAST line per clip winning (the trainer's ``read_manifest`` rule)."""
    p = Path(path)
    if not p.is_file():
        raise Refusal(f"the manifest {p} does not exist (looked there and nowhere else)")
    rows = {}
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            d = json.loads(ln)
            rows[d["clip"]] = d
    return rows


def production_problems(row: dict, clip: str) -> list:
    """Why this manifest row is NOT a production-cache row (empty = it is). The checks are on what the row SAYS about how the stored tokens were made."""
    bad = []
    if row.get("clip") != clip:
        bad.append(f"row is for clip {row.get('clip')!r}, not {clip!r}")
    for k, want in PRODUCTION_ROW.items():
        if row.get(k) != want:
            bad.append(f"{k} = {row.get(k)!r}, production rows read {want!r}")
    if not str(row.get("device", "")).startswith(PRODUCTION_DEVICE_PREFIX):
        bad.append(f"device = {row.get('device')!r}, production rows were encoded on a {PRODUCTION_DEVICE_PREFIX} device")
    if GRID not in (row.get("grids") or []):
        bad.append(f"grids {row.get('grids')!r} has no {GRID}")
    shp = (row.get("shapes") or {}).get("pool" + GRID)
    if not shp or list(shp[1:]) != [40, 1024]:
        bad.append(f"shapes.pool{GRID} = {shp!r}, expected [T, 40, 1024]")
    if shp and row.get("n_frames") != shp[0]:
        bad.append(f"n_frames {row.get('n_frames')!r} != shapes.pool{GRID}[0] {shp[0]!r}")
    for k in ("md5", "content_md5", "file", "t_ep"):
        if row.get(k) in (None, ""):
            bad.append(f"the row has no {k}")
    return bad


# --------------------------------------------------------------------------------------------- the comparison and the verdict
def compare_pooled(mine: torch.Tensor, ref: torch.Tensor) -> dict:
    """``mine`` / ``ref``: fp8 e4m3fn ``[T, 40, 1024]`` (the cache's storage dtype). rel-L2 on float64 norms of the fp8 VALUES; ``frac_bitwise_equal`` on the fp8 BYTES."""
    if mine.dtype != torch.float8_e4m3fn or ref.dtype != torch.float8_e4m3fn:
        raise ValueError(f"compare_pooled takes fp8 e4m3fn tensors, got {mine.dtype} / {ref.dtype}")
    if mine.shape != ref.shape:
        raise ValueError(f"shape {tuple(mine.shape)} != {tuple(ref.shape)}")
    a, b = mine.double(), ref.double()
    d = (a - b)
    nb = float(b.norm())
    if not bool(torch.isfinite(a).all()) or nb == 0.0:
        rel = float("inf")                                    # a non-finite or all-zero reading is never a pass
    else:
        rel = float(d.norm()) / nb
    return {"n_values": int(mine.numel()), "rel_l2": rel, "max_abs": float(d.abs().max()) if d.numel() else 0.0,
            "frac_bitwise_equal": float((mine.view(torch.uint8) == ref.view(torch.uint8)).double().mean())}


def decide(per_clip: dict, *, tau: float = TAU_DEVICE, scope: dict) -> dict:
    """Amendment 2: PASS iff rel-L2 <= tau on EVERY clip AND the reading is qualifying. ``per_clip``: ``{clip: {"split", "rel_l2", ...}}``;
    ``scope``: ``{"device_type", "all_frames", "autocast"}``. Pure: no device, no file."""
    if not per_clip:
        return {"numbers_pass_tau": False, "worst_rel_l2": None, "worst_clip": None, "qualifying": False,
                "qualifying_reasons": ["no clip was compared"], "verdict": "FAIL", "fail_reasons": ["no clip was compared"]}
    worst = max(per_clip, key=lambda c: (math.inf if not math.isfinite(per_clip[c]["rel_l2"]) else per_clip[c]["rel_l2"]))
    over = [c for c, v in per_clip.items() if not (v["rel_l2"] <= tau)]           # NaN is over
    n_tr = sum(1 for v in per_clip.values() if v.get("split") == "train")
    n_ev = sum(1 for v in per_clip.values() if v.get("split") == "eval")
    reasons = []
    if scope.get("device_type") != REQUIRED_DEVICE_TYPE:
        reasons.append(f"device {scope.get('device_type')!r} is not the {REQUIRED_DEVICE_TYPE} device that built cache_full (Amendment 2: the control is SAME-DEVICE)")
    if not scope.get("all_frames"):
        reasons.append("not every cache frame of every clip was compared (a subset batches differently from the builder)")
    if scope.get("autocast"):
        reasons.append("autocast was on (Amendment 2 pins --amp OFF: the builder's pure-bf16 numerics)")
    if n_tr < MIN_TRAIN or n_ev < MIN_EVAL:
        reasons.append(f"{n_tr} train / {n_ev} eval clips, Amendment 2 needs >= {MIN_TRAIN} / >= {MIN_EVAL}")
    fails = []
    if over:
        fails.append(f"rel-L2 > {tau:g} on {len(over)} of {len(per_clip)} clips: " + ", ".join(f"{c[:8]} {per_clip[c]['rel_l2']:.3e}" for c in over))
    fails += reasons
    return {"numbers_pass_tau": not over, "worst_rel_l2": per_clip[worst]["rel_l2"], "worst_clip": worst, "qualifying": not reasons, "qualifying_reasons": reasons,
            "verdict": "PASS" if not fails else "FAIL", "fail_reasons": fails}


# --------------------------------------------------------------------------------------------- the encoder
def build_encoder(device: str):
    """The arm's encoder at B = 0 (rank 32, alpha 32, q / v, 24 blocks), weights loaded in pure bf16 from the local HF cache and the safetensors sha256
    checked against the cache builder's. Tests replace THIS, never :func:`run`."""
    return E.DinoV3LoRAEncoder.from_pretrained(r=32, alpha=32.0, targets=("q", "v"), seed=0, grid=GRID, quantize=True, device=device, dtype=torch.bfloat16,
                                               verify_sha=True)


def apply_mutant(enc, name: str, device) -> callable:
    """Change ONE preprocessing constant of the mirror; -> ``restore()``."""
    attr, val = MUTANTS[name]
    old, saved = getattr(E, attr), (enc.mean, enc.std)
    setattr(E, attr, val)
    enc.mean, enc.std = E.mean_std(saved[0].dtype, device)                  # the mutated constants, through the builder's rounding to the encoder's dtype

    def restore():
        setattr(E, attr, old)
        enc.mean, enc.std = saved
    return restore


def encode_pooled(enc, o: dict, steps, pool_field) -> torch.Tensor:
    """The builder's path for these cache steps: batches of 8, ``inference_mode``, the fp32 field on the CPU, ``pool_field`` over the whole set, THEN fp8."""
    feats = []
    with torch.inference_mode():
        for j in range(0, len(steps), BUILDER_BATCH):
            feats.append(enc.field(E.decode_cache_frames(o, steps[j:j + BUILDER_BATCH])).cpu())
    f = torch.cat(feats)
    if not bool(torch.isfinite(f).all()):
        raise FloatingPointError("the encoder produced a non-finite field")
    return pool_field(f, 4, 10).to(torch.float8_e4m3fn).contiguous()


def spread(n: int, k: int) -> list:
    """``k`` cache steps spread over ``n`` (k <= 0 or >= n: all of them)."""
    if k <= 0 or k >= n:
        return list(range(n))
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


# --------------------------------------------------------------------------------------------- the run
def _load_clip(split: str, clip: str, rows: dict, cache_root: Path, need_tokens: bool) -> dict:
    """Everything one clip needs, with every refusal BEFORE any model work."""
    row = rows.get(clip)
    if row is None:
        raise Refusal(f"{split} clip {clip} has no row in {cache_root / split / 'manifest.jsonl'}")
    bad = production_problems(row, clip)
    if bad:
        raise Refusal(f"{split} clip {clip[:8]}: its manifest row is not a production-cache row: " + "; ".join(bad))
    out = {"row": row, "split": split}
    tok = cache_root / split / row["file"]
    if tok.is_file():
        md5 = file_md5(tok)
        if md5 != row["md5"]:
            raise Refusal(f"{split} clip {clip[:8]}: the stored file {tok} has md5 {md5} but the manifest row says {row['md5']}: it is not the cache the row describes")
        out["stored_path"], out["stored_md5"] = tok, md5
    elif need_tokens:
        raise Refusal(f"{split} clip {clip[:8]}: the production cache file {tok} does not exist (the control compares against the STORED tokens; no other reference is used)")
    return out


def run(a, build=None) -> tuple:
    """-> ``(result dict, rc)``. ``build`` replaces :func:`build_encoder` (tests)."""
    build = build or build_encoder
    t0 = time.time()
    cache_root = Path(a.cache_root)
    res = {"what": "PREREG_V8F_LORA Amendment 2: same-device identity control (B = 0 on-the-fly encoder vs the production cache)", "tau_device": a.tau,
           "mutant": a.mutant, "manifest_only": bool(a.manifest_only), "cache_root": str(cache_root), "argv": sys.argv[1:], "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "driver_sha256": file_sha256(HERE), "clips": {}, "torch": torch.__version__}
    try:
        if a.mutant and a.mutant not in MUTANTS:
            raise Refusal(f"unknown mutant {a.mutant!r}; one of {sorted(MUTANTS)}")
        if not (a.train_clips or a.eval_clips):
            raise Refusal("no clips declared")
        clips = [("train", c) for c in a.train_clips] + [("eval", c) for c in a.eval_clips]
        if len({c for _, c in clips}) != len(clips):
            raise Refusal("a clip is declared twice")
        man = {s: read_manifest(cache_root / s / "manifest.jsonl") for s in sorted({s for s, _ in clips})}
        loaded = {c: _load_clip(s, c, man[s], cache_root, need_tokens=not a.manifest_only) for s, c in clips}
        res["manifest_rows"] = {c: v["row"] for c, v in loaded.items()}
        if a.manifest_only:
            res.update(verdict="MANIFEST-OK", n_clips=len(clips), stored_files_checked=sum(1 for v in loaded.values() if "stored_md5" in v))
            return res, 0
        dev = torch.device(a.device)
        res["device"] = str(dev)
        if dev.type != REQUIRED_DEVICE_TYPE and not a.allow_non_cuda:
            raise Refusal(f"device {dev}: Amendment 2 makes this control SAME-DEVICE on the cuda device that built cache_full; pass --allow-non-cuda for a labelled, NON-QUALIFYING reading")
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise Refusal("--device cuda but torch.cuda.is_available() is False")
        if torch.is_autocast_enabled():
            raise Refusal("autocast is enabled in this process: Amendment 2 pins the builder's pure-bf16 numerics")
        frames = {}
        for s, c in clips:
            root = Path(a.train_frames if s == "train" else a.eval_frames or "")
            p = root / f"{c}.v2ep.pt"
            if not p.is_file():
                raise Refusal(f"{s} clip {c[:8]}: the frames file {p} does not exist (built by the production route and shipped before this runs)")
            o = torch.load(p, map_location="cpu", weights_only=False)             # our own file, as the trainer's FrameStore loads it
            row = loaded[c]["row"]
            if o.get("clip_id") != c or str(o.get("codec")) != "png":
                raise Refusal(f"{p}: clip_id {o.get('clip_id')!r} / codec {o.get('codec')!r}")
            if int(o["jpeg_len"].shape[0]) != int(row["t_ep"]) or E.n_cache_frames(o) != int(row["n_frames"]):
                raise Refusal(f"{p}: {int(o['jpeg_len'].shape[0])} raw / {E.n_cache_frames(o)} cache frames, the production row says t_ep {row['t_ep']} / n_frames {row['n_frames']}: "
                              f"these are not the frames the cache was encoded from")
            frames[c] = (o, file_sha256(p))
        enc = build(str(dev))
        lv = enc.built_levers()
        if not lv["lora_B_zero_at_build"] or enc.trainable_count() != EXPECTED_TRAINABLE:
            raise Refusal(f"the encoder is not the arm's at B = 0 (B zero {lv['lora_B_zero_at_build']}, trainable {enc.trainable_count()})")
        if str(lv["backbone_dtype"]) != "torch.bfloat16":
            raise Refusal(f"the backbone is {lv['backbone_dtype']}, the builder's weights are pure bf16")
        pool_field = E.builder_pool_field()
        restore = apply_mutant(enc, a.mutant, dev) if a.mutant else (lambda: None)
        res.update(cuda=torch.cuda.get_device_name(0) if dev.type == "cuda" else None, dtype=lv["backbone_dtype"], autocast=False, weights_sha256=lv["weights_sha256"],
                   builder_model_sha256=lv["builder_model_sha256"], builder_encode_episode_sha256=E.builder_encode_episode_sha256(), builder_pin=E.BUILDER_ENCODE_EPISODE_SHA256,
                   builder_unchanged=E.builder_encode_episode_sha256() == E.BUILDER_ENCODE_EPISODE_SHA256, batch=BUILDER_BATCH,
                   attn_implementation=getattr(getattr(enc.backbone, "config", None), "_attn_implementation", None),
                   allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32), cudnn_benchmark=bool(torch.backends.cudnn.benchmark))
        per, all_frames = {}, True
        try:
            for s, c in clips:
                o, fsha = frames[c]
                n = E.n_cache_frames(o)
                steps = spread(n, a.frames_per_clip)
                all_frames &= len(steps) == n
                tt = time.time()
                mine = encode_pooled(enc, o, steps, pool_field)
                ref = torch.load(loaded[c]["stored_path"], map_location="cpu", weights_only=True)["pool" + GRID]
                if tuple(ref.shape) != (n, 40, 1024) or ref.dtype != torch.float8_e4m3fn:
                    raise Refusal(f"{c[:8]}: the stored pool{GRID} is {ref.dtype} {tuple(ref.shape)}, expected float8_e4m3fn {(n, 40, 1024)}")
                st = compare_pooled(mine, ref[steps])
                per[c] = {"split": s, "n_frames": n, "n_compared": len(steps), "frames_sha256": fsha, "stored_md5": loaded[c]["stored_md5"], "seconds": time.time() - tt, **st,
                          "manifest_row_matched": {k: loaded[c]["row"][k] for k in ("sha12", "file", "md5", "content_md5", "source", "enc_dtype", "device", "pipeline", "n_frames", "t_ep")
                                                   if k in loaded[c]["row"]}}
                print(f"  {s} {c[:8]}: {len(steps)}/{n} frames rel-L2 {st['rel_l2']:.3e} max abs {st['max_abs']:.4g} bitwise {st['frac_bitwise_equal']:.6f} ({time.time() - tt:.1f} s)", flush=True)
        finally:
            restore()
        res["clips"] = per
        res.update(decide(per, tau=a.tau, scope={"device_type": dev.type, "all_frames": all_frames, "autocast": False}))
        res["scope"] = {"all_frames": all_frames, "frames_per_clip": a.frames_per_clip, "device_type": dev.type}
        if a.mutant:
            res["expected_verdict"] = "FAIL"
            res["mutant_caught"] = bool(res["verdict"] == "FAIL" and res["qualifying"] and not res["numbers_pass_tau"])
        return res, (0 if res["mutant_caught"] else 1) if a.mutant else (0 if res["verdict"] == "PASS" else 1)
    except Refusal as e:
        res.update(verdict="REFUSED", refused=str(e))
        return res, 2
    finally:
        res["seconds"] = time.time() - t0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache-root", required=True, help="the PRODUCTION cache root (cache_full): <root>/{train,eval}/manifest.jsonl and <root>/{train,eval}/clips/<clip>.pt")
    ap.add_argument("--train-clips", nargs="*", default=[], help="full train clip ids")
    ap.add_argument("--eval-clips", nargs="*", default=[], help="full eval clip ids")
    ap.add_argument("--train-frames", default=None, help="directory of <clip>.v2ep.pt for the train clips (production route)")
    ap.add_argument("--eval-frames", default=None, help="directory of <clip>.v2ep.pt for the eval clips (production route)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, help="the json this writes (also on a refusal)")
    ap.add_argument("--tau", type=float, default=TAU_DEVICE, help="rel-L2 bar (Amendment 2: 1e-3)")
    ap.add_argument("--mutant", nargs="?", const=DEFAULT_MUTANT, default=None, help=f"change one preprocessing constant; must FAIL. one of {sorted(MUTANTS)}")
    ap.add_argument("--manifest-only", action="store_true", help="assert the production rows (and md5s where the files exist) and stop")
    ap.add_argument("--frames-per-clip", type=int, default=0, help="0 = every cache frame (the only QUALIFYING reading); k = k frames spread over the clip (a labelled subset)")
    ap.add_argument("--allow-non-cuda", action="store_true", help="run on another device as a labelled NON-QUALIFYING reading (the verdict is FAIL by construction)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.train_clips and not a.train_frames and not a.manifest_only:
        print("REFUSED: --train-clips needs --train-frames")
        return 2
    if a.eval_clips and not a.eval_frames and not a.manifest_only:
        print("REFUSED: --eval-clips needs --eval-frames")
        return 2
    res, rc = run(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    os.replace(tmp, out)
    if res.get("verdict") == "REFUSED":
        print("REFUSED:", res["refused"], flush=True)
    elif a.mutant:
        print(f"MUTANT {a.mutant}: verdict {res['verdict']} worst rel-L2 {res['worst_rel_l2']:.3e} -> {'CAUGHT' if res['mutant_caught'] else 'NOT CAUGHT (the instrument is blind to this change)'}", flush=True)
    elif a.manifest_only:
        print(f"MANIFEST-OK {res['n_clips']} clips, {res['stored_files_checked']} stored files md5-checked", flush=True)
    else:
        print(f"IDENTITY {res['verdict']}: worst rel-L2 {res['worst_rel_l2']:.3e} ({res['worst_clip'][:8]}) tau {a.tau:g} qualifying {res['qualifying']}"
              + ("" if res["qualifying"] else " [" + "; ".join(res["qualifying_reasons"]) + "]"), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
