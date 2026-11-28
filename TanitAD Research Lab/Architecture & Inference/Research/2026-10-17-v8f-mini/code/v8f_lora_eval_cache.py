"""PREREG_V8F_LORA, the EVAL ROUTE: a LoRA checkpoint -> merge -> the CACHE BUILDER re-encodes the eval clips at the merged weights -> the UNCHANGED ``v8f_mini_eval.py``.

    python v8f_lora_eval_cache.py --ckpt <ckpt_last.pt of a --encoder lora run> --eval-eps <dir of eval <clip>.v2ep.pt> --out <new cache root>
           [--base-cache <existing cache root>]  [--clips A B ...]  [--device cuda] [--dtype bf16]
    python v8f_mini_eval.py run --ckpt <the same ckpt> --eval-cache <new cache root> ...        # unchanged; reads ckpt["model"], never ckpt["lora"]

What it does, and nothing else:
  1. ``DinoV3LoRAEncoder.from_pretrained`` with the checkpoint's own r / alpha / targets, ``load_lora_state_dict(ckpt["lora"])``; the encoder weights' sha256 must be the cache builder's.
  2. ``merge()``: ``W' = W + (alpha / r) B A`` per adapted projection (fp32 sum, rounded to the weight's dtype), wrappers removed -- the deployed encoder, exactly the frozen
     DINOv3-L's modules and parameter count. ``--zero-lora`` skips the checkpoint (B = 0): the route's identity control.
  3. For every eval clip, the BUILDER's own ``pull_encode_pooled.encode_clip`` (-> ``dinov3_fp8_encode_ship.encode_episode`` -> ``pool_field`` -> fp8) is called with the MERGED
     backbone as its ``model``; its payload (pool, mean, kinematics) and ``make_meta`` are written in the builder's own file format (``<out>/eval/clips/<clip>.pt``,
     ``manifest.jsonl``, ``meta.json``), so ``v8f_mini_eval.py --eval-cache <out>`` reads it as it reads any cache.
  4. ``--base-cache``: every NON-feature array (v, a, steer, yaw_rate, pose_grid, kin3, p10) of every written clip must equal the base cache's, bit for bit, or the run
     STOPS (the new cache must differ from the old in the pooled tokens ONLY).
Out of scope here: choosing which checkpoint, and any eval number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[5]
sys.path[:0] = [str(REPO / "stack"), str(REPO / "taniteval")]
from tanitad.models import v8f_lora_encoder as E  # noqa: E402

#: tests replace this to run on a tiny backbone; a launch never does
BACKBONE_FACTORY = None
FEATURE_KEYS = ("pool4x10", "pool8x20", "mean")


def build_encoder(meta: dict | None, device: str, dtype: torch.dtype):
    if BACKBONE_FACTORY is not None:
        return BACKBONE_FACTORY(meta, device, dtype)
    m = meta or {"r": 32, "alpha": 32.0, "targets": ["q", "v"]}
    return E.DinoV3LoRAEncoder.from_pretrained(r=m["r"], alpha=m["alpha"], targets=tuple(m["targets"]), device=device, dtype=dtype, verify_sha=True)


def state_sha256(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def merged_for(ckpt_path: str | None, device: str, dtype: torch.dtype):
    """-> (merged plain encoder, provenance). ``ckpt_path`` None = B = 0."""
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "lora" not in ck:
            raise SystemExit(f"{ckpt_path} has no 'lora' entry: it is not an --encoder lora checkpoint (an --encoder cache run has nothing to merge)")
        enc = build_encoder(ck["lora"]["meta"], device, dtype)
        enc.load_lora_state_dict(ck["lora"])
        prov = {"ckpt": str(ckpt_path), "ckpt_step": ck.get("step"), "lora_meta": ck["lora"]["meta"]}
    else:
        enc = build_encoder(None, device, dtype)
        prov = {"ckpt": None, "lora": "ZERO (identity route)"}
    n_lora = enc.trainable_count()
    for p in enc.parameters():
        p.requires_grad_(False)
    merged = enc.merged_encoder()
    prov.update(merged_state_sha256=state_sha256(merged.backbone), unmerged_lora_trainable=n_lora, merged_trainable=merged.trainable_count(),
                merged_params=sum(p.numel() for p in merged.backbone.parameters()), encoder_weights_sha256=enc.weights_sha256)
    return merged, prov


def same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality (NaN rows compare equal to themselves, as in the builder's ``content_md5``)."""
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def write_clip(root: Path, c: str, payload: dict, meta: dict, pep, enc_dtype: str, device: str, secs: float) -> dict:
    blob = dict(payload, meta=meta)
    tmp = root / ".tmp" / f"{c}.pt"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, tmp)
    final = root / "clips" / f"{c}.pt"
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp, final)
    return {"clip": c, "sha12": pep.sha12(c), "file": f"clips/{c}.pt", "bytes": final.stat().st_size, "md5": pep.file_hash(final), "content_md5": pep.content_md5(payload),
            "n_frames": meta["n_frames"], "t_ep": meta["t_ep"], "dtype": meta["dtype"], "enc_dtype": enc_dtype, "device": device, "source": "local256",
            "mean_dtype": "float16", "grids": meta["grids"], "shapes": {k: list(v.shape) for k, v in payload.items() if isinstance(v, torch.Tensor)},
            "enc_s": round(secs, 2), "pipeline": pep.PIPELINE}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=None, help="an --encoder lora checkpoint; omit with --zero-lora")
    ap.add_argument("--zero-lora", action="store_true", help="B = 0: the route's identity control (the merged encoder IS the frozen one)")
    ap.add_argument("--eval-eps", required=True)
    ap.add_argument("--out", required=True, help="new cache root; writes <out>/eval/{clips,manifest.jsonl,meta.json}")
    ap.add_argument("--base-cache", default=None)
    ap.add_argument("--clips", nargs="+", default=None, help="default: every clip in <base-cache>/eval/manifest.jsonl (else every v2ep in --eval-eps)")
    ap.add_argument("--grids", default="4x10")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="encoder dtype (the shipped cache is bf16)")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if bool(a.ckpt) == bool(a.zero_lora):
        raise SystemExit("exactly one of --ckpt / --zero-lora")
    if a.threads:
        torch.set_num_threads(a.threads)
    pep, ship = E.load_builder()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    grids = [g for g in a.grids.split(",") if g]
    root = Path(a.out) / "eval"
    (root / "clips").mkdir(parents=True, exist_ok=True)
    if a.clips:
        clips = a.clips
    elif a.base_cache and (Path(a.base_cache) / "eval" / "manifest.jsonl").is_file():
        clips = [json.loads(l)["clip"] for l in open(Path(a.base_cache) / "eval" / "manifest.jsonl", encoding="utf-8") if l.strip()]
        clips = list(dict.fromkeys(clips))
    else:
        clips = sorted(p.name[:-len(".v2ep.pt")] for p in Path(a.eval_eps).glob("*.v2ep.pt"))
    if not clips:
        raise SystemExit(f"no eval clips (looked in {a.eval_eps} and the base cache manifest)")
    merged, prov = merged_for(a.ckpt, a.device, dtype)
    qdtype = torch.float8_e4m3fn if pep.fp8_ok(Path(a.out)) else torch.float16
    qname = str(qdtype).replace("torch.", "")
    if qdtype is torch.float16:
        print("ZZFP8-FALLBACK-FP16ZZ fp8 unavailable: the cache would be fp16, not the builder's fp8", flush=True)
    meta_json = {"pipeline": pep.PIPELINE, "route": "v8f_lora_eval_cache", "model": E.MID, "torch": torch.__version__, "device": a.device, "enc_dtype": a.dtype,
                 "builder_script_sha256": pep.file_hash(pep.__file__, "sha256"), "encoder_module_sha256": pep.file_hash(ship.__file__, "sha256"),
                 "builder_encode_episode_sha256": E.builder_encode_episode_sha256(), "argv": sys.argv[1:], "started": time.strftime("%Y-%m-%d %H:%M:%S"), **prov}
    (root / "meta.json").write_text(json.dumps(meta_json, indent=1), encoding="utf-8")
    mf = open(root / "manifest.jsonl", "a", encoding="utf-8")
    base_diffs = {}
    for i, c in enumerate(clips):
        o = torch.load(Path(a.eval_eps) / f"{c}.v2ep.pt", map_location="cpu", weights_only=False)
        assert o.get("clip_id") == c, "v2ep clip_id != file stem"
        t0 = time.time()
        with torch.no_grad():
            payload = pep.encode_clip(o, ship, merged.backbone, merged.mean, merged.std, merged.n_special, a.device, grids, qdtype)   # THE BUILDER'S OWN FUNCTION, merged weights
        meta = pep.make_meta(c, o, payload, grids, qdtype)
        meta["source"] = "local256"
        ent = write_clip(root, c, payload, meta, pep, a.dtype, a.device, time.time() - t0)
        ent["lora_merged"] = {"ckpt": prov.get("ckpt"), "merged_state_sha256": prov["merged_state_sha256"]}
        mf.write(json.dumps(ent) + "\n")
        mf.flush()
        if a.base_cache:
            bp = Path(a.base_cache) / "eval" / "clips" / f"{c}.pt"
            if not bp.is_file():
                raise SystemExit(f"--base-cache has no {bp} (looked there)")
            base = torch.load(bp, map_location="cpu", weights_only=True)
            bad = [k for k in payload if k not in FEATURE_KEYS and (k not in base or not same_bits(payload[k], base[k]))]
            if bad:
                raise SystemExit(f"STOP: {c}: non-feature arrays differ from the base cache {bad}: the new cache would differ in more than the pooled tokens")
            base_diffs[c] = {k: float((payload[k].float() - base[k].float()).abs().mean()) for k in FEATURE_KEYS if k in payload and k in base}
        print(f"[{i + 1}/{len(clips)}] {c} {time.time() - t0:.1f} s", flush=True)
    mf.close()
    if a.base_cache:
        (root / "base_feature_diff.json").write_text(json.dumps(base_diffs, indent=1), encoding="utf-8")
    print(f"DONE {len(clips)} clips -> {root}  merged_state_sha256 {prov['merged_state_sha256'][:16]}  params {prov['merged_params']:,}  trainable {prov['merged_trainable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
