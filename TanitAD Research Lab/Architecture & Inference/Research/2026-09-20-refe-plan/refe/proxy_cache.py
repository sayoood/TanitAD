#!/usr/bin/env python3
"""The SHARED dev-box proxy cache: ONE snapshot's frozen encoder side, run ONCE per frame, stored as raw bits.
Consumers: the decoder-side proxy (M3 / M4a / M4b / M6 A/Bs) and the scorer-only fine-tune (measure 5).

WHAT IS FROZEN IN THE PROXY (and therefore cached): everything up to the two decoders' contexts -- the DINOv3 trunk,
its LoRA, the task registers, the PETR pos3d MLP, the register compression and `scene_proj`. What the proxy TRAINS
reads only these contexts plus per-row ego / goal / targets / labels:
  scene_ctx  [64, 256]    = scene_proj(register compression)     -> ego_enc/queries/dec/traj_head (trajectory side)
  visual_ctx [7680, 256]  = scene_proj(visual) (the scorer's sctx) -> score_q_mlp/score_dec/score_head (scorer side)
Both are captured by forward PRE-HOOKS on dec[0] / score_dec[0] during the model's OWN forward on the trainer's OWN
inputs (`train.TargetBank.__getitem__`: images, ego, goal, per-sample rig) -- no re-implemented encoder.

LAYOUT  <out>/
  shard_NNNN.scene.bin / shard_NNNN.visual.bin   raw little-endian bits, [n, 64, 256] / [n, 7680, 256]; bf16 as uint16
  frames.jsonl        one line per frame, APPEND order: key, shard, index in shard, vindex (its row in the shard's
                      visual file; -1 = not stored under --visual-for sets; absent = index), sha256 of its scene bits
  rows.jsonl          one line per (frame, rank): key, rank, ego[7], goal[4], traj[20][3] (targets; labels live apart)
  manifest.json       written LAST on every pass: snapshot + sha256, frozen-in-proxy fingerprint, config, numerics,
                      shard table with sha256, counts, tool sha256. A cache without it is not a cache.
NUMERICS: `--amp bf16` reproduces the live trainer's forward (bf16 autocast); the stored bits ARE the tensors its
decoders saw. `--amp none` for an fp32 eval cache. RESUMABLE: frames already in frames.jsonl are skipped; a pass stops
cleanly at --deadline (GPU windows) and a later pass appends new shards.
CONTROLS (every pass, gates): C1 the trajectory decoder run on the CAPTURED scene_ctx reproduces the full forward's
proposals BIT-FOR-BIT (same device, same autocast); C2 every shard read back from disk equals the in-memory bits.

    python refe/proxy_cache.py build --manifest <proxy_manifest.json> --images <root> --calib <calib_table.json>
                                     --snapshot <snap_epochNNN.pt> --out <dir> [--amp bf16] [--limit N] [--deadline HH:MM]
    python refe/proxy_cache.py verify --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BANK = Path("D:/Projects/TanitAD/data/refe_navtrain10")
SCENE_TOKENS, VISUAL_TOKENS, WIDTH = 64, 7680, 256


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def past(deadline: str | None) -> bool:
    if not deadline:
        return False
    hh, mm = (int(x) for x in deadline.split(":"))
    lt = time.localtime()
    return lt.tm_hour * 60 + lt.tm_min >= hh * 60 + mm


def to_bits(t) -> bytes:
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().tobytes()
    return t.float().numpy().tobytes()


def from_bits(b: bytes, shape, dtype: str):
    import torch
    if dtype == "bf16":
        return torch.from_numpy(np.frombuffer(b, dtype=np.int16).copy()).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(np.frombuffer(b, dtype=np.float32).copy()).reshape(shape)


def frozen_in_proxy_fingerprint(model) -> str:
    """sha256 over every tensor the proxy does NOT train (all but ego_enc/queries/dec/traj_head/score_*)."""
    import torch
    train_prefixes = ("ego_enc.", "queries", "dec.", "traj_head.", "score_q_mlp.", "score_dec.", "score_head.")
    h = hashlib.sha256()
    for n, v in sorted(model.state_dict().items()):
        if n.startswith(train_prefixes):
            continue
        h.update(n.encode())
        h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes() if v.dtype != torch.bool
                 else v.cpu().numpy().tobytes())
    return h.hexdigest()


def rows_for(manifest: dict) -> dict:
    """the manifest's (frame, rank) rows, verbatim from the local live-bank files"""
    want = {(f["log_name"], f["token"], int(f["step"])): set(f["ranks"]) for f in manifest["frames"]}
    out = {}
    for rk, p in (("0", BANK / "r0" / "targets_rank0.jsonl"), ("1", BANK / "aug" / "targets_aug.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            k = (r["log_name"], r["token"], int(r["step"]))
            if k in want and rk in want[k]:
                out[(k, rk)] = r
    return out


def build(a) -> int:
    import torch
    import ckpt_io
    import train as T
    from model import REFe, REFeConfig
    man = json.load(open(a.manifest, encoding="utf-8"))
    frames = man["frames"][:a.limit] if a.limit else man["frames"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    fl = out / "frames.jsonl"
    if fl.exists():
        done = {tuple(json.loads(l)["key"]) for l in open(fl, encoding="utf-8")}
    todo = [f for f in frames if (f["log_name"], f["token"], int(f["step"])) not in done]
    print(f"  frames: {len(frames):,} in the manifest, {len(done):,} cached, {len(todo):,} to do", flush=True)
    rows = rows_for({"frames": todo})
    tmp = Path(tempfile.mkdtemp(prefix="proxycache_rows_"))
    for rk in ("0", "1"):
        with open(tmp / f"targets_rank{rk}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for (k, r_), r in rows.items():
                if r_ == rk:
                    f.write(json.dumps(r) + "\n")
    dev = "cuda" if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    cfg = REFeConfig.for_backbone(a.backbone)
    model = REFe(cfg).to(dev).eval()
    meta: dict = {}
    fmt = ckpt_io.load_for_inference(model, a.snapshot, map_location=dev, backbone=a.backbone, meta_out=meta)
    fp = frozen_in_proxy_fingerprint(model)
    ds = T.TargetBank(str(tmp), None if a.synthetic else a.images, cfg, synthetic=a.synthetic,
                      calib=None if a.synthetic else a.calib)
    by_key = {}
    for i, r in enumerate(ds.rows):
        by_key.setdefault((r["log_name"], r["token"], int(r["step"])), []).append(i)
    cap = {}
    model.dec[0].register_forward_pre_hook(lambda _m, args: cap.__setitem__("scene", args[1].detach()))
    model.score_dec[0].register_forward_pre_hook(lambda _m, args: cap.__setitem__("visual", args[1].detach()))
    amp = a.amp == "bf16" and dev == "cuda"
    dtype = "bf16" if amp else "fp32"
    shard = len(list(out.glob("shard_*.scene.bin")))
    buf_s, buf_v, lines, rlines = [], [], [], []
    c1_worst, n_new, stopped = 0.0, 0, None

    def decode(scene_ctx, ego, goal):
        g = goal.unsqueeze(-1) * model.goal_freqs
        gf = torch.cat([goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        q = model.queries.expand(scene_ctx.shape[0], -1, -1) + model.ego_enc(torch.cat([ego, gf], -1)).unsqueeze(1)
        for blk in model.dec:
            q = blk(q, scene_ctx)
        return model.traj_head(q).view(scene_ctx.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)

    def flush():
        nonlocal shard, buf_s, buf_v, lines
        if not buf_s:
            return
        sb, vb = b"".join(buf_s), b"".join(buf_v)
        ps, pv = out / f"shard_{shard:04d}.scene.bin", out / f"shard_{shard:04d}.visual.bin"
        for p, b in ((ps, sb), (pv, vb)):
            with open(p, "wb") as f:
                f.write(b)
                f.flush()
                os.fsync(f.fileno())
            if sha256_file(p) != sha256_bytes(b):                                  # C2: read back == written
                raise SystemExit(f"C2 FAILED: {p} does not read back as written")
        with open(fl, "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(l) + "\n" for l in lines))
        with open(out / "rows.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write("".join(json.dumps(l) + "\n" for l in rlines))
        shard += 1
        buf_s, buf_v, lines = [], [], []
        rlines.clear()
    t0 = time.time()
    for f in todo:
        if past(a.deadline):
            stopped = f"deadline {a.deadline}"
            break
        k = (f["log_name"], f["token"], int(f["step"]))
        idx = by_key.get(k)
        if not idx:
            continue                                              # dropped by TargetBank (missing frame / calib)
        img, ego, goal, tgt, _ct, _cg, _cm, _ci, cal = ds[idx[0]]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            traj, _score = model(img[None].to(dev), ego[None].to(dev), goal[None].to(dev),
                                 calib=cal[None] if cal.numel() else None)
            dt = decode(cap["scene"], ego[None].to(dev), goal[None].to(dev))
        c1_worst = max(c1_worst, float((dt.float() - traj.float()).abs().max()))
        sbits = to_bits(cap["scene"][0])
        buf_s.append(sbits)
        # --visual-for sets: the scorer's context is stored only for a frame that carries an on-policy set at ANY rank
        # (the proxy reads visual_ctx only for a labelled sample; 3.9 MB/frame in bf16)
        if a.visual_for == "all" or any(v.get("onpolicy") for v in f.get("ranks", {}).values()):
            buf_v.append(to_bits(cap["visual"][0]))
            vindex = len(buf_v) - 1
        else:
            vindex = -1
        lines.append({"key": list(k), "shard": shard, "index": len(buf_s) - 1, "vindex": vindex,
                      "scene_sha256": sha256_bytes(sbits)})
        for i in idx:
            r = ds.rows[i]
            rlines.append({"key": list(k), "rank": int(r.get("rank", 0)), "ego": r["ego"], "goal": r["goal"][:4],
                           "traj": r["traj"]})
        n_new += 1
        if len(buf_s) >= a.shard_frames:
            flush()
    flush()
    PREV_PASSES = (json.load(open(out / "manifest.json", encoding="utf-8")).get("passes", [])
                   if (out / "manifest.json").exists() else [])
    shards = sorted(out.glob("shard_*.scene.bin"))
    mf = {"version": 1, "snapshot": str(a.snapshot), "snapshot_sha256": sha256_file(a.snapshot), "format": fmt,
          "frozen_in_proxy_sha256": fp, "backbone": a.backbone, "dtype": dtype, "amp": a.amp, "device": dev,
          "input_path": "train.TargetBank.__getitem__" + (" (SYNTHETIC images)" if a.synthetic else ""),
          "images": a.images, "calib": a.calib, "manifest": str(a.manifest), "manifest_status": man.get("status"),
          "shapes": {"scene": [SCENE_TOKENS, WIDTH], "visual": [VISUAL_TOKENS, WIDTH]},
          "shards": [{"scene": p.name, "scene_sha256": sha256_file(p), "visual": p.name.replace(".scene.", ".visual."),
                      "visual_sha256": sha256_file(p.with_name(p.name.replace(".scene.", ".visual.")))} for p in shards],
          "frames": sum(1 for _ in open(fl, encoding="utf-8")) if fl.exists() else 0,
          "visual_for": a.visual_for,
          "visual_frames": sum(1 for l in open(fl, encoding="utf-8") if json.loads(l).get("vindex", 0) >= 0)
          if fl.exists() else 0,
          "rows": sum(1 for _ in open(out / "rows.jsonl", encoding="utf-8")) if (out / "rows.jsonl").exists() else 0,
          "last_pass": {"new_frames": n_new, "seconds": round(time.time() - t0, 1), "stopped": stopped,
                        "C1_decode_vs_forward_max_abs": c1_worst},
          "passes": PREV_PASSES + [{"new_frames": n_new, "seconds": round(time.time() - t0, 1), "stopped": stopped,
                        "C1_decode_vs_forward_max_abs": c1_worst}],
          "tool_sha256": sha256_file(__file__), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmpm = out / "manifest.json.tmp"
    with open(tmpm, "w", encoding="utf-8", newline="\n") as f:
        json.dump(mf, f, indent=1)
    os.replace(tmpm, out / "manifest.json")
    ok = c1_worst == 0.0
    print(f"  cached {n_new} new frames ({mf['frames']} total) in {mf['last_pass']['seconds']} s; C1 max|decode - "
          f"forward| {c1_worst:.3g} ({'bit-exact' if ok else 'NOT EXACT'}); stopped: {stopped}")
    print("ZZCACHE_OK" if ok else "ZZCACHE_FAIL C1")
    return 0 if ok else 1


def verify(a) -> int:
    out = Path(a.out)
    mf = json.load(open(out / "manifest.json", encoding="utf-8"))
    bad = [s["scene"] for s in mf["shards"] if sha256_file(out / s["scene"]) != s["scene_sha256"]]
    bad += [s["visual"] for s in mf["shards"] if sha256_file(out / s["visual"]) != s["visual_sha256"]]
    per = 2 if mf["dtype"] == "bf16" else 4
    frames = [json.loads(l) for l in open(out / "frames.jsonl", encoding="utf-8")]
    sizes, vsizes = {}, {}
    for fr in frames:
        sizes[fr["shard"]] = max(sizes.get(fr["shard"], 0), fr["index"] + 1)
        vsizes[fr["shard"]] = max(vsizes.get(fr["shard"], 0), fr.get("vindex", fr["index"]) + 1)   # no vindex = all
    geom = all(os.path.getsize(out / f"shard_{s:04d}.scene.bin") == n * SCENE_TOKENS * WIDTH * per and
               os.path.getsize(out / f"shard_{s:04d}.visual.bin") == vsizes[s] * VISUAL_TOKENS * WIDTH * per
               for s, n in sizes.items())
    spot = 0
    for fr in frames[:: max(1, len(frames) // 20)]:
        with open(out / f"shard_{fr['shard']:04d}.scene.bin", "rb") as f:
            n = SCENE_TOKENS * WIDTH * per
            f.seek(fr["index"] * n)
            spot += sha256_bytes(f.read(n)) == fr["scene_sha256"]
    ok = not bad and geom and spot == len(frames[:: max(1, len(frames) // 20)])
    print(json.dumps({"frames": len(frames), "shards": len(mf["shards"]), "sha_mismatch": bad, "geometry_ok": geom,
                      "frame_spot_checks_ok": spot, "dtype": mf["dtype"], "ok": ok}, indent=1))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "verify"))
    ap.add_argument("--manifest")
    ap.add_argument("--images")
    ap.add_argument("--calib")
    ap.add_argument("--snapshot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--amp", choices=("bf16", "none"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-frames", type=int, default=256)
    ap.add_argument("--deadline", default=None, help="local HH:MM; stop cleanly (resumable) at a GPU window's end")
    ap.add_argument("--synthetic", action="store_true", help="TEST ONLY: noise images (no JPEGs needed)")
    ap.add_argument("--visual-for", choices=("all", "sets"), default="all",
                    help="store the scorer's visual_ctx for every frame, or only for frames carrying an on-policy set "
                         "at any rank (frames.jsonl 'vindex' = its row in the shard's visual file, -1 = not stored)")
    a = ap.parse_args()
    return build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    sys.exit(main())
