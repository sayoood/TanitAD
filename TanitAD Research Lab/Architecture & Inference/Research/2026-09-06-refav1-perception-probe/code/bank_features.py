"""P1 — bank the four probe arms' features + the perception targets.

⭐ WHAT THIS ANSWERS
    Does refav1's FROZEN trunk encode where the lead vehicle is?
    The planner's cost is defined in the token-field latent. If the lead is not
    decodable THERE, no cost in that space can express "keep distance" — the
    objective is UNREPRESENTABLE, not mistuned.

⛔ THE TRUNK IS FROZEN AND THE FREEZE IS PROVEN, not asserted: every trunk
   tensor is hashed before and after, and max|delta| must be 0.0 exactly.

ARMS (identical pooling + identical downstream pipeline; only the source differs)
    pixels        raw decoded frame            -- the FLOOR. A learned
                                                  representation that does not
                                                  beat raw pixels added NOTHING.
    dinov3        frozen encoder patch tokens  -- the EXTERNAL REFERENCE
    refav1_field  refav1 `_last_state` output  -- THE LATENT THE COST LIVES IN
    (constant + time-shuffle controls are built in the probe script, since they
     need the split.)

ALIGNMENT (verified, not assumed): the DINOv3 cache is the 0.2 s grid = every
2nd episode frame, so feature row j <-> episode frame 2*j (201 frames -> 101
rows). The probe script runs an OFFSET SWEEP and the best offset must be 0.

⛔ TRAP HONOURED: the v2ep buffer is named `jpeg_buf` but its `codec` field says
   `png`. We READ THE CODEC FIELD and never trust the buffer's name.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

# ---- geometry / pooling (declared before any result) ----------------------
GRID_H, GRID_W = 16, 40          # DINOv3 token grid over 120 deg HFOV
POOL_H, POOL_W = 8, 20           # -> 20 azimuth bins = 6 deg/bin
PIX_H, PIX_W = 32, 80            # pixel arm raster (same 2.5 aspect)
WIN = 4                          # temporal window = cfg.op_window

# ---- lead criterion: the PROGRAMME'S EXISTING convention -----------------
# from p2_lead_census.json: "cx>0 and |cy|<=1.75 and cx<=30.0"
LEAD_LAT_M = 1.75
LEAD_MAX_GAP_M = 30.0
VEHICLE_CLS = {"automobile", "heavy_truck", "bus", "other_vehicle",
               "trailer", "train_or_tram_car"}

DINO_DIR = "/home/nvidia/data/dinov3-b1-fp8-w120-256x640cyl"
EP_DIRS = ["/home/nvidia/data/physicalai-b1-w120-256x640cyl",
           "/home/nvidia/data/physicalai-train-e438721ae894-w120-256x640cyl"]
CKPT = "/home/nvidia/refav1_lon/ckpt/ckpt.pt"
OUT = "/home/nvidia/percprobe/bank"


# --------------------------------------------------------------------------
def trunk_fingerprint(model):
    """name -> (sha256 of the raw bytes, float64 sum). Used for the freeze proof."""
    fp = {}
    for name, p in model.named_parameters():
        a = p.detach().cpu().contiguous()
        fp[name] = (hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                    float(a.double().sum().item()))
    for name, b in model.named_buffers():
        a = b.detach().cpu().contiguous()
        fp["buf:" + name] = (hashlib.sha256(a.numpy().tobytes()).hexdigest(),
                             float(a.double().sum().item()))
    return fp


def freeze_delta(before, after, model):
    """max |delta| over every trunk tensor, plus a hash-mismatch count."""
    assert set(before) == set(after), "tensor SET changed -- not a freeze"
    nmis = sum(1 for k in before if before[k][0] != after[k][0])
    cur = {}
    for name, p in model.named_parameters():
        cur[name] = p.detach().cpu()
    for name, b in model.named_buffers():
        cur["buf:" + name] = b.detach().cpu()
    return nmis, cur


def pool_grid(x, gh, gw, ph, pw):
    """[N, C] tokens on a gh x gw grid -> [ph*pw, C] block means."""
    n, c = x.shape
    assert n == gh * gw, (n, gh, gw)
    x = x.reshape(gh, gw, c)
    x = x.reshape(ph, gh // ph, pw, gw // pw, c).mean(axis=(1, 3))
    return x.reshape(ph * pw, c)


def decode_frames(ep, idxs):
    """Decode the requested EPISODE frame indices. READS THE CODEC FIELD."""
    from PIL import Image
    codec = ep.get("codec", None)
    if codec is None:
        raise RuntimeError("v2ep carries no `codec` field -- refusing to guess")
    if codec != "png":
        raise RuntimeError("unexpected codec %r (buffer name is a lie by "
                           "design; we read the field)" % codec)
    buf = ep["jpeg_buf"].numpy().tobytes()
    lens = ep["jpeg_len"].tolist()
    offs = np.concatenate([[0], np.cumsum(lens)])
    out = np.zeros((len(idxs), PIX_H, PIX_W, 3), dtype=np.uint8)
    for k, i in enumerate(idxs):
        raw = buf[offs[i]:offs[i] + lens[i]]
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize(
            (PIX_W, PIX_H), Image.BILINEAR)
        out[k] = np.asarray(im)
    return out


def lead_of(agents, vehicles_only, max_gap=LEAD_MAX_GAP_M):
    best = None
    for a in agents:
        if vehicles_only and a["cls"] not in VEHICLE_CLS:
            continue
        if a["cx"] > 0 and abs(a["cy"]) <= LEAD_LAT_M and a["cx"] <= max_gap:
            if best is None or a["cx"] < best["cx"]:
                best = a
    return best


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", default="/home/nvidia/percprobe/raw/train2400_agents.jsonl.xz")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)

    # ---------------- targets from the join --------------------------------
    import lzma
    have = set(os.path.basename(p)[:-3] for p in glob.glob(DINO_DIR + "/*.pt"))
    tgt = {}   # clip -> {frame_idx: rec}
    with lzma.open(args.join, "rt") as f:
        for line in f:
            r = json.loads(line)
            c = r["clip_id"]
            if c not in have:
                continue
            ags = r["agents"]
            ld = lead_of(ags, vehicles_only=True)
            ldany = lead_of(ags, vehicles_only=False)
            # UNCAPPED nearest in-corridor vehicle: lets the probe apply either
            # of the programme's two conventions (30 m census / 80 m gate)
            # without a rebuild.
            ldu = lead_of(ags, vehicles_only=True, max_gap=float("inf"))
            tgt.setdefault(c, {})[r["frame_idx"]] = {
                "gap": float(ld["cx"]) if ld else float("nan"),
                "lat": float(ld["cy"]) if ld else float("nan"),
                "trk": (ldu["track_id"] if ldu else ""),
                "gap_unc": float(ldu["cx"]) if ldu else float("nan"),
                "lat_unc": float(ldu["cy"]) if ldu else float("nan"),
                "gap_any": float(ldany["cx"]) if ldany else float("nan"),
                "present": 1.0 if ld else 0.0,
                "n_vis": float(sum(1 for a in ags if a.get("occ", 1) == 0)),
                "n_all": float(len(ags)),
            }
    clips = sorted(tgt)
    if args.limit:
        clips = clips[:args.limit]
    print("clips to bank: %d" % len(clips), flush=True)

    # ---------------- model ------------------------------------------------
    sys.path.insert(0, "/home/nvidia/refav1_lon/code/stack")
    from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg = RefAV1Config(**cfg) if isinstance(cfg, dict) else cfg
    model = RefAV1(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print("load_state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)),
          flush=True)
    if missing:
        print("  first missing:", list(missing)[:5], flush=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)

    fp_before = trunk_fingerprint(model)
    print("freeze fingerprint: %d tensors" % len(fp_before), flush=True)

    # ---------------- bank -------------------------------------------------
    ep_index = {}
    for d in EP_DIRS:
        for p in glob.glob(d + "/*.v2ep.pt"):
            ep_index.setdefault(os.path.basename(p)[:-8], p)

    meta = []
    t00 = time.time()
    for ci, c in enumerate(clips):
        try:
            feat = torch.load(DINO_DIR + "/" + c + ".pt", map_location="cpu")
            nrow = feat.shape[0]
            epp = ep_index.get(c)
            if epp is None:
                print("  SKIP %s: no v2ep" % c, flush=True)
                continue
            ep = torch.load(epp, map_location="cpu", weights_only=False)
            nfr = int(ep["jpeg_len"].shape[0])

            rows = [j for j in range(nrow) if (2 * j) < nfr and (2 * j) in tgt[c]]
            if not rows:
                continue

            f32 = feat.to(torch.float32)

            # -- refav1 field: window of WIN rows ending at j, take _last_state
            fields = np.zeros((len(rows), POOL_H * POOL_W, cfg.d_state), np.float16)
            dinos = np.zeros((len(rows), POOL_H * POOL_W, cfg.d_enc), np.float16)
            with torch.no_grad():
                B = 8                     # Thor saturates at batch 8
                for s in range(0, len(rows), B):
                    chunk = rows[s:s + B]
                    win = torch.stack([
                        f32[[max(0, j - k) for k in range(WIN - 1, -1, -1)]]
                        for j in chunk])                      # [b, WIN, N, d_enc]
                    fld = model.encode(win.to(dev))           # [b, WIN, N, d_state]
                    last = model._last_state(fld)             # [b, N, d_state]
                    last = last.float().cpu().numpy()
                    for k in range(len(chunk)):
                        fields[s + k] = pool_grid(last[k], GRID_H, GRID_W,
                                                  POOL_H, POOL_W).astype(np.float16)
                    raw = f32[chunk].numpy()
                    for k in range(len(chunk)):
                        dinos[s + k] = pool_grid(raw[k], GRID_H, GRID_W,
                                                 POOL_H, POOL_W).astype(np.float16)

            pix = decode_frames(ep, [2 * j for j in rows])

            T = [tgt[c][2 * j] for j in rows]
            np.savez_compressed(
                os.path.join(args.out, c + ".npz"),
                rows=np.array(rows, np.int32),
                frame_idx=np.array([2 * j for j in rows], np.int32),
                field=fields, dino=dinos, pix=pix,
                gap=np.array([t["gap"] for t in T], np.float32),
                lat=np.array([t["lat"] for t in T], np.float32),
                gap_unc=np.array([t["gap_unc"] for t in T], np.float32),
                lat_unc=np.array([t["lat_unc"] for t in T], np.float32),
                gap_any=np.array([t["gap_any"] for t in T], np.float32),
                present=np.array([t["present"] for t in T], np.float32),
                n_vis=np.array([t["n_vis"] for t in T], np.float32),
                n_all=np.array([t["n_all"] for t in T], np.float32),
                trk=np.array([t["trk"] for t in T]),
            )
            meta.append({"clip": c, "n_rows": len(rows),
                         "n_lead": int(sum(t["present"] for t in T))})
            if (ci + 1) % 20 == 0:
                el = time.time() - t00
                print("  %d/%d clips  %.0fs  (%.1f s/clip)  cuda_max=%.2fGB"
                      % (ci + 1, len(clips), el, el / (ci + 1),
                         torch.cuda.max_memory_allocated() / 1e9), flush=True)
        except Exception as e:
            print("  ERROR %s: %r" % (c, e), flush=True)

    # ---------------- FREEZE PROOF ----------------------------------------
    fp_after = trunk_fingerprint(model)
    nmis, cur = freeze_delta(fp_before, fp_after, model)
    maxdelta = 0.0
    worst = None
    for k in fp_before:
        d = abs(fp_before[k][1] - fp_after[k][1])
        if d > maxdelta:
            maxdelta, worst = d, k
    print("\n=== FREEZE PROOF ===")
    print("trunk tensors checked : %d" % len(fp_before))
    print("sha256 mismatches     : %d" % nmis)
    print("max |delta(sum)|      : %.17g  (worst tensor: %s)" % (maxdelta, worst))
    print("FREEZE %s" % ("HELD (max|delta| = 0 exactly)" if (nmis == 0 and maxdelta == 0.0)
                         else "VIOLATED"))

    with open(os.path.join(args.out, "_bank_meta.json"), "w") as f:
        json.dump({"clips": meta,
                   "n_clips": len(meta),
                   "n_rows": sum(m["n_rows"] for m in meta),
                   "n_lead_rows": sum(m["n_lead"] for m in meta),
                   "pool": [POOL_H, POOL_W], "grid": [GRID_H, GRID_W],
                   "pix": [PIX_H, PIX_W], "win": WIN,
                   "lead_criterion": {"lat_m": LEAD_LAT_M,
                                      "max_gap_m": LEAD_MAX_GAP_M,
                                      "vehicles_only": True},
                   "ckpt": CKPT, "ckpt_step": int(ck["step"]),
                   "freeze": {"n_tensors": len(fp_before),
                              "sha_mismatches": nmis,
                              "max_abs_delta": maxdelta,
                              "held": bool(nmis == 0 and maxdelta == 0.0)}}, f, indent=1)
    print("\nbanked %d clips, %d rows, %d lead rows -> %s"
          % (len(meta), sum(m["n_rows"] for m in meta),
             sum(m["n_lead"] for m in meta), args.out))


if __name__ == "__main__":
    main()
