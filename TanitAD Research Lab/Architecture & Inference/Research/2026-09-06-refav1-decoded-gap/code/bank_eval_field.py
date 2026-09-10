#!/usr/bin/env python3
"""D-REFAV1-DK-DECODED P1 -- bank refav1's own planning latent, the raw-pixel
floor and the frozen-DINOv3 reference over the 141-clip v7.2 EVAL split, with the
B1 EVAL lead block's GT gap as the label.

⭐ WHY THIS FILE EXISTS. `D-REFAV1-DK-COST` proved the distance-keeping cost fires
and re-ranks, but it was fed an ORACLE gap, so the whole result is a CEILING. The
vision-only arm needs a head that decodes the gap from what the planner can SEE.
`M84` measured the gap IS decodable from `_last_state` (+0.3632 [+0.2069,+0.5088]
vs a -0.0513 raw-pixel floor) -- on the TRAIN corpus. This banks the same tensor
on the EVAL corpus, where the cost is actually evaluated.

⛔ THE TENSOR IS THE RIGHT ONE, and it is the same one M84 probed:
`plan()` does `field = self.encode(feats)` then `last = self._last_state(field)`,
and `last` is the root every iCEM rollout starts from. The pooling (16x40 -> 8x20)
is byte-identical to M84's so the two panels are comparable.

⛔ THE TRUNK IS FROZEN AND THE FREEZE IS PROVEN, not asserted: every parameter and
buffer is sha256'd before and after, and `max |delta|` must be 0 EXACTLY.

⛔ THE JOIN KEY IS `(clip_id, RAW frame 2j)` -- the SAME key the cost's plan-time
lookup and the post-roll metric use, so a disagreement between them would be a real
disagreement and not two joins. The alignment is CHECKED, not assumed: the block's
own `speeds` column is compared against the episode's speed at that frame, which is
a LABEL-FREE alignment proof (the same one `refav1_arm.py` runs at plan time).

⚠️ TRAP HONOURED: the v2ep buffer is named `jpeg_buf` but its `codec` field says
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

# ---- geometry / pooling: IDENTICAL to M84's bank so the panels compare -------
GRID_H, GRID_W = 16, 40           # DINOv3 token grid over the 120 deg HFOV
POOL_H, POOL_W = 8, 20            # -> 20 azimuth bins = 6 deg/bin
PIX_H, PIX_W = 32, 80             # pixel-floor raster, same 2.5 aspect
WIN = 4                           # temporal window = cfg.op_window


def trunk_fingerprint(model):
    """name -> (sha256 of raw bytes, float64 sum). The freeze proof's operands."""
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
        raise RuntimeError("unexpected codec %r (the buffer NAME is a lie by "
                           "design; we read the FIELD)" % codec)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="C:/Users/Admin/tanitad-data/"
                                       "refav1-eval141/refav1-fp8-eval")
    ap.add_argument("--episodes", default="C:/Users/Admin/tanitad-data/"
                                          "refav1-eval141/eps")
    ap.add_argument("--block", default="C:/Users/Admin/tanitad-wt/_lead_b1/"
                                       "b1_eval_lead_block.npz")
    ap.add_argument("--ckpt", default="C:/Users/Admin/refav1_eval_slice/"
                                      "ckpt_ep3/ckpt.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)

    # ---- labels: the B1 EVAL lead block, keyed (clip_id, RAW frame) ---------
    blk = np.load(a.block, allow_pickle=True)
    b_cid = blk["clip_id"].astype(str)
    b_fr = blk["frame"].astype(np.int64)
    b_gap = blk["gap0_m"].astype(np.float64)
    b_state = blk["state"].astype(str)
    b_speed = blk["speeds"].astype(np.float64)
    b_rel = blk["rel_speed_mps"].astype(np.float64)
    key = {}
    for i in range(b_cid.size):
        key[(b_cid[i], int(b_fr[i]))] = i
    print("block rows %d over %d clips" % (b_cid.size, len(set(b_cid))),
          flush=True)

    have = sorted(os.path.basename(p)[:-3] for p in glob.glob(a.cache + "/*.pt"))
    clips = [c for c in have if any((c, f) in key for f in range(0, 208, 2))]
    if a.limit:
        clips = clips[:a.limit]
    print("clips to bank: %d" % len(clips), flush=True)

    # ---- model -------------------------------------------------------------
    from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    cfg = RefAV1Config(**cfg) if isinstance(cfg, dict) else cfg
    model = RefAV1(cfg)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print("load_state_dict: missing=%d unexpected=%d step=%s"
          % (len(missing), len(unexpected), ck.get("step")), flush=True)
    if missing:
        print("  first missing:", list(missing)[:5], flush=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    ck_sha = hashlib.sha256()
    with open(a.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            ck_sha.update(chunk)
    ck_sha = ck_sha.hexdigest()

    fp_before = trunk_fingerprint(model)
    print("freeze fingerprint: %d tensors" % len(fp_before), flush=True)

    ep_index = {os.path.basename(p)[:-8]: p
                for p in glob.glob(a.episodes + "/*.v2ep.pt")}

    meta, t00 = [], time.time()
    speed_check_max = 0.0
    speed_ctrl_max, speed_ctrl_mean, n_speed_checked = 0.0, [], 0
    for ci, c in enumerate(clips):
        try:
            feat = torch.load(a.cache + "/" + c + ".pt", map_location="cpu")
            nrow = int(feat.shape[0])
            epp = ep_index.get(c)
            if epp is None:
                print("  SKIP %s: no v2ep" % c, flush=True)
                continue
            ep = torch.load(epp, map_location="cpu", weights_only=False)
            nfr = int(ep["jpeg_len"].shape[0])
            rows = [j for j in range(nrow)
                    if (2 * j) < nfr and (c, 2 * j) in key]
            if not rows:
                continue
            bi = np.array([key[(c, 2 * j)] for j in rows], dtype=np.int64)

            # ⛔ LABEL-FREE ALIGNMENT PROOF, per clip: the episode's own speed at
            # RAW frame 2j against the block's `speeds` at the row we joined. A
            # mis-join would put another clip's traffic into the label and no
            # downstream check could see it.
            # ⛔⛔ THE CHECK MUST BE PROVEN TO HAVE RUN. The first version of this
            # block looked for a `speed`/`speeds` key the v2ep DOES NOT HAVE, so
            # it silently never executed and reported `max delta = 0.000e+00` --
            # a vacuous check reads EXACTLY its own pass value. The speed is
            # `poses[:, 3]` (`refav1_arm.py`: "poses [T_ep, 4] = (x, y, yaw, v)",
            # and the loader refuses if `v0 != poses[2t, 3]`), and the SAME-BREATH
            # MIS-JOIN CONTROL below shifts the join by one cache step and MUST
            # read large -- otherwise the comparison has no power and the zero
            # means nothing.
            if "poses" not in ep:
                raise RuntimeError("v2ep has no `poses` -- the alignment proof "
                                   "cannot run, and a check that cannot run must "
                                   "not report a pass")
            sp_ep = np.asarray(ep["poses"], dtype=np.float64)[:, 3].reshape(-1)
            fr_ok = [2 * j for j in rows]
            d = np.abs(sp_ep[fr_ok] - b_speed[bi])
            speed_check_max = max(speed_check_max, float(np.nanmax(d)))
            n_speed_checked += len(rows)
            off = [min(nfr - 1, 2 * j + 2) for j in rows]        # MIS-JOIN control
            d_off = np.abs(sp_ep[off] - b_speed[bi])
            speed_ctrl_max = max(speed_ctrl_max, float(np.nanmax(d_off)))
            speed_ctrl_mean.append(float(np.nanmean(d_off)))

            f32 = feat.to(torch.float32)
            fields = np.zeros((len(rows), POOL_H * POOL_W, cfg.d_state), np.float16)
            dinos = np.zeros((len(rows), POOL_H * POOL_W, cfg.d_enc), np.float16)
            with torch.no_grad():
                B = a.batch
                for s in range(0, len(rows), B):
                    chunk = rows[s:s + B]
                    win = torch.stack([
                        f32[[max(0, j - k) for k in range(WIN - 1, -1, -1)]]
                        for j in chunk])                     # [b, WIN, N, d_enc]
                    fld = model.encode(win.to(dev))          # [b, WIN, N, d_state]
                    last = model._last_state(fld)            # [b, N, d_state]
                    last = last.float().cpu().numpy()
                    for k in range(len(chunk)):
                        fields[s + k] = pool_grid(last[k], GRID_H, GRID_W,
                                                  POOL_H, POOL_W).astype(np.float16)
                    raw = f32[chunk].numpy()
                    for k in range(len(chunk)):
                        dinos[s + k] = pool_grid(raw[k], GRID_H, GRID_W,
                                                 POOL_H, POOL_W).astype(np.float16)

            pix = decode_frames(ep, [2 * j for j in rows])
            np.savez_compressed(
                os.path.join(a.out, c + ".npz"),
                rows=np.array(rows, np.int32),
                frame=np.array([2 * j for j in rows], np.int32),
                field=fields, dino=dinos, pix=pix,
                gap=b_gap[bi].astype(np.float32),
                state=b_state[bi],
                speed=b_speed[bi].astype(np.float32),
                rel_speed=b_rel[bi].astype(np.float32),
            )
            nlead = int((b_state[bi] == "LEAD").sum())
            meta.append({"clip": c, "n_rows": len(rows), "n_lead": nlead})
            if (ci + 1) % 10 == 0:
                el = time.time() - t00
                print("  %d/%d clips  %.0fs  (%.2f s/clip)  cuda_max=%.2fGB"
                      % (ci + 1, len(clips), el, el / (ci + 1),
                         torch.cuda.max_memory_allocated() / 1e9), flush=True)
        except Exception as e:
            print("  ERROR %s: %r" % (c, e), flush=True)

    fp_after = trunk_fingerprint(model)
    assert set(fp_before) == set(fp_after), "tensor SET changed -- not a freeze"
    nmis = sum(1 for k in fp_before if fp_before[k][0] != fp_after[k][0])
    maxdelta, worst = 0.0, None
    for k in fp_before:
        d = abs(fp_before[k][1] - fp_after[k][1])
        if d > maxdelta:
            maxdelta, worst = d, k
    held = bool(nmis == 0 and maxdelta == 0.0)
    print("\n=== FREEZE PROOF ===")
    print("trunk tensors checked : %d" % len(fp_before))
    print("sha256 mismatches     : %d" % nmis)
    print("max |delta(sum)|      : %.17g  (worst: %s)" % (maxdelta, worst))
    print("FREEZE %s" % ("HELD (max|delta| = 0 exactly)" if held else "VIOLATED"))
    ctrl_mean = float(np.mean(speed_ctrl_mean)) if speed_ctrl_mean else float("nan")
    print("")
    print("=== LABEL-FREE ALIGNMENT PROOF ===")
    print("rows cross-checked    : %d   (0 would mean the check never ran)"
          % n_speed_checked)
    print("TRUE join   max |dv|  : %.6e m/s" % speed_check_max)
    print("MIS-JOIN(+1) max |dv| : %.6e m/s   mean %.6f m/s   <- MUST be large"
          % (speed_ctrl_max, ctrl_mean))
    align_ok = bool(n_speed_checked > 0 and speed_check_max < 1e-3
                    and ctrl_mean > 1e-2)
    print("ALIGNMENT %s" % ("PROVEN (true join exact, mis-join control separates)"
                            if align_ok else "NOT PROVEN"))

    with open(os.path.join(a.out, "_bank_meta.json"), "w") as f:
        json.dump({"clips": meta, "n_clips": len(meta),
                   "n_rows": sum(m["n_rows"] for m in meta),
                   "n_lead_rows": sum(m["n_lead"] for m in meta),
                   "pool": [POOL_H, POOL_W], "grid": [GRID_H, GRID_W],
                   "pix": [PIX_H, PIX_W], "win": WIN,
                   "d_state": int(cfg.d_state), "d_enc": int(cfg.d_enc),
                   "ckpt": a.ckpt, "ckpt_sha256": ck_sha,
                   "ckpt_step": int(ck.get("step", -1)),
                   "block": a.block,
                   "speed_check_max_mps": speed_check_max,
                   "speed_check_n_rows": n_speed_checked,
                   "speed_misjoin_ctrl_max_mps": speed_ctrl_max,
                   "speed_misjoin_ctrl_mean_mps": ctrl_mean,
                   "alignment_proven": align_ok,
                   "freeze": {"n_tensors": len(fp_before),
                              "sha_mismatches": nmis,
                              "max_abs_delta": maxdelta, "held": held}}, f, indent=1)
    print("\nbanked %d clips / %d rows / %d LEAD rows -> %s"
          % (len(meta), sum(m["n_rows"] for m in meta),
             sum(m["n_lead"] for m in meta), a.out), flush=True)


if __name__ == "__main__":
    sys.exit(main())
