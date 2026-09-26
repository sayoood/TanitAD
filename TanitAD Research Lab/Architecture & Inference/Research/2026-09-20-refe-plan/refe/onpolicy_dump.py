#!/usr/bin/env python3
"""On-policy proposal dump, POD-LOCAL (PI decision 2026-09-26, option B -- the paper's supervision).

The student's OWN 64 proposals per training sample, for `onpolicy_label.py` to score. The model is the
live run's newest `ckpt_last.pt` (partial: LoRA + heads over the DINOv3 trunk), reloaded whenever the
trainer writes a newer one; each sample's inputs come from `train.TargetBank.__getitem__` -- the
trainer's OWN input construction (images, ego, goal, per-sample rig) -- and the forward runs under
the trainer's numerics (bf16 autocast + TF32, eval mode, no grad).

  python onpolicy_dump.py --run-dir <run> --targets <train_grow> --images <pixels> --calib <calib.json> \
      --queue <queue dir> [--batch 4 --chunk 32 --max-pending 3]

Writes `<queue>/props_r<rank>_<ckptstep>_<seq>.jsonl`, one line per sample (`kind: onpolicy_props`:
key, ckpt_step, the 64 proposals [M][T][3], the banked teacher target, the row's `aug`), atomically
(tmp + rename). ⛔ BACKPRESSURE: a rank gets a new chunk only while fewer than --max-pending chunks
wait for it, so the A40 -- shared with the live trainer -- runs this only as fast as the CPU
labellers consume, and every chunk uses the checkpoint current at the time it is dumped.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ckpt_io  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402
import train as T  # noqa: E402


def stable_mtime(p: Path, settle_s: float = 5.0):
    """The file's mtime once its size has stopped changing (the trainer writes tmp + rename)."""
    try:
        s1 = p.stat()
        time.sleep(settle_s)
        s2 = p.stat()
    except OSError:
        return None
    return s2.st_mtime if s1.st_size == s2.st_size and s1.st_mtime == s2.st_mtime else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--max-pending", type=int, default=3)
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--max-chunks", type=int, default=0, help="stop after this many (smoke)")
    a = ap.parse_args()
    torch.set_num_threads(1)                      # the pod's CPUs belong to the trainer's loaders
    torch.backends.cuda.matmul.allow_tf32 = True  # the trainer's --tf32
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.queue, exist_ok=True)
    ranks = [int(x) for x in a.ranks.split(",")]
    cfg = REFeConfig.for_backbone(a.backbone)
    model = REFe(cfg)
    ck = Path(a.run_dir) / "ckpt_last.pt"
    m0 = stable_mtime(ck)
    meta: dict = {}
    ckpt_io.load_for_inference(model, str(ck), map_location="cpu", backbone=a.backbone, meta_out=meta)
    st = torch.load(ck, map_location="cpu", weights_only=False)
    ckpt_step = int(st["state"]["step"])
    frozen = (st.get("meta") or {}).get("frozen_sha256")
    del st
    model = model.to("cuda").eval()
    print(f"  model: {a.backbone} from {ck} at step {ckpt_step} (per_sample_calib "
          f"{meta.get('per_sample_calib')})", flush=True)
    # the trainer's dataset, built ONCE: every sample's inputs exactly as training builds them
    ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib)
    by_rank = {r: [i for i, row in enumerate(ds.rows) if int(row.get("rank", 0)) == r] for r in ranks}
    for r in ranks:
        random.Random(a.seed + r).shuffle(by_rank[r])
    cursor = {r: 0 for r in ranks}
    # resume: never re-dump a (sample, checkpoint) that is queued, claimed or done
    seen = set()
    for q in glob.glob(os.path.join(a.queue, "props_r*_*.jsonl*")):
        try:
            with open(q, encoding="utf-8") as f:
                for line in f:
                    rr = json.loads(line)
                    seen.add((rr["log_name"], rr["token"], int(rr["step"]), int(rr["rank"])))
        except (OSError, json.JSONDecodeError):
            continue
    print(f"  bank: {len(ds):,} samples ({', '.join(f'rank {r}: {len(v):,}' for r, v in by_rank.items())}); "
          f"{len(seen):,} already queued/labelled", flush=True)
    seq, n_chunks = int(time.time()), 0
    while True:
        # a newer checkpoint: reload the trainable weights only (the trunk is frozen and verified)
        m1 = stable_mtime(ck, settle_s=2.0)
        if m1 is not None and m0 is not None and m1 > m0:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            ckpt_io.load_partial(model, sd["model_partial"], frozen)
            ckpt_step = int(sd["state"]["step"])
            del sd
            m0 = m1
            print(f"  reloaded {ck.name} at step {ckpt_step}", flush=True)
        produced = False
        for r in ranks:
            pending = len(glob.glob(os.path.join(a.queue, f"props_r{r}_*.jsonl")))
            if pending >= a.max_pending:
                continue
            idx = []
            while len(idx) < a.chunk and cursor[r] < len(by_rank[r]):
                i = by_rank[r][cursor[r]]
                cursor[r] += 1
                row = ds.rows[i]
                if (row["log_name"], row["token"], int(row.get("step", 0)), r) not in seen:
                    idx.append(i)
            if not idx:
                continue
            t0 = time.time()
            lines = []
            for b0 in range(0, len(idx), a.batch):
                items = [ds[i] for i in idx[b0:b0 + a.batch]]
                img = torch.stack([it[0] for it in items]).cuda(non_blocking=True)
                ego = torch.stack([it[1] for it in items]).cuda(non_blocking=True)
                goal = torch.stack([it[2] for it in items]).cuda(non_blocking=True)
                cal = torch.stack([it[8] for it in items]) if items[0][8].numel() else None
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    traj, _score = model(img, ego, goal, calib=cal)
                traj = traj.float().cpu().numpy()
                for j, i in enumerate(idx[b0:b0 + a.batch]):
                    row = ds.rows[i]
                    lines.append(json.dumps({
                        "kind": "onpolicy_props", "log_name": row["log_name"], "token": row["token"],
                        "step": int(row.get("step", 0)), "rank": r, "ckpt_step": ckpt_step,
                        "teacher": row["traj"], "aug": row.get("aug"),
                        "props": [[[round(float(v), 5) for v in p] for p in traj[j, k]]
                                  for k in range(traj.shape[1])]}) + "\n")
                    seen.add((row["log_name"], row["token"], int(row.get("step", 0)), r))
            out = os.path.join(a.queue, f"props_r{r}_{ckpt_step:06d}_{seq:010d}.jsonl")
            seq += 1
            tmp = out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("".join(lines))
            os.replace(tmp, out)
            n_chunks += 1
            produced = True
            print(f"  {os.path.basename(out)}: {len(idx)} samples in {time.time() - t0:.1f} s "
                  f"(rank {r} cursor {cursor[r]:,}/{len(by_rank[r]):,}; GPU mem "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB peak)", flush=True)
            if a.max_chunks and n_chunks >= a.max_chunks:
                print(f"ZZOPDUMP_DONE {n_chunks} chunks")
                return 0
        if all(cursor[r] >= len(by_rank[r]) for r in ranks):
            print("ZZOPDUMP_EXHAUSTED every sample queued once")
            return 0
        if not produced:
            time.sleep(a.poll_s)


if __name__ == "__main__":
    sys.exit(main())
