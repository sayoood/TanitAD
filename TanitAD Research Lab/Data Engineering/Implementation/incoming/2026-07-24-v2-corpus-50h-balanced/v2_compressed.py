"""v2 compressed episode cache — JPEG-encoded f-theta-cropped 256px frames.

Sayed's "FIT THE QUOTA" path: the raw uint8 epcache is ~112 MB/episode (982 GB for
9,000). We store the SAME f-theta-cropped 256px frames (full 256px parity — no
downscale) but JPEG-encoded and UN-stacked (stacking is redundant 3x storage and is
reproduced at load), cutting the cache ~15-25x.

Faithfulness: reuses physicalai._decode_mp4 (identical f-theta crop + per-clip
intrinsics), signals_at (identical poses), maneuvers_for_poses (identical labels),
and comma2k19.stack_frames (identical D-015 stacking) — the ONLY difference from
build_episode's output is JPEG lossiness on the frames. load_compressed() returns a
real ToyEpisode(frames[T-2,9,256,256] uint8, actions, poses, maneuvers).

Modes:
  measure  --root <r> --n 12 [--quality 90]   build+load N clips, report MB/clip
  build    --sel <parquet> --root <r> --out <dir> --egoroot <r> [--quality 90]
           per-chunk fetch camera -> extract selected -> build compressed ->
           delete mp4s+zip. Resumable (skips built clips). Banks incrementally.
"""
from __future__ import annotations
import argparse, io, json, os, sys, time, zipfile
import numpy as np, pandas as pd, torch
import torchvision.io as tvio

_STACK = os.environ.get("TANITAD_STACK", "/workspace/TanitAD/stack")
sys.path.insert(0, _STACK); sys.path.append(os.path.join(_STACK, "scripts"))
from tanitad.data.physicalai import (                          # noqa: E402
    _decode_mp4, signals_at, load_egomotion, maneuvers_for_poses, TARGET_HZ,
    intrinsics_for_clip, _physicalai_root_of)
from tanitad.data.calib import ftheta_crop_resize              # noqa: E402
from tanitad.data.comma2k19 import stack_frames                # noqa: E402
from tanitad.data.toy_driving import ToyEpisode                # noqa: E402

_TN = int(os.environ.get("V2_TORCH_THREADS", "0"))
if _TN > 0:
    torch.set_num_threads(_TN)


def _decode_cropped_selected(mp4, size, frame_idx):
    """f-theta-crop ONLY the frames in frame_idx (the ~201 kept @10Hz), not all
    ~605. build_episode crops every frame then subsamples — 2/3 wasted. Per-frame
    crop is independent, so cropping the kept frames gives a BIT-IDENTICAL result
    (validated). ~3x less grid_sample work — the measured bottleneck under load."""
    import av
    clip_id = os.path.basename(str(mp4)).split(".")[0]
    intr = intrinsics_for_clip(clip_id, _physicalai_root_of(mp4))
    need = set(int(i) for i in frame_idx.tolist())
    batch = int(os.environ.get("PAI_DECODE_BATCH", "16"))
    crops: dict[int, torch.Tensor] = {}
    bidx: list[int] = []; bfr: list[torch.Tensor] = []
    def flush():
        if bfr:
            c = ftheta_crop_resize(torch.stack(bfr), intr, size)
            for j, idx in enumerate(bidx):
                crops[idx] = c[j]
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]; st.thread_type = "AUTO"
        st.thread_count = int(os.environ.get("PAI_DECODE_THREADS", "4"))
        fi = 0
        for fr in c.decode(st):
            if fi in need:
                bfr.append(torch.from_numpy(fr.to_ndarray(format="rgb24")).permute(2, 0, 1))
                bidx.append(fi)
                if len(bfr) >= batch:
                    flush(); bidx, bfr = [], []
            fi += 1
        flush()
    return torch.stack([crops[int(i)] for i in frame_idx.tolist()])   # [n,3,S,S] u8

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
CAM_TMPL = ("camera/camera_front_wide_120fov/"
            "camera_front_wide_120fov.chunk_{chunk_id:04d}.zip")


def _resampled(clip: dict, size: int):
    """build_episode's frames/actions/poses up to vid[:n] (UN-stacked)."""
    ts = pd.read_parquet(clip["timestamps"])
    tcol = next(c for c in ts.columns if "time" in c.lower())
    t_frames = ts[tcol].to_numpy(np.float64)
    ego = load_egomotion(clip["ego_zip"], clip["clip_id"])
    span = t_frames[-1] - t_frames[0]; unit = 1.0
    for cand in (1e9, 1e6, 1e3):
        if span / cand > 1.0:
            unit = cand; break
    n_target = max(int(span / unit * TARGET_HZ), 4)
    t_query = np.linspace(t_frames[0], t_frames[-1], n_target)
    frame_idx = np.searchsorted(t_frames, t_query).clip(0, len(t_frames) - 1)
    vid = _decode_cropped_selected(clip["mp4"], size, frame_idx)   # [n,3,S,S] u8
    actions, poses = signals_at(ego, t_query)
    n = min(vid.shape[0], actions.shape[0])
    return vid[:n].contiguous(), actions[:n], poses[:n]


def build_compressed(clip: dict, out_path: str, size: int = 256,
                     n_stack: int = 3, quality: int = 90) -> int:
    vid, actions, poses = _resampled(clip, size)
    jpegs = [tvio.encode_jpeg(vid[i].contiguous(), quality=quality)
             for i in range(vid.shape[0])]
    lens = torch.tensor([int(j.numel()) for j in jpegs], dtype=torch.int64)
    buf = torch.cat(jpegs) if jpegs else torch.zeros(0, dtype=torch.uint8)
    ep_id = int.from_bytes(clip["clip_id"].encode()[:4].ljust(4, b"\0"), "big")
    tmp = out_path + ".tmp"                                    # atomic: a kill mid-save
    torch.save({"jpeg_buf": buf, "jpeg_len": lens,            # must not leave a corrupt .pt
                "actions": torch.from_numpy(actions), "poses": torch.from_numpy(poses),
                "n_stack": n_stack, "image_size": size, "episode_id": ep_id,
                "clip_id": clip["clip_id"], "quality": quality}, tmp)
    os.replace(tmp, out_path)
    return int(buf.numel())


def load_compressed(path: str) -> ToyEpisode:
    d = torch.load(path, map_location="cpu", weights_only=False)
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    frames = [tvio.decode_jpeg(buf[int(offs[i]):int(offs[i + 1])],
                               mode=tvio.ImageReadMode.RGB) for i in range(len(lens))]
    vid = torch.stack(frames)                                  # [n,3,S,S] u8
    k = d["n_stack"] - 1
    stacked = stack_frames(vid, d["n_stack"])                  # [n-k,9,S,S]
    poses = d["poses"][k:]
    return ToyEpisode(frames=stacked, actions=d["actions"][k:], poses=poses,
                      episode_id=int(d["episode_id"]),
                      maneuvers=maneuvers_for_poses(poses))


def _discover(root: str, sel_ids: set | None = None) -> list[dict]:
    from tanitad.data.physicalai import discover_r0_clips
    clips = discover_r0_clips(root)
    return [c for c in clips if sel_ids is None or c["clip_id"] in sel_ids]


# --------------------------------------------------------------------------- #
def measure(a):
    clips = _discover(a.root)[:a.n]
    print(f"[measure] {len(clips)} clips, quality={a.quality}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    sizes, ns, t0 = [], [], time.time()
    for c in clips:
        p = os.path.join(a.out, f"{c['clip_id']}.v2ep.pt")
        nb = build_compressed(c, p, quality=a.quality)
        ep = load_compressed(p)                                # validate round-trip
        sizes.append(os.path.getsize(p)); ns.append(ep.frames.shape[0])
        print(f"  {c['clip_id'][:8]} frames_stacked={tuple(ep.frames.shape)} "
              f"poses={tuple(ep.poses.shape)} man={tuple(ep.maneuvers.shape)} "
              f"file={os.path.getsize(p)/1e6:.2f}MB", flush=True)
    mb = np.mean(sizes) / 1e6
    print(f"[measure] mean {mb:.3f} MB/clip ({np.mean(ns):.0f} stacked frames); "
          f"PROJECTED 9000 clips = {mb*9000/1024:.1f} GB; "
          f"{(time.time()-t0)/len(clips):.2f}s/clip", flush=True)


def _hf_download(rel, root, dest=None):
    """curl-based resumable download (pod datacenter net)."""
    import subprocess
    zp = dest or os.path.join(root, rel); os.makedirs(os.path.dirname(zp), exist_ok=True)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    # --speed-limit/--speed-time: abort (exit 28) if <1MB/s for 25s -> --retry
    # resumes (-C -). Prevents the 51KB/s "10h ETA" stalls that hang a worker.
    cmd = ["curl", "-fL", "-C", "-", "--retry", "30", "--retry-delay", "5",
           "--connect-timeout", "30", "--speed-limit", "1000000", "--speed-time", "25",
           "-o", zp, url]
    if tok:
        cmd += ["-H", f"Authorization: Bearer {tok}"]
    for _ in range(6):                                        # outer retry beyond curl --retry
        r = subprocess.run(cmd)
        if r.returncode == 0:
            return zp
    raise RuntimeError(f"download failed after retries: {rel}")


def _ensure_ego(root, ch):
    """Egomotion zip for `ch` must exist (poses/maneuvers need it). Fetch if missing."""
    ez = os.path.join(root, "labels", "egomotion", f"egomotion.chunk_{ch:04d}.zip")
    if os.path.exists(ez) and os.path.getsize(ez) > 1_000_000:
        return ez
    return _hf_download(f"labels/egomotion/egomotion.chunk_{ch:04d}.zip", root, dest=ez)


def build(a):
    sel = pd.read_parquet(a.sel)
    by_chunk: dict[int, set] = {}
    for _, r in sel.iterrows():
        by_chunk.setdefault(int(r["chunk"]), set()).add(str(r["clip_id"]))
    chunks = sorted(by_chunk)
    si, sk = 0, 1
    if a.shard:
        si, sk = (int(x) for x in a.shard.split("/"))
        chunks = [c for k, c in enumerate(chunks) if k % sk == si]   # disjoint by chunk
    os.makedirs(a.out, exist_ok=True)
    cam_dir = os.path.join(a.root, "r0", "camera_front_wide"); os.makedirs(cam_dir, exist_ok=True)
    done = {p.split(".v2ep")[0] for p in os.listdir(a.out) if p.endswith(".v2ep.pt")}
    tag = f"[build s{si}/{sk}]"
    print(f"{tag} {len(chunks)} chunks / {sum(len(by_chunk[c]) for c in chunks)} clips; "
          f"{len(done)} built already; quality={a.quality}", flush=True)
    t0, nbuilt, nbytes = time.time(), 0, 0
    for ci, ch in enumerate(chunks):
        want = by_chunk[ch] - done
        if not want:
            continue
        try:
            _ensure_ego(a.root, ch)
            zp = _hf_download(CAM_TMPL.format(chunk_id=ch), a.root,
                              dest=os.path.join(a.root, "r0", f"_cam_{ch:04d}.zip"))
        except Exception as e:
            print(f"{tag} chunk {ch} fetch FAILED: {e}", flush=True); continue
        # extract only selected clips' mp4 + timestamps; track paths per clip_id
        ego_zip = os.path.join(a.root, "labels", "egomotion", f"egomotion.chunk_{ch:04d}.zip")
        got: dict[str, dict] = {}
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                cid = name.split("/")[-1].split(".")[0]
                if cid not in want:
                    continue
                if name.endswith(".mp4"):
                    z.extract(name, cam_dir); got.setdefault(cid, {})["mp4"] = os.path.join(cam_dir, name)
                elif name.endswith(".timestamps.parquet"):
                    z.extract(name, cam_dir); got.setdefault(cid, {})["timestamps"] = os.path.join(cam_dir, name)
        os.unlink(zp)                                          # 1.3 GB — never keep
        for cid, paths in got.items():
            if cid in done or "mp4" not in paths or "timestamps" not in paths:
                continue
            clip = {"clip_id": cid, "mp4": paths["mp4"],
                    "timestamps": paths["timestamps"], "ego_zip": ego_zip}
            p = os.path.join(a.out, f"{cid}.v2ep.pt")
            try:
                nbytes += build_compressed(clip, p, quality=a.quality); nbuilt += 1
                done.add(cid)
            except Exception as e:
                print(f"{tag} clip {cid[:8]} FAILED: {type(e).__name__}: {e}", flush=True)
        for paths in got.values():                            # delete mp4s (cache holds them compressed)
            for f in paths.values():
                try: os.unlink(f)
                except OSError: pass
        print(f"{tag} chunk {ci+1}/{len(chunks)} (#{ch}) built={nbuilt} "
              f"cache={nbytes/1024**3:.2f}GB {time.time()-t0:.0f}s", flush=True)
    print(f"{tag} DONE built={nbuilt} cache={nbytes/1024**3:.2f}GB {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("measure"); m.add_argument("--root", required=True)
    m.add_argument("--out", default="/workspace/tmp/v2measure"); m.add_argument("--n", type=int, default=12)
    m.add_argument("--quality", type=int, default=90)
    b = sub.add_parser("build")
    b.add_argument("--sel", required=True); b.add_argument("--root", required=True)
    b.add_argument("--out", required=True); b.add_argument("--quality", type=int, default=90)
    b.add_argument("--shard", default="", help="i/K — build chunks with index%%K==i")
    a = ap.parse_args()
    (measure if a.mode == "measure" else build)(a)
