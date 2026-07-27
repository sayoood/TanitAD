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
from tanitad.data.calib import (CanonicalFrame, as_frame,      # noqa: E402
                                cylindrical_rectify, ftheta_crop_resize)
from tanitad.data.comma2k19 import stack_frames                # noqa: E402
from tanitad.data.toy_driving import ToyEpisode                # noqa: E402

_TN = int(os.environ.get("V2_TORCH_THREADS", "0"))
if _TN > 0:
    torch.set_num_threads(_TN)


def _remap(vid, intr, frame, projection_mode):
    """Batch -> canonical frame, via the selected resampler (mirrors
    physicalai._remap_batch so the two paths cannot drift)."""
    if projection_mode == "cylindrical":
        return cylindrical_rectify(vid, intr, frame)
    return ftheta_crop_resize(vid, intr, frame=frame)


def _decode_cropped_selected(mp4, size, frame_idx, frame=None,
                             projection_mode="ftheta_crop"):
    """f-theta-crop ONLY the frames in frame_idx (the ~201 kept @10Hz), not all
    ~605. build_episode crops every frame then subsamples — 2/3 wasted. Per-frame
    crop is independent, so cropping the kept frames gives a BIT-IDENTICAL result
    (validated). ~3x less grid_sample work — the measured bottleneck under load.

    ⛔ The name ``fr`` is RESERVED for the CanonicalFrame in this function and the
    PyAV decode loop below MUST NOT reuse it. `fdc5b4f` introduced ``fr =
    as_frame(...)`` here and left the loop as ``for fr in c.decode(st)``, which
    rebound ``fr`` to a VideoFrame that ``flush()`` then closed over — so
    ``_remap`` received a video frame where the geometry was expected and every
    build raised ``AttributeError: 'VideoFrame' object has no attribute
    'half_angle_x_rad'`` **on the deployed path too**. The identical defect was
    fixed in ``physicalai._decode_mp4`` by `4cb37f4`; this second instance
    survived that hotfix and was found by running the real builder (MEASURED on
    pod2, 2026-07-27). Regression test: ``tests/test_v2_compressed_real.py``."""
    import av
    clip_id = os.path.basename(str(mp4)).split(".")[0]
    intr = intrinsics_for_clip(clip_id, _physicalai_root_of(mp4))
    fr = as_frame(frame, size, 266.0)              # the CanonicalFrame — reserved
    need = set(int(i) for i in frame_idx.tolist())
    batch = int(os.environ.get("PAI_DECODE_BATCH", "16"))
    crops: dict[int, torch.Tensor] = {}
    bidx: list[int] = []; bfr: list[torch.Tensor] = []
    def flush():
        if bfr:
            out = _remap(torch.stack(bfr), intr, fr, projection_mode)
            for j, idx in enumerate(bidx):
                crops[idx] = out[j]
    with av.open(str(mp4)) as container:
        st = container.streams.video[0]; st.thread_type = "AUTO"
        st.thread_count = int(os.environ.get("PAI_DECODE_THREADS", "4"))
        fi = 0
        for vframe in container.decode(st):        # NOT `fr` — see the docstring
            if fi in need:
                bfr.append(torch.from_numpy(
                    vframe.to_ndarray(format="rgb24")).permute(2, 0, 1))
                bidx.append(fi)
                if len(bfr) >= batch:
                    flush(); bidx, bfr = [], []
            fi += 1
        flush()
    return torch.stack([crops[int(i)] for i in frame_idx.tolist()])   # [n,3,H,W] u8

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
#: The sensor name PhysicalAI-AV uses in BOTH the chunk path and the per-clip
#: artifact name inside the zip: ``<clip_id>.camera_front_wide_120fov.mp4`` /
#: ``…​.timestamps.parquet``. It is a single constant because the reuse probe in
#: :func:`build` must look for exactly the name the extractor writes — see the
#: comment there for the defect that came from spelling it twice.
CAM_NAME = "camera_front_wide_120fov"
CAM_TMPL = f"camera/{CAM_NAME}/{CAM_NAME}.chunk_{{chunk_id:04d}}.zip"


def _resampled(clip: dict, size: int, frame=None,
               projection_mode="ftheta_crop"):
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
    vid = _decode_cropped_selected(clip["mp4"], size, frame_idx, frame,
                                   projection_mode)                # [n,3,H,W] u8
    actions, poses = signals_at(ego, t_query)
    n = min(vid.shape[0], actions.shape[0])
    return vid[:n].contiguous(), actions[:n], poses[:n]


def build_compressed(clip: dict, out_path: str, size: int = 256,
                     n_stack: int = 3, quality: int = 90, frame=None,
                     projection_mode: str = "ftheta_crop",
                     codec: str = "jpeg") -> int:
    """Build one compressed clip payload.

    ``codec="jpeg"`` (default) is the deployed lossy path. ``codec="png"`` is the
    LOSSLESS alternative — same container, same loader (the decoder is selected
    from the stored codec), so the two are A/B-able without a format fork.
    ``frame`` / ``projection_mode`` carry the canonical geometry; ``None`` ==
    the deployed square frame, byte-identical to the pre-2026-07-27 path.
    """
    fr = as_frame(frame, size, 266.0)
    vid, actions, poses = _resampled(clip, size, frame, projection_mode)
    enc = (tvio.encode_png if codec == "png" else
           (lambda x: tvio.encode_jpeg(x, quality=quality)))
    jpegs = [enc(vid[i].contiguous()) for i in range(vid.shape[0])]
    lens = torch.tensor([int(j.numel()) for j in jpegs], dtype=torch.int64)
    buf = torch.cat(jpegs) if jpegs else torch.zeros(0, dtype=torch.uint8)
    ep_id = int.from_bytes(clip["clip_id"].encode()[:4].ljust(4, b"\0"), "big")
    tmp = out_path + ".tmp"                                    # atomic: a kill mid-save
    torch.save({"jpeg_buf": buf, "jpeg_len": lens,            # must not leave a corrupt .pt
                "actions": torch.from_numpy(actions), "poses": torch.from_numpy(poses),
                "n_stack": n_stack, "image_size": size, "episode_id": ep_id,
                "clip_id": clip["clip_id"], "quality": quality,
                # geometry (2026-07-27). image_size is KEPT so an older reader
                # still works on a square payload; image_h/image_w are what a
                # non-square one needs, and `frame` is the full provenance.
                "image_h": int(fr.height), "image_w": int(fr.width),
                "frame": fr.to_dict(), "projection_mode": projection_mode,
                "codec": codec}, tmp)
    os.replace(tmp, out_path)
    return int(buf.numel())


def load_compressed(path: str) -> ToyEpisode:
    d = torch.load(path, map_location="cpu", weights_only=False)
    lens = d["jpeg_len"]
    offs = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(lens, 0)])
    buf = d["jpeg_buf"]
    dec = (tvio.decode_png if d.get("codec") == "png" else tvio.decode_jpeg)
    frames = [dec(buf[int(offs[i]):int(offs[i + 1])],
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
    frame = _frame_from_args(a)
    print(f"[measure] {len(clips)} clips, codec={a.codec} quality={a.quality} "
          f"frame={'canonical-256sq' if frame is None else frame.tag()} "
          f"projection={a.projection_mode}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    if clips:
        _assert_geometry_deliverable(a, frame, clips[0]["clip_id"], a.root)
    sizes, ns, t0 = [], [], time.time()
    for c in clips:
        p = os.path.join(a.out, f"{c['clip_id']}.v2ep.pt")
        nb = build_compressed(c, p, quality=a.quality, frame=frame,
                              projection_mode=a.projection_mode, codec=a.codec)
        ep = load_compressed(p)                                # validate round-trip
        sizes.append(os.path.getsize(p)); ns.append(ep.frames.shape[0])
        print(f"  {c['clip_id'][:8]} frames_stacked={tuple(ep.frames.shape)} "
              f"poses={tuple(ep.poses.shape)} man={tuple(ep.maneuvers.shape)} "
              f"file={os.path.getsize(p)/1e6:.2f}MB", flush=True)
    mb = np.mean(sizes) / 1e6
    # 2976 = the parity corpus (2376 train + 600 val); 9000 = the v2bal corpus.
    print(f"[measure] mean {mb:.3f} MB/clip ({np.mean(ns):.0f} stacked frames); "
          f"PROJECTED 2976 clips = {mb*2976/1024:.1f} GB; "
          f"9000 clips = {mb*9000/1024:.1f} GB; "
          f"{(time.time()-t0)/max(len(clips),1):.2f}s/clip", flush=True)


def _hf_download(rel, root, dest=None):
    """curl-based resumable download (pod datacenter net).

    🔒 The bearer token is passed on curl's STDIN as a ``--config`` snippet, never
    in argv. It used to be ``-H "Authorization: Bearer <tok>"``, which put a live
    HF token into the process command line — readable by any `ps` on the pod, and
    captured by every process listing this program's own runbooks tell agents to
    take. `CLAUDE.md` §Invariants says tokens are never passed in args; this is
    that rule applied to the one place that was breaking it.
    """
    import subprocess
    zp = dest or os.path.join(root, rel); os.makedirs(os.path.dirname(zp), exist_ok=True)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{rel}"
    # --speed-limit/--speed-time: abort (exit 28) if <1MB/s for 25s -> --retry
    # resumes (-C -). Prevents the 51KB/s "10h ETA" stalls that hang a worker.
    cmd = ["curl", "-fL", "-C", "-", "--retry", "30", "--retry-delay", "5",
           "--connect-timeout", "30", "--speed-limit", "1000000", "--speed-time", "25",
           "-o", zp, url]
    cfg = ""
    if tok:
        cmd += ["--config", "-"]                 # read the header from stdin
        cfg = f'header = "Authorization: Bearer {tok}"\n'
    for _ in range(6):                                        # outer retry beyond curl --retry
        r = subprocess.run(cmd, input=cfg.encode() if cfg else None)
        if r.returncode == 0:
            return zp
    raise RuntimeError(f"download failed after retries: {rel}")


def _ensure_ego(root, ch):
    """Egomotion zip for `ch` must exist (poses/maneuvers need it). Fetch if missing."""
    ez = os.path.join(root, "labels", "egomotion", f"egomotion.chunk_{ch:04d}.zip")
    if os.path.exists(ez) and os.path.getsize(ez) > 1_000_000:
        return ez
    return _hf_download(f"labels/egomotion/egomotion.chunk_{ch:04d}.zip", root, dest=ez)


def _frame_from_args(a):
    """Resolve the CLI geometry to a CanonicalFrame (or None = deployed square).

    ⛔ PARITY/PROVENANCE: ``None`` reproduces the pre-2026-07-27 deployed frame
    BYTE-IDENTICALLY, so an existing v2 corpus keeps its exact meaning. Any
    non-default geometry is a DIFFERENT corpus and must live in a different
    ``--out`` directory.
    """
    if not a.hfov and not a.width and not a.height:
        return None
    h = a.height or 256
    w = a.width or h
    proj = "cylindrical" if a.projection_mode == "cylindrical" else "pinhole"
    if a.hfov:
        return CanonicalFrame.from_hfov(a.hfov, h, w, proj)
    return CanonicalFrame(height=h, width=w, f_ref=a.f_ref or 266.0,
                          projection=proj)


def _assert_geometry_deliverable(a, frame, clip_id, root):
    """PROVE the requested field is DELIVERED before a multi-hour build.

    Mirrors ``build_pai_cache.py``'s pre-decode assert, which exists because a
    crop that clamps at the sensor edge does not widen — it silently ZOOMS
    (MEASURED: 100 deg on a SQUARE 256 frame delivers only 67.1 deg). Without
    this the v2 builder would happily spend hours writing a corpus at the wrong
    field. Decodes nothing: a ``torch.zeros`` probe drives the same ray map.
    """
    if frame is None:
        return {"canonical": True}
    intr = intrinsics_for_clip(clip_id, root)
    probe = torch.zeros(1, 3, intr.height, intr.width, dtype=torch.uint8)
    if a.projection_mode == "cylindrical":
        cylindrical_rectify(probe, intr, frame, require_per_clip=intr.per_clip)
        got_f = float(cylindrical_rectify.last_f_eff)
        obs = float(cylindrical_rectify.last_observed_frac)
    else:
        ftheta_crop_resize(probe, intr, frame=frame)
        got_f = float(ftheta_crop_resize.last_f_eff)
        obs = 1.0
    import math
    got_hfov = math.degrees(2 * (math.atan((frame.width / 2) / got_f)
                                 if frame.projection == "pinhole"
                                 else (frame.width / 2) / got_f))
    rep = {"requested_hfov_deg": round(float(frame.hfov_deg), 4),
           "achieved_hfov_deg": round(got_hfov, 4), "f_eff": round(got_f, 4),
           "observed_frac": round(obs, 6), "frame_tag": frame.tag()}
    print(f"[build] geometry check (clip {clip_id[:8]}): {json.dumps(rep)}",
          flush=True)
    assert abs(got_hfov - frame.hfov_deg) < 0.5, (
        f"ABORT: requested {frame.hfov_deg:.2f}deg but this sensor can only "
        f"deliver {got_hfov:.2f}deg at {frame.height}x{frame.width} — the crop "
        f"CLAMPED at the sensor edge and would have ZOOMED instead of widening. "
        f"Widen the frame (more columns) or lower the requested HFOV.")
    return rep


def build(a):
    sel = pd.read_parquet(a.sel)
    sel["clip_id"] = sel["clip_id"].astype(str)
    if a.only_clips:
        keep = {ln.strip() for ln in open(a.only_clips) if ln.strip()}
        before = len(sel)
        sel = sel[sel["clip_id"].isin(keep)]
        print(f"[build] --only-clips {a.only_clips}: {before} -> {len(sel)} "
              f"clips (list had {len(keep)})", flush=True)
        missing = keep - set(sel["clip_id"])
        if missing:
            raise SystemExit(
                f"[build] REFUSING: {len(missing)} clip ids in --only-clips are "
                f"NOT in --sel. A build over a DIFFERENT clip set than the one "
                f"requested is an episode re-selection; fix the inputs.")
    frame = _frame_from_args(a)
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
          f"{len(done)} built already; codec={a.codec} quality={a.quality} "
          f"frame={'canonical-256sq' if frame is None else frame.tag()} "
          f"projection={a.projection_mode}", flush=True)
    # A geometry manifest next to the payloads: which frame this cache IS.
    # Without it a wide cache is indistinguishable from the deployed one on disk.
    geom_manifest = {
        "frame": None if frame is None else frame.to_dict(),
        "frame_tag": "canonical-256sq" if frame is None else frame.tag(),
        "projection_mode": a.projection_mode, "codec": a.codec,
        "quality": a.quality if a.codec == "jpeg" else None,
        "selection_parquet": str(a.sel), "only_clips": a.only_clips or None,
        "clips_requested": int(len(sel)),
    }
    t0, nbuilt, nbytes = time.time(), 0, 0
    checked_geometry = False
    n_reused = 0
    for ci, ch in enumerate(chunks):
        want = by_chunk[ch] - done
        if not want:
            continue
        # Reuse clips whose mp4 + timestamps are ALREADY on this host. The camera
        # chunk zips are ~2 GB each and this corpus spans 197 of them (~394 GB of
        # egress); a host that already carries part of the raw corpus should not
        # re-download it. MEASURED 2026-07-27: pod2 held 760 of the 3,000 clips,
        # so this skips ~25 % of the download. Purely an IO saving — the clips
        # built are EXACTLY the same set either way, so selection is untouched.
        # ⛔ THE PROBE MUST USE THE NAME THE EXTRACTOR WRITES. The zip entries —
        # and therefore every mp4 that has ever existed in `cam_dir` — are
        # `<clip_id>.camera_front_wide_120fov.mp4`. This looked for
        # `<clip_id>.mp4`, a name NO artifact on this corpus has ever had, so
        # the reuse branch could never fire and the "reuse" was inert.
        # MEASURED 2026-07-27: all 8 shards of the wide TRAIN build logged
        # `reused_local=0` while 761 clips sat decoded on the host — including
        # ALL 600 clips of the parity VAL split. Silent, because a redundant
        # download is only slow, never wrong.
        # ``PAI_NO_LOCAL_REUSE=1`` forces the download path (the recovery route
        # if a host's local copy is suspect).
        local: dict[str, dict] = {}
        if os.environ.get("PAI_NO_LOCAL_REUSE") != "1":
            for cid in sorted(want):
                for stem in (f"{cid}.{CAM_NAME}", cid):
                    mp4 = os.path.join(cam_dir, f"{stem}.mp4")
                    ts = os.path.join(cam_dir, f"{stem}.timestamps.parquet")
                    if os.path.exists(mp4) and os.path.exists(ts):
                        local[cid] = {"mp4": mp4, "timestamps": ts,
                                      "_preexisting": True}
                        break
        need_dl = want - set(local)
        n_reused += len(local)
        zp = None
        if need_dl:
            try:
                _ensure_ego(a.root, ch)
                zp = _hf_download(CAM_TMPL.format(chunk_id=ch), a.root,
                                  dest=os.path.join(a.root, "r0", f"_cam_{ch:04d}.zip"))
            except Exception as e:
                print(f"{tag} chunk {ch} fetch FAILED: {e}", flush=True)
                if not local:
                    continue
        else:
            _ensure_ego(a.root, ch)
            print(f"{tag} chunk {ch}: all {len(local)} clips already local — "
                  f"no download", flush=True)
        # extract only selected clips' mp4 + timestamps; track paths per clip_id
        ego_zip = os.path.join(a.root, "labels", "egomotion", f"egomotion.chunk_{ch:04d}.zip")
        got: dict[str, dict] = dict(local)
        if zp is not None:
            with zipfile.ZipFile(zp) as z:
                for name in z.namelist():
                    cid = name.split("/")[-1].split(".")[0]
                    if cid not in need_dl:
                        continue
                    if name.endswith(".mp4"):
                        z.extract(name, cam_dir); got.setdefault(cid, {})["mp4"] = os.path.join(cam_dir, name)
                    elif name.endswith(".timestamps.parquet"):
                        z.extract(name, cam_dir); got.setdefault(cid, {})["timestamps"] = os.path.join(cam_dir, name)
            os.unlink(zp)                                      # ~2 GB — never keep
        for cid, paths in got.items():
            if cid in done or "mp4" not in paths or "timestamps" not in paths:
                continue
            clip = {"clip_id": cid, "mp4": paths["mp4"],
                    "timestamps": paths["timestamps"], "ego_zip": ego_zip}
            p = os.path.join(a.out, f"{cid}.v2ep.pt")
            if not checked_geometry:      # ABORT before hours of work, not after
                geom_manifest["geometry_check"] = _assert_geometry_deliverable(
                    a, frame, cid, a.root)
                with open(os.path.join(a.out, "_geometry.json"), "w") as fh:
                    json.dump(geom_manifest, fh, indent=1)
                checked_geometry = True
            try:
                nbytes += build_compressed(clip, p, quality=a.quality,
                                           frame=frame,
                                           projection_mode=a.projection_mode,
                                           codec=a.codec); nbuilt += 1
                done.add(cid)
            except Exception as e:
                print(f"{tag} clip {cid[:8]} FAILED: {type(e).__name__}: {e}", flush=True)
        # Delete only mp4s WE extracted — the compressed cache holds them now.
        # ⛔ Never delete a clip that was already on the host: this builder is
        # not entitled to destroy another stream's copy of the raw corpus, and
        # on a partially-populated host that would silently shrink it.
        for paths in got.values():
            if paths.get("_preexisting"):
                continue
            for k, f in paths.items():
                if k.startswith("_"):
                    continue
                try: os.unlink(f)
                except OSError: pass
        print(f"{tag} chunk {ci+1}/{len(chunks)} (#{ch}) built={nbuilt} "
              f"reused_local={n_reused} cache={nbytes/1024**3:.2f}GB "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"{tag} DONE built={nbuilt} cache={nbytes/1024**3:.2f}GB {time.time()-t0:.0f}s", flush=True)


def _add_geometry_args(p):
    """INPUT GEOMETRY + CODEC for the v2 builder (wide-FOV enablement).

    Every default reproduces the deployed 256x256 / f_ref 266 / pinhole /
    ftheta_crop / JPEG-q90 cache byte-identically, so an existing v2 corpus keeps
    its exact meaning. ``build_compressed()`` has accepted ``frame`` /
    ``projection_mode`` / ``codec`` since `fdc5b4f`, but the CLI never passed
    them — so the ONLY cache the command line could produce was the deployed
    square JPEG one, and the wide-FOV build had no entry point at all.
    """
    p.add_argument("--hfov", type=float, default=0.0,
                   help="horizontal field in degrees; solves f_ref for the "
                        "given HxW (0 = deployed canonical frame)")
    p.add_argument("--height", type=int, default=0)
    p.add_argument("--width", type=int, default=0,
                   help="output columns; WIDENING REQUIRES MORE COLUMNS — a "
                        "square frame clamps at the sensor and silently zooms")
    p.add_argument("--f-ref", type=float, default=0.0,
                   help="explicit focal instead of --hfov")
    p.add_argument("--projection-mode", choices=("ftheta_crop", "cylindrical"),
                   default="ftheta_crop",
                   help="ftheta_crop (deployed; replicate-pads rig B) or "
                        "cylindrical (masks instead of fabricating)")
    p.add_argument("--codec", choices=("jpeg", "png"), default="jpeg",
                   help="png = LOSSLESS (bit-exact, ~2.6x; decodes FASTER than "
                        "jpeg); jpeg = deployed lossy path")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("measure"); m.add_argument("--root", required=True)
    m.add_argument("--out", default="/workspace/tmp/v2measure"); m.add_argument("--n", type=int, default=12)
    m.add_argument("--quality", type=int, default=90)
    _add_geometry_args(m)
    b = sub.add_parser("build")
    b.add_argument("--sel", required=True); b.add_argument("--root", required=True)
    b.add_argument("--out", required=True); b.add_argument("--quality", type=int, default=90)
    b.add_argument("--shard", default="", help="i/K — build chunks with index%%K==i")
    b.add_argument("--only-clips", default="",
                   help="path to a newline-separated clip_id list — restricts "
                        "the build to EXACTLY those clips (e.g. the parity "
                        "train split). Refuses if any id is not in --sel.")
    _add_geometry_args(b)
    a = ap.parse_args()
    (measure if a.mode == "measure" else build)(a)
