"""v2 corpus -> PH0 pilot inputs (the bridge the VLM/algorithmic pipeline eats).

Turns compressed ``<clip>.v2ep.pt`` episodes into what ``ph0_pilot.py`` consumes::

    <out>/videos/<clip_id>.mp4   10 Hz, the frames the world model itself sees
    <out>/ego/<clip_id>.npz      poses [T, 4] = x, y, yaw, v  @10 Hz
    <out>/clips.json             the clip list actually written

Selection = clips present in BOTH the Alpamayo ``records.parquet`` AND the v2
corpus, so all four pilot engines (A geometry / B VLM / C SAM / D Alpamayo) fire
on the SAME clip and their outputs are comparable window-for-window.

⚠️ Two things this file exists to get right, both of which bit us once:

1. **It does not decode JPEG itself.** The v2 cache stores ``jpeg_buf``/
   ``jpeg_len``/``n_stack``, NOT a ``frames`` tensor — an earlier version of this
   bridge looked for ``ep["frames"]``, found nothing, and SKIPPED every clip while
   exiting 0 (`BRIDGE_DONE n=0`, which reads like success). It now calls
   ``tanitad.data.v2_dataset.decode_full_episode``, the canonical decoder that is
   byte-identical to ``scripts/v2_compressed.load_compressed``. There are already
   three v2 decode paths in this program; this is deliberately not a fourth.

2. **Channel slice ``[-3:]``, not ``[:3]``.** ``stack_frames`` row *j* is the
   channel-concat of raw frames *j … j+n_stack-1*, and ``decode_full_episode``
   returns poses ``d["poses"][k:]`` with ``k = n_stack-1``. So row *j*'s pose is
   raw frame *j+k*, which is the LAST 3 channels. Taking the first 3 would produce
   a video offset from its own ego trace by ``k`` frames — a silent misalignment
   that would land as a BEV/overlay registration error nobody could trace back.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch


def stacked_to_rgb(frames: torch.Tensor) -> np.ndarray:
    """``[T, 3*n_stack, H, W]`` uint8 -> ``[T, H, W, 3]`` uint8, pose-aligned.

    Takes the LAST raw frame of each stack (see this module's docstring) and
    tolerates the already-unstacked and grayscale cases so a corpus variant
    cannot silently produce a 1- or 12-channel "video"."""
    fr = torch.as_tensor(frames)
    if fr.dtype != torch.uint8:                        # stored float 0..1
        fr = (fr.clamp(0, 1) * 255).to(torch.uint8)
    if fr.ndim != 4:
        raise ValueError(f"expected [T,C,H,W], got {tuple(fr.shape)}")
    c = int(fr.shape[1])
    if c % 3 == 0 and c >= 3:
        fr = fr[:, -3:]                                # pose-aligned raw frame
    elif c == 1:
        fr = fr.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"channel count {c} is neither 1 nor a multiple of 3")
    return fr.permute(0, 2, 3, 1).contiguous().numpy()


def pick_clips(corpus: str, records: str | None, n: int, seed: int) -> list[str]:
    corp = {os.path.basename(p).split(".")[0]: p
            for p in glob.glob(os.path.join(corpus, "*.v2ep.pt"))}
    if not corp:
        raise SystemExit(f"[bridge] no *.v2ep.pt under {corpus}")
    if records:
        import pandas as pd
        ids = set(pd.read_parquet(records)["clip_id"].astype(str).unique())
        pool = sorted(ids & set(corp))
        if not pool:
            # ⛔ do NOT fall back to corpus-only: the whole point of the join is
            # that engine D (Alpamayo) has a record for the clip. A silent
            # fallback would produce a pilot whose D column is empty and whose
            # comparison across engines is meaningless.
            raise SystemExit("[bridge] no clip overlap between records and "
                             "corpus — refusing to run a pilot engine D cannot "
                             "score")
    else:
        pool = sorted(corp)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    return [pool[i] for i in order[:n]], corp


def write_mp4(path: str, frames, fps: int) -> str:
    """Write an RGB uint8 [T,H,W,3] stack to H.264, using whatever the host has.

    ⛔ WHY THIS IS NOT JUST ``imageio.get_writer``. MEASURED 2026-08-13: pod5 —
    the pod that HOLDS the 80 GB corpus — has ``imageio_ffmpeg`` (which bundles
    the ffmpeg BINARY) but not ``imageio``, and no cv2 and no av. The bridge
    failed 2400/2400 clips on ``ModuleNotFoundError: imageio``.

    ⚠️ And installing it was the wrong fix: pod5 was TRAINING at the time, and
    ``uv pip install`` has twice replaced torch with a wheel the driver cannot
    run (CLAUDE.md). The binary is already on disk behind
    ``imageio_ffmpeg.get_ffmpeg_exe()``; piping raw frames to it adds no package
    and cannot touch the running job's environment.

    Order: imageio (nicest when present) -> bundled ffmpeg binary -> raise with
    both reasons, so a failure names what was tried rather than the last error."""
    import subprocess
    import numpy as _np
    arr = _np.ascontiguousarray(_np.asarray(frames, dtype=_np.uint8))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] uint8, got {arr.shape}")
    try:
        import imageio.v2 as _iio
        w = _iio.get_writer(path, fps=fps, macro_block_size=1)
        for f in arr:
            w.append_data(f)
        w.close()
        return "imageio"
    except ImportError as e_iio:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e_bin:                        # noqa: BLE001
            raise RuntimeError(
                f"no mp4 writer: imageio ({e_iio}); "
                f"imageio_ffmpeg ({type(e_bin).__name__}: {e_bin})") from e_iio
        T, H, W, _ = arr.shape
        cmd = [exe, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "18", path]
        pr = subprocess.run(cmd, input=arr.tobytes(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pr.returncode != 0 or not os.path.exists(path):
            raise RuntimeError(
                f"ffmpeg rc={pr.returncode}: "
                f"{pr.stderr.decode('utf-8', 'replace')[:200]}")
        return "imageio_ffmpeg"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("v2_to_pilot")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--records", default=None,
                    help="Alpamayo records.parquet; when given, only clips "
                         "present in BOTH are eligible")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args(argv)

    pick, corp = pick_clips(a.corpus, a.records, a.n, a.seed)
    print(f"[bridge] picked {len(pick)} clips (seed {a.seed})", flush=True)

    vdir, edir = os.path.join(a.out, "videos"), os.path.join(a.out, "ego")
    os.makedirs(vdir, exist_ok=True)
    os.makedirs(edir, exist_ok=True)

    written, failed = [], []
    for cid in pick:
        try:
            # Imported per clip so a broken environment (no torchvision, no
            # imageio) lands in failures.json with its reason and still exits
            # non-zero, rather than raising past the handler and leaving no
            # record of WHICH stage of the bridge was unusable.
            from tanitad.data.v2_dataset import decode_full_episode
            ep = decode_full_episode(corp[cid])
            arr = stacked_to_rgb(ep.frames)
            p = torch.as_tensor(ep.poses).float().numpy()
            if p.shape[0] != arr.shape[0]:
                raise ValueError(f"{arr.shape[0]} frames vs {p.shape[0]} poses "
                                 f"— refusing to write a misaligned clip")
            mp4 = os.path.join(vdir, f"{cid}.mp4")
            write_mp4(mp4, arr, a.fps)
            np.savez(os.path.join(edir, f"{cid}.npz"), poses=p[:, :4],
                     actions=torch.as_tensor(ep.actions).float().numpy())
            written.append(cid)
            print(f"[bridge] {cid}: {arr.shape[0]} frames {arr.shape[1:]} "
                  f"-> mp4; poses {p.shape}", flush=True)
        except Exception as e:                          # per-clip, never global
            failed.append({"clip": cid, "error": f"{type(e).__name__}: {e}"})
            print(f"[bridge] FAILED {cid}: {type(e).__name__}: {e}", flush=True)

    json.dump(written, open(os.path.join(a.out, "clips.json"), "w"), indent=1)
    if failed:
        json.dump(failed, open(os.path.join(a.out, "failures.json"), "w"),
                  indent=1)
    print(f"BRIDGE_DONE n={len(written)} failed={len(failed)}", flush=True)
    # ⛔ zero clips is a FAILURE, not a quiet success — the earlier bridge exited
    # 0 on n=0 and the chain happily launched a 9B VLM against an empty dir.
    return 0 if written else 3


if __name__ == "__main__":
    raise SystemExit(main())
