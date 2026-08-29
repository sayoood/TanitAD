"""Re-time the epcache build against the format the v7 trainer ACTUALLY reads.

My first benchmark priced `ep_*.pt` (raw `frames_u8`) -- the LEGACY epcache. The
v7 trainer's `--v2-cache` reads `*.v2ep.pt`, written by
`scripts/v2_compressed.py::build_compressed`, which stores ENCODED frames
(`jpeg_buf` + `jpeg_len` + `codec`). That adds an encode step my raw-write
benchmark never paid, and it changes the artifact size by ~an order of magnitude.

This measures the ENCODE leg on real remapped frames at the real geometry, and
the resulting BYTES -- so the size projection is ours, not inherited.

⚠️ The buffer is named `jpeg_buf` while `codec` may be "png". Read the codec
field, never the buffer name (the CLAUDE.md naming trap, logged after a decode
raised into a pre-allocated memmap and left 2.76 GB of zeros).
"""
import os
import statistics
import sys
import time

os.environ.setdefault("PAI_DECODE_THREADS", "4")
os.environ.setdefault(
    "TANITAD_PAI_INTRINSICS",
    "C:/Users/Admin/tanitad-data/physicalai/calibration/physicalai_front_wide_intrinsics.csv")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torchvision.io as tvio  # noqa: E402

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FRAME  # noqa: E402
from tanitad.data.physicalai import _decode_mp4  # noqa: E402

CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
# an episode stores ~10 Hz timesteps, not every decoded frame
EP_FRAMES = 201
clips = sorted(CAM.glob("*.mp4"))[:4]
print(f"frame {FRAME.height}x{FRAME.width} {FRAME.projection} | "
      f"episode frames {EP_FRAMES}", flush=True)

rows = []
for p in clips:
    t0 = time.time()
    vid = _decode_mp4(p, 256, frame=FRAME, projection_mode="cylindrical")
    t_dr = time.time() - t0
    # subsample to the episode's 10 Hz grid, as the builder does
    idx = torch.linspace(0, vid.shape[0] - 1, EP_FRAMES).long()
    sub = vid[idx]
    out = {}
    for codec, enc in (("png", tvio.encode_png),
                       ("jpeg", lambda x: tvio.encode_jpeg(x, quality=95))):
        t0 = time.time()
        bufs = [enc(sub[i]) for i in range(sub.shape[0])]
        out[codec] = (time.time() - t0, sum(b.numel() for b in bufs) / 1e6)
    rows.append((p.name[:8], t_dr, out))
    print(f"  {rows[-1][0]} decode+remap {t_dr:5.2f}s | "
          + " | ".join(f"{c} enc {v[0]:5.2f}s -> {v[1]:6.1f} MB"
                       for c, v in out.items()), flush=True)

for codec in ("png", "jpeg"):
    dr = statistics.median(r[1] for r in rows)
    en = statistics.median(r[2][codec][0] for r in rows)
    mb = statistics.median(r[2][codec][1] for r in rows)
    tot = dr + en
    print(f"\n[{codec}] per clip: decode+remap {dr:.2f}s + encode {en:.2f}s "
          f"= {tot:.2f}s ({en/tot*100:.0f}% encode) | {mb:.1f} MB/episode")
    print(f"   4719 clips: " + " | ".join(
        f"{w}w {4719*tot/w/3600:.2f}h" for w in (1, 6, 8)))
    print(f"   corpus artifact: {mb*4719/1000:.0f} GB")
