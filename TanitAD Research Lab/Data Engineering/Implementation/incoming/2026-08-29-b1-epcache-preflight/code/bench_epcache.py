"""MEASURE the epcache per-clip cost for the B1 corpus — scheduling input.

Question from the Master Mind: wall-clock for 4,719 clips at w120 256x640
cylindrical, and the BOTTLENECK (decode / IO / GPU?). Guessing is not admissible
for a decision that orders Thor's queue, so this times the REAL path:
`physicalai._decode_mp4` with the real per-clip intrinsics, at the real frame.

Reports decode-only vs decode+remap so the attribution is measured, not assumed.
"""
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("PAI_DECODE_THREADS", "4")
sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")

import av  # noqa: E402
import torch  # noqa: E402

from tanitad.data.calib import PHYSICALAI_WIDE120_256x640 as FRAME  # noqa: E402
from tanitad.data.physicalai import _decode_mp4  # noqa: E402

CAM = Path("C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
clips = sorted(CAM.glob("*.mp4"))[:N]
print(f"torch threads {torch.get_num_threads()} | PAI_DECODE_THREADS "
      f"{os.environ['PAI_DECODE_THREADS']} | clips {len(clips)}", flush=True)


def decode_only(p):
    n = 0
    with av.open(str(p)) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        st.thread_count = int(os.environ["PAI_DECODE_THREADS"])
        for f in c.decode(video=0):
            f.to_ndarray(format="rgb24")
            n += 1
    return n


rows = []
for p in clips:
    mb = p.stat().st_size / 1e6
    t0 = time.time()
    nfr = decode_only(p)
    t_dec = time.time() - t0
    t0 = time.time()
    out = _decode_mp4(p, 256, frame=FRAME, projection_mode="cylindrical")
    t_full = time.time() - t0
    rows.append((p.name[:8], mb, nfr, t_dec, t_full, tuple(out.shape)))
    print(f"  {rows[-1][0]} {mb:5.1f} MB {nfr:4d} fr | decode {t_dec:6.2f}s | "
          f"decode+remap {t_full:6.2f}s | remap share "
          f"{(t_full-t_dec)/t_full*100:4.1f}% | out {tuple(out.shape)}", flush=True)

full = [r[4] for r in rows]
dec = [r[3] for r in rows]
med = statistics.median(full)
print(f"\nper-clip median {med:.2f}s (decode {statistics.median(dec):.2f}s = "
      f"{statistics.median(dec)/med*100:.0f}%)")
for W in (1, 4, 6, 8, 12):
    print(f"  {W:2d} worker(s): 4719 clips -> {4719*med/W/3600:6.2f} h")
ex = rows[0]
nb = 1
for d in ex[5]:
    nb *= d
print(f"\nartifact: one clip {ex[2]} fr -> {nb/1e6:.1f} MB uint8 "
      f"-> 4719 clips ~ {nb*4719/1e9:.0f} GB raw uint8 "
      f"(the cache stores ENCODED frames, so the real figure is the codec's)")
print(f"GPU used by this path: {'YES' if torch.cuda.is_initialized() else 'NO'}")
