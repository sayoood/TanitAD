"""Frame montages decoded DIRECTLY from the UUID-keyed clip mp4.

⛔ WHY NOT THE EPISODE CACHE. The cache is keyed by the colliding 16-bit
`episode_id_legacy`, and 8 of 39 joins (20.5 %) resolved to the WRONG EPISODE —
which means an earlier frame-by-frame validation inspected other clips' frames
and "confirmed" conclusions about clips it never saw. The mp4 filename IS the
clip UUID, so this path has no join at all and cannot make that error.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack")
from tanitad.data import egomotion_source as ES        # noqa: E402

MP4 = Path("C:/Users/Admin/tanitad-data/physicalai/r0/camera_front_wide")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/montages_mp4")
OUT.mkdir(parents=True, exist_ok=True)

T0_S = 8.0
OFFSETS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
CELL = 236


def frames_at(path: Path, times_s):
    """Decode the frames nearest each requested time. Single sequential pass."""
    want = sorted(times_s)
    got, wi = {}, 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = float(stream.time_base)
        for frame in container.decode(stream):
            if wi >= len(want):
                break
            t = float(frame.pts) * tb
            while wi < len(want) and t >= want[wi] - 1e-6:
                got[want[wi]] = frame.to_ndarray(format="rgb24")
                wi += 1
    if want and wi < len(want):          # clip shorter than asked
        last = got.get(want[wi - 1]) if wi else None
        for k in want[wi:]:
            if last is not None:
                got[k] = last
    return got


def montage(clip_id: str, label: dict | None = None):
    src = MP4 / f"{clip_id}.camera_front_wide_120fov.mp4"
    if not src.exists():
        return None
    times = [max(0.0, T0_S + o) for o in OFFSETS]
    imgs = frames_at(src, times)
    if not imgs:
        return None
    try:
        tr = ES.load(clip_id, max_s=T0_S + 12.0)
        spd = {o: float(tr.poses[min(len(tr.poses) - 1,
                                     int(round((T0_S + o) * ES.HZ))), 3])
               for o in OFFSETS}
    except Exception:                                   # noqa: BLE001
        spd = {o: float("nan") for o in OFFSETS}

    sheet = Image.new("RGB", (CELL * len(OFFSETS), CELL + 24), (16, 20, 26))
    d = ImageDraw.Draw(sheet)
    for k, o in enumerate(OFFSETS):
        arr = imgs.get(max(0.0, T0_S + o))
        if arr is None:
            continue
        im = Image.fromarray(arr)
        w, h = im.size
        s = CELL / max(1, h)
        im = im.resize((int(w * s), CELL), Image.LANCZOS)
        left = max(0, (im.width - CELL) // 2)
        sheet.paste(im.crop((left, 0, left + CELL, CELL)), (k * CELL, 24))
        txt = f"t{o:+.0f}s v={spd[o]:.1f}" if o else f"KEY t0 v={spd[o]:.1f}"
        if o == 0:
            d.rectangle([k * CELL, 24, k * CELL + CELL - 1, CELL + 23],
                        outline=(90, 155, 255), width=3)
        d.text((k * CELL + 5, 6), txt, fill=(210, 225, 240))
    f = OUT / f"{clip_id[:8]}.png"
    sheet.save(f)
    return f


def main() -> None:
    ids = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    made = []
    for c in ids:
        try:
            f = montage(c)
            if f:
                made.append(str(f))
        except Exception as ex:                          # noqa: BLE001
            print(f"  FAIL {c[:8]}: {ex!r}")
    print(f"wrote {len(made)} mp4 montages")
    for m in made:
        print(" ", m)


if __name__ == "__main__":
    main()
