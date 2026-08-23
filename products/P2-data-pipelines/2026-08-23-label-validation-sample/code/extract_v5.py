"""Validation sample v5 — the re-emitted labels, on JOIN-FREE frames.

Frames are decoded from the clip mp4 whose FILENAME IS THE CLIP UUID, and poses
come from the provider egomotion keyed by the same UUID. There is no legacy-id
lookup anywhere in this path, so the 20.5 % mis-join that corrupted the previous
sample cannot recur here.

Every field carries the SOURCE it was filled from (PI request): geometry, the
VLM chain-of-thought, or an explicit abstain.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import av
import numpy as np

REPO = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.data import egomotion_source as ES        # noqa: E402

MP4 = Path("C:/Users/Admin/tanitad-data/physicalai/r0/camera_front_wide")
LBL = Path("C:/Users/Admin/tanitad-wt/_s2build/labels_geom/s2_labels_geom.jsonl")
OUT = Path("C:/Users/Admin/tanitad-wt/_s2build/validation")
T0_S, HZ = 8.0, 10.0
OFFSETS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
STRAT_LO = 8.0


def decode(path: Path, times):
    want, got, wi = sorted(times), {}, 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb = float(st.time_base)
        for fr in c.decode(st):
            if wi >= len(want):
                break
            t = float(fr.pts) * tb
            while wi < len(want) and t >= want[wi] - 1e-6:
                got[want[wi]] = fr.to_ndarray(format="rgb24")
                wi += 1
    if wi and wi < len(want):
        for k in want[wi:]:
            got[k] = got[want[wi - 1]]
    return got


def b64(arr, size=340):
    from PIL import Image
    im = Image.fromarray(arr)
    w, h = im.size
    s = size / max(1, h)
    im = im.resize((int(w * s), size), Image.LANCZOS)
    left = max(0, (im.width - size) // 2)
    im = im.crop((left, 0, left + size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> None:
    labels = {}
    for line in LBL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["clip_id"]] = r

    have = sorted(c for c in labels
                  if (MP4 / f"{c}.camera_front_wide_120fov.mp4").exists())
    print(f"join-free clips available: {len(have)}")

    out = []
    for c in have:
        r = labels[c]
        try:
            tr = ES.load(c, max_s=T0_S + 30.0)
        except Exception as ex:                              # noqa: BLE001
            print(f"  skip {c[:8]}: {ex!r}")
            continue
        times = [max(0.0, T0_S + o) for o in OFFSETS]
        imgs = decode(MP4 / f"{c}.camera_front_wide_120fov.mp4", times)
        if not imgs:
            continue
        p = tr.poses
        k = tr.key_index
        cy, sy = np.cos(-p[k, 2]), np.sin(-p[k, 2])
        dx, dy = p[:, 0] - p[k, 0], p[:, 1] - p[k, 1]
        ex, ey = cy * dx - sy * dy, sy * dx + cy * dy

        frames, meta = {}, []
        for o in OFFSETS:
            a = imgs.get(max(0.0, T0_S + o))
            if a is None:
                continue
            key = f"t{o:+.0f}"
            frames[key] = b64(a)
            i = min(len(p) - 1, int(round((T0_S + o) * HZ)))
            meta.append({"key": key, "offset_s": o, "strategic": o >= STRAT_LO,
                         "v": round(float(p[i, 3]), 1)})

        out.append({
            **{kk: r[kk] for kk in ("clip_id", "g_str", "a_str", "g_tac",
                                    "a_tac", "manoeuvre", "horizon",
                                    "semantics", "_provenance")},
            "frames": frames, "frame_meta": meta,
            "traj": {"px": [round(float(q), 2) for q in ex[max(0, k - 60):k + 1]],
                     "py": [round(float(q), 2) for q in ey[max(0, k - 60):k + 1]],
                     "fx": [round(float(q), 2) for q in ex[k:]],
                     "fy": [round(float(q), 2) for q in ey[k:]],
                     "band_i": int(STRAT_LO * HZ)},
        })

    (OUT / "sample_v5.json").write_text(json.dumps(out), encoding="utf-8")
    slim = [{kk: v for kk, v in r.items() if kk != "frames"} for r in out]
    (OUT / "sample_v5_slim.json").write_text(json.dumps(slim, indent=1),
                                             encoding="utf-8")
    print(f"[v5] {len(out)} clips with join-free frames -> sample_v5.json")


if __name__ == "__main__":
    main()
