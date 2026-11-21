"""Which metadata field is the CSV's `token`? Try them ALL and report the join size for each.

⛔ Do not guess and do not assume the name that sounds right: `scene_token` joined ZERO of 204.
Also counts pkl members BY DIRECTORY, because a `.pkl`-only filter previously invented 2,731
phantom 'unreadable' scenes in the navhard archive.
"""
from __future__ import annotations

import collections
import csv
import io
import pathlib
import re
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

TAR = "C:/Users/Admin/navsim/data/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz"
RAW = pathlib.Path("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                   "2026-09-19-navsim-refcv4b-bridge/raw")
HEX = re.compile(r"^[0-9a-f]{8,}$")
FIELDS = ["scene_token", "initial_token", "corresponding_original_initial_token",
          "corresponding_original_scene"]


def main() -> int:
    scored = set()
    for r in csv.DictReader(open(RAW / "score_STOP_zero.csv", newline="", encoding="utf-8")):
        t = (r.get("token") or "")
        if HEX.match(t):
            scored.add(t)
    print(f"scored tokens in CSV: {len(scored)}  lens={collections.Counter(len(t) for t in scored)}")

    dirs = collections.Counter()
    got = {f: {} for f in FIELDS}
    nframes = collections.Counter()
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".pkl")):
                continue
            dirs[str(pathlib.PurePosixPath(m.name).parent)] += 1
            try:
                o = WinUnpickler(io.BytesIO(tf.extractfile(m).read())).load()
                md = o["scene_metadata"]
                v = o["frames"][-1]["ego_status"]["ego_velocity"]
                sp = round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 6)
                nframes[len(o["frames"])] += 1
                for f in FIELDS:
                    if f in md and isinstance(md[f], str):
                        got[f][md[f]] = sp
            except Exception as e:
                dirs[f"__FAILED__ {type(e).__name__}"] += 1

    print("\npkl members BY DIRECTORY:")
    for d, n in dirs.most_common():
        print(f"  {n:>5}  {d}")
    print(f"\nframes-per-scene histogram: {dict(nframes)}")

    print("\njoin size per candidate field (of %d scored):" % len(scored))
    for f in FIELDS:
        j = scored & set(got[f])
        print(f"  {f:<40} distinct={len(got[f]):>5}  JOINED={len(j):>4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
