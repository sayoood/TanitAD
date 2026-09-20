"""What does a navhard synthetic scene actually carry? Read ONE and print its shape.

Asks specifically: are there FUTURE frames? If the scene carries the human's future poses, the
scene's own affordance (how far the ego actually travels in the 4 s horizon) is measurable with
0 GPU and NO arm -- which is the arm-independent control the prereg commits to.
"""
from __future__ import annotations

import io
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

TAR = ("C:/Users/Admin/navsim/data/navsim-v2/"
       "navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz")


def main() -> int:
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and "/synthetic_scene_pickles/" in m.name
                    and m.name.endswith(".pkl")):
                continue
            o = WinUnpickler(io.BytesIO(tf.extractfile(m).read())).load()
            print("member:", m.name)
            print("top-level keys:", sorted(o.keys()))
            print()
            md = o.get("scene_metadata", {})
            print("scene_metadata keys:", sorted(md.keys()) if hasattr(md, "keys") else type(md))
            for k in sorted(md.keys()) if hasattr(md, "keys") else []:
                v = md[k]
                print(f"   {k} = {v!r}"[:160])
            print()
            fr = o["frames"]
            print("n frames:", len(fr))
            print("frame[0] keys:", sorted(fr[0].keys()))
            print()
            for i, f in enumerate(fr):
                es = f.get("ego_status", {})
                print(f"  frame {i}: ego_status keys={sorted(es.keys())}")
                if i == 0:
                    for k in sorted(es.keys()):
                        print(f"      {k} = {es[k]!r}"[:200])
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
