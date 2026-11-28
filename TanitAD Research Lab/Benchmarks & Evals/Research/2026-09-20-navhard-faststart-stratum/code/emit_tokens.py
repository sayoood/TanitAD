"""Emit `initial_token,scene_token,v0_ms` for a two-stage archive.

⛔ THE JOIN KEY IS `initial_token`, NOT `scene_token`. MEASURED (which_token.py): against the
scorer's per-scene CSVs, `scene_token` joins 0 of 204 and `initial_token` joins 204/204. Both
are emitted so the table can be read either way, but the FIRST column is the one that joins.

Counts members BY DIRECTORY, because a `.pkl`-only filter previously invented phantom
'unreadable' scenes: these archives hold `openscene_meta_datas` beside the scene pickles.
"""
from __future__ import annotations

import collections
import io
import pathlib
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

ARCHIVES = {
    "warmup": ("C:/Users/Admin/navsim/data/navsim-v2/navsim_v2.2_warmup_two_stage.tar.gz",
               "warmup_token_v0.csv"),
    "navhard": ("C:/Users/Admin/navsim/data/navsim-v2/"
                "navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz",
                "navhard_token_v0.csv"),
}
HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    which = sys.argv[1]
    tar, out = ARCHIVES[which]
    rows, dirs = [], collections.Counter()
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and "/synthetic_scene_pickles/" in m.name
                    and m.name.endswith(".pkl")):
                if m.isfile() and m.name.endswith(".pkl"):
                    dirs[str(pathlib.PurePosixPath(m.name).parent)] += 1
                continue
            dirs[str(pathlib.PurePosixPath(m.name).parent)] += 1
            o = WinUnpickler(io.BytesIO(tf.extractfile(m).read())).load()
            md = o["scene_metadata"]
            v = o["frames"][-1]["ego_status"]["ego_velocity"]
            # ⛔ corresponding_original_scene is the CLUSTER key: warmup's 204 synthetic scenes
            # derive from only 16 originals, so a per-scene bootstrap is pseudo-replication.
            rows.append((md["initial_token"], md["scene_token"],
                         md.get("corresponding_original_scene", ""),
                         round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 6)))
            if len(rows) % 1000 == 0:
                print(f"  {len(rows)}", flush=True)

    for d, n in dirs.most_common():
        print(f"  {n:>5} members  {d}")
    assert rows, "control: zero scenes read"
    assert len({r[0] for r in rows}) == len(rows), "control: initial_token not unique"
    assert len({r[1] for r in rows}) == len(rows), "control: scene_token not unique"
    (HERE / out).write_bytes(("initial_token,scene_token,orig_scene,v0_ms\n"
                              + "\n".join(f"{a},{b},{c},{d}" for a, b, c, d in rows)).encode())
    import collections as _c
    cl = _c.Counter(r[2] for r in rows)
    print(f"{which}: {len(rows)} scenes, {len(cl)} original-scene CLUSTERS "
          f"(median {sorted(cl.values())[len(cl)//2]} scenes/cluster) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
