"""navhard's start-speed distribution, read STREAMING from the tar so 8,195 files are never
written to disk.

WHY: `H-NAVHARD-STOP-1` predicts that on a split whose starts are NOT slow, an all-zero STOP plan
must LOSE. Warmup's starts are slow -- median v0 4.14 m/s, 36/204 below 1 m/s -- and that is what
switches on the <= 5 m ego-progress clause (`pdm_scorer.py:231-236`), which MEASURED fires on
37/204 = 18.1 % of warmup stage-2 scenes. This measures the PRECONDITION on navhard.

/!\ SCOPE, stated before the number: warmup's figures are over the 204 STAGE-2 scenes the scorer
ran. This reads EVERY synthetic scene in the navhard archive, which is a different and larger
population. It answers "are navhard's starts broadly faster", not "are the scored subset's".

/!\ The archive holds TWO .pkl directories: synthetic_scene_pickles (5,462, the scenes) and
openscene_meta_datas (2,731, a DIFFERENT object with no ego_status). A first run filtered only on
".pkl", swallowed the KeyError and reported 2,731 "unreadable" -- which reads as a data-quality
problem and was entirely the filter. The scene count now cross-checks against the 5,462 rows of
synthetic_scenes_attributes.csv.

/!\ The pickles carry pathlib.PosixPath and will not unpickle on Windows without the
PurePosixPath mapping in peek.py; they also need the navsim venv, which has `nuplan`.
"""
from __future__ import annotations

import io
import json
import pathlib
import statistics as st
import sys
import tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peek import WinUnpickler  # noqa: E402

TAR = ("C:/Users/Admin/navsim/data/navsim-v2/"
       "navsim_v2.2_navhard_two_stage_scene_pickles.tar.gz")
OUT = pathlib.Path(__file__).resolve().parent / "navhard_startspeed.json"


def main() -> int:
    speeds, n_read, n_fail = [], 0, 0
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not (m.isfile() and "/synthetic_scene_pickles/" in m.name
                    and m.name.endswith(".pkl")):
                continue
            f = tf.extractfile(m)
            if f is None:
                n_fail += 1
                continue
            try:
                o = WinUnpickler(io.BytesIO(f.read())).load()
                v = o["frames"][-1]["ego_status"]["ego_velocity"]
                speeds.append(float((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5))
                n_read += 1
            except Exception:
                n_fail += 1
            if n_read and n_read % 1000 == 0:
                print(f"  {n_read} scenes read", flush=True)

    assert n_read > 0, "control: zero scenes read -- the measurement is about the READER"
    speeds.sort()

    def pct(p):
        return speeds[min(len(speeds) - 1, int(p * len(speeds)))]

    res = {
        "_what": "navhard synthetic-scene start speed |v| at the t0 frame",
        "_evidence_class": "MEASURED (ours, streaming from the archive; no GPU, no scoring)",
        "_scope": ("EVERY synthetic scene in the navhard archive -- NOT the stage-2 subset the "
                   "scorer runs. Warmup's 4.14 m/s median is over its 204 scored stage-2 scenes, "
                   "so the two populations are not identical and the comparison is indicative."),
        "n_scenes": n_read, "n_unreadable": n_fail,
        "median_ms": round(st.median(speeds), 4),
        "mean_ms": round(st.mean(speeds), 4),
        "p10_ms": round(pct(0.10), 4), "p25_ms": round(pct(0.25), 4),
        "p75_ms": round(pct(0.75), 4), "p90_ms": round(pct(0.90), 4),
        "min_ms": round(speeds[0], 4), "max_ms": round(speeds[-1], 4),
        "n_below_1ms": sum(1 for s in speeds if s < 1.0),
        "frac_below_1ms": round(sum(1 for s in speeds if s < 1.0) / n_read, 4),
        "n_below_2ms": sum(1 for s in speeds if s < 2.0),
        "warmup_reference_stage2_only": {
            "median_ms": 4.14, "n_below_1ms": 36, "n_scenes": 204,
            "frac_below_1ms": round(36 / 204, 4),
            "source": "…/2026-09-19-navsim-refcv4b-bridge/RESULT.md (landed 3b02a41)"},
    }
    print(json.dumps(res, indent=1))
    OUT.write_bytes(json.dumps(res, indent=1).encode("utf-8"))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
