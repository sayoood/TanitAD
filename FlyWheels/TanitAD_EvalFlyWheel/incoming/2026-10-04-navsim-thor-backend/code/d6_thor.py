#!/usr/bin/env python3
"""Run D6's ``d6_fan_score.py`` (UNCHANGED, imported from a byte-identical copy) on the Thor backend.

    CUDA_VISIBLE_DEVICES= PYTHONHASHSEED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    PYTHONPATH=/home/nvidia/navsim-thor/code/stubs nice -n 19 ionice -c3 setsid \
    /home/nvidia/venvs/navsim-cpu/bin/python d6_thor.py --d6-code /home/nvidia/navsim-thor/code/d6 -- \
        --fan <fan.jsonl> --hooks <hooks.json> --out <scored.jsonl> --shard k/n [d6 flags ...]

What this shim does, and nothing else: the dev-box env defaults become the Thor tmpfs mirror (set
BEFORE ``d6_rescore`` runs its ``os.environ.setdefault``), ``thor_compat.install()`` (pickled
WindowsPath + Windows map_root in the dev-box-built metric caches), and ``d6_rescore.CACHE`` -> the
tmpfs copy of the SAME navhard cache (tree hash verified). ⚠ UNVALIDATED on Thor until a D6 shard is
re-scored there and compared cell-for-cell with its dev-box output (see RESULT.md); D6 scores plans
directly via ``pdm_score`` (no seam agent, so no AgentInput fingerprint is involved).
"""
from __future__ import annotations

import argparse
import os
import sys

THOR_ENV = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": "/dev/shm/navsim/data/maps",
    "NAVSIM_EXP_ROOT": "/dev/shm/navsim/exp",
    "NAVSIM_DEVKIT_ROOT": "/home/nvidia/navsim-thor/src/navsim-v2",
    "OPENSCENE_DATA_ROOT": "/dev/shm/navsim/data/openscene",
}
CACHE = "/dev/shm/navsim/exp/metric_cache_navhard_two_stage"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d6-code", required=True)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest
    for k, v in THOR_ENV.items():
        os.environ[k] = v
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import thor_compat
    thor_compat.install()
    sys.path.insert(0, a.d6_code)
    import d6_rescore as R
    R.CACHE = a.cache
    import d6_fan_score
    sys.argv = [os.path.join(a.d6_code, "d6_fan_score.py")] + rest
    return d6_fan_score.main()


if __name__ == "__main__":
    raise SystemExit(main())
