"""camblind_roll_r9_e2.py -- launcher: camblind_roll_r9.py (UNCHANGED) with the refcv9 staging proxy installed, for a K2 that holds B4 / T1.  NEW file (E2-ROLL).

    python -X utf8 camblind_roll_r9_e2.py --camblind-script <.../2026-11-03-refcv9-camblind/code/camblind_roll_r9.py> --camera derange --derange-seed 0 \
        --ztac-out <f.npz> --require-seal -- <battery child argv>
The hook fires on the first import of ``r7_roll`` (inside camblind's ``roll_seed_r8_pf`` import), then camblind runs as ``__main__`` with its own argv.
HAZARD (UNVERIFIED; cannot be fixed without editing it): camblind_roll_r9.py hard-codes ``EVAL_REMAP = C:/Users/Admin/refcv8_pf/refcv9_l1/code/refcv8_eval_remap_devbox_l1.json``
(a RELATIVE path on Thor => run with cwd = a dir holding that relative tree) and ``PKG = <incoming>/2026-10-27-refcv9-anticheat`` (must be shipped);
its ``L1_PF_CODE`` / ``L1_R8_CODE`` env pins are required here."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_battery_r9_e2 as R9  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if "--camblind-script" not in argv:
        raise SystemExit("[r9e2-cb] --camblind-script <camblind_roll_r9.py> is required")
    i = argv.index("--camblind-script")
    script = Path(argv[i + 1])
    del argv[i:i + 2]
    R9.require_env(pins=("R8_LAUNCH_TREE", "PF_R8_CODE", "R8_EVAL_KIT", "L1_PF_CODE", "L1_R8_CODE"))
    R9.child_stage_hook()
    sys.argv = [str(script)] + argv
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        st = getattr(sys.modules.get("r7_roll"), "R9E2_STAGER", None)
        print(f"[r9e2-cb] staged forwards: {None if st is None else st.census()}", flush=True)
        if (e.code in (0, None)) and (st is None or not st.n_forwards):
            raise SystemExit("[r9e2-cb] the roll ended OK but nothing was staged")
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
